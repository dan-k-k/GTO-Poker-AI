# unified_api.py
# find . -maxdepth 2 -not -path '*/.*' 
import os
import uuid
import base64
import pickle
import torch
import uvicorn
from io import BytesIO
from contextlib import asynccontextmanager
from typing import Dict, Optional, List
from datetime import datetime
from starlette.status import HTTP_303_SEE_OTHER # Add this import

from fastapi import FastAPI, Request, Response, HTTPException, Depends, Cookie
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# --- App Imports ---
from app.TexasHoldemEnv import TexasHoldemEnv
from app.nfsp_components import NFSPAgent
from app.poker_core import GameState, string_to_card_id
from app.feature_extractor import FeatureExtractor
from app._visuals import create_table_image 

# --- Schema Imports ---
from api_schemas import GameStateInput, ActionResponse, ActionLog

# Global State & Configuration
# ==========================================

# 1. Global AI Model (Loaded once)
GLOBAL_MODEL_AGENT: Optional[NFSPAgent] = None

# 2. In-Memory Session Store for "Play vs Bot"
ACTIVE_SESSIONS: Dict[str, Dict] = {}

templates = Jinja2Templates(directory="templates")

# Define standard config centrally so it can be reused
STANDARD_AGENT_CONFIG = {
    'eta': 0.1, 
    'gamma': 0.99, 
    'batch_size': 128, 
    'update_frequency': 1, 
    'learning_rate': 0.001, 
    'target_update_frequency': 100
}

def load_global_model():
    """Loads the 'Brain' used for the Solver API."""
    print("Loading Global AI Model...")
    buffer_config = {'rl_buffer_capacity': 10000, 'sl_buffer_capacity': 10000}
    
    agent = NFSPAgent(seat_id=1, agent_config=STANDARD_AGENT_CONFIG, buffer_config=buffer_config, random_equity_trials=500, starting_stack=200)
    
    as_path = "training_output/models/nfsp_agent1_as_latest.pt"
    br_path = "training_output/models/nfsp_agent1_br_latest.pt"
    
    if os.path.exists(as_path) and os.path.exists(br_path):
        agent.load_models(br_path=br_path, as_path=as_path)
        agent.opponent_as_network = agent.as_network 
        agent.set_mode('eval')
        print(" Global Model Loaded.")
    else:
        print(" WARNING: Model files not found. Solver will be random.")
    
    return agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global GLOBAL_MODEL_AGENT
    GLOBAL_MODEL_AGENT = load_global_model()
    yield
    # Shutdown
    print("Shutting down...")
    ACTIVE_SESSIONS.clear()

app = FastAPI(title="Poker AI Unified Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper Functions
# ==========================================

def get_game_response(session_data: dict) -> dict:
    """Formats the env state for the UI (Images, valid actions, etc)."""
    env = session_data['env']
    state_dict = env.get_state_dict()
    
    if env.state and not env.state.terminal:
        state_dict['legal_actions'] = env.state.get_legal_actions()
        state_dict['min_raise'] = env.state.get_min_raise_amount()

    if env.state:
        pil_image = create_table_image(env.state, env, show_all_cards=env.state.terminal)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        state_dict['table_image'] = img_str
    
    state_dict['pnl_history'] = session_data.get('pnl_history', [])
    return state_dict

# --- Helper for Solver ---
def convert_input_to_state(data: GameStateInput) -> GameState:
    num_players = len(data.stacks)
    hole_cards_ints = [[string_to_card_id(c) for c in hand] for hand in data.hole_cards]
    community_ints = [string_to_card_id(c) for c in data.community_cards]
    active = [stack > 0 for stack in data.stacks]
    all_in = [stack == 0 and bet > 0 for stack, bet in zip(data.stacks, data.current_bets)]
    
    return GameState(
        num_players=num_players,
        starting_stack=max(data.initial_stacks), 
        small_blind=1, big_blind=2,   
        hole_cards=hole_cards_ints, community=community_ints,
        stacks=data.stacks, current_bets=data.current_bets, pot=data.pot,
        starting_stacks_this_hand=data.initial_stacks, 
        starting_pot_this_round=data.pot - sum(data.current_bets),
        active=active, all_in=all_in, acted=[False] * num_players,
        surviving_players=[i for i in range(num_players)],
        stage=data.stage,
        dealer_pos=data.dealer_id,
        sb_pos=(data.dealer_id + 1) % num_players,
        bb_pos=(data.dealer_id + 2) % num_players,
        to_move=data.to_move,
        initial_bet=0, last_raise_size=2, last_raiser=None,
        terminal=False, winners=None, win_reason=None
    )

def reconstruct_feature_extractor(seat_id: int, past_actions: list) -> FeatureExtractor:
    class MockState:
        def __init__(self, stage, last_raiser=None):
            self.stage = stage
            self.last_raiser = last_raiser
    
    extractor = FeatureExtractor(seat_id=seat_id)
    extractor.new_hand()
    
    current_last_raiser = None
    previous_stage = 0

    for log in past_actions:
        # SAFETY: If we moved to a new street, reset the "last raiser" logic
        if log.stage > previous_stage:
            current_last_raiser = None
            previous_stage = log.stage

        # Create state snapshot reflecting context BEFORE this action occurred
        mock_state = MockState(stage=log.stage, last_raiser=current_last_raiser)
        
        # FIX: Pass 'log.action_type' (int) directly, NOT the tuple
        extractor.update_betting_action(
            player_id=log.seat_id, 
            action=log.action_type, 
            state_before_action=mock_state, 
            stage=log.stage
        )
        
        # If this action was a Bet/Raise (type 2), update the raiser for the next iteration
        if log.action_type == 2:
            current_last_raiser = log.seat_id
            
    return extractor


# ==========================================
# PART 1: The Solver API (Stateless)
# ==========================================

@app.post("/get_optimal_action", 
          response_model=ActionResponse,
          summary="Get Solver Recommendation",
          description="Stateless endpoint. Takes a raw game state, reconstructs history, and returns the NFSP bot's optimal move.")
def predict_optimal_action(input_data: GameStateInput):
    if not GLOBAL_MODEL_AGENT:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        current_state = convert_input_to_state(input_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid State Data: {str(e)}")

    fresh_extractor = reconstruct_feature_extractor(
        seat_id=input_data.to_move,
        past_actions=input_data.past_actions
    )

    features_schema = fresh_extractor.extract_features(current_state, GLOBAL_MODEL_AGENT)
    features_vector = features_schema.to_vector()
    
    network = GLOBAL_MODEL_AGENT.as_network
    action_type, amount, _, _, _ = GLOBAL_MODEL_AGENT._get_action_from_network(
        features_vector, network, current_state
    )

    action_map = {0: "fold", 1: "call", 2: "raise"}
    return ActionResponse(
        action_type=action_map.get(action_type, "unknown"),
        amount=amount,
        policy="NFSP_AS_Inference"
    )


# ==========================================
# PART 2: The Playable Game (Stateful)
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/solver", response_class=HTMLResponse)
async def solver_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")

@app.get("/game_state")
async def get_game_state(request: Request):
    # 1. Retrieve or Generate Session ID
    session_id = request.cookies.get("session_id")
    is_new_session = False
    
    if not session_id:
        session_id = str(uuid.uuid4())
        is_new_session = True
        
    # 2. Initialize Game Session if not exists
    if session_id not in ACTIVE_SESSIONS:
        print(f"Initializing new game for session: {session_id}")
        env = TexasHoldemEnv(num_players=2)
        env.reset()
        
        agent = NFSPAgent(seat_id=1, agent_config=STANDARD_AGENT_CONFIG, buffer_config={'rl_buffer_capacity': 10000, 'sl_buffer_capacity': 10000}, random_equity_trials=500, starting_stack=200)
        
        # Share Global Networks
        if GLOBAL_MODEL_AGENT:
            agent.br_network = GLOBAL_MODEL_AGENT.br_network
            agent.as_network = GLOBAL_MODEL_AGENT.as_network
            agent.opponent_as_network = GLOBAL_MODEL_AGENT.as_network
            agent.set_mode('eval')
            
        ACTIVE_SESSIONS[session_id] = {
            'env': env,
            'agent': agent,
            'pnl_history': [env.starting_stack]
        }
    
    # 3. Construct Response
    data = get_game_response(ACTIVE_SESSIONS[session_id])
    response = JSONResponse(content=data)
    
    # 4. Explicitly Set Cookie on the returned JSONResponse object
    if is_new_session:
        response.set_cookie(key="session_id", value=session_id)

    return response

@app.post("/action")
async def handle_human_action(request: Request, data: dict):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in ACTIVE_SESSIONS:
        return JSONResponse({'error': 'Session expired'}, status_code=400)
    
    session_data = ACTIVE_SESSIONS[session_id]
    env = session_data['env']
    bot_agent = session_data['agent']

    if env.state.terminal or env.state.to_move != 0:
        return JSONResponse({'error': 'Not your turn'}, status_code=400)

    state_before_action = env.state.copy()
    
    try:
        act_type = data.get('action')
        amt = data.get('amount')
        
        if act_type == 'fold': env.step(0)
        elif act_type == 'call': env.step(1)
        elif act_type == 'raise': env.step(2, int(amt))
        else: return JSONResponse({'error': 'Invalid action'}, 400)
        
        action_tuple = (0 if act_type=='fold' else 1 if act_type=='call' else 2, int(amt) if amt else 0)
        bot_agent.observe(action_tuple, 0, state_before_action, env.state)
        
    except ValueError as e:
        return JSONResponse({'error': str(e)}, status_code=400)

    return JSONResponse(get_game_response(session_data))

@app.post("/bot_action")
async def handle_bot_action(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in ACTIVE_SESSIONS:
        return JSONResponse({'error': 'Session expired'}, 400)

    session_data = ACTIVE_SESSIONS[session_id]
    env = session_data['env']
    bot_agent = session_data['agent']

    if env.state.terminal or env.state.to_move != 1:
        return JSONResponse({'error': 'Not bot turn'}, 400)

    state_before_action = env.state.copy()
    
    action, amount, _, _, _ = bot_agent.compute_action(env.state)
    env.step(action, amount)
    
    bot_agent.observe((action, amount), 1, state_before_action, env.state)

    return JSONResponse(get_game_response(session_data))

@app.post("/new_game")
async def new_game(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in ACTIVE_SESSIONS:
        return RedirectResponse(url="/game_state")

    session_data = ACTIVE_SESSIONS[session_id]
    env = session_data['env']
    bot_agent = session_data['agent']

    if env.state.win_reason != 'tournament_winner':
        session_data['pnl_history'].append(env.state.stacks[0])
        env.reset()
        bot_agent.new_hand()
    else:
        del ACTIVE_SESSIONS[session_id]
        # CHANGE: Use status_code=303 to force the browser to switch from POST to GET
        return RedirectResponse(url="/game_state", status_code=HTTP_303_SEE_OTHER)

    return JSONResponse(get_game_response(session_data))

if __name__ == "__main__":
    uvicorn.run("unified_api:app", host="0.0.0.0", port=8000, reload=True)
    print("Go to http://localhost:8000/ (.../solver)")

