# fastapi_app.py
import torch
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from app.TexasHoldemEnv import TexasHoldemEnv
from app.nfsp_components import NFSPAgent
from app.poker_core import GameState, string_to_card_id
from api_schemas import GameStateInput, ActionResponse
from app.feature_extractor import FeatureExtractor
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Helper: Convert "Human" Input -> "Bot" Internal State ---
def convert_input_to_state(data: GameStateInput) -> GameState:
    num_players = len(data.stacks)
    
    # 1. Convert Cards
    hole_cards_ints = [
        [string_to_card_id(c) for c in hand] 
        for hand in data.hole_cards
    ]
    community_ints = [string_to_card_id(c) for c in data.community_cards]

    # 2. Infer "Missing" Game Logic Fields
    active = [stack > 0 for stack in data.stacks]
    all_in = [stack == 0 and bet > 0 for stack, bet in zip(data.stacks, data.current_bets)]
    
    # 3. Construct the Full GameState
    return GameState(
        num_players=num_players,
        starting_stack=max(data.initial_stacks), 
        small_blind=1, 
        big_blind=2,   
        
        hole_cards=hole_cards_ints,
        community=community_ints,
        
        stacks=data.stacks,
        current_bets=data.current_bets,
        pot=data.pot,
        
        # Crucial for correct SPR/Odds calc
        starting_stacks_this_hand=data.initial_stacks, 
        starting_pot_this_round=data.pot - sum(data.current_bets),
        
        active=active,
        all_in=all_in,
        acted=[False] * num_players,
        surviving_players=[i for i in range(num_players)],
        
        stage=data.stage,
        dealer_pos=data.dealer_id,
        sb_pos=(data.dealer_id + 1) % num_players,
        bb_pos=(data.dealer_id + 2) % num_players,
        to_move=data.to_move,
        
        initial_bet=0,
        last_raise_size=2, # Default safe value
        last_raiser=None,
        terminal=False,
        winners=None,
        win_reason=None
    )

# --- Helper: Reconstruct History ---
class MockState:
    """Minimal state wrapper to satisfy FeatureExtractor signature."""
    def __init__(self, stage):
        self.stage = stage

def reconstruct_feature_extractor(seat_id: int, past_actions: list) -> FeatureExtractor:
    """
    Creates a NEW extractor and fast-forwards it through the hand history.
    This ensures the bot 'remembers' previous actions without polluting global state.
    """
    extractor = FeatureExtractor(seat_id=seat_id)
    extractor.new_hand()
    
    for log in past_actions:
        # We create a dummy state because update_betting_action usually checks state.stage
        mock_state = MockState(stage=log.stage)
        
        extractor.update_betting_action(
            player_id=log.seat_id,
            action=(log.action_type, log.amount_added),
            state_before_action=mock_state,
            stage=log.stage
        )
    return extractor

# --- Global Variables ---
global_agent_reference = None 

def load_agent():
    """Initializes the agent and loads PyTorch weights."""
    print("Loading AI Models...")
    
    agent_config = {'eta': 0.1, 'gamma': 0.99, 'batch_size': 128, 
                    'update_frequency': 1, 'learning_rate': 0.001, 
                    'target_update_frequency': 100}
    buffer_config = {'rl_buffer_capacity': 10000, 'sl_buffer_capacity': 10000}
    
    # Initialize Agent
    agent = NFSPAgent(seat_id=1, agent_config=agent_config, 
                      buffer_config=buffer_config, 
                      random_equity_trials=500, intelligent_equity_trials=500)
    
    # Load Weights
    as_path = "training_output/models/nfsp_agent1_as_latest.pt"
    br_path = "training_output/models/nfsp_agent1_br_latest.pt"
    
    if os.path.exists(as_path) and os.path.exists(br_path):
        agent.load_models(br_path=br_path, as_path=as_path)
        agent.opponent_as_network = agent.as_network 
        print("Models loaded successfully.")
    else:
        print("Warning: Models not found. Bot will play randomly.")
    
    return agent

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_agent_reference
    global_agent_reference = load_agent()
    yield
    print("Shutting down...")

# --- Init API ---
app = FastAPI(title="GTO Poker Solver API", lifespan=lifespan)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal Server Error: {str(exc)}", "detail": str(exc)},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": global_agent_reference is not None}

@app.post("/get_optimal_action", response_model=ActionResponse)
def predict(input_data: GameStateInput):
    """
    Reconstructs history, extracts features, and runs inference.
    Does NOT modify the global agent state.
    """
    if not global_agent_reference:
        raise HTTPException(status_code=500, detail="Model not initialized")

    # 1. Reconstruct Physical GameState
    try:
        current_state = convert_input_to_state(input_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid State Data: {str(e)}")

    # 2. Reconstruct Context (The "Brain" History)
    # We create a fresh extractor for THIS specific request
    fresh_extractor = reconstruct_feature_extractor(
        seat_id=input_data.to_move,
        past_actions=input_data.past_actions
    )

    # 3. Manual Inference (Thread-Safe)
    # We use the FRESH extractor with the GLOBAL agent's neural net
    
    # A. Extract Features
    # Note: We pass global_agent_reference so it can access equity trials config if needed
    features_schema = fresh_extractor.extract_features(current_state, global_agent_reference)
    features_vector = features_schema.to_vector()
    
    # B. Run Neural Network
    # We pick the "Average Strategy" (AS) network for stable GTO-like suggestions
    network = global_agent_reference.as_network
    
    # Use the agent's internal helper to process the network output
    action_type, amount, _, predictions = global_agent_reference._get_action_from_network(
        features_vector, 
        network, 
        current_state
    )

    # 4. Format Output
    action_map = {0: "fold", 1: "call", 2: "raise"}
    action_str = action_map.get(action_type, "unknown")
    
    return ActionResponse(
        action_type=action_str,
        amount=amount,
        policy="NFSP_AS_Inference", 
        confidence=None 
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

