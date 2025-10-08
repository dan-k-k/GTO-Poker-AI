# main.py (Corrected Version)

import os
import base64
from io import BytesIO
import pickle 
from flask import Flask, session, redirect, url_for, render_template, request, jsonify
from dataclasses import fields
from flask_session import Session 

from app.TexasHoldemEnv import TexasHoldemEnv
from app.nfsp_components import NFSPAgent
from app._visuals import create_table_image
from app.poker_core import GameState

# --- App initialisation ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Server-side sessions 
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem" 
Session(app) 
# -----------------------------

def get_env_and_agent_from_session() -> tuple:
    """Reconstructs the environment AND the bot agent from the session."""
    game_state_dict = session.get('game_state', {})
    if not game_state_dict:
        return None, None

    # Reconstruct environment
    env = TexasHoldemEnv()
    deck_cards = game_state_dict.pop('deck_cards', None)
    if deck_cards is not None:
        env.deck.cards = deck_cards
    valid_fields = {f.name for f in fields(GameState)}
    filtered_state_dict = {k: v for k, v in game_state_dict.items() if k in valid_fields}
    env.state = GameState(**filtered_state_dict)

    # Reconstruct the agent from pickled data
    pickled_agent = session.get('bot_agent')
    bot_agent = pickle.loads(pickled_agent) if pickled_agent else None
    
    return env, bot_agent

def save_state_to_session(env: TexasHoldemEnv, agent: NFSPAgent):
    """Saves the complete state (env and agent) to the session."""
    session['game_state'] = env.get_state_dict()
    session['bot_agent'] = pickle.dumps(agent)

def get_json_response(env: TexasHoldemEnv):
    """Creates a JSON-serialisable response object from the environment state."""
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
    
    state_dict['pnl_history'] = session.get('pnl_history', [])
    
    return jsonify(state_dict)

# --- Routes ---

@app.route('/')
def index():
    """Renders the main game page shell."""
    return render_template('index.html')

@app.route('/game_state', methods=['GET'])
def get_game_state():
    """Provides the initial game state, creating it if it doesn't exist."""
    env, bot_agent = get_env_and_agent_from_session()

    if not env:

        env = TexasHoldemEnv(num_players=2)
        env.reset()
        session['pnl_history'] = [env.starting_stack]

        # A new agent is created ONLY when a new game starts.
        print("Initializing new bot agent for new game...")
        agent_config = {'eta': 0.1, 'gamma': 0.99, 'batch_size': 128, 'update_frequency': 1, 'learning_rate': 0.001, 'target_update_frequency': 100}
        buffer_config = {'rl_buffer_capacity': 10000, 'sl_buffer_capacity': 10000}
        bot_agent = NFSPAgent(seat_id=1, agent_config=agent_config, buffer_config=buffer_config, random_equity_trials=500, intelligent_equity_trials=500)
        
        as_model_path = "training_output/models/nfsp_agent1_as_latest.pt"
        br_model_path = "training_output/models/nfsp_agent1_br_latest.pt"
        if os.path.exists(as_model_path) and os.path.exists(br_model_path):
            bot_agent.load_models(br_path=br_model_path, as_path=as_model_path)
            # The bot asks itself what it would've done if it were in its opponent's shoes in the previous action (for intelligent_equity/hand strength (HS))
            bot_agent.opponent_as_network = bot_agent.as_network
            print("Trained model loaded and intelligent equity is active.")
        else:
            print("WARNING: Model not found. The bot will play randomly.")

        save_state_to_session(env, bot_agent)

    return get_json_response(env)
    
@app.route('/action', methods=['POST'])
def handle_action():
    """Handles a player's action, updates state, and returns new state."""
    env, bot_agent = get_env_and_agent_from_session()
    if not env or env.state.terminal or env.state.to_move != 0:
        return jsonify({'error': 'Not your turn or game is over'}), 400

    state_before_action = env.state.copy()
    data = request.json
    action_type = data.get('action')
    raw_amount = data.get('amount')
    
    try:
        action_tuple = (0, 0)
        if action_type == 'fold':
            action_tuple = (0, 0)
            env.step(action_tuple[0])
        elif action_type == 'call':
            action_tuple = (1, 0)
            env.step(action_tuple[0])
        elif action_type == 'raise':
            amount = int(raw_amount)
            action_tuple = (2, amount)
            env.step(action_tuple[0], action_tuple[1])
        else:
            return jsonify({'error': 'Invalid action'}), 400
    except (ValueError, TypeError) as e:
        return jsonify({'error': str(e)}), 400

    # Let the bot observe the player's action to update its FeatureExtractor.
    bot_agent.observe(action_tuple, 0, state_before_action, env.state)
    save_state_to_session(env, bot_agent)
    
    return get_json_response(env)

@app.route('/bot_action', methods=['POST'])
def bot_action():
    """Computes and performs the bot's action."""
    env, bot_agent = get_env_and_agent_from_session()
    if not env or env.state.terminal or env.state.to_move != 1:
        return jsonify({'error': 'Not bot\'s turn or game is over'}), 400
        
    state_before_action = env.state.copy()
    action, amount, _, _, _ = bot_agent.compute_action(env.state)
    env.step(action, amount)
    
    # Let the bot observe its own action as well.
    bot_agent.observe((action, amount), 1, state_before_action, env.state)
    
    save_state_to_session(env, bot_agent)
    return get_json_response(env)
    
@app.route('/new_game', methods=['POST'])
def new_game():
    """Resets the game state for a new hand or tournament."""
    env, bot_agent = get_env_and_agent_from_session()
    
    if env and env.state.win_reason != 'tournament_winner':
        # Case 1: Start the next hand in the same game.
        pnl_history = session.get('pnl_history', [])
        pnl_history.append(env.state.stacks[0])
        session['pnl_history'] = pnl_history
        
        env.reset()
        bot_agent.new_hand() 
        save_state_to_session(env, bot_agent)
        
    else:
        # Case 2: Start a brand new tournament.
        session.pop('game_state', None)
        session.pop('bot_agent', None)
        session.pop('pnl_history', None)
        return redirect(url_for('get_game_state'))
    
    return get_json_response(env)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

