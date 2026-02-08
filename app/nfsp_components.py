# app/nfsp_components.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle 
import os
import random
from collections import deque
from typing import Dict, List, Tuple, Optional

from app.poker_agents import NeuralNetworkAgent, GTOPokerNet, ACTION_MAP
from app.poker_feature_schema import PokerFeatureSchema, StreetFeatures
from app.poker_core import GameState
from app.feature_extractor import FeatureExtractor

class ReplayBuffer:
    """RL Replay Buffer for Best Response Policy (DQN-style)."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience tuple."""
        # Ensure data types are correct for PyTorch
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)

class SLBuffer:
    """Reservoir Buffer for Average Strategy."""
    
    def __init__(self, capacity: int):
        self.buffer = [] 
        self.capacity = capacity
        self.total_count = 0 
        
    def push(self, state: np.ndarray, action_index: int):
        """Store state-action_index pair using Reservoir Sampling."""
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action_index))
        else:
            r = random.randint(0, self.total_count)
            if r < self.capacity:
                self.buffer[r] = (state, action_index)
        
        self.total_count += 1
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Sample a batch of state-action pairs."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        
        return states, actions
        
    def __len__(self):
        return len(self.buffer)

class NFSPAgent(NeuralNetworkAgent):
    """
    Neural Fictitious Self-Play Agent.
    Corrected to handle turn-based transitions propertly.
    """
    
    def __init__(self, seat_id: int, agent_config: Dict, buffer_config: Dict, 
                 random_equity_trials: int, intelligent_equity_trials: int):
        super().__init__(seat_id)
        
        self.intelligent_equity_trials = intelligent_equity_trials

        self.feature_extractor = FeatureExtractor(
            seat_id=self.seat_id,
            random_equity_trials=random_equity_trials
        )
        
        self.eta = agent_config['eta']
        self.gamma = agent_config['gamma']
        self.batch_size = agent_config['batch_size']
        self.update_frequency = agent_config['update_frequency']
        lr = agent_config['learning_rate']
        rl_buffer_capacity = buffer_config['rl_buffer_capacity']
        sl_buffer_capacity = buffer_config['sl_buffer_capacity']
        
        input_size = PokerFeatureSchema.get_vector_size()
        self.br_network = GTOPokerNet(input_size=input_size)
        self.br_target_network = GTOPokerNet(input_size=input_size)
        self.br_target_network.load_state_dict(self.br_network.state_dict())
        self.br_target_network.eval()
        self.as_network = GTOPokerNet(input_size=input_size)
        
        self.br_optimizer = optim.Adam(self.br_network.parameters(), lr=lr)
        self.as_optimizer = optim.Adam(self.as_network.parameters(), lr=lr)
        
        self.rl_buffer = ReplayBuffer(capacity=rl_buffer_capacity)
        self.sl_buffer = SLBuffer(capacity=sl_buffer_capacity)
        
        self.step_count = 0
        self.target_update_frequency = agent_config['target_update_frequency']

        # === Opponent Modelling ===
        self.opponent_as_network: Optional[nn.Module] = None
        self.last_opp_state_before_action: Optional[Dict] = None
        self.last_opp_action_index: Optional[int] = None
        
        self.use_average_strategy_this_hand = False

        # === PENDING EXPERIENCE STORE ===
        # We store the state/action from the PREVIOUS turn here.
        # We only push them to the buffer when we reach the NEXT turn (or showdown).
        self.pending_state: Optional[np.ndarray] = None
        self.pending_action: Optional[int] = None

        
    def compute_action(self, state: GameState) -> Tuple[int, Optional[int], Dict, str, 'PokerFeatureSchema']:
        # 1. Extract CURRENT state features (S_t)
        features_schema = self.feature_extractor.extract_features(state, self)
        current_features_vector = features_schema.to_vector()

        # 2. Check if we have a pending experience from the PREVIOUS turn (S_{t-1})
        if self.pending_state is not None and self.pending_action is not None:
            self.rl_buffer.push(
                state=self.pending_state,
                action=self.pending_action,
                reward=0.0,              # Intermediate reward is 0
                next_state=current_features_vector,
                done=False
            )
            
            # Trigger Learning Steps
            self._attempt_learning_step()

        # 3. Select New Action (A_t)
        # Determine policy for this HAND (set in new_hand)
        use_average_strategy = self.use_average_strategy_this_hand
        
        if use_average_strategy:
            action_type, amount, action_index, predictions, is_random = self._get_action_from_network(
                features=current_features_vector, network=self.as_network, state=state, use_greedy=False)
        else:
            action_type, amount, action_index, predictions, is_random = self._get_action_from_network(
                features=current_features_vector, network=self.br_network, state=state, use_greedy=True, epsilon=0.06)
        
        # FIX: Only push to SL buffer if NOT exploring
        if not use_average_strategy and not is_random:
            self.sl_buffer.push(current_features_vector, action_index)
            
        self.pending_state = current_features_vector
        self.pending_action = action_index
        
        policy_name = "AS" if use_average_strategy else "BR"
        return action_type, amount, predictions, policy_name, features_schema
        
    def observe(self, player_action, player_id, state_before_action: GameState, next_state: GameState):
        """
        Passive observation. 
        We NO LONGER use this to record our own trajectory (that happens in compute_action).
        We ONLY use this to update the feature extractor and track opponents.
        """
        action_taken, amount_put_in = player_action
        
        # 1. Update Betting History in Feature Extractor
        self.feature_extractor.update_betting_action(player_id, action_taken, state_before_action, state_before_action.stage)

        # 2. Track Opponent for Intelligent Equity
        if player_id != self.seat_id:
            opp_action_index = self._get_action_index_from_move(state_before_action, action_taken, amount_put_in)
            if opp_action_index is not None:
                self.last_opp_state_before_action = state_before_action
                self.last_opp_action_index = opp_action_index
            
    def observe_showdown(self, showdown_state):
        """
        The hand is over. We must close the loop on the LAST action taken.
        This provides the final reward (S_last -> S_terminal).
        """
        # Calculate the single, final reward.
        my_stack_before = showdown_state.get('stacks_before', {}).get(self.seat_id, 0)
        my_stack_after = showdown_state.get('stacks_after', {}).get(self.seat_id, 0)
        scaling_factor = 200.0 
        raw_reward = my_stack_after - my_stack_before
        final_reward = raw_reward / scaling_factor

        # If we have a pending action that hasn't been closed yet
        if self.pending_state is not None and self.pending_action is not None:
            
            dummy_terminal_state = np.zeros_like(self.pending_state)

            self.rl_buffer.push(
                state=self.pending_state,
                action=self.pending_action,
                reward=final_reward,     # THE REAL REWARD IS HERE
                next_state=dummy_terminal_state,
                done=True                # TERMINAL FLAG
            )
            
            self._attempt_learning_step()

        # Clear pending variables
        self.pending_state = None
        self.pending_action = None
            
    def _attempt_learning_step(self):
        """Helper to check frequency and trigger learning."""
        self.step_count += 1
        if self.step_count % self.update_frequency == 0:
            self.learn_rl()
            self.learn_sl()

    def learn_rl(self):
        """Perform Double DQN learning step for Best Response network."""
        if len(self.rl_buffer) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.rl_buffer.sample(self.batch_size)
        
        # 1. Current Q(s, a)
        current_q_values = self.br_network(states)['q_values']
        current_q_values = torch.gather(current_q_values, 1, actions.unsqueeze(1)).squeeze(1)
        
        # 2. Target = r + gamma * max Q(s', a')
        # Double DQN: Use Main Net to choose action, Target Net to evaluate value
        target_q_values = self.br_network.compute_double_dqn_target(
            next_states, rewards, dones, self.br_target_network, self.gamma
        )
        
        # 3. Loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.br_optimizer.zero_grad()
        loss.backward()
        # Optional: Clip gradients to stabilize training
        torch.nn.utils.clip_grad_norm_(self.br_network.parameters(), 1.0)
        self.br_optimizer.step()
        
        # 4. Sync Target Network
        if self.step_count % self.target_update_frequency == 0:
            self.br_target_network.load_state_dict(self.br_network.state_dict())
        
    def learn_sl(self):
        """Perform supervised learning step for Average Strategy network."""
        if len(self.sl_buffer) < self.batch_size:
            return
            
        states, action_indices = self.sl_buffer.sample(self.batch_size)
        
        predictions = self.as_network(states)
        action_logits = predictions['action_logits']

        loss = nn.CrossEntropyLoss()(action_logits, action_indices)

        self.as_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.as_network.parameters(), 1.0)
        self.as_optimizer.step()
        
    def new_hand(self):
        """Reset for new hand."""
        super().new_hand()
        
        # Reset Pending State (Just in case observe_showdown wasn't called, though it should be)
        self.pending_state = None
        self.pending_action = None

        self.last_opp_state_before_action = None
        self.last_opp_action_index = None
        
        # Decide Policy for the Entire Hand
        self.use_average_strategy_this_hand = random.random() < self.eta

    def _get_current_street_schema(self, schema: PokerFeatureSchema, stage: int) -> StreetFeatures:
        """A helper to get the feature object for the current street."""
        if stage == 1:
            return schema.flop
        elif stage == 2:
            return schema.turn
        elif stage == 3:
            return schema.river
        return schema.preflop
        
    def _get_action_index_from_move(self, state: GameState, action_type: int, amount: Optional[int]) -> Optional[int]:
        """Finds the ACTION_MAP index that best matches a given game action."""
        if action_type == 0: return 0
        if action_type == 1: return 1
        if action_type == 2:
            opp_id = 1 - self.seat_id
            stack = state.stacks[opp_id]
            if stack == 0 or amount is None: return None

            amount_to_call = max(state.current_bets) - state.current_bets[opp_id]
            pot_after_call = state.pot + amount_to_call
            
            if amount == stack: # All-in action
                all_in_index = next((i for i, (_, sizing) in enumerate(ACTION_MAP) if sizing == -1), None)
                return all_in_index

            if pot_after_call <= 0: return None 
            
            raise_amount = (amount - amount_to_call)
            sizing_ratio = raise_amount / pot_after_call

            best_match_index = -1
            min_diff = float('inf')
            for i in range(2, len(ACTION_MAP)):
                _, sizing = ACTION_MAP[i]
                if sizing == -1: continue
                diff = abs(sizing - sizing_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_match_index = i
            return best_match_index
        return None

    def _calculate_intelligent_equity(self, my_hole_cards, community_cards, historical_context=None) -> float:
        # [Preserve your existing logic here, it looked correct]
        trials = self.intelligent_equity_trials

        if self.opponent_as_network is None or self.last_opp_state_before_action is None:
            return 0.5

        opp_seat_id = 1 - self.seat_id
        opp_feature_extractor = FeatureExtractor(
            seat_id=opp_seat_id,
            random_equity_trials=self.feature_extractor.random_equity_trials
        )
        
        if historical_context:
            opp_feature_extractor._betting_history = historical_context['betting_history']
            opp_feature_extractor._action_counts_this_street = historical_context['action_counts']
            opp_feature_extractor.last_aggressor = historical_context['last_aggressor']

        deck = [c for c in range(52) if c not in my_hole_cards and c not in community_cards]
        if len(deck) < 2: return 0.5

        feature_vectors_to_batch = []
        candidate_hands = []
        
        for _ in range(trials):
            opp_hole_candidate = list(np.random.choice(deck, size=2, replace=False))
            
            opp_feature_extractor.new_hand()
            temp_game_state = self.last_opp_state_before_action.copy()
            temp_game_state.hole_cards[opp_seat_id] = opp_hole_candidate

            features_schema = opp_feature_extractor.extract_features(
                temp_game_state, skip_random_equity=True
            )
            feature_vectors_to_batch.append(features_schema.to_vector())
            candidate_hands.append(opp_hole_candidate)

        if not feature_vectors_to_batch:
            return 0.5
            
        batch_tensor = torch.FloatTensor(np.array(feature_vectors_to_batch))

        with torch.no_grad():
            batch_predictions = self.opponent_as_network(batch_tensor)
            
        all_action_probs = batch_predictions['action_probs'].numpy()
        
        weighted_wins = 0.0
        total_weight = 0.0
        
        for i, opp_hole_candidate in enumerate(candidate_hands):
            weight = all_action_probs[i][self.last_opp_action_index]
            
            if weight < 1e-6:
                continue

            current_remaining_deck = [c for c in deck if c not in opp_hole_candidate]
            num_to_deal = 5 - len(community_cards)
            if len(current_remaining_deck) < num_to_deal: continue

            board_runout = np.random.choice(current_remaining_deck, size=num_to_deal, replace=False)
            final_board = community_cards + list(board_runout)

            my_rank = self.feature_extractor.evaluator.best_hand_rank(my_hole_cards, final_board)
            opp_rank = self.feature_extractor.evaluator.best_hand_rank(opp_hole_candidate, final_board)

            outcome = 0.0
            if my_rank > opp_rank: outcome = 1.0
            elif my_rank == opp_rank: outcome = 0.5
            
            weighted_wins += outcome * weight
            total_weight += weight
        
        return weighted_wins / total_weight if total_weight > 0 else 0.5

    # [Add save_models, load_models, save_buffers, load_buffers, etc. here from original code]
    def save_models(self, br_path: str = "nfsp_br.pt", as_path: str = "nfsp_as.pt"):
        torch.save(self.br_network.state_dict(), br_path)
        torch.save(self.as_network.state_dict(), as_path)
        
    def load_models(self, br_path: str = "nfsp_br.pt", as_path: str = "nfsp_as.pt"):
        try:
            self.br_network.load_state_dict(torch.load(br_path, map_location='cpu'))
            self.as_network.load_state_dict(torch.load(as_path, map_location='cpu'))
            self.br_target_network.load_state_dict(self.br_network.state_dict())
            print(f"Loaded NFSP models from {br_path} and {as_path}")
        except Exception as e:
            print(f"Could not load NFSP models: {e}")

    def save_buffers(self, rl_path: str, sl_path: str):
        try:
            with open(rl_path, 'wb') as f:
                pickle.dump(self.rl_buffer, f)
            with open(sl_path, 'wb') as f:
                pickle.dump(self.sl_buffer, f)
        except Exception as e:
            print(f"Error saving buffers: {e}")

    def load_buffers(self, rl_path: str, sl_path: str):
        try:
            if os.path.exists(rl_path) and os.path.exists(sl_path):
                with open(rl_path, 'rb') as f:
                    self.rl_buffer = pickle.load(f)
                with open(sl_path, 'rb') as f:
                    self.sl_buffer = pickle.load(f)
                print(f"Successfully loaded replay buffers.")
            else:
                print("No replay buffer files found, starting with empty buffers.")
        except Exception as e:
            print(f"Error loading buffers: {e}. Starting with empty buffers.")

