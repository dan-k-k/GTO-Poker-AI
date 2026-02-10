# app/poker_agents.py
import torch
import torch.nn as nn
import numpy as np
import random
import os
from typing import Dict, List, Tuple, Optional
from app.feature_extractor import FeatureExtractor
from app.poker_core import GameState
from app.poker_feature_schema import PokerFeatureSchema

# Action indices map to (Action Type, Bet Size as % of Pot)
# FOLD=0, CALL=1, RAISE=2. For RAISE, -1 signifies All-in.
ACTION_MAP = [
    (0, 0),      # Fold
    (1, 0),      # Call
    (2, 0.50),   # Raise Small (50% Pot)
    (2, 1.00),   # Raise Pot
    (2, -1),     # All-in
]
NUM_ACTIONS = len(ACTION_MAP)

class BRNet(nn.Module):
    """
    Dueling Double DQN poker network architecture.
    Combines Dueling DQN (separate value/advantage streams) with Double DQN (action selection/evaluation).
    Used by both training and inference.
    """
    def __init__(self, input_size: int = None):  # Dynamic input size from schema
        super().__init__()
        
        # Get input size dynamically from schema if not provided
        if input_size is None:
            from app.poker_feature_schema import PokerFeatureSchema
            input_size = PokerFeatureSchema.get_vector_size()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Output heads for Dueling DQN
        self.advantage_head = nn.Linear(32, NUM_ACTIONS)  # Action advantages A(s,a)
        self.value_head = nn.Linear(32, 1)  # State value V(s)
        
        # Initialize heads
        nn.init.constant_(self.value_head.bias, 0.0)
        nn.init.normal_(self.value_head.weight, 0.0, 0.1)
        nn.init.normal_(self.advantage_head.weight, 0.0, 0.1)
        
    def forward(self, x):
        shared_out = self.shared(x)
        
        # Get the value and advantage streams
        state_values = self.value_head(shared_out)
        advantages = self.advantage_head(shared_out)
        
        # Combine them to get the Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = state_values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return {
            'action_logits': q_values,  # Q-values serve as action logits
            'action_probs': torch.softmax(q_values, dim=-1),
            'state_values': state_values,  # Return the raw state values
            'q_values': q_values  # Explicit Q-values for Double DQN
        }
    
    def compute_double_dqn_target(self, next_states, rewards, dones, target_network, gamma, next_legal_masks):
        with torch.no_grad():
            # 1. Main Net selection (Select Best LEGAL Action)
            next_q_values_main = self.forward(next_states)['q_values']
            
            # Set illegal actions to negative infinity
            masked_q_main = next_q_values_main.clone()
            masked_q_main[~next_legal_masks] = -float('inf')
            
            next_actions = torch.argmax(masked_q_main, dim=1)
            
            # 2. Target Net evaluation
            next_q_values_target = target_network.forward(next_states)['q_values']
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            targets = rewards + gamma * next_q_values * (1.0 - dones.float())
            
        return targets

class ASNet(nn.Module):
    """Standard Feed-Forward Network for Classification.
    Used for the Average Strategy (Supervised Learning)."""
    def __init__(self, input_size: int = None):
        super().__init__()
        
        if input_size is None:
            from app.poker_feature_schema import PokerFeatureSchema
            input_size = PokerFeatureSchema.get_vector_size()
            
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS) # Direct output to action logits
        )
        
    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        
        return {'action_logits': logits,   # Used for CrossEntropyLoss
                'action_probs': probs,     # Used for sampling actions
                'q_values': logits}
    
class PokerAgent:
    """Base poker agent class. All agents inherit from this."""

    def __init__(self, seat_id: int):
        self.seat_id = seat_id
        
    def compute_action(self, state: GameState) -> Tuple[int, Optional[int], Optional[Dict], str, Optional[PokerFeatureSchema]]:
        """Compute action for the current state.
        Returns (action, amount) where action ∈ {0: fold, 1: call, 2: raise}"""
        raise NotImplementedError
        
    def new_hand(self):
        """Reset for new hand."""
        pass
        
    def observe(self, player_action, player_id, state_before_action: GameState, next_state: GameState):
        """Observe an opponent's action."""
        pass
        
    def observe_showdown(self, showdown_state: Dict):
        """Observe showdown results."""
        pass


class NeuralNetworkAgent(PokerAgent):
    """An agent that uses a neural network and a feature extractor to make decisions.
    Base class for neural network-based agents."""
    def __init__(self, seat_id: int):
        super().__init__(seat_id)
        # The feature extractor is common to both agents
        self.feature_extractor = FeatureExtractor(seat_id=self.seat_id)

    def new_hand(self):
        """Resets the feature extractor for a new hand."""
        super().new_hand()
        self.feature_extractor.new_hand()

    def _get_action_from_network(self, features: np.ndarray, network: nn.Module, state: GameState, use_greedy: bool = False, epsilon: float = 0.0) -> Tuple[int, Optional[int], int, Dict, bool]:
        """Returns: (action_type, amount, action_index, predictions, is_exploring_flag)"""
        network.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            predictions = network(features_tensor)
            q_values = predictions['q_values'][0].numpy()
            action_probs = predictions['action_probs'][0].numpy()
        network.train()
        
        p = action_probs + 1e-9
        predictions['entropy'] = float(-np.sum(p * np.log(p)))

        # 1. Get Mask of Legal Actions
        legal_action_mask = self._get_legal_action_mask(state)
        
        if not np.any(legal_action_mask):
            raise RuntimeError("CRITICAL ERROR: No legal actions found for Player {self.seat_id}.")

        action_index = 0
        is_random_exploration = False # NEW FLAG

        # === PATH A: Best Response (Greedy + Epsilon) ===
        if use_greedy:
            if random.random() < epsilon:
                # --- FIX STARTS HERE ---
                # Don't choose uniformly from all indices. 
                # Categorize: 0=Fold, 1=Call, 2+=Raise
                
                legal_indices = np.where(legal_action_mask)[0]
                
                can_fold = 0 in legal_indices
                can_call = 1 in legal_indices
                raise_indices = [i for i in legal_indices if i >= 2]
                
                # Define probabilities for categories: [Fold, Call, Raise]
                # Adjust these to encourage "normal" poker behavior during exploration
                category_probs = []
                available_categories = []
                
                if can_fold: 
                    available_categories.append('fold')
                    category_probs.append(0.2) # 20% chance to fold
                if can_call: 
                    available_categories.append('call')
                    category_probs.append(0.4) # 40% chance to call
                if raise_indices: 
                    available_categories.append('raise')
                    category_probs.append(0.4) # 40% chance to raise (split among sizes)
                    
                # Normalize probabilities
                total_p = sum(category_probs)
                category_probs = [p / total_p for p in category_probs]
                
                chosen_cat = np.random.choice(available_categories, p=category_probs)
                
                if chosen_cat == 'fold':
                    action_index = 0
                elif chosen_cat == 'call':
                    action_index = 1
                else:
                    action_index = np.random.choice(raise_indices) # Uniform among raise sizes
                
                is_random_exploration = True 
                # --- FIX ENDS HERE ---
                
            else:
                # (Standard Argmax Logic)
                masked_q_values = q_values.copy()
                masked_q_values[~legal_action_mask] = -float('inf')
                action_index = np.argmax(masked_q_values)

        # === PATH B: Average Strategy (Softmax Sampling) ===
        else:
            filtered_probs = action_probs * legal_action_mask
            prob_sum = np.sum(filtered_probs)
            
            if prob_sum > 0:
                filtered_probs /= prob_sum
            else:
                # If network predicts 0 probability for all legal moves, uniform sample legal ones
                filtered_probs = legal_action_mask.astype(float) / np.sum(legal_action_mask)
                
            action_index = np.random.choice(NUM_ACTIONS, p=filtered_probs)
        
        # 3. Convert to game action
        action_type, sizing = ACTION_MAP[action_index]
        amount = None
        
        if action_type == 2:
            amount = self._calculate_bet_amount(sizing, state)
            min_raise = state.get_min_raise_amount()
            current_stack = state.stacks[self.seat_id]
            is_all_in = (amount == current_stack)
            if min_raise is not None and amount < min_raise and not is_all_in:
                raise RuntimeError(f"Agent attempted illegal raise size! Calc: {amount}, Min: {min_raise}, Stack: {current_stack}")
        
        return action_type, amount, action_index, predictions, is_random_exploration
        
    def _get_legal_action_mask(self, state: GameState) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        legal_actions_generic = state.get_legal_actions() # [0, 1, 2]
        
        if 0 in legal_actions_generic: mask[0] = True
        if 1 in legal_actions_generic: mask[1] = True
        if 2 not in legal_actions_generic: return mask

        min_raise = state.get_min_raise_amount()
        player_stack = state.stacks[self.seat_id]
        for i in range(2, NUM_ACTIONS):
            _, sizing = ACTION_MAP[i]
            
            if sizing == -1: 
                mask[i] = True
                continue

            amount_to_call = max(state.current_bets) - state.current_bets[self.seat_id]
            pot_after_call = state.pot + amount_to_call
            raise_amount = int(pot_after_call * sizing)
            total_bet = amount_to_call + raise_amount
            
            if total_bet >= player_stack: mask[i] = False 
            elif min_raise is not None and total_bet < min_raise: mask[i] = False
            else: mask[i] = True
        return mask

    def _calculate_bet_amount(self, sizing: float, state: GameState) -> int:
        stack = state.stacks[self.seat_id]
        if stack == 0: return 0
        if sizing == -1: return stack

        amount_to_call = max(state.current_bets) - state.current_bets[self.seat_id]
        pot_after_call = state.pot + amount_to_call
        raise_amount = int(pot_after_call * sizing)
        total_amount = amount_to_call + raise_amount
        
        min_raise = state.get_min_raise_amount()
        if total_amount >= stack: return stack
        if min_raise is not None and total_amount < min_raise: return min_raise
        return total_amount


class GTOAgent(NeuralNetworkAgent):
    """
    GTO poker agent that uses trained PyTorch models.
    Layer 1 agent.
    """
    
    def __init__(self, seat_id: int, model_path: str = "gto_average_strategy.pt"):
        super().__init__(seat_id)
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"CRITICAL: GTO Model not found at {model_path}. Cannot proceed.")
        self.network = ASNet()
        try:
            self.network.load_state_dict(torch.load(model_path))
        except RuntimeError as e:
            raise RuntimeError(f"CRITICAL: Architecture mismatch for GTO Agent. {e}")
        
    def compute_action(self, state: GameState) -> Tuple[int, Optional[int], None, str, None]:
        features = self.feature_extractor.extract_features(state).to_vector()
        action, amount, _, _, _ = self._get_action_from_network(features, self.network, state)
        return action, amount, None, "GTO", None


class RandomBot(PokerAgent):
    """
    A simple poker agent that makes random legal actions.
    Useful for testing and as a baseline opponent.
    """
    def __init__(self, seat_id: int, aggression: float = 0.3):

        super().__init__(seat_id)
        self.aggression = aggression

    def compute_action(self, state: GameState, env=None) -> Tuple[int, Optional[int], None, str, None]: 
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return 1, None, None, "Random", None # Fallback to call/check

        # Try to be aggressive first
        if 2 in legal_actions and random.random() < self.aggression:
            min_raise = state.get_min_raise_amount() 
            max_raise = state.stacks[self.seat_id]

            if min_raise and min_raise <= max_raise:
                # If min_raise is the only option, just do that (all-in)
                amount = max_raise if min_raise >= max_raise else random.randint(min_raise, max_raise)
                return 2, amount, None, "Random", None

        # If not raising, choose a non-aggressive action
        fallback_actions = [action for action in legal_actions if action != 2]
        if fallback_actions:
            action = random.choice(fallback_actions)
            return action, None, None, "Random", None
        
        # If only raising was legal but chose not to, bot must go all-in
        return 2, state.stacks[self.seat_id], None, "Random", None

# Backward compatibility alias
NeuralPokerAgent = GTOAgent 

