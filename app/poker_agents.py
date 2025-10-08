# app/poker_agents.py
import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from app.feature_extractor import FeatureExtractor
from app.poker_core import GameState
from app.poker_feature_schema import PokerFeatureSchema

# Action indices map to (Action Type, Bet Size as % of Pot)
# FOLD=0, CALL=1, RAISE=2. For RAISE, -1 signifies All-in.
ACTION_MAP = [
    (0, 0),      # 0: Fold
    (1, 0),      # 1: Call/Check
    (2, 0.20),   # 2: Raise 20% pot
    (2, 0.30),   # 3: Raise 30% pot
    (2, 0.40),   # 4: Raise 40% pot
    (2, 0.50),   # 5: Raise 50% pot
    (2, 0.70),   # 6: Raise 70% pot
    (2, 0.90),   # 7: Raise 90% pot
    (2, 1.10),   # 8: Raise 110% pot
    (2, 1.30),   # 9: Raise 130% pot
    (2, 1.50),   # 10: Raise 150% pot
    (2, -1),     # 11: All-in
]
NUM_ACTIONS = len(ACTION_MAP)

class GTOPokerNet(nn.Module):
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
    
    def compute_double_dqn_target(self, next_states, rewards, dones, target_network, gamma=0.99):
        """
        Compute Double DQN targets to reduce overestimation bias.
        Uses main network to select actions, target network to evaluate them.
        """
        with torch.no_grad():
            # Use main network to select best actions
            next_q_values_main = self.forward(next_states)['q_values']
            next_actions = torch.argmax(next_q_values_main, dim=1)
            
            # Use target network to evaluate the selected actions
            next_q_values_target = target_network.forward(next_states)['q_values']
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute targets: r + γ * Q_target(s', argmax_a Q_main(s', a))
            targets = rewards + gamma * next_q_values * (~dones)
            
        return targets

class PokerAgent:
    """
    Base poker agent class.
    All agents inherit from this.
    """
    def __init__(self, seat_id: int):
        self.seat_id = seat_id
        
    def compute_action(self, state: GameState) -> Tuple[int, Optional[int], Optional[Dict], str, Optional[PokerFeatureSchema]]:
        """
        Compute action for the current state.
        Returns (action, amount) where action ∈ {0: fold, 1: call, 2: raise}
        """
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
    """
    An agent that uses a neural network and a feature extractor to make decisions.
    Base class for neural network-based agents.
    """
    def __init__(self, seat_id: int):
        super().__init__(seat_id)
        # The feature extractor is common to both agents
        self.feature_extractor = FeatureExtractor(seat_id=self.seat_id)

    def new_hand(self):
        """Resets the feature extractor for a new hand."""
        super().new_hand()
        self.feature_extractor.new_hand()

    def _get_action_from_network(self, features: np.ndarray, network: nn.Module, state: GameState) -> Tuple[int, Optional[int], int, Dict]:
        """Get action from a network and return internal predictions."""
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            predictions = network(features_tensor)
            action_probs = predictions['action_probs'][0].numpy()
        
        # 1. Filter to get a mask of legal actions
        legal_action_mask = self._get_legal_action_mask(state)
        if not np.any(legal_action_mask):
            # Fallback if no actions are legal (shouldn't happen in a valid state)
            return 1, None, 1, predictions # Default to call/check with action_index 1
            
        # 2. Apply the mask to the network's probabilities
        filtered_probs = action_probs * legal_action_mask
        
        # 3. Normalize and sample
        prob_sum = np.sum(filtered_probs)
        if prob_sum > 0:
            filtered_probs /= prob_sum
        else:
            # If all legal actions have 0 probability, distribute equally
            filtered_probs = legal_action_mask / np.sum(legal_action_mask)
                
        # Sample the action index (0-11)
        action_index = np.random.choice(NUM_ACTIONS, p=filtered_probs)
        
        # 4. Convert action index to game action (fold/call/raise) and amount
        action_type, sizing = ACTION_MAP[action_index]
        amount = None
        if action_type == 2: # If it's a raise
            amount = self._calculate_bet_amount(sizing, state)
        
        return action_type, amount, action_index, predictions
        
    def _get_legal_action_mask(self, state: GameState) -> np.ndarray:
        """Returns a boolean mask of size NUM_ACTIONS indicating which actions are legal."""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        legal_actions_from_state = state.get_legal_actions()
        min_raise = state.get_min_raise_amount()
        
        for i, (action_type, sizing) in enumerate(ACTION_MAP):
            if action_type in legal_actions_from_state:
                if action_type == 2: # Raise
                    potential_amount = self._calculate_bet_amount(sizing, state)
                    if potential_amount is not None and min_raise is not None and potential_amount >= min_raise:
                        mask[i] = True
                else: # Fold or Call
                    mask[i] = True
        return mask

    def _calculate_bet_amount(self, sizing: float, state: GameState) -> Optional[int]:
        stack = state.stacks[self.seat_id]
        if stack == 0: return None
        if sizing == -1: return stack

        amount_to_call = max(state.current_bets) - state.current_bets[self.seat_id]
        pot_after_call = state.pot + amount_to_call
        raise_amount = int(pot_after_call * sizing)
        total_amount = amount_to_call + raise_amount
        
        min_raise = state.get_min_raise_amount()
        if min_raise is None: return None
        
        return max(min_raise, min(total_amount, stack))


class GTOAgent(NeuralNetworkAgent):
    """
    GTO poker agent that uses trained PyTorch models.
    Layer 1 agent.
    """
    
    def __init__(self, seat_id: int, model_path: str = "gto_average_strategy.pt"):
        super().__init__(seat_id)
        self.model_path = model_path
        
        # Load the trained model
        self.network = GTOPokerNet()
        try:
            self.network.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.network.eval()
            print(f" Loaded GTO model from {model_path}")
        except Exception as e:
            print(f" -Could not load GTO model: {e}")
            print("   Using random initialization")
        
    def compute_action(self, state: GameState) -> Tuple[int, Optional[int], None, str, None]:
        features = self.feature_extractor.extract_features(state).to_vector()
        action, amount, _, _ = self._get_action_from_network(features, self.network, state)
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

