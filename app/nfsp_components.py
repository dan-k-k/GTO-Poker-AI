# app/nfsp_components.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle 
import os
import random
from typing import Dict, List, Tuple, Optional
import shutil

from app.poker_agents import NeuralNetworkAgent, BRNet, ASNet, ACTION_MAP, NUM_ACTIONS
from app.poker_feature_schema import PokerFeatureSchema, StreetFeatures
from app.poker_core import GameState
from app.feature_extractor import FeatureExtractor

class ReplayBuffer:
    """RL Replay Buffer for Best Response Policy (DQN-style)."""
    
    def __init__(self, capacity: int, input_size: int, action_dim: int = NUM_ACTIONS):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.input_size = input_size
        self.action_dim = action_dim 
        self._init_buffers()

    def _init_buffers(self):
        self.states = np.zeros((self.capacity, self.input_size), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.input_size), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64) 
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)
        self.masks = np.zeros((self.capacity, self.action_dim), dtype=np.bool_)
        
    def push(self, state, action, reward, next_state, done, next_legal_mask):
        """Stores experience in the pre-allocated arrays."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.masks[self.ptr] = next_legal_mask

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Returns tensors directly to match the original API."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (torch.as_tensor(self.states[idx]), torch.as_tensor(self.actions[idx]), torch.as_tensor(self.rewards[idx]), torch.as_tensor(self.next_states[idx]), torch.as_tensor(self.dones[idx]), torch.as_tensor(self.masks[idx]))

    def __len__(self):
        return self.size
    
    def __getstate__(self):
        """Only save the valid parts of the arrays to save disk space."""
        # Optimization: Only slice if not full.
        if self.size < self.capacity:
            return {'capacity': self.capacity, 'input_size': self.input_size, 'action_dim': self.action_dim, 'ptr': self.ptr, 'size': self.size, 'states': self.states[:self.size], 'next_states': self.next_states[:self.size], 'actions': self.actions[:self.size], 'rewards': self.rewards[:self.size], 'dones': self.dones[:self.size], 'masks': self.masks[:self.size]}
        else:
            return self.__dict__

    def __setstate__(self, state):
        self.capacity = state['capacity']
        self.input_size = state['input_size']
        self.action_dim = state.get('action_dim', NUM_ACTIONS) # Backwards compat
        self.ptr = state['ptr']
        self.size = state['size']
        
        self._init_buffers()        
        load_size = state['states'].shape[0]
        
        self.states[:load_size] = state['states']
        self.next_states[:load_size] = state['next_states']
        self.actions[:load_size] = state['actions']
        self.rewards[:load_size] = state['rewards']
        self.dones[:load_size] = state['dones']
        self.masks[:load_size] = state['masks']

class SLBuffer:
    """Reservoir Buffer for Average Strategy (Optimized & Backward Compatible)."""
    
    def __init__(self, capacity: int, input_size: int = None): # Default None for pickle safety
        self.capacity = capacity
        self.input_size = input_size
        
        # We will initialize these if input_size is provided, otherwise they stay None
        # until the first push or load
        self.state_buffer = None
        self.action_buffer = None
        
        if input_size is not None:
            self._init_buffers(input_size)
            
        self.size = 0        
        self.total_count = 0 
        
    def _init_buffers(self, input_size):
        """Helper to allocate the numpy arrays."""
        self.input_size = input_size
        self.state_buffer = np.zeros((self.capacity, input_size), dtype=np.float32)
        self.action_buffer = np.zeros(self.capacity, dtype=np.int64)

    def push(self, state: np.ndarray, action_index: int):
        """Store state-action_index pair using Reservoir Sampling."""
        
        # Lazy initialization if this is the first push ever
        if self.state_buffer is None:
            self._init_buffers(state.shape[0])

        if self.size < self.capacity:
            self.state_buffer[self.size] = state
            self.action_buffer[self.size] = action_index
            self.size += 1
        else:
            r = random.randint(0, self.total_count)
            if r < self.capacity:
                self.state_buffer[r] = state
                self.action_buffer[r] = action_index
        
        self.total_count += 1
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        real_batch_size = min(self.size, batch_size)
        indices = np.random.randint(0, self.size, size=real_batch_size)
        
        # Return Tensors directly from numpy arrays (Fast!)
        return torch.from_numpy(self.state_buffer[indices]), torch.from_numpy(self.action_buffer[indices])
        
    def __len__(self):
        return self.size

    # Methods to load old buffers
    def __getstate__(self):
        """Called when saving: only save the efficient numpy arrays."""
        return {'capacity': self.capacity,'input_size': self.input_size,'state_buffer': self.state_buffer,'action_buffer': self.action_buffer,'size': self.size,'total_count': self.total_count}

    def __setstate__(self, state):
        """Called when loading: handle both old (list) and new (dict) formats."""
        
        # Loading a new buffer
        if isinstance(state, dict):
            self.__dict__.update(state)
            
        # Loading an old buffer
        elif 'buffer' in state:
            print("  Converting legacy SLBuffer to optimized format...")
            self.capacity = state['capacity']
            self.total_count = state['total_count']
            old_buffer_list = state['buffer']
            self.size = len(old_buffer_list)
            
            # Determine input size from the first element
            if self.size > 0:
                first_state = old_buffer_list[0][0]
                self.input_size = first_state.shape[0]
                
                # Allocate arrays
                self._init_buffers(self.input_size)
                
                # Fill arrays from the list
                for i, (s, a) in enumerate(old_buffer_list):
                    self.state_buffer[i] = s
                    self.action_buffer[i] = a
            else:
                self.input_size = None
                self.state_buffer = None
                self.action_buffer = None

class NFSPAgent(NeuralNetworkAgent):
    """Neural Fictitious Self-Play Agent."""
    
    def __init__(self, seat_id: int, agent_config: Dict, buffer_config: Dict, random_equity_trials: int, starting_stack: int):
        super().__init__(seat_id)
        self.starting_stack = float(starting_stack)
        self.mode = 'train'
        self.feature_extractor = FeatureExtractor(
            seat_id=self.seat_id,
            random_equity_trials=random_equity_trials
        )
        
        self.eta = agent_config['eta']
        self.gamma = agent_config['gamma']
        self.batch_size = agent_config['batch_size']
        self.update_frequency = agent_config['update_frequency']

        self.epsilon_start = agent_config.get('epsilon_start', 0.06)
        self.epsilon_end = agent_config.get('epsilon_end', 0.01)
        self.epsilon_decay_steps = agent_config.get('epsilon_decay_steps', 100000)

        lr = agent_config['learning_rate']
        
        input_size = PokerFeatureSchema.get_vector_size()
        self.br_network = BRNet(input_size=input_size)
        self.br_target_network = BRNet(input_size=input_size)
        self.br_target_network.load_state_dict(self.br_network.state_dict())
        self.br_target_network.eval()
        self.as_network = ASNet(input_size=input_size)
        
        self.br_optimizer = optim.Adam(self.br_network.parameters(), lr=lr)
        self.as_optimizer = optim.Adam(self.as_network.parameters(), lr=lr)
        
        self.target_rl_cap = buffer_config['rl_buffer_capacity']
        self.target_sl_cap = buffer_config['sl_buffer_capacity']
        self.rl_buffer = ReplayBuffer(capacity=self.target_rl_cap, input_size=input_size, action_dim=NUM_ACTIONS)
        self.sl_buffer = SLBuffer(capacity=self.target_sl_cap, input_size=input_size)
        
        self.step_count = 0
        self.target_update_frequency = agent_config['target_update_frequency']
        
        self.use_average_strategy_this_hand = False

        # === EXPERIENCE STORE ===
        self.pending_state: Optional[np.ndarray] = None
        self.pending_action: Optional[int] = None

    def compute_action(self, state: GameState) -> Tuple[int, Optional[int], Dict, str, 'PokerFeatureSchema']:
        # 1. Extract current state features (S_t)
        features_schema = self.feature_extractor.extract_features(state)
        current_features_vector = features_schema.to_vector()
        current_legal_mask = self._get_legal_action_mask(state)

        # 2. Check if we have a pending experience from the previous turn (S_{t-1})
        if self.mode == 'train' and self.pending_state is not None:
            self.rl_buffer.push(
                state=self.pending_state,
                action=self.pending_action,
                reward=0.0,              # Intermediate reward is 0
                next_state=current_features_vector,
                done=False,
                next_legal_mask=current_legal_mask
            )
            self._attempt_learning_step()

        # 3. Select New Action (A_t)
        use_average_strategy = self.use_average_strategy_this_hand
        
        if use_average_strategy:
            action_type, amount, action_index, predictions, is_random = self._get_action_from_network(
                features=current_features_vector, network=self.as_network, state=state, use_greedy=False)
        else:
            current_epsilon = self.get_current_epsilon()
            action_type, amount, action_index, predictions, is_random = self._get_action_from_network(
                features=current_features_vector, network=self.br_network, state=state, use_greedy=True, epsilon=current_epsilon)
        
        # Only push to SL buffer if NOT exploring
        if not use_average_strategy and not is_random:
            self.sl_buffer.push(current_features_vector, action_index)
            
        self.pending_state = current_features_vector
        self.pending_action = action_index
        
        policy_name = "AS" if use_average_strategy else "BR"
        return action_type, amount, predictions, policy_name, features_schema
        
    def observe(self, player_action, player_id, state_before_action: GameState, next_state: GameState):
        """Passive observation. Only use this to update the feature extractor."""
        action_taken, amount_put_in = player_action
        
        # 1. Update Betting History in Feature Extractor
        self.feature_extractor.update_betting_action(player_id, action_taken, state_before_action, state_before_action.stage)
            
    def observe_showdown(self, showdown_state):
        """The hand is over. We must close the loop on the LAST action taken. This provides the final reward (S_last -> S_terminal)."""
        # Calculate the single, final reward.
        my_stack_before = showdown_state.get('stacks_before', {}).get(self.seat_id, 0)
        my_stack_after = showdown_state.get('stacks_after', {}).get(self.seat_id, 0)
        # Dynamic scaling using the stored starting_stack
        raw_reward = my_stack_after - my_stack_before
        final_reward = raw_reward / self.starting_stack

        # If we have a pending action that hasn't been closed yet
        if self.pending_state is not None and self.pending_action is not None:
            
            dummy_terminal_state = np.zeros_like(self.pending_state)

            self.rl_buffer.push(
                state=self.pending_state,
                action=self.pending_action,
                reward=final_reward,     # THE REAL REWARD IS HERE
                next_state=dummy_terminal_state,
                done=True,                # TERMINAL FLAG
                next_legal_mask=np.ones(NUM_ACTIONS, dtype=bool)
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
            
        states, actions, rewards, next_states, dones, next_legal_masks = self.rl_buffer.sample(self.batch_size)
        
        # 1. Current Q(s, a)
        current_q_values = self.br_network(states)['q_values']
        current_q_values = torch.gather(current_q_values, 1, actions.unsqueeze(1)).squeeze(1)
        
        # 2. Target = r + gamma * max Q(s', a')
        # Double DQN: Use Main Net to choose action, Target Net to evaluate value
        target_q_values = self.br_network.compute_double_dqn_target(next_states, rewards, dones, self.br_target_network, self.gamma, next_legal_masks)
        
        # 3. Loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.br_optimizer.zero_grad()
        loss.backward()
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
    
    def get_current_epsilon(self) -> float:
        """Calculates linear decay based on current step_count."""
        if self.step_count >= self.epsilon_decay_steps:
            return self.epsilon_end
        # Linear
        slope = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        return self.epsilon_start - (slope * self.step_count)
    
    def set_mode(self, mode: str): self.mode = mode # 'train' or 'eval'

    def new_hand(self):
        super().new_hand()
        self.pending_state = None
        self.pending_action = None
        
        if hasattr(self, 'mode') and self.mode == 'eval':
            self.use_average_strategy_this_hand = True # ALWAYS use AS in eval
        else:
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
    
    def save_models(self, br_path: str, as_path: str):
        torch.save({'model_state_dict': self.br_network.state_dict(), 'target_model_state_dict': self.br_target_network.state_dict(), 'optimizer_state_dict': self.br_optimizer.state_dict(),'step_count': self.step_count}, br_path)
        torch.save({'model_state_dict': self.as_network.state_dict(), 'optimizer_state_dict': self.as_optimizer.state_dict()}, as_path)
            
    def load_models(self, br_path: str, as_path: str):
        if not os.path.exists(br_path):
            raise FileNotFoundError(f"CRITICAL: Model file not found at {br_path}")
        try:
            checkpoint_br = torch.load(br_path, map_location='cpu')
            self.br_network.load_state_dict(checkpoint_br['model_state_dict'])
            self.br_optimizer.load_state_dict(checkpoint_br['optimizer_state_dict'])
            self.step_count = checkpoint_br.get('step_count', 0)
            
            if 'target_model_state_dict' in checkpoint_br:
                self.br_target_network.load_state_dict(checkpoint_br['target_model_state_dict'])
            else:
                print("Warning: No target network state found. Syncing with main network (May cause instability).")
                self.br_target_network.load_state_dict(self.br_network.state_dict())
            
            checkpoint_as = torch.load(as_path, map_location='cpu')
            self.as_network.load_state_dict(checkpoint_as['model_state_dict'])
            self.as_optimizer.load_state_dict(checkpoint_as['optimizer_state_dict'])
            
            print(f"Loaded NFSP models from {br_path}")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Failed to load models. File corrupt or architecture mismatch. Error: {e}")

    def save_buffers(self, rl_path: str, sl_path: str):
        """Saves buffers atomically to prevent corruption on interrupt."""
        def save_safe(obj, final_path):
            temp_path = final_path + ".tmp"
            try:
                with open(temp_path, 'wb') as f:
                    pickle.dump(obj, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, final_path)
            except Exception as e:
                print(f"Error saving buffer to {final_path}: {e}")
                if os.path.exists(temp_path): os.remove(temp_path)

        save_safe(self.rl_buffer, rl_path)
        save_safe(self.sl_buffer, sl_path)

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

