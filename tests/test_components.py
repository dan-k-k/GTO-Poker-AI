# tests/test_components.py
# python -m unittest tests.test_components
import unittest
import torch
import numpy as np
import random
from unittest.mock import patch, MagicMock

from app.feature_extractor import FeatureExtractor
from app.nfsp_components import NFSPAgent, ReplayBuffer, SLBuffer
from app.poker_feature_schema import PokerFeatureSchema
from app.poker_agents import ACTION_MAP
from app.poker_core import GameState

# --- MOCK DATA AND HELPERS ---
MOCK_CONFIG = {
    'agent': {
        'eta': 0.1, 'gamma': 0.99, 'batch_size': 4,
        'update_frequency': 1, 'learning_rate': 0.001,
        'target_update_frequency': 100
    },
    'buffers': {
        'rl_buffer_capacity': 50, 'sl_buffer_capacity': 50,
    }
}
FEATURE_VECTOR_SIZE = PokerFeatureSchema.get_vector_size()
NUM_ACTIONS = len(ACTION_MAP)

def _create_dummy_vector():
    return np.random.rand(FEATURE_VECTOR_SIZE).astype(np.float32)

def _create_dummy_mask():
    return np.ones(NUM_ACTIONS, dtype=bool)

def _create_mock_state(**kwargs) -> GameState:
    defaults = {
        'num_players': 2, 'starting_stack': 200, 'small_blind': 1, 'big_blind': 2,
        'hole_cards': [[0, 1], [2, 3]], 'community': [], 'stacks': [198, 199],
        'current_bets': [2, 1], 'pot': 3, 'stage': 0, 'to_move': 1, 'active': [True, True],
        'all_in': [False, False], 'acted': [False, False], 'surviving_players': [0, 1],
        'dealer_pos': 1, 'sb_pos': 1, 'bb_pos': 0, 'initial_bet': 2, 'last_raise_size': 2,
        'last_raiser': None, 'terminal': False, 'winners': None, 'win_reason': None,
        'starting_pot_this_round': 3, 'starting_stacks_this_hand': [200, 200]
    }
    defaults.update(kwargs)
    valid_fields = GameState.__dataclass_fields__.keys()
    filtered_args = {k: v for k, v in defaults.items() if k in valid_fields}
    return GameState(**filtered_args)


# --- TEST SUITES ---
class TestBuffers(unittest.TestCase):
    def test_rl_buffer_push_and_capacity(self):
        buffer = ReplayBuffer(capacity=3, input_size=FEATURE_VECTOR_SIZE)
        self.assertEqual(len(buffer), 0)
        for i in range(4):
            buffer.push(_create_dummy_vector(), i, float(i), _create_dummy_vector(), False, _create_dummy_mask())
        self.assertEqual(len(buffer), 3)
        
        actions_in_buffer = buffer.actions[:len(buffer)]
        
        self.assertNotIn(0, actions_in_buffer)
        self.assertIn(1, actions_in_buffer)
        self.assertIn(3, actions_in_buffer)

    def test_rl_buffer_sample(self):
        # FIX: Added input_size argument
        buffer = ReplayBuffer(capacity=10, input_size=FEATURE_VECTOR_SIZE)
        batch_size = 4
        for _ in range(5):
            # Added dummy_mask to match new signature
            buffer.push(_create_dummy_vector(), 0, 0.0, _create_dummy_vector(), False, _create_dummy_mask())
        
        # Unpack the 6 return values (mask added)
        states, actions, rewards, next_states, dones, next_masks = buffer.sample(batch_size)
        
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertEqual(states.shape, (batch_size, FEATURE_VECTOR_SIZE))
        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertEqual(next_states.shape, (batch_size, FEATURE_VECTOR_SIZE))
        self.assertEqual(dones.shape, (batch_size,))
        self.assertEqual(next_masks.shape, (batch_size, NUM_ACTIONS))
        self.assertEqual(states.dtype, torch.float32)
        self.assertEqual(actions.dtype, torch.long)
        self.assertEqual(dones.dtype, torch.bool)
        self.assertEqual(next_masks.dtype, torch.bool)

    def test_sl_buffer_logic(self):
        buffer = SLBuffer(capacity=3)
        self.assertEqual(len(buffer), 0)
        for i in range(4):
            buffer.push(_create_dummy_vector(), i)
        self.assertEqual(len(buffer), 3)
        states, actions = buffer.sample(batch_size=2)
        self.assertEqual(states.shape, (2, FEATURE_VECTOR_SIZE))
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(states.dtype, torch.float32)
        self.assertEqual(actions.dtype, torch.long)


class TestNFSPAgent(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        
        self.agent = NFSPAgent(
            seat_id=0,
            agent_config=MOCK_CONFIG['agent'],
            buffer_config=MOCK_CONFIG['buffers'],
            random_equity_trials=50,
            starting_stack=200  # <--- ADDED THIS ARGUMENT
        )

    def test_initialization(self):
        self.assertEqual(self.agent.seat_id, 0)
        self.assertIsNotNone(self.agent.br_network)
        self.assertIsNotNone(self.agent.as_network)
        self.assertEqual(len(self.agent.rl_buffer), 0)
        self.assertEqual(len(self.agent.sl_buffer), 0)

    def test_rl_learn_updates_weights(self):
        initial_params = [p.clone() for p in self.agent.br_network.parameters()]
        for _ in range(MOCK_CONFIG['agent']['batch_size'] + 1):
            # Added mask to push
            self.agent.rl_buffer.push(_create_dummy_vector(), 0, 1.0, _create_dummy_vector(), False, _create_dummy_mask())
        self.agent.learn_rl()
        updated_params = [p.clone() for p in self.agent.br_network.parameters()]
        params_are_different = any(not torch.equal(p_initial, p_updated) for p_initial, p_updated in zip(initial_params, updated_params))
        self.assertTrue(params_are_different, "BR network weights did not change after learning step.")

    def test_sl_learn_updates_weights(self):
        initial_params = [p.clone() for p in self.agent.as_network.parameters()]
        for i in range(MOCK_CONFIG['agent']['batch_size'] + 1):
            self.agent.sl_buffer.push(_create_dummy_vector(), i % len(ACTION_MAP))
        self.agent.learn_sl()
        updated_params = [p.clone() for p in self.agent.as_network.parameters()]
        params_are_different = any(not torch.equal(p_initial, p_updated) for p_initial, p_updated in zip(initial_params, updated_params))
        self.assertTrue(params_are_different, "AS network weights did not change after learning step.")

    def test_learn_does_not_run_if_buffer_too_small(self):
        initial_params = [p.clone() for p in self.agent.br_network.parameters()]
        for _ in range(MOCK_CONFIG['agent']['batch_size'] - 1):
            # Added mask to push
            self.agent.rl_buffer.push(_create_dummy_vector(), 0, 1.0, _create_dummy_vector(), False, _create_dummy_mask())
        self.agent.learn_rl()
        updated_params = [p.clone() for p in self.agent.br_network.parameters()]
        params_are_identical = all(torch.equal(p_initial, p_updated) for p_initial, p_updated in zip(initial_params, updated_params))
        self.assertTrue(params_are_identical, "Network weights changed even though buffer was too small.")

    def test_observe_showdown_populates_buffer_correctly(self):
        """
        Tests that observe_showdown processes the pending experience and pushes 
        an experience tuple with the correct reward and done flag.
        """
        # Set up a pending state (simulating the state at the agent's last action)
        pending_state = _create_dummy_vector()
        pending_action = 1
        self.agent.pending_state = pending_state
        self.agent.pending_action = pending_action
        
        # Simulating showdown data
        showdown_state = {'stacks_before': {0: 150}, 'stacks_after': {0: 200}}

        self.agent.observe_showdown(showdown_state)
        
        self.assertEqual(len(self.agent.rl_buffer), 1, "Should have pushed exactly one experience.")
        
        # FIX: Access internal arrays instead of unpacking .buffer[0]
        s = self.agent.rl_buffer.states[0]
        a = self.agent.rl_buffer.actions[0]
        r = self.agent.rl_buffer.rewards[0]
        done = self.agent.rl_buffer.dones[0]
        
        self.assertTrue(np.array_equal(s, pending_state))
        self.assertEqual(a, pending_action)
        self.assertEqual(r, 0.25, "The final reward should be net winnings (50) scaled by 200 -> 0.25")
        self.assertTrue(done, "The experience should be marked as terminal (done=True).")
        self.assertIsNone(self.agent.pending_state, "Pending state should be cleared after showdown")

    def test_observe_showdown_after_opponent_folds(self):
        """
        Tests that a correct final reward is assigned when the hand ends due to a fold.
        """
        # Set up a pending state
        pending_state = _create_dummy_vector()
        pending_action = 2
        self.agent.pending_state = pending_state
        self.agent.pending_action = pending_action

        showdown_state = {
            'stacks_before': {0: 180},
            'stacks_after': {0: 210}
        }
        # Profit 30 / Scale 200 = 0.15
        expected_reward = 0.15

        self.agent.observe_showdown(showdown_state)

        self.assertEqual(len(self.agent.rl_buffer), 1, "Should have pushed the final hand experience.")

        # FIX: Access internal arrays instead of unpacking .buffer[0]
        s = self.agent.rl_buffer.states[0]
        a = self.agent.rl_buffer.actions[0]
        r = self.agent.rl_buffer.rewards[0]
        done = self.agent.rl_buffer.dones[0]
        
        self.assertTrue(np.array_equal(s, pending_state), "State vector mismatch.")
        self.assertEqual(a, pending_action, "Action index mismatch.")
        self.assertAlmostEqual(r, expected_reward, places=5, msg="Final reward mismatch.")
        self.assertTrue(done, "The experience should be marked as terminal (done=True).")

    # NOTE: Intelligent Equity tests were removed because the provided NFSPAgent
    # code does not contain the `_calculate_intelligent_equity` method or logic.
    
    def test_new_hand_resets_state(self):
        """
        Tests that new_hand() correctly resets the agent's per-hand state.
        """
        # Manually set pending state
        self.agent.pending_state = _create_dummy_vector()
        self.agent.pending_action = 2
        
        self.agent.new_hand()

        self.assertIsNone(self.agent.pending_state, "pending_state should be reset.")
        self.assertIsNone(self.agent.pending_action, "pending_action should be reset.")

    def test_compute_action_stores_pending_experience(self):
        """
        Tests that compute_action stores the current features as pending_state
        to be used in the NEXT step's transition calculation.
        """
        mock_state = _create_mock_state()
        self.assertIsNone(self.agent.pending_state)

        # Mock the network response to avoid actual inference
        with patch.object(self.agent, '_get_action_from_network') as mock_net_act:
            mock_net_act.return_value = ('call', 0, 1, {}, False) # action_index = 1
            
            # Call compute action
            self.agent.compute_action(mock_state)

        # Verify pending state was populated
        self.assertIsNotNone(self.agent.pending_state)
        self.assertEqual(self.agent.pending_action, 1)
        self.assertEqual(self.agent.pending_state.shape, (FEATURE_VECTOR_SIZE,))

    def test_compute_action_uses_eta_policy(self):
        """
        Tests that compute_action() correctly uses eta to select between
        the Best Response (BR) and Average Strategy (AS) networks.
        """
        mock_state = _create_mock_state()
        
        # --- Case 1: Force Average Strategy (eta = 1.0) ---
        self.agent.eta = 1.0
        # The policy is decided in new_hand(). 
        self.agent.new_hand() 
        
        with patch.object(self.agent, 'as_network', wraps=self.agent.as_network) as spy_as_net, \
             patch.object(self.agent, 'br_network', wraps=self.agent.br_network) as spy_br_net:
            
            self.agent.compute_action(mock_state)
            
            spy_as_net.assert_called_once()
            spy_br_net.assert_not_called()

        # --- Case 2: Force Best Response (eta = 0.0) ---
        self.agent.eta = 0.0
        # Call new_hand again to apply the new eta
        self.agent.new_hand()
        
        with patch.object(self.agent, 'as_network', wraps=self.agent.as_network) as spy_as_net, \
             patch.object(self.agent, 'br_network', wraps=self.agent.br_network) as spy_br_net:

            self.agent.compute_action(mock_state)

            spy_br_net.assert_called_once()
            spy_as_net.assert_not_called()

if __name__ == '__main__':
    unittest.main()

