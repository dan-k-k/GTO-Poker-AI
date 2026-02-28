# tests/test_components2.py
# python -m unittest tests.test_components2

import unittest
import torch
import numpy as np
import random
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock

from app.nfsp_components import NFSPAgent, SLBuffer
from app.poker_feature_schema import PokerFeatureSchema
from app.poker_agents import ACTION_MAP

MOCK_CONFIG = {
    'agent': {
        'eta': 0.1, 
        'gamma': 0.99, 
        'batch_size': 4,
        'update_frequency': 1, 
        'learning_rate': 0.001,
        'target_update_frequency': 5,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay_steps': 100
    },
    'buffers': {
        'rl_buffer_capacity': 50, 
        'sl_buffer_capacity': 20,
    }
}
FEATURE_VECTOR_SIZE = PokerFeatureSchema.get_vector_size()
NUM_ACTIONS = len(ACTION_MAP)

def _create_dummy_vector():
    return np.random.rand(FEATURE_VECTOR_SIZE).astype(np.float32)

def _create_dummy_mask():
    return np.ones(NUM_ACTIONS, dtype=bool)

# --- TEST SUITES 2 ---
class TestAdvancedComponents(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        
        self.test_dir = tempfile.mkdtemp()
        
        self.agent = NFSPAgent(
            seat_id=0,
            agent_config=MOCK_CONFIG['agent'],
            buffer_config=MOCK_CONFIG['buffers'],
            random_equity_trials=50,
            starting_stack=200
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_persistence_save_and_load(self):
        """Test that models and buffers can be saved and loaded without losing state."""
        br_path = os.path.join(self.test_dir, 'br_net.pt')
        as_path = os.path.join(self.test_dir, 'as_net.pt')
        rl_buf_path = os.path.join(self.test_dir, 'rl_buf.pkl')
        sl_buf_path = os.path.join(self.test_dir, 'sl_buf.pkl')

        # Simulate training by modifying agent state
        self.agent.step_count = 123
        dummy_state = _create_dummy_vector()
        self.agent.rl_buffer.push(dummy_state, 1, 1.0, dummy_state, False, _create_dummy_mask())
        self.agent.sl_buffer.push(dummy_state, 2)
        
        with torch.no_grad():
            for p in self.agent.br_network.parameters():
                p.add_(0.5)

        self.agent.save_models(br_path, as_path)
        self.agent.save_buffers(rl_buf_path, sl_buf_path)
        
        # Fresh agent instance
        new_agent = NFSPAgent(
            seat_id=0,
            agent_config=MOCK_CONFIG['agent'],
            buffer_config=MOCK_CONFIG['buffers'],
            random_equity_trials=50,
            starting_stack=200
        )
        
        self.assertEqual(new_agent.step_count, 0) # Fresh agent is empty/default
        self.assertEqual(len(new_agent.rl_buffer), 0)
        
        new_agent.load_models(br_path, as_path)
        new_agent.load_buffers(rl_buf_path, sl_buf_path)
        
        self.assertEqual(new_agent.step_count, 123) # Matches original
        self.assertEqual(len(new_agent.rl_buffer), 1)
        self.assertEqual(len(new_agent.sl_buffer), 1)
        
        for p1, p2 in zip(self.agent.br_network.parameters(), new_agent.br_network.parameters()):
            self.assertTrue(torch.equal(p1, p2), "BR Network weights failed to load correctly.")

    def test_epsilon_decay(self):
        """Test that epsilon decreases linearly and clamps at epsilon_end."""
        self.agent.step_count = 0
        self.assertEqual(self.agent.get_current_epsilon(), 1.0)
        
        self.agent.step_count = 50
        # Should be roughly 0.55 (halfway between 1.0 and 0.1)
        self.assertAlmostEqual(self.agent.get_current_epsilon(), 0.55)
        
        self.agent.step_count = 100
        self.assertEqual(self.agent.get_current_epsilon(), 0.1)
        
        self.agent.step_count = 500
        self.assertEqual(self.agent.get_current_epsilon(), 0.1)

    def test_target_network_sync_frequency(self):
        """Test that target network updates ONLY when frequency is met."""
        self.agent.step_count = 0
        
        with torch.no_grad():
            for p in self.agent.br_network.parameters():
                p.add_(1.0)
        
        initial_params_match = all(
            torch.equal(p1, p2) 
            for p1, p2 in zip(self.agent.br_network.parameters(), self.agent.br_target_network.parameters())
        )
        self.assertFalse(initial_params_match, "Networks should start different for this test.")

        for i in range(1, 5):
            for _ in range(self.agent.batch_size):
                self.agent.rl_buffer.push(_create_dummy_vector(), 0, 0, _create_dummy_vector(), False, _create_dummy_mask())
            
            self.agent._attempt_learning_step()
            
            params_match = all(
                torch.equal(p1, p2) 
                for p1, p2 in zip(self.agent.br_network.parameters(), self.agent.br_target_network.parameters())
            )
            self.assertFalse(params_match, f"Target network synced prematurely at step {self.agent.step_count}")

        for _ in range(self.agent.batch_size):
            self.agent.rl_buffer.push(_create_dummy_vector(), 0, 0, _create_dummy_vector(), False, _create_dummy_mask())
            
        self.agent._attempt_learning_step() # Will hit step_count = 5
        
        params_match = all(
            torch.equal(p1, p2) 
            for p1, p2 in zip(self.agent.br_network.parameters(), self.agent.br_target_network.parameters())
        )
        self.assertTrue(params_match, "Target network failed to sync at the correct frequency.")

    def test_sl_buffer_reservoir_sampling(self):
        """Test that SLBuffer handles capacity correctly using reservoir sampling logic."""
        capacity = 10
        buffer = SLBuffer(capacity=capacity, input_size=FEATURE_VECTOR_SIZE)
        
        for i in range(capacity):
            buffer.push(_create_dummy_vector(), i)
            
        self.assertEqual(len(buffer), capacity)
        self.assertEqual(buffer.total_count, capacity)
        
        buffer.push(_create_dummy_vector(), 999)
        
        self.assertEqual(len(buffer), capacity, "Buffer size should not exceed capacity.")
        self.assertEqual(buffer.total_count, capacity + 1, "Total count should track total insertions.")
        self.assertEqual(buffer.state_buffer.shape[0], capacity)
        self.assertEqual(buffer.action_buffer.shape[0], capacity)

if __name__ == '__main__':
    unittest.main()

