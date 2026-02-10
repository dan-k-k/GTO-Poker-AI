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

# --- MOCK DATA AND HELPERS ---
MOCK_CONFIG = {
    'agent': {
        'eta': 0.1, 
        'gamma': 0.99, 
        'batch_size': 4,
        'update_frequency': 1, 
        'learning_rate': 0.001,
        'target_update_frequency': 5,  # Reduced for easier testing
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

# --- ADVANCED TEST SUITE ---
class TestAdvancedComponents(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        
        # Create a temporary directory for file I/O tests
        self.test_dir = tempfile.mkdtemp()
        
        self.agent = NFSPAgent(
            seat_id=0,
            agent_config=MOCK_CONFIG['agent'],
            buffer_config=MOCK_CONFIG['buffers'],
            random_equity_trials=50,
            starting_stack=200
        )

    def tearDown(self):
        # Cleanup temp directory after each test
        shutil.rmtree(self.test_dir)

    # ----------------------------------------------------------------
    # 1. PERSISTENCE TESTS (Save/Load)
    # ----------------------------------------------------------------
    def test_persistence_save_and_load(self):
        """Test that models and buffers can be saved and loaded without losing state."""
        br_path = os.path.join(self.test_dir, 'br_net.pt')
        as_path = os.path.join(self.test_dir, 'as_net.pt')
        rl_buf_path = os.path.join(self.test_dir, 'rl_buf.pkl')
        sl_buf_path = os.path.join(self.test_dir, 'sl_buf.pkl')

        # Modify agent state (simulate training)
        self.agent.step_count = 123
        
        # Add data to buffers
        dummy_state = _create_dummy_vector()
        self.agent.rl_buffer.push(dummy_state, 1, 1.0, dummy_state, False, _create_dummy_mask())
        self.agent.sl_buffer.push(dummy_state, 2)
        
        # Modify network weights slightly to ensure we aren't loading defaults
        with torch.no_grad():
            for p in self.agent.br_network.parameters():
                p.add_(0.5)

        # Save everything
        self.agent.save_models(br_path, as_path)
        self.agent.save_buffers(rl_buf_path, sl_buf_path)
        
        # Create a fresh agent instance
        new_agent = NFSPAgent(
            seat_id=0,
            agent_config=MOCK_CONFIG['agent'],
            buffer_config=MOCK_CONFIG['buffers'],
            random_equity_trials=50,
            starting_stack=200
        )
        
        # Verify fresh agent is empty/default
        self.assertEqual(new_agent.step_count, 0)
        self.assertEqual(len(new_agent.rl_buffer), 0)
        
        # Load saved state
        new_agent.load_models(br_path, as_path)
        new_agent.load_buffers(rl_buf_path, sl_buf_path)
        
        # Verify loaded state matches original
        self.assertEqual(new_agent.step_count, 123)
        self.assertEqual(len(new_agent.rl_buffer), 1)
        self.assertEqual(len(new_agent.sl_buffer), 1)
        
        # Verify weights match
        for p1, p2 in zip(self.agent.br_network.parameters(), new_agent.br_network.parameters()):
            self.assertTrue(torch.equal(p1, p2), "BR Network weights failed to load correctly.")

    # ----------------------------------------------------------------
    # 2. EPSILON DECAY TEST
    # ----------------------------------------------------------------
    def test_epsilon_decay(self):
        """Test that epsilon decreases linearly and clamps at epsilon_end."""
        # Config: start=1.0, end=0.1, steps=100
        self.agent.step_count = 0
        self.assertEqual(self.agent.get_current_epsilon(), 1.0)
        
        # Halfway point
        self.agent.step_count = 50
        # Should be roughly 0.55 (halfway between 1.0 and 0.1)
        self.assertAlmostEqual(self.agent.get_current_epsilon(), 0.55)
        
        # End point
        self.agent.step_count = 100
        self.assertEqual(self.agent.get_current_epsilon(), 0.1)
        
        # Past end point (should stay at min)
        self.agent.step_count = 500
        self.assertEqual(self.agent.get_current_epsilon(), 0.1)

    # ----------------------------------------------------------------
    # 3. TARGET NETWORK SYNC TEST
    # ----------------------------------------------------------------
    def test_target_network_sync_frequency(self):
        """Test that target network updates ONLY when frequency is met."""
        # target_update_frequency is set to 5 in MOCK_CONFIG
        self.agent.step_count = 0
        
        # Force main and target networks to diverge
        with torch.no_grad():
            for p in self.agent.br_network.parameters():
                p.add_(1.0)
        
        # Verify they are different
        initial_params_match = all(
            torch.equal(p1, p2) 
            for p1, p2 in zip(self.agent.br_network.parameters(), self.agent.br_target_network.parameters())
        )
        self.assertFalse(initial_params_match, "Networks should start different for this test.")

        # -- Steps 1 to 4: Should NOT Sync --
        for i in range(1, 5):
            # We must populate buffer so learn_rl() actually runs
            for _ in range(self.agent.batch_size):
                self.agent.rl_buffer.push(_create_dummy_vector(), 0, 0, _create_dummy_vector(), False, _create_dummy_mask())
            
            # Manually trigger the step check
            self.agent._attempt_learning_step()
            
            # Check equality (Should still be DIFFERENT)
            params_match = all(
                torch.equal(p1, p2) 
                for p1, p2 in zip(self.agent.br_network.parameters(), self.agent.br_target_network.parameters())
            )
            self.assertFalse(params_match, f"Target network synced prematurely at step {self.agent.step_count}")

        # -- Step 5: SHOULD Sync --
        # Populate buffer
        for _ in range(self.agent.batch_size):
            self.agent.rl_buffer.push(_create_dummy_vector(), 0, 0, _create_dummy_vector(), False, _create_dummy_mask())
            
        self.agent._attempt_learning_step() # This will hit step_count = 5
        
        # Check equality (Should now be IDENTICAL)
        params_match = all(
            torch.equal(p1, p2) 
            for p1, p2 in zip(self.agent.br_network.parameters(), self.agent.br_target_network.parameters())
        )
        self.assertTrue(params_match, "Target network failed to sync at the correct frequency.")

    # ----------------------------------------------------------------
    # 4. RESERVOIR SAMPLING LOGIC TEST
    # ----------------------------------------------------------------
    def test_sl_buffer_reservoir_sampling(self):
        """Test that SLBuffer handles capacity correctly using reservoir sampling logic."""
        capacity = 10
        # Create a small buffer for testing
        buffer = SLBuffer(capacity=capacity, input_size=FEATURE_VECTOR_SIZE)
        
        # 1. Fill to capacity
        for i in range(capacity):
            buffer.push(_create_dummy_vector(), i)
            
        self.assertEqual(len(buffer), capacity)
        self.assertEqual(buffer.total_count, capacity)
        
        # 2. Push over capacity
        # Reservoir sampling means total_count increases, but buffer size stays at capacity
        buffer.push(_create_dummy_vector(), 999)
        
        self.assertEqual(len(buffer), capacity, "Buffer size should not exceed capacity.")
        self.assertEqual(buffer.total_count, capacity + 1, "Total count should track total insertions.")
        
        # 3. Verify internal structure integrity
        self.assertEqual(buffer.state_buffer.shape[0], capacity)
        self.assertEqual(buffer.action_buffer.shape[0], capacity)

if __name__ == '__main__':
    unittest.main()

