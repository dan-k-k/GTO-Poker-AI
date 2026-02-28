# tests/test_api_integration.py

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import torch 
import json

from unified_api import app, load_global_model
from app.nfsp_components import NFSPAgent
from app.feature_extractor import FeatureExtractor
from app.poker_core import GameState, string_to_card_id
from app.poker_agents import NUM_ACTIONS

class TestIntegrationFeatures(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Dummy global agent
        dummy_config = {'eta': 0.1, 'gamma': 0.99, 'batch_size': 128, 'update_frequency': 1, 'learning_rate': 0.001, 'target_update_frequency': 100}
        dummy_buffer = {'rl_buffer_capacity': 100, 'sl_buffer_capacity': 100}
        
        cls.dummy_agent = NFSPAgent(1, dummy_config, dummy_buffer, 10, 10)
        
        # Mock network outputs
        mock_out = {
            'action_probs': torch.tensor([[1/NUM_ACTIONS]*NUM_ACTIONS], dtype=torch.float32), 
            'action_logits': torch.zeros((1, NUM_ACTIONS)), 
            'q_values': torch.zeros((1, NUM_ACTIONS)), 
            'state_values': torch.zeros((1, 1))
        }
        cls.dummy_agent.as_network = MagicMock(return_value=mock_out)
        cls.dummy_agent.br_network = MagicMock(return_value=mock_out)
        
        cls.loader_patcher = patch('unified_api.load_global_model', return_value=cls.dummy_agent)
        cls.loader_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.loader_patcher.stop()

    def test_solver_api_time_travel_verification(self):
        """Verifies that the solver API correctly reconstructs past street features from a JSON history log."""
        payload = {
            "pot": 100, "current_bets": [0, 0], "stacks": [150, 150],
            "initial_stacks": [200, 200], 
            "hole_cards": [["Ah", "Kh"], ["2c", "7d"]], 
            "community_cards": ["As", "7d", "2c", "Td", "2h"], 
            "dealer_id": 1, "to_move": 0, "stage": 3,
            "past_actions": [
                # Seat 0 raises first. This is an OPEN.
                {"seat_id": 0, "stage": 0, "action_type": 2, "amount_added": 6}, 
                {"seat_id": 1, "stage": 0, "action_type": 1, "amount_added": 6},
            ]
        }

        with patch("app.feature_extractor.FeatureExtractor.extract_features", 
                   side_effect=FeatureExtractor.extract_features, autospec=True) as mock_extract:
            
            with TestClient(app) as client:
                response = client.post("/get_optimal_action", json=payload)
                
                if response.status_code != 200:
                    self.fail(f"API request failed with {response.status_code}: {response.text}")
                
                extractor_instance = mock_extract.call_args[0][0]
                
                self.assertGreater(extractor_instance._betting_history['my_bets_opened'][0], 0, 
                                   "Integration Fail: API dropped the Preflop open-raise history.")

                captured_state = mock_extract.call_args[0][1]
                result_schema = extractor_instance.extract_features(captured_state)
                self.assertEqual(result_schema.river_cards.board_made_rank_pair, 1.0,
                                 "Integration Fail: API failed to parse community cards correctly.")

    def test_live_session_memory_verification(self):
        """Verifies that the live bot endpoint correctly remembers session stateacross HTTP requests."""
        session_id = "test_live_session"
        
        # Mock environment
        dummy_env = MagicMock()
        dummy_state = GameState(
            num_players=2, starting_stack=200, small_blind=1, big_blind=2,
            hole_cards=[[0, 1], [string_to_card_id('Ks'), string_to_card_id('2s')]], 
            community=[string_to_card_id('Kh'), string_to_card_id('Kd'), string_to_card_id('Kc')],
            stacks=[100, 100], current_bets=[0, 0], pot=200, starting_pot_this_round=200,
            starting_stacks_this_hand=[200, 200], active=[True, True], all_in=[False, False], 
            acted=[False, False], surviving_players=[0, 1], stage=1, dealer_pos=0, sb_pos=0, bb_pos=1,
            to_move=1, initial_bet=0, last_raise_size=0, last_raiser=None, terminal=False, winners=None, win_reason=None
        )
        dummy_env.state = dummy_state
        dummy_env.state.get_legal_actions = MagicMock(return_value=[1, 2])
        dummy_env.state.get_min_raise_amount = MagicMock(return_value=2)
        dummy_env.state.copy = MagicMock(return_value=dummy_state) # Important for observe

        # Return real data, not MagicMock
        dummy_env.get_state_dict.return_value = {
            'pot': 200,
            'stacks': [100, 100],
            'table_image': 'dummy_base64_string', # Mocking the image string
            'legal_actions': [1, 2],
            'min_raise': 2
        }

        import unified_api
        unified_api.ACTIVE_SESSIONS[session_id] = {'env': dummy_env, 'agent': self.dummy_agent, 'pnl_history': []}

        with patch("app.feature_extractor.FeatureExtractor.extract_features", 
                   side_effect=FeatureExtractor.extract_features, autospec=True) as mock_extract:
            
            with TestClient(app) as client:
                client.cookies.set("session_id", session_id)
                response = client.post("/bot_action")

                if response.status_code != 200:
                    self.fail(f"API request failed with {response.status_code}: {response.text}")

                extractor_instance = mock_extract.call_args[0][0]
                schema = extractor_instance.extract_features(dummy_state, self.dummy_agent)
                
                self.assertEqual(schema.flop_cards.made_hand_rank_trips, 1.0, 
                                 "Integration fail: Live bot session did not see its own trips/quads.")

if __name__ == "__main__":
    unittest.main()

