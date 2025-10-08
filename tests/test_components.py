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

def _create_dummy_vector():
    return np.random.rand(FEATURE_VECTOR_SIZE).astype(np.float32)

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
        buffer = ReplayBuffer(capacity=3)
        self.assertEqual(len(buffer), 0)
        for i in range(4):
            buffer.push(_create_dummy_vector(), i, float(i), _create_dummy_vector(), False)
        self.assertEqual(len(buffer), 3)
        actions_in_buffer = [experience[1] for experience in buffer.buffer]
        self.assertNotIn(0, actions_in_buffer)
        self.assertIn(1, actions_in_buffer)
        self.assertIn(3, actions_in_buffer)

    def test_rl_buffer_sample(self):
        buffer = ReplayBuffer(capacity=10)
        batch_size = 4
        for _ in range(5):
            buffer.push(_create_dummy_vector(), 0, 0.0, _create_dummy_vector(), False)
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertEqual(states.shape, (batch_size, FEATURE_VECTOR_SIZE))
        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertEqual(next_states.shape, (batch_size, FEATURE_VECTOR_SIZE))
        self.assertEqual(dones.shape, (batch_size,))
        self.assertEqual(states.dtype, torch.float32)
        self.assertEqual(actions.dtype, torch.long)
        self.assertEqual(dones.dtype, torch.bool)

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
            intelligent_equity_trials=50
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
            self.agent.rl_buffer.push(_create_dummy_vector(), 0, 1.0, _create_dummy_vector(), False)
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
            self.agent.rl_buffer.push(_create_dummy_vector(), 0, 1.0, _create_dummy_vector(), False)
        self.agent.learn_rl()
        updated_params = [p.clone() for p in self.agent.br_network.parameters()]
        params_are_identical = all(torch.equal(p_initial, p_updated) for p_initial, p_updated in zip(initial_params, updated_params))
        self.assertTrue(params_are_identical, "Network weights changed even though buffer was too small.")

    def test_get_action_index_from_move(self):
        """Tests the helper that maps a game action to a policy index."""
        state = _create_mock_state(pot=40, stacks=[100, 100], current_bets=[10, 10])
        
        self.assertEqual(self.agent._get_action_index_from_move(state, 0, None), 0, "Fold should be index 0")
        self.assertEqual(self.agent._get_action_index_from_move(state, 1, None), 1, "Check/Call should be index 1")

        first_raise_index = next((i for i, (action, sizing) in enumerate(ACTION_MAP) if i > 1 and sizing != -1), None)
        self.assertIsNotNone(first_raise_index, "ACTION_MAP must contain at least one non-all-in raise option.")
        sizing_to_test = ACTION_MAP[first_raise_index][1]

        opponent_id = 1 - self.agent.seat_id
        amount_to_call = max(state.current_bets) - state.current_bets[opponent_id]
        pot_after_call = state.pot + amount_to_call
        
        raise_amount = pot_after_call * sizing_to_test
        total_amount = amount_to_call + raise_amount
        
        calculated_index = self.agent._get_action_index_from_move(state, 2, int(total_amount))
        self.assertEqual(calculated_index, first_raise_index)

        all_in_index = next((i for i, (_, sizing) in enumerate(ACTION_MAP) if sizing == -1.0), -1)
        if all_in_index != -1:
             self.assertEqual(self.agent._get_action_index_from_move(state, 2, state.stacks[opponent_id]), all_in_index)

    def test_observe_showdown_populates_buffer_correctly(self):
        """
        Tests that observe_showdown processes the trajectory and pushes an
        experience tuple with the correct reward and done flag.
        """
        last_state_vector = _create_dummy_vector()
        next_state_vector = _create_dummy_vector()
        last_action_index = 1
        
        self.agent.current_hand_trajectory.append(
            (last_state_vector, last_action_index, next_state_vector)
        )
        
        showdown_state = {'stacks_before': {0: 150}, 'stacks_after': {0: 200}}

        self.agent.observe_showdown(showdown_state)
        
        self.assertEqual(len(self.agent.rl_buffer), 1, "Should have pushed exactly one experience.")
        
        s, a, r, next_s, done = self.agent.rl_buffer.buffer[0]
        
        self.assertTrue(np.array_equal(s, last_state_vector))
        self.assertEqual(a, last_action_index)
        self.assertEqual(r, 50.0, "The final reward should be the net winnings.")
        self.assertTrue(np.array_equal(next_s, next_state_vector))
        self.assertTrue(done, "The experience should be marked as terminal (done=True).")

    def test_observe_showdown_after_opponent_folds(self):
        """
        Tests that a correct final reward is assigned from the trajectory
        when the hand ends due to a fold.
        """
        last_state_vector = _create_dummy_vector()
        next_state_vector = _create_dummy_vector()
        last_action_index = 2
        
        self.agent.current_hand_trajectory.append(
            (last_state_vector, last_action_index, next_state_vector)
        )

        showdown_state = {
            'stacks_before': {0: 180},
            'stacks_after': {0: 210}
        }
        expected_reward = 30.0

        self.agent.observe_showdown(showdown_state)

        self.assertEqual(len(self.agent.rl_buffer), 1, "Should have pushed the final hand experience.")

        s, a, r, next_s, done = self.agent.rl_buffer.buffer[0]
        
        self.assertTrue(np.array_equal(s, last_state_vector), "State vector mismatch.")
        self.assertEqual(a, last_action_index, "Action index mismatch.")
        self.assertEqual(r, expected_reward, "Final reward should be the net profit from the hand.")
        self.assertTrue(np.array_equal(next_s, next_state_vector), "Next state vector mismatch.")
        self.assertTrue(done, "The experience should be marked as terminal (done=True).")

    def test_intelligent_equity_batching(self):
        """
        Tests that the equity simulation correctly batches feature vectors
        before calling the network.
        """
        num_trials = 5
        self.agent.last_opp_action_index = 2
        mock_env_state = _create_mock_state(hole_cards=[[0, 1], [4, 5]], community=[8, 9, 10])
        self.agent.last_opp_state_before_action = mock_env_state

        def mock_network_side_effect(feature_batch_tensor):
            # This is the core check: does the network receive a batch of the correct size?
            self.assertEqual(feature_batch_tensor.shape[0], num_trials,
                             "The batch tensor passed to the network has the wrong size.")

            batch_size = feature_batch_tensor.shape[0]
            probs = torch.full((batch_size, len(ACTION_MAP)), 0.01)
            probs[:, 2] = 0.9
            return {'action_probs': probs}

        self.agent.opponent_as_network = MagicMock(side_effect=mock_network_side_effect)

        self.agent.intelligent_equity_trials = num_trials
        self.agent._calculate_intelligent_equity(
            my_hole_cards=mock_env_state.hole_cards[0],
            community_cards=mock_env_state.community
        )

        # Ensure the mock network was actually called
        self.agent.opponent_as_network.assert_called_once()

    def test_intelligent_equity_generates_correct_opponent_schema(self):
        """
        Verifies that the schema generated from the opponent's perspective
        inside the equity calculation is correct.
        """
        agent_seat_id = 0
        opp_seat_id = 1
        
        state_faced_by_opponent = _create_mock_state(
            stage=2,
            community=[8, 16, 24, 28],
            pot=75,
            stacks=[175, 175],
            current_bets=[25, 0],
            last_raiser=agent_seat_id,
            dealer_pos=opp_seat_id,
            to_move=opp_seat_id
        )

        self.agent.opponent_as_network = MagicMock(return_value={
            'action_probs': torch.rand(1, len(ACTION_MAP))
        })
        self.agent.last_opp_state_before_action = state_faced_by_opponent
        
        self.agent.last_opp_action_index = 3 

        simulated_opp_hand = [4, 12] # e.g., Ad, Kd
        
        # This side effect handles the two different calls to np.random.choice
        def mock_choice_side_effect(deck, size, replace):
            if size == 2: # This is the call for the opponent's hand
                return np.array(simulated_opp_hand)
            elif size == 1: # This is the call for the board runout
                card = [c for c in deck if c not in simulated_opp_hand][0]
                return np.array([card])
            # Fallback for other cases
            return np.random.choice(deck, size=size, replace=replace)

        captured_schema = None
        
        original_extract = FeatureExtractor.extract_features
        def spy_on_extraction(self_extractor, state_obj, agent=None, skip_random_equity=False):
            nonlocal captured_schema
            schema = original_extract(self_extractor, state_obj, agent, skip_random_equity)
            if captured_schema is None:
                captured_schema = schema
            return schema

        with patch('app.feature_extractor.FeatureExtractor.extract_features', new=spy_on_extraction), \
             patch('numpy.random.choice', side_effect=mock_choice_side_effect, create=True):
            
            self.agent.intelligent_equity_trials = 1
            self.agent._calculate_intelligent_equity(
                my_hole_cards=[0, 1],
                community_cards=state_faced_by_opponent.community
            )

        self.assertIsNotNone(captured_schema, "The spy did not capture the feature schema.")

        self.assertEqual(captured_schema.hand.is_button, 1.0, "Opponent should be the button.")
        self.assertEqual(captured_schema.dynamic.player_has_initiative, 0.0, "Opponent is facing a bet, so they don't have initiative.")
        
        amount_to_call = 25
        pot_before_call = 75
        final_pot = pot_before_call + amount_to_call
        expected_pot_odds = amount_to_call / final_pot
        self.assertAlmostEqual(captured_schema.dynamic.pot_odds, expected_pot_odds, places=4)
        
        self.assertEqual(captured_schema.turn_cards.made_hand_rank_flush, 1.0, "Schema should detect the opponent's simulated flush.")

    def _set_agent_mid_hand_state(self):
        """Helper to put the agent in a state as if it's mid-hand."""
        self.agent.last_state = _create_dummy_vector()
        self.agent.last_action = 2
        self.agent.current_hand_trajectory.append(
            (_create_dummy_vector(), 1, _create_dummy_vector())
        )

    def test_new_hand_resets_state(self):
        """
        Tests that new_hand() correctly resets the agent's per-hand state. 🧹
        """
        self._set_agent_mid_hand_state()
        self.assertTrue(self.agent.last_state is not None)
        self.assertTrue(len(self.agent.current_hand_trajectory) > 0)

        self.agent.new_hand()

        self.assertIsNone(self.agent.last_state, "last_state should be reset.")
        self.assertIsNone(self.agent.last_action, "last_action should be reset.")
        self.assertEqual(len(self.agent.current_hand_trajectory), 0,
                         "The hand trajectory should be empty.")

    def test_observe_builds_trajectory_correctly(self):
        """
        Tests that the observe() method correctly appends the agent's own
        experiences to the trajectory and ignores the opponent's. 📝
        """
        self._set_agent_mid_hand_state()
        initial_trajectory_len = len(self.agent.current_hand_trajectory)
        
        mock_state = _create_mock_state()

        self.agent.observe(player_action=(2, 20), player_id=self.agent.seat_id,
                           state_before_action=mock_state, next_state=mock_state)

        self.assertEqual(len(self.agent.current_hand_trajectory),
                         initial_trajectory_len + 1,
                         "Trajectory should grow after observing own action.")

        self.agent.observe(player_action=(1, 0), player_id=1 - self.agent.seat_id,
                           state_before_action=mock_state, next_state=mock_state)

        self.assertEqual(len(self.agent.current_hand_trajectory),
                         initial_trajectory_len + 1,
                         "Trajectory should not grow after observing an opponent's action.")


    def test_compute_action_uses_eta_policy(self):
        """
        Tests that compute_action() correctly uses eta to select between
        the Best Response (BR) and Average Strategy (AS) networks.
        """
        mock_state = _create_mock_state()
        
        self.agent.eta = 1.0
        
        with patch.object(self.agent, 'as_network', wraps=self.agent.as_network) as spy_as_net, \
             patch.object(self.agent, 'br_network', wraps=self.agent.br_network) as spy_br_net:
            
            self.agent.compute_action(mock_state)
            
            spy_as_net.assert_called_once()
            spy_br_net.assert_not_called()

        self.agent.eta = 0.0
        
        with patch.object(self.agent, 'as_network', wraps=self.agent.as_network) as spy_as_net, \
             patch.object(self.agent, 'br_network', wraps=self.agent.br_network) as spy_br_net:

            self.agent.compute_action(mock_state)

            spy_br_net.assert_called_once()
            spy_as_net.assert_not_called()

if __name__ == '__main__':
    unittest.main()

