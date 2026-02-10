# tests/test_features.py
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from copy import deepcopy
import torch

from app.feature_extractor import FeatureExtractor
from app.poker_core import GameState, string_to_card_id
from app.poker_feature_schema import PokerFeatureSchema, BettingRoundFeatures, StreetFeatures
from app.nfsp_components import NFSPAgent 

# Helper function to create card lists from strings
def cards(card_strs: list[str]) -> list[int]:
    """Converts a list of card strings (e.g., ['As', 'Ks']) to integer IDs."""
    return [string_to_card_id(s) for s in card_strs]

class TestFeatureExtractor(unittest.TestCase):
    """A comprehensive test suite for the FeatureExtractor."""

    def setUp(self):
        """Set up a fresh feature extractor for player 0 before each test."""
        self.extractor = FeatureExtractor(seat_id=0, num_players=2)

    def _create_base_state(self, **kwargs) -> GameState:
        """Creates a mock GameState object with reasonable defaults for a 2-player game."""
        defaults = {
            'num_players': 2, 'starting_stack': 200, 'small_blind': 1, 'big_blind': 2,
            'hole_cards': [cards(['As', 'Ks']), cards(['Qd', 'Qc'])],
            'community': [], 'stacks': [198, 199], 'current_bets': [2, 1],
            'pot': 3, 'starting_pot_this_round': 3, 'starting_stacks_this_hand': [200, 200],
            'active': [True, True], 'all_in': [False, False], 'acted': [False, False],
            'surviving_players': [0, 1], 'stage': 0, 'dealer_pos': 1,
            'sb_pos': 1, 'bb_pos': 0, 'to_move': 1, 'initial_bet': 2,
            'last_raise_size': 2, 'last_raiser': None, 'terminal': False,
            'winners': None, 'win_reason': None
        }
        defaults.update(kwargs)
        valid_fields = GameState.__dataclass_fields__.keys()
        filtered_args = {k: v for k, v in defaults.items() if k in valid_fields}
        return GameState(**filtered_args)

    def test_hand_static_features(self):
        """Tests features that are static for the entire hand (position, hole cards)."""
        print("\n--- Testing Hand-Static Features ---")
        
        state1 = self._create_base_state(
            hole_cards=[cards(['Ac', 'Ad']), cards(['7h', '2s'])],
            dealer_pos=0
        )
        schema1 = self.extractor.extract_features(state1)
        self.assertEqual(schema1.hand.is_button, 1.0, "Should be button")
        self.assertEqual(schema1.hand.is_pair, 1.0, "Should be a pair")
        self.assertEqual(schema1.hand.is_suited, 0.0, "Should not be suited")
        self.assertAlmostEqual(schema1.hand.high_card_rank, 1.0)

        self.extractor.new_hand()
        state2 = self._create_base_state(
            hole_cards=[cards(['Ks', 'Qs']), cards(['2c', '3d'])],
            dealer_pos=1
        )
        schema2 = self.extractor.extract_features(state2)
        self.assertEqual(schema2.hand.is_button, 0.0, "Should not be button")
        self.assertEqual(schema2.hand.is_pair, 0.0, "Should not be a pair")
        self.assertEqual(schema2.hand.is_suited, 1.0, "Should be suited")
        self.assertAlmostEqual(schema2.hand.high_card_rank, 11/12.0)

    def test_dynamic_features_at_decision_point(self):
        """Tests features that change with every action (stacks, pot odds, SPR)."""
        print("\n--- Testing Action-Dependent Dynamic Features ---")
        state = self._create_base_state(
            stage=1, community=cards(['Jc', '8d', '2s']),
            pot=100, big_blind=10, starting_stack=200, num_players=2,
            stacks=[150, 200], current_bets=[0, 50],
            last_raiser=1 
        )
        
        my_stack_bb_raw = 15.0
        opp_stack_bb_raw = 20.0
        pot_bb_raw = 10.0
        STARTING_STACK_BB = 20.0
        MAX_POT_BB = 40.0
        
        expected_my_stack_norm = np.log1p(my_stack_bb_raw) / np.log1p(STARTING_STACK_BB)
        expected_opp_stack_norm = np.log1p(opp_stack_bb_raw) / np.log1p(STARTING_STACK_BB)
        expected_pot_norm = np.log1p(pot_bb_raw) / np.log1p(MAX_POT_BB)
        expected_spr_norm = min(1.5, 20.0) / 20.0 

        schema = self.extractor.extract_features(state)
        
        self.assertAlmostEqual(schema.dynamic.my_stack_bb, expected_my_stack_norm)
        self.assertAlmostEqual(schema.dynamic.opp_stack_bb, expected_opp_stack_norm)
        self.assertAlmostEqual(schema.dynamic.pot_bb, expected_pot_norm)
        self.assertAlmostEqual(schema.dynamic.effective_stack_bb, expected_my_stack_norm)
        self.assertAlmostEqual(schema.dynamic.pot_odds, 50 / 150, msg="Pot odds incorrect") # it should be 0.333 since pot defined earlier DOES contain opp bet this street.
        self.assertAlmostEqual(schema.dynamic.bet_faced_ratio, 0.5, msg="Bet faced ratio incorrect")
        self.assertAlmostEqual(schema.dynamic.spr, expected_spr_norm, msg="SPR incorrect")
        self.assertEqual(schema.dynamic.player_has_initiative, 0.0, "Initiative should be with opponent")

    def test_card_based_street_features(self):
        """Tests both player-specific and board-only card features (made hands, draws)."""
        print("\n--- Testing Card-Based Street Features (Player and Board) ---")
        
        state_2p = self._create_base_state(stage=1, hole_cards=[cards(['As', 'Kh'])], community=cards(['Ac', 'Kc', '7d']))
        schema_2p = self.extractor.extract_features(state_2p)
        self.assertEqual(schema_2p.flop_cards.made_hand_rank_twopair, 1.0, "Player should have two pair")

        self.extractor.new_hand()
        state_fd = self._create_base_state(stage=1, hole_cards=[cards(['Ah', 'Qh'])], community=cards(['2h', '7h', 'Kc']))
        schema_fd = self.extractor.extract_features(state_fd)
        self.assertEqual(schema_fd.flop_cards.has_4card_flush_draw, 1.0, "Player should have a 4-card flush draw")

        self.extractor.new_hand()
        state_bsd_gapped = self._create_base_state(stage=1, community=cards(['5s', '6h', '8d']))
        schema_bsd_gapped = self.extractor.extract_features(state_bsd_gapped)
        self.assertAlmostEqual(schema_bsd_gapped.flop_cards.is_3card_straight_draw_board, 0.5, msg="Board should have a gapped 3-card straight draw (quality 0.5)")

        self.extractor.new_hand()
        state_bsd_conn = self._create_base_state(stage=1, community=cards(['5s', '6h', '7d']))
        schema_bsd_conn = self.extractor.extract_features(state_bsd_conn)
        self.assertAlmostEqual(schema_bsd_conn.flop_cards.is_3card_straight_draw_board, 1.0, msg="Board should have a consecutive 3-card straight draw (quality 1.0)")

        self.extractor.new_hand()
        state_bp = self._create_base_state(stage=2, community=cards(['Ac', 'Ad', '7s', '2c']))
        schema_bp = self.extractor.extract_features(state_bp)
        self.assertEqual(schema_bp.turn_cards.board_made_rank_pair, 1.0, "Board should have a pair")
        self.assertEqual(schema_bp.flop_cards.board_made_rank_pair, 1.0, "Flop history should also show a board pair")

    def test_tiered_blocker_features_are_correct(self):
        """Tests that the tiered blocker features are calculated correctly."""
        print("\n--- Testing Tiered Blocker Features ---")
        print("  --> Testing 'has_flush_blocker'")

        self.extractor.new_hand()
        state_immediate_flush_blocker = self._create_base_state(
            stage=1, community=cards(['Kh', 'Qh', '2h']), hole_cards=[cards(['Ah', '3d']), cards(['Xx', 'Xx'])]
        )
        schema1 = self.extractor.extract_features(state_immediate_flush_blocker)
        self.assertEqual(schema1.flop_cards.has_flush_blocker, 1.0)

        self.extractor.new_hand()
        state_backdoor_flush_blocker = self._create_base_state(
            stage=1, community=cards(['Kh', 'Qd', '2h']), hole_cards=[cards(['Ah', '3d']), cards(['Xx', 'Xx'])]
        )
        schema2 = self.extractor.extract_features(state_backdoor_flush_blocker)
        self.assertEqual(schema2.flop_cards.has_flush_blocker, 0.5)

        self.extractor.new_hand()
        state_made_flush = self._create_base_state(
            stage=1, community=cards(['Kh', 'Qh', '2h']), hole_cards=[cards(['Ah', 'Th']), cards(['Xx', 'Xx'])]
        )
        schema3 = self.extractor.extract_features(state_made_flush)
        self.assertEqual(schema3.flop_cards.has_flush_blocker, 0.0)
        self.assertEqual(schema3.flop_cards.made_hand_rank_flush, 1.0)

        print("  --> Testing 'straight_blocker_value'")

        self.extractor.new_hand()
        state_oesd_blocker = self._create_base_state(
            stage=1, community=cards(['8s', '9d', 'Td']), hole_cards=[cards(['Js', '2c']), cards(['Xx', 'Xx'])]
        )
        schema4 = self.extractor.extract_features(state_oesd_blocker)
        self.assertEqual(schema4.flop_cards.straight_blocker_value, 1.0)

        self.extractor.new_hand()
        state_gutshot_blocker = self._create_base_state(
            stage=1, community=cards(['8s', '9d', 'Jc']), hole_cards=[cards(['Ts', '2c']), cards(['Xx', 'Xx'])]
        )
        schema5 = self.extractor.extract_features(state_gutshot_blocker)
        self.assertEqual(schema5.flop_cards.straight_blocker_value, 0.5)

        self.extractor.new_hand()
        state_made_straight = self._create_base_state(
            stage=1, community=cards(['8s', '9d', 'Td']), hole_cards=[cards(['Js', '7c']), cards(['Xx', 'Xx'])]
        )
        schema6 = self.extractor.extract_features(state_made_straight)
        self.assertEqual(schema6.flop_cards.straight_blocker_value, 0.0)
        self.assertEqual(schema6.flop_cards.made_hand_rank_straight, 1.0)

        self.extractor.new_hand()
        state_no_blocker = self._create_base_state(
            stage=1, community=cards(['8s', '9d', 'Td']), hole_cards=[cards(['2c', '2h']), cards(['Xx', 'Xx'])]
        )
        schema7 = self.extractor.extract_features(state_no_blocker)
        self.assertEqual(schema7.flop_cards.straight_blocker_value, 0.0)
        self.assertEqual(schema7.flop_cards.has_flush_blocker, 0.0)

    def test_board_texture_features(self):
        """Tests the feature extractor's ability to analyze complex board textures."""
        print("\n--- Testing Complex Board Texture Features ---")

        self.extractor.new_hand()
        opp_extractor = FeatureExtractor(seat_id=1) 

        state_paired = self._create_base_state(
            stage=1, community=cards(['Ac', 'Ad', '7s']),
            hole_cards=[cards(['7h', '7d']), cards(['As', 'Ks'])] 
        )
        schema_p0 = self.extractor.extract_features(state_paired)
        schema_p1 = opp_extractor.extract_features(state_paired)

        self.assertEqual(schema_p0.flop_cards.made_hand_rank_fullhouse, 1.0)
        self.assertEqual(schema_p0.flop_cards.made_hand_rank_trips, 1.0)
        self.assertEqual(schema_p1.flop_cards.made_hand_rank_fullhouse, 0.0)
        self.assertEqual(schema_p1.flop_cards.made_hand_rank_trips, 1.0)
        self.assertEqual(schema_p0.flop_cards.board_made_rank_pair, 1.0)
        self.assertEqual(schema_p1.flop_cards.board_made_rank_pair, 1.0)

        self.extractor.new_hand()
        opp_extractor.new_hand()

        state_monotone = self._create_base_state(
            stage=1, community=cards(['Ah', '7h', '2h']),
            hole_cards=[cards(['Kh', 'Qh']), cards(['As', 'Ac'])]
        )
        schema_p0 = self.extractor.extract_features(state_monotone)
        schema_p1 = opp_extractor.extract_features(state_monotone)

        self.assertEqual(schema_p0.flop_cards.made_hand_rank_flush, 1.0)
        self.assertEqual(schema_p0.flop_cards.made_hand_rank_trips, 0.0)
        self.assertEqual(schema_p1.flop_cards.made_hand_rank_flush, 0.0)
        self.assertEqual(schema_p1.flop_cards.made_hand_rank_trips, 1.0)
        self.assertEqual(schema_p0.flop_cards.is_3card_flush_draw_board, 1.0)

        self.extractor.new_hand()
        state_connected = self._create_base_state(
            stage=2, community=cards(['8s', '9s', 'Td', 'Jc'])
        )
        schema = self.extractor.extract_features(state_connected)
        
        self.assertEqual(schema.turn_cards.is_4card_straight_draw_board, 1.0)
        self.assertEqual(schema.turn_cards.is_3card_flush_draw_board, 0.0)

    def test_full_hand_history_and_separation(self):
        """Simulates a multi-street hand to verify historical vs dynamic data."""
        print("\n--- Testing Full Hand History & Past/Present Separation ---")
        self.extractor.new_hand()
        
        state_p1_open = self._create_base_state(pot=3, stage=0, current_bets=[2, 1])
        self.extractor.update_betting_action(1, 2, state_p1_open, 0)

        state_p0_3bet = self._create_base_state(pot=8, stage=0, current_bets=[2, 6], last_raiser=1)
        self.extractor.update_betting_action(0, 2, state_p0_3bet, 0)
        
        state_p1_call = self._create_base_state(pot=18, stage=0, current_bets=[6, 12])
        self.extractor.update_betting_action(1, 1, state_p1_call, 0)

        state_p0_cbet = self._create_base_state(pot=24, stage=1, current_bets=[0, 0], initial_bet=None)
        self.extractor.update_betting_action(0, 2, state_p0_cbet, 1)

        state_p1_check = self._create_base_state(pot=48, stage=2, current_bets=[0, 0])
        self.extractor.update_betting_action(1, 1, state_p1_check, 2)
        
        state_p0_check = self._create_base_state(pot=48, stage=2, current_bets=[0, 0])
        self.extractor.update_betting_action(0, 1, state_p0_check, 2)
        
        final_state = self._create_base_state(stage=3)
        schema = self.extractor.extract_features(final_state)

        preflop_history = schema.preflop_betting
        self.assertAlmostEqual(preflop_history.opp_bets_opened, 0.1)
        self.assertAlmostEqual(preflop_history.my_raises_made, 0.1)
        self.assertAlmostEqual(preflop_history.actions_on_street, 0.3)

        flop_history = schema.flop_betting
        self.assertAlmostEqual(flop_history.my_bets_opened, 0.1)
        self.assertAlmostEqual(flop_history.actions_on_street, 0.1)

        turn_history = schema.turn_betting
        self.assertAlmostEqual(turn_history.my_bets_opened, 0.0)
        self.assertAlmostEqual(turn_history.actions_on_street, 0.2)

    def test_opponent_perspective_is_mirrored(self):
        """Ensures the FeatureExtractor's logic is symmetrical."""
        print("\n--- Testing Opponent Perspective Mirroring ---")
        my_extractor = FeatureExtractor(seat_id=0)
        opp_extractor = FeatureExtractor(seat_id=1)

        state_p1_open = self._create_base_state(pot=3, stage=0)
        my_extractor.update_betting_action(1, 2, state_p1_open, 0)
        opp_extractor.update_betting_action(1, 2, state_p1_open, 0)
        
        state_p0_call = self._create_base_state(pot=8, current_bets=[6, 6], stage=0)
        my_extractor.update_betting_action(0, 1, state_p0_call, 0)
        opp_extractor.update_betting_action(0, 1, state_p0_call, 0)

        state_p1_check = self._create_base_state(pot=12, current_bets=[0, 0], stage=1)
        my_extractor.update_betting_action(1, 1, state_p1_check, 1)
        opp_extractor.update_betting_action(1, 1, state_p1_check, 1)

        final_state = self._create_base_state(
            stage=1,
            hole_cards=[cards(['8s', '9d']), cards(['As', 'Ks'])],
            community=cards(['Tc', '6h', '2d']),
            pot=12, current_bets=[0, 0], stacks=[188, 188], to_move=0, last_raiser=None 
        )

        my_schema = my_extractor.extract_features(final_state)
        opp_schema = opp_extractor.extract_features(final_state)
        
        self.assertNotEqual(my_schema.hand.is_suited, opp_schema.hand.is_suited)
        self.assertAlmostEqual(my_schema.flop_cards.has_3card_straight_draw, 1.0)
        self.assertAlmostEqual(my_schema.flop_cards.has_4card_straight_draw, 0.5)
        self.assertAlmostEqual(opp_schema.flop_cards.has_3card_straight_draw, 0.5)
        self.assertAlmostEqual(opp_schema.flop_cards.has_4card_straight_draw, 0.0)

    def test_rare_board_made_hands(self):
        """Tests that the extractor correctly identifies rare made hands on the board."""
        print("\n--- Testing Rare Board-Only Made Hands ---")

        self.extractor.new_hand()
        state_flush = self._create_base_state(stage=3, community=cards(['2h', '5h', '8h', 'Th', 'Kh']))
        schema_flush = self.extractor.extract_features(state_flush)
        self.assertEqual(schema_flush.river_cards.board_made_rank_flush, 1.0)

        self.extractor.new_hand()
        state_straight = self._create_base_state(stage=3, community=cards(['5c', '6d', '7h', '8s', '9c']))
        schema_straight = self.extractor.extract_features(state_straight)
        self.assertEqual(schema_straight.river_cards.board_made_rank_straight, 1.0)
        
        self.extractor.new_hand()
        state_fh = self._create_base_state(stage=3, community=cards(['Ac', 'Ad', 'Ah', 'Ks', 'Kd']))
        schema_fh = self.extractor.extract_features(state_fh)
        self.assertEqual(schema_fh.river_cards.board_made_rank_fullhouse, 1.0)

    # REMOVED: test_equity_simulation_uses_imperfect_information
    # Reason: Feature "Intelligent Equity" (opponent range prediction) was removed.

    def test_skip_random_equity_flag_works_as_intended(self):
        """
        Verifies the `skip_random_equity` flag correctly prevents
        the random_strength calculation without altering other features.
        """
        print("\n--- Testing skip_random_equity Flag ---")
        # Arrange: Set up a standard post-flop state
        state = self._create_base_state(
            stage=1, 
            hole_cards=[cards(['As', 'Kh'])], 
            community=cards(['Ac', 'Kc', '7d'])
        )

        extractor_with_equity = FeatureExtractor(seat_id=0)
        extractor_without_equity = FeatureExtractor(seat_id=0)

        # Act: Extract features using the independent instances
        schema_with_equity = extractor_with_equity.extract_features(state, skip_random_equity=False)
        schema_without_equity = extractor_without_equity.extract_features(state, skip_random_equity=True)

        # Assert:
        # 1. The calculation should have run in the first case
        self.assertGreater(schema_with_equity.flop_cards.random_strength, 0, 
                         "random_strength should be calculated when flag is False")
        
        # 2. The calculation should have been skipped in the second case, leaving default values
        self.assertEqual(schema_without_equity.flop_cards.random_strength, 0.0,
                         "random_strength should be 0.0 when flag is True")
        
        # 3. All other features should be identical
        # Temporarily set all equity-related fields to be the same to compare the rest of the object.
        
        # Reset the equity for ALL processed streets.
        schema_with_equity.preflop_cards.random_strength = 0.0
        schema_with_equity.flop_cards.random_strength = 0.0
        # NOTE: Removed schema_with_equity.dynamic.hand_strength = 0.0 (Feature Removed)

        self.assertEqual(schema_with_equity, schema_without_equity,
                         "Other features should not change when skipping random equity")

if __name__ == '__main__':
    unittest.main()

