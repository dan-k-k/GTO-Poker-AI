# tests/test_features.py
# find . -type d -name "__pycache__" -exec rm -r {} +
# python -m unittest tests.test_features
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
        
        # Scenario 1: Player 0 has pocket Aces on the button
        state1 = self._create_base_state(
            hole_cards=[cards(['Ac', 'Ad']), cards(['7h', '2s'])],
            dealer_pos=0
        )
        schema1 = self.extractor.extract_features(state1)
        self.assertEqual(schema1.hand.is_button, 1.0, "Should be button")
        self.assertEqual(schema1.hand.is_pair, 1.0, "Should be a pair")
        self.assertEqual(schema1.hand.is_suited, 0.0, "Should not be suited")
        self.assertAlmostEqual(schema1.hand.high_card_rank, 1.0)

        # Scenario 2: Player 0 has KQs (suited non-pair), not on the button
        self.extractor.new_hand() # Reset for a new hand
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
            pot=100, big_blind=10, starting_stack=200, num_players=2, # Explicitly set for calculation
            stacks=[150, 200], current_bets=[0, 50],
            last_raiser=1 # Opponent was last aggressor
        )
        
        # Manually calculate expected normalized values
        my_stack_bb_raw = 15.0
        opp_stack_bb_raw = 20.0
        pot_bb_raw = 10.0
        STARTING_STACK_BB = 20.0
        MAX_POT_BB = 40.0
        
        expected_my_stack_norm = np.log1p(my_stack_bb_raw) / np.log1p(STARTING_STACK_BB)
        expected_opp_stack_norm = np.log1p(opp_stack_bb_raw) / np.log1p(STARTING_STACK_BB)
        expected_pot_norm = np.log1p(pot_bb_raw) / np.log1p(MAX_POT_BB)
        expected_spr_norm = min(1.5, 20.0) / 20.0 # spr is 150/100 = 1.5

        schema = self.extractor.extract_features(state)
        
        self.assertAlmostEqual(schema.dynamic.my_stack_bb, expected_my_stack_norm)
        self.assertAlmostEqual(schema.dynamic.opp_stack_bb, expected_opp_stack_norm)
        self.assertAlmostEqual(schema.dynamic.pot_bb, expected_pot_norm)
        self.assertAlmostEqual(schema.dynamic.effective_stack_bb, expected_my_stack_norm) # Effective stack is our stack
        self.assertAlmostEqual(schema.dynamic.pot_odds, 50 / 150, msg="Pot odds incorrect")
        self.assertAlmostEqual(schema.dynamic.bet_faced_ratio, 0.5, msg="Bet faced ratio incorrect")
        self.assertAlmostEqual(schema.dynamic.spr, expected_spr_norm, msg="SPR incorrect")
        self.assertEqual(schema.dynamic.player_has_initiative, 0.0, "Initiative should be with opponent")

    def test_card_based_street_features(self):
        """Tests both player-specific and board-only card features (made hands, draws)."""
        print("\n--- Testing Card-Based Street Features (Player and Board) ---")
        
        # Player has Two Pair on the flop
        state_2p = self._create_base_state(stage=1, hole_cards=[cards(['As', 'Kh'])], community=cards(['Ac', 'Kc', '7d']))
        schema_2p = self.extractor.extract_features(state_2p)
        self.assertEqual(schema_2p.flop_cards.made_hand_rank_twopair, 1.0, "Player should have two pair")

        # Player has a 4-card flush draw on the flop
        self.extractor.new_hand()
        state_fd = self._create_base_state(stage=1, hole_cards=[cards(['Ah', 'Qh'])], community=cards(['2h', '7h', 'Kc']))
        schema_fd = self.extractor.extract_features(state_fd)
        self.assertEqual(schema_fd.flop_cards.has_4card_flush_draw, 1.0, "Player should have a 4-card flush draw")

        # Board has a gapped 3-card straight draw on the flop (5-6-8)
        self.extractor.new_hand()
        state_bsd_gapped = self._create_base_state(stage=1, community=cards(['5s', '6h', '8d']))
        schema_bsd_gapped = self.extractor.extract_features(state_bsd_gapped)
        # Assert the quality is 0.5 for a gapped draw
        self.assertAlmostEqual(schema_bsd_gapped.flop_cards.is_3card_straight_draw_board, 0.5, msg="Board should have a gapped 3-card straight draw (quality 0.5)")

        # Board has a consecutive 3-card straight draw on the flop (5-6-7)
        self.extractor.new_hand()
        state_bsd_conn = self._create_base_state(stage=1, community=cards(['5s', '6h', '7d']))
        schema_bsd_conn = self.extractor.extract_features(state_bsd_conn)
        # Assert the quality is 1.0 for a consecutive draw
        self.assertAlmostEqual(schema_bsd_conn.flop_cards.is_3card_straight_draw_board, 1.0, msg="Board should have a consecutive 3-card straight draw (quality 1.0)")

        # Board has a pair on a 4-card board (turn)
        self.extractor.new_hand()
        state_bp = self._create_base_state(stage=2, community=cards(['Ac', 'Ad', '7s', '2c']))
        schema_bp = self.extractor.extract_features(state_bp)
        self.assertEqual(schema_bp.turn_cards.board_made_rank_pair, 1.0, "Board should have a pair")
        self.assertEqual(schema_bp.flop_cards.board_made_rank_pair, 1.0, "Flop history should also show a board pair")

    def test_tiered_blocker_features_are_correct(self):
        """Tests that the tiered blocker features are calculated correctly."""
        print("\n--- Testing Tiered Blocker Features ---")

        # FLUSH BLOCKER TESTS (has_flush_blocker: 1.0, 0.5, or 0.0)
        print("  --> Testing 'has_flush_blocker'")

        # Scenario 1: Player blocks an IMMEDIATE flush draw (3 suited cards on board)
        # Expected value: 1.0
        self.extractor.new_hand()
        state_immediate_flush_blocker = self._create_base_state(
            stage=1, # Flop
            community=cards(['Kh', 'Qh', '2h']), # 3-flush board
            hole_cards=[cards(['Ah', '3d']), cards(['Xx', 'Xx'])] # We hold ONE heart
        )
        schema1 = self.extractor.extract_features(state_immediate_flush_blocker)
        self.assertEqual(schema1.flop_cards.has_flush_blocker, 1.0,
                         msg="FAIL: Should be 1.0 for blocking an immediate flush draw")

        # Scenario 2: Player blocks a BACKDOOR flush draw (2 suited cards on board)
        # Expected value: 0.5
        self.extractor.new_hand()
        state_backdoor_flush_blocker = self._create_base_state(
            stage=1, # Flop
            community=cards(['Kh', 'Qd', '2h']), # 2-flush board
            hole_cards=[cards(['Ah', '3d']), cards(['Xx', 'Xx'])] # We hold ONE heart
        )
        schema2 = self.extractor.extract_features(state_backdoor_flush_blocker)
        self.assertEqual(schema2.flop_cards.has_flush_blocker, 0.5,
                         msg="FAIL: Should be 0.5 for blocking a backdoor flush draw")

        # Scenario 3: Player has a MADE FLUSH - blocker should be OFF
        # Expected value: 0.0 (because made_hand_rank_flush will be 1.0)
        self.extractor.new_hand()
        state_made_flush = self._create_base_state(
            stage=1, # Flop
            community=cards(['Kh', 'Qh', '2h']), # 3-flush board
            hole_cards=[cards(['Ah', 'Th']), cards(['Xx', 'Xx'])] # We hold TWO hearts
        )
        schema3 = self.extractor.extract_features(state_made_flush)
        self.assertEqual(schema3.flop_cards.has_flush_blocker, 0.0,
                         msg="FAIL: Blocker should be 0.0 when a flush is already made")
        self.assertEqual(schema3.flop_cards.made_hand_rank_flush, 1.0,
                         msg="FAIL: made_hand_rank_flush should be 1.0")

        # STRAIGHT BLOCKER TESTS (straight_blocker_value: 1.0, 0.5, or 0.0)
        print("  --> Testing 'straight_blocker_value'")

        # Scenario 4: Player blocks an OPEN-ENDED straight draw
        # Expected value: 1.0
        self.extractor.new_hand()
        state_oesd_blocker = self._create_base_state(
            stage=1, # Flop
            community=cards(['8s', '9d', 'Td']), # OESD board (needs a 7 or J)
            hole_cards=[cards(['Js', '2c']), cards(['Xx', 'Xx'])] # We hold one of the outs
        )
        schema4 = self.extractor.extract_features(state_oesd_blocker)
        self.assertEqual(schema4.flop_cards.straight_blocker_value, 1.0,
                         msg="FAIL: Should be 1.0 for blocking an OESD")

        # Scenario 5: Player blocks a GUTSHOT straight draw
        # Expected value: 0.5
        self.extractor.new_hand()
        state_gutshot_blocker = self._create_base_state(
            stage=1, # Flop
            community=cards(['8s', '9d', 'Jc']), # Gutshot board (needs a T)
            hole_cards=[cards(['Ts', '2c']), cards(['Xx', 'Xx'])] # We hold the gutshot out
        )
        schema5 = self.extractor.extract_features(state_gutshot_blocker)
        self.assertEqual(schema5.flop_cards.straight_blocker_value, 0.5,
                         msg="FAIL: Should be 0.5 for blocking a gutshot")

        # Scenario 6: Player has a MADE STRAIGHT - blocker should be OFF
        # Expected value: 0.0 (because made_hand_rank_straight will be 1.0)
        self.extractor.new_hand()
        state_made_straight = self._create_base_state(
            stage=1, # Flop
            community=cards(['8s', '9d', 'Td']), # OESD board (needs a 7 or J)
            hole_cards=[cards(['Js', '7c']), cards(['Xx', 'Xx'])] # We hold both outs
        )
        schema6 = self.extractor.extract_features(state_made_straight)
        self.assertEqual(schema6.flop_cards.straight_blocker_value, 0.0,
                         msg="FAIL: Blocker should be 0.0 when a straight is already made")
        self.assertEqual(schema6.flop_cards.made_hand_rank_straight, 1.0,
                         msg="FAIL: made_hand_rank_straight should be 1.0")

        # Scenario 7: Player has no relevant blockers on a dangerous board
        self.extractor.new_hand()
        state_no_blocker = self._create_base_state(
            stage=1, # Flop
            community=cards(['8s', '9d', 'Td']), # OESD board
            hole_cards=[cards(['2c', '2h']), cards(['Xx', 'Xx'])] # We hold cards that don't block
        )
        schema7 = self.extractor.extract_features(state_no_blocker)
        self.assertEqual(schema7.flop_cards.straight_blocker_value, 0.0,
                         msg="FAIL: Should have 0 straight blockers")
        self.assertEqual(schema7.flop_cards.has_flush_blocker, 0.0,
                         msg="FAIL: Should have 0 flush blockers")

    def test_board_texture_features(self):
        """
        Tests the feature extractor's ability to analyze complex board textures,
        including paired, monotone, and connected boards.
        """
        print("\n--- Testing Complex Board Texture Features ---")

        # --- Scenario 1: Paired Board (Full House vs. Trips) ---
        self.extractor.new_hand()
        opp_extractor = FeatureExtractor(seat_id=1) # Need opponent's perspective

        state_paired = self._create_base_state(
            stage=1, # Flop
            community=cards(['Ac', 'Ad', '7s']),
            hole_cards=[cards(['7h', '7d']), cards(['As', 'Ks'])] # P0 has Full House, P1 has Trips
        )
        schema_p0 = self.extractor.extract_features(state_paired)
        schema_p1 = opp_extractor.extract_features(state_paired)

        # Assert Player 0's Full House is detected
        self.assertEqual(schema_p0.flop_cards.made_hand_rank_fullhouse, 1.0, "P0 should have a Full House")
        self.assertEqual(schema_p0.flop_cards.made_hand_rank_trips, 1.0, "Full House should also register as Trips")
        
        # Assert Player 1's Trips are detected (and not a full house)
        self.assertEqual(schema_p1.flop_cards.made_hand_rank_fullhouse, 0.0, "P1 should not have a Full House")
        self.assertEqual(schema_p1.flop_cards.made_hand_rank_trips, 1.0, "P1 should have Trips")

        # Assert the board itself is correctly identified as paired
        self.assertEqual(schema_p0.flop_cards.board_made_rank_pair, 1.0, "Board should register as paired for P0")
        self.assertEqual(schema_p1.flop_cards.board_made_rank_pair, 1.0, "Board should register as paired for P1")

        # --- Scenario 2: Monotone Board (Flush vs. Set) ---
        self.extractor.new_hand()
        opp_extractor.new_hand()

        state_monotone = self._create_base_state(
            stage=1, # Flop
            community=cards(['Ah', '7h', '2h']),
            hole_cards=[cards(['Kh', 'Qh']), cards(['As', 'Ac'])] # P0 has a Flush, P1 has a Set
        )
        schema_p0 = self.extractor.extract_features(state_monotone)
        schema_p1 = opp_extractor.extract_features(state_monotone)

        # Assert Player 0's Flush is detected
        self.assertEqual(schema_p0.flop_cards.made_hand_rank_flush, 1.0, "P0 should have a Flush")
        self.assertEqual(schema_p0.flop_cards.made_hand_rank_trips, 0.0, "P0 should not have a Set")

        # Assert Player 1's Set is detected
        self.assertEqual(schema_p1.flop_cards.made_hand_rank_flush, 0.0, "P1 should not have a Flush")
        self.assertEqual(schema_p1.flop_cards.made_hand_rank_trips, 1.0, "P1 should have a Set of Aces")

        # Assert the board is identified as a 3-card flush draw board
        self.assertEqual(schema_p0.flop_cards.is_3card_flush_draw_board, 1.0, "Board should be a 3-flush board")

        # --- Scenario 3: Four-to-a-Straight Board ---
        self.extractor.new_hand()
        state_connected = self._create_base_state(
            stage=2, # Turn
            community=cards(['8s', '9s', 'Td', 'Jc']) # Open-ended straight draw board
        )
        schema = self.extractor.extract_features(state_connected)
        
        # The board has 8-9-T-J, which is an open-ended straight draw
        self.assertEqual(schema.turn_cards.is_4card_straight_draw_board, 1.0, "Board should be a 4-card straight draw (quality 1.0)")
        self.assertEqual(schema.turn_cards.is_3card_flush_draw_board, 0.0, "Board is not a 3-flush draw (only 2 spades)")

    def test_full_hand_history_and_separation(self):
        """
        Simulates a multi-street hand to verify that historical betting
        data is correctly stored in past streets, while dynamic data reflects the current street.
        """
        print("\n--- Testing Full Hand History & Past/Present Separation ---")
        self.extractor.new_hand()
        
        # ACTION SEQUENCE
        # 1. PREFLOP: Opponent (P1) opens, we (P0) 3-bet, Opponent calls.
        
        # MODIFIED: Create GameState objects instead of dictionaries.
        state_p1_open = self._create_base_state(pot=3, stage=0, current_bets=[2, 1])
        self.extractor.update_betting_action(1, 2, state_p1_open, 0)

        state_p0_3bet = self._create_base_state(pot=8, stage=0, current_bets=[2, 6], last_raiser=1)
        self.extractor.update_betting_action(0, 2, state_p0_3bet, 0)
        
        state_p1_call = self._create_base_state(pot=18, stage=0, current_bets=[6, 12])
        self.extractor.update_betting_action(1, 1, state_p1_call, 0)

        # 2. FLOP: We (P0) make a continuation bet.
        state_p0_cbet = self._create_base_state(pot=24, stage=1, current_bets=[0, 0], initial_bet=None)
        self.extractor.update_betting_action(0, 2, state_p0_cbet, 1)

        # 3. TURN: Opponent (P1) checks, we (P0) check back.
        state_p1_check = self._create_base_state(pot=48, stage=2, current_bets=[0, 0])
        self.extractor.update_betting_action(1, 1, state_p1_check, 2)
        
        state_p0_check = self._create_base_state(pot=48, stage=2, current_bets=[0, 0])
        self.extractor.update_betting_action(0, 1, state_p0_check, 2)
        
        # --- VERIFICATION AT THE START OF THE RIVER ---
        final_state = self._create_base_state(stage=3) # Now it's the river
        schema = self.extractor.extract_features(final_state)

        # Verify PREFLOP history (completed street)
        preflop_history = schema.preflop_betting
        self.assertAlmostEqual(preflop_history.opp_bets_opened, 0.1, msg="Preflop: Opponent should have 1 open (normalized)")
        self.assertAlmostEqual(preflop_history.my_raises_made, 0.1, msg="Preflop: We should have 1 raise (normalized)")
        self.assertAlmostEqual(preflop_history.actions_on_street, 0.3, msg="Preflop: Should be 3 total actions (normalized)")

        # Verify FLOP history (completed street)
        flop_history = schema.flop_betting
        self.assertAlmostEqual(flop_history.my_bets_opened, 0.1, msg="Flop: We should have 1 open (normalized)")
        self.assertAlmostEqual(flop_history.actions_on_street, 0.1, msg="Flop: Should be 1 total action (normalized)")

        # Verify TURN history (completed street)
        turn_history = schema.turn_betting
        self.assertAlmostEqual(turn_history.my_bets_opened, 0.0, msg="Turn: No bets were opened")
        self.assertAlmostEqual(turn_history.actions_on_street, 0.2, msg="Turn: Should be 2 total actions (normalized)")

    def _assert_mirrored_betting_features(self, my_features: BettingRoundFeatures, opp_features: BettingRoundFeatures, street_name: str):
        """Helper to assert that betting features are correctly mirrored between two perspectives."""
        self.assertEqual(my_features.my_bets_opened, opp_features.opp_bets_opened, f"{street_name}: my_bets_opened mismatch")
        self.assertEqual(my_features.my_raises_made, opp_features.opp_raises_made, f"{street_name}: my_raises_made mismatch")
        self.assertEqual(my_features.opp_bets_opened, opp_features.my_bets_opened, f"{street_name}: opp_bets_opened mismatch")
        self.assertEqual(my_features.opp_raises_made, opp_features.my_raises_made, f"{street_name}: opp_raises_made mismatch")
        self.assertEqual(my_features.actions_on_street, opp_features.actions_on_street, f"{street_name}: actions_on_street mismatch")

    def _assert_identical_board_features(self, my_features: StreetFeatures, opp_features: StreetFeatures, street_name: str):
        """Helper to assert that all public board features are identical."""
        self.assertEqual(my_features.board_made_rank_pair, opp_features.board_made_rank_pair, f"{street_name}: board pair mismatch")
        self.assertEqual(my_features.board_made_rank_twopair, opp_features.board_made_rank_twopair, f"{street_name}: board two pair mismatch")
        self.assertEqual(my_features.is_3card_flush_draw_board, opp_features.is_3card_flush_draw_board, f"{street_name}: board 3FD mismatch")
        self.assertEqual(my_features.is_4card_straight_draw_board, opp_features.is_4card_straight_draw_board, f"{street_name}: board 4SD mismatch")
        
    def test_opponent_perspective_is_mirrored(self):
        """
        Ensures the FeatureExtractor's logic is symmetrical for a single, known GameState.
        This test uses perfect information by design to verify the tool's correctness.
        """
        print("\n--- Testing Opponent Perspective Mirroring ---")
        my_extractor = FeatureExtractor(seat_id=0)
        opp_extractor = FeatureExtractor(seat_id=1)

        # --- Simulate a shared history for both extractors ---
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
            # Player 0 (us) has a gutshot straight draw (8,9 on a T,6,2 board needs a 7).
            # Player 1 (opponent) has two high cards that don't connect.
            hole_cards=[cards(['8s', '9d']), cards(['As', 'Ks'])],
            # This board is not inherently "drawy".
            community=cards(['Tc', '6h', '2d']),
            pot=12,
            current_bets=[0, 0],
            stacks=[188, 188],
            to_move=0,
            # Note: last_raiser doesn't matter for this specific card-feature test
            last_raiser=None 
        )

        my_schema = my_extractor.extract_features(final_state)
        opp_schema = opp_extractor.extract_features(final_state)
        
        # --- VERIFY MIRRORED FEATURES ---
        self.assertNotEqual(my_schema.hand.is_suited, opp_schema.hand.is_suited)

        # My hand (8,9 on T,6,2) has a 3-card consecutive draw (8,9,T -> quality 1.0)
        # and a 4-card gutshot draw (6,8,9,T -> quality 0.5).
        self.assertAlmostEqual(my_schema.flop_cards.has_3card_straight_draw, 1.0, msg="P0 should have a 3-card consecutive draw")
        self.assertAlmostEqual(my_schema.flop_cards.has_4card_straight_draw, 0.5, msg="P0 should have a 4-card gutshot draw")
        
        # The opponent's hand (A,K on T,6,2) has a gapped 3-card straight draw (T,K,A).
        self.assertAlmostEqual(opp_schema.flop_cards.has_3card_straight_draw, 0.5, msg="P1 should have a gapped 3-card straight draw (quality 0.5)")
        self.assertAlmostEqual(opp_schema.flop_cards.has_4card_straight_draw, 0.0, msg="P1 should have no 4-card straight draw")

    def test_rare_board_made_hands(self):
        """Tests that the extractor correctly identifies rare made hands on the board."""
        print("\n--- Testing Rare Board-Only Made Hands ---")

        # 1. Test for a flush on the board
        self.extractor.new_hand()
        state_flush = self._create_base_state(stage=3, community=cards(['2h', '5h', '8h', 'Th', 'Kh']))
        schema_flush = self.extractor.extract_features(state_flush)
        self.assertEqual(schema_flush.river_cards.board_made_rank_flush, 1.0, "Board should be a flush")

        # 2. Test for a straight on the board
        self.extractor.new_hand()
        state_straight = self._create_base_state(stage=3, community=cards(['5c', '6d', '7h', '8s', '9c']))
        schema_straight = self.extractor.extract_features(state_straight)
        self.assertEqual(schema_straight.river_cards.board_made_rank_straight, 1.0, "Board should be a straight")
        
        # 3. Test for a full house on the board
        self.extractor.new_hand()
        state_fh = self._create_base_state(stage=3, community=cards(['Ac', 'Ad', 'Ah', 'Ks', 'Kd']))
        schema_fh = self.extractor.extract_features(state_fh)
        self.assertEqual(schema_fh.river_cards.board_made_rank_fullhouse, 1.0, "Board should be a full house")


    @patch('app.nfsp_components.NFSPAgent._extract_opponent_hand_features')
    def test_equity_simulation_uses_imperfect_information_optimized(self, mock_extract_hand_features: MagicMock):
        """
        Verifies that the equity simulation loop tries different opponent hands,
        confirming it works with imperfect information under the new optimized logic.
        """
        print("\n--- Testing Equity Simulation for Imperfect Information (Integration Test) ---")
        agent_config = {'eta': 0.1, 'gamma': 0.99, 'batch_size': 32, 'update_frequency': 1, 'learning_rate': 0.001, 'target_update_frequency': 100}
        buffer_config = {'rl_buffer_capacity': 1000, 'sl_buffer_capacity': 1000}
        
        agent = NFSPAgent(
            seat_id=0, 
            agent_config=agent_config, 
            buffer_config=buffer_config,
            random_equity_trials=100,
            intelligent_equity_trials=50
        )

        agent.opponent_as_network = MagicMock()
        def mock_network_side_effect(feature_batch_tensor):
            batch_size = feature_batch_tensor.shape[0]
            return {'action_probs': torch.ones(batch_size, 12) / 12}
        agent.opponent_as_network.side_effect = mock_network_side_effect
        
        mock_extract_hand_features.return_value = np.zeros(PokerFeatureSchema.get_vector_size())

        my_real_hand = cards(['As', 'Ks'])
        opp_real_hand = cards(['Qd', 'Qc'])
        community = cards(['Ac', '5h', '6s'])

        state_before_opp_action = self._create_base_state(
            stage=1,
            community=community,
            hole_cards=[my_real_hand, opp_real_hand]
        )
        
        agent.last_opp_state_before_action = state_before_opp_action
        agent.last_opp_action_index = 5

        agent._calculate_intelligent_equity(
            my_hole_cards=my_real_hand,
            community_cards=community
        )

        self.assertEqual(mock_extract_hand_features.call_count, agent.intelligent_equity_trials + 1,
                         "The hand feature extractor helper was not called for each trial + 1.")

        calls = mock_extract_hand_features.call_args_list
        simulated_hands = set()
        
        for call in calls[1:]:
            # THIS IS THE CORRECTED LINE:
            # The hand list is the first argument (index 0) after `self`.
            simulated_opp_hand_list = call.args[0] 
            simulated_hands.add(tuple(sorted(simulated_opp_hand_list)))

        self.assertGreater(len(simulated_hands), 1, 
                         "Equity simulation should have tried multiple different opponent hands.")

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
        # 1. The calculation should have run in the first case and returned plausible values
        self.assertGreater(schema_with_equity.flop_cards.random_strength, 0, 
                         "random_strength should be calculated when flag is False")
        self.assertGreater(schema_with_equity.dynamic.hand_strength, 0,
                         "dynamic.hand_strength should be populated from random_strength")

        # 2. The calculation should have been skipped in the second case, leaving default values
        self.assertEqual(schema_without_equity.flop_cards.random_strength, 0.0,
                         "random_strength should be 0.0 when flag is True")
        self.assertEqual(schema_without_equity.dynamic.hand_strength, 0.0,
                         "dynamic.hand_strength should be 0.0 when random_strength is skipped")
        
        # 3. All other features should be identical
        # Temporarily set all equity-related fields to be the same to compare the rest of the object.
        
        # Reset the equity for ALL processed streets AND the dynamic feature that depends on it.
        schema_with_equity.preflop_cards.random_strength = 0.0
        schema_with_equity.flop_cards.random_strength = 0.0
        schema_with_equity.dynamic.hand_strength = 0.0 

        self.assertEqual(schema_with_equity, schema_without_equity,
                         "Other features should not change when skipping random equity")

if __name__ == '__main__':
    unittest.main()

