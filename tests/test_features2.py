# tests/test_features2.py
import unittest
import numpy as np
from app.feature_extractor import FeatureExtractor
from app.poker_core import GameState, string_to_card_id

# Helper
def cards(card_strs: list[str]) -> list[int]:
    return [string_to_card_id(s) for s in card_strs]

class TestFeaturesEdgeCases(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(seat_id=0, num_players=2)

    def _create_state(self, **kwargs):
        # Minimal defaults for a valid state
        defaults = {
            'num_players': 2, 'starting_stack': 200, 'small_blind': 1, 'big_blind': 2,
            'hole_cards': [cards(['As', 'Ks']), cards(['Qd', 'Qc'])],
            'community': [], 'stacks': [200, 200], 'current_bets': [0, 0],
            'pot': 0, 'starting_pot_this_round': 0, 'starting_stacks_this_hand': [200, 200],
            'active': [True, True], 'all_in': [False, False], 'acted': [False, False],
            'surviving_players': [0, 1], 'stage': 0, 'dealer_pos': 1,
            'sb_pos': 1, 'bb_pos': 0, 'to_move': 0, 'initial_bet': None,
            'last_raise_size': 0, 'last_raiser': None, 'terminal': False,
            'winners': None, 'win_reason': None
        }
        defaults.update(kwargs)
        # Filter only valid keys for GameState
        valid_keys = GameState.__dataclass_fields__.keys()
        filtered = {k: v for k, v in defaults.items() if k in valid_keys}
        return GameState(**filtered)

    def test_short_stack_pot_odds(self):
        """
        CRITICAL: Tests that pot odds are calculated correctly when facing a bet
        larger than our remaining stack (Effective Stack Logic).
        """
        print("\n--- Testing Short Stack / All-In Pot Odds ---")
        
        # Scenario: 
        # Pot is 10. 
        # Player 0 (Hero) has 20 chips left. 
        # Player 1 (Villain) bets 100 (effectively putting Hero all-in).
        # Total Pot becomes 110.
        
        # Math:
        # Hero can only call 20.
        # Villain's effective bet is 20. The other 80 is "excess" returned to Villain.
        # Effective Pot = 10 (Start) + 20 (Villain) + 20 (Hero Call) = 50.
        # Hero pays 20 to win 50.
        # Pot Odds = 20 / 50 = 0.40.
        
        state = self._create_state(
            pot=110,                # 10 start + 100 bet
            current_bets=[0, 100],  # Hero 0, Villain 100
            stacks=[20, 100],       # Hero has 20 left
            to_move=0,
            stage=1
        )
        
        schema = self.extractor.extract_features(state)
        
        # Verify Pot Odds
        self.assertAlmostEqual(schema.dynamic.pot_odds, 0.40, msg="Pot odds failed to account for stack cap")
        
        # Verify 'Bet Faced Ratio'
        # Effective bet faced is 20. Pot before that was 10.
        # Ratio = 20 / 30 (size of pot created by villain's effective bet) -> 0.666...
        # Or does your logic use (pot - excess)? 
        # Code: actual_to_call / (state.pot - excess_bet)
        # 20 / (110 - 80) = 20 / 30 = 0.666...
        self.assertAlmostEqual(schema.dynamic.bet_faced_ratio, 2/3, msg="Bet faced ratio failed effective stack logic")

    def test_initiative_persistence(self):
        """
        Tests that 'player_has_initiative' correctly tracks who raised last,
        even across streets.
        """
        print("\n--- Testing Initiative Persistence ---")
        
        # 1. Preflop: Hero Raises
        state_pf = self._create_state(stage=0, last_raiser=0)
        
        # Update extractor (simulating the observation of that action)
        state_before_raise = self._create_state(stage=0, last_raiser=None) 
        self.extractor.update_betting_action(0, 2, state_before_raise, 0) # Hero raises
        
        # Extract features for Flop
        state_flop = self._create_state(stage=1, last_raiser=None, community=cards(['Jh', 'Th', '2c']))
        schema_flop = self.extractor.extract_features(state_flop)
        
        self.assertEqual(schema_flop.dynamic.player_has_initiative, 1.0, 
                         "Hero should still have initiative on Flop after Preflop raise")
        
        # 2. Flop: Hero Checks, Villain Bets
        # Hero Checks
        self.extractor.update_betting_action(0, 1, state_flop, 1)
        # Villain Bets (Update requires state before bet)
        state_flop_checked = self._create_state(stage=1, last_raiser=None)
        self.extractor.update_betting_action(1, 2, state_flop_checked, 1)
        
        # 3. Turn: Extract features
        state_turn = self._create_state(stage=2, community=cards(['Jh', 'Th', '2c', '5h']))
        schema_turn = self.extractor.extract_features(state_turn)
        
        self.assertEqual(schema_turn.dynamic.player_has_initiative, 0.0,
                         "Hero should lose initiative after Villain bet on Flop")

    def test_betting_war_counters(self):
        """
        Tests that aggression counters (bets_opened, raises_made) increment correctly
        during a back-and-forth raising war (4-betting).
        """
        print("\n--- Testing Betting War Counters ---")
        
        # Sequence on Flop:
        # 1. Villain Bets (Open)
        # 2. Hero Raises (Raise)
        # 3. Villain 3-bets (Raise)
        # 4. Hero 4-bets (Raise)
        
        stage = 1
        
        # 1. Villain Bets
        s1 = self._create_state(stage=stage, last_raiser=None)
        self.extractor.update_betting_action(1, 2, s1, stage)
        
        # 2. Hero Raises
        s2 = self._create_state(stage=stage, last_raiser=1)
        self.extractor.update_betting_action(0, 2, s2, stage)
        
        # 3. Villain 3-bets
        s3 = self._create_state(stage=stage, last_raiser=0)
        self.extractor.update_betting_action(1, 2, s3, stage)
        
        # 4. Hero 4-bets
        s4 = self._create_state(stage=stage, last_raiser=1)
        self.extractor.update_betting_action(0, 2, s4, stage)
        
        # Check Final Features
        state_final = self._create_state(stage=stage, last_raiser=0)
        schema = self.extractor.extract_features(state_final)
        
        # Normalize Clip checks (values are normalized / 10.0)
        # Hero: 0 Opens, 2 Raises (The initial raise and the 4-bet)
        self.assertAlmostEqual(schema.dynamic.current_betting_round.my_bets_opened, 0.0)
        self.assertAlmostEqual(schema.dynamic.current_betting_round.my_raises_made, 0.2) # 2/10
        
        # Villain: 1 Open, 1 Raise (The 3-bet)
        self.assertAlmostEqual(schema.dynamic.current_betting_round.opp_bets_opened, 0.1) # 1/10
        self.assertAlmostEqual(schema.dynamic.current_betting_round.opp_raises_made, 0.1) # 1/10
        
        # Total Actions: 4 aggressive actions
        self.assertAlmostEqual(schema.dynamic.current_betting_round.actions_on_street, 0.4)

if __name__ == '__main__':
    unittest.main()

