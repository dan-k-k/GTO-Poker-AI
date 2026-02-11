# tests/test_environment4.py
# find . -type d -name "__pycache__" -exec rm -r {} +
# python -m unittest tests.test_environment4
import unittest
from app.TexasHoldemEnv import TexasHoldemEnv, GameState
from app.poker_core import string_to_card_id

def cards(card_strs: list[str]) -> list[int]:
    """Helper to convert a list of card strings to integer IDs for tests."""
    return [string_to_card_id(s) for s in card_strs]

class TestFinalEdgeCases(unittest.TestCase):
    """
    The final battery of tests for the TexasHoldemEnv.
    Focuses on Heads-Up specific turn orders, chip conservation in odd splits,
    and raise re-opening logic.
    """

    def setUp(self):
        """This method is called before each test function."""
        print(f"\n--- Running Final Edge Case Test: {self.id()} ---")

    def _setup_hand(self, num_players=2, hole_cards=None, community=None, stacks=None, dealer_pos=0):
        """A powerful helper to manually set up a specific game state for testing."""
        env = TexasHoldemEnv(num_players=num_players, starting_stack=2000, small_blind=10, big_blind=20)
        
        if stacks is None:
            stacks = [2000] * num_players
        if hole_cards is None:
            all_cards = [cards(['As', 'Ks']), cards(['Ad', 'Kd']), cards(['Ac', 'Kc']), cards(['Ah', 'Kh'])]
            hole_cards = all_cards[:num_players]
            
        starting_stacks_for_hand = stacks.copy()
            
        survivors = [i for i, s in enumerate(stacks) if s > 0]
        if len(survivors) == 2:
            sb_pos = dealer_pos
            bb_pos = (dealer_pos + 1) % 2
            to_move = sb_pos # SB acts first pre-flop
        else:
            sb_pos = (dealer_pos + 1) % num_players
            bb_pos = (dealer_pos + 2) % num_players
            to_move = (dealer_pos + 3) % num_players
            
        current_bets = [0] * num_players
        sb_amount = min(10, stacks[sb_pos]) if stacks[sb_pos] > 0 else 0
        bb_amount = min(20, stacks[bb_pos]) if stacks[bb_pos] > 0 else 0
        
        stacks[sb_pos] -= sb_amount
        stacks[bb_pos] -= bb_amount
        current_bets[sb_pos] = sb_amount
        current_bets[bb_pos] = bb_amount
        pot = sb_amount + bb_amount
        
        env.state = GameState(
            num_players=num_players, starting_stack=2000, small_blind=10, big_blind=20,
            hole_cards=hole_cards, community=community or [],
            stacks=stacks, current_bets=current_bets, pot=pot,
            starting_pot_this_round=pot, 
            starting_stacks_this_hand=starting_stacks_for_hand,
            active=[s > 0 for s in starting_stacks_for_hand], 
            all_in=[s == 0 for s in stacks], 
            acted=[False] * num_players,
            surviving_players=survivors, stage=0 if not community else 1, 
            dealer_pos=dealer_pos, sb_pos=sb_pos, bb_pos=bb_pos, to_move=to_move,
            initial_bet=20, last_raise_size=20, last_raiser=None, terminal=False,
            winners=None, win_reason=None
        )
        return env

    def test_heads_up_post_flop_order(self):
        """
        Verify that in Heads-Up play:
        - Pre-flop: Dealer (SB) acts first.
        - Post-flop: Non-Dealer (BB) acts first.
        """
        env = self._setup_hand(num_players=2, dealer_pos=0) # P0=Dealer/SB, P1=BB
        
        # 1. Pre-flop: Action should be on P0 (SB/Dealer)
        self.assertEqual(env.state.to_move, 0, "Pre-flop: Dealer/SB should act first")
        
        # Action: P0 Calls (10 -> 20), P1 Checks (20). Flop deals.
        env.step(1)
        env.step(1)
        
        self.assertEqual(env.state.stage, 1, "Game should be on the Flop")
        
        # 2. Post-flop: Action should flip to P1 (BB/Non-Dealer)
        self.assertEqual(env.state.to_move, 1, "Post-flop: BB (Non-Dealer) should act first")

    def test_odd_chip_distribution(self):
        """
        Verify that odd chips in split pots are conserved and not lost.
        Scenario: Pot is 25. Two players split it. 
        Result should be 13 and 12 (or vice versa), total 25.
        """
        # Set up a board where both players play the board (Royal Flush) -> Split Pot
        board = cards(['As', 'Ks', 'Qs', 'Js', 'Ts'])
        hole_cards = [cards(['2c', '3c']), cards(['2d', '3d'])]
        
        env = self._setup_hand(num_players=2, hole_cards=hole_cards, community=board, stacks=[100, 100])
        
        # Manually force the pot to be an odd number
        env.state.pot = 25
        env.state.starting_stacks_this_hand = [112, 113] 
        env.state.stacks = [100, 100] 
        env.state.active = [True, True]
        env.state.terminal = True
        
        hand_ranks = env._calculate_showdown_hand_ranks()
        env._distribute_pot_with_side_pots(hand_ranks)
        
        final_total = sum(env.state.stacks)
        self.assertEqual(final_total, 225, "Total chips must be conserved (200 stacks + 25 pot)")
        
        diff = abs(env.state.stacks[0] - env.state.stacks[1])
        self.assertEqual(diff, 1, "Split pot with odd chips should result in 1 chip difference")

    def test_full_raise_reopens_action(self):
        """
        Verify that in a Heads-Up specific engine, a 'full raise' that puts a player all-in
        does NOT allow a re-raise.
        
        Logic: In HU, if the opponent is All-In, raising is functionally identical to Calling.
        To simplify the action space for ML/RL, we mask 'Raise' as illegal in this spot.
        """
        # Setup: P0 (SB/Dealer), P1 (BB).
        # Stacks: P0=2000 (Deep), P1=300 (Short).
        env = self._setup_hand(num_players=2, stacks=[2000, 300], dealer_pos=0)
        
        # P1 posts 20 BB. Remaining stack: 300 - 20 = 280.
        
        # Pre-flop action to get to the Flop
        env.step(1) # P0 Call
        env.step(1) # P1 Check
        self.assertEqual(env.state.stage, 1) # Flop
        
        # Post-flop: P1 acts first (BB). 
        env.step(1) # P1 Checks.
        
        # P0 Bets 100.
        env.step(2, amount=100)
        self.assertEqual(env.state.current_bets[0], 100)
        
        # P1 Raises All-In. 
        p1_stack = env.state.stacks[1] # Should be 280
        env.step(2, amount=p1_stack)
        
        # Action returns to P0.
        self.assertEqual(env.state.to_move, 0)
        
        # Verify P0 CANNOT Raise.
        legal_actions = env.state.get_legal_actions()
        
        # We now Assert that 2 (Raise) is NOT in legal actions
        self.assertNotIn(2, legal_actions, "P0 should NOT be allowed to re-raise an All-In in HU (redundant action)")
        self.assertIn(1, legal_actions, "P0 must be able to Call")
        self.assertIn(0, legal_actions, "P0 must be able to Fold")

    def test_simultaneous_elimination_results_in_tournament_winner(self):
        """
        Tests that a hand eliminating multiple players correctly ends the tournament.
        Uses a rigged deck to ensure deterministic outcome.
        """
        hole_cards = [cards(['Ac', 'Ad']), cards(['Kc', 'Kd']), cards(['Qc', 'Qd'])]
        env = self._setup_hand(num_players=3, hole_cards=hole_cards, stacks=[2000, 100, 100], dealer_pos=0)

        env.deck.cards = cards(['2s', '3h', '4c', '5d', '7s']) 

        # Action: P0 (UTG) raises to 200.
        state, done = env.step(action=2, amount=200)
        self.assertFalse(done)

        # Action: P1 (SB) calls all-in.
        state, done = env.step(action=1) 
        self.assertFalse(done)

        # Action: P2 (BB) calls all-in.
        state, done = env.step(action=1)
        
        self.assertTrue(done, "Hand should be terminal after final all-in call")

        # Assert correct tournament termination
        self.assertEqual(state.win_reason, 'tournament_winner')
        self.assertEqual(state.winners, [0], "Player 0 (AA) should be the sole winner")
        self.assertEqual(state.surviving_players, [0], "Only P0 should be in surviving_players")
        self.assertEqual(state.stacks[0], 2200, "P0 should have all chips")

if __name__ == '__main__':
    unittest.main()

