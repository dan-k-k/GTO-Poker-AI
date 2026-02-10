# tests/test_environment.py
# find . -type d -name "__pycache__" -exec rm -r {} +
# python -m unittest tests.test_environment
import unittest
from app.TexasHoldemEnv import TexasHoldemEnv, GameState
from app.poker_core import string_to_card_id, HandEvaluator

def cards(card_strs: list[str]) -> list[int]:
    """Helper to convert a list of card strings to integer IDs for tests."""
    return [string_to_card_id(s) for s in card_strs]

class TestTexasHoldemEnv(unittest.TestCase):
    """A comprehensive test suite for the TexasHoldemEnv class."""

    def setUp(self):
        """This method is called before each test function."""
        print(f"\n--- Running: {self.id()} ---")
        self.evaluator = HandEvaluator()

    def _setup_hand(self, num_players=2, hole_cards=None, community=None, stacks=None, dealer_pos=0):
        """A powerful helper to manually set up a specific game state for testing."""
        env = TexasHoldemEnv(num_players=num_players, starting_stack=200, small_blind=10, big_blind=20)
        
        # Override the random state with a specific, controlled scenario
        if stacks is None:
            stacks = [2000] * num_players
        if hole_cards is None:
            hole_cards = [cards(['As', 'Ks']), cards(['Ad', 'Kd'])]
            
        # --- FIX START: Remove manually assigned cards from the deck ---
        used_cards = []
        if hole_cards:
            for hand in hole_cards:
                used_cards.extend(hand)
        if community:
            used_cards.extend(community)
            
        # Remove these specific integers from the deck's list of available cards
        env.deck.cards = [c for c in env.deck.cards if c not in used_cards]
        # --- FIX END ---

        # Capture the state of the stacks *before* blinds are posted.
        starting_stacks_for_hand = stacks.copy()
            
        # Determine positions based on dealer
        if num_players == 2:
            sb_pos = dealer_pos
            bb_pos = (dealer_pos + 1) % 2
            to_move = sb_pos # Heads-up, SB is first to act pre-flop
        else:
            sb_pos = (dealer_pos + 1) % num_players
            bb_pos = (dealer_pos + 2) % num_players
            to_move = (dealer_pos + 3) % num_players
            
        # Post blinds (...modifies the 'stacks' list)
        current_bets = [0] * num_players
        pot = 30
        stacks[sb_pos] -= 10
        stacks[bb_pos] -= 20
        current_bets[sb_pos] = 10
        current_bets[bb_pos] = 20
        
        env.state = GameState(
            num_players=num_players, starting_stack=2000, small_blind=10, big_blind=20,
            hole_cards=hole_cards, community=community or [],
            stacks=stacks, current_bets=current_bets, pot=pot,
            starting_pot_this_round=pot, 
            starting_stacks_this_hand=starting_stacks_for_hand, # Use the saved copy
            active=[True] * num_players, all_in=[False] * num_players, acted=[False] * num_players,
            surviving_players=list(range(num_players)), stage=0 if not community else 1, 
            dealer_pos=dealer_pos, sb_pos=sb_pos, bb_pos=bb_pos, to_move=to_move,
            initial_bet=20, last_raise_size=20, last_raiser=None, terminal=False,
            winners=None, win_reason=None
        )
        return env

    def test_initialization_and_blinds(self):
        """Verify that a new game starts with correct stacks and blinds posted."""
        env = TexasHoldemEnv(num_players=3, starting_stack=1000, small_blind=5, big_blind=10)
        state = env.state # Use the state object directly
        
        self.assertEqual(state.dealer_pos, 0, "Dealer should start at seat 0")
        self.assertEqual(state.sb_pos, 1, "SB should be at seat 1")
        self.assertEqual(state.bb_pos, 2, "BB should be at seat 2")
        
        self.assertEqual(state.stacks[1], 995, "SB stack should be debited")
        self.assertEqual(state.stacks[2], 990, "BB stack should be debited")
        self.assertEqual(state.pot, 15, "Pot should contain both blinds")
        self.assertEqual(state.to_move, 0, "Action should be on UTG (seat 0)")

    def test_preflop_betting_street_ends_correctly(self):
        """Crucial tests for the street-ending logic, especially the BB option."""
        # Scenario 1: Limp, BB checks. Street should end.
        env = self._setup_hand(dealer_pos=0) # P0 is dealer/SB, P1 is BB
        env.state.to_move = 0
        state, done = env.step(1) # P0 calls the BB
        self.assertEqual(state.to_move, 1, "Action should be on BB")
        self.assertFalse(done, "Hand should not be over")
        state, done = env.step(1) # P1 checks
        self.assertEqual(state.stage, 1, "Street should end and advance to flop")
        self.assertFalse(done, "Hand should not be over after checks")

        # Scenario 2: Limp, BB raises. Action must return to limper.
        env = self._setup_hand(dealer_pos=0)
        env.state.to_move = 0
        state, done = env.step(1) # P0 calls
        state, done = env.step(2, amount=60) # P1 raises to 60 total
        self.assertEqual(state.to_move, 0, "Action must return to P0 after BB raise")
        self.assertEqual(state.stage, 0, "Street is NOT over yet")
        state, done = env.step(0) # P0 folds
        self.assertTrue(done, "Hand should be over after the fold")
        self.assertEqual(state.winners, [1], "Player 1 should win after the fold")

    def test_postflop_betting(self):
        """Test post-flop bet/check/raise logic."""
        env = self._setup_hand(dealer_pos=0, community=cards(['Ac', 'Kd', '7s']))
        env.state.pot = 40
        env.state.current_bets = [0,0]
        env.state.stacks = [1980, 1980]
        env.state.to_move = 1 # Post-flop, player after dealer acts first
        
        # Scenario: P1 checks, P0 bets, P1 calls
        state, done = env.step(1) # P1 checks
        self.assertEqual(state.to_move, 0, "Action should move to P0")
        state, done = env.step(2, amount=30) # P0 bets 30
        self.assertEqual(state.to_move, 1, "Action should move back to P1")
        state, done = env.step(1) # P1 calls
        self.assertEqual(state.stage, 2, "Should advance to the turn street")
        self.assertEqual(state.pot, 100, "Pot should be 40 + 30 + 30")

    def test_all_in_and_auto_completion(self):
        """Test that the board completes automatically after an all-in."""
        # Setup: P0 is SB, P1 is BB. Stacks are BEFORE blinds are posted.
        env = self._setup_hand(dealer_pos=0, stacks=[200, 2000])
        env.state.to_move = 0
        
        # The helper posts blinds, so P0's actual stack is 200 - 10 (SB) = 190.
        # This is the correct all-in amount.
        p0_all_in_amount = env.state.stacks[0]
        self.assertEqual(p0_all_in_amount, 190, "P0's stack should be 190 after posting SB")

        state, done = env.step(2, amount=p0_all_in_amount) # P0 goes all-in
        self.assertFalse(done, "Hand should not be terminal yet, P1 must act")
        state, done = env.step(1) # P1 calls
        
        self.assertTrue(done, "Hand must be terminal after all-in and call")
        self.assertEqual(len(state.community), 5, "Board should be fully dealt")
        self.assertIsNotNone(state.winners, "A winner should be declared")
        self.assertEqual(state.win_reason, 'all_in_showdown')
        self.assertEqual(state.pot, 0, "Pot should be 0 after distribution")
        
        # Verify chip conservation. Total chips should be 200 + 2000 = 2200.
        self.assertEqual(sum(state.stacks), 2200, "Total chip count must be conserved")

    def test_side_pot_logic(self):
        """The ultimate test: 3 players, one short-stack all-in, creating a side pot."""
        hole_cards = [
            cards(['As', 'Ad']),  # Player 0 (Big Stack) - Best hand
            cards(['Ks', 'Kd']),  # Player 1 (Medium Stack) - Second best hand
            cards(['Qs', 'Qd'])   # Player 2 (Short Stack) - Worst hand
        ]
        stacks = [2000, 1000, 200]
        env = self._setup_hand(num_players=3, hole_cards=hole_cards, stacks=stacks, dealer_pos=0)

        # Action: P0 raises, P1 calls, P2 goes all-in, P0 & P1 call.
        # P2 all-in for 200. P1 calls 200. P0 calls 200.
        # Main Pot: 200*3 = 600. Eligible: P0, P1, P2.
        # P1 all-in for remaining 800. P0 calls 800.
        # Side Pot: 800*2 = 1600. Eligible: P0, P1.
        
        # Simulate a final state for showdown
        env.state.terminal = True
        env.state.active = [True, True, True]
        env.state.starting_stacks_this_hand = [2000, 1000, 200]
        env.state.stacks = [1000, 0, 0] # Stacks after betting
        env.state.pot = 2200
        env.state.community = cards(['2c', '3d', '4h', '5s', '7h']) # No one improves

        # Manually trigger pot distribution
        hand_ranks = env._calculate_showdown_hand_ranks()
        env._distribute_pot_with_side_pots(hand_ranks)
        
        final_stacks = env.state.stacks
        
        # P0 had AA, wins both pots.
        # Main pot (600) + Side pot (1600) = 2200.
        # P0 started with 2000, invested 1000 -> 1000 left.
        # P0 final stack = 1000 (remaining) + 2200 (winnings) = 3200.
        self.assertEqual(final_stacks[0], 3200, "P0 should win both pots")
        self.assertEqual(final_stacks[1], 0, "P1 should be eliminated")
        self.assertEqual(final_stacks[2], 0, "P2 should be eliminated")
        
    def test_split_pot(self):
        """Test a simple split pot scenario."""
        hole_cards = [cards(['As', '2d']), cards(['Ah', '3c'])]
        community = cards(['Kc', 'Kd', 'Qh', 'Qs', 'Jd']) # Board is two pair, K's and Q's
        env = self._setup_hand(hole_cards=hole_cards, community=community, stacks=[0, 0])
        env.state.pot = 1000
        env.state.starting_stacks_this_hand = [500, 500]
        env.state.active = [True, True]
        env.state.terminal = True
        
        # Best hand for both is KKQQA. Split pot.
        hand_ranks = env._calculate_showdown_hand_ranks()
        env._distribute_pot_with_side_pots(hand_ranks)
        final_stacks = env.state.stacks
        
        self.assertEqual(final_stacks[0], 500, "P0 should get half the pot")
        self.assertEqual(final_stacks[1], 500, "P1 should get half the pot")

    def test_tournament_elimination_and_win(self):
        """Test that players are eliminated and a winner is declared."""
        env = TexasHoldemEnv(num_players=2, starting_stack=100)
        
        # Manually set a state where P1 is about to lose
        env.state.stacks = [200, 0]
        env.state.surviving_players = [0, 1]
        
        # This function is called at the end of a hand
        env._check_tournament_winner()
        
        # Check that P1 was eliminated
        self.assertEqual(env.state.surviving_players, [0], "Only player 0 should survive")
        self.assertTrue(env.state.terminal, "Game should be terminal")
        self.assertEqual(env.state.win_reason, 'tournament_winner', "Win reason should be set")
        
        # Now, calling reset should start a completely new tournament.
        state = env.reset()
        
        # Check that the state has been properly reset
        self.assertFalse(state.terminal, "Resetting a finished game should start a new, non-terminal game")
        self.assertIsNone(state.win_reason, "Win reason should be cleared on reset")
        self.assertEqual(sum(state.starting_stacks_this_hand), 200, "Should reset to starting stacks (100 * 2)")

if __name__ == '__main__':
    unittest.main()

