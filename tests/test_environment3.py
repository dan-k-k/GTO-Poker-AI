# tests/test_environment3.py
# find . -type d -name "__pycache__" -exec rm -r {} +
# python -m unittest tests.test_environment3
import unittest
from app.TexasHoldemEnv import TexasHoldemEnv, GameState
from app.poker_core import string_to_card_id

def cards(card_strs: list[str]) -> list[int]:
    """Helper to convert a list of card strings to integer IDs for tests."""
    return [string_to_card_id(s) for s in card_strs]

class TestHyperRigorousHoldemEnv(unittest.TestCase):
    """
    A test suite for "once-in-a-thousand-hands" edge cases in the TexasHoldemEnv.
    This suite focuses on complex side pots, blind/stack scenarios, and state machine integrity.
    """

    def setUp(self):
        """This method is called before each test function."""
        print(f"\n--- Running Rigorous Test: {self.id()} ---")

    def _setup_hand(self, num_players=2, hole_cards=None, community=None, stacks=None, dealer_pos=0):
        """A powerful helper to manually set up a specific game state for testing."""
        env = TexasHoldemEnv(num_players=num_players, starting_stack=2000, small_blind=10, big_blind=20)
        
        if stacks is None:
            stacks = [2000] * num_players
        if hole_cards is None:
            # Provide enough default cards for up to 4 players
            all_cards = [cards(['As', 'Ks']), cards(['Ad', 'Kd']), cards(['Ac', 'Kc']), cards(['Ah', 'Kh'])]
            hole_cards = all_cards[:num_players]
            
        starting_stacks_for_hand = stacks.copy()
            
        survivors = [i for i, s in enumerate(stacks) if s > 0]
        if len(survivors) == 2:
            sb_pos = dealer_pos
            bb_pos = (dealer_pos + 1) % num_players
            to_move = sb_pos
        else:
            # This logic assumes players are indexed contiguously, which works
            sb_pos = (dealer_pos + 1) % num_players
            bb_pos = (dealer_pos + 2) % num_players
            to_move = (dealer_pos + 3) % num_players
            
        current_bets = [0] * num_players
        sb_amount = min(10, stacks[sb_pos])
        bb_amount = min(20, stacks[bb_pos])
        
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

    def test_multiple_side_pots_with_split_pot(self):
        """
        Tests a 4-way all-in with two side pots, where the first side pot is split.
        P3 (short) -> Main Pot
        P1, P2 (medium) -> Side Pot 1
        P0 (deep) -> Side Pot 2
        """
        hole_cards = [
            cards(['Ac', 'Ad']),  # P0 (Deep Stack) - Wins Side Pot 2
            cards(['Kc', '9c']),  # P1 (Medium Stack) - Splits Main Pot & Side Pot 1
            cards(['Kh', '9h']),  # P2 (Medium Stack) - Splits Main Pot & Side Pot 1
            cards(['Qc', 'Qd'])   # P3 (Short Stack) - Loses everything
        ]
        stacks = [2000, 1000, 1000, 200]
        env = self._setup_hand(num_players=4, hole_cards=hole_cards, stacks=stacks, dealer_pos=3)

        # Simulate a final state for showdown after all bets are in
        env.state.terminal = True
        env.state.active = [True, True, True, True]
        env.state.starting_stacks_this_hand = [2000, 1000, 1000, 200]
        env.state.stacks = [0, 0, 0, 0] # All chips are in the pot
        env.state.pot = 4200 # 2000 + 1000 + 1000 + 200
        env.state.community = cards(['Ks', 'Kd', '5d', '5c', '2s'])

        # Manually trigger pot distribution
        hand_ranks = env._calculate_showdown_hand_ranks()
        env._distribute_pot_with_side_pots(hand_ranks)
        
        final_stacks = env.state.stacks

        # Expected Outcome:
        # Main Pot (200 * 4 = 800): P1 and P2 split (Trip Kings > Queens). Each gets 400.
        # Side Pot 1 (800 * 3 = 2400): P1 and P2 split (Trip Kings > Aces Up). Each gets 1200.
        # Side Pot 2 (1000 * 1 = 1000): P0 wins uncontested.
        self.assertEqual(final_stacks[0], 1000, "P0 should win Side Pot 2")
        self.assertEqual(final_stacks[1], 1600, "P1 should get 400 (main) + 1200 (side 1)")
        self.assertEqual(final_stacks[2], 1600, "P2 should get 400 (main) + 1200 (side 1)")
        self.assertEqual(final_stacks[3], 0, "P3 should be eliminated")

    def test_uncalled_bet_is_returned_via_pot_logic(self):
        """Tests that an uncalled bet is correctly returned to the bettor."""
        hole_cards = [cards(['As', 'Ks']), cards(['Ad', 'Kd'])]
        env = self._setup_hand(num_players=2, hole_cards=hole_cards, stacks=[1000, 300])
        
        # Simulate a state where P0 bet 500 and P1 called all-in for 300
        env.state.terminal = True
        env.state.active = [True, True]
        env.state.starting_stacks_this_hand = [1000, 300]
        env.state.stacks = [500, 0] # Stacks after betting
        env.state.pot = 800 # 500 from P0 + 300 from P1
        env.state.community = cards(['2c', '3d', '4h', '5s', '7h']) # No one improves, split pot

        hand_ranks = env._calculate_showdown_hand_ranks()
        env._distribute_pot_with_side_pots(hand_ranks)
        
        # Expected Outcome:
        # The main pot is 300 * 2 = 600. P0 and P1 split this, each gets 300.
        # The uncalled 200 from P0 is returned to P0 as a "pot" they win uncontested.
        # P0 Final Stack: 500 (remaining) + 300 (split) + 200 (returned bet) = 1000.
        # P1 Final Stack: 0 (remaining) + 300 (split) = 300.
        self.assertEqual(env.state.stacks[0], 1000)
        self.assertEqual(env.state.stacks[1], 300)

    def test_walkover_hand_ends_correctly(self):
        """Test that the hand ends correctly when everyone folds to the big blind."""
        env = self._setup_hand(num_players=3, dealer_pos=0) # P1=SB, P2=BB, P0=UTG
        self.assertEqual(env.state.to_move, 0)
        
        state, done = env.step(action=0) # P0 (UTG) folds
        self.assertFalse(done)
        self.assertEqual(state.to_move, 1) # Action on P1 (SB)

        state, done = env.step(action=0) # P1 (SB) folds
        self.assertTrue(done, "Hand should be terminal after SB folds to BB")
        self.assertEqual(state.winners, [2], "Player 2 (BB) should win the pot")
        self.assertEqual(state.win_reason, 'fold')
        
        # P2 started with 2000, posted 20 BB. Wins the 10 SB. Final stack should be 1980 + 10 = 1990
        # The pot was 30. P2's final stack is start_stack - contribution + pot_winnings
        # 2000 - 20 + 30 = 2010. Let's trace stacks: SB=1990, BB=1980. After fold, BB gets pot of 30.
        # BB stack becomes 1980 + 30 = 2010. SB stack remains 1990. Correct.
        self.assertEqual(state.stacks[2], 2010)

    def test_action_when_bb_is_short_stacked_all_in(self):
        """Tests that a short-stacked BB doesn't change the legal call/raise amounts."""
        # Setup: P0=D/UTG, P1=SB, P2=BB. BB is short.
        env = self._setup_hand(num_players=3, stacks=[1000, 1000, 5], dealer_pos=0)
        
        self.assertTrue(env.state.all_in[2], "BB should be all-in")
        self.assertEqual(env.state.current_bets[2], 5, "BB's bet should be 5")
        self.assertEqual(env.state.current_bets[1], 10, "SB's bet should be 10")
        
        # Action is on P0.
        legal_actions = env.state.get_legal_actions()
        
        # Check that fold, call, and raise are all possible
        self.assertIn(0, legal_actions)
        self.assertIn(1, legal_actions)
        self.assertIn(2, legal_actions)
        
        # Now, verify the correct raise amounts by calling the specific state method
        min_raise_amount = env.state.get_min_raise_amount()
        max_raise_amount = env.state.stacks[env.state.to_move]
        
        # The highest bet to match is the SB's 10.
        # The minimum raise increment is the big blind size (20).
        # Therefore, the total amount P0 must bet to min-raise is 10 (call) + 20 (raise) = 30.
        self.assertEqual(min_raise_amount, 30, "Min raise should be call(10) + raise_increment(20)")
        
        # P0 has not posted any blinds, so their full stack is available.
        self.assertEqual(max_raise_amount, 1000, "Max raise should be P0's full stack")

    def test_simultaneous_elimination_results_in_tournament_winner(self):
        """Tests that a hand eliminating multiple players correctly ends the tournament."""
        hole_cards = [cards(['Ac', 'Ad']), cards(['Kc', 'Kd']), cards(['Qc', 'Qd'])]
        # Setup with starting stacks. The helper will correctly post the blinds.
        # P0 is Dealer. P1 is SB (100 chips), P2 is BB (100 chips).
        # After setup: SB posts 10, BB posts 20. Stacks: [2000, 90, 80]. Pot: 30. Turn: P0.
        env = self._setup_hand(num_players=3, hole_cards=hole_cards, stacks=[2000, 100, 100], dealer_pos=0)

        # Action: P0 (UTG) raises to 200.
        # This is an overbet to put both other players all-in.
        state, done = env.step(action=2, amount=200)
        self.assertFalse(done)
        self.assertEqual(state.to_move, 1, "Action should be on P1 (SB)")

        # Action: P1 (SB) calls all-in with their remaining 90 chips.
        state, done = env.step(action=1) # Call is sufficient, env calculates the amount.
        self.assertFalse(done)
        self.assertEqual(state.to_move, 2, "Action should be on P2 (BB)")

        # Action: P2 (BB) calls all-in with their remaining 80 chips.
        # This action closes the betting and should trigger the showdown automatically.
        state, done = env.step(action=1)
        
        # The hand is now over. The final state is returned.
        self.assertTrue(done, "Hand should be terminal after final all-in call")

        # Now, we assert the final state is correct.
        self.assertEqual(state.win_reason, 'tournament_winner')
        self.assertEqual(state.winners, [0], "Player 0 should be the sole winner")
        self.assertEqual(state.surviving_players, [0], "Only P0 should be in surviving_players")
        self.assertEqual(state.stacks[0], 2200, "P0 should have all the chips")
        self.assertEqual(state.stacks[1], 0, "P1 should be eliminated")
        self.assertEqual(state.stacks[2], 0, "P2 should be eliminated")

    def test_step_on_terminal_state_raises_error(self):
        """Tests that calling step() on a finished hand raises a ValueError."""
        env = self._setup_hand(num_players=2)
        
        # Finish the hand
        env.step(action=0) # Player 0 folds
        self.assertTrue(env.state.terminal)

        # Assert that the next action raises an error
        with self.assertRaises(ValueError, msg="Should raise ValueError on step() when terminal"):
            env.step(action=1)


if __name__ == '__main__':
    unittest.main()

