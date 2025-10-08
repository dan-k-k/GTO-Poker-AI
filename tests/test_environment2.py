# tests/test_environment2.py
# find . -type d -name "__pycache__" -exec rm -r {} +
# python -m unittest tests.test_environment2
import unittest
from app.TexasHoldemEnv import TexasHoldemEnv, GameState
from app.poker_core import string_to_card_id

def cards(card_strs: list[str]) -> list[int]:
    """Helper to convert a list of card strings to integer IDs for tests."""
    return [string_to_card_id(s) for s in card_strs]

class TestAdvancedTexasHoldemEnv(unittest.TestCase):
    """
    A second, more advanced test suite for the TexasHoldemEnv.
    This focuses on edge cases, illegal actions, complex tournament dynamics,
    and intricate showdown scenarios to ensure robustness.
    """

    def setUp(self):
        """This method is called before each test function."""
        print(f"\n--- Running Advanced Test: {self.id()} ---")

    def _setup_hand(self, num_players=2, hole_cards=None, community=None, stacks=None, dealer_pos=0):
        """A powerful helper to manually set up a specific game state for testing."""
        env = TexasHoldemEnv(num_players=num_players, starting_stack=2000, small_blind=10, big_blind=20)
        
        if stacks is None:
            stacks = [2000] * num_players
        if hole_cards is None:
            all_cards = [cards(['As', 'Ks']), cards(['Ad', 'Kd']), cards(['Ac', 'Kc']), cards(['Ah', 'Kh'])]
            hole_cards = all_cards[:num_players]
            
        starting_stacks_for_hand = stacks.copy()
            
        survivors = list(range(num_players))
        if len(survivors) == 2:
            sb_pos = dealer_pos
            bb_pos = (dealer_pos + 1) % 2
            to_move = sb_pos
        else:
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
            active=[True] * num_players, all_in=[s == 0 for s in stacks], acted=[False] * num_players,
            surviving_players=survivors, stage=0 if not community else 1, 
            dealer_pos=dealer_pos, sb_pos=sb_pos, bb_pos=bb_pos, to_move=to_move,
            initial_bet=20, last_raise_size=20, last_raiser=None, terminal=False,
            winners=None, win_reason=None
        )
        return env

    def test_illegal_actions_raise_errors(self):
        """Ensure the environment throws ValueError for illegal moves."""
        env = self._setup_hand(dealer_pos=0, stacks=[200, 200]) # P0 is SB, P1 is BB
        env.state.to_move = 0

        with self.assertRaises(ValueError, msg="Should reject betting more chips than available"):
            env.step(2, amount=9999) 

        # P0 has posted 10, faces 20. Min-raise is to 40. A raise of 10 (to 30 total) is illegal.
        # The 'amount' is additional chips. P0 would add 20 to raise to 30 total.
        with self.assertRaises(ValueError, msg="Should reject an illegal under-raise"):
            env.step(2, amount=20)

    def test_positions_rotate_correctly_after_elimination(self):
        """Verify dealer/blind positions skip eliminated players."""
        env = TexasHoldemEnv(num_players=4, starting_stack=1000)
        
        # Set a custom state: dealer is P0, P1 is eliminated
        env.state.dealer_pos = 0
        env.state.stacks = [500, 0, 1500, 2000]
        # Manually set surviving players before the reset
        env.state.surviving_players = [0, 2, 3] 

        # Call reset without arguments. The env will see the game is not
        # terminal and start a new hand, preserving the stacks.
        state = env.reset()
        
        # The new hand should skip player 1 for all positions
        self.assertEqual(state.dealer_pos, 2, "Dealer button should skip eliminated player 1 and move to P2")
        self.assertEqual(state.sb_pos, 3, "SB should be P3")
        self.assertEqual(state.bb_pos, 0, "BB should be P0, wrapping around")
        self.assertNotIn(1, state.surviving_players, "Player 1 should be removed from survivors list")

    def test_under_raise_all_in_does_not_reopen_action(self):
        """Test that a short all-in raise doesn't let prior actors re-raise, and the street ends correctly."""
        # Dealer=1. SB=2, BB=0. UTG (first to act) is P1.
        env = self._setup_hand(num_players=3, stacks=[1000, 1000, 55], dealer_pos=1)
        self.assertEqual(env.state.to_move, 1)

        # 1. P1 (UTG) raises to 60. This is the last aggressive action.
        state, _ = env.step(2, amount=60)
        self.assertEqual(state.to_move, 2)

        # 2. P2 (SB) goes all-in for 45 more (total bet of 55). This is an under-raise and does not reopen action.
        state, _ = env.step(2, amount=45)
        self.assertEqual(state.to_move, 0)

        # 3. P0 (BB) calls the 60.
        state, done = env.step(1)

        # The pre-flop action is now complete because P1's raise was only called.
        # The test should now assert that the game has advanced to the flop.
        self.assertFalse(done, "The hand should not be over, it should proceed to the flop.")
        self.assertEqual(state.stage, 1, "The game should have advanced to the flop (stage 1).")

        # Post-flop, the first active player to the left of the dealer (P1) is P0.
        # P2 is skipped because they are all-in.
        self.assertEqual(state.to_move, 0, "Action on the flop should start with Player 0.")

    def test_side_pot_with_different_winners(self):
        """A showdown where the side pot and main pot are won by different players."""
        hole_cards = [
            cards(['Ks', 'Kd']),  # P0 (Big Stack)
            cards(['As', 'Ad']),  # P1 (Medium Stack)
            cards(['Qs', 'Qd'])   # P2 (Short Stack)
        ]
        stacks = [2000, 1000, 200]
        env = self._setup_hand(num_players=3, hole_cards=hole_cards, stacks=stacks, dealer_pos=0)

        env.state.terminal = True
        env.state.active = [True, True, True]
        env.state.starting_stacks_this_hand = [2000, 1000, 200]
        env.state.stacks = [1000, 0, 0] # Stacks after betting
        env.state.pot = 2200
        env.state.community = cards(['Qc', 'Jd', '7s', '2d', '3h']) 

        hand_ranks = env._calculate_showdown_hand_ranks()
        env._distribute_pot_with_side_pots(hand_ranks)
        final_stacks = env.state.stacks
        
        # Expected Outcome with new board (Qc Jd 7s 2d 3h):
        # Ranks: P2 (Set of Qs) > P1 (Pair of As) > P0 (Pair of Ks)
        # Main Pot (600): P2 wins.
        # Side Pot (1600): P1 wins against P0.
        self.assertEqual(final_stacks[0], 1000, "P0 should have their remaining 1000 chips")
        self.assertEqual(final_stacks[1], 1600, "P1 should win the 1600 side pot")
        self.assertEqual(final_stacks[2], 600, "P2 should win the 600 main pot")

    def test_intricate_showdowns(self):
        """Test showdowns involving kickers and 'playing the board'."""
        
        # Scenario 1: Kicker problem
        hole_kicker = [cards(['As', 'Kc']), cards(['Ad', 'Qh'])]
        board_kicker = cards(['Ac', 'Js', '7d', '5h', '2s'])
        env_kicker = self._setup_hand(hole_cards=hole_kicker, community=board_kicker, stacks=[0, 0])
        env_kicker.state.pot = 1000
        env_kicker.state.starting_stacks_this_hand = [500, 500]
        env_kicker.state.active = [True, True]
        env_kicker.state.terminal = True
        
        ranks_kicker = env_kicker._calculate_showdown_hand_ranks()
        env_kicker._distribute_pot_with_side_pots(ranks_kicker)
        
        self.assertEqual(env_kicker.state.stacks[0], 1000, "P0 should win with the King kicker")
        self.assertEqual(env_kicker.state.stacks[1], 0, "P1 should lose due to weaker kicker")

        # Scenario 2: Playing the board for a split pot
        hole_board = [cards(['2s', '2c']), cards(['3d', '3h'])]
        board_board = cards(['Ts', 'Js', 'Qs', 'Ks', 'As']) # A royal flush on board
        env_board = self._setup_hand(hole_cards=hole_board, community=board_board, stacks=[0, 0])
        env_board.state.pot = 1000
        env_board.state.starting_stacks_this_hand = [500, 500]
        env_board.state.active = [True, True]
        env_board.state.terminal = True

        ranks_board = env_board._calculate_showdown_hand_ranks()
        env_board._distribute_pot_with_side_pots(ranks_board)

        self.assertEqual(env_board.state.stacks[0], 500, "P0 should split pot when playing the board")
        self.assertEqual(env_board.state.stacks[1], 500, "P1 should split pot when playing the board")

if __name__ == '__main__':
    unittest.main()

