# app/poker_core.py
# Core poker classes and utilities - foundation for the poker AI system
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
from itertools import combinations
import random
from treys import Evaluator, Card

class Deck:
    """
    Standard 52-card deck with shuffling and dealing capabilities.
    Maintains compatibility with existing integer-based system.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self):
        """Reset deck to full 52 cards and shuffle."""
        # Create all 52 cards as integers (for compatibility)
        self.cards = list(range(52))
        self.rng.shuffle(self.cards)
    
    def set_deterministic_deck(self, card_strs: List[str]):
        if len(set(card_strs)) != 52:
            raise ValueError("Deterministic deck must contain exactly 52 unique cards.")
        self.cards = [string_to_card_id(s) for s in reversed(card_strs)]
    
    def deal(self, num_cards: int = 1) -> List[int]:
        """Deal specified number of cards as integers."""
        if num_cards > len(self.cards):
            raise ValueError(f"Cannot deal {num_cards} cards, only {len(self.cards)} remaining")
        
        dealt = []
        for _ in range(num_cards):
            dealt.append(self.cards.pop())
        return dealt
    
    def deal_card(self) -> int:
        """Deal a single card as integer."""
        return self.deal(1)[0]
    
    def cards_remaining(self) -> int:
        """Number of cards remaining in deck."""
        return len(self.cards)
    
    def peek_next(self, num_cards: int = 1) -> List[int]:
        """Peek at next cards without dealing them."""
        if num_cards > len(self.cards):
            raise ValueError(f"Cannot peek {num_cards} cards, only {len(self.cards)} remaining")
        return self.cards[-num_cards:]


@dataclass
class GameState:
    """
    Centralized container for all game state variables.
    Separates game data from game logic for cleaner architecture.
    """
    
    # Basic game configuration
    num_players: int
    starting_stack: int
    small_blind: int
    big_blind: int
    
    # Cards and community
    hole_cards: List[List[int]]  # Each player's hole cards as integers
    community: List[int]         # Community cards as integers
    
    # Money and betting
    stacks: List[int]            # Each player's current stack
    current_bets: List[int]      # Current round bets
    pot: int                     # Current pot size
    starting_pot_this_round: int # Pot size at start of current betting round
    starting_stacks_this_hand: List[int] # Each player's stack at start of hand (before blinds)
    
    # Player states
    active: List[bool]           # Is player still in hand
    all_in: List[bool]           # Is player all-in
    acted: List[bool]            # Has player acted this round
    surviving_players: List[int] # List of player IDs still in tournament
    
    # Game flow
    stage: int                   # 0=preflop, 1=flop, 2=turn, 3=river
    dealer_pos: int              # Dealer button position
    sb_pos: int                  # Small blind position
    bb_pos: int                  # Big blind position
    to_move: int                 # Current player to act
    
    # Betting context
    initial_bet: Optional[int]   # Initial bet amount for current round
    last_raise_size: int         # Size of last raise
    last_raiser: Optional[int]   # Player who made last raise
    
    # Game status
    terminal: bool               # Is hand over
    winners: Optional[List[int]] # Winners (if terminal)
    win_reason: Optional[str]    # How hand ended
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        return GameState(
            num_players=self.num_players,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            hole_cards=[cards.copy() for cards in self.hole_cards],
            community=self.community.copy(),
            stacks=self.stacks.copy(),
            current_bets=self.current_bets.copy(),
            pot=self.pot,
            starting_pot_this_round=self.starting_pot_this_round,
            starting_stacks_this_hand=self.starting_stacks_this_hand.copy(), #
            active=self.active.copy(),
            all_in=self.all_in.copy(),
            acted=self.acted.copy(),
            surviving_players=self.surviving_players.copy(), #
            stage=self.stage,
            dealer_pos=self.dealer_pos,
            sb_pos=self.sb_pos,
            bb_pos=self.bb_pos,
            to_move=self.to_move,
            initial_bet=self.initial_bet,
            last_raise_size=self.last_raise_size,
            last_raiser=self.last_raiser,
            terminal=self.terminal,
            winners=self.winners.copy() if self.winners else None,
            win_reason=self.win_reason
        )
    
    def get_min_raise_amount(self) -> Optional[int]:
        """Calculates the minimum legal raise amount for the current player."""
        player = self.to_move
        if not self.active[player] or self.stacks[player] == 0:
            return None

        current_max_bet = max(self.current_bets)
        amount_to_call = current_max_bet - self.current_bets[player]

        # If the player doesn't have enough to even call, they can't raise.
        if self.stacks[player] <= amount_to_call:
            return None

        # The minimum increment for a raise is the size of the last raise,
        # or the big blind if no raise has occurred yet this street.
        min_raise_increment = max(self.last_raise_size, self.big_blind)
        
        # This is the total additional amount the player must put in.
        required_additional_amount = amount_to_call + min_raise_increment

        # If the player can afford a full raise, that's the minimum.
        # Otherwise, their minimum (and only) raise is to go all-in.
        if self.stacks[player] >= required_additional_amount:
            return required_additional_amount
        else:
            return self.stacks[player]  # All-in is the only option

    def get_legal_actions(self) -> List[int]:
        """Get the complete list of legal actions for the current player."""
        player = self.to_move
        if self.terminal or not self.active[player] or self.all_in[player]:
            return []

        legal = []
        current_max_bet = max(self.current_bets)
        amount_to_call = current_max_bet - self.current_bets[player]

        # Action 0: Fold
        # Folding is legal only if there is a bet to call.
        if amount_to_call > 0:
            legal.append(0)

        # Action 1: Check/Call
        # This is always a legal option (checking if amount_to_call is 0).
        legal.append(1)

        # Action 2: Bet/Raise
        # A raise is legal if the new get_min_raise_amount method returns a value.
        if self.get_min_raise_amount() is not None:
            legal.append(2)
            
        return sorted(legal)

class HandEvaluator:
    def __init__(self):
        # Create the evaluator instance once
        self.evaluator = Evaluator()

    def best_hand_rank(self, hand: list[int], board: list[int]) -> int:
        if len(hand) + len(board) < 5:
            return 9999

        # Convert cards to the treys format
        hand_treys = [Card.new(card_to_string(c)) for c in hand]
        board_treys = [Card.new(card_to_string(c)) for c in board]

        # No more splitting! Just pass the arguments directly.
        return -self.evaluator.evaluate(board_treys, hand_treys)

    def get_rank_string(self, rank: int) -> str:
        """Converts a treys integer rank into a human-readable string."""
        if rank == 9999: # Handle the case of not enough cards
            return "No Rank"

        raw_treys_rank = -rank
        # Class (e.g., 1 for SF, 4 for Flush)
        rank_class = self.evaluator.get_rank_class(raw_treys_rank)
        # Convert the class to a string (e.g., "Straight Flush")
        return self.evaluator.class_to_string(rank_class)


def get_betting_order(seat_id: int, dealer_pos: int, num_players: int) -> int:
    """
    Calculates a betting order where higher is better (acts later).
    SB=0, BB=1, ..., Button=num_players-1
    
    This is a centralized utility function used across multiple analyzers
    to ensure consistent betting order calculations.
    """
    relative_position = (seat_id - dealer_pos) % num_players
    return (relative_position - 1 + num_players) % num_players

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def card_to_string(card_id: int) -> str:
    """Convert card ID (0-51) to string representation like '2s'."""
    if card_id < 0 or card_id > 51:
        return 'As'  # Fallback
        
    rank_id = card_id // 4
    suit_id = card_id % 4
    
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['c', 'd', 'h', 's']
    
    return ranks[rank_id] + suits[suit_id]

def string_to_card_id(card_str: str) -> int:
    """Convert card string like '2s' to card ID (0-51)."""
    if len(card_str) < 2:
        return 0
    rank_char = card_str[0]
    suit_char = card_str[1]
    
    rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
               '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
    
    rank = rank_map.get(rank_char, 0)
    suit = suit_map.get(suit_char, 0)
    
    return rank * 4 + suit

def get_street_name(stage: int) -> str:
    """Convert stage number to street name."""
    stage_map = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
    return stage_map.get(stage, 'preflop')

