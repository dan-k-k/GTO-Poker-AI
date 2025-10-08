# app/poker_feature_schema.py
import numpy as np
from dataclasses import dataclass, fields, field

# --- Reusable Blocks ---
# feature dim length prone to changing!

@dataclass
class BettingRoundFeatures:
    """A summary of betting actions for a single street."""
    my_bets_opened: float = 0.0      # How many times we made the FIRST bet in this round
    my_raises_made: float = 0.0      # How many times we raised over another bet this round
    opp_bets_opened: float = 0.0     # How many times the opponent opened this round
    opp_raises_made: float = 0.0     # How many times the opponent raised this round
    actions_on_street: float = 0.0   # Total actions taken on this street

@dataclass
class DynamicFeatures:
    """Features that change with EVERY action and must be recalculated each turn."""
    hand_strength: float = 0.0       # The "intelligent" equity
    my_stack_bb: float = 0.0
    opp_stack_bb: float = 0.0
    pot_bb: float = 0.0
    effective_stack_bb: float = 0.0
    pot_odds: float = 0.0            # Pot odds for calling current bet (call amount / final pot)
    bet_faced_ratio: float = 0.0     # Size of bet faced relative to current pot
    spr: float = 0.0                 # Stack-to-Pot Ratio (effective stack / pot)
    player_has_initiative: float = 0.0  # Whether player was last aggressor
    
    # Current betting round summary for the in-progress street
    current_betting_round: BettingRoundFeatures = field(default_factory=BettingRoundFeatures)

@dataclass
class StreetFeatures:
    """Features that are STATIC for a given street (CARD-BASED ONLY)."""
    random_strength: float = 0.0        # Equity vs a random hand
    made_hand_rank_pair: float = 0.0      # Is our best hand at least a pair?
    made_hand_rank_twopair: float = 0.0   # ... at least two pair?
    made_hand_rank_trips: float = 0.0     # ... etc.
    made_hand_rank_straight: float = 0.0
    made_hand_rank_flush: float = 0.0
    made_hand_rank_fullhouse: float = 0.0
    # Board-only made hand ranks
    board_made_rank_pair: float = 0.0      # Does the board itself have at least a pair?
    board_made_rank_twopair: float = 0.0   # ... at least two pair?
    board_made_rank_trips: float = 0.0     # ... etc.
    board_made_rank_straight: float = 0.0
    board_made_rank_flush: float = 0.0
    board_made_rank_fullhouse: float = 0.0
    
    # Board draw features - public information about the board's drawiness
    is_3card_flush_draw_board: float = 0.0  # e.g., Kh 7h 2h (runner-runner)
    is_4card_flush_draw_board: float = 0.0  # e.g., Kh 7h 2h Ah (immediate threat)
    is_3card_straight_draw_board: float = 0.0  # e.g., 8-9-J (gutshot/runner-runner)
    is_4card_straight_draw_board: float = 0.0  # e.g., 8-9-T-J (open-ended)
    
    # Player draw features - private information about our draws
    has_3card_flush_draw: float = 0.0  # Player has hole + board = 3 suited cards
    has_4card_flush_draw: float = 0.0  # Player has hole + board = 4 suited cards
    has_3card_straight_draw: float = 0.0
    has_4card_straight_draw: float = 0.0
    
    # Blocker features - how our hole cards block opponent draws
    has_flush_blocker: float = 0.0      
    straight_blocker_value: float = 0.0 

@dataclass
class HandFeatures:
    """Features that are STATIC for the entire hand, set once at the beginning."""
    is_button: float = 0.0              # 1.0 if we are the button, 0.0 otherwise
    is_pair: float = 0.0
    is_suited: float = 0.0
    high_card_rank: float = 0.0         # Rank of our highest card (0-1, A=1)
    low_card_rank: float = 0.0          # Rank of our lowest card (0-1, 2=0)


# --- The Main Schema ---

@dataclass
class PokerFeatureSchema:
    """
    The main, self-documenting schema for our entire feature vector.
    This class defines the structure, and the `to_vector` method flattens it
    for the neural network.
    """
    hand: HandFeatures = field(default_factory=HandFeatures)
    
    dynamic: DynamicFeatures = field(default_factory=DynamicFeatures)
    
    # Static card features for each street
    preflop_cards: StreetFeatures = field(default_factory=StreetFeatures)
    flop_cards: StreetFeatures = field(default_factory=StreetFeatures)
    turn_cards: StreetFeatures = field(default_factory=StreetFeatures)
    river_cards: StreetFeatures = field(default_factory=StreetFeatures)
    
    # Finalized betting summary for past streets
    preflop_betting: BettingRoundFeatures = field(default_factory=BettingRoundFeatures)
    flop_betting: BettingRoundFeatures = field(default_factory=BettingRoundFeatures)
    turn_betting: BettingRoundFeatures = field(default_factory=BettingRoundFeatures)

    def to_vector(self) -> np.ndarray:
        """Flattens the entire nested schema into a 1D NumPy array using recursion."""
        vector_parts = []
        self._flatten_recursively(self, vector_parts)
        return np.array(vector_parts, dtype=np.float32)

    def _flatten_recursively(self, item, vector_parts):
        """Helper method to recursively flatten nested dataclasses."""
        if hasattr(item, '__dataclass_fields__'):
            for field_info in fields(item):
                value = getattr(item, field_info.name)
                self._flatten_recursively(value, vector_parts)
        else:
            vector_parts.append(item)

    @staticmethod
    def get_vector_size() -> int:
        """Dynamically calculates the total size of the feature vector."""
        # Create a default instance and get the length of its vector form
        return len(PokerFeatureSchema().to_vector())
    
