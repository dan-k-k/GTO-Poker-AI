# app/feature_extractor.py 
import numpy as np
from app.poker_core import GameState, HandEvaluator
from app.poker_feature_schema import PokerFeatureSchema, StreetFeatures, BettingRoundFeatures
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.nfsp_components import NFSPAgent
from app._preflop_equity import PREFLOP_EQUITY

class FeatureExtractor:
    """
    A stateful feature extractor that populates a 3-tier schema:
    1. HandFeatures (once per hand)
    2. StreetFeatures (once per street)  
    3. DynamicFeatures (on every action)
    """
    VECTOR_SIZE = PokerFeatureSchema.get_vector_size()

    def __init__(self, seat_id: int, num_players: int = 2, random_equity_trials: int = 200):
        self.seat_id = seat_id
        self.num_players = num_players
        self.evaluator = HandEvaluator()
        self.random_equity_trials = random_equity_trials
        
        self.schema = PokerFeatureSchema()
        self._hand_features_extracted = False 
        self._street_features_extracted = [False, False, False, False] 
        
        # === BETTING HISTORY TRACKER ===
        self._betting_history = {
            'my_bets_opened': [0, 0, 0, 0],
            'my_raises_made': [0, 0, 0, 0],
            'opp_bets_opened': [0, 0, 0, 0],
            'opp_raises_made': [0, 0, 0, 0]
        }
        self._action_counts_this_street = [0, 0, 0, 0]  # Track total actions per street
        self.last_aggressor = None  # Track who was the last aggressor

    def _normalize_log(self, value, max_val):
        """Normalizes a value using log scaling, clipped at 0."""
        if value <= 0 or max_val <= 0:
            return 0.0
        return np.log1p(value) / np.log1p(max_val)

    def _normalize_clip(self, value, max_val):
        """Normalizes a value by clipping and dividing."""
        if max_val == 0: return 0.0
        return min(value, max_val) / max_val

    def new_hand(self):
        """Resets the schema for a new hand."""
        self.schema = PokerFeatureSchema()
        self._hand_features_extracted = False
        self._street_features_extracted = [False, False, False, False]  # Reset street cache
        # === RESET THE TRACKER ===
        self._betting_history = {
            'my_bets_opened': [0, 0, 0, 0],
            'my_raises_made': [0, 0, 0, 0],
            'opp_bets_opened': [0, 0, 0, 0],
            'opp_raises_made': [0, 0, 0, 0]
        }
        self._action_counts_this_street = [0, 0, 0, 0]
        self.last_aggressor = None

    def extract_features(self, state: GameState, agent: 'NFSPAgent' = None, skip_random_equity: bool = False) -> PokerFeatureSchema:
        """
        Main extraction method. Populates the clean schema with both historical and real-time data.
        """
        # 1. Extract Hand-Static features ONCE per hand
        if not self._hand_features_extracted:
            self._extract_hand_features(state)
            self._hand_features_extracted = True

        current_stage = state.stage

        # 2. Populate static card features and FINALIZED betting history for PAST streets
        for stage in range(current_stage):  # Note: Loops up to (but not including) the current stage
            # Get the right schema objects for the historical stage
            street_cards_schema = self._get_street_cards_schema(stage)
            betting_history_schema = self._get_betting_history_schema(stage)

            # Only extract static card features if not already done for this street
            if not self._street_features_extracted[stage]:
                self._extract_static_street_features(state, street_cards_schema, stage, skip_random_equity=skip_random_equity)
                self._street_features_extracted[stage] = True
            
            # Populate FINALIZED betting history from our internal tracker
            betting_history_schema.my_bets_opened = self._normalize_clip(self._betting_history['my_bets_opened'][stage], 10.0)
            betting_history_schema.my_raises_made = self._normalize_clip(self._betting_history['my_raises_made'][stage], 10.0)
            betting_history_schema.opp_bets_opened = self._normalize_clip(self._betting_history['opp_bets_opened'][stage], 10.0)
            betting_history_schema.opp_raises_made = self._normalize_clip(self._betting_history['opp_raises_made'][stage], 10.0)
            betting_history_schema.actions_on_street = self._normalize_clip(self._action_counts_this_street[stage], 10.0)

        # 3. Populate static card features for the CURRENT street
        current_street_cards_schema = self._get_street_cards_schema(current_stage)
        # Only extract static card features if not already done for this street
        if not self._street_features_extracted[current_stage]:
            self._extract_static_street_features(state, current_street_cards_schema, current_stage, skip_random_equity=skip_random_equity)
            self._street_features_extracted[current_stage] = True

        # 4. Populate ALL dynamic features for the IMMEDIATE decision
        self._update_dynamic_features(state, agent)

        return self.schema

    def _extract_hand_features(self, state: GameState):
        """Populates the `hand` part of the schema. Called only once per hand."""
        self.schema.hand.is_button = 1.0 if self.seat_id == state.dealer_pos else 0.0
        hole = state.hole_cards[self.seat_id]
        if len(hole) == 2:
            ranks = sorted([c // 4 for c in hole])
            suits = [c % 4 for c in hole]
            self.schema.hand.is_pair = 1.0 if ranks[0] == ranks[1] else 0.0
            self.schema.hand.is_suited = 1.0 if suits[0] == suits[1] else 0.0
            self.schema.hand.high_card_rank = ranks[1] / 12.0
            self.schema.hand.low_card_rank = ranks[0] / 12.0

    def _update_dynamic_features(self, state: GameState, agent: 'NFSPAgent'):
        """Populates the `dynamic` part of the schema. Called on every action."""
        dyn_schema = self.schema.dynamic
        bb_size = state.big_blind if state.big_blind > 0 else 1
        
        # Define normalization constants based on game parameters
        STARTING_STACK_BB = state.starting_stack / bb_size if bb_size > 0 else 200
        MAX_POT_BB = (state.starting_stack * self.num_players) / bb_size if bb_size > 0 else 400

        # --- Stacks and Pot ---
        my_stack_bb_raw = state.stacks[self.seat_id] / bb_size
        opp_stack_bb_raw = state.stacks[1 - self.seat_id] / bb_size
        pot_bb_raw = state.pot / bb_size
        effective_stack_bb_raw = min(my_stack_bb_raw, opp_stack_bb_raw)
        
        dyn_schema.my_stack_bb = self._normalize_log(my_stack_bb_raw, STARTING_STACK_BB)
        dyn_schema.opp_stack_bb = self._normalize_log(opp_stack_bb_raw, STARTING_STACK_BB)
        dyn_schema.pot_bb = self._normalize_log(pot_bb_raw, MAX_POT_BB)
        dyn_schema.effective_stack_bb = self._normalize_log(effective_stack_bb_raw, STARTING_STACK_BB)

        # === HAND STRENGTH ===
        # Case 1: Post-flop with an agent -> Use intelligent equity simulation
        if state.stage > 0 and agent is not None:
            historical_context = {
                'betting_history': self._betting_history,
                'action_counts': self._action_counts_this_street,
                'last_aggressor': self.last_aggressor
            }
            dyn_schema.hand_strength = agent._calculate_intelligent_equity(
                my_hole_cards=state.hole_cards[self.seat_id],
                community_cards=state.community,
                historical_context=historical_context
            )
        # Case 2: Post-flop WITHOUT an agent -> Fallback to the street's random equity
        elif state.stage > 0:
            # Get the schema for the current street
            current_street_cards_schema = self._get_street_cards_schema(state.stage)
            # Use the random_strength that was just calculated (or skipped)
            dyn_schema.hand_strength = current_street_cards_schema.random_strength
        # Case 3: Pre-flop -> Always use the static lookup
        else:
            dyn_schema.hand_strength = self._get_preflop_strength(state.hole_cards[self.seat_id])

        # --- Current betting round summary (for the in-progress street) ---
        stage = state.stage
        dyn_schema.current_betting_round.my_bets_opened = self._normalize_clip(self._betting_history['my_bets_opened'][stage], 10.0)
        dyn_schema.current_betting_round.my_raises_made = self._normalize_clip(self._betting_history['my_raises_made'][stage], 10.0)
        dyn_schema.current_betting_round.opp_bets_opened = self._normalize_clip(self._betting_history['opp_bets_opened'][stage], 10.0)
        dyn_schema.current_betting_round.opp_raises_made = self._normalize_clip(self._betting_history['opp_raises_made'][stage], 10.0)
        dyn_schema.current_betting_round.actions_on_street = self._normalize_clip(self._action_counts_this_street[stage], 10.0)
        
        # In heads-up, pre-flop, the BB is considered the initial aggressor before any action.
        initial_aggressor = state.bb_pos if self.last_aggressor is None and state.stage == 0 else self.last_aggressor
        if initial_aggressor is not None:
            dyn_schema.player_has_initiative = 1.0 if self.seat_id == initial_aggressor else 0.0
        else:
            dyn_schema.player_has_initiative = 0.0
        
        # Pot Odds, SPR, etc.
        amount_to_call = max(state.current_bets) - state.current_bets[self.seat_id]
        if amount_to_call > 0:
            final_pot = state.pot + amount_to_call
            if final_pot > 0:
                dyn_schema.pot_odds = amount_to_call / final_pot
            if state.pot > 0:
                dyn_schema.bet_faced_ratio = amount_to_call / state.pot
        else: # Reset if there's no bet to call
            dyn_schema.pot_odds = 0.0
            dyn_schema.bet_faced_ratio = 0.0

        # --- Stack-to-Pot Ratio (SPR) ---
        if state.stage > 0 and state.pot > 0:
            effective_stack = min(state.stacks[self.seat_id], state.stacks[1 - self.seat_id])
            spr = effective_stack / state.pot
            dyn_schema.spr = self._normalize_clip(spr, 20.0)  # Clip SPR at 20 and scale
        else:
            dyn_schema.spr = 0.0

    def _extract_static_street_features(self, state: GameState, street_schema: StreetFeatures, stage: int, skip_random_equity: bool = False):
        """
        A generic function to populate the STATIC features for ANY street schema.
        This is only called ONCE per street.
        """
        hole = state.hole_cards[self.seat_id]

        cards_on_street = {0: 0, 1: 3, 2: 4, 3: 5}
        community = state.community[:cards_on_street[stage]]

        # --- Get hand class using the treys evaluator ---
        player_hand_class = 0
        if len(hole + community) >= 5:
            # best_hand_rank returns a negative int; negate it back for treys
            player_raw_rank = -self.evaluator.best_hand_rank(hole + community)
            player_hand_class = self.evaluator.evaluator.get_rank_class(player_raw_rank)

        board_hand_class = 0
        if len(community) >= 5:
            board_raw_rank = -self.evaluator.best_hand_rank(community)
            board_hand_class = self.evaluator.evaluator.get_rank_class(board_raw_rank)

        # --- Populate Made Hand Features ---
        self._populate_made_hand_ranks(hole + community, street_schema, is_board=False)
        self._populate_made_hand_ranks(community, street_schema, is_board=True)

        # Only calculate random equity if the flag is not set
        if not skip_random_equity:

            if stage == 0:
                street_schema.random_strength = self._get_preflop_strength(hole)
            else:
                street_schema.random_strength = self._calculate_random_equity(hole, community)

        if stage == 0:
            return

        # --- BOARD DRAW FEATURES (Public Information) ---
        is_board_flush = board_hand_class in {1, 4}
        is_board_straight = board_hand_class in {1, 5}

        if len(community) >= 3 and not is_board_flush:
            community_suits = [c % 4 for c in community]
            suit_counts = np.bincount(community_suits)
            if len(suit_counts) > 0:
                max_suits = np.max(suit_counts)
                if max_suits == 3:
                    street_schema.is_3card_flush_draw_board = 1.0
                elif max_suits == 4:
                    street_schema.is_4card_flush_draw_board = 1.0

        if len(community) >= 3 and not is_board_straight:
            three_q, four_q = self._get_straight_draw_qualities(community)
            street_schema.is_3card_straight_draw_board = three_q
            street_schema.is_4card_straight_draw_board = four_q

        # --- PLAYER DRAW FEATURES (Private Information) ---
        is_player_flush = player_hand_class in {1, 4}
        is_player_straight = player_hand_class in {1, 5}

        # Player's personal flush draws
        if not is_player_flush:
            all_cards = hole + community
            all_suits = [c % 4 for c in all_cards]
            suit_counts = np.bincount(all_suits)
            if len(suit_counts) > 0:
                max_suits = np.max(suit_counts)
                if max_suits == 3:
                    street_schema.has_3card_flush_draw = 1.0
                elif max_suits == 4:
                    street_schema.has_4card_flush_draw = 1.0
        # --- Quality-based player straight draws ---
        if not is_player_straight:
            three_q, four_q = self._get_straight_draw_qualities(hole + community)
            street_schema.has_3card_straight_draw = three_q
            street_schema.has_4card_straight_draw = four_q

        # --- BLOCKER FEATURES ---
        # Calculate how our private hole cards block the public board draws
        hole_ranks = [c // 4 for c in hole]
        hole_suits = [c % 4 for c in hole]

        # 1. Flush Blockers
        street_schema.has_flush_blocker = 0.0

        community_suits = [c % 4 for c in community]
        hole_suits = [c % 4 for c in hole]

        if len(community_suits) > 1:
            suit_counts = np.bincount(community_suits)
            max_suit_count = np.max(suit_counts)
            draw_suit = np.argmax(suit_counts)

            # Holding exactly one card of the potential draw suit
            if hole_suits.count(draw_suit) == 1:
                
                # Scenario for an IMMEDIATE draw blocker
                if max_suit_count == 3:
                    street_schema.has_flush_blocker = 1.0
                
                # Scenario for a BACKDOOR draw blocker
                elif max_suit_count == 2:
                    street_schema.has_flush_blocker = 0.5
        
        # 2. Straight Blockers
        # Default the feature to 0.0
        street_schema.straight_blocker_value = 0.0

        # First, check that we don't already have a straight
        is_player_straight = player_hand_class in {1, 5} 

        if not is_player_straight and len(community) >= 3:
            draw_type, completing_ranks = self._find_best_board_straight_draw(community)
            
            if draw_type:
                hole_ranks = [c // 4 for c in hole]
                has_blocker = any(rank in completing_ranks for rank in hole_ranks)

                if has_blocker:
                    if draw_type == 'OESD':
                        street_schema.straight_blocker_value = 1.0
                    elif draw_type == 'gutshot':
                        street_schema.straight_blocker_value = 0.5
            
    def _populate_made_hand_ranks(self, cards: list, street_schema: StreetFeatures, is_board: bool = False):
        """
        Calculates the made hand rank from a list of cards and updates the
        appropriate fields in the provided schema object.
        """
        if not cards:
            return

        prefix = 'board_made_rank_' if is_board else 'made_hand_rank_'
        # --- Part 1: Component checks (work on any number of cards) ---
        ranks = [c // 4 for c in cards]
        counts = {r: ranks.count(r) for r in set(ranks)}
        freq = sorted(counts.values(), reverse=True)

        if 4 in freq: # Quads are trips and a pair
            setattr(street_schema, f"{prefix}trips", 1.0)
            setattr(street_schema, f"{prefix}pair", 1.0)
        elif 3 in freq: # Trips is a pair
            setattr(street_schema, f"{prefix}trips", 1.0)
            setattr(street_schema, f"{prefix}pair", 1.0)
            if 2 in freq: # This means it's a full house
                setattr(street_schema, f"{prefix}twopair", 1.0)
        elif freq and freq[0] == 2: # At least one pair
            setattr(street_schema, f"{prefix}pair", 1.0)
            if len(freq) > 1 and freq[1] == 2: # This means it's two pair
                setattr(street_schema, f"{prefix}twopair", 1.0)

        # --- Part 2: 5-card hand checks ---
        if len(cards) < 5:
            return # Not enough cards for straights, flushes, etc.

        # 1. Get the single integer rank from our helper.
        raw_rank = -self.evaluator.best_hand_rank(cards)

        # 2. Use the treys evaluator to get the hand class (1=SF, 2=Quads, ..., 9=HC).
        hand_class = self.evaluator.evaluator.get_rank_class(raw_rank) # treys' evaluator

        # 3. Set flags based on the hand class returned by treys.
        if hand_class == 1: # Straight Flush
            setattr(street_schema, f"{prefix}straight", 1.0)
            setattr(street_schema, f"{prefix}flush", 1.0)
        elif hand_class == 3: # Full House
            setattr(street_schema, f"{prefix}fullhouse", 1.0)
        elif hand_class == 4: # Flush
            setattr(street_schema, f"{prefix}flush", 1.0)
        elif hand_class == 5: # Straight
            setattr(street_schema, f"{prefix}straight", 1.0)

# --------------------------------------------------------
# --------------------------------------------------------

    def _get_straight_draw_qualities(self, cards: list) -> tuple:
        """
        Analyzes cards to find the quality of the best 3 and 4-card straight draws.
        Quality: 1.0 for consecutive (strong), 0.5 for gapped (weaker).
        Returns a tuple: (three_card_draw_quality, four_card_draw_quality)
        """
        ranks = sorted(list(set([c // 4 for c in cards])))
        if 12 in ranks: ranks.insert(0, -1) # Handle Ace-low

        three_card_quality = 0.0
        four_card_quality = 0.0

        # 1. Analyze 4-card draws (OESD vs Gutshot)
        if len(ranks) >= 4:
            for i in range(len(ranks) - 3):
                four_ranks = ranks[i:i+4]
                span = four_ranks[3] - four_ranks[0]
                if span == 3: # e.g., 5,6,7,8 -> span is 3. Open-ended.
                    four_card_quality = 1.0
                    break # Found the best possible 4-card draw
                elif span == 4: # e.g., 5,6,8,9 -> span is 4. Gutshot.
                    four_card_quality = max(four_card_quality, 0.5)
        
        # 2. Analyze 3-card draws (consecutive vs gapped)
        if len(ranks) >= 3:
            for i in range(len(ranks) - 2):
                three_ranks = ranks[i:i+3]
                span = three_ranks[2] - three_ranks[0]
                if span == 2: # e.g., 4,5,6 -> span is 2. 3-consecutive.
                    three_card_quality = max(three_card_quality, 1.0)
                # e.g., 4,5,7 (span 3) or 4,6,8 (span 4) are gapped
                elif span <= 4:
                    three_card_quality = max(three_card_quality, 0.5)

        return three_card_quality, four_card_quality

    def _find_best_board_straight_draw(self, community_cards: list) -> tuple:
        """
        Analyzes community cards to find the best possible straight draw.
        Returns: (draw_type, completing_ranks)
        - draw_type: 'OESD', 'gutshot', or None
        - completing_ranks: A list of card ranks that would complete the draw.
        """
        if len(community_cards) < 3:
            return None, []

        ranks = sorted(list(set([c // 4 for c in community_cards])))
        # Add Ace for A-2-3-4-5 straights
        if 12 in ranks:
            ranks.insert(0, -1) # Use -1 for the low Ace

        # 1. Prioritize 4-card open-ended draws (e.g., 5-6-7-8)
        for i in range(len(ranks) - 3):
            four_ranks = ranks[i:i+4]
            if four_ranks[3] - four_ranks[0] == 3: # Consecutive
                low_rank = four_ranks[0]
                high_rank = four_ranks[3]
                completing = []
                if low_rank > -1: completing.append(low_rank - 1)
                if high_rank < 12: completing.append(high_rank + 1)
                # Handle A-2-3-4 case
                if low_rank == 0 and high_rank == 3: completing.append(12)
                return 'OESD', completing

        # 2. Check for 4-card gutshot draws (e.g., 5-6-8-9)
        for i in range(len(ranks) - 3):
            four_ranks = ranks[i:i+4]
            if four_ranks[3] - four_ranks[0] == 4 and len(set(four_ranks)) == 4:
                # The missing rank is the completing card
                all_possible = set(range(four_ranks[0], four_ranks[3] + 1))
                completing = list(all_possible - set(four_ranks))
                return 'gutshot', completing
        
        # 3. Check for 3-card open-ended draws (e.g., 5-6-7)
        for i in range(len(ranks) - 2):
            three_ranks = ranks[i:i+3]
            if three_ranks[2] - three_ranks[0] == 2: # Consecutive
                # This is a simplification. Assume it's open-ended.
                return 'OESD', [three_ranks[0] - 1, three_ranks[2] + 1]

        # 4. Check for 3-card gutshot draws (e.g., 5-7-8)
        for i in range(len(ranks) - 2):
            three_ranks = ranks[i:i+3]
            if three_ranks[2] - three_ranks[0] in [3, 4]: # One or two gaps
                # This is a simplification, but captures the essence
                return 'gutshot', [r for r in range(three_ranks[0]+1, three_ranks[2])]

        return None, []

    def _calculate_random_equity(self, my_hole: list, community: list) -> float:
        """
        Fast, simple Monte Carlo equity calculation against a single random opponent hand.
        """
        # Use the value stored from the config file
        trials = self.random_equity_trials

        if not my_hole:
            return 0.5

        wins = 0
        deck = [c for c in range(52) if c not in my_hole and c not in community]

        for _ in range(trials):
            if len(deck) < 2: break
            
            shuffled_deck = np.random.permutation(deck)
            opp_hole = list(shuffled_deck[:2])
            
            remaining_deck = shuffled_deck[2:]
            
            num_cards_to_deal = 5 - len(community)
            if len(remaining_deck) < num_cards_to_deal: continue

            board_runout = remaining_deck[:num_cards_to_deal]
            final_board = community + list(board_runout)
            
            my_rank = self.evaluator.best_hand_rank(my_hole + final_board)
            opp_rank = self.evaluator.best_hand_rank(opp_hole + final_board)

            if my_rank > opp_rank: wins += 1.0
            elif my_rank == opp_rank: wins += 0.5
        
        return wins / trials if trials > 0 else 0.5
    
    def _get_preflop_equity_key(self, hole_cards: list) -> str:
        """
        Converts two integer hole cards into a standardized string key for equity lookup.
        Example: [51, 47] -> "AA", [49, 45] -> "AKs", [49, 44] -> "AKo"
        """
        ranks_map = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}
        
        ranks = sorted([c // 4 for c in hole_cards], reverse=True)
        suits = [c % 4 for c in hole_cards]
        
        rank1_char = ranks_map[ranks[0]]
        rank2_char = ranks_map[ranks[1]]

        if rank1_char == rank2_char:
            return f"{rank1_char}{rank2_char}" # Pocket pair (e.g., "AA", "77")
        
        is_suited = 's' if suits[0] == suits[1] else 'o'
        return f"{rank1_char}{rank2_char}{is_suited}" # e.g., "AKs" or "T8o"

    def _get_preflop_strength(self, hole_cards):
        """Looks up the pre-computed equity for the given hole cards."""
        if not hole_cards or len(hole_cards) < 2:
            return 0.0

        # Generate the key (e.g., "AKs") from the integer cards
        key = self._get_preflop_equity_key(hole_cards)
        
        # Return the equity from the dictionary, with a fallback of 0.0
        return PREFLOP_EQUITY.get(key, 0.0)
    
    def _get_street_cards_schema(self, stage: int) -> StreetFeatures:
        """Helper to get the correct street cards schema for a given stage."""
        if stage == 0:
            return self.schema.preflop_cards
        elif stage == 1:
            return self.schema.flop_cards
        elif stage == 2:
            return self.schema.turn_cards
        elif stage == 3:
            return self.schema.river_cards
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def _get_betting_history_schema(self, stage: int) -> BettingRoundFeatures:
        """Helper to get the correct betting history schema for a given stage."""
        if stage == 0:
            return self.schema.preflop_betting
        elif stage == 1:
            return self.schema.flop_betting
        elif stage == 2:
            return self.schema.turn_betting
        else:
            raise ValueError(f"Invalid stage for betting history: {stage}")
    
    def update_betting_action(self, player_id: int, action: int, state_before_action, stage: int):
        """
        Update betting history based on the game state *before* the action occurred.
        This function robustly handles state passed as either a GameState object or a dict.
        """
        # Track all actions (not just aggressive ones)
        self._action_counts_this_street[stage] += 1
        
        if action != 2:  # Only track aggressive actions (raises/bets)
            return
        
        # Track the last aggressor
        self.last_aggressor = player_id

        # An open-bet/raise is the first aggressive action on a street. This is
        # indicated by `last_raiser` being None *before* this action occurs.
        is_dict = isinstance(state_before_action, dict)
        last_raiser = state_before_action.get('last_raiser') if is_dict else state_before_action.last_raiser
        is_first_aggressive_action = (last_raiser is None)

        is_my_action = (player_id == self.seat_id)

        if is_my_action:
            if is_first_aggressive_action:
                self._betting_history['my_bets_opened'][stage] += 1
            else:
                self._betting_history['my_raises_made'][stage] += 1
        else: # Opponent's action
            if is_first_aggressive_action:
                self._betting_history['opp_bets_opened'][stage] += 1
            else:
                self._betting_history['opp_raises_made'][stage] += 1
    
