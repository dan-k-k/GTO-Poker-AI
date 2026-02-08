# app/TexasHoldemEnv.py
# Poker environment using GameState and centralized HandEvaluator

from typing import List, Optional, Dict
from app.poker_core import GameState, Deck, HandEvaluator
from dataclasses import asdict

class TexasHoldemEnv:
    """
    Clean poker game environment focused purely on game logic.
    AI-specific tracking moved to FeatureExtractor.
    Uses GameState from poker_core.
    """

    def __init__(self, num_players=2, starting_stack=200, small_blind=1, big_blind=2, seed=None):
        assert 2 <= num_players <= 9
        self._num_players = num_players  # Use a private attribute
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind

        # Use centralized deck and hand evaluator
        self.deck = Deck(seed)
        self.evaluator = HandEvaluator()
        # Game state container

        self.state: Optional[GameState] = None
        self.reset()


    def reset(self):
        """
        Resets the environment. If the tournament is over or hasn't started,
        it begins a new tournament. Otherwise, it just starts the next hand.
        """
        if self.state is None or self.state.win_reason == 'tournament_winner':
            return self._start_new_tournament()
        else:
            return self._start_new_hand(preserve_stacks=True)
    
    def _start_new_tournament(self) -> GameState:
        """Sets up initial stacks for all players and starts the first hand."""
        initial_stacks = [self.starting_stack] * self.num_players
        return self._start_new_hand(preserve_stacks=False, existing_stacks=initial_stacks)
    
    def _start_new_hand(self, preserve_stacks: bool, existing_stacks: Optional[List[int]] = None) -> GameState:
        """Resets the game state for the start of a new hand."""
        if preserve_stacks:
            stacks = self.state.stacks.copy()
            surviving_players = [p for p, s in enumerate(stacks) if s > 0]

            if len(surviving_players) <= 1:
                winner = surviving_players[0] if surviving_players else -1
                self.state.terminal = True
                self.state.winners = [winner] if winner != -1 else []
                self.state.win_reason = 'tournament_winner'
                return self.state
        else:
            stacks = existing_stacks if existing_stacks else [self.starting_stack] * self.num_players
            surviving_players = list(range(self.num_players))

        # 1) Reset deck & shuffle
        self.deck.reset()

        # 2) Deal hole cards
        hole_cards = []
        for _ in range(self.num_players):
            player_cards = self.deck.deal(2)
            hole_cards.append(player_cards)

        # 4) Advance dealer button (clockwise to next surviving player)
        if not hasattr(self, 'state') or self.state is None:
            dealer_pos = surviving_players[0]
        else:
            # Find next surviving player clockwise from current dealer
            current_dealer = self.state.dealer_pos
            next_dealer = (current_dealer + 1) % self.num_players

            # Advance clockwise until we find a surviving player
            while next_dealer not in surviving_players:
                next_dealer = (next_dealer + 1) % self.num_players

            dealer_pos = next_dealer

        # 5) Calculate positions (only among surviving players)
        if len(surviving_players) == 2:
            # Heads-up: Dealer is SB, other player is BB
            dealer_idx = surviving_players.index(dealer_pos)
            sb_pos = surviving_players[dealer_idx]
            bb_pos = surviving_players[(dealer_idx + 1) % 2]
        else:
            # 3+ players: SB is left of dealer, BB is left of SB
            dealer_idx = surviving_players.index(dealer_pos)
            sb_idx = (dealer_idx + 1) % len(surviving_players)
            bb_idx = (dealer_idx + 2) % len(surviving_players)
            sb_pos = surviving_players[sb_idx]
            bb_pos = surviving_players[bb_idx]
        # Store starting stacks before blinds are posted
        initial_stacks = stacks.copy()

        # 6) Post blinds
        sb_amount = min(self.small_blind, stacks[sb_pos])
        bb_amount = min(self.big_blind, stacks[bb_pos])

        stacks[sb_pos] -= sb_amount
        stacks[bb_pos] -= bb_amount

        current_bets = [0] * self.num_players
        current_bets[sb_pos] = sb_amount
        current_bets[bb_pos] = bb_amount

        pot = sb_amount + bb_amount

        # 7) Determine first to act (only among surviving players)
        if len(surviving_players) == 2:
            to_move = sb_pos # SB acts first in heads-up
        else:
            # UTG is left of BB
            bb_idx = surviving_players.index(bb_pos)
            utg_idx = (bb_idx + 1) % len(surviving_players)
            to_move = surviving_players[utg_idx]

        # Set up player states (all players, but only surviving ones matter)
        active = [p in surviving_players for p in range(self.num_players)]
        all_in = [stacks[p] == 0 for p in range(self.num_players)]

        # Ensure first player to act can actually act (not all-in)
        while to_move in surviving_players and all_in[to_move]:
            current_idx = surviving_players.index(to_move)
            next_idx = (current_idx + 1) % len(surviving_players)
            to_move = surviving_players[next_idx]

        # 8) Create GameState
        self.state = GameState(
            num_players=self.num_players,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            hole_cards=hole_cards,
            community=[],
            stacks=stacks,
            current_bets=current_bets,
            pot=pot,
            starting_pot_this_round=pot, # Starting pot for preflop should be blinds total
            starting_stacks_this_hand=initial_stacks, # Starting stacks before blinds
            active=active,
            all_in=all_in,
            acted=[False] * self.num_players,
            surviving_players=surviving_players,
            stage=0, # Preflop
            dealer_pos=dealer_pos,
            sb_pos=sb_pos,
            bb_pos=bb_pos,
            to_move=to_move,
            initial_bet=self.big_blind,
            last_raise_size=self.big_blind,
            last_raiser=None,
            terminal=False,
            winners=None,
            win_reason=None
        )

        return self.state

    def step(self, action, amount=None):
        """Execute an action and advance game state."""
        if self.state.terminal:
            raise ValueError("Cannot step in a terminal state.")

        if self.state is None:
            raise ValueError("Game state not initialized")

        player = self.state.to_move
        current_max = max(self.state.current_bets)
    
        # Execute action
        if action == 0: # FOLD
            self.state.active[player] = False
        elif action == 1: # CHECK/CALL
            to_call = current_max - self.state.current_bets[player]
            if to_call > 0:
                call_amt = min(to_call, self.state.stacks[player])
                self.state.stacks[player] -= call_amt
                self.state.current_bets[player] += call_amt
                self.state.pot += call_amt
        else: # BET/RAISE
            is_all_in_bet = (amount == self.state.stacks[player])
            
            # An all-in is always a legal action, even if it's less than a full min-raise.
            # Validate the size of non-all-in bets.
            if not is_all_in_bet:
                required = self.state.get_min_raise_amount()
                # Strict action validation to prevent illegal under-raises
                if required is None:
                    raise ValueError(f"Player {player} cannot raise (insufficient chips or illegal state)")

                if amount < required:
                    current_total = self.state.current_bets[player] + amount
                    required_total = self.state.current_bets[player] + required
                    raise ValueError(f"Illegal under-raise: Player {player} tried to raise by {amount} (to {current_total} total), minimum raise by {required} (to {required_total} total)")

            if amount > self.state.stacks[player]:
                raise ValueError(f"Illegal bet size: Player {player} tried to bet {amount}, only has {self.state.stacks[player]} chips")

            # Calculate new total bet after this action
            new_total_bet = self.state.current_bets[player] + amount

            # Prevent illegal under-raises relative to current max bet
            if not is_all_in_bet and new_total_bet < current_max:
                raise ValueError(f"Illegal under-raise: New total bet {new_total_bet} is less than current max bet {current_max}")

            # A "full raise" that re-opens action must be a raise (new total > current max) 
            # AND its size (new total - current max) must be at least the last raise size.
            # An all-in that is too small does not reopen action.
            is_a_raise = new_total_bet > current_max
            raise_amount = new_total_bet - current_max
            is_full_raise = is_a_raise and (raise_amount >= self.state.last_raise_size)
            self.state.stacks[player] -= amount
            self.state.pot += amount
            self.state.current_bets[player] += amount

            if is_full_raise:
                self.state.last_raise_size = raise_amount
                self.state.last_raiser = player

                # Reset acted flags for all other players
                for p in range(self.state.num_players):
                    if p != player:
                        self.state.acted[p] = False

        self.state.acted[player] = True
        if self.state.stacks[player] == 0:
            self.state.all_in[player] = True

        # Check for winner by fold (only among surviving players)
        still_in = [p for p in self.state.surviving_players if self.state.active[p]]
        if len(still_in) == 1:
            winner = still_in[0]
            self.state.stacks[winner] += self.state.pot
            self.state.pot = 0 # Reset pot after awarding chips
            self.state.terminal = True
            self.state.winners = [winner]
            self.state.win_reason = 'fold'

            self._check_tournament_winner()
            return self.state, True
            
        # Check if street is over
        street_is_over = self._is_street_over()

        if not street_is_over:
            # Advance to next player in proper turn order
            # Find the index of the current player in the list of survivors
            current_idx_in_survivors = self.state.surviving_players.index(player)
            num_survivors = len(self.state.surviving_players)
            # Check the next players in clockwise order
            for i in range(1, num_survivors + 1):
                next_player_idx = (current_idx_in_survivors + i) % num_survivors
                next_player = self.state.surviving_players[next_player_idx]
                # If this player is active and not all-in, it's their turn
                if self.state.active[next_player] and not self.state.all_in[next_player]:
                    self.state.to_move = next_player
                    break
            return self.state, False
            
        # Street is over - handle all-in situations and advance
        if self._should_auto_complete():
            return self._finish_hand_all_in()
            
        # Reset for next street
        self.state.acted = [False] * self.state.num_players
        self.state.last_raiser = None
        self.state.current_bets = [0] * self.state.num_players
        self.state.initial_bet = None
        self.state.last_raise_size = self.state.big_blind
        
        # Check if hand is over
        if self.state.stage == 3: # River complete
            self._handle_showdown()
            return self.state, True
            
        # Deal next street
        self.state.stage += 1
        if self.state.stage == 1: # Flop
            new_cards = self.deck.deal(3)
            self.state.community.extend(new_cards)
        else: # Turn or River
            new_card = self.deck.deal(1)
            self.state.community.extend(new_card)

        self.state.starting_pot_this_round = self.state.pot
        
        # Set first to act (dealer acts last post-flop, only among surviving players)
        active_survivors = [p for p in self.state.surviving_players if self.state.active[p] and not self.state.all_in[p]]
        if active_survivors:
            self.state.to_move = self._next_active_player(self.state.dealer_pos)
            
        return self.state, False

    def _is_street_over(self) -> bool:
        """
        Check if the current betting street is complete.
        This is true if all active players have either matched the highest bet or are all-in.
        """
        active_players = [p for p in self.state.surviving_players if self.state.active[p]]
        if len(active_players) <= 1:
            return True

        players_who_can_act = [p for p in active_players if not self.state.all_in[p]]
        
        # If no one can act (e.g., everyone is all-in), the betting round is over.
        if not players_who_can_act:
            return True

        highest_bet = max(self.state.current_bets[p] for p in active_players)

        # Special case: Preflop, big blind can act again if there were no raises.
        if self.state.stage == 0 and highest_bet == self.state.big_blind:
            bb_pos = self.state.bb_pos
            # The BB can only act if they haven't already and are not all-in.
            if bb_pos in players_who_can_act and not self.state.acted[bb_pos]:
                return False

        # General case: The street is over if all players who can still act...
        # 1. Have had a turn to act.
        # 2. Have contributed the same amount to the pot as the highest bettor.
        all_acted = all(self.state.acted[p] for p in players_who_can_act)
        bets_match = all(self.state.current_bets[p] == highest_bet for p in players_who_can_act)

        return all_acted and bets_match

    def _should_auto_complete(self):
        """Check if hand should auto-complete due to all-in situation."""
        active_survivors = [p for p in self.state.surviving_players if self.state.active[p]]
        # Auto-complete if 1 or 0 players can still make actions
        can_act_count = sum(1 for p in active_survivors if not self.state.all_in[p])
        return can_act_count <= 1

    def _next_active_player(self, current_player):
        """Find next active player who can act (clockwise, only among surviving players)."""
        if not self.state.surviving_players:
            return -1 # No one left
        
        # Start searching from the player to the left of the current_player (or dealer button post-flop)
        try:
            start_idx = self.state.surviving_players.index(current_player)
        except ValueError:
            # If current_player isn't in survivors (e.g., just busted), start from dealer
            start_idx = self.state.surviving_players.index(self.state.dealer_pos)
            
        num_survivors = len(self.state.surviving_players)

        # Search all other players in a circle, clockwise
        for i in range(1, num_survivors + 1):
            next_idx = (start_idx + i) % num_survivors
            next_player = self.state.surviving_players[next_idx]
            if self.state.active[next_player] and not self.state.all_in[next_player]:
                return next_player
        
        # Fallback if only one player can act, they are the next to move
        can_act = [p for p in self.state.surviving_players if self.state.active[p] and not self.state.all_in[p]]
        return can_act[0] if can_act else -1


    def _finish_hand_all_in(self):
        """Complete hand when all players are all-in or folded."""
        # Deal remaining community cards
        while self.state.stage < 3:
            self.state.stage += 1
            if self.state.stage == 1:
                new_cards = self.deck.deal(3)
                self.state.community.extend(new_cards)
            else:
                new_card = self.deck.deal(1)
                self.state.community.extend(new_card)

        # Determine winners
        self._handle_showdown(from_all_in=True) 
        return self.state, True

    def _calculate_showdown_hand_ranks(self):
        """Calculate hand ranks for all active players at showdown."""
        hand_ranks = {}
        for p in self.state.surviving_players:
            if not self.state.active[p]:
                continue
            hand_ranks[p] = self.evaluator.best_hand_rank(self.state.hole_cards[p], self.state.community)
        return hand_ranks

    def _determine_showdown_winner(self, hand_ranks=None):
        """Determine winner(s) at showdown using pre-calculated hand ranks."""
        if hand_ranks is None:
            hand_ranks = self._calculate_showdown_hand_ranks()
        if not hand_ranks:
            return []
        best_rank = max(hand_ranks.values())
        winners = [p for p, rank in hand_ranks.items() if rank == best_rank]
        return winners

    def _handle_showdown(self, from_all_in: bool = False):
        """Centralized method to process a showdown."""
        self.state.terminal = True

        hand_ranks = self._calculate_showdown_hand_ranks()
        self._distribute_pot_with_side_pots(hand_ranks)
        self.state.winners = self._determine_showdown_winner(hand_ranks)

        is_tournament_over = self._check_tournament_winner()

        if not is_tournament_over:
            self.state.win_reason = 'all_in_showdown' if from_all_in else 'showdown'

    def _check_tournament_winner(self) -> bool: 
        """
        Check if any players went bust and declare tournament winner if only 1 remains.
        Returns True if a tournament winner was found, False otherwise.
        """
        # Update surviving players list - remove anyone with 0 chips
        self.state.surviving_players = [p for p in self.state.surviving_players if self.state.stacks[p] > 0]

        # Check for tournament winner
        if len(self.state.surviving_players) == 1:
            winner = self.state.surviving_players[0]
            self.state.terminal = True
            self.state.winners = [winner]
            self.state.win_reason = 'tournament_winner'
            return True 
        elif len(self.state.surviving_players) == 0:
            # This case handles a scenario where multiple players bust simultaneously leaving no winner
            self.state.terminal = True
            self.state.winners = []
            self.state.win_reason = 'no_survivors'
            return True 
            
        return False 

    def _distribute_pot_with_side_pots(self, hand_ranks=None):
        """
        Handles complex pot distribution for showdowns with multiple all-in players.
        This method creates and awards the main pot and all side pots.
        """
        if hand_ranks is None:
            hand_ranks = self._calculate_showdown_hand_ranks()
        state = self.state

        # Get players involved in the showdown
        showdown_players = [p for p in state.surviving_players if state.active[p] and p in hand_ranks]
        if not showdown_players:
            return # No one to award pot to

        # Calculate each player's total investment for the hand
        investments = {p: state.starting_stacks_this_hand[p] - state.stacks[p] for p in showdown_players}

        if all(v == 0 for v in investments.values()):
            return

        # Create a sorted list of unique investment amounts
        sorted_investments = sorted(list(set(investments.values())))
        
        last_investment_level = 0
        
        # Iterate through each investment level to create and award pots
        for investment_level in sorted_investments:
            # Determine the pot amount for this specific level (side pot)
            pot_increment = investment_level - last_investment_level
            
            # Find all players who contributed to this pot (i.e., invested at least this much)
            eligible_players = [p for p in showdown_players if investments[p] >= investment_level]
            
            if not eligible_players:
                continue

            # Calculate the size of this pot
            current_pot_size = pot_increment * len(eligible_players)
            
            # Find the winner(s) among the eligible players for this pot
            best_rank_in_pot = None  # Use None instead of -1
            winners = []
            for p in eligible_players:
                rank = hand_ranks.get(p)
                if rank is None:
                    continue
                # Compare tuple to tuple (or handle the first player)
                if best_rank_in_pot is None or rank > best_rank_in_pot:
                    best_rank_in_pot = rank
                    winners = [p]
                elif rank == best_rank_in_pot:
                    winners.append(p)
            
            # Distribute this pot to the winner(s)
            if winners:
                share = current_pot_size // len(winners)
                remainder = current_pot_size % len(winners)
                for i, w in enumerate(winners):
                    state.stacks[w] += share
                    if i < remainder:
                        state.stacks[w] += 1
            
            last_investment_level = investment_level

        state.pot = 0 # All money has been distributed

    # Legacy compatibility methods
    @property
    def hole(self):
        """Legacy compatibility - access hole cards."""
        return self.state.hole_cards if self.state else []
        
    @property
    def community(self):
        """Legacy compatibility - access community cards."""
        return self.state.community if self.state else []

    @property
    def num_players(self):
        """Number of players in game."""
        return self._num_players

    @num_players.setter
    def num_players(self, value):
        """Set number of players."""
        self._num_players = value

    def get_state_dict(self) -> Dict:
        """
        Returns a serializable dictionary representation of the current game state,
        INCLUDING the deck's state.
        """
        if self.state is None:
            return {}
        # Use the existing asdict helper
        state_dict = asdict(self.state)
        
        state_dict['deck_cards'] = self.deck.cards # Add remaining cards in deck to dict
        
        return state_dict

