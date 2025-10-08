# app/_hand_history_logger.py
import logging
import dataclasses 
from app.poker_core import card_to_string, HandEvaluator
from app.poker_agents import ACTION_MAP 

class HandHistoryLogger:
    def __init__(self, log_file='hand_history.log', dump_features: bool = False):
        self.logger = logging.getLogger('hand_history')
        self.logger.setLevel(logging.INFO)
        # Prevent logs from propagating to the root logger
        self.logger.propagate = False
        
        # Remove any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Add a file handler
        handler = logging.FileHandler(log_file, mode='w') # 'w' to overwrite on each run
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.evaluator = HandEvaluator()
        # Store the on/off switch
        self.dump_features = dump_features

    def _format_cards(self, cards):
        return [card_to_string(c) for c in cards]

    def _format_probs(self, probs_tensor):
        probs = probs_tensor[0].numpy()
        # FOLD is action 0, CALL is action 1, RAISES are 2-11
        fold_prob = probs[0]
        call_prob = probs[1]
        raise_prob = sum(probs[2:])
        return f"F:{fold_prob:.0%}, C:{call_prob:.0%}, R:{raise_prob:.0%}"

    def log_start_hand(self, episode, hand_num, state):
        p0_cards = self._format_cards(state.hole_cards[0])
        p1_cards = self._format_cards(state.hole_cards[1])
        self.logger.info(f"[Hand {hand_num}] P0 dealt {p0_cards}. P1 dealt {p1_cards}.")

    def log_action(self, player_id, action, amount, state_before, predictions, policy_name, features_schema=None, last_rl_loss: float = 0.0):
        # Identify who is who
        player_name = f"P{player_id}"

        # Get hand info
        hole = self._format_cards(state_before.hole_cards[player_id])
        community = state_before.community

        hand_type_name = "N/A" # Default value
        
        # Case 1: Preflop (no community cards yet)
        if not community:
            hole_cards = state_before.hole_cards[player_id]
            rank1 = hole_cards[0] // 4
            rank2 = hole_cards[1] // 4
            if rank1 == rank2:
                hand_type_name = "Pair"
            else:
                hand_type_name = "High Card"
        # Case 2: Flop, Turn, and River
        else:
            hole_cards = state_before.hole_cards[player_id]
            rank = self.evaluator.best_hand_rank(hole_cards, community)
            hand_type_name = self.evaluator.get_rank_string(rank)

        features_str = ""
        if features_schema:
            # Get intelligent hand strength (always dynamic)
            hs = features_schema.dynamic.hand_strength
            
            # Get the correct random strength for the current street
            stage = state_before.stage
            if stage == 0: rs = features_schema.preflop_cards.random_strength
            elif stage == 1: rs = features_schema.flop_cards.random_strength
            elif stage == 2: rs = features_schema.turn_cards.random_strength
            else: rs = features_schema.river_cards.random_strength
            
            features_str = f"HS:{hs:.2f}, RS:{rs:.2f} | "

        stack_bb = state_before.stacks[player_id] / state_before.big_blind 

        # AI-specific info (if available)
        if predictions:
            probs_str = self._format_probs(predictions['action_probs'])
            e_rew = predictions['state_values'][0].item()
            e_rew_str = f"{e_rew:+.4f}"
        else:
            probs_str = "--"
            e_rew_str = "--"

        log_line = f"    [{player_name}({policy_name}): {' '.join(hole)} ({hand_type_name}) | " \
                f"{features_str}" \
                f"P:[{probs_str}] | x_Rew:{e_rew_str} | Stack: {stack_bb:.1f}BB]"

        self.logger.info(log_line)

        # Log the action itself
        action_str = ""
        if action == 0: 
            action_str = f"  {player_name} folds."
        elif action == 1: 
            action_str = f"  {player_name} calls." if max(state_before.current_bets) > state_before.current_bets[player_id] else f"  {player_name} checks."
        elif action == 2: 
            action_str = f"  {player_name} bets {amount}." if max(state_before.current_bets) == state_before.current_bets[player_id] else f"  {player_name} raises to {amount + state_before.current_bets[player_id]}."

        self.logger.info(action_str)


    def log_street(self, state):
        street_name = {1: "FLOP", 2: "TURN", 3: "RIVER"}.get(state.stage)
        cards = self._format_cards(state.community)
        pot = state.pot
        self.logger.info(f"- {street_name} {cards} (pot: {pot}):")

    # Log rewards for ALL players
    def log_end_hand(self, episode_rewards, state):
        p0_reward = episode_rewards[0]
        p_rew = p0_reward / state.big_blind # Reward in BB
        self.logger.info(f"- HAND PROFIT: {p0_reward} chips (P_Rew: {p_rew:+.4f})\n")

    def log_feature_dump_if_needed(self, state, features_schema, predictions):
        """Checks if conditions are met to dump the feature schema and does so."""
        # 1. Check if the feature is turned on
        if not self.dump_features:
            return
            
        # 2. Check if it's the river
        if state.stage != 3:
            return

        # 3. Check if it's the second player's turn to act on this street
        if features_schema and features_schema.dynamic.bet_faced_ratio > 0.0:
            self.logger.info("  --- Feature Schema Dump (dump_features=True in config.yaml) ---")
            # Pass `predictions` to the dumper method
            self._dump_schema_features(features_schema, predictions)
            self.logger.info("  ---------------------------------")

    def _dump_schema_features(self, schema, predictions):
        """Recursively logs the fields and values of a nested dataclass."""
        self._recursive_dump(schema, indent_level=0)
        
        if predictions:
            q_values = predictions['q_values'][0].numpy()
            probs = predictions['action_probs'][0].numpy()
            
            # 1. Q-Values
            self.logger.info("    --- Q-Values (Expected Reward per Action) ---")
            for i, (action_type, sizing) in enumerate(ACTION_MAP):
                action_name = self._get_action_name(action_type, sizing)
                self.logger.info(f"        {action_name:<12} = {q_values[i]:+.4f}")

            # 2. Full, unrounded action probabilities.
            self.logger.info("    --- Full Action Probabilities ---")
            for i, (action_type, sizing) in enumerate(ACTION_MAP):
                action_name = self._get_action_name(action_type, sizing)
                self.logger.info(f"        {action_name:<12} = {probs[i]:.4%}")

    def _get_action_name(self, action_type, sizing):
        """Creates a readable name for an action from the ACTION_MAP."""
        if action_type == 0: return "Fold"
        if action_type == 1: return "Call/Check"
        if action_type == 2:
            if sizing == -1: return "All-In"
            else: return f"Raise {int(sizing*100)}%"
        return "Unknown"


    def _recursive_dump(self, item, indent_level):
        """Helper function to print nested dataclass fields."""
        indent = "    " * (indent_level + 1)
        if dataclasses.is_dataclass(item):
            for field in dataclasses.fields(item):
                value = getattr(item, field.name)
                # If the field is another dataclass, print its name and recurse
                if dataclasses.is_dataclass(value):
                    self.logger.info(f"{indent}{field.name}:")
                    self._recursive_dump(value, indent_level + 1)
                # Otherwise, print the field name and its value
                else:
                    self.logger.info(f"{indent}{field.name} = {value:.4f}")

