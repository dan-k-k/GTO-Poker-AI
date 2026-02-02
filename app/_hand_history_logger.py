# app/_hand_history_logger.py
import logging
import json
import dataclasses
import numpy as np
import os
from app.poker_core import card_to_string, HandEvaluator
from app.poker_agents import ACTION_MAP 

class HandHistoryLogger:
    def __init__(self, log_file='hand_history.log', data_file='hand_data.jsonl', dump_features: bool = False):
        # ... (Your init code remains the same) ...
        self.logger = logging.getLogger('hand_history')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        self.data_file = data_file
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                pass

        self.evaluator = HandEvaluator()
        self.dump_features = dump_features
        self.current_hand_data = {}

    def _format_cards(self, cards):
        return [card_to_string(c) for c in cards]
    
    def _make_serializable(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _format_probs(self, probs):
        if hasattr(probs, 'detach'):
            probs = probs.detach().cpu().numpy()
        elif hasattr(probs, 'numpy'):
            probs = probs.numpy()
        if hasattr(probs, 'shape') and len(probs.shape) > 1:
            probs = probs.flatten()
        return " ".join([f"{p:.2f}" for p in probs])
    
    def log_start_hand(self, episode, hand_num, state):
        p0_cards = self._format_cards(state.hole_cards[0])
        p1_cards = self._format_cards(state.hole_cards[1])
        self.logger.info(f"[Hand {hand_num}] P0 dealt {p0_cards}. P1 dealt {p1_cards}.")
        
        self.current_hand_data = {
            "episode": episode,
            "hand_num": hand_num,
            "p0_cards": p0_cards,
            "p1_cards": p1_cards,
            "btn_pos": state.dealer_pos,
            "actions": [],
            "rewards": [],
            "winner": None
        }

    def log_action(self, player_id, action, amount, state_before, predictions, policy_name, features_schema=None, last_rl_loss: float = 0.0):
        # --- (Text Logging Logic) ---
        player_name = f"P{player_id}"
        hole = self._format_cards(state_before.hole_cards[player_id])
        community = state_before.community
        
        hand_type_name = "N/A"
        if not community:
            hole_cards = state_before.hole_cards[player_id]
            rank1 = hole_cards[0] // 4
            rank2 = hole_cards[1] // 4
            if rank1 == rank2: hand_type_name = "Pair"
            else: hand_type_name = "High Card"
        else:
            hole_cards = state_before.hole_cards[player_id]
            rank = self.evaluator.best_hand_rank(hole_cards, community)
            hand_type_name = self.evaluator.get_rank_string(rank)

        features_str = ""
        if features_schema:
            hs = features_schema.dynamic.hand_strength
            stage = state_before.stage
            if stage == 0: rs = features_schema.preflop_cards.random_strength
            elif stage == 1: rs = features_schema.flop_cards.random_strength
            elif stage == 2: rs = features_schema.turn_cards.random_strength
            else: rs = features_schema.river_cards.random_strength
            features_str = f"HS:{hs:.2f}, RS:{rs:.2f} | "

        stack_bb = state_before.stacks[player_id] / state_before.big_blind 

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

        action_str = ""
        if action == 0: 
            action_str = f"  {player_name} folds."
        elif action == 1: 
            action_str = f"  {player_name} calls." if max(state_before.current_bets) > state_before.current_bets[player_id] else f"  {player_name} checks."
        elif action == 2: 
            action_str = f"  {player_name} bets {amount}." if max(state_before.current_bets) == state_before.current_bets[player_id] else f"  {player_name} raises to {amount + state_before.current_bets[player_id]}."

        self.logger.info(action_str)

        # --- MERGED DATA LOGGING LOGIC ---
        
        # 1. Calculate Entropy first
        entropy_val = None
        if predictions and 'action_probs' in predictions:
            probs = predictions['action_probs']
            if hasattr(probs, 'detach'): probs = probs.detach().cpu().numpy()
            elif hasattr(probs, 'numpy'): probs = probs.numpy()
            
            probs = np.ravel(probs)
            # Entropy = -sum(p * log(p))
            entropy_val = -np.sum(probs * np.log(probs + 1e-9))

        # 2. Create the ONE dictionary
        action_record = {
            "player": player_id,
            "street": state_before.stage, 
            "action_type": action, 
            "amount": self._make_serializable(amount) if amount else 0,
            "policy": policy_name,
            "pot_before": self._make_serializable(state_before.pot),
            "stack_before": self._make_serializable(state_before.stacks[player_id]),
            "is_aggressor": 1 if action == 2 else 0,
            "entropy": float(entropy_val) if entropy_val is not None else None 
        }
        
        # Optional: Save specific features
        if features_schema:
             action_record["hs"] = self._make_serializable(features_schema.dynamic.hand_strength)

        # 3. Append ONCE
        self.current_hand_data["actions"].append(action_record)

    def log_street(self, state):
        # Text Log
        street_name = {1: "FLOP", 2: "TURN", 3: "RIVER"}.get(state.stage)
        cards = self._format_cards(state.community)
        pot = state.pot
        self.logger.info(f"- {street_name} {cards} (pot: {pot}):")
        
        # Data Log - store community cards as they appear
        stage_key = {1: "flop", 2: "turn", 3: "river"}.get(state.stage)
        if stage_key:
             self.current_hand_data[f"{stage_key}_cards"] = cards

    # Log rewards for ALL players
    def log_end_hand(self, episode_rewards, state):
        # 1. Text Log
        p0_reward = episode_rewards[0]
        p_rew = p0_reward / state.big_blind 
        self.logger.info(f"- HAND PROFIT: {p0_reward} chips (P_Rew: {p_rew:+.4f})\n")

        # 2. Data Log Finalization
        self.current_hand_data["rewards"] = [self._make_serializable(r) for r in episode_rewards]
        self.current_hand_data["final_pot"] = self._make_serializable(state.pot)
        
        # Append line to JSONL file
        with open(self.data_file, 'a') as f:
            f.write(json.dumps(self.current_hand_data) + '\n')
        return self.current_hand_data

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
            self.logger.info("  --- Feature Schema Dump (dump_features: true in config.yaml) ---")
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

