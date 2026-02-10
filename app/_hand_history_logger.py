# app/_hand_history_logger.py
# app/_hand_history_logger.py
import logging
import dataclasses
from app.poker_core import card_to_string, HandEvaluator
from app.poker_agents import ACTION_MAP 

class HandHistoryLogger:
    def __init__(self, log_file='hand_history.log', dump_features: bool = False, verbose: bool = False):
        self.verbose = verbose 
        self.logger = logging.getLogger('hand_history')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if self.logger.hasHandlers(): self.logger.handlers.clear()

        handler = logging.FileHandler(log_file, mode='w')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        self.evaluator = HandEvaluator()
        self.dump_features = dump_features
        self.current_hand_actions = []

    def _format_cards(self, cards):
        return [card_to_string(c) for c in cards]

    def _format_probs(self, p):
        # Handle tensor/numpy conversion safely and succinctly
        p = p.detach().cpu().numpy().flatten() if hasattr(p, 'detach') else p.flatten()
        return f"F:{p[0]:.0%}, C:{p[1]:.0%}, R:{sum(p[2:]):.0%}"

    def log_start_hand(self, episode, hand_num, state):
        self.current_hand_actions = []
        if self.verbose:
            self.logger.info(f"[Hand {hand_num}] P0: {self._format_cards(state.hole_cards[0])} P1: {self._format_cards(state.hole_cards[1])}")

    def log_action(self, player_id, action, amount, state_before, predictions, policy_name, features_schema=None):
        entropy_val = predictions.get('entropy', None) if predictions else None
        if hasattr(entropy_val, 'item'): entropy_val = entropy_val.item()

        self.current_hand_actions.append({'player': player_id, 'action_type': action, 'street': state_before.stage, 'policy': policy_name, 'entropy': entropy_val})
        if not self.verbose: return
        
        # Hand Rank Logic
        p_name = f"P{player_id}"
        hole = self._format_cards(state_before.hole_cards[player_id])
        if not state_before.community:
            r1, r2 = state_before.hole_cards[player_id][0] // 4, state_before.hole_cards[player_id][1] // 4
            hand_name = "Pair" if r1 == r2 else "High Card"
        else:
            rank = self.evaluator.best_hand_rank(state_before.hole_cards[player_id], state_before.community)
            hand_name = self.evaluator.get_rank_string(rank)

        # Feature String
        feat_str = ""
        if features_schema:            
            stage_map = {0: features_schema.preflop_cards, 1: features_schema.flop_cards, 2: features_schema.turn_cards, 3: features_schema.river_cards}
            feat_str = f"RS:{stage_map[state_before.stage].random_strength:.2f} | "

        # AI Info - FIX: Handle missing state_values safely
        probs_str, e_rew_str = "--", "--"
        if predictions:
            if 'action_probs' in predictions: 
                probs_str = self._format_probs(predictions['action_probs'])
            
            # NFSP (DQN) uses 'q_values', Actor-Critic uses 'state_values'
            if 'q_values' in predictions:
                e_rew_str = f"{predictions['q_values'].max().item():+.4f}"
            elif 'state_values' in predictions:
                e_rew_str = f"{predictions['state_values'][0].item():+.4f}"

        stack_bb = state_before.stacks[player_id] / state_before.big_blind 
        self.logger.info(f"    [{p_name}({policy_name}): {' '.join(hole)} ({hand_name}) | {feat_str}P:[{probs_str}] | Rew:{e_rew_str} | Stack: {stack_bb:.1f}BB]")

        # Action String
        cur_bets = state_before.current_bets
        if action == 0: act_str = f"  {p_name} folds."
        elif action == 1: act_str = f"  {p_name} calls." if max(cur_bets) > cur_bets[player_id] else f"  {p_name} checks."
        elif action == 2: act_str = f"  {p_name} bets {amount}." if max(cur_bets) == cur_bets[player_id] else f"  {p_name} raises to {amount + cur_bets[player_id]}."
        self.logger.info(act_str)

    def log_street(self, state):
        if self.verbose:
            st_name = {1: "FLOP", 2: "TURN", 3: "RIVER"}.get(state.stage, "")
            self.logger.info(f"- {st_name} {self._format_cards(state.community)} (pot: {state.pot}):")

    def log_end_hand(self, episode_rewards, state):
        if self.verbose:
            self.logger.info(f"- PROFIT: {episode_rewards[0]} chips (Avg: {episode_rewards[0]/state.big_blind:+.4f})\n")
        return {'actions': self.current_hand_actions, 'rewards': episode_rewards}

    def log_feature_dump_if_needed(self, state, schema, predictions):
        if self.dump_features and state.stage == 3 and schema and schema.dynamic.bet_faced_ratio > 0.0:
            self.logger.info("  --- Feature Schema Dump ---")
            self._dump_schema_features(schema, predictions)
            self.logger.info("  ---------------------------")

    def _dump_schema_features(self, schema, predictions):
        self._recursive_dump(schema, 0)
        if not predictions: return

        if 'q_values' in predictions:
            q = predictions['q_values'][0].detach().cpu().numpy()
            self.logger.info("    --- Q-Values ---")
            for i, (atype, size) in enumerate(ACTION_MAP):
                self.logger.info(f"        {self._get_action_name(atype, size):<12} = {q[i]:+.4f}")

        if 'action_probs' in predictions:
            p = predictions['action_probs'][0].detach().cpu().numpy()
            self.logger.info("    --- Action Probs ---")
            for i, (atype, size) in enumerate(ACTION_MAP):
                self.logger.info(f"        {self._get_action_name(atype, size):<12} = {p[i]:.4%}")

    def _get_action_name(self, atype, size):
        if atype == 0: return "Fold"
        if atype == 1: return "Check/Call"
        return "All-In" if size == -1 else f"Raise {int(size*100)}%"

    def _recursive_dump(self, item, indent_level):
        indent = "    " * (indent_level + 1)
        if dataclasses.is_dataclass(item):
            for field in dataclasses.fields(item):
                val = getattr(item, field.name)
                if dataclasses.is_dataclass(val):
                    self.logger.info(f"{indent}{field.name}:")
                    self._recursive_dump(val, indent_level + 1)
                else:
                    v_str = f"{val:.4f}" if isinstance(val, float) else f"{val}"
                    self.logger.info(f"{indent}{field.name} = {v_str}")

