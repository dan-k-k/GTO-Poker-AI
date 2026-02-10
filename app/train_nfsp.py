# app/train_nfsp.py
# python -m app.train_nfsp
# python -m cProfile -o training.pstats app/train_nfsp.py
# snakviz training.psteats
import os
import torch
import numpy as np
import json
import yaml
import time
import datetime
from typing import Dict, List
from app.TexasHoldemEnv import TexasHoldemEnv
from app.nfsp_components import NFSPAgent
from app.poker_agents import RandomBot
from app._hand_history_logger import HandHistoryLogger
from app.live_plotter import LivePlotter
from collections import deque

class NFSPTrainer:
    """Main training loop for Neural Fictitious Self-Play."""
    
    def __init__(self, config: Dict):
        # Store all hyperparameters
        self.config = config
        self.num_episodes = config['training']['num_episodes']
        self.save_interval = config['training']['save_interval']
        self.eval_interval = config['training']['eval_interval']

        # --- Define a main output directory ---
        self.output_dir = "training_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Read the simulation config values
        self.random_trials = config['simulations']['random_equity_trials']

        # Initialize environments and agents
        stack_size = config['simulations'].get('starting_stack', 200) 

        self.env = TexasHoldemEnv(num_players=2, starting_stack=stack_size)
        self.eval_env = TexasHoldemEnv(num_players=2, starting_stack=stack_size)
        self.random_bot = RandomBot(seat_id=1, aggression=0.3)
        self.agents = [NFSPAgent(i, config['agent'], config['buffers'], config['simulations']['random_equity_trials'], starting_stack=stack_size) for i in range(2)]

        # Training statistics (Modified to be bounded)
        self.stats = {
            # Keep only last 100 rewards for printing progress
            'episode_rewards': [deque(maxlen=100), deque(maxlen=100)], 
            'buffer_sizes_rl': [deque(maxlen=100), deque(maxlen=100)],
            'buffer_sizes_sl': [deque(maxlen=100), deque(maxlen=100)],
            'training_time': 0
        }
        self.prior_training_time = 0

        # Track best performance
        self.best_avg_reward = -float('inf')

        # --- Initialize the logger ---
        csv_path = os.path.join(self.output_dir, "metrics.csv")
        plot_path = os.path.join(self.output_dir, "training_dashboard.png")
        log_path = os.path.join(self.output_dir, "hand_history.log")
        self.logger = HandHistoryLogger(log_file=log_path, dump_features=config['logging']['dump_features'], verbose=config['logging']['verbose'])
        self.hand_counter = 0
        self.plotter = LivePlotter(plot_file=plot_path, csv_file=csv_path)
        
        # === Attempt to load latest models to resume training ===
        self._load_state()
        self._load_buffers()
        self._load_models(suffix="_latest")

    def play_episode(self, env, agents, logger=None, training=False, episode_num=0):
        """Unified game loop for both training and evaluation."""
        state = env.reset()
        for agent in agents:
            agent.new_hand()

        # Logging start
        if logger:
            hand_id = getattr(self, 'hand_counter', 0)
            logger.log_start_hand(episode_num, hand_id, state)

        initial_stacks = state.starting_stacks_this_hand.copy()

        while not state.terminal:
            player_idx = state.to_move
            agent = agents[player_idx]
            state_before = state.copy()

            # Compute Action
            action, amount, preds, policy, schema = agent.compute_action(state)

            # Logging (Only if logger is provided)
            if logger:
                logger.log_action(player_idx, action, amount, state_before, preds, policy, schema)
                if schema:
                    logger.log_feature_dump_if_needed(state_before, schema, preds)

            # Step Environment
            next_state, done = env.step(action, amount)

            # Observations (Crucial for NFSP internal state tracking)
            for obs_agent in agents:
                obs_agent.observe((action, amount), player_idx, state_before, next_state)
            
            # Log Street changes
            if logger and next_state.stage > state_before.stage:
                logger.log_street(next_state)

            state = next_state

        # End of Hand / Showdown
        final_stacks = state.stacks
        rewards = [final_stacks[i] - initial_stacks[i] for i in range(len(agents))]

        showdown_data = {
            'stacks_before': {i: initial_stacks[i] for i in range(len(initial_stacks))},
            'stacks_after': {i: final_stacks[i] for i in range(len(final_stacks))}
        }

        # Finalize Agents (This triggers the final RL buffer push)
        for agent in agents:
            agent.observe_showdown(showdown_data)

        if logger:
            return rewards, logger.log_end_hand(rewards, state)
        
        return rewards, None

    def _update_stats(self, rewards):
            """Helper to update deque stats."""
            for i, r in enumerate(rewards):
                self.stats['episode_rewards'][i].append(r)
                self.stats['buffer_sizes_rl'][i].append(len(self.agents[i].rl_buffer))
                self.stats['buffer_sizes_sl'][i].append(len(self.agents[i].sl_buffer))
                
    def train(self):
        """Main training loop with Resume and Interrupt handling."""
        print("Starting NFSP Training...")
        print(f"Total Target Episodes: {self.num_episodes}")
        
        start_episode = self.hand_counter
        if start_episode >= self.num_episodes:
            print(f"Training already completed ({start_episode}/{self.num_episodes}).")
            return

        print(f"Resuming from Episode {start_episode}...")
        session_start_time = time.time()
        completed_normally = False

        try:
            for episode in range(start_episode, self.num_episodes):
                self.hand_counter = episode 

                # 1. Training Step
                for agent in self.agents:
                    agent.set_mode('train')
                
                rewards, hand_data = self.play_episode(self.env, self.agents, logger=self.logger, training=True, episode_num=episode)
                self._update_stats(rewards)
                if hand_data and (episode % 100 == 0): self.plotter.update(hand_data, episode)

                # 2. Evaluation Step
                if episode > 0 and episode % self.eval_interval == 0:
                    self._evaluate_performance(episode)

                # 3. Save Step
                if episode > 0 and episode % self.save_interval == 0:
                    self._perform_save(session_start_time, episode)

            completed_normally = True

        except KeyboardInterrupt:
            print(f"\nTraining interrupted by user at Episode {self.hand_counter}!")
            # Graceful save ONLY on manual interrupt
            self._perform_save(session_start_time, self.hand_counter)
        
        if completed_normally:
            print("Training completed successfully.")
            self._perform_save(session_start_time, self.hand_counter)

    def _perform_save(self, session_start_time, episode):
        """Helper to centralize saving logic."""
        current_session_duration = time.time() - session_start_time
        total_seconds = self.prior_training_time + current_session_duration
        self.stats['training_time'] = total_seconds
        
        time_str = str(datetime.timedelta(seconds=int(total_seconds)))
        print(f"\n[Saving] Episode {episode} | Total Time: {time_str}")
        
        self._save_models(episode, suffix="_latest")
        self._save_buffers()
        self._save_state()
                    
    def _evaluate_performance(self, episode):
        agent_to_eval = self.agents[0]
        agent_to_eval.set_mode('eval')
        eval_agents = [agent_to_eval, self.random_bot]

        wins = 0
        total_profit = 0
        eval_episodes = 1000 # Increase to 2000+ for higher precision/less luck
        
        bb_size = self.eval_env.big_blind
        for _ in range(eval_episodes):
            rewards, _ = self.play_episode(self.eval_env, eval_agents, logger=None)
            nfsp_reward = rewards[0]
            
            if nfsp_reward > 0: wins += 1
            total_profit += nfsp_reward

        avg_reward = total_profit / eval_episodes
        win_rate = wins / eval_episodes
        bb_per_100 = (avg_reward / bb_size) * 100

        print(f"Eval Episode {episode} vs RandomBot | Win Rate: {win_rate:.2%} | Avg Profit: {avg_reward:.2f} | Strength: {bb_per_100:.2f} bb/100")
        if avg_reward > self.best_avg_reward:
            print(f"New Best Performance found! (Avg Profit: {avg_reward:.2f} > Previous: {self.best_avg_reward:.2f})")
            self.best_avg_reward = avg_reward
            self._save_models(episode, suffix="_best")
            
        agent_to_eval.set_mode('train')

    def _save_state(self):
        """Saves the current training metadata AND stats."""
        state_path = os.path.join(self.output_dir, "training_state.json")
        
        # Convert deques to lists so JSON can handle them
        stats_to_save = {'episode_rewards': [list(d) for d in self.stats['episode_rewards']],
                         'buffer_sizes_rl': [list(d) for d in self.stats['buffer_sizes_rl']],
                         'buffer_sizes_sl': [list(d) for d in self.stats['buffer_sizes_sl']],
                         'training_time': self.stats['training_time']}

        state_data = {"hand_counter": self.hand_counter + 1, "best_avg_reward": self.best_avg_reward, "timestamp": time.time(), "stats": stats_to_save}
        
        with open(state_path, "w") as f:
            json.dump(state_data, f)
        print(f"Training state saved. (Hand {self.hand_counter})")

    def _load_state(self):
        """Loads the last training metadata and restores stats."""
        state_path = os.path.join(self.output_dir, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                data = json.load(f)
                self.hand_counter = data.get("hand_counter", 0)
                self.best_avg_reward = data.get("best_avg_reward", -float('inf'))
                
                if "stats" in data:
                    saved_stats = data["stats"]
                try:
                    self.stats['episode_rewards'] = [deque(l, maxlen=100) for l in saved_stats['episode_rewards']]
                    self.stats['buffer_sizes_rl'] = [deque(l, maxlen=100) for l in saved_stats['buffer_sizes_rl']]
                    self.stats['buffer_sizes_sl'] = [deque(l, maxlen=100) for l in saved_stats['buffer_sizes_sl']]
                    self.prior_training_time = saved_stats['training_time'] # Will KeyError if missing
                except KeyError as e:
                    raise KeyError(f"Corrupt training_state.json: Missing key {e}. Cannot restore training history.")
                    
            print(f"Resuming from Hand {self.hand_counter} (Best Reward: {self.best_avg_reward})")
        else:
            model_dir = os.path.join(self.output_dir, "models")
            buffer_dir = os.path.join(self.output_dir, "buffers")
            
            has_models = os.path.isdir(model_dir) and os.listdir(model_dir)
            has_buffers = os.path.isdir(buffer_dir) and os.listdir(buffer_dir)

            if has_models or has_buffers:
                raise RuntimeError(f"CRITICAL: No 'training_state.json' found, but existing models/buffers were detected in {self.output_dir}. "
                    "Refusing to silently restart and overwrite data. "
                    "Clear the directory to start fresh or restore the state file.")

            print("No previous training state or data found. Starting from Episode 0.")
            self.hand_counter = 0
            self.best_avg_reward = -float('inf')

    def _load_buffers(self):
        """Helper to load replay buffers. CRASHES if state exists but buffers don't."""
        buffer_dir = os.path.join(self.output_dir, "buffers")
        
        if self.hand_counter > 0 and not os.path.isdir(buffer_dir):
            raise FileNotFoundError(f"Resuming from Episode {self.hand_counter}, but buffer directory is missing: {buffer_dir}")
        if not os.path.isdir(buffer_dir): return # Safe to skip only if episode is 0

        print("Loading replay buffers...")
        for i, agent in enumerate(self.agents):
            rl_path = os.path.join(buffer_dir, f"agent{i}_rl_buffer.pkl")
            sl_path = os.path.join(buffer_dir, f"agent{i}_sl_buffer.pkl")
            if not os.path.exists(rl_path) or not os.path.exists(sl_path):
                 raise FileNotFoundError(f"Missing buffer file: {rl_path}")
            agent.load_buffers(rl_path, sl_path)

    def _save_buffers(self):
        """Helper to save replay buffers for all agents."""
        buffer_dir = os.path.join(self.output_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)
        
        print("Saving replay buffers...")
        for i, agent in enumerate(self.agents):
            rl_path = os.path.join(buffer_dir, f"agent{i}_rl_buffer.pkl")
            sl_path = os.path.join(buffer_dir, f"agent{i}_sl_buffer.pkl")
            agent.save_buffers(rl_path, sl_path)

    def _load_models(self, suffix: str):
        """Helper to load models. CRASHES if files are missing but expected."""
        save_dir = os.path.join(self.output_dir, "models")
        
        if self.hand_counter > 0 and not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Training state indicates Episode {self.hand_counter}, but no model directory found at {save_dir}.")

        print(f"Attempting to load models with suffix '{suffix}'...")
        
        for i, agent in enumerate(self.agents):
            br_path = os.path.join(save_dir, f"nfsp_agent{i}_br{suffix}.pt")
            as_path = os.path.join(save_dir, f"nfsp_agent{i}_as{suffix}.pt")
            
            if not os.path.exists(br_path) or not os.path.exists(as_path):
                if self.hand_counter == 0:
                    # Check if ANY model file exists to prevent accidental overwrite
                    if os.path.exists(br_path) or os.path.exists(as_path):
                        raise FileExistsError(f"Partial model files found at {br_path} but expected fresh start. Check directory.")
                    
                    print("No existing models found for Episode 0. Starting fresh.")
                    return
                else:
                    raise FileNotFoundError(f"Missing model file: {br_path}. Cannot resume training.")
            agent.load_models(br_path, as_path)
        print("Models loaded successfully.")
    
    def _save_models(self, episode: int, suffix: str):
        """Save agent models with a specific suffix (e.g., '_best', '_latest')."""
        save_dir = os.path.join(self.output_dir, "models")
        os.makedirs(save_dir, exist_ok=True)

        for i, agent in enumerate(self.agents):
            br_path = os.path.join(save_dir, f"nfsp_agent{i}_br{suffix}.pt")
            as_path = os.path.join(save_dir, f"nfsp_agent{i}_as{suffix}.pt")
            agent.save_models(br_path, as_path)
                    
def main(config_path: str = "config.yaml"):
    """Main training function."""
    # 1. Load configuration from the YAML file path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print("--- Configuration Loaded ---")
    print(json.dumps(config, indent=2))

    # 2. Pass the entire config object to the trainer
    trainer = NFSPTrainer(config=config)
    trainer.train()

if __name__ == "__main__":
    main()

