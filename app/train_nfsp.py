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

        self._load_buffers()

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
        """
        Unified game loop for both training and evaluation.
        """
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
        
        # Determine start point based on loaded state
        start_episode = self.hand_counter
        
        if start_episode >= self.num_episodes:
            print(f"Training already completed ({start_episode}/{self.num_episodes}).")
            print("Increase 'num_episodes' in config to continue training.")
            return

        print(f"Resuming from Episode {start_episode}...")
        session_start_time = time.time()
        try:
            for episode in range(start_episode, self.num_episodes):
                self.hand_counter = episode 

                # 1. Training Step
                for agent in self.agents:
                    agent.set_mode('train')
                
                rewards, hand_data = self.play_episode(self.env, self.agents, logger=self.logger, training=True, episode_num=episode)
                self._update_stats(rewards)
                if hand_data: self.plotter.update(hand_data, episode)

                # 2. Evaluation Step (Uses eval_interval)
                if episode > 0 and episode % self.eval_interval == 0:
                    self._evaluate_performance(episode)

                # 3. Save Step (Uses save_interval)
                if episode > 0 and episode % self.save_interval == 0:
                    current_session_duration = time.time() - session_start_time
                    total_seconds = self.prior_training_time + current_session_duration
                    self.stats['training_time'] = total_seconds
                    
                    time_str = str(datetime.timedelta(seconds=int(total_seconds)))
                    print(f"\n[Auto-Save] Episode {episode} | Total Time: {time_str}")
                    
                    self._save_models(episode, suffix="_latest")
                    self._save_buffers()
                    self._save_state()
                    
                    eps = self.agents[0].get_current_epsilon()
                    print(f"Current BR Epsilon: {eps:.4f}")

        except KeyboardInterrupt:
            print(f"\nTraining interrupted by user at Episode {self.hand_counter}!")
            
        except Exception as e:
            print(f"\nTraining crashed with error: {e}")
            raise e
            
        finally:
            current_session_duration = time.time() - session_start_time
            self.stats['training_time'] = self.prior_training_time + current_session_duration
            print("Saving final state...")
            self._save_models(self.hand_counter, suffix="_latest")
            self._save_buffers()
            self._save_state()
            print("Save complete. Exiting.")
                    
    def _evaluate_performance(self, episode):
        for agent in self.agents:
            agent.set_mode('eval')

        # Variables to track wins and rewards
        wins = [0, 0]
        total_rewards = [0, 0]
        eval_episodes = 1000

        for _ in range(eval_episodes):
            # Use play_episode with the eval environment and NO logger
            rewards, _ = self.play_episode(self.eval_env, self.agents, logger=None)
            
            if rewards[0] > 0: wins[0] += 1
            if rewards[1] > 0: wins[1] += 1
            
            total_rewards[0] += rewards[0]
            total_rewards[1] += rewards[1]

        # Calculate average reward for Agent 0 (or average of both if symmetric)
        avg_reward_agent0 = total_rewards[0] / eval_episodes
        
        print(f"Eval Episode {episode} | Agent 0 Win Rate: {wins[0]/eval_episodes:.2%} | Avg Reward: {avg_reward_agent0:.2f}")
        
        # --- FIX: Save Best Model Logic ---
        if avg_reward_agent0 > self.best_avg_reward:
            print(f"New Best Performance! (Reward: {avg_reward_agent0:.2f} > {self.best_avg_reward:.2f})")
            self.best_avg_reward = avg_reward_agent0
            self._save_models(episode, suffix="_best")
        # ----------------------------------

        # Switch back to train mode
        for agent in self.agents:
            agent.set_mode('train')

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
                    self.stats['episode_rewards'] = [deque(l, maxlen=100) for l in saved_stats.get('episode_rewards', [[],[]])]
                    self.stats['buffer_sizes_rl'] = [deque(l, maxlen=100) for l in saved_stats.get('buffer_sizes_rl', [[],[]])]
                    self.stats['buffer_sizes_sl'] = [deque(l, maxlen=100) for l in saved_stats.get('buffer_sizes_sl', [[],[]])]
                    self.prior_training_time = saved_stats.get('training_time', 0)
                    self.stats['training_time'] = self.prior_training_time
                    print(" - Restored console statistics (smooth graphs will continue).")
                    
            print(f"Resuming from Hand {self.hand_counter} (Best Reward: {self.best_avg_reward})")
        else:
            print("No previous training state found. Starting from Episode 0.")
            self.hand_counter = 0
            self.best_avg_reward = -float('inf')

    def _load_buffers(self):
        """Helper to load replay buffers for all agents."""
        buffer_dir = os.path.join(self.output_dir, "buffers")
        if not os.path.isdir(buffer_dir):
            print("No buffer directory found, skipping buffer loading.")
            return

        print("Attempting to load replay buffers...")
        for i, agent in enumerate(self.agents):
            rl_path = os.path.join(buffer_dir, f"agent{i}_rl_buffer.pkl")
            sl_path = os.path.join(buffer_dir, f"agent{i}_sl_buffer.pkl")
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
        """Helper to load models for all agents with a given suffix."""
        save_dir = os.path.join(self.output_dir, "models")
        if not os.path.isdir(save_dir):
            print("No models directory found, starting from scratch.")
            return

        print(f"Attempting to load models with suffix '{suffix}' to resume training...")
        try:
            # Check if the first agent's model exists as a proxy for all
            br_path_check = os.path.join(save_dir, f"nfsp_agent0_br{suffix}.pt")
            if not os.path.exists(br_path_check):
                print("No models found to load. Starting from scratch.")
                return

            for i, agent in enumerate(self.agents):
                br_path = os.path.join(save_dir, f"nfsp_agent{i}_br{suffix}.pt")
                as_path = os.path.join(save_dir, f"nfsp_agent{i}_as{suffix}.pt")
                agent.load_models(br_path, as_path)
                
            print("Models loaded successfully. Resuming training.")
        except Exception as e:
            print(f"Could not load models: {e}. Starting from scratch.")
    
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
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return # Exit gracefully if config is missing
        
    print("--- Configuration Loaded ---")
    print(json.dumps(config, indent=2))
    print("--------------------------")

    # 2. Pass the entire config object to the trainer
    trainer = NFSPTrainer(config=config)
    trainer.train()

if __name__ == "__main__":
    main()

