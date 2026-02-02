# app/train_nfsp.py
# python -m app.train_nfsp
# python -m cProfile -o training.pstats app/train_nfsp.py
# snakviz training.psteats
import os
import torch
import numpy as np
import json
import time
import yaml
from typing import Dict, List
from app.TexasHoldemEnv import TexasHoldemEnv
from app.nfsp_components import NFSPAgent
from app.poker_agents import RandomBot
from app._hand_history_logger import HandHistoryLogger
from app.live_plotter import LivePlotter

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
        self.intelligent_trials = config['simulations']['intelligent_equity_trials']

        # Initialize environments and agents
        self.env = TexasHoldemEnv(num_players=2, starting_stack=200)
        self.eval_env = TexasHoldemEnv(num_players=2, starting_stack=200)
        self.random_bot = RandomBot(seat_id=1, aggression=0.3)

        self.agents = []
        for i in range(2):
            agent = NFSPAgent(
                seat_id=i,
                agent_config=config['agent'],
                buffer_config=config['buffers'],
                random_equity_trials=self.random_trials,
                intelligent_equity_trials=self.intelligent_trials
            )
            self.agents.append(agent)

        self._load_buffers()

        # === Give agents a reference to their opponent's network ===
        self.agents[0].opponent_as_network = self.agents[1].as_network
        self.agents[1].opponent_as_network = self.agents[0].as_network

        # Training statistics
        self.stats = {
            'episode_rewards': [[], []],
            'buffer_sizes_rl': [[], []],
            'buffer_sizes_sl': [[], []],
            'training_time': 0
        }

        # Track best performance
        self.best_avg_reward = -float('inf')

        # --- Initialize the logger ---
        log_path = os.path.join(self.output_dir, "hand_history.log")
        data_path = os.path.join(self.output_dir, "hand_data.jsonl")
        plot_path = os.path.join(self.output_dir, "training_dashboard.png") # New image file
        self.logger = HandHistoryLogger(log_file=log_path, data_file=data_path, dump_features=config['logging']['dump_features'])
        self.hand_counter = 0
        if os.path.exists(data_path):
            try:
                # Count lines in the file to determine current hand number
                with open(data_path, 'r') as f:
                    self.hand_counter = sum(1 for _ in f)
                print(f"Resuming stats from Hand {self.hand_counter}...")
            except Exception:
                print("Could not read existing data, starting counter at 0.")
        self.plotter = LivePlotter(data_file=data_path, plot_file=plot_path, window_size=1000)
        
        # === Attempt to load latest models to resume training ===
        self._load_models(suffix="_latest")
        
    def train(self):
        """Main training loop."""
        print("Starting NFSP Training...")
        print(f"Episodes: {self.num_episodes}")
        print(f"Agents: {len(self.agents)}")
        
        start_time = time.time()
        
        try:
            for episode in range(self.num_episodes):
                # Pass episode number to _run_episode
                episode_rewards = self._run_episode(episode)
                
                # Update statistics
                for i, reward in enumerate(episode_rewards):
                    self.stats['episode_rewards'][i].append(reward)
                    self.stats['buffer_sizes_rl'][i].append(len(self.agents[i].rl_buffer))
                    self.stats['buffer_sizes_sl'][i].append(len(self.agents[i].sl_buffer))
                
                # Periodic evaluation
                if episode % self.eval_interval == 0 and episode > 0:
                    self._evaluate_performance(episode)
                    self._save_models(episode, suffix="_latest")
                    
                # Progress reporting
                if episode % 100 == 0:
                    avg_reward_0 = np.mean(self.stats['episode_rewards'][0][-100:]) if len(self.stats['episode_rewards'][0]) >= 100 else 0
                    avg_reward_1 = np.mean(self.stats['episode_rewards'][1][-100:]) if len(self.stats['episode_rewards'][1]) >= 100 else 0
                    print(f"Episode {episode}: Agent0={avg_reward_0:.2f}, Agent1={avg_reward_1:.2f}")

        except KeyboardInterrupt:
            print("\nTraining interrupted.")

        finally:
            self.stats['training_time'] = time.time() - start_time
            print(f"\nTraining completed or interrupted after {self.stats['training_time']:.2f} seconds.\nContinue training with python -m app.train_nfsp.")
            
            # Final save of stats and buffers
            self._save_buffers()
        
    def _run_episode(self, episode: int) -> List[float]:
        """Run a single training episode with logging."""
        self.hand_counter += 1
        
        # Start a new hand with preserved stacks if the tournament is ongoing.
        state = self.env.reset()
        
        for agent in self.agents:
            agent.new_hand()
        
        self.logger.log_start_hand(episode, self.hand_counter, state)
        last_stage = 0
        
        episode_rewards = [0.0, 0.0]
        initial_stacks = state.starting_stacks_this_hand.copy()
        
        while not state.terminal:
            player_to_move = state.to_move
            agent = self.agents[player_to_move]
            
            state_before = state.copy()
            
            # Unpack the `features_schema` object
            action, amount, predictions, policy_name, features_schema = agent.compute_action(state)
            
            # Pass the object to the logger
            self.logger.log_action(
                player_to_move, action, amount, state_before, 
                predictions, policy_name, features_schema
            )

            if features_schema:
                # Add the `predictions` object to this call
                self.logger.log_feature_dump_if_needed(state_before, features_schema, predictions)

            next_state, done = self.env.step(action, amount)
            
            # Let all agents observe the action
            for i, obs_agent in enumerate(self.agents):
                obs_agent.observe((action, amount), player_to_move, state_before, next_state)
            
            state = next_state
            
            if state.stage > last_stage:
                self.logger.log_street(state)
                last_stage = state.stage

        # Calculate final rewards and let agents observe showdown
        final_stacks = state.stacks
        for i in range(len(self.agents)):
            episode_rewards[i] = final_stacks[i] - initial_stacks[i]
            
        # Create showdown state for agents
        showdown_state = {
            'stacks_before': {i: initial_stacks[i] for i in range(len(initial_stacks))},
            'stacks_after': {i: final_stacks[i] for i in range(len(final_stacks))}
        }
        
        for agent in self.agents:
            agent.observe_showdown(showdown_state)

        self.logger.log_end_hand(episode_rewards, state)

        # Capture the data returned by the logger
        hand_data = self.logger.log_end_hand(episode_rewards, state)
        
        # Update the live plotter memory
        self.plotter.update(hand_data)
        
        # Update the actual image file periodically (e.g., every 50 hands)
        # Doing this every hand is too slow (IO heavy)
        if self.hand_counter % 50 == 0:
            self.plotter.save_plot()
            
        return episode_rewards
        
        return episode_rewards
        
    def _evaluate_performance(self, episode: int):
        """Evaluate agent performance against random baseline AND save best models."""
        print(f"\n--- Evaluation at Episode {episode} ---")
        
        # Test each agent against RandomBot
        for i, agent in enumerate(self.agents):
            wins = 0
            total_reward = 0
            eval_episodes = 100
            
            self.random_bot.seat_id = 1 - i  # Just update the seat_id
            
            eval_agents = [None, None]
            eval_agents[i] = agent
            eval_agents[1-i] = self.random_bot
            
            for _ in range(eval_episodes):
                # Run episode by just resetting the existing environment
                state = self.eval_env.reset()
                agent.new_hand()
                self.random_bot.new_hand()
                
                initial_stack = state.starting_stacks_this_hand[i]
                
                while not state.terminal:
                    player_to_move = state.to_move
                    current_agent = eval_agents[player_to_move]
                    
                    # Unpack all 5 values to avoid an error
                    action, amount, _, _, _ = current_agent.compute_action(state)
                    state, _ = self.eval_env.step(action, amount)
                    
                final_stack = state.stacks[i]
                reward = final_stack - initial_stack
                total_reward += reward
                
                if reward > 0:
                    wins += 1
                    
            win_rate = wins / eval_episodes
            avg_reward = total_reward / eval_episodes
            
            print(f"Agent {i}: Win Rate = {win_rate:.2%}, Avg Reward = {avg_reward:.2f}")

            # Check if this is the new best performance (track Agent 0)
            if i == 0 and avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                print(f"New best performance for Agent 0! Avg Reward: {avg_reward:.2f}. Saving best models...")

                self._save_models(episode, suffix="_best")
            
        print("--- End Evaluation ---\n")

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
    # The script still runs normally from the command line, using the default "config.yaml"
    main()

