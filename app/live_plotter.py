# app/live_plotter.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

class LivePlotter:
    def __init__(self, data_file, plot_file, window_size=1000):
        self.data_file = data_file
        self.plot_file = plot_file
        self.window_size = window_size
        self.history = []
        self._load_initial_history()

    def _load_initial_history(self):
        if not os.path.exists(self.data_file):
            return

        print("Rebuilding plot history from log file...")
        try:
            # Read line by line to handle potential malformed lines gracefully
            # or just use pandas with error_bad_lines=False (deprecated) or on_bad_lines='skip'
            df = pd.read_json(self.data_file, lines=True)
            if df.empty: return
            
            for _, row in df.iterrows():
                self._process_row(row)
                
            print(f"History loaded: {len(self.history)} hands.")
        except Exception as e:
            print(f"Warning: Could not load plot history: {e}")

    def update(self, hand_data):
        self._process_row(hand_data)

    def _process_row(self, row):
        actions = row['actions']
        rewards = row['rewards']
        
        p0_actions = [a for a in actions if a['player'] == 0]
        if not p0_actions: return

        # Only track Average Strategy (AS) for convergence stats
        current_policy = p0_actions[0].get('policy', 'unknown')
        if current_policy != 'AS': return 

        # --- 1. Basic Stats (VPIP / PFR / Reward) ---
        p0_preflop = [a for a in p0_actions if a['street'] == 0]
        is_vpip = 1 if any(a['action_type'] in [1, 2] for a in p0_preflop) else 0
        is_pfr = 1 if any(a['action_type'] == 2 for a in p0_preflop) else 0
        reward = rewards[0]

        # --- 2. Aggression Frequency (AFq) ---
        # AFq = (Bets + Raises) / (Total Actions)
        # Note: Some definitions exclude checks, but (Aggressive / Total) is a good proxy for neural nets.
        n_actions = len(p0_actions)
        n_aggressive = sum(1 for a in p0_actions if a['action_type'] == 2)
        hand_afq = (n_aggressive / n_actions) if n_actions > 0 else 0.0

        # --- 3. Policy Entropy ---
        # Get average entropy of all actions taken this hand
        entropies = [a.get('entropy') for a in p0_actions if a.get('entropy') is not None]
        avg_entropy = np.mean(entropies) if entropies else None

        self.history.append({
            'vpip': is_vpip,
            'pfr': is_pfr,
            'reward': reward,
            'afq': hand_afq,
            'entropy': avg_entropy
        })

    def save_plot(self):
        if len(self.history) < 10: return

        df = pd.DataFrame(self.history)
        
        # Calculate Rolling Stats
        rolling_df = df.rolling(window=self.window_size, min_periods=10).mean()
        
        # Setup Figure: 2x2 Grid
        fig, axes = plt.subplots(2, 2, figsize=(9.8, 7), sharex=True)
        ax1, ax2 = axes[0, 0], axes[0, 1]
        ax3, ax4 = axes[1, 0], axes[1, 1]
        
        # --- Plot 1: VPIP & PFR ---
        ax1.plot(df.index, rolling_df['vpip'], label='VPIP', color='blue')
        ax1.plot(df.index, rolling_df['pfr'], label='PFR', color='orange')
        ax1.axhline(y=0.65, color='blue', linestyle='--', alpha=0.3, label='Target ~0.65')
        ax1.axhline(y=0.50, color='orange', linestyle='--', alpha=0.3, label='Target ~0.5')
        
        ax1.set_title(f'Preflop Strategy (Rolling {self.window_size})')
        ax1.set_ylabel('Frequency')  # MOVED from xlabel
        ax1.set_ylim(0, 1.0)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.15)
        
        # --- Plot 2: Profitability ---
        ax2.plot(df.index, rolling_df['reward'], label='Avg Reward', color='green')
        ax2.axhline(0, color='black', linewidth=1)
        if len(rolling_df) > 0:
            last_val = rolling_df['reward'].iloc[-1]
            ax2.text(df.index[-1], last_val, f"{last_val:.2f}", ha='right', va='bottom')
            
        ax2.set_title('Profitability')
        ax2.set_ylabel('BB/Hand')  # MOVED from xlabel
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.15)

        # --- Plot 3: Aggression Frequency (AFq) ---
        ax3.plot(df.index, rolling_df['afq'], label='Aggression Freq', color='red')
        ax3.axhline(y=0.50, color='red', linestyle='--', alpha=0.3, label='Target ~0.5')
        
        ax3.set_title('Aggression')
        ax3.set_ylabel('Frequency') # MOVED from xlabel
        ax3.set_xlabel('# Hands') # Correct X-label
        ax3.set_ylim(0, 1.0)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.15)

        # --- Plot 4: Policy Entropy ---
        if 'entropy' in rolling_df and rolling_df['entropy'].notna().any():
            ax4.plot(df.index, rolling_df['entropy'], label='Entropy', color='purple')
            ax4.set_title('Policy Entropy (Uncertainty)')
            ax4.set_ylabel('Nats')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.15)
        else:
            ax4.text(0.5, 0.5, 'No Entropy Data Yet', ha='center', va='center')
            
        ax4.set_xlabel('# Hands') # Added X-label

        plt.tight_layout()
        plt.savefig(self.plot_file)
        plt.close(fig)

