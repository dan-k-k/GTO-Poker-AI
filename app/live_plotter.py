# app/live_plotter.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class LivePlotter:
    def __init__(self, plot_file: str, csv_file: str = "training_metrics.csv", batch_size: int = 100):
        self.plot_file = plot_file
        self.csv_file = csv_file
        self.batch_size = batch_size 

        self.csv_buffer = []
        self.plot_data = deque(maxlen=1000)

        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_file):
            # Create the file with headers
            pd.DataFrame(columns=['vpip', 'pfr', 'reward', 'afq', 'entropy']).to_csv(self.csv_file, index=False)

    def update(self, hand_data: dict):
        stats = self._extract_stats(hand_data)
        if stats:
            self.csv_buffer.append(stats)
            self.plot_data.append(stats) 

        if len(self.csv_buffer) >= self.batch_size:
            self._flush_to_disk()

    def _flush_to_disk(self):
        """Writes current buffer to CSV and clears memory."""
        if not self.csv_buffer:
            return
        
        df = pd.DataFrame(self.csv_buffer)
        df.to_csv(self.csv_file, mode='a', header=False, index=False)
        self.csv_buffer = []

    def save_plot(self, window_size=1000): # Reduced window size since we only have 1000 pts
        """Generates plot from RAM buffer (Fast)."""
        
        if len(self.plot_data) < 10: 
            return

        # Create DataFrame directly from RAM 
        df = pd.DataFrame(self.plot_data)
        
        # Rolling window on the live RAM data
        rolling = df.rolling(window=window_size, min_periods=5).mean()
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
        (ax1, ax2), (ax3, ax4) = axes
        
        # 1. Strategy (VPIP/PFR)
        ax1.plot(df.index, rolling['vpip'], label='VPIP', color='blue', lw=1.5)
        ax1.plot(df.index, rolling['pfr'], label='PFR', color='orange', lw=1.5)
        ax1.set_title('Preflop Strategy')
        ax1.set_ylabel('Freq')
        ax1.legend()
        ax1.grid(alpha=0.2)

        # 2. Profitability
        ax2.plot(df.index, rolling['reward'], color='green', lw=1)
        ax2.axhline(0, color='black', lw=0.5)
        ax2.set_title(f"Avg Reward (Last: {rolling['reward'].iloc[-1]:.2f})")
        ax2.set_ylabel('BB/Hand')
        ax2.grid(alpha=0.2)

        # 3. Aggression
        ax3.plot(df.index, rolling['afq'], color='red', lw=1)
        ax3.set_title('Aggression Frequency')
        ax3.set_ylabel('Freq')
        ax3.set_xlabel('Hands')
        ax3.grid(alpha=0.2)

        # 4. Entropy
        if 'entropy' in rolling and rolling['entropy'].notna().any():
            ax4.plot(df.index, rolling['entropy'], color='purple', lw=1)
        ax4.set_title('Policy Entropy')
        ax4.set_xlabel('Hands')
        ax4.grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig(self.plot_file)
        plt.close(fig)

    def _extract_stats(self, row):
        """Helper to parse raw hand_data safely."""
        try:
            p0_actions = [a for a in row.get('actions', []) if a['player'] == 0]
            if not p0_actions: return None

            # Only track Average Strategy (AS)
            # Safe get for policy name in case it's missing from some actions
            policies = [a.get('policy') for a in p0_actions if 'policy' in a]
            if not policies or 'AS' not in policies[0]: 
                return None

            # Calc Stats
            preflop = [a for a in p0_actions if a['street'] == 0]
            vpip = 1 if any(a['action_type'] in [1, 2] for a in preflop) else 0
            pfr = 1 if any(a['action_type'] == 2 for a in preflop) else 0
            
            aggressions = sum(1 for a in p0_actions if a['action_type'] == 2)
            afq = aggressions / len(p0_actions) if p0_actions else 0

            entropies = [a['entropy'] for a in p0_actions if 'entropy' in a and a['entropy'] is not None]
            avg_ent = np.mean(entropies) if entropies else None

            return {
                'vpip': vpip,
                'pfr': pfr,
                'reward': row['rewards'][0],
                'afq': afq,
                'entropy': avg_ent
            }
        except Exception:
            return None
        
