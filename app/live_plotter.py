# app/live_plotter.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LivePlotter:
    def __init__(self, plot_file: str, csv_file: str = "training_metrics.csv", batch_size: int = 100):
        self.plot_file = plot_file
        self.csv_file = csv_file
        self.batch_size = batch_size 
        self.csv_buffer = []
                
        # Check if CSV exists to ensure headers
        if not os.path.exists(self.csv_file):
            pd.DataFrame(columns=['hand_id', 'vpip', 'pfr', 'reward', 'afq', 'entropy']).to_csv(self.csv_file, index=False)

    def update(self, hand_data: dict, episode_num: int):
        stats = self._extract_stats(hand_data)
        if stats:
            stats['hand_id'] = episode_num
            self.csv_buffer.append(stats)

        if len(self.csv_buffer) >= self.batch_size:
            self._flush_to_disk()
            self.save_plot() 

    def save_plot(self, window_size=1000): 
        if not os.path.exists(self.csv_file): return

        try:
            # Read full history
            df = pd.read_csv(self.csv_file)
            if df.empty or len(df) < 10: return
            
            if 'hand_id' in df.columns:
                df.set_index('hand_id', inplace=True)
            
            # Calculate rolling average
            rolling = df.rolling(window=window_size, min_periods=10).mean()
            
            # Downsample for speed
            if len(rolling) > 5000:
                plot_data = rolling.iloc[::max(1, len(rolling)//2000)]
            else:
                plot_data = rolling

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            (ax1, ax2), (ax3, ax4) = axes
            
            # --- Plot 1: Preflop Strategy (VPIP / PFR) ---
            ax1.plot(plot_data.index, plot_data['vpip'], label='VPIP', color='blue', linewidth=1.5)
            ax1.plot(plot_data.index, plot_data['pfr'], label='PFR', color='orange', linewidth=1.5)
            
            # Reference Lines (Heads Up Norms)
            # VPIP in HU is very high (SB plays ~80-95%)
            ax1.axhline(0.90, color='blue', linestyle='--', alpha=0.3, label='Ref VPIP (~0.9)')
            ax1.axhline(0.55, color='orange', linestyle='--', alpha=0.3, label='Ref PFR (~0.55)')
            ax1.set_title('P0 AS Preflop Strategy')
            ax1.legend(loc='lower right', fontsize='small')
            ax1.grid(True, alpha=0.2)
            # ax1.set_ylim(-0.05, 1.05)

            # --- Plot 2: Reward ---
            ax2.plot(plot_data.index, plot_data['reward'], color='green', linewidth=1)
            ax2.axhline(0, color='black', lw=1, linestyle='-')
            current_reward = rolling['reward'].iloc[-1] if not rolling.empty else 0
            ax2.set_title(f"P0 AS Avg Reward (Last: {current_reward:.2f})")
            ax2.grid(True, alpha=0.2)

            # --- Plot 3: Aggression Frequency (AFq) ---
            ax3.plot(plot_data.index, plot_data['afq'], color='red', linewidth=1.5)
            ax3.axhline(0.50, color='red', linestyle='--', alpha=0.3, label='Ref AFq (0.5)')
            ax3.set_title('P0 AS Aggression Freq')
            ax3.legend(loc='lower right', fontsize='small')
            ax3.grid(True, alpha=0.2)
            # ax3.set_ylim(0, 1.0)

            # --- Plot 4: Entropy ---
            if 'entropy' in plot_data:
                clean_entropy = plot_data['entropy'].fillna(0)
                ax4.plot(plot_data.index, clean_entropy, color='purple', linewidth=1.5)
            ax4.set_title('P0 AS Entropy (Policy Uncertainty)')
            ax4.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(self.plot_file)
            plt.close(fig)
            
        except Exception as e:
            print(f"[LivePlotter] Error generating plot: {e}")

    def _flush_to_disk(self):
        if not self.csv_buffer: return
        
        df = pd.DataFrame(self.csv_buffer)
        expected_cols = ['hand_id', 'vpip', 'pfr', 'reward', 'afq', 'entropy']
        # Ensure columns exist even if data is missing
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        df = df[expected_cols]
        df.to_csv(self.csv_file, mode='a', header=False, index=False)
        self.csv_buffer = []

    def _extract_stats(self, hand_data):
        try:
            # We are tracking Player 0 (The Agent)
            p0_actions = [a for a in hand_data.get('actions', []) if a['player'] == 0]
            if not p0_actions: return None
            if p0_actions[0].get('policy', 'Unknown') == 'BR': return None 

            # Preflop stats
            preflop = [a for a in p0_actions if a['street'] == 0]
            vpip = 1.0 if any(a['action_type'] in [1, 2] for a in preflop) else 0.0
            pfr = 1.0 if any(a['action_type'] == 2 for a in preflop) else 0.0
            
            # Aggression Freq (Raise / (Call + Raise + Fold))
            raise_count = sum(1 for a in p0_actions if a['action_type'] == 2)
            afq = raise_count / len(p0_actions) if p0_actions else 0.0

            # Entropy
            valid_entropies = [a['entropy'] for a in p0_actions if 'entropy' in a and a['entropy'] is not None]
            
            # If no entropy data found, default to 0.0 so the plot isn't blank
            avg_ent = np.mean(valid_entropies) if valid_entropies else 0.0

            return {'vpip': vpip, 'pfr': pfr, 'reward': hand_data['rewards'][0], 'afq': afq, 'entropy': avg_ent}
        except Exception:
            return None
        
