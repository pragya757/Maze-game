import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

# Create training_plots directory if it doesn't exist
plots_dir = Path('training_plots')
plots_dir.mkdir(exist_ok=True)

def find_latest_metrics_file():
    """Find the most recent metrics file"""
    # Look in both current directory and training_results directory
    search_paths = [Path('.'), Path('training_results')]
    metrics_files = []
    
    for path in search_paths:
        metrics_files.extend(list(path.glob('final_training_results.xlsx')))
    
    if not metrics_files:
        return None
    return max(metrics_files, key=os.path.getmtime)

try:
    # Find and read the latest metrics file
    metrics_file = find_latest_metrics_file()
    if not metrics_file:
        raise FileNotFoundError("No metrics file found. Please run the training script first.")
    
    print(f"Reading metrics from: {metrics_file}")
    df = pd.read_excel(metrics_file)
    
    # Set up plot parameters
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    
    # Calculate max episode for x-axis limits
    max_episode = df['Episode'].max()
    x_ticks = np.linspace(0, max_episode, min(10, max_episode), dtype=int)

    # 1. Cumulative Reward per Episode
    plt.figure()
    try:
        plt.plot(df['Episode'], df['Cumulative_Reward'], 'b-', alpha=0.7)
        plt.title('Cumulative Reward per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.xticks(x_ticks)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'cumulative_reward.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot cumulative reward: {e}")

    # 2. Episode Reward per Episode
    plt.figure()
    try:
        plt.plot(df['Episode'], df['Episode_Reward'], 'g-', alpha=0.7)
        plt.title('Episode Reward per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Episode Reward')
        plt.xticks(x_ticks)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'episode_reward.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot episode reward: {e}")

    # 3. Steps per Episode
    plt.figure()
    try:
        plt.plot(df['Episode'], df['Steps'], 'r-', alpha=0.7)
        plt.title('Steps per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Steps')
        plt.xticks(x_ticks)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'steps.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot steps: {e}")

    # 2. Return per Episode
    plt.figure()
    try:
        plt.plot(df['Episode'], df['Return'], 'g-', alpha=0.7)
        plt.title('Return per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Return')
        plt.xticks(x_ticks)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'returns.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot returns: {e}")

    # 3. Epsilon per Episode
    if 'Epsilon' in df.columns:
        plt.figure()
        try:
            plt.plot(df['Episode'], df['Epsilon'], 'm-', alpha=0.7)
            plt.title('Epsilon per Episode')
            plt.xlabel('Episodes')
            plt.ylabel('Epsilon')
            plt.xticks(x_ticks)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / 'epsilon.png')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not plot epsilon: {e}")


    # 5. Loss per Episode (if available)
    if 'Loss' in df.columns:
        plt.figure()
        try:
            plt.plot(df['Episode'], df['Loss'], 'b-', alpha=0.7)
            plt.title('Loss per Episode')
            plt.xlabel('Episodes')
            plt.ylabel('Loss')
            plt.xticks(x_ticks)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / 'temporal_difference.png')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not plot loss: {e}")

    print(f"\nAll plots have been generated in the '{plots_dir}' directory!")

except Exception as e:
    print(f"Error generating plots: {e}")
    sys.exit(1)
