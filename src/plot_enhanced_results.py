"""
Create visualizations:
- Episode-by-episode learning curves (reward and frames over training episodes)
- Final evaluation comparison table
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

FIGS_DIR = "figs"
os.makedirs(FIGS_DIR, exist_ok=True)

def plot_training_progress(episode_log_path: str = "runs/training_episodes.csv", window_size: int = 100):
    """
    Create training progress plots
    Shows average reward and frames over episodes with smoothing
    """
    if not os.path.exists(episode_log_path):
        print(f"Episode log not found: {episode_log_path}")
        return
    
    df = pd.read_csv(episode_log_path)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for algo in df['algo'].unique():
        algo_data = df[df['algo'] == algo].sort_values('episode')
        
        # Rolling average for smoothing
        episodes = algo_data['episode'].values
        rewards = algo_data['reward'].values
        lengths = algo_data['length'].values
        
        # Calculate rolling means
        if len(rewards) >= window_size:
            rewards_smooth = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
            lengths_smooth = pd.Series(lengths).rolling(window=window_size, min_periods=1).mean()
        else:
            rewards_smooth = rewards
            lengths_smooth = lengths
        
        # Plot reward
        ax1.plot(episodes, rewards_smooth, label=algo, linewidth=2, marker='o', 
                 markersize=4, markevery=max(1, len(episodes)//10))
        
        # Plot frames (length)
        ax2.plot(episodes, lengths_smooth, label=algo, linewidth=2, marker='o',
                 markersize=4, markevery=max(1, len(episodes)//10))
    
    # Configure reward plot
    ax1.set_xlabel('Episodes', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title(f'Average total reward (smoothed, window={window_size})', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Configure frames plot
    ax2.set_xlabel('Episodes', fontsize=12)
    ax2.set_ylabel('Frames', fontsize=12)
    ax2.set_title(f'Average number of frames (smoothed, window={window_size})', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(FIGS_DIR, "training_progress_episodes.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training progress plot to {output_path}")
    plt.close()

def plot_raw_training_progress(episode_log_path: str = "runs/training_episodes.csv", sample_every: int = 10):
    """
    Create raw (unsmoothed) training plots showing all episodes
    Useful for seeing variance and instability
    """
    if not os.path.exists(episode_log_path):
        print(f"Episode log not found: {episode_log_path}")
        return
    
    df = pd.read_csv(episode_log_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for algo in df['algo'].unique():
        algo_data = df[df['algo'] == algo].sort_values('episode')
        
        # Sample to reduce clutter
        sampled = algo_data.iloc[::sample_every]
        
        episodes = sampled['episode'].values
        rewards = sampled['reward'].values
        lengths = sampled['length'].values
        
        # Plot with transparency
        ax1.scatter(episodes, rewards, label=algo, alpha=0.5, s=20)
        ax2.scatter(episodes, lengths, label=algo, alpha=0.5, s=20)
    
    ax1.set_xlabel('Episodes', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Episode Rewards (raw, sampled)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Episodes', fontsize=12)
    ax2.set_ylabel('Frames', fontsize=12)
    ax2.set_title('Episode Lengths (raw, sampled)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(FIGS_DIR, "training_progress_raw.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved raw training plot to {output_path}")
    plt.close()

def create_final_comparison_table(summary_path: str = "results/final_evaluation_summary.csv"):
    if not os.path.exists(summary_path):
        print(f"Summary not found: {summary_path}")
        return
    
    df = pd.read_csv(summary_path)
    
    # Select key columns for table
    table_data = df[['algorithm', 'episodes', 'avg_survival_sec', 'avg_score', 'best_score']].copy()
    table_data.columns = ['Algorithm', 'Episodes', 'Avg Survival (s)', 'Avg Score', 'Best Score']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(table_data.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#5DADE2')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#EBF5FB')
    
    plt.title('Final Evaluation Results', fontsize=14, fontweight='bold', pad=20)
    
    output_path = os.path.join(FIGS_DIR, "final_comparison_table.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison table to {output_path}")
    plt.close()

def plot_checkpoint_learning_curves(checkpoint_path: str = "runs/metrics_dino_checkpoints.csv"):
    """
    Plot checkpoint-based learning curves
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint log not found: {checkpoint_path}")
        return
    
    df = pd.read_csv(checkpoint_path)
    
    # Calculate mean per checkpoint
    summary = df.groupby(['algo', 'checkpoint'])['score'].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in summary['algo'].unique():
        algo_data = summary[summary['algo'] == algo].sort_values('checkpoint')
        
        checkpoints = algo_data['checkpoint'].values / 1000  # Convert to K
        means = algo_data['mean'].values
        stds = algo_data['std'].values
        
        ax.plot(checkpoints, means, label=algo.upper(), marker='o', linewidth=2, markersize=8)
        ax.fill_between(checkpoints, means - stds, means + stds, alpha=0.2)
    
    ax.set_xlabel('Training Steps (K)', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('Learning Curves (Checkpoint Evaluations)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(FIGS_DIR, "checkpoint_learning_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved checkpoint learning curves to {output_path}")
    plt.close()

def create_all_visualizations():
    """Generate all visualizations"""
    print("\n" + "="*70)
    print("  CREATING ENHANCED VISUALIZATIONS")
    print("="*70 + "\n")
    
    # 1. Episode-level training progress
    print("[1/5] Training progress by episode...")
    plot_training_progress()
    
    # 2. Raw training progress
    print("[2/5] Raw training progress...")
    plot_raw_training_progress()
    
    # 3. Checkpoint learning curves
    print("[3/5] Checkpoint learning curves...")
    plot_checkpoint_learning_curves()
    
    # 4. Final comparison table
    print("[4/5] Final comparison table...")
    create_final_comparison_table()
    
    # 5. Distribution plots
    print("[5/5] Score distributions...")
    plot_score_distributions()
    
    print("\n" + "="*70)
    print("  VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nAll figures saved to {FIGS_DIR}/")

def plot_score_distributions():
    """Plot score distributions for final evaluation"""
    final_eval_path = "runs/metrics_final_evaluation.csv"
    
    if not os.path.exists(final_eval_path):
        print(f"  (Skipping - file not found: {final_eval_path})")
        return
    
    df = pd.read_csv(final_eval_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algos = df['algo'].unique()
    data_to_plot = [df[df['algo'] == algo]['score'].values for algo in algos]
    
    bp = ax.boxplot(data_to_plot, labels=algos, patch_artist=True, showmeans=True)
    
    # Color boxes
    colors = ['#3498db', '#e74c3c', '#95a5a6']
    for patch, color in zip(bp['boxes'], colors[:len(algos)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Score Distribution (Final Evaluation)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(FIGS_DIR, "final_score_distributions.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved score distributions to {output_path}")
    plt.close()

if __name__ == "__main__":
    create_all_visualizations()
