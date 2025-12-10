"""
Comprehensive evaluation and analysis of training results
Generates learning curves, statistics, and comparison plots
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load all experimental data"""
    # Random baseline
    random_df = pd.read_csv("runs/metrics_dino.csv")
    random_df = random_df[random_df['algo'] == 'random']
    
    # DQN and PPO checkpoints
    checkpoints_df = pd.read_csv("runs/metrics_dino_checkpoints.csv")
    
    return random_df, checkpoints_df

def compute_statistics(df, algo_name):
    """Compute comprehensive statistics for an algorithm"""
    scores = df['score'].values
    steps = df['steps'].values
    
    stats_dict = {
        'Algorithm': algo_name,
        'Episodes': len(scores),
        'Mean Score': np.mean(scores),
        'Std Score': np.std(scores),
        'Median Score': np.median(scores),
        'Min Score': np.min(scores),
        'Max Score': np.max(scores),
        'Mean Steps': np.mean(steps),
        'Std Steps': np.std(steps),
        'Median Steps': np.median(steps),
        'Max Steps': np.max(steps),
    }
    return stats_dict

def plot_learning_curves(checkpoints_df, output_dir="figs"):
    """Plot learning curves for DQN and PPO"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Score learning curve
    for algo in ['dqn', 'ppo']:
        algo_df = checkpoints_df[checkpoints_df['algo'] == algo]
        grouped = algo_df.groupby('checkpoint')['score'].agg(['mean', 'std', 'count'])
        grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
        
        checkpoints = grouped.index / 1000  # Convert to thousands
        means = grouped['mean'].values
        se = grouped['se'].values
        
        axes[0].plot(checkpoints, means, marker='o', label=algo.upper(), linewidth=2)
        axes[0].fill_between(checkpoints, means - se, means + se, alpha=0.2)
    
    axes[0].set_xlabel('Training Steps (×1000)', fontsize=12)
    axes[0].set_ylabel('Average Score', fontsize=12)
    axes[0].set_title('Learning Curve: Score vs Training Steps', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Steps survived learning curve
    for algo in ['dqn', 'ppo']:
        algo_df = checkpoints_df[checkpoints_df['algo'] == algo]
        grouped = algo_df.groupby('checkpoint')['steps'].agg(['mean', 'std', 'count'])
        grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
        
        checkpoints = grouped.index / 1000
        means = grouped['mean'].values
        se = grouped['se'].values
        
        axes[1].plot(checkpoints, means, marker='o', label=algo.upper(), linewidth=2)
        axes[1].fill_between(checkpoints, means - se, means + se, alpha=0.2)
    
    axes[1].set_xlabel('Training Steps (×1000)', fontsize=12)
    axes[1].set_ylabel('Average Steps Survived', fontsize=12)
    axes[1].set_title('Learning Curve: Survival vs Training Steps', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_curves.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved learning curves to {output_dir}/learning_curves.png")
    plt.close()

def plot_checkpoint_comparison(checkpoints_df, random_baseline_mean, output_dir="figs"):
    """Compare all algorithms at each checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique checkpoints
    checkpoints = sorted(checkpoints_df['checkpoint'].unique())
    checkpoint_labels = [f"{c//1000}k" for c in checkpoints]
    
    x = np.arange(len(checkpoints))
    width = 0.35
    
    # DQN bars
    dqn_means = []
    dqn_se = []
    for cp in checkpoints:
        cp_data = checkpoints_df[(checkpoints_df['algo'] == 'dqn') & (checkpoints_df['checkpoint'] == cp)]['score']
        dqn_means.append(cp_data.mean())
        dqn_se.append(cp_data.std() / np.sqrt(len(cp_data)))
    
    # PPO bars
    ppo_means = []
    ppo_se = []
    for cp in checkpoints:
        cp_data = checkpoints_df[(checkpoints_df['algo'] == 'ppo') & (checkpoints_df['checkpoint'] == cp)]['score']
        ppo_means.append(cp_data.mean())
        ppo_se.append(cp_data.std() / np.sqrt(len(cp_data)))
    
    ax.bar(x - width/2, dqn_means, width, yerr=dqn_se, label='DQN', alpha=0.8, capsize=5)
    ax.bar(x + width/2, ppo_means, width, yerr=ppo_se, label='PPO', alpha=0.8, capsize=5)
    ax.axhline(y=random_baseline_mean, color='red', linestyle='--', linewidth=2, label='Random Baseline')
    
    ax.set_xlabel('Training Checkpoint', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('Algorithm Comparison Across Training Checkpoints', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoint_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/checkpoint_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved checkpoint comparison to {output_dir}/checkpoint_comparison.png")
    plt.close()

def plot_score_distributions(random_df, checkpoints_df, output_dir="figs"):
    """Plot score distributions for final checkpoints"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get final checkpoint for each algo
    max_checkpoint = checkpoints_df['checkpoint'].max()
    dqn_final = checkpoints_df[(checkpoints_df['algo'] == 'dqn') & (checkpoints_df['checkpoint'] == max_checkpoint)]['score']
    ppo_final = checkpoints_df[(checkpoints_df['algo'] == 'ppo') & (checkpoints_df['checkpoint'] == max_checkpoint)]['score']
    random_scores = random_df['score']
    
    data_to_plot = [random_scores, dqn_final, ppo_final]
    labels = ['Random', 'DQN (200k)', 'PPO (200k)']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
    
    # Color the boxes
    colors = ['lightgray', 'skyblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Score Distribution Comparison (Final Checkpoint)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_distributions.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved score distributions to {output_dir}/score_distributions.png")
    plt.close()

def statistical_tests(random_df, checkpoints_df):
    """Perform statistical significance tests"""
    print("\n" + "="*70)
    print("  STATISTICAL SIGNIFICANCE TESTS (Mann-Whitney U)")
    print("="*70)
    
    max_checkpoint = checkpoints_df['checkpoint'].max()
    dqn_final = checkpoints_df[(checkpoints_df['algo'] == 'dqn') & (checkpoints_df['checkpoint'] == max_checkpoint)]['score']
    ppo_final = checkpoints_df[(checkpoints_df['algo'] == 'ppo') & (checkpoints_df['checkpoint'] == max_checkpoint)]['score']
    random_scores = random_df['score']
    
    # DQN vs Random
    stat, p = stats.mannwhitneyu(dqn_final, random_scores, alternative='greater')
    print(f"\nDQN vs Random:")
    print(f"  U-statistic: {stat:.2f}")
    print(f"  p-value: {p:.6f}")
    print(f"  Significant: {'Yes' if p < 0.05 else 'No'} (α=0.05)")
    
    # PPO vs Random
    stat, p = stats.mannwhitneyu(ppo_final, random_scores, alternative='greater')
    print(f"\nPPO vs Random:")
    print(f"  U-statistic: {stat:.2f}")
    print(f"  p-value: {p:.6f}")
    print(f"  Significant: {'Yes' if p < 0.05 else 'No'} (α=0.05)")
    
    # DQN vs PPO
    stat, p = stats.mannwhitneyu(dqn_final, ppo_final, alternative='two-sided')
    print(f"\nDQN vs PPO:")
    print(f"  U-statistic: {stat:.2f}")
    print(f"  p-value: {p:.6f}")
    print(f"  Significant: {'Yes' if p < 0.05 else 'No'} (α=0.05)")
    if dqn_final.mean() > ppo_final.mean():
        print(f"  Winner: DQN (mean diff: {dqn_final.mean() - ppo_final.mean():.2f})")
    else:
        print(f"  Winner: PPO (mean diff: {ppo_final.mean() - dqn_final.mean():.2f})")

def generate_summary_table(random_df, checkpoints_df):
    """Generate comprehensive summary statistics table"""
    print("\n" + "="*70)
    print("  COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*70)
    
    # Random baseline
    random_stats = compute_statistics(random_df, "Random")
    
    # DQN at each checkpoint
    dqn_checkpoints = []
    for cp in sorted(checkpoints_df[checkpoints_df['algo'] == 'dqn']['checkpoint'].unique()):
        cp_df = checkpoints_df[(checkpoints_df['algo'] == 'dqn') & (checkpoints_df['checkpoint'] == cp)]
        stats_dict = compute_statistics(cp_df, f"DQN-{cp//1000}k")
        dqn_checkpoints.append(stats_dict)
    
    # PPO at each checkpoint
    ppo_checkpoints = []
    for cp in sorted(checkpoints_df[checkpoints_df['algo'] == 'ppo']['checkpoint'].unique()):
        cp_df = checkpoints_df[(checkpoints_df['algo'] == 'ppo') & (checkpoints_df['checkpoint'] == cp)]
        stats_dict = compute_statistics(cp_df, f"PPO-{cp//1000}k")
        ppo_checkpoints.append(stats_dict)
    
    # Create DataFrame
    all_stats = [random_stats] + dqn_checkpoints + ppo_checkpoints
    summary_df = pd.DataFrame(all_stats)
    
    # Print formatted table
    print("\n" + summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv("results/summary_statistics.csv", index=False)
    print(f"\n✓ Saved summary statistics to results/summary_statistics.csv")

def main():
    """Main analysis pipeline"""
    print("\n" + "="*70)
    print("  ANALYZING TRAINING RESULTS")
    print("="*70)
    
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    random_df, checkpoints_df = load_data()
    print(f"✓ Loaded {len(random_df)} random episodes")
    print(f"✓ Loaded {len(checkpoints_df)} checkpoint evaluation episodes")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_learning_curves(checkpoints_df)
    plot_checkpoint_comparison(checkpoints_df, random_df['score'].mean())
    plot_score_distributions(random_df, checkpoints_df)
    
    # Statistical analysis
    statistical_tests(random_df, checkpoints_df)
    
    # Summary table
    generate_summary_table(random_df, checkpoints_df)
    
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - figs/learning_curves.png")
    print("  - figs/checkpoint_comparison.png")
    print("  - figs/score_distributions.png")
    print("  - results/summary_statistics.csv")

if __name__ == "__main__":
    main()
