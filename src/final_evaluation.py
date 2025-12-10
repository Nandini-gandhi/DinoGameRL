"""
Comprehensive final evaluation for trained models
- Multiple episodes on final models
- Survival time tracking
- Best score tracking
- Statistical summary
"""
import os
import csv
import time
import numpy as np
from typing import Dict, List, Tuple
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from dino_local_env import make_dino_env

SEED = 42
LOG = "runs/metrics_final_evaluation.csv"
SUMMARY = "results/final_evaluation_summary.csv"
os.makedirs("runs", exist_ok=True)
os.makedirs("results", exist_ok=True)

def ensure_header():
    need = not os.path.exists(LOG) or os.path.getsize(LOG) == 0
    if need:
        with open(LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "env", "algo", "seed", "episode", 
                "steps", "survival_time_sec", "score"
            ])

def make_vec_env(train_mode=False):
    base_env = lambda: make_dino_env(train_mode=train_mode)
    vec = DummyVecEnv([base_env])
    vec = VecTransposeImage(vec)
    return vec

def evaluate_model(model, env, n_episodes: int, algo_name: str, fps: int = 15, max_steps: int = 2000) -> Dict:
    """
    Evaluate a trained model comprehensively
    
    Returns:
        Dict with episodes, avg_score, std_score, best_score, avg_survival_sec, std_survival_sec
    """
    ensure_header()
    
    scores = []
    survival_times = []
    steps_list = []
    
    print(f"[EVAL] Evaluating {algo_name} on {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = [False]
        total_reward = 0.0
        steps = 0
        
        while not done[0] and steps < max_steps:
            # Use deterministic policy for evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = env.step(action)
            total_reward += float(reward[0])
            steps += 1
            done = done_arr
        
        survival_sec = round(steps / float(fps), 2)
        scores.append(total_reward)
        survival_times.append(survival_sec)
        steps_list.append(steps)
        
        # Log individual episode
        with open(LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                int(time.time()), "DinoLocal", algo_name, SEED, ep+1,
                steps, survival_sec, round(total_reward, 2)
            ])
        
        if (ep + 1) % 10 == 0:
            print(f"  [{ep+1}/{n_episodes}] Avg score so far: {np.mean(scores):.2f}")
    
    # Calculate statistics
    results = {
        "algorithm": algo_name,
        "episodes": n_episodes,
        "avg_score": round(np.mean(scores), 2),
        "std_score": round(np.std(scores), 2),
        "median_score": round(np.median(scores), 2),
        "best_score": round(np.max(scores), 2),
        "worst_score": round(np.min(scores), 2),
        "avg_survival_sec": round(np.mean(survival_times), 2),
        "std_survival_sec": round(np.std(survival_times), 2),
        "median_survival_sec": round(np.median(survival_times), 2),
        "max_survival_sec": round(np.max(survival_times), 2),
        "avg_steps": round(np.mean(steps_list), 2),
        "success_rate": round(100 * np.sum(np.array(scores) > 0) / len(scores), 1),  # % episodes with positive score
    }
    
    return results

def evaluate_random_baseline(n_episodes: int = 50, fps: int = 15, max_steps: int = 2000) -> Dict:
    """Evaluate random policy as baseline"""
    ensure_header()
    
    env = make_dino_env(train_mode=False)
    scores = []
    survival_times = []
    steps_list = []
    
    print(f"[EVAL] Evaluating Random baseline on {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=SEED + ep)
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
        
        survival_sec = round(steps / float(fps), 2)
        scores.append(total_reward)
        survival_times.append(survival_sec)
        steps_list.append(steps)
        
        # Log individual episode
        with open(LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                int(time.time()), "DinoLocal", "random", SEED, ep+1,
                steps, survival_sec, round(total_reward, 2)
            ])
    
    env.close()
    
    results = {
        "algorithm": "Random",
        "episodes": n_episodes,
        "avg_score": round(np.mean(scores), 2),
        "std_score": round(np.std(scores), 2),
        "median_score": round(np.median(scores), 2),
        "best_score": round(np.max(scores), 2),
        "worst_score": round(np.min(scores), 2),
        "avg_survival_sec": round(np.mean(survival_times), 2),
        "std_survival_sec": round(np.std(survival_times), 2),
        "median_survival_sec": round(np.median(survival_times), 2),
        "max_survival_sec": round(np.max(survival_times), 2),
        "avg_steps": round(np.mean(steps_list), 2),
        "success_rate": round(100 * np.sum(np.array(scores) > 0) / len(scores), 1),
    }
    
    return results

def save_summary_table(results_list: List[Dict]):
    print("\n" + "="*90)
    print("  FINAL EVALUATION SUMMARY")
    print("="*90)
    
    # Print table header
    header = f"{'Algorithm':<12} {'Episodes':>10} {'Avg Survival (s)':>18} {'Avg Score':>12} {'Best Score':>12}"
    print(header)
    print("-" * 90)
    
    # Print rows
    for r in results_list:
        row = f"{r['algorithm']:<12} {r['episodes']:>10} {r['avg_survival_sec']:>18.1f} {r['avg_score']:>12.1f} {r['best_score']:>12.1f}"
        print(row)
    
    print("-" * 90)
    
    # Save to CSV for easy import
    with open(SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
        writer.writeheader()
        writer.writerows(results_list)
    
    print(f"\n✓ Summary saved to {SUMMARY}")
    
    # Print detailed statistics
    print("\n" + "="*90)
    print("  DETAILED STATISTICS")
    print("="*90)
    for r in results_list:
        print(f"\n{r['algorithm']}:")
        print(f"  Score:       {r['avg_score']:.2f} ± {r['std_score']:.2f}  (median: {r['median_score']:.2f}, best: {r['best_score']:.2f})")
        print(f"  Survival:    {r['avg_survival_sec']:.2f}s ± {r['std_survival_sec']:.2f}s  (median: {r['median_survival_sec']:.2f}s, max: {r['max_survival_sec']:.2f}s)")
        print(f"  Steps:       {r['avg_steps']:.1f} average")
        print(f"  Success Rate: {r['success_rate']:.1f}% (episodes with score > 0)")

def run_final_evaluation(
    dqn_model_path: str = "models/checkpoints/dqn_dino_final.zip",
    ppo_model_path: str = "models/checkpoints/ppo_dino_final.zip",
    n_episodes: int = 50,
    include_random: bool = True
):
    """
    Run comprehensive final evaluation on all trained models
    
    Args:
        dqn_model_path: Path to trained DQN model
        ppo_model_path: Path to trained PPO model
        n_episodes: Number of episodes to evaluate per algorithm
        include_random: Whether to include random baseline
    """
    print("="*90)
    print("  FINAL MODEL EVALUATION")
    print("="*90)
    print(f"\nConfiguration:")
    print(f"  - Episodes per algorithm: {n_episodes}")
    print(f"  - Deterministic policy: Yes")
    print(f"  - Max steps per episode: 2000")
    print(f"  - FPS for time calculation: 15")
    print()
    
    results = []
    
    # Evaluate Random baseline
    if include_random:
        print("\n[1/3] Random Baseline")
        print("-" * 90)
        random_results = evaluate_random_baseline(n_episodes=n_episodes)
        results.append(random_results)
        print(f"✓ Random: {random_results['avg_score']:.2f} avg score, {random_results['avg_survival_sec']:.2f}s survival")
    
    # Evaluate DQN
    if os.path.exists(dqn_model_path):
        print("\n[2/3] DQN Model")
        print("-" * 90)
        env = make_vec_env(train_mode=False)
        dqn_model = DQN.load(dqn_model_path, env=env)
        dqn_results = evaluate_model(dqn_model, env, n_episodes, "DQN")
        results.append(dqn_results)
        env.close()
        print(f"✓ DQN: {dqn_results['avg_score']:.2f} avg score, {dqn_results['avg_survival_sec']:.2f}s survival")
    else:
        print(f"\n[2/3] DQN Model NOT FOUND at {dqn_model_path}")
    
    # Evaluate PPO
    if os.path.exists(ppo_model_path):
        print("\n[3/3] PPO Model")
        print("-" * 90)
        env = make_vec_env(train_mode=False)
        ppo_model = PPO.load(ppo_model_path, env=env)
        ppo_results = evaluate_model(ppo_model, env, n_episodes, "PPO")
        results.append(ppo_results)
        env.close()
        print(f"✓ PPO: {ppo_results['avg_score']:.2f} avg score, {ppo_results['avg_survival_sec']:.2f}s survival")
    else:
        print(f"\n[3/3] PPO Model NOT FOUND at {ppo_model_path}")
    
    # Save and display summary
    if results:
        save_summary_table(results)
        print(f"\n✓ Individual episode logs saved to {LOG}")
        print(f"✓ Summary table saved to {SUMMARY}")
    
    print("\n" + "="*90)
    print("  EVALUATION COMPLETE")
    print("="*90)
    
    return results

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    n_episodes = 50  # Default
    if len(sys.argv) > 1:
        n_episodes = int(sys.argv[1])
    
    print(f"\nRunning final evaluation with {n_episodes} episodes per algorithm...\n")
    
    run_final_evaluation(
        n_episodes=n_episodes,
        include_random=True
    )
