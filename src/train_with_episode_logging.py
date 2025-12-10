"""
Enhanced training with episode-level logging for smoother learning curves
"""
import os
import csv
import time
import numpy as np
from typing import Dict, List
from collections import deque

import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from dino_local_env import make_dino_env

SEED = 42
EPISODE_LOG = "runs/training_episodes.csv"
CHECKPOINT_LOG = "runs/metrics_dino_checkpoints.csv"
MODELS = "models/checkpoints"
os.makedirs("runs", exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

def ensure_episode_header():
    need = not os.path.exists(EPISODE_LOG) or os.path.getsize(EPISODE_LOG) == 0
    if need:
        with open(EPISODE_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "algo", "episode", "timesteps", "reward", 
                "length", "survival_time_sec", "best_reward_so_far"
            ])

def ensure_checkpoint_header():
    need = not os.path.exists(CHECKPOINT_LOG) or os.path.getsize(CHECKPOINT_LOG) == 0
    if need:
        with open(CHECKPOINT_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "env", "algo", "checkpoint", "seed", 
                "episode", "steps", "survival_time_sec", "score"
            ])

class EpisodeLoggingCallback(BaseCallback):
    """
    Callback that logs every training episode for detailed learning curves
    """
    def __init__(self, algo_name: str, fps: int = 15, verbose: int = 0):
        super().__init__(verbose)
        self.algo_name = algo_name
        self.fps = fps
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')
        
    def _on_step(self) -> bool:
        # Check if episode just finished
        if self.locals.get('dones') is not None:
            dones = self.locals['dones']
            if any(dones):
                # Get episode info from info dict
                infos = self.locals.get('infos', [])
                for info in infos:
                    if 'episode' in info:
                        self.episode_count += 1
                        ep_reward = info['episode']['r']
                        ep_length = info['episode']['l']
                        survival_sec = round(ep_length / self.fps, 2)
                        
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        
                        # Track best reward
                        if ep_reward > self.best_reward:
                            self.best_reward = ep_reward
                        
                        # Log to CSV
                        ensure_episode_header()
                        with open(EPISODE_LOG, "a", newline="") as f:
                            csv.writer(f).writerow([
                                int(time.time()),
                                self.algo_name,
                                self.episode_count,
                                self.num_timesteps,
                                round(ep_reward, 2),
                                ep_length,
                                survival_sec,
                                round(self.best_reward, 2)
                            ])
                        
                        # Print every 100 episodes
                        if self.episode_count % 100 == 0:
                            recent_rewards = self.episode_rewards[-100:]
                            avg_reward = np.mean(recent_rewards)
                            avg_length = np.mean(self.episode_lengths[-100:])
                            print(f"  Episode {self.episode_count} | Steps: {self.num_timesteps} | "
                                  f"Avg Reward (last 100): {avg_reward:.2f} | "
                                  f"Avg Length: {avg_length:.1f} | Best: {self.best_reward:.2f}")
        
        return True

class CheckpointEvalCallback(BaseCallback):
    """Callback for evaluating at checkpoints"""
    def __init__(self, eval_freq: int, algo_name: str, n_eval_episodes: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.algo_name = algo_name
        self.n_eval_episodes = n_eval_episodes
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            checkpoint = self.n_calls
            print(f"\n{'='*70}")
            print(f"  CHECKPOINT EVALUATION @ {checkpoint} steps")
            print(f"{'='*70}")
            
            # Create eval environment
            base_env = lambda: make_dino_env(train_mode=False)
            eval_env = DummyVecEnv([base_env])
            eval_env = VecTransposeImage(eval_env)
            
            scores = []
            steps_list = []
            survival_times = []
            
            for ep in range(self.n_eval_episodes):
                obs = eval_env.reset()
                done = [False]
                total = 0.0
                steps = 0
                while not done[0] and steps < 2000:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done_arr, info = eval_env.step(action)
                    total += float(reward[0])
                    steps += 1
                    done = done_arr
                
                scores.append(total)
                steps_list.append(steps)
                survival_times.append(steps / 15.0)
                
                # Log to checkpoint CSV
                ensure_checkpoint_header()
                with open(CHECKPOINT_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([
                        int(time.time()), "DinoLocal", self.algo_name.lower(), 
                        checkpoint, SEED, ep+1, steps, 
                        round(steps/15.0, 2), round(total, 2)
                    ])
            
            avg_score = np.mean(scores)
            avg_survival = np.mean(survival_times)
            print(f"\nResults: Avg Score = {avg_score:.2f}, Avg Survival = {avg_survival:.2f}s")
            print(f"{'='*70}\n")
            
            eval_env.close()
        return True

def make_monitored_env(train_mode: bool = True):
    def make_env():
        env = make_dino_env(train_mode=train_mode)
        # Monitor wrapper automatically tracks episode rewards/lengths
        env = Monitor(env)
        return env
    
    vec = DummyVecEnv([make_env])
    vec = VecTransposeImage(vec)
    return vec

def train_dqn_with_logging(
    total_timesteps: int = 200_000,
    checkpoint_freq: int = 25_000,
    resume_from: str = None
):
    """Train DQN with episode-level logging"""
    print("\n" + "="*70)
    print("  TRAINING DQN WITH EPISODE LOGGING")
    print("="*70)
    
    env = make_monitored_env(train_mode=True)
    
    # Create callbacks
    episode_logger = EpisodeLoggingCallback(algo_name="DQN", fps=15, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=MODELS,
        name_prefix="dqn_dino",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )
    eval_callback = CheckpointEvalCallback(
        eval_freq=checkpoint_freq,
        algo_name="DQN",
        n_eval_episodes=20,
        verbose=1
    )
    
    # Create or load model
    if resume_from and os.path.exists(resume_from):
        print(f"[RESUME] Loading from {resume_from}")
        model = DQN.load(resume_from, env=env)
    else:
        print(f"[NEW] Creating DQN model with original hyperparameters")
        model = DQN(
            "CnnPolicy", env, seed=SEED, verbose=0,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=2_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.25,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
        )
    
    print(f"[TRAIN] Training for {total_timesteps} steps...")
    print(f"[TRAIN] Episode data will be logged to {EPISODE_LOG}")
    print(f"[TRAIN] Checkpoint evals will be logged to {CHECKPOINT_LOG}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[episode_logger, checkpoint_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=(resume_from is None)
    )
    
    # Save final model
    final_path = os.path.join(MODELS, "dqn_dino_final.zip")
    model.save(final_path)
    print(f"\n✓ Final model saved to {final_path}")
    
    env.close()
    return model

def train_ppo_with_logging(
    total_timesteps: int = 200_000,
    checkpoint_freq: int = 25_000,
    resume_from: str = None
):
    """Train PPO with episode-level logging"""
    print("\n" + "="*70)
    print("  TRAINING PPO WITH EPISODE LOGGING")
    print("="*70)
    
    env = make_monitored_env(train_mode=True)
    
    # Create callbacks
    episode_logger = EpisodeLoggingCallback(algo_name="PPO", fps=15, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=MODELS,
        name_prefix="ppo_dino",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_callback = CheckpointEvalCallback(
        eval_freq=checkpoint_freq,
        algo_name="PPO",
        n_eval_episodes=20,
        verbose=1
    )
    
    # Create or load model
    if resume_from and os.path.exists(resume_from):
        print(f"[RESUME] Loading from {resume_from}")
        model = PPO.load(resume_from, env=env)
    else:
        print(f"[NEW] Creating PPO model with original hyperparameters")
        model = PPO(
            "CnnPolicy", env, seed=SEED, verbose=0,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
    
    print(f"[TRAIN] Training for {total_timesteps} steps...")
    print(f"[TRAIN] Episode data will be logged to {EPISODE_LOG}")
    print(f"[TRAIN] Checkpoint evals will be logged to {CHECKPOINT_LOG}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[episode_logger, checkpoint_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=(resume_from is None)
    )
    
    # Save final model
    final_path = os.path.join(MODELS, "ppo_dino_final.zip")
    model.save(final_path)
    print(f"\n✓ Final model saved to {final_path}")
    
    env.close()
    return model

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    algo = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    resume = sys.argv[3] if len(sys.argv) > 3 else None
    
    ensure_episode_header()
    ensure_checkpoint_header()
    
    if algo.lower() == "dqn":
        train_dqn_with_logging(total_timesteps=timesteps, resume_from=resume)
    elif algo.lower() == "ppo":
        train_ppo_with_logging(total_timesteps=timesteps, resume_from=resume)
    else:
        print(f"Unknown algorithm: {algo}")
        print("Usage: python train_with_episode_logging.py [dqn|ppo] [timesteps] [resume_path]")
        sys.exit(1)
    
    print("\n✓ Training complete!")
    print(f"  Episode logs: {EPISODE_LOG}")
    print(f"  Checkpoint evals: {CHECKPOINT_LOG}")
