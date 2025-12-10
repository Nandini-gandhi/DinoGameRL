"""Train and evaluate a DQN agent on the Chrome Dino environment.

This script:
- creates a vectorised Chrome Dino environment using make_dino_env
- trains a DQN agent with predefined hyperparameters
- saves the trained model to the models/ directory
- performs a short evaluation run and records videos to media/
- logs evaluation metrics to runs/metrics_dino.csv for later analysis.
"""

import os, csv, time
import numpy as np
from typing import List

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecVideoRecorder

from dino_local_env import make_dino_env

SEED = 42
LOG = "runs/metrics_dino.csv"
VID = "media"
MODELS = "models"
os.makedirs("runs", exist_ok=True)
os.makedirs(VID, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

def ensure_header():
    need = not os.path.exists(LOG) or os.path.getsize(LOG) == 0
    if need:
        with open(LOG, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","env","algo","seed","episode","steps","survival_time_sec","score"])

def make_vec_env(train_mode=True):
    base_env = lambda: make_dino_env(train_mode=train_mode)
    vec = DummyVecEnv([base_env])
    vec = VecTransposeImage(vec)
    return vec

def train_dqn(total_timesteps=50_000):
    env = make_vec_env(train_mode=True)
    model = DQN(
        "CnnPolicy",
        env,
        seed=SEED,
        verbose=0,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,                 # env steps per gradient step
        target_update_interval=1_000, # hard target update frequency
        exploration_fraction=0.25,    # linear eps decay fraction of training
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )
    print(f"[TRAIN] DQN on Dino for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    path = os.path.join(MODELS, "dqn_dino.zip")
    model.save(path)
    print(f"[TRAIN] Saved model -> {path}")
    env.close()
    return path

def eval_and_record(model_path, n_episodes=10, fps=15, max_steps=2000):
    # Record every episode
    base = make_vec_env(train_mode=False)
    rec = VecVideoRecorder(
        base,
        video_folder=VID,
        name_prefix="dqn_dino_eval",
        record_video_trigger=lambda ep_id: True,
        video_length=max_steps,
    )

    model = DQN.load(model_path)
    scores: List[float] = []
    lengths: List[int] = []

    print(f"[EVAL] Evaluating DQN for {n_episodes} episodes with video...")
    for ep in range(1, n_episodes + 1):
        obs = rec.reset()
        done = [False]
        steps = 0
        total = 0.0
        while not done[0] and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = rec.step(action)
            total += float(rewards[0])
            steps += 1
            done = dones
        scores.append(total)
        lengths.append(steps)
        with open(LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                int(time.time()), "DinoLocal", "dqn", SEED, ep, steps,
                round(steps/float(fps),3), round(total,3)
            ])
        print(f"[EVAL][E{ep:02d}] steps={steps:4d}  score={total:7.2f}")

    rec.close()
    avg = float(np.mean(scores)); best = float(np.max(scores))
    print(f"[EVAL] Avg score over {n_episodes} eps: {avg:.2f} | Best: {best:.2f}")

if __name__ == "__main__":
    ensure_header()
    model_path = train_dqn(total_timesteps=50_000) 
    eval_and_record(model_path, n_episodes=5, fps=15, max_steps=1200)
    print("[DONE] DQN training + eval complete.")
