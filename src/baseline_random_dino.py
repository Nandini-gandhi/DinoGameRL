"""Run a random-action baseline on the Chrome Dino environment.

This script:
- creates a Chrome Dino environment in evaluation mode
- runs several episodes choosing actions uniformly at random
- records short gameplay videos to media/
- logs per-episode scores to runs/metrics_dino.csv

It provides a simple performance baseline to compare against the trained agents.
"""
import os, csv, time, random
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from dino_local_env import make_dino_env

SEED = 42
LOG = "runs/metrics_dino.csv"; VID = "media"
os.makedirs("runs", exist_ok=True); os.makedirs(VID, exist_ok=True)

def ensure_header():
    need = not os.path.exists(LOG) or os.path.getsize(LOG) == 0
    if need:
        with open(LOG, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","env","algo","seed","episode","steps","survival_time_sec","score"])

def run_random(episodes=10, max_steps=2000, fps=15):
    ensure_header()
    env = make_dino_env(train_mode=False)
    env = RecordVideo(env, video_folder=VID, name_prefix="random_dino", episode_trigger=lambda e: True)
    print(f"[INFO] Random baseline on Dino | episodes={episodes}")
    for ep in range(1, episodes+1):
        obs, info = env.reset(seed=SEED+ep)
        done = False; steps = 0; score = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            score += float(reward); steps += 1
            done = terminated or truncated or steps >= max_steps
        with open(LOG, "a", newline="") as f:
            csv.writer(f).writerow([int(time.time()), "DinoLocal", "random", SEED, ep, steps, round(steps/float(fps),3), round(score,3)])
        print(f"[E{ep:02d}] steps={steps:4d}  score={score:6.2f}")
    env.close()
    print(f"[INFO] Logged -> {LOG} | Videos -> {VID}/")

if __name__ == "__main__":
    run_random()
