# Deep Reinforcement Learning for Chrome Dino

This repository trains and evaluates deep RL agents on a custom Chrome Dino environment.
It compares **DQN**, **PPO**, and a **random policy** using a consistent logging and evaluation pipeline.

---

## 1. Setup

1. **Create a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

3. **Runtime directories**

   The scripts automatically create them when needed:

   * `runs/` – CSV logs for training, checkpoints, and final evaluation.
   * `results/` – Summary CSVs.
   * `models/` – Saved models and checkpoints.
   * `figs/` – Plots and figures.
   * `media/` – Recorded gameplay / evaluation videos.

---

## 2. How to Run the Code

All commands below assume you are in the **repository root**.

### 2.1 Train baseline agents (quick runs)

These scripts train simple DQN and PPO agents and a random baseline with minimal logging.

```bash
# DQN baseline
python src/train_dqn_dino.py

# PPO baseline
python src/train_ppo_dino.py

# Random policy baseline
python src/baseline_random_dino.py
```

* They use the custom environment from `src/dino_local_env.py`.
* They log summary metrics to `runs/metrics_dino.csv`.
* They save videos to `media/` and models to `models/`.

### 2.2 Main experiments with episode-level logging

Use `train_with_episode_logging.py` for the full experiments described in the report.
This script logs **every episode** and evaluates checkpoints during training.

```bash
# Train DQN with episode-level logging
python src/train_with_episode_logging.py dqn 1000000

# Train PPO with episode-level logging
python src/train_with_episode_logging.py ppo 1000000

# (Optional) Resume from a saved model
python src/train_with_episode_logging.py dqn 1000000 models/dqn_dino_final.zip
```

This script:

* Writes per-episode logs to `runs/training_episodes.csv`.
* Writes checkpoint evaluation logs to `runs/metrics_dino_checkpoints.csv`.
* Saves checkpoint and final models under `models/`.

### 2.3 Final evaluation of trained models

After training, run a larger evaluation to get more stable performance estimates:

```bash
# Default: 50 evaluation episodes per algorithm
python src/final_evaluation.py

# Or choose a number of episodes explicitly
python src/final_evaluation.py 100
```

This script:

* Loads the final models from `models/` (DQN, PPO, and an optional random baseline).
* Runs many evaluation episodes.
* Logs per‑episode metrics to `runs/metrics_final_evaluation.csv`.
* Writes a summary table to `results/final_evaluation_summary.csv`.

### 2.4 Analysis and plots

Once logs and summary CSVs exist, generate the figures for the report:

```bash
# General learning curves, distributions, and summary CSVs
python src/analyze_results.py

# Enhanced plots that match the final report
python src/plot_enhanced_results.py
```

These scripts read from `runs/` and `results/` and save plots into `figs/`.

During the project, I used two additional plotting scripts to generate some of the figures that appear in the written report.
These helper scripts were used  during report preparation to export clean versions of plots. All plots that can be reproduced directly from the experiment logs are generated through the main analysis scripts (`analyze_results.py` and `plot_enhanced_results.py`).
