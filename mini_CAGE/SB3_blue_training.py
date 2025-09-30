"""
SB3_training.py — launch N PPO runs in parallel (one process each)

This has been adjusted for running experiments for the Snakeoil project, case study 1.
Initial focus is on training variance, the effect of varying hyperparameters, and the effect of terminating episodes
before convergence is present.

Following this, the evaluation process will include running bline trained agents against meander agents (and vice
versa). This requires reconstructing the meander agent in the MiniCAGE agent format. Potentially also configuring to
change agent order (although this may be covered just using YT).
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime
from multiprocessing import Process, set_start_method
from pathlib import Path
from typing import Optional
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from single_agent_gym_wrapper import MiniCageBlue
from stable_baselines3.common.callbacks import BaseCallback

# ═══════════════════════════════════════════════════════════════════════
# Training Config
# ═══════════════════════════════════════════════════════════════════════
NUM_RUNS: int = 25
TOTAL_TIMESTEPS: int = 2_500_000

USE_WANDB: bool = True           # flip to False to disable W&B logging
USE_TENSORBOARD: bool = True     # if True, each run gets its own TB dir

WANDB_PROJECT: str = "mini-cage-trial" # Adjust
WANDB_ENTITY: str | None = "YTRewards" # Adjust
EXPERIMENT: str = "default"  # Choose from HP_PRESETS below

CHECKPOINTS = [50_000, 100_000, 250_000, 500_000, 750_000, 1_000_000, 2_000_000, 2_500_000]

HP_PRESETS = {
    "default": {
        "LEARNING_RATE": 0.002,
        "GAMMA": 0.99,
        "CLIP_RANGE": 0.2,
    },
    "alt_alpha": {
        "LEARNING_RATE": 0.02,
        "GAMMA": 0.99,
        "CLIP_RANGE": 0.2,
    },
    "alt_eps": {
        "LEARNING_RATE": 0.002,
        "GAMMA": 0.99,
        "CLIP_RANGE": 1,  # essentially no clipping, super weak
    },
    "alt_gamma": {
        "LEARNING_RATE": 0.002,
        "GAMMA": 0.1,
        "CLIP_RANGE": 0.2,
    },
}

# Pick the HPs based on the EXPERIMENT name
if EXPERIMENT not in HP_PRESETS:
    raise ValueError(f"Unknown EXPERIMENT '{EXPERIMENT}'. Choose one of {list(HP_PRESETS)}")

LEARNING_RATE = HP_PRESETS[EXPERIMENT]["LEARNING_RATE"]
GAMMA = HP_PRESETS[EXPERIMENT]["GAMMA"]
CLIP_RANGE = HP_PRESETS[EXPERIMENT]["CLIP_RANGE"]

GROUP_NAME: str = f"SB3_PPO_{EXPERIMENT}_{TOTAL_TIMESTEPS}"

BASE_DIR: Path = Path("snakeoil_ppo_models")
EXPERIMENT_DIR: Path = BASE_DIR / f"SB3_PPO_{EXPERIMENT}"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

def make_env(seed: Optional[int] = None):
    """Factory that returns a **monitored** MiniCageBlue env."""
    env = MiniCageBlue(red_policy="bline", max_steps=100, remove_bugs=True)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
    env = Monitor(env)
    return env

class CheckpointSaver(BaseCallback):
    """
    Save the model at specific global timesteps.
    Creates a per-agent directory and writes separate .zip files per milestone.
    """
    def __init__(self, checkpoints: list[int], save_root: Path, run_name: str, agent_idx: int, verbose: int = 0):
        super().__init__(verbose)
        self.checkpoints = sorted(set(int(s) for s in checkpoints))
        self.save_root = Path(save_root)
        self.run_name = run_name
        self.agent_idx = agent_idx
        self._already_saved = set()

        # Per-agent dir: snakeoil_ppo_models/SB3_PPO_<EXPERIMENT>/agent_XX
        self.agent_dir = self.save_root / f"agent_{agent_idx:02d}"
        self.agent_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        t = self.model.num_timesteps  # global count across learn()
        if t in self.checkpoints and t not in self._already_saved:
            filename = f"{self.run_name}_step_{t:,}.zip".replace(",", "")
            path = self.agent_dir / filename
            if self.verbose:
                print(f"[CheckpointSaver] Saving checkpoint at {t} steps -> {path}")
            self.model.save(path)
            self._already_saved.add(t)
        return True

def train_worker(idx: int):
    """Launch a single PPO run (executed inside its own process)."""
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_mini_cage_bline_{time_tag}_{idx}"

    # Make environment
    env = DummyVecEnv([lambda: make_env(seed=idx)])
    env = VecMonitor(env)

    # TensorBoard dir
    tb_dir: Optional[str]
    if USE_TENSORBOARD:
        tb_dir = f"./ppo_mini_cage_tensorboard/run_{idx}"
        os.makedirs(tb_dir, exist_ok=True)
    else:
        tb_dir = tempfile.mkdtemp() if USE_WANDB else None

    # Initialise model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=tb_dir,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        clip_range=CLIP_RANGE,
        n_epochs=6,
        seed=idx,  # unique seed per run
        device="auto",
    )

    # Build callbacks
    callback_list = []
    checkpoint_cb = CheckpointSaver(
        checkpoints=CHECKPOINTS,
        save_root=EXPERIMENT_DIR,
        run_name=run_name,
        agent_idx=idx,
        verbose=1,
    )
    callback_list.append(checkpoint_cb)

    if USE_WANDB:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            group=GROUP_NAME,
            monitor_gym=True,
            save_code=True,
            sync_tensorboard=True,
            config=dict(
                algorithm="PPO",
                total_timesteps=TOTAL_TIMESTEPS,
                env="MiniCageBlue",
                seed=idx,
                learning_rate=LEARNING_RATE,
                gamma=GAMMA,
                clip_range=CLIP_RANGE,
                n_epochs=6,
            ),
        )

        callback_list.append(
            WandbCallback(
                gradient_save_freq=1_000,
                model_save_path=str(EXPERIMENT_DIR / f"agent_{idx:02d}"),
                verbose=0,
            )
        )

    # Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        log_interval=10,
    )

    # Final save (per-agent dir as an end-of-training snapshot)
    final_ckpt_path = EXPERIMENT_DIR / f"agent_{idx:02d}" / f"{run_name}_final.zip"
    model.save(final_ckpt_path)

    if USE_WANDB:
        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(str(final_ckpt_path))
        run.log_artifact(artifact)
        run.finish()

    print(f"Run {idx}: finished. Model saved to {final_ckpt_path}")

if __name__ == "__main__":
    try:
        set_start_method("spawn")  # does nothing if already set
    except RuntimeError:
        pass

    START_IDX = 2
    processes: list[Process] = []
    for idx in range(START_IDX, START_IDX + NUM_RUNS):
        p = Process(target=train_worker, args=(idx,), daemon=False)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n All runs finished!")