"""
SB3_training.py — launch N PPO runs in parallel (one process each)

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
NUM_RUNS: int = 1
TOTAL_TIMESTEPS: int = 1_000_000

USE_WANDB: bool = True           # flip to False to disable W&B logging
USE_TENSORBOARD: bool = True     # if True, each run gets its own TB dir

WANDB_PROJECT: str = "mini-cage-trial"
WANDB_ENTITY: str | None = "YTRewards"
GROUP_NAME: str = f"SB3_PPO_{TOTAL_TIMESTEPS}"

# These hyper-parameters are taken from the CC2 cardiff solution
LEARNING_RATE: float = 0.002
GAMMA: float = 0.99
CLIP_RANGE: float = 0.2
N_EPOCHS: int = 6

SAVE_DIR: Path = Path("ppo_models") / GROUP_NAME

SAVE_DIR.mkdir(parents=True, exist_ok=True)

def make_env(seed: Optional[int] = None):
    """Factory that returns a **monitored** MiniCageBlue env."""
    env = MiniCageBlue(red_policy="bline", max_steps=100, remove_bugs=True)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)

    env = Monitor(env)
    return env

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
        # create a temporary directory so SB3 still instantiates the writer
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
        n_epochs=N_EPOCHS,
        seed=idx,  # unique seed per run
        device="auto",
    )

    # Build callbacks
    callback_list = []

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
            sync_tensorboard=True,  # TB writer exists, so sync it
            config=dict(
                algorithm="PPO",
                total_timesteps=TOTAL_TIMESTEPS,
                env="MiniCageBlue",
                seed=idx,
                learning_rate=LEARNING_RATE,
                gamma=GAMMA,
                clip_range=CLIP_RANGE,
                n_epochs=N_EPOCHS,
            ),
        )

        callback_list.append(
            WandbCallback(
                gradient_save_freq=1_000,
                model_save_path=str(SAVE_DIR),
                verbose=0,
            )

        )

    # Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list or None,
        log_interval=10,
    )

    # Save checkpoint
    ckpt_path = SAVE_DIR / f"{run_name}.zip"
    model.save(ckpt_path)

    if USE_WANDB:
        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(str(ckpt_path))
        run.log_artifact(artifact)
        run.finish()

    print(f"Run {idx}: finished. Model saved to {ckpt_path}")


if __name__ == "__main__":
    # sleep
    # time.sleep(5400)

    try:
        set_start_method("spawn")  # does nothing if already set
    except RuntimeError:
        pass


    START_IDX = 21
    processes: list[Process] = []
    for idx in range(START_IDX, START_IDX + NUM_RUNS):
        p = Process(target=train_worker, args=(idx,), daemon=False)
        p.start()
        processes.append(p)

    # Wait for all workers to complete
    for p in processes:
        p.join()

    print("\n All runs finished!")