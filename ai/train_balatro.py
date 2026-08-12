#!/usr/bin/env python3
"""
Balatro RL Training Script

Main training script for teaching AI to play Balatro using Stable-Baselines3.
This script creates the Balatro environment, sets up the RL model, and runs training.

Usage:
    python -m ai.train_balatro                      # interactive
    python -m ai.train_balatro --no-prompt          # unattended / supervised

Requirements:
    - Balatro game running with RLBridge mod
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

# SB3 imports
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# SB3 Contrib for action masking
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Our custom environment
from .environment.balatro_env import BalatroEnv
from .telemetry import EventType, RunSession, TelemetryEmitter, close_emitter, emit, set_emitter

DEFAULT_TIMESTEPS = 250000


def parse_args(argv=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a Balatro RL agent")
    parser.add_argument("--run-name", default="run", help="Label for this run's directory")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                        help=f"Total training timesteps (default: {DEFAULT_TIMESTEPS})")
    parser.add_argument("--device", default=None,
                        help="Torch device: cuda, cpu, or auto (default: cuda when available)")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Skip the interactive prompt; required when supervised")
    parser.add_argument("--runs-dir", default="runs", help="Parent directory for run directories")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh instead of resuming the latest checkpoint")
    parser.add_argument("--save-freq", type=int, default=None,
                        help="Checkpoint interval in timesteps (default: timesteps/20)")
    return parser.parse_args(argv)


def setup_logging(log_path: str = "training.log"):
    """Setup logging for training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def mask_fn(env):
    """Extract action mask from the environment's action_masks() method"""
    return env.action_masks()

def create_environment(monitor_path: str = "training_monitor.csv"):
    """Create and wrap the Balatro environment"""
    # Create base environment
    env = BalatroEnv()

    # Use ActionMasker wrapper
    env = ActionMasker(env, mask_fn)

    # Wrap with Monitor for logging episode stats
    env = Monitor(env, filename=monitor_path)

    return env


def get_device(explicit: Optional[str] = None):
    """
    Resolve the torch device to train on

    Uses CUDA when available. Override with --device or BALATRO_RL_DEVICE=cpu.

    Returns:
        Device string accepted by Stable-Baselines3
    """
    import torch

    device = explicit or os.environ.get("BALATRO_RL_DEVICE")
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class TelemetryCallback(BaseCallback):
    """
    Emit PPO training statistics into the telemetry stream

    Episode statistics are computed from ep_info_buffer so they are always
    current. The train/* scalars are read from the SB3 logger, which at
    rollout end still holds the previous iteration's values -- the per-run
    TensorBoard files remain authoritative for those.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.iteration = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.iteration += 1

        payload = {
            "iteration": self.iteration,
            "timesteps": self.num_timesteps,
        }

        # Fresh episode statistics
        buffer = self.model.ep_info_buffer
        if buffer:
            rewards = [ep["r"] for ep in buffer]
            lengths = [ep["l"] for ep in buffer]
            payload["ep_rew_mean"] = round(sum(rewards) / len(rewards), 4)
            payload["ep_len_mean"] = round(sum(lengths) / len(lengths), 4)
            payload["ep_rew_max"] = round(max(rewards), 4)
            payload["ep_count"] = len(buffer)

        # Best-effort optimiser scalars from the SB3 logger
        scalars = getattr(self.model.logger, "name_to_value", {})
        for key, value in scalars.items():
            if key.startswith(("train/", "time/")):
                try:
                    payload[key] = round(float(value), 6)
                except (TypeError, ValueError):
                    pass

        emit(EventType.ROLLOUT, **payload)


def create_model(env, tb_dir: str = "./tensorboard_logs/", device: Optional[str] = None,
                 model_path=None):
    """
    Create MaskablePPO model for training

    Args:
        env: Balatro environment
        tb_dir: TensorBoard log directory for this run
        device: Torch device override
        model_path: Path to load existing model (optional)

    Returns:
        MaskablePPO model ready for training
    """
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log=tb_dir,
        device=get_device(device)
    )

    # Load existing model if path provided
    if model_path and Path(model_path).exists():
        model.load(model_path)
        print(f"Loaded existing model from {model_path}")

    return model


def create_callbacks(save_freq=1000, checkpoints_dir="./models/"):
    """Create training callbacks for saving and evaluation"""
    callbacks = []

    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoints_dir,
        name_prefix="balatro_model"
    )
    callbacks.append(checkpoint_callback)

    # Telemetry callback - stream training stats to the monitor
    callbacks.append(TelemetryCallback())

    return callbacks


def find_latest_checkpoint(runs_dir: str = "runs") -> Optional[Path]:
    """
    Most recently modified checkpoint across all runs

    Also searches the legacy flat models/ directory so older checkpoints
    remain resumable.
    """
    candidates = list(Path(runs_dir).glob("*/checkpoints/balatro_model_*_steps.zip"))
    candidates += list(Path("models").glob("balatro_model_*_steps.zip"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def train_agent(session: RunSession, total_timesteps=DEFAULT_TIMESTEPS,
                save_path=None, resume_from=None, device=None, save_freq=None):
    """
    Main training function

    Args:
        session: Run session owning this run's directory
        total_timesteps: Number of training steps
        save_path: Where to save final model
        resume_from: Path to checkpoint to resume from
        device: Torch device override
        save_freq: Checkpoint interval in timesteps
    """
    logger = logging.getLogger(__name__)
    resolved_device = get_device(device)
    logger.info(f"Starting Balatro RL training with MaskablePPO")
    logger.info(f"Training for {total_timesteps} timesteps")
    logger.info(f"Device: {resolved_device}")
    logger.info(f"Run directory: {session.run_dir}")

    save_path = save_path or os.path.join(session.run_dir, "balatro_trained")
    env = None

    emit(
        EventType.SESSION_START,
        run_id=session.run_id,
        total_timesteps=total_timesteps,
        device=resolved_device,
        resume_from=str(resume_from) if resume_from else None,
    )

    try:
        # Create environment and model
        env = create_environment(monitor_path=session.monitor_path)

        if resume_from and Path(resume_from).exists():
            logger.info(f"Resuming training from: {resume_from}")
            model = MaskablePPO.load(
                resume_from,
                env=env,
                tensorboard_log=session.tb_dir,
                device=resolved_device
            )
        else:
            logger.info("Starting training from scratch")
            model = create_model(env, tb_dir=session.tb_dir, device=device)

        # Create callbacks
        if save_freq is None:
            save_freq = max(1000, total_timesteps // 20)
        callbacks = create_callbacks(save_freq=save_freq,
                                     checkpoints_dir=session.checkpoints_dir)

        # Train the model
        logger.info("Starting training...")
        start_time = time.time()

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Save final model
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")

        emit(EventType.SESSION_END, status="completed",
             wall_time=round(training_time, 2), model_path=str(save_path))
        session.mark_finished("completed")

        # Clean up environment
        cleanup_env(env)

        return model

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        emit(EventType.SESSION_END, status="interrupted")
        session.mark_finished("interrupted")
        cleanup_env(env)
        return None
    except Exception as e:
        logger.error(f"Training failed: {e}")
        emit(EventType.SESSION_END, status="failed", error=str(e))
        session.mark_finished("failed")
        cleanup_env(env)
        raise


def cleanup_env(env):
    """Release the environment's socket if it was created"""
    if env is None:
        return
    if hasattr(env, 'cleanup'):
        env.cleanup()
    elif hasattr(env, 'env') and hasattr(env.env, 'cleanup'):  # Monitor wrapper
        env.env.cleanup()


def test_trained_model(model_path, num_episodes=5):
    """
    Test a trained model

    Args:
        model_path: Path to trained model
        num_episodes: Number of episodes to test
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Testing model from {model_path}")

    # Create environment and load model
    env = create_environment()
    model = MaskablePPO.load(model_path)

    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                break

        episode_rewards.append(total_reward)
        logger.info(f"Episode {episode + 1}: {steps} steps, reward: {total_reward:.2f}")

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    logger.info(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")

    env.cleanup()
    return episode_rewards


if __name__ == "__main__":
    args = parse_args()

    # Create the run directory first so logs and telemetry land inside it
    session = RunSession.create(
        name=args.run_name,
        runs_dir=args.runs_dir,
        config={
            "total_timesteps": args.timesteps,
            "device": get_device(args.device),
            "algo": "MaskablePPO",
            "policy": "MlpPolicy",
            "learning_rate": 1e-4,
            "n_steps": 4096,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "ent_coef": 0.01,
        },
    )

    setup_logging(os.path.join(session.run_dir, "training.log"))
    set_emitter(TelemetryEmitter(session.events_path))

    # Resolve which checkpoint to resume from
    resume_from = args.resume
    if resume_from is None and not args.no_resume:
        latest = find_latest_checkpoint(args.runs_dir)
        if latest:
            resume_from = str(latest)
            print(f"📂 Found checkpoint: {latest}")

    print("\n🎮 Starting Balatro RL Training!")
    print(f"📁 Run: {session.run_dir}")
    print("Setup steps:")
    print("1. ✅ Balatro is running with RLBridge mod")
    print("2. ✅ Balatro is in menu state")

    if not args.no_prompt:
        input("Press Enter to start training then press 'R' in Balatro)...")

    try:
        model = train_agent(
            session=session,
            total_timesteps=args.timesteps,
            resume_from=resume_from,
            device=args.device,
            save_freq=args.save_freq,
        )

        if model:
            print("\n🎉Training completed successfully! Ready for next training session.")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Check the logs for more details.")
    finally:
        close_emitter()
        print("🧹 Training session ended")
