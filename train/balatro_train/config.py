"""Training configuration: dataclasses + YAML loading.

YAML files mirror the nested dataclass structure (see ``configs/debug.yaml``
and ``configs/default.yaml``).  Unknown keys are rejected so typos fail loudly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    name: str = "mock"            # env_api.make_vec_env registry key
    num_envs: int = 16
    seed: int = 0                 # base seed; env i resets from seed + i
    kwargs: dict = field(default_factory=dict)  # extra env ctor kwargs
    # (e.g. reward_mode: bandit for the mock)


@dataclass
class PolicyConfig:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    ffn_mult: int = 4
    global_hidden: int = 256      # hidden width of the global-scalars MLP
    value_hidden: int = 256       # hidden width of the value head MLP


@dataclass
class PPOConfig:
    total_timesteps: int = 1_000_000_000
    rollout_len: int = 128        # steps per env per iteration
    minibatch_size: int = 32_768  # transitions per gradient step
    update_epochs: int = 4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3.0e-4
    anneal_lr: bool = True        # linear decay of lr to 0 over total_timesteps
    max_grad_norm: float = 0.5
    norm_adv: bool = True         # per-batch advantage normalization
    norm_ret: bool = True         # running return normalization (reward scaling)


@dataclass
class CurriculumConfig:
    """Ante-limit curriculum (see env_api "Ante-limit curriculum").

    The trainer starts the envs at ``win_ante = start_ante`` and every
    ``eval_every_steps`` global steps runs an argmax evaluation on a held-out
    seed block (``promotion_eval_episodes`` episodes at the CURRENT win ante).
    When the win rate reaches ``promote_winrate`` the win ante is promoted to
    the next ``ladder`` entry (env-side it applies at each env's next
    auto-reset).  Independently, every ``milestone_every_steps`` a full-game
    (win_ante = 8) argmax eval is logged for M-milestone tracking.
    """

    enabled: bool = False
    start_ante: int = 1
    ladder: list = field(default_factory=lambda: [1, 2, 3, 5, 8])
    promote_winrate: float = 0.7
    eval_every_steps: int = 2_000_000
    promotion_eval_episodes: int = 200
    eval_num_envs: int = 100          # vec width of the eval env
    eval_seed_base: int = 1_000_000   # held-out block (training uses seed+[0,N))
    milestone_every_steps: int = 25_000_000
    milestone_episodes: int = 200


@dataclass
class SnapshotAnnealConfig:
    """Linear anneal of the snapshot fraction over global steps.

    Fraction is ``snapshots.fraction`` until ``start_step``, then linearly
    reaches ``final_fraction`` at ``end_step`` and stays there.  With
    ``end_step <= start_step`` the fraction is constant (no anneal).
    """

    start_step: int = 0
    end_step: int = 0
    final_fraction: float = 0.0


@dataclass
class SnapshotConfig:
    """Snapshot start-state mixing (M3 backward curriculum).

    When enabled, the trainer loads ``pool_path`` into the env
    (``load_snapshot_pool``) and drives ``set_snapshot_fraction`` each
    iteration from the anneal schedule.  Mixing affects TRAINING envs only:
    promotion/milestone evals build separate envs that never load a pool,
    so eval episodes are always fresh starts.  Env semantics
    (reproducibility, win_ante eligibility, ``info["episode"]
    ["from_snapshot"]``): env_api docstring "Snapshot start-state mixing".
    """

    enabled: bool = False
    pool_path: str = ""           # packed pool file (snapshot_pool.py format)
    fraction: float = 0.3         # P(auto-reset starts from the pool)
    anneal: SnapshotAnnealConfig = field(default_factory=SnapshotAnnealConfig)


@dataclass
class ShapingAnnealConfig:
    """Linear anneal of the reward-shaping scale β over global steps.

    β is ``shaping.beta`` until ``start_step``, then linearly reaches
    ``final_beta`` at ``end_step`` and stays there.  With
    ``end_step <= start_step`` β is constant (no anneal).
    """

    start_step: int = 0
    end_step: int = 0
    final_beta: float = 0.1


@dataclass
class ShapingConfig:
    """Reward-shaping scale (M4).  β multiplies the per-hand progress and
    blind-clear shaping terms env-side (``set_shaping_beta``); the win bonus
    is never scaled.  Plan: anneal 1.0 -> 0.1 once ante-8 win-rate > 10%,
    keeping a small variance-reduction floor rather than going to zero.
    Evals are unaffected (win rates count episodes, not rewards)."""

    beta: float = 1.0
    anneal: ShapingAnnealConfig = field(default_factory=ShapingAnnealConfig)


@dataclass
class PerfConfig:
    """Throughput toggles. ALL defaults preserve the pre-perf behavior
    bit-exactly (M3-era code path); see train/README.md "Throughput".

    * ``bf16_update`` — autocast bf16 for the PPO update forward/backward
      (weights/optimizer stay fp32 master copies; losses are computed in
      fp32 from upcast head outputs).  Rollout autocast is the separate
      top-level ``bf16_rollout`` flag.
    * ``compile`` — ``torch.compile`` the policy forward (rollout + update
      batch shapes each get a specialization; expect ~1-2 min warmup).
    * ``buffer_device`` — where rollout-buffer obs/masks/actions/values
      live: ``"cpu"`` (classic), ``"cuda"`` (kills the per-epoch
      CPU-gather + H2D of every minibatch AND the per-step CPU stores;
      costs ~T*N*4.3KB of VRAM), or ``"auto"`` (cuda when it fits with
      headroom — uses the auto_minibatch probe's peak when available).
    * ``pin_memory`` — pinned staging + ``non_blocking`` H2D for per-step
      obs/mask transfers and (cpu-buffer only) minibatch fetches.
    * ``auto_minibatch`` — at startup, probe ``ppo.minibatch_size`` with a
      synthetic fwd/bwd/step on a throwaway policy copy and halve until it
      fits in VRAM (floor 1024).  Makes one config portable across GPUs.
    """

    bf16_update: bool = False
    compile: bool = False
    compile_mode: str = "default"   # torch.compile mode
    buffer_device: str = "cpu"      # "cpu" | "cuda" | "auto"
    pin_memory: bool = False
    auto_minibatch: bool = False
    fused_adam: bool = False        # fused CUDA Adam (single-kernel step)


@dataclass
class LogConfig:
    run_name: str = "run"         # subdirectory under log_dir / ckpt_dir
    log_dir: str = "runs"
    tensorboard: bool = True
    telemetry: bool = True        # NDJSON event stream read by monitor/
    wandb: bool = False           # optional, off by default
    wandb_project: str = "balatro"
    ckpt_dir: str = "checkpoints"
    save_every_iters: int = 50
    log_every_iters: int = 1


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)
    shaping: ShapingConfig = field(default_factory=ShapingConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 1                 # torch/python RNG seed for training
    device: str = "cuda"          # "cuda" | "cpu"
    bf16_rollout: bool = True     # autocast bf16 for rollout inference


def _build(cls, data: dict[str, Any], path: str):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = data.keys() - fields.keys()
    if unknown:
        raise ValueError(f"unknown config keys at {path}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        ftype = fields[name].type
        if dataclasses.is_dataclass(_DATACLASS_FIELDS.get(ftype, None)):
            kwargs[name] = _build(_DATACLASS_FIELDS[ftype], value, f"{path}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


_DATACLASS_FIELDS = {
    "EnvConfig": EnvConfig,
    "PolicyConfig": PolicyConfig,
    "PPOConfig": PPOConfig,
    "CurriculumConfig": CurriculumConfig,
    "SnapshotConfig": SnapshotConfig,
    "SnapshotAnnealConfig": SnapshotAnnealConfig,
    "ShapingConfig": ShapingConfig,
    "ShapingAnnealConfig": ShapingAnnealConfig,
    "PerfConfig": PerfConfig,
    "LogConfig": LogConfig,
}


def load_config(path: str | Path) -> TrainConfig:
    """Load a TrainConfig from a YAML file (missing keys take defaults)."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return _build(TrainConfig, data, "cfg")


def config_to_dict(cfg: TrainConfig) -> dict:
    return dataclasses.asdict(cfg)
