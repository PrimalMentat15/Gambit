"""M4 reward-shaping anneal: schedule math, config parsing, trainer push."""

import textwrap

from balatro_train.config import ShapingAnnealConfig, ShapingConfig, load_config
from balatro_train.ppo import shaping_beta_at


def _cfg(beta=1.0, start=100, end=200, final=0.1) -> ShapingConfig:
    return ShapingConfig(
        beta=beta,
        anneal=ShapingAnnealConfig(start_step=start, end_step=end, final_beta=final),
    )


def test_schedule_before_during_after():
    c = _cfg()
    assert shaping_beta_at(0, c) == 1.0
    assert shaping_beta_at(100, c) == 1.0
    assert abs(shaping_beta_at(150, c) - 0.55) < 1e-9  # midpoint of 1.0 -> 0.1
    assert shaping_beta_at(200, c) == 0.1
    assert shaping_beta_at(10**12, c) == 0.1


def test_schedule_disabled_is_constant():
    c = _cfg(start=0, end=0)
    for step in (0, 1, 10**9):
        assert shaping_beta_at(step, c) == 1.0


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        textwrap.dedent(
            """
            shaping:
              beta: 1.0
              anneal:
                start_step: 500000000
                end_step: 900000000
                final_beta: 0.1
            """
        )
    )
    cfg = load_config(p)
    assert cfg.shaping.beta == 1.0
    assert cfg.shaping.anneal.start_step == 500_000_000
    assert cfg.shaping.anneal.final_beta == 0.1


def test_default_is_constant_beta_one():
    # Absent config block -> β pinned at 1.0 forever (pre-M4 behavior).
    assert shaping_beta_at(10**12, ShapingConfig()) == 1.0


def test_trainer_pushes_beta_to_env(tmp_path):
    from balatro_train.config import TrainConfig
    from balatro_train.ppo import PPOTrainer

    cfg = load_config("configs/debug.yaml")
    assert isinstance(cfg, TrainConfig)
    cfg.shaping.anneal.start_step = 0
    cfg.shaping.anneal.end_step = 1  # immediately annealed
    cfg.shaping.anneal.final_beta = 0.25
    cfg.log.tensorboard = False
    cfg.log.log_dir = str(tmp_path)
    cfg.log.ckpt_dir = str(tmp_path)
    cfg.device = "cpu"
    trainer = PPOTrainer(cfg)
    # __init__ applies the schedule at step 0 -> beta == cfg.shaping.beta (1.0)
    assert trainer.shaping_beta == 1.0
    trainer.global_step = 10
    trainer._apply_shaping_beta()
    assert trainer.shaping_beta == 0.25
    assert trainer.env._beta == 0.25  # mock env records the push
