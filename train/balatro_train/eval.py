"""Fixed-seed evaluation protocol skeleton.

Protocol (plan §Training infra): a FIXED set of held-out seeds, each played
once per mode (stochastic sampling and argmax), reporting win-rate, ante
histogram, mean shaped return and episode length.  Runs against anything
implementing the env contract — mock today, Rust sim later — so eval code is
also zero-change when the binding lands.

Run:  python -m balatro_train.eval --checkpoint checkpoints/run/ckpt_X.pt \
          --env mock --episodes 128 --mode argmax
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from balatro_train.config import PolicyConfig, TrainConfig, load_config
from balatro_train.env_api import make_vec_env
from balatro_train.policy import BalatroPolicy, actions_to_numpy, obs_to_torch

# Held-out eval seeds are drawn from a fixed offset so they never collide
# with training seeds (training uses cfg.env.seed + [0, num_envs)).
EVAL_SEED_BASE = 1_000_000


def evaluate(
    policy: BalatroPolicy,
    env_name: str = "mock",
    num_envs: int = 64,
    episodes: int = 128,
    deterministic: bool = False,
    seed_base: int = EVAL_SEED_BASE,
    device: torch.device | None = None,
    env_kwargs: dict | None = None,
    max_steps: int = 100_000,
) -> dict:
    """Play >= ``episodes`` full episodes from fixed seeds; return summary stats.

    Envs auto-reset, so each of the ``num_envs`` slots contributes a stream
    of episodes; we stop counting once ``episodes`` have completed.
    """
    device = device or next(policy.parameters()).device
    env = make_vec_env(env_name, num_envs, **(env_kwargs or {}))
    obs, masks = env.reset([seed_base + i for i in range(num_envs)])

    returns, lengths, antes, wins = [], [], [], []
    policy.eval()
    for _ in range(max_steps):
        with torch.no_grad():
            actions, *_ = policy.act(
                obs_to_torch(obs, device), obs_to_torch(masks, device),
                deterministic=deterministic,
            )
        obs, masks, _rew, _done, infos = env.step(actions_to_numpy(actions))
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                returns.append(ep["r"])
                lengths.append(ep["l"])
                antes.append(ep.get("ante", 0))
                wins.append(ep.get("won", False))
        if len(returns) >= episodes:
            break

    returns, antes = np.array(returns), np.array(antes)
    return {
        "episodes": len(returns),
        "mode": "argmax" if deterministic else "stochastic",
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "length_mean": float(np.mean(lengths)),
        "ante_mean": float(antes.mean()),
        "ante_hist": {int(a): int((antes == a).sum()) for a in np.unique(antes)},
        "win_rate": float(np.mean(wins)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help=".pt from ppo.py (fresh policy if omitted)")
    parser.add_argument("--config", default=None, help="YAML config (for policy dims); "
                        "defaults to the config stored in the checkpoint")
    parser.add_argument("--env", default="mock", help="env registry name (mock | sim)")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--mode", choices=["stochastic", "argmax", "both"], default="both")
    parser.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.config:
        pcfg = load_config(args.config).policy
    elif args.checkpoint:
        from balatro_train.config import _build  # reuse strict dict->dataclass

        ckpt_cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
        pcfg = _build(TrainConfig, ckpt_cfg, "ckpt").policy
    else:
        pcfg = PolicyConfig()

    device = torch.device(args.device)
    policy = BalatroPolicy(pcfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        print(f"loaded {args.checkpoint} (step {ckpt['global_step']})")

    modes = [False, True] if args.mode == "both" else [args.mode == "argmax"]
    for deterministic in modes:
        stats = evaluate(
            policy, args.env, args.num_envs, args.episodes,
            deterministic=deterministic, seed_base=args.seed_base, device=device,
        )
        print({k: v for k, v in stats.items()})


if __name__ == "__main__":
    main()
