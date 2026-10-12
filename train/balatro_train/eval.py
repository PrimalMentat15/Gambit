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
import math

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
    max_episode_steps: int = 1000,
) -> dict:
    """Play >= ``episodes`` full episodes from fixed seeds; return summary stats.

    Envs auto-reset, so each of the ``num_envs`` slots contributes a stream
    of episodes; we stop counting once ``episodes`` have completed.

    An episode past ``max_episode_steps`` is a TIMEOUT: it counts as a loss in
    ``win_rate`` and contributes to ``return_mean``/``length_mean``, but it has
    no ante outcome, so it is reported in ``timeouts`` and excluded from
    ``ante_mean``/``ante_hist`` (which therefore sum to ``episodes - timeouts``).
    """
    device = device or next(policy.parameters()).device
    env = make_vec_env(env_name, num_envs, **(env_kwargs or {}))
    obs, masks = env.reset([seed_base + i for i in range(num_envs)])

    # UNBIASED counting: take exactly `per_slot` episodes from EVERY env slot
    # and wait for all slots to reach it. Counting the first-N-to-COMPLETE
    # instead (the old behaviour) is biased toward FAST episodes -- losing runs
    # die at ante 1 while wins take all 8 antes, so the quick finishers are
    # mostly losses and the win rate reads far too low, worst for hard decks.
    # Equal episodes per slot removes that completion-order bias.
    #
    # STRAGGLER CAP (`max_episode_steps`): waiting for every slot means one slot
    # on a pathological long run (episode-length p99 ~356 steps but max ~4,700)
    # can stall the whole call -- and the deck matrix makes 30 calls, which is
    # how a milestone eval ballooned to ~5.8 h. An episode that has not finished
    # within the cap is counted as a LOSS and stopped waiting on; >99% of real
    # wins finish well under the cap, so this barely moves the win rate while
    # bounding wall-clock. `capped` stops the eventual real end of that same
    # episode from being double-counted; run configs keep episodes <= num_envs
    # (per_slot == 1) so a capped slot never has to wait out its long episode
    # to start a second one.
    per_slot = max(1, math.ceil(episodes / num_envs))
    slot_count = [0] * num_envs
    ep_steps = [0] * num_envs
    capped = [False] * num_envs
    returns, lengths, antes, wins = [], [], [], []
    timeouts = 0
    policy.eval()
    for _ in range(max_steps):
        with torch.no_grad():
            actions, *_ = policy.act(
                obs_to_torch(obs, device), obs_to_torch(masks, device),
                deterministic=deterministic,
            )
        obs, masks, _rew, _done, infos = env.step(actions_to_numpy(actions))
        for i, info in enumerate(infos):
            ep_steps[i] += 1
            if "episode" in info:
                if not capped[i] and slot_count[i] < per_slot:
                    ep = info["episode"]
                    returns.append(ep["r"])
                    lengths.append(ep["l"])
                    antes.append(ep.get("ante", 0))
                    wins.append(ep.get("won", False))
                    slot_count[i] += 1
                capped[i] = False
                ep_steps[i] = 0
            elif ep_steps[i] >= max_episode_steps and not capped[i] and slot_count[i] < per_slot:
                # Timed out past the cap: counts as a LOSS for win_rate, but it
                # has no ante outcome -- the episode never ended. It is kept OUT
                # of `antes` and counted in `timeouts` instead, so ante_mean and
                # ante_hist describe only episodes that actually finished.
                # Filing it under ante 0 (the old behaviour) is indistinguishable
                # from a real ante-0 death (Hieroglyph/Petroglyph call
                # ease_ante(-1)) and reads as a catastrophic early death when the
                # truth is the opposite -- the run went too LONG.
                returns.append(0.0)
                lengths.append(ep_steps[i])
                wins.append(False)
                timeouts += 1
                slot_count[i] += 1
                capped[i] = True
        if all(c >= per_slot for c in slot_count):
            break

    returns, antes = np.array(returns), np.array(antes)
    # `episodes` counts timeouts (win_rate must treat them as losses); ante_mean
    # and ante_hist are over COMPLETED episodes only, so they sum to
    # `episodes - timeouts`.
    return {
        "episodes": len(returns),
        "timeouts": timeouts,
        "mode": "argmax" if deterministic else "stochastic",
        "return_mean": float(returns.mean()) if returns.size else 0.0,
        "return_std": float(returns.std()) if returns.size else 0.0,
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "ante_mean": float(antes.mean()) if antes.size else 0.0,
        "ante_hist": {int(a): int((antes == a).sum()) for a in np.unique(antes)},
        "win_rate": float(np.mean(wins)) if wins else 0.0,
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
