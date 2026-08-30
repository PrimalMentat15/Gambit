"""Replay a checkpoint and dump per-action traces for episodes that fail a filter.

Aggregate stats say *how often* the policy loses; they never say *what it did*.
This plays the held-out eval seeds, keeps a bounded per-env trace of the actions
taken, and prints the trace for every episode matching a failure pattern -- the
cards it selected, the chips each hand actually scored, and what was left.

Motivating case (``--filter ante1-small``): ~2.3% of runs end at ante 1 with an
episode length of exactly 9, which on Red Deck is SELECT_BLIND + 4 plays + 4
discards -- every resource spent on the *small* blind. Their median shaped
return implies ~90 of the 300 chips required, about 22 per hand, while the
scripted heuristic bot reaches ~212 on the same failure. Something structural
is happening in those states and only the actions can say what.

    python scripts/replay_failures.py --checkpoint runs/m6/ckpt_5000396800.pt \
        --num-envs 128 --episodes 600 --max-traces 8

Read-only: loads a checkpoint, never writes one, and touches no run directory.
"""

from __future__ import annotations

import argparse
import collections
import pathlib

import numpy as np
import torch

import balatro_train.encoding as E
from balatro_train.config import PolicyConfig, TrainConfig, _build
from balatro_train.env_api import make_vec_env
from balatro_train.eval import EVAL_SEED_BASE
from balatro_train.policy import BalatroPolicy, actions_to_numpy, obs_to_torch

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["S", "H", "C", "D"]
ACTION_NAMES = {int(a): a.name for a in E.ActionType}

#: Episodes never run long before the failures we care about; a bounded trace
#: keeps memory flat no matter how many envs or steps are simulated.
MAX_TRACE = 64


def decode_card(row: np.ndarray) -> str:
    """One hand-token feature row -> e.g. 'AS' or 'TD*' (enhanced)."""
    rank = int(np.argmax(row[E.CARD_RANK_OFF:E.CARD_RANK_OFF + 13]))
    suit = int(np.argmax(row[E.CARD_SUIT_OFF:E.CARD_SUIT_OFF + 4]))
    out = f"{RANKS[rank]}{SUITS[suit]}"
    enh = row[E.CARD_ENH_OFF:E.CARD_ENH_OFF + 9]
    if enh.any() and int(np.argmax(enh)) != 0:
        out += "*"
    return out


def decode_hand(hand: np.ndarray, hand_len) -> list[str]:
    return [decode_card(hand[i]) for i in range(min(int(hand_len), E.HAND_MAX))]


def snap(obs: dict, i: int) -> dict:
    """The bits of env i's observation a human needs to read a trace."""
    blind = obs["blind"][i]
    g = obs["global"][i]
    phase = int(np.argmax(g[E.GLOBAL_PHASE_OFF:E.GLOBAL_PHASE_OFF + E.N_PHASES]))
    return {
        "hand": decode_hand(obs["hand"][i], obs["hand_len"][i]),
        # chips are stored log1p in the blind features
        "scored": float(np.expm1(blind[E.BLIND_SCORED_OFF])),
        "req": float(np.expm1(blind[E.BLIND_REQ_OFF])),
        # stored normalized; undo it so the trace reads in game units
        "hands_left": round(float(g[E.GLOBAL_HANDS_LEFT]) * 4.0),
        "discards_left": round(float(g[E.GLOBAL_DISCARDS_LEFT]) * 4.0),
        "phase": E.Phase(phase).name,
    }


FILTERS = {
    # ante 1, exactly 9 steps: SELECT_BLIND + 4 plays + 4 discards, i.e. every
    # hand and discard spent without clearing the SMALL blind.
    "ante1-small": lambda ep: ep.get("ante") == 1 and ep.get("l") == 9,
    # ante 1, longer: cleared small+big, lost the boss. Much more defensible.
    "ante1-boss": lambda ep: ep.get("ante") == 1 and 20 <= ep.get("l", 0) <= 39,
    "any-loss": lambda ep: not ep.get("won", False),
    # degenerate loops -- 64 of these appeared in m6, the longest 131k steps
    "runaway": lambda ep: ep.get("l", 0) > 1000,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=pathlib.Path)
    p.add_argument("--env", default="sim")
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--filter", default="ante1-small", choices=sorted(FILTERS))
    p.add_argument("--max-traces", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    p.add_argument("--stochastic", action="store_true",
                   help="sample instead of argmax (eval uses argmax)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pcfg: PolicyConfig = _build(TrainConfig, ck["config"], "ckpt").policy
    device = torch.device(args.device)
    policy = BalatroPolicy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    print(f"loaded {args.checkpoint} (step {ck['global_step']:,})")

    env = make_vec_env(args.env, args.num_envs)
    obs, masks = env.reset([args.seed_base + i for i in range(args.num_envs)])

    keep = FILTERS[args.filter]
    traces: dict[int, collections.deque] = {
        i: collections.deque(maxlen=MAX_TRACE) for i in range(args.num_envs)
    }
    matched: list[tuple[int, dict, list]] = []
    n_done = 0
    play_chips: list[float] = []        # per PLAY_HAND, matched episodes only
    play_sizes: collections.Counter = collections.Counter()
    # Control: the same two stats over every play in every episode. Without it
    # "median 15 chips per hand" has nothing to be bad relative to.
    all_chips: list[float] = []
    all_sizes: collections.Counter = collections.Counter()

    for _ in range(args.max_steps):
        with torch.no_grad():
            acts, *_ = policy.act(
                obs_to_torch(obs, device), obs_to_torch(masks, device),
                deterministic=not args.stochastic,
            )
        a = actions_to_numpy(acts)
        before = [snap(obs, i) for i in range(args.num_envs)]
        obs, masks, _rew, done, infos = env.step(a)

        for i in range(args.num_envs):
            at = int(a["action_type"][i])
            n = int(a["n_cards"][i])
            picks = [int(x) for x in a["cards"][i][:n] if x >= 0]
            traces[i].append({
                "action": ACTION_NAMES.get(at, str(at)),
                "picked": [before[i]["hand"][j] for j in picks
                           if j < len(before[i]["hand"])],
                "before": before[i],
                # On the terminating step the env has ALREADY auto-reset, so a
                # post-step observation belongs to the NEXT episode: its chip
                # count is 0 and a naive delta reads as a large negative.
                "after": None if done[i] else snap(obs, i),
            })

        for i in np.flatnonzero(done):
            i = int(i)
            ep = infos[i].get("episode")
            n_done += 1
            hit = bool(ep) and keep(ep)
            if hit and len(matched) < args.max_traces:
                matched.append((args.seed_base + i, ep, list(traces[i])))
            for rec in traces[i]:
                if rec["action"] != "PLAY_HAND" or rec["after"] is None:
                    continue
                gain = rec["after"]["scored"] - rec["before"]["scored"]
                all_chips.append(gain)
                all_sizes[len(rec["picked"])] += 1
                if hit:
                    play_chips.append(gain)
                    play_sizes[len(rec["picked"])] += 1
            traces[i].clear()

        if n_done >= args.episodes:
            break

    print(f"{n_done} episodes; {len(matched)} traces captured "
          f"for filter {args.filter!r}\n")

    for seed, ep, tr in matched:
        print("=" * 72)
        print(f"seed {seed}  ante {ep['ante']}  len {ep['l']}  "
              f"return {ep['r']:.3f}  won {ep['won']}")
        for k, rec in enumerate(tr):
            b, af = rec["before"], rec["after"]
            line = (f"  {k:>2} {rec['action']:<16} "
                    f"hands {b['hands_left']} disc {b['discards_left']}  "
                    f"chips {b['scored']:>7.0f}/{b['req']:<7.0f}")
            if rec["picked"]:
                line += f"  picked {'+'.join(rec['picked'])}"
            if rec["action"] == "PLAY_HAND":
                line += ("  -> (episode ended)" if af is None
                         else f"  -> +{af['scored'] - b['scored']:.0f}")
            print(line)
            if b["hand"]:
                print(f"     hand: {' '.join(b['hand'])}")

    def report(name: str, chips: list[float], sizes: collections.Counter) -> None:
        if not chips:
            return
        arr = np.array(chips)
        tot = sum(sizes.values())
        print(f"\n{name} ({len(arr)} plays):")
        print(f"  chips per PLAY_HAND: mean {arr.mean():7.1f}  "
              f"median {np.median(arr):6.1f}  p90 {np.percentile(arr, 90):7.1f}  "
              f"max {arr.max():.0f}")
        print("  cards per PLAY_HAND: "
              + ", ".join(f"{k}:{100 * v / tot:.0f}%" for k, v in sorted(sizes.items())))

    print("\n" + "=" * 72)
    report(f"matched ({args.filter})", play_chips, play_sizes)
    report("ALL episodes (control)", all_chips, all_sizes)


if __name__ == "__main__":
    main()
