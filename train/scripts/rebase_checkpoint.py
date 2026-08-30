"""Rebase a checkpoint to step 0, keeping the weights but resetting the clock.

Resuming normally continues the global-step counter, which re-anchors every
step-based schedule to where the OLD run left off: the lr anneal ramp, the
shaping beta anneal, the snapshot fraction anneal, and (P7) the deck-curriculum
tier unlocks. Starting a genuinely new training phase from a strong checkpoint
usually wants the opposite -- keep what the policy learned, but let the new
run's schedules start from step 0 so `tier_steps` / lr / anneals mean what the
new config says rather than silently firing in the past.

    python scripts/rebase_checkpoint.py runs/m7/ckpt_<best>.pt -o runs/m8_seed.pt

By default it keeps the optimizer state and the return normalizer (both are
consistent with the weights, so keeping them avoids a re-warm transient) and
only zeroes `global_step` / `iteration`. `--drop-optimizer` starts Adam fresh
instead -- reasonable when the new run uses a very different lr schedule.

Read-only on the input; writes a new file. Does not touch any run directory.
"""

from __future__ import annotations

import argparse
import pathlib

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=pathlib.Path)
    p.add_argument("-o", "--out", type=pathlib.Path, required=True)
    p.add_argument("--drop-optimizer", action="store_true",
                   help="reset Adam moments too (fresh optimizer for the new run)")
    p.add_argument("--drop-ret-norm", action="store_true",
                   help="reset the return normalizer (rarely wanted; reward scale is unchanged)")
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old_step = ck.get("global_step")
    ck["global_step"] = 0
    ck["iteration"] = 0
    if args.drop_optimizer:
        ck["optimizer"] = None
    if args.drop_ret_norm:
        ck["ret_norm"] = None

    torch.save(ck, args.out)
    print(f"rebased {args.checkpoint.name}: global_step {old_step:,} -> 0")
    print(f"  optimizer: {'reset' if args.drop_optimizer else 'kept'}   "
          f"ret_norm: {'reset' if args.drop_ret_norm else 'kept'}   "
          f"policy weights: kept")
    print(f"  wrote {args.out}")
    print("\nResume normally from the new file; every step-based schedule now "
          "starts from 0. Remember the lr ramp restarts at full `ppo.lr` -- set "
          "that deliberately for the new phase.")


if __name__ == "__main__":
    main()
