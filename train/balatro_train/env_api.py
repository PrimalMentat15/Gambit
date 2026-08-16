"""The vectorized environment CONTRACT for the Balatro RL project.

The training stack (policy / buffer / PPO / eval) is written against the
:class:`VecEnv` protocol below and the array specs in
:mod:`balatro_train.encoding`.  The Rust PyO3 vec-env binding (phase P6, sim
side) must implement this protocol so it drops in with ZERO changes to the
training code.  ``MockBalatroEnv`` (:mod:`balatro_train.mock_env`) is the
reference implementation used by tests and smoke training.

===========================================================================
CONTRACT
===========================================================================

Batching
--------
The env steps ``num_envs`` independent games in lockstep (the Rust side
parallelizes internally with rayon, GIL released).  Every array is
batch-first with leading dimension ``N = num_envs``.

reset(seeds) -> (obs, masks)
----------------------------
* ``seeds``: sequence of ``num_envs`` ints; env *i* starts a fresh run from
  ``seeds[i]``.  Identical seeds must yield identical runs (bit-exact) —
  this is what the fixed-seed eval protocol and the determinism gate rely on.
* Returns ``(obs, masks)`` matching ``encoding.OBS_SPEC`` /
  ``encoding.MASK_SPEC`` (keys, shapes, dtypes exact).
* During ``Phase.PACK`` the ``shop_ids`` / ``shop_feats`` slots carry the
  open pack's contents (see encoding.py) — the policy's pack pointer head
  relies on this.

step(actions) -> (obs, masks, reward, done, info)
-------------------------------------------------
* ``actions``: dict matching ``encoding.ACTION_SPEC``.  For each env the
  composite action is:
    - ``action_type[i]`` — always meaningful.
    - conditional params, meaningful ONLY when ``action_type[i]`` needs them
      (see ``encoding.ACTION_NEEDS_*``); every unused field is -1 (``cards``
      is -1-padded past ``n_cards``) and MUST be ignored by the env.
* Returns:
    - ``obs, masks``   — next observation + next legal-action masks.
    - ``reward``       — float32 ``(N,)``.
    - ``done``         — bool ``(N,)``; True when env *i*'s episode ENDED on
      this step (win or loss).
    - ``info``         — list of N dicts.  ``{}`` normally; when
      ``done[i]`` it contains at least
      ``{"episode": {"r": <undiscounted shaped return>, "l": <length>,
      "ante": <ante reached>, "won": <bool>}}``.
* AUTO-RESET: when ``done[i]`` is True, the returned ``obs``/``masks`` row
  *i* is already the FIRST observation of the NEXT episode (fresh seed drawn
  deterministically from the env's internal RNG stream; later also snapshot
  start-states, sampled Rust-side).  The trainer never calls ``reset``
  mid-run and never needs the terminal observation (bootstrap value is
  zeroed on done).

Mask semantics (the M0 invariant)
---------------------------------
Masks describe EXACTLY the legal choices; the policy guarantees it only ever
samples inside them, and the env is entitled to hard-error on any violation
(the mock does, as should the debug build of the Rust binding):

* ``action_type_mask``  — legal level-1 types for the current phase/state.
  At least one type is always legal in a non-terminal state.
* ``card_select_mask``  — hand cards that may be included in a card-subset
  sub-action.  All-False beyond ``hand_len``.  Guarantee: whenever
  PLAY_HAND or DISCARD is legal, at least one card is selectable.
* ``joker_target_mask`` / ``consumable_target_mask`` / ``shop_target_mask``
  / ``pack_target_mask`` — legal single-target slots.  Guarantee: whenever
  an action type needing pointer head H is legal, H's mask has at least one
  True (e.g. BUY_SHOP legal implies >=1 affordable+roomy shop slot;
  affordability/slot-room checks are baked INTO the mask env-side).
* Masks are consumed at sampling time AND stored verbatim in the rollout
  buffer for the PPO recompute — they are never recomputed from obs.

Card-subset sub-action semantics (shared policy<->env contract)
---------------------------------------------------------------
The card head is an autoregressive pointer over hand slots plus a STOP token
(index ``encoding.CARD_STOP_INDEX``).  Conditional sub-step masks are derived
DETERMINISTICALLY from (stored base ``card_select_mask``, action type, picks
so far) — identically at sampling and recompute time:

* sub-step k availability = ``card_select_mask`` minus already-picked slots;
* STOP legal iff ``k >= encoding.CARD_PICK_MIN[action_type]``, and forced
  when no card is available or ``k == MAX_CARD_PICKS``;
* PLAY_HAND / DISCARD therefore select 1..5 cards; USE_CONSUMABLE selects
  0..5 *candidate targets*.
* Contract decision: per-consumable exact target counts are NOT expressible
  in v1 masks.  The env must interpret the USE_CONSUMABLE subset leniently:
  truncate to the consumable's max target count, ignore extras, and apply
  best-effort if fewer than ideal were given (never an invalid-action error).

Rewards (computed ENV-side; shaping lands in Rust later)
--------------------------------------------------------
Scheme from the project plan — every term capped & monotone toward winning:

1. Per played hand: ``beta * Δ min(chips_scored / blind_requirement, 1)`` —
   sums to exactly 1.0 over a cleared blind; overkill earns nothing.
2. Blind-clear bonus: ``beta * 0.5 * (1 + 0.15 * ante)``. No bonus on skip.
3. Win bonus: +15.0 on beating the boss of the CURRICULUM win ante
   (``win_ante``, default 8 = the real game; NOT scaled by beta).  The
   episode ends there with ``info["episode"]["won"] = True``.
   Loss: no penalty (forfeited future reward suffices).
4. NO money reward (the classic Balatro reward hack).
5. ``beta`` is the shaping scale, annealed 1.0 -> 0.1 by the trainer once
   ante-8 win-rate passes 10%; envs expose :meth:`VecEnv.set_shaping_beta`.

Ante-limit curriculum
---------------------
Envs take a ``win_ante`` ctor kwarg (int, 1..=8, default 8) and expose
:meth:`VecEnv.set_win_ante`.  ``set_win_ante(n)`` applies at each env's
NEXT (auto-)reset: episodes already in flight keep the goalpost they
started with, so a promotion never awards a stale +15 mid-episode.  A full
``reset()`` applies the latest value to every env immediately.  On a
curriculum win ``info["episode"]["won"]`` is True and ``ante`` reports the
post-boss ante (``win_ante + 1`` for the Rust sim).

Snapshot start-state mixing (M3 backward curriculum)
----------------------------------------------------
Envs expose :meth:`VecEnv.load_snapshot_pool` (a packed file of serialized
mid-run states, format in :mod:`balatro_train.snapshot_pool`) and
:meth:`VecEnv.set_snapshot_fraction`.  With fraction ``f > 0`` (requires a
loaded pool), each AUTO-reset starts from a uniformly sampled pool snapshot
with probability ``f`` and from a fresh seed otherwise; a full ``reset()``
always starts fresh runs.  Semantics (Rust reference:
``sim/py/src/env.rs::begin_episode``):

* Reproducibility — the include decision, pool index and the re-seed of the
  restored run's future (redeterminize) are all drawn from the env's own
  per-env seed stream, so trajectories stay a pure function of
  (reset seed, pool file, fraction/win_ante schedule).  At ``f == 0.0`` the
  stream is bit-identical to an env without a pool.
* win_ante — only snapshots with ``ante < win_ante`` (the env's current
  curriculum goal) are eligible; if none qualify the reset falls back to a
  fresh run.
* Episode bookkeeping restarts (return/length zero); the finished episode's
  ``info["episode"]`` gains ``"from_snapshot": bool`` (its START origin).
* Promotion/milestone EVALS must use fresh starts only: eval envs are
  constructed separately and never load a pool, so their fraction is 0.

Determinism
-----------
Given the same reset seeds and the same action sequence, obs/masks/rewards
must be bit-identical across processes and runs.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

ObsDict = dict[str, np.ndarray]
MaskDict = dict[str, np.ndarray]
ActionDict = dict[str, np.ndarray]


@runtime_checkable
class VecEnv(Protocol):
    """Protocol every Balatro vec-env (mock today, PyO3 later) implements."""

    @property
    def num_envs(self) -> int:
        """Number of parallel games (fixed for the lifetime of the object)."""
        ...

    def reset(self, seeds: Sequence[int]) -> tuple[ObsDict, MaskDict]:
        """Start ``num_envs`` fresh runs from the given seeds. See module doc."""
        ...

    def step(
        self, actions: ActionDict
    ) -> tuple[ObsDict, MaskDict, np.ndarray, np.ndarray, list[dict]]:
        """Advance every env by one composite action. See module doc."""
        ...

    def set_shaping_beta(self, beta: float) -> None:
        """Set the shaping scale (reward terms 1-2). Trainer anneal hook."""
        ...

    def set_win_ante(self, win_ante: int) -> None:
        """Set the curriculum win ante (applies at next reset per env)."""
        ...

    def load_snapshot_pool(self, path: str) -> int:
        """Load a packed snapshot start-state pool file; returns its size."""
        ...

    def set_snapshot_fraction(self, fraction: float) -> None:
        """P(auto-reset starts from a pool snapshot). Trainer anneal hook."""
        ...


def make_vec_env(name: str, num_envs: int, **kwargs) -> VecEnv:
    """Tiny env registry.

    ``"mock"`` — :class:`balatro_train.mock_env.MockBalatroEnv`.
    ``"sim"``  — the Rust PyO3 vec-env (phase P6 sim side); wired here once
    the ``balatro_sim`` extension module exists.  Nothing else in the
    training stack knows which one it got.
    """
    if name == "mock":
        from balatro_train.mock_env import MockBalatroEnv

        return MockBalatroEnv(num_envs, **kwargs)
    if name == "sim":
        from balatro_train.sim_env import SimBalatroEnv

        return SimBalatroEnv(num_envs, **kwargs)
    raise ValueError(f"unknown env '{name}' (known: mock, sim)")
