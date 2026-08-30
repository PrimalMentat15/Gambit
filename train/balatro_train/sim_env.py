"""`env_api.VecEnv` wrapper over the Rust PyO3 vec-env (``balatro_sim``).

The Rust extension (built from ``sim/py``; see README) does all the work —
N independent `balatro-core` runs stepped in parallel under rayon with the
GIL released, contract-exact obs/mask/reward arrays, strict mask validation,
auto-reset and reward shaping. This wrapper only:

* validates array specs at the boundary (same as the mock does),
* normalizes the ``seeds``/``actions`` containers, and
* exposes the sim-only extras (snapshot/restore, redeterminize, the
  heuristic bot, ``observe``) with numpy-friendly signatures.

Binding decisions (seed derivation, joker state-scalar packing, PLAY/Psychic
masking, USE_CONSUMABLE leniency, snapshot semantics) are documented in
``sim/py/src/*.rs`` module docs and summarized in train/README.md.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from balatro_train import encoding as E

_ACTION_KEYS = tuple(E.ACTION_SPEC.keys())


class SimBalatroEnv:
    """The real (Rust) Balatro vec-env; drop-in for ``MockBalatroEnv``.

    ``strict=True`` (default) makes the env hard-error on any action that
    violates the masks it emitted — the M0 invariant referee. Leave it on;
    it is cheap (a few comparisons per env-step).
    """

    def __init__(self, num_envs: int, strict: bool = True, win_ante: int = 8,
                 deck: int = 0, stake: int = 1):
        import balatro_sim

        # deck/stake are passed ONLY when non-default. They are the binding's
        # own defaults (Red Deck / White Stake), so omitting them is
        # semantically identical -- and it keeps this file working against a
        # binding built before P7 added the arguments. Without that, a trainer
        # restart during the migration window would fail at construction with
        # a TypeError rather than running. Drop the branch once every
        # environment has been rebuilt.
        extra = {}
        if int(deck) != 0:
            extra["deck"] = int(deck)
        if int(stake) != 1:
            extra["stake"] = int(stake)
        self._env = balatro_sim.BalatroVecEnv(num_envs, strict, win_ante, **extra)
        self._num_envs = int(num_envs)
        self._deck = int(deck)
        self._stake = int(stake)

    # ------------------------------------------------------------ VecEnv API

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def set_shaping_beta(self, beta: float) -> None:
        self._env.set_shaping_beta(float(beta))

    def set_win_ante(self, win_ante: int) -> None:
        """Curriculum knob: episodes win (+15, ``info['episode']['won']``)
        once the boss of ante ``win_ante`` is defeated (8 = the real win).
        Applies at each env's NEXT (auto-)reset — in-flight episodes keep
        the goalpost they started with; ``reset()`` applies it everywhere."""
        self._env.set_win_ante(int(win_ante))

    @property
    def deck(self) -> int:
        """Deck index (0 = Red), in `encoding.Deck` order."""
        return self._deck

    @property
    def stake(self) -> int:
        """Stake level, ONE-BASED (1 = White .. 8 = Gold)."""
        return self._stake

    def set_deck(self, deck: int) -> None:
        """Deck every env resets to. Applies at each env's NEXT (auto-)reset,
        exactly like `set_win_ante`."""
        self._env.set_deck(int(deck))
        self._deck = int(deck)

    def set_stake(self, stake: int) -> None:
        """Stake every env resets to (1 = White .. 8 = Gold)."""
        self._env.set_stake(int(stake))
        self._stake = int(stake)

    def set_deck_distribution(self, weights) -> None:
        """Sample a deck per episode from 15 non-negative weights in
        `encoding.Deck` order. An empty list restores the fixed `set_deck`
        value and stops drawing from the seed stream entirely."""
        self._env.set_deck_distribution([float(w) for w in weights])

    def set_stake_distribution(self, weights) -> None:
        """Sample a stake per episode from 8 weights, White first."""
        self._env.set_stake_distribution([float(w) for w in weights])

    def load_snapshot_pool(self, path: str) -> int:
        """Load a packed snapshot start-state pool (validated, one shared
        in-memory copy; format in :mod:`balatro_train.snapshot_pool`).
        Returns the entry count."""
        return int(self._env.load_snapshot_pool(str(path)))

    def set_snapshot_fraction(self, fraction: float) -> None:
        """P(auto-reset starts from a sampled pool snapshot); requires a
        loaded pool when > 0. Applies to auto-resets only — ``reset()``
        always starts fresh runs. Reproducibility/win_ante semantics:
        env_api docstring ("Snapshot start-state mixing")."""
        self._env.set_snapshot_fraction(float(fraction))

    @property
    def snapshot_fraction(self) -> float:
        return float(self._env.snapshot_fraction)

    @property
    def snapshot_pool_len(self) -> int:
        return int(self._env.snapshot_pool_len)

    def reset(self, seeds: Sequence[int]):
        obs, masks = self._env.reset([int(s) for s in seeds])
        E.validate_batch(E.OBS_SPEC, obs, self._num_envs, "obs")
        E.validate_batch(E.MASK_SPEC, masks, self._num_envs, "masks")
        return obs, masks

    def step(self, actions):
        E.validate_batch(E.ACTION_SPEC, actions, self._num_envs, "actions")
        actions = {
            k: np.ascontiguousarray(actions[k], dtype=np.int64) for k in _ACTION_KEYS
        }
        obs, masks, reward, done, infos = self._env.step(actions)
        E.validate_batch(E.OBS_SPEC, obs, self._num_envs, "obs")
        E.validate_batch(E.MASK_SPEC, masks, self._num_envs, "masks")
        return obs, masks, reward, done, infos

    # ------------------------------------------------------------ sim extras

    def observe(self):
        """Re-emit (obs, masks) for the current states (post restore/...)."""
        return self._env.observe()

    def snapshot(self, env_idx: int) -> bytes:
        """Full serialized state of one env (run + RNG + episode stats)."""
        return self._env.snapshot(int(env_idx))

    def restore(self, env_idx: int, snapshot: bytes) -> None:
        self._env.restore(int(env_idx), snapshot)

    def redeterminize(self, env_idx: int, seed: int) -> None:
        """Re-randomize env's unobserved future (deck order + RNG streams)."""
        self._env.redeterminize(int(env_idx), int(seed))

    def bot_actions(self):
        """Batched heuristic-bot actions (``encoding.ACTION_SPEC`` dict)."""
        return self._env.bot_actions()

    def bot_action(self, env_idx: int) -> dict:
        """One env's heuristic-bot action as a dict of Python scalars."""
        return self._env.bot_action(int(env_idx))

    def run_seed(self, env_idx: int) -> str:
        """The Balatro seed string of the env's current run (P5 replays)."""
        return self._env.run_seed(int(env_idx))

    def run_info(self, env_idx: int) -> dict:
        """Scalar run summary ``{seed, ante, round, state}`` — snapshot
        harvest triggers (gen_snapshots.py) without decoding observations."""
        return self._env.run_info(int(env_idx))
