"""Pure-numpy mock Balatro vec-env implementing the ``env_api.VecEnv`` contract.

Used by every test and by smoke training runs until the Rust PyO3 vec-env
lands.  Dynamics are fake-but-plausible random walks; what is EXACT is the
contract surface: array specs, mask semantics (only truly legal choices are
ever marked True), reward shaping scheme, auto-reset, and determinism.

The mock is also a strict referee: every incoming action is validated against
the masks it emitted and the composite-action conventions (unused params must
be -1), raising ``ValueError`` on any violation.  This is what makes the
masking tests meaningful (M0 gate: literal-zero invalid actions).

Two reward modes:

* ``"shaped"`` (default) — the plan's shaping scheme (see env_api docstring).
* ``"bandit"`` — trivially learnable debug reward: the env stays in the
  PLAYING phase (only PLAY_HAND / DISCARD legal, resources never deplete),
  episodes last exactly ``bandit_episode_len`` steps, and DISCARD pays +1.0
  (PLAY_HAND pays 0).  Used by the PPO overfit test.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from balatro_train import encoding as E
from balatro_train.encoding import ActionType, BlindKind, Edition, Phase, ShopKind

_A = ActionType


def _zeros(spec, num_envs):
    return {k: np.zeros((num_envs, *shape), dtype=dt) for k, (shape, dt) in spec.items()}


class _Game:
    """One mock Balatro run. Holds state, emits obs/masks, applies actions."""

    def __init__(self, seed: int, reward_mode: str, bandit_episode_len: int,
                 win_ante: int = 8):
        self.reward_mode = reward_mode
        self.bandit_episode_len = bandit_episode_len
        # Curriculum knob (mirrors the Rust binding): win when the boss of
        # ante `win_ante` falls.  `pending_win_ante` is what set_win_ante
        # writes; it becomes `win_ante` at the next (auto-)reset.
        self.pending_win_ante = win_ante
        # Snapshot start-state mixing (set by the env; see _begin_episode).
        self.snapshot_antes: np.ndarray | None = None
        self.snapshot_fraction = 0.0
        self.reset(seed)

    # -- lifecycle -----------------------------------------------------------

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.ep_return = 0.0
        self.ep_len = 0
        self.won = False
        self.win_ante = self.pending_win_ante
        self.from_snapshot = False

        self.ante = 1
        self.round = 0
        self.money = 4
        self.joker_slots = 5
        self.consumable_slots = 2
        self.hand_size = int(r.integers(7, E.HAND_MAX + 1))
        self.jokers: list[tuple[int, np.ndarray]] = []   # (id, feats)
        self.consumables: list[int] = []
        self.shop: list[tuple[int, int, int, int]] = []  # (kind, id, cost, edition)
        self.pack: list[tuple[int, int]] = []            # (kind, id)
        self.pack_picks_left = 0
        self.reroll_cost = 5
        self.hand_levels = np.zeros((E.N_HAND_TYPES, 2), dtype=np.float64)
        self.deck_counts = np.ones(E.N_DECK_SLOTS, dtype=np.float64)
        self.drawpile = self.deck_counts.copy()
        self.hand: list[tuple[int, int]] = []            # (rank, suit)

        if self.reward_mode == "bandit":
            self.phase = Phase.PLAYING
            self.blind_kind = BlindKind.SMALL
            self.boss_id = 0
            self.blind_req = 300.0
            self.blind_scored = 0.0
            self.hands_left = 4
            self.discards_left = 4
            self.hand_size = 8
            self._deal()
            return

        self.phase = Phase.BLIND_SELECT
        self.blind_kind = BlindKind.SMALL
        self._roll_boss()
        self._set_blind_req()
        self.blind_scored = 0.0
        self.hands_left = 4
        self.discards_left = 3

    def _roll_boss(self) -> None:
        self.boss_id = int(self.rng.integers(1, E.BOSS_VOCAB)) if self.blind_kind == BlindKind.BOSS else 0

    def _set_blind_req(self) -> None:
        mult = {BlindKind.SMALL: 1.0, BlindKind.BIG: 1.5, BlindKind.BOSS: 2.0}[self.blind_kind]
        self.blind_req = 300.0 * (2.2 ** (self.ante - 1)) * mult

    def _deal(self) -> None:
        r = self.rng
        self.hand = [(int(r.integers(E.N_RANKS)), int(r.integers(E.N_SUITS))) for _ in range(self.hand_size)]
        self._consume_drawpile(self.hand_size)

    def _consume_drawpile(self, n: int) -> None:
        avail = np.flatnonzero(self.drawpile > 0)
        if avail.size:
            take = self.rng.choice(avail, size=min(n, avail.size), replace=False)
            self.drawpile[take] -= 1

    def _roll_shop(self) -> None:
        r = self.rng
        self.shop = []
        n_items = int(r.integers(3, E.SHOP_SLOTS + 1))
        kinds = [ShopKind.JOKER, ShopKind.TAROT, ShopKind.PLANET, ShopKind.SPECTRAL,
                 ShopKind.PLAYING_CARD, ShopKind.VOUCHER, ShopKind.PACK]
        for _ in range(n_items):
            kind = int(r.choice(kinds))
            item_id = int(r.integers(1, E.SHOP_VOCAB))
            cost = int(r.integers(2, 9 + self.ante))
            edition = int(r.choice([0, 0, 0, 1, 2, 3]))
            self.shop.append((kind, item_id, cost, edition))

    def _roll_pack(self) -> None:
        r = self.rng
        n = int(r.integers(2, E.PACK_SLOTS + 1))
        pack_kinds = [ShopKind.TAROT, ShopKind.PLANET, ShopKind.SPECTRAL, ShopKind.PLAYING_CARD]
        self.pack = [(int(r.choice(pack_kinds)), int(r.integers(1, E.SHOP_VOCAB))) for _ in range(n)]
        self.pack_picks_left = 1 if r.random() < 0.8 else 2

    # -- observation ---------------------------------------------------------

    def write_obs(self, obs: dict[str, np.ndarray], i: int) -> None:
        hand = obs["hand"][i]
        for slot, (rank, suit) in enumerate(self.hand[: E.HAND_MAX]):
            hand[slot, E.CARD_RANK_OFF + rank] = 1.0
            hand[slot, E.CARD_SUIT_OFF + suit] = 1.0
            hand[slot, E.CARD_ENH_OFF + 0] = 1.0
            hand[slot, E.CARD_EDITION_OFF + 0] = 1.0
            hand[slot, E.CARD_SEAL_OFF + 0] = 1.0
        obs["hand_len"][i] = len(self.hand)

        for slot, (jid, feats) in enumerate(self.jokers[: E.JOKER_SLOTS]):
            obs["joker_ids"][i, slot] = jid
            obs["joker_feats"][i, slot] = feats
        for slot, cid in enumerate(self.consumables[: E.CONSUMABLE_SLOTS]):
            obs["consumable_ids"][i, slot] = cid
        obs["consumables_len"][i] = len(self.consumables)

        if self.phase == Phase.SHOP:
            for slot, (kind, item_id, cost, edition) in enumerate(self.shop[: E.SHOP_SLOTS]):
                obs["shop_ids"][i, slot] = item_id
                sf = obs["shop_feats"][i, slot]
                sf[E.SHOP_KIND_OFF + kind] = 1.0
                sf[E.SHOP_COST_OFF] = np.log1p(cost)
                sf[E.SHOP_EDITION_OFF + edition] = 1.0
        elif self.phase == Phase.PACK:
            # Contract: during Phase.PACK the shop slots carry the OPEN
            # PACK's contents (cost 0), so the policy's pack pointer can
            # attend to real item tokens.
            for slot, (kind, item_id) in enumerate(self.pack[: E.PACK_SLOTS]):
                obs["shop_ids"][i, slot] = item_id
                sf = obs["shop_feats"][i, slot]
                sf[E.SHOP_KIND_OFF + kind] = 1.0
                sf[E.SHOP_EDITION_OFF + Edition.NONE] = 1.0

        blind = obs["blind"][i]
        blind[E.BLIND_KIND_OFF + int(self.blind_kind)] = 1.0
        if self.boss_id > 0:
            blind[E.BLIND_BOSS_OFF + self.boss_id] = 1.0
        blind[E.BLIND_REQ_OFF] = np.log1p(self.blind_req)
        blind[E.BLIND_SCORED_OFF] = np.log1p(self.blind_scored)
        blind[E.BLIND_REWARD_OFF] = np.log1p(3 + int(self.blind_kind))

        g = obs["global"][i]
        g[E.GLOBAL_MONEY] = np.sign(self.money) * np.log1p(abs(self.money))
        g[E.GLOBAL_HANDS_LEFT] = self.hands_left / 4.0
        g[E.GLOBAL_DISCARDS_LEFT] = self.discards_left / 4.0
        g[E.GLOBAL_ANTE_OFF + min(self.ante, 9) - 1] = 1.0
        g[E.GLOBAL_ROUND] = self.round / 24.0
        g[E.GLOBAL_PHASE_OFF + int(self.phase)] = 1.0
        hl = self.hand_levels
        g[E.GLOBAL_HAND_LEVELS_OFF : E.GLOBAL_HAND_LEVELS_OFF + 2 * E.N_HAND_TYPES] = np.log1p(hl).ravel()
        g[E.GLOBAL_REROLL_COST] = np.log1p(self.reroll_cost)
        g[E.GLOBAL_JOKER_SLOT_COUNT] = self.joker_slots / 6.0
        g[E.GLOBAL_CONSUMABLE_SLOT_COUNT] = self.consumable_slots / 3.0
        g[E.GLOBAL_DECK_SIZE] = np.log1p(self.deck_counts.sum())
        g[E.GLOBAL_HAND_SIZE] = self.hand_size / 10.0
        # Deck/stake: the mock only ever models Red/White, but it must still
        # satisfy the contract's one-hot invariant so it stays a valid stand-in
        # for the sim env in shape/consistency tests.
        g[E.GLOBAL_DECK_OFF + int(E.Deck.RED)] = 1.0
        g[E.GLOBAL_STAKE] = (int(E.Stake.WHITE) - 1) / (E.N_STAKES - 1)

        obs["deck_counts"][i] = self.deck_counts
        obs["drawpile_counts"][i] = self.drawpile
        agg = obs["deck_aggregates"][i]
        agg[E.DECK_AGG_ENH_OFF] = np.log1p(self.deck_counts.sum())  # all NONE-enhanced
        agg[E.DECK_AGG_TOTAL] = np.log1p(self.deck_counts.sum())

    # -- masks ---------------------------------------------------------------

    def _shop_slot_buyable(self, kind: int, cost: int) -> bool:
        if cost > self.money:
            return False
        if kind == ShopKind.JOKER and len(self.jokers) >= self.joker_slots:
            return False
        if kind in (ShopKind.TAROT, ShopKind.PLANET, ShopKind.SPECTRAL) and len(
            self.consumables
        ) >= self.consumable_slots:
            return False
        return True

    def _pack_slot_takeable(self, kind: int) -> bool:
        if kind in (ShopKind.TAROT, ShopKind.PLANET, ShopKind.SPECTRAL) and len(
            self.consumables
        ) >= self.consumable_slots:
            return False
        return True

    def write_masks(self, masks: dict[str, np.ndarray], i: int) -> None:
        at = masks["action_type_mask"][i]
        can_use_stuff = self.phase in (Phase.BLIND_SELECT, Phase.PLAYING, Phase.SHOP)

        if self.reward_mode == "bandit":
            at[_A.PLAY_HAND] = True
            at[_A.DISCARD] = True
            masks["card_select_mask"][i, : len(self.hand)] = True
            return

        if self.phase == Phase.BLIND_SELECT:
            at[_A.SELECT_BLIND] = True
            at[_A.SKIP_BLIND] = self.blind_kind != BlindKind.BOSS
        elif self.phase == Phase.PLAYING:
            at[_A.PLAY_HAND] = len(self.hand) > 0
            at[_A.DISCARD] = self.discards_left > 0 and len(self.hand) > 0
            masks["card_select_mask"][i, : len(self.hand)] = True
        elif self.phase == Phase.ROUND_EVAL:
            at[_A.CASH_OUT] = True
        elif self.phase == Phase.SHOP:
            for slot, (kind, _id, cost, _ed) in enumerate(self.shop):
                masks["shop_target_mask"][i, slot] = self._shop_slot_buyable(kind, cost)
            at[_A.BUY_SHOP] = bool(masks["shop_target_mask"][i].any())
            at[_A.REROLL] = self.money >= self.reroll_cost
            at[_A.LEAVE_SHOP] = True
        elif self.phase == Phase.PACK:
            for slot, (kind, _id) in enumerate(self.pack):
                masks["pack_target_mask"][i, slot] = self._pack_slot_takeable(kind)
            at[_A.PICK_PACK] = bool(masks["pack_target_mask"][i].any())
            at[_A.SKIP_PACK] = True

        if can_use_stuff:
            if self.consumables:
                at[_A.USE_CONSUMABLE] = True
                at[_A.SELL_CONSUMABLE] = True
                masks["consumable_target_mask"][i, : len(self.consumables)] = True
            if self.jokers:
                at[_A.SELL_JOKER] = True
                masks["joker_target_mask"][i, : len(self.jokers)] = True

    # -- action validation (strict referee) ----------------------------------

    def validate(self, masks: dict[str, np.ndarray], i: int, actions: dict[str, np.ndarray]) -> None:
        t = int(actions["action_type"][i])
        if not (0 <= t < E.N_ACTION_TYPES) or not masks["action_type_mask"][i, t]:
            raise ValueError(f"env {i}: illegal action_type {t} (phase {self.phase.name})")
        t = ActionType(t)

        def check_ptr(field: str, mask_key: str, needed: bool) -> None:
            v = int(actions[field][i])
            if not needed:
                if v != -1:
                    raise ValueError(f"env {i}: {field}={v} must be -1 for {t.name}")
                return
            if v < 0 or not masks[mask_key][i, v]:
                raise ValueError(f"env {i}: illegal {field}={v} for {t.name}")

        check_ptr("joker_target", "joker_target_mask", t in E.ACTION_NEEDS_JOKER)
        check_ptr("consumable_target", "consumable_target_mask", t in E.ACTION_NEEDS_CONSUMABLE)
        check_ptr("shop_target", "shop_target_mask", t in E.ACTION_NEEDS_SHOP)
        check_ptr("pack_target", "pack_target_mask", t in E.ACTION_NEEDS_PACK)

        n_cards = int(actions["n_cards"][i])
        cards = actions["cards"][i]
        if t not in E.ACTION_NEEDS_CARDS:
            if n_cards != 0 or (cards != -1).any():
                raise ValueError(f"env {i}: card params must be empty for {t.name}")
            return
        lo = E.CARD_PICK_MIN[t]
        if not (lo <= n_cards <= E.MAX_CARD_PICKS):
            raise ValueError(f"env {i}: n_cards={n_cards} out of [{lo},{E.MAX_CARD_PICKS}] for {t.name}")
        picked = cards[:n_cards]
        if (cards[n_cards:] != -1).any():
            raise ValueError(f"env {i}: cards not -1-padded past n_cards")
        if len(set(picked.tolist())) != n_cards:
            raise ValueError(f"env {i}: duplicate card picks {picked}")
        for c in picked:
            if c < 0 or not masks["card_select_mask"][i, c]:
                raise ValueError(f"env {i}: illegal card pick {c}")

    # -- dynamics ------------------------------------------------------------

    def apply(self, actions: dict[str, np.ndarray], i: int, beta: float) -> tuple[float, bool, dict]:
        """Apply env i's composite action. Returns (reward, done, info)."""
        t = ActionType(int(actions["action_type"][i]))
        n_cards = int(actions["n_cards"][i])
        r = self.rng
        reward = 0.0
        done = False

        if self.reward_mode == "bandit":
            reward = 1.0 if t == _A.DISCARD else 0.0
            self._deal()
            self.ep_len += 1
            self.ep_return += reward
            if self.ep_len >= self.bandit_episode_len:
                done = True
            return self._finish(reward, done)

        if t == _A.SELECT_BLIND:
            self.phase = Phase.PLAYING
            self.blind_scored = 0.0
            self.hands_left = 4
            self.discards_left = 3
            self.drawpile = self.deck_counts.copy()
            self._deal()
        elif t == _A.SKIP_BLIND:
            self._advance_blind()
        elif t == _A.PLAY_HAND:
            before = min(self.blind_scored / self.blind_req, 1.0)
            gain = float(r.lognormal(mean=0.0, sigma=0.6)) * self.blind_req / 2.5 * (0.5 + 0.3 * n_cards)
            self.blind_scored += gain
            after = min(self.blind_scored / self.blind_req, 1.0)
            reward += beta * (after - before)
            self.hands_left -= 1
            ht = int(r.integers(E.N_HAND_TYPES))
            self.hand_levels[ht, 1] += 1.0
            if self.blind_scored >= self.blind_req:
                reward += beta * 0.5 * (1 + 0.15 * self.ante)
                if self.blind_kind == BlindKind.BOSS and self.ante >= self.win_ante:
                    reward += 15.0
                    self.won = True
                    done = True
                else:
                    self.phase = Phase.ROUND_EVAL
                    self.round += 1
            elif self.hands_left <= 0:
                done = True  # loss: no penalty
            else:
                self._deal()
        elif t == _A.DISCARD:
            self.discards_left -= 1
            self._deal()
        elif t == _A.CASH_OUT:
            self.money += 3 + int(self.blind_kind) + min(self.money // 5, 5)
            self.phase = Phase.SHOP
            self.reroll_cost = 5
            self._roll_shop()
        elif t == _A.BUY_SHOP:
            slot = int(actions["shop_target"][i])
            kind, item_id, cost, _ed = self.shop.pop(slot)
            self.money -= cost
            if kind == ShopKind.JOKER:
                feats = np.zeros(E.F_JOKER_FEAT, dtype=np.float32)
                feats[E.JOKER_EDITION_OFF] = 1.0
                feats[E.JOKER_SELL_OFF] = np.log1p(max(cost // 2, 1))
                self.jokers.append((1 + item_id % (E.JOKER_VOCAB - 1), feats))
            elif kind in (ShopKind.TAROT, ShopKind.PLANET, ShopKind.SPECTRAL):
                self.consumables.append(1 + item_id % (E.CONSUMABLE_VOCAB - 1))
            elif kind == ShopKind.PACK:
                self.phase = Phase.PACK
                self._roll_pack()
            elif kind == ShopKind.PLAYING_CARD:
                self.deck_counts[int(r.integers(E.N_DECK_SLOTS))] += 1
            # voucher: no modeled effect in the mock
        elif t == _A.REROLL:
            self.money -= self.reroll_cost
            self.reroll_cost += 1
            self._roll_shop()
        elif t == _A.LEAVE_SHOP:
            self._advance_blind()
        elif t == _A.USE_CONSUMABLE:
            slot = int(actions["consumable_target"][i])
            self.consumables.pop(slot)
            effect = r.random()
            if effect < 0.4:
                ht = int(r.integers(E.N_HAND_TYPES))
                self.hand_levels[ht, 0] += 1.0
            elif effect < 0.6:
                self.money += int(r.integers(1, 5))
        elif t == _A.SELL_JOKER:
            slot = int(actions["joker_target"][i])
            _jid, feats = self.jokers.pop(slot)
            self.money += int(np.expm1(feats[E.JOKER_SELL_OFF]))
        elif t == _A.SELL_CONSUMABLE:
            slot = int(actions["consumable_target"][i])
            self.consumables.pop(slot)
            self.money += 1
        elif t == _A.PICK_PACK:
            slot = int(actions["pack_target"][i])
            kind, item_id = self.pack.pop(slot)
            if kind in (ShopKind.TAROT, ShopKind.PLANET, ShopKind.SPECTRAL):
                self.consumables.append(1 + item_id % (E.CONSUMABLE_VOCAB - 1))
            else:
                self.deck_counts[int(r.integers(E.N_DECK_SLOTS))] += 1
            self.pack_picks_left -= 1
            if self.pack_picks_left <= 0 or not self.pack:
                self.phase = Phase.SHOP
        elif t == _A.SKIP_PACK:
            self.phase = Phase.SHOP
        else:  # pragma: no cover - masked out
            raise ValueError(f"unhandled action type {t}")

        self.ep_len += 1
        self.ep_return += reward
        return self._finish(reward, done)

    def _advance_blind(self) -> None:
        if self.blind_kind == BlindKind.BOSS:
            self.ante += 1
            self.blind_kind = BlindKind.SMALL
        else:
            self.blind_kind = BlindKind(int(self.blind_kind) + 1)
        self.phase = Phase.BLIND_SELECT
        self.hand = []
        self._roll_boss()
        self._set_blind_req()
        self.blind_scored = 0.0

    def _finish(self, reward: float, done: bool) -> tuple[float, bool, dict]:
        info: dict = {}
        if done:
            info["episode"] = {
                "r": float(self.ep_return),
                "l": int(self.ep_len),
                "ante": int(self.ante),
                "won": bool(self.won),
                "from_snapshot": bool(self.from_snapshot),
            }
            self._begin_episode()
        return reward, done, info

    def _begin_episode(self) -> None:
        """AUTO-RESET, mirroring the Rust env's ``begin_episode``: with
        probability ``snapshot_fraction``, start from a sampled pool
        snapshot; a fresh-seed episode otherwise.  The mock cannot decode
        the Rust snapshot payloads, so it fakes the essence — a mid-run
        start at the sampled entry's ante (BLIND_SELECT) — which is enough
        for trainer-side tests (from_snapshot bookkeeping, per-source
        stats, anneal plumbing).  Same contract semantics: every draw comes
        from the env's own episode stream, only entries with
        ``ante < win_ante`` are eligible, fresh fallback when none qualify,
        and fraction 0.0 draws nothing extra (bit-identical to pre-M3)."""
        rng = self.rng
        seed = int(rng.integers(0, 2**63 - 1))
        antes = self.snapshot_antes
        if antes is not None and self.snapshot_fraction > 0.0:
            if rng.random() < self.snapshot_fraction:
                eligible = antes[antes < self.pending_win_ante]
                if eligible.size:
                    start = int(eligible[int(rng.integers(eligible.size))])
                    self.reset(seed)
                    self.ante = start
                    self._set_blind_req()
                    self.from_snapshot = True
                    return
        self.reset(seed)


class MockBalatroEnv:
    """`env_api.VecEnv` reference implementation (see module docstring)."""

    def __init__(self, num_envs: int, reward_mode: str = "shaped", bandit_episode_len: int = 32,
                 win_ante: int = 8):
        if reward_mode not in ("shaped", "bandit"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        if not 1 <= win_ante <= 8:
            raise ValueError(f"win_ante must be in 1..=8, got {win_ante}")
        self._num_envs = num_envs
        self._reward_mode = reward_mode
        self._bandit_episode_len = bandit_episode_len
        self._win_ante = win_ante
        self._beta = 1.0
        self._games: list[_Game] | None = None
        self._last_masks: dict[str, np.ndarray] | None = None
        self._pool_antes: np.ndarray | None = None
        self._snapshot_fraction = 0.0

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def set_shaping_beta(self, beta: float) -> None:
        self._beta = float(beta)

    def set_win_ante(self, win_ante: int) -> None:
        """Curriculum knob (see env_api docstring): applies at each env's
        NEXT (auto-)reset; a full ``reset()`` applies it everywhere."""
        if not 1 <= win_ante <= 8:
            raise ValueError(f"win_ante must be in 1..=8, got {win_ante}")
        self._win_ante = int(win_ante)
        for game in self._games or []:
            game.pending_win_ante = self._win_ante

    def load_snapshot_pool(self, path: str) -> int:
        """Mock parity for the sim's pool loading: reads the pool file's
        ante table (the payloads are Rust-only). Returns the entry count."""
        from balatro_train import snapshot_pool

        self._pool_antes = np.asarray(snapshot_pool.read_pool_antes(path),
                                      dtype=np.int64)
        for game in self._games or []:
            game.snapshot_antes = self._pool_antes
        return int(self._pool_antes.size)

    def set_snapshot_fraction(self, fraction: float) -> None:
        """P(auto-reset starts from a pool snapshot); requires a loaded
        pool when > 0. Auto-resets only — ``reset()`` starts fresh runs."""
        f = float(fraction)
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"snapshot fraction must be in [0, 1], got {f}")
        if f > 0.0 and self._pool_antes is None:
            raise ValueError("snapshot fraction > 0 requires load_snapshot_pool() first")
        self._snapshot_fraction = f
        for game in self._games or []:
            game.snapshot_fraction = f

    def reset(self, seeds: Sequence[int]):
        if len(seeds) != self._num_envs:
            raise ValueError(f"need {self._num_envs} seeds, got {len(seeds)}")
        self._games = [
            _Game(int(s), self._reward_mode, self._bandit_episode_len, self._win_ante)
            for s in seeds
        ]
        for game in self._games:
            game.snapshot_antes = self._pool_antes
            game.snapshot_fraction = self._snapshot_fraction
        return self._observe()

    def step(self, actions):
        if self._games is None:
            raise RuntimeError("step() before reset()")
        E.validate_batch(E.ACTION_SPEC, actions, self._num_envs, "actions")
        assert self._last_masks is not None
        rewards = np.zeros(self._num_envs, dtype=np.float32)
        dones = np.zeros(self._num_envs, dtype=np.bool_)
        infos: list[dict] = []
        for i, game in enumerate(self._games):
            game.validate(self._last_masks, i, actions)
            rew, done, info = game.apply(actions, i, self._beta)
            rewards[i] = rew
            dones[i] = done
            infos.append(info)
        obs, masks = self._observe()
        return obs, masks, rewards, dones, infos

    def _observe(self):
        obs = _zeros(E.OBS_SPEC, self._num_envs)
        masks = _zeros(E.MASK_SPEC, self._num_envs)
        assert self._games is not None
        for i, game in enumerate(self._games):
            game.write_obs(obs, i)
            game.write_masks(masks, i)
        E.validate_batch(E.OBS_SPEC, obs, self._num_envs, "obs")
        E.validate_batch(E.MASK_SPEC, masks, self._num_envs, "masks")
        self._last_masks = {k: v.copy() for k, v in masks.items()}
        return obs, masks
