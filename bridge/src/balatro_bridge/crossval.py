"""P5 sim-vs-real-game cross-validation harness.

Drives the SAME seeded run (Red Deck / White Stake) in the Rust sim
(`balatro_sim.CrossvalRun`, direct Run-level interface) and in the live game
(balatrobot JSON-RPC via `BalatroBridgeClient`). A deterministic scripted
policy (seeded random-legal-action, or the sim's greedy bot) chooses actions
from the SIM's legal-action set; each action is applied to the sim first
(sim rejections never touch the game), then mirrored onto the game. After
every action both full states are pulled, normalized to the shared schema
subset, and diffed field-by-field. On divergence a repro bundle is written
to tools/crossval/reports/<seed>_<step>.json and the run stops (unless
--continue-on-divergence).

Usage (in the `balatro` conda env, with `pip install ./sim/py` done):
    balatro-crossval --seeds 1 --policy random --max-antes 2
    balatro-crossval --seeds 5 --policy bot --max-antes 3
    balatro-crossval --dry-run --seeds 3          # sim-vs-sim sanity

Known/expected divergences handled by normalization:
* the run's first shop's guaranteed Buffoon pack picks its art variant with
  an UNSEEDED math.random (common_events.lua:1947) — cosmetic,
  config-identical; the `p_buffoon_normal_[12]` key is collapsed in the
  first shop only;
* absolute card ids (`sort_id`) are session-global in the game — never
  compared; only relative order is;
* SMODS-beta scoring quirk (mod stack, NOT vanilla): when a blind
  (Mouth/Eye/Psychic...) debuffs a played hand, vanilla banks 0 chips, but
  the live smods-1.0.0-beta game banks floor(base_chips*base_mult) of the
  declared hand for every debuffed play AFTER the first one in a round
  (stale SMODS.Scoring_Parameters state). The sim implements the vanilla
  rule; the harness detects the pattern (chips-only diff on a play the sim
  scored as 0), carries the delta as an offset for the rest of the round,
  and records it under "quirks" instead of failing the run.
* SMODS-beta Flint quirk (same family): The Flint's Blind:modify_hand
  halving is dropped by the mod — SMODS's wrapper (overrides.lua:2624)
  writes the halved mult/chips to _G globals, bypassing the
  Scoring_Parameters state its rewritten scoring pipeline reads, so plays
  score UNHALVED live (verified: seed DUGLZE2F step 195, sim 410x140=57400
  vs game 430x146=62780 = exactly the unhalved product). Handled like the
  debuff quirk: chips-only diff during a Flint round with game > sim
  carries the delta. Caveat for both chip quirks: if the scoring gap flips
  whether the blind is CLEARED on that play, sim and game enter different
  states and the run hard-diverges — that is a quirk manifestation, not a
  sim bug.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
from pathlib import Path
from typing import Any

# Comparison happens on these states only; anything else from the game is a
# transient animation state and triggers a re-poll.
SETTLED_STATES = {
    "MENU",
    "BLIND_SELECT",
    "SELECTING_HAND",
    "ROUND_EVAL",
    "SHOP",
    "SMODS_BOOSTER_OPENED",
    "GAME_OVER",
}

HAND_NAMES = [
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush",
    "Full House", "Four of a Kind", "Straight Flush", "Five of a Kind",
    "Flush House", "Flush Five",
]

# Game endpoints exist only for a subset of the sim's action space / states.
# balatrobot: `use`/`sell` require SELECTING_HAND or SHOP; there is no
# boss-reroll or buy-and-use endpoint.
GAME_INVENTORY_STATES = {"SELECTING_HAND", "SHOP"}
UNSUPPORTED_KINDS = {"reroll_boss", "buy_and_use"}


# ---------------------------------------------------------------------------
# Normalization: sim state_json / balatrobot gamestate -> shared schema
# ---------------------------------------------------------------------------


def _lua_table(x: Any) -> dict:
    """Empty Lua tables serialize as [] — coerce to {}."""
    return x if isinstance(x, dict) else {}


def _norm_card_game(c: dict, *, keep_hidden: bool) -> dict:
    mod = _lua_table(c.get("modifier"))
    st = _lua_table(c.get("state"))
    out = {
        "key": c.get("key", ""),
        "enhancement": mod.get("enhancement"),
        "edition": mod.get("edition"),
        "seal": mod.get("seal"),
        "debuff": bool(st.get("debuff", False)),
    }
    if keep_hidden:
        out["hidden"] = bool(st.get("hidden", False))
    return out


def _norm_card_sim(c: dict, *, keep_hidden: bool) -> dict:
    out = {
        "key": c["key"],
        "enhancement": c.get("enhancement"),
        "edition": c.get("edition"),
        "seal": c.get("seal"),
        "debuff": bool(c.get("debuff", False)),
    }
    if keep_hidden:
        out["hidden"] = bool(c.get("hidden", False))
    return out


def _norm_pack_key(key: str, first_shop: bool) -> str:
    if first_shop and key.startswith("p_buffoon_normal_"):
        return "p_buffoon_normal_x"
    return key


def _num(x: Any) -> float | None:
    return None if x is None else float(x)


def normalize_game(gs: dict, *, first_shop: bool) -> dict:
    rd = _lua_table(gs.get("round"))
    out: dict[str, Any] = {
        "state": gs.get("state"),
        "ante": gs.get("ante_num"),
        "round": gs.get("round_num"),
        "money": gs.get("money"),
        "won": bool(gs.get("won", False)),
        "round_info": {
            "hands_left": rd.get("hands_left"),
            "hands_played": rd.get("hands_played"),
            "discards_left": rd.get("discards_left"),
            "discards_used": rd.get("discards_used"),
            "reroll_cost": rd.get("reroll_cost"),
            "chips": _num(rd.get("chips")),
        },
    }

    blinds = _lua_table(gs.get("blinds"))
    out["blinds"] = {
        stage: {
            "name": b.get("name"),
            "score": b.get("score"),
            "status": b.get("status"),
            "tag_name": b.get("tag_name", ""),
        }
        for stage, b in ((s, _lua_table(blinds.get(s))) for s in ("small", "big", "boss"))
    }
    # Boss tags don't exist; the mod leaves them "".
    out["blinds"]["boss"].pop("tag_name", None)

    hand = _lua_table(gs.get("hand"))
    out["hand"] = [_norm_card_game(c, keep_hidden=True) for c in hand.get("cards", [])]
    out["hand_limit"] = hand.get("limit")

    deck = _lua_table(gs.get("cards"))
    out["deck"] = [_norm_card_game(c, keep_hidden=False) for c in deck.get("cards", [])]

    jok = _lua_table(gs.get("jokers"))
    out["jokers"] = [
        {
            "key": c.get("key"),
            "edition": _lua_table(c.get("modifier")).get("edition"),
            "sell": _lua_table(c.get("cost")).get("sell"),
            "debuff": bool(_lua_table(c.get("state")).get("debuff", False)),
        }
        for c in jok.get("cards", [])
    ]
    out["joker_limit"] = jok.get("limit")

    cons = _lua_table(gs.get("consumables"))
    out["consumables"] = [
        {
            "key": c.get("key"),
            "edition": _lua_table(c.get("modifier")).get("edition"),
            "sell": _lua_table(c.get("cost")).get("sell"),
        }
        for c in cons.get("cards", [])
    ]
    out["consumable_limit"] = cons.get("limit")

    hands = _lua_table(gs.get("hands"))
    out["hands_table"] = {
        name: {
            "level": h.get("level"),
            "chips": _num(h.get("chips")),
            "mult": _num(h.get("mult")),
            "played": h.get("played"),
            "played_this_round": h.get("played_this_round"),
        }
        for name, h in ((n, _lua_table(hands.get(n))) for n in HAND_NAMES)
    }

    uv = gs.get("used_vouchers")
    out["used_vouchers"] = sorted(uv.keys()) if isinstance(uv, dict) else []

    if "shop" in gs and gs.get("state") in ("SHOP", "SMODS_BOOSTER_OPENED"):
        out["shop"] = [
            {
                "key": c.get("key"),
                "set": c.get("set"),
                "edition": _lua_table(c.get("modifier")).get("edition"),
                "buy": _lua_table(c.get("cost")).get("buy"),
            }
            for c in _lua_table(gs.get("shop")).get("cards", [])
        ]
        out["shop_vouchers"] = [
            {"key": c.get("key"), "buy": _lua_table(c.get("cost")).get("buy")}
            for c in _lua_table(gs.get("vouchers")).get("cards", [])
        ]
        out["shop_packs"] = [
            {
                "key": _norm_pack_key(c.get("key", ""), first_shop),
                "buy": _lua_table(c.get("cost")).get("buy"),
            }
            for c in _lua_table(gs.get("packs")).get("cards", [])
        ]
    if gs.get("state") == "SMODS_BOOSTER_OPENED":
        # extract_card_modifier reports `ability.effect` ("Hand Upgrade",
        # joker effects, ...) as an "enhancement" for non-playing cards —
        # only playing cards carry a real one.
        out["pack"] = [
            {
                "key": c.get("key"),
                "edition": _lua_table(c.get("modifier")).get("edition"),
                "enhancement": _lua_table(c.get("modifier")).get("enhancement")
                if c.get("set") in ("DEFAULT", "ENHANCED")
                else None,
                "seal": _lua_table(c.get("modifier")).get("seal"),
            }
            for c in _lua_table(gs.get("pack")).get("cards", [])
        ]
    return out


def normalize_sim(sj: dict, *, first_shop: bool) -> dict:
    rd = sj["round"]
    out: dict[str, Any] = {
        "state": sj["state"],
        "ante": sj["ante_num"],
        "round": sj["round_num"],
        "money": sj["money"],
        "won": bool(sj["won"]),
        "round_info": {
            "hands_left": rd["hands_left"],
            "hands_played": rd["hands_played"],
            "discards_left": rd["discards_left"],
            "discards_used": rd["discards_used"],
            "reroll_cost": rd["reroll_cost"],
            "chips": _num(rd["chips"]),
        },
    }
    out["blinds"] = {
        stage: {
            "name": b["name"],
            "score": b["score"],
            "status": b["status"],
            "tag_name": b.get("tag_name", ""),
        }
        for stage, b in ((s, sj["blinds"][s]) for s in ("small", "big", "boss"))
    }
    out["blinds"]["boss"].pop("tag_name", None)

    out["hand"] = [_norm_card_sim(c, keep_hidden=True) for c in sj["hand"]["cards"]]
    out["hand_limit"] = sj["hand"]["limit"]
    out["deck"] = [_norm_card_sim(c, keep_hidden=False) for c in sj["cards"]["cards"]]
    out["jokers"] = [
        {"key": j["key"], "edition": j["edition"], "sell": j["sell"], "debuff": j["debuff"]}
        for j in sj["jokers"]["cards"]
    ]
    out["joker_limit"] = sj["jokers"]["limit"]
    out["consumables"] = [
        {"key": c["key"], "edition": c["edition"], "sell": c["sell"]}
        for c in sj["consumables"]["cards"]
    ]
    out["consumable_limit"] = sj["consumables"]["limit"]
    out["hands_table"] = {
        name: {
            "level": h["level"],
            "chips": _num(h["chips"]),
            "mult": _num(h["mult"]),
            "played": h["played"],
            "played_this_round": h["played_this_round"],
        }
        for name, h in ((n, sj["hands"][n]) for n in HAND_NAMES)
    }
    out["used_vouchers"] = sorted(sj["used_vouchers"])

    if sj["state"] in ("SHOP", "SMODS_BOOSTER_OPENED") and sj.get("shop") is not None:
        out["shop"] = [
            {"key": c["key"], "set": c["set"], "edition": c["edition"], "buy": c["buy"]}
            for c in sj["shop"]["cards"]
        ]
        out["shop_vouchers"] = [
            {"key": c["key"], "buy": c["buy"]} for c in sj["vouchers"]["cards"]
        ]
        out["shop_packs"] = [
            {"key": _norm_pack_key(c["key"], first_shop), "buy": c["buy"]}
            for c in sj["packs"]["cards"]
        ]
    if sj["state"] == "SMODS_BOOSTER_OPENED" and sj.get("pack") is not None:
        out["pack"] = [
            {
                "key": c["key"],
                "edition": c.get("edition"),
                "enhancement": c.get("enhancement"),
                "seal": c.get("seal"),
            }
            for c in sj["pack"]["cards"]
        ]
    return out


def diff_states(sim: Any, game: Any, path: str = "") -> list[dict]:
    """Recursive exact diff (numbers compared as floats — scores/money must
    match exactly; no tolerance)."""
    diffs: list[dict] = []
    if isinstance(sim, dict) and isinstance(game, dict):
        for k in sorted(set(sim) | set(game)):
            p = f"{path}.{k}" if path else str(k)
            if k not in sim:
                diffs.append({"path": p, "sim": "<missing>", "game": game[k]})
            elif k not in game:
                diffs.append({"path": p, "sim": sim[k], "game": "<missing>"})
            else:
                diffs.extend(diff_states(sim[k], game[k], p))
        return diffs
    if isinstance(sim, list) and isinstance(game, list):
        if len(sim) != len(game):
            diffs.append(
                {"path": f"{path}.<len>", "sim": len(sim), "game": len(game)}
            )
        for i, (a, b) in enumerate(zip(sim, game)):
            diffs.extend(diff_states(a, b, f"{path}[{i}]"))
        return diffs
    if isinstance(sim, bool) or isinstance(game, bool):
        if bool(sim) is not bool(game):
            diffs.append({"path": path, "sim": sim, "game": game})
        return diffs
    if isinstance(sim, (int, float)) and isinstance(game, (int, float)):
        if float(sim) != float(game):
            diffs.append({"path": path, "sim": sim, "game": game})
        return diffs
    if sim != game:
        diffs.append({"path": path, "sim": sim, "game": game})
    return diffs


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


class SimDriver:
    def __init__(self, seed: str) -> None:
        import balatro_sim

        self.run = balatro_sim.CrossvalRun(seed)

    def state(self) -> dict:
        return json.loads(self.run.state_json())

    def legal(self) -> list[dict]:
        return json.loads(self.run.legal_actions_json())

    def apply(self, action: dict) -> None:
        self.run.apply_json(json.dumps(action))

    def bot_action(self) -> dict:
        return json.loads(self.run.bot_action_json())

    def is_over(self) -> bool:
        return self.run.is_over()

    def ante(self) -> int:
        return self.run.ante()


class GameDriver:
    """Mirrors direct-action dicts onto balatrobot endpoints."""

    is_sim = False

    def __init__(self, host: str, port: int, timeout: float = 180.0) -> None:
        from balatro_bridge.client import BalatroBridgeClient

        self.client = BalatroBridgeClient(host=host, port=port, timeout=timeout)

    def start(self, seed: str) -> None:
        gs = self.client.gamestate()
        if gs.get("state") != "MENU":
            self.client.menu()
            self._wait_state({"MENU"})
        self.client.start(deck="RED", stake="WHITE", seed=seed)

    def menu(self) -> None:
        try:
            self.client.menu()
        except Exception:
            pass

    def _wait_state(self, states: set[str], timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.client.gamestate().get("state") in states:
                return
            time.sleep(0.5)
        raise TimeoutError(f"game never reached {states}")

    def stable_state(self, timeout: float = 45.0) -> dict:
        """Poll gamestate until it lands on a settled state and THREE
        consecutive reads (~1s window) are identical. Two reads are not
        enough: the cash-out dollar counter ticks with pauses long enough
        to alias a two-read check (observed live: $22 of $24 mid-count)."""
        deadline = time.monotonic() + timeout
        prev: dict | None = None
        stable_reads = 0
        while True:
            gs = self.client.gamestate()
            if gs.get("state") in SETTLED_STATES and prev == gs:
                stable_reads += 1
                if stable_reads >= 2:  # 3 identical reads total
                    return gs
            else:
                stable_reads = 0
                prev = gs if gs.get("state") in SETTLED_STATES else None
            if time.monotonic() >= deadline:
                return gs
            time.sleep(0.45)

    @staticmethod
    def _expected_payout(sim_state: dict) -> int:
        """Upper-bound the round-eval dollar count from the sim's ROUND_EVAL
        state (blind reward + $/remaining hand + interest), to size the
        cash-out animation wait. Overestimates are fine."""
        try:
            blinds = sim_state["blinds"]
            if blinds["boss"]["status"] == "DEFEATED":
                blind_pay = 5
            elif blinds["big"]["status"] == "DEFEATED":
                blind_pay = 4
            else:
                blind_pay = 3
            hands = sim_state["round"]["hands_left"]
            interest = min(sim_state["money"] // 5, 10)
            return blind_pay + hands + interest
        except (KeyError, TypeError):
            return 15

    def apply(self, action: dict, sim_state: dict) -> None:
        kind = action["kind"]
        c = self.client
        if kind == "select_blind":
            c.select()
        elif kind == "skip_blind":
            c.skip()
        elif kind == "play":
            c.play(action["cards"])
        elif kind == "discard":
            c.discard(action["cards"])
        elif kind == "cash_out":
            # Race guard (found live, seed PQD3RI3I step 31): balatrobot
            # presses G.FUNCS.cash_out directly, but the handler pays out
            # G.GAME.current_round.dollars (button_callbacks.lua:2939),
            # which the round-eval UI only stages in the LAST animation
            # event — the same one that creates the button a human would
            # press (common_events.lua:1087). Cash out too early and the
            # game eases in the PREVIOUS round's stale total. gamestate
            # doesn't expose the animation, so wait it out, sized by the
            # payout the sim expects (~0.18s/$ tick plus row/defeat
            # delays; generous because oversleeping is harmless).
            time.sleep(min(8.0, 1.5 + 0.25 * self._expected_payout(sim_state)))
            c.cash_out()
        elif kind == "leave_shop":
            c.next_round()
        elif kind == "reroll":
            c.reroll()
        elif kind == "buy_card":
            c.buy(card=action["slot"])
        elif kind == "redeem_voucher":
            c.buy(voucher=action["slot"])
        elif kind == "buy_pack":
            # Sim pack slots are stable across buys (used=true); the game
            # area drops bought packs — map via the exported unused list.
            packs = sim_state["packs"]["cards"]
            game_idx = next(
                i for i, p in enumerate(packs) if p["slot"] == action["slot"]
            )
            c.buy(pack=game_idx)
        elif kind == "use_consumable":
            c.use(action["slot"], cards=action.get("targets") or None)
        elif kind == "sell_joker":
            c.sell(joker=action["slot"])
        elif kind == "sell_consumable":
            c.sell(consumable=action["slot"])
        elif kind == "pick_pack":
            c.pack(card=action["slot"], targets=action.get("targets") or None)
        elif kind == "skip_pack":
            c.pack(skip=True)
        else:
            raise ValueError(f"unsupported action kind {kind!r}")


class SimAsGameDriver:
    """--dry-run: a second sim instance standing in for the game."""

    is_sim = True

    def __init__(self) -> None:
        self.sim: SimDriver | None = None

    def start(self, seed: str) -> None:
        self.sim = SimDriver(seed)

    def menu(self) -> None:
        pass

    def stable_state(self, timeout: float = 0.0) -> dict:
        assert self.sim is not None
        return self.sim.state()

    def apply(self, action: dict, sim_state: dict) -> None:
        assert self.sim is not None
        a = dict(action)
        if a["kind"] == "buy_pack":
            pass  # sim slots used directly
        self.sim.apply(a)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "select_blind": 8.0,
    "skip_blind": 1.0,
    "play": 6.0,
    "discard": 3.0,
    "cash_out": 10.0,
    "leave_shop": 2.0,
    "buy_card": 3.0,
    "redeem_voucher": 2.0,
    "buy_pack": 3.0,
    "reroll": 1.0,
    "use_consumable": 3.0,
    "sell_joker": 0.5,
    "sell_consumable": 0.5,
    "pick_pack": 5.0,
    "skip_pack": 1.0,
}


def _game_executable(actions: list[dict], state: str) -> list[dict]:
    out = []
    for a in actions:
        if a["kind"] in UNSUPPORTED_KINDS:
            continue
        if (
            a["kind"] in ("use_consumable", "sell_joker", "sell_consumable")
            and state not in GAME_INVENTORY_STATES
        ):
            continue
        out.append(a)
    return out


def random_policy(sim: SimDriver, rng: random.Random) -> dict | None:
    """Pick a random legal action (game-executable subset), apply it to the
    sim, and return it. Sim rejections retry with another candidate."""
    state = json.loads(sim.run.state_json())
    hand_len = len(state["hand"]["cards"])
    for _attempt in range(30):
        candidates = _game_executable(sim.legal(), state["state"])
        if not candidates:
            return None
        weights = [_WEIGHTS.get(a["kind"], 1.0) for a in candidates]
        a = dict(rng.choices(candidates, weights=weights, k=1)[0])
        kind = a["kind"]
        if kind in ("play", "discard"):
            n = rng.randint(1, min(5, max(1, hand_len)))
            picks = rng.sample(range(hand_len), min(n, hand_len))
            forced = a.get("forced")
            if forced is not None and forced not in picks:
                picks[-1] = forced
            a["cards"] = sorted(set(picks))
        elif kind == "use_consumable":
            a["targets"] = _random_targets(a, state, rng)
        elif kind == "pick_pack":
            a["targets"] = _pack_targets(a, state)
        a.pop("forced", None)
        a.pop("key", None)
        a.pop("max_targets", None)
        a.pop("min_targets", None)
        try:
            sim.apply(a)
            return a
        except ValueError:
            continue
    return None


def _random_targets(a: dict, state: dict, rng: random.Random) -> list[int]:
    hand = state["hand"]["cards"]
    maxt, mint = a.get("max_targets", 0), a.get("min_targets", 0)
    if maxt == 0:
        return []
    if a.get("key") == "c_aura":
        cand = [i for i, c in enumerate(hand) if c.get("edition") is None]
        return cand[:1]
    if not hand:
        return []
    k = rng.randint(max(1, mint), min(maxt, len(hand)))
    return sorted(rng.sample(range(len(hand)), k))


def _pack_targets(a: dict, state: dict) -> list[int]:
    # Lowest legal hand indices, mirroring the sim's own PICK_PACK
    # legalization (action.rs).
    hand = state["hand"]["cards"]
    mint = a.get("min_targets", 0)
    if a.get("max_targets", 0) == 0 or not hand:
        return []
    return list(range(min(mint, len(hand))))


def bot_policy(sim: SimDriver, rng: random.Random) -> dict | None:
    a = sim.bot_action()
    sim.apply(a)
    return a


def mixed_policy(sim: SimDriver, rng: random.Random) -> dict | None:
    """Bot with epsilon-random exploration: keeps runs alive deep enough to
    reach shops/packs while still covering off-policy actions."""
    if rng.random() < 0.30:
        return random_policy(sim, rng)
    return bot_policy(sim, rng)


def make_ckpt_policy(ckpt_path: str, device_str: str = "cpu"):
    """A crossval policy driven by a trained PPO checkpoint (argmax).

    Needs `balatro_sim` and the train package installed (both are in the `balatro` conda env). The policy
    reads the sim's encoded obs/masks (CrossvalRun.observe_encoded, batch of
    1 in the frozen env contract), picks the argmax composite action, and
    translates it to a direct action via CrossvalRun.action_json — index
    resolution identical to the vec-env's execution, so the mirrored game
    action is exactly what the policy chose in training terms.
    """
    import torch

    from balatro_train.config import TrainConfig, _build
    from balatro_train.policy import (
        BalatroPolicy,
        actions_to_numpy,
        obs_to_torch,
    )

    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pcfg = _build(TrainConfig, ckpt["config"], "ckpt").policy
    policy_net = BalatroPolicy(pcfg).to(device)
    policy_net.load_state_dict(ckpt["policy"])
    policy_net.eval()
    print(f"ckpt policy: {ckpt_path} (step {ckpt['global_step']})", flush=True)

    from balatro_train.encoding import ActionType

    inventory_types = (
        int(ActionType.USE_CONSUMABLE),
        int(ActionType.SELL_JOKER),
        int(ActionType.SELL_CONSUMABLE),
    )

    def ckpt_policy(sim: SimDriver, rng: random.Random) -> dict | None:
        obs, masks = sim.run.observe_encoded()
        # balatrobot's use/sell methods only work in SELECTING_HAND / SHOP
        # (the sim and the real game UI allow more states) — mask those
        # types elsewhere so every chosen action is game-mirrorable, same
        # constraint _game_executable puts on the scripted policies.
        if sim.run.state() not in GAME_INVENTORY_STATES:
            for t in inventory_types:
                masks["action_type_mask"][0][t] = False
        with torch.no_grad():
            actions, *_ = policy_net.act(
                obs_to_torch(obs, device),
                obs_to_torch(masks, device),
                deterministic=True,
            )
        a = actions_to_numpy(actions)
        n = int(a["n_cards"][0])
        j = sim.run.action_json(
            int(a["action_type"][0]),
            [int(x) for x in a["cards"][0][:n]],
            int(a["joker_target"][0]),
            int(a["consumable_target"][0]),
            int(a["shop_target"][0]),
            int(a["pack_target"][0]),
        )
        act = json.loads(j)
        if act is None:
            # USE_CONSUMABLE no-op leniency: the vec-env would waste the
            # step; argmax would loop forever, so defer to the bot instead.
            print("  [ckpt] no-op consumable pick; bot fallback", flush=True)
            return bot_policy(sim, rng)
        sim.apply(act)
        return act

    return ckpt_policy


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def make_seeds(n: int) -> list[str]:
    rng = random.Random("balatro-p5-crossval")
    alphabet = string.digits + string.ascii_uppercase
    return ["".join(rng.choices(alphabet, k=8)) for _ in range(n)]


def run_one(
    seed: str,
    game: GameDriver | SimAsGameDriver,
    policy_name: str,
    max_antes: int,
    max_steps: int,
    reports_dir: Path,
    continue_on_divergence: bool,
    policy_fn=None,
) -> dict:
    sim = SimDriver(seed)
    game.start(seed)
    policy = policy_fn or {
        "bot": bot_policy, "random": random_policy, "mixed": mixed_policy,
    }[policy_name]
    rng = random.Random(f"{seed}:{policy_name}")
    action_log: list[dict] = []
    quirks: list[dict] = []
    shops_seen = 0
    divergences = 0
    step = 0
    chips_offset = 0.0  # SMODS debuffed-hand quirk (module docstring)
    # `quirks` is shared by reference: appends show up in every return path.
    result: dict[str, Any] = {"seed": seed, "policy": policy_name, "quirks": quirks}

    def compare(action: dict | None, sim_chips_before: float | None = None) -> list[dict]:
        nonlocal divergences, chips_offset
        sim_state = sim.state()
        game_state = game.stable_state()
        first_shop = shops_seen <= 1
        n_sim = normalize_sim(sim_state, first_shop=first_shop)
        norm_other = normalize_sim if game.is_sim else normalize_game
        n_game = norm_other(game_state, first_shop=first_shop)
        if n_sim["state"] == "GAME_OVER" and n_game["state"] == "GAME_OVER":
            return []  # game pauses mid-animation at GAME_OVER; state match suffices
        if (
            n_sim["state"] == "ROUND_EVAL"
            and sim_state["blinds"]["boss"]["name"] == "Crimson Heart"
        ):
            # Crimson Heart re-debuffs a random joker every hand; on boss
            # defeat the sim clears it synchronously but the game clears it
            # in the DELAYED Blind:defeat animation event, which can land
            # after stable_state's window (seen live at a winning step,
            # seed V17H810I step 248 — the flag was already gone one poll
            # later). Timing transient, not logic: ignore the flag here.
            for j in n_sim["jokers"] + n_game["jokers"]:
                if isinstance(j, dict):
                    j["debuff"] = None
        if chips_offset and n_game["round_info"]["chips"] is not None:
            n_game["round_info"]["chips"] -= chips_offset
        diffs = diff_states(n_sim, n_game)
        if (
            diffs
            and all(d["path"] == "round_info.chips" for d in diffs)
            and action is not None
            and action.get("kind") == "play"
            and sim_chips_before is not None
            and n_sim["round_info"]["chips"] == sim_chips_before
        ):
            # Sim scored the play as 0 (blind-debuffed hand): the live
            # modded game banks the declared hand's base score instead —
            # known SMODS-beta quirk. Carry the delta, don't fail.
            delta = float(n_game["round_info"]["chips"]) - float(
                n_sim["round_info"]["chips"]
            )
            chips_offset += delta
            quirks.append(
                {
                    "step": step,
                    "kind": "smods_debuffed_hand_scores_base",
                    "delta": delta,
                }
            )
            print(f"  quirk step {step}: smods debuffed-hand +{delta} chips "
                  f"(offset {chips_offset})", flush=True)
            return []
        if (
            diffs
            and all(d["path"] == "round_info.chips" for d in diffs)
            and action is not None
            and action.get("kind") == "play"
            and sim_state["blinds"]["boss"]["name"] == "The Flint"
            and float(n_game["round_info"]["chips"] or 0)
            > float(n_sim["round_info"]["chips"] or 0)
        ):
            # The Flint round, game banked MORE than the sim on a play:
            # smods-1.0.0-beta drops Blind:modify_hand's halving — its
            # wrapper (overrides.lua:2624) writes the halved values to the
            # _G.mult/_G.hand_chips globals, bypassing the Scoring_Parameters
            # state its rewritten pipeline actually reads, so the game
            # scores the play UNHALVED (verified live, seed DUGLZE2F step
            # 195: sim 410x140=57400 halved vs game 430x146=62780
            # unhalved, exact). Vanilla halves every play; the sim is
            # vanilla-faithful. Carry the delta as a quirk, don't fail.
            delta = float(n_game["round_info"]["chips"]) - float(
                n_sim["round_info"]["chips"]
            )
            chips_offset += delta
            quirks.append(
                {
                    "step": step,
                    "kind": "smods_flint_unhalved_play",
                    "delta": delta,
                }
            )
            print(f"  quirk step {step}: smods Flint-unhalved +{delta} chips "
                  f"(offset {chips_offset})", flush=True)
            return []
        if diffs:
            divergences += 1
            reports_dir.mkdir(parents=True, exist_ok=True)
            bundle = {
                "seed": seed,
                "policy": policy_name,
                "step": step,
                "action": action,
                "first_divergent_field": diffs[0],
                "diffs": diffs[:80],
                "action_log": action_log,
                "normalized_sim": n_sim,
                "normalized_game": n_game,
                "sim_state": sim_state,
                "game_state": game_state,
            }
            path = reports_dir / f"{seed}_{step:04d}.json"
            path.write_text(json.dumps(bundle, indent=1))
            print(f"  DIVERGENCE step {step}: {diffs[0]['path']} "
                  f"sim={diffs[0]['sim']!r} game={diffs[0]['game']!r} -> {path.name}",
                  flush=True)
        return diffs

    if compare(None):
        result.update(status="diverged", steps=0, ante=sim.ante())
        return result

    while step < max_steps:
        if sim.is_over():
            result.update(status="terminal", steps=step, ante=sim.ante(),
                          divergences=divergences)
            return result
        if sim.ante() > max_antes:
            result.update(status="ante_limit", steps=step, ante=sim.ante(),
                          divergences=divergences)
            return result
        sim_state_before = sim.state()
        action = policy(sim, rng)
        if action is None:
            result.update(status="no_action", steps=step, ante=sim.ante())
            return result
        step += 1
        action_log.append(action)
        try:
            game.apply(action, sim_state_before)
        except Exception as e:  # noqa: BLE001 — bundle any game-side error
            reports_dir.mkdir(parents=True, exist_ok=True)
            bundle = {
                "seed": seed, "policy": policy_name, "step": step,
                "action": action, "game_error": repr(e),
                "action_log": action_log, "sim_state": sim.state(),
            }
            (reports_dir / f"{seed}_{step:04d}_gameerror.json").write_text(
                json.dumps(bundle, indent=1)
            )
            print(f"  GAME ERROR step {step} on {action}: {e}", flush=True)
            result.update(status="game_error", steps=step, ante=sim.ante())
            return result
        if action["kind"] == "cash_out":
            shops_seen += 1
        if action["kind"] in ("cash_out", "select_blind"):
            chips_offset = 0.0
        sim_chips_before = sim_state_before["round"]["chips"]
        if compare(action, sim_chips_before) and not continue_on_divergence:
            result.update(status="diverged", steps=step, ante=sim.ante())
            return result
        if step % 10 == 0:
            print(f"  step {step} ante {sim.ante()} state {sim.state()['state']}",
                  flush=True)

    result.update(
        status="step_limit" if not divergences else "diverged",
        steps=step,
        ante=sim.ante(),
        divergences=divergences,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds to run")
    ap.add_argument("--seed-list", type=str, default=None,
                    help="comma-separated explicit seeds (overrides --seeds)")
    ap.add_argument("--policy", choices=["random", "bot", "mixed", "ckpt"],
                    default="random")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="checkpoint .pt for --policy ckpt (demo extra)")
    ap.add_argument("--device", type=str, default="cpu",
                    help="torch device for --policy ckpt")
    ap.add_argument("--max-antes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--continue-on-divergence", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="sim-vs-sim (no game needed); must diff clean")
    ap.add_argument("--redo", action="store_true",
                    help="rerun seeds already recorded in progress.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=12346)
    ap.add_argument("--reports-dir", type=Path, default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    reports_dir = args.reports_dir or repo / "tools" / "crossval" / "reports"
    progress_path = reports_dir / "progress.json"
    progress: dict[str, Any] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())

    seeds = (
        [s.strip().upper() for s in args.seed_list.split(",") if s.strip()]
        if args.seed_list
        else make_seeds(args.seeds)
    )

    policy_fn = None
    if args.policy == "ckpt":
        if not args.ckpt:
            ap.error("--policy ckpt requires --ckpt PATH")
        policy_fn = make_ckpt_policy(args.ckpt, args.device)

    if args.dry_run:
        game: GameDriver | SimAsGameDriver = SimAsGameDriver()
    else:
        game = GameDriver(args.host, args.port)
        game.client.health()

    ok = True
    for seed in seeds:
        key = f"{seed}:{args.policy}:{'dry' if args.dry_run else 'live'}"
        prev = progress.get(key)
        if prev and not args.redo and prev.get("status") not in ("game_error",):
            print(f"[{seed}] skipped (already {prev['status']}, "
                  f"{prev.get('steps', 0)} steps)")
            continue
        print(f"[{seed}] policy={args.policy} max_antes={args.max_antes}", flush=True)
        t0 = time.monotonic()
        res = run_one(
            seed, game, args.policy, args.max_antes, args.max_steps,
            reports_dir, args.continue_on_divergence, policy_fn,
        )
        res["wall_s"] = round(time.monotonic() - t0, 1)
        print(f"[{seed}] {res['status']} after {res.get('steps', 0)} steps "
              f"(ante {res.get('ante')}, {res['wall_s']}s)", flush=True)
        progress[key] = res
        reports_dir.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(progress, indent=1))
        if res["status"] in ("diverged", "game_error"):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
