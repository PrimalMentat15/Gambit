#!/usr/bin/env python3
"""Generates the static data tables of sim/core/src/items.rs from
sim/core/tests/data/centers.tsv (produced by tools/dump_centers.lua from the
game's P_CENTERS/P_TAGS tables). Output goes to stdout; items.rs pulls the
result in via include!("items_gen.rs").

Run:  python3 tools/gen_items_rs.py > sim/core/src/items_gen.rs
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(
    (ROOT / "sim/core/tests/data/centers.tsv").open(), delimiter="\t"))


def variant(key: str, prefix: str) -> str:
    stem = key[len(prefix):]
    parts = stem.split("_")
    out = "".join(p.capitalize() for p in parts)
    if out[0].isdigit():
        # j_8_ball -> EightBall
        out = {"8": "Eight"}[out[0]] + out[1:]
    return out


def by_order(rs):
    return sorted(rs, key=lambda r: int(r["order"]))


def num(s):
    return s if s else "0"


jokers = by_order([r for r in rows if r["set"] == "Joker"])
tarots = by_order([r for r in rows if r["set"] == "Tarot"])
planets = by_order([r for r in rows if r["set"] == "Planet"])
spectrals = by_order([r for r in rows if r["set"] == "Spectral"])
vouchers = by_order([r for r in rows if r["set"] == "Voucher"])
boosters = by_order([r for r in rows if r["set"] == "Booster"])
tags = by_order([r for r in rows if r["set"] == "Tag"])
enhanced = by_order([r for r in rows if r["set"] == "Enhanced"])

w = sys.stdout.write

# ---- JokerId enum + metadata ----
w("/// All 150 base-game jokers, in pool order (`center.order` 1..=150;\n")
w("/// game.lua:368-531). Effects land in P3c; P3b needs identity + pool data.\n")
w("#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]\n")
w('#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]\n')
w("#[repr(u8)]\npub enum JokerId {\n")
for r in jokers:
    w(f"    {variant(r['key'], 'j_')},\n")
w("}\n\n")

w("/// Static per-joker data: shop/pool identity only (no effects).\n")
w("#[derive(Debug, Clone, Copy)]\npub struct JokerMeta {\n")
w("    pub id: JokerId,\n    pub key: &'static str,\n    pub name: &'static str,\n")
w("    /// `center.rarity`: 1 common, 2 uncommon, 3 rare, 4 legendary.\n")
w("    pub rarity: u8,\n    pub cost: i64,\n")
w("    /// `center.unlocked` as shipped (fresh profile). The sim's default\n")
w("    /// profile treats everything as unlocked (see `Profile`).\n")
w("    pub unlocked: bool,\n")
w("    /// `center.enhancement_gate` — the joker only enters pools while a\n")
w("    /// playing card with this enhancement exists (common_events.lua:2018-2024).\n")
w("    pub enhancement_gate: Option<&'static str>,\n")
w("    /// `center.no_pool_flag`/`yes_pool_flag` — G.GAME.pool_flags gates\n")
w("    /// (Gros Michel / Cavendish; common_events.lua:2033-2034).\n")
w("    pub no_pool_flag: Option<&'static str>,\n")
w("    pub yes_pool_flag: Option<&'static str>,\n")
w("    /// `center.blueprint_compat` — UI hint only; the actual Blueprint/\n")
w("    /// Brainstorm copy gating happens per effect branch via the\n")
w("    /// `context.blueprint` checks in Card:calculate_joker (see jokers/).\n")
w("    pub blueprint_compat: bool,\n")
w("    /// `center.perishable_compat` / `center.eternal_compat` — sticker\n")
w("    /// eligibility (common_events.lua:2135-2145; Green/Purple Stake).\n")
w("    pub perishable_compat: bool,\n")
w("    pub eternal_compat: bool,\n}\n\n")

w(f"/// `G.P_CENTER_POOLS.Joker` order (sorted by center.order).\n")
w(f"pub const JOKERS: [JokerMeta; {len(jokers)}] = [\n")
for r in jokers:
    unl = "true" if r["unlocked"] == "1" else "false"
    gate = f"Some(\"{r['enhancement_gate']}\")" if r.get("enhancement_gate") else "None"
    nof = f"Some(\"{r['no_pool_flag']}\")" if r.get("no_pool_flag") else "None"
    yesf = f"Some(\"{r['yes_pool_flag']}\")" if r.get("yes_pool_flag") else "None"
    bp = "true" if r["blueprint_compat"] == "1" else "false"
    pc = "true" if r["perishable_compat"] == "1" else "false"
    ec = "true" if r["eternal_compat"] == "1" else "false"
    w(f"    JokerMeta {{ id: JokerId::{variant(r['key'], 'j_')}, "
      f"key: \"{r['key']}\", name: \"{r['name']}\", rarity: {r['rarity']}, "
      f"cost: {r['cost']}, unlocked: {unl}, enhancement_gate: {gate}, "
      f"no_pool_flag: {nof}, yes_pool_flag: {yesf}, "
      f"blueprint_compat: {bp}, perishable_compat: {pc}, eternal_compat: {ec} }},\n")
w("];\n\n")

# ---- Consumables ----
w("/// Static data for one consumable center (Tarot/Planet/Spectral).\n")
w("#[derive(Debug, Clone, Copy)]\npub struct ConsumableMeta {\n")
w("    pub key: &'static str,\n    pub name: &'static str,\n    pub cost: i64,\n")
w("    /// `config.max_highlighted` (0 = untargeted).\n    pub max_highlighted: u8,\n")
w("    /// `config.min_highlighted` (defaults to 1 when targeted).\n    pub min_highlighted: u8,\n")
w("    /// `config.hidden` — The Soul / Black Hole (never in pools).\n    pub hidden: bool,\n}\n\n")

for const, rs, doc in [
    ("TAROTS", tarots, "`G.P_CENTER_POOLS.Tarot` (game.lua:533-554), pool order."),
    ("PLANETS", planets, "`G.P_CENTER_POOLS.Planet` (game.lua:557-568), pool order."),
    ("SPECTRALS", spectrals, "`G.P_CENTER_POOLS.Spectral` (game.lua:571-589), pool order (Soul/Black Hole last, hidden)."),
]:
    w(f"/// {doc}\n")
    w(f"pub const {const}: [ConsumableMeta; {len(rs)}] = [\n")
    for r in rs:
        maxh = num(r["max_highlighted"])
        minh = r["min_highlighted"] or ("1" if maxh != "0" else "0")
        hid = "true" if r["hidden"] == "1" else "false"
        w(f"    ConsumableMeta {{ key: \"{r['key']}\", name: \"{r['name']}\", "
          f"cost: {r['cost']}, max_highlighted: {maxh}, min_highlighted: {minh}, "
          f"hidden: {hid} }},\n")
    w("];\n\n")

# ---- Vouchers ----
w("/// One voucher center. Tier-2 vouchers `requires` their base voucher to\n")
w("/// have been redeemed THIS RUN (get_current_pool, common_events.lua:1994-2001).\n")
w("#[derive(Debug, Clone, Copy)]\npub struct VoucherMeta {\n")
w("    pub key: &'static str,\n    pub name: &'static str,\n    pub cost: i64,\n")
w("    pub unlocked: bool,\n    pub requires: Option<&'static str>,\n}\n\n")
w(f"/// `G.P_CENTER_POOLS.Voucher` (game.lua:592-632), pool order (base/plus interleaved).\n")
w(f"pub const VOUCHERS: [VoucherMeta; {len(vouchers)}] = [\n")
for r in vouchers:
    unl = "true" if r["unlocked"] == "1" else "false"
    req = f"Some(\"{r['requires']}\")" if r["requires"] else "None"
    w(f"    VoucherMeta {{ key: \"{r['key']}\", name: \"{r['name']}\", "
      f"cost: {r['cost']}, unlocked: {unl}, requires: {req} }},\n")
w("];\n\n")

# ---- Boosters ----
w("/// One booster pack center (game.lua:665-698).\n")
w("#[derive(Debug, Clone, Copy)]\npub struct BoosterMeta {\n")
w("    pub key: &'static str,\n    pub name: &'static str,\n")
w("    /// `center.kind`: Arcana/Celestial/Spectral/Standard/Buffoon.\n")
w("    pub kind: PackKind,\n    /// `center.weight` in the get_pack roll.\n    pub weight: f64,\n")
w("    pub cost: i64,\n    /// `config.extra` — number of cards shown.\n    pub cards: u8,\n")
w("    /// `config.choose` — picks allowed.\n    pub choose: u8,\n}\n\n")
w(f"/// `G.P_CENTER_POOLS.Booster` pool order (sorted by center.order).\n")
w(f"pub const BOOSTERS: [BoosterMeta; {len(boosters)}] = [\n")
for r in boosters:
    w(f"    BoosterMeta {{ key: \"{r['key']}\", name: \"{r['name']}\", "
      f"kind: PackKind::{r['kind']}, weight: {float(r['weight'])!r}, cost: {r['cost']}, "
      f"cards: {r['extra']}, choose: {r['choose']} }},\n")
w("];\n\n")

# ---- Tags ----
w("/// One skip-blind tag (game.lua:224-258). `requires` gates the tag on a\n")
w("/// center being discovered in the profile; `min_ante` on the current ante.\n")
w("#[derive(Debug, Clone, Copy)]\npub struct TagMeta {\n")
w("    pub key: &'static str,\n    pub name: &'static str,\n")
w("    pub min_ante: Option<i64>,\n    pub requires: Option<&'static str>,\n}\n\n")
w(f"/// `G.P_CENTER_POOLS.Tag` pool order (sorted by order 1..=24).\n")
w(f"pub const TAGS: [TagMeta; {len(tags)}] = [\n")
for r in tags:
    ma = f"Some({r['min_ante']})" if r["min_ante"] else "None"
    req = f"Some(\"{r['requires']}\")" if r["requires"] else "None"
    w(f"    TagMeta {{ key: \"{r['key']}\", name: \"{r['name']}\", "
      f"min_ante: {ma}, requires: {req} }},\n")
w("];\n\n")

# ---- Enhancements ----
w("/// `G.P_CENTER_POOLS.Enhanced` pool order (m_bonus..m_lucky, order 2..=9);\n")
w("/// used by Standard-pack generation and the Familiar/Grim/Incantation\n")
w("/// `cen_pool` (which drops m_stone).\n")
w(f"pub const ENHANCED_POOL: [&str; {len(enhanced)}] = [\n")
for r in enhanced:
    w(f"    \"{r['key']}\",\n")
w("];\n")
w("// ----------------------- END GENERATED TABLES -----------------------\n")
