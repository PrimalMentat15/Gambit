//! The item registry (jokers, consumables, vouchers, tags, booster packs)
//! plus the shared pool machinery of `get_current_pool`/`create_card`
//! (functions/common_events.lua:1963-2154), `poll_edition`
//! (common_events.lua:2055-2080) and `get_pack` (common_events.lua:1944-1961).
//!
//! Static tables are generated from the game's `P_CENTERS`/`P_TAGS`
//! (game.lua:216-743) via tools/dump_centers.lua + tools/gen_items_rs.py and
//! cross-checked against tests/data/centers.tsv by the items_registry test.
//!
//! # Profile assumption
//! The sim models a fully unlocked + fully discovered profile (the seed-tool
//! convention, and what P5 cross-validation will pin the live game to):
//! `get_current_pool`'s `v.unlocked ~= false` check always passes and the Tag
//! pool's `requires`-discovered check always passes. The shipped
//! `unlocked: false` flags are kept in the metadata for a future
//! fresh-profile mode.

use crate::cards::{Card, Edition, Enhancement, HandType, Rank, Suit};
use crate::rng::{LuaRandom, RngState};
use crate::scoring::HandsTable;
use std::collections::{HashMap, HashSet};

/// Alias for interned registry keys. Exists (rather than spelling
/// `&'static str`) so serde_derive's syntactic lifetime analysis does not
/// force `'de: 'static` bounds on snapshot-serializable containers (P6).
pub type Key = &'static str;

/// `center.kind` of the booster packs (game.lua:665-698).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PackKind {
    Arcana,
    Celestial,
    Spectral,
    Standard,
    Buffoon,
}

/// Which consumable pool a center belongs to (`center.set`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsumableSet {
    Tarot,
    Planet,
    Spectral,
}

impl ConsumableSet {
    /// The `_type` string used in RNG keys ('soul_'.._type, pool keys).
    pub fn type_str(self) -> &'static str {
        match self {
            ConsumableSet::Tarot => "Tarot",
            ConsumableSet::Planet => "Planet",
            ConsumableSet::Spectral => "Spectral",
        }
    }
}

// ------------------------- GENERATED TABLES -------------------------
// (see module docs; inserted by tools/gen_items_rs.py)
include!("items_gen.rs");
// ----------------------- END GENERATED TABLES -----------------------

impl JokerId {
    /// Metadata row; `JokerId` discriminants equal the pool index (JOKERS is
    /// sorted by `center.order` and the enum is generated in the same order).
    pub fn meta(self) -> &'static JokerMeta {
        &JOKERS[self as usize]
    }

    pub fn key(self) -> &'static str {
        self.meta().key
    }

    pub fn name(self) -> &'static str {
        self.meta().name
    }
}

pub fn joker_by_key(key: &str) -> Option<&'static JokerMeta> {
    JOKERS.iter().find(|m| m.key == key)
}

pub fn voucher_by_key(key: &str) -> Option<&'static VoucherMeta> {
    VOUCHERS.iter().find(|m| m.key == key)
}

pub fn tag_by_key(key: &str) -> Option<&'static TagMeta> {
    TAGS.iter().find(|m| m.key == key)
}

pub fn booster_by_key(key: &str) -> Option<&'static BoosterMeta> {
    BOOSTERS.iter().find(|m| m.key == key)
}

pub fn consumable_by_key(key: &str) -> Option<(&'static ConsumableMeta, ConsumableSet)> {
    if let Some(m) = TAROTS.iter().find(|m| m.key == key) {
        return Some((m, ConsumableSet::Tarot));
    }
    if let Some(m) = PLANETS.iter().find(|m| m.key == key) {
        return Some((m, ConsumableSet::Planet));
    }
    if let Some(m) = SPECTRALS.iter().find(|m| m.key == key) {
        return Some((m, ConsumableSet::Spectral));
    }
    None
}

/// `center.cost` for any purchasable center; playing-card centers (c_base and
/// the m_* enhancements) have no `cost`, and `Card:set_ability` falls back to
/// `center.cost or 1` (card.lua:339).
pub fn center_base_cost(key: &str) -> i64 {
    if let Some(m) = joker_by_key(key) {
        return m.cost;
    }
    if let Some((m, _)) = consumable_by_key(key) {
        return m.cost;
    }
    if let Some(m) = voucher_by_key(key) {
        return m.cost;
    }
    if let Some(m) = booster_by_key(key) {
        return m.cost;
    }
    1
}

/// `Card:set_cost` edition surcharge (card.lua:370-373): holo +3, foil +2,
/// polychrome +5, negative +5.
pub fn edition_extra_cost(edition: Edition) -> i64 {
    match edition {
        Edition::Holo => 3,
        Edition::Foil => 2,
        Edition::Polychrome | Edition::Negative => 5,
        Edition::None => 0,
    }
}

/// `Card:set_cost` (card.lua:369-385) for an item outside challenge modifiers:
/// `max(1, floor((base_cost + extra_cost + 0.5) * (100-discount)/100))`,
/// zeroed by the coupon flag while the card sits in a shop area.
pub fn item_cost(base_cost: i64, edition: Edition, discount_percent: i64, couponed: bool) -> i64 {
    let cost = ((base_cost + edition_extra_cost(edition)) as f64 + 0.5)
        * ((100 - discount_percent) as f64)
        / 100.0;
    let cost = (cost.floor() as i64).max(1);
    if couponed {
        0
    } else {
        cost
    }
}

/// `sell_cost = math.max(1, math.floor(cost/2)) + extra_value`
/// (card.lua:383). `cost` here is the owned-area cost (coupon no longer
/// applies once the card left the shop).
pub fn sell_cost(base_cost: i64, edition: Edition, discount_percent: i64, extra_value: i64) -> i64 {
    let cost = item_cost(base_cost, edition, discount_percent, false);
    (cost / 2).max(1) + extra_value
}

/// `planet_for_hand`'s inverse: the `config.hand_type` of a planet center
/// (game.lua:557-568), for the Planet-pool softlock check.
pub fn hand_for_planet(key: &str) -> Option<HandType> {
    Some(match key {
        "c_mercury" => HandType::Pair,
        "c_venus" => HandType::ThreeOfAKind,
        "c_earth" => HandType::FullHouse,
        "c_mars" => HandType::FourOfAKind,
        "c_jupiter" => HandType::Flush,
        "c_saturn" => HandType::Straight,
        "c_uranus" => HandType::TwoPair,
        "c_neptune" => HandType::StraightFlush,
        "c_pluto" => HandType::HighCard,
        "c_planet_x" => HandType::FiveOfAKind,
        "c_ceres" => HandType::FlushHouse,
        "c_eris" => HandType::FlushFive,
        _ => return None,
    })
}

/// Planets whose `config.softlock` gates them on the hand having been played
/// (game.lua:565-567; get_current_pool common_events.lua:2013-2016).
pub fn planet_softlocked(key: &str) -> bool {
    matches!(key, "c_planet_x" | "c_ceres" | "c_eris")
}

/// Enhancement center key -> our enum (game.lua:648-655).
pub fn enhancement_from_key(key: &str) -> Enhancement {
    match key {
        "m_bonus" => Enhancement::Bonus,
        "m_mult" => Enhancement::Mult,
        "m_wild" => Enhancement::Wild,
        "m_glass" => Enhancement::Glass,
        "m_steel" => Enhancement::Steel,
        "m_stone" => Enhancement::Stone,
        "m_gold" => Enhancement::Gold,
        "m_lucky" => Enhancement::Lucky,
        _ => Enhancement::None,
    }
}

/// `G.handlist` (globals.lua:487-500): hand names top-down, used by the
/// Telescope scan in Celestial packs (card.lua:1706-1713).
pub const HANDLIST: [HandType; 12] = [
    HandType::FlushFive,
    HandType::FlushHouse,
    HandType::FiveOfAKind,
    HandType::StraightFlush,
    HandType::FourOfAKind,
    HandType::FullHouse,
    HandType::Flush,
    HandType::Straight,
    HandType::ThreeOfAKind,
    HandType::TwoPair,
    HandType::Pair,
    HandType::HighCard,
];

/// The 52 `G.P_CARDS` keys in `pseudorandom_element` iteration order: the
/// keys are plain strings, so `pseudorandom_element` sorts them
/// byte-lexicographically (misc_functions.lua:262) — "C_2" < ... < "S_T",
/// with rank chars ordered '2'..'9' < 'A' < 'J' < 'K' < 'Q' < 'T'.
pub fn card_front_from_index(idx: usize) -> (Suit, Rank) {
    const SUITS: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];
    const RANKS: [u8; 13] = [2, 3, 4, 5, 6, 7, 8, 9, 14, 11, 13, 12, 10];
    (SUITS[idx / 13], Rank(RANKS[idx % 13]))
}

// ---------------------------------------------------------------------------
// pseudorandom_element / pool draws
// ---------------------------------------------------------------------------

/// `pseudorandom_element(_t, pseudoseed(key))` over an array of strings: the
/// sort keys are the numeric indices 1..n, so array order is preserved and
/// the draw is `_t[math.random(n)]` (misc_functions.lua:253-268). Returns the
/// 0-based index.
pub fn element_index(rng: &mut RngState, key: &str, len: usize) -> usize {
    let seed = rng.pseudoseed(key);
    (LuaRandom::seeded(seed).random_range(1, len as i64) - 1) as usize
}

/// The 'UNAVAILABLE' + `_resample` loop shared by `create_card`,
/// `get_next_voucher_key` and `get_next_tag_key`
/// (common_events.lua:2113-2119 / :1904-1910 / :1917-1923).
pub fn pool_draw(rng: &mut RngState, pool: &[&'static str], pool_key: &str) -> &'static str {
    let mut center = pool[element_index(rng, pool_key, pool.len())];
    let mut it = 1;
    while center == "UNAVAILABLE" {
        it += 1;
        let key = format!("{pool_key}_resample{it}");
        center = pool[element_index(rng, &key, pool.len())];
    }
    center
}

/// Everything `get_current_pool`'s culling reads from the run.
pub struct PoolArgs<'a> {
    /// `G.GAME.round_resets.ante`.
    pub ante: i64,
    /// `G.GAME.used_jokers` — centers currently instantiated this run,
    /// refcounted by the caller (card.lua:349-355 / 4741-4749).
    pub used_keys: &'a HashMap<String, u32>,
    /// `G.GAME.used_vouchers` — vouchers redeemed this run.
    pub used_vouchers: &'a HashSet<&'static str>,
    /// Keys of vouchers currently displayed in `G.shop_vouchers`
    /// (common_events.lua:2003-2007).
    pub shop_vouchers: &'a [&'static str],
    /// `G.GAME.hands` for the Planet softlock check; None only in bare tests.
    pub hands: Option<&'a HandsTable>,
    /// `G.playing_cards` split by area (deck/hand/discard) for the
    /// enhancement_gate scan (common_events.lua:2018-2024); destroyed cards
    /// are gone from `G.playing_cards`.
    pub playing_cards: [&'a [Card]; 3],
    /// `G.GAME.pool_flags` (gros_michel_extinct — set by the Gros Michel
    /// self-destruct).
    pub pool_flags: &'a HashSet<&'static str>,
    /// `next(find_joker("Showman"))` — while a non-debuffed Showman is
    /// owned, the `used_jokers` culling is bypassed for the whole
    /// non-Enhanced/Tag branch (common_events.lua:1988): jokers,
    /// consumables AND the voucher `used_jokers` (displayed-voucher) check
    /// — duplicates re-enter every pool.
    pub showman: bool,
}

impl PoolArgs<'_> {
    fn used(&self, key: &str) -> bool {
        // `not (G.GAME.used_jokers[v.key] and not next(find_joker("Showman")))`.
        !self.showman && self.used_keys.get(key).copied().unwrap_or(0) > 0
    }

    /// `for kk, vv in pairs(G.playing_cards) do if vv.config.center.key ==
    /// v.enhancement_gate` — any owned playing card with the enhancement.
    fn has_enhancement(&self, gate: &str) -> bool {
        let e = enhancement_from_key(gate);
        self.playing_cards
            .iter()
            .any(|cards| cards.iter().any(|c| c.enhancement == e))
    }
}

/// Which pool to build (`_type`/`_rarity`/`_legendary`/`_append` of
/// get_current_pool).
pub enum PoolSpec<'a> {
    /// `_rarity`: a forced rarity float (Wraith 0.99, tags 0.9/1/0) or None
    /// to roll `pseudorandom('rarity'..ante..append)`.
    Joker {
        rarity: Option<f64>,
        legendary: bool,
        append: &'a str,
    },
    Consumable {
        set: ConsumableSet,
        append: &'a str,
    },
    Voucher,
    Tag,
    Enhanced {
        append: &'a str,
    },
}

/// `get_current_pool` (common_events.lua:1963-2053). Returns the pool (with
/// 'UNAVAILABLE' placeholders, exact length) and the final `_pool_key`
/// (already suffixed with the ante except for legendary jokers).
pub fn get_current_pool(
    rng: &mut RngState,
    spec: &PoolSpec,
    args: &PoolArgs,
) -> (Vec<&'static str>, String) {
    let mut pool: Vec<&'static str> = Vec::new();
    let mut pool_size = 0usize;
    let mut push = |pool: &mut Vec<&'static str>, key: &'static str, add: bool| {
        if add {
            pool.push(key);
            pool_size += 1;
        } else {
            pool.push("UNAVAILABLE");
        }
    };

    let (pool_key, fallback, legendary): (String, &'static str, bool) = match spec {
        PoolSpec::Joker {
            rarity,
            legendary,
            append,
        } => {
            // rarity roll happens even when the pool ends up empty
            // (common_events.lua:1969-1970).
            let r = if *legendary {
                4u8
            } else {
                let poll = match rarity {
                    Some(f) => *f,
                    None => rng.random(&format!("rarity{}{}", args.ante, append)),
                };
                if poll > 0.95 {
                    3
                } else if poll > 0.7 {
                    2
                } else {
                    1
                }
            };
            for m in JOKERS.iter().filter(|m| m.rarity == r) {
                // `not (used_jokers[key] and not Showman)` and
                // `(v.unlocked ~= false or v.rarity == 4)` — the sim models a
                // fully unlocked profile, so the unlock check always passes.
                // enhancement_gate jokers (Steel/Stone/Glass Joker, Lucky
                // Cat, Golden Ticket) need a matching enhanced playing card
                // (common_events.lua:2018-2024).
                let mut add = !args.used(m.key);
                if add {
                    if let Some(gate) = m.enhancement_gate {
                        add = args.has_enhancement(gate);
                    }
                }
                // `v.no_pool_flag and pool_flags[..] -> nil`,
                // `v.yes_pool_flag and not pool_flags[..] -> nil`
                // (common_events.lua:2033-2034).
                if let Some(f) = m.no_pool_flag {
                    if args.pool_flags.contains(f) {
                        add = false;
                    }
                }
                if let Some(f) = m.yes_pool_flag {
                    if !args.pool_flags.contains(f) {
                        add = false;
                    }
                }
                push(&mut pool, m.key, add);
            }
            let append = if *legendary { "" } else { append };
            (format!("Joker{r}{append}"), "j_joker", *legendary)
        }
        PoolSpec::Consumable { set, append } => {
            match set {
                ConsumableSet::Tarot => {
                    for m in TAROTS.iter() {
                        push(&mut pool, m.key, !args.used(m.key));
                    }
                }
                ConsumableSet::Planet => {
                    for m in PLANETS.iter() {
                        // `not v.config.softlock or hands[hand_type].played > 0`
                        // (common_events.lua:2013-2016).
                        let mut add = !args.used(m.key);
                        if add && planet_softlocked(m.key) {
                            let ht = hand_for_planet(m.key).unwrap();
                            add = args.hands.map(|h| h.get(ht).played > 0).unwrap_or(false);
                        }
                        push(&mut pool, m.key, add);
                    }
                }
                ConsumableSet::Spectral => {
                    for m in SPECTRALS.iter() {
                        // Black Hole / The Soul: add = false
                        // (common_events.lua:2029-2031).
                        let add = !m.hidden && !args.used(m.key);
                        push(&mut pool, m.key, add);
                    }
                }
            }
            let fb = match set {
                ConsumableSet::Tarot => "c_strength",
                ConsumableSet::Planet => "c_pluto",
                ConsumableSet::Spectral => "c_incantation",
            };
            (format!("{}{}", set.type_str(), append), fb, false)
        }
        PoolSpec::Voucher => {
            for m in VOUCHERS.iter() {
                // common_events.lua:1993-2011: unredeemed, `requires`
                // redeemed this run, and not currently displayed.
                let mut add = !args.used(m.key) && !args.used_vouchers.contains(m.key);
                if add {
                    if let Some(req) = m.requires {
                        add = args.used_vouchers.contains(req);
                    }
                }
                if add && args.shop_vouchers.contains(&m.key) {
                    add = false;
                }
                push(&mut pool, m.key, add);
            }
            ("Voucher".to_string(), "v_blank", false)
        }
        PoolSpec::Tag => {
            for m in TAGS.iter() {
                // `(not v.requires or requires discovered)` — always true on
                // the modeled fully-discovered profile — `and (not min_ante
                // or min_ante <= ante)` (common_events.lua:1990-1993).
                let add = m.min_ante.map(|a| a <= args.ante).unwrap_or(true);
                push(&mut pool, m.key, add);
            }
            ("Tag".to_string(), "tag_handy", false)
        }
        PoolSpec::Enhanced { append } => {
            for k in ENHANCED_POOL.iter() {
                push(&mut pool, k, true); // add = true unconditionally (:1977)
            }
            (format!("Enhanced{append}"), "m_bonus", false)
        }
    };

    // Empty-pool fallback replaces the whole pool with one entry
    // (common_events.lua:2037-2049); the draw still consumes a seed.
    if pool_size == 0 {
        pool = vec![fallback];
    }
    // `_pool_key..(not _legendary and G.GAME.round_resets.ante or '')`
    let pool_key = if legendary {
        pool_key
    } else {
        format!("{pool_key}{}", args.ante)
    };
    (pool, pool_key)
}

/// Convenience: full `get_current_pool` + resample draw, as done by
/// `create_card`/`get_next_voucher_key`/`get_next_tag_key`.
pub fn roll_from_pool(rng: &mut RngState, spec: &PoolSpec, args: &PoolArgs) -> &'static str {
    let (pool, pool_key) = get_current_pool(rng, spec, args);
    pool_draw(rng, &pool, &pool_key)
}

// ---------------------------------------------------------------------------
// poll_edition / get_pack
// ---------------------------------------------------------------------------

/// `poll_edition(_key, _mod, _no_neg, _guaranteed)`
/// (common_events.lua:2055-2080). `edition_rate` is `G.GAME.edition_rate`
/// (1 base; Hone 2, Glow Up 4). Consumes exactly one draw on `key`'s stream.
#[allow(clippy::eq_op)] // 1.0 - 0.04*25.0 spells out the Lua literally
pub fn poll_edition(
    rng: &mut RngState,
    key: &str,
    mod_: f64,
    no_neg: bool,
    guaranteed: bool,
    edition_rate: f64,
) -> Edition {
    let poll = rng.random(key);
    if guaranteed {
        if poll > 1.0 - 0.003 * 25.0 && !no_neg {
            Edition::Negative
        } else if poll > 1.0 - 0.006 * 25.0 {
            Edition::Polychrome
        } else if poll > 1.0 - 0.02 * 25.0 {
            Edition::Holo
        } else if poll > 1.0 - 0.04 * 25.0 {
            Edition::Foil
        } else {
            Edition::None
        }
    } else if poll > 1.0 - 0.003 * mod_ && !no_neg {
        Edition::Negative
    } else if poll > 1.0 - 0.006 * edition_rate * mod_ {
        Edition::Polychrome
    } else if poll > 1.0 - 0.02 * edition_rate * mod_ {
        Edition::Holo
    } else if poll > 1.0 - 0.04 * edition_rate * mod_ {
        Edition::Foil
    } else {
        Edition::None
    }
}

/// The weighted booster roll of `get_pack(_key, _type)`
/// (common_events.lua:1949-1960), untyped (`_type == nil`) and without
/// banned keys. The first-shop Buffoon shortcut (:1945-1948) is the caller's
/// job. Consumes one draw on `'{key}{ante}'`.
pub fn get_pack_roll(rng: &mut RngState, key: &str, ante: i64) -> &'static BoosterMeta {
    let cume: f64 = BOOSTERS.iter().map(|b| b.weight).sum();
    let poll = rng.random(&format!("{key}{ante}")) * cume;
    let mut it = 0.0f64;
    for b in BOOSTERS.iter() {
        it += b.weight;
        // `if it >= poll and it - v.weight <= poll then center = v; break`
        if it >= poll && it - b.weight <= poll {
            return b;
        }
    }
    // Unreachable for in-range polls; mirror Lua's nil-safety with the last
    // entry (poll == cume lands on the final iteration above).
    BOOSTERS.last().unwrap()
}
