//! Consumable *generation* helpers for the blue/purple seal effects. Since
//! P3b the run keeps real consumable inventory (shop.rs / tarots.rs);
//! `purple_seal_tarot` remains here as the standalone, oracle-tested form of
//! the seal's `create_card('Tarot', ..., '8ba')` roll (the run itself uses
//! the shared pool machinery in items.rs, which produces identical draws).
//!
//! Sources: `create_card`/`get_current_pool` (functions/common_events.lua:
//! 1963-2154), `Card:calculate_seal` (card.lua:2242-2269, purple seal),
//! `Card:get_end_of_round_effect` (card.lua:1033-1065, blue seal).

use crate::cards::HandType;
use crate::rng::{LuaRandom, RngState};
use std::collections::HashSet;

/// `G.consumeables.config.card_limit` — `consumable_slots` in
/// `get_starting_params` (misc_functions.lua:1880).
pub const CONSUMABLE_SLOTS: usize = 2;

/// `G.P_CENTER_POOLS.Tarot` in pool order (game.lua:533-554, sorted by the
/// centers' `order` field 1..=22).
pub const TAROT_POOL: [&str; 22] = [
    "c_fool",
    "c_magician",
    "c_high_priestess",
    "c_empress",
    "c_emperor",
    "c_heirophant", // sic — the game's key is misspelled (game.lua:538)
    "c_lovers",
    "c_chariot",
    "c_justice",
    "c_hermit",
    "c_wheel_of_fortune",
    "c_strength",
    "c_hanged_man",
    "c_death",
    "c_temperance",
    "c_devil",
    "c_tower",
    "c_star",
    "c_moon",
    "c_sun",
    "c_judgement",
    "c_world",
];

/// Planet key for a hand type: the `G.P_CENTER_POOLS.Planet` scan in the blue
/// seal effect (card.lua:1047-1053) matches `v.config.hand_type` against
/// `G.GAME.last_hand_played`. Centers from game.lua:557-568.
pub fn planet_for_hand(ht: HandType) -> &'static str {
    match ht {
        HandType::Pair => "c_mercury",
        HandType::ThreeOfAKind => "c_venus",
        HandType::FullHouse => "c_earth",
        HandType::FourOfAKind => "c_mars",
        HandType::Flush => "c_jupiter",
        HandType::Straight => "c_saturn",
        HandType::TwoPair => "c_uranus",
        HandType::StraightFlush => "c_neptune",
        HandType::HighCard => "c_pluto",
        HandType::FiveOfAKind => "c_planet_x",
        HandType::FlushHouse => "c_ceres",
        HandType::FlushFive => "c_eris",
    }
}

/// The purple seal's Tarot generation: `create_card('Tarot', G.consumeables,
/// nil, nil, nil, nil, nil, '8ba')` (card.lua:2260).
///
/// - `soulable` is nil, so the 0.3% Soul poll (common_events.lua:2088-2094)
///   is NOT rolled — no 'soul_Tarot<ante>' stream consumption.
/// - `get_current_pool('Tarot', nil, nil, '8ba')` builds the 22-entry pool in
///   `P_CENTER_POOLS.Tarot` order; a tarot already held this run
///   (`G.GAME.used_jokers[key]`, set by `Card:set_ability` card.lua:349-355,
///   cleared on `Card:remove` card.lua:4741-4749) becomes 'UNAVAILABLE' but
///   keeps its slot, so the pool length stays 22.
/// - Pool key: 'Tarot' .. '8ba' .. ante (common_events.lua:1972+2052).
/// - `pseudorandom_element(_pool, pseudoseed(_pool_key))`: values are plain
///   strings (no sort_id), keys 1..22 sort numerically — array order is
///   preserved, so the draw is `_pool[math.random(22)]`.
/// - While 'UNAVAILABLE' is drawn: redraw with
///   pseudoseed(_pool_key..'_resample'..it), it = 2, 3, ...
///   (common_events.lua:2115-2119).
pub fn purple_seal_tarot(rng: &mut RngState, ante: i64, used_keys: &HashSet<String>) -> String {
    let pool: Vec<&str> = TAROT_POOL
        .iter()
        .map(|&k| {
            if used_keys.contains(k) {
                "UNAVAILABLE"
            } else {
                k
            }
        })
        .collect();
    let pool_key = format!("Tarot8ba{ante}");
    let seed = rng.pseudoseed(&pool_key);
    let idx = LuaRandom::seeded(seed).random_range(1, pool.len() as i64) - 1;
    let mut center = pool[idx as usize];
    let mut it = 1;
    while center == "UNAVAILABLE" {
        it += 1;
        let seed = rng.pseudoseed(&format!("{pool_key}_resample{it}"));
        let idx = LuaRandom::seeded(seed).random_range(1, pool.len() as i64) - 1;
        center = pool[idx as usize];
    }
    center.to_string()
}
