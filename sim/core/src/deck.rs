//! Deck construction and shuffling, ported from Game:start_run (game.lua)
//! and `pseudoshuffle` (functions/misc_functions.lua:206).
//!
//! # Creation order (== sort_id order)
//! `Game:start_run` (game.lua:2338-2373) builds `card_protos` from
//! `pairs(self.P_CARDS)` — an arbitrary hash order — but then sorts the protos
//! by the concatenated string `s..r` ("C2", "DA", "ST", ...; game.lua:2367).
//! Lua string comparison is byte-wise, so the resulting creation order is:
//!   suits  C < D < H < S
//!   ranks  '2'<'3'<'4'<'5'<'6'<'7'<'8'<'9'<'A'<'J'<'K'<'Q'<'T'  (ASCII)
//! `card_from_control` (misc_functions.lua:1625) then creates one Card per
//! proto in that order; `Card:init` (card.lua:24) assigns
//! `sort_id = (G.sort_id or 0) + 1` from a global counter. Only the *relative*
//! order matters (pseudoshuffle pre-sorts by sort_id), so we number 1..=52.
//!
//! # Deck order and drawing
//! `CardArea:emplace` inserts at index 1 for type=='deck' (cardarea.lua:33-34)
//! and `CardArea:remove_card` for a deck removes `cards[#cards]`
//! (cardarea.lua:76-77), i.e. cards are DRAWN FROM THE END of the `cards`
//! array. We model the deck as a Vec and draw with `pop()`.
//!
//! # pseudoshuffle (misc_functions.lua:206-217)
//! 1. `math.randomseed(seed)` — seed comes from `pseudoseed(key)`, where the
//!    key is 'shuffle' at run start (game.lua:2383), 'nr'..ante at round start
//!    (state_events.lua:344) and 'cashout'..ante at cash out
//!    (button_callbacks.lua:2918).
//! 2. stable pre-sort by ascending `sort_id` (making prior deck order moot),
//! 3. Fisher-Yates: `for i = #list, 2, -1 do j = math.random(i); swap(i, j)`.

use crate::cards::{Card, Rank, Suit};
use crate::rng::LuaRandom;

/// Ranks in `s..r` string-sort order within one suit (see module docs):
/// '2'..'9' then 'A', 'J', 'K', 'Q', 'T'.
const RANKS_PROTO_ORDER: [u8; 13] = [2, 3, 4, 5, 6, 7, 8, 9, 14, 11, 13, 12, 10];

/// Suits in `s..r` string-sort order: C < D < H < S.
const SUITS_PROTO_ORDER: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

/// The 52-card Red Deck in creation order, `sort_id` 1..=52.
pub fn red_deck() -> Vec<Card> {
    let mut cards = Vec::with_capacity(52);
    let mut sort_id = 0u32;
    for &suit in &SUITS_PROTO_ORDER {
        for &rank in &RANKS_PROTO_ORDER {
            sort_id += 1;
            cards.push(Card::new(Rank(rank), suit, sort_id));
        }
    }
    cards
}

/// `pseudoshuffle(list, seed)` (misc_functions.lua:206-217).
///
/// `seed` is the value returned by `pseudoseed(key)`; the caller owns the
/// per-key stream advancement (`RngState::pseudoseed`).
pub fn pseudoshuffle(list: &mut [Card], seed: f64) {
    let mut rng = LuaRandom::seeded(seed);
    // `if list[1] and list[1].sort_id then table.sort(...) end` — playing
    // cards always carry a sort_id. sort_ids are unique so table.sort's
    // instability is unobservable.
    list.sort_by_key(|c| c.sort_id);
    // `for i = #list, 2, -1 do local j = math.random(i); swap end`
    for i in (1..list.len()).rev() {
        // Lua index i+1 (1-based); j = math.random(i+1) in 1..=(i+1).
        let j = rng.random_range(1, (i + 1) as i64) as usize - 1;
        list.swap(i, j);
    }
}
