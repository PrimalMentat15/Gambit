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
use crate::config::{Deck, P_CARDS_SORTED};
use crate::rng::{LuaRandom, RngState};

/// Ranks in `s..r` string-sort order within one suit (see module docs):
/// '2'..'9' then 'A', 'J', 'K', 'Q', 'T'.
const RANKS_PROTO_ORDER: [u8; 13] = [2, 3, 4, 5, 6, 7, 8, 9, 14, 11, 13, 12, 10];

/// Suits in `s..r` string-sort order: C < D < H < S.
const SUITS_PROTO_ORDER: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

/// The 52-card Red Deck in creation order, `sort_id` 1..=52.
pub fn red_deck() -> Vec<Card> {
    build_deck(Deck::Red, &mut RngState::new(""))
}

/// The starting deck for `deck`, in `Game:start_run`'s creation order
/// (game.lua:2310-2357).
///
/// The Lua builds a list of card *protos*, filters it, sorts it by the
/// concatenated `s..r..e..d..g` string, and only then creates Cards — so
/// `sort_id` reflects the post-filter, post-sort order, not the pre-filter one.
/// Three decks change what goes in:
///
/// * **Abandoned** drops K/Q/J before the sort (game.lua:2337), leaving 40
///   cards numbered 1..=40.
/// * **Erratic** replaces each of the 52 slots with a random `G.P_CARDS` entry
///   (game.lua:2324). Duplicates are expected; the deck is still 52 cards.
/// * **Checkered** is *not* a proto change — `apply_to_run` re-suits the cards
///   after creation (back.lua:239-253), so `sort_id` follows the standard
///   layout and only the suits differ.
///
/// `rng` is only touched for Erratic, and only on the `'erratic'` key, so every
/// other deck leaves every RNG stream exactly where the Red Deck left it.
pub fn build_deck(deck: Deck, rng: &mut RngState) -> Vec<Card> {
    // --- proto list -----------------------------------------------------
    let mut protos: Vec<(Suit, Rank)> = Vec::with_capacity(52);
    if deck == Deck::Erratic {
        // `pseudorandom_element(G.P_CARDS, pseudoseed('erratic'))` per slot.
        // The element helper sorts its key list before drawing
        // (misc_functions.lua:260-263), so this depends on P_CARDS_SORTED and
        // never on Lua's `pairs` hash order.
        for _ in 0..52 {
            let i = rng.random_range("erratic", 1, P_CARDS_SORTED.len() as i64) as usize - 1;
            protos.push(P_CARDS_SORTED[i]);
        }
    } else {
        for &suit in &SUITS_PROTO_ORDER {
            for &rank in &RANKS_PROTO_ORDER {
                protos.push((suit, Rank(rank)));
            }
        }
    }

    // `no_faces` filter (game.lua:2337), before the sort.
    if deck == Deck::Abandoned {
        protos.retain(|(_, r)| !matches!(r.0, 11 | 12 | 13));
    }

    // `table.sort` by `s..r` (game.lua:2349-2351). Ties are only possible
    // under Erratic, where tied protos are byte-identical at creation and
    // `pseudoshuffle` pre-sorts by sort_id anyway — so LuaJIT's unstable sort
    // is unobservable and a stable sort here is equivalent.
    protos.sort_by_key(|&(s, r)| (suit_sort_key(s), rank_sort_key(r)));

    let mut cards: Vec<Card> = protos
        .into_iter()
        .enumerate()
        .map(|(i, (suit, rank))| Card::new(rank, suit, i as u32 + 1))
        .collect();

    // Checkered: Clubs -> Spades, Diamonds -> Hearts, after creation
    // (back.lua:239-253). Leaves 26 Spades and 26 Hearts.
    if deck == Deck::Checkered {
        for card in &mut cards {
            card.suit = match card.suit {
                Suit::Clubs => Suit::Spades,
                Suit::Diamonds => Suit::Hearts,
                other => other,
            };
        }
    }

    cards
}

/// Position of a suit in `s..r` byte order: C < D < H < S.
fn suit_sort_key(s: Suit) -> u8 {
    match s {
        Suit::Clubs => 0,
        Suit::Diamonds => 1,
        Suit::Hearts => 2,
        Suit::Spades => 3,
    }
}

/// Position of a rank in `s..r` byte order within a suit (see module docs).
fn rank_sort_key(r: Rank) -> u8 {
    RANKS_PROTO_ORDER
        .iter()
        .position(|&x| x == r.0)
        .unwrap_or(usize::MAX) as u8
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Stake;
    use std::collections::HashMap;

    fn build(d: Deck) -> Vec<Card> {
        build_deck(d, &mut RngState::new("DECKTEST"))
    }

    /// Every deck except the three that change composition must produce the
    /// byte-identical Red Deck. If this ever fails, a deck's effect leaked
    /// into construction when it belongs in `apply_to_run`.
    #[test]
    fn only_three_decks_change_the_starting_cards() {
        let red = red_deck();
        assert_eq!(red.len(), 52);
        for d in Deck::ALL {
            let cards = build(d);
            let differs = matches!(d, Deck::Abandoned | Deck::Checkered | Deck::Erratic);
            let same = cards.len() == red.len()
                && cards
                    .iter()
                    .zip(&red)
                    .all(|(a, b)| a.rank == b.rank && a.suit == b.suit && a.sort_id == b.sort_id);
            assert_eq!(
                same,
                !differs,
                "{d:?} composition unexpectedly {}",
                if same { "unchanged" } else { "changed" }
            );
        }
    }

    #[test]
    fn abandoned_drops_the_face_cards() {
        let cards = build(Deck::Abandoned);
        assert_eq!(cards.len(), 40, "52 - 12 face cards");
        assert!(cards.iter().all(|c| !matches!(c.rank.0, 11 | 12 | 13)));
        // sort_ids renumber over what survives, contiguous from 1.
        let ids: Vec<u32> = cards.iter().map(|c| c.sort_id).collect();
        assert_eq!(ids, (1..=40).collect::<Vec<_>>());
        // Aces and tens survive; each suit keeps 10 ranks.
        let mut per_suit: HashMap<Suit, usize> = HashMap::new();
        for c in &cards {
            *per_suit.entry(c.suit).or_default() += 1;
        }
        assert!(per_suit.values().all(|&n| n == 10), "{per_suit:?}");
    }

    #[test]
    fn checkered_recolours_without_reordering() {
        let red = red_deck();
        let cards = build(Deck::Checkered);
        assert_eq!(cards.len(), 52);

        let mut per_suit: HashMap<Suit, usize> = HashMap::new();
        for c in &cards {
            *per_suit.entry(c.suit).or_default() += 1;
        }
        assert_eq!(per_suit.get(&Suit::Spades), Some(&26));
        assert_eq!(per_suit.get(&Suit::Hearts), Some(&26));
        assert_eq!(per_suit.get(&Suit::Clubs), None);
        assert_eq!(per_suit.get(&Suit::Diamonds), None);

        // The swap happens AFTER creation, so ranks and sort_ids are still
        // exactly the Red Deck's -- only the suit changed.
        for (a, b) in cards.iter().zip(&red) {
            assert_eq!(a.rank, b.rank);
            assert_eq!(a.sort_id, b.sort_id);
            let want = match b.suit {
                Suit::Clubs => Suit::Spades,
                Suit::Diamonds => Suit::Hearts,
                other => other,
            };
            assert_eq!(a.suit, want);
        }
    }

    #[test]
    fn erratic_is_52_cards_seed_determined_and_sorted() {
        let a = build(Deck::Erratic);
        let b = build(Deck::Erratic);
        assert_eq!(a.len(), 52, "always 52, duplicates and all");
        // Same seed -> same deck.
        assert!(a
            .iter()
            .zip(&b)
            .all(|(x, y)| x.rank == y.rank && x.suit == y.suit));

        // Different seed -> almost certainly a different deck.
        let c = build_deck(Deck::Erratic, &mut RngState::new("OTHERSEED"));
        assert!(
            a.iter()
                .zip(&c)
                .any(|(x, y)| x.rank != y.rank || x.suit != y.suit),
            "two seeds produced identical erratic decks"
        );

        // Still emitted in `s..r` order with contiguous sort_ids.
        let ids: Vec<u32> = a.iter().map(|x| x.sort_id).collect();
        assert_eq!(ids, (1..=52).collect::<Vec<_>>());
        let keys: Vec<(u8, u8)> = a
            .iter()
            .map(|x| (suit_sort_key(x.suit), rank_sort_key(x.rank)))
            .collect();
        assert!(keys.windows(2).all(|w| w[0] <= w[1]), "not sorted by s..r");
    }

    /// Erratic is the only deck that draws, and only on its own key -- so no
    /// other deck can shift a stream the frozen vectors depend on.
    #[test]
    fn only_erratic_consumes_rng() {
        for d in Deck::ALL {
            let mut rng = RngState::new("STREAMS");
            let _ = build_deck(d, &mut rng);
            let after = rng.pseudoseed("shuffle");

            let mut fresh = RngState::new("STREAMS");
            let baseline = fresh.pseudoseed("shuffle");
            assert_eq!(after, baseline, "{d:?} disturbed the 'shuffle' stream");
        }
    }

    /// Guards the claim in `RunConfig::resolve`'s docs that deck choice and
    /// stake choice are independent inputs to deck construction.
    #[test]
    fn stake_does_not_affect_construction() {
        let _ = Stake::Gold; // stake is not an input to build_deck at all
        assert_eq!(build(Deck::Red).len(), 52);
    }
}
