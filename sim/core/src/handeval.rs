//! Poker-hand evaluation, ported line-for-line from
//! `evaluate_poker_hand` + `get_X_same`/`get_flush`/`get_straight`/`get_highest`
//! (functions/misc_functions.lua:376-621) and `G.FUNCS.get_poker_hand_info`
//! (functions/state_events.lua:540-569).
//!
//! Cards are referred to by index into the played-card slice so that the
//! object-identity comparisons in the Lua (`if vv == v`) stay exact.
//!
//! Stone cards: `Card:get_id()` (card.lua:957-962) returns
//! `-math.random(100, 1000000)` for Stone cards — a *fresh negative random per
//! call* — so a Stone card never rank-matches anything (including another
//! Stone card, which draws a different random on its own call). We model that
//! as "no id" via `Card::id_for_eval`. Fidelity caveat: the real call also
//! advances the global LuaJIT PRNG, which is unobservable in practice because
//! every consumer of `math.random` reseeds via `pseudoseed` first.

use crate::cards::{Card, HandType};

/// Joker-driven evaluation modifiers, computed live from the owned jokers
/// (`Run::live_eval_mods`): the game reads them via `next(find_joker(...))`
/// inside get_flush/get_straight/Card:is_suit, so debuffed copies count for
/// nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct EvalMods {
    /// Four Fingers: flushes and straights need only 4 cards
    /// (misc_functions.lua:526/553).
    pub four_fingers: bool,
    /// Shortcut: straights may skip one rank gap (misc_functions.lua:554).
    pub shortcut: bool,
    /// Smeared Joker: Hearts/Diamonds and Spades/Clubs count as the same
    /// suit in every `Card:is_suit` branch (card.lua:4071/4084).
    pub smeared: bool,
}

/// The full `results` table of `evaluate_poker_hand`: for each hand type the
/// list of card-index groups, plus `top`. Kept whole because jokers and boss
/// blinds (P3) consume entries other than the top one.
#[derive(Debug, Clone, Default)]
pub struct PokerHands {
    results: [Vec<Vec<usize>>; 12],
    pub top: Option<HandType>,
}

impl PokerHands {
    fn idx(ht: HandType) -> usize {
        HandType::ALL.iter().position(|&h| h == ht).unwrap()
    }

    pub fn get(&self, ht: HandType) -> &Vec<Vec<usize>> {
        &self.results[Self::idx(ht)]
    }

    fn get_mut(&mut self, ht: HandType) -> &mut Vec<Vec<usize>> {
        &mut self.results[Self::idx(ht)]
    }

    fn set_with_top(&mut self, ht: HandType, groups: Vec<Vec<usize>>) {
        if !groups.is_empty() && self.top.is_none() {
            self.top = Some(ht);
        }
        *self.get_mut(ht) = groups;
    }
}

/// `hand[i]:get_id()` filtered like the Lua call sites (`id > 1 and id < 15`
/// in get_straight; never-matching negatives in get_X_same).
fn get_id(card: &Card) -> Option<u8> {
    card.id_for_eval()
}

/// `get_X_same(num, hand)` (misc_functions.lua:592-611).
///
/// Iterates `i = #hand..1`; the *last* write for a rank (smallest i) wins, so
/// each group is ordered: the lowest-index member first, then the rest in
/// ascending hand order — which is simply ascending hand order.
pub fn get_x_same(num: usize, hand: &[Card]) -> Vec<Vec<usize>> {
    // `local vals = {{},...}` — 14 slots, indexed by get_id() (2..=14).
    let mut vals: Vec<Vec<usize>> = vec![Vec::new(); 15];
    for i in (0..hand.len()).rev() {
        let Some(id_i) = get_id(&hand[i]) else {
            // Stone card: curr == {hand[i]} and never grows; `vals[negative]`
            // is written in Lua but the read loop below only covers 1..14.
            continue;
        };
        let mut curr = vec![i];
        for (j, cj) in hand.iter().enumerate() {
            if i != j && get_id(cj) == Some(id_i) {
                curr.push(j);
            }
        }
        if curr.len() == num {
            vals[id_i as usize] = curr;
        }
    }
    // `for i=#vals, 1, -1 do if next(vals[i]) then table.insert(ret, vals[i])`
    let mut ret = Vec::new();
    for i in (1..vals.len()).rev() {
        if !vals[i].is_empty() {
            ret.push(std::mem::take(&mut vals[i]));
        }
    }
    ret
}

/// `get_flush(hand)` (misc_functions.lua:522-546). Suit probe order is
/// Spades, Hearts, Clubs, Diamonds; first suit reaching the threshold wins.
pub fn get_flush(hand: &[Card], mods: &EvalMods) -> Vec<Vec<usize>> {
    let four_fingers = mods.four_fingers;
    let need = 5 - usize::from(four_fingers);
    if hand.len() > 5 || hand.len() < need {
        return Vec::new();
    }
    for suit in crate::cards::Suit::ALL {
        // is_suit(suit, nil, true): flush_calc branch of card.lua:4064.
        let t: Vec<usize> = (0..hand.len())
            .filter(|&i| hand[i].is_suit(suit, mods.smeared))
            .collect();
        if t.len() >= need {
            return vec![t];
        }
    }
    Vec::new()
}

/// `get_straight(hand)` (misc_functions.lua:548-590).
///
/// Ported quirks:
/// - `j == 1 and 14 or j`: the ace is probed FIRST (as rank 1) so A-2-3-4-5
///   works, and AGAIN at j == 14 for broadway; the ace-low copy is discarded
///   by the `t = {}` reset at the first gap since the straight isn't complete
///   yet, so aces are never double-collected.
/// - EVERY duplicate of a rank in the window is pushed into the scoring set
///   (relevant with Four Fingers/Shortcut where dupes can coexist).
/// - after the straight is complete the loop keeps collecting consecutive
///   ranks and only `break`s at the next gap.
pub fn get_straight(hand: &[Card], mods: &EvalMods) -> Vec<Vec<usize>> {
    let four_fingers = mods.four_fingers;
    let need = 5 - usize::from(four_fingers);
    if hand.len() > 5 || hand.len() < need {
        return Vec::new();
    }
    let mut t: Vec<usize> = Vec::new();
    let mut ids: Vec<Vec<usize>> = vec![Vec::new(); 15];
    for (i, card) in hand.iter().enumerate() {
        // `if id > 1 and id < 15` — Stone cards (negative ids) are excluded.
        if let Some(id) = get_id(card) {
            ids[id as usize].push(i);
        }
    }

    let mut straight_length = 0usize;
    let mut straight = false;
    let can_skip = mods.shortcut;
    let mut skipped_rank = false;
    for j in 1..=14usize {
        let probe = if j == 1 { 14 } else { j };
        if !ids[probe].is_empty() {
            straight_length += 1;
            skipped_rank = false;
            for &v in &ids[probe] {
                t.push(v);
            }
        } else if can_skip && !skipped_rank && j != 14 {
            skipped_rank = true;
        } else {
            straight_length = 0;
            skipped_rank = false;
            if !straight {
                t.clear();
            }
            if straight {
                break;
            }
        }
        if straight_length >= need {
            straight = true;
        }
    }
    if !straight {
        return Vec::new();
    }
    vec![t]
}

/// `get_highest(hand)` (misc_functions.lua:613-621): single card with the
/// strictly greatest `get_nominal()`. Ties (only possible for identical
/// rank+suit) resolve to the earlier-created card: `unique_val = 1 -
/// ID/1603301` decreases with creation order and the comparison is strict.
pub fn get_highest(hand: &[Card]) -> Vec<Vec<usize>> {
    let mut highest: Option<usize> = None;
    for (i, card) in hand.iter().enumerate() {
        let beats = match highest {
            None => true,
            Some(h) => {
                let (a, b) = (card.get_nominal(false), hand[h].get_nominal(false));
                a > b || (a == b && card.sort_id < hand[h].sort_id)
            }
        };
        if beats {
            highest = Some(i);
        }
    }
    match highest {
        Some(h) if !hand.is_empty() => vec![vec![h]],
        _ => Vec::new(),
    }
}

/// `evaluate_poker_hand(hand)` (misc_functions.lua:376-520).
pub fn evaluate_poker_hand(hand: &[Card], mods: &EvalMods) -> PokerHands {
    let five = get_x_same(5, hand);
    let four = get_x_same(4, hand);
    let three = get_x_same(3, hand);
    let two = get_x_same(2, hand);
    let flush = get_flush(hand, mods);
    let straight = get_straight(hand, mods);
    let highest = get_highest(hand);

    let mut r = PokerHands::default();

    if !five.is_empty() && !flush.is_empty() {
        r.set_with_top(HandType::FlushFive, five.clone());
    }

    if !three.is_empty() && !two.is_empty() && !flush.is_empty() {
        let mut fh: Vec<usize> = Vec::new();
        fh.extend_from_slice(&three[0]);
        fh.extend_from_slice(&two[0]);
        r.set_with_top(HandType::FlushHouse, vec![fh]);
    }

    if !five.is_empty() {
        r.set_with_top(HandType::FiveOfAKind, five.clone());
    }

    if !flush.is_empty() && !straight.is_empty() {
        // flush cards first, then straight cards not already in the flush set
        // (object identity comparison in the Lua; indices here).
        let mut ret: Vec<usize> = flush[0].clone();
        for &v in &straight[0] {
            if !flush[0].contains(&v) {
                ret.push(v);
            }
        }
        r.set_with_top(HandType::StraightFlush, vec![ret]);
    }

    if !four.is_empty() {
        r.set_with_top(HandType::FourOfAKind, four.clone());
    }

    if !three.is_empty() && !two.is_empty() {
        let mut fh: Vec<usize> = Vec::new();
        fh.extend_from_slice(&three[0]);
        fh.extend_from_slice(&two[0]);
        r.set_with_top(HandType::FullHouse, vec![fh]);
    }

    if !flush.is_empty() {
        r.set_with_top(HandType::Flush, flush.clone());
    }

    if !straight.is_empty() {
        r.set_with_top(HandType::Straight, straight.clone());
    }

    if !three.is_empty() {
        r.set_with_top(HandType::ThreeOfAKind, three.clone());
    }

    // `(#parts._2 == 2) or (#parts._3 == 1 and #parts._2 == 1)` — Two Pair is
    // two pairs, or trips+pair (a Full House also registers as Two Pair).
    if two.len() == 2 || (three.len() == 1 && two.len() == 1) {
        let fh_2a = &two[0];
        let fh_2b: &Vec<usize> = if two.len() >= 2 { &two[1] } else { &three[0] };
        let mut fh: Vec<usize> = Vec::new();
        fh.extend_from_slice(fh_2a);
        fh.extend_from_slice(fh_2b);
        r.set_with_top(HandType::TwoPair, vec![fh]);
    }

    if !two.is_empty() {
        r.set_with_top(HandType::Pair, two.clone());
    }

    if !highest.is_empty() {
        r.set_with_top(HandType::HighCard, highest);
    }

    // Tail overwrites (misc_functions.lua:507-517): "a Five of a Kind is also
    // a Four of a Kind", etc. NOTE the Lua copies the first 4/3/2 *groups* of
    // the higher list (`{results[X][1], results[X][2], ...}` with nil holes
    // dropped), not 4/3/2 cards — so e.g. Four of a Kind ends up containing
    // the full 5-card group. Ported verbatim.
    if !r.get(HandType::FiveOfAKind).is_empty() {
        let mut v = r.get(HandType::FiveOfAKind).clone();
        v.truncate(4);
        *r.get_mut(HandType::FourOfAKind) = v;
    }
    if !r.get(HandType::FourOfAKind).is_empty() {
        let mut v = r.get(HandType::FourOfAKind).clone();
        v.truncate(3);
        *r.get_mut(HandType::ThreeOfAKind) = v;
    }
    if !r.get(HandType::ThreeOfAKind).is_empty() {
        let mut v = r.get(HandType::ThreeOfAKind).clone();
        v.truncate(2);
        *r.get_mut(HandType::Pair) = v;
    }

    r
}

/// `G.FUNCS.get_poker_hand_info(_cards)` (state_events.lua:540-569): pick the
/// best hand by the fixed elseif chain and return its first scoring group.
pub fn get_poker_hand_info(hand: &[Card], mods: &EvalMods) -> (HandType, Vec<usize>, PokerHands) {
    let poker_hands = evaluate_poker_hand(hand, mods);
    const ORDER: [HandType; 12] = [
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
    for ht in ORDER {
        if !poker_hands.get(ht).is_empty() {
            let scoring = poker_hands.get(ht)[0].clone();
            return (ht, scoring, poker_hands);
        }
    }
    // Unreachable for non-empty input (High Card always matches); empty input
    // yields an empty High Card like the Lua's 'NULL' text path.
    (HandType::HighCard, Vec::new(), poker_hands)
}
