//! Scripted baseline bot. Competent-but-simple: best-immediate-value hand
//! selection, potential-keeping discards, planets-first shop policy. Emits
//! contract action rows that go through the SAME `step()` path (and masks)
//! as the learned policy.
//!
//! Policy sketch:
//! * BLIND_SELECT: always SELECT_BLIND (never skips).
//! * PLAYING: enumerate every legal 1..5-card subset of the (observable)
//!   hand, score it as `(level chips + card chip bonuses) * level mult` for
//!   the evaluated hand type; play the best. If the best play cannot clear
//!   the remaining requirement, discards remain and the best hand is weak
//!   (< Three of a Kind), discard up to 5 low cards that are not part of a
//!   pair / the dominant suit. Under The Psychic only 5-card subsets are
//!   considered; under Cerulean Bell only subsets containing the forced
//!   card (and no discards).
//! * ROUND_EVAL: CASH_OUT.
//! * SHOP: use held planets; buy planets (celestial cards) when affordable;
//!   buy the highest-rarity affordable joker (cheaper on ties) while a slot
//!   is free; buy free (couponed) packs; sell a stuck tarot when the
//!   consumable slots are full; otherwise LEAVE_SHOP. Never rerolls.
//! * PACK: prefer the planet of the most-played hand, then any planet, then
//!   a joker, then any usable consumable, then a playing card; else skip.

use balatro_core::cards::HandType;
use balatro_core::consumables::planet_for_hand;
use balatro_core::handeval::get_poker_hand_info;
use balatro_core::items::{consumable_by_key, ConsumableSet};
use balatro_core::run::{Run, State};
use balatro_core::shop::{PackItem, ShopItemKind};

use crate::consts::{action_type as at, HAND_MAX, MAX_CARD_PICKS};
use crate::encode::{EnvMasks, ShopSlotRef};

fn row(t: i64) -> crate::action::ActionRow {
    crate::action::ActionRow {
        action_type: t,
        cards: [-1; MAX_CARD_PICKS],
        n_cards: 0,
        joker_target: -1,
        consumable_target: -1,
        shop_target: -1,
        pack_target: -1,
    }
}

fn with_cards(t: i64, picks: &[usize]) -> crate::action::ActionRow {
    let mut r = row(t);
    r.n_cards = picks.len() as i64;
    for (i, &p) in picks.iter().take(MAX_CARD_PICKS).enumerate() {
        r.cards[i] = p as i64;
    }
    r
}

fn is_planet(key: &str) -> bool {
    matches!(consumable_by_key(key), Some((_, ConsumableSet::Planet)))
}

/// Value estimate of playing `subset` (hand indices): evaluated hand type's
/// level chips/mult with the scoring cards' chip bonuses.
fn play_value(run: &Run, subset: &[usize]) -> (f64, HandType) {
    let cards: Vec<balatro_core::cards::Card> = subset.iter().map(|&i| run.hand()[i]).collect();
    let mods = run.eval_mods();
    let (ht, scoring, _) = get_poker_hand_info(&cards, &mods);
    let rowv = run.hands_table().get(ht);
    let card_chips: f64 = scoring.iter().map(|&i| cards[i].chip_bonus()).sum();
    ((rowv.chips + card_chips) * rowv.mult, ht)
}

/// Enumerate legal subsets and return the best (value, subset, hand type).
fn best_play(run: &Run) -> Option<(f64, Vec<usize>, HandType)> {
    let n = run.hand().len().min(HAND_MAX);
    if n == 0 {
        return None;
    }
    let psychic = run.active_blind().is_some_and(|b| {
        !b.disabled
            && matches!(
                b.proto.debuff,
                balatro_core::blinds::BossDebuff::HandSizeGe(_)
            )
    });
    let forced = run.forced_card_index().filter(|&i| i < n);
    let mut best: Option<(f64, Vec<usize>, HandType)> = None;
    // All subsets of size 1..=5 over <=10 cards (<=637 evals).
    let total = 1u32 << n;
    for bits in 1..total {
        let k = bits.count_ones() as usize;
        if k > MAX_CARD_PICKS || (psychic && k != 5) {
            continue;
        }
        if let Some(f) = forced {
            if bits & (1 << f) == 0 {
                continue;
            }
        }
        let subset: Vec<usize> = (0..n).filter(|i| bits & (1 << i) != 0).collect();
        let (v, ht) = play_value(run, &subset);
        if best.as_ref().is_none_or(|(bv, _, _)| v > *bv) {
            best = Some((v, subset, ht));
        }
    }
    best
}

/// Cards worth keeping: members of a rank pair/trip, or of the dominant
/// suit when 4+ strong. Everything else is discard fodder, lowest first.
fn discard_picks(run: &Run) -> Vec<usize> {
    let n = run.hand().len().min(HAND_MAX);
    let hand = &run.hand()[..n];
    let mut rank_counts = [0u8; 15];
    let mut suit_counts = [0u8; 4];
    for c in hand {
        rank_counts[c.rank.id() as usize] += 1;
        suit_counts[crate::consts::suit_index(c.suit)] += 1;
    }
    let dom_suit = (0..4).max_by_key(|&s| suit_counts[s]).unwrap();
    let flushy = suit_counts[dom_suit] >= 4;
    let mut fodder: Vec<usize> = (0..n)
        .filter(|&i| {
            let c = &hand[i];
            let keep = rank_counts[c.rank.id() as usize] >= 2
                || (flushy && crate::consts::suit_index(c.suit) == dom_suit);
            !keep
        })
        .collect();
    // Lowest nominal value first.
    fodder.sort_by(|&a, &b| {
        hand[a]
            .rank
            .nominal()
            .cmp(&hand[b].rank.nominal())
            .then(a.cmp(&b))
    });
    fodder.truncate(MAX_CARD_PICKS);
    fodder
}

/// Best pack slot to take, by preference class (lower = better).
fn pack_preference(run: &Run, item: &PackItem) -> i64 {
    match item {
        PackItem::Consumable(c) if is_planet(c.key) => {
            if c.key == planet_for_hand(run.most_played_hand()) {
                0
            } else {
                1
            }
        }
        PackItem::Joker(_) => 2,
        PackItem::Consumable(_) => 3,
        PackItem::PlayingCard(_) => 4,
    }
}

pub fn bot_action(run: &Run, masks: &EnvMasks, slots: &[ShopSlotRef]) -> crate::action::ActionRow {
    let m = &masks.action_type;
    let choice = 'choice: {
        // Use a held planet whenever possible (frees a slot, levels a hand).
        if m[at::USE_CONSUMABLE as usize] {
            for (i, c) in run
                .consumables()
                .iter()
                .take(crate::consts::CONSUMABLE_SLOTS)
                .enumerate()
            {
                if is_planet(c.key) && run.consumable_has_any_use(c.key) {
                    let mut r = row(at::USE_CONSUMABLE);
                    r.consumable_target = i as i64;
                    break 'choice r;
                }
            }
        }

        match run.state() {
            State::BlindSelect => row(at::SELECT_BLIND),
            State::RoundEval => row(at::CASH_OUT),
            State::SelectingHand => {
                let Some((value, subset, ht)) = best_play(run) else {
                    break 'choice fallback(masks);
                };
                let remaining = (run.blind_chips() - run.chips()).max(0.0);
                let weak = (ht as usize) < (HandType::ThreeOfAKind as usize);
                if m[at::DISCARD as usize]
                    && value < remaining
                    && weak
                    && run.forced_card_index().is_none()
                {
                    let picks = discard_picks(run);
                    if !picks.is_empty() {
                        break 'choice with_cards(at::DISCARD, &picks);
                    }
                }
                with_cards(at::PLAY_HAND, &subset)
            }
            State::Shop => {
                // Buy planets first, then the best joker, then free packs.
                let mut joker_pick: Option<(usize, u8, i64)> = None; // slot, rarity, cost
                for (s, r) in slots.iter().enumerate() {
                    if !masks.shop_target[s] {
                        continue;
                    }
                    let shop = run.shop().expect("shop");
                    match *r {
                        ShopSlotRef::Card(i) => match &shop.jokers[i].kind {
                            ShopItemKind::Consumable(c) if is_planet(c.key) => {
                                let mut a = row(at::BUY_SHOP);
                                a.shop_target = s as i64;
                                break 'choice a;
                            }
                            ShopItemKind::Joker(j) => {
                                let meta = j.id.meta();
                                let cost = run.shop_item_cost(&shop.jokers[i]);
                                let better = match joker_pick {
                                    None => true,
                                    Some((_, r0, c0)) => {
                                        meta.rarity > r0 || (meta.rarity == r0 && cost < c0)
                                    }
                                };
                                if better {
                                    joker_pick = Some((s, meta.rarity, cost));
                                }
                            }
                            _ => {}
                        },
                        ShopSlotRef::Pack(i) => {
                            if run.pack_cost(&shop.packs[i]) == 0 {
                                let mut a = row(at::BUY_SHOP);
                                a.shop_target = s as i64;
                                break 'choice a;
                            }
                        }
                        ShopSlotRef::Voucher(_) => {}
                    }
                }
                if let Some((s, _, _)) = joker_pick {
                    let mut a = row(at::BUY_SHOP);
                    a.shop_target = s as i64;
                    break 'choice a;
                }
                // Slots full of unusable tarots? Sell one to make room.
                if m[at::SELL_CONSUMABLE as usize]
                    && run.consumables().len() >= run.consumable_slots()
                {
                    for (i, c) in run
                        .consumables()
                        .iter()
                        .take(crate::consts::CONSUMABLE_SLOTS)
                        .enumerate()
                    {
                        if !is_planet(c.key) && masks.consumable_target[i] {
                            let mut a = row(at::SELL_CONSUMABLE);
                            a.consumable_target = i as i64;
                            break 'choice a;
                        }
                    }
                }
                row(at::LEAVE_SHOP)
            }
            State::PackOpen => {
                let Some(pack) = run.pack() else {
                    break 'choice fallback(masks);
                };
                let mut best: Option<(i64, usize)> = None;
                for (i, item) in pack
                    .items
                    .iter()
                    .take(crate::consts::PACK_SLOTS)
                    .enumerate()
                {
                    if !masks.pack_target[i] {
                        continue;
                    }
                    let pref = pack_preference(run, item);
                    if best.is_none_or(|(bp, _)| pref < bp) {
                        best = Some((pref, i));
                    }
                }
                match best {
                    Some((pref, i)) if pref <= 4 => {
                        let mut a = row(at::PICK_PACK);
                        a.pack_target = i as i64;
                        a
                    }
                    _ => row(at::SKIP_PACK),
                }
            }
            _ => fallback(masks),
        }
    };

    // Safety net: never emit something the masks forbid.
    if crate::action::validate(usize::MAX, &choice, masks).is_ok() {
        choice
    } else {
        fallback(masks)
    }
}

/// First legal action type with the first legal pointer/card choices —
/// guaranteed valid whenever any action is legal.
pub fn fallback(masks: &EnvMasks) -> crate::action::ActionRow {
    let t = masks
        .action_type
        .iter()
        .position(|&b| b)
        .unwrap_or(at::SELECT_BLIND as usize) as i64;
    let mut r = row(t);
    let first = |m: &[bool]| m.iter().position(|&b| b).map(|i| i as i64).unwrap_or(-1);
    match t {
        x if x == at::PLAY_HAND || x == at::DISCARD => {
            r.n_cards = 1;
            r.cards[0] = first(&masks.card_select);
        }
        x if x == at::USE_CONSUMABLE => r.consumable_target = first(&masks.consumable_target),
        x if x == at::SELL_CONSUMABLE => r.consumable_target = first(&masks.consumable_target),
        x if x == at::SELL_JOKER => r.joker_target = first(&masks.joker_target),
        x if x == at::BUY_SHOP => r.shop_target = first(&masks.shop_target),
        x if x == at::PICK_PACK => r.pack_target = first(&masks.pack_target),
        _ => {}
    }
    r
}
