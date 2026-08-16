//! Observation + mask encoding from a `balatro_core::run::Run`, matching
//! `train/balatro_train/encoding.py` exactly.
//!
//! Documented binding decisions (see also train/README.md):
//!
//! * **Phase mapping**: BlindSelect→BLIND_SELECT, SelectingHand→PLAYING,
//!   RoundEval→ROUND_EVAL, Shop→SHOP, PackOpen→PACK. GameOver/Won are never
//!   observed (auto-reset).
//! * **Hand clamp**: Turtle Bean / vouchers / The Serpent can push the hand
//!   past 10 cards. The observation shows the FIRST 10 in display order and
//!   `hand_len` is clamped to 10; cards past index 9 are unobservable and
//!   unselectable until the hand shrinks (rare; documented contract clamp).
//! * **Face-down cards** (The Wheel/The Fish/The House/The Mark): the whole
//!   identity block (rank/suit/enhancement/edition/seal) is zeroed, only the
//!   face_down flag (and debuff flag) is set.
//! * **Joker state scalars** (joker_feats[6..10], each sign*log1p):
//!   `[state.mult, state.chips, state.x_mult - 1, state.extra]` — the four
//!   `JokerState` accumulators of the effect engine (Ride the Bus / Runner /
//!   Vampire-style scalers / named counters like Seltzer hands or Turtle
//!   Bean size). To Do List's target hand is not observed (v1 gap).
//! * **Flipped jokers** (Amber Acorn) keep their id visible — a small
//!   observation leak the policy could not exploit adversarially (the flip
//!   only hides the post-shuffle order in the real game's UI).
//! * **Jokers/consumables beyond the 6/3 observable slots** (possible via
//!   Negative editions) are hidden and unactionable until the count drops.
//! * **Shop slot order**: card slots (in shop order), then vouchers, then
//!   unopened packs, truncated to 6. During Phase.PACK the slots carry the
//!   open pack's contents instead (cost 0), per the contract.
//! * **deck_counts / drawpile_counts** count PHYSICAL rank/suit — Stone
//!   cards are counted at their printed identity (their Stone-ness shows up
//!   in deck_aggregates' enhancement counts).
//! * **Blind features**: while a blind is live (PLAYING / ROUND_EVAL) they
//!   describe it (requirement = live `ActiveBlind.chips`, e.g. halved by a
//!   disabled Wall); otherwise they describe the UPCOMING blind
//!   (`blind_on_deck`, requirement from `proto.chips(blind_ante)`).

use balatro_core::cards::{Enhancement, HandType};
use balatro_core::items::consumable_by_key;
use balatro_core::items::ConsumableSet;
use balatro_core::run::{Action, BlindStage, Run, State};
use balatro_core::shop::{PackItem, ShopItemKind};

use crate::consts::*;

/// Per-env observation buffer (scattered into the batched numpy arrays).
#[derive(Clone)]
pub struct EnvObs {
    pub hand: [f32; HAND_MAX * F_CARD],
    pub hand_len: i64,
    pub joker_ids: [i64; JOKER_SLOTS],
    pub joker_feats: [f32; JOKER_SLOTS * F_JOKER_FEAT],
    pub consumable_ids: [i64; CONSUMABLE_SLOTS],
    pub consumables_len: i64,
    pub shop_ids: [i64; SHOP_SLOTS],
    pub shop_feats: [f32; SHOP_SLOTS * F_SHOP_FEAT],
    pub blind: [f32; F_BLIND],
    pub global: [f32; F_GLOBAL],
    pub deck_counts: [f32; N_DECK_SLOTS],
    pub deck_aggregates: [f32; F_DECK_AGG],
    pub drawpile_counts: [f32; N_DECK_SLOTS],
}

impl Default for EnvObs {
    fn default() -> Self {
        EnvObs {
            hand: [0.0; HAND_MAX * F_CARD],
            hand_len: 0,
            joker_ids: [0; JOKER_SLOTS],
            joker_feats: [0.0; JOKER_SLOTS * F_JOKER_FEAT],
            consumable_ids: [0; CONSUMABLE_SLOTS],
            consumables_len: 0,
            shop_ids: [0; SHOP_SLOTS],
            shop_feats: [0.0; SHOP_SLOTS * F_SHOP_FEAT],
            blind: [0.0; F_BLIND],
            global: [0.0; F_GLOBAL],
            deck_counts: [0.0; N_DECK_SLOTS],
            deck_aggregates: [0.0; F_DECK_AGG],
            drawpile_counts: [0.0; N_DECK_SLOTS],
        }
    }
}

/// Per-env mask buffer.
#[derive(Clone, Copy)]
pub struct EnvMasks {
    pub action_type: [bool; N_ACTION_TYPES],
    pub card_select: [bool; HAND_MAX],
    pub joker_target: [bool; JOKER_SLOTS],
    pub consumable_target: [bool; CONSUMABLE_SLOTS],
    pub shop_target: [bool; SHOP_SLOTS],
    pub pack_target: [bool; PACK_SLOTS],
}

impl Default for EnvMasks {
    fn default() -> Self {
        EnvMasks {
            action_type: [false; N_ACTION_TYPES],
            card_select: [false; HAND_MAX],
            joker_target: [false; JOKER_SLOTS],
            consumable_target: [false; CONSUMABLE_SLOTS],
            shop_target: [false; SHOP_SLOTS],
            pack_target: [false; PACK_SLOTS],
        }
    }
}

/// What a visible shop slot points at in the sim's shop areas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShopSlotRef {
    /// `ShopState.jokers[i]` (card slot: joker / consumable / playing card).
    Card(usize),
    /// `ShopState.vouchers[i]`.
    Voucher(usize),
    /// `ShopState.packs[i]` (unopened).
    Pack(usize),
}

/// Contract shop-slot order: card slots, vouchers, unopened packs; first 6.
pub fn shop_slot_map(run: &Run) -> Vec<ShopSlotRef> {
    let mut v = Vec::new();
    if run.state() != State::Shop {
        return v;
    }
    if let Some(shop) = run.shop() {
        for i in 0..shop.jokers.len() {
            v.push(ShopSlotRef::Card(i));
        }
        for i in 0..shop.vouchers.len() {
            v.push(ShopSlotRef::Voucher(i));
        }
        for (i, p) in shop.packs.iter().enumerate() {
            if !p.used {
                v.push(ShopSlotRef::Pack(i));
            }
        }
        v.truncate(SHOP_SLOTS);
    }
    v
}

pub fn phase_of(state: State) -> Phase {
    match state {
        State::BlindSelect => Phase::BlindSelect,
        State::SelectingHand => Phase::Playing,
        State::RoundEval => Phase::RoundEval,
        State::Shop => Phase::Shop,
        State::PackOpen => Phase::Pack,
        _ => Phase::GameOver,
    }
}

fn write_card(feat: &mut [f32], card: &balatro_core::cards::Card) {
    if card.face_down {
        feat[CARD_FACEDOWN_OFF] = 1.0;
        feat[CARD_DEBUFFED_OFF] = if card.debuff { 1.0 } else { 0.0 };
        return;
    }
    if card.enhancement != Enhancement::Stone {
        feat[CARD_RANK_OFF + rank_index(card.rank.id())] = 1.0;
        feat[CARD_SUIT_OFF + suit_index(card.suit)] = 1.0;
    }
    feat[CARD_ENH_OFF + enhancement_index(card.enhancement)] = 1.0;
    feat[CARD_EDITION_OFF + edition_index(card.edition)] = 1.0;
    feat[CARD_SEAL_OFF + seal_index(card.seal)] = 1.0;
    feat[CARD_DEBUFFED_OFF] = if card.debuff { 1.0 } else { 0.0 };
}

fn consumable_shop_kind(key: &str) -> ShopKind {
    match consumable_by_key(key) {
        Some((_, ConsumableSet::Tarot)) => ShopKind::Tarot,
        Some((_, ConsumableSet::Planet)) => ShopKind::Planet,
        _ => ShopKind::Spectral,
    }
}

pub fn encode_obs(run: &Run) -> EnvObs {
    let mut o = EnvObs::default();
    let state = run.state();

    // ---------------------------------------------------------------- hand
    let hand = run.hand();
    let hand_len = hand.len().min(HAND_MAX);
    o.hand_len = hand_len as i64;
    for (slot, card) in hand.iter().take(HAND_MAX).enumerate() {
        write_card(&mut o.hand[slot * F_CARD..(slot + 1) * F_CARD], card);
    }

    // --------------------------------------------------------------- jokers
    for (slot, j) in run.jokers().iter().take(JOKER_SLOTS).enumerate() {
        o.joker_ids[slot] = joker_vocab_id(j.id);
        let f = &mut o.joker_feats[slot * F_JOKER_FEAT..(slot + 1) * F_JOKER_FEAT];
        f[JOKER_EDITION_OFF + edition_index(j.edition)] = 1.0;
        f[JOKER_SELL_OFF] = log1p(run.joker_sell_value(j).max(0) as f64);
        f[JOKER_STATE_OFF] = slog1p(j.state.mult);
        f[JOKER_STATE_OFF + 1] = slog1p(j.state.chips);
        f[JOKER_STATE_OFF + 2] = slog1p(j.state.x_mult - 1.0);
        f[JOKER_STATE_OFF + 3] = slog1p(j.state.extra);
        f[JOKER_DEBUFFED_OFF] = if j.debuffed { 1.0 } else { 0.0 };
    }

    // ---------------------------------------------------------- consumables
    let cons = run.consumables();
    o.consumables_len = cons.len().min(CONSUMABLE_SLOTS) as i64;
    for (slot, c) in cons.iter().take(CONSUMABLE_SLOTS).enumerate() {
        o.consumable_ids[slot] = consumable_vocab_id(c.key);
    }

    // ----------------------------------------------------------- shop / pack
    if state == State::Shop {
        let slots = shop_slot_map(run);
        let shop = run.shop().expect("shop state without shop");
        for (slot, r) in slots.iter().enumerate() {
            let f = &mut o.shop_feats[slot * F_SHOP_FEAT..(slot + 1) * F_SHOP_FEAT];
            match *r {
                ShopSlotRef::Card(i) => {
                    let item = &shop.jokers[i];
                    let cost = run.shop_item_cost(item);
                    f[SHOP_COST_OFF] = log1p(cost.max(0) as f64);
                    match &item.kind {
                        ShopItemKind::Joker(j) => {
                            o.shop_ids[slot] = shop_vocab_joker(j.id);
                            f[SHOP_KIND_OFF + ShopKind::Joker as usize] = 1.0;
                            f[SHOP_EDITION_OFF + edition_index(j.edition)] = 1.0;
                        }
                        ShopItemKind::Consumable(c) => {
                            o.shop_ids[slot] = shop_vocab_consumable(c.key);
                            f[SHOP_KIND_OFF + consumable_shop_kind(c.key) as usize] = 1.0;
                            f[SHOP_EDITION_OFF] = 1.0; // Edition::None
                        }
                        ShopItemKind::PlayingCard(c) => {
                            o.shop_ids[slot] = shop_vocab_card(c.rank.id(), c.suit);
                            f[SHOP_KIND_OFF + ShopKind::PlayingCard as usize] = 1.0;
                            f[SHOP_EDITION_OFF + edition_index(c.edition)] = 1.0;
                        }
                    }
                }
                ShopSlotRef::Voucher(i) => {
                    let v = &shop.vouchers[i];
                    o.shop_ids[slot] = shop_vocab_voucher(v.key);
                    f[SHOP_KIND_OFF + ShopKind::Voucher as usize] = 1.0;
                    f[SHOP_COST_OFF] = log1p(run.voucher_cost(v.key).max(0) as f64);
                }
                ShopSlotRef::Pack(i) => {
                    let p = &shop.packs[i];
                    o.shop_ids[slot] = shop_vocab_pack(p.key);
                    f[SHOP_KIND_OFF + ShopKind::Pack as usize] = 1.0;
                    f[SHOP_COST_OFF] = log1p(run.pack_cost(p).max(0) as f64);
                }
            }
        }
    } else if state == State::PackOpen {
        // Contract: during Phase.PACK the shop slots carry the open pack's
        // contents (cost 0) so the pack pointer head has real tokens.
        if let Some(pack) = run.pack() {
            for (slot, item) in pack.items.iter().take(PACK_SLOTS).enumerate() {
                let f = &mut o.shop_feats[slot * F_SHOP_FEAT..(slot + 1) * F_SHOP_FEAT];
                match item {
                    PackItem::Joker(j) => {
                        o.shop_ids[slot] = shop_vocab_joker(j.id);
                        f[SHOP_KIND_OFF + ShopKind::Joker as usize] = 1.0;
                        f[SHOP_EDITION_OFF + edition_index(j.edition)] = 1.0;
                    }
                    PackItem::Consumable(c) => {
                        o.shop_ids[slot] = shop_vocab_consumable(c.key);
                        f[SHOP_KIND_OFF + consumable_shop_kind(c.key) as usize] = 1.0;
                        f[SHOP_EDITION_OFF] = 1.0;
                    }
                    PackItem::PlayingCard(c) => {
                        o.shop_ids[slot] = shop_vocab_card(c.rank.id(), c.suit);
                        f[SHOP_KIND_OFF + ShopKind::PlayingCard as usize] = 1.0;
                        f[SHOP_EDITION_OFF + edition_index(c.edition)] = 1.0;
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- blind
    {
        let b = &mut o.blind;
        let (proto, req) = match run.active_blind() {
            Some(ab) => (ab.proto, ab.chips),
            None => {
                let proto = match run.blind_on_deck() {
                    BlindStage::Small => &balatro_core::blinds::BL_SMALL,
                    BlindStage::Big => &balatro_core::blinds::BL_BIG,
                    BlindStage::Boss => balatro_core::blinds::boss_by_key(run.boss_choice()),
                };
                (proto, proto.chips(run.blind_ante()))
            }
        };
        let kind = if proto.is_boss {
            2
        } else if proto.key == balatro_core::blinds::BL_BIG.key {
            1
        } else {
            0
        };
        b[BLIND_KIND_OFF + kind] = 1.0;
        if proto.is_boss {
            let idx = boss_onehot_id(proto.key);
            debug_assert!(idx < BOSS_VOCAB);
            b[BLIND_BOSS_OFF + idx] = 1.0;
        }
        b[BLIND_REQ_OFF] = log1p(req.max(0.0));
        b[BLIND_SCORED_OFF] = log1p(run.chips().max(0.0));
        b[BLIND_REWARD_OFF] = log1p(proto.dollars.max(0) as f64);
    }

    // --------------------------------------------------------------- global
    {
        let g = &mut o.global;
        g[GLOBAL_MONEY] = slog1p(run.dollars() as f64);
        g[GLOBAL_HANDS_LEFT] = run.hands_left() as f32 / 4.0;
        g[GLOBAL_DISCARDS_LEFT] = run.discards_left() as f32 / 4.0;
        let ante_bucket = run.ante().clamp(1, N_ANTE_ONEHOT as i64) - 1;
        g[GLOBAL_ANTE_OFF + ante_bucket as usize] = 1.0;
        g[GLOBAL_ROUND] = run.round() as f32 / 24.0;
        g[GLOBAL_PHASE_OFF + phase_of(state) as usize] = 1.0;
        for (i, ht) in HandType::ALL.iter().enumerate() {
            let row = run.hands_table().get(*ht);
            g[GLOBAL_HAND_LEVELS_OFF + 2 * i] = log1p((row.level.max(1) - 1) as f64);
            g[GLOBAL_HAND_LEVELS_OFF + 2 * i + 1] = log1p(row.played as f64);
        }
        g[GLOBAL_REROLL_COST] = log1p(run.reroll_cost().max(0) as f64);
        g[GLOBAL_JOKER_SLOT_COUNT] = run.joker_slots() as f32 / 6.0;
        g[GLOBAL_CONSUMABLE_SLOT_COUNT] = run.consumable_slots() as f32 / 3.0;
        let full_deck = run.deck_len() + run.hand().len() + run.discard_cards().len();
        g[GLOBAL_DECK_SIZE] = log1p(full_deck as f64);
        g[GLOBAL_HAND_SIZE] = run.hand_size() as f32 / 10.0;
        // [48..64) reserved, must stay 0.0.
    }

    // ----------------------------------------------------- deck composition
    {
        let mut enh = [0u32; 9];
        let mut eds = [0u32; 4];
        let mut seals = [0u32; 4];
        let mut total = 0u32;
        for c in run
            .deck_cards()
            .iter()
            .chain(run.hand().iter())
            .chain(run.discard_cards().iter())
        {
            o.deck_counts[deck_slot(c.rank.id(), c.suit)] += 1.0;
            enh[enhancement_index(c.enhancement)] += 1;
            let e = edition_index(c.edition);
            if e > 0 {
                eds[e - 1] += 1;
            }
            let s = seal_index(c.seal);
            if s > 0 {
                seals[s - 1] += 1;
            }
            total += 1;
        }
        for c in run.deck_cards() {
            o.drawpile_counts[deck_slot(c.rank.id(), c.suit)] += 1.0;
        }
        for (i, &n) in enh.iter().enumerate() {
            o.deck_aggregates[DECK_AGG_ENH_OFF + i] = log1p(n as f64);
        }
        for (i, &n) in eds.iter().enumerate() {
            o.deck_aggregates[DECK_AGG_EDITION_OFF + i] = log1p(n as f64);
        }
        for (i, &n) in seals.iter().enumerate() {
            o.deck_aggregates[DECK_AGG_SEAL_OFF + i] = log1p(n as f64);
        }
        o.deck_aggregates[DECK_AGG_TOTAL] = log1p(total as f64);
    }

    o
}

/// Legal-action masks. Mirrors `Run::legal_actions()` with the contract's
/// slot clamps; see module docs for the (documented) places where the
/// factored masks cannot express a cross-head constraint.
pub fn encode_masks(run: &Run, slots: &[ShopSlotRef]) -> EnvMasks {
    use action_type as at;
    let mut m = EnvMasks::default();
    let hand_len = run.hand().len().min(HAND_MAX);

    let mut use_consumable_ok = false;
    for a in run.legal_actions() {
        match a {
            Action::SelectBlind => m.action_type[at::SELECT_BLIND as usize] = true,
            Action::SkipBlind => m.action_type[at::SKIP_BLIND as usize] = true,
            Action::Play => {
                // The sim lists Play whenever hands remain; a play needs >=1
                // card, so gate on a non-empty (observable) hand.
                if hand_len > 0 {
                    m.action_type[at::PLAY_HAND as usize] = true;
                }
            }
            Action::Discard => {
                if hand_len > 0 {
                    m.action_type[at::DISCARD as usize] = true;
                }
            }
            Action::CashOut => m.action_type[at::CASH_OUT as usize] = true,
            Action::BuyShopItem(i) => {
                if let Some(s) = slots.iter().position(|&r| r == ShopSlotRef::Card(i)) {
                    m.shop_target[s] = true;
                }
            }
            Action::RedeemVoucher(i) => {
                if let Some(s) = slots.iter().position(|&r| r == ShopSlotRef::Voucher(i)) {
                    m.shop_target[s] = true;
                }
            }
            Action::BuyPack(i) => {
                if let Some(s) = slots.iter().position(|&r| r == ShopSlotRef::Pack(i)) {
                    m.shop_target[s] = true;
                }
            }
            Action::Reroll => m.action_type[at::REROLL as usize] = true,
            Action::LeaveShop => m.action_type[at::LEAVE_SHOP as usize] = true,
            Action::UseConsumable(i) => {
                if i < CONSUMABLE_SLOTS {
                    use_consumable_ok = true;
                }
            }
            Action::SellConsumable(i) => {
                if i < CONSUMABLE_SLOTS {
                    m.consumable_target[i] = true;
                    m.action_type[at::SELL_CONSUMABLE as usize] = true;
                }
            }
            Action::SellJoker(i) => {
                if i < JOKER_SLOTS {
                    m.joker_target[i] = true;
                    m.action_type[at::SELL_JOKER as usize] = true;
                }
            }
            Action::PickPackItem(i) => {
                if i < PACK_SLOTS {
                    m.pack_target[i] = true;
                    m.action_type[at::PICK_PACK as usize] = true;
                }
            }
            Action::SkipPack => m.action_type[at::SKIP_PACK as usize] = true,
            // RerollBoss (Director's Cut voucher) and BuyAndUseShopItem are
            // not representable in the v1 action space (documented contract
            // friction; the agent can buy-then-use instead).
            _ => {}
        }
    }
    m.action_type[at::BUY_SHOP as usize] = m.shop_target.iter().any(|&b| b);
    // USE_CONSUMABLE shares consumable_target_mask with SELL_CONSUMABLE, so
    // the mask marks every held (observable) consumable; the action type is
    // legal iff at least one of them is usable right now.
    m.action_type[at::USE_CONSUMABLE as usize] = use_consumable_ok;

    // Card sub-action availability: PLAY/DISCARD in a blind, plus
    // USE_CONSUMABLE candidate targets whenever a hand is visible
    // (SelectingHand, or a targeting pack state with a dealt hand).
    let cards_needed = m.action_type[at::PLAY_HAND as usize]
        || m.action_type[at::DISCARD as usize]
        || m.action_type[at::USE_CONSUMABLE as usize];
    if cards_needed {
        for s in m.card_select.iter_mut().take(hand_len) {
            *s = true;
        }
    }
    m
}

/// True if at least one contract action is legal (the M0 invariant). A
/// non-terminal state with no legal action is treated as a loss by the env
/// (theoretically reachable only by destroying the whole deck).
pub fn has_any_legal(run: &Run, slots: &[ShopSlotRef]) -> bool {
    encode_masks(run, slots).action_type.iter().any(|&b| b)
}
