//! Rust mirror of `train/balatro_train/encoding.py` (the FROZEN contract).
//! Every constant, index offset and enum ordering here must match that file
//! exactly — it is the single source of truth.

use balatro_core::cards::{Edition, Enhancement, HandType, Seal, Suit};
use balatro_core::items::{JokerId, BOOSTERS, PLANETS, SPECTRALS, TAROTS, VOUCHERS};

// ---------------------------------------------------------------- entity slots
pub const HAND_MAX: usize = 10;
pub const JOKER_SLOTS: usize = 6;
pub const CONSUMABLE_SLOTS: usize = 3;
pub const SHOP_SLOTS: usize = 6;
pub const PACK_SLOTS: usize = 5;

// ------------------------------------------------------------------ vocab sizes
pub const JOKER_VOCAB: i64 = 160;
pub const CONSUMABLE_VOCAB: i64 = 60;
pub const SHOP_VOCAB: i64 = 400;
pub const BOSS_VOCAB: usize = 32;

// -------------------------------------------------------------- card features
pub const CARD_RANK_OFF: usize = 0;
pub const CARD_SUIT_OFF: usize = 13;
pub const CARD_ENH_OFF: usize = 17;
pub const CARD_EDITION_OFF: usize = 26;
pub const CARD_SEAL_OFF: usize = 31;
pub const CARD_DEBUFFED_OFF: usize = 36;
pub const CARD_FACEDOWN_OFF: usize = 37;
pub const F_CARD: usize = 38;

// ------------------------------------------------------------- joker features
pub const JOKER_EDITION_OFF: usize = 0;
pub const JOKER_SELL_OFF: usize = 5;
pub const JOKER_STATE_OFF: usize = 6;
pub const N_JOKER_STATE: usize = 4;
pub const JOKER_DEBUFFED_OFF: usize = 10;
pub const F_JOKER_FEAT: usize = 11;

// -------------------------------------------------------------- shop features
/// `encoding.ShopKind` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShopKind {
    Joker = 0,
    Tarot = 1,
    Planet = 2,
    Spectral = 3,
    PlayingCard = 4,
    Voucher = 5,
    Pack = 6,
}
pub const SHOP_KIND_OFF: usize = 0;
pub const SHOP_COST_OFF: usize = 7;
pub const SHOP_EDITION_OFF: usize = 8;
pub const F_SHOP_FEAT: usize = 13;

// ------------------------------------------------------------- blind features
pub const BLIND_KIND_OFF: usize = 0;
pub const BLIND_BOSS_OFF: usize = 3;
pub const BLIND_REQ_OFF: usize = 35;
pub const BLIND_SCORED_OFF: usize = 36;
pub const BLIND_REWARD_OFF: usize = 37;
pub const F_BLIND: usize = 38;

// ------------------------------------------------------------- global features
/// `encoding.Phase` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    BlindSelect = 0,
    Playing = 1,
    RoundEval = 2,
    Shop = 3,
    Pack = 4,
    GameOver = 5,
}
pub const N_PHASES: usize = 6;
pub const N_HAND_TYPES: usize = 12;

pub const GLOBAL_MONEY: usize = 0;
pub const GLOBAL_HANDS_LEFT: usize = 1;
pub const GLOBAL_DISCARDS_LEFT: usize = 2;
pub const GLOBAL_ANTE_OFF: usize = 3;
pub const N_ANTE_ONEHOT: usize = 9;
pub const GLOBAL_ROUND: usize = 12;
pub const GLOBAL_PHASE_OFF: usize = 13;
pub const GLOBAL_HAND_LEVELS_OFF: usize = 19;
pub const GLOBAL_REROLL_COST: usize = 43;
pub const GLOBAL_JOKER_SLOT_COUNT: usize = 44;
pub const GLOBAL_CONSUMABLE_SLOT_COUNT: usize = 45;
pub const GLOBAL_DECK_SIZE: usize = 46;
pub const GLOBAL_HAND_SIZE: usize = 47;
pub const F_GLOBAL: usize = 64;

// ---------------------------------------------------------- deck composition
pub const N_DECK_SLOTS: usize = 52;
pub const DECK_AGG_ENH_OFF: usize = 0;
pub const DECK_AGG_EDITION_OFF: usize = 9;
pub const DECK_AGG_SEAL_OFF: usize = 13;
pub const DECK_AGG_TOTAL: usize = 17;
pub const F_DECK_AGG: usize = 18;

// ---------------------------------------------------------------- action space
/// `encoding.ActionType` indices (stable forever).
pub mod action_type {
    pub const PLAY_HAND: i64 = 0;
    pub const DISCARD: i64 = 1;
    pub const SELECT_BLIND: i64 = 2;
    pub const SKIP_BLIND: i64 = 3;
    pub const CASH_OUT: i64 = 4;
    pub const BUY_SHOP: i64 = 5;
    pub const REROLL: i64 = 6;
    pub const LEAVE_SHOP: i64 = 7;
    pub const USE_CONSUMABLE: i64 = 8;
    pub const SELL_JOKER: i64 = 9;
    pub const SELL_CONSUMABLE: i64 = 10;
    pub const PICK_PACK: i64 = 11;
    pub const SKIP_PACK: i64 = 12;
    // 13 = MOVE_JOKER: reserved, never legal in phase 1.
}
pub const N_ACTION_TYPES: usize = 25;
pub const MAX_CARD_PICKS: usize = 5;

// ------------------------------------------------------------------- helpers

/// `log1p` in f32 domain (computed in f64 like numpy, then cast).
pub fn log1p(x: f64) -> f32 {
    x.ln_1p() as f32
}

/// `sign(x) * log1p(|x|)`.
pub fn slog1p(x: f64) -> f32 {
    (x.signum() * x.abs().ln_1p()) as f32
}

pub fn suit_index(s: Suit) -> usize {
    // Contract Suit order == sim Suit order (Spades, Hearts, Clubs, Diamonds).
    match s {
        Suit::Spades => 0,
        Suit::Hearts => 1,
        Suit::Clubs => 2,
        Suit::Diamonds => 3,
    }
}

/// Contract rank one-hot index: 2 -> 0 ... Ace(14) -> 12.
pub fn rank_index(id: u8) -> usize {
    (id - 2) as usize
}

pub fn enhancement_index(e: Enhancement) -> usize {
    // Contract Enhancement order == sim order.
    e as usize
}

pub fn edition_index(e: Edition) -> usize {
    // Contract Edition order == sim order (Holo == HOLOGRAPHIC).
    e as usize
}

pub fn seal_index(s: Seal) -> usize {
    s as usize
}

/// Canonical hand-type row order of the global hand-level table (the Rust
/// side owns this order per the contract): `HandType::ALL`, High Card ..
/// Flush Five.
pub fn hand_type_index(ht: HandType) -> usize {
    HandType::ALL.iter().position(|&h| h == ht).unwrap()
}

/// deck_counts / drawpile_counts slot: rank * 4 + suit.
pub fn deck_slot(rank_id: u8, suit: Suit) -> usize {
    rank_index(rank_id) * 4 + suit_index(suit)
}

// ------------------------------------------------------------- id vocabularies
//
// Shared SHOP_VOCAB layout (0 = EMPTY):
//   1..=150    jokers        (1 + JokerId discriminant)
//   160..=211  consumables   (160 + consumable index: tarots 0..21,
//                             planets 22..33, spectrals 34..51)
//   220..=271  playing cards (220 + deck_slot(rank, suit))
//   280..=311  vouchers      (280 + index in items::VOUCHERS)
//   320..=351  boosters      (320 + index in items::BOOSTERS)
//
// consumable_ids uses CONSUMABLE_VOCAB: 1 + consumable index (1..=52).
// joker_ids uses JOKER_VOCAB: 1 + JokerId discriminant (1..=150).

pub fn joker_vocab_id(id: JokerId) -> i64 {
    1 + id as i64
}

/// Index of a consumable key in the tarot/planet/spectral registries
/// (tarots 0..21, planets 22..33, spectrals 34..51).
pub fn consumable_index(key: &str) -> Option<i64> {
    if let Some(i) = TAROTS.iter().position(|c| c.key == key) {
        return Some(i as i64);
    }
    if let Some(i) = PLANETS.iter().position(|c| c.key == key) {
        return Some(22 + i as i64);
    }
    SPECTRALS
        .iter()
        .position(|c| c.key == key)
        .map(|i| 34 + i as i64)
}

pub fn consumable_vocab_id(key: &str) -> i64 {
    1 + consumable_index(key).unwrap_or(-1).max(0)
}

pub fn shop_vocab_joker(id: JokerId) -> i64 {
    1 + id as i64
}

pub fn shop_vocab_consumable(key: &str) -> i64 {
    160 + consumable_index(key).unwrap_or(0)
}

pub fn shop_vocab_card(rank_id: u8, suit: Suit) -> i64 {
    220 + deck_slot(rank_id, suit) as i64
}

pub fn shop_vocab_voucher(key: &str) -> i64 {
    280 + VOUCHERS.iter().position(|v| v.key == key).unwrap_or(0) as i64
}

pub fn shop_vocab_pack(key: &str) -> i64 {
    320 + BOOSTERS.iter().position(|b| b.key == key).unwrap_or(0) as i64
}

/// Boss-blind one-hot id (1..=28; 0 = not a boss).
pub fn boss_onehot_id(key: &str) -> usize {
    1 + balatro_core::blinds::BOSSES
        .iter()
        .position(|b| b.key == key)
        .unwrap_or(0)
}
