//! Blind data, selection and boss-blind mechanics, ported from
//! `get_blind_amount` (functions/misc_functions.lua:919-954, scaling==1
//! branch — White Stake), the `P_BLINDS` prototypes (game.lua:263-296),
//! `get_new_boss` (functions/common_events.lua:2338-2383) and the `Blind`
//! class (blind.lua): `set_blind`, `press_play`, `modify_hand`,
//! `debuff_hand`, `debuff_card`, `stay_flipped`, `drawn_to_hand`, `disable`.
//!
//! Effects that need run-level state (The Hook's forced discards, The
//! Tooth's money drain, Cerulean Bell's forced selection, the Serpent's
//! draw-3, Water/Needle/Manacle round modifiers) are orchestrated from
//! run.rs; this module owns the per-blind state (`ActiveBlind`) and the
//! pure/blind-local rules.

use crate::cards::{Card, HandType, Suit};
use crate::rng::{LuaRandom, RngState};
use crate::scoring::HandsTable;
use std::collections::HashMap;

/// Lua's `a % b` for floats: `a - floor(a/b)*b`.
fn lua_fmod(a: f64, b: f64) -> f64 {
    a - (a / b).floor() * b
}

/// `get_blind_amount(ante)`, `G.GAME.modifiers.scaling == 1` branch
/// (misc_functions.lua:919-930). All arithmetic in f64 like Lua.
pub fn get_blind_amount(ante: i64) -> f64 {
    let k = 0.75f64;
    const AMOUNTS: [f64; 8] = [
        300.0, 800.0, 2000.0, 5000.0, 11000.0, 20000.0, 35000.0, 50000.0,
    ];
    if ante < 1 {
        return 100.0;
    }
    if ante <= 8 {
        return AMOUNTS[(ante - 1) as usize];
    }
    let a = AMOUNTS[7];
    let b = 1.6f64;
    let c = (ante - 8) as f64;
    let d = 1.0 + 0.2 * (ante - 8) as f64;
    // amount = math.floor(a*(b+(k*c)^d)^c)
    let mut amount = (a * (b + (k * c).powf(d)).powf(c)).floor();
    // amount = amount - amount % (10^math.floor(math.log10(amount)-1))
    amount -= lua_fmod(amount, 10f64.powf((amount.log10() - 1.0).floor()));
    amount
}

/// A `P_BLINDS.debuff` table entry (game.lua:263-296). Only the variants
/// present in the base pool are modeled (`value`/`nominal` debuffs are
/// unused there).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BossDebuff {
    None,
    /// `debuff = {suit = ...}`: The Club (Clubs), The Goad (Spades),
    /// The Head (Hearts), The Window (Diamonds).
    Suit(Suit),
    /// `debuff = {is_face = 'face'}`: The Plant.
    Face,
    /// `debuff = {h_size_ge = n}`: The Psychic (must play 5 cards —
    /// `#cards < n` debuffs the hand, blind.lua:527-530).
    HandSizeGe(usize),
}

/// One `P_BLINDS` prototype (the fields the sim needs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindProto {
    /// The P_BLINDS key, e.g. "bl_hook".
    pub key: &'static str,
    pub name: &'static str,
    pub dollars: i64,
    /// Chip multiplier; stored x2 to stay integral (bl_big is 1.5).
    mult_x2: i64,
    /// boss.min, or 0 for Small/Big.
    pub boss_min: i64,
    /// boss.showdown (ante-8 finisher bosses).
    pub showdown: bool,
    pub is_boss: bool,
    /// The static `debuff` table from P_BLINDS.
    pub debuff: BossDebuff,
}

impl BlindProto {
    pub fn mult(&self) -> f64 {
        self.mult_x2 as f64 / 2.0
    }

    /// `Blind:set_blind` chip requirement (blind.lua:107):
    /// `get_blind_amount(ante) * self.mult * G.GAME.starting_params.ante_scaling`
    /// (ante_scaling == 1 for the Red Deck / White Stake).
    pub fn chips(&self, ante: i64) -> f64 {
        get_blind_amount(ante) * self.mult() * 1.0
    }
}

/// Alias for static blind prototypes (see `items::Key` for why).
pub type BlindRef = &'static BlindProto;

const fn blind(
    key: &'static str,
    name: &'static str,
    dollars: i64,
    mult_x2: i64,
    boss_min: i64,
    showdown: bool,
    debuff: BossDebuff,
) -> BlindProto {
    BlindProto {
        key,
        name,
        dollars,
        mult_x2,
        boss_min,
        showdown,
        is_boss: boss_min > 0,
        debuff,
    }
}

pub const BL_SMALL: BlindProto = blind("bl_small", "Small Blind", 3, 2, 0, false, BossDebuff::None);
pub const BL_BIG: BlindProto = blind("bl_big", "Big Blind", 4, 3, 0, false, BossDebuff::None);

/// All boss prototypes from game.lua:266-294 (key, dollars, mult, boss.min,
/// showdown, debuff). Order here is irrelevant: `get_new_boss` sorts keys.
pub const BOSSES: [BlindProto; 28] = [
    blind("bl_ox", "The Ox", 5, 4, 6, false, BossDebuff::None),
    blind("bl_hook", "The Hook", 5, 4, 1, false, BossDebuff::None),
    blind("bl_mouth", "The Mouth", 5, 4, 2, false, BossDebuff::None),
    blind("bl_fish", "The Fish", 5, 4, 2, false, BossDebuff::None),
    blind(
        "bl_club",
        "The Club",
        5,
        4,
        1,
        false,
        BossDebuff::Suit(Suit::Clubs),
    ),
    blind(
        "bl_manacle",
        "The Manacle",
        5,
        4,
        1,
        false,
        BossDebuff::None,
    ),
    blind("bl_tooth", "The Tooth", 5, 4, 3, false, BossDebuff::None),
    blind("bl_wall", "The Wall", 5, 8, 2, false, BossDebuff::None),
    blind("bl_house", "The House", 5, 4, 2, false, BossDebuff::None),
    blind("bl_mark", "The Mark", 5, 4, 2, false, BossDebuff::None),
    blind(
        "bl_final_bell",
        "Cerulean Bell",
        8,
        4,
        10,
        true,
        BossDebuff::None,
    ),
    blind("bl_wheel", "The Wheel", 5, 4, 2, false, BossDebuff::None),
    blind("bl_arm", "The Arm", 5, 4, 2, false, BossDebuff::None),
    blind(
        "bl_psychic",
        "The Psychic",
        5,
        4,
        1,
        false,
        BossDebuff::HandSizeGe(5),
    ),
    blind(
        "bl_goad",
        "The Goad",
        5,
        4,
        1,
        false,
        BossDebuff::Suit(Suit::Spades),
    ),
    blind("bl_water", "The Water", 5, 4, 2, false, BossDebuff::None),
    blind("bl_eye", "The Eye", 5, 4, 3, false, BossDebuff::None),
    blind("bl_plant", "The Plant", 5, 4, 4, false, BossDebuff::Face),
    blind("bl_needle", "The Needle", 5, 2, 2, false, BossDebuff::None),
    blind(
        "bl_head",
        "The Head",
        5,
        4,
        1,
        false,
        BossDebuff::Suit(Suit::Hearts),
    ),
    blind(
        "bl_final_leaf",
        "Verdant Leaf",
        8,
        4,
        10,
        true,
        BossDebuff::None,
    ),
    blind(
        "bl_final_vessel",
        "Violet Vessel",
        8,
        12,
        10,
        true,
        BossDebuff::None,
    ),
    blind(
        "bl_window",
        "The Window",
        5,
        4,
        1,
        false,
        BossDebuff::Suit(Suit::Diamonds),
    ),
    blind(
        "bl_serpent",
        "The Serpent",
        5,
        4,
        5,
        false,
        BossDebuff::None,
    ),
    blind("bl_pillar", "The Pillar", 5, 4, 1, false, BossDebuff::None),
    blind("bl_flint", "The Flint", 5, 4, 2, false, BossDebuff::None),
    blind(
        "bl_final_acorn",
        "Amber Acorn",
        8,
        4,
        10,
        true,
        BossDebuff::None,
    ),
    blind(
        "bl_final_heart",
        "Crimson Heart",
        8,
        4,
        10,
        true,
        BossDebuff::None,
    ),
];

pub fn boss_by_key(key: &str) -> &'static BlindProto {
    BOSSES
        .iter()
        .find(|b| b.key == key)
        .expect("unknown boss key")
}

/// Any blind (Small/Big/boss) by key — snapshot re-interning (P6).
pub fn blind_by_key(key: &str) -> Option<&'static BlindProto> {
    match key {
        k if k == BL_SMALL.key => Some(&BL_SMALL),
        k if k == BL_BIG.key => Some(&BL_BIG),
        _ => BOSSES.iter().find(|b| b.key == key),
    }
}

/// `get_new_boss()` (common_events.lua:2338-2383) without the
/// perscribed/forced hooks (empty in a normal run).
///
/// Eligibility (common_events.lua:2350-2357):
/// - non-showdown: `boss.min <= max(1, ante)` and
///   `(max(1, ante) % win_ante ~= 0 or ante < 2)`
/// - showdown: `ante % win_ante == 0 and ante >= 2`
///
/// Then only bosses with the minimum `bosses_used` count stay, and
/// `pseudorandom_element(eligible, pseudoseed('boss'))` picks one:
/// keys sorted byte-wise ascending (values are use-count numbers, so the
/// `a.k < b.k` branch of misc_functions.lua:263 applies), then
/// `math.random(#keys)`.
pub fn get_new_boss(
    rng: &mut RngState,
    bosses_used: &mut HashMap<&'static str, i64>,
    ante: i64,
    win_ante: i64,
) -> &'static str {
    let ante_clamped = ante.max(1);
    let mut eligible: Vec<&'static str> = Vec::new();
    for b in &BOSSES {
        let ok = if !b.showdown {
            b.boss_min <= ante_clamped
                && (lua_fmod(ante_clamped as f64, win_ante as f64) != 0.0 || ante < 2)
        } else {
            lua_fmod(ante as f64, win_ante as f64) == 0.0 && ante >= 2
        };
        if ok {
            eligible.push(b.key);
        }
    }
    // (banned_keys is empty outside challenges.)

    // Keep only bosses at the minimum use count.
    let min_use = eligible
        .iter()
        .map(|k| *bosses_used.get(k).unwrap_or(&0))
        .fold(100i64, i64::min);
    eligible.retain(|k| *bosses_used.get(k).unwrap_or(&0) <= min_use);

    // pseudorandom_element: sort keys ascending, one math.random(#keys) draw.
    eligible.sort_unstable();
    let seed = rng.pseudoseed("boss");
    let j = LuaRandom::seeded(seed).random_range(1, eligible.len() as i64);
    let boss = eligible[(j - 1) as usize];
    *bosses_used.entry(boss).or_insert(0) += 1;
    boss
}

// ---------------------------------------------------------------------------
// Live blind state (the `Blind` object fields that matter for gameplay)
// ---------------------------------------------------------------------------

/// The per-round blind instance: `Blind:set_blind` state (blind.lua:78-216)
/// plus the boss-specific fields (`hands` for The Eye, `only_hand` for The
/// Mouth, `prepped` for The Fish, `discards_sub`/`hands_sub` for
/// Water/Needle).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(deserialize = "")))]
pub struct ActiveBlind {
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::blind_proto"))]
    pub proto: BlindRef,
    /// `self.chips` (blind.lua:107): get_blind_amount(ante)*mult*ante_scaling.
    pub chips: f64,
    /// `self.disabled` — set by boss-disable effects (Chicot/Director's Cut
    /// tags, P3b/P3c); every mechanic below checks it like blind.lua does.
    pub disabled: bool,
    /// `self.triggered` — UI flag, kept for observability.
    pub triggered: bool,
    /// `self.prepped`: set true in set_blind (blind.lua:94) then cleared for
    /// The Fish (blind.lua:176-178); press_play re-arms it (blind.lua:494-496)
    /// and `drawn_to_hand` always clears it (blind.lua:602).
    pub prepped: bool,
    /// The Eye's per-hand-type "already played" table (blind.lua:157-172),
    /// indexed by `HandType::ALL` position.
    pub eye_hands: [bool; 12],
    /// The Mouth's `only_hand` (blind.lua:173-175; false until first play).
    pub only_hand: Option<HandType>,
    /// The Water's stashed discards (blind.lua:179-182).
    pub discards_sub: i64,
    /// The Needle's stashed hands (blind.lua:183-186).
    pub hands_sub: i64,
}

impl ActiveBlind {
    /// `Blind:set_blind(blind)` — the state initialization part.
    /// The side effects on the round (Water zeroing discards, Needle leaving
    /// 1 hand, Manacle shrinking the hand) live in `Run::select_blind`.
    pub fn new(proto: &'static BlindProto, ante: i64) -> Self {
        ActiveBlind {
            proto,
            // blind.lua:107, ante_scaling == 1 (Red Deck).
            chips: proto.chips(ante),
            disabled: false,
            triggered: false,
            // blind.lua:94 sets prepped = true; blind.lua:176-178 clears it
            // for The Fish so the round's first deal is face up.
            prepped: proto.name != "The Fish",
            eye_hands: [false; 12],
            only_hand: None,
            discards_sub: 0,
            hands_sub: 0,
        }
    }

    fn is(&self, name: &str) -> bool {
        self.proto.name == name
    }

    /// `Blind:modify_hand` (blind.lua:510-517): The Flint halves base chips
    /// and mult with Lua `math.floor(x*0.5 + 0.5)` rounding, floored at
    /// 1 mult / 0 chips. Returns (mult, chips, modded).
    pub fn modify_hand(&mut self, mult: f64, hand_chips: f64) -> (f64, f64, bool) {
        if self.disabled {
            return (mult, hand_chips, false);
        }
        if self.is("The Flint") {
            self.triggered = true;
            return (
                (mult * 0.5 + 0.5).floor().max(1.0),
                (hand_chips * 0.5 + 0.5).floor().max(0.0),
                true,
            );
        }
        (mult, hand_chips, false)
    }

    /// `Blind:debuff_hand(cards, hand, handname)` (blind.lua:519-570), the
    /// non-`check` call from evaluate_play (state_events.lua:614). Returns
    /// true if the played hand is debuffed (scores 0). Side effects, exactly
    /// as in the Lua:
    /// - The Eye marks the hand type as played (blind.lua:535-541);
    /// - The Mouth locks in the first hand type (blind.lua:542-548);
    /// - The Arm levels the hand down *before* base chips/mult are read
    ///   (blind.lua:550-559 — evaluate_play re-reads G.GAME.hands at
    ///   state_events.lua:640-641 after this call);
    /// - The Ox sets dollars to 0 when the most played hand is played
    ///   (blind.lua:560-569, `ease_dollars(-G.GAME.dollars, true)`).
    pub fn debuff_hand(
        &mut self,
        played_count: usize,
        handname: HandType,
        hands: &mut HandsTable,
        dollars: &mut i64,
        most_played: HandType,
    ) -> bool {
        if self.disabled {
            return false;
        }
        // `if self.debuff then` — the debuff table is always non-nil, so this
        // block always runs for the base pool.
        self.triggered = false;
        // No base boss uses `debuff.hand`; h_size_le is unused too.
        if let BossDebuff::HandSizeGe(n) = self.proto.debuff {
            // blind.lua:527-530: `if self.debuff.h_size_ge and #cards <
            // self.debuff.h_size_ge`.
            if played_count < n {
                self.triggered = true;
                return true;
            }
        }
        if self.is("The Eye") {
            let idx = HandType::ALL.iter().position(|&h| h == handname).unwrap();
            if self.eye_hands[idx] {
                self.triggered = true;
                return true;
            }
            self.eye_hands[idx] = true;
        }
        if self.is("The Mouth") {
            if let Some(only) = self.only_hand {
                if only != handname {
                    self.triggered = true;
                    return true;
                }
            }
            self.only_hand = Some(handname);
        }
        if self.is("The Arm") {
            self.triggered = false;
            if hands.get(handname).level > 1 {
                self.triggered = true;
                // level_up_hand(..., -1) (common_events.lua:464-469).
                hands.level_up(handname, -1);
            }
        }
        if self.is("The Ox") {
            self.triggered = false;
            if handname == most_played {
                self.triggered = true;
                // `ease_dollars(-G.GAME.dollars, true)` — dollars become 0.
                *dollars = 0;
            }
        }
        false
    }

    /// `Blind:debuff_card(card)` (blind.lua:624-652) as a pure predicate for
    /// playing cards (`card.area ~= G.jokers`). run.rs applies the result to
    /// every playing card on set_blind/disable like blind.lua:207-213/407-412.
    pub fn debuff_card(&self, card: &Card, pareidolia: bool, smeared: bool) -> bool {
        if self.disabled {
            return false;
        }
        match self.proto.debuff {
            // blind.lua:626: `card:is_suit(self.debuff.suit, true)` — the
            // bypass_debuff branch, so Smeared widens suit bosses to the
            // whole colour.
            BossDebuff::Suit(s) if card.is_suit_bypass_debuff(s, smeared) => return true,
            // blind.lua:630: `self.debuff.is_face == 'face' and
            // card:is_face(true)` — Pareidolia makes The Plant debuff
            // everything (card.lua:964-970).
            BossDebuff::Face if card.is_face_from_boss(pareidolia) => return true,
            _ => {}
        }
        // blind.lua:634: The Pillar debuffs cards played this ante.
        if self.is("The Pillar") && card.played_this_ante {
            return true;
        }
        // blind.lua:650: Verdant Leaf debuffs ALL playing cards (until a
        // joker is sold, which disables it — P3b).
        if self.is("Verdant Leaf") {
            return true;
        }
        false
    }

    /// `Blind:stay_flipped(area, card)` for `area == G.hand`
    /// (blind.lua:605-622). Consumes the 'wheel' stream only for The Wheel
    /// (Lua `and` short-circuits on the name check).
    #[allow(clippy::too_many_arguments)]
    pub fn stay_flipped(
        &self,
        rng: &mut RngState,
        card: &Card,
        hands_played_round: i64,
        discards_used_round: i64,
        prob_normal: f64,
        pareidolia: bool,
    ) -> bool {
        if self.disabled {
            return false;
        }
        // blind.lua:608: 1 in 7 per card drawn to hand
        // (`pseudorandom(pseudoseed('wheel')) < G.GAME.probabilities.normal/7`
        // — Oops! All 6s doubles it).
        if self.is("The Wheel") && rng.random("wheel") < prob_normal / 7.0 {
            return true;
        }
        // blind.lua:611: the round's first deal is all face down.
        if self.is("The House") && hands_played_round == 0 && discards_used_round == 0 {
            return true;
        }
        // blind.lua:614: face cards drawn face down (is_face(true) —
        // Pareidolia flips everything).
        if self.is("The Mark") && card.is_face_from_boss(pareidolia) {
            return true;
        }
        // blind.lua:617: cards drawn after a played hand are face down
        // (press_play arms `prepped`, blind.lua:494-496).
        if self.is("The Fish") && self.prepped {
            return true;
        }
        false
    }

    /// `Blind:disable()` (blind.lua:356-415) — the blind-local parts.
    /// Restoring discards/hands/hand-size and flipping cards back up needs
    /// run state and is done by `Run::disable_boss`. Wall/Vessel chip
    /// reductions happen here.
    pub fn disable(&mut self) {
        self.disabled = true;
        if self.is("The Wall") {
            // blind.lua:377-380.
            self.chips /= 2.0;
        }
        if self.is("Violet Vessel") {
            // blind.lua:393-396.
            self.chips /= 3.0;
        }
    }
}
