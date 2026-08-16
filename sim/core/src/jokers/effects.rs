//! Per-joker effect branches, ported from `Card:calculate_joker`
//! (card.lua:2291-4056) and `Card:calculate_dollar_bonus`
//! (card.lua:1655-1679): every rarity 1-4 joker in the registry.
//!
//! Every function takes the RESOLVED target joker (after Blueprint/
//! Brainstorm resolution) and the `blueprint` context flag; branches the Lua
//! wraps in `not context.blueprint` are gated on `!bp` here, everything else
//! runs for copies too — exactly like the game.

use crate::cards::{Card, Enhancement, HandType, Suit};
use crate::items::{element_index, ConsumableSet, JokerId};
use crate::rng::RngState;
use crate::run::MOST_PLAYED_SCAN;
use crate::scoring::{BeforeCtx, CardEffect, HandCtx, HandsTable};
use crate::shop::OwnedJoker;

use super::{card_is_face, JokerEnv, JokerOutbox, MainEffect};

/// `ability.type` (`center.config.type`, game.lua:373-382) for the
/// hand-type-gated jokers; `''` (None) otherwise.
pub(crate) fn joker_type(id: JokerId) -> Option<HandType> {
    Some(match id {
        // j_jolly / j_sly — 'Pair'
        JokerId::Jolly | JokerId::Sly => HandType::Pair,
        // j_zany / j_wily — 'Three of a Kind'
        JokerId::Zany | JokerId::Wily => HandType::ThreeOfAKind,
        // j_mad / j_clever — 'Two Pair'
        JokerId::Mad | JokerId::Clever => HandType::TwoPair,
        // j_crazy / j_devious — 'Straight'
        JokerId::Crazy | JokerId::Devious => HandType::Straight,
        // j_droll / j_crafty — 'Flush'
        JokerId::Droll | JokerId::Crafty => HandType::Flush,
        // P3c-2 typed jokers (Duo/Trio/Family/Order/Tribe) also use this.
        JokerId::Duo => HandType::Pair,
        JokerId::Trio => HandType::ThreeOfAKind,
        JokerId::Family => HandType::FourOfAKind,
        JokerId::Order => HandType::Straight,
        JokerId::Tribe => HandType::Flush,
        _ => return None,
    })
}

/// `ability.t_mult` (game.lua:373-377).
fn t_mult(id: JokerId) -> f64 {
    match id {
        JokerId::Jolly => 8.0,
        JokerId::Zany | JokerId::Crazy => 12.0,
        JokerId::Mad | JokerId::Droll => 10.0,
        _ => 0.0,
    }
}

/// `ability.t_chips` (game.lua:378-382).
fn t_chips(id: JokerId) -> f64 {
    match id {
        JokerId::Sly => 50.0,
        JokerId::Wily | JokerId::Devious => 100.0,
        JokerId::Clever | JokerId::Crafty => 80.0,
        _ => 0.0,
    }
}

/// `effect == 'Suit Mult'` jokers: `extra = {s_mult = 3, suit = …}`
/// (game.lua:369-372).
fn suit_mult_suit(id: JokerId) -> Option<Suit> {
    Some(match id {
        JokerId::GreedyJoker => Suit::Diamonds,
        JokerId::LustyJoker => Suit::Hearts,
        JokerId::WrathfulJoker => Suit::Spades,
        JokerId::GluttenousJoker => Suit::Clubs,
        _ => return None,
    })
}

/// The 'to_do' poker-hand roll at Card creation (`Card:set_ability`,
/// card.lua:311-322): candidates are every VISIBLE hand in
/// `pairs(G.GAME.hands)` order (= MOST_PLAYED_SCAN, oracle-verified); the
/// draw repeats while it matches `old` (never on a fresh card, where old is
/// nil).
pub(crate) fn roll_to_do_hand(
    rng: &mut RngState,
    hands: &HandsTable,
    old: Option<HandType>,
) -> HandType {
    let pool: Vec<HandType> = MOST_PLAYED_SCAN
        .iter()
        .copied()
        .filter(|&ht| hands.get(ht).visible)
        .collect();
    loop {
        let pick = pool[element_index(rng, "to_do", pool.len())];
        if Some(pick) != old {
            return pick;
        }
    }
}

/// What a `{before = true}` eval asks the dispatcher to do afterwards.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct BeforeResult {
    /// `effects.jokers.level_up` — Space Joker (state_events.lua:634-637).
    pub level_up: bool,
    /// DNA fired: copy `full_hand[1]` into the hand (card.lua:3501-3524).
    pub dna_copy: bool,
}

/// `{before = true}` (card.lua:3411-3569), branch order as in the Lua.
#[allow(clippy::too_many_arguments)]
pub(super) fn before_hand(
    j: &mut OwnedJoker,
    bp: bool,
    ctx: &mut BeforeCtx,
    pareidolia: bool,
    _smeared: bool,
    hands: &HandsTable,
    env: &JokerEnv,
    out: &mut JokerOutbox,
    dollars: &mut i64,
    rng: &mut RngState,
) -> BeforeResult {
    let mut res = BeforeResult::default();
    match j.id {
        // card.lua:3412-3419 — Spare Trousers: hand CONTAINS a Two Pair (a
        // Full House also registers one). Blueprint-gated.
        JokerId::Trousers if !bp => {
            if !ctx.poker_hands.get(HandType::TwoPair).is_empty()
                || !ctx.poker_hands.get(HandType::FullHouse).is_empty()
            {
                j.state.mult += 2.0; // extra (game.lua:485)
            }
        }
        // card.lua:3420-3426 — Space Joker: 'space' < normal/4 levels the
        // played hand. NOT blueprint-gated: copies roll separately.
        JokerId::Space => {
            if rng.random("space") < env.prob_normal / 4.0 {
                res.level_up = true;
            }
        }
        // card.lua:3427-3434 — Square Joker: played EXACTLY 4 cards.
        JokerId::Square if !bp => {
            if ctx.full_hand.len() == 4 {
                j.state.chips += 4.0; // extra.chip_mod (game.lua:461)
            }
        }
        // card.lua:3435-3442 — Runner: hand contains a Straight.
        JokerId::Runner if !bp => {
            if !ctx.poker_hands.get(HandType::Straight).is_empty() {
                j.state.chips += 15.0; // extra.chip_mod (game.lua:428)
            }
        }
        // card.lua:3443-3464 — Midas Mask: every scored face becomes a Gold
        // card (set_ability m_gold — the enhancement is REPLACED).
        // Blueprint-gated.
        JokerId::MidasMask if !bp => {
            for &i in ctx.scoring {
                if card_is_face(&ctx.full_hand[i], pareidolia) {
                    ctx.full_hand[i].enhancement = Enhancement::Gold;
                }
            }
        }
        // card.lua:3465-3490 — Vampire: strips every scored non-debuffed
        // enhancement (center != c_base) back to base and gains extra (0.1)
        // x_mult per card. Blueprint-gated.
        JokerId::Vampire if !bp => {
            let mut enhanced = 0usize;
            for &i in ctx.scoring {
                let c = &mut ctx.full_hand[i];
                if c.enhancement != Enhancement::None && !c.debuff {
                    enhanced += 1;
                    c.enhancement = Enhancement::None;
                }
            }
            if enhanced > 0 {
                j.state.x_mult += 0.1 * enhanced as f64;
            }
        }
        // card.lua:3491-3500 — To Do List: played hand matches; pays inline
        // via ease_dollars + dollar_buffer (NOT blueprint-gated).
        JokerId::TodoList => {
            if ctx.hand_type == j.state.to_do_hand {
                *dollars += 4; // extra.dollars (game.lua:454)
                out.dollar_buffer += 4;
            }
        }
        // card.lua:3501-3524 — DNA: the round's first hand, exactly 1 card
        // played -> copy it into the hand. NOT blueprint-gated.
        JokerId::Dna => {
            if env.hands_played_round == 0 && ctx.full_hand.len() == 1 {
                res.dna_copy = true;
            }
        }
        // card.lua:3525-3542 — Ride the Bus: reset on any scored face,
        // else +1 mult per hand.
        JokerId::RideTheBus if !bp => {
            let faces = ctx
                .scoring
                .iter()
                .any(|&i| card_is_face(&ctx.full_hand[i], pareidolia));
            if faces {
                j.state.mult = 0.0;
            } else {
                j.state.mult += 1.0; // extra (game.lua:424)
            }
        }
        // card.lua:3543-3562 — Obelisk: reset to X1 when the played hand is
        // (tied for) most played, else +0.2 x_mult. The current play already
        // counts (hands[..].played incremented at state_events.lua:574).
        JokerId::Obelisk if !bp => {
            let play_more_than = hands.get(ctx.hand_type).played;
            let mut reset = true;
            for ht in MOST_PLAYED_SCAN {
                if ht == ctx.hand_type {
                    continue;
                }
                let row = hands.get(ht);
                if row.played >= play_more_than && row.visible {
                    reset = false;
                }
            }
            if reset {
                if j.state.x_mult > 1.0 {
                    j.state.x_mult = 1.0;
                }
            } else {
                j.state.x_mult += 0.2; // extra (game.lua:462)
            }
        }
        // card.lua:3563-3569 — Green Joker: +1 mult per hand played.
        JokerId::GreenJoker if !bp => {
            j.state.mult += 1.0; // extra.hand_add (game.lua:451)
        }
        _ => {}
    }
    res
}

/// `{repetition = true, cardarea = G.play}` (card.lua:3343-3385). Returns
/// extra repetitions. None of these branches is blueprint-gated.
pub(super) fn repetition_play(
    j: &OwnedJoker,
    ctx: &HandCtx,
    card: &Card,
    pareidolia: bool,
    env: &JokerEnv,
    _rng: &mut RngState,
) -> u32 {
    match j.id {
        // card.lua:3344-3351 — Sock and Buskin: every scored face, 1 extra
        // repetition (extra = 1, game.lua:497).
        JokerId::SockAndBuskin => {
            if card_is_face(card, pareidolia) {
                1
            } else {
                0
            }
        }
        // card.lua:3352-3358 — Hanging Chad: the FIRST scoring card, 2
        // extra repetitions (extra = 2, game.lua:486).
        JokerId::HangingChad => {
            let first = ctx.scoring.first().map(|&i| ctx.full_hand[i].sort_id);
            if first == Some(card.sort_id) {
                2
            } else {
                0
            }
        }
        // card.lua:3360-3366 — Dusk: the round's final hand (hands_left is
        // decremented before evaluate_play), extra = 1 (game.lua:389).
        JokerId::Dusk => {
            if env.hands_left == 0 {
                1
            } else {
                0
            }
        }
        // card.lua:3367-3373 — Seltzer: every scored card, hardcoded 1.
        JokerId::Selzer => 1,
        // card.lua:3374-3384 — Hack: scored 2/3/4/5, extra = 1 (game.lua:414).
        JokerId::Hack => {
            if matches!(card.id_for_eval(), Some(2..=5)) {
                1
            } else {
                0
            }
        }
        _ => 0,
    }
}

/// `{individual = true, cardarea = G.play}` (card.lua:3065-3270). One
/// effects-table entry per triggering joker; entries with only a message
/// still count (8 Ball/Hiker/Lucky Cat/Wee).
#[allow(clippy::too_many_arguments)]
pub(super) fn individual_play(
    j: &mut OwnedJoker,
    bp: bool,
    ctx: &HandCtx,
    card: &Card,
    lucky_trigger: bool,
    pareidolia: bool,
    smeared: bool,
    env: &JokerEnv,
    out: &mut JokerOutbox,
    rng: &mut RngState,
) -> Option<CardEffect> {
    let id = card.id_for_eval(); // get_id(); None models the Stone negative
    let mut eff = CardEffect::default();
    match j.id {
        // card.lua:3067-3075 — Hiker: +5 permanent chips on every scored
        // card, every repetition. NOT blueprint-gated.
        JokerId::Hiker => {
            eff.perma_bonus = 5; // extra (game.lua:449)
            return Some(eff);
        }
        // card.lua:3076-3082 — Lucky Cat: +0.25 x_mult per Lucky proc
        // (card.lucky_trigger set by this repetition's rolls).
        // Blueprint-gated.
        JokerId::LuckyCat if !bp => {
            if lucky_trigger {
                j.state.x_mult += 0.25; // extra (game.lua:437)
                return Some(eff); // message-only entry
            }
        }
        // card.lua:3083-3092 — Wee Joker: +8 chips per scored 2, per
        // repetition. Blueprint-gated.
        JokerId::Wee if !bp => {
            if id == Some(2) {
                j.state.chips += 8.0; // extra.chip_mod (game.lua:511)
                return Some(eff); // message-only entry
            }
        }
        // card.lua:3127-3135 — The Idol: X2 on scored cards matching this
        // round's idol rank+suit (is_suit: Wild matches, Smeared widens).
        JokerId::Idol => {
            let (rank, suit) = env.idol_card;
            if id == Some(rank) && card.is_suit_normal(suit, smeared) {
                eff.x_mult = 2.0; // extra (game.lua:515)
                return Some(eff);
            }
        }
        // card.lua:3185-3195 — Fibonacci: A,2,3,5,8 give +8 mult.
        JokerId::Fibonacci => {
            if matches!(id, Some(2) | Some(3) | Some(5) | Some(8) | Some(14)) {
                eff.mult = 8.0; // extra (game.lua:395)
                return Some(eff);
            }
        }
        // card.lua:3224-3232 — Rough Gem: scored Diamonds pay $1 (buffered).
        JokerId::RoughGem => {
            if card.is_suit_normal(Suit::Diamonds, smeared) {
                eff.dollars = 1; // extra (game.lua:501)
                out.dollar_buffer += 1;
                return Some(eff);
            }
        }
        // card.lua:3233-3239 — Onyx Agate: scored Clubs give +7 mult.
        JokerId::OnyxAgate => {
            if card.is_suit_normal(Suit::Clubs, smeared) {
                eff.mult = 7.0; // extra (game.lua:504)
                return Some(eff);
            }
        }
        // card.lua:3240-3246 — Arrowhead: scored Spades give +50 chips.
        JokerId::Arrowhead => {
            if card.is_suit_normal(Suit::Spades, smeared) {
                eff.chips = 50.0; // extra (game.lua:503)
                return Some(eff);
            }
        }
        // card.lua:3247-3254 — Bloodstone: scored Hearts X1.5 half the time
        // ('bloodstone' < normal/2).
        JokerId::Bloodstone => {
            if card.is_suit_normal(Suit::Hearts, smeared)
                && rng.random("bloodstone") < env.prob_normal / 2.0
            {
                eff.x_mult = 1.5; // extra.Xmult (game.lua:502)
                return Some(eff);
            }
        }
        // card.lua:3255-3261 — Ancient Joker: X1.5 on this round's suit.
        JokerId::Ancient => {
            if let Some(s) = env.ancient_suit {
                if card.is_suit_normal(s, smeared) {
                    eff.x_mult = 1.5; // extra (game.lua:489)
                    return Some(eff);
                }
            }
        }
        // card.lua:3262-3269 — Triboulet: scored Kings and Queens X2.
        JokerId::Triboulet => {
            if matches!(id, Some(12) | Some(13)) {
                eff.x_mult = 2.0; // extra (game.lua:522)
                return Some(eff);
            }
        }
        // card.lua:3086-3097 — Photograph: x2 on the first scored face.
        JokerId::Photograph => {
            let first_face = ctx
                .scoring
                .iter()
                .map(|&i| &ctx.full_hand[i])
                .find(|c| card_is_face(c, pareidolia));
            if first_face.map(|c| c.sort_id) == Some(card.sort_id) {
                eff.x_mult = 2.0; // extra (game.lua:465)
                return Some(eff);
            }
        }
        // card.lua:3104-3124 — 8 Ball: slot gate FIRST, then the 8 check
        // and the 1-in-4 roll; queues a 'Tarot'/'8ba' creation.
        JokerId::EightBall => {
            if env.consumables_len + out.consumable_buffer() < env.consumable_slots
                && id == Some(8)
                && rng.random("8ball") < env.prob_normal / 4.0
            {
                out.create_consumable(ConsumableSet::Tarot, "8ba");
                // The returned table only carries the creation func/message;
                // it still occupies an effects entry.
                return Some(eff);
            }
        }
        // card.lua:3135-3141 — Scary Face: +30 chips per scored face.
        JokerId::ScaryFace => {
            if card_is_face(card, pareidolia) {
                eff.chips = 30.0; // extra (game.lua:398)
                return Some(eff);
            }
        }
        // card.lua:3142-3148 — Smiley Face: +5 mult per scored face.
        JokerId::Smiley => {
            if card_is_face(card, pareidolia) {
                eff.mult = 5.0; // extra (game.lua:479)
                return Some(eff);
            }
        }
        // card.lua:3150-3158 — Golden Ticket: $4 per scored Gold card
        // (dollar_buffer maintained, card.lua:3152).
        JokerId::Ticket => {
            if card.enhancement == Enhancement::Gold {
                eff.dollars = 4; // extra (game.lua:482)
                out.dollar_buffer += 4;
                return Some(eff);
            }
        }
        // card.lua:3158-3165 — Scholar: Aces give +20 chips +4 mult.
        JokerId::Scholar => {
            if id == Some(14) {
                eff.chips = 20.0;
                eff.mult = 4.0; // extra (game.lua:410)
                return Some(eff);
            }
        }
        // card.lua:3166-3173 — Walkie Talkie: 10s and 4s, +10 chips +4 mult.
        JokerId::WalkieTalkie => {
            if id == Some(10) || id == Some(4) {
                eff.chips = 10.0;
                eff.mult = 4.0; // extra (game.lua:480)
                return Some(eff);
            }
        }
        // card.lua:3175-3184 — Business Card: faces pay $2 half the time
        // (dollar_buffer maintained, card.lua:3178).
        JokerId::Business => {
            if card_is_face(card, pareidolia) && rng.random("business") < env.prob_normal / 2.0 {
                eff.dollars = 2; // literal 2 (card.lua:3181)
                out.dollar_buffer += 2;
                return Some(eff);
            }
        }
        // card.lua:3196-3205 — Even Steven: 10,8,6,4,2 give +4 mult.
        // (Lua: id <= 10 and id >= 0 and id % 2 == 0 — a Stone card's
        // negative id fails `>= 0`, modeled by id_for_eval = None.)
        JokerId::EvenSteven => {
            if let Some(n) = id {
                if n <= 10 && n % 2 == 0 {
                    eff.mult = 4.0; // extra (game.lua:407)
                    return Some(eff);
                }
            }
        }
        // card.lua:3206-3217 — Odd Todd: A,9,7,5,3 give +31 chips.
        JokerId::OddTodd => {
            if let Some(n) = id {
                if (n <= 10 && n % 2 == 1) || n == 14 {
                    eff.chips = 31.0; // extra (game.lua:408)
                    return Some(eff);
                }
            }
        }
        _ => {
            // card.lua:3217-3223 — effect == 'Suit Mult' (Greedy/Lusty/
            // Wrathful/Gluttonous): +3 mult per scored card of the suit
            // (Card:is_suit — Wild matches anything, Smeared widens).
            if let Some(suit) = suit_mult_suit(j.id) {
                if card.is_suit_normal(suit, smeared) {
                    eff.mult = 3.0; // extra.s_mult (game.lua:369-372)
                    return Some(eff);
                }
            }
        }
    }
    None
}

/// `{individual = true, cardarea = G.hand}` (card.lua:3271-3341 common
/// branches). Runs for debuffed held cards too — the branches carry their
/// own debuff handling (a message-only entry, which still counts toward the
/// red-seal `#effects > 1` retrigger check).
pub(super) fn individual_held(
    j: &OwnedJoker,
    ctx: &HandCtx,
    card: &Card,
    pareidolia: bool,
    env: &JokerEnv,
    out: &mut JokerOutbox,
    rng: &mut RngState,
) -> Option<CardEffect> {
    let mut eff = CardEffect::default();
    match j.id {
        // card.lua:3272-3286 — Shoot the Moon: each held Queen +13 mult
        // (hardcoded 13; a debuffed Queen returns the message-only entry).
        JokerId::ShootTheMoon => {
            if card.rank.id() == 12 && card.enhancement != Enhancement::Stone {
                if !card.debuff {
                    eff.h_mult = 13.0; // card.lua:3282
                }
                return Some(eff);
            }
        }
        // card.lua:3287-3301 — Baron: each held King X1.5 (a debuffed King
        // returns the message-only entry).
        JokerId::Baron => {
            if card.rank.id() == 13 && card.enhancement != Enhancement::Stone {
                if !card.debuff {
                    eff.x_mult = 1.5; // extra (game.lua:490)
                }
                return Some(eff);
            }
        }
        // card.lua:3302-3319 — Reserved Parking: held faces pay $1 half the
        // time. is_face() is nil for debuffed cards, short-circuiting BEFORE
        // the 'parking' roll — debuffed faces consume no draw.
        JokerId::ReservedParking => {
            if card_is_face(card, pareidolia) && rng.random("parking") < env.prob_normal / 2.0 {
                // extra = {odds = 2, dollars = 1} (game.lua:466);
                // dollar_buffer maintained (card.lua:3312).
                eff.dollars = 1;
                out.dollar_buffer += 1;
                return Some(eff);
            }
        }
        // card.lua:3320-3340 — Raised Fist: 2x the rank value of the lowest
        // held card, attached to that card's evaluation. The scan keeps the
        // LAST hand-order card at the minimum id (`temp_ID >= base.id`),
        // skipping Stone cards; the nominal is the rank chip value.
        JokerId::RaisedFist => {
            let mut temp_mult = 15.0f64;
            let mut temp_id = 15u8;
            let mut raised: Option<u32> = None;
            for c in ctx.held {
                if temp_id >= c.rank.id() && c.enhancement != Enhancement::Stone {
                    temp_mult = c.rank.nominal() as f64;
                    temp_id = c.rank.id();
                    raised = Some(c.sort_id);
                }
            }
            if raised == Some(card.sort_id) {
                if !card.debuff {
                    eff.h_mult = 2.0 * temp_mult; // card.lua:3333
                }
                return Some(eff); // debuffed: message-only entry
            }
        }
        _ => {}
    }
    None
}

/// `{joker_main = true}` — the `else / context.cardarea == G.jokers` tail of
/// calculate_joker (card.lua:3631-4056), one branch per joker, ported in the
/// exact elseif order (the generic x_mult branch intercepts every joker whose
/// live x_mult exceeds 1 — including Throwback/Stencil, whose values
/// `live_x_mult` recomputes like Card:update). Field application order lives
/// in `MainEffect::apply`.
#[allow(clippy::too_many_arguments)]
pub(super) fn joker_main(
    j: &mut OwnedJoker,
    _bp: bool,
    ctx: &HandCtx,
    hands: &HandsTable,
    njokers: usize,
    swashbuckler_mult: f64,
    live_x_mult: f64,
    smeared: bool,
    blind_triggered: bool,
    env: &JokerEnv,
    out: &mut JokerOutbox,
    dollars: &mut i64,
    rng: &mut RngState,
) -> Option<MainEffect> {
    let mut eff = MainEffect::default();

    // card.lua:3632-3652 — Loyalty Card: recompute loyalty_remaining from
    // the run-total hands played (pre-increment during evaluation) and X4
    // every 6th hand. The blueprint and direct variants share the formula.
    if j.id == JokerId::LoyaltyCard {
        // (every - 1 - (hands_played - at_create)) % (every + 1), every = 5.
        let n = env.hands_played_total - j.hands_at_create;
        let loyalty_remaining = (4 - n).rem_euclid(6);
        if loyalty_remaining == 5 {
            eff.x_mult_mod = 4.0; // extra.Xmult (game.lua:392)
            return Some(eff);
        }
        return None;
    }

    // card.lua:3653-3659 — the generic accumulated-x_mult branch: every
    // joker except Seeing Double whose (live) x_mult exceeds 1, gated on the
    // typed jokers' hand containment (Duo..Tribe).
    if j.id != JokerId::SeeingDouble && live_x_mult > 1.0 {
        let type_ok = match joker_type(j.id) {
            None => true,
            Some(ht) => !ctx.poker_hands.get(ht).is_empty(),
        };
        if type_ok {
            eff.x_mult_mod = live_x_mult;
            return Some(eff);
        }
    }
    // card.lua:3661-3666 — t_mult jokers (Jolly..Droll): the hand only has
    // to CONTAIN the type (`next(context.poker_hands[type])`).
    if t_mult(j.id) > 0.0 && !ctx.poker_hands.get(joker_type(j.id).unwrap()).is_empty() {
        eff.mult_mod = t_mult(j.id);
        return Some(eff);
    }
    // card.lua:3667-3672 — t_chips jokers (Sly..Crafty).
    if t_chips(j.id) > 0.0 && !ctx.poker_hands.get(joker_type(j.id).unwrap()).is_empty() {
        eff.chip_mod = t_chips(j.id);
        return Some(eff);
    }

    match j.id {
        // card.lua:3673-3678 — Half Joker: 3 or fewer cards PLAYED.
        JokerId::Half => {
            if ctx.full_hand.len() <= 3 {
                eff.mult_mod = 20.0; // extra = {mult = 20, size = 3}
                return Some(eff);
            }
        }
        // card.lua:3679-3688 — Abstract: +3 mult per joker in the area
        // (debuffed jokers still count; this joker itself included).
        JokerId::Abstract => {
            eff.mult_mod = 3.0 * njokers as f64; // extra (game.lua:399)
            return Some(eff);
        }
        // card.lua:3688-3693 — Acrobat: X3 on the round's final hand.
        JokerId::Acrobat => {
            if env.hands_left == 0 {
                eff.x_mult_mod = 3.0; // extra (game.lua:496)
                return Some(eff);
            }
        }
        // card.lua:3695-3700 — Mystic Summit: +15 mult at 0 discards left.
        JokerId::MysticSummit => {
            if env.discards_left == 0 {
                eff.mult_mod = 15.0; // extra.mult, d_remaining = 0
                return Some(eff);
            }
        }
        // card.lua:3701-3707 — Misprint: pseudorandom('misprint', 0, 23).
        JokerId::Misprint => {
            eff.mult_mod = rng.random_range("misprint", 0, 23) as f64;
            return Some(eff);
        }
        // card.lua:3708-3713 — Banner: +30 chips per remaining discard.
        JokerId::Banner => {
            if env.discards_left > 0 {
                eff.chip_mod = 30.0 * env.discards_left as f64; // extra
                return Some(eff);
            }
        }
        // card.lua:3713-3718 — Stuntman: flat +250 chips.
        JokerId::Stuntman => {
            eff.chip_mod = 250.0; // extra.chip_mod (game.lua:524)
            return Some(eff);
        }
        // card.lua:3719-3730 — Matador: $8 when the boss triggered this
        // hand (inline ease_dollars + dollar_buffer; the returned `dollars`
        // field is NOT consumed by the joker_main phase,
        // state_events.lua:905-916 — single payment).
        JokerId::Matador => {
            if blind_triggered {
                *dollars += 8; // extra (game.lua:518)
                out.dollar_buffer += 8;
                return Some(eff); // message-only for scoring purposes
            }
        }
        // card.lua:3731-3736 — Supernova: mult = times the hand has been
        // played this run (already counting this play, state_events.lua:574).
        JokerId::Supernova => {
            eff.mult_mod = hands.get(ctx.hand_type).played as f64;
            return Some(eff);
        }
        // card.lua:3737-3742 — Ceremonial Dagger: the absorbed mult.
        JokerId::Ceremonial => {
            if j.state.mult > 0.0 {
                eff.mult_mod = j.state.mult;
                return Some(eff);
            }
        }
        // card.lua:3743-3761 — Vagabond: a Tarot ('vag') when at or below
        // $4 (raw G.GAME.dollars — payouts are still buffered), slots
        // permitting. Not blueprint-gated.
        JokerId::Vagabond => {
            if env.consumables_len + out.consumable_buffer() < env.consumable_slots
                && env.dollars <= 4
            {
                out.create_consumable(ConsumableSet::Tarot, "vag");
                return Some(eff); // message-only entry
            }
        }
        // card.lua:3762-3786 — Superposition: an Ace in the scoring hand
        // AND a contained Straight create a Tarot ('sup'), slots permitting.
        JokerId::Superposition => {
            if env.consumables_len + out.consumable_buffer() < env.consumable_slots {
                let aces = ctx
                    .scoring
                    .iter()
                    .filter(|&&i| ctx.full_hand[i].id_for_eval() == Some(14))
                    .count();
                if aces >= 1 && !ctx.poker_hands.get(HandType::Straight).is_empty() {
                    out.create_consumable(ConsumableSet::Tarot, "sup");
                    // message-only effects entry
                    return None;
                }
            }
        }
        // card.lua:3787-3806 — Séance: a contained Straight Flush creates a
        // Spectral ('sea'), slots permitting. Not blueprint-gated.
        JokerId::Seance => {
            if env.consumables_len + out.consumable_buffer() < env.consumable_slots
                && !ctx.poker_hands.get(HandType::StraightFlush).is_empty()
            {
                // extra.poker_hand = 'Straight Flush' (game.lua:456).
                out.create_consumable(ConsumableSet::Spectral, "sea");
                return None;
            }
        }
        // card.lua:3807-3839 — Flower Pot: all four suits among the scoring
        // cards; non-Wild cards claim first (bypass_debuff is_suit, elseif
        // chain H/D/S/C), then Wild cards fill gaps (plain is_suit — a
        // debuffed Wild claims nothing).
        JokerId::FlowerPot => {
            const ORDER: [Suit; 4] = [Suit::Hearts, Suit::Diamonds, Suit::Spades, Suit::Clubs];
            let mut suits = [0u32; 4];
            for &i in ctx.scoring {
                let c = &ctx.full_hand[i];
                if c.enhancement != Enhancement::Wild {
                    for (k, &s) in ORDER.iter().enumerate() {
                        if c.is_suit_bypass_debuff(s, smeared) && suits[k] == 0 {
                            suits[k] += 1;
                            break; // elseif chain
                        }
                    }
                }
            }
            for &i in ctx.scoring {
                let c = &ctx.full_hand[i];
                if c.enhancement == Enhancement::Wild {
                    for (k, &s) in ORDER.iter().enumerate() {
                        if c.is_suit_normal(s, smeared) && suits[k] == 0 {
                            suits[k] += 1;
                            break;
                        }
                    }
                }
            }
            if suits.iter().all(|&n| n > 0) {
                eff.x_mult_mod = 3.0; // extra (game.lua:510)
                return Some(eff);
            }
        }
        // card.lua:3840-3872 — Seeing Double: a scoring Club AND any other
        // suit. Non-Wild cards count every suit they match (independent
        // ifs — a Smeared heart is both Hearts and Diamonds); Wild cards
        // fill at most one empty slot in C/D/S/H order.
        JokerId::SeeingDouble => {
            const ORDER: [Suit; 4] = [Suit::Hearts, Suit::Diamonds, Suit::Spades, Suit::Clubs];
            let mut suits = [0u32; 4]; // H D S C
            for &i in ctx.scoring {
                let c = &ctx.full_hand[i];
                if c.enhancement != Enhancement::Wild {
                    for (k, &s) in ORDER.iter().enumerate() {
                        if c.is_suit_normal(s, smeared) {
                            suits[k] += 1;
                        }
                    }
                }
            }
            const WILD_ORDER: [(usize, Suit); 4] = [
                (3, Suit::Clubs),
                (1, Suit::Diamonds),
                (2, Suit::Spades),
                (0, Suit::Hearts),
            ];
            for &i in ctx.scoring {
                let c = &ctx.full_hand[i];
                if c.enhancement == Enhancement::Wild {
                    for &(k, s) in WILD_ORDER.iter() {
                        if c.is_suit_normal(s, smeared) && suits[k] == 0 {
                            suits[k] += 1;
                            break; // elseif chain
                        }
                    }
                }
            }
            if suits[3] > 0 && (suits[0] > 0 || suits[1] > 0 || suits[2] > 0) {
                eff.x_mult_mod = 2.0; // extra (game.lua:517)
                return Some(eff);
            }
        }
        // card.lua:3873-3879 — Wee Joker: accumulated chips (returned even
        // at 0 — a message-only entry).
        JokerId::Wee => {
            eff.chip_mod = j.state.chips;
            return Some(eff);
        }
        // card.lua:3880-3886 — Castle: accumulated chips when > 0.
        JokerId::Castle => {
            if j.state.chips > 0.0 {
                eff.chip_mod = j.state.chips;
                return Some(eff);
            }
        }
        // card.lua:3873-3879 — Blue Joker: +2 chips per card left in deck.
        JokerId::BlueJoker => {
            if env.deck_len > 0 {
                eff.chip_mod = 2.0 * env.deck_len as f64; // extra (game.lua:433)
                return Some(eff);
            }
        }
        // card.lua:3894-3900 — Erosion: +4 mult per card below the
        // starting deck size (Red Deck: 52).
        JokerId::Erosion => {
            let missing = env.starting_deck_size as i64 - env.playing_cards_len as i64;
            if missing > 0 {
                eff.mult_mod = 4.0 * missing as f64; // extra (game.lua:493)
                return Some(eff);
            }
        }
        // card.lua:3887-3893 — Square Joker: accumulated chips.
        JokerId::Square => {
            eff.chip_mod = j.state.chips;
            return Some(eff);
        }
        // card.lua:3894-3900 — Runner: accumulated chips.
        JokerId::Runner => {
            eff.chip_mod = j.state.chips;
            return Some(eff);
        }
        // card.lua:3901-3907 — Ice Cream: current chips (melts in `after`).
        JokerId::IceCream => {
            eff.chip_mod = j.state.chips;
            return Some(eff);
        }
        // card.lua:3922-3928 — Stone Joker: +25 chips per Stone card owned
        // (stone_tally, Card:update card.lua:4197-4201).
        JokerId::Stone => {
            if env.stone_tally > 0 {
                eff.chip_mod = 25.0 * env.stone_tally as f64; // extra (game.lua:475)
                return Some(eff);
            }
        }
        // card.lua:3929-3935 — Steel Joker: X(1 + 0.2 per Steel card owned).
        JokerId::SteelJoker => {
            if env.steel_tally > 0 {
                eff.x_mult_mod = 1.0 + 0.2 * env.steel_tally as f64; // extra (game.lua:397)
                return Some(eff);
            }
        }
        // card.lua:3936-3942 — Bull: +2 chips per dollar of
        // `G.GAME.dollars + dollar_buffer` (payouts queued this hand count;
        // the base total is the pre-hand snapshot).
        JokerId::Bull => {
            let total = env.dollars + out.dollar_buffer;
            if total > 0 {
                eff.chip_mod = 2.0 * total.max(0) as f64; // extra (game.lua:519)
                return Some(eff);
            }
        }
        // card.lua:3943-3950 — Driver's License: X3 with 16+ enhanced cards
        // (driver_tally: non-c_base centers, Card:update card.lua:4179-4184).
        JokerId::DriversLicense => {
            if env.driver_tally >= 16 {
                eff.x_mult_mod = 3.0; // extra (game.lua:527)
                return Some(eff);
            }
        }
        // card.lua:3951-3965 — Blackboard: X3 when every held card is a
        // Spade or Club (flush_calc is_suit: Wild counts, Stone breaks it,
        // debuffed naturals count; an empty hand triggers).
        JokerId::Blackboard => {
            let all_black = ctx
                .held
                .iter()
                .all(|c| c.is_suit(Suit::Clubs, smeared) || c.is_suit(Suit::Spades, smeared));
            if all_black {
                eff.x_mult_mod = 3.0; // extra (game.lua:427)
                return Some(eff);
            }
        }
        // card.lua:3966-3973 — Joker Stencil: the live x_mult (free slots +
        // 1 per Stencil) while any slot is free. Only reached when the
        // generic branch didn't fire (live x_mult <= 1).
        JokerId::Stencil => {
            if env.joker_slots as i64 - njokers as i64 > 0 {
                eff.x_mult_mod = live_x_mult;
                return Some(eff);
            }
        }
        // card.lua:3974-3979 — Swashbuckler: mult = summed sell value of
        // the other jokers (kept live by Card:update, card.lua:4240-4247).
        JokerId::Swashbuckler => {
            if swashbuckler_mult > 0.0 {
                eff.mult_mod = swashbuckler_mult;
                return Some(eff);
            }
        }
        // card.lua:3980-3985 — Joker: flat +4 mult (config.mult).
        JokerId::Joker => {
            eff.mult_mod = 4.0;
            return Some(eff);
        }
        // card.lua:3986-3991 — Spare Trousers: accumulated mult.
        JokerId::Trousers => {
            if j.state.mult > 0.0 {
                eff.mult_mod = j.state.mult;
                return Some(eff);
            }
        }
        // card.lua:3992-3997 — Ride the Bus: accumulated mult.
        JokerId::RideTheBus => {
            if j.state.mult > 0.0 {
                eff.mult_mod = j.state.mult;
                return Some(eff);
            }
        }
        // card.lua:3998-4003 — Flash Card: accumulated mult.
        JokerId::Flash => {
            if j.state.mult > 0.0 {
                eff.mult_mod = j.state.mult;
                return Some(eff);
            }
        }
        // card.lua:4004-4009 — Popcorn: current mult (shrinks each round).
        JokerId::Popcorn => {
            if j.state.mult > 0.0 {
                eff.mult_mod = j.state.mult;
                return Some(eff);
            }
        }
        // card.lua:4010-4015 — Green Joker: accumulated mult.
        JokerId::GreenJoker => {
            if j.state.mult > 0.0 {
                eff.mult_mod = j.state.mult;
                return Some(eff);
            }
        }
        // card.lua:4016-4021 — Fortune Teller: +1 mult per Tarot used this
        // run (G.GAME.consumeable_usage_total.tarot).
        JokerId::FortuneTeller => {
            if env.tarots_used > 0 {
                eff.mult_mod = env.tarots_used as f64;
                return Some(eff);
            }
        }
        // card.lua:4022-4027 — Gros Michel: flat +15 mult (extra.mult).
        JokerId::GrosMichel => {
            eff.mult_mod = 15.0;
            return Some(eff);
        }
        // card.lua:4028-4033 — Cavendish: flat X3 mult (extra.Xmult).
        JokerId::Cavendish => {
            eff.x_mult_mod = 3.0;
            return Some(eff);
        }
        // card.lua:4034-4039 — Red Card: accumulated mult.
        JokerId::RedCard if j.state.mult > 0.0 => {
            eff.mult_mod = j.state.mult;
            return Some(eff);
        }
        // card.lua:4040-4045 — Card Sharp: X3 when the played hand type was
        // already played this round (played_this_round counts this play).
        JokerId::CardSharp => {
            if hands.get(ctx.hand_type).played_this_round > 1 {
                eff.x_mult_mod = 3.0; // extra.Xmult (game.lua:457)
                return Some(eff);
            }
        }
        // card.lua:4046-4051 — Bootstraps: +2 mult per $5 of
        // `dollars + dollar_buffer`.
        JokerId::Bootstraps => {
            let total = env.dollars + out.dollar_buffer;
            let steps = (total as f64 / 5.0).floor();
            if steps >= 1.0 {
                eff.mult_mod = 2.0 * steps; // extra (game.lua:533)
                return Some(eff);
            }
        }
        // card.lua:4052-4057 — Caino: the separate caino_xmult accumulator
        // (the generic x_mult branch never sees it).
        JokerId::Caino if j.state.extra > 1.0 => {
            eff.x_mult_mod = j.state.extra;
            return Some(eff);
        }
        _ => {}
    }
    None
}

/// `{after = true}` (card.lua:3575-3633).
pub(super) fn after_hand(
    j: &mut OwnedJoker,
    bp: bool,
    _ctx: &HandCtx,
    out: &mut JokerOutbox,
    _rng: &mut RngState,
) {
    match j.id {
        // card.lua:3576-3612 — Ice Cream: -5 chips per hand; melts when it
        // would reach 0 (checked BEFORE the decrement). Blueprint-gated.
        JokerId::IceCream if !bp => {
            if j.state.chips - 5.0 <= 0.0 {
                out.destroy_joker(j.sort_id);
            } else {
                j.state.chips -= 5.0; // extra.chip_mod (game.lua:430)
            }
        }
        // card.lua:3601-3630 — Seltzer: one hand fewer; drank when the
        // counter would reach 0 (checked BEFORE the decrement).
        // Blueprint-gated.
        JokerId::Selzer if !bp => {
            if j.state.extra - 1.0 <= 0.0 {
                out.destroy_joker(j.sort_id);
            } else {
                j.state.extra -= 1.0;
            }
        }
        _ => {}
    }
}

/// `{discard = true, other_card = …}` (card.lua:2756-2873). Returns
/// `eval.remove` (Trading Card destroys the discarded card).
#[allow(clippy::too_many_arguments)]
pub(super) fn on_discard_card(
    j: &mut OwnedJoker,
    bp: bool,
    card: &Card,
    selected: &[Card],
    pareidolia: bool,
    smeared: bool,
    env: &JokerEnv,
    out: &mut JokerOutbox,
    dollars: &mut i64,
    _rng: &mut RngState,
) -> bool {
    let last = selected.last().map(|c| c.sort_id) == Some(card.sort_id);
    match j.id {
        // card.lua:2757-2787 — Ramen: -0.01 x_mult per discarded card;
        // eaten when it would reach X1 (checked BEFORE the decrement).
        // Blueprint-gated.
        JokerId::Ramen if !bp => {
            if j.state.x_mult - 0.01 <= 1.0 {
                out.destroy_joker(j.sort_id);
            } else {
                j.state.x_mult -= 0.01; // extra (game.lua:494)
            }
        }
        // card.lua:2788-2801 — Yorick: counts 23 discarded CARDS down, then
        // +1 x_mult and the counter resets. Blueprint-gated.
        JokerId::Yorick if !bp => {
            if j.state.extra <= 1.0 {
                j.state.extra = 23.0; // extra.discards (game.lua:539)
                j.state.x_mult += 1.0; // extra.xmult
            } else {
                j.state.extra -= 1.0;
            }
        }
        // card.lua:2802-2812 — Trading Card: the round's first discard of
        // exactly one card destroys it and pays $3. Blueprint-gated.
        JokerId::Trading if !bp => {
            if env.discards_used_round <= 0 && selected.len() == 1 {
                *dollars += 3; // extra (game.lua:483)
                return true; // eval.remove
            }
        }
        // card.lua:2814-2824 — Castle: +3 chips per discarded card of this
        // round's castle suit. Blueprint-gated.
        JokerId::Castle if !bp => {
            if !card.debuff && card.is_suit_normal(env.castle_suit, smeared) {
                j.state.chips += 3.0; // extra.chip_mod (game.lua:495)
            }
        }
        // card.lua:2825-2834 — Mail-In Rebate: $5 per discarded card of
        // this round's mail rank (inline ease_dollars; NOT blueprint-gated).
        // A Stone card's random negative get_id() never matches.
        JokerId::Mail => {
            if !card.debuff
                && card.enhancement != Enhancement::Stone
                && card.rank.id() == env.mail_card_id
            {
                *dollars += 5; // extra (game.lua:471)
            }
        }
        // card.lua:2835-2845 — Hit the Road: +0.5 x_mult per discarded
        // Jack. Blueprint-gated.
        JokerId::HitTheRoad if !bp => {
            if !card.debuff && card.enhancement != Enhancement::Stone && card.rank.id() == 11 {
                j.state.x_mult += 0.5; // extra (game.lua:520)
            }
        }
        // card.lua:2846-2856 — Green Joker: -1 mult (floored at 0) once per
        // discard action, keyed to the LAST card of the selection.
        JokerId::GreenJoker if !bp => {
            if last {
                j.state.mult = (j.state.mult - 1.0).max(0.0); // extra.discard_sub
            }
        }
        // card.lua:2858-2872 — Faceless Joker: $5 when 3+ faces are
        // discarded together (evaluated on the last card; not bp-gated).
        JokerId::Faceless if last => {
            let faces = selected
                .iter()
                .filter(|c| card_is_face(c, pareidolia))
                .count();
            if faces >= 3 {
                *dollars += 5; // extra = {dollars = 5, faces = 3}
            }
        }
        _ => {}
    }
    false
}

/// `{end_of_round = true}` non-repetition branch (card.lua:2888-3063). The
/// whole block is inside `elseif not context.blueprint`, so the dispatcher
/// never routes copies here. Returns `eval.saved` (Mr. Bones).
pub(super) fn end_of_round(
    j: &mut OwnedJoker,
    hands: &HandsTable,
    game_over: bool,
    env: &JokerEnv,
    out: &mut JokerOutbox,
    rng: &mut RngState,
) -> bool {
    match j.id {
        // card.lua:2889-2895 — Campfire: resets to X1 once a Boss round
        // ends (the raw blind.boss flag — disabled bosses included).
        JokerId::Campfire => {
            if env.blind_is_boss && j.state.x_mult > 1.0 {
                j.state.x_mult = 1.0;
            }
        }
        // card.lua:2896-2902 — Rocket: payout +2 when a Boss round ends.
        JokerId::Rocket => {
            if env.blind_is_boss {
                j.state.extra += 2.0; // extra.increase (game.lua:492)
            }
        }
        // card.lua:2903-2933 — Turtle Bean: -1 hand size per round; eaten
        // when the bonus would reach 0 (checked BEFORE the decrement). The
        // live hand-size change is applied by the Run (destroy via
        // remove_from_deck, decrement via change_size).
        JokerId::TurtleBean => {
            if j.state.extra - 1.0 <= 0.0 {
                out.destroy_joker(j.sort_id);
            } else {
                j.state.extra -= 1.0; // extra.h_mod (game.lua:490)
                out.events.push(super::OutEvent::TurtleBeanShrink);
            }
        }
        // card.lua:2934-2944 — Invisible Joker: one more round on the
        // clock (sell-activation at extra rounds; game.lua:525 extra = 2).
        JokerId::Invisible => {
            j.state.extra += 1.0;
        }
        // card.lua:2945-2974 — Popcorn: -4 mult per round; eaten when it
        // would reach 0 (checked BEFORE the decrement).
        JokerId::Popcorn => {
            if j.state.mult - 4.0 <= 0.0 {
                out.destroy_joker(j.sort_id);
            } else {
                j.state.mult -= 4.0; // extra (game.lua:477)
            }
        }
        // card.lua:2975-2985 — To Do List: re-roll over the visible hands
        // EXCLUDING the current one (single 'to_do' draw).
        JokerId::TodoList => {
            let pool: Vec<HandType> = MOST_PLAYED_SCAN
                .iter()
                .copied()
                .filter(|&ht| hands.get(ht).visible && ht != j.state.to_do_hand)
                .collect();
            j.state.to_do_hand = pool[element_index(rng, "to_do", pool.len())];
        }
        // card.lua:2986-2993 — Egg: +$3 sell value per round.
        JokerId::Egg => {
            j.extra_value += 3; // extra (game.lua:425)
        }
        // card.lua:2993-3010 — Gift Card: +$1 sell value on every joker and
        // consumable (applied by the Run — it owns both areas).
        JokerId::Gift => {
            out.events.push(super::OutEvent::GiftCard);
        }
        // card.lua:3011-3017 — Hit the Road: the Jack bonus resets.
        JokerId::HitTheRoad => {
            if j.state.x_mult > 1.0 {
                j.state.x_mult = 1.0;
            }
        }
        // card.lua:3019-3046 — Gros Michel / Cavendish extinction rolls:
        // 'gros_michel' < 1/6 (odds 6) / 'cavendish' < 1/1000 (odds 1000).
        // Gros Michel going extinct sets the pool flag that retires it and
        // admits Cavendish (game.lua:405/458).
        JokerId::GrosMichel => {
            if rng.random("gros_michel") < env.prob_normal / 6.0 {
                out.destroy_joker(j.sort_id);
                out.events.push(super::OutEvent::SetGrosMichelExtinct);
            }
        }
        JokerId::Cavendish if rng.random("cavendish") < env.prob_normal / 1000.0 => {
            out.destroy_joker(j.sort_id);
        }
        // card.lua:3047-3063 — Mr. Bones: saves a lost run at >= 25% of the
        // blind requirement and dissolves.
        JokerId::MrBones
            if game_over && env.blind_chips > 0.0 && env.game_chips / env.blind_chips >= 0.25 =>
        {
            out.destroy_joker(j.sort_id);
            return true; // eval.saved
        }
        _ => {}
    }
    false
}

/// `Card:calculate_dollar_bonus` (card.lua:1655-1679). No copy resolution —
/// it is a separate function Blueprint never touches.
pub(super) fn dollar_bonus(j: &OwnedJoker, env: &JokerEnv) -> i64 {
    match j.id {
        // card.lua:1658-1660 — Golden Joker: flat $4 (extra, game.lua:476).
        JokerId::Golden => 4,
        // card.lua:1661-1663 — Cloud 9: $1 per 9 owned (nine_tally,
        // Card:update card.lua:4191-4196).
        JokerId::Cloud9 if env.nine_tally > 0 => env.nine_tally,
        // card.lua:1664-1666 — Rocket: the grown payout (extra.dollars).
        JokerId::Rocket => j.state.extra as i64,
        // card.lua:1667-1673 — Satellite: $1 per DISTINCT Planet used this
        // run (G.GAME.consumeable_usage keys with set == 'Planet').
        JokerId::Satellite if env.planets_used > 0 => env.planets_used,
        // card.lua:1674-1676 — Delayed Gratification: $2 per discard if
        // none were used this round.
        JokerId::DelayedGrat if env.discards_used_round == 0 && env.discards_left > 0 => {
            2 * env.discards_left // extra (game.lua:401)
        }
        _ => 0,
    }
}
