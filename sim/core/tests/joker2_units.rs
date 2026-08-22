//! Per-joker exact-arithmetic tests for the P3c-2 jokers (every Uncommon,
//! Rare and Legendary) through `evaluate_play` with the real `EngineHooks`
//! dispatcher, plus direct hook-window tests for discard/end-of-round/payout
//! effects. Expected values are hand-derived from the ported formulas;
//! RNG-driven jokers compare against a fresh clone of the seed's stream.
//! Run-level windows (setting_blind, selling, shop) live in
//! tests/joker2_run_effects.rs.

use std::sync::atomic::{AtomicU32, Ordering};

use balatro_core::blinds::{boss_by_key, ActiveBlind};
use balatro_core::cards::{Card, Edition, Enhancement, HandType, Rank, Seal, Suit};
use balatro_core::handeval::EvalMods;
use balatro_core::items::{ConsumableSet, JokerId};
use balatro_core::jokers::{EngineHooks, JokerEnv, JokerState, OutEvent};
use balatro_core::rng::RngState;
use balatro_core::scoring::{evaluate_play, HandsTable, JokerHooks, PlayResult, ScoreContext};
use balatro_core::shop::OwnedJoker;

static NEXT_ID: AtomicU32 = AtomicU32::new(1000);

fn c(rank: u8, suit: Suit) -> Card {
    Card::new(Rank(rank), suit, NEXT_ID.fetch_add(1, Ordering::Relaxed))
}

fn j(id: JokerId) -> OwnedJoker {
    OwnedJoker {
        id,
        edition: Edition::None,
        sort_id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        extra_value: 0,
        debuffed: false,
        flipped: false,
        eternal: false,
        hands_at_create: 0,
        state: JokerState::initial(id),
    }
}

/// evaluate_play harness with the live joker engine; the post-eval play/held
/// cards (Midas/Vampire/Hiker mutations, DNA copies) are kept for asserts.
struct World {
    rng: RngState,
    hands: HandsTable,
    dollars: i64,
    jokers: Vec<OwnedJoker>,
    env: JokerEnv,
    blind: Option<ActiveBlind>,
    mods: EvalMods,
    out: Vec<OutEvent>,
    played: Vec<Card>,
    held: Vec<Card>,
}

impl World {
    fn new(seed: &str) -> Self {
        World {
            rng: RngState::new(seed),
            hands: HandsTable::new(),
            dollars: 0,
            jokers: Vec::new(),
            env: JokerEnv {
                consumable_slots: 2,
                joker_slots: 5,
                ..JokerEnv::default()
            },
            blind: None,
            mods: EvalMods::default(),
            out: Vec::new(),
            played: Vec::new(),
            held: Vec::new(),
        }
    }

    fn with(seed: &str, ids: &[JokerId]) -> Self {
        let mut w = Self::new(seed);
        w.jokers = ids.iter().map(|&id| j(id)).collect();
        w
    }

    fn play(&mut self, play: &[Card], held: &[Card]) -> PlayResult {
        let mut hooks = EngineHooks::new(&mut self.jokers, self.env.clone());
        let mut play = play.to_vec();
        let mut held = held.to_vec();
        let r = {
            let mut ctx = ScoreContext {
                rng: &mut self.rng,
                hands: &mut self.hands,
                dollars: &mut self.dollars,
                blind: self.blind.as_mut(),
                most_played: HandType::HighCard,
                mods: self.mods,
                prob_normal: self.env.prob_normal,
                plasma_balance: false,
            };
            evaluate_play(&mut ctx, &mut play, &mut held, &mut hooks)
        };
        self.out = hooks.out.events;
        self.played = play;
        self.held = held;
        r
    }

    fn end_of_round(&mut self, game_over: bool) -> bool {
        let mut hooks = EngineHooks::new(&mut self.jokers, self.env.clone());
        let saved = hooks.end_of_round(&self.hands, game_over, &mut self.rng);
        self.out = hooks.out.events;
        saved
    }

    fn discard(&mut self, selected: &[Card]) -> Vec<bool> {
        let mods = self.mods;
        let mut hooks = EngineHooks::new(&mut self.jokers, self.env.clone());
        hooks.pre_discard(selected, false, &mut self.hands, &mods);
        let removed = selected
            .iter()
            .map(|card| hooks.on_discard_card(card, selected, &mut self.dollars, &mut self.rng))
            .collect();
        self.out = hooks.out.events;
        removed
    }

    fn destroyed(&mut self, cards: &[Card]) {
        let mut hooks = EngineHooks::new(&mut self.jokers, self.env.clone());
        hooks.on_cards_destroyed(cards);
        self.out = hooks.out.events;
    }

    fn dollar_bonus(&mut self) -> i64 {
        let mut hooks = EngineHooks::new(&mut self.jokers, self.env.clone());
        hooks.dollar_bonus()
    }
}

fn pair_5s() -> Vec<Card> {
    vec![c(5, Suit::Hearts), c(5, Suit::Spades)]
}

fn pair_kings() -> Vec<Card> {
    vec![c(13, Suit::Hearts), c(13, Suit::Spades)]
}

// ---------------------------------------------------------------------------
// flat / conditional joker_main branches
// ---------------------------------------------------------------------------

#[test]
fn stencil_counts_free_slots_plus_stencils() {
    // 5 slots, 1 joker: x_mult = (5-1) + 1 = 5 (card.lua:4203-4208).
    let mut w = World::with("SEED", &[JokerId::Stencil]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!((r.chips, r.mult), (20.0, 10.0));
}

#[test]
fn stencil_full_slots_with_two_stencils_double_fires() {
    // [St, St, J, J, J] with 5 slots: x = 0 free + 2 stencils = 2; the
    // generic x_mult branch fires for BOTH stencils even at 0 free slots
    // (card.lua:3653-3659 comes before the gated named branch :3966).
    let mut w = World::with(
        "SEED",
        &[
            JokerId::Stencil,
            JokerId::Stencil,
            JokerId::Joker,
            JokerId::Joker,
            JokerId::Joker,
        ],
    );
    let r = w.play(&pair_5s(), &[]);
    // area order: X2, X2, +4, +4, +4 -> 2*2*2 + 12 = 20.
    assert_eq!(r.mult, 20.0);
}

#[test]
fn stencil_single_free_slot_is_inert() {
    // [St, J, J, J, J]: x = 0 + 1 = 1 — generic gate (>1) fails and the
    // named branch's free-slot gate fails too.
    let mut w = World::with(
        "SEED",
        &[
            JokerId::Stencil,
            JokerId::Joker,
            JokerId::Joker,
            JokerId::Joker,
            JokerId::Joker,
        ],
    );
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 18.0); // 2 + 4*4
}

#[test]
fn acrobat_final_hand_only() {
    let mut w = World::with("SEED", &[JokerId::Acrobat]);
    w.env.hands_left = 1;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0);
    w.env.hands_left = 0;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 6.0); // X3 (card.lua:3688-3693)
}

#[test]
fn stuntman_flat_chips() {
    let mut w = World::with("SEED", &[JokerId::Stuntman]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 270.0); // 20 + 250 (card.lua:3713-3718)
}

#[test]
fn loyalty_card_every_sixth_hand() {
    // loyalty_remaining = (4 - n) % 6, trigger at 5 (card.lua:3632-3651):
    // n = hands_played - at_create counts hands BEFORE this one.
    let mut w = World::with("SEED", &[JokerId::LoyaltyCard]);
    w.env.hands_played_total = 4;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0);
    w.env.hands_played_total = 5; // the 6th hand since purchase
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 8.0); // X4
    w.env.hands_played_total = 11; // and every 6 after
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 8.0);
}

#[test]
fn throwback_scales_with_skips() {
    let mut w = World::with("SEED", &[JokerId::Throwback]);
    w.env.skips = 3; // 1 + 3*0.25 = 1.75 (card.lua:4176-4178)
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 3.5);
    w.env.skips = 0;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0); // x_mult 1: generic branch inert
}

#[test]
fn steel_and_stone_joker_tallies() {
    let mut w = World::with("SEED", &[JokerId::SteelJoker]);
    w.env.steel_tally = 2; // X(1 + 0.4)
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.8);

    let mut w = World::with("SEED", &[JokerId::Stone]);
    w.env.stone_tally = 2; // +50 chips
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 70.0);
    w.env.stone_tally = 0;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 20.0);
}

#[test]
fn drivers_license_needs_16_enhanced() {
    let mut w = World::with("SEED", &[JokerId::DriversLicense]);
    w.env.driver_tally = 15;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0);
    w.env.driver_tally = 16;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 6.0); // X3 (card.lua:3943-3950)
}

#[test]
fn erosion_counts_missing_cards() {
    let mut w = World::with("SEED", &[JokerId::Erosion]);
    w.env.playing_cards_len = 49; // 3 below 52: +12 mult
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 14.0);
    w.env.playing_cards_len = 53; // above start: nothing
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0);
}

#[test]
fn bull_counts_dollars() {
    let mut w = World::with("SEED", &[JokerId::Bull]);
    w.env.dollars = 7;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 34.0); // 20 + 2*7
    w.env.dollars = 0;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 20.0);
}

#[test]
fn bull_sees_dollar_buffer_from_todo_list() {
    // To Do List pays $4 inline into the dollar_buffer (card.lua:3491-3500);
    // Bull reads dollars + buffer (card.lua:3936-3942).
    let mut w = World::with("SEED", &[JokerId::TodoList, JokerId::Bull]);
    w.jokers[0].state.to_do_hand = HandType::Pair;
    w.env.dollars = 7;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(w.dollars, 4);
    assert_eq!(r.chips, 20.0 + 2.0 * 11.0);
}

#[test]
fn bootstraps_two_mult_per_five_dollars() {
    let mut w = World::with("SEED", &[JokerId::Bootstraps]);
    w.env.dollars = 13; // floor(13/5) = 2 -> +4
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 6.0);
    w.env.dollars = 4;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0);
}

#[test]
fn blackboard_all_black_held() {
    let mut w = World::with("SEED", &[JokerId::Blackboard]);
    let held = [c(14, Suit::Spades), c(2, Suit::Clubs)];
    let r = w.play(&pair_5s(), &held);
    assert_eq!(r.mult, 6.0); // X3 (card.lua:3951-3965)
                             // A red card breaks it...
    let held = [c(14, Suit::Spades), c(2, Suit::Hearts)];
    let r = w.play(&pair_5s(), &held);
    assert_eq!(r.mult, 2.0);
    // ...a Wild heart counts as black (flush_calc is_suit)...
    let mut wild = c(2, Suit::Hearts);
    wild.enhancement = Enhancement::Wild;
    let r = w.play(&pair_5s(), &[wild]);
    assert_eq!(r.mult, 6.0);
    // ...a Stone card matches no suit and breaks it...
    let mut stone = c(2, Suit::Spades);
    stone.enhancement = Enhancement::Stone;
    let r = w.play(&pair_5s(), &[stone]);
    assert_eq!(r.mult, 2.0);
    // ...and an empty hand trivially triggers (0 == 0).
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 6.0);
}

#[test]
fn card_sharp_second_play_of_type() {
    let mut w = World::with("SEED", &[JokerId::CardSharp]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0); // played_this_round == 1
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 6.0); // X3 (card.lua:4040-4045)
}

// ---------------------------------------------------------------------------
// typed x_mult jokers (Duo..Tribe) through the generic branch
// ---------------------------------------------------------------------------

#[test]
fn duo_trio_family_order_tribe() {
    // The Duo: X2 with a contained Pair.
    let mut w = World::with("SEED", &[JokerId::Duo]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 4.0);
    let r = w.play(&[c(2, Suit::Hearts)], &[]);
    assert_eq!(r.mult, 1.0); // no pair

    // A Full House CONTAINS a pair — Duo fires below the played hand.
    let fh = vec![
        c(5, Suit::Hearts),
        c(5, Suit::Spades),
        c(5, Suit::Clubs),
        c(9, Suit::Hearts),
        c(9, Suit::Spades),
    ];
    let r = w.play(&fh, &[]);
    assert_eq!(r.hand_type, HandType::FullHouse);
    assert_eq!(r.mult, 8.0); // 4 * 2

    // The Trio: X3 with trips.
    let mut w = World::with("SEED", &[JokerId::Trio]);
    let trips = vec![c(5, Suit::Hearts), c(5, Suit::Spades), c(5, Suit::Clubs)];
    let r = w.play(&trips, &[]);
    assert_eq!(r.mult, 9.0);

    // The Family: X4 with quads. Chips 60 + 4*10.
    let mut w = World::with("SEED", &[JokerId::Family]);
    let quads = vec![
        c(13, Suit::Hearts),
        c(13, Suit::Spades),
        c(13, Suit::Clubs),
        c(13, Suit::Diamonds),
    ];
    let r = w.play(&quads, &[]);
    assert_eq!((r.chips, r.mult), (100.0, 28.0));

    // The Order: X3 with a straight.
    let mut w = World::with("SEED", &[JokerId::Order]);
    let straight = vec![
        c(5, Suit::Hearts),
        c(6, Suit::Spades),
        c(7, Suit::Clubs),
        c(8, Suit::Diamonds),
        c(9, Suit::Hearts),
    ];
    let r = w.play(&straight, &[]);
    assert_eq!(r.mult, 12.0);

    // The Tribe: X2 with a flush.
    let mut w = World::with("SEED", &[JokerId::Tribe]);
    let flush = vec![
        c(2, Suit::Hearts),
        c(5, Suit::Hearts),
        c(7, Suit::Hearts),
        c(9, Suit::Hearts),
        c(13, Suit::Hearts),
    ];
    let r = w.play(&flush, &[]);
    assert_eq!(r.mult, 8.0);
}

// ---------------------------------------------------------------------------
// per-scored-card jokers
// ---------------------------------------------------------------------------

#[test]
fn fibonacci_ranks() {
    let mut w = World::with("SEED", &[JokerId::Fibonacci]);
    let r = w.play(&[c(8, Suit::Hearts), c(8, Suit::Spades)], &[]);
    assert_eq!(r.mult, 18.0); // 2 + 8 + 8
    let r = w.play(&[c(9, Suit::Hearts), c(9, Suit::Spades)], &[]);
    assert_eq!(r.mult, 2.0);
    // Ace counts.
    let r = w.play(&[c(14, Suit::Hearts)], &[]);
    assert_eq!(r.mult, 9.0); // 1 + 8
}

#[test]
fn triboulet_kings_and_queens() {
    let mut w = World::with("SEED", &[JokerId::Triboulet]);
    let r = w.play(&pair_kings(), &[]);
    // chips 10 + 10 + 10; mult 2 * 2 * 2.
    assert_eq!((r.chips, r.mult, r.score), (30.0, 8.0, 240.0));
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0);
}

#[test]
fn suit_gems_rough_onyx_arrowhead() {
    // Rough Gem: $1 per scored Diamond (card.lua:3224-3232).
    let mut w = World::with("SEED", &[JokerId::RoughGem]);
    let r = w.play(&[c(5, Suit::Diamonds), c(5, Suit::Diamonds)], &[]);
    assert_eq!(r.dollars_delta, 2);

    // Onyx Agate: +7 mult per scored Club.
    let mut w = World::with("SEED", &[JokerId::OnyxAgate]);
    let r = w.play(&[c(5, Suit::Clubs), c(5, Suit::Clubs)], &[]);
    assert_eq!(r.mult, 16.0);

    // Arrowhead: +50 chips per scored Spade.
    let mut w = World::with("SEED", &[JokerId::Arrowhead]);
    let r = w.play(&[c(5, Suit::Spades), c(5, Suit::Spades)], &[]);
    assert_eq!(r.chips, 120.0);
}

#[test]
fn bloodstone_rolls_per_scored_heart() {
    let mut w = World::with("BLOOD", &[JokerId::Bloodstone]);
    let mut probe = RngState::new("BLOOD");
    let hits = [
        probe.random("bloodstone") < 0.5,
        probe.random("bloodstone") < 0.5,
    ];
    let r = w.play(&[c(5, Suit::Hearts), c(5, Suit::Hearts)], &[]);
    let mut mult = 2.0;
    for h in hits {
        if h {
            mult *= 1.5;
        }
    }
    assert_eq!(r.mult, mult);
}

#[test]
fn idol_matches_rank_and_suit() {
    let mut w = World::with("SEED", &[JokerId::Idol]);
    w.env.idol_card = (5, Suit::Hearts);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 4.0); // only the 5♥ doubles
                             // A Wild card matches any suit (is_suit).
    let mut wild = c(5, Suit::Spades);
    wild.enhancement = Enhancement::Wild;
    let r = w.play(&[c(5, Suit::Hearts), wild], &[]);
    assert_eq!(r.mult, 8.0);
    // Wrong rank: nothing.
    w.env.idol_card = (9, Suit::Hearts);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0);
}

#[test]
fn ancient_joker_suit() {
    let mut w = World::with("SEED", &[JokerId::Ancient]);
    w.env.ancient_suit = Some(Suit::Clubs);
    let r = w.play(&[c(5, Suit::Clubs), c(5, Suit::Hearts)], &[]);
    assert_eq!(r.mult, 3.0); // one X1.5
}

#[test]
fn wee_joker_scales_on_scored_twos() {
    let mut w = World::with("SEED", &[JokerId::Wee]);
    let r = w.play(&[c(2, Suit::Hearts), c(2, Suit::Spades)], &[]);
    // Both 2s bump the accumulator BEFORE joker_main reads it: +16.
    assert_eq!(r.chips, 10.0 + 2.0 + 2.0 + 16.0);
    assert_eq!(w.jokers[0].state.chips, 16.0);
    // Next hand without 2s still carries the chips.
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 20.0 + 16.0);
}

#[test]
fn wee_joker_with_hack_retriggers() {
    // Hack retriggers 2s; Wee gains per repetition: 2 cards x 2 reps = +32.
    let mut w = World::with("SEED", &[JokerId::Hack, JokerId::Wee]);
    let r = w.play(&[c(2, Suit::Hearts), c(2, Suit::Spades)], &[]);
    assert_eq!(w.jokers[1].state.chips, 32.0);
    assert_eq!(r.chips, 10.0 + 4.0 * 2.0 + 32.0);
}

#[test]
fn hiker_perma_bonus_accumulates_on_cards() {
    let mut w = World::with("SEED", &[JokerId::Hiker]);
    let play = pair_5s();
    let r = w.play(&play, &[]);
    assert_eq!(r.chips, 20.0); // upgrade lands AFTER this rep's chip read
    assert_eq!(w.played[0].perma_bonus, 5);
    assert_eq!(w.played[1].perma_bonus, 5);
    // Replaying the mutated cards scores the bonus.
    let mutated = w.played.clone();
    let r = w.play(&mutated, &[]);
    assert_eq!(r.chips, 30.0); // 10 + (5+5) + (5+5)
}

#[test]
fn hiker_retriggered_reps_see_prior_bonus() {
    // Dusk retriggers on the final hand: rep1 reads +0 and adds 5, rep2
    // reads +5 and adds 5 more (card.lua:3067-3075 + eval order).
    let mut w = World::with("SEED", &[JokerId::Dusk, JokerId::Hiker]);
    w.env.hands_left = 0;
    let r = w.play(&[c(5, Suit::Hearts)], &[]);
    // High card 5 chips: rep1 5, rep2 5+5. total 5 + 5 + 10 = 20.
    assert_eq!(r.chips, 20.0);
    assert_eq!(w.played[0].perma_bonus, 10);
}

// ---------------------------------------------------------------------------
// retrigger jokers
// ---------------------------------------------------------------------------

#[test]
fn dusk_retriggers_on_final_hand() {
    let mut w = World::with("SEED", &[JokerId::Dusk]);
    w.env.hands_left = 0;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 30.0); // 10 + 2*(5+5)
    w.env.hands_left = 2;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 20.0);
}

#[test]
fn sock_and_buskin_retriggers_faces() {
    let mut w = World::with("SEED", &[JokerId::SockAndBuskin]);
    let r = w.play(&pair_kings(), &[]);
    assert_eq!(r.chips, 50.0); // 10 + 2*10 + 2*10
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 20.0);
}

#[test]
fn hack_retriggers_low_ranks() {
    let mut w = World::with("SEED", &[JokerId::Hack]);
    let r = w.play(&[c(2, Suit::Hearts), c(2, Suit::Spades)], &[]);
    assert_eq!(r.chips, 18.0); // 10 + 2*2 + 2*2
    let r = w.play(&[c(6, Suit::Hearts), c(6, Suit::Spades)], &[]);
    assert_eq!(r.chips, 22.0); // 6s are not retriggered
}

#[test]
fn seltzer_retriggers_and_decays() {
    let mut w = World::with("SEED", &[JokerId::Selzer]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 30.0);
    assert_eq!(w.jokers[0].state.extra, 9.0); // after-hand decrement
                                              // At 1 hand left the after-window drinks it.
    w.jokers[0].state.extra = 1.0;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 30.0); // still retriggers this hand
    assert_eq!(w.out, vec![OutEvent::DestroyJoker(w.jokers[0].sort_id)]);
}

#[test]
fn mime_retriggers_held_effects() {
    // Steel card in hand: X1.5 twice (card.lua:3386-3394).
    let mut w = World::with("SEED", &[JokerId::Mime]);
    let mut steel = c(9, Suit::Hearts);
    steel.enhancement = Enhancement::Steel;
    let r = w.play(&pair_5s(), &[steel]);
    assert_eq!(r.mult, 4.5); // 2 * 1.5 * 1.5
                             // A plain held card has no effects — Mime does nothing.
    let r = w.play(&pair_5s(), &[c(9, Suit::Hearts)]);
    assert_eq!(r.mult, 2.0);
}

#[test]
fn baron_kings_and_mime_stack() {
    let mut w = World::with("SEED", &[JokerId::Baron]);
    let r = w.play(&pair_5s(), &[c(13, Suit::Hearts)]);
    assert_eq!(r.mult, 3.0); // 2 * 1.5
                             // Debuffed King: message-only entry, no mult.
    let mut dk = c(13, Suit::Hearts);
    dk.debuff = true;
    let r = w.play(&pair_5s(), &[dk]);
    assert_eq!(r.mult, 2.0);
    // Baron + Mime: the King's X1.5 fires twice.
    let mut w = World::with("SEED", &[JokerId::Baron, JokerId::Mime]);
    let r = w.play(&pair_5s(), &[c(13, Suit::Hearts)]);
    assert_eq!(r.mult, 4.5);
}

// ---------------------------------------------------------------------------
// before-window jokers
// ---------------------------------------------------------------------------

#[test]
fn spare_trousers_scales_on_two_pair() {
    let mut w = World::with("SEED", &[JokerId::Trousers]);
    let two_pair = vec![
        c(5, Suit::Hearts),
        c(5, Suit::Spades),
        c(9, Suit::Hearts),
        c(9, Suit::Spades),
    ];
    let r = w.play(&two_pair, &[]);
    // before: +2; joker_main reads it the same hand.
    assert_eq!((r.chips, r.mult), (48.0, 4.0));
    assert_eq!(w.jokers[0].state.mult, 2.0);
    // A Full House also contains a Two Pair.
    let fh = vec![
        c(5, Suit::Hearts),
        c(5, Suit::Spades),
        c(5, Suit::Clubs),
        c(9, Suit::Hearts),
        c(9, Suit::Spades),
    ];
    w.play(&fh, &[]);
    assert_eq!(w.jokers[0].state.mult, 4.0);
    // A plain pair does not.
    w.play(&pair_5s(), &[]);
    assert_eq!(w.jokers[0].state.mult, 4.0);
}

#[test]
fn space_joker_levels_on_roll() {
    let mut w = World::with("SPACE", &[JokerId::Space]);
    let mut probe = RngState::new("SPACE");
    let hit = probe.random("space") < 0.25;
    w.play(&pair_5s(), &[]);
    let expect = if hit { 2 } else { 1 };
    assert_eq!(w.hands.get(HandType::Pair).level, expect);
}

#[test]
fn midas_mask_golds_scored_faces() {
    let mut w = World::with("SEED", &[JokerId::MidasMask]);
    let r = w.play(&pair_kings(), &[]);
    assert_eq!(r.chips, 30.0);
    assert_eq!(w.played[0].enhancement, Enhancement::Gold);
    assert_eq!(w.played[1].enhancement, Enhancement::Gold);
    // Non-faces untouched.
    w.play(&pair_5s(), &[]);
    assert_eq!(w.played[0].enhancement, Enhancement::None);
}

#[test]
fn midas_mask_feeds_golden_ticket() {
    // Midas golds the Kings in the before window; Golden Ticket then pays
    // $4 per scored Gold card in the same hand.
    let mut w = World::with("SEED", &[JokerId::MidasMask, JokerId::Ticket]);
    let r = w.play(&pair_kings(), &[]);
    assert_eq!(r.dollars_delta, 8);
}

#[test]
fn vampire_strips_enhancements_and_scales() {
    let mut w = World::with("SEED", &[JokerId::Vampire]);
    let mut play = pair_5s();
    play[0].enhancement = Enhancement::Bonus;
    play[1].enhancement = Enhancement::Bonus;
    let r = w.play(&play, &[]);
    // Stripped BEFORE scoring: no +30 bonus chips; X1.2.
    assert_eq!((r.chips, r.mult), (20.0, 2.4));
    assert_eq!(w.played[0].enhancement, Enhancement::None);
    assert_eq!(w.jokers[0].state.x_mult, 1.2);
    // Plain cards: no gain.
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.4); // still X1.2 from before
    assert_eq!(w.jokers[0].state.x_mult, 1.2);
}

#[test]
fn dna_copies_first_hand_single_card() {
    let mut w = World::with("SEED", &[JokerId::Dna]);
    w.env.hands_played_round = 0;
    let mut card = c(5, Suit::Hearts);
    card.seal = Seal::Gold;
    card.enhancement = Enhancement::Bonus;
    w.play(&[card], &[]);
    assert_eq!(w.held.len(), 1);
    assert_eq!(w.held[0].rank.id(), 5);
    assert_eq!(w.held[0].seal, Seal::Gold);
    assert_eq!(w.held[0].enhancement, Enhancement::Bonus);
    assert_ne!(w.held[0].sort_id, card.sort_id);
    // Second hand of the round: no copy.
    w.env.hands_played_round = 1;
    w.play(&[card], &[]);
    assert!(w.held.is_empty());
    // Two cards: no copy.
    w.env.hands_played_round = 0;
    w.play(&pair_5s(), &[]);
    assert!(w.held.is_empty());
}

#[test]
fn dna_copy_triggers_hologram() {
    let mut w = World::with("SEED", &[JokerId::Dna, JokerId::Hologram]);
    w.env.hands_played_round = 0;
    let r = w.play(&[c(5, Suit::Hearts)], &[]);
    // Hologram gains +0.25 during the before window and fires this hand.
    assert_eq!(w.jokers[1].state.x_mult, 1.25);
    assert_eq!(r.mult, 1.25);
}

#[test]
fn obelisk_scales_off_most_played_and_resets() {
    let mut w = World::with("SEED", &[JokerId::Obelisk]);
    // Make High Card the most played hand.
    w.hands.get_mut(HandType::HighCard).played = 5;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(w.jokers[0].state.x_mult, 1.2);
    assert_eq!(r.mult, 2.4); // scales in before, applies the same hand
                             // Now play High Card (most played): resets to 1.
    let r = w.play(&[c(9, Suit::Hearts)], &[]);
    assert_eq!(w.jokers[0].state.x_mult, 1.0);
    assert_eq!(r.mult, 1.0);
}

// ---------------------------------------------------------------------------
// consumable creators
// ---------------------------------------------------------------------------

#[test]
fn vagabond_creates_tarot_when_poor() {
    let mut w = World::with("SEED", &[JokerId::Vagabond]);
    w.env.dollars = 4;
    w.play(&pair_5s(), &[]);
    assert_eq!(
        w.out,
        vec![OutEvent::CreateConsumable(ConsumableSet::Tarot, "vag")]
    );
    w.env.dollars = 5;
    w.play(&pair_5s(), &[]);
    assert!(w.out.is_empty());
}

#[test]
fn seance_creates_spectral_on_straight_flush() {
    let mut w = World::with("SEED", &[JokerId::Seance]);
    let sf = vec![
        c(5, Suit::Hearts),
        c(6, Suit::Hearts),
        c(7, Suit::Hearts),
        c(8, Suit::Hearts),
        c(9, Suit::Hearts),
    ];
    let r = w.play(&sf, &[]);
    assert_eq!(r.hand_type, HandType::StraightFlush);
    assert_eq!(
        w.out,
        vec![OutEvent::CreateConsumable(ConsumableSet::Spectral, "sea")]
    );
    // A plain flush does not.
    let flush = vec![
        c(2, Suit::Hearts),
        c(5, Suit::Hearts),
        c(7, Suit::Hearts),
        c(9, Suit::Hearts),
        c(13, Suit::Hearts),
    ];
    w.play(&flush, &[]);
    assert!(w.out.is_empty());
}

#[test]
fn sixth_sense_destroys_single_six() {
    let mut w = World::with("SEED", &[JokerId::SixthSense]);
    w.env.hands_played_round = 0;
    let r = w.play(&[c(6, Suit::Hearts)], &[]);
    assert_eq!(r.destroyed, vec![0]);
    assert_eq!(
        w.out,
        vec![OutEvent::CreateConsumable(ConsumableSet::Spectral, "sixth")]
    );
    // Not the first hand: no destroy.
    w.env.hands_played_round = 1;
    let r = w.play(&[c(6, Suit::Hearts)], &[]);
    assert!(r.destroyed.is_empty());
    // Two cards: no.
    w.env.hands_played_round = 0;
    let r = w.play(&[c(6, Suit::Hearts), c(6, Suit::Spades)], &[]);
    assert!(r.destroyed.is_empty());
    // Slots full: still destroyed, no spectral (card.lua:2604-2619).
    w.env.consumables_len = 2;
    let r = w.play(&[c(6, Suit::Hearts)], &[]);
    assert_eq!(r.destroyed, vec![0]);
    assert!(w.out.is_empty());
}

// ---------------------------------------------------------------------------
// suit-set jokers (Flower Pot / Seeing Double) and Smeared semantics
// ---------------------------------------------------------------------------

#[test]
fn flower_pot_four_suits() {
    let mut w = World::with("SEED", &[JokerId::FlowerPot]);
    // 2♥3♦4♠5♣6♥ is a straight — all five score.
    let play = vec![
        c(2, Suit::Hearts),
        c(3, Suit::Diamonds),
        c(4, Suit::Spades),
        c(5, Suit::Clubs),
        c(6, Suit::Hearts),
    ];
    let r = w.play(&play, &[]);
    assert_eq!(r.mult, 12.0); // 4 * 3
                              // Missing a suit: no trigger.
    let play = vec![
        c(2, Suit::Hearts),
        c(3, Suit::Hearts),
        c(4, Suit::Spades),
        c(5, Suit::Clubs),
        c(6, Suit::Hearts),
    ];
    let r = w.play(&play, &[]);
    assert_eq!(r.mult, 4.0);
    // A Wild card fills the missing suit (second pass).
    let mut wild = c(3, Suit::Hearts);
    wild.enhancement = Enhancement::Wild;
    let play = vec![
        c(2, Suit::Hearts),
        wild,
        c(4, Suit::Spades),
        c(5, Suit::Clubs),
        c(6, Suit::Hearts),
    ];
    let r = w.play(&play, &[]);
    assert_eq!(r.mult, 12.0);
}

#[test]
fn smeared_flower_pot_reds_count_both_ways() {
    // With Smeared, the second Heart claims the Diamonds slot in the
    // H/D/S/C elseif chain (card.lua:3816-3819 + 4071).
    let mut w = World::with("SEED", &[JokerId::FlowerPot, JokerId::Smeared]);
    let play = vec![
        c(2, Suit::Hearts),
        c(3, Suit::Hearts),
        c(4, Suit::Spades),
        c(5, Suit::Clubs),
        c(6, Suit::Hearts),
    ];
    let r = w.play(&play, &[]);
    assert_eq!(r.hand_type, HandType::Straight);
    assert_eq!(r.mult, 12.0);
}

#[test]
fn seeing_double_needs_club_plus_other() {
    let mut w = World::with("SEED", &[JokerId::SeeingDouble]);
    let r = w.play(&[c(5, Suit::Clubs), c(5, Suit::Hearts)], &[]);
    assert_eq!(r.mult, 4.0); // X2
                             // All clubs: no other suit.
    let r = w.play(&[c(5, Suit::Clubs), c(5, Suit::Clubs)], &[]);
    assert_eq!(r.mult, 2.0);
    // No club: nothing.
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0);
    // A Wild card fills the club slot.
    let mut wild = c(5, Suit::Hearts);
    wild.enhancement = Enhancement::Wild;
    let r = w.play(&[c(5, Suit::Hearts), wild], &[]);
    assert_eq!(r.mult, 4.0);
}

#[test]
fn smeared_seeing_double_spade_counts_as_club() {
    // A Spade is_suit('Clubs') under Smeared: both slots filled by ♠ + ♥.
    let mut w = World::with("SEED", &[JokerId::SeeingDouble, JokerId::Smeared]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 4.0);
}

#[test]
fn smeared_suit_mult_and_flush() {
    // Lusty Joker (+3 per Heart) matches Diamonds under Smeared.
    let mut w = World::with("SEED", &[JokerId::LustyJoker, JokerId::Smeared]);
    let r = w.play(&[c(5, Suit::Diamonds), c(5, Suit::Diamonds)], &[]);
    assert_eq!(r.mult, 8.0); // 2 + 3 + 3

    // Smeared makes a red flush: 3♥4♥7♦8♦J♥.
    let mut w = World::with("SEED", &[JokerId::Smeared]);
    w.mods.smeared = true; // Run::live_eval_mods wires this in real runs
    let play = vec![
        c(3, Suit::Hearts),
        c(4, Suit::Hearts),
        c(7, Suit::Diamonds),
        c(8, Suit::Diamonds),
        c(11, Suit::Hearts),
    ];
    let r = w.play(&play, &[]);
    assert_eq!(r.hand_type, HandType::Flush);
}

// ---------------------------------------------------------------------------
// Pareidolia interactions
// ---------------------------------------------------------------------------

#[test]
fn pareidolia_photograph_first_scored_card() {
    let mut w = World::with("SEED", &[JokerId::Pareidolia, JokerId::Photograph]);
    let r = w.play(&[c(2, Suit::Spades)], &[]);
    assert_eq!((r.chips, r.mult), (7.0, 2.0)); // 5+2 chips, 1 * X2
}

#[test]
fn pareidolia_business_card_rolls_on_everything() {
    let mut w = World::with("BIZ", &[JokerId::Pareidolia, JokerId::Business]);
    let mut probe = RngState::new("BIZ");
    let hit = probe.random("business") < 0.5;
    let r = w.play(&[c(2, Suit::Spades)], &[]);
    assert_eq!(r.dollars_delta, if hit { 2 } else { 0 });
}

#[test]
fn pareidolia_sock_and_buskin_retriggers_all() {
    let mut w = World::with("SEED", &[JokerId::Pareidolia, JokerId::SockAndBuskin]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 30.0); // every card is a face now
}

// ---------------------------------------------------------------------------
// Four Fingers / Shortcut
// ---------------------------------------------------------------------------

#[test]
fn four_fingers_four_card_flush_and_straight() {
    let mut w = World::with("SEED", &[JokerId::FourFingers]);
    w.mods.four_fingers = true;
    // 4-card flush.
    let play = vec![
        c(2, Suit::Hearts),
        c(5, Suit::Hearts),
        c(7, Suit::Hearts),
        c(9, Suit::Hearts),
    ];
    let r = w.play(&play, &[]);
    assert_eq!(r.hand_type, HandType::Flush);
    assert_eq!(r.chips, 35.0 + 2.0 + 5.0 + 7.0 + 9.0);
    // 4-card straight with a duplicate rank: the dupe joins the scoring set
    // (the P2 get_straight quirk).
    let play = vec![
        c(5, Suit::Hearts),
        c(5, Suit::Spades),
        c(6, Suit::Clubs),
        c(7, Suit::Hearts),
        c(8, Suit::Spades),
    ];
    let r = w.play(&play, &[]);
    assert_eq!(r.hand_type, HandType::Straight);
    assert_eq!(r.scoring.len(), 5); // both 5s score
    assert_eq!(r.chips, 30.0 + 5.0 + 5.0 + 6.0 + 7.0 + 8.0);
}

#[test]
fn shortcut_gapped_straight() {
    let mut w = World::with("SEED", &[JokerId::Shortcut]);
    w.mods.shortcut = true;
    let play = vec![
        c(2, Suit::Hearts),
        c(3, Suit::Spades),
        c(5, Suit::Clubs),
        c(6, Suit::Hearts),
        c(7, Suit::Spades),
    ];
    let r = w.play(&play, &[]);
    assert_eq!(r.hand_type, HandType::Straight);
}

// ---------------------------------------------------------------------------
// Blueprint / Brainstorm
// ---------------------------------------------------------------------------

#[test]
fn blueprint_copies_right_neighbour() {
    let mut w = World::with("SEED", &[JokerId::Blueprint, JokerId::Joker]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 10.0); // 2 + 4 (copy) + 4
                              // No right neighbour: inert.
    let mut w = World::with("SEED", &[JokerId::Joker, JokerId::Blueprint]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 6.0);
}

#[test]
fn blueprint_chain_copies_through() {
    let mut w = World::with(
        "SEED",
        &[JokerId::Blueprint, JokerId::Blueprint, JokerId::Joker],
    );
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 14.0); // 2 + 4 + 4 + 4
}

#[test]
fn blueprint_reads_scaling_state_but_never_scales_it() {
    // Green Joker: the before-window +1 is blueprint-gated, but the copy
    // reads the target's live accumulator in joker_main.
    let mut w = World::with("SEED", &[JokerId::Blueprint, JokerId::GreenJoker]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(w.jokers[1].state.mult, 1.0); // scaled ONCE
    assert_eq!(r.mult, 4.0); // 2 + 1 (copy) + 1
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(w.jokers[1].state.mult, 2.0);
    assert_eq!(r.mult, 6.0);
}

#[test]
fn blueprint_copies_todo_list_payout() {
    // To Do List's payout is NOT blueprint-gated: copies pay too.
    let mut w = World::with("SEED", &[JokerId::Blueprint, JokerId::TodoList]);
    w.jokers[1].state.to_do_hand = HandType::Pair;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.dollars_delta, 8);
}

#[test]
fn brainstorm_copies_leftmost() {
    // Brainstorm right of Sock and Buskin: faces retrigger twice.
    let mut w = World::with("SEED", &[JokerId::SockAndBuskin, JokerId::Brainstorm]);
    let r = w.play(&pair_kings(), &[]);
    assert_eq!(r.chips, 70.0); // 10 + 3*10 + 3*10
                               // Brainstorm in slot 1 copies itself -> nothing.
    let mut w = World::with("SEED", &[JokerId::Brainstorm, JokerId::Joker]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 6.0);
}

// ---------------------------------------------------------------------------
// discard-window jokers
// ---------------------------------------------------------------------------

#[test]
fn ramen_decays_per_discarded_card_and_gets_eaten() {
    let mut w = World::with("SEED", &[JokerId::Ramen]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 4.0); // starts at X2
    w.discard(&[c(2, Suit::Hearts), c(3, Suit::Hearts)]);
    assert_eq!(w.jokers[0].state.x_mult, 1.98);
    // At X1.01 or below the next discarded card eats it.
    w.jokers[0].state.x_mult = 1.01;
    w.discard(&[c(2, Suit::Hearts)]);
    assert_eq!(w.out, vec![OutEvent::DestroyJoker(w.jokers[0].sort_id)]);
}

#[test]
fn yorick_counts_23_discards() {
    let mut w = World::with("SEED", &[JokerId::Yorick]);
    assert_eq!(w.jokers[0].state.extra, 23.0);
    w.discard(&[c(2, Suit::Hearts), c(3, Suit::Hearts)]);
    assert_eq!(w.jokers[0].state.extra, 21.0);
    assert_eq!(w.jokers[0].state.x_mult, 1.0);
    // Fast-forward to the last two discards.
    w.jokers[0].state.extra = 2.0;
    w.discard(&[c(2, Suit::Hearts), c(3, Suit::Hearts)]);
    assert_eq!(w.jokers[0].state.x_mult, 2.0);
    assert_eq!(w.jokers[0].state.extra, 23.0); // counter reset
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 4.0);
}

#[test]
fn trading_card_destroys_single_first_discard() {
    let mut w = World::with("SEED", &[JokerId::Trading]);
    w.env.discards_used_round = 0;
    let removed = w.discard(&[c(2, Suit::Hearts)]);
    assert_eq!(removed, vec![true]);
    assert_eq!(w.dollars, 3);
    // Two cards: no.
    let removed = w.discard(&[c(2, Suit::Hearts), c(3, Suit::Hearts)]);
    assert_eq!(removed, vec![false, false]);
    // Not the first discard: no.
    w.env.discards_used_round = 1;
    let removed = w.discard(&[c(2, Suit::Hearts)]);
    assert_eq!(removed, vec![false]);
}

#[test]
fn castle_gains_chips_on_suit_discards() {
    let mut w = World::with("SEED", &[JokerId::Castle]);
    w.env.castle_suit = Suit::Hearts;
    w.discard(&[c(5, Suit::Hearts), c(6, Suit::Hearts), c(7, Suit::Spades)]);
    assert_eq!(w.jokers[0].state.chips, 6.0);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 26.0);
    // A Wild card matches the castle suit.
    let mut wild = c(2, Suit::Clubs);
    wild.enhancement = Enhancement::Wild;
    w.discard(&[wild]);
    assert_eq!(w.jokers[0].state.chips, 9.0);
}

#[test]
fn hit_the_road_scales_on_jacks_and_resets() {
    let mut w = World::with("SEED", &[JokerId::HitTheRoad]);
    w.discard(&[c(11, Suit::Hearts), c(11, Suit::Spades), c(5, Suit::Clubs)]);
    assert_eq!(w.jokers[0].state.x_mult, 2.0);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 4.0);
    // End of round: reset (card.lua:3011-3017).
    w.end_of_round(false);
    assert_eq!(w.jokers[0].state.x_mult, 1.0);
}

#[test]
fn burnt_joker_levels_first_discarded_hand() {
    let mut w = World::with("SEED", &[JokerId::Burnt]);
    w.env.discards_used_round = 0;
    w.discard(&pair_5s());
    assert_eq!(w.hands.get(HandType::Pair).level, 2);
    // Second discard of the round: nothing.
    w.env.discards_used_round = 1;
    w.discard(&pair_5s());
    assert_eq!(w.hands.get(HandType::Pair).level, 2);
    // Blueprint copies Burnt (not gated): two level-ups.
    let mut w = World::with("SEED", &[JokerId::Blueprint, JokerId::Burnt]);
    w.env.discards_used_round = 0;
    w.discard(&pair_5s());
    assert_eq!(w.hands.get(HandType::Pair).level, 3);
}

// ---------------------------------------------------------------------------
// destruction-window jokers (Caino / Glass Joker)
// ---------------------------------------------------------------------------

#[test]
fn caino_counts_destroyed_faces() {
    let mut w = World::with("SEED", &[JokerId::Caino]);
    w.destroyed(&[c(13, Suit::Hearts), c(12, Suit::Spades), c(5, Suit::Clubs)]);
    assert_eq!(w.jokers[0].state.extra, 3.0); // 1 + 2 faces
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 6.0); // X3 via caino_xmult
}

#[test]
fn glass_joker_counts_shattered_glass() {
    let mut w = World::with("SEED", &[JokerId::Glass]);
    let mut g = c(5, Suit::Hearts);
    g.enhancement = Enhancement::Glass;
    w.destroyed(&[g, c(9, Suit::Clubs)]);
    assert_eq!(w.jokers[0].state.x_mult, 1.75);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 3.5);
}

// ---------------------------------------------------------------------------
// Matador and the blind.triggered wiring
// ---------------------------------------------------------------------------

#[test]
fn matador_pays_when_boss_triggers() {
    // The Flint halves mult/chips in modify_hand and sets triggered.
    let mut w = World::with("SEED", &[JokerId::Matador]);
    w.blind = Some(ActiveBlind::new(boss_by_key("bl_flint"), 1));
    let r = w.play(&pair_5s(), &[]);
    // Flint halves the BASE 10/2 before the cards score: 5+5+5 / 1.
    assert_eq!((r.chips, r.mult), (15.0, 1.0));
    assert_eq!(r.dollars_delta, 8);
    // A non-triggering boss pays nothing.
    let mut w = World::with("SEED", &[JokerId::Matador]);
    w.blind = Some(ActiveBlind::new(boss_by_key("bl_wall"), 1));
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.dollars_delta, 0);
}

#[test]
fn matador_pays_on_debuffed_hand() {
    // The Psychic debuffs hands with fewer than 5 cards; Matador's
    // debuffed_hand branch pays (card.lua:2735-2747).
    let mut w = World::with("SEED", &[JokerId::Matador]);
    w.blind = Some(ActiveBlind::new(boss_by_key("bl_psychic"), 1));
    let r = w.play(&pair_5s(), &[]);
    assert!(r.debuffed_hand);
    assert_eq!(r.score, 0.0);
    assert_eq!(r.dollars_delta, 8);
}

// ---------------------------------------------------------------------------
// end-of-round window
// ---------------------------------------------------------------------------

#[test]
fn campfire_scales_on_sales_and_resets_on_boss() {
    let mut w = World::with("SEED", &[JokerId::Campfire]);
    w.jokers[0].state.x_mult = 2.0; // after four sales
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 4.0);
    // Non-boss round end: keeps.
    w.env.blind_is_boss = false;
    w.end_of_round(false);
    assert_eq!(w.jokers[0].state.x_mult, 2.0);
    // Boss round end: resets.
    w.env.blind_is_boss = true;
    w.end_of_round(false);
    assert_eq!(w.jokers[0].state.x_mult, 1.0);
}

#[test]
fn rocket_payout_grows_on_boss_defeat() {
    let mut w = World::with("SEED", &[JokerId::Rocket]);
    assert_eq!(w.dollar_bonus(), 1);
    w.env.blind_is_boss = true;
    w.end_of_round(false);
    assert_eq!(w.jokers[0].state.extra, 3.0);
    assert_eq!(w.dollar_bonus(), 3);
    w.env.blind_is_boss = false;
    w.end_of_round(false);
    assert_eq!(w.dollar_bonus(), 3);
}

#[test]
fn turtle_bean_decays_and_is_eaten() {
    let mut w = World::with("SEED", &[JokerId::TurtleBean]);
    w.end_of_round(false);
    assert_eq!(w.jokers[0].state.extra, 4.0);
    assert_eq!(w.out, vec![OutEvent::TurtleBeanShrink]);
    w.jokers[0].state.extra = 1.0;
    w.end_of_round(false);
    assert_eq!(w.out, vec![OutEvent::DestroyJoker(w.jokers[0].sort_id)]);
}

#[test]
fn invisible_joker_counts_rounds() {
    let mut w = World::with("SEED", &[JokerId::Invisible]);
    w.end_of_round(false);
    w.end_of_round(false);
    assert_eq!(w.jokers[0].state.extra, 2.0);
}

#[test]
fn gift_card_event_queued() {
    let mut w = World::with("SEED", &[JokerId::Gift]);
    w.end_of_round(false);
    assert_eq!(w.out, vec![OutEvent::GiftCard]);
}

#[test]
fn mr_bones_saves_at_quarter_chips() {
    let mut w = World::with("SEED", &[JokerId::MrBones]);
    w.env.blind_chips = 1000.0;
    w.env.game_chips = 250.0;
    let saved = w.end_of_round(true);
    assert!(saved);
    assert_eq!(w.out, vec![OutEvent::DestroyJoker(w.jokers[0].sort_id)]);
    // Below 25%: no save.
    let mut w = World::with("SEED", &[JokerId::MrBones]);
    w.env.blind_chips = 1000.0;
    w.env.game_chips = 249.0;
    assert!(!w.end_of_round(true));
    // Not a loss: nothing.
    let mut w = World::with("SEED", &[JokerId::MrBones]);
    w.env.blind_chips = 1000.0;
    w.env.game_chips = 999.0;
    assert!(!w.end_of_round(false));
    assert!(w.out.is_empty());
}

// ---------------------------------------------------------------------------
// dollar bonuses
// ---------------------------------------------------------------------------

#[test]
fn cloud_nine_rocket_satellite_payouts() {
    let mut w = World::with("SEED", &[JokerId::Cloud9]);
    w.env.nine_tally = 4;
    assert_eq!(w.dollar_bonus(), 4);
    w.env.nine_tally = 0;
    assert_eq!(w.dollar_bonus(), 0);

    let mut w = World::with("SEED", &[JokerId::Satellite]);
    w.env.planets_used = 3;
    assert_eq!(w.dollar_bonus(), 3);
    w.env.planets_used = 0;
    assert_eq!(w.dollar_bonus(), 0);
}

// ---------------------------------------------------------------------------
// RNG-driven per-card jokers
// ---------------------------------------------------------------------------

#[test]
fn lucky_cat_scales_on_lucky_triggers() {
    let mut w = World::with("LUCK", &[JokerId::LuckyCat]);
    let mut probe = RngState::new("LUCK");
    let mult_hit = probe.random("lucky_mult") < 0.2;
    let money_hit = probe.random("lucky_money") < 1.0 / 15.0;
    let mut lucky = c(5, Suit::Hearts);
    lucky.enhancement = Enhancement::Lucky;
    w.play(&[lucky], &[]);
    let expect = if mult_hit || money_hit { 1.25 } else { 1.0 };
    assert_eq!(w.jokers[0].state.x_mult, expect);
}

// ---------------------------------------------------------------------------
// Oops! All 6s probability doubling (through ScoreContext.prob_normal)
// ---------------------------------------------------------------------------

#[test]
fn doubled_probability_flips_lucky_roll() {
    // Find a seed region where the base roll misses but the doubled one
    // hits: roll in [1/5, 2/5).
    let mut base = World::new("ODDS");
    let mut lucky = c(5, Suit::Hearts);
    lucky.enhancement = Enhancement::Lucky;
    let mut probe = RngState::new("ODDS");
    let roll = probe.random("lucky_mult");
    let r1 = base.play(&[lucky], &[]);
    let mut doubled = World::new("ODDS");
    doubled.env.prob_normal = 2.0;
    let r2 = doubled.play(&[lucky], &[]);
    // Same stream, different thresholds.
    assert_eq!(r1.mult > 1.0, roll < 0.2);
    assert_eq!(r2.mult > 1.0, roll < 0.4);
}
