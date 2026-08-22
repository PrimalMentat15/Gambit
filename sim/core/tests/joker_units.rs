//! Per-joker unit tests for every Common (rarity 1) joker: exact arithmetic
//! through `evaluate_play` with the real `EngineHooks` dispatcher, plus
//! direct hook-window tests for the discard/end-of-round/payout effects.
//! Expected values are derived by hand from the ported formulas; RNG-driven
//! jokers compare against a fresh clone of the seed's stream (per-key
//! streams are independent, so the n-th draw of e.g. '8ball' is the same
//! whether or not other streams were touched).

use std::sync::atomic::{AtomicU32, Ordering};

use balatro_core::cards::{Card, Edition, Enhancement, HandType, Rank, Suit};
use balatro_core::handeval::EvalMods;
use balatro_core::items::{element_index, ConsumableSet, JokerId};
use balatro_core::jokers::{resolve_effective, EngineHooks, JokerEnv, JokerState, OutEvent};
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

/// evaluate_play harness with the live joker engine.
struct World {
    rng: RngState,
    hands: HandsTable,
    dollars: i64,
    jokers: Vec<OwnedJoker>,
    env: JokerEnv,
    out: Vec<OutEvent>,
}

impl World {
    fn new(seed: &str) -> Self {
        World {
            rng: RngState::new(seed),
            hands: HandsTable::new(),
            dollars: 0,
            jokers: Vec::new(),
            env: JokerEnv {
                prob_normal: 1.0,
                consumable_slots: 2,
                joker_slots: 5,
                ..JokerEnv::default()
            },
            out: Vec::new(),
        }
    }

    fn with(seed: &str, ids: &[JokerId]) -> Self {
        let mut w = Self::new(seed);
        w.jokers = ids.iter().map(|&id| j(id)).collect();
        w
    }

    fn play(&mut self, play: &[Card], held: &[Card]) -> PlayResult {
        let mut hooks = EngineHooks::new(&mut self.jokers, self.env.clone());
        let mut ctx = ScoreContext {
            rng: &mut self.rng,
            hands: &mut self.hands,
            dollars: &mut self.dollars,
            blind: None,
            most_played: HandType::HighCard,
            mods: EvalMods::default(),
            prob_normal: 1.0,
            plasma_balance: false,
        };
        let mut play = play.to_vec();
        let mut held = held.to_vec();
        let r = evaluate_play(&mut ctx, &mut play, &mut held, &mut hooks);
        self.out = hooks.out.events;
        r
    }

    fn end_of_round(&mut self) {
        let mut hooks = EngineHooks::new(&mut self.jokers, self.env.clone());
        hooks.end_of_round(&self.hands, false, &mut self.rng);
        self.out = hooks.out.events;
    }

    fn discard(&mut self, selected: &[Card]) {
        let mut hooks = EngineHooks::new(&mut self.jokers, self.env.clone());
        for card in selected {
            hooks.on_discard_card(card, selected, &mut self.dollars, &mut self.rng);
        }
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

// ---------------------------------------------------------------------------
// flat / hand-type / suit jokers
// ---------------------------------------------------------------------------

#[test]
fn joker_flat_mult() {
    // Pair of 5s: 10 + 5 + 5 = 20 chips; mult 2 + 4 (Joker) = 6.
    let mut w = World::with("SEED", &[JokerId::Joker]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!((r.chips, r.mult, r.score), (20.0, 6.0, 120.0));
}

#[test]
fn suit_mult_jokers() {
    // Greedy: +3 mult per scored Diamond.
    let mut w = World::with("SEED", &[JokerId::GreedyJoker]);
    let r = w.play(&[c(5, Suit::Diamonds), c(5, Suit::Diamonds)], &[]);
    assert_eq!((r.chips, r.mult), (20.0, 8.0)); // 2 + 3 + 3

    // Lusty/Wrathful/Gluttonous on a mixed pair: exactly one match each.
    for (id, suit) in [
        (JokerId::LustyJoker, Suit::Hearts),
        (JokerId::WrathfulJoker, Suit::Spades),
        (JokerId::GluttenousJoker, Suit::Clubs),
    ] {
        let mut w = World::with("SEED", &[id]);
        let other = if suit == Suit::Clubs {
            Suit::Hearts
        } else {
            Suit::Clubs
        };
        let r = w.play(&[c(9, suit), c(9, other)], &[]);
        assert_eq!(r.mult, 5.0, "{id:?}"); // 2 + 3
    }

    // A Wild card matches any suit (card.lua:4069).
    let mut w = World::with("SEED", &[JokerId::GreedyJoker]);
    let mut wild = c(5, Suit::Clubs);
    wild.enhancement = Enhancement::Wild;
    let r = w.play(&[wild, c(5, Suit::Clubs)], &[]);
    assert_eq!(r.mult, 5.0); // only the Wild matches Diamonds
}

#[test]
fn type_mult_and_chip_jokers() {
    // Jolly on a plain pair: +8 mult.
    let mut w = World::with("SEED", &[JokerId::Jolly]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 10.0); // 2 + 8

    // A Full House CONTAINS a Pair/Three of a Kind/Two Pair —
    // `next(context.poker_hands[type])` (card.lua:3661-3671).
    let fh = vec![
        c(9, Suit::Hearts),
        c(9, Suit::Spades),
        c(9, Suit::Clubs),
        c(4, Suit::Hearts),
        c(4, Suit::Spades),
    ];
    for (id, add) in [
        (JokerId::Jolly, 8.0),
        (JokerId::Zany, 12.0),
        (JokerId::Mad, 10.0),
    ] {
        let mut w = World::with("SEED", &[id]);
        let r = w.play(&fh.clone(), &[]);
        assert_eq!(r.mult, 4.0 + add, "{id:?}"); // Full House base mult 4
    }
    // Crazy (Straight) and Droll (Flush) do NOT trigger on a full house.
    for id in [JokerId::Crazy, JokerId::Droll] {
        let mut w = World::with("SEED", &[id]);
        let r = w.play(&fh.clone(), &[]);
        assert_eq!(r.mult, 4.0, "{id:?}");
    }

    // Chip versions: Sly (+50), Wily (+100), Clever (+80) on the full house
    // (40 base + 9*3 + 4*2 = 75 chips).
    for (id, add) in [
        (JokerId::Sly, 50.0),
        (JokerId::Wily, 100.0),
        (JokerId::Clever, 80.0),
    ] {
        let mut w = World::with("SEED", &[id]);
        let r = w.play(&fh.clone(), &[]);
        assert_eq!(r.chips, 75.0 + add, "{id:?}");
    }
    // Devious (Straight) and Crafty (Flush) on matching hands.
    let straight = vec![
        c(2, Suit::Hearts),
        c(3, Suit::Spades),
        c(4, Suit::Clubs),
        c(5, Suit::Hearts),
        c(6, Suit::Diamonds),
    ];
    let mut w = World::with("SEED", &[JokerId::Devious]);
    let r = w.play(&straight, &[]);
    assert_eq!(r.chips, 30.0 + 20.0 + 100.0); // base 30 + ranks 2..6 + 100
    let flush = vec![
        c(2, Suit::Hearts),
        c(4, Suit::Hearts),
        c(6, Suit::Hearts),
        c(8, Suit::Hearts),
        c(10, Suit::Hearts),
    ];
    let mut w = World::with("SEED", &[JokerId::Crafty]);
    let r = w.play(&flush, &[]);
    assert_eq!(r.chips, 35.0 + 30.0 + 80.0);
    let mut w = World::with("SEED", &[JokerId::Crazy]);
    let r = w.play(&straight, &[]);
    assert_eq!(r.mult, 4.0 + 12.0);
    let mut w = World::with("SEED", &[JokerId::Droll]);
    let r = w.play(&flush, &[]);
    assert_eq!(r.mult, 4.0 + 10.0);
}

#[test]
fn half_joker() {
    // 3 or fewer cards PLAYED: +20 mult (card.lua:3673-3678).
    let mut w = World::with("SEED", &[JokerId::Half]);
    let r = w.play(
        &[c(5, Suit::Hearts), c(5, Suit::Clubs), c(9, Suit::Spades)],
        &[],
    );
    assert_eq!(r.mult, 22.0); // pair base 2 + 20
    let r = w.play(
        &[
            c(5, Suit::Hearts),
            c(5, Suit::Clubs),
            c(9, Suit::Spades),
            c(3, Suit::Spades),
        ],
        &[],
    );
    assert_eq!(r.mult, 2.0); // 4 played: no bonus
}

// ---------------------------------------------------------------------------
// counter / env readers
// ---------------------------------------------------------------------------

#[test]
fn banner_and_mystic_summit() {
    let mut w = World::with("SEED", &[JokerId::Banner]);
    w.env.discards_left = 3;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 20.0 + 90.0); // +30 per remaining discard

    let mut w = World::with("SEED", &[JokerId::MysticSummit]);
    w.env.discards_left = 0;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 17.0); // 2 + 15
    let mut w = World::with("SEED", &[JokerId::MysticSummit]);
    w.env.discards_left = 1;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0);
}

#[test]
fn abstract_counts_all_jokers() {
    // Abstract first: mult 2 + 2*3 (two jokers) + 4 (Joker) = 12.
    let mut w = World::with("SEED", &[JokerId::Abstract, JokerId::Joker]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 12.0);
}

#[test]
fn blue_joker_deck_size() {
    let mut w = World::with("SEED", &[JokerId::BlueJoker]);
    w.env.deck_len = 52;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 20.0 + 104.0);
    w.env.deck_len = 0;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 20.0);
}

#[test]
fn supernova_counts_this_play() {
    let mut w = World::with("SEED", &[JokerId::Supernova]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 3.0); // played = 1 already includes this hand
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 4.0); // played = 2
}

#[test]
fn fortune_teller_reads_tarot_usage() {
    let mut w = World::with("SEED", &[JokerId::FortuneTeller]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0); // 0 tarots used: no effect (mult > 0 gate)
    w.env.tarots_used = 3;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 5.0);
}

#[test]
fn swashbuckler_sums_other_sell_values() {
    // Joker costs $2 -> sell 1. Swashbuckler alone: no effect.
    let mut w = World::with("SEED", &[JokerId::Swashbuckler]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0);
    let mut w = World::with("SEED", &[JokerId::Swashbuckler, JokerId::Joker]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0 + 1.0 + 4.0); // swash +1, then Joker +4
}

#[test]
fn misprint_rolls_its_stream() {
    let mut w = World::with("SEED", &[JokerId::Misprint]);
    let expected = RngState::new("SEED").random_range("misprint", 0, 23) as f64;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 2.0 + expected);
}

// ---------------------------------------------------------------------------
// per-card scoring jokers
// ---------------------------------------------------------------------------

#[test]
fn even_steven_and_odd_todd_and_friends() {
    // 10 (even) + 9 (odd) + A (odd) high-card-ish: play as three cards.
    let play = vec![c(10, Suit::Hearts), c(10, Suit::Spades), c(9, Suit::Clubs)];
    let mut w = World::with("SEED", &[JokerId::EvenSteven]);
    let r = w.play(&play.clone(), &[]);
    // Pair of 10s scores (9 kicker not in scoring set): +4 per even card.
    assert_eq!(r.mult, 2.0 + 8.0);

    let mut w = World::with("SEED", &[JokerId::OddTodd]);
    let r = w.play(
        &[c(9, Suit::Hearts), c(9, Suit::Clubs), c(14, Suit::Clubs)],
        &[],
    );
    // Pair of 9s: two odd scoring cards (+31 chips each); the Ace kicker
    // does not score.
    assert_eq!(r.chips, 10.0 + 18.0 + 62.0);

    let mut w = World::with("SEED", &[JokerId::Scholar]);
    let r = w.play(&[c(14, Suit::Hearts), c(14, Suit::Clubs)], &[]);
    // Pair of aces: chips 10 + 11 + 20 + 11 + 20; mult 2 + 4 + 4.
    assert_eq!((r.chips, r.mult), (72.0, 10.0));

    let mut w = World::with("SEED", &[JokerId::WalkieTalkie]);
    let r = w.play(
        &[c(10, Suit::Hearts), c(4, Suit::Clubs), c(4, Suit::Spades)],
        &[],
    );
    // Pair of 4s scores + the 10 kicker does not... 4s: +10c +4m each.
    assert_eq!((r.chips, r.mult), (10.0 + 8.0 + 20.0, 2.0 + 8.0));
}

#[test]
fn scary_face_and_smiley() {
    let play = vec![c(13, Suit::Hearts), c(13, Suit::Clubs)];
    let mut w = World::with("SEED", &[JokerId::ScaryFace]);
    let r = w.play(&play.clone(), &[]);
    assert_eq!(r.chips, 10.0 + 10.0 + 10.0 + 60.0); // +30 per face

    let mut w = World::with("SEED", &[JokerId::Smiley]);
    let r = w.play(&play, &[]);
    assert_eq!(r.mult, 2.0 + 10.0); // +5 per face
}

#[test]
fn golden_ticket_pays_for_gold_cards() {
    let mut w = World::with("SEED", &[JokerId::Ticket]);
    let mut g = c(5, Suit::Hearts);
    g.enhancement = Enhancement::Gold;
    let r = w.play(&[g, c(5, Suit::Clubs)], &[]);
    assert_eq!(r.dollars_delta, 4);
}

#[test]
fn photograph_first_face_only() {
    // K K pair: the FIRST scored face gets x2, applied at that card's eval:
    // chips 10+10 -> mult 2*2=4 -> second K chips +10. Score (30)*4.
    let mut w = World::with("SEED", &[JokerId::Photograph]);
    let r = w.play(&[c(13, Suit::Hearts), c(13, Suit::Clubs)], &[]);
    assert_eq!((r.chips, r.mult, r.score), (30.0, 4.0, 120.0));
}

#[test]
fn business_card_rolls_per_face_rep() {
    let mut w = World::with("SEED", &[JokerId::Business]);
    let mut mirror = RngState::new("SEED");
    let expected: i64 = (0..2).filter(|_| mirror.random("business") < 0.5).count() as i64 * 2;
    let r = w.play(&[c(12, Suit::Hearts), c(12, Suit::Clubs)], &[]);
    assert_eq!(r.dollars_delta, expected);
}

#[test]
fn eight_ball_creates_tarots_with_gate() {
    let mut w = World::with("SEED", &[JokerId::EightBall]);
    let mut mirror = RngState::new("SEED");
    let expected = (0..3).filter(|_| mirror.random("8ball") < 0.25).count();
    let r = w.play(
        &[c(8, Suit::Hearts), c(8, Suit::Clubs), c(8, Suit::Spades)],
        &[],
    );
    assert_eq!(r.hand_type, HandType::ThreeOfAKind);
    let creates = w
        .out
        .iter()
        .filter(|e| matches!(e, OutEvent::CreateConsumable(ConsumableSet::Tarot, "8ba")))
        .count();
    assert_eq!(creates, expected.min(2)); // capped by the 2 consumable slots

    // No slots -> the gate short-circuits BEFORE the roll: no '8ball' draws.
    let mut w = World::with("SEED2", &[JokerId::EightBall]);
    w.env.consumable_slots = 0;
    w.play(&[c(8, Suit::Hearts), c(8, Suit::Clubs)], &[]);
    assert!(w.out.is_empty());
    // The '8ball' stream was never touched: its next draw equals a fresh one.
    assert_eq!(
        w.rng.random("8ball"),
        RngState::new("SEED2").random("8ball")
    );
}

#[test]
fn superposition_needs_ace_and_straight() {
    let mut w = World::with("SEED", &[JokerId::Superposition]);
    let wheel = vec![
        c(14, Suit::Hearts),
        c(2, Suit::Clubs),
        c(3, Suit::Spades),
        c(4, Suit::Hearts),
        c(5, Suit::Diamonds),
    ];
    w.play(&wheel, &[]);
    assert_eq!(
        w.out,
        vec![OutEvent::CreateConsumable(ConsumableSet::Tarot, "sup")]
    );
    // Straight without an ace: nothing.
    let mut w = World::with("SEED", &[JokerId::Superposition]);
    let s = vec![
        c(6, Suit::Hearts),
        c(2, Suit::Clubs),
        c(3, Suit::Spades),
        c(4, Suit::Hearts),
        c(5, Suit::Diamonds),
    ];
    w.play(&s, &[]);
    assert!(w.out.is_empty());
}

// ---------------------------------------------------------------------------
// held-card jokers
// ---------------------------------------------------------------------------

#[test]
fn raised_fist_doubles_lowest_held_rank() {
    // Held K, 3, 7 -> lowest is the 3: h_mult 2*3 = 6.
    let mut w = World::with("SEED", &[JokerId::RaisedFist]);
    let held = vec![c(13, Suit::Hearts), c(3, Suit::Clubs), c(7, Suit::Spades)];
    let r = w.play(&[c(2, Suit::Hearts)], &held);
    // High card 2: chips 5 + 2 = 7, mult 1 + 6 = 7.
    assert_eq!((r.chips, r.mult), (7.0, 7.0));

    // Ace is the HIGHEST id (14) — a lone held ace pays 2*11 = 22.
    let mut w = World::with("SEED", &[JokerId::RaisedFist]);
    let r = w.play(&[c(2, Suit::Hearts)], &[c(14, Suit::Clubs)]);
    assert_eq!(r.mult, 23.0);

    // Stone cards are skipped by the scan (card.lua:3325).
    let mut w = World::with("SEED", &[JokerId::RaisedFist]);
    let mut stone = c(2, Suit::Clubs);
    stone.enhancement = Enhancement::Stone;
    let r = w.play(&[c(2, Suit::Hearts)], &[stone, c(9, Suit::Clubs)]);
    assert_eq!(r.mult, 1.0 + 18.0);
}

#[test]
fn shoot_the_moon_pays_held_queens() {
    let mut w = World::with("SEED", &[JokerId::ShootTheMoon]);
    let r = w.play(
        &[c(2, Suit::Hearts)],
        &[c(12, Suit::Clubs), c(12, Suit::Spades), c(5, Suit::Hearts)],
    );
    assert_eq!(r.mult, 1.0 + 13.0 + 13.0);

    // A debuffed Queen yields the message-only entry (no mult).
    let mut w = World::with("SEED", &[JokerId::ShootTheMoon]);
    let mut q = c(12, Suit::Clubs);
    q.debuff = true;
    let r = w.play(&[c(2, Suit::Hearts)], &[q]);
    assert_eq!(r.mult, 1.0);
}

#[test]
fn reserved_parking_rolls_per_held_face() {
    let mut w = World::with("SEED", &[JokerId::ReservedParking]);
    let mut mirror = RngState::new("SEED");
    let expected: i64 = (0..2).filter(|_| mirror.random("parking") < 0.5).count() as i64;
    let r = w.play(
        &[c(2, Suit::Hearts)],
        &[c(13, Suit::Clubs), c(11, Suit::Spades)],
    );
    assert_eq!(r.dollars_delta, expected);

    // Debuffed faces never roll (is_face is nil before the 'parking' draw).
    let mut w = World::with("SEED3", &[JokerId::ReservedParking]);
    let mut k = c(13, Suit::Clubs);
    k.debuff = true;
    w.play(&[c(2, Suit::Hearts)], &[k]);
    assert_eq!(
        w.rng.random("parking"),
        RngState::new("SEED3").random("parking")
    );
}

// ---------------------------------------------------------------------------
// scaling jokers across multiple hands / discards / rounds
// ---------------------------------------------------------------------------

#[test]
fn ride_the_bus_scales_and_resets() {
    let mut w = World::with("SEED", &[JokerId::RideTheBus]);
    // Hand 1, no faces: before-hand bumps to 1, joker_main adds it.
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 3.0);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 4.0); // counter now 2
                             // A scored face resets BEFORE the main effect: no bonus at all.
    let r = w.play(&[c(13, Suit::Hearts)], &[]);
    assert_eq!(r.mult, 1.0);
    assert_eq!(w.jokers[0].state.mult, 0.0);
}

#[test]
fn green_joker_scales_up_and_down() {
    let mut w = World::with("SEED", &[JokerId::GreenJoker]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 3.0); // +1 hand, then main
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 4.0);
    // One discard action: -1, keyed to the last selected card.
    let sel = vec![c(9, Suit::Hearts), c(4, Suit::Clubs)];
    w.discard(&sel);
    assert_eq!(w.jokers[0].state.mult, 1.0);
    // Floors at 0.
    w.discard(&sel);
    w.discard(&sel);
    assert_eq!(w.jokers[0].state.mult, 0.0);
}

#[test]
fn square_joker_counts_4_card_hands() {
    let mut w = World::with("SEED", &[JokerId::Square]);
    let four = vec![
        c(5, Suit::Hearts),
        c(5, Suit::Clubs),
        c(9, Suit::Spades),
        c(3, Suit::Diamonds),
    ];
    let r = w.play(&four.clone(), &[]);
    assert_eq!(r.chips, 10.0 + 10.0 + 4.0); // pair of 5s + fresh +4
    let r = w.play(&four, &[]);
    assert_eq!(r.chips, 10.0 + 10.0 + 8.0);
    // 5 cards played: no growth (chips stay 8).
    let r = w.play(
        &[
            c(5, Suit::Hearts),
            c(5, Suit::Clubs),
            c(9, Suit::Spades),
            c(3, Suit::Diamonds),
            c(2, Suit::Diamonds),
        ],
        &[],
    );
    assert_eq!(r.chips, 10.0 + 10.0 + 8.0);
}

#[test]
fn runner_grows_on_straights() {
    let mut w = World::with("SEED", &[JokerId::Runner]);
    let straight = vec![
        c(2, Suit::Hearts),
        c(3, Suit::Spades),
        c(4, Suit::Clubs),
        c(5, Suit::Hearts),
        c(6, Suit::Diamonds),
    ];
    // First straight: before-hand +15, main applies it the same hand.
    let r = w.play(&straight.clone(), &[]);
    assert_eq!(r.chips, 30.0 + 20.0 + 15.0);
    // Non-straight: no growth, accumulated chips still paid.
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 10.0 + 10.0 + 15.0);
    let r = w.play(&straight, &[]);
    assert_eq!(r.chips, 30.0 + 20.0 + 30.0);
}

#[test]
fn ice_cream_melts() {
    let mut w = World::with("SEED", &[JokerId::IceCream]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 20.0 + 100.0);
    assert_eq!(w.jokers[0].state.chips, 95.0); // -5 in `after`
    w.jokers[0].state.chips = 5.0;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 20.0 + 5.0); // still scores its last hand
    assert_eq!(w.out, vec![OutEvent::DestroyJoker(w.jokers[0].sort_id)]);
}

#[test]
fn popcorn_shrinks_each_round() {
    let mut w = World::with("SEED", &[JokerId::Popcorn]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 22.0); // starts at 20
    w.end_of_round();
    assert_eq!(w.jokers[0].state.mult, 16.0);
    for _ in 0..3 {
        w.end_of_round();
    }
    assert_eq!(w.jokers[0].state.mult, 4.0);
    // 4 - 4 <= 0: eaten (checked before the decrement).
    w.end_of_round();
    assert_eq!(w.out, vec![OutEvent::DestroyJoker(w.jokers[0].sort_id)]);
}

// ---------------------------------------------------------------------------
// end-of-round / payout jokers
// ---------------------------------------------------------------------------

#[test]
fn egg_gains_sell_value() {
    let mut w = World::with("SEED", &[JokerId::Egg]);
    w.end_of_round();
    w.end_of_round();
    assert_eq!(w.jokers[0].extra_value, 6);
}

#[test]
fn gros_michel_and_cavendish_extinction() {
    // Find a seed whose first 'gros_michel' roll goes extinct, and one that
    // stays safe, then assert both behaviours.
    let mut extinct_seed = None;
    let mut safe_seed = None;
    for i in 0..200 {
        let s = format!("GM{i}");
        let roll = RngState::new(&s).random("gros_michel");
        if roll < 1.0 / 6.0 && extinct_seed.is_none() {
            extinct_seed = Some(s);
        } else if roll >= 1.0 / 6.0 && safe_seed.is_none() {
            safe_seed = Some(s);
        }
    }
    let (es, ss) = (extinct_seed.unwrap(), safe_seed.unwrap());

    let mut w = World::with(&es, &[JokerId::GrosMichel]);
    let sid = w.jokers[0].sort_id;
    w.end_of_round();
    assert_eq!(
        w.out,
        vec![OutEvent::DestroyJoker(sid), OutEvent::SetGrosMichelExtinct]
    );

    let mut w = World::with(&ss, &[JokerId::GrosMichel]);
    w.end_of_round();
    assert!(w.out.is_empty());
    // Its flat +15 mult while alive.
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 17.0);

    // Cavendish: X3 mult, 1-in-1000 destruction.
    let mut w = World::with("SEED", &[JokerId::Cavendish]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 6.0);
    let survives = RngState::new("SEED").random("cavendish") >= 1.0 / 1000.0;
    w.end_of_round();
    assert_eq!(w.out.is_empty(), survives);
}

#[test]
fn to_do_list_pays_and_rerolls() {
    let mut w = World::with("SEED", &[JokerId::TodoList]);
    w.jokers[0].state.to_do_hand = HandType::Pair;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.dollars_delta, 4);
    let r = w.play(&[c(2, Suit::Hearts)], &[]);
    assert_eq!(r.dollars_delta, 0);

    // End-of-round re-roll: single 'to_do' draw over the visible hands
    // minus the current one (pairs() order = MOST_PLAYED_SCAN).
    let pool = [
        HandType::HighCard,
        HandType::StraightFlush,
        HandType::FourOfAKind,
        HandType::FullHouse,
        HandType::Flush,
        HandType::Straight,
        HandType::ThreeOfAKind,
        HandType::TwoPair,
    ];
    let mut mirror = w.rng.clone();
    let expected = pool[element_index(&mut mirror, "to_do", pool.len())];
    w.end_of_round();
    assert_eq!(w.jokers[0].state.to_do_hand, expected);
    assert_ne!(w.jokers[0].state.to_do_hand, HandType::Pair);
}

#[test]
fn golden_and_delayed_gratification_payouts() {
    let mut w = World::with("SEED", &[JokerId::Golden]);
    assert_eq!(w.dollar_bonus(), 4);

    let mut w = World::with("SEED", &[JokerId::DelayedGrat]);
    w.env.discards_left = 3;
    w.env.discards_used_round = 0;
    assert_eq!(w.dollar_bonus(), 6);
    w.env.discards_used_round = 1;
    assert_eq!(w.dollar_bonus(), 0);
    w.env.discards_used_round = 0;
    w.env.discards_left = 0;
    assert_eq!(w.dollar_bonus(), 0);
}

// ---------------------------------------------------------------------------
// discard jokers
// ---------------------------------------------------------------------------

#[test]
fn faceless_joker_pays_for_face_dumps() {
    let mut w = World::with("SEED", &[JokerId::Faceless]);
    let sel = vec![c(11, Suit::Hearts), c(12, Suit::Clubs), c(13, Suit::Spades)];
    w.discard(&sel);
    assert_eq!(w.dollars, 5);
    // Two faces: nothing.
    let mut w = World::with("SEED", &[JokerId::Faceless]);
    let sel = vec![c(11, Suit::Hearts), c(12, Suit::Clubs), c(5, Suit::Spades)];
    w.discard(&sel);
    assert_eq!(w.dollars, 0);
}

#[test]
fn mail_in_rebate_pays_per_matching_rank() {
    let mut w = World::with("SEED", &[JokerId::Mail]);
    w.env.mail_card_id = 7;
    let sel = vec![c(7, Suit::Hearts), c(7, Suit::Clubs), c(5, Suit::Spades)];
    w.discard(&sel);
    assert_eq!(w.dollars, 10); // $5 per 7

    // Stone cards never match (random negative get_id).
    let mut w = World::with("SEED", &[JokerId::Mail]);
    w.env.mail_card_id = 7;
    let mut stone = c(7, Suit::Hearts);
    stone.enhancement = Enhancement::Stone;
    w.discard(&[stone]);
    assert_eq!(w.dollars, 0);
}

// ---------------------------------------------------------------------------
// splash / retriggers
// ---------------------------------------------------------------------------

#[test]
fn splash_scores_every_played_card() {
    // Pair of 5s + 9 kicker: without Splash the 9 does not score.
    let play = vec![c(5, Suit::Hearts), c(5, Suit::Clubs), c(9, Suit::Spades)];
    let mut w = World::new("SEED");
    let r = w.play(&play.clone(), &[]);
    assert_eq!((r.scoring.len(), r.chips), (2, 20.0));

    let mut w = World::with("SEED", &[JokerId::Splash]);
    let r = w.play(&play, &[]);
    assert_eq!((r.scoring.len(), r.chips), (3, 29.0));

    // A debuffed Splash does nothing (find_joker skips debuffed jokers).
    let mut w = World::with("SEED", &[JokerId::Splash]);
    w.jokers[0].debuffed = true;
    let r = w.play(
        &[c(5, Suit::Hearts), c(5, Suit::Clubs), c(9, Suit::Spades)],
        &[],
    );
    assert_eq!(r.scoring.len(), 2);
}

#[test]
fn hanging_chad_retriggers_first_scoring_card_twice() {
    // Lone 5: 1 + 2 repetitions -> chips 5 + 3*5 = 20.
    let mut w = World::with("SEED", &[JokerId::HangingChad]);
    let r = w.play(&[c(5, Suit::Hearts)], &[]);
    assert_eq!(r.chips, 20.0);
    // Pair: only the first card (play order) retriggers.
    let mut w = World::with("SEED", &[JokerId::HangingChad]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.chips, 10.0 + 3.0 * 5.0 + 5.0);
}

// ---------------------------------------------------------------------------
// dispatcher: order, editions, debuffs
// ---------------------------------------------------------------------------

#[test]
fn joker_order_matters_for_xmult() {
    // [Joker, Cavendish]: (2+4)*3 = 18; [Cavendish, Joker]: 2*3+4 = 10.
    let mut w = World::with("SEED", &[JokerId::Joker, JokerId::Cavendish]);
    assert_eq!(w.play(&pair_5s(), &[]).mult, 18.0);
    let mut w = World::with("SEED", &[JokerId::Cavendish, JokerId::Joker]);
    assert_eq!(w.play(&pair_5s(), &[]).mult, 10.0);
}

#[test]
fn joker_edition_positions() {
    // Foil: +50 chips BEFORE the joker's own effect.
    let mut w = World::with("SEED", &[JokerId::Joker]);
    w.jokers[0].edition = Edition::Foil;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!((r.chips, r.mult), (70.0, 6.0));
    // Holo: +10 mult before the +4.
    let mut w = World::with("SEED", &[JokerId::Joker]);
    w.jokers[0].edition = Edition::Holo;
    assert_eq!(w.play(&pair_5s(), &[]).mult, 16.0);
    // Polychrome: x1.5 AFTER the joker's own effect: (2+4)*1.5 = 9.
    let mut w = World::with("SEED", &[JokerId::Joker]);
    w.jokers[0].edition = Edition::Polychrome;
    assert_eq!(w.play(&pair_5s(), &[]).mult, 9.0);
    // Polychrome Cavendish: own X3 then edition x1.5: 2*3*1.5 = 9.
    let mut w = World::with("SEED", &[JokerId::Cavendish]);
    w.jokers[0].edition = Edition::Polychrome;
    assert_eq!(w.play(&pair_5s(), &[]).mult, 9.0);
}

#[test]
fn debuffed_jokers_are_fully_inert_but_flipped_ones_work() {
    let mut w = World::with("SEED", &[JokerId::Joker]);
    w.jokers[0].debuffed = true;
    w.jokers[0].edition = Edition::Foil;
    let r = w.play(&pair_5s(), &[]);
    assert_eq!((r.chips, r.mult), (20.0, 2.0)); // no edition, no effect

    let mut w = World::with("SEED", &[JokerId::Joker]);
    w.jokers[0].flipped = true; // Amber Acorn flip: still calculates
    assert_eq!(w.play(&pair_5s(), &[]).mult, 6.0);
}

#[test]
fn observatory_planets_multiply_matching_hand() {
    let mut w = World::new("SEED");
    w.env.observatory = true;
    w.env.consumable_keys = vec!["c_mercury"]; // Pair planet
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(r.mult, 3.0); // 2 * 1.5
    let r = w.play(&[c(2, Suit::Hearts)], &[]);
    assert_eq!(r.mult, 1.0); // High Card: no match
}

// ---------------------------------------------------------------------------
// copy resolution (Blueprint / Brainstorm groundwork)
// ---------------------------------------------------------------------------

#[test]
fn resolve_effective_chains() {
    let jk = |id| j(id);
    // Blueprint copies its right neighbour.
    let v = vec![jk(JokerId::Blueprint), jk(JokerId::Joker)];
    assert_eq!(resolve_effective(&v, 0), Some((1, true)));
    assert_eq!(resolve_effective(&v, 1), Some((1, false)));
    // Blueprint with no right neighbour: nothing.
    let v = vec![jk(JokerId::Joker), jk(JokerId::Blueprint)];
    assert_eq!(resolve_effective(&v, 1), None);
    // Brainstorm copies the leftmost; leftmost Brainstorm is self -> None.
    let v = vec![jk(JokerId::Joker), jk(JokerId::Brainstorm)];
    assert_eq!(resolve_effective(&v, 1), Some((0, true)));
    let v = vec![jk(JokerId::Brainstorm), jk(JokerId::Joker)];
    assert_eq!(resolve_effective(&v, 0), None);
    // Chained copies: Blueprint -> Blueprint -> Joker.
    let v = vec![
        jk(JokerId::Blueprint),
        jk(JokerId::Blueprint),
        jk(JokerId::Joker),
    ];
    assert_eq!(resolve_effective(&v, 0), Some((2, true)));
    // Cycle: Blueprint -> Brainstorm -> Blueprint -> ... terminates at
    // depth > len + 1 (card.lua:2312).
    let v = vec![jk(JokerId::Blueprint), jk(JokerId::Brainstorm)];
    assert_eq!(resolve_effective(&v, 0), None);
    assert_eq!(resolve_effective(&v, 1), None);
    // Debuffed links kill the chain (card.lua:2292).
    let mut v = vec![jk(JokerId::Blueprint), jk(JokerId::Joker)];
    v[1].debuffed = true;
    assert_eq!(resolve_effective(&v, 0), None);
    v[1].debuffed = false;
    v[0].debuffed = true;
    assert_eq!(resolve_effective(&v, 0), None);
}

#[test]
fn blueprint_copies_common_effects_with_gates() {
    // Blueprint left of Joker: both fire (+4 each).
    let mut w = World::with("SEED", &[JokerId::Blueprint, JokerId::Joker]);
    assert_eq!(w.play(&pair_5s(), &[]).mult, 10.0);

    // Brainstorm right of Joker copies the leftmost.
    let mut w = World::with("SEED", &[JokerId::Joker, JokerId::Brainstorm]);
    assert_eq!(w.play(&pair_5s(), &[]).mult, 10.0);

    // Blueprint + Green Joker: the copy reads the target's accumulated mult
    // (joker_main has no blueprint gate) but the scaling itself happens only
    // once (`not context.blueprint` on the before-hand branch).
    let mut w = World::with("SEED", &[JokerId::Blueprint, JokerId::GreenJoker]);
    let r = w.play(&pair_5s(), &[]);
    assert_eq!(w.jokers[1].state.mult, 1.0); // scaled once, not twice
    assert_eq!(r.mult, 2.0 + 1.0 + 1.0); // paid by copy AND original
}
