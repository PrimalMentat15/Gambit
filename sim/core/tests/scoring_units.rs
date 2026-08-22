//! Unit tests for the full scoring pipeline: known hands with
//! enhancements/editions/seals computing exact scores. Every expected value
//! is derived by hand from the ported formulas (comments show the
//! arithmetic; state_events.lua/card.lua refs in scoring.rs).

use balatro_core::blinds::{boss_by_key, ActiveBlind};
use balatro_core::cards::{Card, Edition, Enhancement, HandType, Rank, Seal, Suit};
use balatro_core::handeval::EvalMods;
use balatro_core::rng::RngState;
use balatro_core::scoring::{evaluate_play, HandsTable, NoJokers, PlayResult, ScoreContext};

fn c(rank: u8, suit: Suit) -> Card {
    Card::new(Rank(rank), suit, rank as u32 + 100)
}

struct World {
    rng: RngState,
    hands: HandsTable,
    dollars: i64,
    blind: Option<ActiveBlind>,
    most_played: HandType,
}

impl World {
    fn new(seed: &str) -> Self {
        World {
            rng: RngState::new(seed),
            hands: HandsTable::new(),
            dollars: 0,
            blind: None,
            most_played: HandType::HighCard,
        }
    }

    fn play(&mut self, play: &[Card], held: &[Card]) -> PlayResult {
        let mut ctx = ScoreContext {
            rng: &mut self.rng,
            hands: &mut self.hands,
            dollars: &mut self.dollars,
            blind: self.blind.as_mut(),
            most_played: self.most_played,
            mods: EvalMods::default(),
            prob_normal: 1.0,
            plasma_balance: false,
        };
        let mut play = play.to_vec();
        let mut held = held.to_vec();
        evaluate_play(&mut ctx, &mut play, &mut held, &mut NoJokers)
    }
}

#[test]
fn mult_card_and_foil_edition() {
    // Pair of 5s: base 10 chips x2 mult. Card 1 has the Mult enhancement
    // (+4 mult, m_mult game.lua:649), card 2 is Foil (+50 chips,
    // e_foil game.lua:659).
    let mut w = World::new("TESTSEED");
    let mut c1 = c(5, Suit::Hearts);
    c1.enhancement = Enhancement::Mult;
    let mut c2 = c(5, Suit::Spades);
    c2.edition = Edition::Foil;
    let r = w.play(&[c1, c2], &[]);
    assert_eq!(r.hand_type, HandType::Pair);
    // chips = 10 + 5 (c1) + 5 (c2) + 50 (foil) = 70
    // mult  = 2 + 4 (mult card) = 6
    assert_eq!(r.chips, 70.0);
    assert_eq!(r.mult, 6.0);
    assert_eq!(r.score, 420.0); // floor(70 * 6)
}

#[test]
fn holo_and_polychrome_editions() {
    // High Card ace (Holo: +10 mult): chips = 5 + 11 = 16, mult = 1 + 10 = 11.
    let mut w = World::new("TESTSEED");
    let mut a = c(14, Suit::Spades);
    a.edition = Edition::Holo;
    let r = w.play(&[a], &[]);
    assert_eq!((r.chips, r.mult, r.score), (16.0, 11.0, 176.0));

    // Polychrome (x1.5 mult): mult = 1 * 1.5 = 1.5; floor(16 * 1.5) = 24.
    let mut w = World::new("TESTSEED");
    let mut a = c(14, Suit::Spades);
    a.edition = Edition::Polychrome;
    let r = w.play(&[a], &[]);
    assert_eq!((r.chips, r.mult, r.score), (16.0, 1.5, 24.0));
}

#[test]
fn glass_times_two_before_polychrome() {
    // Glass (x2, applied at the x_mult slot) comes BEFORE the same card's
    // edition x1.5 (state_events.lua:751-776): mult = ((1)*2)*1.5 = 3.
    let mut w = World::new("TESTSEED");
    let mut a = c(14, Suit::Spades);
    a.enhancement = Enhancement::Glass;
    a.edition = Edition::Polychrome;

    // Predict the glass-break roll from a clone of the stream.
    let mut probe = RngState::new("TESTSEED");
    let breaks = probe.random("glass") < 1.0 / 4.0;

    let r = w.play(&[a], &[]);
    assert_eq!((r.chips, r.mult), (16.0, 3.0));
    assert_eq!(r.score, 48.0);
    assert_eq!(r.destroyed.is_empty(), !breaks, "break roll must decide");
    // Exactly one 'glass' draw was consumed.
    assert_eq!(w.rng.random("glass"), probe.random("glass"));
}

#[test]
fn debuffed_glass_neither_scores_nor_rolls() {
    let mut w = World::new("TESTSEED");
    let mut a = c(14, Suit::Spades);
    a.enhancement = Enhancement::Glass;
    a.debuff = true;
    let mut k = c(13, Suit::Spades);
    k.rank = Rank(14); // second ace -> Pair, both score
    let r = w.play(&[a, k], &[]);
    assert_eq!(r.hand_type, HandType::Pair);
    // Debuffed ace contributes nothing: chips = 10 (base) + 11 (other ace).
    assert_eq!(r.chips, 21.0);
    assert_eq!(r.mult, 2.0);
    assert!(r.destroyed.is_empty());
    // The 'glass' stream was never touched: first draw equals a fresh one.
    let mut probe = RngState::new("TESTSEED");
    assert_eq!(w.rng.random("glass"), probe.random("glass"));
}

#[test]
fn gold_seal_and_red_seal_retrigger() {
    // High Card ace with Gold seal (+$3 on scoring, card.lua:1071-1073) and
    // Red seal (1 retrigger, card.lua:2244-2252): two full evaluations.
    let mut w = World::new("TESTSEED");
    let mut a = c(14, Suit::Spades);
    a.seal = Seal::Red;
    let mut g = c(14, Suit::Hearts);
    g.seal = Seal::Gold;
    let r = w.play(&[a, g], &[]);
    assert_eq!(r.hand_type, HandType::Pair);
    // Red-sealed ace scores twice, gold ace once:
    // chips = 10 + 11 + 11 (red) + 11 (gold) = 43
    assert_eq!(r.chips, 43.0);
    assert_eq!(w.dollars, 3);
    assert_eq!(r.dollars_delta, 3);

    // A Red seal on a Gold-seal card is impossible (one seal slot), but a
    // Red seal retriggers its OWN card's gold-seal-free evaluation; verify a
    // red-sealed Lucky card rolls both lucky streams twice (fresh rolls per
    // repetition, state_events.lua:692).
    let mut probe = RngState::new("TESTSEED");
    let m1 = probe.random("lucky_mult") < 1.0 / 5.0;
    let _ = probe.random("lucky_money");
    let m2 = probe.random("lucky_mult") < 1.0 / 5.0;
    let _ = probe.random("lucky_money");

    let mut w = World::new("TESTSEED");
    let mut lucky_red = c(7, Suit::Clubs);
    lucky_red.enhancement = Enhancement::Lucky;
    lucky_red.seal = Seal::Red;
    let r = w.play(&[lucky_red], &[]);
    let expected_mult = 1.0 + if m1 { 20.0 } else { 0.0 } + if m2 { 20.0 } else { 0.0 };
    assert_eq!(r.mult, expected_mult);
    // Both streams sit exactly two draws in.
    assert_eq!(w.rng.random("lucky_mult"), probe.random("lucky_mult"));
    assert_eq!(w.rng.random("lucky_money"), probe.random("lucky_money"));
}

#[test]
fn steel_held_and_red_seal_held_retrigger() {
    // Play a lone ace; hold Steel K with Red seal + plain Steel Q.
    // Steel: x1.5 mult held (m_steel game.lua:652). Red seal retriggers held
    // effects (state_events.lua:809-830): K applies twice, Q once.
    // mult = 1 * 1.5 * 1.5 * 1.5 = 3.375; chips = 16.
    let mut w = World::new("TESTSEED");
    let a = c(14, Suit::Spades);
    let mut k = c(13, Suit::Hearts);
    k.enhancement = Enhancement::Steel;
    k.seal = Seal::Red;
    let mut q = c(12, Suit::Hearts);
    q.enhancement = Enhancement::Steel;
    let r = w.play(&[a], &[k, q]);
    assert_eq!(r.chips, 16.0);
    assert_eq!(r.mult, 3.375);
    assert_eq!(r.score, (16.0f64 * 3.375).floor());

    // Red seal on a plain held card retriggers nothing (no held effect).
    let mut w = World::new("TESTSEED");
    let mut plain = c(13, Suit::Hearts);
    plain.seal = Seal::Red;
    let r = w.play(&[a], &[plain]);
    assert_eq!(r.mult, 1.0);
}

#[test]
fn lucky_card_consumes_both_streams() {
    // A Lucky card rolls 'lucky_mult' (< 1/5 => +20 mult) then 'lucky_money'
    // (< 1/15 => +$20) once per evaluation (card.lua:987-993, 1074-1083).
    let mut probe = RngState::new("TESTSEED");
    let mult_hit = probe.random("lucky_mult") < 1.0 / 5.0;
    let money_hit = probe.random("lucky_money") < 1.0 / 15.0;

    let mut w = World::new("TESTSEED");
    let mut l = c(7, Suit::Clubs);
    l.enhancement = Enhancement::Lucky;
    let r = w.play(&[l], &[]);
    // High Card base 5 chips 1 mult; +7 rank chips.
    assert_eq!(r.chips, 12.0);
    assert_eq!(r.mult, if mult_hit { 21.0 } else { 1.0 });
    assert_eq!(w.dollars, if money_hit { 20 } else { 0 });
    // Both streams advanced exactly once.
    assert_eq!(w.rng.random("lucky_mult"), probe.random("lucky_mult"));
    assert_eq!(w.rng.random("lucky_money"), probe.random("lucky_money"));
}

#[test]
fn bonus_and_stone_cards() {
    // Bonus card: rank nominal + 30 (m_bonus game.lua:648).
    let mut w = World::new("TESTSEED");
    let mut b = c(9, Suit::Diamonds);
    b.enhancement = Enhancement::Bonus;
    let r = w.play(&[b], &[]);
    assert_eq!(r.chips, 5.0 + 9.0 + 30.0);

    // Stone card: always scores 50, never identifies a hand
    // (card.lua:957-961, 976-980).
    let mut w = World::new("TESTSEED");
    let mut s = c(2, Suit::Clubs);
    s.enhancement = Enhancement::Stone;
    let ace = c(14, Suit::Spades);
    let r = w.play(&[ace, s], &[]);
    assert_eq!(r.hand_type, HandType::HighCard);
    assert_eq!(r.scoring, vec![0, 1]); // ace + appended stone
    assert_eq!(r.chips, 5.0 + 11.0 + 50.0);
}

#[test]
fn arm_levels_down_before_scoring() {
    // The Arm: level 2 Pair -> level 1 BEFORE the base is read
    // (blind.lua:550-559, state_events.lua:640-641).
    let mut w = World::new("TESTSEED");
    w.hands.level_up(HandType::Pair, 1); // level 2: 25 chips x3 mult
    w.blind = Some(ActiveBlind::new(boss_by_key("bl_arm"), 2));
    let r = w.play(&[c(5, Suit::Hearts), c(5, Suit::Spades)], &[]);
    assert_eq!(w.hands.get(HandType::Pair).level, 1);
    // Level 1 base: 10 chips x2 mult; +5+5 rank chips.
    assert_eq!((r.chips, r.mult), (20.0, 2.0));

    // At level 1 The Arm does nothing (level > 1 guard).
    let r = w.play(&[c(5, Suit::Hearts), c(5, Suit::Spades)], &[]);
    assert_eq!(w.hands.get(HandType::Pair).level, 1);
    assert_eq!((r.chips, r.mult), (20.0, 2.0));
}

#[test]
fn flint_halves_base_chips_and_mult() {
    // The Flint (blind.lua:510-517): mult = max(floor(m*0.5+0.5), 1),
    // chips = max(floor(c*0.5+0.5), 0), applied to the base BEFORE card
    // chips are added.
    let mut w = World::new("TESTSEED");
    w.blind = Some(ActiveBlind::new(boss_by_key("bl_flint"), 2));
    let r = w.play(&[c(14, Suit::Spades)], &[]);
    // High Card base 5 chips 1 mult -> floor(2.5+0.5)=3 chips, max(1,1)=1.
    // chips = 3 + 11 = 14.
    assert_eq!((r.chips, r.mult, r.score), (14.0, 1.0, 14.0));

    // Pair (10c x2m) -> floor(5.5)=5 chips, floor(1.5)=1 mult.
    let r = w.play(&[c(5, Suit::Hearts), c(5, Suit::Spades)], &[]);
    assert_eq!((r.chips, r.mult), (5.0 + 5.0 + 5.0, 1.0));
}

#[test]
fn psychic_debuffs_hands_under_five_cards() {
    let mut w = World::new("TESTSEED");
    w.blind = Some(ActiveBlind::new(boss_by_key("bl_psychic"), 1));
    let r = w.play(&[c(14, Suit::Spades), c(5, Suit::Hearts)], &[]);
    assert!(r.debuffed_hand);
    assert_eq!((r.chips, r.mult, r.score), (0.0, 0.0, 0.0));
    // The hand still counts as played (state_events.lua:574, before the
    // debuff check).
    assert_eq!(w.hands.get(HandType::HighCard).played, 1);

    // Five cards pass.
    let five = [
        c(2, Suit::Hearts),
        c(5, Suit::Spades),
        c(7, Suit::Clubs),
        c(9, Suit::Diamonds),
        c(11, Suit::Hearts),
    ];
    let r = w.play(&five, &[]);
    assert!(!r.debuffed_hand);
    assert!(r.score > 0.0);
}

#[test]
fn eye_blocks_repeat_hand_types() {
    let mut w = World::new("TESTSEED");
    w.blind = Some(ActiveBlind::new(boss_by_key("bl_eye"), 3));
    let pair = [c(5, Suit::Hearts), c(5, Suit::Spades)];
    let r = w.play(&pair, &[]);
    assert!(!r.debuffed_hand);
    let r = w.play(&pair, &[]);
    assert!(r.debuffed_hand, "second Pair must be debuffed");
    let r = w.play(&[c(14, Suit::Spades)], &[]);
    assert!(!r.debuffed_hand, "a fresh hand type is fine");
}

#[test]
fn mouth_allows_only_first_hand_type() {
    let mut w = World::new("TESTSEED");
    w.blind = Some(ActiveBlind::new(boss_by_key("bl_mouth"), 2));
    let pair = [c(5, Suit::Hearts), c(5, Suit::Spades)];
    let r = w.play(&pair, &[]);
    assert!(!r.debuffed_hand);
    let r = w.play(&pair, &[]);
    assert!(!r.debuffed_hand, "same hand type stays allowed");
    let r = w.play(&[c(14, Suit::Spades)], &[]);
    assert!(r.debuffed_hand, "other hand types are debuffed");
}

#[test]
fn ox_takes_all_money_without_debuffing() {
    let mut w = World::new("TESTSEED");
    w.dollars = 23;
    w.most_played = HandType::HighCard;
    w.blind = Some(ActiveBlind::new(boss_by_key("bl_ox"), 6));
    let r = w.play(&[c(14, Suit::Spades)], &[]);
    assert!(!r.debuffed_hand, "the Ox never debuffs the hand");
    assert!(r.score > 0.0);
    assert_eq!(w.dollars, 0, "ease_dollars(-G.GAME.dollars, true)");
    assert_eq!(r.dollars_delta, -23);
}

#[test]
fn debuffed_cards_still_identify_hands() {
    // A debuffed card contributes rank AND natural suit to hand identity
    // (get_id/get_flush bypass debuff), it just scores nothing.
    let mut w = World::new("TESTSEED");
    let mut cards = [
        c(2, Suit::Hearts),
        c(4, Suit::Hearts),
        c(6, Suit::Hearts),
        c(8, Suit::Hearts),
        c(10, Suit::Hearts),
    ];
    for card in &mut cards {
        card.debuff = true;
    }
    let r = w.play(&cards, &[]);
    assert_eq!(r.hand_type, HandType::Flush);
    // Base flush values only: 35 chips x4 mult, no card contributes.
    assert_eq!((r.chips, r.mult, r.score), (35.0, 4.0, 140.0));

    // ... but a debuffed WILD card loses its any-suit power in flush calc
    // (card.lua:4069 requires `not self.debuff`).
    let mut w = World::new("TESTSEED");
    let mut wild = c(3, Suit::Spades);
    wild.enhancement = Enhancement::Wild;
    wild.debuff = true;
    let mut cards2 = cards;
    cards2[0] = wild; // spade wild, debuffed -> no heart flush
    let r = w.play(&cards2, &[]);
    assert_ne!(r.hand_type, HandType::Flush);
}

#[test]
fn level_up_formula_matches_level_up_hand() {
    // common_events.lua:464-469 recompute with clamps.
    let mut hands = HandsTable::new();
    hands.level_up(HandType::Pair, 1);
    assert_eq!(hands.get(HandType::Pair).level, 2);
    assert_eq!(hands.get(HandType::Pair).chips, 25.0); // 10 + 15
    assert_eq!(hands.get(HandType::Pair).mult, 3.0); // 2 + 1
    hands.level_up(HandType::Pair, -1);
    assert_eq!(hands.get(HandType::Pair).chips, 10.0);
    // Clamp: level 0 => mult = max(2 + 1*(0-1), 1) = 1,
    //        chips = max(10 + 15*(0-1), 0) = 0.
    hands.level_up(HandType::Pair, -1);
    assert_eq!(hands.get(HandType::Pair).level, 0);
    assert_eq!(hands.get(HandType::Pair).mult, 1.0);
    assert_eq!(hands.get(HandType::Pair).chips, 0.0);
}
