//! Unit tests for the evaluate_poker_hand port, exercising the quirks called
//! out in misc_functions.lua:376-621.

use balatro_core::cards::{Card, Enhancement, HandType, Rank, Suit};
use balatro_core::handeval::{evaluate_poker_hand, get_poker_hand_info, EvalMods};
use balatro_core::rng::RngState;
use balatro_core::scoring::{evaluate_play, HandsTable, NoJokers, PlayResult, ScoreContext};

/// P2's `score_play` shim: evaluate_play with no blind, no jokers, no held
/// cards (the extra machinery is exercised in tests/scoring_units.rs).
fn score_play(
    play: &[balatro_core::cards::Card],
    hands: &mut HandsTable,
    mods: &EvalMods,
) -> PlayResult {
    let mut rng = RngState::new("TESTSEED");
    let mut dollars = 0i64;
    let mut ctx = ScoreContext {
        rng: &mut rng,
        hands,
        dollars: &mut dollars,
        blind: None,
        most_played: HandType::HighCard,
        mods: *mods,
        prob_normal: 1.0,
        plasma_balance: false,
    };
    let mut play = play.to_vec();
    let mut held = Vec::new();
    evaluate_play(&mut ctx, &mut play, &mut held, &mut NoJokers)
}

fn c(rank: u8, suit: Suit) -> Card {
    // sort_id increments per call so ties resolve like creation order.
    use std::sync::atomic::{AtomicU32, Ordering};
    static NEXT: AtomicU32 = AtomicU32::new(1);
    Card::new(Rank(rank), suit, NEXT.fetch_add(1, Ordering::Relaxed))
}

fn stone(rank: u8, suit: Suit) -> Card {
    let mut card = c(rank, suit);
    card.enhancement = Enhancement::Stone;
    card
}

const M: EvalMods = EvalMods {
    four_fingers: false,
    shortcut: false,
    smeared: false,
};

#[test]
fn wheel_straight_a2345() {
    let hand = [
        c(14, Suit::Spades),
        c(2, Suit::Hearts),
        c(3, Suit::Clubs),
        c(4, Suit::Diamonds),
        c(5, Suit::Spades),
    ];
    let (ht, scoring, _) = get_poker_hand_info(&hand, &M);
    assert_eq!(ht, HandType::Straight);
    let mut s = scoring.clone();
    s.sort_unstable();
    assert_eq!(s, vec![0, 1, 2, 3, 4]);
    // The ace-low probe runs first: scoring order starts with the ace.
    assert_eq!(scoring[0], 0);
}

#[test]
fn broadway_straight_tjqka() {
    let hand = [
        c(10, Suit::Spades),
        c(11, Suit::Hearts),
        c(12, Suit::Clubs),
        c(13, Suit::Diamonds),
        c(14, Suit::Spades),
    ];
    let (ht, mut scoring, _) = get_poker_hand_info(&hand, &M);
    assert_eq!(ht, HandType::Straight);
    scoring.sort_unstable();
    assert_eq!(scoring, vec![0, 1, 2, 3, 4]);
}

#[test]
fn ace_does_not_wrap_ka234() {
    let hand = [
        c(13, Suit::Spades),
        c(14, Suit::Hearts),
        c(2, Suit::Clubs),
        c(3, Suit::Diamonds),
        c(4, Suit::Spades),
    ];
    let (ht, scoring, _) = get_poker_hand_info(&hand, &M);
    assert_eq!(ht, HandType::HighCard);
    // Highest by get_nominal: the ace.
    assert_eq!(scoring, vec![1]);
}

#[test]
fn straight_collects_duplicate_ranks_with_four_fingers() {
    // 9,9,T,J,Q: a 4-length straight window (Four Fingers) whose duplicate 9
    // is ALSO swept into the scoring set (get_straight pushes every card of
    // each rank in the window).
    let mods = EvalMods {
        four_fingers: true,
        shortcut: false,
        smeared: false,
    };
    let hand = [
        c(9, Suit::Spades),
        c(9, Suit::Hearts),
        c(10, Suit::Clubs),
        c(11, Suit::Diamonds),
        c(12, Suit::Spades),
    ];
    let (ht, mut scoring, _) = get_poker_hand_info(&hand, &mods);
    assert_eq!(ht, HandType::Straight);
    scoring.sort_unstable();
    assert_eq!(scoring, vec![0, 1, 2, 3, 4], "both 9s must score");
}

#[test]
fn flush_five_and_tail_overwrites() {
    let hand = [
        c(14, Suit::Spades),
        c(14, Suit::Spades),
        c(14, Suit::Spades),
        c(14, Suit::Spades),
        c(14, Suit::Spades),
    ];
    let (ht, scoring, ph) = get_poker_hand_info(&hand, &M);
    assert_eq!(ht, HandType::FlushFive);
    assert_eq!(scoring.len(), 5);
    // Tail-overwrite quirk (misc_functions.lua:507-517): Four of a Kind /
    // Three of a Kind / Pair get the *whole 5-card group* copied in.
    assert_eq!(ph.get(HandType::FourOfAKind)[0].len(), 5);
    assert_eq!(ph.get(HandType::ThreeOfAKind)[0].len(), 5);
    assert_eq!(ph.get(HandType::Pair)[0].len(), 5);
}

#[test]
fn five_of_a_kind_mixed_suits() {
    let hand = [
        c(14, Suit::Spades),
        c(14, Suit::Hearts),
        c(14, Suit::Hearts),
        c(14, Suit::Clubs),
        c(14, Suit::Diamonds),
    ];
    let (ht, scoring, _) = get_poker_hand_info(&hand, &M);
    assert_eq!(ht, HandType::FiveOfAKind);
    assert_eq!(scoring.len(), 5);
}

#[test]
fn flush_house() {
    let hand = [
        c(7, Suit::Diamonds),
        c(7, Suit::Diamonds),
        c(7, Suit::Diamonds),
        c(4, Suit::Diamonds),
        c(4, Suit::Diamonds),
    ];
    let (ht, mut scoring, ph) = get_poker_hand_info(&hand, &M);
    assert_eq!(ht, HandType::FlushHouse);
    scoring.sort_unstable();
    assert_eq!(scoring, vec![0, 1, 2, 3, 4]);
    // Scoring group order: trips first, then the pair (_3[1] .. _2[1]).
    assert!(!ph.get(HandType::FullHouse).is_empty());
    assert!(!ph.get(HandType::Flush).is_empty());
}

#[test]
fn full_house_and_two_pair_from_3_plus_2() {
    let hand = [
        c(13, Suit::Hearts),
        c(13, Suit::Clubs),
        c(13, Suit::Diamonds),
        c(2, Suit::Spades),
        c(2, Suit::Diamonds),
    ];
    let (ht, scoring, ph) = get_poker_hand_info(&hand, &M);
    assert_eq!(ht, HandType::FullHouse);
    assert_eq!(scoring.len(), 5);
    // The `_3 + _2` Two Pair branch fires too: 5 cards in the Two Pair group.
    assert_eq!(ph.get(HandType::TwoPair)[0].len(), 5);
}

#[test]
fn two_pair_higher_pair_first() {
    let hand = [
        c(4, Suit::Hearts),
        c(14, Suit::Hearts),
        c(14, Suit::Diamonds),
        c(12, Suit::Clubs),
        c(4, Suit::Clubs),
    ];
    let (ht, scoring, _) = get_poker_hand_info(&hand, &M);
    assert_eq!(ht, HandType::TwoPair);
    // get_X_same returns groups by descending rank: aces then fours.
    assert_eq!(scoring.len(), 4);
    assert_eq!(&scoring[..2], &[1, 2], "ace pair first");
    let mut rest = scoring[2..].to_vec();
    rest.sort_unstable();
    assert_eq!(rest, vec![0, 4]);
}

#[test]
fn straight_flush_merge() {
    let hand = [
        c(8, Suit::Spades),
        c(9, Suit::Spades),
        c(10, Suit::Spades),
        c(11, Suit::Spades),
        c(12, Suit::Spades),
    ];
    let (ht, mut scoring, ph) = get_poker_hand_info(&hand, &M);
    assert_eq!(ht, HandType::StraightFlush);
    scoring.sort_unstable();
    assert_eq!(scoring, vec![0, 1, 2, 3, 4]);
    assert!(!ph.get(HandType::Flush).is_empty());
    assert!(!ph.get(HandType::Straight).is_empty());
}

#[test]
fn four_card_hands_never_straight_or_flush() {
    // 4 hearts: no flush without Four Fingers.
    let hearts = [
        c(2, Suit::Hearts),
        c(5, Suit::Hearts),
        c(9, Suit::Hearts),
        c(13, Suit::Hearts),
    ];
    let ph = evaluate_poker_hand(&hearts, &M);
    assert!(ph.get(HandType::Flush).is_empty());
    assert_eq!(ph.top, Some(HandType::HighCard));

    // 2,3,4,5: no straight from 4 cards.
    let run4 = [
        c(2, Suit::Hearts),
        c(3, Suit::Spades),
        c(4, Suit::Clubs),
        c(5, Suit::Diamonds),
    ];
    let ph = evaluate_poker_hand(&run4, &M);
    assert!(ph.get(HandType::Straight).is_empty());
    assert_eq!(ph.top, Some(HandType::HighCard));
}

#[test]
fn six_card_hands_early_out() {
    let hand = [
        c(2, Suit::Hearts),
        c(3, Suit::Hearts),
        c(4, Suit::Hearts),
        c(5, Suit::Hearts),
        c(6, Suit::Hearts),
        c(7, Suit::Hearts),
    ];
    let ph = evaluate_poker_hand(&hand, &M);
    // `#hand > 5` early-outs in get_flush/get_straight.
    assert!(ph.get(HandType::Flush).is_empty());
    assert!(ph.get(HandType::Straight).is_empty());
    assert_eq!(ph.top, Some(HandType::HighCard));
}

#[test]
fn stone_cards_break_flush_and_rank_matching() {
    // 4 spades + a stone "spade": Stone fails is_suit, so no flush.
    let hand = [
        c(2, Suit::Spades),
        c(5, Suit::Spades),
        c(9, Suit::Spades),
        c(13, Suit::Spades),
        stone(7, Suit::Spades),
    ];
    let ph = evaluate_poker_hand(&hand, &M);
    assert!(ph.get(HandType::Flush).is_empty());
    assert_eq!(ph.top, Some(HandType::HighCard));

    // Two stones never pair (get_id returns a fresh negative random each call).
    let hand = [stone(7, Suit::Spades), stone(7, Suit::Spades)];
    let ph = evaluate_poker_hand(&hand, &M);
    assert!(ph.get(HandType::Pair).is_empty());
    assert_eq!(ph.top, Some(HandType::HighCard));
}

#[test]
fn stone_cards_join_the_scoring_set() {
    // Pair of kings + stone: hand is a Pair, but the stone card is appended
    // as a "pure bonus" card (evaluate_play, state_events.lua:580-599) and
    // contributes its 50 chips.
    let play = [
        c(13, Suit::Hearts),
        c(13, Suit::Clubs),
        stone(2, Suit::Spades),
    ];
    let mut hands = HandsTable::new();
    let result = score_play(&play, &mut hands, &M);
    assert_eq!(result.hand_type, HandType::Pair);
    assert_eq!(result.scoring, vec![0, 1, 2]);
    // Pair base 10 chips x2 mult; +10+10 kings, +50 stone.
    assert_eq!(result.chips, 10.0 + 10.0 + 10.0 + 50.0);
    assert_eq!(result.mult, 2.0);
    assert_eq!(result.score, 160.0);
}

#[test]
fn high_card_scoring_and_base_values() {
    let play = [c(9, Suit::Diamonds), c(14, Suit::Spades), c(3, Suit::Clubs)];
    let mut hands = HandsTable::new();
    let result = score_play(&play, &mut hands, &M);
    assert_eq!(result.hand_type, HandType::HighCard);
    assert_eq!(result.scoring, vec![1]); // only the ace scores
    assert_eq!(result.chips, 5.0 + 11.0);
    assert_eq!(result.mult, 1.0);
    assert_eq!(result.score, 16.0);
    assert_eq!(hands.get(HandType::HighCard).played, 1);
}
