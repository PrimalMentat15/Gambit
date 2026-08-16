//! Boss-blind effects observed through the Run API on fixed seeds.
//!
//! Harness: `force_boss` pins the desired boss for the ante-1 Boss round
//! (bypassing the min-ante eligibility the way a test should — no 'boss'
//! stream is consumed). Hand levels are raised via `level_up_hand` (the real
//! `level_up_hand` formula) so scripted single-card plays clear Small/Big
//! but do NOT one-shot the boss, letting us observe mid-round behavior:
//! High Card at level 4 scores (35 + rank) * 4 ≈ 150-190 per play vs the
//! 600-chip ante-1 boss.

use balatro_core::cards::HandType;
use balatro_core::rng::RngState;
use balatro_core::run::{BlindStage, Run, RunError, State};

/// Level every hand to level+3 (High Card: 35 chips x4 mult).
fn power_small(run: &mut Run) {
    for ht in HandType::ALL {
        run.level_up_hand(ht, 3);
    }
}

/// Level every hand sky-high (any play scores ~10^7).
fn power_big(run: &mut Run) {
    for ht in HandType::ALL {
        run.level_up_hand(ht, 2000);
    }
}

/// First `n` hand indices, always including Cerulean Bell's forced card.
fn selection(run: &Run, n: usize) -> Vec<usize> {
    let mut sel: Vec<usize> = Vec::new();
    if let Some(f) = run.forced_card_index() {
        sel.push(f);
    }
    let mut i = 0;
    while sel.len() < n && i < run.hand().len() {
        if !sel.contains(&i) {
            sel.push(i);
        }
        i += 1;
    }
    sel.sort_unstable();
    sel
}

/// Beat Small and Big with single-card plays (High Card), then force `boss`
/// and enter the ante-1 Boss round.
fn enter_ante1_boss(seed: &str, boss: &'static str) -> Run {
    let mut run = Run::new(seed);
    power_small(&mut run);
    loop {
        match run.state() {
            State::BlindSelect => {
                if run.blind_on_deck() == BlindStage::Boss {
                    run.force_boss(boss);
                    run.select_blind().unwrap();
                    return run;
                }
                run.select_blind().unwrap();
            }
            State::SelectingHand => {
                let sel = selection(&run, 1);
                run.play(&sel).unwrap();
            }
            State::RoundEval => run.cash_out().unwrap(),
            State::Shop => run.leave_shop().unwrap(),
            s => panic!("unexpected state {s:?} before boss"),
        }
    }
}

/// Play 5-card hands at full power up to the ante-8 Boss selection.
fn enter_ante8_boss(seed: &str, boss: &'static str) -> Run {
    let mut run = Run::new(seed);
    power_big(&mut run);
    loop {
        match run.state() {
            State::BlindSelect => {
                if run.blind_on_deck() == BlindStage::Boss && run.ante() == 8 {
                    run.force_boss(boss);
                    run.select_blind().unwrap();
                    return run;
                }
                run.select_blind().unwrap();
            }
            State::SelectingHand => {
                let sel = selection(&run, 5);
                run.play(&sel).unwrap();
            }
            State::RoundEval => run.cash_out().unwrap(),
            State::Shop => run.leave_shop().unwrap(),
            s => panic!("unexpected state {s:?} before ante 8"),
        }
    }
}

#[test]
fn hook_discards_two_random_cards() {
    let mut run = enter_ante1_boss("HOOKSEED", "bl_hook");
    let deck_before = run.deck_len();
    run.play(&[0]).unwrap();
    assert_eq!(run.state(), State::SelectingHand, "180 chips must not win");
    // 1 played + 2 hooked leave the hand; redraw refills 3 from the deck.
    assert_eq!(run.hand().len(), 8);
    assert_eq!(run.deck_len(), deck_before - 3);
    assert_eq!(run.discard_pile_len(), 3);
}

#[test]
fn tooth_takes_a_dollar_per_played_card() {
    let mut run = enter_ante1_boss("TOOTHSEED", "bl_tooth");
    let before = run.dollars();
    run.play(&[0]).unwrap();
    assert_eq!(run.dollars(), before - 1);
    let before = run.dollars();
    run.play(&[0, 1, 2]).unwrap();
    assert_eq!(run.dollars(), before - 3);
}

#[test]
fn ox_wipes_money_on_most_played_hand() {
    // most_played_poker_hand starts as 'High Card' (game.lua:1964) and is
    // only recomputed at Boss round ends, so the forced ante-1 Ox targets
    // single-card plays.
    let mut run = enter_ante1_boss("OXSEED12", "bl_ox");
    assert_eq!(run.most_played_hand(), HandType::HighCard);
    assert!(run.dollars() > 0);
    let r = run.play(&[0]).unwrap().clone();
    assert!(!r.debuffed_hand, "the Ox does not debuff");
    assert!(r.score > 0.0);
    assert_eq!(run.dollars(), 0);
}

#[test]
fn arm_levels_down_the_played_hand() {
    let mut run = enter_ante1_boss("ARMSEED1", "bl_arm");
    assert_eq!(run.hands_table().get(HandType::HighCard).level, 4);
    run.play(&[0]).unwrap();
    assert_eq!(run.hands_table().get(HandType::HighCard).level, 3);
}

#[test]
fn wall_and_flint_and_vessel_chip_requirements() {
    // The Wall: mult 4 => 300 * 4 = 1200 at ante 1 (P_BLINDS bl_wall).
    let run = enter_ante1_boss("WALLSEED", "bl_wall");
    assert_eq!(run.blind_chips(), 1200.0);

    // The Flint: normal 2x requirement (600), but base chips AND mult are
    // halved on every play (blind.lua:510-517).
    let mut run = enter_ante1_boss("FLINTSED", "bl_flint");
    assert_eq!(run.blind_chips(), 600.0);
    let r = run.play(&[0]).unwrap();
    // High Card lvl 4: 35 chips x4 mult -> floor(17.5+0.5)=18 chips,
    // floor(2+0.5)=2 mult; chips = 18 + rank of the played card.
    assert_eq!(r.mult, 2.0);
    assert!(r.chips >= 18.0 + 2.0 && r.chips <= 18.0 + 11.0);
}

#[test]
fn water_needle_manacle_round_modifiers() {
    let run = enter_ante1_boss("WATERSED", "bl_water");
    assert_eq!(run.discards_left(), 0, "The Water: 0 discards");
    assert_eq!(run.hands_left(), 4);

    let run = enter_ante1_boss("NEEDLSED", "bl_needle");
    assert_eq!(run.hands_left(), 1, "The Needle: 1 hand");
    assert_eq!(run.discards_left(), 4);
    // The Needle's chip requirement uses mult 1 => 300.
    assert_eq!(run.blind_chips(), 300.0);

    let run = enter_ante1_boss("MANACSED", "bl_manacle");
    assert_eq!(run.hand_size(), 7, "The Manacle: -1 hand size");
    assert_eq!(run.hand().len(), 7);
}

#[test]
fn manacle_restores_hand_size_after_win() {
    let mut run = enter_ante1_boss("MANACWIN", "bl_manacle");
    power_big(&mut run); // finish the boss quickly
    run.play(&selection(&run, 5)).unwrap();
    assert_eq!(run.state(), State::RoundEval);
    assert_eq!(run.hand_size(), 8, "Blind:defeat gives the size back");
}

#[test]
fn psychic_requires_five_cards() {
    let mut run = enter_ante1_boss("PSYCHSED", "bl_psychic");
    let r = run.play(&[0, 1, 2, 3]).unwrap().clone();
    assert!(r.debuffed_hand);
    assert_eq!(r.score, 0.0);
    let r = run.play(&[0, 1, 2, 3, 4]).unwrap().clone();
    assert!(!r.debuffed_hand);
    assert!(r.score > 0.0);
}

#[test]
fn eye_debuffs_repeat_hand_types() {
    let mut run = enter_ante1_boss("EYESEED1", "bl_eye");
    let r = run.play(&[0]).unwrap().clone();
    assert!(!r.debuffed_hand);
    assert_eq!(run.state(), State::SelectingHand);
    let r = run.play(&[0]).unwrap().clone();
    assert!(r.debuffed_hand, "second High Card is blocked by The Eye");
    assert_eq!(r.score, 0.0);
}

#[test]
fn mouth_locks_the_first_hand_type() {
    let mut run = enter_ante1_boss("MOUTHSED", "bl_mouth");
    let r = run.play(&[0]).unwrap().clone();
    assert!(!r.debuffed_hand);
    // Find a pair in hand (discarding junk until one shows up).
    loop {
        let hand = run.hand();
        let mut pair: Option<(usize, usize)> = None;
        'outer: for i in 0..hand.len() {
            for j in i + 1..hand.len() {
                if hand[i].rank == hand[j].rank {
                    pair = Some((i, j));
                    break 'outer;
                }
            }
        }
        if let Some((i, j)) = pair {
            let r = run.play(&[i, j]).unwrap().clone();
            assert_eq!(r.hand_type, HandType::Pair);
            assert!(r.debuffed_hand, "The Mouth blocks non-High-Card hands");
            assert_eq!(r.score, 0.0);
            return;
        }
        assert!(run.discards_left() > 0, "seed must offer a pair in time");
        run.discard(&[7]).unwrap();
    }
}

#[test]
fn wheel_flips_cards_by_oracle_pattern() {
    // The Wheel consumes one 'wheel' draw per card drawn to hand
    // (blind.lua:608). Nothing touched the stream before the boss deal, so
    // the 8 dealt cards follow the first 8 draws of a fresh stream.
    let seed = "WHEELSED";
    let mut probe = RngState::new(seed);
    let expected: usize = (0..8).filter(|_| probe.random("wheel") < 1.0 / 7.0).count();
    let run = enter_ante1_boss(seed, "bl_wheel");
    let flipped = run.hand().iter().filter(|c| c.face_down).count();
    assert_eq!(flipped, expected);
}

#[test]
fn house_first_deal_face_down() {
    let mut run = enter_ante1_boss("HOUSESED", "bl_house");
    assert!(run.hand().iter().all(|c| c.face_down), "whole deal flipped");
    // After a discard the House condition (hands_played == 0 AND
    // discards_used == 0) no longer holds: the replacement is face up.
    run.discard(&[0]).unwrap();
    assert_eq!(run.hand().iter().filter(|c| !c.face_down).count(), 1);
}

#[test]
fn mark_flips_face_cards() {
    let run = enter_ante1_boss("MARKSEED", "bl_mark");
    for c in run.hand() {
        assert_eq!(
            c.face_down,
            c.rank.is_face(),
            "The Mark flips exactly the face cards: {c:?}"
        );
    }
}

#[test]
fn fish_deals_face_down_after_a_play() {
    let mut run = enter_ante1_boss("FISHSEED", "bl_fish");
    assert!(
        run.hand().iter().all(|c| !c.face_down),
        "first deal face up"
    );
    run.play(&[0]).unwrap();
    assert_eq!(run.state(), State::SelectingHand);
    assert_eq!(
        run.hand().iter().filter(|c| c.face_down).count(),
        1,
        "the single replacement card is face down"
    );
    // A discard's replacement is also face down while prepped... prepped was
    // cleared by drawn_to_hand (blind.lua:602); only post-play draws flip.
    run.discard(&[7]).unwrap();
    assert_eq!(run.hand().iter().filter(|c| c.face_down).count(), 1);
}

#[test]
fn serpent_draws_exactly_three() {
    let mut run = enter_ante1_boss("SERPSEED", "bl_serpent");
    assert_eq!(run.hand().len(), 8);
    // After a discard (discards_used > 0) the Serpent always deals 3
    // (state_events.lua:363-368) — the hand overfills.
    run.discard(&[0]).unwrap();
    assert_eq!(run.hand().len(), 10);
    // And after a play of 5 it deals 3 again: 10 - 5 + 3 = 8.
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    if run.state() == State::SelectingHand {
        assert_eq!(run.hand().len(), 8);
    }
}

#[test]
fn suit_and_face_debuff_bosses() {
    use balatro_core::cards::Suit;
    for (boss, suit) in [
        ("bl_club", Suit::Clubs),
        ("bl_goad", Suit::Spades),
        ("bl_head", Suit::Hearts),
        ("bl_window", Suit::Diamonds),
    ] {
        let run = enter_ante1_boss("SUITSEED", boss);
        for c in run.hand() {
            assert_eq!(c.debuff, c.suit == suit, "{boss}: {c:?}");
        }
    }
    let run = enter_ante1_boss("PLANTSED", "bl_plant");
    for c in run.hand() {
        assert_eq!(c.debuff, c.rank.is_face(), "The Plant debuffs faces");
    }
}

#[test]
fn pillar_debuffs_cards_played_this_ante() {
    let mut run = enter_ante1_boss("PILLASED", "bl_pillar");
    assert!(
        run.hand().iter().any(|c| c.played_this_ante),
        "seed must redeal some already-played cards"
    );
    for c in run.hand() {
        assert_eq!(c.debuff, c.played_this_ante, "Pillar: {c:?}");
    }
    // Win the boss; the flags are wiped on Boss defeat
    // (state_events.lua:265-267).
    power_big(&mut run);
    run.play(&selection(&run, 5)).unwrap();
    assert_eq!(run.state(), State::RoundEval);
    run.cash_out().unwrap();
    run.leave_shop().unwrap();
    run.select_blind().unwrap();
    assert!(
        run.hand().iter().all(|c| !c.played_this_ante),
        "played_this_ante cleared after the boss"
    );
}

#[test]
fn disable_boss_reverts_effects() {
    // The Water refunds the stashed discards on disable (blind.lua:361-363).
    let mut run = enter_ante1_boss("WATERDIS", "bl_water");
    assert_eq!(run.discards_left(), 0);
    run.disable_boss();
    assert_eq!(run.discards_left(), 4);

    // The Wall halves its requirement on disable (blind.lua:377-380).
    let mut run = enter_ante1_boss("WALLDIS1", "bl_wall");
    assert_eq!(run.blind_chips(), 1200.0);
    run.disable_boss();
    assert_eq!(run.blind_chips(), 600.0);

    // The Manacle: hand size restored and one card drawn (blind.lua:386-390).
    let mut run = enter_ante1_boss("MANADIS1", "bl_manacle");
    assert_eq!(run.hand().len(), 7);
    run.disable_boss();
    assert_eq!(run.hand_size(), 8);
    assert_eq!(run.hand().len(), 8);
}

// ---------------------------------------------------------------------------
// Ante-8 showdown finishers
// ---------------------------------------------------------------------------

#[test]
fn cerulean_bell_forces_a_card() {
    let mut run = enter_ante8_boss("BELLSEED", "bl_final_bell");
    let forced = run.forced_card_index();
    assert!(forced.is_some(), "a card must be forced on deal");
    // Playing without the forced card is rejected.
    let other = (0..run.hand().len()).find(|&i| Some(i) != forced).unwrap();
    assert!(matches!(
        run.play(&[other]),
        Err(RunError::BadCardSelection(_))
    ));
    // Including it works, and the win at ante 8 ends the run.
    let sel = selection(&run, 5);
    run.play(&sel).unwrap();
    assert_eq!(run.state(), State::RoundEval);
    assert!(run.won());
    run.cash_out().unwrap();
    assert_eq!(run.state(), State::Won);
}

#[test]
fn verdant_leaf_debuffs_every_card() {
    let mut run = enter_ante8_boss("LEAFSEED", "bl_final_leaf");
    assert!(run.hand().iter().all(|c| c.debuff));
    // Cards score nothing; only the (huge, leveled) base counts.
    let r = run.play(&[0]).unwrap().clone();
    let base = run.hands_table().get(HandType::HighCard).chips;
    assert_eq!(r.chips, base, "no card chips under Verdant Leaf");
}

#[test]
fn violet_vessel_requirement_and_showdown_rewards() {
    let run = enter_ante8_boss("VESSSEED", "bl_final_vessel");
    // mult 6 => 50000 * 6 (get_blind_amount(8) = 50000).
    assert_eq!(run.blind_chips(), 300_000.0);
    assert_eq!(run.current_blind().unwrap().dollars, 8);
}

#[test]
fn acorn_and_heart_are_joker_only_no_ops_in_p3a() {
    // Amber Acorn (flip+shuffle jokers) and Crimson Heart (debuff a joker
    // per hand) have no observable effect without jokers; the round must
    // still play out and win the run.
    for (seed, boss) in [
        ("ACRNSEED", "bl_final_acorn"),
        ("HEARTSED", "bl_final_heart"),
    ] {
        let mut run = enter_ante8_boss(seed, boss);
        assert_eq!(run.blind_chips(), 100_000.0); // 50000 * 2
        run.play(&selection(&run, 5)).unwrap();
        assert_eq!(run.state(), State::RoundEval);
        assert!(run.won());
    }
}
