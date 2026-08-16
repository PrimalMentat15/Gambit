//! Run-level tests for the round-lifecycle card effects: Gold cards paying
//! at end of round, blue seals generating Planets, purple seals generating
//! Tarots on discard (including The Hook's forced discards), all against
//! the consumable-slot limit and the exact RNG streams.

use balatro_core::cards::{Enhancement, HandType, Seal};
use balatro_core::consumables::{planet_for_hand, purple_seal_tarot};
use balatro_core::rng::RngState;
use balatro_core::run::{BlindStage, Run, State};
use std::collections::HashSet;

fn power_big(run: &mut Run) {
    for ht in HandType::ALL {
        run.level_up_hand(ht, 2000);
    }
}

fn all_cards(run: &mut Run, f: impl Fn(&mut balatro_core::cards::Card) + Copy) {
    for sid in 1..=52 {
        run.modify_card(sid, f);
    }
}

#[test]
fn gold_cards_pay_three_at_end_of_round() {
    let mut run = Run::new("GOLDSEED");
    power_big(&mut run);
    all_cards(&mut run, |c| c.enhancement = Enhancement::Gold);
    run.select_blind().unwrap();
    let before = run.dollars();
    // One winning play; 3 Gold cards remain held at end of round
    // (card.lua:1036-1039: h_dollars = 3 each).
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    assert_eq!(run.state(), State::RoundEval);
    assert_eq!(run.dollars(), before + 3 * 3);
    // The payout (pending_cashout) is separate and unclaimed yet.
    assert!(run.pending_cashout() > 0);
}

#[test]
fn red_seal_doubles_gold_card_payout() {
    let mut run = Run::new("GOLDSEED");
    power_big(&mut run);
    all_cards(&mut run, |c| {
        c.enhancement = Enhancement::Gold;
        c.seal = Seal::Red;
    });
    run.select_blind().unwrap();
    let before = run.dollars();
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    // Red seal retriggers the end-of-round effect (state_events.lua:189-197):
    // $6 per held Gold card.
    assert_eq!(run.dollars(), before + 3 * 6);
}

#[test]
fn blue_seals_generate_planets_up_to_the_slot_limit() {
    let mut run = Run::new("BLUESEED");
    power_big(&mut run);
    all_cards(&mut run, |c| c.seal = Seal::Blue);
    run.select_blind().unwrap();
    let r = run.play(&[0, 1, 2, 3, 4]).unwrap().clone();
    assert_eq!(run.state(), State::RoundEval);
    // 3 blue seals held, but only 2 consumable slots
    // (card.lua:1040: #cards + buffer < card_limit).
    let planet = planet_for_hand(r.hand_type);
    assert_eq!(
        run.pending_consumables(),
        &[planet.to_string(), planet.to_string()]
    );
}

#[test]
fn purple_seals_generate_tarots_on_discard() {
    let seed = "PURPSEED";
    // Oracle expectation: two draws from the untouched 'Tarot8ba1' stream,
    // the second resampling against the first key.
    let mut probe = RngState::new(seed);
    let mut used = HashSet::new();
    let t1 = purple_seal_tarot(&mut probe, 1, &used);
    used.insert(t1.clone());
    let t2 = purple_seal_tarot(&mut probe, 1, &used);

    let mut run = Run::new(seed);
    all_cards(&mut run, |c| c.seal = Seal::Purple);
    run.select_blind().unwrap();
    // Discard 3 purple seals: the third is over the 2-slot limit
    // (card.lua:2254).
    run.discard(&[0, 1, 2]).unwrap();
    assert_eq!(run.pending_consumables(), &[t1, t2]);
    // Further discards generate nothing while the slots stay full.
    run.discard(&[0]).unwrap();
    assert_eq!(run.pending_consumables().len(), 2);
}

#[test]
fn hook_discards_trigger_purple_seals() {
    let seed = "HOOKPURP";
    let mut probe = RngState::new(seed);
    let mut used = HashSet::new();
    let t1 = purple_seal_tarot(&mut probe, 1, &used);
    used.insert(t1.clone());
    let t2 = purple_seal_tarot(&mut probe, 1, &used);

    let mut run = Run::new(seed);
    all_cards(&mut run, |c| c.seal = Seal::Purple);
    // Beat Small and Big fast, then face The Hook.
    power_big(&mut run);
    loop {
        match run.state() {
            State::BlindSelect => {
                if run.blind_on_deck() == BlindStage::Boss {
                    run.force_boss("bl_hook");
                    run.select_blind().unwrap();
                    break;
                }
                run.select_blind().unwrap();
            }
            State::SelectingHand => {
                run.play(&[0, 1, 2, 3, 4]).unwrap();
            }
            State::RoundEval => run.cash_out().unwrap(),
            State::Shop => run.leave_shop().unwrap(),
            s => panic!("unexpected {s:?}"),
        }
    }
    assert!(run.pending_consumables().is_empty());
    // The play wins instantly, but press_play runs first: The Hook force-
    // discards 2 purple-seal cards -> 2 tarots (state_events.lua:400 runs
    // for hook discards too).
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    assert_eq!(run.pending_consumables(), &[t1, t2]);
}

#[test]
fn glass_cards_shatter_and_leave_the_deck() {
    // All-Glass deck: over a few rounds some cards must shatter (1 in 4 per
    // scored card) and card conservation shifts to destroyed_cards.
    let mut run = Run::new("GLASSSED");
    power_big(&mut run);
    all_cards(&mut run, |c| c.enhancement = Enhancement::Glass);
    let mut rounds = 0;
    while run.state() == State::BlindSelect && rounds < 6 {
        run.select_blind().unwrap();
        while run.state() == State::SelectingHand {
            run.play(&[0, 1, 2, 3, 4]).unwrap();
        }
        if run.state() == State::RoundEval {
            run.cash_out().unwrap();
        }
        rounds += 1;
    }
    let total = run.deck_len() + run.hand().len() + run.discard_pile_len();
    assert_eq!(total + run.destroyed_cards().len(), 52, "conservation");
    assert!(
        !run.destroyed_cards().is_empty(),
        "some glass must have shattered across {rounds} rounds"
    );
}
