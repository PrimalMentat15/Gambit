//! End-to-end smoke tests: a scripted policy driving the run state machine
//! must be deterministic and never panic — now with live boss effects
//! (forced selections, shrunk hands, zero discards, forced discards...).

use balatro_core::cards::HandType;
use balatro_core::run::{BlindStage, Run, State};

/// First `n` (or fewer) hand indices, always including Cerulean Bell's
/// forced card.
fn selection(run: &Run, n: usize) -> Vec<usize> {
    let mut sel: Vec<usize> = Vec::new();
    if let Some(f) = run.forced_card_index() {
        sel.push(f);
    }
    let mut i = 0;
    while sel.len() < n.min(run.hand().len()) && i < run.hand().len() {
        if !sel.contains(&i) {
            sel.push(i);
        }
        i += 1;
    }
    sel.sort_unstable();
    sel
}

/// Play the first five cards of the (desc-sorted) hand every time; cash out
/// when offered. Returns a trace string for determinism comparison.
fn scripted_run(seed: &str, max_steps: usize) -> (Run, String) {
    let mut run = Run::new(seed);
    let mut trace = String::new();
    trace.push_str(&format!("boss1={}\n", run.boss_choice()));
    for _ in 0..max_steps {
        match run.state() {
            State::BlindSelect => {
                trace.push_str(&format!(
                    "select ante={} stage={:?} boss={}\n",
                    run.ante(),
                    run.blind_on_deck(),
                    run.boss_choice()
                ));
                run.select_blind().unwrap();
                trace.push_str(&format!("blind_chips={}\n", run.blind_chips()));
                let hand: Vec<String> = run
                    .hand()
                    .iter()
                    .map(|c| {
                        format!(
                            "{}{}{}{}",
                            c.rank.key(),
                            c.suit.key(),
                            if c.face_down { "?" } else { "" },
                            if c.debuff { "!" } else { "" }
                        )
                    })
                    .collect();
                trace.push_str(&format!("hand={}\n", hand.join(",")));
            }
            State::SelectingHand => {
                let sel = selection(&run, 5);
                let result = run.play(&sel).unwrap().clone();
                trace.push_str(&format!(
                    "play {:?} chips={} mult={} score={} total={} $={}\n",
                    result.hand_type,
                    result.chips,
                    result.mult,
                    result.score,
                    run.chips(),
                    run.dollars(),
                ));
            }
            State::RoundEval => {
                trace.push_str(&format!(
                    "cashout pending={} dollars_before={}\n",
                    run.pending_cashout(),
                    run.dollars()
                ));
                run.cash_out().unwrap();
            }
            State::GameOver => {
                trace.push_str("game_over\n");
                break;
            }
            State::Won => {
                trace.push_str("won\n");
                break;
            }
            other => panic!("unexpected state {other:?}"),
        }
    }
    (run, trace)
}

#[test]
fn deterministic_and_panic_free() {
    let (run_a, trace_a) = scripted_run("TESTSEED", 10_000);
    let (run_b, trace_b) = scripted_run("TESTSEED", 10_000);
    assert_eq!(trace_a, trace_b, "same seed must replay identically");
    assert_eq!(run_a.round(), run_b.round());
    assert_eq!(run_a.dollars(), run_b.dollars());

    // The run must have ended cleanly (won or lost, never stuck).
    assert!(
        matches!(run_a.state(), State::GameOver | State::Won),
        "run did not terminate: {:?}",
        run_a.state()
    );
    // The naive policy always clears at least the ante-1 Small Blind before
    // dying; a 300-chip target falls to four 5-card junk hands.
    assert!(run_a.round() >= 1);
}

#[test]
fn different_seeds_diverge() {
    let (_, trace_a) = scripted_run("TESTSEED", 10_000);
    let (_, trace_b) = scripted_run("AAAAAAAA", 10_000);
    assert_ne!(trace_a, trace_b);
}

#[test]
fn invariants_hold_throughout() {
    let mut run = Run::new("XEQH7CP9");
    let mut steps = 0;
    loop {
        steps += 1;
        assert!(steps < 10_000, "run did not terminate");
        // Card conservation: 52 cards across deck/hand/discard/destroyed at
        // all times (no Glass in a plain Red Deck, so destroyed stays empty).
        let total = run.deck_len()
            + run.hand().len()
            + run.discard_pile_len()
            + run.destroyed_cards().len();
        assert_eq!(total, 52, "card count drifted at step {steps}");
        assert!(run.destroyed_cards().is_empty());
        match run.state() {
            State::BlindSelect => {
                run.select_blind().unwrap();
                // The Manacle shrinks the hand size for its round.
                assert_eq!(
                    run.hand().len() as i64,
                    run.hand_size(),
                    "must deal to hand size"
                );
                // The Needle leaves 1 hand, The Water 0 discards.
                let boss = run.current_blind().unwrap().name;
                let want_hands = if boss == "The Needle" { 1 } else { 4 };
                let want_discards = if boss == "The Water" { 0 } else { 4 };
                assert_eq!(run.hands_left(), want_hands);
                assert_eq!(run.discards_left(), want_discards);
                // Hand arrives sorted descending by get_nominal.
                let noms: Vec<f64> = run.hand().iter().map(|c| c.get_nominal(false)).collect();
                assert!(
                    noms.windows(2).all(|w| w[0] >= w[1]),
                    "hand not sorted desc"
                );
            }
            State::SelectingHand => {
                // Alternate discard and play to exercise both paths; keep the
                // selection legal under Cerulean Bell's forced card.
                if run.discards_left() > 0 && run.hands_left() == 4 {
                    let sel = match run.forced_card_index() {
                        Some(fi) => vec![fi],
                        None => vec![run.hand().len() - 1],
                    };
                    run.discard(&sel).unwrap();
                } else {
                    run.play(&selection(&run, 5)).unwrap();
                }
            }
            State::RoundEval => {
                assert!(run.pending_cashout() >= 0);
                run.cash_out().unwrap();
            }
            State::GameOver | State::Won => break,
            other => panic!("unexpected state {other:?}"),
        }
    }
}

#[test]
fn action_validation() {
    use balatro_core::run::RunError;
    let mut run = Run::new("TESTSEED");
    assert!(matches!(run.play(&[0]), Err(RunError::WrongState)));
    assert!(matches!(run.cash_out(), Err(RunError::WrongState)));
    run.select_blind().unwrap();
    assert!(matches!(run.select_blind(), Err(RunError::WrongState)));
    assert!(matches!(run.play(&[]), Err(RunError::BadCardSelection(_))));
    assert!(matches!(
        run.play(&[0, 1, 2, 3, 4, 5]),
        Err(RunError::BadCardSelection(_))
    ));
    assert!(matches!(
        run.play(&[0, 0]),
        Err(RunError::BadCardSelection(_))
    ));
    assert!(matches!(run.play(&[8]), Err(RunError::BadCardSelection(_))));
    // Burn all discards; the fifth must fail.
    run.discard(&[0]).unwrap();
    run.discard(&[0]).unwrap();
    run.discard(&[0]).unwrap();
    run.discard(&[0]).unwrap();
    assert!(matches!(run.discard(&[0]), Err(RunError::NoDiscardsLeft)));
}

/// Scripted power runs to ante 8 with every boss active along the way: the
/// boss pool gets exercised across seeds and each run ends in a win.
#[test]
fn powered_runs_reach_ante_8_with_bosses() {
    for seed in ["TESTSEED", "AAAAAAAA", "7NLLGSMA", "OOPS1234"] {
        let mut run = Run::new(seed);
        for ht in HandType::ALL {
            run.level_up_hand(ht, 2000);
        }
        let mut bosses_seen: Vec<&'static str> = Vec::new();
        let mut steps = 0;
        loop {
            steps += 1;
            assert!(steps < 10_000, "seed {seed}: run did not terminate");
            match run.state() {
                State::BlindSelect => {
                    if run.blind_on_deck() == BlindStage::Boss {
                        bosses_seen.push(run.boss_choice());
                    }
                    run.select_blind().unwrap();
                }
                State::SelectingHand => {
                    let sel = selection(&run, 5);
                    run.play(&sel).unwrap();
                }
                State::RoundEval => run.cash_out().unwrap(),
                State::Shop => run.leave_shop().unwrap(),
                State::Won => break,
                State::GameOver => panic!("seed {seed}: powered run lost?"),
                other => panic!("unexpected state {other:?}"),
            }
        }
        assert!(run.won(), "seed {seed} must win");
        assert_eq!(bosses_seen.len(), 8, "one boss per ante");
        assert!(
            bosses_seen[7].starts_with("bl_final_"),
            "ante 8 must be a showdown finisher, got {}",
            bosses_seen[7]
        );
    }
}
