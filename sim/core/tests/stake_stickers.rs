//! P7 Stage 2: eternal / perishable / rental stickers (Black / Orange / Gold
//! stakes). Covers both the RNG-faithful application in `create_joker` and the
//! per-round tick in `end_round`.

use balatro_core::cards::{Edition, HandType};
use balatro_core::config::{Deck, RunConfig, Stake, PERISHABLE_ROUNDS, RENTAL_RATE};
use balatro_core::items::{joker_by_key, JokerId};
use balatro_core::run::{Run, State};
use balatro_core::shop::ShopItemKind;

/// Drive a fresh run to its first shop, winning every blind with pumped hands.
fn run_to_shop(cfg: RunConfig, seed: &str) -> Run {
    let mut run = Run::with_config(seed, cfg);
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    loop {
        match run.state() {
            State::BlindSelect => run.select_blind().unwrap(),
            State::SelectingHand => run.play(&[0, 1, 2, 3, 4]).map(|_| ()).unwrap(),
            State::RoundEval => run.cash_out().unwrap(),
            State::Shop => return run,
            s => panic!("unexpected {s:?}"),
        }
    }
}

fn win_next_blind(run: &mut Run) {
    loop {
        match run.state() {
            State::BlindSelect => run.select_blind().unwrap(),
            State::SelectingHand => run.play(&[0, 1, 2, 3, 4]).map(|_| ()).unwrap(),
            State::RoundEval => {
                run.cash_out().unwrap();
                return;
            }
            State::Shop => run.leave_shop().unwrap(),
            s => panic!("unexpected {s:?}"),
        }
    }
}

/// Collect every shop joker across a block of seeds at a given stake, opening
/// the shop and rerolling a few times to widen the sample.
fn sample_shop_jokers(stake: Stake, seeds: u64) -> Vec<(JokerId, bool, bool, i64, bool)> {
    let mut out = Vec::new();
    for s in 0..seeds {
        let mut run = run_to_shop(RunConfig::new(Deck::Red, stake), &format!("STK{s:05}"));
        for _ in 0..4 {
            if let Some(shop) = run.shop() {
                for item in &shop.jokers {
                    if let ShopItemKind::Joker(j) = &item.kind {
                        out.push((j.id, j.eternal, j.perishable, j.perish_tally, j.rental));
                    }
                }
            }
            if run.reroll_shop().is_err() {
                break;
            }
        }
    }
    out
}

#[test]
fn white_stake_has_no_stickers() {
    let js = sample_shop_jokers(Stake::White, 40);
    assert!(!js.is_empty(), "no shop jokers sampled");
    for (id, e, p, t, r) in js {
        assert!(
            !e && !p && !r && t == 0,
            "{id:?} carried a sticker on White Stake"
        );
    }
}

#[test]
fn gold_stake_applies_all_three_stickers_within_compat() {
    // Gold enables eternals, perishables AND rentals (stake cascade).
    let js = sample_shop_jokers(Stake::Gold, 120);
    assert!(js.len() > 200, "sample too small: {}", js.len());

    let (mut e, mut p, mut r) = (0, 0, 0);
    for (id, eternal, perishable, tally, rental) in &js {
        let meta = joker_by_key(id.key()).unwrap();
        if *eternal {
            e += 1;
            assert!(
                meta.eternal_compat,
                "{id:?} eternal despite !eternal_compat"
            );
            assert!(!*perishable, "{id:?} both eternal and perishable");
        }
        if *perishable {
            p += 1;
            assert!(
                meta.perishable_compat,
                "{id:?} perishable despite !perishable_compat"
            );
            assert_eq!(
                *tally, PERISHABLE_ROUNDS,
                "{id:?} perish_tally not initialized"
            );
        } else {
            assert_eq!(*tally, 0, "{id:?} tally set without perishable");
        }
        if *rental {
            r += 1;
        }
    }
    // The bands are ~30% eternal / ~30% perishable / ~30% rental of eligible
    // draws; over 200+ jokers each must appear.
    assert!(
        e > 0 && p > 0 && r > 0,
        "missing sticker kind: e={e} p={p} r={r}"
    );
}

#[test]
fn rental_drains_three_dollars_per_round() {
    // Two identical runs differing only in a rental flag on an
    // economy-neutral joker. The GREEN DECK is used so `no_interest` is set:
    // otherwise the drained dollars would earn less interest and the gap would
    // compound past the raw drain (a real effect, but it makes the exact
    // amount awkward to assert). With interest gone the gap is purely rental.
    let seed = "RENTAL01";
    let setup = |rental: bool| {
        let mut run = Run::with_config(seed, RunConfig::new(Deck::Green, Stake::White));
        for ht in HandType::ALL {
            run.level_up_hand(ht, 5000);
        }
        run.debug_add_joker(JokerId::Joker, Edition::None);
        if rental {
            run.debug_joker_mut(0).rental = true;
        }
        run
    };
    let mut plain = setup(false);
    let mut rented = setup(true);
    for _ in 0..3 {
        win_next_blind(&mut plain);
        win_next_blind(&mut rented);
    }
    assert_eq!(
        plain.dollars() - rented.dollars(),
        3 * RENTAL_RATE,
        "3 rounds of rental should cost exactly 3 x {RENTAL_RATE} with no interest"
    );
}

#[test]
fn rental_drain_compounds_through_interest() {
    // The flip side of the note above: WITH interest, the gap exceeds the raw
    // drain because the rented run keeps less money and so earns less interest
    // each round. This pins that the drain happens before the interest step.
    let setup = |rental: bool| {
        let mut run = Run::new("RENTAL02");
        for ht in HandType::ALL {
            run.level_up_hand(ht, 5000);
        }
        run.debug_add_joker(JokerId::Joker, Edition::None);
        if rental {
            run.debug_joker_mut(0).rental = true;
        }
        run
    };
    let mut plain = setup(false);
    let mut rented = setup(true);
    for _ in 0..3 {
        win_next_blind(&mut plain);
        win_next_blind(&mut rented);
    }
    let gap = plain.dollars() - rented.dollars();
    assert!(
        gap > 3 * RENTAL_RATE,
        "interest should widen the gap: {gap}"
    );
}

#[test]
fn perishable_counts_down_and_debuffs_at_zero() {
    let mut run = Run::new("PERISH01");
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    run.debug_add_joker(JokerId::Joker, Edition::None);
    {
        let j = run.debug_joker_mut(0);
        j.perishable = true;
        j.perish_tally = PERISHABLE_ROUNDS;
    }

    // Each won round decrements the tally by one; it debuffs when it hits 0.
    for round in 1..=PERISHABLE_ROUNDS {
        win_next_blind(&mut run);
        let j = &run.jokers()[0];
        let want = PERISHABLE_ROUNDS - round;
        assert_eq!(j.perish_tally, want, "after {round} rounds");
        assert_eq!(
            j.debuffed,
            want == 0,
            "debuff state wrong after {round} rounds (tally {want})"
        );
    }
}

#[test]
fn only_gold_stake_draws_the_rental_stream() {
    // Below Gold the rental poll is never drawn, so the run seed maps to the
    // same sequence of shop jokers as White -- proof the added stream does not
    // perturb anything on lower stakes. (Eternal/perishable draw the SAME
    // etper poll on every stake, so only their APPLICATION differs, never the
    // stream position.)
    let ids = |stake: Stake| -> Vec<JokerId> {
        let run = run_to_shop(RunConfig::new(Deck::Red, stake), "STREAM07");
        run.shop()
            .unwrap()
            .jokers
            .iter()
            .filter_map(|it| match &it.kind {
                ShopItemKind::Joker(j) => Some(j.id),
                _ => None,
            })
            .collect()
    };
    // Orange (7) enables perishables+eternals but NOT rentals, so its stream
    // matches White's joker identities exactly.
    assert_eq!(
        ids(Stake::White),
        ids(Stake::Orange),
        "stream shifted below Gold"
    );
}

#[test]
fn rental_joker_sells_for_one() {
    // A rental joker's cost is forced to 1 (card.lua:381), so it always sells
    // for $1 -- not half its base price.
    let mut run = Run::new("SELL01");
    run.debug_add_joker(JokerId::Blueprint, Edition::None); // base cost 10
    let plain = run.joker_sell_value(&run.jokers()[0]);
    run.debug_joker_mut(0).rental = true;
    let rented = run.joker_sell_value(&run.jokers()[0]);
    assert!(plain > 1, "base sell value should exceed 1: {plain}");
    assert_eq!(rented, 1, "rental joker must sell for exactly 1");
}
