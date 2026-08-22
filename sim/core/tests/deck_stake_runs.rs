//! Starting state for every deck and stake, through the real `Run` constructor.
//!
//! `config.rs`'s unit tests cover `RunConfig::resolve` in isolation; these go
//! through `Run::with_config` so a param that resolves correctly but never
//! reaches the run state still fails.

use balatro_core::config::{Deck, RunConfig, Scaling, Stake};
use balatro_core::run::Run;

const SEED: &str = "DECKRUNS";

fn run_with(deck: Deck, stake: Stake) -> Run {
    Run::with_config(SEED, RunConfig::new(deck, stake))
}

/// `Run::new` must stay exactly `Run::with_config(.., Red/White)` — this is
/// what keeps every frozen oracle vector valid across the P7 refactor.
#[test]
fn new_is_red_white() {
    let a = Run::new(SEED);
    let b = run_with(Deck::Red, Stake::White);
    assert_eq!(a.deck_kind(), Deck::Red);
    assert_eq!(a.stake(), Stake::White);
    assert_eq!(a.dollars(), b.dollars());
    assert_eq!(a.hands_left(), b.hands_left());
    assert_eq!(a.discards_left(), b.discards_left());
    assert_eq!(a.hand_size(), b.hand_size());
    assert_eq!(a.joker_slots(), b.joker_slots());
    assert_eq!(a.consumable_slots(), b.consumable_slots());
    assert_eq!(a.deck_len(), b.deck_len());
    assert_eq!(a.ante(), b.ante());
}

/// The pre-P7 hardcoded baseline, restated as an executable claim.
#[test]
fn red_white_baseline_is_unchanged() {
    let r = Run::new(SEED);
    assert_eq!(r.dollars(), 4);
    assert_eq!(r.hands_left(), 4);
    assert_eq!(r.discards_left(), 4, "base 3 + Red Deck +1");
    assert_eq!(r.hand_size(), 8);
    assert_eq!(r.joker_slots(), 5);
    assert_eq!(r.consumable_slots(), 2);
}

#[test]
fn every_deck_reaches_the_run_state() {
    // (deck, dollars, hands, discards, hand_size, joker_slots, consumables)
    let cases = [
        (Deck::Red, 4, 4, 4, 8, 5, 2),
        (Deck::Blue, 4, 5, 3, 8, 5, 2),
        (Deck::Yellow, 14, 4, 3, 8, 5, 2),
        (Deck::Green, 4, 4, 3, 8, 5, 2),
        (Deck::Black, 4, 3, 3, 8, 6, 2),
        (Deck::Nebula, 4, 4, 3, 8, 5, 1),
        (Deck::Painted, 4, 4, 3, 10, 4, 2),
        (Deck::Plasma, 4, 4, 3, 8, 5, 2),
        (Deck::Erratic, 4, 4, 3, 8, 5, 2),
    ];
    for (deck, dollars, hands, discards, hand_size, jokers, cons) in cases {
        let r = run_with(deck, Stake::White);
        assert_eq!(r.deck_kind(), deck);
        assert_eq!(r.dollars(), dollars, "{deck:?} dollars");
        assert_eq!(r.hands_left(), hands, "{deck:?} hands");
        assert_eq!(r.discards_left(), discards, "{deck:?} discards");
        assert_eq!(r.hand_size(), hand_size, "{deck:?} hand size");
        assert_eq!(r.joker_slots(), jokers, "{deck:?} joker slots");
        assert_eq!(r.consumable_slots(), cons, "{deck:?} consumable slots");
    }
}

#[test]
fn deck_composition_reaches_the_draw_pile() {
    // Abandoned is the only deck whose starting pile is not 52 cards.
    for deck in Deck::ALL {
        let r = run_with(deck, Stake::White);
        let want = if deck == Deck::Abandoned { 40 } else { 52 };
        assert_eq!(r.deck_len(), want, "{deck:?} draw pile");
    }
}

#[test]
fn stake_modifiers_reach_the_run_state() {
    // Only Blue Stake's -1 discard is observable in the starting state; the
    // rest live in `modifiers` until the systems that read them land.
    assert_eq!(run_with(Deck::Red, Stake::Black).discards_left(), 4);
    assert_eq!(run_with(Deck::Red, Stake::Blue).discards_left(), 3);
    assert_eq!(run_with(Deck::Red, Stake::Gold).discards_left(), 3);

    let m = *run_with(Deck::Red, Stake::Gold).modifiers();
    assert_eq!(m.scaling, Scaling::Three);
    assert!(m.enable_eternals_in_shop);
    assert!(m.enable_perishables_in_shop);
    assert!(m.enable_rentals_in_shop);
    assert!(m.no_blind_reward[0]);

    let w = *run_with(Deck::Red, Stake::White).modifiers();
    assert_eq!(w.scaling, Scaling::One);
    assert!(!w.enable_eternals_in_shop);
    assert!(!w.no_blind_reward[0]);
}

/// Deck and stake are independent axes; the full grid must construct.
#[test]
fn the_whole_grid_constructs() {
    for deck in Deck::ALL {
        for stake in Stake::ALL {
            let r = run_with(deck, stake);
            assert_eq!(r.deck_kind(), deck);
            assert_eq!(r.stake(), stake);
            assert!(r.deck_len() >= 40);
            assert!(r.hands_left() >= 1, "{deck:?}/{stake:?} unplayable");
            assert!(r.discards_left() >= 0);
            assert!(r.joker_slots() >= 1);
        }
    }
}

/// Only the Erratic Deck touches the RNG during construction, so every other
/// deck must produce the Red Deck's ante-1 boss for a given seed.
#[test]
fn deck_choice_does_not_shift_the_boss_roll() {
    let red = Run::new(SEED).ante();
    for deck in Deck::ALL {
        let r = run_with(deck, Stake::White);
        assert_eq!(r.ante(), red);
    }
}

/// Magic / Nebula / Zodiac pre-redeem vouchers; Magic / Ghost start holding
/// consumables (back.lua:176-196, :232-238).
#[test]
fn starting_vouchers_and_consumables_land() {
    // Magic: Crystal Ball is redeemed, so 2 base slots become 3, and it holds
    // two Fools.
    let m = run_with(Deck::Magic, Stake::White);
    assert!(m.used_vouchers().contains("v_crystal_ball"));
    assert_eq!(m.consumable_slots(), 3, "base 2 + Crystal Ball");
    let held: Vec<&str> = m.consumables().iter().map(|c| c.key).collect();
    assert_eq!(held, vec!["c_fool", "c_fool"]);

    // Nebula: Telescope is a no-op in apply_to_run (card.lua:1910-1911); the
    // slot count comes purely from the deck's consumable_slot = -1.
    let n = run_with(Deck::Nebula, Stake::White);
    assert!(n.used_vouchers().contains("v_telescope"));
    assert_eq!(n.consumable_slots(), 1);
    assert!(n.consumables().is_empty());

    // Ghost: one Hex, and the shop's spectral weight is live.
    let g = run_with(Deck::Ghost, Stake::White);
    let held: Vec<&str> = g.consumables().iter().map(|c| c.key).collect();
    assert_eq!(held, vec!["c_hex"]);
    assert_eq!(g.modifiers().spectral_rate, 2.0);

    // Zodiac: all three redeemed.
    let z = run_with(Deck::Zodiac, Stake::White);
    for key in ["v_tarot_merchant", "v_planet_merchant", "v_overstock_norm"] {
        assert!(z.used_vouchers().contains(key), "{key} not redeemed");
    }

    // Decks with neither leave both empty.
    let r = run_with(Deck::Red, Stake::White);
    assert!(r.used_vouchers().is_empty());
    assert!(r.consumables().is_empty());
}

/// The deck's vouchers are redeemed BEFORE the run-start 'Voucher1' roll
/// (game.lua:2043 vs :2178), so they cannot be offered back to the player.
#[test]
fn pre_redeemed_vouchers_leave_the_shop_pool() {
    for deck in [Deck::Magic, Deck::Nebula, Deck::Zodiac] {
        let r = run_with(deck, Stake::White);
        let offered = r.current_voucher().expect("run-start voucher");
        for &owned in deck.starting_vouchers() {
            assert_ne!(
                offered, owned,
                "{deck:?} was offered {owned}, which it already owns --                  apply_to_run must precede the Voucher1 roll"
            );
        }
    }
}
