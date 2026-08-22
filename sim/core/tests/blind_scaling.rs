//! The three `get_blind_amount` curves (misc_functions.lua:919-954) and the
//! independent `ante_scaling` multiplier (blind.lua:107).

use balatro_core::blinds::{
    boss_by_key, get_blind_amount, get_blind_amount_scaled, BL_BIG, BL_SMALL,
};
use balatro_core::config::{Deck, RunConfig, Scaling, Stake};
use balatro_core::run::Run;

/// Verbatim from misc_functions.lua:922-924 / :932-935 / :943-946.
const TABLES: [(Scaling, [f64; 8]); 3] = [
    (
        Scaling::One,
        [
            300.0, 800.0, 2000.0, 5000.0, 11000.0, 20000.0, 35000.0, 50000.0,
        ],
    ),
    (
        Scaling::Two,
        [
            300.0, 900.0, 2600.0, 8000.0, 20000.0, 36000.0, 60000.0, 100000.0,
        ],
    ),
    (
        Scaling::Three,
        [
            300.0, 1000.0, 3200.0, 9000.0, 25000.0, 60000.0, 110000.0, 200000.0,
        ],
    ),
];

#[test]
fn ante_1_to_8_matches_each_table() {
    for (scaling, table) in TABLES {
        for (i, want) in table.iter().enumerate() {
            let ante = i as i64 + 1;
            assert_eq!(
                get_blind_amount_scaled(ante, scaling),
                *want,
                "{scaling:?} ante {ante}"
            );
        }
        // ante < 1 is a flat 100 on every curve (misc_functions.lua:925).
        assert_eq!(get_blind_amount_scaled(0, scaling), 100.0);
    }
}

/// The base curve must be untouched — every frozen oracle vector depends on it.
#[test]
fn unscaled_entry_point_is_the_base_curve() {
    for ante in 1..=16 {
        assert_eq!(
            get_blind_amount(ante),
            get_blind_amount_scaled(ante, Scaling::One),
            "ante {ante}"
        );
    }
}

/// Past ante 8 all three branches run the identical tail formula, anchored on
/// their own table's last entry — so the curves stay ordered and none collapses.
#[test]
fn tail_formula_is_shared_and_curves_stay_ordered() {
    for ante in 9..=20 {
        let one = get_blind_amount_scaled(ante, Scaling::One);
        let two = get_blind_amount_scaled(ante, Scaling::Two);
        let three = get_blind_amount_scaled(ante, Scaling::Three);
        assert!(
            one.is_finite() && two.is_finite() && three.is_finite(),
            "ante {ante}"
        );
        assert!(one < two && two < three, "ante {ante}: {one} {two} {three}");
        // Each curve is truncated to two significant figures INDEPENDENTLY
        // (`amount - amount % 10^floor(log10(amount)-1)`,
        // misc_functions.lua:928-929). That is why the curves are not exactly
        // proportional past ante 8 even though the tail formula is shared:
        // at ante 9 the ratio is 230000/110000 = 2.0909..., not 2.
        // Only meaningful while the values are exactly representable: past
        // ~2^53 an f64 is no longer an exact integer and `%` against a power
        // of ten returns noise. The implementation itself uses `lua_fmod`
        // exactly as Lua does, so the game has the same behaviour up there.
        if one < 9e15 {
            for amount in [one, two, three] {
                let step = 10f64.powf((amount.log10() - 1.0).floor());
                assert_eq!(amount % step, 0.0, "ante {ante}: {amount} not 2 s.f.");
            }
        }
    }
}

#[test]
fn blind_multipliers_apply_on_every_curve() {
    // Small x1, Big x1.5 (game.lua:264-265).
    assert_eq!(BL_SMALL.chips_scaled(2, Scaling::Two, 1.0), 900.0);
    assert_eq!(BL_BIG.chips_scaled(2, Scaling::Two, 1.0), 1350.0);
    assert_eq!(BL_SMALL.chips_scaled(2, Scaling::Three, 1.0), 1000.0);
    // A x2 boss on the base curve, unchanged from pre-P7.
    assert_eq!(
        boss_by_key("bl_hook").chips_scaled(1, Scaling::One, 1.0),
        600.0
    );
}

/// `ante_scaling` (Plasma) multiplies whatever curve the stake selected — the
/// two are independent inputs, not alternatives.
#[test]
fn ante_scaling_is_independent_of_the_stake_curve() {
    assert_eq!(BL_SMALL.chips_scaled(2, Scaling::One, 2.0), 1600.0);
    assert_eq!(BL_SMALL.chips_scaled(2, Scaling::Two, 2.0), 1800.0);
    assert_eq!(BL_SMALL.chips_scaled(2, Scaling::Three, 2.0), 2000.0);
}

/// The same three facts, but reached through a real `Run` so a config that
/// resolves correctly yet never reaches blind selection still fails.
#[test]
fn runs_apply_their_own_deck_and_stake() {
    let chips = |deck, stake, ante| {
        Run::with_config("BLINDSCALE", RunConfig::new(deck, stake)).proto_chips(&BL_SMALL, ante)
    };

    // Red / White: the pre-P7 baseline.
    assert_eq!(chips(Deck::Red, Stake::White, 2), 800.0);
    // Plasma doubles it.
    assert_eq!(chips(Deck::Plasma, Stake::White, 2), 1600.0);
    // Green Stake moves to curve 2; Purple to curve 3.
    assert_eq!(chips(Deck::Red, Stake::Green, 2), 900.0);
    assert_eq!(chips(Deck::Red, Stake::Purple, 2), 1000.0);
    // Both at once: Plasma on Gold Stake.
    assert_eq!(chips(Deck::Plasma, Stake::Gold, 2), 2000.0);
}
