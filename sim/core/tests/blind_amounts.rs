//! get_blind_amount fidelity: hardcoded antes 1-8 plus bit-exact comparison
//! against LuaJIT-oracle vectors for the ante>8 formula.
//! Regenerate with:
//!   tools/oracle/luajit tools/gen_blind_vectors.lua > sim/core/tests/data/blind_amounts.tsv

use balatro_core::blinds::{get_blind_amount, BL_BIG, BL_SMALL};

const VECTORS: &str = include_str!("data/blind_amounts.tsv");

#[test]
fn antes_1_to_8_table() {
    let expected = [
        300.0, 800.0, 2000.0, 5000.0, 11000.0, 20000.0, 35000.0, 50000.0,
    ];
    for (i, want) in expected.iter().enumerate() {
        assert_eq!(get_blind_amount(i as i64 + 1), *want, "ante {}", i + 1);
    }
    assert_eq!(get_blind_amount(0), 100.0);
    assert_eq!(get_blind_amount(-3), 100.0);
}

#[test]
fn matches_luajit_vectors() {
    let mut count = 0usize;
    for line in VECTORS.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f[0], "blind_amount");
        let ante: i64 = f[1].parse().unwrap();
        let want = f64::from_bits(u64::from_str_radix(f[2], 16).unwrap());
        let got = get_blind_amount(ante);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "ante {ante}: got {got}, want {want}"
        );
        count += 1;
    }
    assert!(count >= 20, "blind vectors: {count}");
}

#[test]
fn blind_chip_requirements() {
    // Small = 1x, Big = 1.5x of the ante base amount.
    assert_eq!(BL_SMALL.chips(1), 300.0);
    assert_eq!(BL_BIG.chips(1), 450.0);
    assert_eq!(BL_SMALL.chips(2), 800.0);
    assert_eq!(BL_BIG.chips(2), 1200.0);
    // Bosses are mostly 2x; The Wall 4x, Violet Vessel 6x, The Needle 1x.
    assert_eq!(balatro_core::blinds::boss_by_key("bl_hook").chips(1), 600.0);
    assert_eq!(
        balatro_core::blinds::boss_by_key("bl_wall").chips(2),
        3200.0
    );
    assert_eq!(
        balatro_core::blinds::boss_by_key("bl_final_vessel").chips(8),
        300000.0
    );
    assert_eq!(
        balatro_core::blinds::boss_by_key("bl_needle").chips(2),
        800.0
    );
}
