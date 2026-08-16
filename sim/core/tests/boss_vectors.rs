//! get_new_boss fidelity against LuaJIT-oracle vectors.
//! Regenerate with:
//!   tools/oracle/luajit tools/gen_boss_vectors.lua > sim/core/tests/data/boss_vectors.tsv

use balatro_core::blinds::get_new_boss;
use balatro_core::rng::RngState;
use std::collections::HashMap;

const VECTORS: &str = include_str!("data/boss_vectors.tsv");

#[test]
fn matches_luajit_boss_rolls() {
    let mut states: HashMap<String, (RngState, HashMap<&'static str, i64>)> = HashMap::new();
    let mut count = 0usize;

    for line in VECTORS.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f[0], "boss");
        let (seed, ante, want) = (f[1], f[2].parse::<i64>().unwrap(), f[3]);
        let (rng, used) = states
            .entry(seed.to_string())
            .or_insert_with(|| (RngState::new(seed), HashMap::new()));
        let got = get_new_boss(rng, used, ante, 8);
        assert_eq!(got, want, "seed {seed:?} ante {ante}");
        count += 1;
    }
    assert!(count >= 72, "boss vectors: {count}");
}

#[test]
fn run_uses_boss_stream_for_ante_1() {
    // Run::new must roll the ante-1 boss exactly like the oracle sequence.
    let run = balatro_core::run::Run::new("TESTSEED");
    assert_eq!(run.boss_choice(), "bl_head");
    let run = balatro_core::run::Run::new("TUTORIAL");
    assert_eq!(run.boss_choice(), "bl_hook");
}
