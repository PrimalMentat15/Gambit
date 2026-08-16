//! Oracle parity for the P3a RNG streams: lucky_mult / lucky_money / glass /
//! wheel draws, The Hook's and Cerulean Bell's pseudorandom_element picks,
//! and the purple-seal tarot pool (with used-key resampling).
//!
//! Vectors from tools/gen_scoring_vectors.lua run under tools/oracle/luajit
//! (the system luajit produces different math.random output).

use balatro_core::consumables::purple_seal_tarot;
use balatro_core::rng::{LuaRandom, RngState};
use std::collections::HashSet;

const DATA: &str = include_str!("data/scoring_vectors.tsv");

fn hex_bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

/// The oracle's `{ sort_id = ((i*m) % n) + 1 }` hand construction.
fn scrambled_hand(n: usize, m: usize) -> Vec<u32> {
    (1..=n).map(|i| (((i * m) % n) + 1) as u32).collect()
}

/// `pseudorandom_element` over cards with sort_ids: sort ascending, one
/// math.random(#keys) draw, return the picked sort_id.
fn element_pick(rng: &mut RngState, key: &str, ids: &[u32]) -> u32 {
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    let seed = rng.pseudoseed(key);
    let j = LuaRandom::seeded(seed).random_range(1, sorted.len() as i64);
    sorted[(j - 1) as usize]
}

#[test]
fn matches_luajit_scoring_streams() {
    let mut checked = 0usize;
    for line in DATA.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        // ("hook"/"bell"/... lines are handled in dedicated passes below.)
        if f[0] == "stream" {
            let (seed, key, draws) = (f[1], f[2], f[3]);
            let mut rng = RngState::new(seed);
            for (i, expect) in draws.split(',').enumerate() {
                let got = rng.random(key);
                assert_eq!(hex_bits(got), expect, "stream {key} seed {seed} draw {i}");
                checked += 1;
            }
        }
    }

    // hook lines share stream state per seed (three consecutive plays).
    let mut rng: Option<(String, RngState)> = None;
    for line in DATA.lines().filter(|l| l.starts_with("hook\t")) {
        let f: Vec<&str> = line.split('\t').collect();
        let (seed, n, picks) = (f[1], f[2].parse::<usize>().unwrap(), f[3]);
        if rng.as_ref().map(|(s, _)| s.as_str()) != Some(seed) {
            rng = Some((seed.to_string(), RngState::new(seed)));
        }
        let (_, r) = rng.as_mut().unwrap();
        let mut ids = scrambled_hand(n, 7);
        let mut got: Vec<String> = Vec::new();
        for i in 0..2usize {
            if n > i {
                let sid = element_pick(r, "hook", &ids);
                ids.retain(|&s| s != sid);
                got.push(sid.to_string());
            }
        }
        assert_eq!(got.join(","), picks, "hook seed {seed} n {n}");
        checked += 1;
    }

    for line in DATA.lines().filter(|l| l.starts_with("bell\t")) {
        let f: Vec<&str> = line.split('\t').collect();
        let (seed, picks) = (f[1], f[2]);
        let mut r = RngState::new(seed);
        for entry in picks.split(',') {
            let (n, expect) = entry.split_once(':').unwrap();
            let ids = scrambled_hand(n.parse().unwrap(), 5);
            let sid = element_pick(&mut r, "cerulean_bell", &ids);
            assert_eq!(sid.to_string(), expect, "bell seed {seed} entry {entry}");
            checked += 1;
        }
    }

    for line in DATA.lines().filter(|l| l.starts_with("tarot\t")) {
        let f: Vec<&str> = line.split('\t').collect();
        let (seed, ante, keys) = (f[1], f[2].parse::<i64>().unwrap(), f[3]);
        let mut r = RngState::new(seed);
        let mut used: HashSet<String> = HashSet::new();
        for expect in keys.split(',') {
            let got = purple_seal_tarot(&mut r, ante, &used);
            assert_eq!(got, expect, "tarot seed {seed} ante {ante}");
            used.insert(got);
            checked += 1;
        }
    }

    // Threshold-decision parity: the < comparisons themselves.
    for line in DATA.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (kind, threshold, key): (&str, f64, &str) = match f[0] {
            "wheelflip" => ("wheelflip", 1.0 / 7.0, "wheel"),
            "glassbreak" => ("glassbreak", 1.0 / 4.0, "glass"),
            "luckymult" => ("luckymult", 1.0 / 5.0, "lucky_mult"),
            "luckymoney" => ("luckymoney", 1.0 / 15.0, "lucky_money"),
            _ => continue,
        };
        let (seed, flags) = (f[1], f[2]);
        let mut r = RngState::new(seed);
        for (i, expect) in flags.split(',').enumerate() {
            let got = r.random(key) < threshold;
            assert_eq!(
                if got { "1" } else { "0" },
                expect,
                "{kind} seed {seed} draw {i}"
            );
            checked += 1;
        }
    }

    assert!(checked > 500, "vector file too small? checked {checked}");
}
