//! pseudoshuffle bit-exactness against LuaJIT-oracle vectors.
//! Regenerate with:
//!   tools/oracle/luajit tools/gen_shuffle_vectors.lua > sim/core/tests/data/shuffle_vectors.tsv

use balatro_core::cards::{Card, Rank, Suit};
use balatro_core::deck::pseudoshuffle;
use balatro_core::rng::RngState;
use std::collections::HashMap;

const VECTORS: &str = include_str!("data/shuffle_vectors.tsv");

#[test]
fn matches_luajit_shuffles() {
    // Per seed: a persistent pseudoseed state and a persistent 52-card list,
    // exactly like the game's deck. Initial order mirrors deck emplacement
    // (front-insert => cards[0] carries the highest sort_id).
    let mut states: HashMap<String, (RngState, Vec<Card>)> = HashMap::new();
    let mut count = 0usize;

    for line in VECTORS.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f[0], "shuffle", "unexpected vector type {:?}", f[0]);
        let (seed, key, op) = (f[1], f[2], f[3]);
        let (rng, list) = states.entry(seed.to_string()).or_insert_with(|| {
            let list: Vec<Card> = (0..52)
                .map(|i| Card::new(Rank(2), Suit::Spades, 52 - i))
                .collect();
            (RngState::new(seed), list)
        });

        let s = rng.pseudoseed(key);
        pseudoshuffle(list, s);

        let got: Vec<String> = list.iter().map(|c| c.sort_id.to_string()).collect();
        assert_eq!(
            got.join(","),
            f[4],
            "seed {seed:?} key {key:?} op {op}: permutation mismatch"
        );
        count += 1;
    }
    assert!(count >= 60, "shuffle vectors: {count}");
}
