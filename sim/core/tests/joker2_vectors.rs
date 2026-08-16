//! Oracle-vector parity for the P3c-2 joker RNG streams
//! (tools/gen_joker2_vectors.lua -> data/joker2_vectors.tsv): the plain
//! probability rolls ('space'/'bloodstone'/'certsl'), the joker/consumable
//! picks over sort_id-ordered tables ('madness'/'invisible'/'perkeo') and
//! the P_CARDS front picks ('cert_fr'/'marb_fr').

use balatro_core::items::{card_front_from_index, element_index};
use balatro_core::rng::RngState;

const DATA: &str = include_str!("data/joker2_vectors.tsv");

fn bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

#[test]
fn joker2_streams_match_oracle() {
    let mut rng: Option<(String, RngState)> = None;
    let mut checked = 0usize;
    for line in DATA.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let seed = f[1];
        if rng.as_ref().map(|(s, _)| s.as_str()) != Some(seed) {
            rng = Some((seed.to_string(), RngState::new(seed)));
        }
        let r = &mut rng.as_mut().unwrap().1;
        match f[0] {
            "roll" => {
                assert_eq!(bits(r.random(f[2])), f[4], "{line}");
            }
            "joker_pick" => {
                // pseudorandom_element over Card tables sorts by sort_id:
                // the pick with 0-based index k has sort_id k+1.
                let key = f[2];
                let n: usize = f[4].parse().unwrap();
                let expect: usize = f[5].parse().unwrap();
                let idx = element_index(r, key, n);
                assert_eq!(idx + 1, expect, "{line}");
            }
            "front" => {
                // pseudorandom_element over the P_CARDS string keys —
                // byte-lexicographic order == card_front_from_index.
                let key = f[2];
                let idx = element_index(r, key, 52);
                let (suit, rank) = card_front_from_index(idx);
                let got = format!("{}_{}", suit.key(), rank.key());
                assert_eq!(got, f[4], "{line}");
            }
            other => panic!("unknown row {other}"),
        }
        checked += 1;
    }
    assert_eq!(checked, 190);
}
