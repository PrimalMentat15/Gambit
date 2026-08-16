//! Oracle-vector parity for the common-joker RNG streams
//! (tools/gen_joker_vectors.lua -> data/joker_vectors.tsv): the plain
//! probability rolls ('8ball'/'business'/'parking'/'gros_michel'/
//! 'cavendish'/'halu<ante>'), Misprint's ranged roll, To Do List's
//! string-array pick, the mail/idol/castle card picks (sort_id ordering) and
//! Ancient Joker's suit pick.

use balatro_core::items::element_index;
use balatro_core::rng::RngState;

const DATA: &str = include_str!("data/joker_vectors.tsv");

/// pairs(G.GAME.hands) iteration order, as in the Lua harness.
const HANDLIST: [&str; 12] = [
    "Pair",
    "High Card",
    "Flush Five",
    "Flush House",
    "Five of a Kind",
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
];

fn bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

#[test]
fn joker_streams_match_oracle() {
    let mut rng: Option<(String, RngState)> = None;
    let mut checked = 0usize;
    // The mock card table of the Lua harness: 52 cards, sort_id
    // ((i*17) % 52) + 1 — pseudorandom_element sorts by sort_id, so the
    // pick with index k (0-based) has sort_id k+1.
    // Ancient pick state per seed:
    let mut ancient: Option<&str> = None;

    for line in DATA.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let seed = f[1];
        if rng.as_ref().map(|(s, _)| s.as_str()) != Some(seed) {
            rng = Some((seed.to_string(), RngState::new(seed)));
            ancient = None;
        }
        let r = &mut rng.as_mut().unwrap().1;
        match f[0] {
            "roll" => {
                // pseudorandom(key) — one math.random() in [0,1).
                let key = f[2];
                assert_eq!(bits(r.random(key)), f[4], "{line}");
            }
            "misprint" => {
                // pseudorandom('misprint', 0, 23) (card.lua:3701).
                let v: i64 = f[3].parse().unwrap();
                assert_eq!(r.random_range("misprint", 0, 23), v, "{line}");
            }
            "to_do" => {
                // pseudorandom_element over a string array: array order is
                // preserved (numeric keys), one math.random(n) draw.
                let idx = element_index(r, "to_do", HANDLIST.len());
                assert_eq!(HANDLIST[idx], f[3], "{line}");
            }
            "card_pick" => {
                // pseudorandom_element over Card tables: sorted by sort_id
                // 1..=52, one math.random(52) draw -> sort_id == idx + 1.
                let key = f[2];
                let want: usize = f[4].parse().unwrap();
                let idx = element_index(r, key, 52);
                assert_eq!(idx + 1, want, "{line}");
            }
            "ancient" => {
                // reset_ancient_card (common_events.lua:2303-2310): the
                // pool excludes the current suit.
                const SUITS: [&str; 4] = ["Spades", "Hearts", "Clubs", "Diamonds"];
                let pool: Vec<&str> = SUITS
                    .iter()
                    .copied()
                    .filter(|&s| Some(s) != ancient)
                    .collect();
                let idx = element_index(r, "anc1", pool.len());
                ancient = Some(pool[idx]);
                assert_eq!(pool[idx], f[3], "{line}");
            }
            other => panic!("unknown row kind {other}"),
        }
        checked += 1;
    }
    assert_eq!(checked, DATA.lines().count());
    assert!(checked >= 300);
}
