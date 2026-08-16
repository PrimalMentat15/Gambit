//! Cross-checks the generated item registry (sim/core/src/items_gen.rs)
//! against the raw dump of game.lua's P_CENTERS/P_TAGS
//! (tests/data/centers.tsv, produced by tools/dump_centers.lua).

use balatro_core::items::{
    booster_by_key, consumable_by_key, joker_by_key, tag_by_key, voucher_by_key, BOOSTERS, JOKERS,
    PLANETS, SPECTRALS, TAGS, TAROTS, VOUCHERS,
};

const DATA: &str = include_str!("data/centers.tsv");

fn rows(set: &str) -> Vec<Vec<&'static str>> {
    DATA.lines()
        .skip(1)
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .filter(|f| f[1] == set)
        .collect()
}

#[test]
fn joker_registry_matches_dump() {
    let rows = rows("Joker");
    assert_eq!(rows.len(), 150);
    assert_eq!(JOKERS.len(), 150);
    for f in &rows {
        let m = joker_by_key(f[0]).unwrap_or_else(|| panic!("missing joker {}", f[0]));
        assert_eq!(m.name, f[3]);
        assert_eq!(m.cost.to_string(), f[4]);
        assert_eq!(m.rarity.to_string(), f[5]);
        assert_eq!(m.unlocked, f[6] == "1");
        assert_eq!(m.enhancement_gate.unwrap_or(""), f[23]);
        assert_eq!(m.blueprint_compat, f[26] == "1");
        assert_eq!(m.perishable_compat, f[27] == "1");
        assert_eq!(m.eternal_compat, f[28] == "1");
    }
    // JOKERS is sorted by center.order, and JokerId discriminants match.
    let mut by_order: Vec<(&str, i64)> =
        rows.iter().map(|f| (f[0], f[2].parse().unwrap())).collect();
    by_order.sort_by_key(|&(_, o)| o);
    for (i, (key, _)) in by_order.iter().enumerate() {
        assert_eq!(JOKERS[i].key, *key, "pool order at {i}");
        assert_eq!(JOKERS[i].id as usize, i);
    }
    // Rarity tallies: 61 common, 64 uncommon, 20 rare, 5 legendary.
    for (rarity, n) in [(1u8, 61), (2, 64), (3, 20), (4, 5)] {
        assert_eq!(JOKERS.iter().filter(|m| m.rarity == rarity).count(), n);
    }
}

#[test]
fn consumable_registry_matches_dump() {
    for (set, table) in [
        ("Tarot", &TAROTS[..]),
        ("Planet", &PLANETS[..]),
        ("Spectral", &SPECTRALS[..]),
    ] {
        let rows = rows(set);
        assert_eq!(rows.len(), table.len(), "{set} count");
        let mut by_order: Vec<(&str, i64)> =
            rows.iter().map(|f| (f[0], f[2].parse().unwrap())).collect();
        by_order.sort_by_key(|&(_, o)| o);
        for (i, (key, _)) in by_order.iter().enumerate() {
            assert_eq!(table[i].key, *key, "{set} pool order at {i}");
        }
        for f in &rows {
            let (m, _) = consumable_by_key(f[0]).unwrap();
            assert_eq!(m.name, f[3]);
            assert_eq!(m.cost.to_string(), f[4]);
            assert_eq!(
                m.max_highlighted.to_string(),
                if f[11].is_empty() { "0" } else { f[11] },
                "{} max_highlighted",
                f[0]
            );
            assert_eq!(m.hidden, f[20] == "1", "{} hidden", f[0]);
        }
    }
}

#[test]
fn voucher_booster_tag_registries_match_dump() {
    let vrows = rows("Voucher");
    assert_eq!(vrows.len(), 32);
    assert_eq!(VOUCHERS.len(), 32);
    for f in &vrows {
        let m = voucher_by_key(f[0]).unwrap();
        assert_eq!(m.name, f[3]);
        assert_eq!(m.cost.to_string(), f[4]);
        assert_eq!(m.unlocked, f[6] == "1");
        assert_eq!(m.requires.unwrap_or(""), f[21]);
    }
    let brows = rows("Booster");
    assert_eq!(brows.len(), 32);
    assert_eq!(BOOSTERS.len(), 32);
    for f in &brows {
        let m = booster_by_key(f[0]).unwrap();
        assert_eq!(m.name, f[3]);
        assert_eq!(m.cost.to_string(), f[4]);
        // The dump prints %.17g; compare the parsed doubles bit-exactly.
        assert_eq!(
            m.weight.to_bits(),
            f[7].parse::<f64>().unwrap().to_bits(),
            "{} weight",
            f[0]
        );
        assert_eq!(m.cards.to_string(), f[9], "{} cards", f[0]);
        assert_eq!(m.choose.to_string(), f[10], "{} choose", f[0]);
    }
    let trows = rows("Tag");
    assert_eq!(trows.len(), 24);
    assert_eq!(TAGS.len(), 24);
    for f in &trows {
        let m = tag_by_key(f[0]).unwrap();
        assert_eq!(m.name, f[3]);
        let want_min = if f[22].is_empty() {
            None
        } else {
            Some(f[22].parse::<i64>().unwrap())
        };
        assert_eq!(m.min_ante, want_min, "{} min_ante", f[0]);
    }
    // get_pack's total weight (all boosters): 22.42.
    let total: f64 = BOOSTERS.iter().map(|b| b.weight).sum();
    assert!((total - 22.42).abs() < 1e-9, "total weight {total}");
}
