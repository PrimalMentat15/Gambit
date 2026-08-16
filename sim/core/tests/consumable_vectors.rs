//! Oracle vectors for the consumable-usage streams (scenario 3 of
//! tools/gen_shop_vectors.lua): Emperor/High Priestess pools, Judgement/
//! Soul/Wraith joker creation, Wheel of Fortune, Ankh/Hex/Ectoplasm picks,
//! Sigil/Ouija conversions, Aura editions, Familiar/Grim/Incantation
//! destroy-and-create, and Immolate's shuffle-destroy.

use balatro_core::cards::{Edition, Enhancement};
use balatro_core::items::JokerId;
use balatro_core::run::Run;

const DATA: &str = include_str!("data/shop_vectors.tsv");

fn row<'a>(kind: &str, seed: &str) -> Vec<&'a str>
where
    'static: 'a,
{
    DATA.lines()
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .find(|f| f[0] == kind && f[1] == seed)
        .unwrap_or_else(|| panic!("row {kind} {seed}"))
}

fn edition_from(s: &str) -> Edition {
    match s {
        "none" => Edition::None,
        "foil" => Edition::Foil,
        "holo" => Edition::Holo,
        "polychrome" => Edition::Polychrome,
        "negative" => Edition::Negative,
        _ => panic!("bad edition {s}"),
    }
}

fn enhancement_from(s: &str) -> Enhancement {
    match s {
        "m_bonus" => Enhancement::Bonus,
        "m_mult" => Enhancement::Mult,
        "m_wild" => Enhancement::Wild,
        "m_glass" => Enhancement::Glass,
        "m_steel" => Enhancement::Steel,
        "m_gold" => Enhancement::Gold,
        "m_lucky" => Enhancement::Lucky,
        _ => panic!("bad enhancement {s}"),
    }
}

fn use_from_inventory(run: &mut Run, key: &'static str, targets: &[usize]) {
    run.debug_add_consumable(key);
    let idx = run.consumables().len() - 1;
    run.use_consumable(idx, targets).unwrap_or_else(|e| {
        panic!("using {key}: {e}");
    });
}

/// Sorted-by-sort_id ids of the current hand (the pseudorandom_element
/// candidate order for Card tables).
fn sorted_hand_ids(run: &Run) -> Vec<u32> {
    let mut ids: Vec<u32> = run.hand().iter().map(|c| c.sort_id).collect();
    ids.sort_unstable();
    ids
}

#[test]
fn consumable_streams_match_lua() {
    for seed in ["TUTORIAL", "AAAAAAAA", "TAROTSED"] {
        let mut run = Run::new(seed);
        run.debug_set_dollars(1000);

        // --- The Emperor: 2 tarots off 'Tarotemp1' ---
        let exp = row("cemp", seed);
        use_from_inventory(&mut run, "c_emperor", &[]);
        let got: Vec<String> = run.pending_consumables();
        assert_eq!(
            got,
            [exp[2].to_string(), exp[3].to_string()],
            "{seed} emperor"
        );
        run.sell_consumable(0).unwrap();
        run.sell_consumable(0).unwrap();

        // --- High Priestess: 2 planets off 'Planetpri1' ---
        let exp = row("cpri", seed);
        use_from_inventory(&mut run, "c_high_priestess", &[]);
        let got: Vec<String> = run.pending_consumables();
        assert_eq!(
            got,
            [exp[2].to_string(), exp[3].to_string()],
            "{seed} priestess"
        );
        run.sell_consumable(0).unwrap();
        run.sell_consumable(0).unwrap();

        // --- Judgement / The Soul / Wraith ---
        let exp = row("cjok", seed);
        use_from_inventory(&mut run, "c_judgement", &[]);
        use_from_inventory(&mut run, "c_soul", &[]);
        use_from_inventory(&mut run, "c_wraith", &[]);
        for (i, want) in exp[2..5].iter().enumerate() {
            let (wkey, wed) = want.split_once(':').unwrap();
            let j = &run.jokers()[i];
            assert_eq!(j.id.key(), wkey, "{seed} joker {i}");
            assert_eq!(j.edition, edition_from(wed), "{seed} joker {i} edition");
        }
        assert_eq!(run.dollars(), 0, "{seed} Wraith wipes money");
        run.debug_set_dollars(1000);
        for _ in 0..3 {
            run.sell_joker(0).unwrap();
        }

        // --- Wheel of Fortune x6, one eligible joker each ---
        let exp = row("cwheel", seed);
        for want in exp[2].split(',') {
            let (hit, ed) = want.split_once(':').unwrap();
            run.debug_add_joker(JokerId::Joker, Edition::None);
            use_from_inventory(&mut run, "c_wheel_of_fortune", &[]);
            let got = run.jokers()[0].edition;
            if hit == "hit" {
                assert_eq!(got, edition_from(ed), "{seed} wheel hit");
            } else {
                assert_eq!(got, Edition::None, "{seed} wheel miss");
            }
            run.sell_joker(0).unwrap();
        }

        // --- Ankh x3 over three jokers ---
        let exp = row("cankh", seed);
        let trio = [JokerId::Joker, JokerId::GreedyJoker, JokerId::LustyJoker];
        for pick in exp[2].split(',') {
            let pick: usize = pick.parse().unwrap();
            for id in trio {
                run.debug_add_joker(id, Edition::None);
            }
            use_from_inventory(&mut run, "c_ankh", &[]);
            assert_eq!(run.jokers().len(), 2, "{seed} ankh leaves chosen + copy");
            assert_eq!(run.jokers()[0].id, trio[pick - 1], "{seed} ankh pick");
            assert_eq!(run.jokers()[1].id, trio[pick - 1], "{seed} ankh copy");
            run.sell_joker(0).unwrap();
            run.sell_joker(0).unwrap();
        }

        // --- Hex x3 over three editionless jokers ---
        let exp = row("chex", seed);
        for pick in exp[2].split(',') {
            let pick: usize = pick.parse().unwrap();
            for id in trio {
                run.debug_add_joker(id, Edition::None);
            }
            use_from_inventory(&mut run, "c_hex", &[]);
            assert_eq!(run.jokers().len(), 1, "{seed} hex destroys the rest");
            assert_eq!(run.jokers()[0].id, trio[pick - 1], "{seed} hex pick");
            assert_eq!(run.jokers()[0].edition, Edition::Polychrome);
            run.sell_joker(0).unwrap();
        }

        // --- into the round for the hand-targeted group ---
        run.select_blind().unwrap();
        assert_eq!(run.hand().len(), 8);

        // Sigil x4.
        let exp = row("csigil", seed);
        for want in exp[2].split(',') {
            use_from_inventory(&mut run, "c_sigil", &[]);
            for c in run.hand() {
                assert_eq!(c.suit.key(), want.chars().next().unwrap(), "{seed} sigil");
            }
        }
        // Ouija x4.
        let exp = row("couija", seed);
        for want in exp[2].split(',') {
            use_from_inventory(&mut run, "c_ouija", &[]);
            for c in run.hand() {
                assert_eq!(c.rank.key(), want.chars().next().unwrap(), "{seed} ouija");
            }
        }
        // Aura x3 on targets 0/1/2.
        let exp = row("caura", seed);
        for (t, want) in exp[2].split(',').enumerate() {
            use_from_inventory(&mut run, "c_aura", &[t]);
            assert_eq!(run.hand()[t].edition, edition_from(want), "{seed} aura {t}");
        }

        // Familiar (8 -> 10), Grim (10 -> 11), Incantation (11 -> 14).
        for (kind, key, extra) in [
            ("cfam", "c_familiar", 3usize),
            ("cgrim", "c_grim", 2),
            ("cinc", "c_incantation", 4),
        ] {
            let exp = row(kind, seed);
            let didx: usize = exp[2].parse().unwrap();
            let before = sorted_hand_ids(&run);
            use_from_inventory(&mut run, key, &[]);
            let after: std::collections::HashSet<u32> =
                run.hand().iter().map(|c| c.sort_id).collect();
            let destroyed: Vec<usize> = before
                .iter()
                .enumerate()
                .filter(|(_, id)| !after.contains(id))
                .map(|(i, _)| i + 1)
                .collect();
            assert_eq!(destroyed, [didx], "{seed} {key} destroy pick");
            let tail = &run.hand()[run.hand().len() - extra..];
            let want: Vec<&str> = exp[3].split(';').collect();
            for (c, w) in tail.iter().zip(want) {
                let parts: Vec<&str> = w.split(':').collect();
                assert_eq!(
                    c.rank.key(),
                    parts[0].chars().next().unwrap(),
                    "{seed} {key} rank"
                );
                assert_eq!(
                    c.suit.key(),
                    parts[1].chars().next().unwrap(),
                    "{seed} {key} suit"
                );
                assert_eq!(
                    c.enhancement,
                    enhancement_from(parts[2]),
                    "{seed} {key} cen"
                );
            }
        }

        // Immolate: 5 destroyed out of the 14-card hand.
        let exp = row("cimmo", seed);
        let want: std::collections::HashSet<usize> =
            exp[2].split(',').map(|s| s.parse().unwrap()).collect();
        let before = sorted_hand_ids(&run);
        assert_eq!(before.len(), 14);
        let dollars_before = run.dollars();
        use_from_inventory(&mut run, "c_immolate", &[]);
        assert_eq!(run.dollars(), dollars_before + 20, "{seed} immolate $20");
        let after: std::collections::HashSet<u32> = run.hand().iter().map(|c| c.sort_id).collect();
        let destroyed: std::collections::HashSet<usize> = before
            .iter()
            .enumerate()
            .filter(|(_, id)| !after.contains(id))
            .map(|(i, _)| i + 1)
            .collect();
        assert_eq!(destroyed, want, "{seed} immolate picks");

        // Ectoplasm x3 over two editionless jokers.
        let exp = row("cecto", seed);
        for pick in exp[2].split(',') {
            let pick: usize = pick.parse().unwrap();
            run.debug_add_joker(JokerId::Joker, Edition::None);
            run.debug_add_joker(JokerId::GreedyJoker, Edition::None);
            let hand_size_before = run.hand_size();
            use_from_inventory(&mut run, "c_ectoplasm", &[]);
            let negatives: Vec<usize> = run
                .jokers()
                .iter()
                .enumerate()
                .filter(|(_, j)| j.edition == Edition::Negative)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(negatives, [pick - 1], "{seed} ectoplasm pick");
            assert!(run.hand_size() < hand_size_before, "{seed} ecto hand size");
            run.sell_joker(0).unwrap();
            run.sell_joker(0).unwrap();
        }
    }
}
