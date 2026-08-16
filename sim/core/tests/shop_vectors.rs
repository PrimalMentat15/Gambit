//! Oracle vectors for the P3b shop/booster/voucher/tag streams: a scripted
//! no-buy run (open both packs skipping everything, one reroll per shop) is
//! replayed against tools/gen_shop_vectors.lua's output, which executes the
//! game's own get_current_pool/poll_edition/get_pack/get_next_voucher_key/
//! get_next_tag_key verbatim.
//!
//! The driver levels every poker hand sky-high at run start (stream-free)
//! so any 5-card play clears any blind; gameplay streams ('nr', 'hook',
//! 'wheel', ...) are per-key independent of the shop streams, so the shop
//! contents must match the harness exactly.

use balatro_core::cards::{Card, Edition, Enhancement, HandType, Seal};
use balatro_core::items::consumable_by_key;
use balatro_core::run::{Run, State};
use balatro_core::shop::{PackItem, ShopItem, ShopItemKind};

const DATA: &str = include_str!("data/shop_vectors.tsv");

fn edition_str(e: Edition) -> &'static str {
    match e {
        Edition::None => "none",
        Edition::Foil => "foil",
        Edition::Holo => "holo",
        Edition::Polychrome => "polychrome",
        Edition::Negative => "negative",
    }
}

fn seal_str(s: Seal) -> &'static str {
    match s {
        Seal::None => "none",
        Seal::Gold => "Gold",
        Seal::Red => "Red",
        Seal::Blue => "Blue",
        Seal::Purple => "Purple",
    }
}

fn enhancement_key(e: Enhancement) -> &'static str {
    match e {
        Enhancement::None => "c_base",
        Enhancement::Bonus => "m_bonus",
        Enhancement::Mult => "m_mult",
        Enhancement::Wild => "m_wild",
        Enhancement::Glass => "m_glass",
        Enhancement::Steel => "m_steel",
        Enhancement::Stone => "m_stone",
        Enhancement::Gold => "m_gold",
        Enhancement::Lucky => "m_lucky",
    }
}

fn front_key(c: &Card) -> String {
    format!("{}_{}", c.suit.key(), c.rank.key())
}

/// The harness' item_str encoding for a shop card slot.
fn shop_item_str(run: &Run, item: &ShopItem) -> String {
    match &item.kind {
        ShopItemKind::Joker(j) => format!("Joker:{}:{}", j.id.key(), edition_str(j.edition)),
        ShopItemKind::Consumable(c) => {
            let (_, set) = consumable_by_key(c.key).unwrap();
            format!("{}:{}:none", set.type_str(), c.key)
        }
        ShopItemKind::PlayingCard(c) => {
            let _ = run;
            format!(
                "{}:{}:{}:{}",
                if c.enhancement == Enhancement::None {
                    "Base"
                } else {
                    "Enhanced"
                },
                enhancement_key(c.enhancement),
                edition_str(c.edition),
                front_key(c)
            )
        }
    }
}

/// The harness' item_str for a pack card.
fn pack_item_str(item: &PackItem) -> String {
    match item {
        PackItem::Joker(j) => format!("Joker:{}:{}", j.id.key(), edition_str(j.edition)),
        PackItem::Consumable(c) => {
            let (_, set) = consumable_by_key(c.key).unwrap();
            format!("{}:{}:none", set.type_str(), c.key)
        }
        PackItem::PlayingCard(c) => format!(
            "{}:{}:{}:{}:{}",
            if c.enhancement == Enhancement::None {
                "Base"
            } else {
                "Enhanced"
            },
            enhancement_key(c.enhancement),
            edition_str(c.edition),
            front_key(c),
            seal_str(c.seal)
        ),
    }
}

fn current_shop_str(run: &Run) -> String {
    let shop = run.shop().expect("shop");
    shop.jokers
        .iter()
        .map(|i| shop_item_str(run, i))
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Default)]
struct Expected {
    /// ante -> voucher key
    vouchers: Vec<(i64, String)>,
    /// ante -> (small, big)
    tags: Vec<(i64, String, String)>,
    /// shop_idx -> (ante, slots, packs)
    shops: Vec<(i64, String, String)>,
    /// (shop_idx, pack_slot) -> (key, contents)
    packs: Vec<(usize, usize, String, String)>,
    /// shop_idx -> slots after reroll
    rerolls: Vec<String>,
}

fn parse(prefix: &str, seed: &str) -> Expected {
    let mut e = Expected::default();
    for line in DATA.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 2 || f[1] != seed {
            continue;
        }
        let Some(kind) = f[0].strip_prefix(prefix) else {
            continue;
        };
        match kind {
            "voucher" => e.vouchers.push((f[2].parse().unwrap(), f[3].to_string())),
            "tags" => e
                .tags
                .push((f[2].parse().unwrap(), f[3].to_string(), f[4].to_string())),
            "shop" => e
                .shops
                .push((f[3].parse().unwrap(), f[4].to_string(), f[5].to_string())),
            "pack" => e.packs.push((
                f[2].parse::<usize>().unwrap(),
                f[3].parse::<usize>().unwrap(),
                f[4].to_string(),
                f[5].to_string(),
            )),
            "reroll" => e.rerolls.push(f[3].to_string()),
            _ => {}
        }
    }
    e
}

#[test]
fn scripted_run_matches_lua_shop_streams() {
    let seeds = [
        "TUTORIAL", "AAAAAAAA", "7NLLGSMA", "XEQH7CP9", "OOPS1234", "P3BSHOPS",
    ];
    let mut asserts = 0usize;
    for seed in seeds {
        asserts += scripted_run("", seed, &[]);
    }
    // ~600 comparisons of full shop/pack contents across 6 seeds x 11 shops.
    assert!(asserts >= 400, "only {asserts} assertions ran");
}

/// Scenario 2: shops with Tarot/Planet Merchant, Hone, Magic Trick,
/// Illusion, Overstock, Crystal Ball and Omen Globe active — exercises the
/// 'illusion', 'Enhancedsho', 'frontsho', 'omen_globe' and 'Spectralar2'
/// streams plus the 3-slot Overstock shop and boosted rates.
#[test]
fn voucher_modified_shops_match_lua_streams() {
    let vouchers = [
        "v_tarot_merchant",
        "v_planet_merchant",
        "v_hone",
        "v_magic_trick",
        "v_illusion",
        "v_overstock_norm",
        "v_crystal_ball",
        "v_omen_globe",
    ];
    let mut asserts = 0usize;
    for seed in ["TUTORIAL", "AAAAAAAA", "VOUCHER1"] {
        asserts += scripted_run("v", seed, &vouchers);
    }
    assert!(asserts >= 150, "only {asserts} assertions ran");
}

fn scripted_run(prefix: &str, seed: &str, vouchers: &[&'static str]) -> usize {
    {
        let exp = parse(prefix, seed);
        assert!(!exp.shops.is_empty(), "no vectors for {prefix}{seed}");
        let mut asserts = 0usize;

        let mut run = Run::new(seed);
        // Stream-free hand pumping: any 5-card play beats any blind.
        for ht in HandType::ALL {
            run.level_up_hand(ht, 5000);
        }
        // Scenario vouchers granted right after the run-start rolls (the
        // harness mirrors this ordering).
        for v in vouchers {
            run.debug_apply_voucher(v);
        }

        // Run-start voucher and blind tags (ante 1 rows).
        let mut vi = 0usize; // voucher row cursor
        let mut ti = 0usize; // tag row cursor
        assert_eq!(exp.vouchers[vi].0, 1);
        assert_eq!(
            run.current_voucher().unwrap(),
            exp.vouchers[vi].1,
            "{seed} run-start voucher"
        );
        vi += 1;
        assert_eq!(
            (run.blind_tags().0, run.blind_tags().1),
            (exp.tags[ti].1.as_str(), exp.tags[ti].2.as_str()),
            "{seed} ante-1 tags"
        );
        ti += 1;
        asserts += 2;

        let mut shop_idx = 0usize;
        let mut seen_ante = 1i64;
        while shop_idx < exp.shops.len() {
            match run.state() {
                State::BlindSelect => run.select_blind().unwrap(),
                State::SelectingHand => {
                    run.play(&[0, 1, 2, 3, 4]).unwrap();
                }
                State::RoundEval => run.cash_out().unwrap(),
                State::Shop => {
                    let (ante, slots, packs) = &exp.shops[shop_idx];
                    assert_eq!(run.ante(), *ante, "{seed} shop {shop_idx} ante");
                    if *ante > seen_ante {
                        // New ante: boss-defeat voucher + tag rolls happened.
                        seen_ante = *ante;
                        assert_eq!(exp.vouchers[vi].0, *ante);
                        assert_eq!(
                            run.current_voucher().unwrap(),
                            exp.vouchers[vi].1,
                            "{seed} ante {ante} voucher"
                        );
                        vi += 1;
                        assert_eq!(exp.tags[ti].0, *ante);
                        assert_eq!(
                            (run.blind_tags().0, run.blind_tags().1),
                            (exp.tags[ti].1.as_str(), exp.tags[ti].2.as_str()),
                            "{seed} ante {ante} tags"
                        );
                        ti += 1;
                        asserts += 2;
                    }
                    assert_eq!(
                        &current_shop_str(&run),
                        slots,
                        "{seed} shop {shop_idx} slots"
                    );
                    let got_packs: Vec<&str> =
                        run.shop().unwrap().packs.iter().map(|p| p.key).collect();
                    assert_eq!(&got_packs.join(","), packs, "{seed} shop {shop_idx} packs");
                    asserts += 2;

                    run.debug_set_dollars(100_000);
                    for slot in 0..2usize {
                        let (pk_shop, _pk_slot, pk_key, pk_contents) = exp
                            .packs
                            .iter()
                            .find(|(s, sl, _, _)| *s == shop_idx + 1 && *sl == slot + 1)
                            .expect("pack row");
                        assert_eq!(*pk_shop, shop_idx + 1);
                        run.buy_pack(slot).unwrap();
                        let got: Vec<String> = run
                            .pack()
                            .unwrap()
                            .items
                            .iter()
                            .map(pack_item_str)
                            .collect();
                        assert_eq!(run.pack().unwrap().key, pk_key.as_str());
                        assert_eq!(
                            &got.join(","),
                            pk_contents,
                            "{seed} shop {} pack {} ({})",
                            shop_idx + 1,
                            slot + 1,
                            pk_key
                        );
                        asserts += 2;
                        run.skip_pack().unwrap();
                    }

                    run.reroll_shop().unwrap();
                    assert_eq!(
                        &current_shop_str(&run),
                        &exp.rerolls[shop_idx],
                        "{seed} shop {shop_idx} reroll"
                    );
                    asserts += 1;

                    run.leave_shop().unwrap();
                    shop_idx += 1;
                }
                s => panic!("unexpected state {s:?}"),
            }
        }
        asserts
    }
}
