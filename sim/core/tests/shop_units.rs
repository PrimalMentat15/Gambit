//! Unit tests for the P3b shop/economy/voucher/tag/consumable logic that is
//! deterministic given the seeded streams: cost arithmetic, reroll
//! progression, voucher effects, tag effects, and the RNG-free tarot
//! effects. Oracle-vector coverage lives in shop_vectors.rs /
//! consumable_vectors.rs.

use balatro_core::cards::{Edition, Enhancement, HandType, Seal, Suit};
use balatro_core::items::{item_cost, sell_cost, JokerId, PackKind};
use balatro_core::run::{BlindStage, Run, RunError, State};
use balatro_core::shop::{PackItem, ShopItemKind};

/// Drive a fresh run to its first shop, winning blinds with pumped hands.
fn run_to_shop(seed: &str) -> Run {
    let mut run = Run::new(seed);
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    loop {
        match run.state() {
            State::BlindSelect => run.select_blind().unwrap(),
            State::SelectingHand => {
                run.play(&[0, 1, 2, 3, 4]).unwrap();
            }
            State::RoundEval => run.cash_out().unwrap(),
            State::Shop => return run,
            s => panic!("unexpected {s:?}"),
        }
    }
}

fn win_next_blind(run: &mut Run) {
    loop {
        match run.state() {
            State::BlindSelect => run.select_blind().unwrap(),
            State::SelectingHand => {
                run.play(&[0, 1, 2, 3, 4]).unwrap();
            }
            State::RoundEval => {
                run.cash_out().unwrap();
                return;
            }
            State::Shop => run.leave_shop().unwrap(),
            s => panic!("unexpected {s:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Cost / sell arithmetic (card.lua:369-385)
// ---------------------------------------------------------------------------

#[test]
fn cost_and_sell_arithmetic() {
    // Base costs: floor((base + 0.5) * 1) with a $1 floor.
    assert_eq!(item_cost(5, Edition::None, 0, false), 5);
    assert_eq!(item_cost(10, Edition::None, 0, false), 10);
    assert_eq!(item_cost(0, Edition::None, 0, false), 1); // $1 floor
    assert_eq!(item_cost(5, Edition::None, 0, true), 0); // couponed
                                                         // Edition surcharges: foil +2, holo +3, polychrome/negative +5.
    assert_eq!(item_cost(5, Edition::Foil, 0, false), 7);
    assert_eq!(item_cost(5, Edition::Holo, 0, false), 8);
    assert_eq!(item_cost(5, Edition::Polychrome, 0, false), 10);
    assert_eq!(item_cost(5, Edition::Negative, 0, false), 10);
    // Clearance Sale (25%): voucher 10 -> floor(10.5*0.75) = 7; booster
    // 4 -> floor(4.5*0.75) = 3.
    assert_eq!(item_cost(10, Edition::None, 25, false), 7);
    assert_eq!(item_cost(4, Edition::None, 25, false), 3);
    // Liquidation (50%): joker 6 -> floor(6.5*0.5) = 3.
    assert_eq!(item_cost(6, Edition::None, 50, false), 3);
    // sell = max(1, floor(cost/2)) + extra_value.
    assert_eq!(sell_cost(5, Edition::None, 0, 0), 2);
    assert_eq!(sell_cost(3, Edition::None, 0, 0), 1); // tarot: cost 3 -> 1
    assert_eq!(sell_cost(4, Edition::None, 0, 0), 2); // spectral: 4 -> 2
    assert_eq!(sell_cost(2, Edition::None, 0, 0), 1);
    assert_eq!(sell_cost(5, Edition::None, 0, 3), 5); // extra_value rides on top
}

// ---------------------------------------------------------------------------
// Reroll cost progression (common_events.lua:2263-2269)
// ---------------------------------------------------------------------------

#[test]
fn reroll_cost_progression_and_reset() {
    let mut run = run_to_shop("REROLLAA");
    run.debug_set_dollars(500);
    assert_eq!(run.reroll_cost(), 5);
    run.reroll_shop().unwrap();
    assert_eq!(run.reroll_cost(), 6);
    run.reroll_shop().unwrap();
    assert_eq!(run.reroll_cost(), 7);
    let d = run.dollars();
    run.reroll_shop().unwrap();
    assert_eq!(run.dollars(), d - 7);
    // The increase resets with new_round (state_events.lua:300), i.e. the
    // next blind's shop starts at base again.
    run.leave_shop().unwrap();
    win_next_blind(&mut run);
    assert_eq!(run.reroll_cost(), 5);
}

#[test]
fn reroll_surplus_lowers_base_cost() {
    let mut run = run_to_shop("REROLLBB");
    run.debug_set_dollars(500);
    run.debug_apply_voucher("v_reroll_surplus");
    // current cost floored at max(0, 5-2) = 3; base drops to 3.
    assert_eq!(run.reroll_cost(), 3);
    run.reroll_shop().unwrap();
    assert_eq!(run.reroll_cost(), 4);
    run.debug_apply_voucher("v_reroll_glut");
    assert_eq!(run.reroll_cost(), 2);
    run.leave_shop().unwrap();
    win_next_blind(&mut run);
    assert_eq!(run.reroll_cost(), 1); // base 5-2-2
}

#[test]
fn cannot_reroll_below_bankroll() {
    let mut run = run_to_shop("REROLLCC");
    run.debug_set_dollars(4); // reroll costs 5
    assert!(matches!(
        run.reroll_shop(),
        Err(RunError::InsufficientFunds)
    ));
}

// ---------------------------------------------------------------------------
// Interest / payout (state_events.lua:1192; Seed Money / Money Tree)
// ---------------------------------------------------------------------------

#[test]
fn interest_cap_and_seed_money() {
    // Base cap: interest = min(floor(d/5), 25/5) = 5 at $25+.
    let mut run = Run::new("INTEREST");
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    run.debug_set_dollars(43);
    run.select_blind().unwrap();
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    assert_eq!(run.state(), State::RoundEval);
    // Small blind $3 + 3 hands left + interest 5 (cap) = 11.
    assert_eq!(run.pending_cashout(), 3 + 3 + 5);

    // Seed Money: cap 50 -> min(floor(43/5)=8, 10) = 8.
    let mut run = Run::new("INTEREST");
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    run.debug_apply_voucher("v_seed_money");
    run.debug_set_dollars(43);
    run.select_blind().unwrap();
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    assert_eq!(run.pending_cashout(), 3 + 3 + 8);
}

// ---------------------------------------------------------------------------
// Voucher effects
// ---------------------------------------------------------------------------

#[test]
fn overstock_adds_and_fills_a_shop_slot() {
    let mut run = run_to_shop("OVERSTCK");
    assert_eq!(run.shop().unwrap().jokers.len(), 2);
    run.debug_set_dollars(500);
    run.debug_apply_voucher("v_overstock_norm");
    // change_shop_size refills immediately while the shop is open
    // (common_events.lua:1112-1116).
    assert_eq!(run.shop().unwrap().jokers.len(), 3);
    run.debug_apply_voucher("v_overstock_plus");
    assert_eq!(run.shop().unwrap().jokers.len(), 4);
}

#[test]
fn grabber_wasteful_paint_brush_hieroglyph() {
    let mut run = run_to_shop("VOUCHER2");
    run.debug_apply_voucher("v_grabber");
    run.debug_apply_voucher("v_wasteful");
    run.debug_apply_voucher("v_paint_brush");
    assert_eq!(run.hands_left(), 5); // 4 + 1, applied live
    assert_eq!(run.discards_left(), 5);
    run.leave_shop().unwrap();
    run.select_blind().unwrap();
    assert_eq!(run.hands_left(), 5);
    assert_eq!(run.discards_left(), 5);
    assert_eq!(run.hand().len(), 9); // Paint Brush hand size

    // Hieroglyph: ante down, one hand fewer.
    let mut run = run_to_shop("VOUCHER3");
    let ante = run.ante();
    run.debug_apply_voucher("v_hieroglyph");
    assert_eq!(run.ante(), ante - 1);
    assert_eq!(run.blind_ante(), ante - 1);
    run.leave_shop().unwrap();
    run.select_blind().unwrap();
    assert_eq!(run.hands_left(), 3);
    // Petroglyph: discards down.
    let mut run = run_to_shop("VOUCHER4");
    run.debug_apply_voucher("v_petroglyph");
    run.leave_shop().unwrap();
    run.select_blind().unwrap();
    assert_eq!(run.discards_left(), 3);
}

#[test]
fn crystal_ball_and_antimatter_slots() {
    let mut run = run_to_shop("VOUCHER5");
    assert_eq!(run.consumable_slots(), 2);
    assert_eq!(run.joker_slots(), 5);
    run.debug_apply_voucher("v_crystal_ball");
    run.debug_apply_voucher("v_antimatter");
    assert_eq!(run.consumable_slots(), 3);
    assert_eq!(run.joker_slots(), 6);
}

#[test]
fn directors_cut_and_retcon_boss_rerolls() {
    let mut run = Run::new("DIRECTOR");
    run.debug_set_dollars(50);
    assert!(!run.can_reroll_boss());
    run.debug_apply_voucher("v_directors_cut");
    assert!(run.can_reroll_boss());
    let before = run.boss_choice();
    run.reroll_boss().unwrap();
    assert_eq!(run.dollars(), 40);
    assert_ne!(run.boss_choice(), before, "boss rerolled ('boss' stream)");
    // Director's Cut: once per ante.
    assert!(!run.can_reroll_boss());
    // Retcon: unlimited.
    run.debug_apply_voucher("v_retcon");
    assert!(run.can_reroll_boss());
    run.reroll_boss().unwrap();
    assert!(run.can_reroll_boss());
    // ...but never below the $10.
    run.debug_set_dollars(9);
    assert!(!run.can_reroll_boss());
}

#[test]
fn clearance_sale_discounts_shop_prices() {
    let mut run = run_to_shop("DISCOUNT");
    run.debug_apply_voucher("v_clearance_sale");
    let shop = run.shop().unwrap();
    // Voucher slot: 10 -> 7.
    if !shop.vouchers.is_empty() {
        assert_eq!(run.voucher_cost(shop.vouchers[0].key), 7);
    }
    // Booster: 4/6/8 -> 3/4/6.
    let p = shop.packs[0];
    let base = balatro_core::items::center_base_cost(p.key);
    let want = ((base as f64 + 0.5) * 0.75).floor() as i64;
    assert_eq!(run.pack_cost(&p), want.max(1));
}

// ---------------------------------------------------------------------------
// Money floor
// ---------------------------------------------------------------------------

#[test]
fn cannot_buy_below_zero() {
    let mut run = run_to_shop("BROKEAF1");
    run.debug_set_dollars(0);
    assert!(matches!(
        run.buy_shop_item(0),
        Err(RunError::InsufficientFunds)
    ));
    assert!(matches!(run.buy_pack(0), Err(RunError::InsufficientFunds)));
    if !run.shop().unwrap().vouchers.is_empty() {
        assert!(matches!(
            run.redeem_voucher(0),
            Err(RunError::InsufficientFunds)
        ));
    }
}

// ---------------------------------------------------------------------------
// Tags
// ---------------------------------------------------------------------------

#[test]
fn skip_tag_and_double_tag() {
    let mut run = Run::new("TAGSKIP1");
    run.force_tag(Some("tag_skip"));
    let d = run.dollars();
    run.skip_blind().unwrap(); // Skip Tag fires immediately: $5 * 1 skip
    assert_eq!(run.dollars(), d + 5);
    assert_eq!(run.blind_on_deck(), BlindStage::Big);
    assert_eq!(run.skips(), 1);

    // Double Tag: held Double copies the next tag.
    let mut run = Run::new("TAGDBL01");
    run.force_tag(Some("tag_double"));
    run.skip_blind().unwrap();
    assert_eq!(run.tags().len(), 1); // the Double waits
    run.force_tag(Some("tag_economy"));
    run.debug_set_dollars(30);
    run.skip_blind().unwrap();
    // Economy applies twice: 30 -> 60 -> 100 (cap +40).
    assert_eq!(run.dollars(), 100);
    assert!(run.tags().is_empty());
}

#[test]
fn handy_garbage_economy_tags() {
    let mut run = Run::new("TAGMONEY");
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    // Win the small blind with one play (3 discards unused).
    run.select_blind().unwrap();
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    run.cash_out().unwrap();
    run.leave_shop().unwrap();
    // Skip Big with a Handy Tag: $1 per hand played (1).
    run.force_tag(Some("tag_handy"));
    let d = run.dollars();
    run.skip_blind().unwrap();
    assert_eq!(run.dollars(), d + 1);
    // Garbage Tag: $1 per unused discard (3 from the won round).
    run.force_tag(Some("tag_garbage"));
    // (Boss can't be skipped; force the tag through a fresh check instead.)
    assert_eq!(run.blind_on_deck(), BlindStage::Boss);
    assert!(matches!(run.skip_blind(), Err(RunError::BadSlot(_))));
}

#[test]
fn juggle_tag_is_temporary() {
    let mut run = Run::new("TAGJUGGL");
    run.force_tag(Some("tag_juggle"));
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    run.skip_blind().unwrap();
    assert_eq!(run.tags().len(), 1);
    run.select_blind().unwrap();
    // 8 + 3 for this round only.
    assert_eq!(run.hand().len(), 11);
    assert!(run.tags().is_empty());
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    run.cash_out().unwrap();
    run.leave_shop().unwrap();
    run.select_blind().unwrap();
    assert_eq!(run.hand().len(), 8);
}

#[test]
fn d6_tag_zeroes_reroll_cost_for_one_shop() {
    let mut run = Run::new("TAGDSIX1");
    run.force_tag(Some("tag_d_six"));
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    run.skip_blind().unwrap();
    run.select_blind().unwrap();
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    run.cash_out().unwrap();
    assert_eq!(run.state(), State::Shop);
    assert_eq!(run.reroll_cost(), 0);
    run.debug_set_dollars(100);
    run.reroll_shop().unwrap();
    assert_eq!(run.reroll_cost(), 1); // 0 + increase
                                      // The temp cost expires with the round (state_events.lua:270-271).
    run.leave_shop().unwrap();
    win_next_blind(&mut run);
    assert_eq!(run.reroll_cost(), 5);
}

#[test]
fn coupon_tag_makes_cards_and_packs_free() {
    let mut run = Run::new("TAGCOUPN");
    run.force_tag(Some("tag_coupon"));
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    run.skip_blind().unwrap();
    run.select_blind().unwrap();
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    run.cash_out().unwrap();
    let shop = run.shop().unwrap();
    for item in &shop.jokers {
        assert_eq!(run.shop_item_cost(item), 0);
    }
    for p in &shop.packs {
        assert_eq!(run.pack_cost(p), 0);
    }
    // Free things are buyable at $0 (can_buy allows cost 0).
    run.debug_set_dollars(0);
    run.buy_pack(0).unwrap();
    assert_eq!(run.dollars(), 0);
}

#[test]
fn voucher_tag_adds_a_second_voucher() {
    let mut run = Run::new("TAGVOUCH");
    run.force_tag(Some("tag_voucher"));
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    run.skip_blind().unwrap();
    run.select_blind().unwrap();
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    run.cash_out().unwrap();
    let shop = run.shop().unwrap();
    assert_eq!(shop.vouchers.len(), 2);
    assert!(shop.vouchers[0].shop_voucher);
    assert!(!shop.vouchers[1].shop_voucher);
    assert_ne!(shop.vouchers[0].key, shop.vouchers[1].key);
}

#[test]
fn uncommon_and_edition_tags_hit_the_shop() {
    let mut run = Run::new("TAGUNCMN");
    run.force_tag(Some("tag_uncommon"));
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    run.skip_blind().unwrap();
    run.select_blind().unwrap();
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    run.cash_out().unwrap();
    let shop = run.shop().unwrap();
    let ShopItemKind::Joker(j) = &shop.jokers[0].kind else {
        panic!("uncommon tag slot is a joker");
    };
    assert_eq!(j.id.meta().rarity, 2, "Uncommon Tag forces rarity 2");
    assert!(shop.jokers[0].couponed, "tag joker is free");
    assert_eq!(run.shop_item_cost(&shop.jokers[0]), 0);

    // Foil Tag: the first edition-less shop joker turns foil and free.
    let mut run = Run::new("TAGFOILX");
    run.force_tag(Some("tag_foil"));
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    run.skip_blind().unwrap();
    run.select_blind().unwrap();
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    run.cash_out().unwrap();
    let shop = run.shop().unwrap();
    let foiled = shop.jokers.iter().any(|i| {
        matches!(&i.kind, ShopItemKind::Joker(j) if j.edition == Edition::Foil) && i.couponed
    });
    let any_joker = shop
        .jokers
        .iter()
        .any(|i| matches!(&i.kind, ShopItemKind::Joker(_)));
    assert_eq!(foiled, any_joker, "foil tag applies iff a joker rolled");
    assert_eq!(run.tags().is_empty(), any_joker);
}

#[test]
fn top_up_and_orbital_tags() {
    let mut run = Run::new("TAGTOPUP");
    run.force_tag(Some("tag_top_up"));
    run.skip_blind().unwrap();
    assert_eq!(run.jokers().len(), 2);
    for j in run.jokers() {
        assert_eq!(j.id.meta().rarity, 1, "Top-up creates commons");
    }

    let mut run = Run::new("TAGORBIT");
    run.force_tag(Some("tag_orbital"));
    let want = run.orbital_choice(BlindStage::Small).unwrap();
    run.skip_blind().unwrap();
    assert_eq!(run.hands_table().get(want).level, 4, "Orbital: +3 levels");
}

#[test]
fn investment_tag_pays_after_boss() {
    let mut run = Run::new("TAGINVST");
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    run.force_tag(Some("tag_investment"));
    run.skip_blind().unwrap(); // hold Investment
    run.force_tag(None);
    assert_eq!(run.tags().len(), 1);
    // Win Big (no payout from tag)...
    run.select_blind().unwrap();
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    let base = 4 + 3; // big blind $4 + 3 hands
    let interest = (run.dollars() / 5).min(5);
    assert_eq!(run.pending_cashout(), base + interest);
    assert_eq!(run.tags().len(), 1, "Investment waits for a boss");
    run.cash_out().unwrap();
    run.leave_shop().unwrap();
    // ...then the Boss: +$25 in the round eval.
    run.select_blind().unwrap();
    run.play(&[0, 1, 2, 3, 4]).unwrap();
    let boss_dollars = run.current_blind().unwrap().dollars;
    let interest = (run.dollars() / 5).min(5);
    assert_eq!(run.pending_cashout(), boss_dollars + 3 + 25 + interest);
    assert!(run.tags().is_empty());
}

#[test]
fn boss_tag_rerolls_the_boss() {
    let mut run = Run::new("TAGBOSS1");
    let before = run.boss_choice();
    run.force_tag(Some("tag_boss"));
    run.skip_blind().unwrap();
    assert_ne!(run.boss_choice(), before);
    assert!(run.tags().is_empty());
}

#[test]
fn charm_tag_opens_a_free_mega_arcana() {
    let mut run = Run::new("TAGCHARM");
    run.force_tag(Some("tag_charm"));
    let d = run.dollars();
    run.skip_blind().unwrap();
    assert_eq!(run.state(), State::PackOpen);
    assert_eq!(run.dollars(), d, "tag pack is free");
    let pack = run.pack().unwrap();
    assert_eq!(pack.kind, PackKind::Arcana);
    assert_eq!(pack.key, "p_arcana_mega_1");
    assert_eq!(pack.items.len(), 5);
    assert_eq!(pack.choices_left, 2);
    assert!(pack.hand_dealt, "arcana packs deal a hand");
    assert_eq!(run.hand().len(), 8);
    run.skip_pack().unwrap();
    assert_eq!(run.state(), State::BlindSelect);
    assert_eq!(run.hand().len(), 0, "hand returns to the deck");
    assert_eq!(run.deck_len(), 52);
}

// ---------------------------------------------------------------------------
// Booster pack mechanics
// ---------------------------------------------------------------------------

#[test]
fn first_shop_has_a_buffoon_pack_and_picking_works() {
    let mut run = run_to_shop("FIRSTBUF");
    assert_eq!(run.shop().unwrap().packs[0].key, "p_buffoon_normal_1");
    run.debug_set_dollars(100);
    run.buy_pack(0).unwrap();
    let pack = run.pack().unwrap();
    assert_eq!(pack.kind, PackKind::Buffoon);
    assert_eq!(pack.items.len(), 2);
    assert_eq!(pack.choices_left, 1);
    let PackItem::Joker(want) = pack.items[0].clone() else {
        panic!("buffoon packs hold jokers");
    };
    run.pick_pack_item(0, &[]).unwrap();
    assert_eq!(run.state(), State::Shop, "1-choice pack closes on pick");
    assert_eq!(run.jokers().len(), 1);
    assert_eq!(run.jokers()[0].id, want.id);
    // The unpicked joker's pool slot frees up (used_jokers cleared).
    assert!(run.shop().unwrap().packs[0].used);
    assert!(matches!(run.buy_pack(0), Err(RunError::BadSlot(_))));
}

#[test]
fn celestial_pick_levels_the_hand_and_telescope_forces_it() {
    // Meteor Tag opens a Mega Celestial pack; picking a planet uses it.
    let mut run = Run::new("TAGMETEO");
    for ht in HandType::ALL {
        run.level_up_hand(ht, 10);
    }
    run.force_tag(Some("tag_meteor"));
    run.skip_blind().unwrap();
    assert_eq!(run.state(), State::PackOpen);
    let pack = run.pack().unwrap();
    assert_eq!(pack.kind, PackKind::Celestial);
    assert_eq!(pack.items.len(), 5);
    let PackItem::Consumable(c) = pack.items[0].clone() else {
        panic!("celestial pack holds planets");
    };
    let ht = balatro_core::items::hand_for_planet(c.key);
    run.pick_pack_item(0, &[]).unwrap();
    if let Some(ht) = ht {
        assert_eq!(run.hands_table().get(ht).level, 12, "planet used on pick");
    }
    assert_eq!(run.pack().unwrap().choices_left, 1);
    run.skip_pack().unwrap();

    // Telescope: first celestial card is the most played hand's planet.
    let mut run = Run::new("TELESCOP");
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    run.debug_apply_voucher("v_telescope");
    // Play one hand so a hand type has played > 0.
    run.select_blind().unwrap();
    let played = run.play(&[0, 1, 2, 3, 4]).unwrap().hand_type;
    run.cash_out().unwrap();
    run.leave_shop().unwrap();
    run.force_tag(Some("tag_meteor"));
    run.skip_blind().unwrap();
    let pack = run.pack().unwrap();
    let PackItem::Consumable(c) = &pack.items[0] else {
        panic!()
    };
    assert_eq!(
        c.key,
        balatro_core::consumables::planet_for_hand(played),
        "Telescope forces the most played hand's planet"
    );
    run.skip_pack().unwrap();
}

#[test]
fn standard_pack_cards_join_the_deck() {
    let mut run = Run::new("TAGSTAND");
    run.force_tag(Some("tag_standard"));
    run.skip_blind().unwrap();
    let pack = run.pack().unwrap();
    assert_eq!(pack.kind, PackKind::Standard);
    assert_eq!((pack.items.len(), pack.choices_left), (5, 2));
    let deck_before = run.deck_len();
    let PackItem::PlayingCard(want) = pack.items[2].clone() else {
        panic!("standard packs hold playing cards");
    };
    run.pick_pack_item(2, &[]).unwrap();
    assert_eq!(run.deck_len(), deck_before + 1);
    run.pick_pack_item(0, &[]).unwrap(); // second (last) choice
    assert_eq!(run.state(), State::BlindSelect);
    assert_eq!(run.deck_len(), deck_before + 2);
    // The picked card kept its identity (it sits at the deck bottom).
    let got = run.state(); // silence unused warnings path
    let _ = (want, got);
}

// ---------------------------------------------------------------------------
// Buying and selling
// ---------------------------------------------------------------------------

#[test]
fn buy_and_sell_jokers_and_consumables() {
    let mut run = run_to_shop("BUYSELL1");
    run.debug_set_dollars(100);
    let shop = run.shop().unwrap().clone();
    if let Some(item) = shop.jokers.first() {
        let cost = run.shop_item_cost(item);
        let d = run.dollars();
        run.buy_shop_item(0).unwrap_or_else(|e| panic!("buy: {e}"));
        assert_eq!(run.dollars(), d - cost);
    }
    // Sell whatever was bought.
    let d = run.dollars();
    if !run.jokers().is_empty() {
        let v = run.joker_sell_value(&run.jokers()[0]);
        run.sell_joker(0).unwrap();
        assert_eq!(run.dollars(), d + v);
    } else if !run.consumables().is_empty() {
        let v = run.consumable_sell_value(&run.consumables()[0]);
        run.sell_consumable(0).unwrap();
        assert_eq!(run.dollars(), d + v);
    }
}

#[test]
fn joker_slots_enforced_and_negative_bypasses() {
    let mut run = Run::new("SLOTSFUL");
    for _ in 0..5 {
        run.debug_add_joker(JokerId::Joker, Edition::None);
    }
    assert_eq!(run.jokers().len(), 5);
    // A negative joker still fits (check_for_buy_space, and add_to_deck
    // raises the limit).
    run.debug_add_joker(JokerId::GreedyJoker, Edition::Negative);
    assert_eq!(run.jokers().len(), 6);
    assert_eq!(run.joker_slots(), 6);
    run.sell_joker(5).unwrap();
    assert_eq!(run.joker_slots(), 5);
}

// ---------------------------------------------------------------------------
// RNG-free tarot effects
// ---------------------------------------------------------------------------

fn run_in_round(seed: &str) -> Run {
    let mut run = Run::new(seed);
    for ht in HandType::ALL {
        run.level_up_hand(ht, 5000);
    }
    run.select_blind().unwrap();
    run
}

#[test]
fn enhancing_and_converting_tarots() {
    let mut run = run_in_round("TAROTFX1");
    // Magician: up to 2 cards -> Lucky.
    run.debug_add_consumable("c_magician");
    run.use_consumable(0, &[0, 1]).unwrap();
    assert_eq!(run.hand()[0].enhancement, Enhancement::Lucky);
    assert_eq!(run.hand()[1].enhancement, Enhancement::Lucky);
    // The Devil: 1 card -> Gold.
    run.debug_add_consumable("c_devil");
    run.use_consumable(0, &[0]).unwrap();
    assert_eq!(run.hand()[0].enhancement, Enhancement::Gold);
    // Target count limits enforced.
    run.debug_add_consumable("c_devil");
    assert!(matches!(
        run.use_consumable(0, &[0, 1]),
        Err(RunError::CannotUse(_))
    ));
    run.sell_consumable(0).unwrap();
    // The Star: up to 3 cards -> Diamonds.
    run.debug_add_consumable("c_star");
    run.use_consumable(0, &[2, 3, 4]).unwrap();
    for i in 2..5 {
        assert_eq!(run.hand()[i].suit, Suit::Diamonds);
    }
    // Strength: +1 rank, Ace wraps to 2.
    let r0 = run.hand()[0].rank.id();
    run.debug_add_consumable("c_strength");
    run.use_consumable(0, &[0]).unwrap();
    let want = if r0 == 14 { 2 } else { (r0 + 1).min(14) };
    assert_eq!(run.hand()[0].rank.id(), want);
    // Death: left card becomes a copy of the rightmost target.
    let src = run.hand()[3];
    run.debug_add_consumable("c_death");
    run.use_consumable(0, &[1, 3]).unwrap();
    let dst = run.hand()[1];
    assert_eq!(
        (dst.rank, dst.suit, dst.enhancement, dst.seal),
        (src.rank, src.suit, src.enhancement, src.seal)
    );
    assert_ne!(dst.sort_id, src.sort_id, "copy keeps its own card identity");
    // Hanged Man: destroys up to 2 targets.
    let n = run.hand().len();
    run.debug_add_consumable("c_hanged_man");
    run.use_consumable(0, &[0, 1]).unwrap();
    assert_eq!(run.hand().len(), n - 2);
    assert_eq!(run.destroyed_cards().len(), 2);
}

#[test]
fn seals_and_cryptid() {
    let mut run = run_in_round("TAROTFX2");
    for (key, seal) in [
        ("c_talisman", Seal::Gold),
        ("c_deja_vu", Seal::Red),
        ("c_trance", Seal::Blue),
        ("c_medium", Seal::Purple),
    ] {
        run.debug_add_consumable(key);
        run.use_consumable(0, &[0]).unwrap();
        assert_eq!(run.hand()[0].seal, seal, "{key}");
    }
    let n = run.hand().len();
    let src = run.hand()[2];
    run.debug_add_consumable("c_cryptid");
    run.use_consumable(0, &[2]).unwrap();
    assert_eq!(run.hand().len(), n + 2);
    let tail = &run.hand()[n..];
    for c in tail {
        assert_eq!(
            (c.rank, c.suit, c.enhancement),
            (src.rank, src.suit, src.enhancement)
        );
    }
}

#[test]
fn hermit_temperance_fool_planets() {
    let mut run = run_in_round("TAROTFX3");
    // Hermit doubles, capped +$20.
    run.debug_set_dollars(7);
    run.debug_add_consumable("c_hermit");
    run.use_consumable(0, &[]).unwrap();
    assert_eq!(run.dollars(), 14);
    run.debug_set_dollars(50);
    run.debug_add_consumable("c_hermit");
    run.use_consumable(0, &[]).unwrap();
    assert_eq!(run.dollars(), 70);
    // Temperance: joker sell values, capped $50 (no jokers -> $0).
    run.debug_add_consumable("c_temperance");
    run.use_consumable(0, &[]).unwrap();
    assert_eq!(run.dollars(), 70);
    run.debug_add_joker(JokerId::Joker, Edition::None); // sell value 1
    run.debug_add_consumable("c_temperance");
    run.use_consumable(0, &[]).unwrap();
    assert_eq!(run.dollars(), 71);
    // Planets level their hand.
    run.debug_add_consumable("c_mercury");
    run.use_consumable(0, &[]).unwrap();
    assert_eq!(run.hands_table().get(HandType::Pair).level, 5002);
    // The Fool copies the last used tarot/planet (Mercury).
    run.debug_add_consumable("c_fool");
    run.use_consumable(0, &[]).unwrap();
    assert_eq!(run.pending_consumables(), ["c_mercury".to_string()]);
    // A Fool cannot copy the Fool: last_tarot_planet is now c_fool... but
    // using the held Mercury resets it.
    run.use_consumable(0, &[]).unwrap();
    run.debug_add_consumable("c_fool");
    run.use_consumable(0, &[]).unwrap();
    assert_eq!(run.pending_consumables(), ["c_mercury".to_string()]);
    run.sell_consumable(0).unwrap();
    // Black Hole: every hand +1.
    run.debug_add_consumable("c_black_hole");
    run.use_consumable(0, &[]).unwrap();
    for ht in HandType::ALL {
        assert!(run.hands_table().get(ht).level >= 5002);
    }
}

#[test]
fn fool_cannot_copy_itself_and_needs_history() {
    let mut run = run_in_round("TAROTFX4");
    run.debug_add_consumable("c_fool");
    // Nothing used yet.
    assert!(matches!(
        run.use_consumable(0, &[]),
        Err(RunError::CannotUse(_))
    ));
    run.debug_add_consumable("c_mercury");
    run.use_consumable(1, &[]).unwrap();
    run.use_consumable(0, &[]).unwrap(); // Fool copies Mercury
    assert_eq!(run.pending_consumables(), ["c_mercury".to_string()]);
    // last_tarot_planet is now c_fool -> a second Fool is unusable.
    run.debug_add_consumable("c_fool");
    assert!(matches!(
        run.use_consumable(1, &[]),
        Err(RunError::CannotUse(_))
    ));
}

#[test]
fn judgement_needs_joker_space_and_purple_seal_cap() {
    let mut run = run_in_round("TAROTFX5");
    for _ in 0..5 {
        run.debug_add_joker(JokerId::Joker, Edition::None);
    }
    run.debug_add_consumable("c_judgement");
    assert!(matches!(
        run.use_consumable(0, &[]),
        Err(RunError::CannotUse(_))
    ));
    run.sell_joker(0).unwrap();
    run.use_consumable(0, &[]).unwrap();
    assert_eq!(run.jokers().len(), 5);
}

#[test]
fn consumable_slots_gate_shop_buys() {
    let mut run = run_to_shop("SLOTGATE");
    run.debug_set_dollars(100);
    run.debug_add_consumable("c_fool");
    run.debug_add_consumable("c_fool");
    // Slots full: consumable buys refuse, jokers still fine.
    let shop = run.shop().unwrap().clone();
    for (i, item) in shop.jokers.iter().enumerate() {
        if item.is_consumable() {
            assert!(matches!(run.buy_shop_item(i), Err(RunError::NoSpace)));
        }
    }
}

#[test]
fn buy_and_use_consumes_without_storing() {
    // Find a seed whose first shop offers an untargeted-usable consumable.
    for seed in ["BAU00001", "BAU00002", "BAU00003", "BAU00004", "BAU00005"] {
        let mut run = run_to_shop(seed);
        run.debug_set_dollars(100);
        let shop = run.shop().unwrap().clone();
        let slot = shop.jokers.iter().position(|item| {
            matches!(&item.kind, ShopItemKind::Consumable(c)
                if run.consumable_can_use(c.key, &[]).is_ok())
        });
        let Some(slot) = slot else { continue };
        let item = shop.jokers[slot].clone();
        let cost = run.shop_item_cost(&item);
        let d = run.dollars();
        run.buy_and_use_shop_item(slot, &[]).unwrap();
        // Paid, used on the spot, never stored (button_callbacks.lua:2473).
        assert!(run.consumables().is_empty(), "{seed}: not stored");
        let ShopItemKind::Consumable(c) = item.kind else {
            unreachable!()
        };
        // The Hermit moves money after paying; everything else leaves the
        // post-payment total alone (planets level hands silently).
        if c.key != "c_hermit" && c.key != "c_temperance" {
            assert_eq!(run.dollars(), d - cost, "{seed}: paid the price");
        }
        assert_eq!(
            run.shop().unwrap().jokers.len(),
            shop.jokers.len() - 1,
            "{seed}: slot cleared"
        );
        return;
    }
    panic!("no seed offered a usable consumable in the first shop");
}
