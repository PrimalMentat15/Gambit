//! Extended smoke runs through the full P3b loop: blind -> shop -> buy /
//! reroll / packs / consumables -> skip blinds -> next ante, on fixed seeds
//! with a deterministic greedy policy. Asserts state-machine sanity and
//! economy invariants at every step.

use balatro_core::cards::HandType;
use balatro_core::run::{Action, Run, State};
use balatro_core::shop::{PackItem, ShopItemKind};

/// Greedy scripted policy: buy the first affordable thing once per shop,
/// open the first affordable pack and pick the first pickable item, use any
/// usable held consumable (untargeted), skip every Small blind, and play
/// max-size hands.
fn drive(seed: &str, pump_levels: i64) -> Run {
    let mut run = Run::new(seed);
    for ht in HandType::ALL {
        run.level_up_hand(ht, pump_levels);
    }
    let mut steps = 0usize;
    let mut bought_this_shop = false;
    loop {
        steps += 1;
        assert!(steps < 5000, "{seed}: runaway loop");
        // Invariants that must hold in every state.
        assert!(
            run.jokers().len() <= run.joker_slots() + 1,
            "{seed}: joker overflow"
        );
        match run.state() {
            State::BlindSelect => {
                let legal = run.legal_actions();
                assert!(legal.contains(&Action::SelectBlind));
                if legal.contains(&Action::SkipBlind) && run.ante() % 2 == 1 {
                    run.skip_blind().unwrap();
                } else {
                    run.select_blind().unwrap();
                }
            }
            State::SelectingHand => {
                let n = run.hand().len().clamp(1, 5);
                let sel: Vec<usize> = if let Some(f) = run.forced_card_index() {
                    let mut s: Vec<usize> = (0..n).collect();
                    if !s.contains(&f) {
                        s[0] = f;
                        s.sort_unstable();
                    }
                    s
                } else {
                    (0..n).collect()
                };
                run.play(&sel).unwrap();
            }
            State::RoundEval => {
                bought_this_shop = false;
                run.cash_out().unwrap();
            }
            State::Shop => {
                let legal = run.legal_actions();
                assert!(legal.contains(&Action::LeaveShop));
                // Use/sell any held consumable that offers itself.
                if let Some(Action::UseConsumable(i)) =
                    legal.iter().find(|a| matches!(a, Action::UseConsumable(_)))
                {
                    let key = run.consumables()[*i].key;
                    if run.consumable_can_use(key, &[]).is_ok() {
                        run.use_consumable(*i, &[]).unwrap();
                        continue;
                    }
                }
                if !bought_this_shop {
                    if let Some(Action::BuyShopItem(i)) =
                        legal.iter().find(|a| matches!(a, Action::BuyShopItem(_)))
                    {
                        let d = run.dollars();
                        let item = run.shop().unwrap().jokers[*i].clone();
                        let cost = run.shop_item_cost(&item);
                        run.buy_shop_item(*i).unwrap();
                        assert_eq!(run.dollars(), d - cost, "{seed}: buy price");
                        assert!(run.dollars() >= 0, "{seed}: bought below $0");
                        bought_this_shop = true;
                        match item.kind {
                            ShopItemKind::Joker(_) => {
                                assert!(!run.jokers().is_empty())
                            }
                            ShopItemKind::Consumable(_) => {
                                assert!(!run.consumables().is_empty())
                            }
                            ShopItemKind::PlayingCard(_) => {}
                        }
                        continue;
                    }
                    if let Some(Action::BuyPack(i)) =
                        legal.iter().find(|a| matches!(a, Action::BuyPack(_)))
                    {
                        run.buy_pack(*i).unwrap();
                        continue;
                    }
                }
                if legal.contains(&Action::Reroll) && run.dollars() > 30 {
                    run.reroll_shop().unwrap();
                    continue;
                }
                run.leave_shop().unwrap();
            }
            State::PackOpen => {
                let pack = run.pack().unwrap().clone();
                let pick = (0..pack.items.len()).find(|&i| {
                    run.can_pick_pack_item(i)
                        && match &pack.items[i] {
                            // Only pick consumables the policy can use
                            // untargeted, to stay deterministic.
                            PackItem::Consumable(c) => run.consumable_can_use(c.key, &[]).is_ok(),
                            _ => true,
                        }
                });
                match pick {
                    Some(i) => run.pick_pack_item(i, &[]).unwrap(),
                    None => run.skip_pack().unwrap(),
                }
            }
            State::GameOver | State::Won => return run,
            s => panic!("unexpected state {s:?}"),
        }
    }
}

#[test]
fn greedy_buyer_wins_with_pumped_hands() {
    for seed in ["SMOKE001", "SMOKE002", "SMOKE003", "SMOKE004"] {
        let run = drive(seed, 5000);
        assert_eq!(run.state(), State::Won, "{seed} should win at ante 8");
        assert_eq!(run.ante(), 9, "ante increments past the win");
    }
}

#[test]
fn unpumped_run_terminates() {
    // Without pumped hands the greedy player dies somewhere; the machine
    // must still terminate cleanly.
    for seed in ["SMOKE005", "SMOKE006"] {
        let run = drive(seed, 0);
        assert!(matches!(run.state(), State::GameOver | State::Won));
    }
}

#[test]
fn snapshot_clone_is_deterministic() {
    // Run::clone is the P6 snapshot primitive: a cloned run must evolve
    // identically.
    let mut a = drive_to_first_shop("SNAPSHOT");
    let mut b = a.clone();
    for r in [&mut a, &mut b] {
        r.debug_set_dollars(100);
        r.buy_pack(0).unwrap();
    }
    assert_eq!(a.pack(), b.pack());
    assert_eq!(a.dollars(), b.dollars());
}

fn drive_to_first_shop(seed: &str) -> Run {
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
