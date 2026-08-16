//! Run-level joker engine tests: passive add_to_deck effects, the
//! setting_blind/open_booster/skipping_booster hook windows, the per-round
//! mail/idol/ancient/castle re-rolls, payout wiring, and smoke runs with
//! random common-joker loadouts.

use balatro_core::cards::Edition;
use balatro_core::deck::red_deck;
use balatro_core::items::{element_index, JokerId, JOKERS};
use balatro_core::rng::RngState;
use balatro_core::run::{Run, State};

fn selection(run: &Run, n: usize) -> Vec<usize> {
    let mut sel: Vec<usize> = Vec::new();
    if let Some(f) = run.forced_card_index() {
        sel.push(f);
    }
    let mut i = 0;
    while sel.len() < n.min(run.hand().len()) && i < run.hand().len() {
        if !sel.contains(&i) {
            sel.push(i);
        }
        i += 1;
    }
    sel.sort_unstable();
    sel
}

/// Play first-5 until the current round ends (RoundEval / GameOver).
fn play_round(run: &mut Run) {
    while run.state() == State::SelectingHand {
        let sel = selection(run, 5);
        run.play(&sel).unwrap();
    }
}

/// Level every hand so a scripted first-5 policy always beats the blind.
fn overlevel(run: &mut Run) {
    for ht in balatro_core::cards::HandType::ALL {
        run.level_up_hand(ht, 30);
    }
}

#[test]
fn credit_card_moves_bankrupt_at() {
    let mut run = Run::new("PASSIVES");
    assert_eq!(run.bankrupt_at(), 0);
    run.debug_add_joker(JokerId::CreditCard, Edition::None);
    assert_eq!(run.bankrupt_at(), -20); // card.lua:593-595
    run.sell_joker(0).unwrap();
    assert_eq!(run.bankrupt_at(), 0);
}

#[test]
fn juggler_and_drunkard_passives() {
    let mut run = Run::new("PASSIVES");
    assert_eq!(run.hand_size(), 8);
    run.debug_add_joker(JokerId::Juggler, Edition::None);
    assert_eq!(run.hand_size(), 9); // h_size = 1 (card.lua:586-587)
    run.debug_add_joker(JokerId::Drunkard, Edition::None);
    assert_eq!(run.discards_left(), 5); // ease_discard(+1) immediately
    run.select_blind().unwrap();
    assert_eq!(run.discards_left(), 5); // round_resets.discards 4 + 1
    assert_eq!(run.hand().len(), 9);
    play_round(&mut run);
}

#[test]
fn chaos_gives_free_rerolls() {
    let mut run = Run::new("CHAOS");
    run.debug_add_joker(JokerId::Chaos, Edition::None);
    // add_to_deck bumps free_rerolls and recomputes (card.lua:602-605).
    assert_eq!(run.reroll_cost(), 0);
    run.select_blind().unwrap(); // new_round: free_rerolls = #Chaos
    assert_eq!(run.reroll_cost(), 0);
    run.sell_joker(0).unwrap();
    assert_eq!(run.reroll_cost(), 5);
}

#[test]
fn riff_raff_creates_common_jokers_at_blind_select() {
    let mut run = Run::new("RIFFRAFF");
    run.debug_add_joker(JokerId::RiffRaff, Edition::None);
    run.select_blind().unwrap();
    // card.lua:2534-2551 — up to 2 commons ('Joker1rif1' pool).
    assert_eq!(run.jokers().len(), 3);
    for j in &run.jokers()[1..] {
        assert_eq!(j.id.meta().rarity, 1, "{:?}", j.id);
    }
    play_round(&mut run);

    // Slot gating: with 4 slots already taken only 1 is created.
    let mut run = Run::new("RIFFRAFF2");
    for _ in 0..3 {
        run.debug_add_joker(JokerId::Joker, Edition::None);
    }
    run.debug_add_joker(JokerId::RiffRaff, Edition::None);
    run.select_blind().unwrap();
    assert_eq!(run.jokers().len(), 5);
}

#[test]
fn red_card_scales_on_skipped_packs() {
    let mut run = Run::new("REDCARD");
    run.force_tag(Some("tag_buffoon"));
    run.debug_add_joker(JokerId::RedCard, Edition::None);
    run.skip_blind().unwrap(); // Buffoon Tag opens a mega pack
    assert_eq!(run.state(), State::PackOpen);
    run.skip_pack().unwrap();
    // card.lua:2442-2457 — +3 mult per skipped booster.
    assert_eq!(run.jokers()[0].state.mult, 3.0);
}

#[test]
fn hallucination_rolls_on_pack_open() {
    // Find a seed whose first 'halu1' roll succeeds (< 1/2) and one that
    // fails; the tag-opened Arcana pack must create a tarot accordingly.
    let mut hit = None;
    let mut miss = None;
    for i in 0..50 {
        let s = format!("HALU{i}");
        if RngState::new(&s).random("halu1") < 0.5 {
            hit.get_or_insert(s);
        } else {
            miss.get_or_insert(s);
        }
    }
    for (seed, expect) in [(hit.unwrap(), 1usize), (miss.unwrap(), 0usize)] {
        let mut run = Run::new(&seed);
        run.force_tag(Some("tag_charm"));
        run.debug_add_joker(JokerId::Hallucination, Edition::None);
        run.skip_blind().unwrap();
        assert_eq!(run.state(), State::PackOpen);
        assert_eq!(run.consumables().len(), expect, "seed {seed}");
    }
}

#[test]
fn egg_raises_sell_value_each_round() {
    let mut run = Run::new("EGGSEED");
    overlevel(&mut run);
    run.debug_add_joker(JokerId::Egg, Edition::None);
    run.select_blind().unwrap();
    play_round(&mut run);
    assert_eq!(run.state(), State::RoundEval);
    let egg = &run.jokers()[0];
    assert_eq!(egg.extra_value, 3); // card.lua:2986-2993
    assert_eq!(run.joker_sell_value(egg), 2 + 3); // floor(4/2) + extra_value
}

#[test]
fn mail_rank_rolls_at_run_start_and_each_round() {
    // Run start: 'mail1' element over the 52 deck cards sorted by sort_id
    // (creation order), after game.lua:2383's shuffle.
    let seed = "MAILSEED";
    let run = Run::new(seed);
    let deck = red_deck();
    let mut mirror = RngState::new(seed);
    // Same stream, same order: idol1 is drawn first (game.lua:2385).
    let _ = element_index(&mut mirror, &format!("idol{}", 1), 52);
    let idx = element_index(&mut mirror, &format!("mail{}", 1), 52);
    assert_eq!(run.mail_rank_id(), deck[idx].rank.id());

    // After a won round the rank re-rolls (state_events.lua:274).
    let mut run = Run::new(seed);
    let before = run.mail_rank_id();
    run.select_blind().unwrap();
    play_round(&mut run);
    if run.state() == State::RoundEval {
        let after = run.mail_rank_id();
        // The second 'mail1' draw — recompute from the mirror.
        let idx2 = element_index(&mut mirror, &format!("mail{}", 1), 52);
        assert_eq!(after, deck[idx2].rank.id());
        let _ = before;
    }
}

#[test]
fn golden_joker_and_delayed_grat_pay_at_round_eval() {
    // Identical scripted rounds; the only difference is the payout joker
    // (debug_add_joker consumes no streams for these).
    let base = {
        let mut run = Run::new("PAYOUT");
        overlevel(&mut run);
        run.select_blind().unwrap();
        play_round(&mut run);
        assert_eq!(run.state(), State::RoundEval);
        run.pending_cashout()
    };
    let with_golden = {
        let mut run = Run::new("PAYOUT");
        overlevel(&mut run);
        run.debug_add_joker(JokerId::Golden, Edition::None);
        run.select_blind().unwrap();
        play_round(&mut run);
        run.pending_cashout()
    };
    assert_eq!(with_golden - base, 4); // card.lua:1658-1660
    let with_grat = {
        let mut run = Run::new("PAYOUT");
        overlevel(&mut run);
        run.debug_add_joker(JokerId::DelayedGrat, Edition::None);
        run.select_blind().unwrap();
        play_round(&mut run); // never discards
        run.pending_cashout()
    };
    assert_eq!(with_grat - base, 2 * 4); // $2 x 4 unused discards
}

#[test]
fn tarot_usage_counts_for_fortune_teller() {
    let mut run = Run::new("FORTUNE");
    assert_eq!(run.tarots_used(), 0);
    run.debug_add_consumable("c_hermit"); // untargeted tarot
    run.use_consumable(0, &[]).unwrap();
    assert_eq!(run.tarots_used(), 1);
}

#[test]
fn to_do_list_rolls_at_creation() {
    // debug_add_joker mirrors Card creation: set_ability rolls 'to_do'
    // (card.lua:311-322) over the 9 initially-visible hands.
    let mut run = Run::new("TODO");
    let mut mirror = RngState::new("TODO");
    run.debug_add_joker(JokerId::TodoList, Edition::None);
    let idx = element_index(&mut mirror, "to_do", 9);
    // MOST_PLAYED_SCAN order, secret hands hidden.
    use balatro_core::cards::HandType::*;
    let pool = [
        Pair,
        HighCard,
        StraightFlush,
        FourOfAKind,
        FullHouse,
        Flush,
        Straight,
        ThreeOfAKind,
        TwoPair,
    ];
    assert_eq!(run.jokers()[0].state.to_do_hand, pool[idx]);
}

/// Smoke: scripted runs with rotating common-joker loadouts across seeds —
/// deterministic, panic-free, and invariant-respecting.
#[test]
fn smoke_runs_with_common_joker_loadouts() {
    let commons: Vec<JokerId> = JOKERS
        .iter()
        .filter(|m| m.rarity == 1)
        .map(|m| m.id)
        .collect();
    assert_eq!(commons.len(), 61);

    for (si, seed) in ["SMOKE1", "SMOKE2", "SMOKE3", "SMOKE4"].iter().enumerate() {
        let mut run = Run::new(seed);
        let mut next = si * 13; // rotate which commons get loaded
        for step in 0..3000usize {
            match run.state() {
                State::BlindSelect => {
                    while run.jokers().len() < run.joker_slots() {
                        run.debug_add_joker(commons[next % commons.len()], Edition::None);
                        next += 7;
                    }
                    run.select_blind().unwrap();
                }
                State::SelectingHand => {
                    // Discard once per round when possible to exercise the
                    // discard hooks, then play.
                    if run.discards_left() > 0 && step % 3 == 0 {
                        let sel = selection(&run, 3);
                        run.discard(&sel).unwrap();
                    } else {
                        let sel = selection(&run, 5);
                        let r = run.play(&sel).unwrap();
                        assert!(r.score.is_finite() && r.score >= 0.0);
                    }
                }
                State::RoundEval => run.cash_out().unwrap(),
                State::Shop => {
                    // Sell one joker occasionally to exercise passives-off.
                    if step % 5 == 0 && !run.jokers().is_empty() {
                        run.sell_joker(0).unwrap();
                    }
                    run.leave_shop().unwrap();
                }
                State::PackOpen => run.skip_pack().unwrap(),
                State::GameOver | State::Won => break,
                _ => unreachable!(),
            }
            assert!(run.jokers().len() <= run.joker_slots());
        }
        // Determinism: replay produces the same end state.
        let mut replay = Run::new(seed);
        let mut next2 = si * 13;
        for step in 0..3000usize {
            match replay.state() {
                State::BlindSelect => {
                    while replay.jokers().len() < replay.joker_slots() {
                        replay.debug_add_joker(commons[next2 % commons.len()], Edition::None);
                        next2 += 7;
                    }
                    replay.select_blind().unwrap();
                }
                State::SelectingHand => {
                    if replay.discards_left() > 0 && step % 3 == 0 {
                        let sel = selection(&replay, 3);
                        replay.discard(&sel).unwrap();
                    } else {
                        let sel = selection(&replay, 5);
                        replay.play(&sel).unwrap();
                    }
                }
                State::RoundEval => replay.cash_out().unwrap(),
                State::Shop => {
                    if step % 5 == 0 && !replay.jokers().is_empty() {
                        replay.sell_joker(0).unwrap();
                    }
                    replay.leave_shop().unwrap();
                }
                State::PackOpen => replay.skip_pack().unwrap(),
                State::GameOver | State::Won => break,
                _ => unreachable!(),
            }
        }
        assert_eq!(run.state(), replay.state(), "seed {seed}");
        assert_eq!(run.dollars(), replay.dollars(), "seed {seed}");
        assert_eq!(run.round(), replay.round(), "seed {seed}");
    }
}
