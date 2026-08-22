//! Pure-Rust tests of the vec-env logic: encoder invariants, mask
//! guarantees, action legalization, determinism and snapshot round-trips —
//! driven by a random masked agent over full episodes.

use balatro_sim::action::ActionRow;
use balatro_sim::consts::*;
use balatro_sim::encode::{EnvMasks, EnvObs};
use balatro_sim::env::{seed_string, splitmix64, EnvSlot};

/// Tiny deterministic RNG for the random agent.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        splitmix64(&mut self.0)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn random_masked_action(m: &EnvMasks, rng: &mut Rng) -> ActionRow {
    use balatro_sim::consts::action_type as at;
    let legal: Vec<usize> = (0..N_ACTION_TYPES).filter(|&i| m.action_type[i]).collect();
    assert!(!legal.is_empty(), "no legal action type");
    let t = legal[rng.below(legal.len())] as i64;
    let mut row = ActionRow {
        action_type: t,
        cards: [-1; MAX_CARD_PICKS],
        n_cards: 0,
        joker_target: -1,
        consumable_target: -1,
        shop_target: -1,
        pack_target: -1,
    };
    let pick = |mask: &[bool], rng: &mut Rng| -> i64 {
        let opts: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
        assert!(!opts.is_empty(), "action {t} legal but pointer mask empty");
        opts[rng.below(opts.len())] as i64
    };
    match t {
        x if x == at::PLAY_HAND || x == at::DISCARD || x == at::USE_CONSUMABLE => {
            if t == at::USE_CONSUMABLE {
                row.consumable_target = pick(&m.consumable_target, rng);
            }
            let mut avail: Vec<usize> = (0..HAND_MAX).filter(|&i| m.card_select[i]).collect();
            let lo = if t == at::USE_CONSUMABLE { 0 } else { 1 };
            if lo == 1 {
                assert!(!avail.is_empty(), "PLAY/DISCARD legal but no cards");
            }
            let hi = avail.len().min(MAX_CARD_PICKS);
            let n = if hi == 0 {
                0
            } else {
                lo + rng.below(hi - lo + 1)
            };
            for k in 0..n {
                let j = rng.below(avail.len());
                row.cards[k] = avail.swap_remove(j) as i64;
            }
            row.n_cards = n as i64;
        }
        x if x == at::SELL_JOKER => row.joker_target = pick(&m.joker_target, rng),
        x if x == at::SELL_CONSUMABLE => row.consumable_target = pick(&m.consumable_target, rng),
        x if x == at::BUY_SHOP => row.shop_target = pick(&m.shop_target, rng),
        x if x == at::PICK_PACK => row.pack_target = pick(&m.pack_target, rng),
        _ => {}
    }
    row
}

fn onehot_sum(v: &[f32]) -> f32 {
    v.iter().sum()
}

fn check_obs_invariants(o: &EnvObs) {
    // Finite everywhere.
    for (name, arr) in [
        ("hand", &o.hand[..]),
        ("joker_feats", &o.joker_feats[..]),
        ("shop_feats", &o.shop_feats[..]),
        ("blind", &o.blind[..]),
        ("global", &o.global[..]),
        ("deck_counts", &o.deck_counts[..]),
        ("deck_aggregates", &o.deck_aggregates[..]),
        ("drawpile_counts", &o.drawpile_counts[..]),
    ] {
        assert!(arr.iter().all(|x| x.is_finite()), "{name} has non-finite");
    }
    // Card one-hot blocks: occupied slots have exactly-one-hot enh/edition/
    // seal (unless face down), empty slots are all-zero.
    for slot in 0..HAND_MAX {
        let f = &o.hand[slot * F_CARD..(slot + 1) * F_CARD];
        if (slot as i64) < o.hand_len {
            if f[CARD_FACEDOWN_OFF] == 1.0 {
                assert_eq!(onehot_sum(&f[..CARD_SEAL_OFF + 5]), 0.0, "face-down leaks");
            } else {
                assert_eq!(onehot_sum(&f[CARD_ENH_OFF..CARD_ENH_OFF + 9]), 1.0);
                assert_eq!(onehot_sum(&f[CARD_EDITION_OFF..CARD_EDITION_OFF + 5]), 1.0);
                assert_eq!(onehot_sum(&f[CARD_SEAL_OFF..CARD_SEAL_OFF + 5]), 1.0);
                let rs = onehot_sum(&f[CARD_RANK_OFF..CARD_RANK_OFF + 13]);
                let ss = onehot_sum(&f[CARD_SUIT_OFF..CARD_SUIT_OFF + 4]);
                assert!((rs == 1.0 && ss == 1.0) || (rs == 0.0 && ss == 0.0));
            }
        } else {
            assert_eq!(onehot_sum(f), 0.0, "padding slot {slot} not empty");
        }
    }
    // Global: phase, ante and deck one-hots; stake ordinal in range.
    let g = &o.global;
    assert_eq!(
        onehot_sum(&g[GLOBAL_ANTE_OFF..GLOBAL_ANTE_OFF + N_ANTE_ONEHOT]),
        1.0
    );
    assert_eq!(
        onehot_sum(&g[GLOBAL_PHASE_OFF..GLOBAL_PHASE_OFF + N_PHASES]),
        1.0
    );
    // The deck block is exactly one-hot in every state: the deck is fixed for
    // the whole run and must never read as "no deck".
    assert_eq!(
        onehot_sum(&g[GLOBAL_DECK_OFF..GLOBAL_DECK_OFF + N_DECKS]),
        1.0,
        "deck one-hot not set"
    );
    assert!(
        (0.0..=1.0).contains(&g[GLOBAL_STAKE]),
        "stake ordinal out of range: {}",
        g[GLOBAL_STAKE]
    );
    // Blind: kind one-hot.
    assert_eq!(
        onehot_sum(&o.blind[BLIND_KIND_OFF..BLIND_KIND_OFF + 3]),
        1.0
    );
    // Ids in vocab range.
    assert!(o.joker_ids.iter().all(|&i| (0..JOKER_VOCAB).contains(&i)));
    assert!(o
        .consumable_ids
        .iter()
        .all(|&i| (0..CONSUMABLE_VOCAB).contains(&i)));
    assert!(o.shop_ids.iter().all(|&i| (0..SHOP_VOCAB).contains(&i)));
    // drawpile <= deck counts.
    for i in 0..N_DECK_SLOTS {
        assert!(o.drawpile_counts[i] <= o.deck_counts[i] + 1e-6);
    }
}

fn check_mask_invariants(m: &EnvMasks, o: &EnvObs) {
    use balatro_sim::consts::action_type as at;
    assert!(m.action_type.iter().any(|&b| b), "no legal action");
    // Pointer-head guarantee.
    if m.action_type[at::BUY_SHOP as usize] {
        assert!(m.shop_target.iter().any(|&b| b));
    }
    if m.action_type[at::PICK_PACK as usize] {
        assert!(m.pack_target.iter().any(|&b| b));
    }
    if m.action_type[at::SELL_JOKER as usize] {
        assert!(m.joker_target.iter().any(|&b| b));
    }
    if m.action_type[at::USE_CONSUMABLE as usize] || m.action_type[at::SELL_CONSUMABLE as usize] {
        assert!(m.consumable_target.iter().any(|&b| b));
    }
    if m.action_type[at::PLAY_HAND as usize] || m.action_type[at::DISCARD as usize] {
        assert!(m.card_select.iter().any(|&b| b));
    }
    // Card mask never exceeds hand_len.
    for i in 0..HAND_MAX {
        if (i as i64) >= o.hand_len {
            assert!(!m.card_select[i], "card mask past hand_len");
        }
    }
    // MOVE_JOKER + reserved types never legal.
    assert!(!m.action_type[13..].iter().any(|&b| b));
}

/// Drive one env with a random masked agent, checking invariants each step.
fn soak_one(seed: i64, steps: usize, check: bool) -> (u64, u64) {
    let mut slot = EnvSlot::new(seed);
    let mut rng = Rng(seed as u64 ^ 0xDEADBEEF);
    let (mut episodes, mut wins) = (0u64, 0u64);
    for step in 0..steps {
        if check {
            let o = slot.observe();
            check_obs_invariants(&o);
            check_mask_invariants(&slot.masks, &o);
        }
        let row = random_masked_action(&slot.masks, &mut rng);
        let (_r, done, info) = slot
            .step(0, &row, 1.0, true)
            .unwrap_or_else(|e| panic!("step {step}: {e}"));
        if done {
            episodes += 1;
            let ep = info.expect("done without episode info");
            assert!(ep.l > 0 && ep.ante >= 1);
            if ep.won {
                wins += 1;
            }
        }
    }
    (episodes, wins)
}

#[test]
fn seed_string_is_valid_balatro_seed() {
    let mut s = 12345u64;
    for _ in 0..100 {
        let seed = seed_string(splitmix64(&mut s));
        assert_eq!(seed.len(), 8);
        assert!(seed
            .bytes()
            .all(|b| b.is_ascii_digit() || b.is_ascii_uppercase()));
    }
    assert_eq!(seed_string(0), "00000000");
}

#[test]
fn random_agent_invariant_soak() {
    // 20 envs x 2000 steps with full invariant checking (the multi-million
    // step soak runs from Python against the release build).
    for i in 0..20 {
        soak_one(1000 + i, 2000, true);
    }
}

#[test]
fn determinism_bitexact() {
    // Same seed + same action stream twice -> identical obs bytes.
    let run = |[seed, rng_seed]: [i64; 2]| -> Vec<u8> {
        let mut slot = EnvSlot::new(seed);
        let mut rng = Rng(rng_seed as u64);
        let mut hash = Vec::new();
        for _ in 0..3000 {
            let row = random_masked_action(&slot.masks, &mut rng);
            let (r, done, _) = slot.step(0, &row, 1.0, true).unwrap();
            hash.extend_from_slice(&r.to_le_bytes());
            hash.push(done as u8);
            let o = slot.observe();
            for x in o.global {
                hash.extend_from_slice(&x.to_le_bytes());
            }
            for x in o.hand {
                hash.extend_from_slice(&x.to_le_bytes());
            }
            for x in o.shop_ids {
                hash.extend_from_slice(&x.to_le_bytes());
            }
        }
        hash
    };
    assert_eq!(run([42, 7]), run([42, 7]));
    assert_ne!(run([42, 7]), run([43, 7]));
}

#[test]
fn snapshot_restore_roundtrip() {
    let mut slot = EnvSlot::new(77);
    let mut rng = Rng(99);
    for _ in 0..500 {
        let row = random_masked_action(&slot.masks, &mut rng);
        slot.step(0, &row, 1.0, true).unwrap();
    }
    let snap = slot.snapshot_bytes();

    // Continue 300 more steps from the snapshot twice; trajectories must be
    // bit-identical (RngState included in the snapshot).
    let replay = |snap: &[u8]| -> Vec<u8> {
        let mut s2 = EnvSlot::new(1); // overwritten by restore
        s2.restore_bytes(snap).unwrap();
        let mut rng = Rng(1234);
        let mut hash = Vec::new();
        for _ in 0..300 {
            let row = random_masked_action(&s2.masks, &mut rng);
            let (r, done, _) = s2.step(0, &row, 1.0, true).unwrap();
            hash.extend_from_slice(&r.to_le_bytes());
            hash.push(done as u8);
            for x in s2.observe().global {
                hash.extend_from_slice(&x.to_le_bytes());
            }
        }
        hash
    };
    assert_eq!(replay(&snap), replay(&snap));

    // Restoring reproduces the exact observation.
    let mut s3 = EnvSlot::new(2);
    s3.restore_bytes(&snap).unwrap();
    assert_eq!(slot.observe().global, s3.observe().global);
    assert_eq!(slot.observe().hand, s3.observe().hand);
    assert_eq!(slot.run.seed(), s3.run.seed());
}

#[test]
fn redeterminize_keeps_observables() {
    let mut slot = EnvSlot::new(31337);
    let mut rng = Rng(5);
    for _ in 0..200 {
        let row = random_masked_action(&slot.masks, &mut rng);
        slot.step(0, &row, 1.0, true).unwrap();
    }
    let before = slot.observe();
    let masks_before = slot.masks;
    slot.redeterminize(4242);
    let after = slot.observe();
    // Every observable is unchanged (drawpile COUNTS too — only order and
    // future streams moved).
    assert_eq!(before.hand, after.hand);
    assert_eq!(before.global, after.global);
    assert_eq!(before.drawpile_counts, after.drawpile_counts);
    assert_eq!(before.deck_counts, after.deck_counts);
    assert_eq!(masks_before.action_type, slot.masks.action_type);
}

#[test]
fn validation_rejects_mask_violations() {
    let slot = EnvSlot::new(7);
    // Fresh run starts at BLIND_SELECT: PLAY_HAND must be illegal.
    let row = ActionRow {
        action_type: 0,
        cards: [0, -1, -1, -1, -1],
        n_cards: 1,
        joker_target: -1,
        consumable_target: -1,
        shop_target: -1,
        pack_target: -1,
    };
    assert!(balatro_sim::action::validate(0, &row, &slot.masks).is_err());
    // SELECT_BLIND with a stray pointer must be rejected too.
    let row = ActionRow {
        action_type: 2,
        cards: [-1; 5],
        n_cards: 0,
        joker_target: 0,
        consumable_target: -1,
        shop_target: -1,
        pack_target: -1,
    };
    assert!(balatro_sim::action::validate(0, &row, &slot.masks).is_err());
}

#[test]
fn bot_plays_legally_and_finishes_episodes() {
    let mut episodes = 0u64;
    for seed in 0..10 {
        let mut slot = EnvSlot::new(seed);
        for _ in 0..3000 {
            let row = balatro_sim::bot::bot_action(&slot.run, &slot.masks, &slot.shop_slots);
            let (_r, done, info) = slot.step(0, &row, 1.0, true).expect("bot action rejected");
            if done {
                episodes += 1;
                assert!(info.unwrap().ante >= 1);
            }
        }
    }
    assert!(episodes > 0, "bot never finished an episode");
}

#[test]
fn win_ante_curriculum_ends_episodes_early() {
    // With win_ante = 1 the bot must never see ante 3+, every win reports
    // the post-boss ante 2 with the +15 bonus in the return, and across a
    // handful of seeds it wins at least once.
    let (mut episodes, mut wins) = (0u64, 0u64);
    for seed in 0..10 {
        let mut slot = EnvSlot::with_win_ante(seed, 1);
        for _ in 0..2000 {
            assert!(slot.run.ante() <= 2, "episode outlived win_ante 1");
            let row = balatro_sim::bot::bot_action(&slot.run, &slot.masks, &slot.shop_slots);
            let (_r, done, info) = slot.step(0, &row, 1.0, true).expect("bot action rejected");
            if done {
                episodes += 1;
                let ep = info.unwrap();
                assert!(ep.ante <= 2);
                assert_eq!(ep.won, ep.ante == 2, "won iff the ante-1 boss fell");
                if ep.won {
                    wins += 1;
                    assert!(ep.r >= 15.0, "win bonus missing from return: {}", ep.r);
                }
            }
        }
    }
    assert!(episodes > 0);
    assert!(wins > 0, "bot never beat ante 1 across 10 seeds");
}

#[test]
fn set_win_ante_applies_at_next_reset() {
    let mut slot = EnvSlot::with_win_ante(3, 8);
    let mut rng = Rng(11);
    slot.set_win_ante(1);
    assert_eq!(slot.win_ante, 8, "in-flight episode must keep its goalpost");
    // Drive to the end of the current episode; the next one uses 1.
    for step in 0..200_000 {
        let row = random_masked_action(&slot.masks, &mut rng);
        let (_r, done, _) = slot.step(0, &row, 1.0, true).unwrap();
        if done {
            break;
        }
        assert!(step < 199_999, "random agent never finished an episode");
    }
    assert_eq!(slot.win_ante, 1);

    // Snapshot round-trip preserves the episode's win_ante.
    let snap = slot.snapshot_bytes();
    let mut s2 = EnvSlot::new(1);
    s2.restore_bytes(&snap).unwrap();
    assert_eq!(s2.win_ante, 1);
}

// ------------------------------------------------------------------------
// Snapshot start-state mixing (M3): pool format + auto-reset sampling.

/// Bot-drive a fresh slot until it reaches `ante` in Shop or BlindSelect;
/// returns (ante, snapshot bytes). Panics if the bot dies first.
fn harvest_snapshot(seed: i64, ante: i64) -> Option<(u8, Vec<u8>)> {
    use balatro_core::run::State;
    let mut slot = EnvSlot::new(seed);
    for _ in 0..3000 {
        if slot.run.ante() == ante && matches!(slot.run.state(), State::Shop | State::BlindSelect) {
            return Some((ante as u8, slot.snapshot_bytes()));
        }
        let row = balatro_sim::bot::bot_action(&slot.run, &slot.masks, &slot.shop_slots);
        let (_r, done, _) = slot.step(0, &row, 1.0, true).unwrap();
        if done {
            return None; // bot lost (or won) before reaching the target
        }
    }
    None
}

fn build_test_pool(ante: i64, want: usize) -> balatro_sim::pool::SnapshotPool {
    let mut entries = Vec::new();
    let mut seed = 0i64;
    while entries.len() < want && seed < 200 {
        if let Some(e) = harvest_snapshot(seed, ante) {
            entries.push(e);
        }
        seed += 1;
    }
    assert_eq!(
        entries.len(),
        want,
        "could not harvest {want} ante-{ante} snapshots"
    );
    balatro_sim::pool::SnapshotPool::parse(&balatro_sim::pool::pack_pool(&entries)).unwrap()
}

#[test]
fn pool_pack_parse_roundtrip_and_eligibility() {
    use balatro_sim::pool::{pack_pool, SnapshotPool};
    let e1 = harvest_snapshot(3, 1).expect("ante-1 snapshot");
    let e2 = harvest_snapshot(5, 2).expect("ante-2 snapshot");
    let entries = vec![e1.clone(), e2.clone()];
    let pool = SnapshotPool::parse(&pack_pool(&entries)).unwrap();
    assert_eq!(pool.len(), 2);
    assert_eq!((pool.ante(0), pool.ante(1)), (1, 2));
    assert_eq!(pool.get(0), &e1.1[..]);
    assert_eq!(pool.get(1), &e2.1[..]);
    // Eligibility: win_ante 1 -> nothing; 2 -> only the ante-1 entry;
    // 8 -> both (uniform over {0, 1}).
    assert_eq!(pool.sample_eligible(1, 12345), None);
    for r in 0..10u64 {
        assert_eq!(pool.sample_eligible(2, r), Some(0));
    }
    let picks: Vec<usize> = (0..10u64)
        .map(|r| pool.sample_eligible(8, r).unwrap())
        .collect();
    assert!(picks.contains(&0) && picks.contains(&1));

    // Corrupt inputs are rejected.
    assert!(SnapshotPool::parse(b"NOTAPOOL").is_err());
    assert!(SnapshotPool::parse(&[]).is_err());
    let mut bad_ante = entries.clone();
    bad_ante[0].0 = 4; // header says 4, run says 1
    assert!(SnapshotPool::parse(&pack_pool(&bad_ante)).is_err());
    let mut bad_bytes = entries;
    bad_bytes[1].1[10] ^= 0xFF;
    assert!(SnapshotPool::parse(&pack_pool(&bad_bytes)).is_err());
}

#[test]
fn snapshot_mixing_fraction_1_starts_from_pool_and_is_deterministic() {
    let pool = build_test_pool(2, 3);

    let run_stream = |fraction: f64| -> (Vec<u8>, u64, u64) {
        let mut slot = EnvSlot::new(4711);
        let mut rng = Rng(0xF00D);
        let mut hash = Vec::new();
        let (mut from_snap, mut fresh) = (0u64, 0u64);
        for _ in 0..6000 {
            let row = random_masked_action(&slot.masks, &mut rng);
            let (r, done, info) = slot
                .step_with_pool(0, &row, 1.0, true, Some(&pool), fraction)
                .unwrap();
            hash.extend_from_slice(&r.to_le_bytes());
            hash.push(done as u8);
            for x in slot.observe().global {
                hash.extend_from_slice(&x.to_le_bytes());
            }
            if done {
                // The NEXT episode's origin is visible on the slot.
                if slot.from_snapshot {
                    from_snap += 1;
                    // A pool start resumes mid-run: ante >= the entry's.
                    assert!(slot.run.ante() >= 2, "pool start at ante 1");
                    assert_eq!(slot.ep_len, 0);
                    assert_eq!(slot.ep_return, 0.0);
                } else {
                    fresh += 1;
                    assert_eq!(slot.run.ante(), 1);
                }
                // Episode info reports the FINISHED episode's origin.
                assert!(info.is_some());
            }
        }
        (hash, from_snap, fresh)
    };

    // fraction 1: every auto-reset starts from the pool.
    let (h1, snap1, fresh1) = run_stream(1.0);
    assert!(snap1 > 0, "no episodes finished in 6000 steps");
    assert_eq!(fresh1, 0);
    // Deterministic: identical stream on replay.
    let (h2, ..) = run_stream(1.0);
    assert_eq!(h1, h2);

    // fraction 0: never from the pool, and bit-identical to the legacy
    // no-pool step() stream.
    let (h0, snap0, fresh0) = run_stream(0.0);
    assert_eq!(snap0, 0);
    assert!(fresh0 > 0);
    let legacy = {
        let mut slot = EnvSlot::new(4711);
        let mut rng = Rng(0xF00D);
        let mut hash = Vec::new();
        for _ in 0..6000 {
            let row = random_masked_action(&slot.masks, &mut rng);
            let (r, done, _) = slot.step(0, &row, 1.0, true).unwrap();
            hash.extend_from_slice(&r.to_le_bytes());
            hash.push(done as u8);
            for x in slot.observe().global {
                hash.extend_from_slice(&x.to_le_bytes());
            }
        }
        hash
    };
    assert_eq!(h0, legacy);
}

#[test]
fn snapshot_mixing_respects_win_ante_and_redeterminizes() {
    let pool = build_test_pool(2, 2);

    // win_ante 1: every pool entry (ante 2) is at/past the goal ->
    // fresh-start fallback even at fraction 1.
    let mut slot = EnvSlot::with_win_ante(99, 1);
    let mut rng = Rng(0xBEEF);
    let mut episodes = 0u64;
    for _ in 0..6000 {
        let row = random_masked_action(&slot.masks, &mut rng);
        let (_r, done, info) = slot
            .step_with_pool(0, &row, 1.0, true, Some(&pool), 1.0)
            .unwrap();
        if done {
            episodes += 1;
            assert!(!slot.from_snapshot, "ineligible snapshot was used");
            assert!(!info.unwrap().from_snapshot);
        }
    }
    assert!(episodes > 0);

    // win_ante 3: ante-2 entries are eligible; episode info flags them and
    // the +15 arrives on clearing the ante-3 boss. Two envs sharing the
    // seed stream but different pool draws diverge only via the stream —
    // same seed twice must match (redeterminize seed included).
    let drive = |seed: i64| -> Vec<(bool, bool, i64)> {
        let mut slot = EnvSlot::with_win_ante(seed, 3);
        let mut rng = Rng(0x1234);
        let mut eps = Vec::new();
        for _ in 0..8000 {
            let row = random_masked_action(&slot.masks, &mut rng);
            let (_r, done, info) = slot
                .step_with_pool(0, &row, 1.0, true, Some(&pool), 1.0)
                .unwrap();
            if done {
                let ep = info.unwrap();
                eps.push((ep.from_snapshot, ep.won, ep.ante));
            }
        }
        eps
    };
    let a = drive(7);
    let b = drive(7);
    assert_eq!(a, b);
    assert!(
        a.iter().any(|&(fs, ..)| fs),
        "no from-snapshot episode finished"
    );
    for &(_, won, ante) in &a {
        assert!(ante <= 4, "episode outlived win_ante 3");
        assert_eq!(won, ante == 4);
    }
}

#[test]
fn aura_and_every_usable_consumable_legalizes() {
    use balatro_core::run::State;
    use balatro_sim::action::legalize_consumable_targets;

    // Direct regression for c_aura (registry meta has max_highlighted == 0;
    // the legalizer must still produce its 1 edition-less target).
    let mut slot = EnvSlot::new(4242);
    slot.run.select_blind().unwrap();
    assert_eq!(slot.run.state(), State::SelectingHand);
    slot.run.debug_add_consumable("c_aura");
    slot.refresh();
    assert!(slot.run.consumable_has_any_use("c_aura"));
    let targets = legalize_consumable_targets(&slot.run, "c_aura", &[])
        .expect("aura usable per mask but legalize failed");
    assert_eq!(targets.len(), 1);

    // ...and USE_CONSUMABLE through the real step path applies an edition.
    let idx = slot
        .run
        .consumables()
        .iter()
        .position(|c| c.key == "c_aura")
        .unwrap();
    let row = ActionRow {
        action_type: 8,
        cards: [-1; 5],
        n_cards: 0,
        joker_target: -1,
        consumable_target: idx as i64,
        shop_target: -1,
        pack_target: -1,
    };
    slot.step(0, &row, 1.0, true).unwrap();
    assert!(slot.run.consumables().iter().all(|c| c.key != "c_aura"));
    assert!(slot
        .run
        .hand()
        .iter()
        .any(|c| c.edition != balatro_core::cards::Edition::None));

    // Property over a random soak: whenever the sim says a held or
    // pack consumable has ANY use, the auto-legalizer must find targets.
    let mut rng = Rng(0xA0A0);
    let mut slot = EnvSlot::new(999);
    for _ in 0..4000 {
        let mut keys: Vec<&'static str> = slot.run.consumables().iter().map(|c| c.key).collect();
        if let Some(p) = slot.run.pack() {
            for it in &p.items {
                if let balatro_core::shop::PackItem::Consumable(c) = it {
                    keys.push(c.key);
                }
            }
        }
        for key in keys {
            if slot.run.consumable_has_any_use(key) {
                assert!(
                    legalize_consumable_targets(&slot.run, key, &[]).is_some(),
                    "{key}: has_any_use but legalize failed"
                );
            }
        }
        let row = random_masked_action(&slot.masks, &mut rng);
        slot.step(0, &row, 1.0, true).unwrap();
    }
}
