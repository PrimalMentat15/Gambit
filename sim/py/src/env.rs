//! The per-env state machine: seeding, stepping with auto-reset, episode
//! bookkeeping, snapshot/restore and redeterminize.
//!
//! # Seed derivation (documented scheme)
//! `reset(seeds)` gives env *i* an independent SplitMix64 stream seeded with
//! `seeds[i] as u64`. Every episode (including the first) draws the next
//! u64 from that stream and maps it to a Balatro seed string:
//! `x % 36^8` written big-endian in base-36 (`0-9A-Z`), zero-padded to 8
//! chars — always a valid in-game seed (`[A-Z0-9]{1,8}`), so P5 can replay
//! any sim episode live. Auto-reset draws the next seed from the same
//! stream, keeping the whole trajectory a pure function of `seeds[i]`.

use balatro_core::run::{Run, State};
use serde::{Deserialize, Serialize};

use crate::action::{apply, validate, ActionRow};
use crate::encode::{encode_masks, encode_obs, shop_slot_map, EnvMasks, EnvObs, ShopSlotRef};
use crate::pool::SnapshotPool;

pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

const SEED_CHARS: &[u8; 36] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// u64 -> 8-char base-36 Balatro seed (valid in the real game).
pub fn seed_string(x: u64) -> String {
    let mut n = x % 36u64.pow(8);
    let mut out = [b'0'; 8];
    for slot in out.iter_mut().rev() {
        *slot = SEED_CHARS[(n % 36) as usize];
        n /= 36;
    }
    String::from_utf8(out.to_vec()).unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Episode {
    pub r: f64,
    pub l: i64,
    pub ante: i64,
    pub won: bool,
    /// True when the episode STARTED from a sampled pool snapshot (M3
    /// start-state mixing) rather than a fresh seeded run.
    pub from_snapshot: bool,
}

/// Snapshot payload: the full run plus the episode bookkeeping and the
/// env's reset-seed stream, so a restored env continues bit-identically.
#[derive(Serialize, Deserialize)]
pub struct Snapshot {
    pub run: Run,
    pub ep_return: f64,
    pub ep_len: i64,
    pub seed_stream: u64,
    /// Curriculum win ante of the snapshotted episode.
    pub win_ante: u8,
}

pub struct EnvSlot {
    pub run: Run,
    pub seed_stream: u64,
    pub ep_return: f64,
    pub ep_len: i64,
    /// Curriculum knob: the episode ends with the win bonus once the boss
    /// blind of THIS ante is defeated (8 = the real game's win). Fixed for
    /// the lifetime of an episode.
    pub win_ante: u8,
    /// `set_win_ante` writes here; `win_ante` picks it up at the next
    /// (auto-)reset so an in-flight episode keeps its goalpost.
    pub pending_win_ante: u8,
    /// Shop-slot mapping as of the LAST emitted observation (what the
    /// agent's shop_target indexes refer to).
    pub shop_slots: Vec<ShopSlotRef>,
    /// Masks emitted with the last observation (strict validation).
    pub masks: EnvMasks,
    /// Whether the CURRENT episode started from a pool snapshot.
    pub from_snapshot: bool,
}

impl EnvSlot {
    pub fn new(seed: i64) -> Self {
        Self::with_win_ante(seed, 8)
    }

    pub fn with_win_ante(seed: i64, win_ante: u8) -> Self {
        let mut stream = seed as u64;
        let run_seed = seed_string(splitmix64(&mut stream));
        let mut slot = EnvSlot {
            run: Run::new(&run_seed),
            seed_stream: stream,
            ep_return: 0.0,
            ep_len: 0,
            win_ante,
            pending_win_ante: win_ante,
            shop_slots: Vec::new(),
            masks: EnvMasks::default(),
            from_snapshot: false,
        };
        slot.refresh();
        slot
    }

    /// Schedule a new curriculum win ante; takes effect at the next
    /// (auto-)reset of this env — the current episode is unaffected.
    pub fn set_win_ante(&mut self, win_ante: u8) {
        self.pending_win_ante = win_ante;
    }

    /// Curriculum win: the real win flag, or the boss of `win_ante` was
    /// defeated. `Run::ante()` increments exactly once per boss defeat
    /// (`ease_ante(1)` in end_round; Hieroglyph only ever lowers it), so
    /// `ante > win_ante` first holds on the step that cleared `win_ante`'s
    /// boss.
    pub fn won(&self) -> bool {
        self.run.won() || self.run.ante() > i64::from(self.win_ante)
    }

    /// Recompute the slot map + masks for the current run state.
    pub fn refresh(&mut self) {
        self.shop_slots = shop_slot_map(&self.run);
        self.masks = encode_masks(&self.run, &self.shop_slots);
    }

    pub fn observe(&self) -> EnvObs {
        encode_obs(&self.run)
    }

    fn fresh_episode(&mut self) {
        let run_seed = seed_string(splitmix64(&mut self.seed_stream));
        self.run = Run::new(&run_seed);
        self.ep_return = 0.0;
        self.ep_len = 0;
        self.win_ante = self.pending_win_ante;
        self.from_snapshot = false;
    }

    /// Auto-reset: begin the next episode. With probability `fraction`,
    /// start from a uniformly sampled pool snapshot; a fresh seeded run
    /// otherwise (and always when `fraction == 0.0` or no pool is set).
    ///
    /// # Reproducibility (documented scheme)
    /// Every random choice here is drawn from the env's own SplitMix64
    /// seed stream, in a fixed order: (1) the include-snapshot decision as
    /// a 53-bit uniform in [0,1); (2) the pool index; (3) the
    /// redeterminize seed. A fresh episode draws only its run seed. The
    /// whole trajectory therefore stays a pure function of
    /// (`seeds[i]`, pool file, the fraction/win_ante schedule). With
    /// `fraction == 0.0` (or no pool) NOTHING extra is drawn — behavior is
    /// bit-identical to the pre-M3 env.
    ///
    /// # win_ante interaction
    /// Only snapshots with `ante < win_ante` (the env's CURRENT pending
    /// curriculum goal) are eligible — an episode must never start at or
    /// past its goalpost. Sampling is uniform over the eligible subset
    /// (`SnapshotPool::sample_eligible`); if none qualifies the episode
    /// falls back to a fresh start.
    ///
    /// # Episode identity of a snapshot start
    /// The restored run is `redeterminize`d with a stream-drawn seed:
    /// undrawn deck order + all future RNG streams are re-randomized, so
    /// episodes sharing a pool entry diverge (and none replays the
    /// generator's exact future). Episode bookkeeping restarts
    /// (`ep_return = 0`, `ep_len = 0`), the env keeps its OWN seed stream
    /// and curriculum goalpost — the snapshot's are ignored.
    fn begin_episode(&mut self, pool: Option<&SnapshotPool>, fraction: f64) {
        if let Some(pool) = pool {
            if fraction > 0.0 {
                let u = (splitmix64(&mut self.seed_stream) >> 11) as f64 / (1u64 << 53) as f64;
                if u < fraction {
                    let idx_raw = splitmix64(&mut self.seed_stream);
                    if let Some(idx) = pool.sample_eligible(self.pending_win_ante, idx_raw) {
                        let redet = splitmix64(&mut self.seed_stream);
                        // Pool entries are validated at load; decode cannot
                        // fail here.
                        let snap: Snapshot =
                            bincode::deserialize(pool.get(idx)).expect("validated pool entry");
                        self.run = snap.run;
                        self.run.redeterminize(&seed_string(redet));
                        self.ep_return = 0.0;
                        self.ep_len = 0;
                        self.win_ante = self.pending_win_ante;
                        self.from_snapshot = true;
                        return;
                    }
                }
            }
        }
        self.fresh_episode();
    }

    /// Apply one composite action; auto-reset on episode end (fresh-seed
    /// episodes only — the pre-M3 behavior). Returns (reward, done,
    /// episode-info) — the caller encodes obs/masks after.
    pub fn step(
        &mut self,
        env_idx: usize,
        row: &ActionRow,
        beta: f64,
        strict: bool,
    ) -> Result<(f32, bool, Option<Episode>), String> {
        self.step_with_pool(env_idx, row, beta, strict, None, 0.0)
    }

    /// `step` with snapshot start-state mixing on auto-reset (see
    /// `begin_episode`).
    pub fn step_with_pool(
        &mut self,
        env_idx: usize,
        row: &ActionRow,
        beta: f64,
        strict: bool,
        pool: Option<&SnapshotPool>,
        snapshot_fraction: f64,
    ) -> Result<(f32, bool, Option<Episode>), String> {
        if strict {
            validate(env_idx, row, &self.masks)?;
        }
        let reward = apply(
            &mut self.run,
            row,
            &self.shop_slots,
            beta,
            i64::from(self.win_ante),
        )?;
        self.ep_len += 1;
        self.ep_return += reward;

        // Episode end: win as soon as the curriculum goal is reached (the
        // `win_ante` boss round survived, Mr. Bones saves included; ante 8 =
        // the real win), or game over. A non-terminal state with no legal
        // action (deck destroyed to nothing) is treated as a loss.
        self.shop_slots = shop_slot_map(&self.run);
        let state = self.run.state();
        let won = self.won();
        let done = won
            || matches!(state, State::GameOver | State::Won)
            || !crate::encode::has_any_legal(&self.run, &self.shop_slots);

        let mut info = None;
        if done {
            info = Some(Episode {
                r: self.ep_return,
                l: self.ep_len,
                ante: self.run.ante(),
                won,
                from_snapshot: self.from_snapshot,
            });
            self.begin_episode(pool, snapshot_fraction);
            self.shop_slots = shop_slot_map(&self.run);
        }
        self.masks = encode_masks(&self.run, &self.shop_slots);
        Ok((reward as f32, done, info))
    }

    pub fn snapshot_bytes(&self) -> Vec<u8> {
        let snap = Snapshot {
            run: self.run.clone(),
            ep_return: self.ep_return,
            ep_len: self.ep_len,
            seed_stream: self.seed_stream,
            win_ante: self.win_ante,
        };
        bincode::serialize(&snap).expect("snapshot serialize")
    }

    pub fn restore_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        let snap: Snapshot =
            bincode::deserialize(bytes).map_err(|e| format!("snapshot decode: {e}"))?;
        self.run = snap.run;
        self.ep_return = snap.ep_return;
        self.ep_len = snap.ep_len;
        self.seed_stream = snap.seed_stream;
        // Restore the snapshotted episode's goalpost; `pending_win_ante`
        // (this env's scheduled curriculum ante) and `from_snapshot` (a
        // property of how the episode was STARTED, not stored in the
        // snapshot) are left as-is.
        self.win_ante = snap.win_ante;
        self.refresh();
        Ok(())
    }

    /// See `Run::redeterminize`: reshuffles the undrawn deck and replaces
    /// every future RNG stream with a fresh seed's; observable state
    /// (and therefore the masks) is unchanged.
    pub fn redeterminize(&mut self, seed: i64) {
        let mut s = seed as u64;
        let seed_str = seed_string(splitmix64(&mut s));
        self.run.redeterminize(&seed_str);
        self.refresh();
    }
}
