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

use balatro_core::config::{Deck, RunConfig, Stake};
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
    /// The deck/stake this episode was actually played on. Reported per
    /// episode rather than read off the env, because a distribution can
    /// change what the NEXT episode uses before the trainer reads this one.
    pub deck: Deck,
    pub stake: Stake,
}

/// Per-episode deck/stake sampling for a vec env.
///
/// `None` on an axis means "use the env's fixed value and draw NOTHING".
/// That is load-bearing: with no distribution set, `begin_episode` consumes
/// exactly as many numbers from the seed stream as it did before P7, so
/// single-config runs stay bit-identical to the pre-P7 env — the same
/// guarantee snapshot mixing already makes for `fraction == 0.0`.
#[derive(Debug, Clone, Default)]
pub struct ConfigDist {
    /// Cumulative weights, ascending, last == 1.0.
    pub deck_cdf: Option<Vec<(Deck, f64)>>,
    pub stake_cdf: Option<Vec<(Stake, f64)>>,
}

impl ConfigDist {
    pub fn is_empty(&self) -> bool {
        self.deck_cdf.is_none() && self.stake_cdf.is_none()
    }

    /// Build a CDF from non-negative weights. Zero-weight entries are kept
    /// (they simply never get drawn); an all-zero or empty list is rejected.
    pub fn cdf<T: Copy>(items: &[T], weights: &[f64]) -> Result<Vec<(T, f64)>, String> {
        if items.len() != weights.len() {
            return Err(format!(
                "expected {} weights, got {}",
                items.len(),
                weights.len()
            ));
        }
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err("weights must be finite and non-negative".into());
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err("weights sum to zero".into());
        }
        let mut acc = 0.0;
        let mut out = Vec::with_capacity(items.len());
        for (i, w) in weights.iter().enumerate() {
            acc += w / total;
            out.push((items[i], acc));
        }
        // Guard against float drift leaving the last bound below a drawn u.
        if let Some(last) = out.last_mut() {
            last.1 = 1.0;
        }
        Ok(out)
    }
}

fn pick<T: Copy>(cdf: &[(T, f64)], u: f64) -> T {
    for (item, bound) in cdf {
        if u < *bound {
            return *item;
        }
    }
    cdf[cdf.len() - 1].0
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
    /// Deck/stake the snapshotted episode was played on. A restored run must
    /// keep its own config -- the observation's deck one-hot comes from the
    /// `Run`, so resampling here would make the two disagree.
    pub config: RunConfig,
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
    /// Deck/stake of the CURRENT episode. Mirrors `self.run.config()`; kept
    /// here so `Episode` can report it after the run has been replaced.
    pub config: RunConfig,
    /// Scheduled deck/stake, applied at the next (auto-)reset — same
    /// contract as `pending_win_ante`, so an in-flight episode never has its
    /// rules changed underneath it.
    pub pending_config: RunConfig,
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
        Self::with_config(seed, win_ante, RunConfig::default())
    }

    pub fn with_config(seed: i64, win_ante: u8, config: RunConfig) -> Self {
        Self::with_config_dist(seed, win_ante, config, &ConfigDist::default())
    }

    /// Build a slot, sampling deck/stake from `dist` exactly the way
    /// `fresh_episode` does — same draw order, same "draw nothing when the
    /// axis is unset" rule. Without this an explicit `reset()` would pin
    /// every env's FIRST episode to the fixed config while auto-resets
    /// sampled, quietly skewing the mixture at the start of every run.
    pub fn with_config_dist(seed: i64, win_ante: u8, config: RunConfig, dist: &ConfigDist) -> Self {
        let mut stream = seed as u64;
        let run_seed = seed_string(splitmix64(&mut stream));
        let mut config = config;
        if let Some(cdf) = &dist.deck_cdf {
            let u = (splitmix64(&mut stream) >> 11) as f64 / (1u64 << 53) as f64;
            config.deck = pick(cdf, u);
        }
        if let Some(cdf) = &dist.stake_cdf {
            let u = (splitmix64(&mut stream) >> 11) as f64 / (1u64 << 53) as f64;
            config.stake = pick(cdf, u);
        }
        let mut slot = EnvSlot {
            run: Run::with_config(&run_seed, config),
            seed_stream: stream,
            ep_return: 0.0,
            ep_len: 0,
            win_ante,
            pending_win_ante: win_ante,
            config,
            pending_config: config,
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

    /// Schedule a new deck/stake; takes effect at the next (auto-)reset.
    pub fn set_config(&mut self, config: RunConfig) {
        self.pending_config = config;
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

    /// Start a fresh seeded episode, applying the scheduled deck/stake.
    ///
    /// # Draw order (documented scheme)
    /// The run seed is drawn FIRST, then the deck, then the stake — and the
    /// deck/stake draws happen only when a distribution is configured for
    /// that axis. So with no distribution the stream advances exactly as it
    /// did pre-P7, and even with one the run-seed sequence is untouched,
    /// which keeps seeds comparable across mixture settings.
    ///
    /// # win_ante is deliberately NOT copied into the RunConfig
    /// `EnvSlot::win_ante` is the curriculum goalpost — it decides when the
    /// episode ENDS. `RunConfig::win_ante` feeds `get_new_boss`, which gates
    /// showdown bosses on `ante == win_ante`, so writing the curriculum's
    /// value into it would change the boss pool (and therefore the RNG
    /// stream) on every rung below 8. The two stay separate: the config
    /// keeps the real game's 8 unless a caller sets it explicitly.
    fn fresh_episode(&mut self, dist: &ConfigDist) {
        let run_seed = seed_string(splitmix64(&mut self.seed_stream));
        let mut config = self.pending_config;
        if let Some(cdf) = &dist.deck_cdf {
            let u = (splitmix64(&mut self.seed_stream) >> 11) as f64 / (1u64 << 53) as f64;
            config.deck = pick(cdf, u);
        }
        if let Some(cdf) = &dist.stake_cdf {
            let u = (splitmix64(&mut self.seed_stream) >> 11) as f64 / (1u64 << 53) as f64;
            config.stake = pick(cdf, u);
        }
        self.run = Run::with_config(&run_seed, config);
        self.config = config;
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
    pub fn begin_episode(&mut self, pool: Option<&SnapshotPool>, fraction: f64, dist: &ConfigDist) {
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
                        // The snapshot's deck/stake win: the observation's
                        // deck one-hot is encoded from the `Run`, so
                        // overriding here would make the two disagree. A
                        // mixture therefore needs per-deck pool coverage, not
                        // resampling.
                        self.config = self.run.config();
                        self.ep_return = 0.0;
                        self.ep_len = 0;
                        self.win_ante = self.pending_win_ante;
                        self.from_snapshot = true;
                        return;
                    }
                }
            }
        }
        self.fresh_episode(dist);
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
        self.step_with_pool(
            env_idx,
            row,
            beta,
            strict,
            None,
            0.0,
            &ConfigDist::default(),
        )
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
        dist: &ConfigDist,
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
                // Read BEFORE begin_episode replaces the run: this describes
                // the episode that just ended, not the one starting.
                deck: self.config.deck,
                stake: self.config.stake,
            });
            self.begin_episode(pool, snapshot_fraction, dist);
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
            config: self.config,
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
        // Same reasoning as the pool branch of `begin_episode`: the run
        // carries its own deck/stake and the observation is encoded from it.
        self.config = snap.config;
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
