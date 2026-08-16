//! File-backed snapshot start-state pool (the plan's M3 backward
//! curriculum): ~50k serialized mid-run states that auto-resets can start
//! episodes from (see `env::EnvSlot::begin_episode`).
//!
//! # File format (little-endian), version 1
//! Written by `train/balatro_train/snapshot_pool.py` (the Python side owns
//! writing; this module only reads):
//!
//! ```text
//! magic    8 bytes   b"BLTRPOOL"
//! version  u32       1
//! count    u32       >= 1
//! antes    count x u8            entry i's run.ante() at snapshot time
//! offsets  (count+1) x u64       byte offsets into the blob section
//!                                (offsets[0] == 0, strictly increasing,
//!                                offsets[count] == blob length)
//! blob     concatenated `env::Snapshot` bincode payloads
//! ```
//!
//! # Sharing
//! One pool is loaded per vec-env and handed to every env slot behind an
//! `Arc` — 50k x ~KB matters at N=2048, so there is exactly one in-memory
//! copy per `BalatroVecEnv` (eval envs never load a pool at all).
//!
//! # win_ante-eligible sampling
//! A snapshot at `ante >= win_ante` would start an episode at/past the
//! curriculum goal, so only entries with `ante < win_ante` are eligible.
//! Entry indices are pre-sorted by ante at load time; the eligible set for
//! any `win_ante` is a prefix of that order, giving O(1) uniform sampling.

use rayon::prelude::*;

use crate::env::Snapshot;

pub const POOL_MAGIC: &[u8; 8] = b"BLTRPOOL";
pub const POOL_VERSION: u32 = 1;

pub struct SnapshotPool {
    antes: Vec<u8>,
    /// `offsets[i]..offsets[i+1]` is entry i's byte range in `blob`.
    offsets: Vec<usize>,
    blob: Vec<u8>,
    /// Entry indices sorted by ante (stable).
    by_ante: Vec<u32>,
    /// `cum[a]` = number of entries with ante < a — the eligible prefix
    /// length of `by_ante` when the curriculum goal is `win_ante = a`.
    cum: [usize; 10],
}

impl SnapshotPool {
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("snapshot pool {path:?}: {e}"))?;
        Self::parse(&bytes).map_err(|e| format!("snapshot pool {path:?}: {e}"))
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut pos = 0usize;
        let take = |pos: &mut usize, n: usize| -> Result<&[u8], String> {
            let end = pos
                .checked_add(n)
                .filter(|&e| e <= bytes.len())
                .ok_or_else(|| "truncated pool file".to_string())?;
            let s = &bytes[*pos..end];
            *pos = end;
            Ok(s)
        };

        if take(&mut pos, 8)? != POOL_MAGIC {
            return Err("bad magic (not a BLTRPOOL file)".into());
        }
        let version = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap());
        if version != POOL_VERSION {
            return Err(format!("unsupported pool version {version}"));
        }
        let count = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
        if count == 0 {
            return Err("empty pool".into());
        }

        let antes = take(&mut pos, count)?.to_vec();
        for (i, &a) in antes.iter().enumerate() {
            if !(1..=8).contains(&a) {
                return Err(format!("entry {i}: ante {a} out of 1..=8"));
            }
        }

        let mut offsets = Vec::with_capacity(count + 1);
        for _ in 0..=count {
            offsets.push(u64::from_le_bytes(take(&mut pos, 8)?.try_into().unwrap()) as usize);
        }
        let blob = bytes[pos..].to_vec();
        if offsets[0] != 0 || offsets[count] != blob.len() {
            return Err("offset table does not span the blob".into());
        }
        if offsets.windows(2).any(|w| w[0] >= w[1]) {
            return Err("offsets not strictly increasing".into());
        }

        // Validate every entry once at load: it must decode as a Snapshot
        // and its run's ante must match the header (the eligibility table
        // is trusted at reset time).
        let errs: Vec<String> = (0..count)
            .into_par_iter()
            .filter_map(|i| {
                let entry = &blob[offsets[i]..offsets[i + 1]];
                match bincode::deserialize::<Snapshot>(entry) {
                    Err(e) => Some(format!("entry {i}: decode failed: {e}")),
                    Ok(snap) if snap.run.ante() != i64::from(antes[i]) => Some(format!(
                        "entry {i}: header ante {} != run ante {}",
                        antes[i],
                        snap.run.ante()
                    )),
                    Ok(_) => None,
                }
            })
            .collect();
        if let Some(e) = errs.first() {
            return Err(e.clone());
        }

        let mut by_ante: Vec<u32> = (0..count as u32).collect();
        by_ante.sort_by_key(|&i| antes[i as usize]);
        let mut cum = [0usize; 10];
        for a in 1..10usize {
            cum[a] = cum[a - 1] + antes.iter().filter(|&&x| x as usize == a - 1).count();
        }

        Ok(SnapshotPool {
            antes,
            offsets,
            blob,
            by_ante,
            cum,
        })
    }

    pub fn len(&self) -> usize {
        self.antes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.antes.is_empty()
    }

    pub fn ante(&self, i: usize) -> u8 {
        self.antes[i]
    }

    pub fn get(&self, i: usize) -> &[u8] {
        &self.blob[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Uniformly sample an entry with `ante < win_ante` using the raw u64
    /// `r` (drawn from the env's seed stream). `None` when no entry
    /// qualifies — the caller falls back to a fresh episode.
    pub fn sample_eligible(&self, win_ante: u8, r: u64) -> Option<usize> {
        let n = self.cum[usize::from(win_ante.min(9))];
        if n == 0 {
            return None;
        }
        Some(self.by_ante[(r % n as u64) as usize] as usize)
    }
}

/// Test/tooling helper: pack `(ante, snapshot_bytes)` entries into the
/// version-1 pool format (the mirror of `snapshot_pool.py`'s writer).
pub fn pack_pool(entries: &[(u8, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(POOL_MAGIC);
    out.extend_from_slice(&POOL_VERSION.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (ante, _) in entries {
        out.push(*ante);
    }
    let mut off = 0u64;
    out.extend_from_slice(&off.to_le_bytes());
    for (_, bytes) in entries {
        off += bytes.len() as u64;
        out.extend_from_slice(&off.to_le_bytes());
    }
    for (_, bytes) in entries {
        out.extend_from_slice(bytes);
    }
    out
}
