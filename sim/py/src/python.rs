//! PyO3 surface: `balatro_sim.BalatroVecEnv`, implementing the
//! `balatro_train.env_api.VecEnv` contract (through the thin
//! `balatro_train.sim_env` wrapper).

use std::sync::Arc;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rayon::prelude::*;

use crate::action::ActionRow;
use crate::bot::bot_action;
use crate::consts::*;
use crate::encode::{EnvMasks, EnvObs};
use crate::env::{EnvSlot, Episode};
use crate::pool::SnapshotPool;

#[pyclass]
pub struct BalatroVecEnv {
    slots: Vec<EnvSlot>,
    num_envs: usize,
    beta: f64,
    strict: bool,
    win_ante: u8,
    /// Snapshot start-state mixing (M3): one shared pool + the probability
    /// that an auto-reset starts from it. See `env::EnvSlot::begin_episode`.
    pool: Option<Arc<SnapshotPool>>,
    snapshot_fraction: f64,
}

fn check_win_ante(win_ante: u8) -> PyResult<()> {
    if !(1..=8).contains(&win_ante) {
        return Err(PyValueError::new_err(format!(
            "win_ante must be in 1..=8, got {win_ante}"
        )));
    }
    Ok(())
}

type ObsMasks<'py> = (Bound<'py, PyDict>, Bound<'py, PyDict>);

fn build_obs_dict<'py>(py: Python<'py>, obs: &[EnvObs]) -> PyResult<Bound<'py, PyDict>> {
    let n = obs.len();
    let d = PyDict::new(py);

    macro_rules! f32_3d {
        ($key:literal, $field:ident, $d1:expr, $d2:expr) => {{
            let mut v: Vec<f32> = Vec::with_capacity(n * $d1 * $d2);
            for o in obs {
                v.extend_from_slice(&o.$field);
            }
            let arr = PyArray1::from_vec(py, v).reshape([n, $d1, $d2])?;
            d.set_item($key, arr)?;
        }};
    }
    macro_rules! f32_2d {
        ($key:literal, $field:ident, $d1:expr) => {{
            let mut v: Vec<f32> = Vec::with_capacity(n * $d1);
            for o in obs {
                v.extend_from_slice(&o.$field);
            }
            let arr = PyArray1::from_vec(py, v).reshape([n, $d1])?;
            d.set_item($key, arr)?;
        }};
    }
    macro_rules! i64_2d {
        ($key:literal, $field:ident, $d1:expr) => {{
            let mut v: Vec<i64> = Vec::with_capacity(n * $d1);
            for o in obs {
                v.extend_from_slice(&o.$field);
            }
            let arr = PyArray1::from_vec(py, v).reshape([n, $d1])?;
            d.set_item($key, arr)?;
        }};
    }
    macro_rules! i64_1d {
        ($key:literal, $field:ident) => {{
            let v: Vec<i64> = obs.iter().map(|o| o.$field).collect();
            d.set_item($key, PyArray1::from_vec(py, v))?;
        }};
    }

    f32_3d!("hand", hand, HAND_MAX, F_CARD);
    i64_1d!("hand_len", hand_len);
    i64_2d!("joker_ids", joker_ids, JOKER_SLOTS);
    f32_3d!("joker_feats", joker_feats, JOKER_SLOTS, F_JOKER_FEAT);
    i64_2d!("consumable_ids", consumable_ids, CONSUMABLE_SLOTS);
    i64_1d!("consumables_len", consumables_len);
    i64_2d!("shop_ids", shop_ids, SHOP_SLOTS);
    f32_3d!("shop_feats", shop_feats, SHOP_SLOTS, F_SHOP_FEAT);
    f32_2d!("blind", blind, F_BLIND);
    f32_2d!("global", global, F_GLOBAL);
    f32_2d!("deck_counts", deck_counts, N_DECK_SLOTS);
    f32_2d!("deck_aggregates", deck_aggregates, F_DECK_AGG);
    f32_2d!("drawpile_counts", drawpile_counts, N_DECK_SLOTS);
    Ok(d)
}

fn build_mask_dict<'py>(py: Python<'py>, masks: &[EnvMasks]) -> PyResult<Bound<'py, PyDict>> {
    let n = masks.len();
    let d = PyDict::new(py);
    macro_rules! bool_2d {
        ($key:literal, $field:ident, $d1:expr) => {{
            let mut v: Vec<bool> = Vec::with_capacity(n * $d1);
            for m in masks {
                v.extend_from_slice(&m.$field);
            }
            let arr = PyArray1::from_vec(py, v).reshape([n, $d1])?;
            d.set_item($key, arr)?;
        }};
    }
    bool_2d!("action_type_mask", action_type, N_ACTION_TYPES);
    bool_2d!("card_select_mask", card_select, HAND_MAX);
    bool_2d!("joker_target_mask", joker_target, JOKER_SLOTS);
    bool_2d!(
        "consumable_target_mask",
        consumable_target,
        CONSUMABLE_SLOTS
    );
    bool_2d!("shop_target_mask", shop_target, SHOP_SLOTS);
    bool_2d!("pack_target_mask", pack_target, PACK_SLOTS);
    Ok(d)
}

fn action_row_dict<'py>(py: Python<'py>, r: &ActionRow) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("action_type", r.action_type)?;
    d.set_item("cards", r.cards.to_vec())?;
    d.set_item("n_cards", r.n_cards)?;
    d.set_item("joker_target", r.joker_target)?;
    d.set_item("consumable_target", r.consumable_target)?;
    d.set_item("shop_target", r.shop_target)?;
    d.set_item("pack_target", r.pack_target)?;
    Ok(d)
}

fn parse_actions(actions: &Bound<'_, PyDict>, n: usize) -> PyResult<Vec<ActionRow>> {
    fn get1(actions: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<i64>> {
        let item = actions
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("actions missing key {key:?}")))?;
        let arr: PyReadonlyArray1<i64> = item.extract()?;
        Ok(arr.as_array().iter().copied().collect())
    }
    let item = actions
        .get_item("cards")?
        .ok_or_else(|| PyValueError::new_err("actions missing key 'cards'"))?;
    let cards_arr: PyReadonlyArray2<i64> = item.extract()?;
    let cards = cards_arr.as_array();
    if cards.shape() != [n, MAX_CARD_PICKS] {
        return Err(PyValueError::new_err(format!(
            "actions['cards'] shape {:?} != ({n}, {MAX_CARD_PICKS})",
            cards.shape()
        )));
    }

    let action_type = get1(actions, "action_type")?;
    let n_cards = get1(actions, "n_cards")?;
    let joker_target = get1(actions, "joker_target")?;
    let consumable_target = get1(actions, "consumable_target")?;
    let shop_target = get1(actions, "shop_target")?;
    let pack_target = get1(actions, "pack_target")?;
    for (key, v) in [
        ("action_type", &action_type),
        ("n_cards", &n_cards),
        ("joker_target", &joker_target),
        ("consumable_target", &consumable_target),
        ("shop_target", &shop_target),
        ("pack_target", &pack_target),
    ] {
        if v.len() != n {
            return Err(PyValueError::new_err(format!(
                "actions[{key:?}] length {} != num_envs {n}",
                v.len()
            )));
        }
    }

    Ok((0..n)
        .map(|i| {
            let mut row = ActionRow {
                action_type: action_type[i],
                cards: [-1; MAX_CARD_PICKS],
                n_cards: n_cards[i],
                joker_target: joker_target[i],
                consumable_target: consumable_target[i],
                shop_target: shop_target[i],
                pack_target: pack_target[i],
            };
            for k in 0..MAX_CARD_PICKS {
                row.cards[k] = cards[[i, k]];
            }
            row
        })
        .collect())
}

impl BalatroVecEnv {
    fn observe_all<'py>(&self, py: Python<'py>) -> PyResult<ObsMasks<'py>> {
        let obs: Vec<EnvObs> =
            py.allow_threads(|| self.slots.par_iter().map(|s| s.observe()).collect());
        let masks: Vec<EnvMasks> = self.slots.iter().map(|s| s.masks).collect();
        Ok((build_obs_dict(py, &obs)?, build_mask_dict(py, &masks)?))
    }

    fn check_started(&self) -> PyResult<()> {
        if self.slots.is_empty() {
            return Err(PyRuntimeError::new_err("step()/observe() before reset()"));
        }
        Ok(())
    }

    fn check_idx(&self, env_idx: usize) -> PyResult<()> {
        self.check_started()?;
        if env_idx >= self.num_envs {
            return Err(PyValueError::new_err(format!(
                "env_idx {env_idx} out of range ({})",
                self.num_envs
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl BalatroVecEnv {
    #[new]
    #[pyo3(signature = (num_envs, strict = true, win_ante = 8))]
    fn new(num_envs: usize, strict: bool, win_ante: u8) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be >= 1"));
        }
        check_win_ante(win_ante)?;
        Ok(BalatroVecEnv {
            slots: Vec::new(),
            num_envs,
            beta: 1.0,
            strict,
            win_ante,
            pool: None,
            snapshot_fraction: 0.0,
        })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.num_envs
    }

    fn set_shaping_beta(&mut self, beta: f64) {
        self.beta = beta;
    }

    /// Curriculum knob: episodes end with the +15 win bonus once the boss
    /// blind of ante `win_ante` is defeated (8 = the real game's win).
    /// Applies at each env's NEXT (auto-)reset — episodes already in flight
    /// keep the goalpost they started with. A full `reset()` applies it to
    /// every env immediately.
    fn set_win_ante(&mut self, win_ante: u8) -> PyResult<()> {
        check_win_ante(win_ante)?;
        self.win_ante = win_ante;
        for slot in &mut self.slots {
            slot.set_win_ante(win_ante);
        }
        Ok(())
    }

    #[getter]
    fn win_ante(&self) -> u8 {
        self.win_ante
    }

    /// Load a snapshot start-state pool from a packed pool file (format:
    /// `sim/py/src/pool.rs` docs; written by
    /// `train/balatro_train/snapshot_pool.py`). The pool is validated and
    /// held in ONE in-memory copy shared by every env slot. Returns the
    /// entry count.
    fn load_snapshot_pool(&mut self, py: Python<'_>, path: &str) -> PyResult<usize> {
        let pool = py
            .allow_threads(|| SnapshotPool::load(path))
            .map_err(PyValueError::new_err)?;
        let n = pool.len();
        self.pool = Some(Arc::new(pool));
        Ok(n)
    }

    /// Set the probability that an auto-reset starts from a sampled pool
    /// snapshot instead of a fresh seeded run (trainer anneal hook).
    /// Requires a loaded pool when `f > 0`. Applies to auto-resets only;
    /// `reset()` always starts fresh runs. Reproducibility + win_ante
    /// semantics: `env::EnvSlot::begin_episode` docs.
    fn set_snapshot_fraction(&mut self, f: f64) -> PyResult<()> {
        if !(0.0..=1.0).contains(&f) || !f.is_finite() {
            return Err(PyValueError::new_err(format!(
                "snapshot fraction must be in [0, 1], got {f}"
            )));
        }
        if f > 0.0 && self.pool.is_none() {
            return Err(PyValueError::new_err(
                "snapshot fraction > 0 requires load_snapshot_pool() first",
            ));
        }
        self.snapshot_fraction = f;
        Ok(())
    }

    #[getter]
    fn snapshot_fraction(&self) -> f64 {
        self.snapshot_fraction
    }

    #[getter]
    fn snapshot_pool_len(&self) -> usize {
        self.pool.as_ref().map_or(0, |p| p.len())
    }

    /// Start fresh runs from `seeds` (one per env). Returns (obs, masks).
    fn reset<'py>(&mut self, py: Python<'py>, seeds: Vec<i64>) -> PyResult<ObsMasks<'py>> {
        if seeds.len() != self.num_envs {
            return Err(PyValueError::new_err(format!(
                "need {} seeds, got {}",
                self.num_envs,
                seeds.len()
            )));
        }
        let win_ante = self.win_ante;
        self.slots = py.allow_threads(|| {
            seeds
                .into_par_iter()
                .map(|s| EnvSlot::with_win_ante(s, win_ante))
                .collect::<Vec<_>>()
        });
        self.observe_all(py)
    }

    /// Advance every env one composite action.
    /// Returns (obs, masks, reward, done, infos).
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyDict>,
    ) -> PyResult<(
        Bound<'py, PyDict>,
        Bound<'py, PyDict>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyList>,
    )> {
        self.check_started()?;
        let rows = parse_actions(actions, self.num_envs)?;
        let beta = self.beta;
        let strict = self.strict;
        let pool = self.pool.clone();
        let fraction = self.snapshot_fraction;

        type StepRow = Result<(f32, bool, Option<Episode>, EnvObs, EnvMasks), String>;
        let results: Vec<StepRow> = py.allow_threads(|| {
            self.slots
                .par_iter_mut()
                .zip(rows.par_iter())
                .enumerate()
                .map(|(i, (slot, row))| {
                    let (reward, done, info) =
                        slot.step_with_pool(i, row, beta, strict, pool.as_deref(), fraction)?;
                    Ok((reward, done, info, slot.observe(), slot.masks))
                })
                .collect()
        });

        let mut rewards = Vec::with_capacity(self.num_envs);
        let mut dones = Vec::with_capacity(self.num_envs);
        let mut obs = Vec::with_capacity(self.num_envs);
        let mut masks = Vec::with_capacity(self.num_envs);
        let infos = PyList::empty(py);
        for r in results {
            let (reward, done, info, o, m) = r.map_err(PyValueError::new_err)?;
            rewards.push(reward);
            dones.push(done);
            obs.push(o);
            masks.push(m);
            let d = PyDict::new(py);
            if let Some(ep) = info {
                let e = PyDict::new(py);
                e.set_item("r", ep.r)?;
                e.set_item("l", ep.l)?;
                e.set_item("ante", ep.ante)?;
                e.set_item("won", ep.won)?;
                e.set_item("from_snapshot", ep.from_snapshot)?;
                d.set_item("episode", e)?;
            }
            infos.append(d)?;
        }

        Ok((
            build_obs_dict(py, &obs)?,
            build_mask_dict(py, &masks)?,
            PyArray1::from_vec(py, rewards),
            PyArray1::from_vec(py, dones),
            infos,
        ))
    }

    /// Re-emit (obs, masks) for the current states (after restore /
    /// redeterminize, or for eval harnesses).
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<ObsMasks<'py>> {
        self.check_started()?;
        self.observe_all(py)
    }

    /// Full serialized state of env `env_idx` (run + episode bookkeeping +
    /// reset-seed stream), restorable in any process.
    fn snapshot<'py>(&self, py: Python<'py>, env_idx: usize) -> PyResult<Bound<'py, PyBytes>> {
        self.check_idx(env_idx)?;
        Ok(PyBytes::new(py, &self.slots[env_idx].snapshot_bytes()))
    }

    fn restore(&mut self, env_idx: usize, snapshot: &[u8]) -> PyResult<()> {
        self.check_idx(env_idx)?;
        self.slots[env_idx]
            .restore_bytes(snapshot)
            .map_err(PyValueError::new_err)
    }

    /// Re-randomize the unobserved future of env `env_idx` (undrawn deck
    /// order + all RNG stream states) from a fresh seed; observable state
    /// is unchanged. For determinized search.
    fn redeterminize(&mut self, env_idx: usize, seed: i64) -> PyResult<()> {
        self.check_idx(env_idx)?;
        self.slots[env_idx].redeterminize(seed);
        Ok(())
    }

    /// The scripted baseline's action for env `env_idx`, as a scalar action
    /// dict (feed through `step` alongside other envs' actions).
    fn bot_action<'py>(&self, py: Python<'py>, env_idx: usize) -> PyResult<Bound<'py, PyDict>> {
        self.check_idx(env_idx)?;
        let s = &self.slots[env_idx];
        let row = bot_action(&s.run, &s.masks, &s.shop_slots);
        action_row_dict(py, &row)
    }

    /// Batched bot actions for every env, matching `encoding.ACTION_SPEC`.
    fn bot_actions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.check_started()?;
        let rows: Vec<ActionRow> = py.allow_threads(|| {
            self.slots
                .par_iter()
                .map(|s| bot_action(&s.run, &s.masks, &s.shop_slots))
                .collect()
        });
        let n = rows.len();
        let d = PyDict::new(py);
        let mut cards = Vec::with_capacity(n * MAX_CARD_PICKS);
        for r in &rows {
            cards.extend_from_slice(&r.cards);
        }
        d.set_item(
            "cards",
            PyArray1::from_vec(py, cards).reshape([n, MAX_CARD_PICKS])?,
        )?;
        macro_rules! field {
            ($key:literal, $f:ident) => {
                d.set_item(
                    $key,
                    PyArray1::from_vec(py, rows.iter().map(|r| r.$f).collect::<Vec<i64>>()),
                )?;
            };
        }
        field!("action_type", action_type);
        field!("n_cards", n_cards);
        field!("joker_target", joker_target);
        field!("consumable_target", consumable_target);
        field!("shop_target", shop_target);
        field!("pack_target", pack_target);
        Ok(d)
    }

    /// The Balatro seed string of env `env_idx`'s current run (valid in the
    /// real game — P5 live replay).
    fn run_seed(&self, env_idx: usize) -> PyResult<String> {
        self.check_idx(env_idx)?;
        Ok(self.slots[env_idx].run.seed().to_string())
    }

    /// Cheap scalar summary of env `env_idx`'s current run — snapshot
    /// harvesting (gen_snapshots.py) uses it to detect trigger points
    /// without decoding observations. Keys: seed (str), ante (int),
    /// round (int), state (balatrobot state name, e.g. "SHOP").
    fn run_info<'py>(&self, py: Python<'py>, env_idx: usize) -> PyResult<Bound<'py, PyDict>> {
        self.check_idx(env_idx)?;
        let run = &self.slots[env_idx].run;
        let d = PyDict::new(py);
        d.set_item("seed", run.seed())?;
        d.set_item("ante", run.ante())?;
        d.set_item("round", run.round())?;
        d.set_item("state", crate::export::state_name(run.state()))?;
        Ok(d)
    }

    /// Full observable run state of env `env_idx` as a JSON document whose
    /// schema maps onto balatrobot's `gamestate` (see export.rs) — P5
    /// cross-validation.
    fn state_json(&self, env_idx: usize) -> PyResult<String> {
        self.check_idx(env_idx)?;
        Ok(crate::export::state_json(&self.slots[env_idx].run).to_string())
    }
}

/// P5 cross-validation: a single seeded run driven through the DIRECT
/// Run-level action interface (no RL obs/action encoding). Action documents
/// are JSON dicts (`{"kind": "play", "cards": [0, 1]}` ...) matching
/// `legal_actions_json` / `bot_action_json` output; state documents map
/// onto balatrobot's `gamestate` schema.
#[pyclass]
pub struct CrossvalRun {
    run: balatro_core::run::Run,
}

#[pymethods]
impl CrossvalRun {
    #[new]
    fn new(seed: &str) -> PyResult<Self> {
        if seed.is_empty() || seed.len() > 8 || !seed.bytes().all(|b| b.is_ascii_alphanumeric()) {
            return Err(PyValueError::new_err(format!(
                "seed must be 1-8 alphanumeric chars, got {seed:?}"
            )));
        }
        Ok(CrossvalRun {
            run: balatro_core::run::Run::new(&seed.to_uppercase()),
        })
    }

    fn seed(&self) -> String {
        self.run.seed().to_string()
    }

    /// Current state name (balatrobot enum: SELECTING_HAND, SHOP, ...).
    fn state(&self) -> &'static str {
        crate::export::state_name(self.run.state())
    }

    fn ante(&self) -> i64 {
        self.run.ante()
    }

    fn is_over(&self) -> bool {
        matches!(
            self.run.state(),
            balatro_core::run::State::GameOver | balatro_core::run::State::Won
        )
    }

    /// Full observable state as a JSON string (schema: export.rs docs).
    fn state_json(&self) -> String {
        crate::export::state_json(&self.run).to_string()
    }

    /// Legal action descriptors as a JSON array.
    fn legal_actions_json(&self) -> String {
        crate::export::legal_actions_json(&self.run).to_string()
    }

    /// Apply one direct action (JSON dict string). Raises ValueError on an
    /// illegal action; the run state is unchanged in that case.
    fn apply_json(&mut self, action: &str) -> PyResult<()> {
        let v: serde_json::Value = serde_json::from_str(action)
            .map_err(|e| PyValueError::new_err(format!("bad action JSON: {e}")))?;
        crate::export::apply_action(&mut self.run, &v).map_err(PyValueError::new_err)
    }

    /// The scripted baseline's next action as a JSON dict string with fully
    /// resolved indices (mirror it verbatim onto the game).
    fn bot_action_json(&self) -> PyResult<String> {
        crate::export::bot_direct_action(&self.run)
            .map(|v| v.to_string())
            .map_err(PyValueError::new_err)
    }

    /// Encoded (obs, masks) for the current state as batch-of-1 dicts in the
    /// frozen RL env contract — feed straight to a trained policy.
    fn observe_encoded<'py>(&self, py: Python<'py>) -> PyResult<ObsMasks<'py>> {
        let slots = crate::encode::shop_slot_map(&self.run);
        let obs = [crate::encode::encode_obs(&self.run)];
        let masks = [crate::encode::encode_masks(&self.run, &slots)];
        Ok((build_obs_dict(py, &obs)?, build_mask_dict(py, &masks)?))
    }

    /// Translate one RL-contract action row (a trained policy's output for
    /// this state) into a direct-action JSON dict string, resolving indices
    /// exactly like the vec-env would execute it. Returns "null" for the
    /// USE_CONSUMABLE no-op leniency (legal per mask, unusable right now).
    #[pyo3(signature = (action_type, cards, joker_target=-1, consumable_target=-1, shop_target=-1, pack_target=-1))]
    fn action_json(
        &self,
        action_type: i64,
        cards: Vec<i64>,
        joker_target: i64,
        consumable_target: i64,
        shop_target: i64,
        pack_target: i64,
    ) -> PyResult<String> {
        if cards.len() > MAX_CARD_PICKS {
            return Err(PyValueError::new_err(format!(
                "cards has {} picks, max {MAX_CARD_PICKS}",
                cards.len()
            )));
        }
        let mut row = ActionRow {
            action_type,
            cards: [-1; MAX_CARD_PICKS],
            n_cards: cards.len() as i64,
            joker_target,
            consumable_target,
            shop_target,
            pack_target,
        };
        row.cards[..cards.len()].copy_from_slice(&cards);
        crate::export::row_direct_action(&self.run, &row)
            .map(|v| v.to_string())
            .map_err(PyValueError::new_err)
    }
}

#[pymodule]
fn balatro_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BalatroVecEnv>()?;
    m.add_class::<CrossvalRun>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
