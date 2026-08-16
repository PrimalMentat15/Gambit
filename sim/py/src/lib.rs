//! `balatro-sim-py`: PyO3 vectorized Balatro RL environment (project phase
//! P6) binding `balatro-core` to the frozen training contract in
//! `train/balatro_train/encoding.py` / `env_api.py`.
//!
//! Crate layout:
//! * [`consts`]  — Rust mirror of encoding.py (offsets, vocab id maps).
//! * [`encode`]  — Run -> observation/mask encoders + shop slot mapping.
//! * [`action`]  — composite action dict -> sim calls, leniency rules,
//!   reward shaping.
//! * [`env`]     — per-env slot: seeding, auto-reset, snapshot/restore,
//!   redeterminize, snapshot start-state mixing.
//! * [`pool`]    — file-backed snapshot start-state pool (M3).
//! * [`bot`]     — scripted baseline emitting contract actions.
//! * `python`    — the `balatro_sim` extension module (feature `python`).
//!
//! The pure-Rust modules carry the unit tests (`cargo test` runs without
//! linking libpython); maturin builds the cdylib with
//! `--features extension-module`.

pub mod action;
pub mod bot;
pub mod consts;
pub mod encode;
pub mod env;
pub mod export;
pub mod pool;
#[cfg(feature = "python")]
mod python;
