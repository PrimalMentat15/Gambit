pub mod blinds;
pub mod cards;
pub mod config;
pub mod consumables;
pub mod deck;
pub mod handeval;
pub mod items;
pub mod jokers;
// rng.rs intentionally spells out LuaJIT's PI/E seeding constants verbatim
// (lib_math.c); silence the related clippy lints here without touching the
// validated file.
#[allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::doc_overindented_list_items
)]
pub mod rng;
pub mod run;
pub mod scoring;
pub mod shop;
#[cfg(feature = "serde")]
pub mod snapshot;
pub mod tarots;
