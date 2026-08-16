//! Serde helpers for full-`Run` snapshot/restore (P6 PyO3 vec-env).
//!
//! The run state holds `&'static str` keys (and one `&'static BlindProto`)
//! pointing into the item registries. They serialize as plain strings and
//! deserialize by re-interning against the registries, so a snapshot taken
//! by one process restores bit-identically in another.
//!
//! Only compiled with the `serde` feature; nothing here touches game logic.

use std::collections::{HashMap, HashSet};

use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::blinds::{blind_by_key, BlindProto};
use crate::items::{booster_by_key, consumable_by_key, joker_by_key, tag_by_key, voucher_by_key};

/// Re-intern a serialized key against every static registry the run state
/// can reference. `""` is the empty-tag placeholder used before the first
/// blind tags are rolled; `gros_michel_extinct` is the only pool flag.
pub fn intern_key(s: &str) -> Option<&'static str> {
    if s.is_empty() {
        return Some("");
    }
    if s == "gros_michel_extinct" {
        return Some("gros_michel_extinct");
    }
    if let Some(t) = tag_by_key(s) {
        return Some(t.key);
    }
    if let Some(v) = voucher_by_key(s) {
        return Some(v.key);
    }
    if let Some(b) = booster_by_key(s) {
        return Some(b.key);
    }
    if let Some((c, _)) = consumable_by_key(s) {
        return Some(c.key);
    }
    if let Some(j) = joker_by_key(s) {
        return Some(j.key);
    }
    blind_by_key(s).map(|b| b.key)
}

fn intern_or_err<E: DeError>(s: &str) -> Result<&'static str, E> {
    intern_key(s).ok_or_else(|| E::custom(format!("unknown static key {s:?}")))
}

/// `&'static str` field.
pub mod key {
    use super::*;

    pub fn serialize<S: Serializer>(v: &str, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(v)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<&'static str, D::Error> {
        let s = String::deserialize(d)?;
        intern_or_err(&s)
    }
}

/// `Option<&'static str>` field.
pub mod opt_key {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Option<&'static str>, s: S) -> Result<S::Ok, S::Error> {
        v.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<&'static str>, D::Error> {
        let s = Option::<String>::deserialize(d)?;
        s.map(|s| intern_or_err(&s)).transpose()
    }
}

/// `HashMap<&'static str, V>` field.
pub mod key_map {
    use super::*;

    pub fn serialize<S: Serializer, V: Serialize>(
        v: &HashMap<&'static str, V>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        // Sort for stable bytes (HashMap iteration order is nondeterministic).
        let mut entries: Vec<(&str, &V)> = v.iter().map(|(k, val)| (*k, val)).collect();
        entries.sort_by_key(|(k, _)| *k);
        entries.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>, V: Deserialize<'de>>(
        d: D,
    ) -> Result<HashMap<&'static str, V>, D::Error> {
        let entries = Vec::<(String, V)>::deserialize(d)?;
        entries
            .into_iter()
            .map(|(k, v)| Ok((intern_or_err(&k)?, v)))
            .collect()
    }
}

/// `HashSet<&'static str>` field.
pub mod key_set {
    use super::*;

    pub fn serialize<S: Serializer>(v: &HashSet<&'static str>, s: S) -> Result<S::Ok, S::Error> {
        let mut entries: Vec<&str> = v.iter().copied().collect();
        entries.sort_unstable();
        entries.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<HashSet<&'static str>, D::Error> {
        let entries = Vec::<String>::deserialize(d)?;
        entries.iter().map(|k| intern_or_err(k)).collect()
    }
}

/// `&'static BlindProto` field (serialized by key).
pub mod blind_proto {
    use super::*;

    pub fn serialize<S: Serializer>(v: &&'static BlindProto, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(v.key)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<&'static BlindProto, D::Error> {
        let s = String::deserialize(d)?;
        blind_by_key(&s).ok_or_else(|| D::Error::custom(format!("unknown blind key {s:?}")))
    }
}
