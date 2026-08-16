//! Bit-exact port of Balatro's RNG layer.
//!
//! Three pieces, mirroring `functions/misc_functions.lua` and LuaJIT's
//! `math.random`:
//!  1. `pseudohash`   — string -> f64 in [0,1) (or NaN on "toxic" inputs)
//!  2. `RngState`     — per-key `pseudoseed` streams (`G.GAME.pseudorandom`)
//!  3. `LuaRandom`    — LuaJIT 2.1 Tausworthe PRNG behind
//!                      `math.randomseed`/`math.random`
//!
//! Every arithmetic op preserves the Lua evaluation order; f64 IEEE ops are
//! deterministic, so results are bit-identical to the game.

use std::collections::HashMap;

/// `pseudohash(str)` from misc_functions.lua.
///
/// Lua iterates bytes from the end of the string with 1-based index `i`:
/// `num = ((1.1239285023/num)*byte*pi + pi*i) % 1`
pub fn pseudohash(s: &str) -> f64 {
    let bytes = s.as_bytes();
    let mut num: f64 = 1.0;
    for i in (0..bytes.len()).rev() {
        let byte = bytes[i] as f64;
        let lua_i = (i + 1) as f64;
        num = lua_mod1(
            (1.1239285023 / num) * byte * std::f64::consts::PI + std::f64::consts::PI * lua_i,
        );
    }
    num
}

/// Lua's `x % 1` (`x - floor(x/1)*1`). NaN/inf propagate to NaN like LuaJIT.
#[inline]
fn lua_mod1(x: f64) -> f64 {
    x - x.floor()
}

/// Lua's `tonumber(string.format("%.13f", x))` used in `pseudoseed`.
/// Both C printf and Rust `{:.13}` produce the correctly-rounded decimal of
/// the exact binary value, so the round-trip is identical.
#[inline]
fn round13(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    format!("{x:.13}").parse().expect("round13 parse")
}

/// The per-run keyed RNG stream state: `G.GAME.pseudorandom`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RngState {
    seed: String,
    hashed_seed: f64,
    states: HashMap<String, f64>,
}

impl RngState {
    /// Mirrors `start_run`: `hashed_seed = pseudohash(seed)`; per-key streams
    /// are lazily initialized as `pseudohash(key .. seed)`.
    pub fn new(seed: &str) -> Self {
        RngState {
            seed: seed.to_string(),
            hashed_seed: pseudohash(seed),
            states: HashMap::new(),
        }
    }

    pub fn seed(&self) -> &str {
        &self.seed
    }

    /// `pseudoseed(key)`: advance the key's stream and return the value to
    /// feed `math.randomseed`.
    pub fn pseudoseed(&mut self, key: &str) -> f64 {
        let state = self
            .states
            .entry(key.to_string())
            .or_insert_with(|| pseudohash(&format!("{key}{}", self.seed)));
        *state = round13(lua_mod1(2.134453429141 + *state * 1.72431234)).abs();
        (*state + self.hashed_seed) / 2.0
    }

    /// `pseudorandom(key)`: fresh LuaJIT PRNG seeded from the key's stream,
    /// one `math.random()` draw in [0,1).
    pub fn random(&mut self, key: &str) -> f64 {
        LuaRandom::seeded(self.pseudoseed(key)).random()
    }

    /// `pseudorandom(key, min, max)`: one `math.random(min, max)` draw.
    pub fn random_range(&mut self, key: &str, min: i64, max: i64) -> i64 {
        LuaRandom::seeded(self.pseudoseed(key)).random_range(min, max)
    }
}

/// LuaJIT 2.1 `math.random` PRNG: four combined LFSR (Tausworthe) generators,
/// period 2^223. Ported from LuaJIT `lib_math.c` (MIT).
#[derive(Debug, Clone)]
pub struct LuaRandom {
    gen: [u64; 4],
}

impl LuaRandom {
    /// `math.randomseed(d)` — seeds all four generators from a double and
    /// discards 10 warm-up steps.
    pub fn seeded(d: f64) -> Self {
        // 64 - k[i] for k = {63,58,55,47}, packed as four bytes.
        let mut r: u32 = 0x11090601;
        let mut gen = [0u64; 4];
        let mut d = d;
        for slot in gen.iter_mut() {
            let m: u64 = 1u64 << (r & 255);
            r >>= 8;
            d = d * 3.14159265358979323846 + 2.7182818284590452354;
            let mut u = d.to_bits();
            if u < m {
                // Ensure the k[i] MSBs of the generator state are non-zero.
                u += m;
            }
            *slot = u;
        }
        let mut rs = LuaRandom { gen };
        for _ in 0..10 {
            rs.step();
        }
        rs
    }

    #[inline]
    fn step(&mut self) -> u64 {
        let mut r: u64 = 0;
        macro_rules! tw223 {
            ($i:expr, $k:expr, $q:expr, $s:expr) => {{
                let mut z = self.gen[$i];
                z = (((z << $q) ^ z) >> ($k - $s)) ^ ((z & (u64::MAX << (64 - $k))) << $s);
                r ^= z;
                self.gen[$i] = z;
            }};
        }
        tw223!(0, 63, 31, 18);
        tw223!(1, 58, 19, 28);
        tw223!(2, 55, 24, 7);
        tw223!(3, 47, 21, 8);
        r
    }

    /// `math.random()` -> [0,1). LuaJIT builds a double in [1,2) from the top
    /// 52 random bits and subtracts 1.
    pub fn random(&mut self) -> f64 {
        let u = (self.step() & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000;
        f64::from_bits(u) - 1.0
    }

    /// `math.random(m, n)` -> integer in [m, n].
    pub fn random_range(&mut self, m: i64, n: i64) -> i64 {
        let d = self.random();
        (d * ((n - m + 1) as f64)).floor() as i64 + m
    }
}
