//! Run configuration: the deck (Back) and stake a run is played on.
//!
//! Everything here mirrors three places in the game source:
//!
//! * `G.P_CENTERS.b_*` (game.lua:628-642) — the 15 deck prototypes and their
//!   `config` tables.
//! * `Back:apply_to_run` (back.lua:174-278) — how a deck's `config` becomes
//!   run state.
//! * `Game:start_run`'s stake cascade (game.lua:2031-2041) — a flat run of
//!   `>=` tests, which is why [`Stake`] is ordinal and one-based.
//!
//! # Ordering is a wire format
//!
//! [`Deck`] and [`Stake`] discriminants are mirrored in the observation
//! contract (`train/balatro_train/encoding.py`, `sim/py/src/consts.rs`) and
//! encoded into the policy's input as a one-hot and an ordinal. Renumbering
//! them silently invalidates every trained checkpoint, so the values are
//! frozen: [`Deck`] follows each prototype's `order` field zero-based, and
//! [`Stake`] follows `stake_level` one-based.
//!
//! # Why the stake block runs before `apply_to_run`
//!
//! game.lua:2031-2043 applies the stake modifiers first and *then* calls
//! `selected_back:apply_to_run()`. For the base decks the two are commutative
//! (Blue Stake's `discards - 1` and the Red Deck's `discards + 1` land on the
//! same field in either order), but [`RunConfig::resolve`] preserves the game's
//! order rather than relying on that.

use crate::cards::{Rank, Suit};

/// Deck (Back) identity. Discriminants are `order - 1` from game.lua:628-642.
///
/// `b_challenge` (game.lua:644) carries `omit = true` and never enters the
/// Back pool, so it is not represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Deck {
    #[default]
    Red = 0,
    Blue = 1,
    Yellow = 2,
    Green = 3,
    Black = 4,
    Magic = 5,
    Nebula = 6,
    Ghost = 7,
    Abandoned = 8,
    Checkered = 9,
    Zodiac = 10,
    Painted = 11,
    Anaglyph = 12,
    Plasma = 13,
    Erratic = 14,
}

impl Deck {
    pub const ALL: [Deck; 15] = [
        Deck::Red,
        Deck::Blue,
        Deck::Yellow,
        Deck::Green,
        Deck::Black,
        Deck::Magic,
        Deck::Nebula,
        Deck::Ghost,
        Deck::Abandoned,
        Deck::Checkered,
        Deck::Zodiac,
        Deck::Painted,
        Deck::Anaglyph,
        Deck::Plasma,
        Deck::Erratic,
    ];

    /// The `b_*` key, as it appears in `G.P_CENTERS`.
    pub fn key(self) -> &'static str {
        match self {
            Deck::Red => "b_red",
            Deck::Blue => "b_blue",
            Deck::Yellow => "b_yellow",
            Deck::Green => "b_green",
            Deck::Black => "b_black",
            Deck::Magic => "b_magic",
            Deck::Nebula => "b_nebula",
            Deck::Ghost => "b_ghost",
            Deck::Abandoned => "b_abandoned",
            Deck::Checkered => "b_checkered",
            Deck::Zodiac => "b_zodiac",
            Deck::Painted => "b_painted",
            Deck::Anaglyph => "b_anaglyph",
            Deck::Plasma => "b_plasma",
            Deck::Erratic => "b_erratic",
        }
    }

    /// The in-game display name. `Back:trigger_effect` and the Checkered branch
    /// of `apply_to_run` dispatch on this string rather than on `config`
    /// (back.lua:111, :121, :239), so it is load-bearing, not cosmetic.
    pub fn name(self) -> &'static str {
        match self {
            Deck::Red => "Red Deck",
            Deck::Blue => "Blue Deck",
            Deck::Yellow => "Yellow Deck",
            Deck::Green => "Green Deck",
            Deck::Black => "Black Deck",
            Deck::Magic => "Magic Deck",
            Deck::Nebula => "Nebula Deck",
            Deck::Ghost => "Ghost Deck",
            Deck::Abandoned => "Abandoned Deck",
            Deck::Checkered => "Checkered Deck",
            Deck::Zodiac => "Zodiac Deck",
            Deck::Painted => "Painted Deck",
            Deck::Anaglyph => "Anaglyph Deck",
            Deck::Plasma => "Plasma Deck",
            Deck::Erratic => "Erratic Deck",
        }
    }

    pub fn from_index(i: usize) -> Option<Deck> {
        Deck::ALL.get(i).copied()
    }

    /// Vouchers pre-redeemed at run start (`config.voucher` /
    /// `config.vouchers`, back.lua:176-180 and :232-238). Each is marked used,
    /// so it also leaves the shop's voucher pool.
    pub fn starting_vouchers(self) -> &'static [&'static str] {
        match self {
            Deck::Magic => &["v_crystal_ball"],
            Deck::Nebula => &["v_telescope"],
            Deck::Zodiac => &["v_tarot_merchant", "v_planet_merchant", "v_overstock_norm"],
            _ => &[],
        }
    }

    /// Consumables held at run start (`config.consumables`, back.lua:184-196).
    ///
    /// The Lua passes set `'Tarot'` even for Ghost's `c_hex`, which is a
    /// Spectral — the forced key wins, so the set argument is inert.
    pub fn starting_consumables(self) -> &'static [&'static str] {
        match self {
            Deck::Magic => &["c_fool", "c_fool"],
            Deck::Ghost => &["c_hex"],
            _ => &[],
        }
    }
}

/// Stake difficulty. Discriminants are `stake_level` (game.lua:253-260).
///
/// One-based on purpose: every modifier in game.lua:2031-2041 is written as a
/// `>=` test against this number, and [`Stake::at_least`] reproduces that
/// directly rather than re-deriving a level from a zero-based index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Stake {
    #[default]
    White = 1,
    Red = 2,
    Green = 3,
    Black = 4,
    Blue = 5,
    Purple = 6,
    Orange = 7,
    Gold = 8,
}

impl Stake {
    pub const ALL: [Stake; 8] = [
        Stake::White,
        Stake::Red,
        Stake::Green,
        Stake::Black,
        Stake::Blue,
        Stake::Purple,
        Stake::Orange,
        Stake::Gold,
    ];

    pub fn level(self) -> u8 {
        self as u8
    }

    /// `G.GAME.stake >= n`, the shape every stake rule is written in.
    pub fn at_least(self, level: u8) -> bool {
        self.level() >= level
    }

    pub fn key(self) -> &'static str {
        match self {
            Stake::White => "stake_white",
            Stake::Red => "stake_red",
            Stake::Green => "stake_green",
            Stake::Black => "stake_black",
            Stake::Blue => "stake_blue",
            Stake::Purple => "stake_purple",
            Stake::Orange => "stake_orange",
            Stake::Gold => "stake_gold",
        }
    }

    pub fn from_level(level: u8) -> Option<Stake> {
        Stake::ALL.get(level.checked_sub(1)? as usize).copied()
    }
}

/// Which blind-size curve `get_blind_amount` uses
/// (misc_functions.lua:919-954), selected by `G.GAME.modifiers.scaling`.
///
/// Note this is *assigned*, not accumulated: Green/Black/Blue all sit at
/// [`Scaling::Two`] and Purple/Orange/Gold at [`Scaling::Three`]. There is no
/// fourth curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Scaling {
    #[default]
    One = 1,
    Two = 2,
    Three = 3,
}

/// `G.GAME.starting_params` (misc_functions.lua:1868-1881), after the stake
/// cascade and `Back:apply_to_run` have both been applied.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StartingParams {
    pub dollars: i64,
    pub hand_size: i64,
    pub discards: i64,
    pub hands: i64,
    pub reroll_cost: i64,
    pub joker_slots: usize,
    pub consumable_slots: usize,
    /// Multiplies every blind's chip requirement (blind.lua:107). 2 on Plasma.
    pub ante_scaling: f64,
    /// Abandoned Deck: K/Q/J are filtered out of the starting deck
    /// (game.lua:2337).
    pub no_faces: bool,
    /// Erratic Deck: every one of the 52 slots is re-rolled (game.lua:2324).
    pub erratic_suits_and_ranks: bool,
}

impl Default for StartingParams {
    /// `get_starting_params()` — the true base, with no deck or stake applied.
    ///
    /// Note `discards` is 3 here, not 4. The 4 the sim used to hardcode was
    /// base 3 plus the Red Deck's `config.discards = 1`; with a real deck
    /// system that bonus belongs to the deck.
    fn default() -> Self {
        StartingParams {
            dollars: 4,
            hand_size: 8,
            discards: 3,
            hands: 4,
            reroll_cost: 5,
            joker_slots: 5,
            consumable_slots: 2,
            ante_scaling: 1.0,
            no_faces: false,
            erratic_suits_and_ranks: false,
        }
    }
}

/// `G.GAME.modifiers` — the subset the base decks and stakes can set.
///
/// Challenge-only modifiers (`all_eternal`, `inflation`, `discard_cost`,
/// `chips_dollar_cap`, ...) are deliberately absent; challenges are out of
/// scope and adding a field nothing sets invites the reader to assume it works.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Modifiers {
    /// Blind types that pay $0 (blind.lua:84). Red Stake zeroes the Small
    /// Blind; indexed by [`BlindKind`] order (Small, Big, Boss).
    pub no_blind_reward: [bool; 3],
    pub scaling: Scaling,
    /// Black Stake and above: shop/pack jokers can roll the eternal sticker.
    pub enable_eternals_in_shop: bool,
    /// Orange Stake and above: ... and the perishable sticker.
    pub enable_perishables_in_shop: bool,
    /// Gold Stake: ... and the rental sticker.
    pub enable_rentals_in_shop: bool,
    /// Green Deck: no interest at cash out (state_events.lua:1191).
    pub no_interest: bool,
    /// $ per unused hand at cash out; 1 unless a deck says otherwise
    /// (state_events.lua:1165-1168). Green Deck sets 2.
    pub money_per_hand: i64,
    /// $ per unused discard; 0 unless set (state_events.lua:1170-1173).
    /// Green Deck sets 1.
    pub money_per_discard: i64,
    /// Ghost Deck: weight of Spectral cards in the shop type poll
    /// (`G.GAME.spectral_rate`, game.lua:1886 / back.lua:206).
    pub spectral_rate: f64,
    /// Anaglyph Deck: a Double Tag after every boss blind (back.lua:111-120).
    pub anaglyph_double_tag: bool,
    /// Plasma Deck: chips and mult are averaged in the final scoring step
    /// (back.lua:125-171).
    pub plasma_balance: bool,
    /// Checkered Deck: Clubs become Spades and Diamonds become Hearts
    /// (back.lua:239-253).
    pub checkered_suits: bool,
}

impl Modifiers {
    /// `G.GAME.modifiers` defaults, before deck and stake.
    fn base() -> Self {
        Modifiers {
            money_per_hand: 1,
            ..Default::default()
        }
    }
}

/// Everything that distinguishes one run's rules from another's.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RunConfig {
    pub deck: Deck,
    pub stake: Stake,
    /// Ante whose boss ends the run in a win (`G.GAME.win_ante`,
    /// game.lua:1884). The trainer's curriculum lowers it; 8 is the real game.
    pub win_ante: i64,
    /// Keep playing past `win_ante` instead of ending the run. The blind curve
    /// past ante 8 is already the game's own (misc_functions.lua:927-930).
    pub endless: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            deck: Deck::Red,
            stake: Stake::White,
            win_ante: 8,
            endless: false,
        }
    }
}

impl RunConfig {
    pub fn new(deck: Deck, stake: Stake) -> Self {
        RunConfig {
            deck,
            stake,
            ..Default::default()
        }
    }

    /// Apply the stake cascade and then the deck, in `Game:start_run`'s order
    /// (game.lua:2031-2043).
    pub fn resolve(&self) -> (StartingParams, Modifiers) {
        let mut p = StartingParams::default();
        let mut m = Modifiers::base();

        // --- stake cascade (game.lua:2031-2041) -------------------------
        // Cumulative by construction: every rule is `>=`, so stake N gets
        // every rule for levels <= N.
        if self.stake.at_least(2) {
            m.no_blind_reward[0] = true; // Small
        }
        if self.stake.at_least(3) {
            m.scaling = Scaling::Two;
        }
        if self.stake.at_least(4) {
            m.enable_eternals_in_shop = true;
        }
        if self.stake.at_least(5) {
            p.discards -= 1;
        }
        if self.stake.at_least(6) {
            // Assignment, not accumulation — this replaces Scaling::Two.
            m.scaling = Scaling::Three;
        }
        if self.stake.at_least(7) {
            m.enable_perishables_in_shop = true;
        }
        if self.stake.at_least(8) {
            m.enable_rentals_in_shop = true;
        }

        // --- Back:apply_to_run (back.lua:174-278) -----------------------
        match self.deck {
            Deck::Red => p.discards += 1,
            Deck::Blue => p.hands += 1,
            Deck::Yellow => p.dollars += 10,
            Deck::Green => {
                m.no_interest = true;
                m.money_per_hand = 2;
                m.money_per_discard = 1;
            }
            Deck::Black => {
                p.hands -= 1;
                p.joker_slots += 1;
            }
            Deck::Magic => {
                // v_crystal_ball is redeemed at run start and grants the slot.
            }
            Deck::Nebula => p.consumable_slots -= 1,
            Deck::Ghost => m.spectral_rate = 2.0,
            Deck::Abandoned => p.no_faces = true,
            Deck::Checkered => m.checkered_suits = true,
            Deck::Zodiac => {}
            Deck::Painted => {
                p.hand_size += 2;
                p.joker_slots -= 1;
            }
            Deck::Anaglyph => m.anaglyph_double_tag = true,
            Deck::Plasma => {
                p.ante_scaling = 2.0;
                m.plasma_balance = true;
            }
            Deck::Erratic => p.erratic_suits_and_ranks = true,
        }

        (p, m)
    }
}

/// The 52 `G.P_CARDS` keys in the order `pseudorandom_element` sees them.
///
/// `pseudorandom_element` builds a key list and **sorts it**
/// (misc_functions.lua:260-263) before drawing, so the Erratic Deck never
/// depends on Lua's `pairs` hash order — only on this byte-wise ordering of
/// the `"<suit>_<rank>"` keys. Suits sort `C < D < H < S`; ranks sort
/// `'2'..'9' < 'A' < 'J' < 'K' < 'Q' < 'T'` (ASCII).
pub const P_CARDS_SORTED: [(Suit, Rank); 52] = {
    const SUITS: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];
    const RANKS: [u8; 13] = [2, 3, 4, 5, 6, 7, 8, 9, 14, 11, 13, 12, 10];
    let mut out = [(Suit::Clubs, Rank(2)); 52];
    let mut i = 0;
    let mut s = 0;
    while s < 4 {
        let mut r = 0;
        while r < 13 {
            out[i] = (SUITS[s], Rank(RANKS[r]));
            i += 1;
            r += 1;
        }
        s += 1;
    }
    out
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deck_and_stake_discriminants_are_frozen() {
        // These are a wire format shared with the observation contract; a
        // renumbering silently invalidates every trained checkpoint.
        assert_eq!(Deck::Red as usize, 0);
        assert_eq!(Deck::Plasma as usize, 13);
        assert_eq!(Deck::Erratic as usize, 14);
        assert_eq!(Stake::White as u8, 1);
        assert_eq!(Stake::Gold as u8, 8);
        for (i, d) in Deck::ALL.iter().enumerate() {
            assert_eq!(*d as usize, i);
            assert_eq!(Deck::from_index(i), Some(*d));
        }
        for (i, s) in Stake::ALL.iter().enumerate() {
            assert_eq!(s.level() as usize, i + 1);
            assert_eq!(Stake::from_level(s.level()), Some(*s));
        }
    }

    #[test]
    fn red_white_reproduces_the_previous_hardcoded_constants() {
        // The pre-P7 sim hardcoded these; Red/White must still produce them
        // exactly or every existing test vector shifts.
        let (p, m) = RunConfig::default().resolve();
        assert_eq!(p.dollars, 4);
        assert_eq!(p.hand_size, 8);
        assert_eq!(p.hands, 4);
        assert_eq!(p.discards, 4, "base 3 + Red Deck's +1");
        assert_eq!(p.reroll_cost, 5);
        assert_eq!(p.joker_slots, 5);
        assert_eq!(p.consumable_slots, 2);
        assert_eq!(p.ante_scaling, 1.0);
        assert!(!p.no_faces && !p.erratic_suits_and_ranks);
        assert_eq!(m.scaling, Scaling::One);
        assert_eq!(m.money_per_hand, 1);
        assert_eq!(m.money_per_discard, 0);
        assert_eq!(m.spectral_rate, 0.0);
        assert!(!m.no_interest);
        assert_eq!(m.no_blind_reward, [false; 3]);
        assert!(!m.enable_eternals_in_shop);
        assert!(!m.enable_perishables_in_shop);
        assert!(!m.enable_rentals_in_shop);
    }

    #[test]
    fn stake_rules_are_cumulative_and_scaling_is_assigned() {
        let at = |s: Stake| RunConfig::new(Deck::Red, s).resolve();

        assert_eq!(at(Stake::White).1.scaling, Scaling::One);
        assert!(!at(Stake::White).1.no_blind_reward[0]);

        // Red introduces the Small-blind zero and every later stake keeps it.
        for s in [
            Stake::Red,
            Stake::Green,
            Stake::Black,
            Stake::Blue,
            Stake::Purple,
            Stake::Orange,
            Stake::Gold,
        ] {
            assert!(at(s).1.no_blind_reward[0], "{s:?} lost the Red rule");
        }

        // scaling is assigned, not accumulated: 2 for green/black/blue,
        // 3 from purple up, and never 4.
        assert_eq!(at(Stake::Green).1.scaling, Scaling::Two);
        assert_eq!(at(Stake::Blue).1.scaling, Scaling::Two);
        assert_eq!(at(Stake::Purple).1.scaling, Scaling::Three);
        assert_eq!(at(Stake::Gold).1.scaling, Scaling::Three);

        // Blue Stake's -1 discard against the Red Deck's +1: 3 - 1 + 1.
        assert_eq!(at(Stake::Black).0.discards, 4);
        assert_eq!(at(Stake::Blue).0.discards, 3);

        assert!(at(Stake::Black).1.enable_eternals_in_shop);
        assert!(!at(Stake::Black).1.enable_perishables_in_shop);
        assert!(at(Stake::Orange).1.enable_perishables_in_shop);
        assert!(!at(Stake::Orange).1.enable_rentals_in_shop);
        assert!(at(Stake::Gold).1.enable_rentals_in_shop);
    }

    #[test]
    fn deck_effects_match_their_config_tables() {
        let at = |d: Deck| RunConfig::new(d, Stake::White).resolve();

        assert_eq!(at(Deck::Blue).0.hands, 5);
        assert_eq!(at(Deck::Yellow).0.dollars, 14);
        assert_eq!(at(Deck::Black).0.hands, 3);
        assert_eq!(at(Deck::Black).0.joker_slots, 6);
        assert_eq!(at(Deck::Painted).0.hand_size, 10);
        assert_eq!(at(Deck::Painted).0.joker_slots, 4);
        assert_eq!(at(Deck::Nebula).0.consumable_slots, 1);
        assert_eq!(at(Deck::Plasma).0.ante_scaling, 2.0);
        assert!(at(Deck::Plasma).1.plasma_balance);
        assert!(at(Deck::Abandoned).0.no_faces);
        assert!(at(Deck::Erratic).0.erratic_suits_and_ranks);
        assert!(at(Deck::Checkered).1.checkered_suits);
        assert!(at(Deck::Anaglyph).1.anaglyph_double_tag);
        assert_eq!(at(Deck::Ghost).1.spectral_rate, 2.0);

        // Green's whole identity is economic.
        let (_, g) = at(Deck::Green);
        assert!(g.no_interest);
        assert_eq!(g.money_per_hand, 2);
        assert_eq!(g.money_per_discard, 1);

        // Only these decks pre-redeem vouchers / hold consumables.
        assert_eq!(Deck::Magic.starting_vouchers(), &["v_crystal_ball"]);
        assert_eq!(Deck::Nebula.starting_vouchers(), &["v_telescope"]);
        assert_eq!(Deck::Zodiac.starting_vouchers().len(), 3);
        assert_eq!(Deck::Magic.starting_consumables(), &["c_fool", "c_fool"]);
        assert_eq!(Deck::Ghost.starting_consumables(), &["c_hex"]);
        assert!(Deck::Red.starting_vouchers().is_empty());
        assert!(Deck::Red.starting_consumables().is_empty());
    }

    #[test]
    fn stake_block_precedes_the_deck() {
        // Blue Stake (-1 discard) with the Black Deck (-1 hand, +1 joker):
        // both land, and the discard floor is not clamped in between.
        let (p, _) = RunConfig::new(Deck::Black, Stake::Blue).resolve();
        assert_eq!(p.discards, 2, "base 3, Blue Stake -1, Black Deck adds none");
        assert_eq!(p.hands, 3);
        assert_eq!(p.joker_slots, 6);
    }

    #[test]
    fn p_cards_sorted_matches_lua_key_order() {
        // "<suit>_<rank>" byte order: C < D < H < S, and within a suit
        // '2'..'9' < 'A' < 'J' < 'K' < 'Q' < 'T'.
        assert_eq!(P_CARDS_SORTED.len(), 52);
        assert_eq!(P_CARDS_SORTED[0], (Suit::Clubs, Rank(2)));
        assert_eq!(P_CARDS_SORTED[8], (Suit::Clubs, Rank(14)), "C_A after C_9");
        assert_eq!(P_CARDS_SORTED[9], (Suit::Clubs, Rank(11)), "C_J");
        assert_eq!(P_CARDS_SORTED[10], (Suit::Clubs, Rank(13)), "C_K");
        assert_eq!(P_CARDS_SORTED[11], (Suit::Clubs, Rank(12)), "C_Q");
        assert_eq!(P_CARDS_SORTED[12], (Suit::Clubs, Rank(10)), "C_T last");
        assert_eq!(P_CARDS_SORTED[13], (Suit::Diamonds, Rank(2)));
        assert_eq!(P_CARDS_SORTED[51], (Suit::Spades, Rank(10)));

        // Every (suit, rank) appears exactly once.
        let mut seen = std::collections::HashSet::new();
        for entry in P_CARDS_SORTED {
            assert!(seen.insert(entry), "duplicate {entry:?}");
        }
        assert_eq!(seen.len(), 52);
    }
}
