//! Card primitives. Mirrors the game's card identity model:
//! `get_id()` = 2..14 (Ace high = 14), `get_nominal()` = chip base value.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Suit {
    // Order matters: get_flush checks suits in this order.
    Spades,
    Hearts,
    Clubs,
    Diamonds,
}

impl Suit {
    pub const ALL: [Suit; 4] = [Suit::Spades, Suit::Hearts, Suit::Clubs, Suit::Diamonds];

    /// The game's card-key prefix ("S", "H", "C", "D") as in `P_CARDS` keys like "H_A".
    pub fn key(self) -> char {
        match self {
            Suit::Spades => 'S',
            Suit::Hearts => 'H',
            Suit::Clubs => 'C',
            Suit::Diamonds => 'D',
        }
    }

    /// `base.suit_nominal` from card.lua:138-141 (sort tie-break fractions).
    pub fn suit_nominal(self) -> f64 {
        match self {
            Suit::Diamonds => 0.01,
            Suit::Clubs => 0.02,
            Suit::Hearts => 0.03,
            Suit::Spades => 0.04,
        }
    }

    /// `base.suit_nominal_original` from card.lua:138-141.
    pub fn suit_nominal_original(self) -> f64 {
        match self {
            Suit::Diamonds => 0.001,
            Suit::Clubs => 0.002,
            Suit::Hearts => 0.003,
            Suit::Spades => 0.004,
        }
    }
}

/// Rank with the game's id numbering: 2..=10 face value, J=11, Q=12, K=13, A=14.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rank(pub u8);

impl Rank {
    pub const ACE: Rank = Rank(14);

    pub fn id(self) -> u8 {
        self.0
    }

    /// `get_nominal()`-style base chip value: 2-10 face, J/Q/K = 10, A = 11.
    pub fn nominal(self) -> u32 {
        match self.0 {
            14 => 11,
            11..=13 => 10,
            n => n as u32,
        }
    }

    pub fn is_face(self) -> bool {
        matches!(self.0, 11..=13)
    }

    /// `base.face_nominal` from card.lua:122-134 (0.1/0.2/0.3/0.4 for J/Q/K/A).
    pub fn face_nominal(self) -> f64 {
        match self.0 {
            11 => 0.1,
            12 => 0.2,
            13 => 0.3,
            14 => 0.4,
            _ => 0.0,
        }
    }

    /// The game's card-key suffix as in "H_A", "S_T", "D_2".
    pub fn key(self) -> char {
        match self.0 {
            14 => 'A',
            13 => 'K',
            12 => 'Q',
            11 => 'J',
            10 => 'T',
            n => (b'0' + n) as char,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Enhancement {
    #[default]
    None,
    Bonus,
    Mult,
    Wild,
    Glass,
    Steel,
    Stone,
    Gold,
    Lucky,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Edition {
    #[default]
    None,
    Foil,
    Holo,
    Polychrome,
    Negative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Seal {
    #[default]
    None,
    Gold,
    Red,
    Blue,
    Purple,
}

/// A playing card instance. `sort_id` is the creation-order id the game
/// assigns (`card.sort_id`) and is the pre-shuffle sort key in `pseudoshuffle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
    pub enhancement: Enhancement,
    pub edition: Edition,
    pub seal: Seal,
    pub sort_id: u32,
    /// `card.debuff` (card.lua:526-538), recomputed by `Blind:debuff_card`
    /// on every `set_blind`/`disable` (blind.lua:207-213, 407-412).
    pub debuff: bool,
    /// Face-down in hand: `card.ability.wheel_flipped` semantics
    /// (cardarea.lua:41-43). Pure observation flag — the card still plays,
    /// discards and scores normally; only its identity is hidden from the
    /// player/agent. Cleared when the card leaves the hand (cards flip face
    /// up when emplaced into the play area, cardarea.lua:38-40) and at round
    /// start (state_events.lua:307-309).
    pub face_down: bool,
    /// `card.ability.played_this_ante` (state_events.lua:481), the Pillar's
    /// debuff criterion; cleared when a Boss blind is defeated
    /// (state_events.lua:265-267).
    pub played_this_ante: bool,
    /// `ability.perma_bonus` (Hiker, card.lua:3067-3069): permanent extra
    /// chips added to `get_chip_bonus` (card.lua:976-982). Always a whole
    /// number in the base game (+5 per Hiker trigger).
    pub perma_bonus: i64,
}

impl Card {
    pub fn new(rank: Rank, suit: Suit, sort_id: u32) -> Self {
        Card {
            rank,
            suit,
            enhancement: Enhancement::None,
            edition: Edition::None,
            seal: Seal::None,
            sort_id,
            debuff: false,
            face_down: false,
            played_this_ante: false,
            perma_bonus: 0,
        }
    }

    /// `get_id()`; Stone cards have no rank for hand evaluation (id -1 in game,
    /// filtered out by the `id > 1 and id < 15` check).
    pub fn id_for_eval(&self) -> Option<u8> {
        if self.enhancement == Enhancement::Stone {
            None
        } else {
            Some(self.rank.id())
        }
    }

    /// Smeared Joker's suit identity (card.lua:4071/4084): both suits red or
    /// both black.
    #[inline]
    fn smeared_match(&self, suit: Suit) -> bool {
        let red = |s: Suit| matches!(s, Suit::Hearts | Suit::Diamonds);
        red(self.suit) == red(suit)
    }

    /// `Card:is_suit(suit, nil, true)` — the `flush_calc` branch used by
    /// `get_flush` (card.lua:4064-4075): Stone cards match no suit, Wild
    /// cards match any suit *unless debuffed* ("Wild Card" check requires
    /// `not self.debuff`, card.lua:4069), then the Smeared Joker red/black
    /// identity (card.lua:4071), everything else — including debuffed cards
    /// — matches its natural `base.suit`. So debuffed cards still count for
    /// flush identification.
    pub fn is_suit(&self, suit: Suit, smeared: bool) -> bool {
        match self.enhancement {
            Enhancement::Stone => false,
            Enhancement::Wild if !self.debuff => true,
            _ => (smeared && self.smeared_match(suit)) || self.suit == suit,
        }
    }

    /// `Card:is_suit(suit)` — the plain branch (card.lua:4076-4088) used by
    /// the per-card joker suit checks (Greedy..Gluttonous, Rough Gem, The
    /// Idol, Castle, Ancient Joker, Seeing Double's non-Wild pass): debuffed
    /// cards match nothing, Stone cards nothing, Wild cards everything, then
    /// Smeared/natural suit.
    pub fn is_suit_normal(&self, suit: Suit, smeared: bool) -> bool {
        if self.debuff {
            return false;
        }
        match self.enhancement {
            Enhancement::Stone => false,
            Enhancement::Wild => true,
            _ => (smeared && self.smeared_match(suit)) || self.suit == suit,
        }
    }

    /// `Card:is_suit(suit, true)` — bypass_debuff branch (card.lua:4076-4088)
    /// used by `Blind:debuff_card` (blind.lua:626) and Flower Pot's non-Wild
    /// pass. Wild cards match ANY suit here, so every suit-debuff boss
    /// debuffs Wild cards; Smeared widens boss suit debuffs to the colour.
    pub fn is_suit_bypass_debuff(&self, suit: Suit, smeared: bool) -> bool {
        match self.enhancement {
            Enhancement::Stone => false,
            Enhancement::Wild => true,
            _ => (smeared && self.smeared_match(suit)) || self.suit == suit,
        }
    }

    /// `Card:is_face(true)` — the `from_boss` call (card.lua:964-970) used by
    /// The Plant's debuff and The Mark's stay_flipped: bypasses the debuff
    /// check; Stone cards are never faces (`get_id` returns a negative,
    /// card.lua:957-961) — but the Pareidolia `or` still makes EVERY card a
    /// face while it is owned, Stone cards included.
    pub fn is_face_from_boss(&self, pareidolia: bool) -> bool {
        (self.enhancement != Enhancement::Stone && self.rank.is_face()) || pareidolia
    }

    pub fn nominal(&self) -> u32 {
        if self.enhancement == Enhancement::Stone {
            0
        } else {
            self.rank.nominal()
        }
    }

    /// `Card:get_nominal(mod)` (card.lua:950-955), the sort key used by
    /// `CardArea:sort('desc')` and `get_highest`:
    ///   nominal + suit_nominal*mult + suit_nominal_original*0.0001*mult
    ///           + face_nominal + 0.000001*unique_val
    /// with mult = 1000 for mod=='suit', -1000 for Stone cards, else 1.
    /// The `unique_val` term (1 - ID/1603301, a per-object creation counter) is
    /// omitted here; callers must tie-break equal nominals by creation order
    /// (lower `sort_id` == larger unique_val == sorts first in 'desc').
    pub fn get_nominal(&self, suit_mod: bool) -> f64 {
        let mut mult = 1.0;
        if suit_mod {
            mult = 1000.0;
        }
        if self.enhancement == Enhancement::Stone {
            mult = -1000.0;
        }
        // base.nominal is the rank chip value even for Stone cards (the Stone
        // override lives in get_chip_bonus, not in base.nominal).
        self.rank.nominal() as f64
            + self.suit.suit_nominal() * mult
            + self.suit.suit_nominal_original() * 0.0001 * mult
            + self.rank.face_nominal()
    }

    /// `Card:get_chip_bonus()` (card.lua:976-982). Debuffed cards give 0;
    /// Stone Cards score `ability.bonus + perma_bonus` (50, from the m_stone
    /// center config in game.lua:653) instead of the rank nominal; Bonus
    /// cards add 30 (m_bonus, game.lua:648); Hiker's `perma_bonus` adds in
    /// every case.
    pub fn chip_bonus(&self) -> f64 {
        if self.debuff {
            return 0.0;
        }
        let base = match self.enhancement {
            Enhancement::Stone => 50.0,
            Enhancement::Bonus => self.rank.nominal() as f64 + 30.0,
            _ => self.rank.nominal() as f64,
        };
        base + self.perma_bonus as f64
    }
}

impl Enhancement {
    /// `ability.mult` from the center config: m_mult gives 4 (game.lua:649),
    /// m_lucky gives 20 gated behind the 'lucky_mult' roll (game.lua:655,
    /// card.lua:987-993). The roll itself lives in scoring.rs.
    pub fn mult_value(self) -> f64 {
        match self {
            Enhancement::Mult => 4.0,
            Enhancement::Lucky => 20.0,
            _ => 0.0,
        }
    }
}

impl Edition {
    /// `Card:get_edition()` chip/mult/x_mult mods (card.lua:1016-1031) with
    /// the values wired in `Card:set_edition` (card.lua:387-404) from the
    /// e_foil/e_holo/e_polychrome center configs (game.lua:659-661):
    /// Foil +50 chips, Holographic +10 mult, Polychrome x1.5 mult.
    pub fn scoring_mods(self) -> (Option<f64>, Option<f64>, Option<f64>) {
        match self {
            Edition::Foil => (Some(50.0), None, None),
            Edition::Holo => (None, Some(10.0), None),
            Edition::Polychrome => (None, None, Some(1.5)),
            // Negative has no scoring effect on playing cards (config
            // extra = 1 is the joker-slot bonus only).
            Edition::Negative | Edition::None => (None, None, None),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HandType {
    HighCard,
    Pair,
    TwoPair,
    ThreeOfAKind,
    Straight,
    Flush,
    FullHouse,
    FourOfAKind,
    StraightFlush,
    FiveOfAKind,
    FlushHouse,
    FlushFive,
}

impl HandType {
    pub const ALL: [HandType; 12] = [
        HandType::HighCard,
        HandType::Pair,
        HandType::TwoPair,
        HandType::ThreeOfAKind,
        HandType::Straight,
        HandType::Flush,
        HandType::FullHouse,
        HandType::FourOfAKind,
        HandType::StraightFlush,
        HandType::FiveOfAKind,
        HandType::FlushHouse,
        HandType::FlushFive,
    ];

    /// Base (level 1) chips/mult and per-level increments, from the
    /// `hands` table in game.lua (`s_chips/s_mult` and `l_chips/l_mult`).
    pub fn base_values(self) -> HandValues {
        // (base_chips, base_mult, level_chips, level_mult)
        let (c, m, lc, lm) = match self {
            HandType::FlushFive => (160, 16, 50, 3),
            HandType::FlushHouse => (140, 14, 40, 4),
            HandType::FiveOfAKind => (120, 12, 35, 3),
            HandType::StraightFlush => (100, 8, 40, 4),
            HandType::FourOfAKind => (60, 7, 30, 3),
            HandType::FullHouse => (40, 4, 25, 2),
            HandType::Flush => (35, 4, 15, 2),
            HandType::Straight => (30, 4, 30, 3),
            HandType::ThreeOfAKind => (30, 3, 20, 2),
            HandType::TwoPair => (20, 2, 20, 1),
            HandType::Pair => (10, 2, 15, 1),
            HandType::HighCard => (5, 1, 10, 1),
        };
        HandValues {
            base_chips: c,
            base_mult: m,
            level_chips: lc,
            level_mult: lm,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HandValues {
    pub base_chips: u32,
    pub base_mult: u32,
    pub level_chips: u32,
    pub level_mult: u32,
}
