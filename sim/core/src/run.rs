//! The run state machine: Red Deck, White Stake, full scoring pipeline
//! (enhancements/editions/seals/held effects), all boss blinds, and — since
//! P3b — the full between-blind layer: shop, economy, booster packs,
//! consumable usage, vouchers and skip-blind tags. Joker EFFECTS are still
//! stubbed (P3c); jokers exist as inert items. Mirrors the `G.STATES` flow:
//!
//!   BLIND_SELECT -> (select | skip) -> DRAW_TO_HAND -> SELECTING_HAND
//!     -> play/discard ... -> NEW_ROUND/end_round -> ROUND_EVAL
//!     -> (cash out) -> SHOP <-> *_PACK -> BLIND_SELECT
//!
//! Sources: game.lua `Game:start_run`/`update_*`, state_events.lua
//! `new_round`/`end_round`/`evaluate_play`/`G.FUNCS.evaluate_round`,
//! blind.lua (all boss mechanics), button_callbacks.lua (`cash_out`,
//! `buy_from_shop`, `reroll_shop`, `skip_blind`, `use_card`, ...),
//! common_events.lua (`reset_blinds`, `create_card`, `get_current_pool`),
//! card.lua (`use_consumeable`, `open`, `redeem`, `sell_card`), tag.lua.
//!
//! RNG stream usage of the core loop (shop/pack/consumable streams are
//! documented in shop.rs / tarots.rs; per-key streams are independent):
//!   'boss'           run start + after every boss defeat + boss rerolls
//!   'Voucher<ante>'  run start + every boss defeat (get_next_voucher_key)
//!   'Tag<ante>'      run start + every boss defeat, two draws (blind tags)
//!   'orbital'        3 draws at the first blind-select entry of each ante
//!   'shuffle'        once at run start (game.lua:2383)
//!   'nr<ante>'       every round start (state_events.lua:344)
//!   'cashout<ante>'  every cash out (button_callbacks.lua:2918)
//!   'wheel'          The Wheel, per card drawn to hand (blind.lua:608)
//!   'hook'           The Hook, two draws per played hand (blind.lua:475)
//!   'cerulean_bell'  Cerulean Bell, per draw-to-hand without a forced card
//!   'aajk'           Amber Acorn, 3 joker shuffles at set_blind
//!   'crimson_heart'  Crimson Heart, per draw-to-hand while prepped
//!   'lucky_mult'/'lucky_money'/'glass'  scoring (see scoring.rs)
//!   'Tarot8ba<ante>' (+ '_resample<n>')  purple-seal tarot generation

use std::collections::{HashMap, HashSet};

use crate::blinds::{boss_by_key, get_new_boss, ActiveBlind, BlindProto, BL_BIG, BL_SMALL};
use crate::cards::{Card, Enhancement, HandType, Seal, Suit};
use crate::consumables::planet_for_hand;
use crate::deck::{pseudoshuffle, red_deck};
use crate::handeval::EvalMods;
use crate::items::{element_index, ConsumableSet, JokerId};
use crate::jokers::{resolve_effective, EngineHooks, JokerEnv, JokerOutbox, OutEvent};
use crate::rng::{LuaRandom, RngState};
use crate::scoring::{evaluate_play, HandsTable, JokerHooks, PlayResult, ScoreContext};
use crate::shop::{
    JokerArea, OwnedConsumable, OwnedJoker, PackState, RunTag, ShopState, TagContext,
};

/// `get_starting_params()` (misc_functions.lua:1868), Red Deck / White Stake.
const STARTING_DOLLARS: i64 = 4;
const HAND_SIZE: i64 = 8;
const STARTING_HANDS: i64 = 4;
/// base 3 (misc_functions.lua:2026) + 1 from the Red Deck back:
/// `Back:apply_to_run` adds `config.discards` (= 1 for b_red, game.lua:643)
/// to starting_params.discards (back.lua:264-265). Found by P5 live
/// cross-validation — every Red Deck round has 4 discards.
const STARTING_DISCARDS: i64 = 4;
pub(crate) const STARTING_REROLL_COST: i64 = 5;
pub(crate) const JOKER_SLOTS: usize = 5;
/// game.lua:1884 / :1909-1910.
const WIN_ANTE: i64 = 8;
/// `G.hand.config.highlighted_limit` — at most 5 cards played/discarded.
const HIGHLIGHT_LIMIT: usize = 5;

/// The `pairs(G.GAME.hands)` iteration order of end_round's most-played scan
/// (state_events.lua:130-137), reproduced with the oracle LuaJIT
/// (tools/oracle/luajit) over the exact game.lua:2001-2014 table constructor.
/// The scan's `_order` variable is never updated (it stays 100 > every hand's
/// order), so on a played-count tie the LAST hand in this iteration order
/// wins.
pub(crate) const MOST_PLAYED_SCAN: [HandType; 12] = [
    HandType::Pair,
    HandType::HighCard,
    HandType::FlushFive,
    HandType::FlushHouse,
    HandType::FiveOfAKind,
    HandType::StraightFlush,
    HandType::FourOfAKind,
    HandType::FullHouse,
    HandType::Flush,
    HandType::Straight,
    HandType::ThreeOfAKind,
    HandType::TwoPair,
];

/// Subset of `G.STATES` reachable through P3b. `#[non_exhaustive]` leaves
/// room for P3c additions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum State {
    BlindSelect,
    SelectingHand,
    RoundEval,
    /// `G.STATES.SHOP`.
    Shop,
    /// One of the `*_PACK` states — the open pack's detail is in `pack()`.
    PackOpen,
    GameOver,
    /// Ante-8 boss defeated. The real game offers endless mode here; P3 stops.
    Won,
}

/// Which blind is on deck (`G.GAME.blind_on_deck`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlindStage {
    Small,
    Big,
    Boss,
}

/// Legal action descriptors. Card/target subsets are passed to the action
/// methods themselves (`play(&[..])`, `use_consumable(i, &[..])`, ...), as
/// established by Play/Discard in P3a — `legal_actions` enumerates slots but
/// not target subsets.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Action {
    SelectBlind,
    /// Skip the Small/Big blind for its tag (`G.FUNCS.skip_blind`).
    SkipBlind,
    /// Reroll the boss for $10 (Director's Cut / Retcon).
    RerollBoss,
    /// Play 1-5 cards by hand index.
    Play,
    /// Discard 1-5 cards by hand index (needs discards_left > 0).
    Discard,
    CashOut,
    /// Buy the joker/consumable/playing card in shop card slot `i`.
    BuyShopItem(usize),
    /// Buy-and-use the consumable in shop card slot `i`.
    BuyAndUseShopItem(usize),
    /// Redeem the voucher in voucher slot `i`.
    RedeemVoucher(usize),
    /// Open the booster in pack slot `i`.
    BuyPack(usize),
    /// Reroll the shop card slots.
    Reroll,
    LeaveShop,
    /// Use the consumable in inventory slot `i` (targets go to the method).
    UseConsumable(usize),
    SellConsumable(usize),
    SellJoker(usize),
    /// Pick pack item `i` (consumables are used immediately; targets go to
    /// the method).
    PickPackItem(usize),
    SkipPack,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RunError {
    WrongState,
    BadCardSelection(String),
    NoDiscardsLeft,
    /// Slot index out of range / empty.
    BadSlot(String),
    /// `G.GAME.dollars - G.GAME.bankrupt_at` cannot cover the cost.
    InsufficientFunds,
    /// Joker/consumable area is full (`check_for_buy_space`).
    NoSpace,
    /// `Card:can_use_consumeable` said no (wrong targets, no eligible ...).
    CannotUse(String),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::WrongState => write!(f, "action not legal in current state"),
            RunError::BadCardSelection(s) => write!(f, "bad card selection: {s}"),
            RunError::NoDiscardsLeft => write!(f, "no discards left"),
            RunError::BadSlot(s) => write!(f, "bad slot: {s}"),
            RunError::InsufficientFunds => write!(f, "not enough money"),
            RunError::NoSpace => write!(f, "no space"),
            RunError::CannotUse(s) => write!(f, "cannot use: {s}"),
        }
    }
}

impl std::error::Error for RunError {}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(deserialize = "")))]
pub struct Run {
    pub(crate) rng: RngState,
    pub(crate) state: State,

    pub(crate) deck: Vec<Card>,
    pub(crate) hand: Vec<Card>,
    pub(crate) discard_pile: Vec<Card>,
    /// Cards destroyed for good (shattered Glass / Hanged Man...).
    pub(crate) destroyed_cards: Vec<Card>,

    pub(crate) hands_table: HandsTable,

    pub(crate) ante: i64,
    /// `G.GAME.round_resets.blind_ante` — the ante used for blind chip
    /// amounts; tracks `ante` in vanilla (Hieroglyph/Petroglyph shift both).
    pub(crate) blind_ante: i64,
    pub(crate) round: i64,
    pub(crate) dollars: i64,
    /// `G.GAME.bankrupt_at` (0; Credit Card lowers it in P3c).
    pub(crate) bankrupt_at: i64,
    /// `G.GAME.chips` — score accumulated against the current blind.
    pub(crate) chips: f64,
    pub(crate) hands_left: i64,
    pub(crate) discards_left: i64,
    pub(crate) unused_discards: i64,
    /// `G.hand.config.card_limit` (The Manacle shrinks it for its round;
    /// Paint Brush/Juggle/Ouija/Ectoplasm move it too).
    pub(crate) hand_size: i64,

    /// `G.GAME.round_resets.hands` / `.discards` (Grabber/Wasteful/
    /// Hieroglyph/Petroglyph adjust these).
    pub(crate) round_resets_hands: i64,
    pub(crate) round_resets_discards: i64,
    /// `G.GAME.round_resets.temp_handsize` — Juggle Tag, removed at round end.
    pub(crate) temp_handsize: Option<i64>,

    /// Reroll bookkeeping: `G.GAME.round_resets.reroll_cost` (base, vouchers
    /// lower it), `current_round.reroll_cost(_increase)`, D6's
    /// `temp_reroll_cost`, Chaos' `free_rerolls`.
    pub(crate) base_reroll_cost: i64,
    pub(crate) reroll_cost: i64,
    pub(crate) reroll_cost_increase: i64,
    pub(crate) temp_reroll_cost: Option<i64>,
    pub(crate) free_rerolls: i64,

    /// `G.GAME.interest_cap` / `interest_amount` (Seed Money/Money Tree; To
    /// the Moon in P3c).
    pub(crate) interest_cap: i64,
    pub(crate) interest_amount: i64,
    /// `G.GAME.discount_percent` (Clearance Sale 25 / Liquidation 50).
    pub(crate) discount_percent: i64,
    /// `G.GAME.edition_rate` (Hone 2 / Glow Up 4).
    pub(crate) edition_rate: f64,
    /// Shop item-type weights (game.lua:1900-1905 + voucher effects).
    pub(crate) joker_rate: f64,
    pub(crate) tarot_rate: f64,
    pub(crate) planet_rate: f64,
    pub(crate) playing_card_rate: f64,
    pub(crate) spectral_rate: f64,

    /// `G.jokers.config.card_limit` (5; Antimatter/negative editions +1).
    pub(crate) joker_slots: usize,
    /// `G.consumeables.config.card_limit` (2; Crystal Ball +1).
    pub(crate) consumable_slots: usize,
    /// `G.GAME.shop.joker_max` (2; Overstock(+) +1 each).
    pub(crate) shop_joker_max: usize,

    /// `G.GAME.current_round.hands_played` / `.discards_used`.
    pub(crate) hands_played_round: i64,
    pub(crate) discards_used_round: i64,
    /// `G.GAME.hands_played` — run total (state_events.lua:523; Handy Tag).
    pub(crate) hands_played_total: i64,
    /// `G.GAME.skips` (button_callbacks.lua:2755; Skip Tag).
    pub(crate) skips: i64,

    pub(crate) blind_on_deck: BlindStage,
    /// Live blind state while a round is running (and until cash out).
    pub(crate) active_blind: Option<ActiveBlind>,
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::key"))]
    pub(crate) boss_choice: crate::items::Key,
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::key_map"))]
    pub(crate) bosses_used: HashMap<crate::items::Key, i64>,
    pub(crate) small_defeated: bool,
    pub(crate) big_defeated: bool,
    pub(crate) small_skipped: bool,
    pub(crate) big_skipped: bool,
    /// `G.GAME.round_resets.boss_rerolled` (Director's Cut limiter).
    pub(crate) boss_rerolled: bool,
    /// Test/tooling override mirroring `G.FORCE_TAG` (see `force_tag`).
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::opt_key"))]
    pub(crate) forced_tag: Option<crate::items::Key>,
    /// `G.GAME.last_blind.boss` (blind.lua:97-99; Investment Tag).
    pub(crate) last_blind_boss: bool,

    /// `G.GAME.current_round.most_played_poker_hand` (game.lua:1964),
    /// recomputed at the end of every won Boss round (state_events.lua:129-138).
    pub(crate) most_played_hand: HandType,
    /// `G.GAME.last_hand_played` (state_events.lua:576).
    pub(crate) last_hand_played: Option<HandType>,

    /// Cerulean Bell's forced card, by sort_id (`ability.forced_selection`).
    pub(crate) forced_card: Option<u32>,

    /// Inert jokers (effects in P3c) and held consumables.
    pub(crate) jokers: Vec<OwnedJoker>,
    pub(crate) consumables: Vec<OwnedConsumable>,
    /// `G.GAME.used_jokers`, refcounted by live Card instances
    /// (card.lua:349-355 marks on set_ability, :4741-4749 clears on remove
    /// when no same-named card remains).
    pub(crate) used_keys: HashMap<String, u32>,
    /// `G.GAME.used_vouchers`.
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::key_set"))]
    pub(crate) used_vouchers: HashSet<crate::items::Key>,
    /// `G.GAME.pool_flags` (game.lua:1932) — gros_michel_extinct in P3c.
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::key_set"))]
    pub(crate) pool_flags: HashSet<crate::items::Key>,

    /// The live shop (Some while in Shop/PackOpen after a cash out).
    pub(crate) shop: Option<ShopState>,
    /// The open booster pack (Some in State::PackOpen).
    pub(crate) pack: Option<PackState>,
    /// `G.GAME.current_round.voucher` — rolled at run start and on every
    /// boss defeat; persists across shops of the ante until redeemed.
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::opt_key"))]
    pub(crate) current_voucher: Option<crate::items::Key>,
    /// `G.GAME.round_resets.blind_tags` (Small, Big).
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::key"))]
    pub(crate) blind_tag_small: crate::items::Key,
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::key"))]
    pub(crate) blind_tag_big: crate::items::Key,
    /// `G.GAME.tags` — held (not yet triggered) tags, in acquisition order.
    pub(crate) tags: Vec<RunTag>,
    /// `G.GAME.orbital_choices[ante][Small/Big/Boss]` (UI_definitions.lua:1506).
    pub(crate) orbital_choices: HashMap<i64, [Option<HandType>; 3]>,
    /// `G.GAME.first_shop_buffoon` (common_events.lua:1945).
    pub(crate) first_shop_buffoon: bool,
    /// `G.GAME.shop_free` / `shop_d6ed` (Coupon/D6 once-per-shop latches).
    pub(crate) shop_free: bool,
    pub(crate) shop_d6ed: bool,
    /// `G.GAME.last_tarot_planet` (misc_functions.lua:1219; The Fool).
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::opt_key"))]
    pub(crate) last_tarot_planet: Option<crate::items::Key>,
    /// `G.GAME.consumeable_usage` counts by center key (Fool gating uses
    /// last_tarot_planet; Satellite reads this in P3c).
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::key_map"))]
    pub(crate) consumable_usage: HashMap<crate::items::Key, u32>,
    /// `G.GAME.consumeable_usage_total.tarot` (misc_functions.lua:1196-1206)
    /// — Fortune Teller's mult.
    pub(crate) tarots_used: i64,
    /// `G.GAME.ecto_minus` (card.lua:1499-1502).
    pub(crate) ecto_minus: i64,
    /// `G.GAME.probabilities.normal` (game.lua:1890; Oops! All 6s doubles it
    /// in P3c-2). Every joker/enhancement odds check reads it.
    pub(crate) prob_normal: f64,

    /// `G.GAME.current_round.mail_card` / `idol_card` / `ancient_card` /
    /// `castle_card` — re-rolled at run start (game.lua:2385-2389) and at
    /// the end of every won round (state_events.lua:273-276). The streams
    /// ('mail'/'idol'/'anc'/'cas' + ante) are consumed whether or not the
    /// matching joker is owned.
    pub(crate) mail_card_id: u8,
    pub(crate) idol_card: (u8, Suit),
    pub(crate) ancient_suit: Option<Suit>,
    pub(crate) castle_suit: Suit,

    /// `G.sort_id` (card.lua:24) — every Card creation we model advances it;
    /// relative order drives pseudoshuffle pre-sorts and joker-target picks.
    pub(crate) sort_id_counter: u32,
    /// `G.playing_card` — playing-card instance counter.
    pub(crate) playing_card_counter: u32,

    pub(crate) pending_cashout: i64,
    pub(crate) last_play: Option<PlayResult>,
    pub(crate) won: bool,
}

impl Run {
    /// Mirrors `Game:start_run` for a seeded Red Deck / White Stake run.
    pub fn new(seed: &str) -> Self {
        let mut rng = RngState::new(seed);
        let mut deck = red_deck();

        // game.lua:2177 — the ante-1 boss is rolled at run start, before the
        // first shuffle.
        let mut bosses_used: HashMap<&'static str, i64> = HashMap::new();
        let boss_choice = get_new_boss(&mut rng, &mut bosses_used, 1, WIN_ANTE);

        let mut run = Run {
            rng,
            state: State::BlindSelect,
            deck: Vec::new(),
            hand: Vec::new(),
            discard_pile: Vec::new(),
            destroyed_cards: Vec::new(),
            hands_table: HandsTable::new(),
            ante: 1,
            blind_ante: 1,
            round: 0,
            dollars: STARTING_DOLLARS,
            bankrupt_at: 0,
            chips: 0.0,
            // game.lua:2492-2493 — start_run pre-fills current_round's
            // hands/discards from round_resets before the first blind.
            hands_left: STARTING_HANDS,
            discards_left: STARTING_DISCARDS,
            unused_discards: 0,
            hand_size: HAND_SIZE,
            round_resets_hands: STARTING_HANDS,
            round_resets_discards: STARTING_DISCARDS,
            temp_handsize: None,
            base_reroll_cost: STARTING_REROLL_COST,
            reroll_cost: STARTING_REROLL_COST,
            reroll_cost_increase: 0,
            temp_reroll_cost: None,
            free_rerolls: 0,
            // game.lua:1895-1911 — G.GAME economy defaults.
            interest_cap: 25,
            interest_amount: 1,
            discount_percent: 0,
            edition_rate: 1.0,
            joker_rate: 20.0,
            tarot_rate: 4.0,
            planet_rate: 4.0,
            playing_card_rate: 0.0,
            spectral_rate: 0.0,
            joker_slots: JOKER_SLOTS,
            consumable_slots: 2,
            shop_joker_max: 2, // G.GAME.shop.joker_max (game.lua:1985)
            hands_played_round: 0,
            discards_used_round: 0,
            hands_played_total: 0,
            skips: 0,
            blind_on_deck: BlindStage::Small,
            active_blind: None,
            boss_choice,
            bosses_used,
            small_defeated: false,
            big_defeated: false,
            small_skipped: false,
            big_skipped: false,
            boss_rerolled: false,
            forced_tag: None,
            last_blind_boss: false,
            most_played_hand: HandType::HighCard, // game.lua:1964
            last_hand_played: None,
            forced_card: None,
            jokers: Vec::new(),
            consumables: Vec::new(),
            used_keys: HashMap::new(),
            used_vouchers: HashSet::new(),
            pool_flags: HashSet::new(),
            shop: None,
            pack: None,
            current_voucher: None,
            blind_tag_small: "",
            blind_tag_big: "",
            tags: Vec::new(),
            orbital_choices: HashMap::new(),
            first_shop_buffoon: false,
            shop_free: false,
            shop_d6ed: false,
            last_tarot_planet: None,
            consumable_usage: HashMap::new(),
            tarots_used: 0,
            ecto_minus: 1,
            prob_normal: 1.0,
            // game.lua:1949-1952 — the current_round defaults before the
            // start_run resets below.
            mail_card_id: 14,
            idol_card: (14, Suit::Spades),
            ancient_suit: None,
            castle_suit: Suit::Spades,
            sort_id_counter: 52, // the 52 deck cards hold sort_id 1..=52
            playing_card_counter: 52,
            pending_cashout: 0,
            last_play: None,
            won: false,
        };

        // game.lua:2178-2180 — right after the boss roll, the run-start shop
        // voucher and the ante-1 Small/Big skip tags are rolled ('Voucher1',
        // 'Tag1' x2). P3a documented these as gaps; wired now.
        run.current_voucher = Some(run.next_voucher_key(false));
        run.blind_tag_small = run.next_tag_key();
        run.blind_tag_big = run.next_tag_key();

        // game.lua:2383 — `self.deck:shuffle()` with the default 'shuffle'
        // key. Gameplay-inert (the round-start 'nr1' shuffle re-sorts by
        // sort_id first) but kept for stream fidelity.
        let seed_v = run.rng.pseudoseed("shuffle");
        pseudoshuffle(&mut deck, seed_v);
        run.deck = deck;

        // game.lua:2385-2389 — right after the start-run shuffle, the
        // per-round joker target cards are rolled (idol, mail, ancient —
        // with its previous suit cleared — then castle).
        run.reset_idol_card();
        run.reset_mail_rank();
        run.ancient_suit = None;
        run.reset_ancient_card();
        run.reset_castle_card();

        // Game:update_blind_select — the blind-choice UI rolls the ante's
        // Orbital-tag hands ('orbital' x3) on first display
        // (UI_definitions.lua:1506-1516). No tags exist yet, so the
        // 'immediate'/'new_blind_choice' passes are no-ops.
        run.enter_blind_select();
        run
    }

    // ------------------------------------------------------------------
    // Actions
    // ------------------------------------------------------------------

    /// `G.FUNCS.select_blind` -> `new_round()` (button_callbacks.lua:2504,
    /// state_events.lua:290-353) -> `Blind:set_blind` -> deal.
    pub fn select_blind(&mut self) -> Result<(), RunError> {
        if self.state != State::BlindSelect {
            return Err(RunError::WrongState);
        }
        self.round += 1; // ease_round(1)
        let proto: &'static BlindProto = match self.blind_on_deck {
            BlindStage::Small => &BL_SMALL,
            BlindStage::Big => &BL_BIG,
            BlindStage::Boss => boss_by_key(self.boss_choice),
        };
        // new_round(): hands/discards reset (state_events.lua:296-299);
        // round_bonus is 0 outside challenges/jokers.
        self.hands_left = self.round_resets_hands.max(1);
        self.discards_left = self.round_resets_discards.max(0);
        self.hands_played_round = 0;
        self.discards_used_round = 0;
        // state_events.lua:300-302 — reroll bookkeeping resets;
        // `free_rerolls = #find_joker('Chaos the Clown')` (:311; non-debuffed
        // Chaos jokers, misc_functions.lua:903-918).
        self.reroll_cost_increase = 0;
        self.free_rerolls = self
            .jokers
            .iter()
            .filter(|j| j.id == JokerId::Chaos && !j.debuffed)
            .count() as i64;
        self.calculate_reroll_cost(true);
        self.hands_table.reset_played_this_round();
        // blind.lua:97-99 — Blind:set_blind records G.GAME.last_blind
        // (Investment Tag's boss check).
        self.last_blind_boss = proto.is_boss;
        // state_events.lua:307-309 — wheel_flipped cleared on every card.
        for c in self
            .deck
            .iter_mut()
            .chain(self.hand.iter_mut())
            .chain(self.discard_pile.iter_mut())
        {
            c.face_down = false;
        }

        // Blind:set_blind (state_events.lua:333, blind.lua:78-216).
        let mut blind = ActiveBlind::new(proto, self.ante);
        if !blind.disabled {
            match proto.name {
                // blind.lua:179-182 — The Water: 0 discards.
                "The Water" => {
                    blind.discards_sub = self.discards_left;
                    self.discards_left -= blind.discards_sub;
                }
                // blind.lua:183-186 — The Needle: play only 1 hand
                // (round_resets.hands - 1 subtracted).
                "The Needle" => {
                    blind.hands_sub = STARTING_HANDS - 1;
                    self.hands_left -= blind.hands_sub;
                }
                // blind.lua:187-189 — The Manacle: -1 hand size.
                "The Manacle" => {
                    self.hand_size -= 1;
                }
                // blind.lua:190-205 — Amber Acorn flips every joker and,
                // with at least 2 jokers, shuffles them three times on the
                // 'aajk' stream. Each CardArea:shuffle pre-sorts by sort_id
                // (misc_functions.lua:209-211), so only the third seed
                // determines the final order — but all three draws happen.
                "Amber Acorn" => {
                    for j in self.jokers.iter_mut() {
                        j.flipped = true;
                    }
                    if self.jokers.len() > 1 {
                        let mut last_seed = 0.0;
                        for _ in 0..3 {
                            last_seed = self.rng.pseudoseed("aajk");
                        }
                        self.jokers.sort_by_key(|j| j.sort_id);
                        let mut lr = LuaRandom::seeded(last_seed);
                        for i in (1..self.jokers.len()).rev() {
                            let j = lr.random_range(1, (i + 1) as i64) as usize - 1;
                            self.jokers.swap(i, j);
                        }
                    }
                }
                _ => {}
            }
        }
        // blind.lua:207-213 — (re)stamp debuff flags on every playing card
        // (Pareidolia/Smeared feed the is_face/is_suit checks).
        let pareidolia = self.joker_owned(JokerId::Pareidolia);
        let smeared = self.joker_owned(JokerId::Smeared);
        for c in self
            .deck
            .iter_mut()
            .chain(self.hand.iter_mut())
            .chain(self.discard_pile.iter_mut())
        {
            c.debuff = blind.debuff_card(c, pareidolia, smeared);
        }
        self.active_blind = Some(blind);

        // state_events.lua:335-337 — `{setting_blind = true}` per joker
        // (Chicot/Madness/Burglar/Riff-raff/Cartomancer/Ceremonial Dagger/
        // Marble; see EngineHooks::setting_blind). The queued events run
        // BEFORE the 'nr' shuffle event (they were queued first).
        let env = self.joker_env();
        let out = {
            let mut hooks = EngineHooks::new(&mut self.jokers, env);
            hooks.setting_blind(&mut self.rng);
            hooks.out
        };
        self.apply_joker_outbox(out);

        // state_events.lua:344 — `G.deck:shuffle('nr'..ante)`.
        let seed = self.rng.pseudoseed(&format!("nr{}", self.ante));
        pseudoshuffle(&mut self.deck, seed);
        // Game:update_draw_to_hand (game.lua:3216-3218) — round_start_bonus
        // tags (Juggle: +h_size for this round via temp_handsize) fire on
        // entering DRAW_TO_HAND, before the deal.
        self.apply_tags(TagContext::RoundStartBonus);
        // DRAW_TO_HAND -> deal to hand size (state_events.lua:355-377,
        // game.lua:3208-3246).
        self.draw_to_hand();
        self.state = State::SelectingHand;
        Ok(())
    }

    /// Play 1-5 cards (hand indices). Mirrors
    /// `G.FUNCS.play_cards_from_highlighted` (state_events.lua:450-538) +
    /// `evaluate_play` + `Game:update_hand_played` (game.lua:3187-3206).
    pub fn play(&mut self, indices: &[usize]) -> Result<&PlayResult, RunError> {
        if self.state != State::SelectingHand {
            return Err(RunError::WrongState);
        }
        let sel = self.validate_selection(indices)?;
        // state_events.lua:459-461 — forced_selection cleared on every card.
        self.forced_card = None;
        self.hands_left -= 1; // ease_hands_played(-1)

        // `table.sort(G.hand.highlighted, a.T.x < b.T.x)` — screen order ==
        // hand order, so the played cards keep ascending hand-index order.
        let mut played = self.remove_from_hand(&sel);
        for card in &mut played {
            // state_events.lua:481 — Pillar bookkeeping.
            card.played_this_ante = true;
            // Cards flip face up when emplaced into the play area
            // (cardarea.lua:38-40).
            card.face_down = false;
        }

        // Blind:press_play (state_events.lua:488, blind.lua:464-508).
        self.press_play(played.len());

        // evaluate_play (state_events.lua:571-1086). The env's playing-card
        // tallies include the cards sitting in G.play.
        let blind_chips;
        let env = self.joker_env_with(&played);
        let mods = self.live_eval_mods();
        let (result, out) = {
            let mut hooks = EngineHooks::new(&mut self.jokers, env);
            let mut ctx = ScoreContext {
                rng: &mut self.rng,
                hands: &mut self.hands_table,
                dollars: &mut self.dollars,
                blind: self.active_blind.as_mut(),
                most_played: self.most_played_hand,
                mods,
                prob_normal: self.prob_normal,
            };
            let r = evaluate_play(&mut ctx, &mut played, &mut self.hand, &mut hooks);
            blind_chips = self.active_blind.as_ref().map(|b| b.chips).unwrap_or(0.0);
            (r, hooks.out)
        };
        // The events the jokers queued mid-evaluation (8 Ball/Superposition
        // tarots, Ice Cream melting) run once the synchronous part of
        // evaluate_play is done — before the round outcome is processed.
        self.apply_joker_outbox(out);
        // state_events.lua:576 — set even for debuffed hands.
        self.last_hand_played = Some(result.hand_type);
        // state_events.lua:1049 — chips += floor(hand_chips * mult).
        self.chips += result.score;

        // draw_from_play_to_discard (state_events.lua:1088-1097): shattered/
        // destroyed cards never reach the discard pile.
        for (i, card) in played.into_iter().enumerate() {
            if result.destroyed.contains(&i) {
                self.destroyed_cards.push(card);
            } else {
                self.discard_pile.push(card);
            }
        }
        // state_events.lua:523-524 — current_round.hands_played and the
        // run-total G.GAME.hands_played (Handy Tag reads the latter).
        self.hands_played_round += 1;
        self.hands_played_total += 1;
        self.last_play = Some(result);

        // Game:update_hand_played (game.lua:3196-3200): the round ends when
        // the blind is beaten OR no hands remain — end_round() itself
        // decides game over vs continue (Mr. Bones can save).
        if self.chips - blind_chips >= 0.0 || self.hands_left < 1 {
            self.end_round();
        } else {
            self.draw_to_hand();
        }
        Ok(self.last_play.as_ref().unwrap())
    }

    /// Discard 1-5 cards (hand indices); redraw.
    /// `G.FUNCS.discard_cards_from_highlighted` (state_events.lua:379-448).
    pub fn discard(&mut self, indices: &[usize]) -> Result<(), RunError> {
        if self.state != State::SelectingHand {
            return Err(RunError::WrongState);
        }
        if self.discards_left <= 0 {
            return Err(RunError::NoDiscardsLeft);
        }
        let sel = self.validate_selection(indices)?;
        // state_events.lua:384-386.
        self.forced_card = None;
        let cards = self.remove_from_hand(&sel);
        self.discard_selected(cards, false);
        self.discards_left -= 1; // ease_discard(-1)
        self.discards_used_round += 1; // state_events.lua:437
        self.draw_to_hand();
        Ok(())
    }

    /// `G.FUNCS.cash_out` (button_callbacks.lua:2912-2956) + `reset_blinds`
    /// (common_events.lua:2326-2336), landing in the shop.
    pub fn cash_out(&mut self) -> Result<(), RunError> {
        if self.state != State::RoundEval {
            return Err(RunError::WrongState);
        }
        // button_callbacks.lua:2918 — shuffle keyed on the *current* ante
        // (already incremented if a boss was just defeated).
        let seed = self.rng.pseudoseed(&format!("cashout{}", self.ante));
        pseudoshuffle(&mut self.deck, seed);

        // :2933-2934 — hands/discards restocked from round_resets (+ the
        // round_bonus fields, 0 outside challenges/jokers).
        self.discards_left = self.round_resets_discards.max(0);
        self.hands_left = self.round_resets_hands.max(1);
        // :2937-2938 — the per-shop Coupon/D6 latches clear.
        self.shop_free = false;
        self.shop_d6ed = false;

        self.dollars += self.pending_cashout; // ease_dollars(current_round.dollars)
        self.pending_cashout = 0;
        self.chips = 0.0; // ease_chips(0)

        // Advance the blind track.
        let was_boss = matches!(
            self.active_blind.as_ref().map(|b| b.proto),
            Some(b) if b.is_boss
        );
        match self.active_blind.as_ref().map(|b| b.proto) {
            Some(b) if b.key == BL_SMALL.key => self.small_defeated = true,
            Some(b) if b.key == BL_BIG.key => self.big_defeated = true,
            _ => {}
        }
        self.active_blind = None;
        if was_boss {
            // :2950-2953 — after a boss, blind_ante snaps to the (new) ante
            // and the next Small/Big skip tags are rolled ('Tag<ante>' x2)...
            self.blind_ante = self.ante;
            self.blind_tag_small = self.next_tag_key();
            self.blind_tag_big = self.next_tag_key();
            // ...then reset_blinds (common_events.lua:2328-2335) rolls the
            // next boss and clears the reroll latch.
            self.small_defeated = false;
            self.big_defeated = false;
            self.small_skipped = false;
            self.big_skipped = false;
            let boss = get_new_boss(&mut self.rng, &mut self.bosses_used, self.ante, WIN_ANTE);
            self.boss_choice = boss;
            self.boss_rerolled = false;
        }
        // UI_definitions.lua:1441-1444 — blind_on_deck from the states.
        self.blind_on_deck = if !self.small_defeated && !self.small_skipped {
            BlindStage::Small
        } else if !self.big_defeated && !self.big_skipped {
            BlindStage::Big
        } else {
            BlindStage::Boss
        };

        if self.won {
            self.state = State::Won;
        } else {
            // G.STATE = G.STATES.SHOP (:2936); Game:update_shop generates
            // the shop contents on entry.
            self.state = State::Shop;
            self.generate_shop();
        }
        Ok(())
    }

    /// `Blind:disable()` (blind.lua:356-415) — the hook boss-disabling
    /// effects (Chicot, Director's Cut ban...) will call in P3b/P3c.
    pub fn disable_boss(&mut self) {
        let Some(blind) = self.active_blind.as_mut() else {
            return;
        };
        if blind.disabled {
            return;
        }
        let name = blind.proto.name;
        blind.disable(); // Wall chips/2, Vessel chips/3

        // blind.lua:361-363 — The Water gives the discards back.
        if name == "The Water" {
            self.discards_left += blind.discards_sub;
        }
        // blind.lua:364-373 — face-down cards in hand flip back up.
        for c in self.hand.iter_mut() {
            c.face_down = false;
        }
        // blind.lua:374-376 — The Needle gives the hands back.
        if name == "The Needle" {
            self.hands_left += blind.hands_sub;
        }
        // blind.lua:381-385 — Cerulean Bell releases the forced card.
        if name == "Cerulean Bell" {
            self.forced_card = None;
        }
        // blind.lua:386-390 — The Manacle: hand size restored + 1 card drawn.
        if name == "The Manacle" {
            self.hand_size += 1;
            if let Some(card) = self.deck.pop() {
                // stay_flipped is false for a disabled blind.
                self.hand.push(card);
                self.sort_hand_desc();
            }
        }
        // blind.lua:407-412 — debuffs recomputed (all clear when disabled).
        for c in self
            .deck
            .iter_mut()
            .chain(self.hand.iter_mut())
            .chain(self.discard_pile.iter_mut())
        {
            c.debuff = false;
        }
        // blind.lua:397-406 — if the (halved) requirement is now met, the
        // round ends immediately.
        let met = self
            .active_blind
            .as_ref()
            .is_some_and(|b| b.proto.is_boss && self.chips - b.chips >= 0.0);
        if met && self.state == State::SelectingHand {
            self.end_round();
        }
    }

    /// Test/tooling helper: override the upcoming boss (must be in
    /// BlindSelect). The real game rolls it via the 'boss' stream; forcing a
    /// key here does not consume any stream.
    pub fn force_boss(&mut self, key: &'static str) {
        assert!(self.state == State::BlindSelect);
        let _ = boss_by_key(key); // validate
        self.boss_choice = key;
    }

    /// Test/tooling helper mirroring the game's `G.FORCE_TAG` debug flag:
    /// while set, `get_next_tag_key` returns this key WITHOUT consuming the
    /// 'Tag' stream (common_events.lua:1915).
    pub fn force_tag(&mut self, key: Option<&'static str>) {
        if let Some(k) = key {
            assert!(crate::items::tag_by_key(k).is_some(), "unknown tag {k}");
            // As if G.FORCE_TAG had been set before the run-start rolls:
            // the already-rolled blind tags take the forced key too.
            self.blind_tag_small = k;
            self.blind_tag_big = k;
        }
        self.forced_tag = key;
    }

    /// Test/tooling helper: set the money counter directly (no stream use).
    /// Used by the oracle-vector drivers and, later, RL domain
    /// randomization.
    pub fn debug_set_dollars(&mut self, dollars: i64) {
        self.dollars = dollars;
    }

    /// `level_up_hand` (common_events.lua:464-469) — exposed for tests and
    /// as the entry point the P3b Planet cards will use.
    pub fn level_up_hand(&mut self, ht: HandType, amount: i64) {
        self.hands_table.level_up(ht, amount);
    }

    /// Test/tooling helper: mutate a card everywhere it lives (deck, hand,
    /// discard pile), matching by sort_id. Stands in for the P3b Tarot layer.
    pub fn modify_card(&mut self, sort_id: u32, f: impl Fn(&mut Card)) {
        for c in self
            .deck
            .iter_mut()
            .chain(self.hand.iter_mut())
            .chain(self.discard_pile.iter_mut())
        {
            if c.sort_id == sort_id {
                f(c);
            }
        }
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    /// `Blind:press_play` (blind.lua:464-508), invoked from
    /// play_cards_from_highlighted after the cards moved to the play area
    /// (state_events.lua:488) and before evaluate_play.
    fn press_play(&mut self, played_count: usize) {
        let Some(blind) = self.active_blind.as_mut() else {
            return;
        };
        if blind.disabled {
            return;
        }
        match blind.proto.name {
            // blind.lua:466-487 — The Hook: discard 2 random cards from the
            // remaining hand. `pseudorandom_element(_cards,
            // pseudoseed('hook'))` sorts the candidates by sort_id (they are
            // Card tables), draws math.random(#cards), and removes the pick
            // before the second draw. `if G.hand.cards[i]` gates each pick on
            // the ORIGINAL hand size (highlighting does not remove cards).
            "The Hook" => {
                blind.triggered = true;
                let mut candidates: Vec<u32> = self.hand.iter().map(|c| c.sort_id).collect();
                let mut picked: Vec<u32> = Vec::new();
                for i in 0..2 {
                    if self.hand.len() > i {
                        let mut sorted = candidates.clone();
                        sorted.sort_unstable();
                        let seed = self.rng.pseudoseed("hook");
                        let j = LuaRandom::seeded(seed).random_range(1, sorted.len() as i64);
                        let sid = sorted[(j - 1) as usize];
                        candidates.retain(|&s| s != sid);
                        picked.push(sid);
                    }
                }
                if !picked.is_empty() {
                    // discard_cards_from_highlighted(nil, hook=true)
                    // (blind.lua:482): highlighted re-sorted by T.x == hand
                    // order (state_events.lua:392).
                    let mut idx: Vec<usize> = self
                        .hand
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| picked.contains(&c.sort_id))
                        .map(|(i, _)| i)
                        .collect();
                    idx.sort_unstable();
                    let cards = self.remove_from_hand(&idx);
                    self.discard_selected(cards, true);
                }
            }
            // blind.lua:497-507 — The Tooth: lose $1 per card played
            // (`ease_dollars(-1)` per play card; money can go negative).
            "The Tooth" => {
                blind.triggered = true;
                self.dollars -= played_count as i64;
            }
            // blind.lua:494-496 — The Fish arms `prepped`: the next draw to
            // hand comes face down.
            "The Fish" => {
                blind.prepped = true;
            }
            // blind.lua:488-493 — Crimson Heart arms `prepped` when at least
            // one joker exists; drawn_to_hand then debuffs a random joker
            // ('crimson_heart' stream).
            "Crimson Heart" if !self.jokers.is_empty() => {
                blind.prepped = true;
            }
            _ => {}
        }
    }

    /// Shared discard path: `G.FUNCS.discard_cards_from_highlighted`
    /// (state_events.lua:379-448) minus the hand-index bookkeeping the caller
    /// did. `hook == true` is The Hook's forced discard (no discard spent,
    /// state_events.lua:432-446) — the per-card joker hooks still run
    /// (Mail-In Rebate/Green Joker/Faceless trigger on hooked discards).
    fn discard_selected(&mut self, cards: Vec<Card>, hook: bool) {
        let env = self.joker_env();
        let mods = self.live_eval_mods();
        let mut hooks = EngineHooks::new(&mut self.jokers, env);
        // state_events.lua:394-396 — Burnt Joker levels the selection's hand.
        hooks.pre_discard(&cards, hook, &mut self.hands_table, &mods);
        let mut purple_triggers = 0usize;
        let mut destroyed_now: Vec<Card> = Vec::new();
        // Purple-seal buffer bookkeeping: `#G.consumeables.cards +
        // G.GAME.consumeable_buffer < G.consumeables.config.card_limit`
        // (card.lua:2254-2255) is checked synchronously per card; the tarot
        // creations run as queued events afterwards.
        for card in &cards {
            // Card:calculate_seal{discard = true} (state_events.lua:400,
            // card.lua:2253-2268). Debuffed cards do nothing (card.lua:2243).
            if card.seal == Seal::Purple
                && !card.debuff
                && self.consumables.len() + purple_triggers < self.consumable_slots
            {
                purple_triggers += 1;
            }
            // Per-card joker discard hook (state_events.lua:402-409);
            // eval.remove (Trading Card) destroys the card.
            let removed = hooks.on_discard_card(card, &cards, &mut self.dollars, &mut self.rng);
            if removed {
                destroyed_now.push(*card);
                self.destroyed_cards.push(*card);
            } else {
                let mut c = *card;
                c.face_down = false;
                self.discard_pile.push(c);
            }
        }
        // state_events.lua:424-428 — destroyed discards fire the
        // remove_playing_cards window (Caino counts faces, Glass Joker
        // counts shattered Glass — a Trading-destroyed Glass card shatters).
        if !destroyed_now.is_empty() {
            hooks.on_cards_destroyed(&destroyed_now);
        }
        let out = hooks.out;
        self.apply_joker_outbox(out);
        // The queued create_card('Tarot', …, '8ba') events (card.lua:2260):
        // each consumes the 'Tarot8ba<ante>' stream (with resamples) and
        // marks its key used before the next one rolls. `soulable` is nil,
        // so no 'soul_Tarot' poll.
        for _ in 0..purple_triggers {
            let key = self.create_consumable_key(ConsumableSet::Tarot, "8ba", false);
            self.add_consumable(key);
        }
    }

    /// `end_round()` (state_events.lua:87-288): joker end_of_round evals
    /// (with the game_over flag and Mr. Bones' save), then either GAME_OVER
    /// or the full win path — most-played update, held-card end-of-round
    /// effects (Gold cards, blue seals), hand -> discard, ante-up on boss,
    /// everything back to the deck, round-eval payout.
    fn end_round(&mut self) {
        // state_events.lua:92-97 — game_over unless the blind was beaten.
        let mut game_over = self
            .active_blind
            .as_ref()
            .map(|b| self.chips - b.chips < 0.0)
            .unwrap_or(false);
        #[cfg(feature = "p5-debug")]
        eprintln!(
            "end_round: chips={} blind={:?} game_over={game_over}",
            self.chips,
            self.active_blind.as_ref().map(|b| (b.proto.name, b.chips))
        );
        // state_events.lua:99-110 — joker end_of_round evals come first
        // (Gros Michel/Cavendish rolls, food decay, To Do List re-roll,
        // Mr. Bones); the destruction events they queue run before anything
        // else looks at the joker area.
        let env = self.joker_env();
        let (saved, out) = {
            let mut hooks = EngineHooks::new(&mut self.jokers, env);
            let saved = hooks.end_of_round(&self.hands_table, game_over, &mut self.rng);
            (saved, hooks.out)
        };
        self.apply_joker_outbox(out);
        #[cfg(feature = "p5-debug")]
        eprintln!("end_round: saved={saved}");
        if saved {
            game_over = false;
        }
        if game_over {
            self.state = State::GameOver;
            return;
        }
        // state_events.lua:124.
        self.unused_discards += self.discards_left;
        let blind = self.active_blind.as_ref().expect("round without blind");
        let blind_proto = blind.proto;
        let blind_disabled = blind.disabled;
        // Win check happens BEFORE ease_ante(1) (state_events.lua:111-114);
        // a Mr. Bones save on the ante-8 boss still wins the run
        // (state_events.lua:112 runs whenever the round survived).
        if self.ante == WIN_ANTE && blind_proto.is_boss {
            self.won = true;
        }
        // state_events.lua:129-138 — most played poker hand, recomputed only
        // when a Boss round is won (feeds The Ox and its blind text).
        if blind_proto.is_boss {
            let mut best = HandType::HighCard;
            let mut best_played: i64 = -1;
            for ht in MOST_PLAYED_SCAN {
                let played = self.hands_table.get(ht).played as i64;
                // `v.played > _played or (v.played == _played and 100 >
                // v.order)` — the tie branch always passes (see
                // MOST_PLAYED_SCAN docs).
                if played >= best_played {
                    best_played = played;
                    best = ht;
                }
            }
            self.most_played_hand = best;
        }

        // state_events.lua:171-233 — per held card end-of-round effects, in
        // hand (display) order, with red-seal/joker retriggers.
        let env = self.joker_env();
        let mut hooks = EngineHooks::new(&mut self.jokers, env);
        let mut planet_triggers = 0usize;
        for i in 0..self.hand.len() {
            let card = self.hand[i];
            let mut total = 1usize;
            let mut j = 0usize;
            while j < total {
                // Card:get_end_of_round_effect (card.lua:1033-1065).
                let mut h_dollars = 0i64;
                let mut blue_fired = false;
                if !card.debuff {
                    if card.enhancement == Enhancement::Gold {
                        // ability.h_dollars = 3 (m_gold, game.lua:654).
                        h_dollars = 3;
                    }
                    if card.seal == Seal::Blue
                        && self.consumables.len() + planet_triggers < self.consumable_slots
                    {
                        // card.lua:1040-1062: queue a Planet for
                        // G.GAME.last_hand_played (forced key -> NO RNG).
                        planet_triggers += 1;
                        blue_fired = true;
                    }
                }
                let joker_effs = hooks.end_of_round_card(&card, &mut self.rng);
                if j == 0 {
                    // Red-seal retrigger (state_events.lua:189-197): only if
                    // the card produced any effect; Mime uses the same gate.
                    let has = h_dollars > 0 || blue_fired || !joker_effs.is_empty();
                    if card.seal == Seal::Red && !card.debuff && has {
                        total += 1;
                    }
                    total += hooks.end_of_round_card_repetitions(&card, has) as usize;
                }
                // state_events.lua:221-224 — ease_dollars(h_dollars).
                self.dollars += h_dollars;
                for eff in &joker_effs {
                    self.dollars += eff.h_dollars;
                }
                j += 1;
            }
        }
        // Joker payouts (state_events.lua:1175-1182) — per joker,
        // Card:calculate_dollar_bonus; no RNG and no state writes, so the
        // total is captured here and added into the payout below.
        let joker_dollars = hooks.dollar_bonus();
        let out = hooks.out;
        self.apply_joker_outbox(out);

        // The queued blue-seal events (card.lua:1043-1060) each create the
        // Planet matching last_hand_played; forced keys consume no RNG.
        if let Some(last) = self.last_hand_played {
            for _ in 0..planet_triggers {
                let key = planet_for_hand(last);
                self.add_consumable(key);
            }
        }

        // G.FUNCS.evaluate_round (state_events.lua:1135-1208): the payout is
        // computed off the post-end-of-round dollar total (Gold cards pay
        // before interest is counted). A disabled blind still pays its base
        // reward (Blind:disable doesn't touch self.dollars). On a Mr. Bones
        // save (chips < blind.chips) the blind reward row pays 0
        // (state_events.lua:985-992, `saved = true`) — found by P5 live
        // cross-validation.
        let blind_chips_req = self.active_blind.as_ref().map(|b| b.chips).unwrap_or(0.0);
        let mut dollars = if self.chips - blind_chips_req >= 0.0 {
            blind_proto.dollars
        } else {
            0
        };
        if self.hands_left > 0 {
            // `hands_left * (G.GAME.modifiers.money_per_hand or 1)`
            dollars += self.hands_left;
        }
        // Joker payouts (state_events.lua:1175-1182).
        dollars += joker_dollars;
        // Tag payouts (state_events.lua:1183-1190) — Investment Tag pays $25
        // when the just-defeated blind was a boss (tag.lua:117-131); every
        // held Investment triggers (no break in the evaluate_round loop).
        dollars += self.apply_eval_tags();
        if self.dollars >= 5 {
            // interest_amount * min(floor(dollars/5), interest_cap/5)
            // (state_events.lua:1192); Seed Money/Money Tree raise the cap.
            dollars += self.interest_amount
                * (((self.dollars as f64) / 5.0).floor() as i64).min(self.interest_cap / 5);
        }
        self.pending_cashout = dollars;

        // Blind:defeat (state_events.lua via evaluate_round:1148-1155,
        // blind.lua:341-343): The Manacle returns the hand size.
        if blind_proto.name == "The Manacle" && !blind_disabled {
            self.hand_size += 1;
        }
        // Blind:defeat also queues `set_blind(nil, nil, true)`
        // (blind.lua:365) whose "add new debuffs" pass re-stamps every
        // playing card against the EMPTY blind — i.e. all card debuffs
        // clear on a won round (P5 live cross-validation).
        for c in self
            .deck
            .iter_mut()
            .chain(self.hand.iter_mut())
            .chain(self.discard_pile.iter_mut())
        {
            c.debuff = false;
        }

        // draw_from_hand_to_discard, then on a boss ease_ante(1), then
        // draw_from_discard_to_deck (state_events.lua:237-250).
        for c in &mut self.hand {
            c.face_down = false;
        }
        self.discard_pile.append(&mut self.hand);
        if blind_proto.is_boss {
            self.ante += 1;
            // state_events.lua:263-267 — on Boss defeat the next shop
            // voucher is rolled ('Voucher'+NEW ante, get_next_voucher_key)
            // and played_this_ante clears on every card.
            self.current_voucher = Some(self.next_voucher_key(false));
            for c in self.deck.iter_mut().chain(self.discard_pile.iter_mut()) {
                c.played_this_ante = false;
            }
        }
        // draw_from_discard_to_deck drains the discard pile from its END
        // and `CardArea:emplace` inserts every card at index 1 for deck
        // areas (cardarea.lua:50-51), so the block lands — order intact —
        // at the FRONT (bottom) of the deck. Inert for gameplay (the next
        // pseudoshuffle pre-sorts by sort_id) but P5 diffs the raw order.
        let mut returned = std::mem::take(&mut self.discard_pile);
        returned.append(&mut self.deck);
        self.deck = returned;
        // state_events.lua:270-271 — Juggle's temp hand size is removed and
        // D6's temp reroll cost expires (reroll_cost recomputed).
        if let Some(t) = self.temp_handsize.take() {
            self.hand_size -= t;
        }
        if self.temp_reroll_cost.take().is_some() {
            self.calculate_reroll_cost(true);
        }
        // state_events.lua:273-276 — the per-round joker target cards
        // re-roll ('idol'/'mail'/'anc'/'cas' + the post-boss ante). The
        // draws happen whether or not the jokers are owned.
        self.reset_idol_card();
        self.reset_mail_rank();
        self.reset_ancient_card();
        self.reset_castle_card();
        // Joker flips (Amber Acorn) and debuffs (Crimson Heart) end with the
        // round (the next set_blind re-stamps debuffs, blind.lua:211-213);
        // un-debuffing restores the joker's passives (Card:set_debuff ->
        // add_to_deck(true), card.lua:526-538).
        for j in self.jokers.iter_mut() {
            j.flipped = false;
        }
        for i in 0..self.jokers.len() {
            self.set_joker_debuffed(i, false);
        }
        // state_events.lua:277-280.
        self.forced_card = None;

        self.state = State::RoundEval;
    }

    /// `G.FUNCS.draw_from_deck_to_hand` (state_events.lua:355-377): draw
    /// `min(#deck, hand_size - #hand)` cards — or exactly `min(#deck, 3)`
    /// while The Serpent is live and a hand/discard was already used — from
    /// the END of the deck (cardarea.lua:76-77). Each card consults
    /// `Blind:stay_flipped` (common_events.lua:400-408); afterwards
    /// `Blind:drawn_to_hand` runs (game.lua:3238).
    fn draw_to_hand(&mut self) {
        let serpent = self.active_blind.as_ref().is_some_and(|b| {
            b.proto.name == "The Serpent"
                && !b.disabled
                && (self.hands_played_round > 0 || self.discards_used_round > 0)
        });
        let hand_space = if serpent {
            // state_events.lua:363-368 — always 3, even past the hand limit.
            self.deck.len().min(3)
        } else {
            self.deck
                .len()
                .min((self.hand_size - self.hand.len() as i64).max(0) as usize)
        };
        for _ in 0..hand_space {
            let mut card = self.deck.pop().expect("deck underflow");
            let pareidolia = self.joker_owned(JokerId::Pareidolia);
            let flipped = match self.active_blind.as_mut() {
                Some(b) => b.stay_flipped(
                    &mut self.rng,
                    &card,
                    self.hands_played_round,
                    self.discards_used_round,
                    self.prob_normal,
                    pareidolia,
                ),
                None => false,
            };
            card.face_down = flipped;
            self.hand.push(card);
        }
        self.sort_hand_desc();

        // game.lua:3226-3231 — first_hand_drawn joker hook (Certificate
        // creates a random sealed card into the hand; DNA/Trading only
        // juice).
        if self.hands_played_round == 0 && self.discards_used_round == 0 {
            let env = self.joker_env();
            let out = {
                let mut hooks = EngineHooks::new(&mut self.jokers, env);
                hooks.first_hand_drawn();
                hooks.out
            };
            self.apply_joker_outbox(out);
        }

        // Blind:drawn_to_hand (game.lua:3238, blind.lua:572-603).
        let mut crimson = false;
        if let Some(blind) = self.active_blind.as_mut() {
            if !blind.disabled && blind.proto.name == "Cerulean Bell" {
                let has_forced = self
                    .forced_card
                    .is_some_and(|sid| self.hand.iter().any(|c| c.sort_id == sid));
                if !has_forced && !self.hand.is_empty() {
                    // pseudorandom_element(G.hand.cards,
                    // pseudoseed('cerulean_bell')) — candidates sorted by
                    // sort_id, one math.random(#hand) draw (blind.lua:583).
                    let mut ids: Vec<u32> = self.hand.iter().map(|c| c.sort_id).collect();
                    ids.sort_unstable();
                    let seed = self.rng.pseudoseed("cerulean_bell");
                    let j = LuaRandom::seeded(seed).random_range(1, ids.len() as i64);
                    self.forced_card = Some(ids[(j - 1) as usize]);
                }
            }
            // blind.lua:588-600 — Crimson Heart (while prepped) clears all
            // joker debuffs and debuffs one at random: candidates are the
            // previously non-debuffed jokers (all of them when fewer than 2),
            // picked by pseudorandom_element sorted by sort_id.
            crimson = !blind.disabled && blind.proto.name == "Crimson Heart" && blind.prepped;
            // blind.lua:602 — prepped always cleared after a draw.
            blind.prepped = false;
        }
        if crimson {
            let mut cand: Vec<u32> = self
                .jokers
                .iter()
                .filter(|j| !j.debuffed || self.jokers.len() < 2)
                .map(|j| j.sort_id)
                .collect();
            // Un-debuffing (and debuffing below) toggles the joker's
            // passive add_to_deck effects (Card:set_debuff,
            // card.lua:526-538).
            for i in 0..self.jokers.len() {
                self.set_joker_debuffed(i, false);
            }
            if !cand.is_empty() {
                cand.sort_unstable();
                let seed = self.rng.pseudoseed("crimson_heart");
                let i = LuaRandom::seeded(seed).random_range(1, cand.len() as i64) - 1;
                let sid = cand[i as usize];
                if let Some(idx) = self.jokers.iter().position(|j| j.sort_id == sid) {
                    self.set_joker_debuffed(idx, true);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Joker engine glue
    // ------------------------------------------------------------------

    /// Snapshot of the run state the joker dispatcher reads (see
    /// `JokerEnv`). Rebuilt right before every hook window.
    pub(crate) fn joker_env(&self) -> JokerEnv {
        self.joker_env_with(&[])
    }

    /// `joker_env` with extra owned playing cards counted into the
    /// Card:update tallies — during evaluate_play the played cards are in
    /// G.play but still members of `G.playing_cards`.
    pub(crate) fn joker_env_with(&self, in_play: &[Card]) -> JokerEnv {
        let mut steel = 0i64;
        let mut stone = 0i64;
        let mut driver = 0i64;
        let mut nine = 0i64;
        let mut len = 0usize;
        for c in self
            .deck
            .iter()
            .chain(self.hand.iter())
            .chain(self.discard_pile.iter())
            .chain(in_play.iter())
        {
            len += 1;
            match c.enhancement {
                Enhancement::Steel => steel += 1,
                Enhancement::Stone => stone += 1,
                _ => {}
            }
            // Driver's License: any non-c_base center (card.lua:4179-4184).
            if c.enhancement != Enhancement::None {
                driver += 1;
            }
            // Cloud 9: get_id() == 9 — Stone cards never match.
            if c.enhancement != Enhancement::Stone && c.rank.id() == 9 {
                nine += 1;
            }
        }
        // Satellite: distinct Planet keys used this run (card.lua:1667-1673).
        let planets_used = self
            .consumable_usage
            .keys()
            .filter(|k| {
                matches!(
                    crate::items::consumable_by_key(k),
                    Some((_, ConsumableSet::Planet))
                )
            })
            .count() as i64;
        let blind = self.active_blind.as_ref();
        JokerEnv {
            ante: self.ante,
            prob_normal: self.prob_normal,
            deck_len: self.deck.len(),
            discards_left: self.discards_left,
            discards_used_round: self.discards_used_round,
            hands_left: self.hands_left,
            hands_played_round: self.hands_played_round,
            hands_played_total: self.hands_played_total,
            dollars: self.dollars,
            skips: self.skips,
            consumables_len: self.consumables.len(),
            consumable_slots: self.consumable_slots,
            joker_slots: self.joker_slots,
            tarots_used: self.tarots_used,
            planets_used,
            mail_card_id: self.mail_card_id,
            idol_card: self.idol_card,
            ancient_suit: self.ancient_suit,
            castle_suit: self.castle_suit,
            discount_percent: self.discount_percent,
            observatory: self.used_vouchers.contains("v_observatory"),
            consumable_keys: self.consumables.iter().map(|c| c.key).collect(),
            playing_cards_len: len,
            starting_deck_size: 52,
            steel_tally: steel,
            stone_tally: stone,
            driver_tally: driver,
            nine_tally: nine,
            blind_is_boss: blind.map(|b| b.proto.is_boss).unwrap_or(false),
            blind_chips: blind.map(|b| b.chips).unwrap_or(0.0),
            game_chips: self.chips,
            sort_id_base: self.sort_id_counter,
        }
    }

    /// The live joker-driven hand-evaluation modifiers
    /// (`next(find_joker(...))` — debuffed copies count for nothing).
    pub(crate) fn live_eval_mods(&self) -> EvalMods {
        EvalMods {
            four_fingers: self.joker_owned(JokerId::FourFingers),
            shortcut: self.joker_owned(JokerId::Shortcut),
            smeared: self.joker_owned(JokerId::Smeared),
        }
    }

    /// `next(find_joker(name))` for a joker id (non-debuffed).
    pub(crate) fn joker_owned(&self, id: JokerId) -> bool {
        self.jokers.iter().any(|j| j.id == id && !j.debuffed)
    }

    /// `playing_card_joker_effects(cards)` at Run level (shop buys, pack
    /// picks, tarot creations): Hologram +0.25 x_mult per card
    /// (card.lua:2456-2461, blueprint-gated).
    pub(crate) fn joker_playing_cards_added(&mut self, count: usize) {
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(&self.jokers, i) else {
                continue;
            };
            if self.jokers[t].id == JokerId::Hologram && !bp {
                self.jokers[t].state.x_mult += 0.25 * count as f64;
            }
        }
    }

    /// `{using_consumeable = true}` per joker (button_callbacks.lua:2220):
    /// Constellation +0.1 x_mult per Planet used (card.lua:2727-2733),
    /// Glass Joker +0.75 per Glass card destroyed by The Hanged Man
    /// (card.lua:2709-2721). Both blueprint-gated.
    pub(crate) fn joker_consumable_used(&mut self, key: &str, hanged_glass: usize) {
        let is_planet = matches!(
            crate::items::consumable_by_key(key),
            Some((_, ConsumableSet::Planet))
        );
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(&self.jokers, i) else {
                continue;
            };
            if bp {
                continue;
            }
            match self.jokers[t].id {
                JokerId::Constellation if is_planet => {
                    self.jokers[t].state.x_mult += 0.1; // extra (game.lua:446)
                }
                JokerId::Glass if key == "c_hanged_man" && hanged_glass > 0 => {
                    // extra (0.75) per shattered glass (game.lua:507).
                    self.jokers[t].state.x_mult += 0.75 * hanged_glass as f64;
                }
                _ => {}
            }
        }
    }

    /// Executes the events a hook window queued, in FIFO order — the
    /// `G.E_MANAGER` events run once evaluate_play/end_round's synchronous
    /// body is done.
    pub(crate) fn apply_joker_outbox(&mut self, out: JokerOutbox) {
        // In-window Card creations (DNA) consumed sort/playing-card ids.
        self.sort_id_counter += out.sort_ids_used;
        self.playing_card_counter += out.playing_cards_delta;
        for ev in out.events {
            match ev {
                OutEvent::CreateConsumable(set, append) => {
                    // create_card(set, G.consumeables, …, append): the pool
                    // draw happens at event time, so earlier creations
                    // already count as used for the culling.
                    let key = self.create_consumable_key(set, append, false);
                    self.add_consumable(key);
                }
                OutEvent::DestroyJoker(sort_id) => {
                    if let Some(i) = self.jokers.iter().position(|j| j.sort_id == sort_id) {
                        self.remove_joker_at(i);
                    }
                }
                OutEvent::SetGrosMichelExtinct => {
                    // card.lua:3037.
                    self.pool_flags.insert("gros_michel_extinct");
                }
                OutEvent::CreateJokers(count) => {
                    // Riff-raff's queued batch (card.lua:2532-2542):
                    // create_card('Joker', G.jokers, nil, 0, …, 'rif') —
                    // forced rarity 0 rolls the common pool
                    // 'Joker1rif<ante>' and the edition 'edirif<ante>';
                    // area G.jokers = no etper poll.
                    for _ in 0..count {
                        let j = self.create_joker(Some(0.0), false, "rif", JokerArea::Direct);
                        self.add_joker(j);
                    }
                }
                OutEvent::DisableBoss => self.disable_boss(),
                OutEvent::Burglar => {
                    // card.lua:2523-2527 — ease_discard(-discards_left),
                    // ease_hands_played(extra = 3).
                    self.discards_left = 0;
                    self.hands_left += 3;
                }
                OutEvent::CreateMarbleStone => {
                    // card.lua:2582-2598 — a Stone card with a 'marb_fr'
                    // front, emplaced through G.play into the deck (bottom).
                    let idx = element_index(&mut self.rng, "marb_fr", 52);
                    let (suit, rank) = crate::items::card_front_from_index(idx);
                    self.playing_card_counter += 1;
                    let sort_id = self.next_sort_id();
                    let mut card = Card::new(rank, suit, sort_id);
                    card.enhancement = Enhancement::Stone;
                    self.deck.insert(0, card);
                }
                OutEvent::CreateCertificate => {
                    // card.lua:2463-2479 — create_playing_card with a
                    // 'cert_fr' front into the HAND, then the 'certsl' seal
                    // poll, then the blind's debuff check + hand sort.
                    let idx = element_index(&mut self.rng, "cert_fr", 52);
                    let (suit, rank) = crate::items::card_front_from_index(idx);
                    self.playing_card_counter += 1;
                    let sort_id = self.next_sort_id();
                    let mut card = Card::new(rank, suit, sort_id);
                    let seal_type = self.rng.random("certsl");
                    card.seal = if seal_type > 0.75 {
                        Seal::Red
                    } else if seal_type > 0.5 {
                        Seal::Blue
                    } else if seal_type > 0.25 {
                        Seal::Gold
                    } else {
                        Seal::Purple
                    };
                    let pareidolia = self.joker_owned(JokerId::Pareidolia);
                    let smeared = self.joker_owned(JokerId::Smeared);
                    if let Some(b) = self.active_blind.as_ref() {
                        card.debuff = b.debuff_card(&card, pareidolia, smeared);
                    }
                    self.hand.push(card);
                    self.sort_hand_desc();
                }
                OutEvent::GiftCard => {
                    // card.lua:2993-3009 — +extra (1) sell value on every
                    // joker and consumable.
                    for j in self.jokers.iter_mut() {
                        j.extra_value += 1;
                    }
                    for c in self.consumables.iter_mut() {
                        c.extra_value += 1;
                    }
                }
                OutEvent::TurtleBeanShrink => {
                    // card.lua:2926-2927 — G.hand:change_size(-1).
                    self.hand_size -= 1;
                }
            }
        }
    }

    /// `G.jokers:remove_card` + `Card:remove` for a self-destructing or
    /// tarot-destroyed joker: passives off (remove_from_deck,
    /// card.lua:645-701), negative slot returned, used_jokers cleared. No
    /// money (that is sell_card's job).
    pub(crate) fn remove_joker_at(&mut self, idx: usize) {
        let j = self.jokers.remove(idx);
        if !j.debuffed {
            self.joker_passives_off(&j);
        }
        if j.edition == crate::cards::Edition::Negative {
            self.joker_slots -= 1;
        }
        self.remove_used(j.id.key());
    }

    /// The passive `Card:add_to_deck` effects (card.lua:564-644): the
    /// generic h_size/d_size handling plus every named branch.
    pub(crate) fn joker_passives_on(&mut self, j: &OwnedJoker) {
        match j.id {
            // config h_size = 1 (game.lua:487).
            JokerId::Juggler => self.hand_size += 1,
            // config d_size = 1 (game.lua:488); card.lua:588-591.
            JokerId::Drunkard => {
                self.round_resets_discards += 1;
                self.discards_left += 1;
            }
            // j_merry_andy config = {d_size = 3, h_size = -1} (game.lua:513):
            // both fall out of the generic handling (card.lua:586-592).
            JokerId::MerryAndy => {
                self.hand_size -= 1;
                self.round_resets_discards += 3;
                self.discards_left += 3;
            }
            // card.lua:593-595, extra = 20.
            JokerId::CreditCard => self.bankrupt_at -= 20,
            // card.lua:596-600 — Chicot disables a live boss on entry (also
            // fires when Crimson Heart un-debuffs it mid-round).
            JokerId::Chicot => {
                let live_boss = self
                    .active_blind
                    .as_ref()
                    .is_some_and(|b| b.proto.is_boss && !b.disabled);
                if live_boss {
                    self.disable_boss();
                }
            }
            // card.lua:602-605.
            JokerId::Chaos => {
                self.free_rerolls += 1;
                self.calculate_reroll_cost(true);
            }
            // card.lua:605-607 — the CURRENT (decaying) bean size.
            JokerId::TurtleBean => self.hand_size += j.state.extra as i64,
            // card.lua:608-612 — every probability doubles.
            JokerId::Oops => self.prob_normal *= 2.0,
            // card.lua:613-615 — interest_amount += extra (1).
            JokerId::ToTheMoon => self.interest_amount += 1,
            // card.lua:623-626 — Troubadour: +2 hand size, -1 hand per round.
            JokerId::Troubadour => {
                self.hand_size += 2;
                self.round_resets_hands -= 1;
            }
            // card.lua:627-629 — Stuntman: -2 hand size.
            JokerId::Stuntman => self.hand_size -= 2,
            // Astronomer's set_cost refresh (card.lua:616-622) is implicit —
            // costs are computed live.
            _ => {}
        }
    }

    /// `Card:remove_from_deck` (card.lua:645-701), inverse of the above
    /// (minus Chicot, whose disable is one-way).
    pub(crate) fn joker_passives_off(&mut self, j: &OwnedJoker) {
        match j.id {
            JokerId::Juggler => self.hand_size -= 1,
            JokerId::Drunkard => {
                self.round_resets_discards -= 1;
                self.discards_left -= 1;
            }
            JokerId::MerryAndy => {
                self.hand_size += 1;
                self.round_resets_discards -= 3;
                self.discards_left -= 3;
            }
            JokerId::CreditCard => self.bankrupt_at += 20,
            JokerId::Chaos => {
                self.free_rerolls -= 1;
                self.calculate_reroll_cost(true);
            }
            JokerId::TurtleBean => self.hand_size -= j.state.extra as i64,
            JokerId::Oops => self.prob_normal /= 2.0,
            JokerId::ToTheMoon => self.interest_amount -= 1,
            JokerId::Troubadour => {
                self.hand_size -= 2;
                self.round_resets_hands += 1;
            }
            JokerId::Stuntman => self.hand_size += 2,
            _ => {}
        }
    }

    /// `Card:set_debuff` for a joker (card.lua:526-538): flipping the flag
    /// toggles the passive add_to_deck effects (`from_debuff = true` leaves
    /// negative slots alone via queue_negative_removal).
    pub(crate) fn set_joker_debuffed(&mut self, idx: usize, debuff: bool) {
        if self.jokers[idx].debuffed == debuff {
            return;
        }
        self.jokers[idx].debuffed = debuff;
        let j = self.jokers[idx];
        if debuff {
            self.joker_passives_off(&j);
        } else {
            self.joker_passives_on(&j);
        }
    }

    /// All owned playing cards (`G.playing_cards`: deck + hand + discard;
    /// destroyed cards are gone), non-Stone, as (sort_id, rank, suit) —
    /// pseudorandom_element sorts Card tables by sort_id
    /// (misc_functions.lua:260-262).
    fn valid_reset_cards(&self) -> Vec<(u32, u8, Suit)> {
        let mut v: Vec<(u32, u8, Suit)> = self
            .deck
            .iter()
            .chain(self.hand.iter())
            .chain(self.discard_pile.iter())
            .filter(|c| c.enhancement != Enhancement::Stone)
            .map(|c| (c.sort_id, c.rank.id(), c.suit))
            .collect();
        v.sort_unstable_by_key(|&(sid, _, _)| sid);
        v
    }

    /// `reset_idol_card` (common_events.lua:2271-2286): 'idol<ante>' pick
    /// over the non-Stone playing cards; Ace of Spades fallback.
    pub(crate) fn reset_idol_card(&mut self) {
        self.idol_card = (14, Suit::Spades);
        let cards = self.valid_reset_cards();
        if !cards.is_empty() {
            let key = format!("idol{}", self.ante);
            let (_, rank, suit) = cards[element_index(&mut self.rng, &key, cards.len())];
            self.idol_card = (rank, suit);
        }
    }

    /// `reset_mail_rank` (common_events.lua:2288-2301): 'mail<ante>' pick;
    /// Ace fallback. Mail-In Rebate matches `get_id()` against this.
    pub(crate) fn reset_mail_rank(&mut self) {
        self.mail_card_id = 14;
        let cards = self.valid_reset_cards();
        if !cards.is_empty() {
            let key = format!("mail{}", self.ante);
            let (_, rank, _) = cards[element_index(&mut self.rng, &key, cards.len())];
            self.mail_card_id = rank;
        }
    }

    /// `reset_ancient_card` (common_events.lua:2303-2310): 'anc<ante>' over
    /// {Spades, Hearts, Clubs, Diamonds} minus the current suit.
    pub(crate) fn reset_ancient_card(&mut self) {
        const ORDER: [Suit; 4] = [Suit::Spades, Suit::Hearts, Suit::Clubs, Suit::Diamonds];
        let pool: Vec<Suit> = ORDER
            .iter()
            .copied()
            .filter(|&s| Some(s) != self.ancient_suit)
            .collect();
        let key = format!("anc{}", self.ante);
        self.ancient_suit = Some(pool[element_index(&mut self.rng, &key, pool.len())]);
    }

    /// `reset_castle_card` (common_events.lua:2312-2325): 'cas<ante>' pick
    /// over the non-Stone playing cards; Spades fallback.
    pub(crate) fn reset_castle_card(&mut self) {
        self.castle_suit = Suit::Spades;
        let cards = self.valid_reset_cards();
        if !cards.is_empty() {
            let key = format!("cas{}", self.ante);
            let (_, _, suit) = cards[element_index(&mut self.rng, &key, cards.len())];
            self.castle_suit = suit;
        }
    }

    /// `CardArea:sort('desc')` (cardarea.lua:579-580): descending
    /// `get_nominal()`; equal nominals (same rank+suit) break by creation
    /// order via the unique_val term — earlier card first.
    pub(crate) fn sort_hand_desc(&mut self) {
        self.hand.sort_by(|a, b| {
            b.get_nominal(false)
                .partial_cmp(&a.get_nominal(false))
                .unwrap()
                .then(a.sort_id.cmp(&b.sort_id))
        });
    }

    fn validate_selection(&self, indices: &[usize]) -> Result<Vec<usize>, RunError> {
        if indices.is_empty() || indices.len() > HIGHLIGHT_LIMIT {
            return Err(RunError::BadCardSelection(format!(
                "must select 1..=5 cards, got {}",
                indices.len()
            )));
        }
        let mut sel = indices.to_vec();
        sel.sort_unstable();
        sel.dedup();
        if sel.len() != indices.len() {
            return Err(RunError::BadCardSelection("duplicate indices".into()));
        }
        if let Some(&max) = sel.last() {
            if max >= self.hand.len() {
                return Err(RunError::BadCardSelection(format!(
                    "index {max} out of range (hand size {})",
                    self.hand.len()
                )));
            }
        }
        // Cerulean Bell: the forced card cannot be deselected, so every
        // play/discard must include it (blind.lua:574-587).
        if let Some(sid) = self.forced_card {
            if let Some(fi) = self.hand.iter().position(|c| c.sort_id == sid) {
                if !sel.contains(&fi) {
                    return Err(RunError::BadCardSelection(format!(
                        "forced card (hand index {fi}) must be selected"
                    )));
                }
            }
        }
        Ok(sel)
    }

    /// Remove the selected cards from the hand, returning them in ascending
    /// hand order (== the play order the game derives from screen x).
    fn remove_from_hand(&mut self, sorted_sel: &[usize]) -> Vec<Card> {
        let cards: Vec<Card> = sorted_sel.iter().map(|&i| self.hand[i]).collect();
        for &i in sorted_sel.iter().rev() {
            self.hand.remove(i);
        }
        cards
    }

    // ------------------------------------------------------------------
    // State getters
    // ------------------------------------------------------------------

    pub fn state(&self) -> State {
        self.state
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        let mut v = Vec::new();
        match self.state {
            State::BlindSelect => {
                v.push(Action::SelectBlind);
                if self.blind_on_deck != BlindStage::Boss {
                    v.push(Action::SkipBlind);
                }
                if self.can_reroll_boss() {
                    v.push(Action::RerollBoss);
                }
                self.push_inventory_actions(&mut v);
            }
            State::SelectingHand => {
                v.push(Action::Play);
                if self.discards_left > 0 {
                    v.push(Action::Discard);
                }
                self.push_inventory_actions(&mut v);
            }
            State::RoundEval => {
                v.push(Action::CashOut);
                self.push_inventory_actions(&mut v);
            }
            State::Shop => {
                v.push(Action::LeaveShop);
                if let Some(shop) = &self.shop {
                    for (i, item) in shop.jokers.iter().enumerate() {
                        if self.can_buy_shop_item(item) {
                            v.push(Action::BuyShopItem(i));
                            if item.is_consumable() && self.affordable(self.shop_item_cost(item)) {
                                v.push(Action::BuyAndUseShopItem(i));
                            }
                        }
                    }
                    for (i, vo) in shop.vouchers.iter().enumerate() {
                        if self.affordable(self.voucher_cost(vo.key)) {
                            v.push(Action::RedeemVoucher(i));
                        }
                    }
                    for (i, p) in shop.packs.iter().enumerate() {
                        if !p.used && self.pack_affordable(p) {
                            v.push(Action::BuyPack(i));
                        }
                    }
                }
                if self.can_reroll_shop() {
                    v.push(Action::Reroll);
                }
                self.push_inventory_actions(&mut v);
            }
            State::PackOpen => {
                if let Some(pack) = &self.pack {
                    for i in 0..pack.items.len() {
                        if self.can_pick_pack_item(i) {
                            v.push(Action::PickPackItem(i));
                        }
                    }
                }
                v.push(Action::SkipPack);
                self.push_inventory_actions(&mut v);
            }
            State::GameOver | State::Won => {}
        }
        v
    }

    /// Sell/use actions available in every interactive state
    /// (`can_sell_card`/`can_use_consumeable` allow SELECTING_HAND, SHOP,
    /// BLIND_SELECT, ROUND_EVAL and the pack states).
    fn push_inventory_actions(&self, v: &mut Vec<Action>) {
        for i in 0..self.jokers.len() {
            v.push(Action::SellJoker(i));
        }
        for (i, c) in self.consumables.iter().enumerate() {
            v.push(Action::SellConsumable(i));
            if self.consumable_has_any_use(c.key) {
                v.push(Action::UseConsumable(i));
            }
        }
    }

    /// Current hand, sorted the way the game displays it ('desc').
    /// `face_down` cards should have their identity masked in observations.
    pub fn hand(&self) -> &[Card] {
        &self.hand
    }

    pub fn deck_len(&self) -> usize {
        self.deck.len()
    }

    /// The undrawn draw pile, bottom-to-top (drawing pops from the END).
    /// P6 observation encoding reads card *counts* from this; peeking at the
    /// order is the caller's responsibility to avoid.
    pub fn deck_cards(&self) -> &[Card] {
        &self.deck
    }

    /// The discard pile (cards return to the deck at round end).
    pub fn discard_cards(&self) -> &[Card] {
        &self.discard_pile
    }

    pub fn discard_pile_len(&self) -> usize {
        self.discard_pile.len()
    }

    /// Public alias of the live joker-driven hand-evaluation modifiers
    /// (Four Fingers / Shortcut / Smeared) — P6 heuristic bot.
    pub fn eval_mods(&self) -> EvalMods {
        self.live_eval_mods()
    }

    /// P6 determinized-search support: re-randomize everything the agent
    /// has not yet observed, leaving owned state untouched. Concretely:
    /// * the entire keyed RNG stream state is replaced with a fresh
    ///   `RngState::new(seed)` — every FUTURE pseudoseed draw (shop rolls,
    ///   pack contents, boss rolls, lucky/glass procs, shuffles, ...) now
    ///   comes from the new seed's streams;
    /// * the undrawn draw pile is reshuffled once on the new seed's
    ///   'redet' stream (pseudoshuffle pre-sorts by sort_id, so the result
    ///   is independent of the previous hidden order).
    ///
    /// Hand, discard pile, jokers, consumables, shop contents already on
    /// display, money, blinds and all other observable state are unchanged.
    pub fn redeterminize(&mut self, seed: &str) {
        self.rng = RngState::new(seed);
        let s = self.rng.pseudoseed("redet");
        pseudoshuffle(&mut self.deck, s);
    }

    /// Cards permanently destroyed this run (shattered Glass etc.).
    pub fn destroyed_cards(&self) -> &[Card] {
        &self.destroyed_cards
    }

    pub fn ante(&self) -> i64 {
        self.ante
    }

    pub fn round(&self) -> i64 {
        self.round
    }

    pub fn dollars(&self) -> i64 {
        self.dollars
    }

    /// `G.GAME.chips` scored against the current blind.
    pub fn chips(&self) -> f64 {
        self.chips
    }

    pub fn hands_left(&self) -> i64 {
        self.hands_left
    }

    pub fn discards_left(&self) -> i64 {
        self.discards_left
    }

    /// Current hand-size limit (`G.hand.config.card_limit`).
    pub fn hand_size(&self) -> i64 {
        self.hand_size
    }

    pub fn blind_on_deck(&self) -> BlindStage {
        self.blind_on_deck
    }

    /// The blind being fought right now (None between rounds).
    pub fn current_blind(&self) -> Option<&'static BlindProto> {
        self.active_blind.as_ref().map(|b| b.proto)
    }

    /// Live blind state (Eye/Mouth/Fish bookkeeping, disabled flag...).
    pub fn active_blind(&self) -> Option<&ActiveBlind> {
        self.active_blind.as_ref()
    }

    /// Chip requirement of the current blind (0 between rounds).
    pub fn blind_chips(&self) -> f64 {
        self.active_blind.as_ref().map(|b| b.chips).unwrap_or(0.0)
    }

    /// The boss key selected for this ante (e.g. "bl_hook").
    pub fn boss_choice(&self) -> &'static str {
        self.boss_choice
    }

    /// Hand index of Cerulean Bell's forced card, if any.
    pub fn forced_card_index(&self) -> Option<usize> {
        self.forced_card
            .and_then(|sid| self.hand.iter().position(|c| c.sort_id == sid))
    }

    /// `G.GAME.current_round.most_played_poker_hand` (The Ox's target).
    pub fn most_played_hand(&self) -> HandType {
        self.most_played_hand
    }

    pub fn last_hand_played(&self) -> Option<HandType> {
        self.last_hand_played
    }

    /// Held consumable center keys, in slot order. (P3a compat name: seals
    /// used to buffer their creations here; they now land in `consumables`.)
    pub fn pending_consumables(&self) -> Vec<String> {
        self.consumables.iter().map(|c| c.key.to_string()).collect()
    }

    /// Held consumables.
    pub fn consumables(&self) -> &[OwnedConsumable] {
        &self.consumables
    }

    /// Held (inert until P3c) jokers, in area order.
    pub fn jokers(&self) -> &[OwnedJoker] {
        &self.jokers
    }

    pub fn joker_slots(&self) -> usize {
        self.joker_slots
    }

    pub fn consumable_slots(&self) -> usize {
        self.consumable_slots
    }

    /// The live shop (Some in State::Shop / State::PackOpen).
    pub fn shop(&self) -> Option<&ShopState> {
        self.shop.as_ref()
    }

    /// The open booster pack (Some in State::PackOpen).
    pub fn pack(&self) -> Option<&PackState> {
        self.pack.as_ref()
    }

    /// Current shop reroll cost (`G.GAME.current_round.reroll_cost`).
    pub fn reroll_cost(&self) -> i64 {
        self.reroll_cost
    }

    /// Vouchers redeemed this run.
    pub fn used_vouchers(&self) -> &HashSet<&'static str> {
        &self.used_vouchers
    }

    /// Held (untriggered) tags in acquisition order.
    pub fn tags(&self) -> &[RunTag] {
        &self.tags
    }

    /// The tags offered for skipping the (Small, Big) blinds this ante.
    pub fn blind_tags(&self) -> (&'static str, &'static str) {
        (self.blind_tag_small, self.blind_tag_big)
    }

    /// `G.GAME.current_round.voucher` — the ante's shop voucher, if unredeemed.
    pub fn current_voucher(&self) -> Option<&'static str> {
        self.current_voucher
    }

    /// `G.GAME.round_resets.blind_ante` — the ante blind sizes scale from.
    pub fn blind_ante(&self) -> i64 {
        self.blind_ante
    }

    pub fn skips(&self) -> i64 {
        self.skips
    }

    /// Dollars awaiting collection on the RoundEval screen.
    pub fn pending_cashout(&self) -> i64 {
        self.pending_cashout
    }

    pub fn last_play(&self) -> Option<&PlayResult> {
        self.last_play.as_ref()
    }

    pub fn hands_table(&self) -> &HandsTable {
        &self.hands_table
    }

    /// `G.GAME.bankrupt_at` (Credit Card lowers it to -20 while owned).
    pub fn bankrupt_at(&self) -> i64 {
        self.bankrupt_at
    }

    /// `G.GAME.consumeable_usage_total.tarot` (Fortune Teller's mult).
    pub fn tarots_used(&self) -> i64 {
        self.tarots_used
    }

    /// `G.GAME.current_round.mail_card.id` — the rank id Mail-In Rebate
    /// pays for this round.
    pub fn mail_rank_id(&self) -> u8 {
        self.mail_card_id
    }

    /// `G.GAME.current_round.idol_card` (rank id, suit) — The Idol (P3c-2).
    pub fn idol_card(&self) -> (u8, Suit) {
        self.idol_card
    }

    /// `G.GAME.current_round.ancient_card.suit` — Ancient Joker (P3c-2).
    pub fn ancient_suit(&self) -> Option<Suit> {
        self.ancient_suit
    }

    /// `G.GAME.current_round.castle_card.suit` — Castle (P3c-2).
    pub fn castle_suit(&self) -> Suit {
        self.castle_suit
    }

    pub fn hands_played_this_round(&self) -> i64 {
        self.hands_played_round
    }

    /// `G.GAME.current_round.discards_used`.
    pub fn discards_used_this_round(&self) -> i64 {
        self.discards_used_round
    }

    /// `G.GAME.round_resets.blind_states.Small == 'Defeated'` equivalents —
    /// P5 cross-validation exports the game's blind-status trio from these.
    pub fn small_defeated(&self) -> bool {
        self.small_defeated
    }

    pub fn big_defeated(&self) -> bool {
        self.big_defeated
    }

    pub fn small_skipped(&self) -> bool {
        self.small_skipped
    }

    pub fn big_skipped(&self) -> bool {
        self.big_skipped
    }

    pub fn won(&self) -> bool {
        self.won
    }

    pub fn seed(&self) -> &str {
        self.rng.seed()
    }
}
