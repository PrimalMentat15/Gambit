//! The joker effect engine: per-joker mutable state, the `JokerHooks`
//! dispatcher over the joker area, and Blueprint/Brainstorm copy resolution.
//!
//! Spec: `Card:calculate_joker` (card.lua:2291-4056) — every per-joker
//! branch ported in effects.rs with file:line citations — plus the
//! joker-area phase of `evaluate_play` (state_events.lua:877-944), which the
//! dispatcher's `joker_main` reproduces:
//!
//! # Joker-area order of operations (state_events.lua:877-944)
//! For each card in `G.jokers.cards` followed by `G.consumeables.cards`, in
//! area order:
//!  1. edition pre-effects (`eval_card{edition = true}` -> `Card:get_edition`,
//!     card.lua:1016-1031): Foil `chip_mod` +50 is applied first (:884-891),
//!     then Holo `mult_mod` +10 (:892-899). Polychrome's `x_mult_mod` is
//!     captured here but held until step 4.
//!  2. the joker's own `joker_main` effect (:905-916), field order within
//!     the one returned table: `mult_mod` added, `chip_mod` added,
//!     `Xmult_mod` multiplied (:909-911). Held Planet cards participate via
//!     the Observatory branch (card.lua:2293-2302).
//!  3. joker-on-joker effects (:920-931): every joker gets
//!     `{other_joker = _card}` (Baseball Card); same field order.
//!  4. edition post-effect (:934-943): Polychrome `x_mult_mod` (x1.5)
//!     multiplied after steps 2-3.
//!
//! A debuffed joker (Crimson Heart) contributes nothing anywhere: both
//! `Card:calculate_joker` (card.lua:2292) and `Card:get_edition`
//! (card.lua:1017) return nil under `self.debuff`. Flipped jokers (Amber
//! Acorn) still work — no facing check exists in the Lua.
//!
//! # Copy resolution (Blueprint card.lua:2304-2320, Brainstorm :2321-2334)
//! Blueprint re-dispatches the context to the joker on its right, Brainstorm
//! to the leftmost joker; chains recurse with a shared `context.blueprint`
//! depth counter that aborts past `#G.jokers.cards + 1`, and Brainstorm in
//! slot 1 (`other_joker == self`) falls through to nothing. The resolved
//! target's branches then run with `context.blueprint` set — every branch
//! guarded by `not context.blueprint` in the Lua is skipped for copies
//! (state-scaling and self-destruction stay on the copied joker only).
//! `JokerMeta::blueprint_compat` is UI metadata; the game never gates on it.

pub mod effects;

use std::collections::HashSet;

use crate::cards::{Card, Edition, Enhancement, HandType, Suit};
use crate::items::{sell_cost, ConsumableSet, JokerId};
use crate::rng::{LuaRandom, RngState};
use crate::scoring::{BeforeCtx, CardEffect, HandCtx, HandsTable, JokerHooks};
use crate::shop::OwnedJoker;

/// Per-joker mutable ability state — the mutable subset of `self.ability`
/// that the base game's jokers touch (card.lua:280-345 initialises it from
/// `center.config`). Plain `Copy` data: this is RL-observation and snapshot
/// surface.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JokerState {
    /// `ability.mult` (Ride the Bus/Green Joker/Red Card/Popcorn/Ceremonial
    /// Dagger/Flash Card/Spare Trousers accumulators; `center.config.mult`
    /// start).
    pub mult: f64,
    /// `ability.extra.chips` accumulators (Runner/Square/Ice Cream/Wee
    /// Joker/Castle).
    pub chips: f64,
    /// `ability.x_mult` (`center.config.Xmult or 1`, card.lua:288) —
    /// Vampire/Obelisk/Madness/Hologram/Campfire/Glass/Lucky Cat/
    /// Constellation/Hit the Road/Yorick/Ramen scalers plus the typed
    /// Duo..Tribe constants, all consumed by the generic `x_mult > 1`
    /// joker_main branch (card.lua:3653-3659).
    pub x_mult: f64,
    /// `ability.extra` when it is a mutable scalar, or the named mutable
    /// counter of the joker: Seltzer hands left, Turtle Bean `extra.h_size`,
    /// Rocket `extra.dollars`, Invisible `invis_rounds`, Yorick
    /// `yorick_discards`, Caino `caino_xmult`.
    pub extra: f64,
    /// `ability.to_do_poker_hand` (To Do List) — rolled on the 'to_do'
    /// stream at Card creation (card.lua:311-322) and re-rolled each round
    /// end (card.lua:2975-2985).
    pub to_do_hand: HandType,
}

impl JokerState {
    /// `Card:set_ability` defaults from `center.config` (game.lua:368-531):
    /// `x_mult = config.Xmult or 1` plus the named counter initialisers
    /// (card.lua:308-333).
    pub fn initial(id: JokerId) -> Self {
        let mut s = JokerState {
            mult: 0.0,
            chips: 0.0,
            x_mult: 1.0,
            extra: 0.0,
            to_do_hand: HandType::HighCard,
        };
        match id {
            // j_popcorn config = {mult = 20, extra = 4} (game.lua:477).
            JokerId::Popcorn => s.mult = 20.0,
            // j_ice_cream extra = {chips = 100, chip_mod = 5} (game.lua:430).
            JokerId::IceCream => s.chips = 100.0,
            // j_selzer extra = 10 hands (game.lua:481).
            JokerId::Selzer => s.extra = 10.0,
            // config.Xmult starts (game.lua): j_ramen Xmult = 2,
            // j_duo/trio/family/order/tribe = 2/3/4/3/2.
            JokerId::Ramen | JokerId::Duo | JokerId::Tribe => s.x_mult = 2.0,
            JokerId::Trio | JokerId::Order => s.x_mult = 3.0,
            JokerId::Family => s.x_mult = 4.0,
            // j_turtle_bean extra = {h_size = 5, h_mod = 1} (game.lua:490).
            JokerId::TurtleBean => s.extra = 5.0,
            // j_rocket extra = {dollars = 1, increase = 2} (game.lua:492).
            JokerId::Rocket => s.extra = 1.0,
            // ability.yorick_discards = extra.discards = 23 (card.lua:327-329).
            JokerId::Yorick => s.extra = 23.0,
            // ability.caino_xmult = 1 (card.lua:324-326).
            JokerId::Caino => s.extra = 1.0,
            // ability.invis_rounds = 0 (card.lua:308-310).
            JokerId::Invisible => s.extra = 0.0,
            _ => {}
        }
        s
    }
}

/// Run-state snapshot the dispatcher reads during a flow. Rebuilt by
/// `Run::joker_env()` right before each hook window; everything here is
/// stable for the duration of that window (deck/discard counts do not move
/// during `evaluate_play`).
#[derive(Debug, Clone)]
pub struct JokerEnv {
    /// `G.GAME.round_resets.ante` (per-ante stream suffixes).
    pub ante: i64,
    /// `G.GAME.probabilities.normal` (game.lua:1890; Oops! All 6s doubles).
    pub prob_normal: f64,
    /// `#G.deck.cards` (Blue Joker).
    pub deck_len: usize,
    /// `G.GAME.current_round.discards_left` (Banner/Mystic Summit/Delayed
    /// Gratification).
    pub discards_left: i64,
    /// `G.GAME.current_round.discards_used` (Delayed Gratification/Burnt/
    /// Trading Card).
    pub discards_used_round: i64,
    /// `G.GAME.current_round.hands_left` (Dusk/Acrobat).
    pub hands_left: i64,
    /// `G.GAME.current_round.hands_played` (DNA/Sixth Sense first-hand
    /// gates) — NOT yet counting the hand being evaluated
    /// (state_events.lua:524 runs after evaluate_play).
    pub hands_played_round: i64,
    /// `G.GAME.hands_played` run total (Loyalty Card) — same pre-increment
    /// timing (state_events.lua:523).
    pub hands_played_total: i64,
    /// `G.GAME.dollars` — unchanged for the whole evaluate_play window (all
    /// payouts are queued events); Bull/Bootstraps add `dollar_buffer`,
    /// Vagabond reads it raw (card.lua:3744).
    pub dollars: i64,
    /// `G.GAME.skips` (Throwback's Card:update, card.lua:4176-4178).
    pub skips: i64,
    /// `#G.consumeables.cards` / `.config.card_limit` — consumable-creation
    /// gates (8 Ball/Superposition; card.lua:3105, 3768).
    pub consumables_len: usize,
    pub consumable_slots: usize,
    /// `G.jokers.config.card_limit` (Riff-raff/Stencil/Invisible gates).
    pub joker_slots: usize,
    /// `G.GAME.consumeable_usage_total.tarot` (Fortune Teller,
    /// misc_functions.lua:1196-1206).
    pub tarots_used: i64,
    /// Distinct Planet keys in `G.GAME.consumeable_usage` (Satellite,
    /// card.lua:1667-1673).
    pub planets_used: i64,
    /// `G.GAME.current_round.mail_card.id` (Mail-In Rebate).
    pub mail_card_id: u8,
    /// `G.GAME.current_round.idol_card` (The Idol).
    pub idol_card: (u8, Suit),
    /// `G.GAME.current_round.ancient_card.suit` (Ancient Joker).
    pub ancient_suit: Option<Suit>,
    /// `G.GAME.current_round.castle_card.suit` (Castle).
    pub castle_suit: Suit,
    /// `G.GAME.discount_percent` — Swashbuckler reads live sell costs.
    pub discount_percent: i64,
    /// `G.GAME.used_vouchers.v_observatory` — held Planets x1.5 the matching
    /// hand in the joker-area phase (card.lua:2293-2302).
    pub observatory: bool,
    /// Held consumable keys in slot order (the `G.consumeables.cards` tail
    /// of the joker-area loop).
    pub consumable_keys: Vec<&'static str>,
    /// `#G.playing_cards` (all owned playing cards: deck + hand + discard +
    /// play) — Erosion (card.lua:3894-3900).
    pub playing_cards_len: usize,
    /// `G.GAME.starting_deck_size` (52 for the Red Deck, game.lua:2375).
    pub starting_deck_size: usize,
    /// Card:update tallies over `G.playing_cards` (card.lua:4179-4202):
    /// Steel Joker / Stone Joker / Driver's License (non-c_base centers) /
    /// Cloud 9 (rank id 9, Stone cards excluded via get_id).
    pub steel_tally: i64,
    pub stone_tally: i64,
    pub driver_tally: i64,
    pub nine_tally: i64,
    /// `G.GAME.blind.boss` — the proto flag, live even for disabled bosses
    /// (Campfire/Rocket/Chicot/Madness read it).
    pub blind_is_boss: bool,
    /// `G.GAME.blind.chips` / `G.GAME.chips` (Mr. Bones' 25% check).
    pub blind_chips: f64,
    pub game_chips: f64,
    /// `G.sort_id` at window construction — in-window Card creations (DNA)
    /// take `sort_id_base + n`; the Run advances its counter by
    /// `JokerOutbox::sort_ids_used` afterwards.
    pub sort_id_base: u32,
}

impl Default for JokerEnv {
    fn default() -> Self {
        JokerEnv {
            ante: 0,
            prob_normal: 1.0,
            deck_len: 0,
            discards_left: 0,
            discards_used_round: 0,
            hands_left: 0,
            hands_played_round: 0,
            hands_played_total: 0,
            dollars: 0,
            skips: 0,
            consumables_len: 0,
            consumable_slots: 0,
            joker_slots: 0,
            tarots_used: 0,
            planets_used: 0,
            mail_card_id: 14,
            idol_card: (14, Suit::Spades),
            ancient_suit: None,
            castle_suit: Suit::Spades,
            discount_percent: 0,
            observatory: false,
            consumable_keys: Vec::new(),
            playing_cards_len: 0,
            starting_deck_size: 52,
            steel_tally: 0,
            stone_tally: 0,
            driver_tally: 0,
            nine_tally: 0,
            blind_is_boss: false,
            blind_chips: 0.0,
            game_chips: 0.0,
            sort_id_base: 100_000,
        }
    }
}

/// Deferred side effects the hooks buffer for the `Run` to apply after the
/// flow — the `G.E_MANAGER` events the Lua queues mid-evaluation. FIFO order
/// is preserved (queued events execute in order once `evaluate_play`'s
/// synchronous body finishes).
#[derive(Debug, Clone, PartialEq)]
pub enum OutEvent {
    /// `create_card(set, G.consumeables, ..., append)` +
    /// `G.consumeables:emplace` — consumes the '<Set><append><ante>' pool
    /// stream at execution time.
    CreateConsumable(ConsumableSet, &'static str),
    /// `G.jokers:remove_card(self); self:remove()` — self-destructing food
    /// jokers and Madness/Ceremonial victims, identified by sort_id.
    DestroyJoker(u32),
    /// `G.GAME.pool_flags.gros_michel_extinct = true` (card.lua:3037).
    SetGrosMichelExtinct,
    /// Riff-raff's queued `create_card('Joker', …, 0, …, 'rif')` batch
    /// (card.lua:2529-2544): `count` common jokers, count fixed at queue
    /// time.
    CreateJokers(usize),
    /// Chicot's setting_blind `G.GAME.blind:disable()` (card.lua:2492-2501).
    DisableBoss,
    /// Burglar (card.lua:2522-2528): `ease_discard(-discards_left)` +
    /// `ease_hands_played(3)` at event time.
    Burglar,
    /// Marble Joker (card.lua:2580-2601): a Stone card ('marb_fr' front)
    /// materialized into the deck.
    CreateMarbleStone,
    /// Certificate (card.lua:2463-2482): a random playing card with a random
    /// seal ('cert_fr'/'certsl') into the hand.
    CreateCertificate,
    /// Gift Card (card.lua:2993-3009): +1 `extra_value` on every joker and
    /// consumable.
    GiftCard,
    /// Turtle Bean's per-round `G.hand:change_size(-1)` (card.lua:2926-2927)
    /// — the state decrement already happened on the joker.
    TurtleBeanShrink,
}

#[derive(Debug, Default)]
pub struct JokerOutbox {
    pub events: Vec<OutEvent>,
    /// `G.GAME.dollar_buffer` — inline `ease_dollars` payouts queued this
    /// window (Golden Ticket/Business/Reserved Parking/Matador/To Do List/
    /// Rough Gem); Bull and Bootstraps read `dollars + dollar_buffer`
    /// (card.lua:3936-3942, 4046-4050). Reset by queued events after the
    /// window, i.e. per-outbox.
    pub dollar_buffer: i64,
    /// `G.GAME.joker_buffer` — Riff-raff's queued creations count against
    /// the joker slots; Ceremonial Dagger decrements it (card.lua:2569).
    pub joker_buffer: i64,
    /// Card creations performed inside the window (DNA): the Run advances
    /// `G.sort_id` / `G.playing_card` by these after the window.
    pub sort_ids_used: u32,
    pub playing_cards_delta: u32,
}

impl JokerOutbox {
    /// `G.GAME.consumeable_buffer` — creations already queued this window
    /// count against the consumable slots (card.lua:3105 etc.).
    pub fn consumable_buffer(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, OutEvent::CreateConsumable(..)))
            .count()
    }

    fn create_consumable(&mut self, set: ConsumableSet, append: &'static str) {
        self.events.push(OutEvent::CreateConsumable(set, append));
    }

    fn destroy_joker(&mut self, sort_id: u32) {
        self.events.push(OutEvent::DestroyJoker(sort_id));
    }
}

/// Blueprint/Brainstorm chain resolution (card.lua:2304-2334). Starting from
/// the joker in slot `start`, follows copies until a non-copying joker is
/// reached. Returns the effective slot index plus whether the context is a
/// copy (`context.blueprint` set — gates every `not context.blueprint`
/// branch). `None` when the chain dead-ends: the root or any link is
/// debuffed (card.lua:2292), Blueprint has no right neighbour, Brainstorm is
/// leftmost (`other_joker == self`), or the depth counter exceeds
/// `#G.jokers.cards + 1` (card.lua:2312/2326).
pub fn resolve_effective(jokers: &[OwnedJoker], start: usize) -> Option<(usize, bool)> {
    let mut idx = start;
    let mut depth = 0usize; // context.blueprint
    let mut blueprint = false;
    loop {
        let j = jokers.get(idx)?;
        if j.debuffed {
            return None;
        }
        match j.id {
            JokerId::Blueprint => {
                // card.lua:2305-2308 — other_joker = G.jokers.cards[i+1].
                if idx + 1 >= jokers.len() {
                    return None;
                }
                depth += 1;
                // card.lua:2312 — `if context.blueprint > #G.jokers.cards + 1`.
                if depth > jokers.len() + 1 {
                    return None;
                }
                blueprint = true;
                idx += 1;
            }
            JokerId::Brainstorm => {
                // card.lua:2322-2325 — other_joker = G.jokers.cards[1];
                // leftmost Brainstorm fails `other_joker ~= self` and falls
                // through to no other branch.
                if idx == 0 {
                    return None;
                }
                depth += 1;
                if depth > jokers.len() + 1 {
                    return None;
                }
                blueprint = true;
                idx = 0;
            }
            _ => return Some((idx, blueprint)),
        }
    }
}

/// `Card:is_face` (card.lua:964-970): debuffed cards are never faces; any
/// non-debuffed card is one while a (non-debuffed) Pareidolia is owned —
/// including Stone cards, whose random negative `get_id()` merely fails the
/// 11/12/13 test before the Pareidolia `or`.
pub(crate) fn card_is_face(card: &Card, pareidolia: bool) -> bool {
    if card.debuff {
        return false;
    }
    (card.enhancement != Enhancement::Stone && card.rank.is_face()) || pareidolia
}

/// `next(find_joker(name))` (misc_functions.lua:903-918): any non-debuffed
/// joker with the name (consumables can never carry a joker name).
pub(crate) fn find_joker(jokers: &[OwnedJoker], id: JokerId) -> bool {
    jokers.iter().any(|j| j.id == id && !j.debuffed)
}

/// One joker_main effect table: `mult_mod`/`chip_mod`/`Xmult_mod`, applied
/// in exactly that field order (state_events.lua:909-911).
#[derive(Debug, Clone, Copy, Default)]
pub struct MainEffect {
    pub mult_mod: f64,
    pub chip_mod: f64,
    pub x_mult_mod: f64,
}

impl MainEffect {
    fn apply(&self, chips: &mut f64, mult: &mut f64) {
        // state_events.lua:909-911 — 0 is truthy in Lua, so a returned
        // `chip_mod = 0` still "applies"; adding 0 is identity either way.
        *mult += self.mult_mod;
        *chips += self.chip_mod;
        if self.x_mult_mod != 0.0 {
            *mult *= self.x_mult_mod;
        }
    }
}

/// The live joker engine: implements every `JokerHooks` call site over the
/// joker area in order. Constructed per hook window from disjoint `Run`
/// field borrows; side effects that the Lua queues as events land in `out`
/// for `Run::apply_joker_outbox` to execute afterwards.
pub struct EngineHooks<'a> {
    pub jokers: &'a mut Vec<OwnedJoker>,
    pub env: JokerEnv,
    pub out: JokerOutbox,
}

impl<'a> EngineHooks<'a> {
    pub fn new(jokers: &'a mut Vec<OwnedJoker>, env: JokerEnv) -> Self {
        EngineHooks {
            jokers,
            env,
            out: JokerOutbox::default(),
        }
    }

    fn pareidolia(&self) -> bool {
        find_joker(self.jokers, JokerId::Pareidolia)
    }

    fn smeared(&self) -> bool {
        find_joker(self.jokers, JokerId::Smeared)
    }

    /// `ability.mult` of Swashbuckler as maintained by `Card:update`
    /// (card.lua:4240-4247): the summed sell_cost of every OTHER joker in
    /// the area.
    fn swashbuckler_mult(&self, own_idx: usize) -> f64 {
        self.jokers
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != own_idx)
            .map(|(_, j)| {
                sell_cost(
                    j.id.meta().cost,
                    j.edition,
                    self.env.discount_percent,
                    j.extra_value,
                ) as f64
            })
            .sum()
    }

    /// The live `ability.x_mult` the generic joker_main branch reads
    /// (card.lua:3653-3659): Throwback's and Joker Stencil's values are
    /// recomputed every frame by `Card:update` (card.lua:4176-4178,
    /// 4203-4208); everyone else's is the stored accumulator.
    fn live_x_mult(&self, t: usize) -> f64 {
        match self.jokers[t].id {
            // 1 + skips * extra (0.25).
            JokerId::Throwback => 1.0 + self.env.skips as f64 * 0.25,
            // (card_limit - #jokers) + 1 per Joker Stencil in the area.
            JokerId::Stencil => {
                let stencils = self
                    .jokers
                    .iter()
                    .filter(|j| j.id == JokerId::Stencil)
                    .count();
                (self.env.joker_slots as f64 - self.jokers.len() as f64) + stencils as f64
            }
            _ => self.jokers[t].state.x_mult,
        }
    }

    /// `playing_card_joker_effects(cards)` (misc_functions.lua:1580-1584):
    /// every joker gets `{playing_card_added = true, cards = …}` — Hologram
    /// (`not context.blueprint`) gains extra (0.25) per card
    /// (card.lua:2456-2461).
    pub fn playing_cards_added(&mut self, count: usize) {
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            if self.jokers[t].id == JokerId::Hologram && !bp {
                self.jokers[t].state.x_mult += 0.25 * count as f64;
            }
        }
    }
}

impl JokerHooks for EngineHooks<'_> {
    fn splash(&self) -> bool {
        // state_events.lua:582 — next(find_joker('Splash')).
        find_joker(self.jokers, JokerId::Splash)
    }

    fn setting_blind(&mut self, rng: &mut RngState) {
        // state_events.lua:335-337 — `{setting_blind = true}` per joker.
        // The joker branches queue their real work as events
        // (card.lua:2491-2602); `getting_sliced` guards interlock the
        // Madness/Ceremonial destructions within the same window.
        let mut sliced: HashSet<u32> = HashSet::new();
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            // `elseif context.setting_blind and not self.getting_sliced`
            // (card.lua:2491) — a sliced TARGET does nothing at all.
            if sliced.contains(&self.jokers[t].sort_id) {
                continue;
            }
            // `(context.blueprint_card or self)` — the chain root.
            let root_sliced = sliced.contains(&self.jokers[i].sort_id);
            match self.jokers[t].id {
                // card.lua:2492-2501 — Chicot disables the incoming boss.
                // Blueprint-gated.
                JokerId::Chicot if !bp && self.env.blind_is_boss => {
                    self.out.events.push(OutEvent::DisableBoss);
                }
                // card.lua:2503-2521 — Madness (non-boss blinds): +0.5
                // x_mult, then destroys a random other non-eternal joker
                // ('madness' stream; candidates exclude already-sliced).
                JokerId::Madness if !bp && !self.env.blind_is_boss => {
                    self.jokers[t].state.x_mult += 0.5;
                    let me = self.jokers[t].sort_id;
                    let mut cand: Vec<u32> = self
                        .jokers
                        .iter()
                        .filter(|j| j.sort_id != me && !j.eternal && !sliced.contains(&j.sort_id))
                        .map(|j| j.sort_id)
                        .collect();
                    if !cand.is_empty() {
                        cand.sort_unstable();
                        let seed = rng.pseudoseed("madness");
                        let pick = cand[(LuaRandom::seeded(seed).random_range(1, cand.len() as i64)
                            - 1) as usize];
                        if !root_sliced {
                            sliced.insert(pick);
                            self.out.destroy_joker(pick);
                        }
                    }
                }
                // card.lua:2522-2528 — Burglar: 0 discards, +3 hands. NOT
                // blueprint-gated (copies fire too).
                JokerId::Burglar if !root_sliced => {
                    self.out.events.push(OutEvent::Burglar);
                }
                // card.lua:2529-2544 — Riff-raff: up to 2 common jokers
                // ('rif'), gated on space incl. the in-window joker_buffer.
                JokerId::RiffRaff
                    if !root_sliced
                        && (self.jokers.len() as i64 + self.out.joker_buffer)
                            < self.env.joker_slots as i64 =>
                {
                    let count = 2.min(
                        self.env.joker_slots as i64
                            - (self.jokers.len() as i64 + self.out.joker_buffer),
                    ) as usize;
                    self.out.joker_buffer += count as i64;
                    self.out.events.push(OutEvent::CreateJokers(count));
                }
                // card.lua:2545-2560 — Cartomancer: one Tarot ('car'),
                // gated on consumable space incl. the buffer. Not bp-gated.
                JokerId::Cartomancer
                    if !root_sliced
                        && self.env.consumables_len + self.out.consumable_buffer()
                            < self.env.consumable_slots =>
                {
                    self.out.create_consumable(ConsumableSet::Tarot, "car");
                }
                // card.lua:2561-2579 — Ceremonial Dagger: destroy the right
                // neighbour, absorb 2x its sell value. Blueprint-gated.
                JokerId::Ceremonial if !bp => {
                    let my_pos = t;
                    if my_pos + 1 < self.jokers.len() {
                        let victim = self.jokers[my_pos + 1];
                        if !victim.eternal && !sliced.contains(&victim.sort_id) {
                            sliced.insert(victim.sort_id);
                            // card.lua:2569 — joker_buffer decrement (feeds a
                            // later Riff-raff's space gate this window).
                            self.out.joker_buffer -= 1;
                            let sc = sell_cost(
                                victim.id.meta().cost,
                                victim.edition,
                                self.env.discount_percent,
                                victim.extra_value,
                            );
                            self.jokers[t].state.mult += 2.0 * sc as f64;
                            self.out.destroy_joker(victim.sort_id);
                        }
                    }
                }
                // card.lua:2580-2601 — Marble Joker: a Stone card into the
                // deck ('marb_fr' at event time); the
                // playing_card_joker_effects({true}) pass runs synchronously.
                JokerId::Marble if !root_sliced => {
                    self.out.events.push(OutEvent::CreateMarbleStone);
                    self.playing_cards_added(1);
                }
                _ => {}
            }
        }
    }

    fn first_hand_drawn(&mut self) {
        // game.lua:3226-3231 — `{first_hand_drawn = true}` per joker.
        // Certificate (card.lua:2463-2482, not blueprint-gated): queues the
        // card creation ('cert_fr'/'certsl' at event time) and runs
        // playing_card_joker_effects({true}) synchronously. DNA/Trading Card
        // only juice here.
        for i in 0..self.jokers.len() {
            let Some((t, _bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            if self.jokers[t].id == JokerId::Certificate {
                self.out.events.push(OutEvent::CreateCertificate);
                self.playing_cards_added(1);
            }
        }
    }

    fn before_hand(
        &mut self,
        ctx: &mut BeforeCtx,
        hands: &mut HandsTable,
        dollars: &mut i64,
        rng: &mut RngState,
    ) {
        // state_events.lua:628-638 — one eval per joker, in area order;
        // `effects.jokers.level_up` levels the played hand (Space Joker) and
        // `playing_cards_created` triggers the Hologram pass immediately
        // (common_events.lua:922-924).
        for i in 0..self.jokers.len() {
            let pareidolia = self.pareidolia();
            let smeared = self.smeared();
            let Some((t, bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            let res = effects::before_hand(
                &mut self.jokers[t],
                bp,
                ctx,
                pareidolia,
                smeared,
                hands,
                &self.env,
                &mut self.out,
                dollars,
                rng,
            );
            if res.level_up {
                hands.level_up(ctx.hand_type, 1);
            }
            if res.dna_copy {
                // card.lua:3501-3524 — copy_card(full_hand[1]) with a fresh
                // sort_id/playing_card id, emplaced into the hand;
                // G.playing_cards grows.
                let mut copy = ctx.full_hand[0];
                self.out.sort_ids_used += 1;
                copy.sort_id = self.env.sort_id_base + self.out.sort_ids_used;
                self.out.playing_cards_delta += 1;
                ctx.held.push(copy);
                // playing_cards_created -> playing_card_joker_effects({true}).
                self.playing_cards_added(1);
            }
        }
    }

    fn scoring_card_repetitions(&mut self, ctx: &HandCtx, card: &Card, rng: &mut RngState) -> u32 {
        // state_events.lua:676-684.
        let pareidolia = self.pareidolia();
        let mut reps = 0u32;
        for i in 0..self.jokers.len() {
            let Some((t, _bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            reps +=
                effects::repetition_play(&self.jokers[t], ctx, card, pareidolia, &self.env, rng);
        }
        reps
    }

    fn on_scoring_card(
        &mut self,
        ctx: &HandCtx,
        card: &Card,
        lucky_trigger: bool,
        rng: &mut RngState,
    ) -> Vec<CardEffect> {
        // state_events.lua:693-699 — every joker appends at most one entry.
        let pareidolia = self.pareidolia();
        let smeared = self.smeared();
        let mut out = Vec::new();
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            if let Some(eff) = effects::individual_play(
                &mut self.jokers[t],
                bp,
                ctx,
                card,
                lucky_trigger,
                pareidolia,
                smeared,
                &self.env,
                &mut self.out,
                rng,
            ) {
                out.push(eff);
            }
        }
        out
    }

    fn on_held_card(&mut self, ctx: &HandCtx, card: &Card, rng: &mut RngState) -> Vec<CardEffect> {
        // state_events.lua:800-807.
        let pareidolia = self.pareidolia();
        let mut out = Vec::new();
        for i in 0..self.jokers.len() {
            let Some((t, _bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            if let Some(eff) = effects::individual_held(
                &self.jokers[t],
                ctx,
                card,
                pareidolia,
                &self.env,
                &mut self.out,
                rng,
            ) {
                out.push(eff);
            }
        }
        out
    }

    fn held_card_repetitions(
        &mut self,
        _ctx: &HandCtx,
        _card: &Card,
        has_effects: bool,
        _rng: &mut RngState,
    ) -> u32 {
        // state_events.lua:820-829 — Mime (card.lua:3386-3394): +1 rep per
        // resolving Mime when the card produced any effect entry. Not
        // blueprint-gated.
        let mut reps = 0;
        if !has_effects {
            return 0;
        }
        for i in 0..self.jokers.len() {
            let Some((t, _bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            if self.jokers[t].id == JokerId::Mime {
                reps += 1; // extra = 1 (game.lua:387)
            }
        }
        reps
    }

    fn joker_main(
        &mut self,
        ctx: &HandCtx,
        hands: &HandsTable,
        chips: &mut f64,
        mult: &mut f64,
        dollars: &mut i64,
        blind_triggered: bool,
        rng: &mut RngState,
    ) {
        // state_events.lua:877-944 — see the module docs for the numbered
        // order of operations.
        let njokers = self.jokers.len();
        let total = njokers + self.env.consumable_keys.len();
        for i in 0..total {
            // 1. edition pre-effects (chip_mod, then mult_mod); x_mult held
            //    for step 4. Consumables carry no editions in the sim
            //    (negative is the only consumable edition and has no scoring
            //    mod). Debuffed jokers return no edition (card.lua:1017).
            let edition = if i < njokers && !self.jokers[i].debuffed {
                self.jokers[i].edition
            } else {
                Edition::None
            };
            let (e_chips, e_mult, e_x) = edition.scoring_mods();
            if let Some(c) = e_chips {
                *chips += c;
            }
            if let Some(m) = e_mult {
                *mult += m;
            }

            // 2. the card's own joker_main effect.
            if i < njokers {
                if let Some((t, bp)) = resolve_effective(self.jokers, i) {
                    let swash = self.swashbuckler_mult(t);
                    let live_x = self.live_x_mult(t);
                    let smeared = self.smeared();
                    if let Some(eff) = effects::joker_main(
                        &mut self.jokers[t],
                        bp,
                        ctx,
                        hands,
                        njokers,
                        swash,
                        live_x,
                        smeared,
                        blind_triggered,
                        &self.env,
                        &mut self.out,
                        dollars,
                        rng,
                    ) {
                        eff.apply(chips, mult);
                    }
                }
            } else {
                // Held Planet + Observatory (card.lua:2293-2302):
                // Xmult = v_observatory.config.extra = 1.5 (game.lua:613)
                // when the planet's hand matches the scoring hand.
                let key = self.env.consumable_keys[i - njokers];
                if self.env.observatory && crate::items::hand_for_planet(key) == Some(ctx.hand_type)
                {
                    *mult *= 1.5;
                }
            }

            // 3. joker-on-joker (state_events.lua:920-931): every joker
            //    evaluates `{other_joker = _card}`. Baseball Card (rare —
            //    ported with the engine since Judgement/Wraith can create
            //    it): x1.5 per OTHER uncommon joker (card.lua:3398-3410).
            if i < njokers {
                let other_rarity = self.jokers[i].id.meta().rarity;
                let other_sort_id = self.jokers[i].sort_id;
                for v in 0..njokers {
                    let Some((t, _bp)) = resolve_effective(self.jokers, v) else {
                        continue;
                    };
                    if self.jokers[t].id == JokerId::Baseball
                        && other_rarity == 2
                        && self.jokers[t].sort_id != other_sort_id
                    {
                        *mult *= 1.5;
                    }
                }
            }

            // 4. edition post-effect: Polychrome x1.5.
            if let Some(x) = e_x {
                *mult *= x;
            }
        }
    }

    fn destroys_card(&mut self, _card: &Card, full_hand: &[Card], rng: &mut RngState) -> bool {
        // state_events.lua:956-959 — the loop breaks at the first joker
        // that destroys. Sixth Sense (card.lua:2603-2621, blueprint-gated):
        // a single played 6 on the round's first hand.
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            if bp {
                // `elseif context.destroying_card and not context.blueprint`.
                continue;
            }
            if self.jokers[t].id == JokerId::SixthSense
                && full_hand.len() == 1
                && full_hand[0].id_for_eval() == Some(6)
                && self.env.hands_played_round == 0
            {
                // Slot-gated spectral ('sixth'); the destroy returns true
                // regardless of consumable space (card.lua:2604-2619).
                if self.env.consumables_len + self.out.consumable_buffer()
                    < self.env.consumable_slots
                {
                    self.out.create_consumable(ConsumableSet::Spectral, "sixth");
                }
                return true;
            }
        }
        let _ = rng;
        false
    }

    fn on_cards_destroyed(&mut self, destroyed: &[Card]) {
        // state_events.lua:974-976 — `{remove_playing_cards = true,
        // removed = …}` per joker: Caino counts destroyed faces
        // (card.lua:2673-2685), Glass Joker counts shattered Glass cards
        // (card.lua:2687-2707). Both blueprint-gated.
        let pareidolia = self.pareidolia();
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            if bp {
                continue;
            }
            match self.jokers[t].id {
                JokerId::Caino => {
                    // is_face() is nil for debuffed cards.
                    let faces = destroyed
                        .iter()
                        .filter(|c| card_is_face(c, pareidolia))
                        .count();
                    // caino_xmult += faces * extra (1).
                    self.jokers[t].state.extra += faces as f64;
                }
                JokerId::Glass => {
                    // `val.shattered` — every destroyed Glass card shatters
                    // (state_events.lua:966-970).
                    let glass = destroyed
                        .iter()
                        .filter(|c| c.enhancement == Enhancement::Glass)
                        .count();
                    // x_mult += extra (0.75) per shattered glass.
                    self.jokers[t].state.x_mult += 0.75 * glass as f64;
                }
                _ => {}
            }
        }
    }

    fn after_hand(&mut self, ctx: &HandCtx, rng: &mut RngState) {
        // state_events.lua:1068-1075 — `{after = true}` per joker.
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            effects::after_hand(&mut self.jokers[t], bp, ctx, &mut self.out, rng);
        }
    }

    fn on_debuffed_hand(&mut self, _ctx: &HandCtx, blind_triggered: bool, dollars: &mut i64) {
        // state_events.lua:1017-1027 — Matador pays extra (8) when the boss
        // triggered (card.lua:2735-2747). Not blueprint-gated.
        if !blind_triggered {
            return;
        }
        for i in 0..self.jokers.len() {
            let Some((t, _bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            if self.jokers[t].id == JokerId::Matador {
                *dollars += 8;
                self.out.dollar_buffer += 8;
            }
        }
    }

    fn pre_discard(
        &mut self,
        selected: &[Card],
        hook: bool,
        hands: &mut HandsTable,
        mods: &crate::handeval::EvalMods,
    ) {
        // state_events.lua:394-396 — Burnt Joker (card.lua:2749-2755, not
        // blueprint-gated): the round's first discard levels the selection's
        // hand once per resolving joker.
        if hook || self.env.discards_used_round > 0 {
            return;
        }
        for i in 0..self.jokers.len() {
            let Some((t, _bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            if self.jokers[t].id == JokerId::Burnt {
                let (ht, _, _) = crate::handeval::get_poker_hand_info(selected, mods);
                hands.level_up(ht, 1);
            }
        }
    }

    fn on_discard_card(
        &mut self,
        card: &Card,
        selected: &[Card],
        dollars: &mut i64,
        rng: &mut RngState,
    ) -> bool {
        // state_events.lua:402-409 — per joker; `eval.remove` destroys the
        // card (Trading Card).
        let pareidolia = self.pareidolia();
        let smeared = self.smeared();
        let mut removed = false;
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            removed |= effects::on_discard_card(
                &mut self.jokers[t],
                bp,
                card,
                selected,
                pareidolia,
                smeared,
                &self.env,
                &mut self.out,
                dollars,
                rng,
            );
        }
        removed
    }

    fn end_of_round(&mut self, hands: &HandsTable, game_over: bool, rng: &mut RngState) -> bool {
        // state_events.lua:99-110 — `{end_of_round = true, game_over = …}`
        // per joker. Every non-repetition end_of_round branch sits inside
        // `elseif not context.blueprint` (card.lua:2888), so copies do
        // nothing at all. `eval.saved` (Mr. Bones) flips game_over off for
        // the remaining jokers (state_events.lua:103-105).
        let mut game_over = game_over;
        let mut saved_any = false;
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            if bp {
                continue;
            }
            let saved = effects::end_of_round(
                &mut self.jokers[t],
                hands,
                game_over,
                &self.env,
                &mut self.out,
                rng,
            );
            #[cfg(feature = "p5-debug")]
            eprintln!(
                "  eor joker i={i} t={t} id={:?} saved={saved}",
                self.jokers[t].id
            );
            if saved {
                game_over = false;
                saved_any = true;
            }
        }
        saved_any
    }

    fn end_of_round_card_repetitions(&mut self, _card: &Card, has_effects: bool) -> u32 {
        // state_events.lua:199-208 — Mime again (card.lua:2877-2887), same
        // effects gate, not blueprint-gated.
        if !has_effects {
            return 0;
        }
        let mut reps = 0;
        for i in 0..self.jokers.len() {
            let Some((t, _bp)) = resolve_effective(self.jokers, i) else {
                continue;
            };
            if self.jokers[t].id == JokerId::Mime {
                reps += 1;
            }
        }
        reps
    }

    fn dollar_bonus(&mut self) -> i64 {
        // state_events.lua:1175-1182 — `Card:calculate_dollar_bonus`
        // (card.lua:1655-1679) per joker; no Blueprint resolution (it is a
        // separate function, never copied) but debuff still nils it.
        let mut total = 0;
        for j in self.jokers.iter() {
            if j.debuffed {
                continue;
            }
            total += effects::dollar_bonus(j, &self.env);
        }
        total
    }
}
