//! The shop, booster packs, vouchers and skip-blind tags — everything of
//! `G.STATES.SHOP` and the `*_PACK` states.
//!
//! Sources: `Game:update_shop` (game.lua:3072-3186), `create_card_for_shop`
//! (UI_definitions.lua:742-802), `create_card`/`get_current_pool`/`get_pack`/
//! `poll_edition`/`get_next_voucher_key`/`get_next_tag_key`/
//! `calculate_reroll_cost` (common_events.lua), `Card:open`/`Card:redeem`/
//! `Card:apply_to_run`/`Card:sell_card`/`Card:set_cost` (card.lua),
//! `G.FUNCS.buy_from_shop`/`reroll_shop`/`toggle_shop`/`skip_blind`/
//! `reroll_boss`/`use_card`/`skip_booster`/`end_consumeable`
//! (button_callbacks.lua), `Tag:apply_to_run` (tag.lua).
//!
//! # RNG stream keys consumed here (all per-ante keys use
//! `G.GAME.round_resets.ante` at roll time)
//!
//! Shop generation, per card slot (create_card_for_shop):
//!   'cdt<ante>'              item-type roll (UI_definitions.lua:768)
//!   'illusion'               with Illusion: Enhanced-vs-Base roll (>0.6),
//!                            then per playing card: edition gate (>0.8) and
//!                            edition pick (poly >0.85 / holo >0.5 / foil)
//!   'rarity<ante>sho'        joker rarity roll (common_events.lua:1969)
//!   'Joker<r>sho<ante>'      joker pool (+ '_resample<n>')
//!   'Tarotsho<ante>'         shop tarot pool (no soul poll: soulable=nil)
//!   'Planetsho<ante>'        shop planet pool
//!   'Spectralsho<ante>'      shop spectral pool (Ghost Deck only)
//!   'Enhancedsho<ante>'      enhancement pool for Illusion playing cards
//!   'frontsho<ante>'         playing-card front (52-key P_CARDS order)
//!   'etperpoll<ante>'        eternal/perishable poll — consumed for every
//!                            shop joker even on White Stake
//!                            (common_events.lua:2138)
//!   'edisho<ante>'           shop joker edition poll
//!   'shop_pack<ante>'        booster slot roll (skipped for the run's first
//!                            pack: the guaranteed Buffoon Pack)
//!   'Voucher<ante>'          (+ '_resample<n>') voucher key — run start &
//!                            every boss defeat
//!   'Voucher_fromtag'        (+ '_resample<n>') Voucher Tag's extra voucher
//!   'Tag<ante>'              (+ '_resample<n>') blind skip tags — run start
//!                            & every boss defeat, two draws
//!
//! Tag-generated shop jokers: 'Joker3rta<ante>'/'edirta<ante>' (Rare Tag),
//! 'Joker2uta<ante>'/'ediuta<ante>' (Uncommon Tag) — both with an
//! 'etperpoll<ante>' draw; no 'rarity' roll (forced rarity).
//!
//! Booster pack contents (Card:open, card.lua:1697-1758), per card:
//!   Arcana:    ['omen_globe' with Omen Globe] -> 'soul_Tarot<ante>' (0.3%
//!              The Soul) -> 'Tarotar1<ante>' pool; a spectral via Omen
//!              Globe: 'soul_Spectral<ante>' x2 -> 'Spectralar2<ante>'
//!   Celestial: 'soul_Planet<ante>' (0.3% Black Hole) -> 'Planetpl1<ante>';
//!              with Telescope the first card is forced (no rolls)
//!   Spectral:  'soul_Spectral<ante>' x2 (Soul, then Black Hole — the same
//!              stream twice, common_events.lua:2087-2099) ->
//!              'Spectralspe<ante>'
//!   Standard:  'stdset<ante>' (>0.6 Enhanced) -> ['Enhancedsta<ante>'] ->
//!              'frontsta<ante>' -> 'standard_edition<ante>' (mod 2, no
//!              negative) -> 'stdseal<ante>' (>0.8) -> ['stdsealtype<ante>']
//!   Buffoon:   'rarity<ante>buf' -> 'Joker<r>buf<ante>' ->
//!              'packetper<ante>' -> 'edibuf<ante>'
//!
//! Tag streams: 'orbital' (blind-select entry, one per undecided blind of
//! the ante), 'boss' (Boss Tag reroll), 'Joker1top<ante>'/'editop<ante>'
//! (Top-up Tag, no etper roll: area = G.jokers).
//!
//! # Modeled divergences (cosmetic only)
//! The run's first shop Buffoon pack and the Charm/Meteor tag mega packs
//! pick their art variant with a raw (unseeded) `math.random(1, 2)`
//! (common_events.lua:1947, tag.lua:214/230) whose state depends on VFX
//! call counts; the variants are config-identical, so the sim always uses
//! variant `_1`.

use crate::blinds::get_new_boss;
use crate::cards::{Card, Edition, Enhancement, HandType, Rank, Suit};
use crate::items::{
    self, booster_by_key, joker_by_key, ConsumableSet, JokerId, PackKind, PoolArgs, PoolSpec,
};
use crate::jokers::{effects::roll_to_do_hand, resolve_effective, JokerState};
use crate::rng::LuaRandom;
use crate::run::{BlindStage, Run, RunError, State};

/// A joker instance: identity + the mutable ability state the effect
/// engine (`crate::jokers`) reads and writes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OwnedJoker {
    pub id: JokerId,
    pub edition: Edition,
    /// `card.sort_id` — creation-order id; pseudorandom_element over jokers
    /// sorts by it (Wheel of Fortune, Ankh, Crimson Heart...).
    pub sort_id: u32,
    /// `ability.extra_value` (Egg each round; Gift Card); raises sell value.
    pub extra_value: i64,
    /// Crimson Heart's rotating debuff — nils every calculate_joker /
    /// get_edition and suspends the passive add_to_deck effects.
    pub debuffed: bool,
    /// Amber Acorn's face-down flip (observation masking only — flipped
    /// jokers still calculate).
    pub flipped: bool,
    /// `ability.eternal` — never set on White Stake, but Madness/Ceremonial
    /// check it and the copy paths (Ankh/Invisible) carry it
    /// (copy_card copies the whole ability table, common_events.lua:2161).
    pub eternal: bool,
    /// `ability.hands_played_at_create` (card.lua:337) — Loyalty Card's
    /// anchor; copies inherit the source value.
    pub hands_at_create: i64,
    /// Mutable ability state (counters/accumulators; see `JokerState`).
    pub state: JokerState,
}

/// A held consumable instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(deserialize = "")))]
pub struct OwnedConsumable {
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::key"))]
    pub key: crate::items::Key,
    pub sort_id: u32,
    /// Negative edition (Perkeo copies, card.lua:2413-2426): +1 consumable
    /// slot while held (card.lua:630-640/687-697).
    pub negative: bool,
    /// `ability.extra_value` (Gift Card) — raises sell value.
    pub extra_value: i64,
}

/// What sits in one shop card slot.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ShopItemKind {
    Joker(OwnedJoker),
    Consumable(OwnedConsumable),
    /// Base/Enhanced playing card (Magic Trick / Illusion).
    PlayingCard(Card),
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShopItem {
    pub kind: ShopItemKind,
    /// `ability.couponed` (Coupon Tag / tag-created jokers): cost 0 while in
    /// the shop.
    pub couponed: bool,
}

impl ShopItem {
    pub fn is_consumable(&self) -> bool {
        matches!(self.kind, ShopItemKind::Consumable(_))
    }

    /// The center key (jokers/consumables) or "c_base"/enhancement key.
    pub fn base_cost(&self) -> i64 {
        match &self.kind {
            ShopItemKind::Joker(j) => j.id.meta().cost,
            ShopItemKind::Consumable(c) => items::center_base_cost(c.key),
            // c_base / m_* centers: `center.cost or 1` (card.lua:339).
            ShopItemKind::PlayingCard(_) => 1,
        }
    }

    fn edition(&self) -> Edition {
        match &self.kind {
            ShopItemKind::Joker(j) => j.edition,
            ShopItemKind::Consumable(_) => Edition::None,
            ShopItemKind::PlayingCard(c) => c.edition,
        }
    }
}

/// A voucher on display in `G.shop_vouchers`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(deserialize = "")))]
pub struct VoucherOffer {
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::key"))]
    pub key: crate::items::Key,
    pub sort_id: u32,
    /// `card.shop_voucher` — true for the ante voucher (redeeming clears
    /// `G.GAME.current_round.voucher`), false for Voucher-Tag extras.
    pub shop_voucher: bool,
}

/// A booster pack on display in `G.shop_booster`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(deserialize = "")))]
pub struct PackOffer {
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::key"))]
    pub key: crate::items::Key,
    pub sort_id: u32,
    pub used: bool,
    pub couponed: bool,
}

/// The live shop: card slots, voucher slot(s), two booster slots.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShopState {
    pub jokers: Vec<ShopItem>,
    pub vouchers: Vec<VoucherOffer>,
    pub packs: Vec<PackOffer>,
}

/// One card inside an opened booster pack.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PackItem {
    Joker(OwnedJoker),
    Consumable(OwnedConsumable),
    PlayingCard(Card),
}

/// Where control returns when a pack closes (`G.GAME.PACK_INTERRUPT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PackReturn {
    Shop,
    BlindSelect,
}

/// An opened booster pack (`G.STATES.*_PACK`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(deserialize = "")))]
pub struct PackState {
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::key"))]
    pub key: crate::items::Key,
    pub kind: PackKind,
    /// `G.GAME.pack_choices` — picks left.
    pub choices_left: u8,
    pub items: Vec<PackItem>,
    pub return_to: PackReturn,
    /// Arcana/Spectral packs deal a hand for targeting (game.lua:3375/3427).
    pub hand_dealt: bool,
}

/// A held skip tag (`G.GAME.tags[i]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(deserialize = "")))]
pub struct RunTag {
    #[cfg_attr(feature = "serde", serde(with = "crate::snapshot::key"))]
    pub key: crate::items::Key,
    /// Orbital Tag's chosen hand (`ability.orbital_hand`), set at creation
    /// from `G.GAME.orbital_choices[ante][blind]` (tag.lua:104-112).
    pub orbital_hand: Option<HandType>,
}

/// `Tag:apply_to_run` context types (`config.type` values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TagContext {
    /// Blind select entry / after skip: money tags, Top-up, Orbital.
    /// (The 'new_blind_choice' pass has its own function:
    /// `apply_new_blind_choice_tags` — it breaks on the first match.)
    Immediate,
    /// Round start: Juggle.
    RoundStartBonus,
    /// Shop entry: D6.
    ShopStart,
    /// Shop generation, after packs: Voucher Tag.
    VoucherAdd,
    /// Shop generation, last: Coupon.
    ShopFinalPass,
}

/// Where a joker Card is created (`area` in create_card): decides the
/// eternal/perishable poll stream (common_events.lua:2135-2145).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JokerArea {
    Shop,
    Pack,
    /// `G.jokers` (Judgement/Soul/Wraith/Top-up) — no etper poll.
    Direct,
}

/// Builds a `PoolArgs` from disjoint `Run` fields so `&mut self.rng` can be
/// borrowed alongside it.
macro_rules! pool_args {
    ($run:expr) => {
        PoolArgs {
            ante: $run.ante,
            used_keys: &$run.used_keys,
            used_vouchers: &$run.used_vouchers,
            shop_vouchers: &[],
            hands: Some(&$run.hands_table),
            playing_cards: [&$run.deck, &$run.hand, &$run.discard_pile],
            pool_flags: &$run.pool_flags,
            showman: $run
                .jokers
                .iter()
                .any(|j| j.id == JokerId::RingMaster && !j.debuffed),
        }
    };
}

impl Run {
    // ------------------------------------------------------------------
    // Card-instance bookkeeping
    // ------------------------------------------------------------------

    /// `G.sort_id` advance for every modeled Card creation (card.lua:24).
    pub(crate) fn next_sort_id(&mut self) -> u32 {
        self.sort_id_counter += 1;
        self.sort_id_counter
    }

    /// `Card:set_ability` marks `G.GAME.used_jokers` for every center that
    /// shares the card's name (card.lua:348-355); our keys are name-unique
    /// for everything that reaches a pool, so refcount per key.
    pub(crate) fn add_used(&mut self, key: &str) {
        *self.used_keys.entry(key.to_string()).or_insert(0) += 1;
    }

    /// `Card:remove` clears the used_jokers entry when no same-named card
    /// remains (card.lua:4741-4749).
    pub(crate) fn remove_used(&mut self, key: &str) {
        if let Some(n) = self.used_keys.get_mut(key) {
            *n -= 1;
            if *n == 0 {
                self.used_keys.remove(key);
            }
        }
    }

    /// Create the Card for a consumable center and put it in the consumable
    /// area (used by seals, The Fool, Emperor/High Priestess).
    pub(crate) fn add_consumable(&mut self, key: &'static str) {
        let sort_id = self.next_sort_id();
        self.add_used(key);
        self.consumables.push(OwnedConsumable {
            key,
            sort_id,
            negative: false,
            extra_value: 0,
        });
    }

    /// `next(find_joker('Showman'))` — the pool-culling bypass
    /// (common_events.lua:1988) and the duplicate-allowing checks.
    pub(crate) fn showman_owned(&self) -> bool {
        self.jokers
            .iter()
            .any(|j| j.id == JokerId::RingMaster && !j.debuffed)
    }

    /// `next(find_joker('Astronomer'))` — Planets and Celestial packs cost 0
    /// (Card:set_cost, card.lua:380).
    pub(crate) fn astronomer_owned(&self) -> bool {
        self.jokers
            .iter()
            .any(|j| j.id == JokerId::Astronomer && !j.debuffed)
    }

    /// The center-key part of `create_card` for a consumable
    /// (common_events.lua:2082-2122): optional Soul/Black Hole polls, then
    /// the pool draw. Rolls only — the caller creates the Card.
    pub(crate) fn create_consumable_key(
        &mut self,
        set: ConsumableSet,
        append: &str,
        soulable: bool,
    ) -> &'static str {
        let mut forced: Option<&'static str> = None;
        if soulable {
            // common_events.lua:2087-2099. Each poll is gated on the target
            // card not being instantiated (used_jokers) — a blocked gate
            // consumes no draw. Spectral rolls the same stream twice.
            let soul_key = format!("soul_{}{}", set.type_str(), self.ante);
            if matches!(set, ConsumableSet::Tarot | ConsumableSet::Spectral)
                && !self.used_keys.contains_key("c_soul")
                && self.rng.random(&soul_key) > 0.997
            {
                forced = Some("c_soul");
            }
            if matches!(set, ConsumableSet::Planet | ConsumableSet::Spectral)
                && !self.used_keys.contains_key("c_black_hole")
                && self.rng.random(&soul_key) > 0.997
            {
                forced = Some("c_black_hole");
            }
        }
        if let Some(k) = forced {
            return k;
        }
        let spec = PoolSpec::Consumable { set, append };
        items::roll_from_pool(&mut self.rng, &spec, &pool_args!(self))
    }

    /// The Joker branch of `create_card` (common_events.lua:2102-2153):
    /// rarity roll + pool draw, Card creation, the eternal/perishable poll
    /// (White Stake: consumed, unused), and the edition poll.
    pub(crate) fn create_joker(
        &mut self,
        rarity: Option<f64>,
        legendary: bool,
        append: &str,
        area: JokerArea,
    ) -> OwnedJoker {
        let spec = PoolSpec::Joker {
            rarity,
            legendary,
            append,
        };
        let key = items::roll_from_pool(&mut self.rng, &spec, &pool_args!(self));
        let id = joker_by_key(key).expect("joker key").id;
        let sort_id = self.next_sort_id();
        self.add_used(key);
        // Card:set_ability runs at Card creation, BEFORE the sticker/edition
        // polls: To Do List rolls its poker hand on 'to_do' here
        // (card.lua:311-322) — even for shop cards that are never bought.
        let mut state = JokerState::initial(id);
        if id == JokerId::TodoList {
            state.to_do_hand = roll_to_do_hand(&mut self.rng, &self.hands_table, None);
        }
        // common_events.lua:2135-2141 — the poll happens for shop/pack areas
        // regardless of stake modifiers; the rental poll ('ssjr'/'packssjr')
        // is short-circuited away below stake 8.
        match area {
            JokerArea::Shop => {
                let _ = self.rng.random(&format!("etperpoll{}", self.ante));
            }
            JokerArea::Pack => {
                let _ = self.rng.random(&format!("packetper{}", self.ante));
            }
            JokerArea::Direct => {}
        }
        // common_events.lua:2148-2149.
        let edition = items::poll_edition(
            &mut self.rng,
            &format!("edi{}{}", append, self.ante),
            1.0,
            false,
            false,
            self.edition_rate,
        );
        OwnedJoker {
            id,
            edition,
            sort_id,
            extra_value: 0,
            debuffed: false,
            flipped: false,
            eternal: false,
            hands_at_create: self.hands_played_total,
            state,
        }
    }

    /// `get_next_voucher_key(_from_tag)` (common_events.lua:1901-1912).
    pub(crate) fn next_voucher_key(&mut self, from_tag: bool) -> &'static str {
        let displayed: Vec<&'static str> = self
            .shop
            .as_ref()
            .map(|s| s.vouchers.iter().map(|v| v.key).collect())
            .unwrap_or_default();
        let args = PoolArgs {
            ante: self.ante,
            used_keys: &self.used_keys,
            used_vouchers: &self.used_vouchers,
            shop_vouchers: &displayed,
            hands: Some(&self.hands_table),
            playing_cards: [&self.deck, &self.hand, &self.discard_pile],
            pool_flags: &self.pool_flags,
            showman: self.showman_owned(),
        };
        let (pool, mut pool_key) =
            items::get_current_pool(&mut self.rng, &PoolSpec::Voucher, &args);
        if from_tag {
            // `_pool_key = 'Voucher_fromtag'` — no ante suffix.
            pool_key = "Voucher_fromtag".to_string();
        }
        items::pool_draw(&mut self.rng, &pool, &pool_key)
    }

    /// `get_next_tag_key()` (common_events.lua:1914-1925). The `G.FORCE_TAG`
    /// debug override returns without touching the 'Tag' stream (:1915).
    pub(crate) fn next_tag_key(&mut self) -> &'static str {
        if let Some(k) = self.forced_tag {
            return k;
        }
        items::roll_from_pool(&mut self.rng, &PoolSpec::Tag, &pool_args!(self))
    }

    /// `calculate_reroll_cost(skip_increment)` (common_events.lua:2263-2269).
    pub(crate) fn calculate_reroll_cost(&mut self, skip_increment: bool) {
        if self.free_rerolls < 0 {
            self.free_rerolls = 0;
        }
        if self.free_rerolls > 0 {
            self.reroll_cost = 0;
            return;
        }
        if !skip_increment {
            self.reroll_cost_increase += 1;
        }
        self.reroll_cost =
            self.temp_reroll_cost.unwrap_or(self.base_reroll_cost) + self.reroll_cost_increase;
    }

    /// `cost <= dollars - bankrupt_at` (`G.FUNCS.can_buy`; free items are
    /// always buyable).
    pub(crate) fn affordable(&self, cost: i64) -> bool {
        cost <= 0 || cost <= self.dollars - self.bankrupt_at
    }

    // ------------------------------------------------------------------
    // Blind select entry: orbital rolls + tag passes
    // ------------------------------------------------------------------

    /// Entering BLIND_SELECT (`Game:update_blind_select`,
    /// game.lua:3260-3312): the blind-choice UI rolls each undecided blind's
    /// Orbital hand for this ante (Small, Big, Boss order;
    /// UI_definitions.lua:1447-1449 + 1506-1516), then tags apply
    /// 'immediate' and 'new_blind_choice'.
    pub(crate) fn enter_blind_select(&mut self) {
        self.state = State::BlindSelect;
        self.roll_orbital_choices();
        self.apply_tags(TagContext::Immediate);
        self.apply_new_blind_choice_tags();
    }

    fn roll_orbital_choices(&mut self) {
        // `_poker_hands`: visible hands in pairs(G.GAME.hands) order — the
        // same iteration order as end_round's most-played scan
        // (run.rs MOST_PLAYED_SCAN, oracle-verified).
        for slot in 0..3usize {
            let done = self
                .orbital_choices
                .get(&self.ante)
                .map(|c| c[slot].is_some())
                .unwrap_or(false);
            if done {
                continue;
            }
            let pool: Vec<HandType> = crate::run::MOST_PLAYED_SCAN
                .iter()
                .copied()
                .filter(|&ht| self.hands_table.get(ht).visible)
                .collect();
            let idx = items::element_index(&mut self.rng, "orbital", pool.len());
            let entry = self.orbital_choices.entry(self.ante).or_insert([None; 3]);
            entry[slot] = Some(pool[idx]);
        }
    }

    /// The chosen Orbital hand for a blind of this ante (rolled at blind
    /// select display).
    pub fn orbital_choice(&self, stage: BlindStage) -> Option<HandType> {
        let slot = match stage {
            BlindStage::Small => 0,
            BlindStage::Big => 1,
            BlindStage::Boss => 2,
        };
        self.orbital_choices.get(&self.ante).and_then(|c| c[slot])
    }

    // ------------------------------------------------------------------
    // Tags
    // ------------------------------------------------------------------

    /// `add_tag(_tag)` (UI_definitions.lua:1252-1272): existing Double Tags
    /// trigger 'tag_add' and queue copies (which are appended after the new
    /// tag; their own tag_add pass finds no untriggered Double left).
    pub(crate) fn add_tag(&mut self, key: &'static str, orbital_hand: Option<HandType>) {
        let mut copies = 0usize;
        if key != "tag_double" {
            // tag.lua:335-349 — every held Double Tag triggers (no break).
            self.tags.retain(|t| {
                if t.key == "tag_double" {
                    copies += 1;
                    false
                } else {
                    true
                }
            });
        }
        self.tags.push(RunTag { key, orbital_hand });
        for _ in 0..copies {
            self.tags.push(RunTag { key, orbital_hand });
        }
    }

    /// Non-branching tag passes (each matching tag triggers and is removed;
    /// the game loops all tags without breaking for these contexts).
    pub(crate) fn apply_tags(&mut self, ctx: TagContext) {
        let mut i = 0;
        while i < self.tags.len() {
            let tag = self.tags[i];
            if self.apply_one_tag(tag, ctx) {
                self.tags.remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// 'new_blind_choice' pass: only the FIRST matching tag triggers per
    /// pass (`if tags[i]:apply_to_run(...) then break`); pack tags interrupt
    /// into a pack state and the pass re-runs when the pack closes
    /// (end_consumeable, button_callbacks.lua:2620-2622). Boss Tag rerolls
    /// and re-runs the pass immediately (reroll_boss,
    /// button_callbacks.lua:2846-2848).
    pub(crate) fn apply_new_blind_choice_tags(&mut self) {
        loop {
            let Some(pos) = self.tags.iter().position(|t| {
                matches!(
                    t.key,
                    "tag_charm"
                        | "tag_meteor"
                        | "tag_ethereal"
                        | "tag_standard"
                        | "tag_buffoon"
                        | "tag_boss"
                )
            }) else {
                return;
            };
            let tag = self.tags.remove(pos);
            match tag.key {
                // tag.lua:206-296 — free mega/normal packs, opened on the
                // spot (cost 0, from_tag). Variant `_1` (see module docs).
                "tag_charm" => {
                    self.open_pack_key("p_arcana_mega_1", true);
                    return;
                }
                "tag_meteor" => {
                    self.open_pack_key("p_celestial_mega_1", true);
                    return;
                }
                "tag_ethereal" => {
                    self.open_pack_key("p_spectral_normal_1", true);
                    return;
                }
                "tag_standard" => {
                    self.open_pack_key("p_standard_mega_1", true);
                    return;
                }
                "tag_buffoon" => {
                    self.open_pack_key("p_buffoon_mega_1", true);
                    return;
                }
                // tag.lua:297-315 — Boss Tag: free boss reroll (sets the
                // Director's Cut latch too, button_callbacks.lua:2803).
                "tag_boss" => {
                    self.boss_rerolled = true;
                    self.boss_choice =
                        get_new_boss(&mut self.rng, &mut self.bosses_used, self.ante, 8);
                    // reroll_boss re-runs the new_blind_choice pass — the
                    // loop continues.
                }
                _ => unreachable!(),
            }
        }
    }

    /// One tag against one context; true = triggered (remove it).
    fn apply_one_tag(&mut self, tag: RunTag, ctx: TagContext) -> bool {
        match (ctx, tag.key) {
            // tag.lua:133-146 — Top-up: up to 2 common jokers (rarity forced
            // 0, append 'top', area G.jokers: no etper poll).
            (TagContext::Immediate, "tag_top_up") => {
                for _ in 0..2 {
                    if self.jokers.len() < self.joker_slots {
                        let j = self.create_joker(Some(0.0), false, "top", JokerArea::Direct);
                        self.add_joker(j);
                    }
                }
                true
            }
            // tag.lua:148-156 — Skip Tag: $5 per skip (incl. this one).
            (TagContext::Immediate, "tag_skip") => {
                self.dollars += 5 * self.skips;
                true
            }
            // tag.lua:158-166 — Garbage: $1 per unused discard this run.
            (TagContext::Immediate, "tag_garbage") => {
                self.dollars += self.unused_discards;
                true
            }
            // tag.lua:168-176 — Handy: $1 per hand played this run.
            (TagContext::Immediate, "tag_handy") => {
                self.dollars += self.hands_played_total;
                true
            }
            // tag.lua:178-192 — Economy: double money, max +$40.
            (TagContext::Immediate, "tag_economy") => {
                // math.min(max, math.max(0, dollars)) (tag.lua:186).
                self.dollars += self.dollars.clamp(0, 40);
                true
            }
            // tag.lua:193-206 — Orbital: +3 levels of the rolled hand.
            (TagContext::Immediate, "tag_orbital") => {
                if let Some(ht) = tag.orbital_hand {
                    self.hands_table.level_up(ht, 3);
                }
                true
            }
            // tag.lua:351-360 — Juggle: +3 hand size this round.
            (TagContext::RoundStartBonus, "tag_juggle") => {
                self.hand_size += 3;
                self.temp_handsize = Some(self.temp_handsize.unwrap_or(0) + 3);
                true
            }
            // tag.lua:404-414 — D6: rerolls start at $0 this shop.
            (TagContext::ShopStart, "tag_d_six") => {
                if !self.shop_d6ed {
                    self.shop_d6ed = true;
                    self.temp_reroll_cost = Some(0);
                    self.calculate_reroll_cost(true);
                    true
                } else {
                    false
                }
            }
            // tag.lua:317-333 — Voucher Tag: an extra voucher for this shop.
            (TagContext::VoucherAdd, "tag_voucher") => {
                let key = self.next_voucher_key(true);
                let sort_id = self.next_sort_id();
                self.add_used(key);
                if let Some(shop) = self.shop.as_mut() {
                    shop.vouchers.push(VoucherOffer {
                        key,
                        sort_id,
                        shop_voucher: false,
                    });
                }
                true
            }
            // tag.lua:466-484 — Coupon: card slots + boosters free.
            (TagContext::ShopFinalPass, "tag_coupon") if !self.shop_free => {
                self.shop_free = true;
                if let Some(shop) = self.shop.as_mut() {
                    for item in shop.jokers.iter_mut() {
                        item.couponed = true;
                    }
                    for p in shop.packs.iter_mut() {
                        p.couponed = true;
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// The 'eval' tag pass of `G.FUNCS.evaluate_round`
    /// (state_events.lua:1183-1190): Investment Tags pay $25 each after a
    /// boss (tag.lua:117-131). Returns the payout total.
    pub(crate) fn apply_eval_tags(&mut self) -> i64 {
        let mut dollars = 0;
        if self.last_blind_boss {
            let before = self.tags.len();
            self.tags.retain(|t| t.key != "tag_investment");
            dollars += 25 * (before - self.tags.len()) as i64;
        }
        dollars
    }

    // ------------------------------------------------------------------
    // Skip blind / boss reroll
    // ------------------------------------------------------------------

    /// `G.FUNCS.skip_blind` (button_callbacks.lua:2740-2782): take the
    /// blind's tag, advance the track, then run the tag passes.
    pub fn skip_blind(&mut self) -> Result<(), RunError> {
        if self.state != State::BlindSelect {
            return Err(RunError::WrongState);
        }
        if self.blind_on_deck == BlindStage::Boss {
            return Err(RunError::BadSlot("cannot skip the boss".into()));
        }
        self.skips += 1; // :2755
        let stage = self.blind_on_deck;
        let (key, orbital) = match stage {
            BlindStage::Small => (self.blind_tag_small, self.orbital_choice(BlindStage::Small)),
            BlindStage::Big => (self.blind_tag_big, self.orbital_choice(BlindStage::Big)),
            BlindStage::Boss => unreachable!(),
        };
        // Orbital hand rides on the tag (Tag:set_ability, tag.lua:104-112).
        let orbital = if key == "tag_orbital" { orbital } else { None };
        self.add_tag(key, orbital);
        match stage {
            BlindStage::Small => {
                self.small_skipped = true;
                self.blind_on_deck = BlindStage::Big;
            }
            BlindStage::Big => {
                self.big_skipped = true;
                self.blind_on_deck = BlindStage::Boss;
            }
            BlindStage::Boss => unreachable!(),
        }
        // :2768-2777 — jokers' skip_blind hook (P3c), then the tag passes.
        self.apply_tags(TagContext::Immediate);
        self.apply_new_blind_choice_tags();
        Ok(())
    }

    /// `G.FUNCS.reroll_boss_button` availability
    /// (button_callbacks.lua:2784-2799).
    pub fn can_reroll_boss(&self) -> bool {
        self.state == State::BlindSelect
            && self.dollars - self.bankrupt_at - 10 >= 0
            && (self.used_vouchers.contains("v_retcon")
                || (self.used_vouchers.contains("v_directors_cut") && !self.boss_rerolled))
    }

    /// `G.FUNCS.reroll_boss` (button_callbacks.lua:2801-2853), the paid
    /// Director's Cut / Retcon path ($10; consumes a 'boss' draw).
    pub fn reroll_boss(&mut self) -> Result<(), RunError> {
        if self.state != State::BlindSelect {
            return Err(RunError::WrongState);
        }
        if !self.can_reroll_boss() {
            return Err(RunError::CannotUse("boss reroll unavailable".into()));
        }
        self.boss_rerolled = true;
        self.dollars -= 10;
        self.boss_choice = get_new_boss(&mut self.rng, &mut self.bosses_used, self.ante, 8);
        // :2846-2848 — the new_blind_choice pass re-runs.
        self.apply_new_blind_choice_tags();
        Ok(())
    }

    // ------------------------------------------------------------------
    // Shop generation
    // ------------------------------------------------------------------

    /// `Game:update_shop` first-entry generation (game.lua:3090-3169).
    pub(crate) fn generate_shop(&mut self) {
        // game.lua:3093-3095 — shop_start tags (D6) first.
        self.apply_tags(TagContext::ShopStart);
        self.shop = Some(ShopState::default());

        // Card slots (game.lua:3111-3113).
        for _ in 0..self.shop_joker_max {
            let item = self.create_card_for_shop();
            self.shop.as_mut().unwrap().jokers.push(item);
        }
        // The store_joker_modify tags queue one pass per created card
        // (create_card_for_shop, UI_definitions.lua:782-789), which runs
        // after this synchronous block.
        self.apply_store_joker_modify_tags();

        // Voucher slot (game.lua:3126-3135).
        if let Some(key) = self.current_voucher {
            let sort_id = self.next_sort_id();
            self.add_used(key);
            self.shop.as_mut().unwrap().vouchers.push(VoucherOffer {
                key,
                sort_id,
                shop_voucher: true,
            });
        }

        // Booster slots (game.lua:3148-3163): the run's first pack is a
        // Buffoon Pack (get_pack, common_events.lua:1945-1948; no
        // 'shop_pack' draw), everything else rolls 'shop_pack<ante>'.
        for _ in 0..2 {
            let key = if !self.first_shop_buffoon {
                self.first_shop_buffoon = true;
                "p_buffoon_normal_1"
            } else {
                items::get_pack_roll(&mut self.rng, "shop_pack", self.ante).key
            };
            let sort_id = self.next_sort_id();
            self.shop.as_mut().unwrap().packs.push(PackOffer {
                key,
                sort_id,
                used: false,
                couponed: false,
            });
        }

        // game.lua:3165-3170 — voucher_add (Voucher Tag), then
        // shop_final_pass (Coupon).
        self.apply_tags(TagContext::VoucherAdd);
        self.apply_tags(TagContext::ShopFinalPass);
    }

    /// `create_card_for_shop(G.shop_jokers)` (UI_definitions.lua:742-802).
    fn create_card_for_shop(&mut self) -> ShopItem {
        // store_joker_create tags replace the whole roll
        // (UI_definitions.lua:755-766): first untriggered Uncommon/Rare Tag.
        if let Some(pos) = self
            .tags
            .iter()
            .position(|t| matches!(t.key, "tag_uncommon" | "tag_rare"))
        {
            let tag = self.tags.remove(pos);
            let forced = match tag.key {
                // tag.lua:362-386 — Rare Tag nopes when every rare is owned.
                "tag_rare" => {
                    let owned_rares: std::collections::HashSet<JokerId> = self
                        .jokers
                        .iter()
                        .filter(|j| j.id.meta().rarity == 3)
                        .map(|j| j.id)
                        .collect();
                    if owned_rares.len() < items::JOKERS.iter().filter(|m| m.rarity == 3).count() {
                        Some(self.create_joker(Some(1.0), false, "rta", JokerArea::Shop))
                    } else {
                        None
                    }
                }
                // tag.lua:387-398 — Uncommon Tag.
                "tag_uncommon" => Some(self.create_joker(Some(0.9), false, "uta", JokerArea::Shop)),
                _ => unreachable!(),
            };
            if let Some(j) = forced {
                let mut item = ShopItem {
                    kind: ShopItemKind::Joker(j),
                    couponed: true, // ability.couponed (tag.lua:377/392)
                };
                // The forced card gets its store_joker_modify pass inline
                // (UI_definitions.lua:759-762).
                self.modify_tags_for_item(&mut item);
                return item;
            }
            // Nope'd Rare Tag: fall through to the normal roll.
        }

        // UI_definitions.lua:767-770 — the type roll.
        let total_rate = self.joker_rate
            + self.tarot_rate
            + self.planet_rate
            + self.playing_card_rate
            + self.spectral_rate;
        let polled_rate = self.rng.random(&format!("cdt{}", self.ante)) * total_rate;
        // The type table constructor evaluates the Illusion roll before the
        // walk, whenever v_illusion is redeemed (UI_definitions.lua:776).
        let playing_card_enhanced =
            self.used_vouchers.contains("v_illusion") && self.rng.random("illusion") > 0.6;

        let rates = [
            self.joker_rate,
            self.tarot_rate,
            self.planet_rate,
            self.playing_card_rate,
            self.spectral_rate,
        ];
        let mut check_rate = 0.0f64;
        let mut chosen = 0usize;
        for (i, val) in rates.iter().enumerate() {
            if polled_rate > check_rate && polled_rate <= check_rate + val {
                chosen = i;
                break;
            }
            check_rate += val;
        }

        let kind = match chosen {
            0 => ShopItemKind::Joker(self.create_joker(None, false, "sho", JokerArea::Shop)),
            1 => {
                let key = self.create_consumable_key(ConsumableSet::Tarot, "sho", false);
                let sort_id = self.next_sort_id();
                self.add_used(key);
                ShopItemKind::Consumable(OwnedConsumable {
                    key,
                    sort_id,
                    negative: false,
                    extra_value: 0,
                })
            }
            2 => {
                let key = self.create_consumable_key(ConsumableSet::Planet, "sho", false);
                let sort_id = self.next_sort_id();
                self.add_used(key);
                ShopItemKind::Consumable(OwnedConsumable {
                    key,
                    sort_id,
                    negative: false,
                    extra_value: 0,
                })
            }
            4 => {
                let key = self.create_consumable_key(ConsumableSet::Spectral, "sho", false);
                let sort_id = self.next_sort_id();
                self.add_used(key);
                ShopItemKind::Consumable(OwnedConsumable {
                    key,
                    sort_id,
                    negative: false,
                    extra_value: 0,
                })
            }
            // Playing card (Magic Trick rate): Base or Enhanced.
            _ => {
                let enhancement = if playing_card_enhanced {
                    let spec = PoolSpec::Enhanced { append: "sho" };
                    let key = items::roll_from_pool(&mut self.rng, &spec, &pool_args!(self));
                    enhancement_from_key(key)
                } else {
                    Enhancement::None
                };
                // create_card front roll (common_events.lua:2124).
                let idx =
                    items::element_index(&mut self.rng, &format!("frontsho{}", self.ante), 52);
                let (suit, rank) = items::card_front_from_index(idx);
                let sort_id = self.next_sort_id();
                let mut card = Card::new(rank, suit, sort_id);
                card.enhancement = enhancement;
                // UI_definitions.lua:791-799 — Illusion edition rolls.
                if self.used_vouchers.contains("v_illusion") && self.rng.random("illusion") > 0.8 {
                    let poll = self.rng.random("illusion");
                    card.edition = if poll > 1.0 - 0.15 {
                        Edition::Polychrome
                    } else if poll > 0.5 {
                        Edition::Holo
                    } else {
                        Edition::Foil
                    };
                }
                ShopItemKind::PlayingCard(card)
            }
        };
        ShopItem {
            kind,
            couponed: false,
        }
    }

    /// The queued per-card 'store_joker_modify' pass: for each shop card in
    /// creation order, the first untriggered edition tag applies to an
    /// edition-less joker, makes it free, and is consumed
    /// (tag.lua:416-465).
    fn apply_store_joker_modify_tags(&mut self) {
        let Some(mut shop) = self.shop.take() else {
            return;
        };
        for item in shop.jokers.iter_mut() {
            self.modify_tags_for_item(item);
        }
        self.shop = Some(shop);
    }

    fn modify_tags_for_item(&mut self, item: &mut ShopItem) {
        let ShopItemKind::Joker(j) = &mut item.kind else {
            return;
        };
        if j.edition != Edition::None {
            return;
        }
        let Some(pos) = self.tags.iter().position(|t| {
            matches!(
                t.key,
                "tag_foil" | "tag_holo" | "tag_polychrome" | "tag_negative"
            )
        }) else {
            return;
        };
        let tag = self.tags.remove(pos);
        j.edition = match tag.key {
            "tag_foil" => Edition::Foil,
            "tag_holo" => Edition::Holo,
            "tag_polychrome" => Edition::Polychrome,
            "tag_negative" => Edition::Negative,
            _ => unreachable!(),
        };
        item.couponed = true; // ability.couponed (tag.lua:430 etc.)
    }

    // ------------------------------------------------------------------
    // Shop costs / affordability
    // ------------------------------------------------------------------

    /// `Card:set_cost` for a shop card slot item. Astronomer zeroes Planet
    /// cards (card.lua:380).
    pub fn shop_item_cost(&self, item: &ShopItem) -> i64 {
        if let ShopItemKind::Consumable(c) = &item.kind {
            if self.astronomer_owned()
                && matches!(
                    items::consumable_by_key(c.key),
                    Some((_, ConsumableSet::Planet))
                )
            {
                return 0;
            }
        }
        items::item_cost(
            item.base_cost(),
            item.edition(),
            self.discount_percent,
            item.couponed,
        )
    }

    /// Voucher cost (vouchers are never couponed; discount applies).
    pub fn voucher_cost(&self, key: &str) -> i64 {
        items::item_cost(
            items::center_base_cost(key),
            Edition::None,
            self.discount_percent,
            false,
        )
    }

    /// Booster pack cost. Astronomer zeroes Celestial packs (card.lua:380
    /// — the name:find('Celestial') check).
    pub fn pack_cost(&self, offer: &PackOffer) -> i64 {
        if self.astronomer_owned() && offer.key.starts_with("p_celestial") {
            return 0;
        }
        items::item_cost(
            items::center_base_cost(offer.key),
            Edition::None,
            self.discount_percent,
            offer.couponed,
        )
    }

    pub(crate) fn pack_affordable(&self, offer: &PackOffer) -> bool {
        self.affordable(self.pack_cost(offer))
    }

    /// `G.FUNCS.check_for_buy_space` (button_callbacks.lua:2391-2402) +
    /// `can_buy` affordability.
    pub(crate) fn can_buy_shop_item(&self, item: &ShopItem) -> bool {
        if !self.affordable(self.shop_item_cost(item)) {
            return false;
        }
        match &item.kind {
            ShopItemKind::Joker(j) => {
                let extra = if j.edition == Edition::Negative { 1 } else { 0 };
                self.jokers.len() < self.joker_slots + extra
            }
            ShopItemKind::Consumable(_) => self.consumables.len() < self.consumable_slots,
            ShopItemKind::PlayingCard(_) => true,
        }
    }

    pub(crate) fn can_reroll_shop(&self) -> bool {
        self.state == State::Shop
            && ((self.dollars - self.bankrupt_at) - self.reroll_cost >= 0 || self.reroll_cost == 0)
    }

    // ------------------------------------------------------------------
    // Shop actions
    // ------------------------------------------------------------------

    /// Add a joker to the joker area (`add_to_deck` + emplace): negative
    /// editions raise the joker slot count (card.lua:633-641) and the
    /// passive effects apply (Juggler/Drunkard/Credit Card/Chaos,
    /// card.lua:564-644).
    pub(crate) fn add_joker(&mut self, j: OwnedJoker) {
        if j.edition == Edition::Negative {
            self.joker_slots += 1;
        }
        self.joker_passives_on(&j);
        self.jokers.push(j);
    }

    /// `G.FUNCS.buy_from_shop` (button_callbacks.lua:2404-2479).
    pub fn buy_shop_item(&mut self, slot: usize) -> Result<(), RunError> {
        self.buy_shop_item_inner(slot, false, &[])
    }

    /// The 'buy_and_use' path for shop consumables: pays, then uses the card
    /// straight from the shop (button_callbacks.lua:2473-2475).
    pub fn buy_and_use_shop_item(
        &mut self,
        slot: usize,
        targets: &[usize],
    ) -> Result<(), RunError> {
        self.buy_shop_item_inner(slot, true, targets)
    }

    fn buy_shop_item_inner(
        &mut self,
        slot: usize,
        and_use: bool,
        targets: &[usize],
    ) -> Result<(), RunError> {
        if self.state != State::Shop {
            return Err(RunError::WrongState);
        }
        let shop = self.shop.as_ref().ok_or(RunError::WrongState)?;
        let item = shop
            .jokers
            .get(slot)
            .ok_or_else(|| RunError::BadSlot(format!("shop slot {slot}")))?
            .clone();
        let cost = self.shop_item_cost(&item);
        if !self.affordable(cost) {
            return Err(RunError::InsufficientFunds);
        }
        if and_use {
            let ShopItemKind::Consumable(c) = &item.kind else {
                return Err(RunError::CannotUse("not a consumable".into()));
            };
            self.consumable_can_use(c.key, targets)
                .map_err(RunError::CannotUse)?;
        } else {
            // check_for_buy_space (buy-and-use bypasses it).
            match &item.kind {
                ShopItemKind::Joker(j) => {
                    let extra = if j.edition == Edition::Negative { 1 } else { 0 };
                    if self.jokers.len() >= self.joker_slots + extra {
                        return Err(RunError::NoSpace);
                    }
                }
                ShopItemKind::Consumable(_) => {
                    if self.consumables.len() >= self.consumable_slots {
                        return Err(RunError::NoSpace);
                    }
                }
                ShopItemKind::PlayingCard(_) => {}
            }
        }
        // Commit: remove from the shop, emplace, pay.
        self.shop.as_mut().unwrap().jokers.remove(slot);
        match item.kind {
            ShopItemKind::PlayingCard(card) => {
                // buy_from_shop: G.deck:emplace(c1) — deck emplace inserts at
                // index 1, the bottom (cardarea.lua:33-34); playing_card
                // counter advances and playing_card_joker_effects fires
                // (button_callbacks.lua:2425-2431 — Hologram).
                self.playing_card_counter += 1;
                self.deck.insert(0, card);
                self.joker_playing_cards_added(1);
            }
            ShopItemKind::Joker(j) => {
                if and_use {
                    return Err(RunError::CannotUse("not a consumable".into()));
                }
                self.add_joker(j);
            }
            ShopItemKind::Consumable(c) => {
                if and_use {
                    // Pay first, then use (use_card runs at the end of the
                    // buy event, button_callbacks.lua:2466-2475).
                    self.dollars -= cost;
                    self.use_consumable_key(c.key, targets)?;
                    self.remove_used(c.key);
                    return Ok(());
                }
                self.consumables.push(c);
            }
        }
        if cost != 0 {
            self.dollars -= cost;
        }
        Ok(())
    }

    /// `G.FUNCS.use_card` on a shop voucher -> `Card:redeem`
    /// (card.lua:1813-1878) + `Card:apply_to_run` (card.lua:1880-1971).
    pub fn redeem_voucher(&mut self, slot: usize) -> Result<(), RunError> {
        if self.state != State::Shop {
            return Err(RunError::WrongState);
        }
        let shop = self.shop.as_ref().ok_or(RunError::WrongState)?;
        let offer = *shop
            .vouchers
            .get(slot)
            .ok_or_else(|| RunError::BadSlot(format!("voucher slot {slot}")))?;
        let cost = self.voucher_cost(offer.key);
        if self.dollars - self.bankrupt_at < cost {
            return Err(RunError::InsufficientFunds);
        }
        self.shop.as_mut().unwrap().vouchers.remove(slot);
        self.remove_used(offer.key);
        // card.lua:1822/1854 — used_vouchers set, current_round.voucher
        // cleared, money paid, then apply_to_run.
        self.used_vouchers.insert(offer.key);
        self.dollars -= cost;
        self.current_voucher = None;
        self.apply_voucher(offer.key);
        Ok(())
    }

    /// `Card:apply_to_run` (card.lua:1880-1971) for the redeemed voucher.
    pub(crate) fn apply_voucher(&mut self, key: &'static str) {
        match key {
            // change_shop_size(1) (common_events.lua:1097-1118): +1 slot,
            // refilled immediately while the shop is open.
            "v_overstock_norm" | "v_overstock_plus" => {
                self.shop_joker_max += 1;
                if self.shop.is_some() {
                    let n = self
                        .shop_joker_max
                        .saturating_sub(self.shop.as_ref().map(|s| s.jokers.len()).unwrap_or(0));
                    for _ in 0..n {
                        let item = self.create_card_for_shop();
                        self.shop.as_mut().unwrap().jokers.push(item);
                    }
                    self.apply_store_joker_modify_tags();
                }
            }
            // G.GAME.tarot_rate = 4 * extra (9.6/4 or 32/4).
            "v_tarot_merchant" => self.tarot_rate = 9.6,
            "v_tarot_tycoon" => self.tarot_rate = 32.0,
            "v_planet_merchant" => self.planet_rate = 9.6,
            "v_planet_tycoon" => self.planet_rate = 32.0,
            // G.GAME.edition_rate = extra.
            "v_hone" => self.edition_rate = 2.0,
            "v_glow_up" => self.edition_rate = 4.0,
            // G.GAME.playing_card_rate = extra.
            "v_magic_trick" | "v_illusion" => self.playing_card_rate = 4.0,
            // Telescope/Observatory alter pack generation / scoring only.
            "v_telescope" | "v_observatory" => {}
            // +1 consumable slot.
            "v_crystal_ball" => self.consumable_slots += 1,
            // G.GAME.discount_percent = extra; costs recompute on the fly.
            "v_clearance_sale" => self.discount_percent = 25,
            "v_liquidation" => self.discount_percent = 50,
            // round_resets.reroll_cost -= 2; current cost floored at 0.
            "v_reroll_surplus" | "v_reroll_glut" => {
                self.base_reroll_cost -= 2;
                self.reroll_cost = (self.reroll_cost - 2).max(0);
            }
            // G.GAME.interest_cap = extra.
            "v_seed_money" => self.interest_cap = 50,
            "v_money_tree" => self.interest_cap = 100,
            // +extra hands per round, applied to the live count too.
            "v_grabber" | "v_nacho_tong" => {
                self.round_resets_hands += 1;
                self.hands_left += 1;
            }
            // Paint Brush/Palette: +1 hand size.
            "v_paint_brush" | "v_palette" => self.hand_size += 1,
            // +extra discards per round.
            "v_wasteful" | "v_recyclomancy" => {
                self.round_resets_discards += 1;
                self.discards_left += 1;
            }
            "v_blank" => {}
            // +1 joker slot.
            "v_antimatter" => self.joker_slots += 1,
            // ease_ante(-1); blind_ante follows; Hieroglyph -1 hand,
            // Petroglyph -1 discard (both round_resets and live counts).
            "v_hieroglyph" | "v_petroglyph" => {
                self.ante -= 1;
                self.blind_ante -= 1;
                if key == "v_hieroglyph" {
                    self.round_resets_hands -= 1;
                    self.hands_left -= 1;
                } else {
                    self.round_resets_discards -= 1;
                    self.discards_left -= 1;
                }
            }
            // Boss-reroll enablers: state checked in can_reroll_boss.
            "v_directors_cut" | "v_retcon" => {}
            _ => {}
        }
    }

    /// Test/tooling helper mirroring the challenge-run voucher grant
    /// (`Game:start_run`, game.lua:2092-2101: `used_vouchers[id] = true` +
    /// `Card.apply_to_run(nil, center)`): applies a voucher without a shop
    /// offer, cost, or stream use.
    pub fn debug_apply_voucher(&mut self, key: &'static str) {
        assert!(
            items::voucher_by_key(key).is_some(),
            "unknown voucher {key}"
        );
        self.used_vouchers.insert(key);
        self.apply_voucher(key);
    }

    /// Test/tooling helper: create a joker as the debug console would (a
    /// plain Card creation — sort_id, used_jokers, negative slot bump,
    /// passives). Set_ability still rolls 'to_do' for To Do List
    /// (card.lua:311-322); no other stream use.
    pub fn debug_add_joker(&mut self, id: JokerId, edition: Edition) {
        let sort_id = self.next_sort_id();
        self.add_used(id.key());
        let mut state = JokerState::initial(id);
        if id == JokerId::TodoList {
            state.to_do_hand = roll_to_do_hand(&mut self.rng, &self.hands_table, None);
        }
        self.add_joker(OwnedJoker {
            id,
            edition,
            sort_id,
            extra_value: 0,
            debuffed: false,
            flipped: false,
            eternal: false,
            hands_at_create: self.hands_played_total,
            state,
        });
    }

    /// Test/tooling helper: mutable access to an owned joker (stickers,
    /// counters) — no game flow attached.
    pub fn debug_joker_mut(&mut self, idx: usize) -> &mut OwnedJoker {
        &mut self.jokers[idx]
    }

    /// Test/tooling helper: create a held consumable (forced-key
    /// `create_card` — no stream use).
    pub fn debug_add_consumable(&mut self, key: &'static str) {
        assert!(
            items::consumable_by_key(key).is_some(),
            "unknown consumable {key}"
        );
        self.add_consumable(key);
    }

    /// `G.FUNCS.reroll_shop` (button_callbacks.lua:2855-2910).
    pub fn reroll_shop(&mut self) -> Result<(), RunError> {
        if self.state != State::Shop {
            return Err(RunError::WrongState);
        }
        if !self.can_reroll_shop() {
            return Err(RunError::InsufficientFunds);
        }
        if self.reroll_cost > 0 {
            self.dollars -= self.reroll_cost;
        }
        let final_free = self.free_rerolls > 0;
        self.free_rerolls = (self.free_rerolls - 1).max(0);
        self.calculate_reroll_cost(final_free);
        // button_callbacks.lua:2866-2868 — `{reroll_shop = true}` per joker:
        // Flash Card +2 mult (card.lua:2403-2411, blueprint-gated).
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(&self.jokers, i) else {
                continue;
            };
            if self.jokers[t].id == JokerId::Flash && !bp {
                self.jokers[t].state.mult += 2.0; // extra (game.lua:487)
            }
        }
        // Remove every card-slot item (Card:remove clears used_jokers) ...
        let old = std::mem::take(&mut self.shop.as_mut().unwrap().jokers);
        for item in old {
            match item.kind {
                ShopItemKind::Joker(j) => self.remove_used(j.id.key()),
                ShopItemKind::Consumable(c) => self.remove_used(c.key),
                ShopItemKind::PlayingCard(_) => {}
            }
        }
        // ... then refill to joker_max (:2884-2887).
        for _ in 0..self.shop_joker_max {
            let item = self.create_card_for_shop();
            self.shop.as_mut().unwrap().jokers.push(item);
        }
        self.apply_store_joker_modify_tags();
        Ok(())
    }

    /// `G.FUNCS.toggle_shop` (button_callbacks.lua:2481-2511): leave for the
    /// blind select of the (possibly new) ante.
    pub fn leave_shop(&mut self) -> Result<(), RunError> {
        if self.state != State::Shop {
            return Err(RunError::WrongState);
        }
        // button_callbacks.lua:2484-2487 — `{ending_shop = true}` per joker:
        // Perkeo copies a random held consumable as a Negative
        // ('perkeo' stream; card.lua:2412-2426, not blueprint-gated).
        for i in 0..self.jokers.len() {
            let Some((t, _bp)) = resolve_effective(&self.jokers, i) else {
                continue;
            };
            if self.jokers[t].id == JokerId::Perkeo && !self.consumables.is_empty() {
                let mut ids: Vec<u32> = self.consumables.iter().map(|c| c.sort_id).collect();
                ids.sort_unstable();
                let seed = self.rng.pseudoseed("perkeo");
                let pick =
                    ids[(LuaRandom::seeded(seed).random_range(1, ids.len() as i64) - 1) as usize];
                let chosen = *self.consumables.iter().find(|c| c.sort_id == pick).unwrap();
                let sort_id = self.next_sort_id();
                // set_edition(negative) + add_to_deck: +1 consumable slot
                // (card.lua:2418-2420, 630-640).
                self.consumable_slots += 1;
                self.add_used(chosen.key);
                self.consumables.push(OwnedConsumable {
                    key: chosen.key,
                    sort_id,
                    negative: true,
                    extra_value: chosen.extra_value,
                });
            }
        }
        // The shop UIBox removal removes its cards (used_jokers entries
        // clear for unbought items).
        if let Some(shop) = self.shop.take() {
            for item in shop.jokers {
                match item.kind {
                    ShopItemKind::Joker(j) => self.remove_used(j.id.key()),
                    ShopItemKind::Consumable(c) => self.remove_used(c.key),
                    ShopItemKind::PlayingCard(_) => {}
                }
            }
            for v in shop.vouchers {
                self.remove_used(v.key);
            }
        }
        self.enter_blind_select();
        Ok(())
    }

    // ------------------------------------------------------------------
    // Selling
    // ------------------------------------------------------------------

    /// Current sell value of a held joker.
    pub fn joker_sell_value(&self, j: &OwnedJoker) -> i64 {
        items::sell_cost(
            j.id.meta().cost,
            j.edition,
            self.discount_percent,
            j.extra_value,
        )
    }

    /// Current sell value of a held consumable. With Astronomer a Planet's
    /// cost is 0, so its sell value floors at 1 (+ extra_value).
    pub fn consumable_sell_value(&self, c: &OwnedConsumable) -> i64 {
        if self.astronomer_owned()
            && matches!(
                items::consumable_by_key(c.key),
                Some((_, ConsumableSet::Planet))
            )
        {
            return 1 + c.extra_value;
        }
        items::sell_cost(
            items::center_base_cost(c.key),
            Edition::None,
            self.discount_percent,
            c.extra_value,
        )
    }

    fn can_sell_now(&self) -> bool {
        matches!(
            self.state,
            State::Shop
                | State::BlindSelect
                | State::RoundEval
                | State::SelectingHand
                | State::PackOpen
        )
    }

    /// `Card:sell_card` (card.lua:1590-1638) for a joker, plus
    /// `G.FUNCS.sell_card`'s `{selling_card = true}` pass over the other
    /// jokers (button_callbacks.lua:2319-2325).
    pub fn sell_joker(&mut self, idx: usize) -> Result<(), RunError> {
        if !self.can_sell_now() {
            return Err(RunError::WrongState);
        }
        if idx >= self.jokers.len() {
            return Err(RunError::BadSlot(format!("joker {idx}")));
        }
        // The pre-sell pass can end the round (Luchador -> disable_boss ->
        // the halved requirement is already met), and end_round destroys
        // self-consuming jokers, so `idx` may not survive it. Lua removes the
        // card by identity; track the sort_id and re-find it.
        let selling_sort_id = self.jokers[idx].sort_id;
        // card.lua:1599 — `self:calculate_joker{selling_self = true}` runs
        // BEFORE the money and removal; a sold Blueprint/Brainstorm resolves
        // its copy target (Luchador/Diet Cola are not blueprint-gated).
        if let Some((t, bp)) = resolve_effective(&self.jokers, idx) {
            match self.jokers[t].id {
                // card.lua:2355-2360 — Luchador: disable a live boss.
                JokerId::Luchador => {
                    let live_boss = self
                        .active_blind
                        .as_ref()
                        .is_some_and(|b| b.proto.is_boss && !b.disabled);
                    if live_boss {
                        self.disable_boss();
                    }
                }
                // card.lua:2361-2370 — Diet Cola: a free Double Tag.
                JokerId::DietCola => {
                    self.add_tag("tag_double", None);
                }
                // card.lua:2371-2394 — Invisible Joker: after `extra` (2)
                // full rounds, duplicate a random OTHER joker ('invisible'
                // stream). Blueprint-gated.
                JokerId::Invisible if !bp && self.jokers[t].state.extra >= 2.0 => {
                    let me = self.jokers[t].sort_id;
                    let mut others: Vec<u32> = self
                        .jokers
                        .iter()
                        .filter(|j| j.sort_id != me)
                        .map(|j| j.sort_id)
                        .collect();
                    // `#G.jokers.cards <= card_limit` — self still counted.
                    if !others.is_empty() && self.jokers.len() <= self.joker_slots {
                        others.sort_unstable();
                        let seed = self.rng.pseudoseed("invisible");
                        let pick = others[(LuaRandom::seeded(seed)
                            .random_range(1, others.len() as i64)
                            - 1) as usize];
                        let chosen = *self.jokers.iter().find(|j| j.sort_id == pick).unwrap();
                        // copy_card (common_events.lua:2156-2181) with
                        // strip_edition = source is negative: the copy keeps
                        // foil/holo/poly but a negative source copies
                        // edition-less (no extra slot). set_ability of the
                        // copy rolls 'to_do' for a To Do List
                        // (card.lua:311-322) before the state is overwritten.
                        if chosen.id == JokerId::TodoList {
                            let _ = crate::jokers::effects::roll_to_do_hand(
                                &mut self.rng,
                                &self.hands_table,
                                None,
                            );
                        }
                        let sort_id = self.next_sort_id();
                        let mut copy = OwnedJoker {
                            sort_id,
                            edition: if chosen.edition == Edition::Negative {
                                Edition::None
                            } else {
                                chosen.edition
                            },
                            ..chosen
                        };
                        // `if card.ability.invis_rounds then … = 0`
                        // (card.lua:2385) — only Invisible carries it.
                        if copy.id == JokerId::Invisible {
                            copy.state.extra = 0.0;
                        }
                        self.add_used(copy.id.key());
                        self.add_joker(copy);
                    }
                }
                _ => {}
            }
        }
        let Some(idx) = self
            .jokers
            .iter()
            .position(|j| j.sort_id == selling_sort_id)
        else {
            // Already destroyed by the pre-sell pass; nothing left to sell.
            return Ok(());
        };
        let j = self.jokers.remove(idx);
        self.dollars += self.joker_sell_value(&j);
        if j.edition == Edition::Negative {
            self.joker_slots -= 1; // remove_from_deck (card.lua:693-701)
        }
        // remove_from_deck passive effects (card.lua:645-701); a debuffed
        // joker's passives are already off (set_debuff).
        if !j.debuffed {
            self.joker_passives_off(&j);
        }
        self.remove_used(j.id.key());
        // card.lua:1616-1622 — selling a joker during Verdant Leaf disables
        // the boss.
        if self
            .active_blind
            .as_ref()
            .is_some_and(|b| b.proto.name == "Verdant Leaf" && !b.disabled)
        {
            self.disable_boss();
        }
        // button_callbacks.lua:2319-2325 — every OTHER joker gets
        // `{selling_card = true}`: Campfire +0.25 x_mult per card sold
        // (card.lua:2395-2401, blueprint-gated).
        self.joker_selling_card_pass();
        Ok(())
    }

    /// The `{selling_card = true}` pass (button_callbacks.lua:2319-2325) —
    /// jokers and consumables both route through G.FUNCS.sell_card.
    pub(crate) fn joker_selling_card_pass(&mut self) {
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(&self.jokers, i) else {
                continue;
            };
            if self.jokers[t].id == JokerId::Campfire && !bp {
                self.jokers[t].state.x_mult += 0.25; // extra (game.lua:498)
            }
        }
    }

    /// `Card:sell_card` for a held consumable.
    pub fn sell_consumable(&mut self, idx: usize) -> Result<(), RunError> {
        if !self.can_sell_now() {
            return Err(RunError::WrongState);
        }
        if idx >= self.consumables.len() {
            return Err(RunError::BadSlot(format!("consumable {idx}")));
        }
        let c = self.consumables.remove(idx);
        self.dollars += self.consumable_sell_value(&c);
        if c.negative {
            // remove_from_deck (card.lua:687-697).
            self.consumable_slots -= 1;
        }
        self.remove_used(c.key);
        self.joker_selling_card_pass();
        Ok(())
    }

    // ------------------------------------------------------------------
    // Booster packs
    // ------------------------------------------------------------------

    /// `G.FUNCS.use_card` on a shop booster -> `Card:open`
    /// (card.lua:1681-1811).
    pub fn buy_pack(&mut self, slot: usize) -> Result<(), RunError> {
        if self.state != State::Shop {
            return Err(RunError::WrongState);
        }
        let shop = self.shop.as_ref().ok_or(RunError::WrongState)?;
        let offer = *shop
            .packs
            .get(slot)
            .ok_or_else(|| RunError::BadSlot(format!("pack slot {slot}")))?;
        if offer.used {
            return Err(RunError::BadSlot("pack already opened".into()));
        }
        let cost = self.pack_cost(&offer);
        if !self.affordable(cost) {
            return Err(RunError::InsufficientFunds);
        }
        // used_packs[i] = 'USED' (button_callbacks.lua:2243).
        self.shop.as_mut().unwrap().packs[slot].used = true;
        if cost > 0 {
            self.dollars -= cost;
        }
        self.open_pack(offer.key, PackReturn::Shop);
        Ok(())
    }

    /// Open a tag-granted pack from blind select (cost 0, from_tag).
    fn open_pack_key(&mut self, key: &'static str, _from_tag: bool) {
        // The tag creates a Card for the pack itself (tag.lua:210-216).
        let _ = self.next_sort_id();
        self.open_pack(key, PackReturn::BlindSelect);
    }

    /// `Card:open` content generation (card.lua:1697-1758) + the pack-state
    /// entry (`update_arcana_pack` etc. deal a hand for Arcana/Spectral).
    fn open_pack(&mut self, key: &'static str, return_to: PackReturn) {
        let meta = booster_by_key(key).expect("booster key");
        let mut items_v: Vec<PackItem> = Vec::new();
        for i in 0..meta.cards {
            let item = match meta.kind {
                PackKind::Arcana => {
                    // card.lua:1717-1721 — Omen Globe: 20% spectral.
                    if self.used_vouchers.contains("v_omen_globe")
                        && self.rng.random("omen_globe") > 0.8
                    {
                        let k = self.create_consumable_key(ConsumableSet::Spectral, "ar2", true);
                        self.new_pack_consumable(k)
                    } else {
                        let k = self.create_consumable_key(ConsumableSet::Tarot, "ar1", true);
                        self.new_pack_consumable(k)
                    }
                }
                PackKind::Celestial => {
                    // card.lua:1723-1740 — Telescope forces the first card
                    // to the most played visible hand's planet (G.handlist
                    // order, strict > beats ties).
                    let mut forced: Option<&'static str> = None;
                    if self.used_vouchers.contains("v_telescope") && i == 0 {
                        let mut best: Option<HandType> = None;
                        let mut tally = 0u32;
                        for ht in items::HANDLIST {
                            let row = self.hands_table.get(ht);
                            if row.visible && row.played > tally {
                                best = Some(ht);
                                tally = row.played;
                            }
                        }
                        forced = best.map(crate::consumables::planet_for_hand);
                    }
                    match forced {
                        Some(k) => self.new_pack_consumable(k),
                        None => {
                            let k = self.create_consumable_key(ConsumableSet::Planet, "pl1", true);
                            self.new_pack_consumable(k)
                        }
                    }
                }
                PackKind::Spectral => {
                    let k = self.create_consumable_key(ConsumableSet::Spectral, "spe", true);
                    self.new_pack_consumable(k)
                }
                PackKind::Buffoon => {
                    let j = self.create_joker(None, false, "buf", JokerArea::Pack);
                    PackItem::Joker(j)
                }
                PackKind::Standard => self.new_standard_pack_card(),
            };
            items_v.push(item);
        }

        // `{open_booster = true}` per joker (Card:open, card.lua:1796-1798).
        // Common branch: Hallucination (card.lua:2335-2351) — slot gate
        // (counting the creations already buffered) THEN the
        // 'halu<ante>' < normal/2 roll; each success queues a Tarot on the
        // 'Tarothal<ante>' pool. Not blueprint-gated: copies roll too.
        let mut halu_creates = 0usize;
        for i in 0..self.jokers.len() {
            let Some((t, _bp)) = resolve_effective(&self.jokers, i) else {
                continue;
            };
            if self.jokers[t].id == JokerId::Hallucination
                && self.consumables.len() + halu_creates < self.consumable_slots
                && self.rng.random(&format!("halu{}", self.ante)) < self.prob_normal / 2.0
            {
                halu_creates += 1;
            }
        }
        for _ in 0..halu_creates {
            let k = self.create_consumable_key(ConsumableSet::Tarot, "hal", false);
            self.add_consumable(k);
        }

        self.pack = Some(PackState {
            key,
            kind: meta.kind,
            choices_left: meta.choose,
            items: items_v,
            return_to,
            hand_dealt: false,
        });
        self.state = State::PackOpen;

        // Game:update_arcana_pack / update_spectral_pack
        // (game.lua:3374/3426) — draw_from_deck_to_hand for targeting. No
        // blind is live, so no flip rolls; the hand sorts 'desc' as always.
        if matches!(meta.kind, PackKind::Arcana | PackKind::Spectral) {
            let n = self
                .deck
                .len()
                .min((self.hand_size - self.hand.len() as i64).max(0) as usize);
            for _ in 0..n {
                let card = self.deck.pop().expect("deck underflow");
                self.hand.push(card);
            }
            self.sort_hand_desc();
            self.pack.as_mut().unwrap().hand_dealt = true;
        }
    }

    fn new_pack_consumable(&mut self, key: &'static str) -> PackItem {
        let sort_id = self.next_sort_id();
        self.add_used(key);
        PackItem::Consumable(OwnedConsumable {
            key,
            sort_id,
            negative: false,
            extra_value: 0,
        })
    }

    /// One Standard-pack playing card (card.lua:1742-1756).
    fn new_standard_pack_card(&mut self) -> PackItem {
        // 'stdset<ante>' > 0.6 -> Enhanced, else Base.
        let enhanced = self.rng.random(&format!("stdset{}", self.ante)) > 0.6;
        let enhancement = if enhanced {
            let spec = PoolSpec::Enhanced { append: "sta" };
            let key = items::roll_from_pool(&mut self.rng, &spec, &pool_args!(self));
            enhancement_from_key(key)
        } else {
            Enhancement::None
        };
        let idx = items::element_index(&mut self.rng, &format!("frontsta{}", self.ante), 52);
        let (suit, rank) = items::card_front_from_index(idx);
        let sort_id = self.next_sort_id();
        let mut card = Card::new(rank, suit, sort_id);
        card.enhancement = enhancement;
        // edition_rate = 2, no negative (card.lua:1745-1746).
        let ed = items::poll_edition(
            &mut self.rng,
            &format!("standard_edition{}", self.ante),
            2.0,
            true,
            false,
            self.edition_rate,
        );
        card.edition = ed;
        // seal_rate = 10: 'stdseal' > 0.8, then 'stdsealtype'
        // (card.lua:1747-1756).
        let seal_poll = self.rng.random(&format!("stdseal{}", self.ante));
        if seal_poll > 1.0 - 0.02 * 10.0 {
            let seal_type = self.rng.random(&format!("stdsealtype{}", self.ante));
            card.seal = if seal_type > 0.75 {
                crate::cards::Seal::Red
            } else if seal_type > 0.5 {
                crate::cards::Seal::Blue
            } else if seal_type > 0.25 {
                crate::cards::Seal::Gold
            } else {
                crate::cards::Seal::Purple
            };
        }
        PackItem::PlayingCard(card)
    }

    /// Can pack item `i` be picked right now? Consumables gate on
    /// `can_use_consumeable` (UI_definitions.lua:265), jokers on slot space
    /// (`can_select_card`, button_callbacks.lua:2112-2119).
    pub fn can_pick_pack_item(&self, idx: usize) -> bool {
        let Some(pack) = self.pack.as_ref() else {
            return false;
        };
        match pack.items.get(idx) {
            Some(PackItem::Joker(j)) => {
                let extra = if j.edition == Edition::Negative { 1 } else { 0 };
                self.jokers.len() < self.joker_slots + extra
            }
            Some(PackItem::Consumable(c)) => self.consumable_has_any_use(c.key),
            Some(PackItem::PlayingCard(_)) => true,
            None => false,
        }
    }

    /// `G.FUNCS.use_card` on a pack card: jokers/playing cards are added,
    /// consumables are used on the spot (targets = hand indices).
    pub fn pick_pack_item(&mut self, idx: usize, targets: &[usize]) -> Result<(), RunError> {
        if self.state != State::PackOpen {
            return Err(RunError::WrongState);
        }
        let pack = self.pack.as_ref().ok_or(RunError::WrongState)?;
        let item = pack
            .items
            .get(idx)
            .ok_or_else(|| RunError::BadSlot(format!("pack item {idx}")))?
            .clone();
        match &item {
            PackItem::Joker(j) => {
                let extra = if j.edition == Edition::Negative { 1 } else { 0 };
                if self.jokers.len() >= self.joker_slots + extra {
                    return Err(RunError::NoSpace);
                }
            }
            PackItem::Consumable(c) => {
                self.consumable_can_use(c.key, targets)
                    .map_err(RunError::CannotUse)?;
            }
            PackItem::PlayingCard(_) => {}
        }
        self.pack.as_mut().unwrap().items.remove(idx);
        match item {
            PackItem::Joker(j) => self.add_joker(j),
            PackItem::Consumable(c) => {
                self.use_consumable_key(c.key, targets)?;
                self.remove_used(c.key);
            }
            PackItem::PlayingCard(card) => {
                // use_card Enhanced/Default branch (button_callbacks.lua:
                // 2224-2235): straight into the deck;
                // playing_card_joker_effects({card}) — Hologram.
                self.playing_card_counter += 1;
                self.deck.insert(0, card);
                self.joker_playing_cards_added(1);
            }
        }
        // button_callbacks.lua:2272-2283 — decrement pack_choices or end.
        let pack = self.pack.as_mut().unwrap();
        if pack.choices_left > 1 && !pack.items.is_empty() {
            pack.choices_left -= 1;
        } else {
            self.finish_pack(false);
        }
        Ok(())
    }

    /// `G.FUNCS.skip_booster` (button_callbacks.lua:2558-2563).
    pub fn skip_pack(&mut self) -> Result<(), RunError> {
        if self.state != State::PackOpen {
            return Err(RunError::WrongState);
        }
        // `{skipping_booster = true}` per joker (button_callbacks.lua:2560).
        // Common branch: Red Card +3 mult (card.lua:2442-2457), gated on
        // `not context.blueprint` — copies do nothing.
        for i in 0..self.jokers.len() {
            let Some((t, bp)) = resolve_effective(&self.jokers, i) else {
                continue;
            };
            if self.jokers[t].id == JokerId::RedCard && !bp {
                self.jokers[t].state.mult += 3.0; // extra (game.lua:459)
            }
        }
        self.finish_pack(true);
        Ok(())
    }

    /// `G.FUNCS.end_consumeable` (button_callbacks.lua:2565-2625): hand back
    /// to the deck, state restored, and — from blind select — another
    /// new_blind_choice tag pass.
    fn finish_pack(&mut self, _skipped: bool) {
        let pack = self.pack.take().expect("no pack");
        // Unpicked pack cards are removed (used_jokers entries clear).
        for item in &pack.items {
            match item {
                PackItem::Joker(j) => self.remove_used(j.id.key()),
                PackItem::Consumable(c) => self.remove_used(c.key),
                PackItem::PlayingCard(_) => {}
            }
        }
        if pack.hand_dealt {
            // draw_from_hand_to_deck (state_events.lua:1121-1126): each hand
            // card emplaces at deck index 1 (the bottom), front-to-back.
            let cards: Vec<Card> = self.hand.drain(..).collect();
            for card in cards {
                self.deck.insert(0, card);
            }
        }
        match pack.return_to {
            PackReturn::Shop => self.state = State::Shop,
            PackReturn::BlindSelect => {
                self.state = State::BlindSelect;
                self.apply_new_blind_choice_tags();
            }
        }
    }
}

pub(crate) use crate::items::enhancement_from_key;

/// Suit/rank helpers for tarot conversions.
pub(crate) fn suit_from_char(c: char) -> Suit {
    match c {
        'S' => Suit::Spades,
        'H' => Suit::Hearts,
        'D' => Suit::Diamonds,
        _ => Suit::Clubs,
    }
}

pub(crate) fn rank_from_char(c: char) -> Rank {
    Rank(match c {
        'A' => 14,
        'K' => 13,
        'Q' => 12,
        'J' => 11,
        'T' => 10,
        n => n.to_digit(10).unwrap() as u8,
    })
}
