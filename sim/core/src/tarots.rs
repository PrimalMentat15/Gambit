//! Consumable usage: `Card:use_consumeable` (card.lua:1091-1521) and
//! `Card:can_use_consumeable` (card.lua:1523-1580) for Tarots, Planets and
//! Spectrals. Targets are hand indices into the current (display-ordered)
//! hand; the Lua `G.hand.highlighted` order is the click order, which only
//! Death observes (rightmost = highest display index).
//!
//! # RNG stream keys
//!   'wheel_of_fortune'   chance roll (<1/4), then the joker pick, then the
//!                        guaranteed-edition poll — three draws on one
//!                        stream when it hits (card.lua:1471-1494)
//!   'aura'               guaranteed-edition poll (card.lua:1196)
//!   'sigil' / 'ouija'    suit / rank pick (card.lua:1235/1250)
//!   'random_destroy'     Familiar/Grim/Incantation destroyed card
//!   'familiar_create'    rank then suit per created card (2 draws)
//!   'grim_create'        suit per created card
//!   'incantation_create' rank then suit per created card
//!   'spe_card'           enhancement per created card (7-entry pool,
//!                        m_stone excluded)
//!   'immolate'           one pseudoshuffle of the hand copy
//!   'ankh_choice'        Ankh's joker pick (over ALL jokers)
//!   'ectoplasm' / 'hex'  joker pick (editionless jokers)
//!   'Tarotemp<ante>'     The Emperor's tarots (+resamples; no soul poll)
//!   'Planetpri<ante>'    High Priestess' planets (+resamples; no soul poll)
//!   'rarity<ante>jud' / 'Joker<r>jud<ante>' / 'edijud<ante>'   Judgement
//!   'Joker4' / 'edisou<ante>'                                  The Soul
//!   'rarity n/a (0.99 forced) / 'Joker3wra<ante>' / 'ediwra<ante>'  Wraith
//!
//! The Fool copies `G.GAME.last_tarot_planet`, which is only updated AFTER
//! the effect resolves (the nested-event trick in set_consumeable_usage,
//! misc_functions.lua:1211-1226) — so the Fool never copies itself.

use crate::cards::{Card, Edition, Enhancement, HandType, Seal};
use crate::deck::pseudoshuffle;
use crate::items::{self, consumable_by_key, ConsumableSet};
use crate::rng::LuaRandom;
use crate::run::{Run, RunError, State};
use crate::shop::{enhancement_from_key, rank_from_char, suit_from_char, JokerArea, OwnedJoker};

impl Run {
    /// Does the current state admit hand-targeted consumables?
    /// card.lua:1565: SELECTING_HAND or the Tarot/Spectral/Planet pack states.
    fn targeting_state(&self) -> bool {
        match self.state {
            State::SelectingHand => true,
            State::PackOpen => matches!(
                self.pack.as_ref().map(|p| p.kind),
                Some(items::PackKind::Arcana)
                    | Some(items::PackKind::Spectral)
                    | Some(items::PackKind::Celestial)
            ),
            _ => false,
        }
    }

    fn editionless_jokers(&self) -> Vec<u32> {
        self.jokers
            .iter()
            .filter(|j| j.edition == Edition::None)
            .map(|j| j.sort_id)
            .collect()
    }

    /// `Card:can_use_consumeable` (card.lua:1523-1580) + `Card:check_use`
    /// (Ankh space, card.lua:1581-1588), for a card with chosen targets.
    pub fn consumable_can_use(&self, key: &str, targets: &[usize]) -> Result<(), String> {
        // The outer state gate: not HAND_PLAYED/DRAW_TO_HAND/PLAY_TAROT —
        // all our discrete states pass except terminal ones.
        if matches!(self.state, State::GameOver | State::Won) {
            return Err("terminal state".into());
        }
        let mut sorted = targets.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != targets.len() || sorted.iter().any(|&i| i >= self.hand.len()) {
            return Err("bad targets".into());
        }
        let (meta, set) = consumable_by_key(key).ok_or("unknown consumable")?;

        // Untargeted, any state (card.lua:1530-1532).
        if matches!(
            key,
            "c_hermit" | "c_temperance" | "c_black_hole" | "c_high_priestess" | "c_emperor"
        ) || set == ConsumableSet::Planet
        {
            // Emperor/High Priestess additionally need a free slot unless
            // used from the consumable area (card.lua:1551-1553) — held use
            // always passes; shop buy-and-use / pack use need space.
            if matches!(key, "c_high_priestess" | "c_emperor")
                && !self.consumables.iter().any(|c| c.key == key)
                && self.consumables.len() >= self.consumable_slots
            {
                return Err("no consumable space".into());
            }
            return Ok(());
        }
        match key {
            "c_wheel_of_fortune" => {
                if self.editionless_jokers().is_empty() {
                    return Err("no editionless joker".into());
                }
            }
            "c_ankh" => {
                if self.jokers.is_empty() || self.joker_slots <= 1 {
                    return Err("needs a joker and >1 slots".into());
                }
                // check_use: blocked at full joker slots (card.lua:1582-1587).
                if self.jokers.len() >= self.joker_slots {
                    return Err("joker slots full".into());
                }
            }
            "c_aura" => {
                if !(self.targeting_state()
                    && targets.len() == 1
                    && self.hand[targets[0]].edition == Edition::None)
                {
                    return Err("needs 1 edition-less highlighted card".into());
                }
            }
            "c_ectoplasm" | "c_hex" => {
                if self.editionless_jokers().is_empty() {
                    return Err("no editionless joker".into());
                }
            }
            "c_fool" => {
                if !self.consumables.iter().any(|c| c.key == "c_fool")
                    && self.consumables.len() >= self.consumable_slots
                {
                    return Err("no consumable space".into());
                }
                match self.last_tarot_planet {
                    Some(k) if k != "c_fool" => {}
                    _ => return Err("nothing to copy".into()),
                }
            }
            "c_judgement" | "c_soul" | "c_wraith" => {
                if self.jokers.len() >= self.joker_slots {
                    return Err("joker slots full".into());
                }
            }
            _ => {
                // Targeted tarots/spectrals (max_highlighted) and the
                // whole-hand spectrals (card.lua:1565-1578).
                if meta.max_highlighted > 0 {
                    if !self.targeting_state() {
                        return Err("no hand to target".into());
                    }
                    // mod_num = min(5, max_highlighted) (card.lua:4155).
                    let mod_num = meta.max_highlighted.min(5) as usize;
                    let min = meta.min_highlighted.max(1) as usize;
                    if targets.len() > mod_num || targets.len() < min {
                        return Err(format!(
                            "needs {}..={} targets, got {}",
                            min,
                            mod_num,
                            targets.len()
                        ));
                    }
                } else if matches!(
                    key,
                    "c_familiar"
                        | "c_grim"
                        | "c_incantation"
                        | "c_immolate"
                        | "c_sigil"
                        | "c_ouija"
                ) {
                    if !(self.targeting_state() && self.hand.len() > 1) {
                        return Err("needs a hand of 2+ cards".into());
                    }
                } else {
                    return Err(format!("{key} has no usable effect here"));
                }
            }
        }
        Ok(())
    }

    /// Is there ANY legal way to use this consumable right now? (Action-list
    /// helper; target enumeration is the caller's business.)
    pub fn consumable_has_any_use(&self, key: &str) -> bool {
        if self.consumable_can_use(key, &[]).is_ok() {
            return true;
        }
        let Some((meta, _)) = consumable_by_key(key) else {
            return false;
        };
        if key == "c_aura" {
            return self.targeting_state() && self.hand.iter().any(|c| c.edition == Edition::None);
        }
        if meta.max_highlighted > 0 {
            let min = meta.min_highlighted.max(1) as usize;
            return self.targeting_state() && self.hand.len() >= min;
        }
        false
    }

    /// Use the held consumable in slot `idx` on `targets` (hand indices).
    pub fn use_consumable(&mut self, idx: usize, targets: &[usize]) -> Result<(), RunError> {
        let Some(c) = self.consumables.get(idx).copied() else {
            return Err(RunError::BadSlot(format!("consumable {idx}")));
        };
        self.consumable_can_use(c.key, targets)
            .map_err(RunError::CannotUse)?;
        // use_card removes the card from its area before use_consumeable
        // (button_callbacks.lua:2210) — Emperor/Fool slot math depends on it.
        // A Negative consumable returns its bonus slot (remove_from_deck,
        // card.lua:687-697).
        let neg = self.consumables[idx].negative;
        self.consumables.remove(idx);
        if neg {
            self.consumable_slots -= 1;
        }
        self.use_consumable_key(c.key, targets)?;
        self.remove_used(c.key);
        Ok(())
    }

    /// `Card:use_consumeable` (card.lua:1091-1521). The caller has validated
    /// with `consumable_can_use` and removed the card from its area.
    pub(crate) fn use_consumable_key(
        &mut self,
        key: &'static str,
        targets: &[usize],
    ) -> Result<(), RunError> {
        let (_meta, set) = consumable_by_key(key).expect("consumable key");
        // The Hanged Man's highlighted-glass count, captured before the
        // destruction for Glass Joker's using_consumeable branch
        // (card.lua:2709-2721 reads G.hand.highlighted).
        let mut hanged_glass = 0usize;
        // set_consumeable_usage (card.lua:1093): usage counts now,
        // last_tarot_planet only after the effect (see module docs).
        // consumeable_usage_total.tarot (misc_functions.lua:1196-1206) feeds
        // Fortune Teller.
        *self.consumable_usage.entry(key).or_insert(0) += 1;
        if set == ConsumableSet::Tarot {
            self.tarots_used += 1;
        }

        match key {
            // -------------------- enhancers / converters --------------------
            "c_magician" => self.enhance_targets(targets, "m_lucky"),
            "c_empress" => self.enhance_targets(targets, "m_mult"),
            "c_heirophant" => self.enhance_targets(targets, "m_bonus"),
            "c_lovers" => self.enhance_targets(targets, "m_wild"),
            "c_chariot" => self.enhance_targets(targets, "m_steel"),
            "c_justice" => self.enhance_targets(targets, "m_glass"),
            "c_devil" => self.enhance_targets(targets, "m_gold"),
            "c_tower" => self.enhance_targets(targets, "m_stone"),
            // card.lua:1112-1120 — Death: the rightmost highlighted card
            // (highest screen x = highest display index) is copied onto the
            // others (copy_card keeps only the target's sort_id; the ability
            // table — perma_bonus included — copies over).
            "c_death" => {
                let right = *targets.iter().max().expect("targets");
                let src = self.hand[right];
                for &i in targets {
                    if i != right {
                        let dst = &mut self.hand[i];
                        dst.rank = src.rank;
                        dst.suit = src.suit;
                        dst.enhancement = src.enhancement;
                        dst.edition = src.edition;
                        dst.seal = src.seal;
                        dst.perma_bonus = src.perma_bonus;
                    }
                }
            }
            // card.lua:1121-1137 — Strength: rank +1, Ace wraps to 2.
            "c_strength" => {
                for &i in targets {
                    let id = self.hand[i].rank.id();
                    let new = if id == 14 { 2 } else { (id + 1).min(14) };
                    self.hand[i].rank = crate::cards::Rank(new);
                }
            }
            // card.lua:1138-1141 — suit conversions.
            "c_star" => self.convert_suits(targets, crate::cards::Suit::Diamonds),
            "c_moon" => self.convert_suits(targets, crate::cards::Suit::Clubs),
            "c_sun" => self.convert_suits(targets, crate::cards::Suit::Hearts),
            "c_world" => self.convert_suits(targets, crate::cards::Suit::Spades),

            // ------------------------- seals / editions ---------------------
            // card.lua:1179-1194.
            "c_talisman" => self.hand[targets[0]].seal = Seal::Gold,
            "c_deja_vu" => self.hand[targets[0]].seal = Seal::Red,
            "c_trance" => self.hand[targets[0]].seal = Seal::Blue,
            "c_medium" => self.hand[targets[0]].seal = Seal::Purple,
            // card.lua:1195-1204 — Aura: guaranteed non-negative edition.
            "c_aura" => {
                let ed =
                    items::poll_edition(&mut self.rng, "aura", 1.0, true, true, self.edition_rate);
                self.hand[targets[0]].edition = ed;
            }

            // --------------------------- hand-wide ---------------------------
            // card.lua:1226-1245 — Sigil: every hand card to one random suit.
            "c_sigil" => {
                let suits = ['S', 'H', 'D', 'C'];
                let i = items::element_index(&mut self.rng, "sigil", suits.len());
                let suit = suit_from_char(suits[i]);
                for c in self.hand.iter_mut() {
                    c.suit = suit;
                }
            }
            // card.lua:1246-1259 — Ouija: one random rank; -1 hand size.
            "c_ouija" => {
                let ranks = [
                    '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
                ];
                let i = items::element_index(&mut self.rng, "ouija", ranks.len());
                let rank = rank_from_char(ranks[i]);
                for c in self.hand.iter_mut() {
                    c.rank = rank;
                }
                self.hand_size -= 1;
            }
            // card.lua:1211-1225 — Cryptid: 2 copies of the target into hand.
            "c_cryptid" => {
                let src = self.hand[targets[0]];
                for _ in 0..2 {
                    self.playing_card_counter += 1;
                    let sort_id = self.next_sort_id();
                    let mut copy = src;
                    copy.sort_id = sort_id;
                    self.hand.push(copy);
                }
                // playing_card_joker_effects(new_cards) (card.lua:1217).
                self.joker_playing_cards_added(2);
            }

            // -------------------------- destroyers ---------------------------
            // card.lua:1279-1301 — Hanged Man: destroy the highlighted cards.
            "c_hanged_man" => {
                hanged_glass = targets
                    .iter()
                    .filter(|&&i| self.hand[i].enhancement == Enhancement::Glass)
                    .count();
                let mut idx: Vec<usize> = targets.to_vec();
                idx.sort_unstable();
                for &i in idx.iter().rev() {
                    let card = self.hand.remove(i);
                    self.destroyed_cards.push(card);
                }
            }
            // card.lua:1302-1355 — Familiar/Grim/Incantation: destroy 1
            // random hand card, create enhanced face/Ace/numbered cards.
            "c_familiar" => self.destroy_and_create(3, CreateKind::Familiar),
            "c_grim" => self.destroy_and_create(2, CreateKind::Grim),
            "c_incantation" => self.destroy_and_create(4, CreateKind::Incantation),
            // card.lua:1356-1383 — Immolate: destroy 5 random, +$20.
            "c_immolate" => {
                let mut temp: Vec<Card> = self.hand.clone();
                let seed = self.rng.pseudoseed("immolate");
                pseudoshuffle(&mut temp, seed);
                let n = 5.min(temp.len());
                for victim in temp.iter().take(n) {
                    if let Some(pos) = self.hand.iter().position(|c| c.sort_id == victim.sort_id) {
                        let card = self.hand.remove(pos);
                        self.destroyed_cards.push(card);
                    }
                }
                self.dollars += 20;
            }

            // ---------------------------- economy ----------------------------
            // card.lua:1401-1410 — Hermit: double money, max +$20.
            "c_hermit" => self.dollars += self.dollars.clamp(0, 20),
            // card.lua:1411-1420 — Temperance: joker sell values, max $50.
            "c_temperance" => {
                let total: i64 = self.jokers.iter().map(|j| self.joker_sell_value(j)).sum();
                self.dollars += total.min(50);
            }

            // ------------------------ consumable makers -----------------------
            // card.lua:1387-1400 — The Fool: copy of the last Tarot/Planet.
            "c_fool" => {
                if self.consumables.len() < self.consumable_slots {
                    let copy = self.last_tarot_planet.expect("checked in can_use");
                    self.add_consumable(copy);
                }
            }
            // card.lua:1421-1435 — Emperor/High Priestess: up to 2 new cards.
            "c_emperor" | "c_high_priestess" => {
                let (cset, append) = if key == "c_emperor" {
                    (ConsumableSet::Tarot, "emp")
                } else {
                    (ConsumableSet::Planet, "pri")
                };
                let n = 2i64.min(self.consumable_slots as i64 - self.consumables.len() as i64);
                for _ in 0..n.max(0) {
                    if self.consumables.len() < self.consumable_slots {
                        let k = self.create_consumable_key(cset, append, false);
                        self.add_consumable(k);
                    }
                }
            }

            // -------------------------- joker makers --------------------------
            // card.lua:1436-1449 — Judgement / The Soul.
            "c_judgement" => {
                let j = self.create_joker(None, false, "jud", JokerArea::Direct);
                self.add_joker(j);
            }
            "c_soul" => {
                let j = self.create_joker(None, true, "sou", JokerArea::Direct);
                self.add_joker(j);
            }
            // card.lua:1479-1493 (creation) — Wraith: rare joker, money to 0.
            "c_wraith" => {
                let j = self.create_joker(Some(0.99), false, "wra", JokerArea::Direct);
                self.add_joker(j);
                self.dollars = 0;
            }
            // card.lua:1450-1478 — Ankh: copy one random joker (negative
            // stripped), destroy the other non-eternal jokers.
            "c_ankh" => {
                let mut ids: Vec<u32> = self.jokers.iter().map(|j| j.sort_id).collect();
                ids.sort_unstable();
                let seed = self.rng.pseudoseed("ankh_choice");
                let pick =
                    ids[(LuaRandom::seeded(seed).random_range(1, ids.len() as i64) - 1) as usize];
                let chosen = *self.jokers.iter().find(|j| j.sort_id == pick).unwrap();
                // Destroy the others (none are eternal on White Stake):
                // Card:remove strips passives/negative slots/used_jokers.
                while let Some(i) = self.jokers.iter().position(|j| j.sort_id != pick) {
                    self.remove_joker_at(i);
                }
                // copy_card (card.lua:1459): the copy's Card() constructor
                // runs set_ability — a To Do List copy consumes one 'to_do'
                // draw (card.lua:311-322) — before `ability` is overwritten
                // with the source's table (state carries over).
                if chosen.id == crate::items::JokerId::TodoList {
                    let _ = crate::jokers::effects::roll_to_do_hand(
                        &mut self.rng,
                        &self.hands_table,
                        None,
                    );
                }
                let sort_id = self.next_sort_id();
                // copy_card copies the whole ability table — extra_value,
                // eternal, hands_played_at_create and the counters ride
                // along; only a Negative edition is stripped.
                let copy = OwnedJoker {
                    sort_id,
                    edition: if chosen.edition == Edition::Negative {
                        Edition::None
                    } else {
                        chosen.edition
                    },
                    debuffed: false,
                    flipped: false,
                    ..chosen
                };
                self.add_used(copy.id.key());
                // add_to_deck + emplace (passives apply; edition already
                // stripped of negative).
                self.add_joker(copy);
            }

            // ----------------------- joker edition givers ---------------------
            // card.lua:1466-1520 — Wheel of Fortune / Ectoplasm / Hex.
            "c_wheel_of_fortune" => {
                // normal/4 chance (Oops! All 6s doubles it).
                if self.rng.random("wheel_of_fortune") < self.prob_normal / 4.0 {
                    let mut pool = self.editionless_jokers();
                    pool.sort_unstable();
                    let i = items::element_index(&mut self.rng, "wheel_of_fortune", pool.len());
                    let sid = pool[i];
                    let ed = items::poll_edition(
                        &mut self.rng,
                        "wheel_of_fortune",
                        1.0,
                        true,
                        true,
                        self.edition_rate,
                    );
                    if let Some(j) = self.jokers.iter_mut().find(|j| j.sort_id == sid) {
                        j.edition = ed;
                    }
                }
                // else: "Nope!" — no further stream use.
            }
            "c_ectoplasm" => {
                let mut pool = self.editionless_jokers();
                pool.sort_unstable();
                let i = items::element_index(&mut self.rng, "ectoplasm", pool.len());
                let sid = pool[i];
                if let Some(j) = self.jokers.iter_mut().find(|j| j.sort_id == sid) {
                    j.edition = Edition::Negative;
                }
                // set_edition on an added card raises the slot count
                // (card.lua:406-414).
                self.joker_slots += 1;
                // card.lua:1499-1502 — growing hand-size penalty.
                self.hand_size -= self.ecto_minus;
                self.ecto_minus += 1;
            }
            "c_hex" => {
                let mut pool = self.editionless_jokers();
                pool.sort_unstable();
                let i = items::element_index(&mut self.rng, "hex", pool.len());
                let sid = pool[i];
                if let Some(j) = self.jokers.iter_mut().find(|j| j.sort_id == sid) {
                    j.edition = Edition::Polychrome;
                }
                let removed: Vec<OwnedJoker> = self
                    .jokers
                    .iter()
                    .copied()
                    .filter(|j| j.sort_id != sid)
                    .collect();
                self.jokers.retain(|j| j.sort_id == sid);
                for j in removed {
                    if j.edition == Edition::Negative {
                        self.joker_slots -= 1;
                    }
                    self.remove_used(j.id.key());
                }
            }

            // ----------------------------- levels -----------------------------
            // card.lua:1153-1177 — Black Hole: every hand +1 level.
            "c_black_hole" => {
                for ht in HandType::ALL {
                    self.hands_table.level_up(ht, 1);
                }
            }
            // Planets: level the configured hand (card.lua:1260-1264).
            _ if set == ConsumableSet::Planet => {
                let ht = items::hand_for_planet(key).expect("planet key");
                self.hands_table.level_up(ht, 1);
            }
            _ => {
                return Err(RunError::CannotUse(format!("{key} not implemented")));
            }
        }

        // misc_functions.lua:1211-1226 — last_tarot_planet updates after the
        // effect for Tarots and Planets.
        if matches!(set, ConsumableSet::Tarot | ConsumableSet::Planet) {
            self.last_tarot_planet = Some(key);
        }
        // button_callbacks.lua:2218-2221 — `{using_consumeable = true}` per
        // joker, after the effect (Constellation / Glass Joker).
        self.joker_consumable_used(key, hanged_glass);
        Ok(())
    }

    fn enhance_targets(&mut self, targets: &[usize], center: &str) {
        let e = enhancement_from_key(center);
        for &i in targets {
            self.hand[i].enhancement = e;
        }
    }

    fn convert_suits(&mut self, targets: &[usize], suit: crate::cards::Suit) {
        for &i in targets {
            self.hand[i].suit = suit;
        }
    }

    /// Familiar/Grim/Incantation (card.lua:1302-1355): destroy one random
    /// hand card ('random_destroy'), then create `extra` playing cards into
    /// the hand with per-card rank/suit rolls and a 'spe_card' enhancement
    /// (m_stone excluded from the pool).
    fn destroy_and_create(&mut self, extra: usize, kind: CreateKind) {
        // pseudorandom_element over Card tables sorts by sort_id.
        let mut ids: Vec<u32> = self.hand.iter().map(|c| c.sort_id).collect();
        ids.sort_unstable();
        let seed = self.rng.pseudoseed("random_destroy");
        let sid = ids[(LuaRandom::seeded(seed).random_range(1, ids.len() as i64) - 1) as usize];
        if let Some(pos) = self.hand.iter().position(|c| c.sort_id == sid) {
            let card = self.hand.remove(pos);
            self.destroyed_cards.push(card);
        }
        // cen_pool: P_CENTER_POOLS.Enhanced minus m_stone (7 entries, pool
        // order); centers carry no sort_id, so element order is array order.
        let cen_pool: Vec<&'static str> = items::ENHANCED_POOL
            .iter()
            .copied()
            .filter(|&k| k != "m_stone")
            .collect();
        for _ in 0..extra {
            let (rank_c, suit_c) = match kind {
                CreateKind::Familiar => {
                    let ranks = ['J', 'Q', 'K'];
                    let r = ranks[items::element_index(&mut self.rng, "familiar_create", 3)];
                    let suits = ['S', 'H', 'D', 'C'];
                    let s = suits[items::element_index(&mut self.rng, "familiar_create", 4)];
                    (r, s)
                }
                CreateKind::Grim => {
                    let suits = ['S', 'H', 'D', 'C'];
                    let s = suits[items::element_index(&mut self.rng, "grim_create", 4)];
                    ('A', s)
                }
                CreateKind::Incantation => {
                    let ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T'];
                    let r = ranks[items::element_index(&mut self.rng, "incantation_create", 9)];
                    let suits = ['S', 'H', 'D', 'C'];
                    let s = suits[items::element_index(&mut self.rng, "incantation_create", 4)];
                    (r, s)
                }
            };
            let cen = cen_pool[items::element_index(&mut self.rng, "spe_card", cen_pool.len())];
            // create_playing_card (common_events.lua:1927-1942).
            self.playing_card_counter += 1;
            let sort_id = self.next_sort_id();
            let mut card = Card::new(rank_from_char(rank_c), suit_from_char(suit_c), sort_id);
            card.enhancement = enhancement_from_key(cen);
            self.hand.push(card);
        }
        // playing_card_joker_effects(cards) (card.lua:1338) — Hologram.
        self.joker_playing_cards_added(extra);
    }
}

#[derive(Debug, Clone, Copy)]
enum CreateKind {
    Familiar,
    Grim,
    Incantation,
}
