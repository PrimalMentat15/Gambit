//! The full scoring pipeline of `G.FUNCS.evaluate_play`
//! (functions/state_events.lua:571-1086): enhancements, editions, seals,
//! held-in-hand effects, boss-blind hand modification, and the joker hook
//! points for P3b.
//!
//! # Order of operations (state_events.lua line refs)
//!  1. `get_poker_hand_info` → hand type + scoring group (:572)
//!  2. hands\[text\].played/played_this_round += 1, last_hand_played (:574-577)
//!  3. append played Stone cards not in the scoring set, re-sort by play
//!     order (:580-600)
//!  4. `Blind:debuff_hand` (:614) — Psychic/Eye/Mouth debuff; The Arm levels
//!     the hand down and The Ox takes all money here (side effects even
//!     though those two never debuff)
//!  5. debuffed hand → chips = mult = 0, `debuffed_hand` joker hook
//!     (:997-1027), skip to 15
//!  6. before-hand joker hook (:628-638)
//!  7. mult/chips (re)read from G.GAME.hands\[text\] (:640-641) — this is
//!     where the level-aware base lands, after The Arm/joker level changes
//!  8. `Blind:modify_hand` — The Flint halves both (:645-646)
//!  9. per scoring card, in play order (:648-780): skip debuffed cards
//!     (blind.triggered = true); otherwise red-seal/joker repetitions, and
//!     per repetition: chip bonus → mult (Lucky roll) → x_mult (Glass) →
//!     $ (Gold seal, Lucky $ roll) → per-card joker hook → edition
//!     (Foil/Holo/Polychrome), applied in exactly that field order
//! 10. per held card, in hand order (:784-872): held effects (Steel x1.5)
//!     with red-seal/joker retriggers (red seal only retriggers if the card
//!     actually has a held effect)
//! 11. joker area phase (:877-944): joker editions + main effects (hook)
//! 12. deck-back `final_scoring_step` (:946-948) — Plasma only, no-op for
//!     the Red Deck
//! 13. glass break rolls per scoring card in order (:950-973), after the
//!     joker `destroying_card` hook; debuffed Glass neither scores nor rolls
//! 14. `remove_playing_cards` joker hook (:974-976)
//! 15. score = floor(hand_chips * mult) added to G.GAME.chips (:1049)
//! 16. after-hand joker hook (:1068-1075) — runs for debuffed hands too
//!
//! `mod_chips`/`mod_mult` are identity outside challenges
//! (misc_functions.lua:684-693). All chip/mult math is Lua-number f64.

use crate::blinds::ActiveBlind;
use crate::cards::{Card, Enhancement, HandType, Seal};
use crate::handeval::{get_poker_hand_info, EvalMods, PokerHands};
use crate::rng::RngState;

/// Per-hand-type run state: the `G.GAME.hands[...]` row (game.lua:2001-2014).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HandTableRow {
    pub level: u32,
    pub chips: f64,
    pub mult: f64,
    pub played: u32,
    pub played_this_round: u32,
    /// `hands[...].visible` — the secret hands (Flush Five/Flush House/Five
    /// of a Kind) start hidden and reveal when first played
    /// (state_events.lua:578). Gates the Orbital-tag hand pool.
    pub visible: bool,
}

/// `G.GAME.hands`: one row per hand type, all at level 1 initially.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HandsTable {
    rows: [HandTableRow; 12],
}

impl Default for HandsTable {
    fn default() -> Self {
        Self::new()
    }
}

impl HandsTable {
    pub fn new() -> Self {
        let rows = HandType::ALL.map(|ht| {
            let v = ht.base_values();
            HandTableRow {
                level: 1,
                chips: v.base_chips as f64,
                mult: v.base_mult as f64,
                played: 0,
                played_this_round: 0,
                // game.lua:2001-2003 — only the secret hands start hidden.
                visible: !matches!(
                    ht,
                    HandType::FlushFive | HandType::FlushHouse | HandType::FiveOfAKind
                ),
            }
        });
        HandsTable { rows }
    }

    fn idx(ht: HandType) -> usize {
        HandType::ALL.iter().position(|&h| h == ht).unwrap()
    }

    pub fn get(&self, ht: HandType) -> &HandTableRow {
        &self.rows[Self::idx(ht)]
    }

    pub fn get_mut(&mut self, ht: HandType) -> &mut HandTableRow {
        &mut self.rows[Self::idx(ht)]
    }

    /// `level_up_hand(card, hand, instant, amount)` (common_events.lua:464-469)
    /// — the exact recompute, not an incremental add:
    ///   level = max(0, level + amount)
    ///   mult  = max(s_mult + l_mult*(level-1), 1)
    ///   chips = max(s_chips + l_chips*(level-1), 0)
    /// (The Arm passes amount = -1; it only fires at level > 1, but Planets
    /// and jokers use this too.)
    pub fn level_up(&mut self, ht: HandType, amount: i64) {
        let v = ht.base_values();
        let row = self.get_mut(ht);
        row.level = (row.level as i64 + amount).max(0) as u32;
        let lvl = row.level as f64;
        row.mult = (v.base_mult as f64 + v.level_mult as f64 * (lvl - 1.0)).max(1.0);
        row.chips = (v.base_chips as f64 + v.level_chips as f64 * (lvl - 1.0)).max(0.0);
    }

    pub fn reset_played_this_round(&mut self) {
        for row in &mut self.rows {
            row.played_this_round = 0;
        }
    }
}

/// `mod_chips` (misc_functions.lua:684-689): identity without the
/// `chips_dollar_cap` challenge modifier.
#[inline]
fn mod_chips(chips: f64) -> f64 {
    chips
}

/// `mod_mult` (misc_functions.lua:691-693): identity.
#[inline]
fn mod_mult(mult: f64) -> f64 {
    mult
}

// ---------------------------------------------------------------------------
// Joker hook points (P3b plugs a real joker engine into these)
// ---------------------------------------------------------------------------

/// Immutable context handed to joker hooks: the pieces of
/// `{cardarea, full_hand, scoring_hand, scoring_name, poker_hands}` the
/// P3a flow already computes.
#[derive(Debug)]
pub struct HandCtx<'a> {
    pub full_hand: &'a [Card],
    /// Indices into `full_hand`, in scoring order.
    pub scoring: &'a [usize],
    pub hand_type: HandType,
    /// The cards left in hand (`G.hand.cards`), in display order — Raised
    /// Fist/Blackboard scan it.
    pub held: &'a [Card],
    /// The full `poker_hands` results table — hand-type jokers gate on
    /// `next(context.poker_hands[type])` (card.lua:3661-3671), which can
    /// match below the played hand (a Full House also contains a Pair).
    pub poker_hands: &'a PokerHands,
}

/// Mutable context for the `{before = true}` window: Midas Mask/Vampire
/// rewrite the played cards' enhancements (card.lua:3443-3490) and DNA
/// emplaces a copy of the played card into the live hand (card.lua:3501-3524)
/// — all before any card scores.
#[derive(Debug)]
pub struct BeforeCtx<'a> {
    pub full_hand: &'a mut [Card],
    pub scoring: &'a [usize],
    pub hand_type: HandType,
    /// `G.hand.cards` — DNA's copy is emplaced here mid-window and
    /// participates in the held-card loop of the same play.
    pub held: &'a mut Vec<Card>,
    pub poker_hands: &'a PokerHands,
}

/// One entry of the `effects` list in evaluate_play: the fields a card or a
/// per-card joker effect can carry, applied in the exact field order of
/// state_events.lua:702-777 (play) / :832-869 (held). A `0.0` x_mult means
/// "absent" (the Lua uses nil).
#[derive(Debug, Clone, Copy, Default)]
pub struct CardEffect {
    pub chips: f64,
    pub mult: f64,
    pub p_dollars: i64,
    pub dollars: i64,
    pub x_mult: f64,
    /// Edition mods ride on the same effects entry (state_events.lua:759-776)
    /// and are applied last: chips += edition_chips; mult += edition_mult;
    /// mult *= edition_x_mult.
    pub edition_chips: f64,
    pub edition_mult: f64,
    pub edition_x_mult: f64,
    /// Held-in-hand `h_mult` (state_events.lua:850-855).
    pub h_mult: f64,
    /// End-of-round `h_dollars` (state_events.lua:221-224).
    pub h_dollars: i64,
    /// Hiker's permanent chip upgrade of the evaluated card
    /// (card.lua:3067-3069) — applied to the card by evaluate_play so the
    /// NEXT repetition's `get_chip_bonus` already sees it.
    pub perma_bonus: i64,
}

impl CardEffect {
    pub fn is_empty(&self) -> bool {
        self.chips == 0.0
            && self.mult == 0.0
            && self.p_dollars == 0
            && self.dollars == 0
            && self.x_mult == 0.0
            && self.edition_chips == 0.0
            && self.edition_mult == 0.0
            && self.edition_x_mult == 0.0
            && self.h_mult == 0.0
            && self.h_dollars == 0
            && self.perma_bonus == 0
    }
}

/// The joker calculation points of the scoring/round flow, one method per
/// `Card:calculate_joker`/`eval_card` call site. P3a wires every call at the
/// exact position the game invokes it, with the no-op `NoJokers` behind it;
/// P3b replaces that with the real joker engine.
pub trait JokerHooks {
    /// `calculate_joker{setting_blind = true}` — new_round,
    /// state_events.lua:335-337 (Chicot/Madness/Burglar/Riff-raff/
    /// Cartomancer/Ceremonial Dagger/Marble Joker).
    fn setting_blind(&mut self, _rng: &mut RngState) {}
    /// `calculate_joker{first_hand_drawn = true}` — game.lua:3226-3231
    /// (first deal of the round only; Certificate).
    fn first_hand_drawn(&mut self) {}
    /// `next(find_joker('Splash'))` — evaluate_play's scoring-set override
    /// (state_events.lua:582-584): every played card scores.
    fn splash(&self) -> bool {
        false
    }
    /// `eval_card(joker, {before = true})` — state_events.lua:628-638.
    /// May level hands (Space Joker et al.), pay out (To Do List calls
    /// ease_dollars inline, card.lua:3487-3496), rewrite played cards
    /// (Midas Mask/Vampire) and emplace a hand copy (DNA).
    fn before_hand(
        &mut self,
        _ctx: &mut BeforeCtx,
        _hands: &mut HandsTable,
        _dollars: &mut i64,
        _rng: &mut RngState,
    ) {
    }
    /// Scoring-card retriggers: `eval_card(joker, {other_card=…,
    /// repetition = true})` per joker — state_events.lua:676-684. Returns
    /// total extra repetitions.
    fn scoring_card_repetitions(
        &mut self,
        _ctx: &HandCtx,
        _card: &Card,
        _rng: &mut RngState,
    ) -> u32 {
        0
    }
    /// Per-scoring-card individual joker effects:
    /// `calculate_joker{individual = true, cardarea = G.play}` —
    /// state_events.lua:693-699. Runs once per repetition;
    /// `lucky_trigger` mirrors `card.lucky_trigger` (set by this
    /// repetition's Lucky rolls, card.lua:988/1076, read by Lucky Cat and
    /// cleared per rep at state_events.lua:700).
    fn on_scoring_card(
        &mut self,
        _ctx: &HandCtx,
        _card: &Card,
        _lucky_trigger: bool,
        _rng: &mut RngState,
    ) -> Vec<CardEffect> {
        Vec::new()
    }
    /// Per-held-card individual joker effects (`cardarea = G.hand,
    /// individual = true`) — state_events.lua:800-807.
    fn on_held_card(
        &mut self,
        _ctx: &HandCtx,
        _card: &Card,
        _rng: &mut RngState,
    ) -> Vec<CardEffect> {
        Vec::new()
    }
    /// Held-card retriggers (`cardarea = G.hand, repetition = true`) —
    /// state_events.lua:820-829. Mime only fires when the card produced any
    /// effect entry (`next(card_effects[1]) or #card_effects > 1`,
    /// card.lua:3387-3394), hence `has_effects`.
    fn held_card_repetitions(
        &mut self,
        _ctx: &HandCtx,
        _card: &Card,
        _has_effects: bool,
        _rng: &mut RngState,
    ) -> u32 {
        0
    }
    /// The whole joker-area phase — state_events.lua:877-944: per joker (and
    /// consumable), edition pre-effects (foil/holo), `joker_main` effects,
    /// joker-on-joker effects, edition post-x_mult. The dispatcher owns the
    /// internal ordering; P3a only fixed where the phase sits in the
    /// pipeline. `hands` is read by Supernova/Card Sharp; `dollars` is
    /// written by To Do List-style inline `ease_dollars` effects (Matador,
    /// card.lua:3719-3730); `blind_triggered` is the live
    /// `G.GAME.blind.triggered` Matador reads (card.lua:3720).
    #[allow(clippy::too_many_arguments)]
    fn joker_main(
        &mut self,
        _ctx: &HandCtx,
        _hands: &HandsTable,
        _chips: &mut f64,
        _mult: &mut f64,
        _dollars: &mut i64,
        _blind_triggered: bool,
        _rng: &mut RngState,
    ) {
    }
    /// `calculate_joker({destroying_card = …, full_hand = …})` —
    /// state_events.lua:956-959 (Sixth Sense reads full_hand).
    fn destroys_card(&mut self, _card: &Card, _full_hand: &[Card], _rng: &mut RngState) -> bool {
        false
    }
    /// `eval_card(joker, {remove_playing_cards = true, removed = …})` —
    /// state_events.lua:974-976 (also discard path :424-428).
    fn on_cards_destroyed(&mut self, _destroyed: &[Card]) {}
    /// `eval_card(joker, {after = true})` — state_events.lua:1068-1075.
    fn after_hand(&mut self, _ctx: &HandCtx, _rng: &mut RngState) {}
    /// `eval_card(joker, {debuffed_hand = true})` — state_events.lua:1017-1027
    /// (Matador pays inline when `G.GAME.blind.triggered`).
    fn on_debuffed_hand(&mut self, _ctx: &HandCtx, _blind_triggered: bool, _dollars: &mut i64) {}
    /// `calculate_joker({pre_discard = true, hook = …})` —
    /// state_events.lua:394-396 (Burnt Joker levels the selection's hand,
    /// card.lua:2749-2755 — hence the hands table and eval mods).
    fn pre_discard(
        &mut self,
        _selected: &[Card],
        _hook: bool,
        _hands: &mut HandsTable,
        _mods: &crate::handeval::EvalMods,
    ) {
    }
    /// `calculate_joker({discard = true, other_card = …})` per discarded card
    /// — state_events.lua:402-409. Returns `eval.remove` (Trading Card).
    /// Mail-In Rebate/Faceless pay via inline `ease_dollars`
    /// (card.lua:2825-2834 / 2874-2887), hence `dollars`.
    fn on_discard_card(
        &mut self,
        _card: &Card,
        _selected: &[Card],
        _dollars: &mut i64,
        _rng: &mut RngState,
    ) -> bool {
        false
    }
    /// `calculate_joker({end_of_round = true, game_over = …})` per joker —
    /// state_events.lua:99-110. Gros Michel/Cavendish roll their extinction
    /// odds here (card.lua:3019-3046); To Do List re-rolls its hand over the
    /// visible entries of `hands` (card.lua:2975-2985); Mr. Bones returns
    /// `saved` on a ≥25% loss (card.lua:3047-3063) — the return value, which
    /// also flips `game_over` off for the remaining jokers
    /// (state_events.lua:103-105).
    fn end_of_round(&mut self, _hands: &HandsTable, _game_over: bool, _rng: &mut RngState) -> bool {
        false
    }
    /// Per-held-card end-of-round joker effects — state_events.lua:181-187.
    /// (`context.end_of_round + individual` is an empty branch in
    /// calculate_joker, card.lua:2875-2876 — no base-game joker uses it.)
    fn end_of_round_card(&mut self, _card: &Card, _rng: &mut RngState) -> Vec<CardEffect> {
        Vec::new()
    }
    /// End-of-round held-card retriggers — state_events.lua:199-208 (Mime,
    /// gated on the card having effects — card.lua:2879-2886).
    fn end_of_round_card_repetitions(&mut self, _card: &Card, _has_effects: bool) -> u32 {
        0
    }
    /// `Card:calculate_dollar_bonus` — state_events.lua:1175-1182
    /// (Rocket/Delayed Gratification…).
    fn dollar_bonus(&mut self) -> i64 {
        0
    }
}

/// The P3a placeholder joker engine: every hook is the trait default no-op.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoJokers;

impl JokerHooks for NoJokers {}

// ---------------------------------------------------------------------------
// evaluate_play
// ---------------------------------------------------------------------------

/// Everything evaluate_play mutates besides the play/held slices.
pub struct ScoreContext<'a> {
    pub rng: &'a mut RngState,
    pub hands: &'a mut HandsTable,
    /// `G.GAME.dollars` — Gold seals / Lucky $ / The Ox write it mid-scoring.
    pub dollars: &'a mut i64,
    /// The live blind (None only in bare unit tests; the game always has
    /// one, possibly a Small/Big blind with no effects).
    pub blind: Option<&'a mut ActiveBlind>,
    /// `G.GAME.current_round.most_played_poker_hand` (for The Ox).
    pub most_played: HandType,
    pub mods: EvalMods,
    /// `G.GAME.probabilities.normal` (Oops! All 6s doubles it) — Lucky rolls
    /// and the Glass break roll read it (card.lua:988/1076,
    /// state_events.lua:961).
    pub prob_normal: f64,
}

/// Result of scoring one played hand.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlayResult {
    pub hand_type: HandType,
    /// Indices (into the played slice) of the scoring cards, in scoring order.
    pub scoring: Vec<usize>,
    pub chips: f64,
    pub mult: f64,
    /// `math.floor(hand_chips * mult)` — the amount added to `G.GAME.chips`.
    pub score: f64,
    /// `Blind:debuff_hand` returned true: hand scored 0.
    pub debuffed_hand: bool,
    /// Play-slice indices of cards destroyed at step 13 (shattered Glass /
    /// joker-destroyed). These cards must not return to the deck.
    pub destroyed: Vec<usize>,
    /// Net dollars added/removed during scoring (Gold seal +3s, Lucky +20s,
    /// The Ox's wipe), for observability.
    pub dollars_delta: i64,
}

/// `eval_card` with `cardarea == G.play` (common_events.lua:592-622) minus
/// the (nil for playing cards) `calculate_joker` slot: one scoring pass over
/// a single card. Consumes 'lucky_mult'/'lucky_money' streams for Lucky
/// cards. Debuffed cards never reach this (skipped upstream), but every
/// getter also debuff-guards like card.lua does.
fn eval_card_play(card: &Card, prob_normal: f64, rng: &mut RngState) -> (CardEffect, bool) {
    let mut eff = CardEffect::default();
    let mut lucky_trigger = false;
    if card.debuff {
        return (eff, false);
    }
    // get_chip_bonus (card.lua:976-982); only registered when > 0
    // (common_events.lua:593-596).
    let chips = card.chip_bonus();
    if chips > 0.0 {
        eff.chips = chips;
    }
    // get_chip_mult (card.lua:984-997): Lucky rolls
    // `pseudorandom('lucky_mult') < G.GAME.probabilities.normal/5` for
    // +ability.mult (20, game.lua:655) and sets `lucky_trigger`; plain Mult
    // cards give 4.
    let mult = match card.enhancement {
        Enhancement::Lucky => {
            if rng.random("lucky_mult") < prob_normal / 5.0 {
                lucky_trigger = true;
                card.enhancement.mult_value()
            } else {
                0.0
            }
        }
        e => e.mult_value(),
    };
    if mult > 0.0 {
        eff.mult = mult;
    }
    // get_chip_x_mult (card.lua:999-1004): ability.x_mult (Glass = 2,
    // game.lua:651) when > 1.
    if card.enhancement == Enhancement::Glass {
        eff.x_mult = 2.0;
    }
    // get_p_dollars (card.lua:1068-1089): Gold seal +$3; Lucky rolls
    // `pseudorandom('lucky_money') < normal/15` for +ability.p_dollars (20),
    // also setting lucky_trigger.
    let mut p_dollars = 0i64;
    if card.seal == Seal::Gold {
        p_dollars += 3;
    }
    if card.enhancement == Enhancement::Lucky && rng.random("lucky_money") < prob_normal / 15.0 {
        lucky_trigger = true;
        p_dollars += 20;
    }
    if p_dollars > 0 {
        eff.p_dollars = p_dollars;
    }
    // get_edition (card.lua:1016-1031).
    let (e_chips, e_mult, e_x) = card.edition.scoring_mods();
    eff.edition_chips = e_chips.unwrap_or(0.0);
    eff.edition_mult = e_mult.unwrap_or(0.0);
    eff.edition_x_mult = e_x.unwrap_or(0.0);
    (eff, lucky_trigger)
}

/// `eval_card` with `cardarea == G.hand` (common_events.lua:624-639):
/// held-in-hand effects. Steel = h_x_mult 1.5 (m_steel, game.lua:652;
/// card.lua:1011-1014); base cards have no h_mult.
fn eval_card_held(card: &Card) -> CardEffect {
    let mut eff = CardEffect::default();
    if card.debuff {
        return eff;
    }
    if card.enhancement == Enhancement::Steel {
        // get_chip_h_x_mult > 0 lands in ret.x_mult (common_events.lua:630-633).
        eff.x_mult = 1.5;
    }
    eff
}

/// Apply one effects-table entry in the play-loop field order
/// (state_events.lua:702-777).
fn apply_play_effect(eff: &CardEffect, chips: &mut f64, mult: &mut f64, dollars: &mut i64) {
    if eff.chips != 0.0 {
        *chips = mod_chips(*chips + eff.chips);
    }
    if eff.mult != 0.0 {
        *mult = mod_mult(*mult + eff.mult);
    }
    if eff.p_dollars != 0 {
        *dollars += eff.p_dollars; // ease_dollars (state_events.lua:720-724)
    }
    if eff.dollars != 0 {
        *dollars += eff.dollars; // state_events.lua:727-731
    }
    if eff.x_mult != 0.0 {
        *mult = mod_mult(*mult * eff.x_mult); // state_events.lua:751-756
    }
    // Edition, last (state_events.lua:759-776): chips add, mult add, then
    // the x_mult multiply — all inside one entry.
    if eff.edition_chips != 0.0 || eff.edition_mult != 0.0 || eff.edition_x_mult != 0.0 {
        *chips = mod_chips(*chips + eff.edition_chips);
        *mult += eff.edition_mult;
        *mult = mod_mult(
            *mult
                * (if eff.edition_x_mult != 0.0 {
                    eff.edition_x_mult
                } else {
                    1.0
                }),
        );
    }
}

/// Apply one held-card effects entry (state_events.lua:832-869): dollars,
/// h_mult, x_mult, in that order.
fn apply_held_effect(eff: &CardEffect, mult: &mut f64, dollars: &mut i64) {
    if eff.dollars != 0 {
        *dollars += eff.dollars;
    }
    if eff.h_mult != 0.0 {
        *mult = mod_mult(*mult + eff.h_mult);
    }
    if eff.x_mult != 0.0 {
        *mult = mod_mult(*mult * eff.x_mult);
    }
}

/// `G.FUNCS.evaluate_play` (state_events.lua:571-1086). `play` is the played
/// cards in play order (debuff flags already stamped by the blind) — mutable
/// because Midas Mask/Vampire rewrite enhancements and Hiker stamps
/// perma_bonus; `held` the cards remaining in hand, in hand (display) order
/// — mutable because DNA emplaces its copy there mid-evaluation.
pub fn evaluate_play(
    ctx: &mut ScoreContext<'_>,
    play: &mut [Card],
    held: &mut Vec<Card>,
    hooks: &mut dyn JokerHooks,
) -> PlayResult {
    let dollars_before = *ctx.dollars;
    let (hand_type, mut scoring, poker_hands) = get_poker_hand_info(play, &ctx.mods);

    // state_events.lua:574-578 (:578 reveals secret hands).
    {
        let row = ctx.hands.get_mut(hand_type);
        row.played += 1;
        row.played_this_round += 1;
        row.visible = true;
    }
    // (G.GAME.last_hand_played = text, :576 — the caller records it from the
    // returned hand_type; it is set even for debuffed hands.)

    // Splash (state_events.lua:582-584): `scoring_hand[i] = G.play.cards[i]`
    // for every played card — the whole play is the scoring set. Otherwise
    // pure-bonus Stone cards join the scoring set (:585-599). Either way the
    // set is re-sorted by T.x == play order (:600).
    if hooks.splash() {
        scoring = (0..play.len()).collect();
    } else {
        for (i, card) in play.iter().enumerate() {
            if card.enhancement == Enhancement::Stone && !scoring.contains(&i) {
                scoring.push(i);
            }
        }
    }
    scoring.sort_unstable();

    // Blind:debuff_hand (:614) — Arm/Ox side effects happen in here.
    let debuffed_hand = match ctx.blind.as_deref_mut() {
        Some(b) => b.debuff_hand(
            play.len(),
            hand_type,
            ctx.hands,
            ctx.dollars,
            ctx.most_played,
        ),
        None => false,
    };

    // The HandCtx borrows are scoped per hook call: the before window and
    // Hiker's perma_bonus writes mutate `play` between calls.
    macro_rules! hctx {
        () => {
            HandCtx {
                full_hand: &*play,
                scoring: &scoring,
                hand_type,
                held: &*held,
                poker_hands: &poker_hands,
            }
        };
    }

    let mut hand_chips: f64;
    let mut mult: f64;
    let mut destroyed: Vec<usize> = Vec::new();

    if !debuffed_hand {
        // (:615-616 read mult/chips once here; both are overwritten by the
        // re-read at :640-641, so only that one is materialized below.)

        // Before-hand joker hook (:628-638) — may rewrite play/held.
        {
            let mut bctx = BeforeCtx {
                full_hand: &mut *play,
                scoring: &scoring,
                hand_type,
                held,
                poker_hands: &poker_hands,
            };
            hooks.before_hand(&mut bctx, ctx.hands, ctx.dollars, ctx.rng);
        }

        // Level-aware base (:640-641) — after The Arm and before-jokers.
        mult = mod_mult(ctx.hands.get(hand_type).mult);
        hand_chips = mod_chips(ctx.hands.get(hand_type).chips);

        // Blind:modify_hand — The Flint (:645-646).
        if let Some(b) = ctx.blind.as_deref_mut() {
            let (m, c, _modded) = b.modify_hand(mult, hand_chips);
            mult = mod_mult(m);
            hand_chips = mod_chips(c);
        }

        // Per scoring card, in play order (:648-780).
        for &i in &scoring {
            if play[i].debuff {
                // :655-663 — debuffed cards contribute nothing at all (no
                // seals, no editions, no repetitions), only blind.triggered.
                if let Some(b) = ctx.blind.as_deref_mut() {
                    b.triggered = true;
                }
                continue;
            }
            // Repetitions are counted up front (:666-684): red seal
            // (calculate_seal{repetition} → 1, card.lua:2244-2252), then
            // jokers. Total evaluations = 1 + reps.
            let mut reps = 1usize;
            if play[i].seal == Seal::Red {
                reps += 1;
            }
            reps += {
                let card = play[i];
                hooks.scoring_card_repetitions(&hctx!(), &card, ctx.rng) as usize
            };
            for _ in 0..reps {
                // effects[1] = eval_card (fresh Lucky/$ rolls each rep, :692;
                // lucky_trigger set here and cleared per rep, :700).
                let (base_eff, lucky) = eval_card_play(&play[i], ctx.prob_normal, ctx.rng);
                // per-card joker effects appended (:693-699)
                let joker_effs = {
                    let card = play[i];
                    hooks.on_scoring_card(&hctx!(), &card, lucky, ctx.rng)
                };
                apply_play_effect(&base_eff, &mut hand_chips, &mut mult, ctx.dollars);
                for eff in &joker_effs {
                    apply_play_effect(eff, &mut hand_chips, &mut mult, ctx.dollars);
                    // Hiker (card.lua:3067-3069): the upgrade lands on the
                    // card before the next repetition's get_chip_bonus.
                    play[i].perma_bonus += eff.perma_bonus;
                }
            }
        }

        // Held cards, in hand order (:784-872). The hand length is fixed
        // during this loop (DNA's copy joined in the before window).
        for hi in 0..held.len() {
            // reps grow only on the first iteration (j == 1, :809-830).
            let mut total = 1usize;
            let mut j = 0usize;
            while j < total {
                let card = held[hi];
                let base_eff = eval_card_held(&card);
                let joker_effs = hooks.on_held_card(&hctx!(), &card, ctx.rng);
                if j == 0 {
                    // Red seal retriggers held effects only when the card has
                    // any (`next(effects[1]) or #effects > 1`, :813-818);
                    // Mime's check is the same (card.lua:3387-3394).
                    let has_effects = !base_eff.is_empty() || !joker_effs.is_empty();
                    if card.seal == Seal::Red && !card.debuff && has_effects {
                        total += 1;
                    }
                    total +=
                        hooks.held_card_repetitions(&hctx!(), &card, has_effects, ctx.rng) as usize;
                }
                apply_held_effect(&base_eff, &mut mult, ctx.dollars);
                for eff in &joker_effs {
                    apply_held_effect(eff, &mut mult, ctx.dollars);
                }
                j += 1;
            }
        }

        // Joker area phase (:877-944). Matador reads the live
        // `G.GAME.blind.triggered` (card.lua:3720).
        let blind_triggered = ctx.blind.as_deref().map(|b| b.triggered).unwrap_or(false);
        hooks.joker_main(
            &hctx!(),
            ctx.hands,
            &mut hand_chips,
            &mut mult,
            ctx.dollars,
            blind_triggered,
            ctx.rng,
        );

        // Deck-back final_scoring_step (:946-948): Plasma Deck balances
        // chips/mult here; the Red Deck's trigger_effect is a no-op.

        // Destruction pass (:950-973): joker destroying_card hook first, then
        // the Glass break roll — `pseudorandom('glass') <
        // G.GAME.probabilities.normal/ability.extra` (extra = 4, game.lua:651)
        // — which rolls for every non-debuffed Glass scoring card even if a
        // joker already destroyed it (separate `if`s in the Lua).
        for &i in &scoring {
            let card = play[i];
            let mut d = hooks.destroys_card(&card, play, ctx.rng);
            if card.enhancement == Enhancement::Glass
                && !card.debuff
                && ctx.rng.random("glass") < ctx.prob_normal / 4.0
            {
                d = true;
            }
            if d {
                destroyed.push(i);
            }
        }
        if !destroyed.is_empty() {
            let cards: Vec<Card> = destroyed.iter().map(|&i| play[i]).collect();
            hooks.on_cards_destroyed(&cards);
        }
    } else {
        // :997-1027 — "Not Allowed!": zero score, debuffed_hand joker hook.
        mult = mod_mult(0.0);
        hand_chips = mod_chips(0.0);
        let blind_triggered = ctx.blind.as_deref().map(|b| b.triggered).unwrap_or(false);
        hooks.on_debuffed_hand(&hctx!(), blind_triggered, ctx.dollars);
    }

    // :1049 — G.GAME.chips += math.floor(hand_chips * mult).
    let score = (hand_chips * mult).floor();

    // After-hand joker hook (:1068-1075) — outside the debuff branch.
    hooks.after_hand(&hctx!(), ctx.rng);

    PlayResult {
        hand_type,
        scoring,
        chips: hand_chips,
        mult,
        score,
        debuffed_hand,
        destroyed,
        dollars_delta: *ctx.dollars - dollars_before,
    }
}
