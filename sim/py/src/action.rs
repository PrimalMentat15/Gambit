//! Composite action dict -> sim `Run` calls, with the contract's leniency
//! rules and the reward shaping scheme.
//!
//! Leniency / auto-legalization decisions (documented per the P6 spec):
//!
//! * **PLAY/Psychic**: the sim (like the game's `evaluate_play`) ACCEPTS a
//!   <5-card play under The Psychic and scores it 0 (debuffed hand) — the
//!   game UI merely disables the button. So PLAY_HAND is legal whenever the
//!   hand is non-empty, any 1..5 subset is a legal play, and the policy pays
//!   for wasted hands through the reward, not through masking.
//! * **Cerulean Bell**: the forced card must be part of every play/discard
//!   (the sim rejects selections without it) and per-card masks cannot
//!   express "must include". The env auto-includes the forced card:
//!   appended when <5 picks, replacing the last pick otherwise.
//! * **USE_CONSUMABLE targets**: truncated to the consumable's
//!   `max_highlighted`, padded from the lowest hand indices up to
//!   `min_highlighted`, and for Aura re-pointed at an edition-less card.
//!   Untargeted consumables ignore the picks entirely.
//! * **USE_CONSUMABLE on a currently-unusable slot**: the shared
//!   consumable_target_mask covers SELL (always legal) and USE (sometimes
//!   not — e.g. a targeted tarot in the shop), a cross-head constraint the
//!   factored masks cannot express. If after legalization the consumable
//!   still cannot be used, the action is a NO-OP (never an error).
//! * **PICK_PACK of a hand-targeting consumable**: the contract's PICK_PACK
//!   carries no card sub-action, so targets are chosen automatically
//!   (lowest legal hand indices / first edition-less card for Aura). The
//!   pack mask already guarantees a legal assignment exists
//!   (`can_pick_pack_item` -> `consumable_has_any_use`).

use balatro_core::cards::Edition;
use balatro_core::items::consumable_by_key;
use balatro_core::run::{Run, State};

use crate::consts::{action_type as at, MAX_CARD_PICKS, N_ACTION_TYPES};
use crate::encode::{EnvMasks, ShopSlotRef};

/// One env's slice of the batched action dict.
#[derive(Debug, Clone, Copy, Default)]
pub struct ActionRow {
    pub action_type: i64,
    pub cards: [i64; MAX_CARD_PICKS],
    pub n_cards: i64,
    pub joker_target: i64,
    pub consumable_target: i64,
    pub shop_target: i64,
    pub pack_target: i64,
}

fn needs_cards(t: i64) -> bool {
    matches!(t, at::PLAY_HAND | at::DISCARD | at::USE_CONSUMABLE)
}

fn card_pick_min(t: i64) -> i64 {
    match t {
        at::PLAY_HAND | at::DISCARD => 1,
        _ => 0,
    }
}

/// Strict referee, mirroring `mock_env._Game.validate` exactly: the action
/// must lie inside the masks the env emitted with the previous observation.
pub fn validate(env_idx: usize, row: &ActionRow, m: &EnvMasks) -> Result<(), String> {
    let t = row.action_type;
    if !(0..N_ACTION_TYPES as i64).contains(&t) || !m.action_type[t as usize] {
        return Err(format!("env {env_idx}: illegal action_type {t}"));
    }

    let check_ptr = |name: &str, v: i64, mask: &[bool], needed: bool| -> Result<(), String> {
        if !needed {
            if v != -1 {
                return Err(format!("env {env_idx}: {name}={v} must be -1 for type {t}"));
            }
            return Ok(());
        }
        if v < 0 || v as usize >= mask.len() || !mask[v as usize] {
            return Err(format!("env {env_idx}: illegal {name}={v} for type {t}"));
        }
        Ok(())
    };
    check_ptr(
        "joker_target",
        row.joker_target,
        &m.joker_target,
        t == at::SELL_JOKER,
    )?;
    check_ptr(
        "consumable_target",
        row.consumable_target,
        &m.consumable_target,
        matches!(t, at::USE_CONSUMABLE | at::SELL_CONSUMABLE),
    )?;
    check_ptr(
        "shop_target",
        row.shop_target,
        &m.shop_target,
        t == at::BUY_SHOP,
    )?;
    check_ptr(
        "pack_target",
        row.pack_target,
        &m.pack_target,
        t == at::PICK_PACK,
    )?;

    if !needs_cards(t) {
        if row.n_cards != 0 || row.cards.iter().any(|&c| c != -1) {
            return Err(format!(
                "env {env_idx}: card params must be empty for type {t}"
            ));
        }
        return Ok(());
    }
    let lo = card_pick_min(t);
    if row.n_cards < lo || row.n_cards > MAX_CARD_PICKS as i64 {
        return Err(format!(
            "env {env_idx}: n_cards={} out of [{lo},{MAX_CARD_PICKS}] for type {t}",
            row.n_cards
        ));
    }
    let n = row.n_cards as usize;
    if row.cards[n..].iter().any(|&c| c != -1) {
        return Err(format!("env {env_idx}: cards not -1-padded past n_cards"));
    }
    for (i, &c) in row.cards[..n].iter().enumerate() {
        if c < 0 || c as usize >= m.card_select.len() || !m.card_select[c as usize] {
            return Err(format!("env {env_idx}: illegal card pick {c}"));
        }
        if row.cards[..i].contains(&c) {
            return Err(format!("env {env_idx}: duplicate card pick {c}"));
        }
    }
    Ok(())
}

/// The picked hand indices (deduped, clamped to the live hand), with the
/// Cerulean Bell forced card auto-included.
pub fn legalize_card_selection(run: &Run, row: &ActionRow) -> Vec<usize> {
    let hand_len = run.hand().len();
    let mut v: Vec<usize> = Vec::with_capacity(MAX_CARD_PICKS);
    for &c in &row.cards[..(row.n_cards.max(0) as usize).min(MAX_CARD_PICKS)] {
        let c = c as usize;
        if c < hand_len && !v.contains(&c) {
            v.push(c);
        }
    }
    if let Some(fi) = run.forced_card_index() {
        if !v.contains(&fi) {
            if v.len() >= MAX_CARD_PICKS {
                v.pop();
            }
            v.push(fi);
        }
    }
    if v.is_empty() && hand_len > 0 {
        v.push(0); // unreachable through the masks; belt-and-braces
    }
    v
}

/// Best-effort target list for a consumable (see module docs). `None` means
/// the consumable cannot be used right now even after legalization.
pub fn legalize_consumable_targets(run: &Run, key: &str, picks: &[usize]) -> Option<Vec<usize>> {
    let (meta, _set) = consumable_by_key(key)?;
    let hand_len = run.hand().len();
    let mut v: Vec<usize> = Vec::new();
    for &p in picks {
        if p < hand_len && !v.contains(&p) {
            v.push(p);
        }
    }
    if key == "c_aura" {
        // Special-cased by the sim (its registry meta has max_highlighted 0):
        // exactly 1 edition-less highlighted card.
        v.retain(|&i| run.hand()[i].edition == Edition::None);
        v.truncate(1);
        if v.is_empty() {
            if let Some(i) = run.hand().iter().position(|c| c.edition == Edition::None) {
                v.push(i);
            }
        }
    } else if meta.max_highlighted > 0 {
        let maxn = (meta.max_highlighted as usize).min(MAX_CARD_PICKS);
        let minn = (meta.min_highlighted as usize).max(1);
        v.truncate(maxn);
        let mut i = 0usize;
        while v.len() < minn && i < hand_len {
            if !v.contains(&i) {
                v.push(i);
            }
            i += 1;
        }
    } else {
        v.clear();
    }
    if run.consumable_can_use(key, &v).is_ok() {
        Some(v)
    } else {
        None
    }
}

/// Apply one composite action. Returns the shaped reward. `win_ante` is the
/// curriculum goal (8 = the real game); the +15 win bonus fires on the step
/// that defeats that ante's boss. Errors are internal invariant violations
/// (a masked-legal action the sim rejected) and abort the step with a
/// Python-side exception.
pub fn apply(
    run: &mut Run,
    row: &ActionRow,
    slots: &[ShopSlotRef],
    beta: f64,
    win_ante: i64,
) -> Result<f64, String> {
    let t = row.action_type;
    let in_blind = run.state() == State::SelectingHand;
    let ante0 = run.ante();
    let req0 = run.blind_chips();
    let phi0 = if in_blind && req0 > 0.0 {
        (run.chips() / req0).min(1.0)
    } else {
        0.0
    };

    let internal = |e: balatro_core::run::RunError| format!("sim rejected masked action {t}: {e}");

    match t {
        at::PLAY_HAND => {
            let sel = legalize_card_selection(run, row);
            run.play(&sel).map_err(internal)?;
        }
        at::DISCARD => {
            let sel = legalize_card_selection(run, row);
            run.discard(&sel).map_err(internal)?;
        }
        at::SELECT_BLIND => run.select_blind().map_err(internal)?,
        at::SKIP_BLIND => run.skip_blind().map_err(internal)?,
        at::CASH_OUT => run.cash_out().map_err(internal)?,
        at::BUY_SHOP => {
            let s = row.shop_target as usize;
            match slots.get(s).copied() {
                Some(ShopSlotRef::Card(i)) => run.buy_shop_item(i).map_err(internal)?,
                Some(ShopSlotRef::Voucher(i)) => run.redeem_voucher(i).map_err(internal)?,
                Some(ShopSlotRef::Pack(i)) => run.buy_pack(i).map_err(internal)?,
                None => return Err(format!("BUY_SHOP target {s} out of slot map")),
            }
        }
        at::REROLL => run.reroll_shop().map_err(internal)?,
        at::LEAVE_SHOP => run.leave_shop().map_err(internal)?,
        at::USE_CONSUMABLE => {
            let i = row.consumable_target as usize;
            let key = match run.consumables().get(i) {
                Some(c) => c.key,
                None => return Err(format!("USE_CONSUMABLE target {i} out of range")),
            };
            let picks = legalize_card_selection_raw(run, row);
            // `None` = the documented no-op leniency (unusable right now).
            if let Some(targets) = legalize_consumable_targets(run, key, &picks) {
                run.use_consumable(i, &targets).map_err(internal)?;
            }
        }
        at::SELL_JOKER => run
            .sell_joker(row.joker_target as usize)
            .map_err(internal)?,
        at::SELL_CONSUMABLE => run
            .sell_consumable(row.consumable_target as usize)
            .map_err(internal)?,
        at::PICK_PACK => {
            let i = row.pack_target as usize;
            let key = match run.pack().and_then(|p| p.items.get(i)) {
                Some(balatro_core::shop::PackItem::Consumable(c)) => Some(c.key),
                Some(_) => None,
                None => return Err(format!("PICK_PACK target {i} out of range")),
            };
            let targets = match key {
                Some(k) => legalize_consumable_targets(run, k, &[])
                    .ok_or_else(|| format!("pack consumable {k} unusable despite mask"))?,
                None => Vec::new(),
            };
            run.pick_pack_item(i, &targets).map_err(internal)?;
        }
        at::SKIP_PACK => run.skip_pack().map_err(internal)?,
        _ => return Err(format!("unhandled action type {t}")),
    }

    // ---------------------------------------------------------- reward shaping
    // 1. per played hand: beta * delta min(chips/req, 1) (overkill earns 0);
    // 2. blind clear: beta * 0.5 * (1 + 0.15 * ante-at-clear), whatever
    //    action cleared it (a Luchador sell disabling the boss counts);
    // 3. win: +15 unscaled by beta (real win OR the curriculum `win_ante`
    //    boss defeated — `Run::ante()` increments exactly on boss defeat,
    //    so `ante > win_ante` first holds on this step). Loss: nothing.
    let mut reward = 0.0f64;
    if in_blind {
        match run.state() {
            State::SelectingHand => {
                if t == at::PLAY_HAND && req0 > 0.0 {
                    let phi1 = (run.chips() / req0).min(1.0);
                    reward += beta * (phi1 - phi0);
                }
            }
            State::RoundEval => {
                // Blind cleared this step: complete the progress term, then
                // the clear bonus (ante as it was when the blind was fought).
                reward += beta * (1.0 - phi0);
                reward += beta * 0.5 * (1.0 + 0.15 * ante0 as f64);
                if run.won() || run.ante() > win_ante {
                    reward += 15.0;
                }
            }
            _ => {}
        }
    }
    Ok(reward)
}

/// The raw (deduped, clamped) picks without forced-card injection — used as
/// consumable target candidates.
pub fn legalize_card_selection_raw(run: &Run, row: &ActionRow) -> Vec<usize> {
    let hand_len = run.hand().len();
    let mut v: Vec<usize> = Vec::with_capacity(MAX_CARD_PICKS);
    for &c in &row.cards[..(row.n_cards.max(0) as usize).min(MAX_CARD_PICKS)] {
        let c = c as usize;
        if c < hand_len && !v.contains(&c) {
            v.push(c);
        }
    }
    v
}
