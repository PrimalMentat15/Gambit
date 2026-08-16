//! P5 cross-validation state export: a JSON document of the full observable
//! run state, with keys/enums designed to map 1:1 onto the balatrobot
//! `gamestate` schema (bridge/src/balatro_bridge/types.py) wherever the two
//! overlap. Fields the mod does not expose (discard pile, held tags,
//! last_hand_played, internal joker counters, ...) live under sim-only keys
//! so the Python harness can diff the shared subset and still bundle the
//! full sim view into divergence reports.
//!
//! Also here: the direct Run-level action interface used by the harness
//! (`legal_actions_json`, `apply_action`, `bot_direct_action`) — P5 drives
//! the sim through these instead of the RL obs/action encoding.

use balatro_core::cards::{Card, Edition, Enhancement, HandType, Seal};
use balatro_core::items::{consumable_by_key, tag_by_key};
use balatro_core::run::{Action, BlindStage, Run, State};
use balatro_core::shop::{OwnedConsumable, OwnedJoker, PackItem, PackReturn, ShopItemKind};
use serde_json::{json, Map, Value};

use crate::action::{legalize_card_selection_raw, legalize_consumable_targets, ActionRow};
use crate::bot::bot_action;
use crate::consts::action_type as at;
use crate::encode::{encode_masks, shop_slot_map, ShopSlotRef};

/// Sim `State` -> balatrobot state name (utils/gamestate.lua get_state_name;
/// pack states are reported as SMODS_BOOSTER_OPENED by the mod).
pub fn state_name(s: State) -> &'static str {
    match s {
        State::BlindSelect => "BLIND_SELECT",
        State::SelectingHand => "SELECTING_HAND",
        State::RoundEval => "ROUND_EVAL",
        State::Shop => "SHOP",
        State::PackOpen => "SMODS_BOOSTER_OPENED",
        State::GameOver => "GAME_OVER",
        // The real game enters ROUND_EVAL with G.GAME.won == true (the
        // win_game overlay is UI); the sim's terminal Won maps there.
        State::Won => "ROUND_EVAL",
        _ => "UNKNOWN",
    }
}

/// The game's display name for a poker hand (`G.GAME.hands` keys).
pub fn hand_name(ht: HandType) -> &'static str {
    match ht {
        HandType::HighCard => "High Card",
        HandType::Pair => "Pair",
        HandType::TwoPair => "Two Pair",
        HandType::ThreeOfAKind => "Three of a Kind",
        HandType::Straight => "Straight",
        HandType::Flush => "Flush",
        HandType::FullHouse => "Full House",
        HandType::FourOfAKind => "Four of a Kind",
        HandType::StraightFlush => "Straight Flush",
        HandType::FiveOfAKind => "Five of a Kind",
        HandType::FlushHouse => "Flush House",
        HandType::FlushFive => "Flush Five",
    }
}

fn enhancement_name(e: Enhancement) -> Value {
    match e {
        Enhancement::None => Value::Null,
        Enhancement::Bonus => json!("BONUS"),
        Enhancement::Mult => json!("MULT"),
        Enhancement::Wild => json!("WILD"),
        Enhancement::Glass => json!("GLASS"),
        Enhancement::Steel => json!("STEEL"),
        Enhancement::Stone => json!("STONE"),
        Enhancement::Gold => json!("GOLD"),
        Enhancement::Lucky => json!("LUCKY"),
    }
}

fn edition_name(e: Edition) -> Value {
    match e {
        Edition::None => Value::Null,
        Edition::Foil => json!("FOIL"),
        Edition::Holo => json!("HOLO"),
        Edition::Polychrome => json!("POLYCHROME"),
        Edition::Negative => json!("NEGATIVE"),
    }
}

fn seal_name(s: Seal) -> Value {
    match s {
        Seal::None => Value::Null,
        Seal::Gold => json!("GOLD"),
        Seal::Red => json!("RED"),
        Seal::Blue => json!("BLUE"),
        Seal::Purple => json!("PURPLE"),
    }
}

/// A playing card as the mod reports it: `key` = "H_A" style front key,
/// modifiers under the balatrobot names.
fn card_json(c: &Card) -> Value {
    json!({
        "key": format!("{}_{}", c.suit.key(), c.rank.key()),
        "rank": c.rank.key().to_string(),
        "suit": c.suit.key().to_string(),
        "enhancement": enhancement_name(c.enhancement),
        "edition": edition_name(c.edition),
        "seal": seal_name(c.seal),
        "debuff": c.debuff,
        "hidden": c.face_down,
    })
}

fn joker_json(run: &Run, j: &OwnedJoker) -> Value {
    json!({
        "key": j.id.key(),
        "set": "JOKER",
        "edition": edition_name(j.edition),
        "sell": run.joker_sell_value(j),
        "debuff": j.debuffed,
        "hidden": j.flipped,
        "eternal": j.eternal,
        // Sim-only: the mutable ability state (the mod only exposes these
        // through the rendered description text).
        "state": {
            "mult": j.state.mult,
            "chips": j.state.chips,
            "x_mult": j.state.x_mult,
            "extra": j.state.extra,
            "to_do_hand": hand_name(j.state.to_do_hand),
            "extra_value": j.extra_value,
        },
    })
}

fn consumable_json(run: &Run, c: &OwnedConsumable) -> Value {
    let set = consumable_by_key(c.key)
        .map(|(_, s)| s.type_str().to_uppercase())
        .unwrap_or_default();
    json!({
        "key": c.key,
        "set": set,
        "edition": if c.negative { json!("NEGATIVE") } else { Value::Null },
        "sell": run.consumable_sell_value(c),
    })
}

fn area_json(cards: Vec<Value>, limit: Option<i64>) -> Value {
    let mut m = Map::new();
    m.insert("count".into(), json!(cards.len()));
    if let Some(l) = limit {
        m.insert("limit".into(), json!(l));
    }
    m.insert("cards".into(), Value::Array(cards));
    Value::Object(m)
}

/// Which stage the live blind belongs to, if a round is running.
fn active_stage(run: &Run) -> Option<BlindStage> {
    run.current_blind().map(|p| {
        if p.key == "bl_small" {
            BlindStage::Small
        } else if p.key == "bl_big" {
            BlindStage::Big
        } else {
            BlindStage::Boss
        }
    })
}

/// The mod's `blind_states` enum for one stage, derived from the sim flags.
/// Timing choices mirror the game: 'Defeated' is visible from ROUND_EVAL on
/// (end_round stamps it before cash out), 'Select' only while the
/// blind-select UI would be up (including a tag pack opened from it).
fn blind_status(run: &Run, stage: BlindStage) -> &'static str {
    if active_stage(run) == Some(stage) {
        return match run.state() {
            State::RoundEval | State::Won => "DEFEATED",
            _ => "CURRENT",
        };
    }
    match stage {
        BlindStage::Small if run.small_defeated() => return "DEFEATED",
        BlindStage::Small if run.small_skipped() => return "SKIPPED",
        BlindStage::Big if run.big_defeated() => return "DEFEATED",
        BlindStage::Big if run.big_skipped() => return "SKIPPED",
        _ => {}
    }
    let at_select = match run.state() {
        State::BlindSelect => true,
        State::PackOpen => run
            .pack()
            .map(|p| p.return_to == PackReturn::BlindSelect)
            .unwrap_or(false),
        _ => false,
    };
    if at_select && run.blind_on_deck() == stage {
        "SELECT"
    } else {
        "UPCOMING"
    }
}

fn blind_json(
    run: &Run,
    stage: BlindStage,
    proto: &'static balatro_core::blinds::BlindProto,
    tag_key: Option<&str>,
) -> Value {
    let ty = match stage {
        BlindStage::Small => "SMALL",
        BlindStage::Big => "BIG",
        BlindStage::Boss => "BOSS",
    };
    // The mod computes score from get_blind_amount(round_resets.ante),
    // which ease_ante bumps at end_round — BEFORE blind_ante snaps to it
    // at cash out. Mirror the mod exactly.
    let score = proto.chips(run.ante()).floor() as i64;
    json!({
        "type": ty,
        "status": blind_status(run, stage),
        "name": proto.name,
        "score": score,
        "tag_key": tag_key,
        "tag_name": tag_key.and_then(|k| tag_by_key(k).map(|m| m.name)).unwrap_or(""),
    })
}

fn shop_json(run: &Run) -> (Value, Value, Value) {
    let Some(shop) = run.shop() else {
        return (Value::Null, Value::Null, Value::Null);
    };
    let cards: Vec<Value> = shop
        .jokers
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let cost = run.shop_item_cost(item);
            let mut v = match &item.kind {
                ShopItemKind::Joker(j) => json!({
                    "key": j.id.key(),
                    "set": "JOKER",
                    "edition": edition_name(j.edition),
                }),
                ShopItemKind::Consumable(c) => {
                    let set = consumable_by_key(c.key)
                        .map(|(_, s)| s.type_str().to_uppercase())
                        .unwrap_or_default();
                    json!({
                        "key": c.key,
                        "set": set,
                        "edition": if c.negative { json!("NEGATIVE") } else { Value::Null },
                    })
                }
                ShopItemKind::PlayingCard(c) => {
                    let mut v = card_json(c);
                    let set = if c.enhancement == Enhancement::None {
                        "DEFAULT"
                    } else {
                        "ENHANCED"
                    };
                    v["set"] = json!(set);
                    v
                }
            };
            v["buy"] = json!(cost);
            v["slot"] = json!(i);
            v["couponed"] = json!(item.couponed);
            v
        })
        .collect();
    let vouchers: Vec<Value> = shop
        .vouchers
        .iter()
        .enumerate()
        .map(|(i, vo)| {
            json!({
                "key": vo.key,
                "set": "VOUCHER",
                "buy": run.voucher_cost(vo.key),
                "slot": i,
            })
        })
        .collect();
    // Bought packs stay in the sim vec (used = true) but leave the game's
    // area — export only unopened offers; `slot` keeps the sim index.
    let packs: Vec<Value> = shop
        .packs
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.used)
        .map(|(i, p)| {
            json!({
                "key": p.key,
                "set": "BOOSTER",
                "buy": run.pack_cost(p),
                "slot": i,
            })
        })
        .collect();
    (
        area_json(cards, None),
        area_json(vouchers, None),
        area_json(packs, None),
    )
}

fn pack_item_json(run: &Run, item: &PackItem) -> Value {
    match item {
        PackItem::Joker(j) => joker_json(run, j),
        PackItem::Consumable(c) => consumable_json(run, c),
        PackItem::PlayingCard(c) => {
            let mut v = card_json(c);
            v["set"] = json!(if c.enhancement == Enhancement::None {
                "DEFAULT"
            } else {
                "ENHANCED"
            });
            v
        }
    }
}

/// The full observable run state as JSON (see module docs for schema).
pub fn state_json(run: &Run) -> Value {
    let mut root = Map::new();
    root.insert("state".into(), json!(state_name(run.state())));
    root.insert("seed".into(), json!(run.seed()));
    root.insert("deck".into(), json!("RED"));
    root.insert("stake".into(), json!("WHITE"));
    root.insert("ante_num".into(), json!(run.ante()));
    root.insert("round_num".into(), json!(run.round()));
    root.insert("money".into(), json!(run.dollars()));
    root.insert("won".into(), json!(run.won()));

    root.insert(
        "round".into(),
        json!({
            "hands_left": run.hands_left(),
            "hands_played": run.hands_played_this_round(),
            "discards_left": run.discards_left(),
            "discards_used": run.discards_used_this_round(),
            "reroll_cost": run.reroll_cost(),
            "chips": run.chips(),
        }),
    );

    let (tag_small, tag_big) = run.blind_tags();
    let boss = balatro_core::blinds::boss_by_key(run.boss_choice());
    root.insert(
        "blinds".into(),
        json!({
            "small": blind_json(run, BlindStage::Small, &balatro_core::blinds::BL_SMALL, Some(tag_small)),
            "big": blind_json(run, BlindStage::Big, &balatro_core::blinds::BL_BIG, Some(tag_big)),
            "boss": blind_json(run, BlindStage::Boss, boss, None),
        }),
    );

    root.insert(
        "hand".into(),
        area_json(
            run.hand().iter().map(card_json).collect(),
            Some(run.hand_size()),
        ),
    );
    // balatrobot calls the undrawn-deck area "cards"; order is
    // bottom-to-top (drawing pops from the END on both sides).
    root.insert(
        "cards".into(),
        area_json(run.deck_cards().iter().map(card_json).collect(), None),
    );
    root.insert(
        "jokers".into(),
        area_json(
            run.jokers().iter().map(|j| joker_json(run, j)).collect(),
            Some(run.joker_slots() as i64),
        ),
    );
    root.insert(
        "consumables".into(),
        area_json(
            run.consumables()
                .iter()
                .map(|c| consumable_json(run, c))
                .collect(),
            Some(run.consumable_slots() as i64),
        ),
    );

    let (shop, vouchers, packs) = shop_json(run);
    if run.state() == State::Shop || run.state() == State::PackOpen {
        root.insert("shop".into(), shop);
        root.insert("vouchers".into(), vouchers);
        root.insert("packs".into(), packs);
    }
    if let Some(pack) = run.pack() {
        root.insert(
            "pack".into(),
            json!({
                "key": pack.key,
                "kind": format!("{:?}", pack.kind),
                "choices_left": pack.choices_left,
                "count": pack.items.len(),
                "cards": pack.items.iter().map(|i| pack_item_json(run, i)).collect::<Vec<_>>(),
                "hand_dealt": pack.hand_dealt,
                "return_to": match pack.return_to {
                    PackReturn::Shop => "SHOP",
                    PackReturn::BlindSelect => "BLIND_SELECT",
                },
            }),
        );
    }

    let mut hands = Map::new();
    for ht in HandType::ALL {
        let row = run.hands_table().get(ht);
        hands.insert(
            hand_name(ht).into(),
            json!({
                "level": row.level,
                "chips": row.chips,
                "mult": row.mult,
                "played": row.played,
                "played_this_round": row.played_this_round,
            }),
        );
    }
    root.insert("hands".into(), Value::Object(hands));

    let mut used: Vec<&str> = run.used_vouchers().iter().copied().collect();
    used.sort_unstable();
    root.insert("used_vouchers".into(), json!(used));

    // ---- Sim-only fields (not exposed by balatrobot; carried for repro
    // bundles and debugging, never diffed against the game).
    root.insert(
        "sim".into(),
        json!({
            "discard": run.discard_cards().iter().map(card_json).collect::<Vec<_>>(),
            "destroyed": run.destroyed_cards().iter().map(card_json).collect::<Vec<_>>(),
            "tags": run.tags().iter().map(|t| json!({
                "key": t.key,
                "name": tag_by_key(t.key).map(|m| m.name).unwrap_or(""),
                "orbital_hand": t.orbital_hand.map(hand_name),
            })).collect::<Vec<_>>(),
            "last_hand_played": run.last_hand_played().map(hand_name),
            "most_played_hand": hand_name(run.most_played_hand()),
            "blind_on_deck": match run.blind_on_deck() {
                BlindStage::Small => "SMALL",
                BlindStage::Big => "BIG",
                BlindStage::Boss => "BOSS",
            },
            "boss_key": run.boss_choice(),
            "blind_chips": run.blind_chips(),
            "blind_disabled": run.active_blind().map(|b| b.disabled),
            "hand_size": run.hand_size(),
            "pending_cashout": run.pending_cashout(),
            "bankrupt_at": run.bankrupt_at(),
            "current_voucher": run.current_voucher(),
            "skips": run.skips(),
            "tarots_used": run.tarots_used(),
        }),
    );

    Value::Object(root)
}

// ---------------------------------------------------------------------------
// Direct action interface
// ---------------------------------------------------------------------------

/// The sim's legal actions as JSON descriptors the harness policy can pick
/// from. Card/target subsets are the policy's job; targeted consumables get
/// their min/max target counts attached.
pub fn legal_actions_json(run: &Run) -> Value {
    let acts: Vec<Value> = run
        .legal_actions()
        .iter()
        .map(|a| match a {
            Action::SelectBlind => json!({"kind": "select_blind"}),
            Action::SkipBlind => json!({"kind": "skip_blind"}),
            Action::RerollBoss => json!({"kind": "reroll_boss"}),
            Action::Play => json!({
                "kind": "play",
                "forced": run.forced_card_index(),
            }),
            Action::Discard => json!({
                "kind": "discard",
                "forced": run.forced_card_index(),
            }),
            Action::CashOut => json!({"kind": "cash_out"}),
            Action::BuyShopItem(i) => json!({"kind": "buy_card", "slot": i}),
            Action::BuyAndUseShopItem(i) => json!({"kind": "buy_and_use", "slot": i}),
            Action::RedeemVoucher(i) => json!({"kind": "redeem_voucher", "slot": i}),
            Action::BuyPack(i) => json!({"kind": "buy_pack", "slot": i}),
            Action::Reroll => json!({"kind": "reroll"}),
            Action::LeaveShop => json!({"kind": "leave_shop"}),
            Action::UseConsumable(i) => {
                let key = run.consumables()[*i].key;
                let (maxt, mint) = consumable_targets_range(key);
                json!({
                    "kind": "use_consumable",
                    "slot": i,
                    "key": key,
                    "max_targets": maxt,
                    "min_targets": mint,
                })
            }
            Action::SellConsumable(i) => json!({"kind": "sell_consumable", "slot": i}),
            Action::SellJoker(i) => json!({"kind": "sell_joker", "slot": i}),
            Action::PickPackItem(i) => {
                let (maxt, mint) = match run.pack().and_then(|p| p.items.get(*i)) {
                    Some(PackItem::Consumable(c)) => consumable_targets_range(c.key),
                    _ => (0, 0),
                };
                json!({
                    "kind": "pick_pack",
                    "slot": i,
                    "max_targets": maxt,
                    "min_targets": mint,
                })
            }
            Action::SkipPack => json!({"kind": "skip_pack"}),
            _ => json!({"kind": "unknown"}),
        })
        .collect();
    Value::Array(acts)
}

/// (max_highlighted, min_highlighted) for a consumable key; Aura's sim meta
/// says 0 but it needs exactly 1 edition-less card.
fn consumable_targets_range(key: &str) -> (u8, u8) {
    if key == "c_aura" {
        return (1, 1);
    }
    consumable_by_key(key)
        .map(|(m, _)| {
            (
                m.max_highlighted,
                m.min_highlighted.max(1) * (m.max_highlighted > 0) as u8,
            )
        })
        .unwrap_or((0, 0))
}

fn get_usize(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| format!("action missing integer field {key:?}"))
}

fn get_indices(v: &Value, key: &str) -> Result<Vec<usize>, String> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| {
                x.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("non-integer entry in {key:?}"))
            })
            .collect(),
        Some(_) => Err(format!("{key:?} must be an array of indices")),
    }
}

/// Apply one direct action (the JSON dialect emitted by
/// `legal_actions_json` / `bot_direct_action`). Errors do not mutate the
/// run (every sim action validates before mutating).
pub fn apply_action(run: &mut Run, action: &Value) -> Result<(), String> {
    let kind = action
        .get("kind")
        .and_then(Value::as_str)
        .ok_or("action missing 'kind'")?;
    let e = |err: balatro_core::run::RunError| format!("{kind}: {err}");
    match kind {
        "select_blind" => run.select_blind().map_err(e),
        "skip_blind" => run.skip_blind().map_err(e),
        "reroll_boss" => run.reroll_boss().map_err(e),
        "play" => run
            .play(&get_indices(action, "cards")?)
            .map(|_| ())
            .map_err(e),
        "discard" => run.discard(&get_indices(action, "cards")?).map_err(e),
        "cash_out" => run.cash_out().map_err(e),
        "buy_card" => run.buy_shop_item(get_usize(action, "slot")?).map_err(e),
        "buy_and_use" => run
            .buy_and_use_shop_item(get_usize(action, "slot")?, &get_indices(action, "targets")?)
            .map_err(e),
        "redeem_voucher" => run.redeem_voucher(get_usize(action, "slot")?).map_err(e),
        "buy_pack" => run.buy_pack(get_usize(action, "slot")?).map_err(e),
        "reroll" => run.reroll_shop().map_err(e),
        "leave_shop" => run.leave_shop().map_err(e),
        "use_consumable" => run
            .use_consumable(get_usize(action, "slot")?, &get_indices(action, "targets")?)
            .map_err(e),
        "sell_joker" => run.sell_joker(get_usize(action, "slot")?).map_err(e),
        "sell_consumable" => run.sell_consumable(get_usize(action, "slot")?).map_err(e),
        "pick_pack" => run
            .pick_pack_item(get_usize(action, "slot")?, &get_indices(action, "targets")?)
            .map_err(e),
        "skip_pack" => run.skip_pack().map_err(e),
        other => Err(format!("unknown action kind {other:?}")),
    }
}

/// An RL-contract action row (as a trained policy emits it), translated into
/// the direct-action dialect with card/target indices resolved EXACTLY like
/// the vec-env's `action::apply` — so mirroring the result onto the live
/// game reproduces the sim mutation bit-for-bit. Returns `Value::Null` for
/// the USE_CONSUMABLE no-op leniency (masked legal but unusable right now),
/// which `apply` treats as a wasted step.
pub fn row_direct_action(run: &Run, row: &ActionRow) -> Result<Value, String> {
    let slots = shop_slot_map(run);
    let t = row.action_type;
    Ok(match t {
        at::PLAY_HAND => {
            json!({"kind": "play", "cards": crate::action::legalize_card_selection(run, row)})
        }
        at::DISCARD => {
            json!({"kind": "discard", "cards": crate::action::legalize_card_selection(run, row)})
        }
        at::SELECT_BLIND => json!({"kind": "select_blind"}),
        at::SKIP_BLIND => json!({"kind": "skip_blind"}),
        at::CASH_OUT => json!({"kind": "cash_out"}),
        at::REROLL => json!({"kind": "reroll"}),
        at::LEAVE_SHOP => json!({"kind": "leave_shop"}),
        at::BUY_SHOP => match slots.get(row.shop_target as usize) {
            Some(ShopSlotRef::Card(i)) => json!({"kind": "buy_card", "slot": i}),
            Some(ShopSlotRef::Voucher(i)) => json!({"kind": "redeem_voucher", "slot": i}),
            Some(ShopSlotRef::Pack(i)) => json!({"kind": "buy_pack", "slot": i}),
            None => return Err(format!("BUY_SHOP target {} unmapped", row.shop_target)),
        },
        at::USE_CONSUMABLE => {
            let i = row.consumable_target as usize;
            let key = run
                .consumables()
                .get(i)
                .map(|c| c.key)
                .ok_or("USE_CONSUMABLE target out of range")?;
            let picks = legalize_card_selection_raw(run, row);
            match legalize_consumable_targets(run, key, &picks) {
                Some(targets) => json!({"kind": "use_consumable", "slot": i, "targets": targets}),
                None => Value::Null,
            }
        }
        at::SELL_JOKER => json!({"kind": "sell_joker", "slot": row.joker_target}),
        at::SELL_CONSUMABLE => json!({"kind": "sell_consumable", "slot": row.consumable_target}),
        at::PICK_PACK => {
            let i = row.pack_target as usize;
            let key = match run.pack().and_then(|p| p.items.get(i)) {
                Some(PackItem::Consumable(c)) => Some(c.key),
                Some(_) => None,
                None => return Err("PICK_PACK target out of range".into()),
            };
            let targets = match key {
                Some(k) => legalize_consumable_targets(run, k, &[])
                    .ok_or_else(|| format!("pack consumable {k} unusable despite mask"))?,
                None => Vec::new(),
            };
            json!({"kind": "pick_pack", "slot": i, "targets": targets})
        }
        at::SKIP_PACK => json!({"kind": "skip_pack"}),
        other => return Err(format!("unhandled action type {other}")),
    })
}

/// The scripted baseline's next action, translated from the RL contract row
/// into the direct-action dialect (with fully resolved card/target indices,
/// so the harness can mirror the EXACT same action onto the game).
pub fn bot_direct_action(run: &Run) -> Result<Value, String> {
    let slots = shop_slot_map(run);
    let masks = encode_masks(run, &slots);
    let row = bot_action(run, &masks, &slots);
    let t = row.action_type;
    let picks = crate::action::legalize_card_selection(run, &row);
    Ok(match t {
        at::PLAY_HAND => json!({"kind": "play", "cards": picks}),
        at::DISCARD => json!({"kind": "discard", "cards": picks}),
        at::SELECT_BLIND => json!({"kind": "select_blind"}),
        at::SKIP_BLIND => json!({"kind": "skip_blind"}),
        at::CASH_OUT => json!({"kind": "cash_out"}),
        at::REROLL => json!({"kind": "reroll"}),
        at::LEAVE_SHOP => json!({"kind": "leave_shop"}),
        at::BUY_SHOP => match slots.get(row.shop_target as usize) {
            Some(ShopSlotRef::Card(i)) => json!({"kind": "buy_card", "slot": i}),
            Some(ShopSlotRef::Voucher(i)) => json!({"kind": "redeem_voucher", "slot": i}),
            Some(ShopSlotRef::Pack(i)) => json!({"kind": "buy_pack", "slot": i}),
            None => return Err(format!("bot BUY_SHOP target {} unmapped", row.shop_target)),
        },
        at::USE_CONSUMABLE => {
            let i = row.consumable_target as usize;
            let key = run
                .consumables()
                .get(i)
                .map(|c| c.key)
                .ok_or("bot USE_CONSUMABLE out of range")?;
            let targets = legalize_consumable_targets(run, key, &[])
                .ok_or_else(|| format!("bot consumable {key} unusable"))?;
            json!({"kind": "use_consumable", "slot": i, "targets": targets})
        }
        at::SELL_JOKER => json!({"kind": "sell_joker", "slot": row.joker_target}),
        at::SELL_CONSUMABLE => json!({"kind": "sell_consumable", "slot": row.consumable_target}),
        at::PICK_PACK => {
            let i = row.pack_target as usize;
            let key = match run.pack().and_then(|p| p.items.get(i)) {
                Some(PackItem::Consumable(c)) => Some(c.key),
                Some(_) => None,
                None => return Err("bot PICK_PACK out of range".into()),
            };
            let targets = match key {
                Some(k) => legalize_consumable_targets(run, k, &[])
                    .ok_or_else(|| format!("bot pack consumable {k} unusable"))?,
                None => Vec::new(),
            };
            json!({"kind": "pick_pack", "slot": i, "targets": targets})
        }
        at::SKIP_PACK => json!({"kind": "skip_pack"}),
        other => return Err(format!("bot emitted unhandled action type {other}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_run_state_json_shape() {
        let run = Run::new("P5TESTAA");
        let v = state_json(&run);
        assert_eq!(v["state"], "BLIND_SELECT");
        assert_eq!(v["ante_num"], 1);
        assert_eq!(v["round_num"], 0);
        assert_eq!(v["money"], 4);
        assert_eq!(v["won"], false);
        assert_eq!(v["seed"], "P5TESTAA");
        assert_eq!(v["cards"]["count"], 52);
        assert_eq!(v["hand"]["count"], 0);
        assert_eq!(v["blinds"]["small"]["status"], "SELECT");
        assert_eq!(v["blinds"]["small"]["score"], 300);
        assert_eq!(v["blinds"]["big"]["score"], 450);
        assert_eq!(v["blinds"]["big"]["status"], "UPCOMING");
        assert_eq!(v["hands"]["Pair"]["chips"], 10.0);
        assert_eq!(v["hands"]["Pair"]["level"], 1);
        // Every deck card is a valid front key.
        for c in v["cards"]["cards"].as_array().unwrap() {
            let key = c["key"].as_str().unwrap();
            assert_eq!(key.len(), 3);
        }
    }

    #[test]
    fn direct_actions_drive_a_round() {
        let mut run = Run::new("P5TESTAA");
        apply_action(&mut run, &json!({"kind": "select_blind"})).unwrap();
        let v = state_json(&run);
        assert_eq!(v["state"], "SELECTING_HAND");
        assert_eq!(v["hand"]["count"], 8);
        assert_eq!(v["blinds"]["small"]["status"], "CURRENT");
        // Legal actions include play + discard.
        let acts = legal_actions_json(&run);
        let kinds: Vec<&str> = acts
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["kind"].as_str().unwrap())
            .collect();
        assert!(kinds.contains(&"play"));
        assert!(kinds.contains(&"discard"));
        // Errors don't mutate.
        let err = apply_action(&mut run, &json!({"kind": "play", "cards": [0, 99]}));
        assert!(err.is_err());
        assert_eq!(state_json(&run)["hand"]["count"], 8);
        apply_action(&mut run, &json!({"kind": "play", "cards": [0, 1, 2, 3, 4]})).unwrap();
        assert!(run.hands_played_this_round() == 1 || run.state() != State::SelectingHand);
    }

    #[test]
    fn bot_direct_action_runs_to_round_eval() {
        let mut run = Run::new("P5TESTAA");
        for _ in 0..200 {
            match run.state() {
                State::GameOver | State::Won => break,
                _ => {}
            }
            if run.ante() > 1 {
                break;
            }
            let a = bot_direct_action(&run).unwrap();
            apply_action(&mut run, &a).unwrap();
        }
        assert!(run.ante() > 1 || matches!(run.state(), State::GameOver | State::Won));
    }
}
