"""Observation / mask / action encoding spec for the Balatro RL environment.

THIS MODULE IS THE SINGLE SOURCE OF TRUTH shared between the Python training
stack and the (future) Rust PyO3 vec-env binding.  The Rust side must produce
arrays that match the shapes, dtypes, index layouts and enum orderings defined
here EXACTLY.  Nothing in `policy.py` / `buffer.py` / `ppo.py` hardcodes a
number that lives here.

Conventions
-----------
* All observation float features are ``float32``; all ids/indices are
  ``int64``; all masks are ``bool``.
* Everything is batch-first over ``N = num_envs``; per-env shapes below omit
  the leading ``(N,)``.
* Index 0 of every id vocabulary (jokers / consumables / shop items) is
  reserved for EMPTY (empty slot / padding).  Real items start at id 1.
* "log" scalars are ``log1p(x)`` unless stated otherwise; signed quantities
  use ``sign(x) * log1p(|x|)``.
* Count arrays (``deck_counts``, ``drawpile_counts``) are RAW counts cast to
  float32 (the network normalizes internally); ``deck_aggregates`` is already
  log1p-normalized env-side (see :data:`DECK_AGG_LAYOUT`).
"""

from __future__ import annotations

import enum

import numpy as np

# ---------------------------------------------------------------------------
# Entity slot counts
# ---------------------------------------------------------------------------
HAND_MAX = 10          # max hand cards observed / selectable
JOKER_SLOTS = 6        # max joker slots observed
CONSUMABLE_SLOTS = 3   # max consumable slots observed
SHOP_SLOTS = 6         # max simultaneously-visible shop items (cards+vouchers+packs)
PACK_SLOTS = 5         # max items offered inside an opened booster pack

# Id vocabularies (embedding table sizes).  Sized with headroom from day one so
# pool expansion is a fine-tune, not an architecture change (plan §RL design).
JOKER_VOCAB = 160        # 0 = EMPTY
CONSUMABLE_VOCAB = 60    # 0 = EMPTY; tarot + planet + spectral share this table
SHOP_VOCAB = 400         # 0 = EMPTY; shared vocab across all shop item kinds
BOSS_VOCAB = 32          # boss-blind id one-hot width (0 used when not a boss)

# ---------------------------------------------------------------------------
# Card feature layout (hand tokens): F_CARD = 38
# ---------------------------------------------------------------------------


class Rank(enum.IntEnum):
    """Card rank one-hot index. 2 = 0 ... Ace = 12."""

    TWO = 0; THREE = 1; FOUR = 2; FIVE = 3; SIX = 4; SEVEN = 5; EIGHT = 6
    NINE = 7; TEN = 8; JACK = 9; QUEEN = 10; KING = 11; ACE = 12


class Suit(enum.IntEnum):
    SPADES = 0; HEARTS = 1; CLUBS = 2; DIAMONDS = 3


class Enhancement(enum.IntEnum):
    NONE = 0; BONUS = 1; MULT = 2; WILD = 3; GLASS = 4; STEEL = 5
    STONE = 6; GOLD = 7; LUCKY = 8


class Edition(enum.IntEnum):
    NONE = 0; FOIL = 1; HOLOGRAPHIC = 2; POLYCHROME = 3; NEGATIVE = 4


class Seal(enum.IntEnum):
    NONE = 0; GOLD = 1; RED = 2; BLUE = 3; PURPLE = 4


N_RANKS = 13
N_SUITS = 4
N_ENHANCEMENTS = 9
N_EDITIONS = 5
N_SEALS = 5

# Per-card feature vector layout (contiguous blocks, in this order):
#   [0:13)   rank one-hot
#   [13:17)  suit one-hot (Stone cards: all-zero suit AND all-zero rank)
#   [17:26)  enhancement one-hot (NONE is an explicit class at 17)
#   [26:31)  edition one-hot
#   [31:36)  seal one-hot
#   [36]     debuffed flag (0/1)
#   [37]     face_down flag (0/1; face-down cards hide rank/suit -> zero one-hots)
CARD_RANK_OFF = 0
CARD_SUIT_OFF = 13
CARD_ENH_OFF = 17
CARD_EDITION_OFF = 26
CARD_SEAL_OFF = 31
CARD_DEBUFFED_OFF = 36
CARD_FACEDOWN_OFF = 37
F_CARD = 38

# ---------------------------------------------------------------------------
# Joker features: ids separate (int64), float feats F_JOKER_FEAT = 11
# ---------------------------------------------------------------------------
# joker_ids : (JOKER_SLOTS,) int64, 0 = empty slot.
# joker_feats layout:
#   [0:5)  edition one-hot
#   [5]    sell value, log1p
#   [6:10) 4 generic per-joker state scalars, each sign*log1p normalized.
#          Semantics are joker-dependent (e.g. accumulated mult, counter,
#          rounds remaining, sticker flags packed by the Rust side); the
#          binding must document its per-joker packing but the SHAPE is fixed.
#   [10]   debuffed flag (0/1)
JOKER_EDITION_OFF = 0
JOKER_SELL_OFF = 5
JOKER_STATE_OFF = 6
N_JOKER_STATE = 4
JOKER_DEBUFFED_OFF = 10
F_JOKER_FEAT = 11

# ---------------------------------------------------------------------------
# Shop item features: ids separate (int64), float feats F_SHOP_FEAT = 13
# ---------------------------------------------------------------------------


class ShopKind(enum.IntEnum):
    """Kind of an item occupying a shop slot.

    NOTE (contract decision): the plan sketch said one-hot(6) but listed seven
    kinds; seven it is.  F_SHOP_FEAT sized accordingly.
    """

    JOKER = 0
    TAROT = 1
    PLANET = 2
    SPECTRAL = 3
    PLAYING_CARD = 4
    VOUCHER = 5
    PACK = 6


N_SHOP_KINDS = 7
# shop_ids : (SHOP_SLOTS,) int64 in the shared SHOP_VOCAB, 0 = empty slot.
#
# CONTRACT: during Phase.PACK the shop slots carry the OPEN PACK's contents
# instead (first PACK_SLOTS slots; cost feature 0) — shop and pack are never
# actionable simultaneously, and this gives the pack pointer real tokens to
# attend to.  Outside SHOP/PACK phases all shop slots are empty (id 0).
#
# shop_feats layout:
#   [0:7)   kind one-hot (all-zero when slot empty)
#   [7]     cost in $, log1p
#   [8:13)  edition one-hot (all-zero when N/A)
SHOP_KIND_OFF = 0
SHOP_COST_OFF = 7
SHOP_EDITION_OFF = 8
F_SHOP_FEAT = 13

# ---------------------------------------------------------------------------
# Blind features: F_BLIND = 38
# ---------------------------------------------------------------------------


class BlindKind(enum.IntEnum):
    SMALL = 0; BIG = 1; BOSS = 2


N_BLIND_KINDS = 3
# blind layout:
#   [0:3)   blind-kind one-hot (the CURRENT or UPCOMING blind)
#   [3:35)  boss id one-hot (BOSS_VOCAB=32; all-zero when not a boss blind)
#   [35]    log1p(chip requirement)
#   [36]    log1p(chips scored so far this blind)
#   [37]    blind cash reward ($), log1p
BLIND_KIND_OFF = 0
BLIND_BOSS_OFF = 3
BLIND_REQ_OFF = 35
BLIND_SCORED_OFF = 36
BLIND_REWARD_OFF = 37
F_BLIND = 38

# ---------------------------------------------------------------------------
# Game phase + global scalar vector: F_GLOBAL = 64
# ---------------------------------------------------------------------------


class Phase(enum.IntEnum):
    BLIND_SELECT = 0   # choosing to play or skip the upcoming blind
    PLAYING = 1        # in a blind, playing/discarding hands
    ROUND_EVAL = 2     # blind beaten, cash-out screen
    SHOP = 3           # in the shop
    PACK = 4           # a booster pack is open
    GAME_OVER = 5      # terminal; never observed by the agent (auto-reset)


N_PHASES = 6
N_HAND_TYPES = 12  # poker-hand-level table rows (High Card ... Flush Five)

# global layout (indices):
#   [0]      money, sign*log1p (money can go negative via certain effects)
#   [1]      hands left this blind / 4.0
#   [2]      discards left this blind / 4.0
#   [3:12)   ante one-hot, antes 1..9 (ante>=9 clamps to the last bucket)
#   [12]     round number / 24.0
#   [13:19)  phase one-hot (Phase enum order)
#   [19:43)  hand-level table: 12 x (level-1 log1p, times-played log1p),
#            row-major in hand-type order (Rust side owns the canonical
#            hand-type order and must keep it stable)
#   [43]     current reroll cost, log1p
#   [44]     joker slots total / 6.0
#   [45]     consumable slots total / 3.0
#   [46]     full deck size, log1p
#   [47]     current hand size / 10.0
#   [48:64)  reserved, must be 0.0 (headroom for tags, vouchers, stake, ...)
GLOBAL_MONEY = 0
GLOBAL_HANDS_LEFT = 1
GLOBAL_DISCARDS_LEFT = 2
GLOBAL_ANTE_OFF = 3
N_ANTE_ONEHOT = 9
GLOBAL_ROUND = 12
GLOBAL_PHASE_OFF = 13
GLOBAL_HAND_LEVELS_OFF = 19          # 12 * 2 = 24 slots
GLOBAL_REROLL_COST = 43
GLOBAL_JOKER_SLOT_COUNT = 44
GLOBAL_CONSUMABLE_SLOT_COUNT = 45
GLOBAL_DECK_SIZE = 46
GLOBAL_HAND_SIZE = 47
GLOBAL_RESERVED_OFF = 48
F_GLOBAL = 64

# ---------------------------------------------------------------------------
# Deck composition features
# ---------------------------------------------------------------------------
# deck_counts     : (52,) raw count of each (rank, suit) in the FULL deck,
#                   index = rank * 4 + suit (Rank/Suit enum order).
# drawpile_counts : (52,) same layout, but cards REMAINING in the draw pile
#                   for the current blind (this is what makes the state
#                   effectively Markov -- card counting handed to the net).
# deck_aggregates : (18,) log1p-normalized full-deck aggregates:
#   [0:9)   enhancement counts (Enhancement enum order, incl. NONE at 0)
#   [9:13)  edition counts: FOIL, HOLOGRAPHIC, POLYCHROME, NEGATIVE
#   [13:17) seal counts: GOLD, RED, BLUE, PURPLE
#   [17]    total deck size
N_DECK_SLOTS = 52
DECK_AGG_ENH_OFF = 0
DECK_AGG_EDITION_OFF = 9
DECK_AGG_SEAL_OFF = 13
DECK_AGG_TOTAL = 17
F_DECK_AGG = 18


def deck_slot(rank: int, suit: int) -> int:
    """Index into deck_counts / drawpile_counts for (rank, suit)."""
    return int(rank) * N_SUITS + int(suit)


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------


class ActionType(enum.IntEnum):
    """Level-1 discrete action head (25 slots; trailing slots reserved).

    The Rust binding must keep these indices STABLE forever (checkpoints and
    rollout data bake them in).  New action types claim RESERVED_* slots.
    """

    PLAY_HAND = 0        # play the selected card subset  (cards: 1..5)
    DISCARD = 1          # discard the selected subset    (cards: 1..5)
    SELECT_BLIND = 2     # blind-select phase: play the upcoming blind
    SKIP_BLIND = 3       # blind-select phase: skip (small/big only, never boss)
    CASH_OUT = 4         # round-eval phase: collect winnings, go to shop
    BUY_SHOP = 5         # buy shop item             (shop_target)
    REROLL = 6           # reroll the shop
    LEAVE_SHOP = 7       # leave shop, advance to next blind-select
    USE_CONSUMABLE = 8   # use consumable            (consumable_target, cards: 0..5)
    SELL_JOKER = 9       # sell a joker              (joker_target)
    SELL_CONSUMABLE = 10  # sell a consumable        (consumable_target)
    PICK_PACK = 11       # take an item from the open pack (pack_target)
    SKIP_PACK = 12       # close the open pack without taking (remaining picks)
    MOVE_JOKER = 13      # RESERVED for later curriculum stage; never legal in
    #                      phase 1 (sim auto-orders jokers).  When enabled it
    #                      will consume joker_target (+ a second pointer, TBD).
    RESERVED_14 = 14
    RESERVED_15 = 15
    RESERVED_16 = 16
    RESERVED_17 = 17
    RESERVED_18 = 18
    RESERVED_19 = 19
    RESERVED_20 = 20
    RESERVED_21 = 21
    RESERVED_22 = 22
    RESERVED_23 = 23
    RESERVED_24 = 24


N_ACTION_TYPES = 25

# Which conditional heads each action type consumes.  The policy samples ONLY
# the heads listed here for the sampled type; all other action-dict fields are
# filled with -1 and MUST be ignored by the env.
ACTION_NEEDS_CARDS = frozenset(
    {ActionType.PLAY_HAND, ActionType.DISCARD, ActionType.USE_CONSUMABLE}
)
ACTION_NEEDS_JOKER = frozenset({ActionType.SELL_JOKER})
ACTION_NEEDS_CONSUMABLE = frozenset(
    {ActionType.USE_CONSUMABLE, ActionType.SELL_CONSUMABLE}
)
ACTION_NEEDS_SHOP = frozenset({ActionType.BUY_SHOP})
ACTION_NEEDS_PACK = frozenset({ActionType.PICK_PACK})

# Card-subset sub-action: autoregressive pointer over hand tokens with STOP.
MAX_CARD_PICKS = 5
CARD_STOP_INDEX = HAND_MAX  # STOP is the (HAND_MAX+1)-th choice of each sub-step

# Minimum number of card picks before STOP becomes legal, per action type.
# PLAY/DISCARD require >=1 card; USE_CONSUMABLE allows 0 (untargeted
# consumables: the agent samples STOP immediately).
CARD_PICK_MIN = {
    ActionType.PLAY_HAND: 1,
    ActionType.DISCARD: 1,
    ActionType.USE_CONSUMABLE: 0,
}

# ---------------------------------------------------------------------------
# Array specs (per-env shapes; prepend (N,) for the batched VecEnv arrays)
# ---------------------------------------------------------------------------
OBS_SPEC: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "hand": ((HAND_MAX, F_CARD), np.dtype(np.float32)),
    "hand_len": ((), np.dtype(np.int64)),
    "joker_ids": ((JOKER_SLOTS,), np.dtype(np.int64)),
    "joker_feats": ((JOKER_SLOTS, F_JOKER_FEAT), np.dtype(np.float32)),
    "consumable_ids": ((CONSUMABLE_SLOTS,), np.dtype(np.int64)),
    "consumables_len": ((), np.dtype(np.int64)),
    "shop_ids": ((SHOP_SLOTS,), np.dtype(np.int64)),
    "shop_feats": ((SHOP_SLOTS, F_SHOP_FEAT), np.dtype(np.float32)),
    "blind": ((F_BLIND,), np.dtype(np.float32)),
    "global": ((F_GLOBAL,), np.dtype(np.float32)),
    "deck_counts": ((N_DECK_SLOTS,), np.dtype(np.float32)),
    "deck_aggregates": ((F_DECK_AGG,), np.dtype(np.float32)),
    "drawpile_counts": ((N_DECK_SLOTS,), np.dtype(np.float32)),
}

MASK_SPEC: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "action_type_mask": ((N_ACTION_TYPES,), np.dtype(np.bool_)),
    "card_select_mask": ((HAND_MAX,), np.dtype(np.bool_)),
    "joker_target_mask": ((JOKER_SLOTS,), np.dtype(np.bool_)),
    "consumable_target_mask": ((CONSUMABLE_SLOTS,), np.dtype(np.bool_)),
    "shop_target_mask": ((SHOP_SLOTS,), np.dtype(np.bool_)),
    "pack_target_mask": ((PACK_SLOTS,), np.dtype(np.bool_)),
}

ACTION_SPEC: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "action_type": ((), np.dtype(np.int64)),
    # picked hand indices, -1 padded past n_cards
    "cards": ((MAX_CARD_PICKS,), np.dtype(np.int64)),
    "n_cards": ((), np.dtype(np.int64)),
    "joker_target": ((), np.dtype(np.int64)),       # -1 when unused
    "consumable_target": ((), np.dtype(np.int64)),  # -1 when unused
    "shop_target": ((), np.dtype(np.int64)),        # -1 when unused
    "pack_target": ((), np.dtype(np.int64)),        # -1 when unused
}


def validate_batch(
    spec: dict[str, tuple[tuple[int, ...], np.dtype]],
    batch: dict[str, np.ndarray],
    num_envs: int,
    what: str = "batch",
) -> None:
    """Assert a batched dict matches a spec exactly (keys, shapes, dtypes)."""
    missing = spec.keys() - batch.keys()
    extra = batch.keys() - spec.keys()
    if missing or extra:
        raise ValueError(f"{what}: missing keys {sorted(missing)}, extra keys {sorted(extra)}")
    for key, (shape, dtype) in spec.items():
        arr = batch[key]
        want = (num_envs, *shape)
        if arr.shape != want:
            raise ValueError(f"{what}[{key}]: shape {arr.shape} != {want}")
        if arr.dtype != dtype:
            raise ValueError(f"{what}[{key}]: dtype {arr.dtype} != {dtype}")
