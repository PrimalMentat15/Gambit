"""Types and enums for the balatrobot JSON-RPC API (v1.5.2).

Mirrors the upstream OpenRPC spec / docs:
- https://coder.github.io/balatrobot/latest/api/
- mod source of truth: ~/.local/share/Balatro/Mods/balatrobot/src/lua/utils/openrpc.json
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

# --- Enums (string literals, matching the API exactly) ---------------------

Deck = Literal[
    "RED", "BLUE", "YELLOW", "GREEN", "BLACK",
    "MAGIC", "NEBULA", "GHOST", "ABANDONED", "CHECKERED",
    "ZODIAC", "PAINTED", "ANAGLYPH", "PLASMA", "ERRATIC",
]

Stake = Literal[
    "WHITE", "RED", "GREEN", "BLACK", "BLUE", "PURPLE", "ORANGE", "GOLD",
]

State = Literal[
    "MENU",
    "BLIND_SELECT",
    "SELECTING_HAND",
    "ROUND_EVAL",
    "SHOP",
    "SMODS_BOOSTER_OPENED",
    "GAME_OVER",
]

Seal = Literal["RED", "BLUE", "GOLD", "PURPLE"]
Edition = Literal["FOIL", "HOLO", "POLYCHROME", "NEGATIVE"]
Enhancement = Literal[
    "BONUS", "MULT", "WILD", "GLASS", "STEEL", "STONE", "GOLD", "LUCKY",
]

BlindType = Literal["SMALL", "BIG", "BOSS"]
BlindStatus = Literal["SELECT", "CURRENT", "UPCOMING", "DEFEATED", "SKIPPED"]

Suit = Literal["H", "D", "C", "S"]
Rank = Literal["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

CardSet = Literal[
    "DEFAULT", "ENHANCED", "JOKER", "TAROT", "PLANET", "SPECTRAL",
    "VOUCHER", "BOOSTER",
]

# --- Response shapes (partial typing; the wire format is JSON) -------------


class CardValue(TypedDict, total=False):
    suit: Suit
    rank: Rank
    effect: str


class CardModifier(TypedDict, total=False):
    seal: Seal | None
    edition: Edition | None
    enhancement: Enhancement | None
    eternal: bool
    perishable: int | None
    rental: bool


class CardState(TypedDict, total=False):
    debuff: bool
    hidden: bool
    highlight: bool


class CardCost(TypedDict, total=False):
    sell: int
    buy: int


class Card(TypedDict, total=False):
    id: int
    key: str
    set: CardSet
    label: str
    value: CardValue
    modifier: CardModifier
    state: CardState
    cost: CardCost


class Area(TypedDict, total=False):
    """A card area (hand, jokers, consumables, shop, ...)."""

    count: int
    limit: int
    highlighted_limit: int
    cards: list[Card]


class Round(TypedDict, total=False):
    hands_left: int
    hands_played: int
    discards_left: int
    discards_used: int
    reroll_cost: int
    chips: int


class Blind(TypedDict, total=False):
    type: BlindType
    status: BlindStatus
    name: str
    effect: str
    score: int
    tag_name: str
    tag_effect: str


class PokerHandInfo(TypedDict, total=False):
    order: int
    level: int
    chips: int
    mult: int
    played: int
    played_this_round: int
    example: list[list[Any]]


class GameState(TypedDict, total=False):
    """Complete game state returned by most methods."""

    state: State
    round_num: int
    ante_num: int
    money: int
    deck: Deck
    stake: Stake
    seed: str
    won: bool
    used_vouchers: dict[str, Any]
    hands: dict[str, PokerHandInfo]
    round: Round
    blinds: dict[str, Blind]
    jokers: Area
    consumables: Area
    cards: Area
    hand: Area
    shop: Area
    vouchers: Area
    packs: Area
    pack: Area


class PathResult(TypedDict, total=False):
    """Result of save/load/screenshot."""

    success: bool
    path: str


class HealthResult(TypedDict, total=False):
    status: str
