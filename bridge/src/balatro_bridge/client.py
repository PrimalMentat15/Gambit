"""Typed client for the balatrobot JSON-RPC 2.0 API.

This is a thin adapter over the upstream generic client
(`balatrobot.cli.client.BalatroClient`, from the `balatrobot` PyPI package,
pinned to the same release as the installed Lua mod). The upstream client
handles the JSON-RPC envelope, request ids, and error unwrapping; this module
adds one typed method per API endpoint, client-side validation of
"exactly one of" parameter groups, and a readiness poll.

API reference: https://coder.github.io/balatrobot/latest/api/
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, cast

import httpx
from balatrobot import APIError, BalatroClient

from balatro_bridge.types import (
    Deck,
    Edition,
    Enhancement,
    GameState,
    HealthResult,
    PathResult,
    Seal,
    Stake,
)

__all__ = ["APIError", "BalatroBridgeClient"]


def _exactly_one(**kwargs: Any) -> None:
    given = [name for name, value in kwargs.items() if value is not None]
    if len(given) != 1:
        names = ", ".join(kwargs)
        raise ValueError(
            f"exactly one of ({names}) must be given, got: {given or 'none'}"
        )


class BalatroBridgeClient:
    """Typed, synchronous client for a running balatrobot game instance."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12346,
        timeout: float = 30.0,
    ) -> None:
        self._client = BalatroClient(host=host, port=port, timeout=timeout)

    @property
    def url(self) -> str:
        return self._client.url

    # --- transport ----------------------------------------------------------

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Raw JSON-RPC call (escape hatch). Raises APIError on API errors."""
        return self._client.call(method, params)

    def wait_until_ready(self, timeout: float = 120.0, interval: float = 2.0) -> None:
        """Poll `health` until the game answers or `timeout` seconds elapse.

        Useful right after launching the game: Proton + Balatro boot can take
        a while before the mod's HTTP server starts listening.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.health()
                return
            except (httpx.HTTPError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"balatrobot API at {self.url} not ready after {timeout:.0f}s"
                    ) from None
                time.sleep(interval)

    # --- meta ---------------------------------------------------------------

    def health(self) -> HealthResult:
        """Health check. Returns {"status": "ok"}."""
        return cast(HealthResult, self.call("health"))

    def gamestate(self) -> GameState:
        """Get the complete current game state."""
        return cast(GameState, self.call("gamestate"))

    def rpc_discover(self) -> dict[str, Any]:
        """Return the OpenRPC specification for the running API."""
        return cast(dict[str, Any], self.call("rpc.discover"))

    # --- run lifecycle --------------------------------------------------------

    def start(self, deck: Deck, stake: Stake, seed: str | None = None) -> GameState:
        """Start a new run (requires MENU state; result state: BLIND_SELECT)."""
        params: dict[str, Any] = {"deck": deck, "stake": stake}
        if seed is not None:
            params["seed"] = seed
        return cast(GameState, self.call("start", params))

    def menu(self) -> GameState:
        """Return to the main menu from any state."""
        return cast(GameState, self.call("menu"))

    def save(self, path: str) -> PathResult:
        """Save the current run to a file (path as seen by the game process)."""
        return cast(PathResult, self.call("save", {"path": path}))

    def load(self, path: str) -> PathResult:
        """Load a saved run from a file."""
        return cast(PathResult, self.call("load", {"path": path}))

    # --- blind selection ------------------------------------------------------

    def select(self) -> GameState:
        """Select the current blind (BLIND_SELECT -> SELECTING_HAND)."""
        return cast(GameState, self.call("select"))

    def skip(self) -> GameState:
        """Skip the current blind (Small or Big only; requires BLIND_SELECT)."""
        return cast(GameState, self.call("skip"))

    # --- playing --------------------------------------------------------------

    def play(self, cards: Sequence[int]) -> GameState:
        """Play 1-5 cards from hand by 0-based index (requires SELECTING_HAND)."""
        return cast(GameState, self.call("play", {"cards": list(cards)}))

    def discard(self, cards: Sequence[int]) -> GameState:
        """Discard cards from hand by 0-based index (requires SELECTING_HAND)."""
        return cast(GameState, self.call("discard", {"cards": list(cards)}))

    def rearrange(
        self,
        hand: Sequence[int] | None = None,
        jokers: Sequence[int] | None = None,
        consumables: Sequence[int] | None = None,
    ) -> GameState:
        """Rearrange one area; pass exactly one permutation of current indices."""
        _exactly_one(hand=hand, jokers=jokers, consumables=consumables)
        params: dict[str, Any] = {}
        if hand is not None:
            params["hand"] = list(hand)
        if jokers is not None:
            params["jokers"] = list(jokers)
        if consumables is not None:
            params["consumables"] = list(consumables)
        return cast(GameState, self.call("rearrange", params))

    def use(self, consumable: int, cards: Sequence[int] | None = None) -> GameState:
        """Use a consumable by 0-based index, optionally targeting hand cards."""
        params: dict[str, Any] = {"consumable": consumable}
        if cards is not None:
            params["cards"] = list(cards)
        return cast(GameState, self.call("use", params))

    # --- round / shop -----------------------------------------------------------

    def cash_out(self) -> GameState:
        """Cash out round rewards (ROUND_EVAL -> SHOP)."""
        return cast(GameState, self.call("cash_out"))

    def next_round(self) -> GameState:
        """Leave the shop (SHOP -> BLIND_SELECT)."""
        return cast(GameState, self.call("next_round"))

    def reroll(self) -> GameState:
        """Reroll shop items (costs money; requires SHOP)."""
        return cast(GameState, self.call("reroll"))

    def buy(
        self,
        card: int | None = None,
        voucher: int | None = None,
        pack: int | None = None,
    ) -> GameState:
        """Buy exactly one of: shop card, voucher, or pack (0-based index)."""
        _exactly_one(card=card, voucher=voucher, pack=pack)
        params = {
            k: v
            for k, v in {"card": card, "voucher": voucher, "pack": pack}.items()
            if v is not None
        }
        return cast(GameState, self.call("buy", params))

    def pack(
        self,
        card: int | None = None,
        targets: Sequence[int] | None = None,
        skip: bool | None = None,
    ) -> GameState:
        """Pick a card from an opened booster pack, or skip it.

        Requires SMODS_BOOSTER_OPENED. Pass `card` (with optional `targets`
        for consumables that need them) or `skip=True`.
        """
        _exactly_one(card=card, skip=skip)
        params: dict[str, Any] = {}
        if card is not None:
            params["card"] = card
        if targets is not None:
            params["targets"] = list(targets)
        if skip is not None:
            params["skip"] = skip
        return cast(GameState, self.call("pack", params))

    def sell(
        self, joker: int | None = None, consumable: int | None = None
    ) -> GameState:
        """Sell exactly one of: a joker or a consumable (0-based index)."""
        _exactly_one(joker=joker, consumable=consumable)
        params = {
            k: v
            for k, v in {"joker": joker, "consumable": consumable}.items()
            if v is not None
        }
        return cast(GameState, self.call("sell", params))

    # --- debug / testing ----------------------------------------------------

    def add(
        self,
        key: str,
        seal: Seal | None = None,
        edition: Edition | None = None,
        enhancement: Enhancement | None = None,
        eternal: bool | None = None,
        perishable: int | None = None,
        rental: bool | None = None,
    ) -> GameState:
        """Add a card by key (debug/testing), e.g. 'j_joker', 'c_fool', 'H_A'."""
        params: dict[str, Any] = {"key": key}
        for name, value in (
            ("seal", seal),
            ("edition", edition),
            ("enhancement", enhancement),
            ("eternal", eternal),
            ("perishable", perishable),
            ("rental", rental),
        ):
            if value is not None:
                params[name] = value
        return cast(GameState, self.call("add", params))

    def set(
        self,
        money: int | None = None,
        chips: int | None = None,
        ante: int | None = None,
        round: int | None = None,
        hands: int | None = None,
        discards: int | None = None,
        shop: bool | None = None,
    ) -> GameState:
        """Set in-game values (debug/testing); at least one argument required."""
        params = {
            k: v
            for k, v in {
                "money": money,
                "chips": chips,
                "ante": ante,
                "round": round,
                "hands": hands,
                "discards": discards,
                "shop": shop,
            }.items()
            if v is not None
        }
        if not params:
            raise ValueError("set() requires at least one argument")
        return cast(GameState, self.call("set", params))

    def screenshot(self, path: str) -> PathResult:
        """Save a PNG screenshot of the game to `path`."""
        return cast(PathResult, self.call("screenshot", {"path": path}))
