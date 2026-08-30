"""Tiered deck-mixture curriculum for the P7 multi-deck run.

Two orthogonal ideas, kept as PURE functions so they unit-test without torch
(the same shape as ``ppo.shaping_beta_at`` / ``snapshot_fraction_at``):

1. **Tiers** -- decks are introduced a group at a time, in order of how far
   they sit from Red, so early per-deck data density stays high instead of
   being split 15 ways from step 0. :func:`active_decks` says which are live
   at a given step.

2. **Prioritized reweighting** -- among the live decks, sampling weight leans
   toward the ones with the lowest win rate, so training spends its budget
   where the policy is weakest. :func:`deck_weights` produces the 15-length
   vector ``set_deck_distribution`` wants. A uniform blend guarantees no live
   deck starves, however well the others are doing.

Neither function reads global state; the trainer passes the step and the
per-deck win rates in, and pushes the result to the env.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from balatro_train import encoding as E

# Deck indices grouped by distance from Red (encoding.Deck order). Red is
# always live; each later tier is unlocked at the matching `tier_steps` entry.
#   1: resource tweaks (hands/discards/money/slots) -- nearest to Red
#   2: deck-composition changes (already visible in deck_counts / drawpile)
#   3: economy / pool changes (starting vouchers, spectral rate, extra tags)
#   4: Plasma alone -- it inverts the scoring heuristic, so add it last and by
#      itself to keep its effect on the rest measurable
_DEFAULT_TIERS = [
    [E.Deck.RED],
    [E.Deck.BLUE, E.Deck.YELLOW, E.Deck.GREEN, E.Deck.BLACK, E.Deck.PAINTED],
    [E.Deck.ABANDONED, E.Deck.CHECKERED, E.Deck.ERRATIC],
    [E.Deck.MAGIC, E.Deck.NEBULA, E.Deck.ZODIAC, E.Deck.GHOST, E.Deck.ANAGLYPH],
    [E.Deck.PLASMA],
]


@dataclass
class DeckCurriculumConfig:
    """Config for the tiered deck mixture.

    ``tiers`` is a list of deck-index groups; ``tiers[0]`` is always live and
    ``tier_steps[i]`` is the global step at which ``tiers[i+1]`` turns on, so
    ``len(tier_steps) == len(tiers) - 1``.
    """

    enabled: bool = False
    tiers: list = field(default_factory=lambda: [[int(d) for d in t] for t in _DEFAULT_TIERS])
    tier_steps: list = field(default_factory=lambda: [500_000_000, 1_000_000_000,
                                                      1_500_000_000, 2_000_000_000])
    #: weight_d proportional to (max(0, 1 - winrate_d) + floor) ** alpha.
    #: alpha > 1 sharpens the lean toward weak decks; floor keeps a solved deck
    #: (winrate ~ 1) from dropping to zero weight.
    priority_alpha: float = 1.0
    priority_floor: float = 0.05
    #: Fraction of the mixture spread uniformly over live decks, so the weakest
    #: deck can never crowd the rest out entirely. 0 = pure priority.
    uniform_mix: float = 0.2

    def __post_init__(self) -> None:
        if len(self.tier_steps) != len(self.tiers) - 1:
            raise ValueError(
                f"tier_steps has {len(self.tier_steps)} entries, expected "
                f"{len(self.tiers) - 1} (one per tier after the first)"
            )
        if not (0.0 <= self.uniform_mix <= 1.0):
            raise ValueError("uniform_mix must be in [0, 1]")


def active_decks(step: int, cfg: DeckCurriculumConfig) -> list[int]:
    """Sorted deck indices live at ``step`` (tier 0 always included)."""
    live = set(cfg.tiers[0])
    for unlock, group in zip(cfg.tier_steps, cfg.tiers[1:]):
        if step >= unlock:
            live.update(group)
    return sorted(live)


def deck_weights(step: int, cfg: DeckCurriculumConfig,
                 win_rates: dict[int, float] | None = None) -> list[float]:
    """A 15-length weight vector for ``set_deck_distribution``.

    Inactive decks get exactly 0. Among the live decks the weight blends a
    priority term (lower win rate -> more weight) with a uniform floor. The
    result is normalized to sum to 1; ``win_rates`` may omit decks (treated as
    win rate 0, i.e. maximum priority -- the right default for a deck that has
    just unlocked and has no history yet).
    """
    win_rates = win_rates or {}
    live = active_decks(step, cfg)
    n = len(live)
    priority = {}
    for d in live:
        wr = win_rates.get(d, 0.0)
        priority[d] = (max(0.0, 1.0 - wr) + cfg.priority_floor) ** cfg.priority_alpha
    total = sum(priority.values())

    w = [0.0] * E.N_DECKS
    for d in live:
        q = priority[d] / total if total > 0 else 1.0 / n
        w[d] = (1.0 - cfg.uniform_mix) * q + cfg.uniform_mix / n
    # Already sums to 1 by construction, but renormalize against float drift.
    s = sum(w)
    return [x / s for x in w]
