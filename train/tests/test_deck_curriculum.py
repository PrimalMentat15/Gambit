"""Pure-function tests for the tiered deck-mixture curriculum. No torch."""

import pytest

import balatro_train.encoding as E
from balatro_train.deck_curriculum import (
    DeckCurriculumConfig,
    active_decks,
    deck_weights,
)


def test_tiers_unlock_in_order():
    cfg = DeckCurriculumConfig()
    assert active_decks(0, cfg) == [int(E.Deck.RED)]
    # Just before the first unlock: still only Red.
    assert active_decks(cfg.tier_steps[0] - 1, cfg) == [int(E.Deck.RED)]
    # After each unlock the active set only grows.
    prev = active_decks(0, cfg)
    for step in cfg.tier_steps:
        cur = active_decks(step, cfg)
        assert set(prev) <= set(cur), "a tier unlock removed a deck"
        prev = cur
    # Past the last unlock: all 15.
    assert active_decks(cfg.tier_steps[-1], cfg) == list(range(E.N_DECKS))


def test_plasma_unlocks_last_and_alone():
    cfg = DeckCurriculumConfig()
    before_last = active_decks(cfg.tier_steps[-1] - 1, cfg)
    assert int(E.Deck.PLASMA) not in before_last, "Plasma should be the final tier"
    assert set(active_decks(cfg.tier_steps[-1], cfg)) - set(before_last) == {int(E.Deck.PLASMA)}


def test_weights_are_a_normalized_distribution_over_active_decks():
    cfg = DeckCurriculumConfig()
    step = cfg.tier_steps[0]
    w = deck_weights(step, cfg, {})
    assert len(w) == E.N_DECKS
    assert abs(sum(w) - 1.0) < 1e-9
    live = set(active_decks(step, cfg))
    for d in range(E.N_DECKS):
        if d in live:
            assert w[d] > 0.0
        else:
            assert w[d] == 0.0, f"inactive deck {d} got weight"


def test_priority_leans_toward_weak_decks():
    cfg = DeckCurriculumConfig()
    step = cfg.tier_steps[0]
    # Two live decks with very different win rates.
    wr = {int(E.Deck.RED): 0.95, int(E.Deck.BLUE): 0.20}
    w = deck_weights(step, cfg, wr)
    assert w[int(E.Deck.BLUE)] > w[int(E.Deck.RED)]


def test_uniform_mix_guarantees_a_floor():
    # With uniform_mix, even a fully-solved deck keeps at least its uniform
    # share, so nothing starves.
    cfg = DeckCurriculumConfig(uniform_mix=0.3)
    step = cfg.tier_steps[0]
    live = active_decks(step, cfg)
    wr = {d: 1.0 for d in live}  # everything solved
    wr[live[0]] = 0.0            # except one
    w = deck_weights(step, cfg, wr)
    floor = cfg.uniform_mix / len(live)
    for d in live:
        assert w[d] >= floor - 1e-9, f"deck {d} fell below the uniform floor"


def test_pure_priority_when_uniform_mix_zero():
    cfg = DeckCurriculumConfig(uniform_mix=0.0)
    step = cfg.tier_steps[0]
    live = active_decks(step, cfg)
    # A solved deck with floor -> small but nonzero; a fully-weak deck -> large.
    wr = {d: 1.0 for d in live}
    w = deck_weights(step, cfg, wr)
    # All equally solved -> uniform.
    assert max(w[d] for d in live) - min(w[d] for d in live) < 1e-9


def test_missing_winrate_treated_as_zero_max_priority():
    cfg = DeckCurriculumConfig()
    step = cfg.tier_steps[0]
    live = active_decks(step, cfg)
    # A deck absent from win_rates should get the SAME weight as one explicitly
    # at win rate 0 -- both are "no evidence, prioritize".
    known = {live[0]: 0.0}
    w = deck_weights(step, cfg, known)
    assert abs(w[live[0]] - w[live[1]]) < 1e-9


def test_config_rejects_mismatched_tier_steps():
    with pytest.raises(ValueError, match="tier_steps"):
        DeckCurriculumConfig(tiers=[[0], [1], [2]], tier_steps=[100])  # needs 2
    with pytest.raises(ValueError, match="uniform_mix"):
        DeckCurriculumConfig(uniform_mix=1.5)


def test_schedule_output_is_accepted_by_the_env():
    """The weights the schedule produces must be a distribution the env's
    set_deck_distribution accepts -- 15 non-negative numbers with positive
    sum. Uses the mock, which validates exactly as the Rust binding does."""
    from balatro_train.mock_env import MockBalatroEnv

    cfg = DeckCurriculumConfig(enabled=True)
    env = MockBalatroEnv(4)
    for step in (0, cfg.tier_steps[0], cfg.tier_steps[-1]):
        w = deck_weights(step, cfg, {})
        # Would raise if the shape/values were invalid.
        env.set_deck_distribution(w)
        assert len(w) == E.N_DECKS and all(x >= 0 for x in w) and sum(w) > 0
