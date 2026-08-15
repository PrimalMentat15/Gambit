"""
Autoregressive card-selection head

The two properties that matter:

1. **Every sampled action is legal.** 1-5 cards, no duplicates, only positions
   that exist in hand. If this holds, the clamping in ``BalatroActionMapper``
   never fires and action projection stops distorting credit assignment.
2. **log_prob reproduces exactly.** PPO's ratio compares the log-probability
   stored during rollout against one recomputed in ``evaluate_actions``. If the
   sampling and scoring paths disagree even slightly, the ratio is wrong and
   training is silently corrupted -- with no error to notice.

    venv/Scripts/python.exe tests/test_autoregressive.py
"""

import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gymnasium import spaces  # noqa: E402

from ai.policies import AutoregressiveCardPolicy  # noqa: E402

MAX_ACTIONS = 3
MAX_CARDS = 8
MAX_PICKS = 5
STOP = MAX_CARDS
OBS = 216


def make_policy():
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS,), dtype=np.float32)
    act_space = spaces.MultiDiscrete([MAX_ACTIONS] + [MAX_CARDS + 1] * MAX_PICKS)
    return AutoregressiveCardPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        max_cards=MAX_CARDS,
        max_picks=MAX_PICKS,
    )


def make_masks(batch, hand_size=MAX_CARDS, allowed_types=(0, 1, 2)):
    """Mirror BalatroEnv._create_action_mask"""
    types = [i in allowed_types for i in range(MAX_ACTIONS)]
    block = [i < hand_size for i in range(MAX_CARDS)] + [True]
    row = types + block * MAX_PICKS
    return np.array([row] * batch, dtype=bool)


def decode(action):
    """Cards chosen, stopping at the first STOP -- same rule as the mapper"""
    cards = []
    for value in action[1:]:
        if int(value) >= STOP:
            break
        cards.append(int(value))
    return cards


def test_actions_are_always_legal():
    """No sampled action can violate Balatro's selection rules"""
    policy = make_policy()
    batch = 512
    obs = th.randn(batch, OBS)

    for hand_size in [1, 2, 4, 5, 8]:
        masks = make_masks(batch, hand_size=hand_size)
        with th.no_grad():
            actions, _, _ = policy(obs, action_masks=masks)
        actions = actions.numpy()

        for action in actions:
            action_type = int(action[0])
            cards = decode(action)

            if action_type == 0:  # SELECT_HAND
                assert 1 <= len(cards) <= MAX_PICKS, f"illegal count {len(cards)}"
                assert len(set(cards)) == len(cards), f"duplicate cards {cards}"
                assert all(0 <= c < hand_size for c in cards), (
                    f"card outside a {hand_size}-card hand: {cards}")
            else:
                # PLAY/DISCARD take no card params; slots must be canonical STOP
                assert cards == [], f"{action_type} carried cards {cards}"
                assert all(int(v) == STOP for v in action[1:]), action

            # Everything after a STOP must also be STOP
            slots = [int(v) for v in action[1:]]
            if STOP in slots:
                first = slots.index(STOP)
                assert all(v == STOP for v in slots[first:]), slots

    print(f"legality OK: {batch} samples x 5 hand sizes, "
          f"no illegal count/duplicate/out-of-hand selection")


def test_respects_action_type_mask():
    """An action type the env says is illegal is never sampled"""
    policy = make_policy()
    batch = 256
    obs = th.randn(batch, OBS)

    for allowed in [(0,), (1,), (2,), (1, 2)]:
        masks = make_masks(batch, allowed_types=allowed)
        with th.no_grad():
            actions, _, _ = policy(obs, action_masks=masks)
        got = set(int(a[0]) for a in actions.numpy())
        assert got <= set(allowed), f"sampled {got}, only {allowed} were legal"

    print("action-type masking OK: never sampled a masked-out type")


def test_log_prob_is_reproducible():
    """
    Sampling and PPO's recompute must agree exactly

    This is the property that, if broken, corrupts training with no visible
    error -- PPO would compute its ratio against a log-prob the policy never
    actually assigned.
    """
    policy = make_policy()
    batch = 256
    obs = th.randn(batch, OBS)
    masks = make_masks(batch, hand_size=6)

    with th.no_grad():
        actions, values, log_prob = policy(obs, action_masks=masks)
        values2, log_prob2, entropy = policy.evaluate_actions(
            obs, actions, action_masks=th.as_tensor(masks)
        )

    max_diff = (log_prob - log_prob2).abs().max().item()
    assert max_diff < 1e-5, f"log_prob mismatch of {max_diff}"
    assert th.allclose(values, values2, atol=1e-6), "value head disagreed"
    assert entropy is not None and entropy.shape == (batch,), entropy
    assert th.isfinite(log_prob).all(), "non-finite log_prob"
    assert th.isfinite(entropy).all(), "non-finite entropy"

    print(f"log_prob reproducible: max |sample - recompute| = {max_diff:.2e}")
    print(f"entropy finite, mean {entropy.mean().item():.3f}")


def test_forced_actions_contribute_no_log_prob():
    """
    PLAY/DISCARD card slots are forced, so they must not be scored

    If forced STOPs contributed log-probability, PPO would be crediting the
    policy for a 'decision' it never had any freedom in.
    """
    policy = make_policy()
    obs = th.randn(64, OBS)

    # Only PLAY_HAND legal -> every card slot is forced to STOP
    masks = make_masks(64, allowed_types=(1,))
    with th.no_grad():
        actions, _, log_prob = policy(obs, action_masks=masks)

    # With one legal action type and all card slots forced, the only source of
    # probability is a certain choice, so log_prob must be ~0
    assert th.allclose(log_prob, th.zeros_like(log_prob), atol=1e-4), (
        f"forced action had non-zero log_prob: max {log_prob.abs().max().item()}")
    print("forced slots contribute no log_prob (max |lp| "
          f"= {log_prob.abs().max().item():.2e})")


def test_gradients_flow():
    """Both heads receive gradient, so the card head is actually trainable"""
    policy = make_policy()
    obs = th.randn(64, OBS)
    masks = make_masks(64)

    actions, _, _ = policy(obs, action_masks=masks)
    _, log_prob, entropy = policy.evaluate_actions(
        obs, actions, action_masks=th.as_tensor(masks)
    )
    loss = -(log_prob.mean() + 0.01 * entropy.mean())
    loss.backward()

    for name, module in [("action_type_net", policy.action_type_net),
                         ("card_net", policy.card_net),
                         ("card_embed", policy.card_embed)]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"{name} received no gradient"
        total = sum(float(g.abs().sum()) for g in grads)
        assert total > 0, f"{name} gradient is all zeros"
        print(f"  {name:<16} grad magnitude {total:.4f}")

    print("gradients flow to both heads")


def test_deterministic_is_stable():
    """Deterministic mode returns the same action twice"""
    policy = make_policy()
    obs = th.randn(32, OBS)
    masks = make_masks(32)

    with th.no_grad():
        a1, _, _ = policy(obs, action_masks=masks, deterministic=True)
        a2, _, _ = policy(obs, action_masks=masks, deterministic=True)
    assert th.equal(a1, a2), "deterministic sampling was not reproducible"
    print("deterministic mode stable")


def test_mapper_never_clamps():
    """
    End-to-end: the mapper's clamp backstop must stay untriggered

    The clamp counters were the symptom the whole redesign targets -- 58.5% of
    SELECT steps previously needed correcting. With legality guaranteed upstream
    they should now read exactly zero.
    """
    from ai.utils.mappers import BalatroActionMapper

    policy = make_policy()
    mapper = BalatroActionMapper(
        {"action_selection": slice(0, 1), "card_indices": slice(1, 1 + MAX_PICKS)}
    )

    batch = 1000
    obs = th.randn(batch, OBS)
    for hand_size in [3, 5, 8]:
        masks = make_masks(batch, hand_size=hand_size)
        with th.no_grad():
            actions, _, _ = policy(obs, action_masks=masks)
        for action in actions.numpy():
            response = mapper.process_action(action)
            params = response["params"]
            if response["action"] == 1:  # SELECT_HAND
                assert 1 <= len(params) <= 5, params
                assert len(set(params)) == len(params), params
                assert all(1 <= p <= hand_size for p in params), (hand_size, params)
            else:
                assert params == [], response

    assert mapper.clamped_count == 0, (
        f"clamping fired {mapper.clamped_count} times -- the policy emitted an "
        f"illegal selection, so its masking has a bug")
    assert mapper.empty_count == 0, (
        f"empty fallback fired {mapper.empty_count} times")
    print(f"mapper clamp backstop never fired across {batch * 3:,} actions "
          f"(clamped={mapper.clamped_count}, empty={mapper.empty_count})")


def test_incompatible_checkpoints_are_skipped():
    """
    A checkpoint from an older action space must not abort the run

    Changing the action space invalidates every earlier checkpoint. SB3 does
    notice, but only once it is far enough into load() to raise, which kills the
    whole session -- exactly what happened on the first run after the
    autoregressive switch. Auto-resume must skip incompatible checkpoints and
    start fresh instead.
    """
    import tempfile
    from stable_baselines3.common.save_util import save_to_zip_file
    from ai.train_balatro import checkpoint_is_compatible, find_latest_checkpoint

    class Env:
        observation_space = spaces.Box(-np.inf, np.inf, (OBS,), np.float32)
        action_space = spaces.MultiDiscrete([MAX_ACTIONS] + [MAX_CARDS + 1] * MAX_PICKS)

    class OldEnv:
        observation_space = spaces.Box(-np.inf, np.inf, (OBS,), np.float32)
        action_space = spaces.MultiDiscrete([MAX_ACTIONS] + [2] * MAX_CARDS)

    root = tempfile.mkdtemp()
    ckpt_dir = os.path.join(root, "runs", "old_run", "checkpoints")
    os.makedirs(ckpt_dir)
    old_path = os.path.join(ckpt_dir, "balatro_model_100_steps.zip")

    # A checkpoint carrying the pre-autoregressive action space
    save_to_zip_file(old_path, data={
        "observation_space": OldEnv.observation_space,
        "action_space": OldEnv.action_space,
    })

    assert not checkpoint_is_compatible(old_path, Env()), (
        "old-action-space checkpoint should be rejected")
    assert checkpoint_is_compatible(old_path, OldEnv()), (
        "should still accept a checkpoint matching its own env -- otherwise the "
        "check is just refusing everything")

    found = find_latest_checkpoint(os.path.join(root, "runs"), env=Env())
    assert found is None, f"should have skipped the incompatible checkpoint, got {found}"

    print("incompatible checkpoints skipped rather than crashing the run")


if __name__ == "__main__":
    th.manual_seed(0)
    test_actions_are_always_legal()
    test_respects_action_type_mask()
    test_log_prob_is_reproducible()
    test_forced_actions_contribute_no_log_prob()
    test_gradients_flow()
    test_deterministic_is_stable()
    test_mapper_never_clamps()
    test_incompatible_checkpoints_are_skipped()
    print("\nALL AUTOREGRESSIVE TESTS PASSED")
