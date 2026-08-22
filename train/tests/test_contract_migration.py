"""P7 contract migration: widening joker/shop features must not change policy
output on a pre-P7 state.

This is the correctness claim the whole Stage 0a migration rests on -- if it
ever stops holding, resuming a pre-P7 checkpoint silently degrades the policy
instead of failing loudly.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re

import pytest
import torch

import balatro_train.encoding as E
from balatro_train.config import PolicyConfig
from balatro_train.policy import BalatroPolicy

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "migrate_contract_p7.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("migrate_contract_p7", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRE_P7_JOKER = 11
PRE_P7_SHOP = 13


def test_contract_widths_are_append_only():
    """The new features must be APPENDED -- a zero-pad is only valid if every
    pre-P7 index still means what it meant before."""
    assert E.F_JOKER_FEAT > PRE_P7_JOKER
    assert E.F_SHOP_FEAT > PRE_P7_SHOP
    # Pre-P7 offsets unmoved.
    assert (E.JOKER_EDITION_OFF, E.JOKER_SELL_OFF, E.JOKER_STATE_OFF,
            E.JOKER_DEBUFFED_OFF) == (0, 5, 6, 10)
    assert (E.SHOP_KIND_OFF, E.SHOP_COST_OFF, E.SHOP_EDITION_OFF) == (0, 7, 8)
    # New features occupy only the appended tail.
    assert E.JOKER_ETERNAL_OFF == PRE_P7_JOKER
    assert E.SHOP_ETERNAL_OFF == PRE_P7_SHOP


def test_global_deck_stake_fits_the_reserved_block():
    """Deck + stake must consume [48:64) exactly, so F_GLOBAL is unchanged and
    the global MLP's weight shape survives the migration."""
    assert E.GLOBAL_DECK_OFF == 48
    assert E.GLOBAL_DECK_OFF + E.N_DECKS == E.GLOBAL_STAKE
    assert E.GLOBAL_STAKE == E.F_GLOBAL - 1
    assert E.F_GLOBAL == 64


def test_migrated_policy_is_bit_identical():
    """The whole Stage 0 claim, end to end.

    Build a genuine pre-P7 policy, migrate its state dict, and assert the
    migrated model encodes pre-P7 states to exactly the same tokens -- despite
    wider features AND every token group sitting at a shifted index.
    """
    mod = _load_script()
    cfg = PolicyConfig(d_model=64, n_layers=2, n_heads=4)

    with mod.pre_p7_contract() as P_old:
        torch.manual_seed(0)
        old_sd = {k: v.clone() for k, v in P_old.BalatroPolicy(cfg).state_dict().items()}
        assert old_sd["joker_proj.weight"].shape[1] == PRE_P7_JOKER
        assert old_sd["shop_proj.weight"].shape[1] == PRE_P7_SHOP
        assert old_sd["slot_embed.weight"].shape[0] == sum(mod.PRE_P7_CAPS.values())

    new_sd, changed = mod.widen(old_sd)
    assert len(changed) == 3, changed  # joker_proj, shop_proj, slot_embed

    # Raises on any divergence; also exercises the slot-index remapping.
    worst = mod.verify(old_sd, new_sd, cfg, n=8)
    # float32 attention over 44 keys instead of 27 reorders the reduction, so
    # agreement is to ~1 ULP, not exact. Anything larger means a real change.
    assert worst < 1e-5, worst


def test_slot_bump_difference_is_rounding_not_structure():
    """Distinguish "reduction reordered" from "the model actually changed".

    If the residual came from the extra tokens genuinely influencing the
    result, it would persist at float64. If it is rounding, it shrinks with
    machine epsilon. Assert it collapses by orders of magnitude.
    """
    mod = _load_script()
    cfg = PolicyConfig(d_model=64, n_layers=2, n_heads=4)
    with mod.pre_p7_contract() as P_old:
        torch.manual_seed(0)
        old_sd = {k: v.clone() for k, v in P_old.BalatroPolicy(cfg).state_dict().items()}
    new_sd, _ = mod.widen(old_sd)

    f32 = mod.verify(old_sd, new_sd, cfg, n=8, dtype=torch.float32)
    f64 = mod.verify(old_sd, new_sd, cfg, n=8, atol=1e-12, dtype=torch.float64)
    assert f64 < f32 / 1e6, f"f32={f32:g} f64={f64:g} -- residual is structural, not rounding"


def test_slot_embed_remap_moves_groups_and_seeds_new_rows():
    """Raising a cap SHIFTS later groups -- rows must be moved, not appended."""
    mod = _load_script()
    old = mod.PRE_P7_CAPS
    new = mod._caps(E.HAND_MAX, E.JOKER_SLOTS, E.CONSUMABLE_SLOTS)
    w = torch.randn(sum(old.values()), 4)
    out = mod.remap_slot_embed(w, old, new)

    assert out.shape[0] == sum(new.values())
    o_span, n_span = mod._spans(old), mod._spans(new)
    for name in old:
        keep = min(len(o_span[name]), len(n_span[name]))
        src = w[o_span[name].start:o_span[name].start + keep]
        dst = out[n_span[name].start:n_span[name].start + keep]
        assert torch.equal(src, dst), f"group {name} not carried over"
        # New rows are seeded with that group's mean, not left at zero.
        extra = out[n_span[name].start + keep:n_span[name].stop]
        if extra.numel():
            want = w[o_span[name].start:o_span[name].stop].mean(dim=0)
            assert torch.allclose(extra, want.expand_as(extra))

    # The joker group really did move.
    assert n_span["joker"].start != o_span["joker"].start


def test_widen_is_idempotent():
    mod = _load_script()
    sd = {
        "joker_proj.weight": torch.randn(8, PRE_P7_JOKER),
        "shop_proj.weight": torch.randn(8, PRE_P7_SHOP),
    }
    once, changed = mod.widen(sd)
    assert len(changed) == 2
    twice, changed2 = mod.widen(once)
    assert changed2 == []
    assert torch.equal(once["joker_proj.weight"], twice["joker_proj.weight"])


def test_widen_rejects_unexpected_width():
    mod = _load_script()
    sd = {
        "joker_proj.weight": torch.randn(8, 99),
        "shop_proj.weight": torch.randn(8, PRE_P7_SHOP),
    }
    with pytest.raises(ValueError, match="input width 99"):
        mod.widen(sd)

def test_rust_mirror_matches_encoding():
    """`sim/py/src/consts.rs` must mirror `encoding.py` exactly.

    consts.rs says so in its first line, but nothing enforced it -- and a
    partial edit to one side is invisible until the binding is rebuilt and the
    boundary validator rejects a shape at runtime, potentially long after the
    change. Parsing the Rust constants keeps the two halves honest at source
    level, and needs no compiled module (so it still runs when the installed
    binding is stale).
    """
    rust_path = pathlib.Path(__file__).resolve().parents[2] / "sim" / "py" / "src" / "consts.rs"
    if not rust_path.is_file():
        pytest.skip(f"Rust mirror not present at {rust_path}")
    rust = rust_path.read_text(encoding="utf-8")

    names = [
        "HAND_MAX", "JOKER_SLOTS", "CONSUMABLE_SLOTS", "SHOP_SLOTS", "PACK_SLOTS",
        "F_CARD", "F_JOKER_FEAT", "F_SHOP_FEAT", "F_BLIND", "F_GLOBAL",
        "JOKER_EDITION_OFF", "JOKER_SELL_OFF", "JOKER_STATE_OFF", "JOKER_DEBUFFED_OFF",
        "JOKER_ETERNAL_OFF", "JOKER_PERISHABLE_OFF", "JOKER_PERISH_TALLY_OFF",
        "JOKER_RENTAL_OFF",
        "SHOP_KIND_OFF", "SHOP_COST_OFF", "SHOP_EDITION_OFF",
        "SHOP_ETERNAL_OFF", "SHOP_PERISHABLE_OFF", "SHOP_RENTAL_OFF",
        "BLIND_KIND_OFF", "BLIND_BOSS_OFF", "BLIND_REQ_OFF",
        "GLOBAL_MONEY", "GLOBAL_ANTE_OFF", "GLOBAL_PHASE_OFF", "GLOBAL_HAND_SIZE",
        "GLOBAL_DECK_OFF", "GLOBAL_STAKE", "N_DECKS", "N_STAKES",
        "N_DECK_SLOTS", "F_DECK_AGG", "N_ACTION_TYPES", "MAX_CARD_PICKS",
    ]

    mismatched, missing = [], []
    for name in names:
        m = re.search(r"pub const %s:\s*usize\s*=\s*(\d+)\s*;" % name, rust)
        py = getattr(E, name, None)
        if m is None or py is None:
            missing.append(f"{name} (rust={m and m.group(1)}, py={py})")
        elif int(m.group(1)) != int(py):
            mismatched.append(f"{name}: rust={m.group(1)} != encoding.py={py}")

    assert not missing, "constants absent from one side: " + ", ".join(missing)
    assert not mismatched, (
        "consts.rs has drifted from encoding.py: " + "; ".join(mismatched))


def test_deck_and_stake_enums_match_rust():
    """Deck/Stake discriminants are a frozen wire ordering, mirrored in Rust."""
    rust_path = pathlib.Path(__file__).resolve().parents[2] / "sim" / "py" / "src" / "consts.rs"
    if not rust_path.is_file():
        pytest.skip("Rust mirror not present")
    rust = rust_path.read_text(encoding="utf-8")

    for enum_name, py_enum in (("Deck", E.Deck), ("Stake", E.Stake)):
        block = re.search(r"pub enum %s \{(.*?)\}" % enum_name, rust, re.S)
        assert block, f"enum {enum_name} not found in consts.rs"
        pairs = dict(re.findall(r"(\w+)\s*=\s*(\d+)", block.group(1)))
        assert len(pairs) == len(py_enum), (
            f"{enum_name}: rust has {len(pairs)} variants, python has {len(py_enum)}")
        for member in py_enum:
            rust_val = pairs.get(member.name.capitalize())
            assert rust_val is not None, f"{enum_name}.{member.name} missing in consts.rs"
            assert int(rust_val) == int(member.value), (
                f"{enum_name}.{member.name}: rust={rust_val} != python={member.value}")


def test_verify_gate_accepts_rounding_and_rejects_a_real_change():
    """`--verify`'s gate is the epsilon collapse, not a float32 threshold.

    A fixed float32 tolerance is a function of activation magnitude, so it
    fires on a trained checkpoint with nothing structural changed. Perturbing
    a migrated weight is the case that must still be caught.
    """
    mod = _load_script()
    cfg = PolicyConfig(d_model=64, n_layers=2, n_heads=4)
    with mod.pre_p7_contract() as P_old:
        torch.manual_seed(0)
        old_sd = {k: v.clone() for k, v in P_old.BalatroPolicy(cfg).state_dict().items()}
    new_sd, _ = mod.widen(old_sd)

    mod.verify_is_rounding(old_sd, new_sd, cfg, n=8)

    # A non-zero column on an APPENDED feature is invisible on pre-P7 states
    # (the input is 0 there), so break a column the old contract did use.
    broken = {k: v.clone() for k, v in new_sd.items()}
    broken["joker_proj.weight"][:, 0] += 0.1
    with pytest.raises(AssertionError, match="structural"):
        mod.verify_is_rounding(old_sd, broken, cfg, n=8)


def test_optimizer_moments_are_migrated_with_the_weights():
    """Widening the weights alone does not resume: Adam's moments must move too.

    The optimizer's state is keyed by parameter POSITION, so a stale
    `slot_embed` moment is not merely the wrong shape -- its rows would be
    attached to the wrong slots, which resumes silently.
    """
    mod = _load_script()
    cfg = PolicyConfig(d_model=32, n_layers=1, n_heads=2)
    with mod.pre_p7_contract() as P_old:
        torch.manual_seed(0)
        old_policy = P_old.BalatroPolicy(cfg)
        names = [n for n, _ in old_policy.named_parameters()]
        opt = torch.optim.Adam(old_policy.parameters(), lr=1e-3)
        for p in old_policy.parameters():
            p.grad = torch.randn_like(p)
        opt.step()
        opt_sd = opt.state_dict()

    mod.widen_optimizer(opt_sd, cfg)
    state = opt_sd["state"]
    for key, old_in, new_in in mod.WIDENED:
        st = state[names.index(key)]
        assert st["exp_avg"].shape[1] == new_in
        # Appended columns carry no history, and the surviving ones are intact.
        assert torch.equal(st["exp_avg"][:, old_in:],
                           torch.zeros_like(st["exp_avg"][:, old_in:]))
        assert st["exp_avg_sq"].shape[1] == new_in

    new_caps = mod._caps(E.HAND_MAX, E.JOKER_SLOTS, E.CONSUMABLE_SLOTS)
    st = state[names.index("slot_embed.weight")]
    assert st["exp_avg"].shape[0] == sum(new_caps.values())

    # The migrated moments load into a real P7 optimizer and step cleanly.
    new_policy = BalatroPolicy(cfg)
    new_policy.load_state_dict(mod.widen(old_policy.state_dict())[0])
    new_opt = torch.optim.Adam(new_policy.parameters(), lr=1e-3)
    new_opt.load_state_dict(opt_sd)
    for p in new_policy.parameters():
        p.grad = torch.randn_like(p)
    new_opt.step()
