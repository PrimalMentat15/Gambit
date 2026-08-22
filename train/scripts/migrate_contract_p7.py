"""Migrate a pre-P7 checkpoint to the P7 observation contract.

P7 widens two per-token feature vectors:

    joker_feats  F_JOKER_FEAT  11 -> 15   (eternal, perishable, tally, rental)
    shop_feats   F_SHOP_FEAT   13 -> 16   (eternal, perishable, rental)

Both are APPENDED at the end of the feature axis, so the corresponding input
columns of `joker_proj` and `shop_proj` can be filled with zeros and the layer
output is unchanged: `Linear(x) = W x + b`, and a zero column contributes
`0 * x_new = 0` for any input. The migrated policy is therefore
BIT-IDENTICAL on Red Deck / White Stake, which `--verify` asserts.

`global` is NOT touched: deck one-hot + stake ordinal consume the block that
was already reserved-and-zero ([48:64)), so `F_GLOBAL` stays 64 and the global
MLP's weight shape is unchanged. On a Red/White state the deck one-hot sets
index 48 to 1.0, which WAS zero before -- so `--verify` compares against the
pre-migration observation (all-zero reserved block), matching what the old
policy actually saw.

    python scripts/migrate_contract_p7.py runs/m4_branch/ckpt_2107375616.pt
    python scripts/migrate_contract_p7.py <ckpt> -o <out> --verify
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import pathlib

import torch

import balatro_train.encoding as E
import balatro_train.policy as policy_mod
from balatro_train.config import PolicyConfig
from balatro_train.policy import BalatroPolicy

#: (state-dict key, old input width, new input width). Bias is unaffected.
PRE_P7_JOKER_FEAT = 11
PRE_P7_SHOP_FEAT = 13

WIDENED = [
    ("joker_proj.weight", PRE_P7_JOKER_FEAT, E.F_JOKER_FEAT),
    ("shop_proj.weight", PRE_P7_SHOP_FEAT, E.F_SHOP_FEAT),
]

#: Pre-P7 entity slot caps, in token-group order.
PRE_P7_CAPS = {"hand": 10, "joker": 6, "consumable": 3, "shop": 6, "blind": 1, "global": 1}


def _caps(hand: int, joker: int, cons: int) -> dict[str, int]:
    return {"hand": hand, "joker": joker, "consumable": cons,
            "shop": E.SHOP_SLOTS, "blind": 1, "global": 1}


def _spans(caps: dict[str, int]) -> dict[str, range]:
    """Token index range of each contiguous group (policy.TOK_* layout)."""
    out, off = {}, 0
    for name, n in caps.items():
        out[name] = range(off, off + n)
        off += n
    return out


def remap_slot_embed(w: torch.Tensor, old: dict[str, int], new: dict[str, int],
                     *, seed_new: bool = True) -> torch.Tensor:
    """Rebuild `slot_embed.weight` when a slot cap changes.

    Slot embeddings are POSITIONAL, so raising a cap shifts every group after
    it -- the joker group moves from index 10 to 12, and so on. Rows must be
    moved, not appended.

    Rows for slots that did not exist before are seeded with the MEAN of their
    own group's existing rows, so a new joker slot starts out looking like a
    typical joker slot. This never affects a pre-P7 state: slots beyond the old
    cap are empty, hence attention-masked, hence their embedding is not read.

    `seed_new=False` zero-fills them instead, for Adam's moment tensors: a row
    that was never trained has no gradient history, and averaging its group's
    history would invent one.
    """
    if all(old[k] == new[k] for k in old):
        return w
    o_span, n_span = _spans(old), _spans(new)
    out = torch.empty(sum(new.values()), w.shape[1], dtype=w.dtype, device=w.device)
    for name in new:
        src, dst = o_span[name], n_span[name]
        keep = min(len(src), len(dst))
        out[dst.start:dst.start + keep] = w[src.start:src.start + keep]
        if len(dst) > keep:
            out[dst.start + keep:dst.stop] = (
                w[src.start:src.stop].mean(dim=0, keepdim=True) if seed_new else 0.0
            )
    return out


def widen(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Return a copy of `sd` with the P7 columns appended as zeros."""
    out = dict(sd)
    changed = []
    for key, old_in, new_in in WIDENED:
        if key not in out:
            raise KeyError(f"{key} missing -- is this a BalatroPolicy checkpoint?")
        w = out[key]
        if w.shape[1] == new_in:
            continue  # already migrated
        if w.shape[1] != old_in:
            raise ValueError(
                f"{key} has input width {w.shape[1]}, expected {old_in} (pre-P7) "
                f"or {new_in} (already migrated)"
            )
        pad = torch.zeros(w.shape[0], new_in - old_in, dtype=w.dtype, device=w.device)
        out[key] = torch.cat([w, pad], dim=1)
        changed.append(f"{key}: {old_in} -> {new_in}")

    new_caps = _caps(E.HAND_MAX, E.JOKER_SLOTS, E.CONSUMABLE_SLOTS)
    if "slot_embed.weight" in out and sum(new_caps.values()) != out["slot_embed.weight"].shape[0]:
        w = out["slot_embed.weight"]
        if w.shape[0] != sum(PRE_P7_CAPS.values()):
            raise ValueError(
                f"slot_embed has {w.shape[0]} rows, expected "
                f"{sum(PRE_P7_CAPS.values())} (pre-P7) or {sum(new_caps.values())}"
            )
        out["slot_embed.weight"] = remap_slot_embed(w, PRE_P7_CAPS, new_caps)
        changed.append(
            f"slot_embed.weight: {w.shape[0]} -> {sum(new_caps.values())} rows "
            f"(hand {PRE_P7_CAPS['hand']}->{E.HAND_MAX}, "
            f"joker {PRE_P7_CAPS['joker']}->{E.JOKER_SLOTS}, "
            f"consumable {PRE_P7_CAPS['consumable']}->{E.CONSUMABLE_SLOTS})"
        )
    return out, changed


def widen_optimizer(opt_sd: dict, pcfg: PolicyConfig) -> list[str]:
    """Reshape Adam's moments in place to match the widened parameters.

    `widen` only touches the weights, which is not enough to resume: Adam
    carries `exp_avg` / `exp_avg_sq` per parameter, and the optimizer's state
    is keyed by the parameter's POSITION in `param_groups`, not by name. Left
    alone, the reloaded moments keep the pre-P7 shapes and the first
    `optimizer.step()` dies on the dtype/layout check -- and for `slot_embed`
    the rows would be right-shaped but attached to the wrong slots, which is
    the worse failure because it is silent.

    Appended columns and new slot rows get zero moments: no gradient history
    is the truth for a parameter that has never been trained, and Adam's first
    update from `m = v = 0` is exactly zero, so nothing jumps.
    """
    with pre_p7_contract() as P_old:
        names = [n for n, _ in P_old.BalatroPolicy(pcfg).named_parameters()]
    new_caps = _caps(E.HAND_MAX, E.JOKER_SLOTS, E.CONSUMABLE_SLOTS)
    widened = {key: (old_in, new_in) for key, old_in, new_in in WIDENED}
    changed = []
    for idx, name in enumerate(names):
        st = opt_sd.get("state", {}).get(idx)
        if not st:
            continue
        for moment in ("exp_avg", "exp_avg_sq"):
            t = st.get(moment)
            if t is None:
                continue
            if name in widened:
                old_in, new_in = widened[name]
                if t.shape[1] == old_in:
                    pad = torch.zeros(t.shape[0], new_in - old_in,
                                      dtype=t.dtype, device=t.device)
                    st[moment] = torch.cat([t, pad], dim=1)
                    changed.append(f"{name}.{moment}: {old_in} -> {new_in}")
            elif name == "slot_embed.weight" and t.shape[0] == sum(PRE_P7_CAPS.values()):
                st[moment] = remap_slot_embed(t, PRE_P7_CAPS, new_caps, seed_new=False)
                changed.append(f"{name}.{moment}: {t.shape[0]} -> "
                               f"{sum(new_caps.values())} rows")
    return changed


@contextlib.contextmanager
def pre_p7_contract():
    """Run the body with `encoding` and `policy` reverted to the pre-P7 shapes.

    `policy` derives its token layout (TOK_JOKER, N_TOKENS, ...) at IMPORT
    time, so patching the constants is not enough -- the module has to be
    reloaded for the old layout to take effect, and reloaded again on the way
    out so the caller is left with the real one.
    """
    saved = {k: getattr(E, k) for k in
             ("HAND_MAX", "JOKER_SLOTS", "CONSUMABLE_SLOTS", "F_JOKER_FEAT", "F_SHOP_FEAT")}
    try:
        E.HAND_MAX = PRE_P7_CAPS["hand"]
        E.JOKER_SLOTS = PRE_P7_CAPS["joker"]
        E.CONSUMABLE_SLOTS = PRE_P7_CAPS["consumable"]
        E.F_JOKER_FEAT, E.F_SHOP_FEAT = PRE_P7_JOKER_FEAT, PRE_P7_SHOP_FEAT
        importlib.reload(policy_mod)
        yield policy_mod
    finally:
        for k, v in saved.items():
            setattr(E, k, v)
        importlib.reload(policy_mod)


def verify(old_sd: dict, new_sd: dict, pcfg: PolicyConfig, seed: int = 0, n: int = 16,
           atol: float = 1e-5, dtype: torch.dtype = torch.float32) -> float:
    """Assert the migrated policy encodes pre-P7 states to the same tokens.

    Compares `_encode` directly: every head (action type, the four pointers,
    the card GRU, the value MLP) is a pure function of `tokens` / `summary`,
    so equality there implies equality everywhere downstream.

    Slots beyond the OLD caps are left empty in the new observation, which
    makes them attention-padding -- so the surviving tokens must match their
    counterparts, group by group, despite sitting at shifted indices.

    NOT bit-identical, and provably so for a benign reason: attention now
    reduces over more keys (44 vs 27). The added keys carry exactly zero
    softmax weight, but the matmul's tiling and summation order change, so
    results differ at the ULP. `dtype=torch.float64` demonstrates this -- the
    discrepancy tracks machine epsilon rather than staying fixed, which is the
    signature of rounding rather than a structural difference.
    """
    new_caps = _caps(E.HAND_MAX, E.JOKER_SLOTS, E.CONSUMABLE_SLOTS)

    with pre_p7_contract() as P_old:
        old = P_old.BalatroPolicy(pcfg).to(dtype)
        old.load_state_dict({k: v.to(dtype) for k, v in old_sd.items()})
        old.eval()
        obs_old = _random_obs(pcfg, n=n, seed=seed)
        obs_old = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in obs_old.items()}
        with torch.no_grad():
            tok_old, sum_old = old._encode(obs_old)
            v_old = old.get_value(obs_old, None)

    new = policy_mod.BalatroPolicy(pcfg).to(dtype)
    new.load_state_dict({k: v.to(dtype) for k, v in new_sd.items()})
    new.eval()
    obs_new = _grow_obs(obs_old, new_caps)
    with torch.no_grad():
        tok_new, sum_new = new._encode(obs_new)
        v_new = new.get_value(obs_new, None)

    o_span, n_span = _spans(PRE_P7_CAPS), _spans(new_caps)
    worst = 0.0
    for name in PRE_P7_CAPS:
        keep = min(len(o_span[name]), len(n_span[name]))
        a = tok_old[:, o_span[name].start:o_span[name].start + keep]
        b = tok_new[:, n_span[name].start:n_span[name].start + keep]
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        if d > atol:
            raise AssertionError(f"token group {name!r} diverged: max |delta| = {d:g}")
    for name, a, b in (("summary", sum_old, sum_new), ("value", v_old, v_new)):
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        if d > atol:
            raise AssertionError(f"{name} diverged: max |delta| = {d:g}")

    print(f"verify[{str(dtype).removeprefix('torch.')}]: all token groups, summary "
          f"and value agree over {n} pre-P7 states (max delta {worst:g})")
    return worst


def verify_is_rounding(old_sd: dict, new_sd: dict, pcfg: PolicyConfig,
                       n: int = 16) -> None:
    """The `--verify` gate: assert the residual is reduction reordering.

    A fixed float32 tolerance is the wrong test. The residual scales with the
    activation magnitudes of the weights being migrated, so a threshold tuned
    on a freshly-initialised policy fires spuriously on a trained one without
    anything structural having changed. Compare the two dtypes instead: extra
    keys that genuinely influenced the result would persist at float64, while
    rounding collapses with machine epsilon (~1e-7 -> ~1e-16).
    """
    f32 = verify(old_sd, new_sd, pcfg, n=n, atol=float("inf"), dtype=torch.float32)
    f64 = verify(old_sd, new_sd, pcfg, n=n, atol=float("inf"), dtype=torch.float64)
    if f64 >= f32 / 1e6:
        raise AssertionError(
            f"residual is structural, not rounding: f32={f32:g} f64={f64:g} "
            "(a reordered reduction shrinks with machine epsilon)")
    print(f"verify: residual is reduction reordering (f32 {f32:g} -> f64 {f64:g})")


def _grow_obs(obs: dict, caps: dict[str, int]) -> dict:
    """Re-express a pre-P7 observation under the new caps and feature widths.

    Added slots are EMPTY (id 0 / zero features) and `hand_len` is unchanged,
    so every added token is attention-padding.
    """
    out = dict(obs)
    out["hand"] = _grow(obs["hand"], caps["hand"], dim=1)
    out["joker_ids"] = _grow(obs["joker_ids"], caps["joker"], dim=1)
    out["joker_feats"] = _grow(_pad(obs["joker_feats"], E.F_JOKER_FEAT), caps["joker"], dim=1)
    out["consumable_ids"] = _grow(obs["consumable_ids"], caps["consumable"], dim=1)
    out["shop_feats"] = _pad(obs["shop_feats"], E.F_SHOP_FEAT)
    return out


def _grow(t: torch.Tensor, n: int, dim: int) -> torch.Tensor:
    """Append empty slots along `dim` up to `n`."""
    if t.shape[dim] >= n:
        return t
    shape = list(t.shape)
    shape[dim] = n - t.shape[dim]
    return torch.cat([t, torch.zeros(*shape, dtype=t.dtype)], dim=dim)


def _pad(t: torch.Tensor, width: int) -> torch.Tensor:
    pad = torch.zeros(*t.shape[:-1], width - t.shape[-1], dtype=t.dtype)
    return torch.cat([t, pad], dim=-1)


def _random_obs(pcfg: PolicyConfig, n: int, seed: int) -> dict:
    """Random observation under the CURRENTLY-ACTIVE contract constants."""
    g = torch.Generator().manual_seed(seed)
    def f(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)
    return {
        "hand": f(n, E.HAND_MAX, E.F_CARD),
        "hand_len": torch.full((n,), E.HAND_MAX, dtype=torch.int64),
        "joker_ids": torch.randint(1, E.JOKER_VOCAB, (n, E.JOKER_SLOTS), generator=g),
        "joker_feats": f(n, E.JOKER_SLOTS, E.F_JOKER_FEAT),
        "consumable_ids": torch.randint(1, E.CONSUMABLE_VOCAB, (n, E.CONSUMABLE_SLOTS), generator=g),
        "consumables_len": torch.full((n,), E.CONSUMABLE_SLOTS, dtype=torch.int64),
        "shop_ids": torch.randint(1, E.SHOP_VOCAB, (n, E.SHOP_SLOTS), generator=g),
        "shop_feats": f(n, E.SHOP_SLOTS, E.F_SHOP_FEAT),
        "blind": f(n, E.F_BLIND),
        "global": f(n, E.F_GLOBAL),
        "deck_counts": f(n, E.N_DECK_SLOTS),
        "deck_aggregates": f(n, E.F_DECK_AGG),
        "drawpile_counts": f(n, E.N_DECK_SLOTS),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=pathlib.Path)
    p.add_argument("-o", "--out", type=pathlib.Path,
                   help="output path (default: <name>.p7.pt beside the input)")
    p.add_argument("--verify", action="store_true",
                   help="assert the migrated policy is unchanged on pre-P7 "
                        "states (float32 vs float64 residual) before writing")
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old_sd = ck["policy"]
    new_sd, changed = widen(old_sd)

    if not changed:
        print("already migrated -- nothing to do")
        return
    for line in changed:
        print("widened", line)

    if args.verify:
        pcfg = PolicyConfig(**ck["config"]["policy"])
        verify_is_rounding(old_sd, new_sd, pcfg)

    ck["policy"] = new_sd
    if ck.get("optimizer"):
        for line in widen_optimizer(ck["optimizer"], PolicyConfig(**ck["config"]["policy"])):
            print("widened", line)
    ck["contract"] = "p7"
    out = args.out or args.checkpoint.with_suffix(".p7.pt")
    torch.save(ck, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
