"""Entity-token transformer policy with masked autoregressive action heads.

Architecture (plan §Network, ~1.5-2.5M params):

* Tokens (27 = 10 hand + 6 jokers + 3 consumables + 6 shop + 1 blind +
  1 global), each projected/embedded to d_model with type + slot embeddings;
  the global token is an MLP over global scalars + deck/drawpile count
  features.  Empty slots are attention-masked padding.
* 3-layer pre-LN transformer encoder, d_model=128, 4 heads, FFN x4.
* Attention-pooled summary -> action-type head (masked categorical) and
  value head MLP.
* Pointer heads (joker/consumable/shop/pack) = dot-product attention of a
  summary-derived query against the group's token keys, masked by the env's
  target masks.
* Card-subset head: autoregressive pointer over hand tokens + STOP, with a
  1-layer GRU carrying within-action context.  Conditional sub-step masks
  are derived deterministically from (base card mask, action type, picks so
  far) exactly as documented in env_api.

Sampling (`act`) and PPO recompute (`evaluate_actions`) share one code path
(`_forward`) so the stored joint logprob is exactly reproducible from stored
obs + stored masks + stored actions.  Joint logprob / entropy are the SUM of
the conditional (per-sub-step) terms along the sampled branch.

Masking is exact: illegal logits are set to -inf before the softmax, so
illegal choices have probability exactly 0 and legal probabilities are
renormalized exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from balatro_train import encoding as E
from balatro_train.config import PolicyConfig

# Token index layout (contiguous groups).
TOK_HAND = 0
TOK_JOKER = TOK_HAND + E.HAND_MAX                 # 10
TOK_CONSUMABLE = TOK_JOKER + E.JOKER_SLOTS        # 16
TOK_SHOP = TOK_CONSUMABLE + E.CONSUMABLE_SLOTS    # 19
TOK_BLIND = TOK_SHOP + E.SHOP_SLOTS               # 25
TOK_GLOBAL = TOK_BLIND + 1                        # 26
N_TOKENS = TOK_GLOBAL + 1                         # 27
N_TOKEN_TYPES = 6

_GLOBAL_IN = E.F_GLOBAL + E.N_DECK_SLOTS + E.F_DECK_AGG + E.N_DECK_SLOTS  # 186


class MaskedCategorical:
    """Categorical over masked logits. Illegal probability is exactly 0.

    ``mask`` is bool with True = legal; every row must have >= 1 legal entry.
    """

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        logits = logits.masked_fill(~mask, float("-inf"))
        self.logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        self.mask = mask

    @property
    def probs(self) -> torch.Tensor:
        return self.logp.exp()

    def sample(self) -> torch.Tensor:
        # multinomial needs float32; probabilities of illegal entries are 0.
        return torch.multinomial(self.probs.float(), 1).squeeze(-1)

    def mode(self) -> torch.Tensor:
        return self.logp.argmax(dim=-1)

    def log_prob(self, index: torch.Tensor) -> torch.Tensor:
        return self.logp.gather(-1, index.unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> torch.Tensor:
        # -sum p*logp. Illegal entries have p == 0 exactly but logp == -inf;
        # sanitize logp BEFORE the multiply (0 * -inf = nan would poison the
        # forward, and even a torch.where around the product leaves a nan in
        # the multiply node's backward: 0-grad * -inf-operand = nan).
        safe_logp = self.logp.masked_fill(~self.mask, 0.0)
        return -(self.probs * safe_logp).sum(dim=-1)


def _needs_table(action_set) -> torch.Tensor:
    t = torch.zeros(E.N_ACTION_TYPES, dtype=torch.bool)
    for a in action_set:
        t[int(a)] = True
    return t


def _safe(mask: torch.Tensor) -> torch.Tensor:
    """Make all-False rows sample slot 0 (their contribution is zeroed out)."""
    dead = ~mask.any(dim=-1)
    safe = mask.clone()
    safe[dead, 0] = True
    return safe


class BalatroPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig | None = None):
        super().__init__()
        cfg = cfg or PolicyConfig()
        self.cfg = cfg
        d = cfg.d_model

        # -- token input projections ----------------------------------------
        self.hand_proj = nn.Linear(E.F_CARD, d)
        self.joker_embed = nn.Embedding(E.JOKER_VOCAB, d, padding_idx=0)
        self.joker_proj = nn.Linear(E.F_JOKER_FEAT, d)
        self.consumable_embed = nn.Embedding(E.CONSUMABLE_VOCAB, d, padding_idx=0)
        self.shop_embed = nn.Embedding(E.SHOP_VOCAB, d, padding_idx=0)
        self.shop_proj = nn.Linear(E.F_SHOP_FEAT, d)
        self.blind_proj = nn.Linear(E.F_BLIND, d)
        self.global_mlp = nn.Sequential(
            nn.Linear(_GLOBAL_IN, cfg.global_hidden),
            nn.ReLU(),
            nn.Linear(cfg.global_hidden, d),
        )
        self.type_embed = nn.Embedding(N_TOKEN_TYPES, d)
        self.slot_embed = nn.Embedding(N_TOKENS, d)

        # -- transformer encoder (pre-LN) ------------------------------------
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=d * cfg.ffn_mult,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(d)

        # -- attention pooling -> summary -------------------------------------
        self.pool_query = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.pool_key = nn.Linear(d, d)

        # -- heads -------------------------------------------------------------
        self.action_type_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, E.N_ACTION_TYPES)
        )
        self.ptr_q = nn.ModuleDict(
            {g: nn.Linear(d, d) for g in ("joker", "consumable", "shop", "pack")}
        )
        self.ptr_k = nn.ModuleDict(
            {g: nn.Linear(d, d) for g in ("joker", "consumable", "shop", "pack")}
        )
        # Contract decision: during Phase.PACK the shop_ids/shop_feats slots
        # carry the OPEN PACK's contents (shop and pack are never actionable
        # simultaneously), so the pack pointer attends to the first
        # PACK_SLOTS shop tokens.

        self.card_gru = nn.GRUCell(d, d)
        self.card_h0 = nn.Linear(d, d)
        self.card_start = nn.Parameter(torch.zeros(d))
        self.card_q = nn.Linear(d, d)
        self.card_k = nn.Linear(d, d)
        self.card_stop_key = nn.Parameter(torch.randn(d) / math.sqrt(d))

        self.value_head = nn.Sequential(
            nn.Linear(d, cfg.value_hidden), nn.ReLU(), nn.Linear(cfg.value_hidden, 1)
        )

        # -- action-type -> conditional-head tables (buffers follow .to(device))
        self.register_buffer("needs_cards", _needs_table(E.ACTION_NEEDS_CARDS), persistent=False)
        self.register_buffer("needs_joker", _needs_table(E.ACTION_NEEDS_JOKER), persistent=False)
        self.register_buffer(
            "needs_consumable", _needs_table(E.ACTION_NEEDS_CONSUMABLE), persistent=False
        )
        self.register_buffer("needs_shop", _needs_table(E.ACTION_NEEDS_SHOP), persistent=False)
        self.register_buffer("needs_pack", _needs_table(E.ACTION_NEEDS_PACK), persistent=False)
        pick_min = torch.zeros(E.N_ACTION_TYPES, dtype=torch.long)
        for a, lo in E.CARD_PICK_MIN.items():
            pick_min[int(a)] = lo
        self.register_buffer("card_pick_min", pick_min, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for head in (self.action_type_head, self.value_head):
            last = head[-1]
            nn.init.orthogonal_(last.weight, gain=0.01)
            nn.init.zeros_(last.bias)

    # ------------------------------------------------------------------ encode

    def _encode(self, obs: dict[str, torch.Tensor]):
        """obs (torch tensors, batch-first) -> (tokens (B,27,d), summary (B,d))."""
        B = obs["hand"].shape[0]
        d = self.cfg.d_model

        hand = self.hand_proj(obs["hand"])                                    # (B,10,d)
        joker = self.joker_embed(obs["joker_ids"]) + self.joker_proj(obs["joker_feats"])
        cons = self.consumable_embed(obs["consumable_ids"])                   # (B,3,d)
        shop = self.shop_embed(obs["shop_ids"]) + self.shop_proj(obs["shop_feats"])
        blind = self.blind_proj(obs["blind"]).unsqueeze(1)                    # (B,1,d)
        glob_in = torch.cat(
            [
                obs["global"],
                obs["deck_counts"] / 4.0,
                obs["deck_aggregates"],
                obs["drawpile_counts"] / 4.0,
            ],
            dim=-1,
        )
        glob = self.global_mlp(glob_in).unsqueeze(1)                          # (B,1,d)

        tokens = torch.cat([hand, joker, cons, shop, blind, glob], dim=1)     # (B,27,d)
        type_ids = tokens.new_zeros((N_TOKENS,), dtype=torch.long)
        type_ids[TOK_JOKER:TOK_CONSUMABLE] = 1
        type_ids[TOK_CONSUMABLE:TOK_SHOP] = 2
        type_ids[TOK_SHOP:TOK_BLIND] = 3
        type_ids[TOK_BLIND] = 4
        type_ids[TOK_GLOBAL] = 5
        slot_ids = torch.arange(N_TOKENS, device=tokens.device)
        tokens = tokens + self.type_embed(type_ids) + self.slot_embed(slot_ids)

        # padding: True = ignore this token in attention
        arange_h = torch.arange(E.HAND_MAX, device=tokens.device)
        pad = torch.cat(
            [
                arange_h.unsqueeze(0) >= obs["hand_len"].unsqueeze(1),        # hand
                obs["joker_ids"] == 0,                                        # jokers
                obs["consumable_ids"] == 0,                                   # consumables
                obs["shop_ids"] == 0,                                         # shop
                torch.zeros(B, 2, dtype=torch.bool, device=tokens.device),    # blind+global
            ],
            dim=1,
        )

        out = self.encoder(tokens, src_key_padding_mask=pad)
        out = self.out_norm(out)

        # attention pooling over non-pad tokens
        scores = (self.pool_key(out) @ self.pool_query) / math.sqrt(d)        # (B,27)
        scores = scores.masked_fill(pad, float("-inf"))
        summary = (F.softmax(scores, dim=-1).unsqueeze(1) @ out).squeeze(1)   # (B,d)
        return out, summary

    # ---------------------------------------------------------------- pointers

    def _pointer(self, group: str, tokens: torch.Tensor, summary: torch.Tensor,
                 mask: torch.Tensor, needed: torch.Tensor,
                 given: torch.Tensor | None, deterministic: bool,
                 need_entropy: bool = True):
        """One single-target pointer head. Returns (target, logp, ent, dist)."""
        d = self.cfg.d_model
        span = {
            "joker": (TOK_JOKER, TOK_CONSUMABLE),
            "consumable": (TOK_CONSUMABLE, TOK_SHOP),
            "shop": (TOK_SHOP, TOK_BLIND),
            # pack contents live in the shop token slots during Phase.PACK
            "pack": (TOK_SHOP, TOK_SHOP + E.PACK_SLOTS),
        }[group]
        keys = self.ptr_k[group](tokens[:, span[0]: span[1]])
        q = self.ptr_q[group](summary)
        logits = torch.einsum("bd,bkd->bk", q, keys) / math.sqrt(d)
        dist = MaskedCategorical(logits, _safe(mask))
        if given is not None:
            target = given.clamp(min=0)  # -1 (unused) -> dummy 0, deselected below
        elif deterministic:
            target = dist.mode()
        else:
            target = dist.sample()
        # torch.where (not *) so an unused head's dummy target can be illegal
        # (-inf logprob) without poisoning the joint logprob with NaN.
        zero = summary.new_zeros(())
        logp = torch.where(needed, dist.log_prob(target), zero)
        ent = (torch.where(needed, dist.entropy(), zero) if need_entropy
               else summary.new_zeros(logp.shape))
        target = torch.where(needed, target, target.new_full((), -1))
        return target, logp, ent, dist

    # -------------------------------------------------------------- card picks

    def _card_substeps(self, tokens: torch.Tensor, summary: torch.Tensor,
                       base_mask: torch.Tensor, action_type: torch.Tensor,
                       given_cards: torch.Tensor | None,
                       given_n: torch.Tensor | None, deterministic: bool,
                       need_entropy: bool = True):
        """Autoregressive card-subset head (see env_api contract).

        Returns (cards (B,5), n_cards (B,), logp (B,), ent (B,), step_dists).
        """
        B = tokens.shape[0]
        d = self.cfg.d_model
        device = tokens.device
        hand_tok = tokens[:, TOK_HAND: TOK_HAND + E.HAND_MAX]                 # (B,10,d)
        keys = self.card_k(hand_tok)                                          # (B,10,d)
        stop_key = self.card_stop_key.expand(B, 1, d)

        needed = self.needs_cards[action_type]                                # (B,)
        min_picks = self.card_pick_min[action_type]                           # (B,)
        picked = torch.zeros(B, E.HAND_MAX, dtype=torch.bool, device=device)
        active = needed.clone()
        cards = torch.full((B, E.MAX_CARD_PICKS), -1, dtype=torch.long, device=device)
        n_cards = torch.zeros(B, dtype=torch.long, device=device)
        logp = summary.new_zeros(B)
        ent = summary.new_zeros(B)
        dists: list[MaskedCategorical] = []

        h = torch.tanh(self.card_h0(summary))
        inp = self.card_start.expand(B, d)

        for k in range(E.MAX_CARD_PICKS):
            h = self.card_gru(inp, h)
            q = self.card_q(h)
            logits = torch.cat(
                [torch.einsum("bd,bkd->bk", q, keys), torch.einsum("bd,bkd->bk", q, stop_key)],
                dim=-1,
            ) / math.sqrt(d)

            avail = base_mask & ~picked & active.unsqueeze(1)                 # (B,10)
            stop_ok = (k >= min_picks) | ~avail.any(dim=-1)
            step_mask = torch.cat([avail, stop_ok.unsqueeze(1)], dim=-1)      # (B,11)
            dist = MaskedCategorical(logits, _safe(step_mask))
            dists.append(dist)

            if given_cards is not None:
                assert given_n is not None
                choice = torch.where(
                    (k < given_n) & active,
                    given_cards[:, k].clamp(min=0),
                    torch.full_like(n_cards, E.CARD_STOP_INDEX),
                )
            elif deterministic:
                choice = dist.mode()
            else:
                choice = dist.sample()

            zero = logp.new_zeros(())
            logp = logp + torch.where(active, dist.log_prob(choice), zero)
            if need_entropy:
                ent = ent + torch.where(active, dist.entropy(), zero)

            is_pick = active & (choice != E.CARD_STOP_INDEX)
            safe_choice = choice.clamp(max=E.HAND_MAX - 1)
            picked[torch.arange(B, device=device), safe_choice] |= is_pick
            cards[:, k] = torch.where(is_pick, choice, cards[:, k])
            n_cards = n_cards + is_pick.long()

            # feed the picked card's token back into the GRU (rows that
            # stopped no longer contribute, value irrelevant)
            inp = hand_tok[torch.arange(B, device=device), safe_choice]
            active = is_pick

        return cards, n_cards, logp, ent, dists

    # ----------------------------------------------------------------- forward

    def _forward(self, obs, masks, given=None, deterministic=False, need_aux=False,
                 need_entropy=True):
        tokens, summary = self._encode(obs)

        type_logits = self.action_type_head(summary)
        type_dist = MaskedCategorical(type_logits, masks["action_type_mask"])
        if given is not None:
            action_type = given["action_type"]
        elif deterministic:
            action_type = type_dist.mode()
        else:
            action_type = type_dist.sample()
        logp = type_dist.log_prob(action_type)
        ent_parts = {"action_type": type_dist.entropy() if need_entropy
                     else logp.new_zeros(logp.shape)}
        logp_parts = {"action_type": logp.clone()}
        dists = {"action_type": type_dist} if need_aux else None

        actions: dict[str, torch.Tensor] = {"action_type": action_type}
        for group, needs, mask_key, field in (
            ("joker", self.needs_joker, "joker_target_mask", "joker_target"),
            ("consumable", self.needs_consumable, "consumable_target_mask", "consumable_target"),
            ("shop", self.needs_shop, "shop_target_mask", "shop_target"),
            ("pack", self.needs_pack, "pack_target_mask", "pack_target"),
        ):
            needed = needs[action_type]
            target, p_logp, p_ent, dist = self._pointer(
                group, tokens, summary, masks[mask_key], needed,
                given[field] if given is not None else None, deterministic,
                need_entropy,
            )
            actions[field] = target
            logp = logp + p_logp
            ent_parts[group] = p_ent
            logp_parts[group] = p_logp
            if need_aux:
                dists[group] = dist

        cards, n_cards, c_logp, c_ent, card_dists = self._card_substeps(
            tokens, summary, masks["card_select_mask"], action_type,
            given["cards"] if given is not None else None,
            given["n_cards"] if given is not None else None,
            deterministic, need_entropy,
        )
        if given is not None:
            cards, n_cards = given["cards"], given["n_cards"]
        actions["cards"] = cards
        actions["n_cards"] = n_cards
        logp = logp + c_logp
        ent_parts["cards"] = c_ent
        logp_parts["cards"] = c_logp

        entropy = sum(ent_parts.values())
        value = self.value_head(summary).squeeze(-1)
        aux = {"entropy_parts": ent_parts, "logp_parts": logp_parts}
        if need_aux:
            aux["dists"] = dists
            aux["card_dists"] = card_dists
        return actions, logp, entropy, value, aux

    # -------------------------------------------------------------- public API

    @torch.no_grad()
    def act(self, obs, masks, deterministic: bool = False, need_aux: bool = False,
            need_entropy: bool = True):
        """Sample a composite action. Returns (actions, logp, entropy, value, aux).

        ``need_entropy=False`` skips the per-head entropy kernels (the rollout
        collector discards entropy); actions/logp/value are unaffected."""
        return self._forward(obs, masks, deterministic=deterministic,
                             need_aux=need_aux, need_entropy=need_entropy)

    def evaluate_actions(self, obs, masks, actions, need_aux: bool = False):
        """Recompute (logp, entropy, value, aux) for stored actions under STORED masks."""
        _, logp, entropy, value, aux = self._forward(obs, masks, given=actions, need_aux=need_aux)
        return logp, entropy, value, aux

    def get_value(self, obs, masks) -> torch.Tensor:
        _, summary = self._encode(obs)
        return self.value_head(summary).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def obs_to_torch(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """numpy obs/mask/action dict -> torch tensors on device."""
    return {k: torch.as_tensor(v).to(device) for k, v in obs.items()}


def actions_to_numpy(actions: dict[str, torch.Tensor]) -> dict:
    import numpy as np

    return {k: v.detach().cpu().numpy().astype(np.int64) for k, v in actions.items()}
