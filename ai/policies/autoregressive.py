"""
Autoregressive card-selection policy

Replaces the independent-bits card action space with a sequential one: the policy
picks up to 5 cards one at a time, each pick conditioned on the picks already
made and masked so it cannot repeat a card or exceed the limit. A STOP choice
ends the selection early.

Why this exists
---------------
Balatro accepts a hand selection of 1-5 cards. The previous action space was 8
independent binary bits, and a MultiDiscrete action mask is per-dimension, so
"at most 5 of 8" was inexpressible: the policy emitted illegal selections freely
and they were clamped after the fact. Clamping meant the executed action differed
from the sampled one, which muddies PPO's credit assignment -- and it was firing
on 58.5% of SELECT_HAND steps, so it was the common case rather than an edge
case.

Here every sampled action is legal by construction, so nothing is ever clamped.

Why it does not cost extra environment steps
--------------------------------------------
All 5 sub-decisions happen inside one forward pass. The environment still sees a
single action and performs a single round-trip per ``step()``. That matters: the
game round-trip is ~99% of wall-clock time, so turning one decision into five
environment steps would have been a large regression.

Action layout
-------------
``MultiDiscrete([3, 9, 9, 9, 9, 9])``

- index 0: action type (0=SELECT_HAND, 1=PLAY_HAND, 2=DISCARD_HAND)
- indices 1-5: card slots. Values 0-7 are hand positions, 8 is STOP.

PLAY_HAND and DISCARD_HAND take no card parameters, so their card slots are
forced to STOP and contribute no log-probability -- the policy is not asked to
make, or be graded on, a decision the game ignores.
"""

from typing import Optional, Tuple

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# Very negative rather than -inf: -inf produces NaN gradients if an entire row is
# masked, and softmax over all -inf is undefined
NEG_INF = -1e8


class AutoregressiveCardPolicy(MaskableActorCriticPolicy):
    """
    Policy with a sequential, self-masking card-selection head

    Attributes:
        max_cards: Hand slots addressable (card values 0..max_cards-1)
        max_picks: Most cards selectable in one action (Balatro's limit is 5)
        min_picks: Fewest cards a SELECT_HAND may choose (the mod rejects 0)
        stop_index: The action value meaning "stop selecting"
    """

    SELECT_HAND_INDEX = 0  # position of SELECT_HAND in the action-type dimension

    def __init__(
        self,
        *args,
        max_cards: int = 8,
        max_picks: int = 5,
        min_picks: int = 1,
        card_embed_dim: int = 16,
        **kwargs,
    ):
        self.max_cards = max_cards
        self.max_picks = max_picks
        self.min_picks = min_picks
        self.stop_index = max_cards
        self.card_options = max_cards + 1  # cards + STOP
        self.card_embed_dim = card_embed_dim

        # Set before super().__init__ because _build() runs inside it
        super().__init__(*args, **kwargs)

    # --- Network construction ---

    def _build(self, lr_schedule) -> None:
        """
        Build the shared trunk, the two action heads and the value head

        Overridden rather than extended: the base class builds a single
        ``action_net`` sized for the flattened MultiDiscrete space, which does
        not apply here because the card head is queried once per pick with a
        different input each time.
        """
        self._build_mlp_extractor()

        latent_dim = self.mlp_extractor.latent_dim_pi

        self.action_type_net = nn.Linear(latent_dim, self.action_space.nvec[0])

        # Conditioned on the trunk plus a summary of the picks made so far, so
        # pick N genuinely depends on picks 1..N-1 rather than only being masked
        # by them
        self.card_net = nn.Linear(latent_dim + self.card_embed_dim, self.card_options)
        self.card_embed = nn.Embedding(self.card_options, self.card_embed_dim)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            from functools import partial
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_type_net: 0.01,
                self.card_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        """Not used: sampling is sequential, see _rollout"""
        raise NotImplementedError(
            "AutoregressiveCardPolicy samples sequentially; use forward() or "
            "evaluate_actions() rather than a single action distribution."
        )

    # --- Masking helpers ---

    def _split_masks(self, action_masks, batch_size: int, device) -> Tuple[th.Tensor, th.Tensor]:
        """
        Split the flat env mask into its action-type and card-availability parts

        The environment emits ``sum(nvec)`` booleans: 3 for the action type and
        then one ``card_options``-wide block per card slot. The blocks are
        identical (they describe which hand positions exist); the per-pick
        constraints are applied here instead, because they depend on what has
        already been picked and so cannot be known when the env builds the mask.

        Returns:
            (action_type_mask [B, 3], card_available [B, card_options])
        """
        n_types = int(self.action_space.nvec[0])

        if action_masks is None:
            types = th.ones(batch_size, n_types, dtype=th.bool, device=device)
            cards = th.ones(batch_size, self.card_options, dtype=th.bool, device=device)
            return types, cards

        masks = action_masks
        if isinstance(masks, np.ndarray):
            masks = th.as_tensor(masks, device=device)
        masks = masks.to(device=device, dtype=th.bool).reshape(batch_size, -1)

        types = masks[:, :n_types]
        cards = masks[:, n_types:n_types + self.card_options]
        return types, cards

    @staticmethod
    def _masked_categorical(logits: th.Tensor, mask: th.Tensor) -> Categorical:
        """
        Build a Categorical over only the legal options

        A row with nothing legal would give a degenerate distribution and NaN
        losses, so such rows fall back to uniform. That should be unreachable
        given the mask construction below; it is here so a mask bug surfaces as
        odd behaviour rather than as NaNs propagating through training.
        """
        safe = mask.clone()
        empty_rows = ~safe.any(dim=-1)
        if empty_rows.any():
            safe[empty_rows] = True
        return Categorical(logits=th.where(safe, logits, th.full_like(logits, NEG_INF)))

    # --- The shared sampling / scoring pass ---

    def _rollout(
        self,
        latent_pi: th.Tensor,
        action_masks,
        actions: Optional[th.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Run the action type head then the card head once per pick

        One method serves both sampling (``actions=None``) and scoring given
        actions (teacher forcing). Sharing the path is deliberate: PPO compares
        a stored log-probability against a recomputed one, so any divergence
        between the two would silently corrupt the ratio.

        Args:
            latent_pi: Policy trunk output [B, latent]
            action_masks: Flat env masks, or None
            actions: Actions to score. If None, actions are sampled.
            deterministic: Take the argmax instead of sampling

        Returns:
            (actions [B, 6], log_prob [B], entropy [B])
        """
        batch = latent_pi.shape[0]
        device = latent_pi.device
        scoring = actions is not None

        type_mask, card_available = self._split_masks(action_masks, batch, device)

        # --- Action type ---
        type_dist = self._masked_categorical(self.action_type_net(latent_pi), type_mask)
        if scoring:
            action_type = actions[:, 0].long()
        elif deterministic:
            action_type = th.argmax(type_dist.probs, dim=-1)
        else:
            action_type = type_dist.sample()

        log_prob = type_dist.log_prob(action_type)
        entropy = type_dist.entropy()

        # --- Card picks ---
        selects = action_type == self.SELECT_HAND_INDEX
        # Only SELECT_HAND consumes card params; the others are forced to STOP so
        # the stored action is canonical and carries no spurious log-probability
        finished = ~selects
        picked = th.zeros(batch, self.card_options, dtype=th.bool, device=device)
        context = th.zeros(batch, self.card_embed_dim, device=device)

        chosen = []
        for pick_index in range(self.max_picks):
            logits = self.card_net(th.cat([latent_pi, context], dim=-1))

            # A card is legal if it exists in hand and has not been taken
            mask = card_available.clone()
            mask[:, self.stop_index] = False
            mask &= ~picked

            # STOP becomes legal once the minimum is met, and is the only legal
            # option for rows that already stopped or are not selecting
            allow_stop = finished | (th.full((batch,), pick_index, device=device) >= self.min_picks)
            # If no card remains (hand smaller than max_picks), STOP must be
            # legal or the row would have nothing to choose
            allow_stop = allow_stop | ~mask.any(dim=-1)
            mask[:, self.stop_index] = allow_stop
            mask[finished] = False
            mask[finished, self.stop_index] = True

            dist = self._masked_categorical(logits, mask)
            if scoring:
                pick = actions[:, 1 + pick_index].long()
            elif deterministic:
                pick = th.argmax(dist.probs, dim=-1)
            else:
                pick = dist.sample()

            # Rows already finished contribute nothing: their "choice" is forced
            active = ~finished
            log_prob = log_prob + dist.log_prob(pick) * active
            entropy = entropy + dist.entropy() * active

            pick = th.where(active, pick, th.full_like(pick, self.stop_index))
            chosen.append(pick)

            took_card = active & (pick != self.stop_index)
            picked = picked | (
                nn.functional.one_hot(pick, self.card_options).bool() & took_card.unsqueeze(-1)
            )
            context = context + self.card_embed(pick) * took_card.unsqueeze(-1)
            finished = finished | (pick == self.stop_index)

        out_actions = th.stack([action_type] + chosen, dim=1)
        return out_actions, log_prob, entropy

    # --- SB3 interface ---

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        values = self.value_net(latent_vf)
        actions, log_prob, _ = self._rollout(
            latent_pi, action_masks, deterministic=deterministic
        )
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        _, log_prob, entropy = self._rollout(
            latent_pi, action_masks, actions=actions.long()
        )
        values = self.value_net(latent_vf)
        return values, log_prob, entropy

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        actions, _, _ = self.forward(
            observation, deterministic=deterministic, action_masks=action_masks
        )
        return actions
