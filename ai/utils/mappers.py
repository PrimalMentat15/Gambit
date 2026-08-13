"""
Mappers for converting between Balatro game data and RL-compatible formats.

This module handles the two-way data transformation:
1. BalatroStateMapper: Converts incoming Balatro JSON to normalized RL observations
2. BalatroActionMapper: Converts RL actions to Balatro command JSON
"""

import numpy as np
from typing import Dict, List, Any
from ..utils.validation import GameStateValidator, ResponseValidator
import logging


def make_onehot(value: int, num_classes: int) -> List[float]:
    """
    Create one-hot encoding for categorical values.
    
    Args:
        value: The category index (0-based)
        num_classes: Total number of possible categories
        
    Returns:
        One-hot encoded list where only the value position is 1.0
        
    Example:
        make_onehot(2, 5) -> [0.0, 0.0, 1.0, 0.0, 0.0]
    """
    onehot = [0.0] * num_classes
    if 0 <= value < num_classes:
        onehot[value] = 1.0
    return onehot


def make_mask(available_items: List[int], total_slots: int) -> List[float]:
    """
    Create binary mask for available items/actions.
    
    Args:
        available_items: List of available indices
        total_slots: Total number of possible slots
        
    Returns:
        Binary mask where available positions are 1.0
        
    Example:
        make_mask([1, 3, 5], 6) -> [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    """
    mask = [0.0] * total_slots
    for item in available_items:
        if 0 <= item < total_slots:
            mask[item] = 1.0
    return mask


def normalize(value: float, max_value: float) -> float:
    """
    Normalize a value to 0-1 range.
    
    Args:
        value: Value to normalize
        max_value: Maximum possible value for scaling
        
    Returns:
        Normalized value between 0.0 and 1.0
        
    Example:
        normalize(1500, 3000) -> 0.5  # Halfway to max
        normalize(50, 100) -> 0.5     # Also halfway
    """
    return value / max_value if max_value > 0 else 0.0



class BalatroStateMapper:
    """
    Converts raw Balatro game state JSON to normalized RL observations.
    
    Handles:
    - Card data normalization
    - Game state parsing
    - Observation space formatting
    """
    # 1 highlighted + 5 suit one-hot + 14 value one-hot + 1 nominal
    FEATURES_PER_CARD = 21

    def __init__(self, observation_size: int, max_actions: int, max_cards: int = 8):
        self.observation_size = observation_size
        self.max_actions = max_actions

        # How many cards the observation has room for. Passed in rather than
        # hardcoded so it stays in step with the environment's action space.
        self.max_cards = max_cards

        # Hand-size telemetry. A hand larger than max_cards is encoded out and
        # the remainder dropped, so this needs to be visible rather than silent.
        self.last_hand_size = 0
        self.last_hand_truncated = 0
        self.truncated_hands = 0
        self.resized_observations = 0

        # Logger
        self.logger = logging.getLogger(__name__)

        # GameStateValidator
        self.game_state_validator = GameStateValidator()

    def process_game_state(self, raw_state: Dict[str, Any] | None) -> np.ndarray:
        """
        Convert Balatro's raw JSON game state into neural network input format
        converting into standardized numerical arrays that neural networks can 
        process.
        
        Args:
            raw_state: Raw game state from Balatro mod JSON
            
        Returns:
            Processed numpy array state suitable for RL training
        """
        # Handle gracefully 
        if not raw_state:
            return np.zeros(self.observation_size, dtype=np.float32)
        
        # Validate game state request 
        try:
            self.game_state_validator.validate_game_state(raw_state)
        except ValueError as e:
            self.logger.error(f"Invalid game state: {e}")

        features = []
        
        features.extend(self._extract_game_features(raw_state.get('game_state', {})))
        features.extend(self._extract_available_actions(raw_state.get('available_actions', [])))

        # Last line of defence: the observation space is declared as a fixed
        # Box, so anything else is a silent contract violation that would either
        # crash deep inside SB3 or feed the policy misaligned features. The
        # per-card padding above should make this unreachable, so a hit here is
        # a bug worth surfacing rather than quietly tolerating.
        if len(features) != self.observation_size:
            self.resized_observations += 1
            self.logger.error(
                f"Observation was {len(features)} features, expected "
                f"{self.observation_size}; padding/truncating. This means a "
                f"feature extractor changed size without observation_size "
                f"being updated."
            )
            if len(features) < self.observation_size:
                features.extend([0.0] * (self.observation_size - len(features)))
            else:
                features = features[:self.observation_size]

        return np.array(features, dtype=np.float32)
    
    def _extract_available_actions(self, available_actions: List[int]) -> List[float]:
        """
        Convert available actions into a fixed-size mask
        Args:
            available_actions: Available actions list from Balatro game state
        Returns:
            Fixed-size list of available action features
        """
        return make_mask(available_actions, self.max_actions)
    
    def _extract_hand_features(self, hand: Dict[str, Any]) -> List[float]:
        """
        Convert Balatro hand data into numerical features for neural network
        
        Transforms card dictionaries into fixed-size numerical arrays:
        - Card values: 2-14 (2 through Ace)
        - Suits: 0-3 (Hearts, Diamonds, Clubs, Spades)
        - Card abilities: chips, mult, special effects
        - Pads/truncates to fixed hand size for consistent input
        
        Args:
            hand: Hand dictionary from Balatro game state
            
        Returns:
            Fixed-size list of hand features
        """
        features = []

        cards = hand.get('cards', [])

        # Hand size is a runtime value, not a constant. It grows with jokers and
        # vouchers, and consumables such as Cryptid insert copies directly into
        # hand -- cardarea.lua only auto-expands card_limit for the deck, so a
        # hand can hold far more than its nominal limit. Anything beyond
        # max_cards is encoded out here; the tail is dropped rather than allowed
        # to change the observation length.
        self.last_hand_size = len(cards)
        self.last_hand_truncated = max(0, len(cards) - self.max_cards)
        if self.last_hand_truncated:
            self.truncated_hands += 1

        features.append(float(hand.get('size', 0)))
        features.append(float(hand.get('highlighted_count', 0)))

        for card in cards[:self.max_cards]:
            card_features = []
            card_features.append(float(card.get('highlighted', False)))
            
            # Suit one-hot
            suits_mapping = {"Hearts": 0, "Diamonds": 1, "Spades": 2, "Clubs": 3}
            suit = card.get('suit', 'Unknown')
            card_features.extend(make_onehot(suits_mapping.get(suit, 4), 5))
            
            # Card value one-hot
            base = card.get('base', {})
            values_mapping = {
                '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8,
                'Jack': 9, 'Queen': 10, 'King': 11, 'Ace': 12
            }
            value = base.get('value')
            card_features.extend(make_onehot(values_mapping.get(value, 13), 14))
            
            # Nominal value (actual chip value used in game calculations)
            card_features.append(base.get('nominal', 0.0))

            features.extend(card_features)

        # Pad the card block out to max_cards so the observation is the same
        # length whatever the hand holds. Previously only a completely empty
        # hand was padded, so any size other than 0 or max_cards silently
        # produced a wrongly-sized observation.
        encoded = min(len(cards), self.max_cards)
        missing = (self.max_cards - encoded) * self.FEATURES_PER_CARD
        if missing:
            features.extend([0.0] * missing)

        return features
    
    def _extract_game_features(self, state: Dict[str, Any]) -> List[float]:
        """
        Extract numerical game-level features from Balatro state
        
        Converts game metadata into neural network inputs:
        - Current game state (menu, selecting hand, etc.)
        - Available actions count
        - Money, chips, round progression
        - Remaining hands/discards
        
        Args:
            state: Full Balatro game state dictionary
            
        Returns:
            List of normalized game features
        """
        features = []
        features.extend(self._extract_round_features(state.get('round', {})))
        features.append(float(state.get('blind_chips', 0)))
        features.append(float(state.get('chips', 0)))
        features.extend(make_onehot(state.get('state', 0), 20))
        features.append(float(state.get('game_over', 0)))
        features.append(float(state.get('game_win', 0)))
        features.append(float(state.get('retry_count', 0)))
        features.extend(self._extract_hand_features(state.get('hand', {})))
        features.extend(self._extract_current_hand_scoring(state.get('current_hand', {})))

        return features

    
    def _extract_round_features(self, round: Dict[str, Any]) -> List[float]:
        """
        Extract information relating to rounds 
        
        Args:
            round: Round state inside of the game_state dictionary
        Returns:
            List of round features
        """
        features = []
        features.append(float(round.get('hands_left', 0)))
        features.append(float(round.get('discards_left', 0)))
        return features



    def _extract_current_hand_scoring(self, current_hand: Dict[str, Any]) -> List[float]:
        """
        Extract current hand scoring information (chips, mult, score, hand type)
        
        Args:
            current_hand: Current hand scoring data from game state
            
        Returns:
            List with chips, mult, score, and one-hot encoded hand type (16 dimensions total)
        """
        features = []
        
        # Raw scoring values
        features.append(float(current_hand.get('chips', 0)))
        features.append(float(current_hand.get('mult', 0)))  
        features.append(float(current_hand.get('score', 0)))
        
        # Hand type one-hot encoding
        hand_types = [
            "None",          # 0 - No hand played yet
            "High Card",     # 1
            "Pair",          # 2
            "Two Pair",      # 3
            "Three of a Kind", # 4
            "Straight",      # 5
            "Flush",         # 6
            "Full House",    # 7
            "Four of a Kind", # 8
            "Straight Flush", # 9
            "Five of a Kind", # 10
            "Flush House",   # 11
            "Flush Five"     # 12
        ]
        
        hand_name = current_hand.get('handname', 'None')
        if not hand_name:
            hand_name = "None"
        
        try:
            hand_index = hand_types.index(hand_name)
        except ValueError:
            hand_index = 0  # Default to "None" if hand type not found

        features.extend(make_onehot(hand_index, len(hand_types)))
        
        return features
    
class BalatroActionMapper:
    """
    Converts RL actions to Balatro command JSON.

    Handles:
    - Binary action conversion to card indices
    - Action validation
    - JSON response formatting
    """

    # Balatro highlights at most 5 cards (cardarea.lua highlighted_limit), and
    # RLBridge/input.lua rejects anything outside 1-5 outright.
    MIN_CARDS = 1
    MAX_SELECTED_CARDS = 5

    # Action id from RLBridge/actions.lua; the only one that takes card params
    SELECT_HAND = 1

    def __init__(self, action_slices: Dict[str, slice]):
        self.slices = action_slices

        # Validator
        self.response_validator = ResponseValidator()

        # Counts of actions the game would have rejected, projected into legal
        # ones. Read by the environment for telemetry: if the policy is learning
        # the constraint these should fall towards zero.
        self.clamped_count = 0
        self.empty_count = 0
        self.last_dropped = 0
        self.last_was_empty = False

        # Logger
        self.logger = logging.getLogger(__name__)

    def process_action(self, rl_action: np.ndarray) -> Dict[str, Any]:
        """
        Convert RL action to Balatro JSON.
        
        Args:
            rl_action: Binary action array from RL agent
            game_state: Current game state for validation
            
        Returns:
            JSON response formatted for Balatro mod
        """
        ai_action = rl_action[self.slices["action_selection"]].tolist()[0]

        # Map AI indices to Balatro action IDs: 0->1, 1->2, 2->3
        ai_to_balatro_mapping = {0: 1, 1: 2, 2: 3}  # SELECT_HAND, PLAY_HAND, DISCARD_HAND
        balatro_action_id = ai_to_balatro_mapping.get(ai_action, self.SELECT_HAND)

        # Only SELECT_HAND consumes card indices. play_hand() and discard_hand()
        # in RLBridge/input.lua take no arguments -- they act on whatever is
        # already highlighted -- and the action mask forces the card bits to 0
        # on those steps anyway. Extracting params there would read that as an
        # empty selection and fabricate a card the mod immediately discards.
        if balatro_action_id == self.SELECT_HAND:
            params = self._extract_select_hand_params(rl_action)
        else:
            params = []
            self.last_dropped = 0
            self.last_was_empty = False

        response_data = {
            "action": balatro_action_id,
            "params": params,
        }

        # Validate action structure
        try:
            self.response_validator.validate_response(response_data)
        except ValueError as e:
            self.logger.error(f"Invalid response: {e}")

        return response_data

    def _extract_select_hand_params(self, raw_action: np.ndarray) -> List[int]:
        """
        Converts the raw action to a list of Lua card indices

        The action space is 8 independent binary bits, so the policy can select
        any number of cards, but Balatro accepts only 1-5. A MultiDiscrete action
        mask is per-dimension and cannot express "at most 5 of 8", so the
        constraint is enforced here instead: over-long selections are truncated
        and empty ones fall back to a single card.

        This is action projection -- what gets executed differs from what was
        sampled -- which slightly muddies credit assignment. The alternative was
        worse: previously these actions were rejected by the mod and the whole
        env step was wasted, which accounted for 15% of all steps.

        Args:
            raw_action: The whole action from the RL agent
        Returns:
            List of 1-based card indices for Lua, always between 1 and 5 long
        """
        card_indices = raw_action[self.slices["card_indices"]]
        selected = [i + 1 for i, val in enumerate(card_indices) if val == 1]

        self.last_dropped = 0
        self.last_was_empty = False

        if len(selected) > self.MAX_SELECTED_CARDS:
            self.last_dropped = len(selected) - self.MAX_SELECTED_CARDS
            self.clamped_count += 1
            selected = selected[:self.MAX_SELECTED_CARDS]
        elif not selected:
            # No bits set: there is nothing to truncate, so pick the first card
            # rather than send a selection the mod will reject
            self.last_was_empty = True
            self.empty_count += 1
            selected = [1]

        return selected


