"""
Balatro RL Environment
Wraps the socket-based communication with Balatro mod in a standard RL interface.
This acts as a translator between Balatro's JSON socket communication and
RL libraries that expect gym-style step()/reset() methods.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
import os
import time
import gymnasium as gym
from gymnasium import spaces

# How long reset() waits for Balatro to (re)connect before giving up. Closing the
# game mid-run used to kill the trainer outright; relaunching it now resumes.
RECONNECT_TIMEOUT = float(os.environ.get("BALATRO_RL_RECONNECT_TIMEOUT", "300"))

# How long each individual accept() waits, so the deadline is checked regularly
RECONNECT_POLL = 5.0

from ..utils.communication import BalatroSocketIO
from .reward import BalatroRewardCalculator
from ..utils.mappers import BalatroStateMapper, BalatroActionMapper
from ..utils.replay import ReplaySystem
from ..telemetry import EventType, Stopwatch, emit


class BalatroEnv(gym.Env):
    """
    Standard RL Environment wrapper for Balatro
    
    Translates between:
    - Balatro mod's line-delimited JSON over a localhost TCP socket
    - Standard RL interface (step, reset, observation spaces)

    This allows RL libraries like Stable-Baselines3 to train on Balatro
    without knowing about the underlying socket communication system.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.current_state = None
        self.prev_state = None
        self.game_over = False
        self.restart_pending = False

        # Telemetry counters (reset() advances the episode index to 0)
        self.episode_index = -1
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.episode_started = time.time()
        self.total_steps = 0

        # Initialize communication and reward systems
        self.pipe_io = BalatroSocketIO()
        self.reward_calculator = BalatroRewardCalculator()

        # Replay System
        self.replay_system = ReplaySystem()
        self.actions_taken = []

        # Define Gymnasium spaces
        # Action Spaces; This should describe the type and shape of the action
        # Constants - Core gameplay actions only (SELECT_HAND=1, PLAY_HAND=2, DISCARD_HAND=3)
        self.MAX_ACTIONS = 3
        self.MAX_CARDS = 8  # Hand slots the observation and actions address
        self.MAX_PICKS = 5  # Balatro's highlighted_limit; a hand is 1-5 cards
        self.MIN_PICKS = 1  # input.lua rejects an empty selection
        self.STOP_INDEX = self.MAX_CARDS  # card slot value meaning "stop picking"

        # Autoregressive selection: one action-type choice followed by up to
        # MAX_PICKS card choices, each of which may instead be STOP. The policy
        # masks each pick against the cards already taken, so an illegal
        # selection (too many cards, duplicates, or none) cannot be represented.
        # The previous space was MAX_CARDS independent binary bits, which could
        # express selections the game rejects and required clamping after the
        # fact on 58.5% of SELECT_HAND steps.
        action_selection = np.array([self.MAX_ACTIONS])
        card_slots = np.array([self.MAX_CARDS + 1] * self.MAX_PICKS)
        self.action_space = spaces.MultiDiscrete(np.concatenate([
            action_selection,
            card_slots
        ]))
        ACTION_SLICE_LAYOUT = [
            ("action_selection", 1),
            ("card_indices", self.MAX_PICKS)
        ]
        slices = self._build_action_slices(ACTION_SLICE_LAYOUT)
        
        # Observation space: This should describe the type and shape of the observation
        # Constants
        self.OBSERVATION_SIZE = 216
        self.observation_space = spaces.Box(
            low=-np.inf, # lowest bound of observation data
            high=np.inf, # highest bound of observation data
            shape=(self.OBSERVATION_SIZE,), # Adjust based on actual state size which  This is a 1D array 
            dtype=np.float32 # Data type of the numbers
        )

        # Initialize mappers
        self.state_mapper = BalatroStateMapper(
            observation_size=self.OBSERVATION_SIZE,
            max_actions=self.MAX_ACTIONS,
            max_cards=self.MAX_CARDS,
        )
        self.action_mapper = BalatroActionMapper(action_slices=slices)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode
        
        In Balatro context, this means starting a new run.
        Communicates with Balatro mod via pipes to initiate reset.
        
        Returns:
            Initial observation/game state
        """
        self.current_state = None
        self.prev_state = None
        self.game_over = False
        self.restart_pending = False
        self.actions_taken = []

        # Per-episode telemetry counters
        self.episode_index += 1
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.episode_started = time.time()

        # Reset reward tracking
        self.reward_calculator.reset()
        
        # Wait for initial request from Balatro (game start)
        initial_request = self._await_initial_request()
        if not initial_request:
            raise RuntimeError(
                f"Failed to receive initial request from Balatro after "
                f"{RECONNECT_TIMEOUT:.0f}s. Is the game running with the mod loaded?"
            )

        # Process initial state for SB3
        self.current_state = initial_request
        initial_observation = self.state_mapper.process_game_state(self.current_state)
        
        # Create initial action mask
        initial_available_actions = initial_request.get('available_actions', [])
        initial_hand = (initial_request.get('game_state', {}).get('hand', {}) or {})
        initial_action_mask = self._create_action_mask(
            initial_available_actions, initial_hand.get('size')
        )
        self._action_masks = initial_action_mask
        
        return initial_observation, {}
    
    def _await_initial_request(self):
        """
        Wait for Balatro, tolerating the game being closed and relaunched

        A multi-hour run should survive the game crashing or being restarted, so
        this retries until a deadline rather than failing on the first missing
        request. Bounded rather than infinite: a game that is never coming back
        should surface as an error instead of hanging silently.

        Returns:
            The initial request dict, or None if the deadline passed
        """
        deadline = time.time() + RECONNECT_TIMEOUT
        attempt = 0

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                emit(EventType.COMM, event="reconnect_timeout",
                     waited=round(RECONNECT_TIMEOUT, 1), attempts=attempt)
                return None

            request = self.pipe_io.wait_for_request(
                accept_timeout=min(RECONNECT_POLL, remaining)
            )
            if request:
                if attempt:
                    self.logger.info("Balatro reconnected; resuming")
                    emit(EventType.COMM, event="reconnected", attempts=attempt)
                return request

            attempt += 1
            if attempt == 1:
                self.logger.warning(
                    "Balatro is not connected. Waiting up to "
                    f"{RECONNECT_TIMEOUT:.0f}s for it to (re)connect - "
                    "relaunching the game will resume this run."
                )
                emit(EventType.COMM, event="awaiting_reconnect",
                     timeout=round(RECONNECT_TIMEOUT, 1))

    def step(self, action):
        """
        Take an action in the Balatro environment
        Sends action to Balatro mod via JSON pipe, waits for response,
        calculates reward, and returns standard RL step format.
        
        Args:
            action: Action dictionary (e.g., {"action": 1, "params": {...}})
            
        Returns:
            Tuple of (observation, reward, done, info) where:
            - observation: Processed game state for neural network
            - reward: Calculated reward for this step
            - done: Whether episode is finished (game over)
            - info: Additional debug information
        """
        timings = {}

        # Store previous state for reward calculation
        self.prev_state = self.current_state

        # Send action response to Balatro mod
        watch = Stopwatch().start()
        response_data = self.action_mapper.process_action(rl_action=action)
        timings['t_map'] = watch.stop()

        self.actions_taken.append(response_data)
        success = self.pipe_io.send_response(response_data)
        if not success:
            raise RuntimeError("Failed to send response to Balatro")

        # Wait for next request with new game state
        next_request = self.pipe_io.wait_for_request()
        timings.update(self.pipe_io.timings())

        if not next_request:
            self.game_over = True
            observation = self.state_mapper.process_game_state(self.current_state)
            reward = 0.0
            self._emit_step(response_data, {}, reward, timings, {})
            self._emit_episode_end('disconnect', {})
            return observation, reward, True, False, {"timeout": True}

        # Update current state
        self.current_state = next_request
        game_state = self.current_state.get('game_state', {})

        # Check for game over condition
        game_over_flag = game_state.get('game_over', 0)
        if game_over_flag == 1:
            watch.start()
            observation = self.state_mapper.process_game_state(self.current_state)
            timings['t_obs'] = watch.stop()

            watch.start()
            reward = self.reward_calculator.calculate_reward(
                current_state=self.current_state,
                prev_state=self.prev_state if self.prev_state else {}
            )
            timings['t_reward'] = watch.stop()

            # Auto-send restart command to Balatro
            restart_response = {"action": 6, "params": []}
            self.pipe_io.send_response(restart_response)

            info = self._build_info(game_state, response_data, next_request, outcome='loss')
            self._emit_step(response_data, game_state, reward, timings, next_request)
            self._emit_episode_end('loss', game_state)

            return observation, reward, True, False, info

        # Check for game win condition
        game_win_flag = game_state.get('game_win', 0)
        if game_win_flag == 1:
            watch.start()
            observation = self.state_mapper.process_game_state(self.current_state)
            timings['t_obs'] = watch.stop()

            watch.start()
            reward = self.reward_calculator.calculate_reward(
                current_state=self.current_state,
                prev_state=self.prev_state if self.prev_state else {}
            )
            timings['t_reward'] = watch.stop()

            # Save replay
            self.replay_system.try_save_replay(
                file_path=self.replay_system.REPLAY_FILE_PATH,
                seed=game_state.get('seed', ''),
                actions=self.actions_taken,
                score=reward,
                chips=game_state.get('chips', 0)
            )

            # Auto-send restart command to Balatro
            restart_response = {"action": 6, "params": []}
            self.pipe_io.send_response(restart_response)

            info = self._build_info(game_state, response_data, next_request, outcome='win')
            self._emit_step(response_data, game_state, reward, timings, next_request)
            self._emit_episode_end('win', game_state)

            return observation, reward, True, False, info


        # Process new state for SB3
        watch.start()
        observation = self.state_mapper.process_game_state(self.current_state)
        timings['t_obs'] = watch.stop()

        # Calculate reward using expert reward calculator
        watch.start()
        reward = self.reward_calculator.calculate_reward(
            current_state=self.current_state,
            prev_state=self.prev_state if self.prev_state else {}
        )
        timings['t_reward'] = watch.stop()

        done = False

        terminated = done
        truncated = False  # Not using time limits for now

        # Create action mask for MaskablePPO
        available_actions = next_request.get('available_actions', [])
        action_mask = self._create_action_mask(
            available_actions, (game_state.get('hand', {}) or {}).get('size')
        )

        info = self._build_info(game_state, response_data, next_request)
        self._emit_step(response_data, game_state, reward, timings, next_request)

        # Store action mask for MaskablePPO
        self._action_masks = action_mask

        return observation, reward, terminated, truncated, info

    def _build_info(self, game_state, response_data, request, outcome=None):
        """
        Assemble the per-step info dict

        Exposes game and reward detail that would otherwise be visible only in
        stdout, so wrappers and callbacks can consume it.

        Args:
            game_state: Inner game state from the mod
            response_data: The action we sent
            request: Full request envelope from the mod
            outcome: 'win', 'loss' or None for a non-terminal step

        Returns:
            Info dictionary for the Gymnasium step return
        """
        round_info = game_state.get('round', {})
        hand_info = game_state.get('hand', {})
        current_hand = game_state.get('current_hand', {})

        info = {
            'action': response_data.get('action'),
            'chips': game_state.get('chips', 0),
            'blind_chips': game_state.get('blind_chips', 0),
            'hands_left': round_info.get('hands_left', 0),
            'discards_left': round_info.get('discards_left', 0),
            'hand_size': hand_info.get('size', 0),
            'highlighted': hand_info.get('highlighted_count', 0),
            'hand_type': current_hand.get('handname', 'None'),
            'retry_count': game_state.get('retry_count', 0),
            'state': game_state.get('state'),
            'available_actions': request.get('available_actions', []),
            'reward_components': self.reward_calculator.last_breakdown,
            'cards_dropped': self.action_mapper.last_dropped,
            'empty_selection': self.action_mapper.last_was_empty,
        }
        if outcome:
            info['outcome'] = outcome
        return info

    def _emit_step(self, response_data, game_state, reward, timings, request):
        """Emit a step event with latency breakdown and game context"""
        self.episode_steps += 1
        self.total_steps += 1
        self.episode_reward += reward

        round_info = game_state.get('round', {})
        current_hand = game_state.get('current_hand', {})

        emit(
            EventType.STEP,
            episode=self.episode_index,
            step=self.episode_steps,
            total_step=self.total_steps,
            action=response_data.get('action'),
            params=response_data.get('params'),
            reward=round(reward, 4),
            chips=game_state.get('chips', 0),
            blind_chips=game_state.get('blind_chips', 0),
            hands_left=round_info.get('hands_left', 0),
            discards_left=round_info.get('discards_left', 0),
            hand_type=current_hand.get('handname', 'None'),
            retry_count=game_state.get('retry_count', 0),
            state=game_state.get('state'),
            available_actions=request.get('available_actions', []),
            reward_components=self.reward_calculator.last_breakdown,
            timings={k: round(v, 6) for k, v in timings.items()},
            game_timing=request.get('timing', {}),
            # Illegal selections projected into legal ones. These should trend
            # towards zero if the policy is learning the 5-card limit.
            cards_dropped=self.action_mapper.last_dropped,
            empty_selection=self.action_mapper.last_was_empty,
            clamped_total=self.action_mapper.clamped_count,
            empty_total=self.action_mapper.empty_count,
            # Hand size was previously only in the info dict, so it was invisible
            # in recorded runs. hand_truncated flags a hand too big to encode.
            hand_size=self.state_mapper.last_hand_size,
            hand_truncated=self.state_mapper.last_hand_truncated,
            truncated_total=self.state_mapper.truncated_hands,
        )

    def _emit_episode_end(self, outcome, game_state):
        """Emit an episode summary event"""
        emit(
            EventType.EPISODE_END,
            episode=self.episode_index,
            outcome=outcome,
            steps=self.episode_steps,
            reward=round(self.episode_reward, 4),
            wall_time=round(time.time() - self.episode_started, 3),
            chips=game_state.get('chips', 0),
            blind_chips=game_state.get('blind_chips', 0),
            seed=game_state.get('seed', ''),
            hands_played=list(self.reward_calculator.hands_played),
            # wins is incremented by reward_calculator.reset(), which has not run
            # yet for this episode, so count the in-flight one here
            wins_total=self.reward_calculator.wins + (1 if outcome == 'win' else 0),
            episodes_total=self.reward_calculator.episode_count + 1,
        )

    def cleanup(self):
        """
        Clean up environment resources
        
        Call this when shutting down to clean up pipe communication.
        """
        self.pipe_io.cleanup()

    # Action Masks for MaskablePPO and for ActionWrapper
    def action_masks(self):
        """Required method for MaskablePPO"""
        if hasattr(self, '_action_masks'):
            return np.array(self._action_masks, dtype=bool)
        else:
            return np.array([True] * sum(self.action_space.nvec), dtype=bool)
    
    def _create_action_mask(self, available_actions, hand_size=None):
        """
        Build the flat action mask for MaskablePPO

        Layout matches the action space: MAX_ACTIONS action-type entries, then
        one (MAX_CARDS + 1)-wide block per card pick.

        The env only reports *which hand positions exist*. The constraints that
        depend on the choices being made -- no repeating a card, at least
        MIN_PICKS, at most MAX_PICKS, everything after STOP is STOP -- are
        applied inside the policy during sampling, because they cannot be known
        here: they depend on picks that have not happened yet when this is built.

        Args:
            available_actions: Balatro action ids currently legal
            hand_size: Cards actually in hand; defaults to a full hand

        Returns:
            Flat list of bools, length sum(action_space.nvec)
        """
        # Action selection mask (SELECT_HAND=1, PLAY_HAND=2, DISCARD_HAND=3)
        # Map Balatro action IDs to AI indices: 1->0, 2->1, 3->2
        action_selection_mask = [False] * self.MAX_ACTIONS
        balatro_to_ai_mapping = {1: 0, 2: 1, 3: 2}

        for action_id in available_actions:
            if action_id in balatro_to_ai_mapping:
                action_selection_mask[balatro_to_ai_mapping[action_id]] = True

        # A row with no legal action type would make the distribution
        # degenerate; fall back to SELECT_HAND, which is what the mod defaults to
        if not any(action_selection_mask):
            action_selection_mask[0] = True

        if hand_size is None:
            hand_size = self.MAX_CARDS
        usable = max(0, min(int(hand_size), self.MAX_CARDS))

        # One block per pick: existing hand positions, plus a STOP slot the
        # policy gates on how many picks have been made
        card_block = [i < usable for i in range(self.MAX_CARDS)] + [True]

        mask = list(action_selection_mask)
        for _ in range(self.MAX_PICKS):
            mask.extend(card_block)
        return mask

    @staticmethod
    def _build_action_slices(layout: List[Tuple[str, int]]) -> Dict[str, slice]:
        """
        Create slices for our actions so that we can precisely extract the
        right params to send over to balatro
        
        Args:
            layout: Our ACTION_SLICE_LAYOUT that contains action name and size
        Return:
            A dictionary containing a key being our action space slice name, and  
            the slice 
        """
        slices = {}
        start = 0
        for action_name, size in layout:
            slices[action_name] = slice(start, start + size)
            start += size
        return slices
