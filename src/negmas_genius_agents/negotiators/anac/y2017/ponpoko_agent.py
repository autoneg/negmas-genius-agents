"""PonPokoAgent from ANAC 2017."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["PonPokoAgent"]


class PonPokoAgent(SAONegotiator):
    """
    PonPokoAgent from ANAC 2017 - The winning agent.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This is a reimplementation of PonPokoAgent, the winning agent from the
    Automated Negotiating Agents Competition (ANAC) 2017.

    PonPokoAgent achieved 1st place in ANAC 2017 with a deceptively simple
    strategy using randomized threshold patterns. The name "PonPoko" comes from
    the Japanese folklore about tanuki (raccoon dogs).

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/nego/node/7

    **Offering Strategy:**
        Randomly selects one of 5 threshold patterns at negotiation start:
        - Pattern 0: Oscillating with sin(40t), -0.1t base decay.
        - Pattern 1: Linear from 1 to 0.78 (simple concession).
        - Pattern 2: Oscillating with sin(20t), larger amplitude.
        - Pattern 3: Conservative (-0.1t) until t>0.99, then -0.3t.
        - Pattern 4: Time-modulated oscillation with sin(20t).
        Selects random bids from the utility range [threshold_low, threshold_high].

    **Acceptance Strategy:**
        Simple threshold-based acceptance: accepts any offer with utility
        >= threshold_low. No special late-game handling - relies entirely
        on the threshold pattern for deadline behavior.

    **Opponent Modeling:**
        None. PonPokoAgent deliberately avoids opponent modeling, relying
        instead on the diversity of threshold patterns to handle different
        opponent types. This simplicity proved highly effective in competition.

    Args:
        pattern: Which threshold pattern to use (0-4, or None for random).
        late_concession_threshold: Time threshold for pattern 3 late concession (default 0.99).
        pattern0_high_decay: Base decay rate of the high threshold in pattern 0 (default 0.1).
        pattern0_low_decay: Base decay rate of the low threshold in pattern 0 (default 0.1).
        pattern0_oscillation_freq: Frequency of the low-threshold oscillation in
            pattern 0 (default 40.0).
        pattern0_oscillation_amp: Amplitude of the low-threshold oscillation in
            pattern 0 (default 0.1).
        pattern1_low_decay: Decay rate of the low threshold in pattern 1 (default 0.22).
        pattern2_high_decay: Base decay rate of the high threshold in pattern 2 (default 0.1).
        pattern2_low_decay: Base decay rate of the low threshold in pattern 2 (default 0.1).
        pattern2_oscillation_freq: Frequency of the low-threshold oscillation in
            pattern 2 (default 20.0).
        pattern2_oscillation_amp: Amplitude of the low-threshold oscillation in
            pattern 2 (default 0.15).
        pattern3_high_decay: Decay rate of the high threshold in pattern 3 (default 0.05).
        pattern3_low_decay: Decay rate of the low threshold in pattern 3 before the
            late concession kicks in (default 0.1).
        pattern3_late_decay: Decay rate of the low threshold in pattern 3 after the
            late concession threshold (default 0.3).
        pattern4_high_decay: Decay rate of the high threshold in pattern 4 (default 0.15).
        pattern4_low_decay: Decay rate of the low threshold in pattern 4 (default 0.21).
        pattern4_oscillation_freq: Frequency of the time-modulated oscillation in
            pattern 4 (default 20.0).
        default_high_decay: Base decay rate of the high threshold in the fallback
            default pattern (default 0.1).
        default_oscillation_freq: Frequency of the low-threshold oscillation in the
            fallback default pattern (default 40.0).
        default_oscillation_amp: Amplitude of the low-threshold oscillation in the
            fallback default pattern (default 0.2).
        threshold_decrement: Step size used to lower the low threshold when no
            bids are found in the current range (default 0.01).
        preferences: NegMAS preferences/utility function.
        ufun: Utility function (overrides preferences if given).
        name: Negotiator name.
        parent: Parent controller.
        owner: Agent that owns this negotiator.
        id: Unique identifier.
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(
        self,
        pattern: int | None = None,
        late_concession_threshold: float = 0.99,
        pattern0_high_decay: float = 0.1,
        pattern0_low_decay: float = 0.1,
        pattern0_oscillation_freq: float = 40.0,
        pattern0_oscillation_amp: float = 0.1,
        pattern1_low_decay: float = 0.22,
        pattern2_high_decay: float = 0.1,
        pattern2_low_decay: float = 0.1,
        pattern2_oscillation_freq: float = 20.0,
        pattern2_oscillation_amp: float = 0.15,
        pattern3_high_decay: float = 0.05,
        pattern3_low_decay: float = 0.1,
        pattern3_late_decay: float = 0.3,
        pattern4_high_decay: float = 0.15,
        pattern4_low_decay: float = 0.21,
        pattern4_oscillation_freq: float = 20.0,
        default_high_decay: float = 0.1,
        default_oscillation_freq: float = 40.0,
        default_oscillation_amp: float = 0.2,
        threshold_decrement: float = 0.01,
        preferences: BaseUtilityFunction | None = None,
        ufun: BaseUtilityFunction | None = None,
        name: str | None = None,
        parent: Controller | None = None,
        owner: Agent | None = None,
        id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            preferences=preferences,
            ufun=ufun,
            name=name,
            parent=parent,
            owner=owner,
            id=id,
            **kwargs,
        )
        self._pattern = pattern if pattern is not None else random.randint(0, 4)
        self._late_concession_threshold = late_concession_threshold
        self._pattern0_high_decay = pattern0_high_decay
        self._pattern0_low_decay = pattern0_low_decay
        self._pattern0_oscillation_freq = pattern0_oscillation_freq
        self._pattern0_oscillation_amp = pattern0_oscillation_amp
        self._pattern1_low_decay = pattern1_low_decay
        self._pattern2_high_decay = pattern2_high_decay
        self._pattern2_low_decay = pattern2_low_decay
        self._pattern2_oscillation_freq = pattern2_oscillation_freq
        self._pattern2_oscillation_amp = pattern2_oscillation_amp
        self._pattern3_high_decay = pattern3_high_decay
        self._pattern3_low_decay = pattern3_low_decay
        self._pattern3_late_decay = pattern3_late_decay
        self._pattern4_high_decay = pattern4_high_decay
        self._pattern4_low_decay = pattern4_low_decay
        self._pattern4_oscillation_freq = pattern4_oscillation_freq
        self._default_high_decay = default_high_decay
        self._default_oscillation_freq = default_oscillation_freq
        self._default_oscillation_amp = default_oscillation_amp
        self._threshold_decrement = threshold_decrement
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Thresholds
        self._threshold_low = 0.99
        self._threshold_high = 1.0

        # State
        self._last_received_bid: Outcome | None = None

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._last_received_bid = None
        self._threshold_low = 0.99
        self._threshold_high = 1.0

    def _update_thresholds(self, time: float) -> None:
        """Update thresholds based on the selected pattern."""
        if self._pattern == 0:
            self._threshold_high = 1 - self._pattern0_high_decay * time
            self._threshold_low = 1 - self._pattern0_low_decay * time - (
                self._pattern0_oscillation_amp
                * abs(math.sin(time * self._pattern0_oscillation_freq))
            )
        elif self._pattern == 1:
            self._threshold_high = 1.0
            self._threshold_low = 1 - self._pattern1_low_decay * time
        elif self._pattern == 2:
            self._threshold_high = 1 - self._pattern2_high_decay * time
            self._threshold_low = 1 - self._pattern2_low_decay * time - (
                self._pattern2_oscillation_amp
                * abs(math.sin(time * self._pattern2_oscillation_freq))
            )
        elif self._pattern == 3:
            self._threshold_high = 1 - self._pattern3_high_decay * time
            self._threshold_low = 1 - self._pattern3_low_decay * time
            if time > self._late_concession_threshold:
                self._threshold_low = 1 - self._pattern3_late_decay * time
        elif self._pattern == 4:
            self._threshold_high = 1 - self._pattern4_high_decay * time * abs(
                math.sin(time * self._pattern4_oscillation_freq)
            )
            self._threshold_low = 1 - self._pattern4_low_decay * time * abs(
                math.sin(time * self._pattern4_oscillation_freq)
            )
        else:
            # Default fallback
            self._threshold_high = 1 - self._default_high_decay * time
            self._threshold_low = 1 - self._default_oscillation_amp * abs(
                math.sin(time * self._default_oscillation_freq)
            )

        # Ensure valid range
        self._threshold_low = max(0.0, min(self._threshold_low, self._threshold_high))

    def _select_bid(self) -> Outcome | None:
        """Select a bid within the threshold range."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        # Get bids in range
        candidates = self._outcome_space.get_bids_in_range(
            self._threshold_low, self._threshold_high
        )

        if not candidates:
            # Lower threshold until we find something
            temp_low = self._threshold_low
            while not candidates and temp_low > 0:
                temp_low -= self._threshold_decrement
                candidates = self._outcome_space.get_bids_in_range(
                    temp_low, self._threshold_high
                )

        if candidates:
            return random.choice(candidates).bid

        # Fallback to best bid
        return self._outcome_space.outcomes[0].bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        self._update_thresholds(time)

        return self._select_bid()

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        self._last_received_bid = offer
        time = state.relative_time
        self._update_thresholds(time)

        offer_utility = float(self.ufun(offer))

        # Accept if above lower threshold
        if offer_utility >= self._threshold_low:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER