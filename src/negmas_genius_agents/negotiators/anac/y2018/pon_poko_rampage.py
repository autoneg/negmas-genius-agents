"""PonPokoRampage from ANAC 2018."""

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

__all__ = ["PonPokoRampage"]


class PonPokoRampage(SAONegotiator):
    """
    PonPokoRampage from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    PonPokoRampage builds on PonPokoAgent (ANAC 2017) with more aggressive
    concession patterns. The agent uses one of five randomized threshold
    patterns featuring oscillating behavior via sin functions for unpredictability.

    **Offering Strategy:**
        Uses dynamic threshold ranges [threshold_low, threshold_high] updated by
        one of 5 patterns (randomly selected at start):
        - Pattern 0: Aggressive oscillating with sin(t*30)
        - Pattern 1: Linear aggressive (up to -0.3*t)
        - Pattern 2: Oscillating with higher amplitude sin(t*25)
        - Pattern 3: Conservative then aggressive drop at t>0.95
        - Pattern 4: Time-scaled oscillation sin(t*25)
        Bids are randomly selected from within the current threshold range.

    **Acceptance Strategy:**
        Simple threshold-based: accepts any offer with utility >= threshold_low.
        The oscillating thresholds create varying acceptance behavior over time,
        sometimes accepting lower utility offers during oscillation dips.

    **Opponent Modeling:**
        No explicit opponent modeling. Strategy relies on randomized threshold
        patterns and oscillating behavior to remain unpredictable while
        gradually conceding through the threshold adjustments.

    Args:
        pattern: Which threshold pattern to use (0-4, or None for random).
        aggressive_drop_time: Time threshold for aggressive drop in pattern 3 (default 0.95).
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
        aggressive_drop_time: float = 0.95,
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
        self._pattern = pattern
        self._aggressive_drop_time = aggressive_drop_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Thresholds
        self._threshold_low = 0.95
        self._threshold_high = 1.0

        # State
        self._last_received_bid: Outcome | None = None
        self._best_bid: Outcome | None = None

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._last_received_bid = None
        self._threshold_low = 0.95
        self._threshold_high = 1.0
        # Randomize pattern if not set
        if self._pattern is None:
            self._pattern = random.randint(0, 4)

    def _update_thresholds(self, time: float) -> None:
        """Update thresholds based on the selected pattern - more aggressive than PonPoko."""
        if self._pattern == 0:
            # Aggressive oscillating
            self._threshold_high = 1 - 0.15 * time
            self._threshold_low = 1 - 0.15 * time - 0.15 * abs(math.sin(time * 30))
        elif self._pattern == 1:
            # Linear aggressive
            self._threshold_high = 1 - 0.05 * time
            self._threshold_low = 1 - 0.3 * time
        elif self._pattern == 2:
            # Oscillating with higher amplitude
            self._threshold_high = 1 - 0.12 * time
            self._threshold_low = 1 - 0.12 * time - 0.2 * abs(math.sin(time * 25))
        elif self._pattern == 3:
            # Conservative then aggressive drop
            self._threshold_high = 1 - 0.08 * time
            self._threshold_low = 1 - 0.15 * time
            if time > self._aggressive_drop_time:
                self._threshold_low = 1 - 0.4 * time
        elif self._pattern == 4:
            # Time-scaled oscillation
            self._threshold_high = 1 - 0.2 * time * abs(math.sin(time * 25))
            self._threshold_low = 1 - 0.28 * time * abs(math.sin(time * 25))
        else:
            # Default aggressive
            self._threshold_high = 1 - 0.15 * time
            self._threshold_low = 1 - 0.25 * abs(math.sin(time * 35))

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
                temp_low -= 0.02
                candidates = self._outcome_space.get_bids_in_range(
                    temp_low, self._threshold_high
                )

        if candidates:
            return random.choice(candidates).bid

        # Fallback to best bid
        return self._best_bid

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
