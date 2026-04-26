"""RandomDance from ANAC 2015."""

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

__all__ = ["RandomDance"]


class RandomDance(SAONegotiator):
    """
    RandomDance negotiation agent from ANAC 2015 - 3rd place agent.

    RandomDance uses a creative "dancing" strategy that adds unpredictability
    to the negotiation while maintaining reasonable utility thresholds.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.RandomDance.RandomDance

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Linear time-dependent concession:
          threshold = max_util - (max_util - min_util) * t
        - "Dance factor": random variance of +/- dance_variance (default 10%)
          applied to threshold each round
        - Random selection from candidates above adjusted threshold
        - Threshold clamped to [min_utility, max_utility]

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= acceptance threshold
        - Final phase (t>0.9): Acceptance threshold decreases more
          aggressively with urgency factor
        - Near deadline (t>0.95): Accepts if offer >= best opponent utility
          AND offer >= minimum utility

    **Opponent Modeling:**
        - Minimal modeling: tracks opponent bids and best utility
        - Uses best opponent utility for end-game acceptance
        - No preference estimation or behavior classification

    Args:
        min_utility: Minimum acceptable utility (default 0.5)
        dance_variance: How much to vary concession rate (default 0.1)
        final_phase_time_threshold: Time after which final phase acceptance logic applies (default 0.9)
        deadline_time_threshold: Time after which end-game acceptance triggers (default 0.95)
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
        min_utility: float = 0.5,
        dance_variance: float = 0.1,
        final_phase_time_threshold: float = 0.9,
        deadline_time_threshold: float = 0.95,
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
        self._min_utility = min_utility
        self._dance_variance = dance_variance
        self._final_phase_time_threshold = final_phase_time_threshold
        self._deadline_time_threshold = deadline_time_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State tracking
        self._max_utility: float = 1.0
        self._current_dance_factor: float = 0.0

        # Opponent tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._best_opponent_utility: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_bids = []
        self._best_opponent_utility = 0.0
        self._current_dance_factor = 0.0

    def _do_dance(self) -> None:
        """Randomly adjust the dance factor for unpredictable concession."""
        # Randomly shift the dance factor
        self._current_dance_factor = random.uniform(
            -self._dance_variance, self._dance_variance
        )

    def _compute_threshold(self, time: float) -> float:
        """
        Compute utility threshold with dancing behavior.

        The threshold decreases linearly with time, but the dance factor
        adds random variation to make the agent less predictable.
        """
        # Base linear concession
        base_threshold = self._max_utility - (
            (self._max_utility - self._min_utility) * time
        )

        # Apply dance factor
        threshold = base_threshold + self._current_dance_factor

        # Ensure threshold stays within bounds
        threshold = max(self._min_utility, min(self._max_utility, threshold))

        return threshold

    def _get_acceptance_threshold(self, time: float) -> float:
        """
        Get the threshold for accepting offers.

        Near the deadline, we become more willing to accept to avoid timeout.
        """
        base_threshold = self._compute_threshold(time)

        # In final phase (last 10% of time), accept if better than best seen
        if time > self._final_phase_time_threshold:
            # Acceptance threshold decreases more aggressively
            urgency = (time - self._final_phase_time_threshold) / (
                1.0 - self._final_phase_time_threshold
            )  # 0 to 1 in final phase
            # At deadline, accept anything above min_utility
            final_threshold = self._min_utility + (1 - urgency) * (
                base_threshold - self._min_utility
            )
            return max(final_threshold, self._min_utility)

        return base_threshold

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent bids to avoid bad deals."""
        self._opponent_bids.append((bid, utility))
        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

    def _select_random_bid(self, threshold: float) -> Outcome | None:
        """Select a random bid above the threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        # Get all bids above threshold
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            # Try with lower threshold
            candidates = self._outcome_space.get_bids_above(
                threshold - self._dance_variance
            )

        if not candidates:
            # Fall back to best bid
            return self._outcome_space.outcomes[0].bid

        # Random selection - this is the "dance"
        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal with dancing behavior."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        # Do the dance - randomly adjust our concession
        self._do_dance()

        threshold = self._compute_threshold(time)
        bid = self._select_random_bid(threshold)

        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        # Track opponent behavior
        self._update_opponent_model(offer, offer_utility)

        # Get acceptance threshold
        accept_threshold = self._get_acceptance_threshold(time)

        # Accept if above threshold
        if offer_utility >= accept_threshold:
            return ResponseType.ACCEPT_OFFER

        # In final moments, accept if it's the best we've seen
        if (
            time > self._deadline_time_threshold
            and offer_utility >= self._best_opponent_utility
        ):
            if offer_utility >= self._min_utility:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
