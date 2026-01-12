"""Agent33 from ANAC 2018."""

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

__all__ = ["Agent33"]


class Agent33(SAONegotiator):
    """
    Agent33 from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Agent33 uses a simple but effective two-phase time-dependent strategy with
    linear concession. The agent maintains high aspirations early, then linearly
    concedes toward the minimum utility threshold while tracking the best
    received offer for deadline decisions.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.agent33.Agent33

    **Offering Strategy:**
        Two-phase approach: (1) Before concession_start, maintains high initial
        target utility. (2) After concession_start, linearly decreases from
        initial_target to min_utility. Bids are randomly selected from candidates
        within +/-0.03 of the target for unpredictability.

    **Acceptance Strategy:**
        Accepts offers meeting the current target utility. Under time pressure
        (t >= 0.9), accepts offers above the minimum utility parameter.
        Near deadline (t >= 0.98), accepts offers that are at least 95% of the
        best received utility, or anything above the minimum utility floor.

    **Opponent Modeling:**
        No explicit opponent modeling. Tracks the best received utility from
        opponent offers to inform deadline acceptance decisions.

    Args:
        initial_target: Initial target utility (default 0.95).
        min_utility: Minimum utility threshold (default 0.6).
        concession_start: Time to start conceding (default 0.1).
        time_pressure_threshold: Time threshold for time pressure acceptance (default 0.9).
        deadline_threshold: Time threshold for deadline acceptance (default 0.98).
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
        initial_target: float = 0.95,
        min_utility: float = 0.6,
        concession_start: float = 0.1,
        time_pressure_threshold: float = 0.9,
        deadline_threshold: float = 0.98,
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
        self._initial_target = initial_target
        self._min_utility_param = min_utility
        self._concession_start = concession_start
        self._time_pressure_threshold = time_pressure_threshold
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._last_received_offer: Outcome | None = None
        self._best_received_utility: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = self._outcome_space.min_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._last_received_offer = None
        self._best_received_utility = 0.0

    def _get_target_utility(self, time: float) -> float:
        """Get target utility based on linear concession."""
        if time < self._concession_start:
            # Initial phase - stay high
            target = self._initial_target
        else:
            # Linear concession
            normalized_time = (time - self._concession_start) / (
                1.0 - self._concession_start
            )
            target = (
                self._initial_target
                - (self._initial_target - self._min_utility_param) * normalized_time
            )

        # Scale to utility range
        scaled = self._min_utility + (self._max_utility - self._min_utility) * target

        return max(scaled, self._min_utility_param)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid near target utility."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)

        # Get candidates
        candidates = self._outcome_space.get_bids_in_range(target - 0.03, target + 0.03)

        if not candidates:
            bid_details = self._outcome_space.get_bid_near_utility(target)
            return bid_details.bid if bid_details else self._best_bid

        # Random selection for unpredictability
        return random.choice(candidates).bid

    def _is_acceptable(self, offer_utility: float, time: float) -> bool:
        """Check if an offer is acceptable."""
        target = self._get_target_utility(time)

        # Accept if meets target
        if offer_utility >= target:
            return True

        # Time pressure
        if time >= self._time_pressure_threshold:
            # Accept if above minimum
            if offer_utility >= self._min_utility_param:
                return True

        # Near deadline
        if time >= self._deadline_threshold:
            # Accept if better than best received
            if offer_utility >= self._best_received_utility * 0.95:
                return True
            # Accept anything above reservation
            if offer_utility >= self._min_utility:
                return True

        return False

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        if self._last_received_offer is None:
            return self._best_bid

        time = state.relative_time
        return self._select_bid(time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        self._last_received_offer = offer
        offer_utility = float(self.ufun(offer))

        # Track best received
        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility

        time = state.relative_time

        if self._is_acceptable(offer_utility, time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
