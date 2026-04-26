"""
AgentF - ANAC 2017 Finalist.

This module contains the reimplementation of AgentF from ANAC 2017.
Original: agents.anac.y2017.agentf.AgentF

References:
    ANAC 2017 competition proceedings.
    https://ii.tudelft.nl/nego/node/7
"""

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

__all__ = ["AgentF"]


class AgentF(SAONegotiator):
    """
    AgentF from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    AgentF uses a simple but effective time-dependent strategy with
    linear concession and opponent adaptation.

    **Offering Strategy:**
        Uses linear time-dependent concession from maximum utility down to
        minimum utility. Selects bids near the target utility with slight
        randomization among similar-value bids to avoid predictability.

    **Acceptance Strategy:**
        Accepts offers meeting or exceeding the current target utility.
        Near the deadline (>98% of time), accepts any offer above the
        minimum utility threshold.

    **Opponent Modeling:**
        Tracks the best utility received from the opponent. This simple
        model helps inform late-game acceptance decisions.

    Args:
        min_utility: Minimum acceptable utility (default 0.6).
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
        min_utility: float = 0.6,
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
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

        # Opponent tracking
        self._best_opponent_utility: float = 0.0

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

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._best_opponent_utility = 0.0

    def _calculate_target_utility(self, time: float) -> float:
        """Calculate target utility using linear concession."""
        # Linear concession from max to min
        target = self._max_utility - time * (self._max_utility - self._min_utility)
        return max(target, self._min_utility)

    def _select_bid(self, target_utility: float) -> Outcome | None:
        """Select a bid near the target utility."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        # Get bid closest to target
        bid_details = self._outcome_space.get_bid_near_utility(target_utility)

        if bid_details is not None:
            # Add some randomness among similar bids
            candidates = self._outcome_space.get_bids_in_range(
                bid_details.utility - 0.02, bid_details.utility + 0.02
            )
            if candidates:
                return random.choice(candidates).bid
            return bid_details.bid

        return self._best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        target = self._calculate_target_utility(time)

        return self._select_bid(target)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)

        time = state.relative_time
        target = self._calculate_target_utility(time)

        # Accept if offer is at least as good as what we'd propose
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if above minimum
        if time > 0.98 and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
