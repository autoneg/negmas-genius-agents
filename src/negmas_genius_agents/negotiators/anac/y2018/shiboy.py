"""
Shiboy from ANAC 2018.

This module implements Shiboy, a negotiating agent that competed in the
Automated Negotiating Agents Competition (ANAC) 2018. Shiboy uses a very
conservative Boulware strategy with best offer tracking and rapid deadline
concession.

References:
    - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
    - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
      Competition (ANAC 2018)." IJCAI 2019.
    - Genius framework: https://ii.tudelft.nl/genius/
    - Original package: agents.anac.y2018.shiboy.Shiboy
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

__all__ = ["Shiboy"]


class Shiboy(SAONegotiator):
    """
    Shiboy from ANAC 2018.

    Shiboy uses a conservative time-dependent strategy with opponent awareness:

    1. Very slow (Boulware) concession in early negotiation
    2. Tracks best offer received from opponent
    3. Uses opponent best offer to guide acceptance
    4. Accelerates concession near deadline

    Key features:
    - Conservative early strategy to maximize utility
    - Tracks opponent's best offer for reference
    - Adaptive acceptance threshold
    - Rapid concession in final phase

    Args:
        e: Concession exponent (default 10.0, very Boulware)
        min_utility: Minimum acceptable utility (default 0.55)
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
        e: float = 10.0,
        min_utility: float = 0.55,
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
        self._e = e
        self._min_utility_param = min_utility
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._last_received_offer: Outcome | None = None
        self._best_received_utility: float = 0.0
        self._best_received_bid: Outcome | None = None

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
        self._best_received_bid = None

    def _get_target_utility(self, time: float) -> float:
        """Get target utility based on Boulware concession."""
        # Very conservative Boulware curve
        if time < 0.85:
            # Stay high for most of negotiation
            target = 1.0 - math.pow(time, self._e)
        else:
            # Accelerate concession near deadline
            normalized_time = (time - 0.85) / 0.15
            high_val = 1.0 - math.pow(0.85, self._e)
            low_val = self._min_utility_param
            target = high_val - (high_val - low_val) * math.pow(normalized_time, 2)

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

        # Near deadline, consider best received
        if time >= 0.9:
            # Accept if better than our target and above minimum
            if offer_utility >= self._min_utility_param:
                return True

        # Very near deadline
        if time >= 0.99:
            # Accept anything above reservation
            if offer_utility >= self._min_utility:
                return True
            # Accept if at least as good as best received
            if offer_utility >= self._best_received_utility * 0.98:
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
            self._best_received_bid = offer

        time = state.relative_time

        if self._is_acceptable(offer_utility, time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
