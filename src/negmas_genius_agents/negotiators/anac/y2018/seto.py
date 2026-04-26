"""
Seto from ANAC 2018 - 3rd Place.

This module implements Seto, the 3rd place agent in the Automated Negotiating
Agents Competition (ANAC) 2018. Seto uses a simple but effective three-phase
time-dependent strategy with conservative early behavior and rapid concession
near the deadline.

References:
    - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
    - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
      Competition (ANAC 2018)." IJCAI 2019.
    - Genius framework: https://ii.tudelft.nl/genius/
    - Original package: agents.anac.y2018.seto.Seto
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

__all__ = ["Seto"]


class Seto(SAONegotiator):
    """
    Seto from ANAC 2018 - 3rd Place.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Seto achieved 3rd place in ANAC 2018 with a simple but effective three-phase
    time-dependent strategy. The agent maintains conservative behavior early,
    linearly concedes in the middle phase, and rapidly concedes near the
    deadline to secure agreements.

    **Offering Strategy:**
        Three-phase concession strategy:
        - Phase 1 (t < 0.3): Stays at initial_target (0.95), offering best bids
        - Phase 2 (0.3 <= t < 0.9): Linear concession from initial_target toward
          middle point (average of initial and min target)
        - Phase 3 (t >= 0.9): Exponential concession (sqrt curve) from middle
          point to min_target for rapid final concession
        Bids randomly selected from candidates within +/-0.03 of target.

    **Acceptance Strategy:**
        Multi-condition acceptance:
        - Accepts offers meeting scaled target utility
        - Late (t >= 0.95): accepts above scaled min_target
        - Deadline (t >= 0.99): accepts if >= 95% of best received utility,
          or anything above minimum utility floor

    **Opponent Modeling:**
        No explicit opponent preference modeling. Tracks best received utility
        to inform deadline acceptance decisions. Strategy relies on predictable
        concession pattern rather than opponent adaptation.

    Args:
        initial_target: Initial target utility (default 0.95).
        min_target: Minimum target utility floor (default 0.6).
        concession_start: Time to start conceding (default 0.3).
        aggressive_time: Time to start aggressive concession (default 0.9).
        late_acceptance_time: Time threshold for late acceptance (default 0.95).
        deadline_threshold: Time threshold for deadline acceptance (default 0.99).
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
        min_target: float = 0.6,
        concession_start: float = 0.3,
        aggressive_time: float = 0.9,
        late_acceptance_time: float = 0.95,
        deadline_threshold: float = 0.99,
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
        self._min_target = min_target
        self._concession_start = concession_start
        self._aggressive_time = aggressive_time
        self._late_acceptance_time = late_acceptance_time
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
        """
        Get target utility based on time with three-phase concession.

        Phase 1 (t < concession_start): Stay at initial target
        Phase 2 (concession_start <= t < aggressive_time): Linear concession
        Phase 3 (t >= aggressive_time): Aggressive concession to min_target
        """
        if time < self._concession_start:
            # Phase 1: Conservative - stay at initial target
            return self._initial_target

        if time < self._aggressive_time:
            # Phase 2: Linear concession
            # Linearly decrease from initial_target to a middle point
            middle_target = (self._initial_target + self._min_target) / 2
            phase_progress = (time - self._concession_start) / (
                self._aggressive_time - self._concession_start
            )
            return (
                self._initial_target
                - (self._initial_target - middle_target) * phase_progress
            )

        # Phase 3: Aggressive concession
        # Quickly decrease to min_target using exponential decay
        phase_progress = (time - self._aggressive_time) / (1.0 - self._aggressive_time)
        middle_target = (self._initial_target + self._min_target) / 2
        return middle_target - (middle_target - self._min_target) * math.pow(
            phase_progress, 0.5
        )

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid based on target utility."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)

        # Scale target to actual utility range
        scaled_target = (
            self._min_utility + (self._max_utility - self._min_utility) * target
        )

        # Get candidates near target
        candidates = self._outcome_space.get_bids_in_range(
            scaled_target - 0.03, scaled_target + 0.03
        )

        if not candidates:
            bid_details = self._outcome_space.get_bid_near_utility(scaled_target)
            return bid_details.bid if bid_details else self._best_bid

        # Random selection among candidates for unpredictability
        return random.choice(candidates).bid

    def _is_acceptable(self, offer_utility: float, time: float) -> bool:
        """
        Check if an offer is acceptable.

        Acceptance conditions:
        1. Offer meets current target utility
        2. Near deadline and offer is above minimum threshold
        3. Very near deadline and offer is better than best received
        """
        target = self._get_target_utility(time)
        scaled_target = (
            self._min_utility + (self._max_utility - self._min_utility) * target
        )

        # Accept if offer meets target
        if offer_utility >= scaled_target:
            return True

        # Near deadline, be more lenient
        if time >= self._late_acceptance_time:
            scaled_min = (
                self._min_utility
                + (self._max_utility - self._min_utility) * self._min_target
            )
            if offer_utility >= scaled_min:
                return True

        # Very near deadline, accept anything reasonable
        if time >= self._deadline_threshold:
            # Accept if better than what we've seen
            if offer_utility >= self._best_received_utility * 0.95:
                return True
            # Accept if above reservation (min utility)
            if offer_utility >= self._min_utility:
                return True

        return False

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        # First move: offer best bid
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

        # Track best received offer
        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility

        time = state.relative_time

        if self._is_acceptable(offer_utility, time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
