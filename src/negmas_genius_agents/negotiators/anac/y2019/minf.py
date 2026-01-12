"""
MINF from ANAC 2019.

This module contains the Python reimplementation of MINF
(Minimum Information), a minimalist agent that deliberately
avoids complex opponent modeling for robustness.

References:
    Baarslag, T., Fujita, K., Gerding, E. H., Hindriks, K., Ito, T.,
    Jennings, N. R., ... & Williams, C. R. (2019). The Tenth International
    Automated Negotiating Agents Competition (ANAC 2019).
    In Proceedings of the International Joint Conference on Autonomous
    Agents and Multiagent Systems (AAMAS).

Original Genius class: agents.anac.y2019.minf.MINF
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

__all__ = ["MINF"]


class MINF(SAONegotiator):
    """
    MINF from ANAC 2019.

    MINF (Minimum Information) implements a minimalist negotiation
    strategy that deliberately avoids complex opponent modeling.
    The philosophy is that simple, predictable behavior can be more
    robust than sophisticated but potentially brittle strategies.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    **Offering Strategy:**
        - Polynomial concession: target = max - (max - min) * t^rate
        - With default rate=1.0, this is linear concession
        - Searches for bids within +/- 0.03 of target utility
        - Randomly selects from candidate bids (no preference)
        - First offer is always the best available bid
        - No opponent modeling influences bid selection

    **Acceptance Strategy:**
        - Accepts offers meeting or exceeding the polynomial target
        - Very near deadline (t >= 0.99): Accepts offers above minimum
        - Simple threshold comparison without adaptive adjustments

    **Opponent Modeling:**
        - Deliberately minimal - only tracks offer count
        - No opponent preference estimation
        - No adaptation to opponent behavior
        - Philosophy: robustness through simplicity
        - Avoids overfitting to noisy opponent signals

    Args:
        concession_rate: Polynomial exponent for concession (default 1.0, linear)
        min_utility: Minimum acceptable utility (default 0.6)
        deadline_threshold: Time threshold for deadline acceptance (default 0.99)
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
        concession_rate: float = 1.0,
        min_utility: float = 0.6,
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
        self._concession_rate = concession_rate
        self._min_utility = min_utility
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._opponent_offers_count: int = 0

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
        self._opponent_offers_count = 0

    def _get_target_utility(self, time: float) -> float:
        """Get target utility using simple concession."""
        # Polynomial concession
        target = self._max_utility - (self._max_utility - self._min_utility) * (
            time**self._concession_rate
        )
        return max(target, self._min_utility)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid near target."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return self._best_bid

        target = self._get_target_utility(time)

        # Simple random selection from near-target bids
        candidates = self._outcome_space.get_bids_in_range(target - 0.03, target + 0.03)

        if not candidates:
            bid_detail = self._outcome_space.get_bid_near_utility(target)
            return bid_detail.bid if bid_detail else self._best_bid

        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        if self._opponent_offers_count == 0:
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

        self._opponent_offers_count += 1

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        time = state.relative_time
        target = self._get_target_utility(time)

        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        if time >= self._deadline_threshold and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
