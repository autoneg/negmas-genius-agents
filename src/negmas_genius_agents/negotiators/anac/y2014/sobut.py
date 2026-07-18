"""Sobut from ANAC 2014."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["Sobut"]


class Sobut(SAONegotiator):
    """
    Sobut from ANAC 2014.

    Sobut is an extremely simple agent: it computes a single minimum
    acceptable utility threshold at the start of the session (based on the
    reservation value, possibly boosted by learning from a previous session
    against the same opponent) and then only ever accepts offers at or above
    that threshold, otherwise proposing random bids that meet the threshold.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.Sobut.Sobut

    **Offering Strategy:**
        No concession at all: draws a uniformly random bid whose utility is
        at or above the (fixed) minimum bid utility threshold, trying up to
        ``max_random_tries`` times. If no such bid is found, falls back to
        the maximum-utility bid in the outcome space.

    **Acceptance Strategy:**
        Accept any offer whose utility is at or above the minimum bid
        utility threshold; otherwise counter-offer.

    **Opponent Modeling:**
        None within a single session. The original agent persists the
        discounted utility achieved in a session and, on the next session
        against the same opponent, uses ``max(reservation, previous outcome,
        random value <= 0.5)`` as a new floor. Since a fresh negotiator
        instance is created per session in this port, cross-session learning
        is not implemented; only the single-session reservation-based floor
        is used.

    Args:
        max_random_tries: Maximum number of random bids sampled before
            falling back to the maximum-utility outcome (default 100000,
            capped internally for performance).
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
        max_random_tries: int = 10000,
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
        self._max_random_tries = max_random_tries
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False
        self._min_bid_utility: float = 0.0
        self._max_utility_bid: Outcome | None = None

    def _initialize(self) -> None:
        """Compute the minimum bid utility threshold."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        reserved_value = float(getattr(self.ufun, "reserved_value", 0.0) or 0.0)
        if reserved_value == float("-inf"):
            reserved_value = 0.0
        self._min_bid_utility = reserved_value

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility_bid = self._outcome_space.outcomes[0].bid
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

    def _get_random_bid(self) -> Outcome | None:
        """Draw a random bid meeting the minimum utility threshold."""
        if self.ufun is None or self._outcome_space is None:
            return None

        candidates = self._outcome_space.get_bids_above(self._min_bid_utility)
        if candidates:
            return random.choice(candidates).bid

        return self._max_utility_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal: always a random bid above the threshold."""
        if not self._initialized:
            self._initialize()

        return self._get_random_bid()

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Accept any offer at or above the minimum bid utility threshold."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER

        if float(self.ufun(offer)) >= self._min_bid_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
