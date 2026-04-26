"""Y2015Group2 from ANAC 2015."""

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

__all__ = ["Y2015Group2"]


class Y2015Group2(SAONegotiator):
    """
    Y2015Group2 negotiation agent from ANAC 2015.

    Group2 uses a balanced time-dependent concession strategy with
    simple but effective opponent tracking.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.group2.Group2

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Standard Boulware-like concession (e=0.2):
          threshold = max_util - (max_util - min_acceptable) * t^(1/e)
        - Minimum threshold capped at 50% utility
        - Random selection from candidates above threshold

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - End-game (t>0.95): Accepts if offer >= best opponent utility
          AND offer >= min_acceptable (50%)

    **Opponent Modeling:**
        - Minimal modeling: tracks only best opponent utility
        - Stores bid history for end-game reference
        - No preference estimation or behavior classification

    Args:
        e: Concession exponent controlling concession speed (default 0.2)
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
        e: float = 0.2,
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
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._min_acceptable: float = 0.5

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
            self._min_utility = self._outcome_space.min_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._best_opponent_utility = 0.0

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent best offer."""
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

    def _compute_threshold(self, time: float) -> float:
        """Compute threshold with balanced concession."""
        f_t = math.pow(time, 1 / self._e) if self._e > 0 else time
        target = self._max_utility - (self._max_utility - self._min_acceptable) * f_t
        return max(target, self._min_acceptable)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid based on threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        return self._select_bid(state.relative_time)

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

        self._update_opponent_model(offer, offer_utility)

        threshold = self._compute_threshold(time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # End-game: accept best opponent offer
        if time > 0.95 and offer_utility >= self._best_opponent_utility:
            if offer_utility >= self._min_acceptable:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
