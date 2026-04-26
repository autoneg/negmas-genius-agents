"""AgentHP from ANAC 2015."""

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

__all__ = ["AgentHP"]


class AgentHP(SAONegotiator):
    """
    AgentHP (High Performance) negotiation agent from ANAC 2015.

    AgentHP emphasizes computational efficiency with bid caching and
    streamlined decision-making for fast negotiation performance.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.AgentHP.AgentHP

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Time-dependent Boulware-like concession with formula:
          threshold = max_util - (max_util - min_util) * 0.5 * t^(1/e)
        - Efficient bid caching: reuses candidates when threshold changes
          by less than 0.05
        - Random selection from top 10 candidates above threshold

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - End-game (t>0.95): Accepts if offer >= best opponent utility
          AND offer >= min_utility + 0.1

    **Opponent Modeling:**
        - Minimal opponent tracking for performance
        - Tracks only best opponent utility and offer count
        - No complex preference estimation to maintain speed

    Args:
        e: Concession exponent controlling concession speed (default 0.15)
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
        e: float = 0.15,
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

        # Caching
        self._last_threshold: float = 1.0
        self._cached_candidates: list = []

        # Opponent tracking
        self._best_opponent_utility: float = 0.0
        self._opponent_count: int = 0

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

        self._last_threshold = 1.0
        self._cached_candidates = []
        self._best_opponent_utility = 0.0
        self._opponent_count = 0

    def _update_opponent(self, utility: float) -> None:
        """Simple opponent tracking."""
        self._opponent_count += 1
        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

    def _compute_threshold(self, time: float) -> float:
        """Efficient threshold computation."""
        f_t = math.pow(time, 1 / self._e) if self._e > 0 else time
        threshold = (
            self._max_utility - (self._max_utility - self._min_utility) * 0.5 * f_t
        )
        return max(threshold, self._min_utility + 0.05)

    def _get_candidates(self, threshold: float) -> list:
        """Get candidates with caching."""
        if self._outcome_space is None:
            return []

        # Use cache if threshold is close
        if abs(threshold - self._last_threshold) < 0.05 and self._cached_candidates:
            return self._cached_candidates

        candidates = self._outcome_space.get_bids_above(threshold)
        self._last_threshold = threshold
        self._cached_candidates = candidates
        return candidates

    def _select_bid(self, time: float) -> Outcome | None:
        """Efficient bid selection."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._get_candidates(threshold)

        if not candidates:
            candidates = self._get_candidates(threshold * 0.9)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Simple selection from top candidates
        n = min(10, len(candidates))
        return random.choice(candidates[:n]).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal efficiently."""
        if not self._initialized:
            self._initialize()

        return self._select_bid(state.relative_time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Fast response evaluation."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        self._update_opponent(offer_utility)

        threshold = self._compute_threshold(time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # End-game acceptance
        if time > 0.95 and offer_utility >= self._best_opponent_utility:
            if offer_utility >= self._min_utility + 0.1:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
