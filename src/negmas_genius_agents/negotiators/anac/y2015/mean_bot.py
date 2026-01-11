"""
MeanBot from ANAC 2015.

MeanBot uses a mean-based strategy:
1. Tracks mean of opponent offers
2. Adjusts threshold based on average opponent behavior
3. Time-dependent concession with mean consideration
4. Statistical approach to acceptance

Original: agents.anac.y2015.meanBot.meanBot

References:
    - https://ii.tudelft.nl/negotiation/node/12 (ANAC 2015)
    - Genius negotiation framework: https://ii.tudelft.nl/genius/
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

__all__ = ["MeanBot"]


class MeanBot(SAONegotiator):
    """
    MeanBot from ANAC 2015.

    MeanBot uses a mean-based strategy:
    1. Tracks mean of opponent offers
    2. Adjusts threshold based on average opponent behavior
    3. Time-dependent concession with mean consideration
    4. Statistical approach to acceptance

    Key features:
    - Statistical analysis of opponent offers
    - Mean-based threshold adjustment
    - Adaptive concession rate
    - Data-driven acceptance

    Args:
        e: Concession exponent (default 0.2)
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

        # Opponent statistics
        self._opponent_utilities: list[float] = []
        self._mean_opponent_utility: float = 0.0
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

        self._opponent_utilities = []
        self._mean_opponent_utility = 0.0
        self._best_opponent_utility = 0.0

    def _update_statistics(self, utility: float) -> None:
        """Update statistical measures of opponent behavior."""
        self._opponent_utilities.append(utility)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Update mean
        self._mean_opponent_utility = sum(self._opponent_utilities) / len(
            self._opponent_utilities
        )

    def _compute_threshold(self, time: float) -> float:
        """Compute threshold using mean-based strategy."""
        # Adjust e based on mean opponent utility
        e = self._e
        if self._mean_opponent_utility > 0.5:
            e *= 0.8  # Opponent generous, stay firm
        elif self._mean_opponent_utility < 0.3:
            e *= 1.3  # Opponent tough, concede more

        if time < 0.2:
            # Early: stay high
            return self._max_utility * 0.93
        elif time < 0.8:
            # Main phase: concede toward mean-based target
            progress = (time - 0.2) / 0.6
            f_t = math.pow(progress, 1 / e)

            # Target is adjusted based on mean
            target_min = max(
                self._mean_opponent_utility + 0.1 if self._opponent_utilities else 0.6,
                self._min_acceptable,
            )

            return (
                self._max_utility * 0.93 - (self._max_utility * 0.93 - target_min) * f_t
            )
        else:
            # End phase: more aggressive
            progress = (time - 0.8) / 0.2
            base = max(self._mean_opponent_utility + 0.1, self._min_acceptable)
            target = self._min_acceptable
            return base - (base - target) * progress * 0.5

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

        self._update_statistics(offer_utility)

        threshold = self._compute_threshold(time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Accept if significantly above mean
        if offer_utility >= self._mean_opponent_utility + 0.2:
            if offer_utility >= self._min_acceptable:
                return ResponseType.ACCEPT_OFFER

        # End-game
        if time > 0.95:
            if offer_utility >= max(self._best_opponent_utility, self._min_acceptable):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
