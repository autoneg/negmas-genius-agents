"""MeanBot from ANAC 2015."""

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
    MeanBot negotiation agent from ANAC 2015.

    MeanBot uses a mean-based strategy that tracks opponent offer statistics
    to adjust its negotiation threshold and acceptance decisions.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.meanBot.meanBot

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Three-phase concession with mean-based targets:
          * Early (t<0.2): Stays high at 93% of max utility
          * Main (0.2<t<0.8): Concedes toward mean-based target
            (mean opponent utility + 0.1 or 0.6, whichever is higher)
          * End (t>0.8): More aggressive concession toward minimum
        - Adaptive rate: firmer (e * 0.8) if mean opponent > 50%,
          more flexible (e * 1.3) if mean opponent < 30%
        - Random selection from candidates above threshold

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - Statistical acceptance: Accepts if offer >= mean + 0.2
          AND offer >= minimum acceptable
        - End-game (t>0.95): Accepts if offer >= best opponent utility
          OR offer >= minimum acceptable

    **Opponent Modeling:**
        - Tracks all opponent utilities to compute running mean
        - Maintains best opponent utility for end-game reference
        - Uses mean to adjust target utility in main phase
        - Statistical approach influences both offering and acceptance

    Args:
        e: Concession exponent (default 0.2)
        early_time_threshold: Time before which agent stays high at 93% (default 0.2)
        main_time_threshold: Time before which agent is in main phase (default 0.8)
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
        e: float = 0.2,
        early_time_threshold: float = 0.2,
        main_time_threshold: float = 0.8,
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
        self._e = e
        self._early_time_threshold = early_time_threshold
        self._main_time_threshold = main_time_threshold
        self._deadline_time_threshold = deadline_time_threshold
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

        if time < self._early_time_threshold:
            # Early: stay high
            return self._max_utility * 0.93
        elif time < self._main_time_threshold:
            # Main phase: concede toward mean-based target
            progress = (time - self._early_time_threshold) / (
                self._main_time_threshold - self._early_time_threshold
            )
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
            progress = (time - self._main_time_threshold) / (
                1.0 - self._main_time_threshold
            )
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
        if time > self._deadline_time_threshold:
            if offer_utility >= max(self._best_opponent_utility, self._min_acceptable):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
