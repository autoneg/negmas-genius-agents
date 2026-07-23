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
        min_acceptable: Minimum acceptable utility (default 0.5)
        generous_threshold: Mean opponent utility above which agent stays firmer (default 0.5)
        tough_threshold: Mean opponent utility below which agent concedes more (default 0.3)
        firm_multiplier: Multiplier applied to e when opponent is generous (default 0.8)
        flexible_multiplier: Multiplier applied to e when opponent is tough (default 1.3)
        max_utility_ratio: Ratio of max utility used as firm starting threshold (default 0.93)
        mean_target_margin: Margin added to mean opponent utility for target (default 0.1)
        mean_target_default: Default target when no opponent utilities observed (default 0.6)
        end_phase_concession_factor: Fraction of concession applied in end phase (default 0.5)
        fallback_threshold_ratio: Ratio used when lowering threshold if no candidates (default 0.9)
        statistical_acceptance_margin: Margin above mean for statistical acceptance (default 0.2)
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
        min_acceptable: float = 0.5,
        generous_threshold: float = 0.5,
        tough_threshold: float = 0.3,
        firm_multiplier: float = 0.8,
        flexible_multiplier: float = 1.3,
        max_utility_ratio: float = 0.93,
        mean_target_margin: float = 0.1,
        mean_target_default: float = 0.6,
        end_phase_concession_factor: float = 0.5,
        fallback_threshold_ratio: float = 0.9,
        statistical_acceptance_margin: float = 0.2,
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
        self._min_acceptable = min_acceptable
        self._generous_threshold = generous_threshold
        self._tough_threshold = tough_threshold
        self._firm_multiplier = firm_multiplier
        self._flexible_multiplier = flexible_multiplier
        self._max_utility_ratio = max_utility_ratio
        self._mean_target_margin = mean_target_margin
        self._mean_target_default = mean_target_default
        self._end_phase_concession_factor = end_phase_concession_factor
        self._fallback_threshold_ratio = fallback_threshold_ratio
        self._statistical_acceptance_margin = statistical_acceptance_margin
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0

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
        if self._mean_opponent_utility > self._generous_threshold:
            e *= self._firm_multiplier  # Opponent generous, stay firm
        elif self._mean_opponent_utility < self._tough_threshold:
            e *= self._flexible_multiplier  # Opponent tough, concede more

        if time < self._early_time_threshold:
            # Early: stay high
            return self._max_utility * self._max_utility_ratio
        elif time < self._main_time_threshold:
            # Main phase: concede toward mean-based target
            progress = (time - self._early_time_threshold) / (
                self._main_time_threshold - self._early_time_threshold
            )
            f_t = math.pow(progress, 1 / e)

            # Target is adjusted based on mean
            target_min = max(
                self._mean_opponent_utility + self._mean_target_margin
                if self._opponent_utilities
                else self._mean_target_default,
                self._min_acceptable,
            )

            return (
                self._max_utility * self._max_utility_ratio
                - (self._max_utility * self._max_utility_ratio - target_min) * f_t
            )
        else:
            # End phase: more aggressive
            progress = (time - self._main_time_threshold) / (
                1.0 - self._main_time_threshold
            )
            base = max(
                self._mean_opponent_utility + self._mean_target_margin,
                self._min_acceptable,
            )
            target = self._min_acceptable
            return base - (base - target) * progress * self._end_phase_concession_factor

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid based on threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(
                threshold * self._fallback_threshold_ratio
            )

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
        if offer_utility >= self._mean_opponent_utility + self._statistical_acceptance_margin:
            if offer_utility >= self._min_acceptable:
                return ResponseType.ACCEPT_OFFER

        # End-game
        if time > self._deadline_time_threshold:
            if offer_utility >= max(self._best_opponent_utility, self._min_acceptable):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER