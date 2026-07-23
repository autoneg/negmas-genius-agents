"""Kawaii from ANAC 2015."""

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

__all__ = ["Kawaii"]


class Kawaii(SAONegotiator):
    """
    Kawaii negotiation agent from ANAC 2015 (Fairy agent).

    Kawaii uses a soft, adaptive strategy that adjusts concession based on
    opponent behavior patterns.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.fairy.kawaii

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Two-phase adaptive concession:
          * Early phase (t<0.3): Stays high with no concession (f_t = 0)
          * Main phase (t>=0.3): Smooth Boulware-like concession with
            adaptive exponent based on opponent behavior
        - Concedes up to 40% of utility range based on time
        - Adaptive rate: firmer (e * 0.7) if opponent is conceding,
          more flexible (e * 1.5) if opponent is tough
        - Gentle randomization in bid selection

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - Near deadline (t>0.95): Accepts if offer >= 30% of utility range
          above minimum

    **Opponent Modeling:**
        - Tracks opponent utility history to estimate concession rate
        - Calculates average change in opponent offers
        - Uses concession rate to adapt own exponent:
          * Positive rate (opponent conceding): stay firmer
          * Negative rate (opponent tough): concede more

    Args:
        e: Initial concession exponent (default 0.2)
        early_time_threshold: Time before which agent stays high with no concession (default 0.3)
        deadline_time_threshold: Time after which agent becomes more accepting (default 0.95)
        concession_rate_min_bids: Minimum opponent bids to estimate concession rate (default 3)
        adaptation_min_observations: Minimum opponent bids before adapting exponent (default 5)
        conceding_rate_threshold: Magnitude of concession rate considered significant (default 0.01)
        firm_multiplier: Multiplier applied to e when opponent is conceding (default 0.7)
        flexible_multiplier: Multiplier applied to e when opponent is tough (default 1.5)
        concession_factor: Fraction of utility range conceded over time (default 0.4)
        fallback_threshold_ratio: Ratio used when lowering threshold if no candidates (default 0.9)
        deadline_acceptance_fraction: Fraction of utility range used for near-deadline acceptance (default 0.3)
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
        early_time_threshold: float = 0.3,
        deadline_time_threshold: float = 0.95,
        concession_rate_min_bids: int = 3,
        adaptation_min_observations: int = 5,
        conceding_rate_threshold: float = 0.01,
        firm_multiplier: float = 0.7,
        flexible_multiplier: float = 1.5,
        concession_factor: float = 0.4,
        fallback_threshold_ratio: float = 0.9,
        deadline_acceptance_fraction: float = 0.3,
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
        self._deadline_time_threshold = deadline_time_threshold
        self._concession_rate_min_bids = concession_rate_min_bids
        self._adaptation_min_observations = adaptation_min_observations
        self._conceding_rate_threshold = conceding_rate_threshold
        self._firm_multiplier = firm_multiplier
        self._flexible_multiplier = flexible_multiplier
        self._concession_factor = concession_factor
        self._fallback_threshold_ratio = fallback_threshold_ratio
        self._deadline_acceptance_fraction = deadline_acceptance_fraction
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._reservation_value: float = 0.0

        # Opponent tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._opponent_utilities: list[float] = []
        self._opponent_concession_rate: float = 0.0

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
        self._opponent_utilities = []
        self._opponent_concession_rate = 0.0

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent behavior and estimate concession rate."""
        self._opponent_bids.append((bid, utility))
        self._opponent_utilities.append(utility)

        if len(self._opponent_utilities) >= self._concession_rate_min_bids:
            # Calculate average change in utility offered
            changes = []
            for i in range(1, len(self._opponent_utilities)):
                changes.append(
                    self._opponent_utilities[i] - self._opponent_utilities[i - 1]
                )
            self._opponent_concession_rate = sum(changes) / len(changes)

    def _adaptive_e(self, time: float) -> float:
        """Adjust concession exponent based on opponent behavior."""
        base_e = self._e

        if len(self._opponent_utilities) < self._adaptation_min_observations:
            return base_e

        # If opponent is conceding (giving us better deals), we can be firmer
        if self._opponent_concession_rate > self._conceding_rate_threshold:
            return base_e * self._firm_multiplier  # More boulware-like

        # If opponent is tough, we need to concede more
        if self._opponent_concession_rate < -self._conceding_rate_threshold:
            return base_e * self._flexible_multiplier  # More conceder-like

        return base_e

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold with adaptive concession."""
        e = self._adaptive_e(time)

        # Kawaii formula: gentle but persistent
        if time < self._early_time_threshold:
            # Early phase: stay high
            f_t = 0.0
        else:
            # Main phase: smooth concession
            adjusted_time = (time - self._early_time_threshold) / (
                1.0 - self._early_time_threshold
            )
            f_t = math.pow(adjusted_time, 1 / e) if e != 0 else adjusted_time

        target = self._max_utility - (
            self._max_utility - self._min_utility
        ) * self._concession_factor * f_t
        return max(target, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid to offer."""
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

        # Gentle randomization
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

        # Accept if meets threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, be more accepting
        if time > self._deadline_time_threshold:
            if offer_utility >= self._min_utility + self._deadline_acceptance_fraction * (
                self._max_utility - self._min_utility
            ):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER