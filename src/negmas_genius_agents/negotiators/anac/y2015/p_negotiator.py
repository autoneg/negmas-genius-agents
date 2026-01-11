"""
PNegotiator from ANAC 2015.

PNegotiator uses a probabilistic strategy:
1. Probability-based bid selection
2. Time-dependent threshold with probabilistic acceptance
3. Opponent modeling for probability estimation
4. Expected utility optimization

Original: agents.anac.y2015.pnegotiator.PNegotiator

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

__all__ = ["PNegotiator"]


class PNegotiator(SAONegotiator):
    """
    PNegotiator from ANAC 2015.

    PNegotiator uses a probabilistic strategy:
    1. Probability-based bid selection
    2. Time-dependent threshold with probabilistic acceptance
    3. Opponent modeling for probability estimation
    4. Expected utility optimization

    Key features:
    - Probabilistic bid selection
    - Expected value calculations
    - Opponent acceptance probability estimation
    - Risk-aware decision making

    Args:
        e: Concession exponent (default 0.18)
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
        e: float = 0.18,
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

        # Opponent model for probability estimation
        self._opponent_utilities: list[float] = []
        self._best_opponent_utility: float = 0.0
        self._estimated_opponent_threshold: float = 0.5

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
        self._best_opponent_utility = 0.0
        self._estimated_opponent_threshold = 0.5

    def _update_opponent_model(self, utility: float) -> None:
        """Update opponent model for probability estimation."""
        self._opponent_utilities.append(utility)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Estimate opponent's minimum acceptable utility
        if len(self._opponent_utilities) >= 3:
            # Opponent probably won't accept below their worst offer to us
            min_offer = min(self._opponent_utilities)
            self._estimated_opponent_threshold = max(
                0.3,
                1.0 - self._best_opponent_utility,  # Rough inverse estimate
            )

    def _estimate_acceptance_probability(
        self, our_utility: float, time: float
    ) -> float:
        """Estimate probability opponent will accept our offer."""
        # Higher our utility = lower probability they accept
        # As time increases, they're more likely to accept

        time_factor = math.pow(time, 2)  # Increases acceptance probability over time
        utility_factor = 1.0 - our_utility  # Lower our utility = higher acceptance

        prob = 0.2 + 0.6 * utility_factor + 0.2 * time_factor
        return max(0.0, min(1.0, prob))

    def _compute_threshold(self, time: float) -> float:
        """Compute threshold with probabilistic approach."""
        f_t = math.pow(time, 1 / self._e) if self._e > 0 else time

        target = self._max_utility - (self._max_utility - self._min_acceptable) * f_t
        return max(target, self._min_acceptable)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid based on expected utility."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.85)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Select based on expected utility
        if len(candidates) > 5:
            best_bid = None
            best_expected = -1.0

            for bd in candidates[:30]:
                prob = self._estimate_acceptance_probability(bd.utility, time)
                expected = bd.utility * prob
                if expected > best_expected:
                    best_expected = expected
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

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

        self._update_opponent_model(offer_utility)

        threshold = self._compute_threshold(time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Probabilistic acceptance: sometimes accept sub-threshold offers
        if time > 0.7:
            accept_prob = (offer_utility - self._min_acceptable) / (
                threshold - self._min_acceptable + 0.01
            )
            accept_prob *= time  # Higher probability as deadline approaches
            if random.random() < accept_prob * 0.3:
                if offer_utility >= self._min_acceptable:
                    return ResponseType.ACCEPT_OFFER

        # End-game
        if time > 0.95:
            if offer_utility >= max(self._best_opponent_utility, self._min_acceptable):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
