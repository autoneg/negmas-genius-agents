"""PNegotiator from ANAC 2015."""

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
    PNegotiator negotiation agent from ANAC 2015.

    PNegotiator uses a probabilistic strategy with expected utility
    optimization and risk-aware decision making.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.pnegotiator.PNegotiator

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Standard Boulware-like concession:
          threshold = max_util - (max_util - min_acceptable) * t^(1/e)
        - Expected utility bid selection: for each candidate, estimates
          acceptance probability and computes expected value
        - Acceptance probability formula: 0.2 + 0.6*(1-our_util) + 0.2*t^2
        - Selects bid with highest expected utility from top 30 candidates

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - Probabilistic acceptance (t>0.7): May accept sub-threshold offers
          with probability proportional to (offer - min) / (threshold - min) * t
        - End-game (t>0.95): Accepts if offer >= best opponent utility
          OR offer >= minimum acceptable

    **Opponent Modeling:**
        - Tracks opponent utilities to estimate opponent threshold
        - Estimates opponent's minimum acceptable as max(0.3, 1 - best_opponent_utility)
        - Uses opponent model in acceptance probability estimation

    Args:
        e: Concession exponent (default 0.18)
        probabilistic_accept_time_threshold: Time after which probabilistic acceptance starts (default 0.7)
        deadline_time_threshold: Time after which end-game acceptance triggers (default 0.95)
        min_acceptable: Minimum acceptable utility (default 0.5)
        opponent_model_min_bids: Minimum opponent bids before estimating opponent threshold (default 3)
        opponent_threshold_floor: Floor for estimated opponent threshold (default 0.3)
        time_exponent: Exponent applied to time in acceptance probability (default 2)
        base_acceptance_prob: Base acceptance probability component (default 0.2)
        utility_weight: Weight of utility factor in acceptance probability (default 0.6)
        time_weight: Weight of time factor in acceptance probability (default 0.2)
        fallback_threshold_ratio: Ratio used when lowering threshold if no candidates (default 0.85)
        expected_utility_min_candidates: Minimum candidates to use expected utility selection (default 5)
        expected_utility_search_count: Number of top candidates searched for expected utility (default 30)
        acceptance_epsilon: Epsilon added to threshold-min denominator (default 0.01)
        acceptance_probability_scale: Scale applied to acceptance probability (default 0.3)
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
        probabilistic_accept_time_threshold: float = 0.7,
        deadline_time_threshold: float = 0.95,
        min_acceptable: float = 0.5,
        opponent_model_min_bids: int = 3,
        opponent_threshold_floor: float = 0.3,
        time_exponent: float = 2,
        base_acceptance_prob: float = 0.2,
        utility_weight: float = 0.6,
        time_weight: float = 0.2,
        fallback_threshold_ratio: float = 0.85,
        expected_utility_min_candidates: int = 5,
        expected_utility_search_count: int = 30,
        acceptance_epsilon: float = 0.01,
        acceptance_probability_scale: float = 0.3,
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
        self._probabilistic_accept_time_threshold = probabilistic_accept_time_threshold
        self._deadline_time_threshold = deadline_time_threshold
        self._min_acceptable = min_acceptable
        self._opponent_model_min_bids = opponent_model_min_bids
        self._opponent_threshold_floor = opponent_threshold_floor
        self._time_exponent = time_exponent
        self._base_acceptance_prob = base_acceptance_prob
        self._utility_weight = utility_weight
        self._time_weight = time_weight
        self._fallback_threshold_ratio = fallback_threshold_ratio
        self._expected_utility_min_candidates = expected_utility_min_candidates
        self._expected_utility_search_count = expected_utility_search_count
        self._acceptance_epsilon = acceptance_epsilon
        self._acceptance_probability_scale = acceptance_probability_scale
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0

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
        if len(self._opponent_utilities) >= self._opponent_model_min_bids:
            # Opponent probably won't accept below their worst offer to us
            min_offer = min(self._opponent_utilities)
            self._estimated_opponent_threshold = max(
                self._opponent_threshold_floor,
                1.0 - self._best_opponent_utility,  # Rough inverse estimate
            )

    def _estimate_acceptance_probability(
        self, our_utility: float, time: float
    ) -> float:
        """Estimate probability opponent will accept our offer."""
        # Higher our utility = lower probability they accept
        # As time increases, they're more likely to accept

        time_factor = math.pow(time, self._time_exponent)  # Increases acceptance probability over time
        utility_factor = 1.0 - our_utility  # Lower our utility = higher acceptance

        prob = (
            self._base_acceptance_prob
            + self._utility_weight * utility_factor
            + self._time_weight * time_factor
        )
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
            candidates = self._outcome_space.get_bids_above(
                threshold * self._fallback_threshold_ratio
            )

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Select based on expected utility
        if len(candidates) > self._expected_utility_min_candidates:
            best_bid = None
            best_expected = -1.0

            for bd in candidates[: self._expected_utility_search_count]:
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
        if time > self._probabilistic_accept_time_threshold:
            accept_prob = (offer_utility - self._min_acceptable) / (
                threshold - self._min_acceptable + self._acceptance_epsilon
            )
            accept_prob *= time  # Higher probability as deadline approaches
            if random.random() < accept_prob * self._acceptance_probability_scale:
                if offer_utility >= self._min_acceptable:
                    return ResponseType.ACCEPT_OFFER

        # End-game
        if time > self._deadline_time_threshold:
            if offer_utility >= max(self._best_opponent_utility, self._min_acceptable):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER