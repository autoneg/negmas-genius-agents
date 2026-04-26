"""
BetaOne - ANAC 2017 3rd Place.

This module contains the reimplementation of BetaOne, the 3rd place agent
from the Automated Negotiating Agents Competition (ANAC) 2017.

References:
    ANAC 2017 competition proceedings.
    https://ii.tudelft.nl/nego/node/7
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

__all__ = ["BetaOne"]


class BetaOne(SAONegotiator):
    """
    BetaOne from ANAC 2017 - 3rd place agent.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    BetaOne uses a sophisticated strategy combining Bayesian opponent modeling
    with adaptive concession across three negotiation phases.

    **Offering Strategy:**
        Three-phase approach:
        - Phase 1 (0-40%): Conservative, slow concession while gathering data.
        - Phase 2 (40-80%): Adaptive concession based on opponent type estimate.
        - Phase 3 (80-100%): Accelerated concession to reach agreement.
        Bids are selected to explore near the Pareto frontier, weighted by
        estimated opponent preferences.

    **Acceptance Strategy:**
        Accepts offers above a dynamically calculated threshold that varies
        by phase and opponent type. Near the deadline (>95%), accepts offers
        close to the best opponent offer if above minimum utility.

    **Opponent Modeling:**
        Uses Bayesian inference to estimate opponent type (hard vs. soft).
        Updates probability based on offer utilities relative to expected
        values for each type. Tracks concession rate over recent offers.
        Hard opponent estimate leads to slower concession; soft opponent
        estimate allows more patience.

    Args:
        min_utility: Minimum acceptable utility floor (default 0.65).
        initial_threshold: Starting threshold for acceptance (default 0.95).
        phase1_end: End of phase 1 (information gathering) (default 0.4).
        phase2_end: End of phase 2 (adaptive phase) (default 0.8).
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
        min_utility: float = 0.65,
        initial_threshold: float = 0.95,
        phase1_end: float = 0.4,
        phase2_end: float = 0.8,
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
        self._min_utility = min_utility
        self._initial_threshold = initial_threshold
        self._phase1_end = phase1_end
        self._phase2_end = phase2_end
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling state
        self._opponent_bids: list[tuple[float, float]] = []  # (time, utility)
        self._opponent_type_prob: float = 0.5  # P(hard) - probability opponent is hard
        self._opponent_concession_rate: float = 0.0

        # Negotiation state
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._current_threshold: float = initial_threshold
        self._best_opponent_utility: float = 0.0

        # Pareto frontier approximation
        self._pareto_bids: list[Outcome] = []

    def _initialize(self) -> None:
        """Initialize the outcome space and Pareto frontier."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility
            # Initialize Pareto bids as top utility bids
            self._pareto_bids = [
                ob.bid
                for ob in self._outcome_space.outcomes[
                    : min(50, len(self._outcome_space.outcomes))
                ]
            ]

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_bids = []
        self._opponent_type_prob = 0.5
        self._opponent_concession_rate = 0.0
        self._current_threshold = self._initial_threshold
        self._best_opponent_utility = 0.0

    def _update_opponent_model(self, offer: Outcome, time: float) -> None:
        """
        Update Bayesian opponent model based on received offer.

        Uses offer utility and timing to estimate if opponent is hard or soft.
        """
        if self.ufun is None:
            return

        offer_utility = float(self.ufun(offer))

        # Track best opponent offer (from our perspective)
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)

        # Record bid history
        self._opponent_bids.append((time, offer_utility))

        if len(self._opponent_bids) < 2:
            return

        # Calculate opponent's concession rate
        recent_bids = self._opponent_bids[-5:]  # Last 5 bids
        if len(recent_bids) >= 2:
            utility_change = recent_bids[-1][1] - recent_bids[0][1]
            time_change = recent_bids[-1][0] - recent_bids[0][0]

            if time_change > 0:
                self._opponent_concession_rate = utility_change / time_change

        # Bayesian update for opponent type
        # Hard opponent: low concession, low utility offers
        # Soft opponent: higher concession, higher utility offers

        # Likelihood of this offer given hard opponent
        expected_hard = 0.3 + 0.2 * (1 - time)  # Hard opponents offer low utility
        likelihood_hard = math.exp(-2 * (offer_utility - expected_hard) ** 2)

        # Likelihood of this offer given soft opponent
        expected_soft = 0.5 + 0.3 * (1 - time)  # Soft opponents offer higher utility
        likelihood_soft = math.exp(-2 * (offer_utility - expected_soft) ** 2)

        # Bayesian update
        prior_hard = self._opponent_type_prob
        prior_soft = 1 - prior_hard

        posterior_hard = likelihood_hard * prior_hard
        posterior_soft = likelihood_soft * prior_soft

        total = posterior_hard + posterior_soft
        if total > 0:
            self._opponent_type_prob = posterior_hard / total
        else:
            self._opponent_type_prob = 0.5

    def _get_phase(self, time: float) -> int:
        """Determine negotiation phase based on time."""
        if time < self._phase1_end:
            return 1  # Information gathering
        elif time < self._phase2_end:
            return 2  # Adaptive phase
        else:
            return 3  # Concession phase

    def _calculate_threshold(self, time: float) -> float:
        """
        Calculate acceptance threshold based on time and opponent model.

        Adapts concession based on:
        - Current negotiation phase
        - Estimated opponent type
        - Best offer received so far
        """
        phase = self._get_phase(time)

        if phase == 1:
            # Phase 1: Conservative, slow concession
            base_threshold = self._initial_threshold - 0.05 * time
        elif phase == 2:
            # Phase 2: Adaptive based on opponent type
            # Against hard opponent: concede slower
            # Against soft opponent: can concede more
            hardness_factor = self._opponent_type_prob
            concession_rate = 0.2 * (1 - 0.5 * hardness_factor)
            base_threshold = self._initial_threshold - concession_rate * time
        else:
            # Phase 3: More aggressive concession to reach agreement
            time_pressure = (time - self._phase2_end) / (
                1.0 - self._phase2_end
            )  # 0 to 1 in phase 3
            base_threshold = self._initial_threshold - 0.3 - 0.15 * time_pressure

            # If opponent is making good offers, be willing to accept
            if self._best_opponent_utility > base_threshold:
                base_threshold = min(base_threshold, self._best_opponent_utility - 0.02)

        # Apply minimum utility floor
        return max(base_threshold, self._min_utility)

    def _select_pareto_bid(self, threshold: float) -> Outcome | None:
        """
        Select a bid from near the Pareto frontier above threshold.

        Prefers bids that are good for us but might also be acceptable
        to the opponent based on their observed behavior.
        """
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        # Get candidates above threshold
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            # Lower threshold if nothing found
            candidates = self._outcome_space.get_bids_above(self._min_utility)

        if not candidates:
            return self._best_bid

        # If we have opponent history, prefer bids similar to what they offered
        if self._opponent_bids and len(candidates) > 1:
            # Weight candidates by how recently they match opponent's pattern
            # This is a heuristic for Pareto exploration
            weighted_candidates: list[tuple[float, Outcome]] = []
            avg_opponent_utility = sum(u for _, u in self._opponent_bids[-5:]) / min(
                5, len(self._opponent_bids)
            )

            for candidate in candidates:
                utility = candidate.utility
                # Prefer bids that are good for us but not too far from opponent range
                distance_from_opponent = abs(utility - avg_opponent_utility)
                # Higher weight for bids closer to opponent's range but still good for us
                weight = utility * math.exp(-0.5 * distance_from_opponent)
                weighted_candidates.append((weight, candidate.bid))

            # Probabilistic selection based on weights
            total_weight = sum(w for w, _ in weighted_candidates)
            if total_weight > 0:
                r = random.random() * total_weight
                cumulative = 0.0
                for weight, bid in weighted_candidates:
                    cumulative += weight
                    if r <= cumulative:
                        return bid

        # Fallback: random selection from candidates
        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        self._current_threshold = self._calculate_threshold(time)

        return self._select_pareto_bid(self._current_threshold)

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

        # Update opponent model
        self._update_opponent_model(offer, time)

        # Calculate current threshold
        self._current_threshold = self._calculate_threshold(time)

        offer_utility = float(self.ufun(offer))

        # Accept if above threshold
        if offer_utility >= self._current_threshold:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if above minimum and better than we might get
        if time > 0.95 and offer_utility >= self._min_utility:
            # Accept if this is the best we've seen
            if offer_utility >= self._best_opponent_utility - 0.01:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
