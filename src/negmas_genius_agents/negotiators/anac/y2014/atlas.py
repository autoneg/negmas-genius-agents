"""Atlas from ANAC 2014."""

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

__all__ = ["Atlas"]


class Atlas(SAONegotiator):
    """
    Atlas from ANAC 2014.

    Atlas is an early version of the Atlas agent series that would later
    evolve into Atlas3 (winner of ANAC 2015). It uses opponent modeling
    combined with smooth time-dependent concession and Pareto frontier
    estimation for intelligent bid selection.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.Atlas.Atlas

    **Offering Strategy:**
        Smooth S-curve concession with Pareto estimation:
        - Sigmoid function: 1 / (1 + exp(-10 * (t - 0.5)))
        - Adjusted by concession speed: sigmoid^(1/speed)
        - Base target = max - adjusted * range * 0.5
        - Pareto estimation: tracks (own_util, opp_util) points from offers,
          identifies non-dominated points, averages their utilities
        - Final target = 0.7 * base + 0.3 * pareto_target

        Bid selection uses Nash product approach:
        score = own_utility * estimated_opponent_utility
        maximizing joint welfare among candidates meeting target.

    **Acceptance Strategy:**
        Target-based acceptance with deadline safety:
        1. Accept if offer utility meets blended (base + Pareto) target
        2. Accept if offer matches or exceeds next planned bid utility
        3. Near-deadline acceptance (t > 0.98) for offers above minimum
        The Pareto-informed target helps find mutually acceptable outcomes.

    **Opponent Modeling:**
        Time-weighted frequency analysis for utility estimation:
        - Value frequencies weighted by time (weight = 1 + time)
        - Later bids receive higher weights as they better reflect
          opponent's true preferences
        - Estimated opponent utility feeds into Pareto frontier tracking
        - Nash product scoring promotes Pareto-efficient bid selection

    Args:
        concession_speed: Speed of S-curve concession (default 1.5).
        min_target: Minimum target utility threshold (default 0.6).
        sigmoid_steepness: Steepness of the sigmoid function (default 10).
        sigmoid_midpoint: Midpoint of the sigmoid function (default 0.5).
        concession_multiplier: Multiplier for base concession (default 0.5).
        base_target_weight: Weight for base target in blending (default 0.7).
        pareto_target_weight: Weight for Pareto target in blending (default 0.3).
        lowered_threshold_factor: Factor for lowering target when no candidates (default 0.95).
        top_candidates_divisor: Divisor for selecting top candidates (default 3).
        deadline_acceptance_time: Time threshold for near-deadline acceptance (default 0.98).
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
        concession_speed: float = 1.5,
        min_target: float = 0.6,
        sigmoid_steepness: float = 10,
        sigmoid_midpoint: float = 0.5,
        concession_multiplier: float = 0.5,
        base_target_weight: float = 0.7,
        pareto_target_weight: float = 0.3,
        lowered_threshold_factor: float = 0.95,
        top_candidates_divisor: int = 3,
        deadline_acceptance_time: float = 0.98,
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
        self._concession_speed = concession_speed
        self._min_target = min_target
        self._sigmoid_steepness = sigmoid_steepness
        self._sigmoid_midpoint = sigmoid_midpoint
        self._concession_multiplier = concession_multiplier
        self._base_target_weight = base_target_weight
        self._pareto_target_weight = pareto_target_weight
        self._lowered_threshold_factor = lowered_threshold_factor
        self._top_candidates_divisor = top_candidates_divisor
        self._deadline_acceptance_time = deadline_acceptance_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_value_freq: dict[int, dict] = {}

        # Pareto estimation
        self._estimated_pareto: list[tuple[float, float]] = []

        # State
        self._min_utility: float = 0.5
        self._max_utility: float = 1.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = max(self._min_target, self._outcome_space.min_utility)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_value_freq = {}
        self._estimated_pareto = []

    def _update_opponent_model(self, bid: Outcome, time: float) -> None:
        """Update opponent model and Pareto estimation."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append(bid)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        # Update value frequencies
        weight = 1.0 + time
        for i, value in enumerate(bid):
            if i not in self._opponent_value_freq:
                self._opponent_value_freq[i] = {}
            self._opponent_value_freq[i][value] = (
                self._opponent_value_freq[i].get(value, 0) + weight
            )

        # Update Pareto estimation
        opp_util = self._estimate_opponent_utility(bid)
        self._estimated_pareto.append((utility, opp_util))

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility using frequency model."""
        if not self._opponent_value_freq:
            return 0.5

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            if i in self._opponent_value_freq:
                freq = self._opponent_value_freq[i].get(value, 0)
                max_freq = max(self._opponent_value_freq[i].values())
                if max_freq > 0:
                    total_score += freq / max_freq

        return total_score / num_issues if num_issues > 0 else 0.5

    def _get_pareto_target(self) -> float:
        """Estimate target from Pareto frontier."""
        if not self._estimated_pareto:
            return self._max_utility

        # Find points on estimated Pareto frontier
        # Sort by own utility, keep only non-dominated
        sorted_points = sorted(self._estimated_pareto, key=lambda x: -x[0])

        max_opp = 0.0
        pareto_points = []
        for own, opp in sorted_points:
            if opp > max_opp:
                pareto_points.append((own, opp))
                max_opp = opp

        if pareto_points:
            # Target is average of Pareto-optimal utilities
            avg_util = sum(p[0] for p in pareto_points) / len(pareto_points)
            return max(self._min_utility, avg_util)

        return self._max_utility

    def _compute_target_utility(self, time: float) -> float:
        """Compute target utility using smooth concession curve."""
        # Smooth S-curve concession
        # Uses sigmoid-like function: slow at start, faster middle, slow at end
        sigmoid_time = 1 / (
            1 + math.exp(-self._sigmoid_steepness * (time - self._sigmoid_midpoint))
        )

        # Adjust with concession speed
        adjusted_time = sigmoid_time ** (1 / self._concession_speed)

        base_target = (
            self._max_utility
            - adjusted_time
            * (self._max_utility - self._min_utility)
            * self._concession_multiplier
        )

        # Consider Pareto estimation
        pareto_target = self._get_pareto_target()

        # Blend base and Pareto targets
        target = (
            self._base_target_weight * base_target
            + self._pareto_target_weight * pareto_target
        )

        return max(self._min_utility, min(self._max_utility, target))

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid considering opponent preferences."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._compute_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            lowered = target * self._lowered_threshold_factor
            candidates = self._outcome_space.get_bids_above(lowered)
            if not candidates:
                return self._outcome_space.outcomes[0].bid

        # Use opponent model to select bid
        if self._opponent_value_freq:
            best_bid = None
            best_score = -1.0

            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                # Nash-like product
                score = bd.utility * opp_util
                if score > best_score:
                    best_score = score
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        return random.choice(
            candidates[: max(1, len(candidates) // self._top_candidates_divisor)]
        ).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        return self._select_bid(time)

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
        self._update_opponent_model(offer, time)

        target = self._compute_target_utility(time)

        # Accept if above target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than our next offer
        next_bid = self._select_bid(time)
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid))
            if offer_utility >= next_utility:
                return ResponseType.ACCEPT_OFFER

        # Near deadline
        if time > self._deadline_acceptance_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
