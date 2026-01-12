"""TUDelftGroup2 from ANAC 2014."""

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

__all__ = ["TUDelftGroup2"]


class TUDelftGroup2(SAONegotiator):
    """
    TUDelftGroup2 (Group2Agent) from ANAC 2014.

    Developed by a team at Delft University of Technology, this agent combines
    polynomial time-dependent concession with sophisticated opponent modeling
    that learns both value preferences and issue importance.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation:
          agents.anac.y2014.TUDelftGroup2.Group2Agent

    **Offering Strategy:**
        Polynomial concession with issue-weighted opponent consideration:
        - Polynomial curve: concession = t^beta * range * 0.5
        - Acceleration near deadline (t > 0.95): extra -= progress * 0.1
        - Target = max_utility - concession - late_extra

        Bid selection uses weighted sum approach:
        score = alpha * own_utility + (1 - alpha) * estimated_opponent_utility
        where alpha defaults to 0.6, favoring own utility while considering
        opponent satisfaction for efficient outcomes.

    **Acceptance Strategy:**
        Threshold and comparative acceptance:
        1. Accept if offer utility meets polynomial target threshold
        2. Accept if offer matches or exceeds next planned bid utility
        3. Late-game acceptance (t > 0.99) for offers above minimum
        The polynomial curve ensures smooth concession toward agreement.

    **Opponent Modeling:**
        Dual-layer preference learning (value + issue importance):
        - Value weights: time-weighted frequencies (weight = 1 + time * 3)
        - Issue importance: estimated from value variance in opponent bids
          (fewer unique values = higher importance, 1 / unique_count)
        - Utility estimation: weighted sum normalized by total importance

        The issue importance estimation identifies which issues matter most
        to the opponent, allowing strategic concessions on less important
        issues while maintaining ground on critical ones.

    Args:
        alpha: Weight for own utility in bid selection (default 0.6).
        beta: Exponent for polynomial concession curve (default 0.5).
        acceleration_time: Time threshold for accelerated concession (default 0.95).
        late_acceptance_time: Time threshold for late-game acceptance (default 0.99).
        time_weight_multiplier: Multiplier for time-based weighting (default 3.0).
        concession_multiplier: Multiplier for base concession (default 0.5).
        acceleration_factor: Factor for late-game acceleration (default 0.1).
        top_candidates_divisor: Divisor for selecting top candidates (default 3).
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
        alpha: float = 0.6,
        beta: float = 0.5,
        acceleration_time: float = 0.95,
        late_acceptance_time: float = 0.99,
        time_weight_multiplier: float = 3.0,
        concession_multiplier: float = 0.5,
        acceleration_factor: float = 0.1,
        top_candidates_divisor: int = 3,
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
        self._alpha = alpha
        self._beta = beta
        self._acceleration_time = acceleration_time
        self._late_acceptance_time = late_acceptance_time
        self._time_weight_multiplier = time_weight_multiplier
        self._concession_multiplier = concession_multiplier
        self._acceleration_factor = acceleration_factor
        self._top_candidates_divisor = top_candidates_divisor
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_value_weights: dict[int, dict] = {}
        self._opponent_issue_weights: dict[int, float] = {}

        # State
        self._min_utility: float = 0.5
        self._max_utility: float = 1.0
        self._round_count: int = 0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = max(0.5, self._outcome_space.min_utility)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_value_weights = {}
        self._opponent_issue_weights = {}
        self._round_count = 0

    def _update_opponent_model(self, bid: Outcome, time: float) -> None:
        """Update opponent model with weighted value frequencies."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append(bid)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        # Weight by time (later bids are more informative)
        weight = 1.0 + time * self._time_weight_multiplier

        for i, value in enumerate(bid):
            if i not in self._opponent_value_weights:
                self._opponent_value_weights[i] = {}
            self._opponent_value_weights[i][value] = (
                self._opponent_value_weights[i].get(value, 0) + weight
            )

        # Estimate issue weights from variance
        self._estimate_issue_weights()

    def _estimate_issue_weights(self) -> None:
        """Estimate opponent's issue weights from value variance."""
        if len(self._opponent_bids) < 3:
            return

        for i in range(len(self._opponent_bids[0])):
            values = [bid[i] for bid in self._opponent_bids]
            unique_values = len(set(values))
            # More unique values = less important issue (opponent is flexible)
            # Fewer unique values = more important (opponent is fixed)
            self._opponent_issue_weights[i] = 1.0 / max(unique_values, 1)

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility using learned model."""
        if not self._opponent_value_weights:
            return 0.5

        total_score = 0.0
        total_weight = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            issue_weight = self._opponent_issue_weights.get(i, 1.0)
            if i in self._opponent_value_weights:
                value_weight = self._opponent_value_weights[i].get(value, 0)
                max_weight = max(self._opponent_value_weights[i].values())
                if max_weight > 0:
                    total_score += issue_weight * (value_weight / max_weight)
                    total_weight += issue_weight

        if total_weight > 0:
            return total_score / total_weight
        return 0.5

    def _compute_target_utility(self, time: float) -> float:
        """Compute target utility using polynomial concession."""
        # Polynomial curve: starts at max, ends near min
        concession = (
            (time**self._beta)
            * (self._max_utility - self._min_utility)
            * self._concession_multiplier
        )
        target = self._max_utility - concession

        # Accelerate near deadline
        if time > self._acceleration_time:
            extra = (
                (time - self._acceleration_time)
                / (1.0 - self._acceleration_time)
                * self._acceleration_factor
            )
            target -= extra

        return max(self._min_utility, target)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid using weighted sum approach."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._compute_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            # Fallback to best available
            return self._outcome_space.outcomes[0].bid

        # Score candidates by weighted sum of own + opponent utility
        if self._opponent_value_weights:
            best_bid = None
            best_score = -1.0

            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                # Weighted combination
                score = self._alpha * bd.utility + (1 - self._alpha) * opp_util
                if score > best_score:
                    best_score = score
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        # No model yet, random from top candidates
        return random.choice(
            candidates[: max(1, len(candidates) // self._top_candidates_divisor)]
        ).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        self._round_count += 1

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

        # Late game acceptance
        if time > self._late_acceptance_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
