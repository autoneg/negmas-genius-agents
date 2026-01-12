"""Aster from ANAC 2014."""

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

__all__ = ["Aster"]


class Aster(SAONegotiator):
    """
    Aster from ANAC 2014.

    Aster employs a "star pattern" strategy that simultaneously considers
    multiple decision criteria. Named for its multi-pointed approach to
    negotiation decisions, it balances own utility, opponent satisfaction,
    and time pressure in an integrated framework.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.Aster.Aster

    **Offering Strategy:**
        Adaptive threshold-based bidding with opponent consideration:
        - Base target computed from time-decayed threshold (t^decay * range)
        - Decay rate adapts to opponent behavior trends:
          * Opponent improving offers: decay *= 0.8 (slow down)
          * Opponent hardening: decay *= 1.2 (speed up)
        - Target = max_util - adaptive_decay * 0.6

        Bid selection uses star pattern scoring:
        score = own_weight * own_utility + opponent_weight * estimated_opp_util
        where own_weight = 1 - opponent_weight (default opponent_weight=0.4)

    **Acceptance Strategy:**
        Multi-criteria acceptance with four conditions (star points):
        1. Primary: Accept if offer utility meets adaptive target
        2. Comparative: Accept if offer >= next planned bid utility
        3. Historical: Accept if offer >= best opponent bid and t > 0.9
        4. Emergency: Accept if t > 0.98 and offer > minimum threshold
        Each criterion addresses different negotiation scenarios.

    **Opponent Modeling:**
        Enhanced frequency analysis with importance weighting:
        - Time-weighted value frequencies (weight = 1 + time * 2)
        - Issue importance estimated from value variance in opponent bids
          (fewer unique values = more important issue for opponent)
        - Combined utility estimate: weighted sum of value scores
        - Trend detection from recent offers to adapt concession rate

    Args:
        threshold_decay: Rate of threshold decay over time (default 0.3).
        opponent_weight: Weight for opponent utility in star pattern (default 0.4).
        time_weight_multiplier: Multiplier for time-based weighting (default 2.0).
        positive_trend_threshold: Threshold for detecting positive trend (default 0.1).
        negative_trend_threshold: Threshold for detecting negative trend (default -0.1).
        positive_trend_decay_factor: Decay multiplier for positive trend (default 0.8).
        negative_trend_decay_factor: Decay multiplier for negative trend (default 1.2).
        decay_multiplier: Multiplier for final decay calculation (default 0.6).
        lowered_threshold_factor: Factor for lowering target when no candidates (default 0.95).
        top_candidates_divisor: Divisor for selecting top candidates (default 3).
        historical_acceptance_time: Time threshold for historical acceptance (default 0.9).
        emergency_acceptance_time: Time threshold for emergency acceptance (default 0.98).
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
        threshold_decay: float = 0.3,
        opponent_weight: float = 0.4,
        time_weight_multiplier: float = 2.0,
        positive_trend_threshold: float = 0.1,
        negative_trend_threshold: float = -0.1,
        positive_trend_decay_factor: float = 0.8,
        negative_trend_decay_factor: float = 1.2,
        decay_multiplier: float = 0.6,
        lowered_threshold_factor: float = 0.95,
        top_candidates_divisor: int = 3,
        historical_acceptance_time: float = 0.9,
        emergency_acceptance_time: float = 0.98,
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
        self._threshold_decay = threshold_decay
        self._opponent_weight = opponent_weight
        self._time_weight_multiplier = time_weight_multiplier
        self._positive_trend_threshold = positive_trend_threshold
        self._negative_trend_threshold = negative_trend_threshold
        self._positive_trend_decay_factor = positive_trend_decay_factor
        self._negative_trend_decay_factor = negative_trend_decay_factor
        self._decay_multiplier = decay_multiplier
        self._lowered_threshold_factor = lowered_threshold_factor
        self._top_candidates_divisor = top_candidates_divisor
        self._historical_acceptance_time = historical_acceptance_time
        self._emergency_acceptance_time = emergency_acceptance_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_value_freq: dict[int, dict] = {}
        self._opponent_issue_importance: dict[int, float] = {}

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
            self._min_utility = max(0.5, self._outcome_space.min_utility)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_value_freq = {}
        self._opponent_issue_importance = {}

    def _update_opponent_model(self, bid: Outcome, time: float) -> None:
        """Update opponent model with frequency and importance analysis."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append(bid)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        # Time-weighted frequency update
        weight = 1.0 + time * self._time_weight_multiplier
        for i, value in enumerate(bid):
            if i not in self._opponent_value_freq:
                self._opponent_value_freq[i] = {}
            self._opponent_value_freq[i][value] = (
                self._opponent_value_freq[i].get(value, 0) + weight
            )

        # Estimate issue importance from value variance
        self._estimate_issue_importance()

    def _estimate_issue_importance(self) -> None:
        """Estimate opponent's issue importance from bid variance."""
        if len(self._opponent_bids) < 3:
            return

        for i in range(len(self._opponent_bids[0])):
            values = [bid[i] for bid in self._opponent_bids]
            unique_count = len(set(values))
            # Fewer unique values = more important issue
            self._opponent_issue_importance[i] = 1.0 / max(unique_count, 1)

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility using frequency and importance."""
        if not self._opponent_value_freq:
            return 0.5

        total_score = 0.0
        total_weight = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            importance = self._opponent_issue_importance.get(i, 1.0)
            if i in self._opponent_value_freq:
                freq = self._opponent_value_freq[i].get(value, 0)
                max_freq = max(self._opponent_value_freq[i].values())
                if max_freq > 0:
                    total_score += importance * (freq / max_freq)
                    total_weight += importance

        if total_weight > 0:
            return total_score / total_weight
        return 0.5

    def _compute_target_utility(self, time: float) -> float:
        """Compute target utility with adaptive decay."""
        # Base decay curve
        decay = (time**self._threshold_decay) * (self._max_utility - self._min_utility)

        # Adapt based on opponent behavior
        if self._opponent_bids and len(self._opponent_bids) > 5:
            recent_utils = [
                float(self.ufun(b)) if self.ufun else 0.5
                for b in self._opponent_bids[-5:]
            ]
            trend = recent_utils[-1] - recent_utils[0]
            if trend > self._positive_trend_threshold:
                # Opponent improving offers, slow decay
                decay *= self._positive_trend_decay_factor
            elif trend < self._negative_trend_threshold:
                # Opponent hardening, speed up decay
                decay *= self._negative_trend_decay_factor

        target = self._max_utility - decay * self._decay_multiplier
        return max(self._min_utility, target)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid using star pattern (multiple criteria)."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._compute_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            lowered = target * self._lowered_threshold_factor
            candidates = self._outcome_space.get_bids_above(lowered)
            if not candidates:
                return self._outcome_space.outcomes[0].bid

        # Star pattern: weight multiple criteria
        if self._opponent_value_freq:
            best_bid = None
            best_score = -1.0

            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                # Star pattern score: own utility + opponent utility
                own_weight = 1.0 - self._opponent_weight
                score = own_weight * bd.utility + self._opponent_weight * opp_util

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
        """Respond using multi-criteria acceptance."""
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

        # Criterion 1: Above target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Criterion 2: Better than our next offer
        next_bid = self._select_bid(time)
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid))
            if offer_utility >= next_utility:
                return ResponseType.ACCEPT_OFFER

        # Criterion 3: Best opponent offer and close to deadline
        if (
            time > self._historical_acceptance_time
            and offer_utility >= self._best_opponent_utility
        ):
            return ResponseType.ACCEPT_OFFER

        # Criterion 4: Emergency acceptance
        if (
            time > self._emergency_acceptance_time
            and offer_utility >= self._min_utility
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
