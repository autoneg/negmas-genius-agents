"""Gangster from ANAC 2014."""

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

__all__ = ["Gangster"]


class Gangster(SAONegotiator):
    """
    Gangster - 3rd Place ANAC 2014.

    Gangster achieved third place in ANAC 2014 using a distinctive multi-strategy
    voting approach. A "gang" of internal strategies vote on acceptance decisions,
    with votes aggregated using time-dependent weights.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.Gangster.Gangster

    **Offering Strategy:**
        Conservative-to-aggressive concession with opponent consideration:
        - Three-phase threshold computation:
          * Early (t < 0.8): Slow concession = rate * (t/0.8) * 0.3
          * Middle (0.8 <= t < 0.95): Medium = base + rate * 0.4 * progress
          * Late (t >= 0.95): Rapid = base + rate * 0.3 * progress
        - Pressure factor adjusts concession based on opponent behavior
          (conceding opponent -> tougher stance, tough opponent -> faster concede)
        - Bids selected to maximize estimated opponent utility among candidates
          meeting threshold, promoting efficient agreements

    **Acceptance Strategy:**
        Gang voting mechanism with five internal strategies:
        1. Time Strategy: Accept if offer >= time-adjusted threshold
        2. Threshold Strategy: Accept if offer >= current computed threshold
        3. Relative Strategy: Score = offer / best_opponent_utility (0-1)
        4. Pressure Strategy: Accept if offer >= threshold / pressure_factor
        5. Risk Strategy: Accept if offer >= 0.7 + 0.25 * (1 - time)

        Votes aggregated with time-dependent weights:
        - Early: Favor threshold (0.35) and pressure (0.20)
        - Middle: Balanced weights
        - Late: Favor risk (0.30) and time (0.25) strategies
        Accept if weighted sum >= 0.5. Also accept if offer >= next bid utility.

    **Opponent Modeling:**
        Dual-weighted frequency analysis:
        - Recency weight: 1 + num_bids * 0.1 (recent bids matter more)
        - Time weight: 1 + time * 2 (late bids reveal more)
        - Combined weight = recency * time for frequency updates
        - Pressure factor tracking: compares first/second half of recent
          utilities to detect opponent concession trends
          * Improving: pressure -= 0.05 (min 0.5)
          * Hardening: pressure += 0.05 (max 2.0)

    Args:
        initial_threshold: Starting acceptance threshold (default 0.95).
        concession_rate: Base rate of concession over time (default 0.1).
        slow_concession_end: Time threshold ending slow concession phase (default 0.8).
        medium_concession_end: Time threshold ending medium concession phase (default 0.95).
        early_vote_end: Time threshold ending early voting weights (default 0.5).
        late_vote_start: Time threshold starting late voting weights (default 0.85).
        recency_weight_increment: Increment for recency weight per bid (default 0.1).
        time_weight_multiplier: Multiplier for time-based weight (default 2.0).
        slow_concession_multiplier: Concession multiplier for early phase (default 0.3).
        medium_concession_multiplier: Concession multiplier for middle phase (default 0.4).
        fast_concession_base: Base concession for late phase (default 0.7).
        fast_concession_additional: Additional concession for late phase (default 0.3).
        pressure_adjustment: Step size for pressure factor adjustment (default 0.05).
        min_pressure_factor: Minimum value for pressure factor (default 0.5).
        max_pressure_factor: Maximum value for pressure factor (default 2.0).
        time_strategy_decay: Decay factor for time strategy threshold (default 0.4).
        risk_strategy_base: Base utility for risk strategy (default 0.7).
        risk_strategy_time_factor: Time factor for risk strategy (default 0.25).
        vote_threshold: Threshold for weighted vote acceptance (default 0.5).
        top_candidate_divisor: Divisor for selecting top candidates (default 3).
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
        initial_threshold: float = 0.95,
        concession_rate: float = 0.1,
        slow_concession_end: float = 0.8,
        medium_concession_end: float = 0.95,
        early_vote_end: float = 0.5,
        late_vote_start: float = 0.85,
        recency_weight_increment: float = 0.1,
        time_weight_multiplier: float = 2.0,
        slow_concession_multiplier: float = 0.3,
        medium_concession_multiplier: float = 0.4,
        fast_concession_base: float = 0.7,
        fast_concession_additional: float = 0.3,
        pressure_adjustment: float = 0.05,
        min_pressure_factor: float = 0.5,
        max_pressure_factor: float = 2.0,
        time_strategy_decay: float = 0.4,
        risk_strategy_base: float = 0.7,
        risk_strategy_time_factor: float = 0.25,
        vote_threshold: float = 0.5,
        top_candidate_divisor: int = 3,
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
        self._initial_threshold = initial_threshold
        self._concession_rate = concession_rate
        self._slow_concession_end = slow_concession_end
        self._medium_concession_end = medium_concession_end
        self._early_vote_end = early_vote_end
        self._late_vote_start = late_vote_start
        self._recency_weight_increment = recency_weight_increment
        self._time_weight_multiplier = time_weight_multiplier
        self._slow_concession_multiplier = slow_concession_multiplier
        self._medium_concession_multiplier = medium_concession_multiplier
        self._fast_concession_base = fast_concession_base
        self._fast_concession_additional = fast_concession_additional
        self._pressure_adjustment = pressure_adjustment
        self._min_pressure_factor = min_pressure_factor
        self._max_pressure_factor = max_pressure_factor
        self._time_strategy_decay = time_strategy_decay
        self._risk_strategy_base = risk_strategy_base
        self._risk_strategy_time_factor = risk_strategy_time_factor
        self._vote_threshold = vote_threshold
        self._top_candidate_divisor = top_candidate_divisor
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_value_weights: dict[int, dict] = {}
        self._total_opponent_weight: float = 0.0

        # Gang strategy state
        self._gang_votes: dict[str, float] = {}
        self._current_threshold: float = initial_threshold
        self._last_bid: Outcome | None = None

        # Time pressure tracking
        self._pressure_factor: float = 1.0
        self._offer_count: int = 0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_bids = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_value_weights = {}
        self._total_opponent_weight = 0.0
        self._gang_votes = {}
        self._current_threshold = self._initial_threshold
        self._last_bid = None
        self._pressure_factor = 1.0
        self._offer_count = 0

    def _update_opponent_model(self, bid: Outcome, time: float) -> None:
        """Update opponent model with weighted frequency analysis."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append(bid)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        # Weight increases with time (later bids reveal more preferences)
        # Also weight recent bids more heavily
        recency_weight = 1.0 + len(self._opponent_bids) * self._recency_weight_increment
        time_weight = 1.0 + time * self._time_weight_multiplier
        weight = recency_weight * time_weight

        self._total_opponent_weight += weight

        # Update value weights per issue
        for i, value in enumerate(bid):
            if i not in self._opponent_value_weights:
                self._opponent_value_weights[i] = {}
            self._opponent_value_weights[i][value] = (
                self._opponent_value_weights[i].get(value, 0.0) + weight
            )

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """
        Estimate opponent's utility for a bid using frequency analysis.

        Higher frequency values are assumed to be more preferred by opponent.
        """
        if not self._opponent_value_weights or self._total_opponent_weight == 0:
            return 0.5  # No info, assume neutral

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            if i in self._opponent_value_weights:
                issue_weights = self._opponent_value_weights[i]
                value_weight = issue_weights.get(value, 0.0)
                max_weight = max(issue_weights.values()) if issue_weights else 1.0
                # Normalize to [0, 1]
                if max_weight > 0:
                    total_score += value_weight / max_weight
                else:
                    total_score += 0.5
            else:
                total_score += 0.5

        return total_score / num_issues if num_issues > 0 else 0.5

    def _compute_threshold(self, time: float) -> float:
        """
        Compute acceptance threshold using time-pressure based adjustment.

        The threshold decreases over time with acceleration near deadline.
        """
        # Base concession curve (conservative early, faster late)
        if time < self._slow_concession_end:
            # Slow concession in first phase
            concession = (
                self._concession_rate
                * (time / self._slow_concession_end)
                * self._slow_concession_multiplier
            )
        elif time < self._medium_concession_end:
            # Medium concession
            base = self._concession_rate * self._slow_concession_multiplier
            additional = (
                self._concession_rate
                * self._medium_concession_multiplier
                * (
                    (time - self._slow_concession_end)
                    / (self._medium_concession_end - self._slow_concession_end)
                )
            )
            concession = base + additional
        else:
            # Rapid concession near deadline
            base = self._concession_rate * self._fast_concession_base
            additional = (
                self._concession_rate
                * self._fast_concession_additional
                * (
                    (time - self._medium_concession_end)
                    / (1.0 - self._medium_concession_end)
                )
            )
            concession = base + additional

        # Apply pressure factor (increases if opponent is tough)
        pressure_adjusted = concession * self._pressure_factor

        threshold = self._initial_threshold - pressure_adjusted

        # Ensure reasonable bounds
        return max(0.5, min(0.99, threshold))

    def _update_pressure_factor(self) -> None:
        """Update pressure factor based on opponent behavior."""
        if len(self._opponent_bids) < 5:
            return

        # Check if opponent is conceding
        recent_utilities = []
        for bid in self._opponent_bids[-10:]:
            if self.ufun:
                recent_utilities.append(float(self.ufun(bid)))

        if len(recent_utilities) < 2:
            return

        # Calculate trend
        first_half = sum(recent_utilities[: len(recent_utilities) // 2]) / (
            len(recent_utilities) // 2
        )
        second_half = sum(recent_utilities[len(recent_utilities) // 2 :]) / (
            len(recent_utilities) - len(recent_utilities) // 2
        )

        if second_half > first_half:
            # Opponent is conceding, we can be tougher
            self._pressure_factor = max(
                self._min_pressure_factor,
                self._pressure_factor - self._pressure_adjustment,
            )
        else:
            # Opponent is tough, increase pressure to concede
            self._pressure_factor = min(
                self._max_pressure_factor,
                self._pressure_factor + self._pressure_adjustment,
            )

    def _gang_vote(self, time: float, offer_utility: float) -> dict[str, float]:
        """
        Have the gang of strategies vote on the decision.

        Returns votes from different strategy perspectives.
        """
        votes = {}

        # Strategy 1: Time-based (conservative early, lenient late)
        time_threshold = self._initial_threshold * (
            1 - time * self._time_strategy_decay
        )
        votes["time_strategy"] = 1.0 if offer_utility >= time_threshold else 0.0

        # Strategy 2: Threshold-based (strict threshold)
        votes["threshold_strategy"] = (
            1.0 if offer_utility >= self._current_threshold else 0.0
        )

        # Strategy 3: Best-so-far (compare to best opponent offer)
        if self._best_opponent_utility > 0:
            relative = offer_utility / self._best_opponent_utility
            votes["relative_strategy"] = min(1.0, relative)
        else:
            votes["relative_strategy"] = 0.5

        # Strategy 4: Pressure-aware (adjust based on opponent behavior)
        pressure_threshold = self._current_threshold / self._pressure_factor
        votes["pressure_strategy"] = 1.0 if offer_utility >= pressure_threshold else 0.0

        # Strategy 5: Risk-averse (accept good offers to avoid deadline risk)
        risk_threshold = self._risk_strategy_base + self._risk_strategy_time_factor * (
            1 - time
        )
        votes["risk_strategy"] = 1.0 if offer_utility >= risk_threshold else 0.0

        return votes

    def _aggregate_votes(self, votes: dict[str, float], time: float) -> bool:
        """
        Aggregate gang votes to make final decision.

        Weight strategies differently based on negotiation phase.
        """
        # Weight strategies based on time
        if time < self._early_vote_end:
            # Early: favor threshold and conservative strategies
            weights = {
                "time_strategy": 0.15,
                "threshold_strategy": 0.35,
                "relative_strategy": 0.15,
                "pressure_strategy": 0.20,
                "risk_strategy": 0.15,
            }
        elif time < self._late_vote_start:
            # Middle: balanced approach
            weights = {
                "time_strategy": 0.20,
                "threshold_strategy": 0.25,
                "relative_strategy": 0.20,
                "pressure_strategy": 0.20,
                "risk_strategy": 0.15,
            }
        else:
            # Late: favor risk-aware and time strategies
            weights = {
                "time_strategy": 0.25,
                "threshold_strategy": 0.15,
                "relative_strategy": 0.15,
                "pressure_strategy": 0.15,
                "risk_strategy": 0.30,
            }

        weighted_sum = sum(votes[k] * weights[k] for k in votes)

        # Majority vote with weighted sum
        return weighted_sum >= self._vote_threshold

    def _select_bid(self, time: float) -> Outcome | None:
        """
        Select a bid that maximizes opponent utility while meeting own threshold.
        """
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        self._current_threshold = threshold

        # Get bids meeting our threshold
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            # Fallback: get best available
            if self._outcome_space.outcomes:
                return self._outcome_space.outcomes[0].bid
            return None

        # If we have opponent model, select bid maximizing estimated opponent utility
        if self._opponent_value_weights:
            best_bid = None
            best_opponent_util = -1.0

            for bd in candidates:
                opponent_util = self._estimate_opponent_utility(bd.bid)
                if opponent_util > best_opponent_util:
                    best_opponent_util = opponent_util
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        # No opponent model yet, return random candidate from top tier
        top_candidates = candidates[
            : max(1, len(candidates) // self._top_candidate_divisor)
        ]
        return random.choice(top_candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        self._offer_count += 1
        self._update_pressure_factor()

        bid = self._select_bid(time)
        self._last_bid = bid

        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond using gang voting mechanism."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        # Update opponent model
        self._update_opponent_model(offer, time)
        self._update_pressure_factor()

        # Update threshold
        self._current_threshold = self._compute_threshold(time)

        # Gang voting
        votes = self._gang_vote(time, offer_utility)
        self._gang_votes = votes

        if self._aggregate_votes(votes, time):
            return ResponseType.ACCEPT_OFFER

        # Also accept if offer >= our next bid utility
        next_bid = self._select_bid(time)
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid)) if self.ufun else 0.0
            if offer_utility >= next_utility:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
