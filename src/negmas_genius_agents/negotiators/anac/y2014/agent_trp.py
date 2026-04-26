"""AgentTRP from ANAC 2014."""

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

__all__ = ["AgentTRP"]


class AgentTRP(SAONegotiator):
    """
    AgentTRP (Trade-off, Risk, and Pressure) from ANAC 2014.

    AgentTRP employs a multi-criteria decision framework that explicitly
    balances three key factors: finding mutually beneficial trade-offs,
    managing negotiation failure risk, and responding to time pressure.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.AgentTRP.AgentTRP

    **Offering Strategy:**
        Combines time pressure with trade-off analysis for bid selection:
        - Time pressure increases exponentially near deadline (pressure_phase1_end, pressure_phase2_end thresholds)
        - Base target computed from pressure-adjusted concession
        - Risk adjustment modifies target based on perceived opponent volatility
        - Final bids selected using trade-off score: own_weight * own_util +
          trade_off_weight * opponent_util, with risk bonus for well-understood
          opponent preferences.

    **Acceptance Strategy:**
        Risk-aware multi-condition acceptance:
        1. Accept if offer utility meets target threshold
        2. Accept below target if risk is high and time > risk_acceptance_time (risk threshold
           = target - perceived_risk * risk_aversion * risk_adjustment_factor)
        3. Accept if offer matches or exceeds next planned bid utility
        4. Emergency acceptance near deadline (t > deadline_acceptance_time) above minimum
        This approach prevents missed deals when facing unpredictable opponents.

    **Opponent Modeling:**
        Frequency-based preference learning combined with risk perception:
        - Tracks value frequencies per issue to estimate opponent utility
        - Computes perceived risk from variance in recent opponent offers
        - High variance indicates unpredictable opponent, increasing risk
        - Risk perception influences both acceptance flexibility and
          target utility adjustment (high risk -> more accommodating)

    Args:
        risk_aversion: Level of risk aversion [0-1] (default 0.5).
        trade_off_weight: Weight for opponent utility in bid selection (default 0.4).
        pressure_phase1_end: Time threshold ending first pressure phase (default 0.8).
        pressure_phase2_end: Time threshold ending second pressure phase (default 0.95).
        risk_acceptance_time: Time threshold for risk-aware acceptance (default 0.7).
        deadline_acceptance_time: Time threshold for near-deadline acceptance (default 0.98).
        min_utility_floor: Floor value for minimum utility (default 0.5).
        phase1_pressure_factor: Pressure factor during phase 1 (default 0.5).
        phase1_to_phase2_base: Base pressure at start of phase 2 (default 0.4).
        phase1_to_phase2_range: Pressure range during phase 2 (default 0.3).
        phase2_to_end_base: Base pressure at start of final phase (default 0.7).
        phase2_to_end_range: Pressure range during final phase (default 0.3).
        risk_base: Base risk perception value (default 0.3).
        variance_multiplier: Multiplier for variance in risk calculation (default 2.0).
        lowered_threshold_factor: Factor to lower threshold when no candidates (default 0.95).
        high_opponent_util_threshold: Threshold for high opponent utility (default 0.6).
        risk_bonus_factor: Bonus factor for well-understood preferences (default 0.1).
        risk_adjustment_factor: Factor for risk adjustment in acceptance (default 0.1).
        top_candidates_divisor: Divisor to select top candidates (default 3).
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
        risk_aversion: float = 0.5,
        trade_off_weight: float = 0.4,
        pressure_phase1_end: float = 0.8,
        pressure_phase2_end: float = 0.95,
        risk_acceptance_time: float = 0.7,
        deadline_acceptance_time: float = 0.98,
        min_utility_floor: float = 0.5,
        phase1_pressure_factor: float = 0.5,
        phase1_to_phase2_base: float = 0.4,
        phase1_to_phase2_range: float = 0.3,
        phase2_to_end_base: float = 0.7,
        phase2_to_end_range: float = 0.3,
        risk_base: float = 0.3,
        variance_multiplier: float = 2.0,
        lowered_threshold_factor: float = 0.95,
        high_opponent_util_threshold: float = 0.6,
        risk_bonus_factor: float = 0.1,
        risk_adjustment_factor: float = 0.1,
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
        self._risk_aversion = risk_aversion
        self._trade_off_weight = trade_off_weight
        self._pressure_phase1_end = pressure_phase1_end
        self._pressure_phase2_end = pressure_phase2_end
        self._risk_acceptance_time = risk_acceptance_time
        self._deadline_acceptance_time = deadline_acceptance_time
        self._min_utility_floor = min_utility_floor
        self._phase1_pressure_factor = phase1_pressure_factor
        self._phase1_to_phase2_base = phase1_to_phase2_base
        self._phase1_to_phase2_range = phase1_to_phase2_range
        self._phase2_to_end_base = phase2_to_end_base
        self._phase2_to_end_range = phase2_to_end_range
        self._risk_base = risk_base
        self._variance_multiplier = variance_multiplier
        self._lowered_threshold_factor = lowered_threshold_factor
        self._high_opponent_util_threshold = high_opponent_util_threshold
        self._risk_bonus_factor = risk_bonus_factor
        self._risk_adjustment_factor = risk_adjustment_factor
        self._top_candidates_divisor = top_candidates_divisor
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_value_freq: dict[int, dict] = {}

        # State
        self._min_utility: float = min_utility_floor
        self._max_utility: float = 1.0
        self._perceived_risk: float = 0.5

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = max(
                self._min_utility_floor, self._outcome_space.min_utility
            )
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_value_freq = {}
        self._perceived_risk = 0.5

    def _update_opponent_model(self, bid: Outcome, time: float) -> None:
        """Update opponent model and risk perception."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append(bid)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        # Update value frequencies
        for i, value in enumerate(bid):
            if i not in self._opponent_value_freq:
                self._opponent_value_freq[i] = {}
            self._opponent_value_freq[i][value] = (
                self._opponent_value_freq[i].get(value, 0) + 1
            )

        # Update risk perception based on opponent behavior
        if len(self._opponent_bids) > 3:
            recent_utils = [float(self.ufun(b)) for b in self._opponent_bids[-5:]]
            variance = sum(
                (u - sum(recent_utils) / len(recent_utils)) ** 2 for u in recent_utils
            ) / len(recent_utils)
            # High variance = unpredictable opponent = higher risk
            self._perceived_risk = min(
                1.0, self._risk_base + variance * self._variance_multiplier
            )

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

    def _compute_time_pressure(self, time: float) -> float:
        """Compute time pressure factor."""
        # Pressure increases exponentially near deadline
        if time < self._pressure_phase1_end:
            return time * self._phase1_pressure_factor
        elif time < self._pressure_phase2_end:
            return (
                self._phase1_to_phase2_base
                + (time - self._pressure_phase1_end)
                / (self._pressure_phase2_end - self._pressure_phase1_end)
                * self._phase1_to_phase2_range
            )
        else:
            return (
                self._phase2_to_end_base
                + (time - self._pressure_phase2_end)
                / (1.0 - self._pressure_phase2_end)
                * self._phase2_to_end_range
            )

    def _compute_target_utility(self, time: float) -> float:
        """Compute target utility balancing trade-off, risk, and pressure."""
        pressure = self._compute_time_pressure(time)

        # Base target from time pressure
        base_target = self._max_utility - pressure * (
            self._max_utility - self._min_utility
        )

        # Adjust for risk
        risk_adjustment = (
            self._risk_aversion * self._perceived_risk * self._risk_adjustment_factor
        )
        if self._perceived_risk > self._high_opponent_util_threshold:
            # High risk, be more accommodating
            base_target -= risk_adjustment
        else:
            # Low risk, can be tougher
            base_target += risk_adjustment * 0.5

        return max(self._min_utility, min(self._max_utility, base_target))

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid using trade-off analysis."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._compute_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            lowered = target * self._lowered_threshold_factor
            candidates = self._outcome_space.get_bids_above(lowered)
            if not candidates:
                return self._outcome_space.outcomes[0].bid

        # Trade-off: balance own utility with opponent utility
        if self._opponent_value_freq:
            best_bid = None
            best_score = -1.0

            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                # Trade-off score
                own_weight = 1.0 - self._trade_off_weight
                score = own_weight * bd.utility + self._trade_off_weight * opp_util

                # Risk bonus for well-understood opponent preferences
                if opp_util > self._high_opponent_util_threshold:
                    score += (1 - self._perceived_risk) * self._risk_bonus_factor

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
        """Respond using risk-aware acceptance."""
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

        # Risk-aware: accept slightly below target if risk is high
        risk_threshold = (
            target
            - self._perceived_risk * self._risk_aversion * self._risk_adjustment_factor
        )
        if offer_utility >= risk_threshold and time > self._risk_acceptance_time:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than our next offer
        next_bid = self._select_bid(time)
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid))
            if offer_utility >= next_utility:
                return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if reasonable
        if time > self._deadline_acceptance_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
