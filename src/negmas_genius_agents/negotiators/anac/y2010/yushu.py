"""Yushu from ANAC 2010."""

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

__all__ = ["Yushu"]


class Yushu(SAONegotiator):
    """
    Yushu from ANAC 2010 - 2nd place agent.

    This agent uses an adaptive time-dependent strategy with sophisticated round
    estimation and opponent tracking.

    .. warning::
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This implementation faithfully reproduces Yushu's core strategies:

    - Time-dependent concession with eagerness parameter
    - Best-10 opponent bid tracking for fallback decisions
    - Response time estimation for remaining rounds prediction
    - Dynamic minimum utility adjustment based on opponent behavior

    References:
        Original Genius class: ``agents.anac.y2010.Yushu.Yushu``

        ANAC 2010: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        - Time-dependent target: `target = max - (max-min) * time^eagerness`
        - Eagerness=1.2 (slightly conceder, faster than linear)
        - Bid selection from candidates in [0.96*target, 1.08*target] range
        - Avoids repeating recent bids (last 4)
        - Near deadline: offers best opponent bid if rounds_left < 1.6

    **Acceptance Strategy:**
        - Accept if offer >= target utility
        - Accept if offer >= 0.96*max (when rounds_left >= 15)
        - Accept if offer >= 0.92*max (when rounds_left < 15)
        - Also accepts if offer >= planned counter-offer utility when rounds_left < 8

    **Opponent Modeling:**
        - Tracks all opponent bids with utilities
        - Maintains sorted list of best 10 opponent bids
        - Calculates average opponent utility for min_utility adjustment
        - Estimates remaining rounds from response time patterns

    Minimum utility adjustments:
        - rounds_left > 6.7: min = 0.93 * max
        - rounds_left > 5.0: min = 0.90 * max
        - rounds_left > 3.0: min = 0.86 * max
        - rounds_left > 2.3: min = 0.80 * max
        - otherwise: min = 0.60 * max

    Args:
        eagerness: Concession rate exponent (default 1.2, slightly faster than linear)
    """

    def __init__(
        self,
        eagerness: float = 1.2,
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
        self._eagerness = eagerness
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent tracking
        self._opponent_history: list[Outcome] = []
        self._opponent_utilities: list[float] = []
        self._best_ten_indices: list[int] = []
        self._total_opponent_utility: float = 0.0

        # Own history
        self._my_history: list[Outcome] = []
        self._my_last_bid: Outcome | None = None
        self._suggest_bid: Outcome | None = None

        # Response time tracking
        self._response_times: list[float] = []
        self._last_time: float = 0.0

        # Utility bounds
        self._max_utility: float = 1.0
        self._min_utility: float = 0.5

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.outcomes[0].utility
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_history = []
        self._opponent_utilities = []
        self._best_ten_indices = []
        self._total_opponent_utility = 0.0
        self._my_history = []
        self._my_last_bid = None
        self._suggest_bid = None
        self._response_times = []
        self._last_time = 0.0
        self._min_utility = 0.5

    def _update_belief(self, bid: Outcome, utility: float) -> None:
        """Update opponent model with new bid."""
        idx = len(self._opponent_history)
        self._opponent_history.append(bid)
        self._opponent_utilities.append(utility)
        self._total_opponent_utility += utility

        if len(self._best_ten_indices) < 10:
            self._best_ten_indices.append(idx)
            self._best_ten_indices.sort(
                key=lambda i: self._opponent_utilities[i], reverse=True
            )
        elif utility > self._opponent_utilities[self._best_ten_indices[-1]]:
            self._best_ten_indices[-1] = idx
            self._best_ten_indices.sort(
                key=lambda i: self._opponent_utilities[i], reverse=True
            )

    def _average_response_time(self) -> float:
        """Calculate average response time."""
        if len(self._response_times) < 2:
            return 0.01
        diffs = [
            self._response_times[i] - self._response_times[i - 1]
            for i in range(1, len(self._response_times))
        ]
        return sum(diffs) / len(diffs) if diffs else 0.01

    def _estimate_rounds_left(self, time: float) -> float:
        """Estimate remaining rounds based on response time."""
        avg_time = self._average_response_time()
        if avg_time <= 0:
            return 100
        time_left = 1.0 - time
        return time_left / avg_time

    def _update_min_utility(self, time: float, rounds_left: float) -> None:
        """Update minimum utility based on time and estimated rounds."""
        avg_opp_util = (
            self._total_opponent_utility / len(self._opponent_history)
            if self._opponent_history
            else 0.5
        )

        if rounds_left > 6.7:
            self._min_utility = 0.93 * self._max_utility
        elif rounds_left > 5:
            self._min_utility = 0.90 * self._max_utility
        elif rounds_left > 3:
            self._min_utility = 0.86 * self._max_utility
        elif rounds_left > 2.3:
            self._min_utility = 0.80 * self._max_utility
        else:
            self._min_utility = 0.60 * self._max_utility

        if avg_opp_util < 0.75 and len(self._opponent_history) > 3:
            self._min_utility -= (0.75 - avg_opp_util) / 2.5

        self._min_utility = max(0.50, self._min_utility)
        scale_factor = min(0.75, avg_opp_util) / 3 + 0.75
        self._min_utility *= scale_factor
        self._min_utility = max(self._min_utility, avg_opp_util)

    def _get_target_utility(self, time: float) -> float:
        """Calculate target utility using time-dependent formula."""
        return self._max_utility - (self._max_utility - self._min_utility) * math.pow(
            time, self._eagerness
        )

    def _get_next_bid(self, target: float, time: float) -> Outcome | None:
        """Find a bid near the target utility."""
        if self._outcome_space is None:
            return None

        lower = 0.96 * target
        upper = 1.08 * target

        candidates: list[Outcome] = []
        for bd in self._outcome_space.outcomes:
            if lower <= bd.utility <= upper:
                candidates.append(bd.bid)
            elif bd.utility < lower:
                break

        if self._best_ten_indices and self._estimate_rounds_left(time) > 30:
            best_opp_util = self._opponent_utilities[self._best_ten_indices[0]]
            for bd in self._outcome_space.outcomes:
                if bd.utility >= best_opp_util and bd.bid not in candidates:
                    candidates.append(bd.bid)
                elif bd.utility < best_opp_util:
                    break

        if not candidates:
            if self._outcome_space.outcomes:
                return self._outcome_space.outcomes[0].bid
            return None

        recent = set(
            tuple(b) if b else () for b in self._my_history[-4:] if b is not None
        )
        filtered = [c for c in candidates if tuple(c) not in recent]

        if filtered and len(self._my_history) > 10:
            return random.choice(filtered)
        elif filtered:
            return min(
                filtered,
                key=lambda b: abs(float(self.ufun(b)) - target) if self.ufun else 0,
            )
        elif candidates:
            return candidates[0]

        return (
            self._outcome_space.outcomes[0].bid
            if self._outcome_space.outcomes
            else None
        )

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        self._response_times.append(time)
        rounds_left = self._estimate_rounds_left(time)
        self._update_min_utility(time, rounds_left)

        target = self._get_target_utility(time)

        if rounds_left < 1.6 and self._best_ten_indices:
            self._suggest_bid = self._opponent_history[self._best_ten_indices[0]]
        else:
            self._suggest_bid = self._get_next_bid(target, time)

        if self._suggest_bid is not None:
            self._my_history.append(self._suggest_bid)
            self._my_last_bid = self._suggest_bid

        return self._suggest_bid

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
        self._response_times.append(time)

        offer_utility = float(self.ufun(offer))
        self._update_belief(offer, offer_utility)

        rounds_left = self._estimate_rounds_left(time)
        self._update_min_utility(time, rounds_left)

        target = self._get_target_utility(time)

        acceptable_u = (
            0.96 * self._max_utility if rounds_left >= 15 else 0.92 * self._max_utility
        )

        if offer_utility >= target or offer_utility >= acceptable_u:
            if self._suggest_bid is None:
                return ResponseType.ACCEPT_OFFER
            suggest_util = float(self.ufun(self._suggest_bid)) if self.ufun else 0
            if rounds_left < 8 or suggest_util <= offer_utility:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
