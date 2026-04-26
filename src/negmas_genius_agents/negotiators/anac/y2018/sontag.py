"""Sontag from ANAC 2018."""

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

__all__ = ["Sontag"]


class Sontag(SAONegotiator):
    """
    Sontag from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Sontag implements a sophisticated strategy combining:

    1. Adaptive concession rate based on opponent behavior
    2. Tit-for-tat style response to opponent concessions
    3. Window-based analysis of recent opponent behavior
    4. Nash bargaining solution estimation for bid selection

    Key features:
    - Monitors opponent's concession behavior over sliding window
    - Adjusts own concession rate to match opponent cooperation
    - Uses estimated Nash product for bid selection
    - Conservative initial phase followed by adaptive concession

    Args:
        window_size: Number of offers to analyze (default 10)
        min_utility: Minimum utility threshold (default 0.6)
        initial_phase_end: Time to end initial phase (default 0.2).
        time_pressure_start: Time when time pressure begins (default 0.9).
        time_pressure_threshold: Time threshold for time pressure acceptance (default 0.95).
        deadline_threshold: Time threshold for deadline acceptance (default 0.99).
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
        window_size: int = 10,
        min_utility: float = 0.6,
        initial_phase_end: float = 0.2,
        time_pressure_start: float = 0.9,
        time_pressure_threshold: float = 0.95,
        deadline_threshold: float = 0.99,
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
        self._window_size = window_size
        self._min_utility_param = min_utility
        self._initial_phase_end = initial_phase_end
        self._time_pressure_start = time_pressure_start
        self._time_pressure_threshold = time_pressure_threshold
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._value_frequencies: dict[int, dict[str, int]] = {}
        self._total_opponent_offers: int = 0
        self._opponent_utilities: list[float] = []

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._last_received_offer: Outcome | None = None
        self._concession_rate: float = 0.1  # Initial concession rate

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = self._outcome_space.min_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._value_frequencies = {}
        self._total_opponent_offers = 0
        self._opponent_utilities = []
        self._last_received_offer = None
        self._concession_rate = 0.1

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update opponent model with frequency analysis."""
        if bid is None:
            return

        self._total_opponent_offers += 1
        self._opponent_utilities.append(utility)

        for i, value in enumerate(bid):
            if i not in self._value_frequencies:
                self._value_frequencies[i] = {}

            value_str = str(value)
            if value_str not in self._value_frequencies[i]:
                self._value_frequencies[i][value_str] = 0
            self._value_frequencies[i][value_str] += 1

        # Update concession rate based on opponent behavior
        self._update_concession_rate()

    def _update_concession_rate(self) -> None:
        """Adjust concession rate based on opponent's behavior."""
        if len(self._opponent_utilities) < 3:
            return

        # Analyze recent window
        window = self._opponent_utilities[-self._window_size :]
        if len(window) < 2:
            return

        # Calculate opponent's average concession (our utility change)
        avg_change = (window[-1] - window[0]) / len(window)

        # Tit-for-tat: if opponent concedes, we concede less; if hardball, concede more
        if avg_change > 0.02:
            # Opponent conceding - slow down our concession
            self._concession_rate = max(0.05, self._concession_rate - 0.02)
        elif avg_change < -0.01:
            # Opponent hardening - speed up concession to find agreement
            self._concession_rate = min(0.3, self._concession_rate + 0.02)

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility based on frequency model."""
        if self._total_opponent_offers == 0 or bid is None:
            return 0.0

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            value_str = str(value)
            if i in self._value_frequencies and value_str in self._value_frequencies[i]:
                freq = self._value_frequencies[i][value_str]
                total_score += freq / self._total_opponent_offers

        return total_score / num_issues if num_issues > 0 else 0.0

    def _get_target_utility(self, time: float) -> float:
        """Get target utility with adaptive concession."""
        # Base target with adaptive rate
        if time < self._initial_phase_end:
            target = 1.0
        else:
            normalized_time = (time - self._initial_phase_end) / (
                1.0 - self._initial_phase_end
            )
            target = 1.0 - self._concession_rate * normalized_time

        # Time pressure near deadline
        if time > self._time_pressure_start:
            pressure = (time - self._time_pressure_start) / (
                1.0 - self._time_pressure_start
            )
            target = target - 0.2 * pressure

        # Scale to utility range
        scaled = self._min_utility + (self._max_utility - self._min_utility) * target

        return max(scaled, self._min_utility_param)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid using Nash product estimation."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)

        # Get candidates near target
        candidates = self._outcome_space.get_bids_in_range(target - 0.05, target + 0.05)

        if not candidates:
            bid_details = self._outcome_space.get_bid_near_utility(target)
            return bid_details.bid if bid_details else self._best_bid

        if len(candidates) == 1:
            return candidates[0].bid

        # Use Nash product for selection
        if self._total_opponent_offers > 5:
            best_score = -1.0
            best_bid = candidates[0].bid
            for bd in candidates:
                own_util = bd.utility
                opp_util = self._estimate_opponent_utility(bd.bid)
                nash_score = own_util * opp_util
                if nash_score > best_score:
                    best_score = nash_score
                    best_bid = bd.bid
            return best_bid
        else:
            return random.choice(candidates).bid

    def _is_acceptable(self, offer_utility: float, time: float) -> bool:
        """Check if an offer is acceptable."""
        target = self._get_target_utility(time)

        if offer_utility >= target:
            return True

        if (
            time >= self._time_pressure_threshold
            and offer_utility >= self._min_utility_param
        ):
            return True

        if time >= self._deadline_threshold and offer_utility >= self._min_utility:
            return True

        return False

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        if self._last_received_offer is None:
            return self._best_bid

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

        self._last_received_offer = offer
        offer_utility = float(self.ufun(offer))
        self._update_opponent_model(offer, offer_utility)

        time = state.relative_time

        if self._is_acceptable(offer_utility, time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
