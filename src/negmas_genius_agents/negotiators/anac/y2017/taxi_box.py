"""TaxiBox from ANAC 2017."""

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

__all__ = ["TaxiBox"]


class TaxiBox(SAONegotiator):
    """
    TaxiBox from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This is a reimplementation of TaxiBox from ANAC 2017.
    Original: agents.anac.y2017.tangxun.taxibox.TaxiBox

    TaxiBox uses a "fare meter" inspired strategy where concession
    accumulates over time like a taxi fare increases during a ride.

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/nego/node/7

    **Offering Strategy:**
        Accumulates concession continuously: accumulated += time_delta * rate.
        Rate adapts to opponent: cooperative (concession >0.1) slows rate
        to 70%, hardening (<-0.05) increases to 120%. Near deadline (>80%)
        accelerates further. Threshold = max - accumulated * range. Bids
        selected from a shrinking window that narrows as threshold drops.

    **Acceptance Strategy:**
        Uses "accept-next" criterion: accepts if offer utility >= what we
        would propose next. Also accepts above the fare-based threshold
        and considers opponent's best offer as a floor. Near deadline
        (>98%) accepts above minimum utility.

    **Opponent Modeling:**
        Tracks opponent utility history with timestamps. Calculates
        concession rate from recent offers (last 5) to adjust our
        fare accumulation rate. Also maintains opponent's best offer
        as a floor for acceptance threshold.

    Args:
        min_utility: Minimum acceptable utility (default 0.55).
        base_rate: Base concession rate per time unit (default 0.3).
        deadline_threshold: Time threshold for accelerated concession (default 0.8).
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
        min_utility: float = 0.55,
        base_rate: float = 0.3,
        deadline_threshold: float = 0.8,
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
        self._base_rate = base_rate
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

        # Fare/concession tracking
        self._accumulated_concession: float = 0.0
        self._last_time: float = 0.0

        # Opponent tracking
        self._opponent_utilities: list[tuple[float, float]] = []  # (time, utility)
        self._best_opponent_utility: float = 0.0
        self._opponent_concession_rate: float = 0.0

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

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._accumulated_concession = 0.0
        self._last_time = 0.0
        self._opponent_utilities = []
        self._best_opponent_utility = 0.0
        self._opponent_concession_rate = 0.0

    def _update_opponent_model(self, offer: Outcome, time: float) -> None:
        """Track opponent behavior."""
        if self.ufun is None:
            return

        offer_utility = float(self.ufun(offer))
        self._opponent_utilities.append((time, offer_utility))
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)

        # Estimate opponent's concession rate
        if len(self._opponent_utilities) >= 3:
            recent = self._opponent_utilities[-5:]
            utility_change = recent[-1][1] - recent[0][1]
            time_change = recent[-1][0] - recent[0][0]
            if time_change > 0:
                self._opponent_concession_rate = utility_change / time_change

    def _update_fare(self, time: float) -> None:
        """Update accumulated concession (fare meter)."""
        time_delta = time - self._last_time
        if time_delta <= 0:
            return

        # Adjust rate based on opponent behavior
        rate = self._base_rate
        if self._opponent_concession_rate > 0.1:
            # Opponent is conceding, slow down our concession
            rate *= 0.7
        elif self._opponent_concession_rate < -0.05:
            # Opponent is hardening, speed up slightly
            rate *= 1.2

        # Accelerate near deadline
        if time > self._deadline_threshold:
            late_factor = 1 + (time - self._deadline_threshold) / (
                1.0 - self._deadline_threshold
            )
            rate *= late_factor

        # Accumulate concession
        self._accumulated_concession += time_delta * rate
        self._last_time = time

    def _calculate_threshold(self, time: float) -> float:
        """Calculate threshold based on accumulated fare."""
        self._update_fare(time)

        utility_range = self._max_utility - self._min_utility
        threshold = self._max_utility - self._accumulated_concession * utility_range

        # Don't go below what opponent has offered
        if self._best_opponent_utility > self._min_utility:
            threshold = max(threshold, self._best_opponent_utility - 0.05)

        return max(threshold, self._min_utility)

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Select a bid in the current utility window."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        # Window shrinks as threshold decreases
        window_width = max(0.05, (threshold - self._min_utility) * 0.3)

        candidates = self._outcome_space.get_bids_in_range(
            threshold, threshold + window_width
        )

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold)

        if candidates:
            return random.choice(candidates).bid

        return self._best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        threshold = self._calculate_threshold(time)

        return self._select_bid(threshold)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer using accept-next criterion."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        self._update_opponent_model(offer, time)

        threshold = self._calculate_threshold(time)
        offer_utility = float(self.ufun(offer))

        # Accept if above threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Accept-next: accept if offer is better than what we'd propose
        next_bid = self._select_bid(threshold)
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid))
            if offer_utility >= next_utility:
                return ResponseType.ACCEPT_OFFER

        # Near deadline
        if time > 0.98 and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
