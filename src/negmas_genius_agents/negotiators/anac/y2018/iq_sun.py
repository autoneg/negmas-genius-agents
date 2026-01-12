"""IQSun2018 from ANAC 2018."""

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

__all__ = ["IQSun2018"]


class IQSun2018(SAONegotiator):
    """
    IQSun2018 from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    IQSun2018 uses a hybrid strategy combining Boulware-style time-dependent
    concession with frequency-based opponent modeling. The agent maintains
    conservative early behavior and uses opponent preference estimates to
    select bids that maximize likelihood of acceptance.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.iqson.IQSun2018

    **Offering Strategy:**
        Uses Boulware concession: target = 1 - t^(1/e) where e=0.2 creates
        very slow early concession that accelerates near deadline. Bid selection
        narrows margin over time: 0.05 * (1 - t), meaning more precise targeting
        as negotiation progresses. Among candidates, selects the bid with highest
        estimated opponent utility to increase acceptance probability.

    **Acceptance Strategy:**
        Multi-phase acceptance based on negotiation stage:
        - Normal: accepts offers meeting current target utility
        - Late (t >= 0.95): accepts offers above reservation threshold
        - Deadline (t >= 0.99): accepts anything above minimum utility floor
        Also tracks best received utility for reference.

    **Opponent Modeling:**
        Frequency-based model tracking value occurrences per issue. Estimates
        opponent utility as average normalized frequency across issues. No issue
        weighting - assumes equal importance. Model is used for bid selection
        once > 3 opponent offers are received.

    Args:
        e: Concession exponent (default 0.2 for Boulware-like behavior).
        reservation: Reservation utility threshold (default 0.65).
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
        e: float = 0.2,
        reservation: float = 0.65,
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
        self._e = e
        self._reservation = reservation
        self._time_pressure_threshold = time_pressure_threshold
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._value_frequencies: dict[int, dict[str, int]] = {}
        self._total_opponent_offers: int = 0
        self._opponent_best_utility: float = 0.0

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._last_received_offer: Outcome | None = None
        self._best_received_utility: float = 0.0

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
        self._opponent_best_utility = 0.0
        self._last_received_offer = None
        self._best_received_utility = 0.0

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update frequency-based opponent model."""
        if bid is None:
            return

        self._total_opponent_offers += 1

        for i, value in enumerate(bid):
            if i not in self._value_frequencies:
                self._value_frequencies[i] = {}

            value_str = str(value)
            if value_str not in self._value_frequencies[i]:
                self._value_frequencies[i][value_str] = 0
            self._value_frequencies[i][value_str] += 1

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
        """Get target utility based on Boulware concession."""
        # Boulware-like: slow concession early, faster later
        target = 1.0 - math.pow(time, 1.0 / self._e)

        # Scale to actual utility range
        scaled_target = (
            self._min_utility + (self._max_utility - self._min_utility) * target
        )

        return max(scaled_target, self._reservation)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid based on target utility and opponent model."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)

        # Get candidates near target
        margin = 0.05 * (1 - time)  # Narrower margin as time progresses
        candidates = self._outcome_space.get_bids_in_range(
            target - margin, target + margin
        )

        if not candidates:
            bid_details = self._outcome_space.get_bid_near_utility(target)
            return bid_details.bid if bid_details else self._best_bid

        if len(candidates) == 1:
            return candidates[0].bid

        # Use opponent model if enough data
        if self._total_opponent_offers > 3:
            best_opp_util = -1.0
            best_bid = candidates[0].bid
            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                if opp_util > best_opp_util:
                    best_opp_util = opp_util
                    best_bid = bd.bid
            return best_bid
        else:
            return random.choice(candidates).bid

    def _is_acceptable(self, offer_utility: float, time: float) -> bool:
        """Check if an offer is acceptable."""
        target = self._get_target_utility(time)

        # Accept if offer meets target
        if offer_utility >= target:
            return True

        # Near deadline acceptance
        if time >= self._time_pressure_threshold and offer_utility >= self._reservation:
            return True

        # Very near deadline - accept anything reasonable
        if time >= self._deadline_threshold and offer_utility >= self._min_utility:
            return True

        return False

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        # First move: offer best bid
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
        self._update_opponent_model(offer)

        offer_utility = float(self.ufun(offer))

        # Track best received
        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility

        time = state.relative_time

        if self._is_acceptable(offer_utility, time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
