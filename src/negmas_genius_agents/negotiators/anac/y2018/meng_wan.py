"""MengWan from ANAC 2018."""

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

__all__ = ["MengWan"]


class MengWan(SAONegotiator):
    """
    MengWan (Agent36) from ANAC 2018 - 2nd Place.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    MengWan achieved 2nd place in ANAC 2018 with a very conservative Boulware
    strategy combined with frequency-based opponent modeling. The agent uses
    extremely slow concession (e=5) and time-varying acceptance thresholds that
    become more lenient as the deadline approaches.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.meng_wan.Agent36

    **Offering Strategy:**
        Uses very conservative Boulware concession: target = 1 - t^5 which
        maintains high utility for most of the negotiation. Among candidates
        near the target utility (+/-0.05), selects the bid with highest
        estimated opponent utility to maximize agreement probability.

    **Acceptance Strategy:**
        Time-dependent change thresholds that gradually decrease:
        - t < 0.95: accepts above 0.80 threshold
        - 0.95 <= t < 0.98: accepts above 0.75 threshold
        - t >= 0.98: accepts above 0.70 threshold
        Always accepts offers meeting the current target utility.
        Near deadline (t >= 0.99), accepts above min_threshold.

    **Opponent Modeling:**
        Frequency-based model tracking value occurrences per issue. Estimates
        opponent utility as average normalized frequency. No issue weighting.
        Model is used after 5+ opponent offers to bias bid selection toward
        higher estimated opponent satisfaction.

    Args:
        e: Concession exponent (default 5.0 for very slow Boulware).
        min_threshold: Minimum utility threshold (default 0.7).
        early_threshold: Acceptance threshold before late phase (default 0.80).
        mid_threshold: Acceptance threshold in middle late phase (default 0.75).
        late_threshold: Acceptance threshold in very late phase (default 0.70).
        mid_time_threshold: Time threshold for mid acceptance (default 0.95).
        late_time_threshold: Time threshold for late acceptance (default 0.98).
        time_pressure_threshold: Time threshold for time pressure acceptance (default 0.9).
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
        e: float = 5.0,
        min_threshold: float = 0.7,
        early_threshold: float = 0.80,
        mid_threshold: float = 0.75,
        late_threshold: float = 0.70,
        mid_time_threshold: float = 0.95,
        late_time_threshold: float = 0.98,
        time_pressure_threshold: float = 0.9,
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
        self._min_threshold = min_threshold
        self._early_threshold = early_threshold
        self._mid_threshold = mid_threshold
        self._late_threshold = late_threshold
        self._mid_time_threshold = mid_time_threshold
        self._late_time_threshold = late_time_threshold
        self._time_pressure_threshold = time_pressure_threshold
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._value_frequencies: dict[int, dict[str, int]] = {}
        self._total_opponent_offers: int = 0

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._accepted_bids: list[Outcome] = []
        self._last_received_offer: Outcome | None = None

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
        self._accepted_bids = []
        self._last_received_offer = None

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
        # Boulware: utility = 1 - t^e
        target = 1.0 - math.pow(time, self._e)

        # Apply range scaling
        target = self._min_utility + (self._max_utility - self._min_utility) * target

        return max(target, self._min_threshold)

    def _get_change_threshold(self, time: float) -> float:
        """Get the acceptance change threshold based on time."""
        if time >= self._late_time_threshold:
            return self._late_threshold
        elif time >= self._mid_time_threshold:
            return self._mid_threshold
        else:
            return self._early_threshold

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid based on target utility and opponent model."""
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

        # Select bid with highest estimated opponent utility
        if self._total_opponent_offers > 5:
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

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        target = self._get_target_utility(time)
        change_threshold = self._get_change_threshold(time)

        # Accept if offer meets target utility
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if above change threshold and late in negotiation
        if time >= self._time_pressure_threshold and offer_utility >= change_threshold:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if above minimum
        if time >= self._deadline_threshold and offer_utility >= self._min_threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
