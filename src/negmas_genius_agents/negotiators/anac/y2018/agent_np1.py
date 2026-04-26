"""AgentNP1 from ANAC 2018."""

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

__all__ = ["AgentNP1"]


class AgentNP1(SAONegotiator):
    """
    AgentNP1 (Nash Product 1) from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    AgentNP1 focuses on achieving fair outcomes by maximizing the Nash product
    during bid selection. The agent uses frequency-based opponent modeling to
    estimate opponent preferences and selects bids that balance own utility
    with estimated opponent satisfaction.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.agentnp1.AgentNP1

    **Offering Strategy:**
        Uses a polynomial concession curve: target = 1 - t^e. Candidates are
        collected from bids above the target threshold, then the bid maximizing
        Nash product (own_utility + 0.01) * (opp_utility + 0.01) is selected.
        In the early phase (< 3 opponent offers), randomly selects from top 5
        candidates to explore the outcome space.

    **Acceptance Strategy:**
        Accepts offers meeting the current target utility. Under time pressure
        (t >= 0.9), accepts offers above the minimum utility parameter.
        Near deadline (t >= 0.98), accepts anything above the minimum utility
        in the outcome space.

    **Opponent Modeling:**
        Builds a frequency model tracking how often each value appears for each
        issue in opponent offers. Opponent utility is estimated as the average
        normalized frequency across all issues, with 0.5 as the neutral estimate
        when no data is available.

    Args:
        e: Concession exponent (default 2.0, moderate concession speed).
        min_utility: Minimum utility threshold (default 0.55).
        time_pressure_threshold: Time threshold for time pressure acceptance (default 0.9).
        deadline_threshold: Time threshold for deadline acceptance (default 0.98).
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
        e: float = 2.0,
        min_utility: float = 0.55,
        time_pressure_threshold: float = 0.9,
        deadline_threshold: float = 0.98,
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
        self._min_utility_param = min_utility
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
            return 0.5  # Neutral estimate

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            value_str = str(value)
            if i in self._value_frequencies and value_str in self._value_frequencies[i]:
                freq = self._value_frequencies[i][value_str]
                total_score += freq / self._total_opponent_offers

        return total_score / num_issues if num_issues > 0 else 0.5

    def _get_target_utility(self, time: float) -> float:
        """Get target utility based on concession curve."""
        # Polynomial concession
        target = 1.0 - math.pow(time, self._e)

        # Scale to utility range
        scaled = self._min_utility + (self._max_utility - self._min_utility) * target

        return max(scaled, self._min_utility_param)

    def _compute_nash_product(self, own_util: float, opp_util: float) -> float:
        """Compute Nash product with safety margins."""
        # Use small offset to handle zero utilities
        return (own_util + 0.01) * (opp_util + 0.01)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid maximizing Nash product above threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)

        # Get candidates above target
        candidates = self._outcome_space.get_bids_in_range(target - 0.05, 1.0)

        if not candidates:
            bid_details = self._outcome_space.get_bid_near_utility(target)
            return bid_details.bid if bid_details else self._best_bid

        if len(candidates) == 1:
            return candidates[0].bid

        # Select bid maximizing Nash product
        if self._total_opponent_offers > 3:
            best_nash = -1.0
            best_bid = candidates[0].bid
            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                nash = self._compute_nash_product(bd.utility, opp_util)
                if nash > best_nash:
                    best_nash = nash
                    best_bid = bd.bid
            return best_bid
        else:
            # Early phase - random from top candidates
            top = candidates[: min(5, len(candidates))]
            return random.choice(top).bid

    def _is_acceptable(self, offer_utility: float, time: float) -> bool:
        """Check if an offer is acceptable."""
        target = self._get_target_utility(time)

        if offer_utility >= target:
            return True

        # Time pressure
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
        self._update_opponent_model(offer)

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        if self._is_acceptable(offer_utility, time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
