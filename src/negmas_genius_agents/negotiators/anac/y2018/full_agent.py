"""FullAgent from ANAC 2018."""

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

__all__ = ["FullAgent"]


class FullAgent(SAONegotiator):
    """
    FullAgent from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    FullAgent uses a comprehensive strategy combining domain-adaptive concession,
    opponent modeling with issue weight estimation, and Nash welfare maximization.
    The agent adjusts its concession rate based on domain size and uses frequency
    analysis to estimate opponent preferences for mutually beneficial bid selection.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.fullagent.FullAgent

    **Offering Strategy:**
        Uses domain-adaptive exponential concession: target = 1 - t^(1/effective_beta)
        where effective_beta = beta * domain_size_factor. Smaller domains (< 500
        outcomes) get factor 1.2 (slower), medium (500-2000) get 1.0, larger
        domains get 0.8 (faster). Bids are selected to maximize Nash welfare
        (own_utility * opp_utility) among candidates near the target.

    **Acceptance Strategy:**
        Accepts offers meeting the current target utility. Uses graduated
        acceptance near deadline: t >= 0.9 accepts above min_threshold,
        t >= 0.98 accepts above minimum utility floor.

    **Opponent Modeling:**
        Weighted frequency model using standard deviation for issue importance.
        Tracks value frequencies per issue, computes standard deviation of
        frequencies, and uses higher std_dev as weight (more varied = more
        important issue to opponent). Opponent utility combines weighted
        frequencies to estimate preference satisfaction.

    Args:
        beta: Concession curve parameter (default 0.05, conservative).
        min_threshold: Minimum acceptance threshold (default 0.65).
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
        beta: float = 0.05,
        min_threshold: float = 0.65,
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
        self._beta = beta
        self._min_threshold = min_threshold
        self._time_pressure_threshold = time_pressure_threshold
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._value_frequencies: dict[int, dict[str, int]] = {}
        self._issue_weights: dict[int, float] = {}
        self._total_opponent_offers: int = 0

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._last_received_offer: Outcome | None = None
        self._domain_size_factor: float = 1.0

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

            # Adjust for domain size
            domain_size = len(self._outcome_space.outcomes)
            if domain_size < 500:
                self._domain_size_factor = 1.2
            elif domain_size < 2000:
                self._domain_size_factor = 1.0
            else:
                self._domain_size_factor = 0.8

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._value_frequencies = {}
        self._issue_weights = {}
        self._total_opponent_offers = 0
        self._last_received_offer = None

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent model with frequency analysis."""
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

        # Update issue weights using standard deviation
        self._update_issue_weights()

    def _update_issue_weights(self) -> None:
        """Compute issue weights from frequency standard deviations."""
        for issue_idx, freq_map in self._value_frequencies.items():
            if not freq_map:
                self._issue_weights[issue_idx] = 0.0
                continue

            values = list(freq_map.values())
            n = len(values)
            if n == 0:
                self._issue_weights[issue_idx] = 0.0
                continue

            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std_dev = math.sqrt(variance)

            self._issue_weights[issue_idx] = std_dev

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility using weighted frequency model."""
        if self._total_opponent_offers == 0 or bid is None:
            return 0.0

        total_weight = sum(self._issue_weights.values())
        if total_weight == 0:
            total_weight = len(self._value_frequencies)
            if total_weight == 0:
                return 0.0
            weights = {i: 1.0 for i in range(len(bid))}
        else:
            weights = self._issue_weights

        total_score = 0.0
        for i, value in enumerate(bid):
            value_str = str(value)
            if i in self._value_frequencies and value_str in self._value_frequencies[i]:
                freq = self._value_frequencies[i][value_str]
                weight = weights.get(i, 0.0) / total_weight if total_weight > 0 else 0.0
                total_score += (freq / self._total_opponent_offers) * weight

        return total_score

    def _get_target_utility(self, time: float) -> float:
        """Get target utility based on exponential concession."""
        # Exponential concession adjusted by domain size
        effective_beta = self._beta * self._domain_size_factor
        target = 1.0 - math.pow(time, 1.0 / effective_beta)

        # Scale to utility range
        scaled = self._min_utility + (self._max_utility - self._min_utility) * target

        return max(scaled, self._min_threshold)

    def _calculate_nash_welfare(self, own_utility: float, opp_utility: float) -> float:
        """Calculate Nash welfare (product of utilities)."""
        if own_utility <= 0 or opp_utility <= 0:
            return 0.0
        return own_utility * opp_utility

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid maximizing Nash welfare."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)

        # Get candidates near target
        margin = 0.06
        candidates = self._outcome_space.get_bids_in_range(
            target - margin, target + margin
        )

        if not candidates:
            bid_details = self._outcome_space.get_bid_near_utility(target)
            return bid_details.bid if bid_details else self._best_bid

        if len(candidates) == 1:
            return candidates[0].bid

        # Use Nash welfare for selection if we have opponent data
        if self._total_opponent_offers > 5:
            best_welfare = -1.0
            best_bid = candidates[0].bid
            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                welfare = self._calculate_nash_welfare(bd.utility, opp_util)
                if welfare > best_welfare:
                    best_welfare = welfare
                    best_bid = bd.bid
            return best_bid
        else:
            return random.choice(candidates).bid

    def _is_acceptable(self, offer_utility: float, time: float) -> bool:
        """Check if an offer is acceptable."""
        target = self._get_target_utility(time)

        if offer_utility >= target:
            return True

        # Graduated acceptance near deadline
        if (
            time >= self._time_pressure_threshold
            and offer_utility >= self._min_threshold
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

        offer_utility = float(self.ufun(offer))
        time = state.relative_time

        if self._is_acceptable(offer_utility, time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
