"""AgreeableAgent2018 from ANAC 2018."""

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

__all__ = ["AgreeableAgent2018"]


class AgreeableAgent2018(SAONegotiator):
    """
    AgreeableAgent2018 from ANAC 2018 - Competition Winner (1st Place).

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    AgreeableAgent2018 won ANAC 2018 by combining sophisticated opponent modeling
    with strategic bid selection. The agent uses frequency analysis to estimate
    opponent preferences, computes issue weights from frequency variance, and
    employs roulette wheel selection to find mutually beneficial outcomes.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.agreeableagent2018.AgreeableAgent2018

    **Offering Strategy:**
        Uses a Boulware-style concession curve: target = 1 - f(t) where
        f(t) = k + (1-k) * t^(1/concession_factor). Stays at maximum utility
        until time_to_concede, then gradually concedes. Bids are selected from
        a neighborhood around the target utility. After the model becomes usable,
        roulette wheel selection biases toward bids with higher estimated
        opponent utility, increasing agreement probability.

    **Acceptance Strategy:**
        Accepts offers that meet or exceed the utility of the next planned bid.
        Near deadline (t >= 0.99), accepts any offer above the reservation value.
        This "accept if better than what I'd offer" approach ensures the agent
        never rejects an offer it would have accepted if offered as a counter.

    **Opponent Modeling:**
        Tracks frequency of each value for each issue. Computes issue weights
        from the standard deviation of frequencies - higher variance indicates
        more important issues to the opponent. The model is only used after
        a time threshold (0.2-0.4 depending on domain size) and minimum 5 offers
        to ensure reliability. Estimated opponent utility combines weighted
        frequencies across issues.

    Args:
        time_to_concede: Time before starting to concede (default 0.2).
        concession_factor: Concession curve parameter (default 0.1, Boulware-like).
        minimum_utility: Floor for target utility (default 0.8).
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
        time_to_concede: float = 0.2,
        concession_factor: float = 0.1,
        minimum_utility: float = 0.8,
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
        self._time_to_concede = time_to_concede
        self._concession_factor = concession_factor
        self._minimum_utility = minimum_utility
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._value_frequencies: dict[int, dict[str, int]] = {}
        self._issue_weights: dict[int, float] = {}
        self._total_opponent_offers: int = 0
        self._time_for_using_model: float = 0.2

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._best_bid: Outcome | None = None
        self._last_received_offer: Outcome | None = None
        self._reservation_value: float = 0.0
        self._k: float = 0.0  # Initial offset

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

        # Adjust model timing based on domain size
        domain_size = len(self._outcome_space.outcomes) if self._outcome_space else 0
        if domain_size < 1000:
            self._time_for_using_model = 0.2
        elif domain_size < 10000:
            self._time_for_using_model = 0.3
        else:
            self._time_for_using_model = 0.4

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
        """Update frequency-based opponent model."""
        if bid is None:
            return

        self._total_opponent_offers += 1

        # Count frequencies for each issue value
        for i, value in enumerate(bid):
            if i not in self._value_frequencies:
                self._value_frequencies[i] = {}

            value_str = str(value)
            if value_str not in self._value_frequencies[i]:
                self._value_frequencies[i][value_str] = 0
            self._value_frequencies[i][value_str] += 1

        # Update issue weights based on frequency standard deviation
        self._update_issue_weights()

    def _update_issue_weights(self) -> None:
        """Compute issue weights from frequency standard deviations."""
        for issue_idx, freq_map in self._value_frequencies.items():
            if not freq_map:
                self._issue_weights[issue_idx] = 0.0
                continue

            # Compute standard deviation of frequencies
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
        """Estimate opponent utility based on frequency model."""
        if self._total_opponent_offers == 0:
            return 0.0

        total_weight = sum(self._issue_weights.values())
        if total_weight == 0:
            # Equal weights
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

    def _f(self, t: float) -> float:
        """Concession function."""
        if self._concession_factor == 0:
            return self._k
        return self._k + (1 - self._k) * math.pow(t, 1.0 / self._concession_factor)

    def _get_utility_by_time(self, time: float) -> float:
        """Compute target utility based on time."""
        if time < self._time_to_concede:
            return 1.0

        # Normalize time after concession starts
        normalized_time = (time - self._time_to_concede) / (1 - self._time_to_concede)
        target = self._min_utility + (self._max_utility - self._min_utility) * (
            1 - self._f(normalized_time)
        )

        return max(target, self._minimum_utility)

    def _is_model_usable(self, time: float) -> bool:
        """Check if opponent model has enough data."""
        return time >= self._time_for_using_model and self._total_opponent_offers > 5

    def _get_explorable_neighborhood(self, time: float) -> float:
        """Get the utility range for neighborhood exploration."""
        if time < self._time_to_concede:
            return 0.0

        # Factor of 0.05 for neighborhood exploration
        return 0.05 * (
            1
            - (
                self._min_utility
                + (self._max_utility - self._min_utility) * (1 - self._f(time))
            )
        )

    def _select_bid_by_opponent_model(
        self, target_utility: float, time: float
    ) -> Outcome | None:
        """Select bid using opponent model with roulette wheel selection."""
        if self._outcome_space is None:
            return None

        # Get neighborhood around target
        neighborhood = self._get_explorable_neighborhood(time)
        range_min = target_utility - neighborhood
        range_max = target_utility + neighborhood

        candidates = self._outcome_space.get_bids_in_range(range_min, range_max)

        if not candidates:
            bid_details = self._outcome_space.get_bid_near_utility(target_utility)
            return bid_details.bid if bid_details else None

        if len(candidates) == 1:
            return candidates[0].bid

        # Roulette wheel selection based on opponent utility
        opponent_utilities = []
        for bd in candidates:
            opp_util = self._estimate_opponent_utility(bd.bid)
            opponent_utilities.append(opp_util)

        total = sum(opponent_utilities)
        if total == 0:
            return random.choice(candidates).bid

        # Normalize and select
        r = random.random()
        cumulative = 0.0
        for i, bd in enumerate(candidates):
            cumulative += opponent_utilities[i] / total
            if r <= cumulative:
                return bd.bid

        return candidates[-1].bid

    def _get_next_bid(self, time: float) -> Outcome | None:
        """Get the next bid to offer."""
        if self._outcome_space is None:
            return None

        target_utility = self._get_utility_by_time(time)

        if self._is_model_usable(time):
            return self._select_bid_by_opponent_model(target_utility, time)
        else:
            bid_details = self._outcome_space.get_bid_near_utility(target_utility)
            return bid_details.bid if bid_details else None

    def _is_acceptable(
        self, offer_utility: float, my_bid_utility: float, time: float
    ) -> bool:
        """Check if an offer is acceptable."""
        if offer_utility >= my_bid_utility:
            return True

        # Near deadline, accept if above reservation
        if (
            time >= self._deadline_threshold
            and offer_utility >= self._reservation_value
        ):
            return True

        return False

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        if self._last_received_offer is None:
            # First move: offer best bid
            return self._best_bid

        time = state.relative_time
        return self._get_next_bid(time)

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

        my_bid = self._get_next_bid(time)
        my_bid_utility = float(self.ufun(my_bid)) if my_bid else self._max_utility

        if self._is_acceptable(offer_utility, my_bid_utility, time):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER
