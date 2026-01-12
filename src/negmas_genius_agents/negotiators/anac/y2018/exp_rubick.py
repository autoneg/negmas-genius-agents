"""ExpRubick from ANAC 2018."""

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

__all__ = ["ExpRubick"]


class ExpRubick(SAONegotiator):
    """
    ExpRubick from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    ExpRubick is an experimental evolution of Rubick (ANAC 2017) with enhanced
    opponent modeling and adaptive concession. The agent learns issue weights
    from frequency variance, adapts its concession rate based on opponent
    behavior, and uses Nash product optimization for bid selection.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.exp_rubick.Exp_Rubick
        - Based on Rubick from ANAC 2017

    **Offering Strategy:**
        Uses adaptive exponential concession: target = 1 - t^(1/factor) where
        the concession_factor (initially 0.2) is adjusted based on opponent
        behavior. When opponent concedes (utility change > 0.05), factor decreases
        (slower concession). When opponent hardens (change < -0.02), factor
        increases (faster concession). Bids are selected to maximize Nash product
        with weighted opponent utility estimation.

    **Acceptance Strategy:**
        Accepts offers meeting the current target utility. Under time pressure
        (t >= 0.9), accepts offers above minimum utility parameter.
        Near deadline (t >= 0.98), accepts anything above minimum utility floor.

    **Opponent Modeling:**
        Sophisticated three-component model:
        1. Frequency tracking: counts value occurrences per issue
        2. Issue weight estimation: weights = 1 + sqrt(variance) where higher
           variance indicates more important issues to opponent
        3. Concession tracking: monitors last 5 utilities to adapt own strategy
        Opponent utility combines weighted frequencies with learned importance.

    Args:
        min_utility: Minimum utility threshold (default 0.55).
        learning_rate: Rate of strategy adaptation (default 0.1).
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
        min_utility: float = 0.55,
        learning_rate: float = 0.1,
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
        self._min_utility_param = min_utility
        self._learning_rate = learning_rate
        self._time_pressure_threshold = time_pressure_threshold
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._value_frequencies: dict[int, dict[str, int]] = {}
        self._issue_weights: dict[int, float] = {}
        self._total_opponent_offers: int = 0
        self._opponent_utilities: list[float] = []

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._last_received_offer: Outcome | None = None
        self._concession_factor: float = 0.2  # Adaptive

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
        self._issue_weights = {}
        self._total_opponent_offers = 0
        self._opponent_utilities = []
        self._last_received_offer = None
        self._concession_factor = 0.2

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update opponent model with weight estimation."""
        if bid is None:
            return

        self._total_opponent_offers += 1
        self._opponent_utilities.append(utility)

        # Update frequencies
        for i, value in enumerate(bid):
            if i not in self._value_frequencies:
                self._value_frequencies[i] = {}

            value_str = str(value)
            if value_str not in self._value_frequencies[i]:
                self._value_frequencies[i][value_str] = 0
            self._value_frequencies[i][value_str] += 1

        # Estimate issue weights from variance
        self._update_issue_weights()

        # Adapt concession rate
        self._adapt_concession()

    def _update_issue_weights(self) -> None:
        """Estimate issue weights from frequency variance."""
        for issue_idx, freq_map in self._value_frequencies.items():
            if not freq_map:
                self._issue_weights[issue_idx] = 1.0
                continue

            values = list(freq_map.values())
            n = len(values)
            if n <= 1:
                self._issue_weights[issue_idx] = 1.0
                continue

            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n

            # Higher variance = more important issue
            self._issue_weights[issue_idx] = 1.0 + math.sqrt(variance)

    def _adapt_concession(self) -> None:
        """Adapt concession factor based on opponent behavior."""
        if len(self._opponent_utilities) < 5:
            return

        recent = self._opponent_utilities[-5:]
        change = recent[-1] - recent[0]

        # If opponent conceding (our utility increasing), slow down
        if change > 0.05:
            self._concession_factor = max(
                0.1, self._concession_factor - self._learning_rate
            )
        # If opponent hardening, speed up
        elif change < -0.02:
            self._concession_factor = min(
                0.5, self._concession_factor + self._learning_rate
            )

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility with weighted issues."""
        if self._total_opponent_offers == 0 or bid is None:
            return 0.0

        total_score = 0.0
        total_weight = (
            sum(self._issue_weights.values()) if self._issue_weights else len(bid)
        )

        for i, value in enumerate(bid):
            value_str = str(value)
            weight = self._issue_weights.get(i, 1.0)

            if i in self._value_frequencies and value_str in self._value_frequencies[i]:
                freq = self._value_frequencies[i][value_str]
                # Weighted frequency score
                total_score += (freq / self._total_opponent_offers) * weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _get_target_utility(self, time: float) -> float:
        """Get target utility with adaptive concession."""
        # Exponential concession with adaptive factor
        target = 1.0 - math.pow(time, 1.0 / self._concession_factor)

        # Scale to utility range
        scaled = self._min_utility + (self._max_utility - self._min_utility) * target

        return max(scaled, self._min_utility_param)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid maximizing Nash product."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)

        # Get candidates
        candidates = self._outcome_space.get_bids_in_range(target - 0.05, target + 0.05)

        if not candidates:
            bid_details = self._outcome_space.get_bid_near_utility(target)
            return bid_details.bid if bid_details else self._best_bid

        if len(candidates) == 1:
            return candidates[0].bid

        # Nash product selection
        if self._total_opponent_offers > 5:
            best_nash = -1.0
            best_bid = candidates[0].bid
            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                nash = bd.utility * opp_util
                if nash > best_nash:
                    best_nash = nash
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
