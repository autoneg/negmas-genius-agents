"""AteamAgent from ANAC 2018."""

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

__all__ = ["AteamAgent"]


class AteamAgent(SAONegotiator):
    """
    AteamAgent from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    AteamAgent implements a team-inspired collaborative strategy using sigmoid-based
    concession that allows rapid adaptation in the late game. The agent monitors
    opponent cooperation and adjusts its behavior accordingly, seeking win-win
    outcomes through opponent modeling.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.ateamagent.ATeamAgent

    **Offering Strategy:**
        Uses a sigmoid concession curve: S(t) = 1 / (1 + e^(k*(t-0.7))) which
        maintains high utility early and drops rapidly around t=0.7. The target
        is adjusted -0.05 when opponent appears cooperative (recent utility
        increasing). Bids are selected using a weighted score: 0.7*own_utility +
        0.3*estimated_opponent_utility to balance self-interest with cooperation.

    **Acceptance Strategy:**
        Accepts offers meeting the current target utility. Under time pressure
        (t >= 0.9), accepts offers above the minimum utility parameter.
        Near deadline (t >= 0.98), accepts anything above the minimum utility
        in the outcome space.

    **Opponent Modeling:**
        Tracks frequency of values in opponent offers. Monitors cooperation by
        checking if utilities from opponent's last 5 offers are trending upward.
        Uses frequency-based utility estimation to select bids that may appeal
        to the opponent while maintaining acceptable own utility.

    Args:
        min_utility: Minimum utility threshold (default 0.5).
        sigmoid_steepness: Steepness of sigmoid curve (default 10.0).
        cooperation_adjustment_time: Time after which to adjust for cooperative opponent (default 0.3).
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
        min_utility: float = 0.5,
        sigmoid_steepness: float = 10.0,
        cooperation_adjustment_time: float = 0.3,
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
        self._sigmoid_steepness = sigmoid_steepness
        self._cooperation_adjustment_time = cooperation_adjustment_time
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

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update opponent model."""
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

    def _is_opponent_cooperative(self) -> bool:
        """Check if opponent appears cooperative."""
        if len(self._opponent_utilities) < 5:
            return False

        # Check if utility trend is positive
        recent = self._opponent_utilities[-5:]
        return recent[-1] > recent[0]

    def _get_target_utility(self, time: float) -> float:
        """Get target utility using sigmoid concession."""
        # Sigmoid function: stays high early, drops rapidly near midpoint
        # S(t) = 1 / (1 + e^(k*(t-0.7)))
        sigmoid = 1.0 / (1.0 + math.exp(self._sigmoid_steepness * (time - 0.7)))

        # Map to utility range
        range_span = 1.0 - self._min_utility_param
        target = self._min_utility_param + range_span * sigmoid

        # Adjust for cooperative opponent
        if self._is_opponent_cooperative() and time > self._cooperation_adjustment_time:
            target = max(target - 0.05, self._min_utility_param)

        # Scale to actual utility range
        scaled = self._min_utility + (self._max_utility - self._min_utility) * target

        return max(scaled, self._min_utility_param)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid with opponent consideration."""
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

        # Use opponent model for selection
        if self._total_opponent_offers > 5:
            best_score = -1.0
            best_bid = candidates[0].bid
            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                # Weighted combination: own utility + opponent utility
                score = 0.7 * bd.utility + 0.3 * opp_util
                if score > best_score:
                    best_score = score
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
