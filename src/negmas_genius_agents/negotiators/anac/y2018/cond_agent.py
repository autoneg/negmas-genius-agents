"""ConDAgent from ANAC 2018."""

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

__all__ = ["ConDAgent"]


class ConDAgent(SAONegotiator):
    """
    ConDAgent (Conditional Dependent Agent) from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    ConDAgent adapts its negotiation strategy based on opponent cooperation level.
    The agent monitors opponent concession patterns and switches between cooperative
    (faster concession) and competitive (slower concession) modes accordingly.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.condagent.ConDAgent

    **Offering Strategy:**
        Uses exponential concession: target = 1 - t^(1/rate) where the rate is
        conditionally adjusted. When opponent is cooperative (conceding), the
        rate is multiplied by 1.5 (faster concession). When opponent is competitive,
        the rate is multiplied by 0.7 (slower concession). Bids are randomly
        selected from candidates within +/-0.04 of the target utility.

    **Acceptance Strategy:**
        Conditional thresholds based on opponent behavior. Near deadline (t >= 0.85):
        - Cooperative opponent: accepts offers >= target * 0.9 (more lenient)
        - Competitive opponent: accepts offers >= target * 0.95 (stricter)
        Under time pressure (t >= 0.95), accepts above minimum utility parameter.
        At deadline (t >= 0.99), accepts anything above minimum utility floor.

    **Opponent Modeling:**
        Tracks utilities of last 5 opponent offers and computes average change.
        Opponent is classified as cooperative if average change > 0.02 threshold
        (indicating their offers are improving for us). No explicit preference
        model - focuses purely on concession behavior.

    Args:
        base_rate: Base concession rate (default 0.08).
        min_utility: Minimum utility threshold (default 0.6).
        conditional_acceptance_time: Time threshold for conditional acceptance (default 0.85).
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
        base_rate: float = 0.08,
        min_utility: float = 0.6,
        conditional_acceptance_time: float = 0.85,
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
        self._base_rate = base_rate
        self._min_utility_param = min_utility
        self._conditional_acceptance_time = conditional_acceptance_time
        self._time_pressure_threshold = time_pressure_threshold
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent tracking
        self._opponent_utilities: list[float] = []
        self._is_cooperative: bool = False
        self._cooperation_threshold: float = 0.02

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
        self._opponent_utilities = []
        self._is_cooperative = False
        self._last_received_offer = None

    def _update_cooperation_status(self) -> None:
        """Determine if opponent is being cooperative."""
        if len(self._opponent_utilities) < 5:
            self._is_cooperative = False
            return

        # Check if opponent utilities (to us) are increasing
        recent = self._opponent_utilities[-5:]
        avg_change = (recent[-1] - recent[0]) / len(recent)

        self._is_cooperative = avg_change > self._cooperation_threshold

    def _get_concession_rate(self) -> float:
        """Get current concession rate based on cooperation."""
        if self._is_cooperative:
            # Opponent is cooperative, concede faster
            return self._base_rate * 1.5
        else:
            # Opponent is competitive, concede slower
            return self._base_rate * 0.7

    def _get_target_utility(self, time: float) -> float:
        """Get target utility with conditional concession."""
        rate = self._get_concession_rate()

        # Exponential concession with conditional rate
        target = 1.0 - math.pow(time, 1.0 / rate)

        # Scale to utility range
        scaled = self._min_utility + (self._max_utility - self._min_utility) * target

        return max(scaled, self._min_utility_param)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid near target utility."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)

        # Get candidates
        margin = 0.04
        candidates = self._outcome_space.get_bids_in_range(
            target - margin, target + margin
        )

        if not candidates:
            bid_details = self._outcome_space.get_bid_near_utility(target)
            return bid_details.bid if bid_details else self._best_bid

        return random.choice(candidates).bid

    def _is_acceptable(self, offer_utility: float, time: float) -> bool:
        """Check if an offer is acceptable with conditional thresholds."""
        target = self._get_target_utility(time)

        # Basic acceptance
        if offer_utility >= target:
            return True

        # Conditional acceptance near deadline
        if time >= self._conditional_acceptance_time:
            if self._is_cooperative:
                # Be more lenient with cooperative opponent
                if offer_utility >= target * 0.9:
                    return True
            else:
                # Be stricter with competitive opponent
                if offer_utility >= target * 0.95:
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
        offer_utility = float(self.ufun(offer))

        # Track opponent behavior
        self._opponent_utilities.append(offer_utility)
        self._update_cooperation_status()

        time = state.relative_time

        if self._is_acceptable(offer_utility, time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
