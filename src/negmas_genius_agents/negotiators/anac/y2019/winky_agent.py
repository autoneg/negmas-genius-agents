"""
WinkyAgent from ANAC 2019.

This agent won the Nash-based category in ANAC 2019. WinkyAgent
uses Nash equilibrium estimation combined with time-dependent
concession for effective negotiation.

References:
    Baarslag, T., Fujita, K., Gerding, E. H., Hindriks, K., Ito, T.,
    Jennings, N. R., ... & Williams, C. R. (2019). The Tenth International
    Automated Negotiating Agents Competition (ANAC 2019).
    In Proceedings of the International Joint Conference on Autonomous
    Agents and Multiagent Systems (AAMAS).

Original Genius class: agents.anac.y2019.winkyagent.winkyAgent
"""

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

__all__ = ["WinkyAgent"]


class WinkyAgent(SAONegotiator):
    """
    WinkyAgent from ANAC 2019 - Nash-based category winner.

    WinkyAgent won the Nash-based category in ANAC 2019 by combining
    Nash product optimization with time-dependent concession. The agent
    aims to find mutually beneficial outcomes that approximate the
    Nash bargaining solution.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    **Offering Strategy:**
        - Polynomial concession: target = max - (max - min) * t^(1/e)
        - With default e=0.2, this is a conceder-like curve
        - After 3 opponent offers, selects bids maximizing Nash product
        - Nash product = own_utility * estimated_opponent_utility
        - Before sufficient data, randomly selects from good candidates
        - First offer is always the best available bid

    **Acceptance Strategy:**
        - Accepts offers meeting or exceeding the polynomial target
        - Near deadline (t >= 0.99): Accepts offers above 0.5 utility
          or 95% of best received offer
        - Tracks best received offer for fallback decisions

    **Opponent Modeling:**
        - Frequency-based model tracking issue value occurrences
        - Estimates opponent utility as normalized frequency scores
        - Higher frequency = higher estimated preference
        - Used to compute Nash product for bid selection
        - Aims to select Pareto-efficient outcomes near Nash point

    Args:
        e: Concession exponent (default 0.2, conceder-like curve)
        deadline_threshold: Time threshold for deadline acceptance (default 0.99)
        deadline_min_utility: Minimum utility for deadline acceptance (default 0.5)
        deadline_best_ratio: Ratio of best received utility for deadline acceptance (default 0.95)
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
        deadline_threshold: float = 0.99,
        deadline_min_utility: float = 0.5,
        deadline_best_ratio: float = 0.95,
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
        self._deadline_threshold = deadline_threshold
        self._deadline_min_utility = deadline_min_utility
        self._deadline_best_ratio = deadline_best_ratio
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_value_freq: dict[int, dict[str, int]] = {}
        self._opponent_offers_count: int = 0

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._best_received_offer: Outcome | None = None
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
        self._opponent_value_freq = {}
        self._opponent_offers_count = 0
        self._best_received_offer = None
        self._best_received_utility = 0.0

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update frequency-based opponent model."""
        if bid is None:
            return

        self._opponent_offers_count += 1

        for i, value in enumerate(bid):
            if i not in self._opponent_value_freq:
                self._opponent_value_freq[i] = {}
            value_str = str(value)
            if value_str not in self._opponent_value_freq[i]:
                self._opponent_value_freq[i][value_str] = 0
            self._opponent_value_freq[i][value_str] += 1

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility based on frequency model."""
        if bid is None or self._opponent_offers_count == 0:
            return 0.5

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            value_str = str(value)
            if i in self._opponent_value_freq:
                freq = self._opponent_value_freq[i].get(value_str, 0)
                max_freq = (
                    max(self._opponent_value_freq[i].values())
                    if self._opponent_value_freq[i]
                    else 1
                )
                total_score += freq / max_freq if max_freq > 0 else 0

        return total_score / num_issues if num_issues > 0 else 0.5

    def _compute_nash_product(self, bid: Outcome) -> float:
        """Compute Nash product for bid selection."""
        if bid is None or self.ufun is None:
            return 0.0

        own_utility = float(self.ufun(bid))
        opp_utility = self._estimate_opponent_utility(bid)

        # Nash product: u1 * u2
        return own_utility * opp_utility

    def _get_target_utility(self, time: float) -> float:
        """Get target utility based on concession curve."""
        # Polynomial concession: starts high, decreases over time
        # u(t) = 1 - t^(1/e)
        min_target = max(0.5, self._min_utility)
        target = self._max_utility - (self._max_utility - min_target) * (
            time ** (1.0 / self._e)
        )
        return max(target, min_target)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid maximizing Nash product above target utility."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return self._best_bid

        target = self._get_target_utility(time)

        # Get candidates above target
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            # Fallback to best available
            bid_detail = self._outcome_space.get_bid_near_utility(target)
            return bid_detail.bid if bid_detail else self._best_bid

        if len(candidates) == 1:
            return candidates[0].bid

        # Select bid with highest Nash product
        if self._opponent_offers_count >= 3:
            best_nash = -1.0
            best_bid = candidates[0].bid
            for bd in candidates:
                nash = self._compute_nash_product(bd.bid)
                if nash > best_nash:
                    best_nash = nash
                    best_bid = bd.bid
            return best_bid
        else:
            # Early in negotiation, just pick randomly from good bids
            return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        # First move: best bid
        if self._opponent_offers_count == 0:
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

        # Update opponent model
        self._update_opponent_model(offer)

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        time = state.relative_time

        # Track best received offer
        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility
            self._best_received_offer = offer

        # Get target utility
        target = self._get_target_utility(time)

        # Accept if above target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if reasonable
        if time >= self._deadline_threshold:
            if offer_utility >= self._deadline_min_utility:
                return ResponseType.ACCEPT_OFFER
            if offer_utility >= self._best_received_utility * self._deadline_best_ratio:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
