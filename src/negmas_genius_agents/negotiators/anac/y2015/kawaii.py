"""
Kawaii negotiation agent from ANAC 2015 (Fairy agent).

This module implements Kawaii (from the Fairy agent package), a negotiation
agent that competed in the Sixth International Automated Negotiating Agents
Competition (ANAC 2015). Kawaii uses an adaptive strategy that adjusts
concession based on opponent behavior patterns.

Original Java class: agents.anac.y2015.fairy.kawaii

References:
    ANAC 2015 competition:
    https://ii.tudelft.nl/negotiation/node/12

    Aydogan, R., Festen, D., Hindriks, K., & Jonker, C. (2017).
    Alternating Offers Protocols for Multilateral Negotiation.
    In Modern Approaches to Agent-based Complex Automated Negotiation.
    Springer. (ANAC 2015 Proceedings)
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

__all__ = ["Kawaii"]


class Kawaii(SAONegotiator):
    """
    Kawaii from ANAC 2015 - Fairy agent.

    Kawaii uses a soft, adaptive strategy:
    1. Starts with high expectations but adapts quickly
    2. Uses opponent concession rate to adjust own behavior
    3. Has gentle but persistent negotiation style
    4. Accepts deals that show mutual benefit

    Key features:
    - Adaptive concession based on opponent behavior
    - Tracks opponent's concession pattern
    - Friendly acceptance criteria near deadline
    - Smooth utility transitions

    Args:
        e: Initial concession exponent (default 0.2)
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
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._reservation_value: float = 0.0

        # Opponent tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._opponent_utilities: list[float] = []
        self._opponent_concession_rate: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = self._outcome_space.min_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._opponent_utilities = []
        self._opponent_concession_rate = 0.0

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent behavior and estimate concession rate."""
        self._opponent_bids.append((bid, utility))
        self._opponent_utilities.append(utility)

        if len(self._opponent_utilities) >= 3:
            # Calculate average change in utility offered
            changes = []
            for i in range(1, len(self._opponent_utilities)):
                changes.append(
                    self._opponent_utilities[i] - self._opponent_utilities[i - 1]
                )
            self._opponent_concession_rate = sum(changes) / len(changes)

    def _adaptive_e(self, time: float) -> float:
        """Adjust concession exponent based on opponent behavior."""
        base_e = self._e

        if len(self._opponent_utilities) < 5:
            return base_e

        # If opponent is conceding (giving us better deals), we can be firmer
        if self._opponent_concession_rate > 0.01:
            return base_e * 0.7  # More boulware-like

        # If opponent is tough, we need to concede more
        if self._opponent_concession_rate < -0.01:
            return base_e * 1.5  # More conceder-like

        return base_e

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold with adaptive concession."""
        e = self._adaptive_e(time)

        # Kawaii formula: gentle but persistent
        if time < 0.3:
            # Early phase: stay high
            f_t = 0.0
        else:
            # Main phase: smooth concession
            adjusted_time = (time - 0.3) / 0.7
            f_t = math.pow(adjusted_time, 1 / e) if e != 0 else adjusted_time

        target = self._max_utility - (self._max_utility - self._min_utility) * 0.4 * f_t
        return max(target, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid to offer."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Gentle randomization
        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        return self._select_bid(state.relative_time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        self._update_opponent_model(offer, offer_utility)

        threshold = self._compute_threshold(time)

        # Accept if meets threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, be more accepting
        if time > 0.95:
            if offer_utility >= self._min_utility + 0.3 * (
                self._max_utility - self._min_utility
            ):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
