"""
AgentKN - ANAC 2017 Finalist.

This module contains the reimplementation of AgentKN from ANAC 2017.
Original: agents.anac.y2017.agentkn.AgentKN

AgentKN is part of the "Agent K" family of agents that have competed
in multiple ANAC competitions, building on the success of AgentK (2010)
and AgentK2 (2011).

References:
    ANAC 2017 competition proceedings.
    https://ii.tudelft.nl/nego/node/7
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

__all__ = ["AgentKN"]


class AgentKN(SAONegotiator):
    """
    AgentKN from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    AgentKN uses a sophisticated time-dependent strategy with adaptive
    concession, building on the legacy of the Agent K family.

    **Offering Strategy:**
        Uses a sigmoid-based concession curve that starts slow (conservative
        early game), accelerates in the middle phase, and slows down near
        the reservation value. Bids are selected randomly from a range
        around the current threshold.

    **Acceptance Strategy:**
        Accepts offers above the dynamically calculated threshold. The
        threshold adapts based on opponent behavior - becomes more lenient
        against aggressive opponents and more demanding against cooperative
        ones. Late-game (>95% time) considers accepting opponent's best offer.

    **Opponent Modeling:**
        Tracks opponent's average offer utility and best offer. Uses this
        information to adapt concession rate: cooperative opponents (avg >0.6)
        result in higher patience, while aggressive opponents (avg <0.3)
        trigger faster concession.

    Args:
        min_utility: Minimum acceptable utility (default 0.55).
        concession_rate: Controls steepness of sigmoid concession (default 5.0).
        late_game_threshold: Time threshold for late game pressure (default 0.95).
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
        concession_rate: float = 5.0,
        late_game_threshold: float = 0.95,
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
        self._min_utility = min_utility
        self._concession_rate = concession_rate
        self._late_game_threshold = late_game_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

        # Opponent tracking
        self._opponent_best_utility: float = 0.0
        self._opponent_bid_count: int = 0
        self._opponent_utility_sum: float = 0.0

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

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._opponent_best_utility = 0.0
        self._opponent_bid_count = 0
        self._opponent_utility_sum = 0.0

    def _update_opponent_model(self, offer: Outcome) -> None:
        """Update opponent tracking."""
        if self.ufun is None:
            return

        offer_utility = float(self.ufun(offer))
        self._opponent_best_utility = max(self._opponent_best_utility, offer_utility)
        self._opponent_bid_count += 1
        self._opponent_utility_sum += offer_utility

    def _get_opponent_avg_utility(self) -> float:
        """Get average utility of opponent's offers."""
        if self._opponent_bid_count == 0:
            return 0.5
        return self._opponent_utility_sum / self._opponent_bid_count

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for smooth concession."""
        return 1.0 / (1.0 + math.exp(-self._concession_rate * (x - 0.5)))

    def _calculate_threshold(self, time: float) -> float:
        """Calculate acceptance threshold using sigmoid concession."""
        # Sigmoid maps [0,1] -> ~[0,1] with S-curve
        sigmoid_val = self._sigmoid(time)

        # Map sigmoid output to utility range
        utility_range = self._max_utility - self._min_utility
        threshold = self._max_utility - sigmoid_val * utility_range

        # Adjust based on opponent behavior
        opponent_avg = self._get_opponent_avg_utility()
        if opponent_avg > 0.6:
            # Opponent is cooperative, we can be slightly more patient
            threshold = min(threshold + 0.03, self._max_utility)
        elif opponent_avg < 0.3:
            # Opponent is aggressive, need to concede more
            threshold = max(threshold - 0.03, self._min_utility)

        # Late game pressure
        if time > self._late_game_threshold:
            # Consider accepting opponent's best if it's reasonable
            if self._opponent_best_utility > self._min_utility:
                threshold = min(threshold, self._opponent_best_utility)

        return max(threshold, self._min_utility)

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Select a bid near the threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        # Get bids in a range around the threshold
        candidates = self._outcome_space.get_bids_in_range(threshold, threshold + 0.1)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(self._min_utility)

        if candidates:
            return random.choice(candidates).bid

        return self._best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        threshold = self._calculate_threshold(time)

        return self._select_bid(threshold)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        self._update_opponent_model(offer)

        time = state.relative_time
        threshold = self._calculate_threshold(time)
        offer_utility = float(self.ufun(offer))

        # Accept if above threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
