"""AgentLarry from ANAC 2019."""

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

__all__ = ["AgentLarry"]


class AgentLarry(SAONegotiator):
    """
    AgentLarry from ANAC 2019.

    AgentLarry implements a simple, robust time-dependent concession
    strategy using linear concession. Its simplicity makes it predictable
    but also reliable across different negotiation scenarios.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        .. code-block:: bibtex

            @inproceedings{baarslag2019tenth,
                title={The Tenth International Automated Negotiating Agents
                       Competition (ANAC 2019)},
                author={Baarslag, Tim and Fujita, Katsuhide and Gerding,
                        Enrico H and Hindriks, Koen and Ito, Takayuki and
                        Jennings, Nicholas R and others},
                booktitle={Proceedings of the International Joint Conference
                           on Autonomous Agents and Multiagent Systems (AAMAS)},
                year={2019}
            }

        Original Genius class: ``agents.anac.y2019.agentlarry.AgentLarry``

    **Offering Strategy:**
        - Linear concession: target = initial - (initial - min) * t
        - Starts at initial_target (0.95) and linearly decreases to min_target (0.6)
        - Selects bids nearest to the current target utility
        - First offer is always the best available bid
        - No opponent modeling for bid selection - purely time-based

    **Acceptance Strategy:**
        - Accepts offers meeting or exceeding the current target utility
        - Very near deadline (t >= 0.99): Accepts offers above minimum target
        - Simple threshold-based decision without adaptive adjustments

    **Opponent Modeling:**
        - Minimal opponent modeling - only tracks offer count
        - Does not adapt strategy based on opponent behavior
        - Relies entirely on time-based concession for robustness

    Args:
        initial_target: Starting target utility (default 0.95)
        min_target: Minimum acceptable utility (default 0.6)
        deadline_threshold: Time threshold for deadline acceptance (default 0.99)
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
        initial_target: float = 0.95,
        min_target: float = 0.6,
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
        self._initial_target = initial_target
        self._min_target = min_target
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._opponent_offers_count: int = 0

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
        self._opponent_offers_count = 0

    def _get_target_utility(self, time: float) -> float:
        """Get target utility based on linear concession."""
        # Linear concession
        target = self._initial_target - (self._initial_target - self._min_target) * time
        return max(target, self._min_target)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid near target utility."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return self._best_bid

        target = self._get_target_utility(time)
        bid_detail = self._outcome_space.get_bid_near_utility(target)
        return bid_detail.bid if bid_detail else self._best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        # First offer: best bid
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

        self._opponent_offers_count += 1

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        time = state.relative_time
        target = self._get_target_utility(time)

        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        if time >= self._deadline_threshold and offer_utility >= self._min_target:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
