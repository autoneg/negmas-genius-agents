"""EAgent from ANAC 2019."""

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

__all__ = ["EAgent"]


class EAgent(SAONegotiator):
    """
    EAgent from ANAC 2019.

    EAgent implements an exponential decay concession strategy where
    the concession rate is controlled by an exponential function.
    This creates rapid initial concession that slows over time,
    the opposite of Boulware-style strategies.

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

        Original Genius class: ``agents.anac.y2019.eagent.EAgent``

    **Offering Strategy:**
        - Exponential decay: target = min + (max - min) * e^(-rate * t)
        - With default rate=3.0, ~95% concession occurs by t=1.0
        - Concedes rapidly early, then stabilizes near minimum
        - Selects bids nearest to the computed target utility
        - First offer is always the best available bid

    **Acceptance Strategy:**
        - Accepts offers meeting or exceeding the exponential target
        - Near deadline (t >= 0.95): Accepts offers above minimum utility
        - Very near deadline (t >= 0.99): Accepts 90% of best received offer
        - Tracks best received utility for deadline fallback

    **Opponent Modeling:**
        - Minimal modeling - only tracks offer count and best received
        - Does not use opponent model for bid selection
        - Relies on time-based strategy for robustness

    Args:
        decay_rate: Exponential decay rate (default 3.0, rapid decay)
        min_utility: Minimum acceptable utility (default 0.5)
        near_deadline_time: Time threshold for near deadline (default 0.95)
        final_deadline_time: Time threshold for final deadline (default 0.99)
        final_best_ratio: Ratio of best received utility for final deadline acceptance (default 0.9)
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
        decay_rate: float = 3.0,
        min_utility: float = 0.5,
        near_deadline_time: float = 0.95,
        final_deadline_time: float = 0.99,
        final_best_ratio: float = 0.9,
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
        self._decay_rate = decay_rate
        self._min_utility = min_utility
        self._near_deadline_time = near_deadline_time
        self._final_deadline_time = final_deadline_time
        self._final_best_ratio = final_best_ratio
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._opponent_offers_count: int = 0
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

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._opponent_offers_count = 0
        self._best_received_utility = 0.0

    def _get_target_utility(self, time: float) -> float:
        """Get target utility based on exponential decay."""
        # Exponential decay: target = min + (max - min) * e^(-rate * time)
        decay = math.exp(-self._decay_rate * time)
        target = self._min_utility + (self._max_utility - self._min_utility) * decay
        return max(target, self._min_utility)

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

        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility

        target = self._get_target_utility(time)

        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        if time >= self._near_deadline_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        if time >= self._final_deadline_time:
            if offer_utility >= self._best_received_utility * self._final_best_ratio:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
