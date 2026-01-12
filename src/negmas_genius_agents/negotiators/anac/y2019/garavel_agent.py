"""GaravelAgent from ANAC 2019."""

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

__all__ = ["GaravelAgent"]


class GaravelAgent(SAONegotiator):
    """
    GaravelAgent from ANAC 2019.

    GaravelAgent implements a tit-for-tat inspired negotiation strategy
    that responds to opponent concession patterns. If the opponent
    concedes, the agent remains firm; if the opponent hardens, the
    agent concedes to maintain negotiation progress.

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

        Original Genius class: ``agents.anac.y2019.garavelagent.GaravelAgent``

    **Offering Strategy:**
        - Polynomial base concession: target = initial - (initial - min) * t^1.5
        - Adaptive adjustment based on opponent's total concession:
          - If opponent conceded significantly (delta > 0.1): +0.05 (stay firm)
          - If opponent hardened (delta < 0): -0.05 (concede more)
        - Selects bids nearest to the adjusted target utility
        - First offer is always the best available bid

    **Acceptance Strategy:**
        - Accepts offers meeting or exceeding the adaptive target
        - Near deadline (t >= 0.98): Accepts offers above minimum target
        - Very near deadline (t >= 0.99): Accepts 95% of best received offer
        - Tracks all opponent offer utilities for comparison

    **Opponent Modeling:**
        - Tracks utility (for self) of all opponent offers in sequence
        - Estimates opponent concession as: last_offer - first_offer
        - Positive value indicates opponent is giving better offers over time
        - Simple but effective measure of opponent flexibility
        - Does not model issue-level preferences

    Args:
        initial_target: Starting target utility (default 0.95)
        min_target: Minimum acceptable utility (default 0.55)
        near_deadline_time: Time threshold for near deadline (default 0.98)
        final_deadline_time: Time threshold for final deadline (default 0.99)
        final_best_ratio: Ratio of best received utility for final deadline acceptance (default 0.95)
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
        min_target: float = 0.55,
        near_deadline_time: float = 0.98,
        final_deadline_time: float = 0.99,
        final_best_ratio: float = 0.95,
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
        self._near_deadline_time = near_deadline_time
        self._final_deadline_time = final_deadline_time
        self._final_best_ratio = final_best_ratio
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent tracking
        self._opponent_offers: list[float] = []
        self._opponent_offers_count: int = 0

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

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
        self._opponent_offers = []
        self._opponent_offers_count = 0

    def _estimate_opponent_concession(self) -> float:
        """Estimate how much the opponent has conceded."""
        if len(self._opponent_offers) < 2:
            return 0.0

        # Compare first and last offers (improvement for us = concession by opponent)
        first = self._opponent_offers[0]
        last = self._opponent_offers[-1]

        return last - first

    def _get_target_utility(self, time: float) -> float:
        """Get target utility based on time and opponent behavior."""
        # Base concession (polynomial)
        base_target = self._initial_target - (
            self._initial_target - self._min_target
        ) * (time**1.5)

        # Adjust based on opponent concession
        opp_concession = self._estimate_opponent_concession()

        if opp_concession > 0.1:
            # Opponent conceded a lot, we can stay firmer
            adjustment = 0.05
        elif opp_concession < 0:
            # Opponent is getting harder, we need to concede more
            adjustment = -0.05
        else:
            adjustment = 0.0

        target = base_target + adjustment
        return max(min(target, self._max_utility), self._min_target)

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
        self._opponent_offers.append(offer_utility)

        time = state.relative_time
        target = self._get_target_utility(time)

        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        if time >= self._near_deadline_time and offer_utility >= self._min_target:
            return ResponseType.ACCEPT_OFFER

        if time >= self._final_deadline_time and self._opponent_offers:
            if offer_utility >= max(self._opponent_offers) * self._final_best_ratio:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
