"""
SACRA from ANAC 2019.

This module contains the Python reimplementation of SACRA
(Simulated Annealing-based Concession Rate controlling Agent).

References:
    Baarslag, T., Fujita, K., Gerding, E. H., Hindriks, K., Ito, T.,
    Jennings, N. R., ... & Williams, C. R. (2019). The Tenth International
    Automated Negotiating Agents Competition (ANAC 2019).
    In Proceedings of the International Joint Conference on Autonomous
    Agents and Multiagent Systems (AAMAS).

Original Genius class: agents.anac.y2019.sacra.SACRA
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["SACRA"]


class SACRA(SAONegotiator):
    """
    SACRA from ANAC 2019.

    SACRA (Simulated Annealing-based Concession Rate controlling Agent)
    derives its name from using simulated annealing to estimate the
    opponent's utility space under preference uncertainty. Since NegMAS
    negotiators are given an exact utility function, that estimation step
    is not needed here; this port focuses on SACRA's runtime bidding and
    acceptance strategy, which controls its concession rate based on how
    much the opponent itself has conceded so far.

    Note:
        This is an AI-generated reimplementation based on the original Java
        code from the Genius framework. It may not behave identically to
        the original.

    **Offering Strategy:**
        - First offer: the maximum-utility bid.
        - Afterwards, tracks the utility of the very first bid received
          from the opponent and the most recent one.
        - ``concession_rate = max(0, u(last) - u(first)) / u(max) * 0.7``
          -- SACRA only concedes proportionally to how much the opponent
          itself has conceded (reciprocal concession), scaled down by 0.7.
        - ``target_utility = u(max) - concession_rate``
        - Offers a random bid with utility >= target_utility.

    **Acceptance Strategy:**
        - Computes an acceptance probability:
          ``(u(last) - target) / (u(max) - target)``
        - Accepts with that probability (stochastic acceptance, higher
          probability the closer the offer is to our own best bid).
        - Degenerates safely to "reject" if the opponent has not conceded
          at all yet (denominator is zero).

    **Opponent Modeling:**
        - No explicit issue-weight opponent model is used at runtime;
          SACRA's opponent model in Genius is only used to *estimate its own*
          utility function under preference uncertainty (not applicable
          here since our ufun is known exactly).

    Args:
        concession_scale: Multiplier applied to the opponent's own
            concession to determine our concession (default 0.7).
        denominator_epsilon: Epsilon below which the acceptance-probability
            denominator is treated as zero (default 1e-9).
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
        concession_scale: float = 0.7,
        denominator_epsilon: float = 1e-9,
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
        self._concession_scale = concession_scale
        self._denominator_epsilon = denominator_epsilon
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        self._max_utility_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._first_received_bid: Outcome | None = None
        self._last_received_bid: Outcome | None = None

    def _initialize(self) -> None:
        if self._initialized:
            return
        if self.ufun is None:
            return
        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._initialize()
        self._first_received_bid = None
        self._last_received_bid = None

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        if not self._initialized:
            self._initialize()

        if self.ufun is None:
            return self._max_utility_bid

        if self._last_received_bid is None or self._first_received_bid is None:
            return self._max_utility_bid

        u_first = float(self.ufun(self._first_received_bid))
        u_last = float(self.ufun(self._last_received_bid))
        u_max = self._max_utility if self._max_utility > 0 else 1.0

        concession_rate = max(0.0, u_last - u_first) / u_max * self._concession_scale
        target_utility = u_max - concession_rate

        candidates = (
            self._outcome_space.get_bids_above(target_utility)
            if self._outcome_space is not None
            else []
        )
        if not candidates:
            return self._max_utility_bid
        return random.choice(candidates).bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self._first_received_bid is None:
            self._first_received_bid = offer
        self._last_received_bid = offer

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        u_first = float(self.ufun(self._first_received_bid))
        u_last = float(self.ufun(offer))
        u_max = self._max_utility if self._max_utility > 0 else 1.0

        concession_rate = max(0.0, u_last - u_first) / u_max * self._concession_scale
        target_utility = u_max - concession_rate

        denominator = u_max - target_utility
        if denominator <= self._denominator_epsilon:
            # Opponent has not conceded at all: only accept its best bid.
            accept_probability = 1.0 if u_last >= u_max else 0.0
        else:
            accept_probability = (u_last - target_utility) / denominator
            accept_probability = max(0.0, min(1.0, accept_probability))

        if random.random() < accept_probability:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER
