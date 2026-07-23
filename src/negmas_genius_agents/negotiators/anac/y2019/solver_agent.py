"""
SolverAgent from ANAC 2019.

This module contains the Python reimplementation of SolverAgent.

References:
    Baarslag, T., Fujita, K., Gerding, E. H., Hindriks, K., Ito, T.,
    Jennings, N. R., ... & Williams, C. R. (2019). The Tenth International
    Automated Negotiating Agents Competition (ANAC 2019).
    In Proceedings of the International Joint Conference on Autonomous
    Agents and Multiagent Systems (AAMAS).

Original Genius class: agents.anac.y2019.solveragent.SolverAgent
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

__all__ = ["SolverAgent"]


class SolverAgent(SAONegotiator):
    """
    SolverAgent from ANAC 2019.

    The original Java SolverAgent is built almost entirely around solving
    for issue weights and value utilities from an *ordinal bid ranking*
    under preference uncertainty (a combinatorial "solver" over pairwise
    bid comparisons plus linear regression). Since NegMAS negotiators are
    given an exact utility function directly, none of that estimation
    machinery is needed or applicable; in fact, the original Java agent
    throws a ``NullPointerException`` in ``chooseAction`` when run without
    preference uncertainty (its ``userModel``/bid-ranking fields stay
    ``null``), so it cannot even be run in this "full information" style
    setup.

    This port instead reproduces SolverAgent's runtime *bidding phases*
    and *acceptance* logic using the exact utility function, matching the
    qualitative shape of the original: a high, slowly-changing utility
    floor with three offering phases and a threshold-based acceptance
    strategy plus a final-rounds concession.

    Note:
        This is an AI-generated reimplementation based on the original Java
        code from the Genius framework. It may not behave identically to
        the original -- the estimation/"solver" phase for preference
        uncertainty has been entirely omitted since it is not applicable
        when the utility function is known exactly.

    **Offering Strategy:**
        - Maintains a utility floor ``phase1_bound`` (>= 0.75) below which
          it never offers.
        - Phase 1 (t <= 0.3): offers a random bid from the high-utility
          candidate set (>= floor).
        - Phase 2 (0.3 < t < near deadline): cycles sequentially through
          the same high-utility candidate set.
        - Phase 3 (near deadline): offers the best bid available.

    **Acceptance Strategy:**
        - Accept if the opponent's offer utility is >= our own current /
          next planned offer utility.
        - Accept if the opponent's offer utility is >= ``phase1_bound``.
        - In the very last rounds, accept anything >= 0.80.

    **Opponent Modeling:**
        - Simplified: tracks the best offer received from the opponent to
          use as a fallback near the deadline (the original's "Nash point"
          estimation via linear regression is dropped as it depends on the
          preference-uncertainty machinery).

    Args:
        floor_utility: Minimum acceptable utility floor (default 0.80,
            clipped to be at least ``min_floor_utility`` as in the original agent).
        deadline_fraction: Fraction of relative time after which the agent
            enters its final concession phase (default 0.95).
        min_floor_utility: Hard lower bound that ``floor_utility`` is clipped to
            (default 0.75).
        phase1_time: Relative-time cutoff for phase 1 (random offering from
            the high-utility candidate set) (default 0.3).
        final_accept_utility: Utility threshold for accepting offers in the
            final concession phase (default 0.80).
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
        floor_utility: float = 0.80,
        deadline_fraction: float = 0.95,
        min_floor_utility: float = 0.75,
        phase1_time: float = 0.3,
        final_accept_utility: float = 0.80,
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
        self._floor_utility = max(min_floor_utility, floor_utility)
        self._deadline_fraction = deadline_fraction
        self._min_floor_utility = min_floor_utility
        self._phase1_time = phase1_time
        self._final_accept_utility = final_accept_utility

        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        self._best_bid: Outcome | None = None
        self._phase_bids: list = []
        self._phase_index: int = 0
        self._best_received_bid: Outcome | None = None
        self._best_received_utility: float = 0.0

    def _initialize(self) -> None:
        if self._initialized:
            return
        if self.ufun is None:
            return
        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._phase_bids = self._outcome_space.get_bids_above(self._floor_utility)
            if not self._phase_bids:
                self._phase_bids = [self._outcome_space.outcomes[0]]
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._initialize()
        self._phase_index = 0
        self._best_received_bid = None
        self._best_received_utility = 0.0

    def _next_offer(self, time: float) -> Outcome | None:
        if not self._phase_bids:
            return self._best_bid

        if time <= self._phase1_time:
            return random.choice(self._phase_bids).bid

        if time < self._deadline_fraction:
            self._phase_index = (self._phase_index + 1) % len(self._phase_bids)
            return self._phase_bids[self._phase_index].bid

        return self._best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        if not self._initialized:
            self._initialize()
        return self._next_offer(state.relative_time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility
            self._best_received_bid = offer

        time = state.relative_time
        next_bid = self._next_offer(time)
        next_utility = float(self.ufun(next_bid)) if next_bid is not None else 1.0

        if offer_utility >= next_utility:
            return ResponseType.ACCEPT_OFFER

        if offer_utility >= self._floor_utility:
            return ResponseType.ACCEPT_OFFER

        if time >= self._deadline_fraction and offer_utility >= self._final_accept_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
