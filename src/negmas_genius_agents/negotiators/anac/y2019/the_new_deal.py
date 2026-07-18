"""
TheNewDeal from ANAC 2019.

This module contains the Python reimplementation of TheNewDeal.

References:
    Baarslag, T., Fujita, K., Gerding, E. H., Hindriks, K., Ito, T.,
    Jennings, N. R., ... & Williams, C. R. (2019). The Tenth International
    Automated Negotiating Agents Competition (ANAC 2019).
    In Proceedings of the International Joint Conference on Autonomous
    Agents and Multiagent Systems (AAMAS).

Original Genius class: agents.anac.y2019.thenewdeal.TheNewDeal
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["TheNewDeal"]


class TheNewDeal(SAONegotiator):
    """
    TheNewDeal from ANAC 2019.

    Much of the original Java ``TheNewDeal`` implements an ordinal
    "equal-difference" utility estimator over a preference-uncertainty
    bid ranking (used to derive per-issue-value average weights when the
    exact utility function is unknown). Since NegMAS negotiators are given
    an exact utility function directly, that estimation step collapses:
    the agent instead reduces to what its ``chooseAction`` loop does with
    an exact utility ranking -- a slow, oscillating concession that steps
    down through the top half of the sorted outcome space and wraps back
    to the best bid, offering the same bid for several consecutive rounds
    (``ratio`` rounds) before stepping down.

    Note:
        This is an AI-generated reimplementation based on the original Java
        code from the Genius framework. It may not behave identically to
        the original -- the ordinal bid-ranking estimation phase (relevant
        only under preference uncertainty) has been omitted since it does
        not apply when the utility function is known exactly. In fact,
        the original Java agent throws a ``NullPointerException`` in
        ``chooseAction`` when run without preference uncertainty (its
        ``sortedUtilityArray``/bid-ranking fields stay ``null``), so it
        cannot even be run in this "full information" style setup.

    **Offering Strategy:**
        - Bids are indexed from the best (index 0) to the worst outcome.
        - ``ratio = (n_steps - 1) // n_outcomes`` rounds are spent
          offering the same bid before moving to the next-lower-utility
          bid (mirroring the original agent's integer-division ratio,
          including the degenerate ``ratio == 0`` case which effectively
          keeps the agent hard-headed on its best bid for large domains).
        - The bid index wraps back to 0 (the best bid) once it reaches
          half of the outcome space, so the agent never explores below
          the top 50% of outcomes.

    **Acceptance Strategy:**
        - Accepts the opponent's offer if its utility is >= the utility of
          our current (or next) planned offer.
        - In the final round, accepts the opponent's last offer (or offers
          our best bid if we appear to be the first mover and have not
          received anything yet).

    **Opponent Modeling:**
        - None -- TheNewDeal does not build an opponent model at runtime;
          its "opponent awareness" in Genius is limited to reading the
          opponent's offered utility directly from the (estimated) utility
          space.

    Args:
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
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        self._best_bid: Outcome | None = None
        self._n_outcomes: int = 0
        self._ratio: int = 0
        self._counter: int = 1
        self._round_counter: int = 0
        self._bid_index: int = 0
        self._n_steps: int | None = None
        self._first_mover: bool = False
        self._received_any: bool = False

    def _initialize(self, state: SAOState) -> None:
        if self._initialized:
            return
        if self.ufun is None:
            return
        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._n_outcomes = len(self._outcome_space.outcomes)

        try:
            self._n_steps = self.nmi.n_steps
        except Exception:
            self._n_steps = None

        if self._n_steps and self._n_outcomes:
            self._ratio = int((self._n_steps - 1) // self._n_outcomes)
        else:
            self._ratio = 1

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._initialize(state)
        self._counter = 1
        self._round_counter = 0
        self._bid_index = 0
        self._first_mover = False
        self._received_any = False

    def _current_bid(self) -> Outcome | None:
        if not self._outcome_space or not self._outcome_space.outcomes:
            return self._best_bid
        idx = min(self._bid_index, self._n_outcomes - 1)
        return self._outcome_space.outcomes[idx].bid

    def _step_bid_index(self) -> None:
        half = max(1, self._n_outcomes // 2)
        if self._counter >= self._ratio:
            self._counter = 1
            self._bid_index += 1
            if self._bid_index >= half:
                self._bid_index = 0
        self._counter += 1
        self._round_counter += 1

    def _near_deadline(self, state: SAOState) -> bool:
        if self._n_steps is None or state.step is None:
            return False
        return state.step >= self._n_steps - 2

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        if not self._initialized:
            self._initialize(state)

        if self._near_deadline(state):
            if self._first_mover or not self._received_any:
                return self._best_bid
            # Otherwise we would have accepted in respond(); fall back
            # to our best bid if we must still propose.
            return self._best_bid

        bid = self._current_bid()
        self._step_bid_index()
        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        if not self._initialized:
            self._initialize(state)

        offer = state.current_offer
        if offer is None:
            self._first_mover = True
            return ResponseType.REJECT_OFFER

        self._received_any = True

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))

        if self._near_deadline(state):
            return ResponseType.ACCEPT_OFFER

        our_bid = self._current_bid()
        our_utility = float(self.ufun(our_bid)) if our_bid is not None else 1.0

        our_bid2 = None
        if self._outcome_space and self._outcome_space.outcomes:
            idx2 = min(self._bid_index + 1, self._n_outcomes - 1)
            our_bid2 = self._outcome_space.outcomes[idx2].bid
        our_utility2 = float(self.ufun(our_bid2)) if our_bid2 is not None else our_utility

        if offer_utility >= our_utility or offer_utility >= our_utility2:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
