"""
Time-dependent negotiating agents reimplemented from Genius.

This module contains Python reimplementations of the classic time-dependent
negotiation strategies from Genius, including Boulware, Conceder, and Linear agents.

Note:
    These are AI-generated reimplementations based on the original Java code
    from the Genius framework. They may not behave identically to the originals.

The time-dependent strategy uses the formula:
    f(t) = k + (1 - k) * t^(1/e)

Where:
    - t is the normalized time (0 at start, 1 at deadline)
    - e is the concession exponent
    - k is the initial concession (typically 0)

The target utility at time t is:
    p(t) = Pmin + (Pmax - Pmin) * (1 - f(t))

References:
    S. Shaheen Fatima, Michael Wooldridge, Nicholas R. Jennings
    "Optimal Negotiation Strategies for Agents with Incomplete Information"
    http://eprints.ecs.soton.ac.uk/6151/1/atal01.pdf
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

__all__ = [
    "TimeDependentAgent",
    "TimeDependentAgentBoulware",
    "TimeDependentAgentConceder",
    "TimeDependentAgentLinear",
    "TimeDependentAgentHardliner",
]


class TimeDependentAgent(SAONegotiator):
    """
    Base class for time-dependent negotiation strategies.

    This is a Python reimplementation of Genius's TimeDependentAgent.

    Time-dependent agents use a concession function based on time to determine
    their target utility. The concession rate is controlled by parameter 'e':

    - e < 1: Boulware (reluctant to concede, tough negotiator)
    - e = 1: Linear (constant concession rate)
    - e > 1: Conceder (eager to concede, cooperative negotiator)
    - e = 0: Hardliner (never concedes)

    Args:
        e: Concession exponent controlling how fast the agent concedes.
        k: Initial constant (typically 0, meaning start at max utility).
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
        e: float = 1.0,
        k: float = 0.0,
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
        self._k = k
        self._outcome_space: SortedOutcomeSpace | None = None
        self._pmax: float = 1.0
        self._pmin: float = 0.0
        self._last_received_bid: Outcome | None = None
        self._initialized = False

    @property
    def e(self) -> float:
        """
        Get the concession exponent.

        Returns:
            The e value controlling concession speed.
        """
        return self._e

    def _initialize(self) -> None:
        """Initialize the outcome space and utility bounds."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)

        # Get reservation value
        reservation = getattr(self.ufun, "reserved_value", 0.0)
        if reservation is None or reservation == float("-inf"):
            reservation = 0.0

        # Set utility bounds
        self._pmax = self._outcome_space.max_utility
        self._pmin = max(self._outcome_space.min_utility, reservation)

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

    def f(self, t: float) -> float:
        """
        Compute the concession function value at time t.

        The concession function determines how much the agent has conceded
        from its initial position. f(0) = k (initial), f(1) = 1 (full concession).

        Args:
            t: Normalized time in [0, 1].

        Returns:
            Concession value in [k, 1].
        """
        if self._e == 0:
            # Hardliner: never concede
            return self._k

        return self._k + (1 - self._k) * (t ** (1 / self._e))

    def p(self, t: float) -> float:
        """
        Compute the target utility at time t.

        This maps the concession function to actual utility values:
        - At t=0: p(0) = Pmax (start with best possible utility)
        - At t=1: p(1) = Pmin (end at reservation/minimum utility)

        Args:
            t: Normalized time in [0, 1].

        Returns:
            Target utility value.
        """
        return self._pmin + (self._pmax - self._pmin) * (1 - self.f(t))

    def _make_bid(self, t: float) -> Outcome | None:
        """
        Generate a bid for the given time.

        Args:
            t: Normalized time in [0, 1].

        Returns:
            An outcome near the target utility, or None if unavailable.
        """
        if self._outcome_space is None:
            return None

        target_utility = self.p(t)
        bid_details = self._outcome_space.get_bid_near_utility(target_utility)

        if bid_details is None:
            return None

        return bid_details.bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """
        Generate a proposal based on current time.

        Args:
            state: Current negotiation state.
            dest: Destination negotiator ID (ignored).

        Returns:
            Outcome to propose, or None.
        """
        if not self._initialized:
            self._initialize()

        t = state.relative_time
        return self._make_bid(t)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """
        Respond to an offer.

        Accepts if the offered utility is >= the utility of our planned counter-bid.

        Args:
            state: Current negotiation state.
            source: Source negotiator ID (ignored).

        Returns:
            ResponseType indicating acceptance or rejection.
        """
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        self._last_received_bid = offer

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        # Get utility of the offer
        offer_utility = float(self.ufun(offer))

        # Get our planned counter-bid
        t = state.relative_time
        my_bid = self._make_bid(t)

        if my_bid is None:
            # If we can't make a bid, accept if offer is reasonable
            return (
                ResponseType.ACCEPT_OFFER
                if offer_utility >= self._pmin
                else ResponseType.REJECT_OFFER
            )

        my_bid_utility = float(self.ufun(my_bid))

        # Accept if offer is at least as good as what we would offer
        if offer_utility >= my_bid_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


class TimeDependentAgentBoulware(TimeDependentAgent):
    """
    Boulware (tough) negotiation strategy.

    A Boulware agent is reluctant to concede. It maintains its initial offer
    for most of the negotiation and only concedes near the deadline.

    Uses e=0.2 by default, which means concessions are slow and happen mainly near the end.

    This is a reimplementation of Genius's TimeDependentAgentBoulware.

    Args:
        e: Concession exponent (default 0.2 for Boulware behavior).
        k: Initial concession constant (default 0.0).
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
        k: float = 0.0,
        preferences: BaseUtilityFunction | None = None,
        ufun: BaseUtilityFunction | None = None,
        name: str | None = None,
        parent: Controller | None = None,
        owner: Agent | None = None,
        id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            e=e,
            k=k,
            preferences=preferences,
            ufun=ufun,
            name=name,
            parent=parent,
            owner=owner,
            id=id,
            **kwargs,
        )


class TimeDependentAgentConceder(TimeDependentAgent):
    """
    Conceder (cooperative) negotiation strategy.

    A Conceder agent is eager to concede. It moves quickly toward its
    reservation value early in the negotiation.

    Uses e=2.0 by default, which means fast concessions early in the negotiation.

    This is a reimplementation of Genius's TimeDependentAgentConceder.

    Args:
        e: Concession exponent (default 2.0 for Conceder behavior).
        k: Initial concession constant (default 0.0).
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
        e: float = 2.0,
        k: float = 0.0,
        preferences: BaseUtilityFunction | None = None,
        ufun: BaseUtilityFunction | None = None,
        name: str | None = None,
        parent: Controller | None = None,
        owner: Agent | None = None,
        id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            e=e,
            k=k,
            preferences=preferences,
            ufun=ufun,
            name=name,
            parent=parent,
            owner=owner,
            id=id,
            **kwargs,
        )


class TimeDependentAgentLinear(TimeDependentAgent):
    """
    Linear negotiation strategy.

    A Linear agent concedes at a constant rate throughout the negotiation.

    Uses e=1.0 by default, which means linear concession over time.

    Args:
        e: Concession exponent (default 1.0 for Linear behavior).
        k: Initial concession constant (default 0.0).
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
        e: float = 1.0,
        k: float = 0.0,
        preferences: BaseUtilityFunction | None = None,
        ufun: BaseUtilityFunction | None = None,
        name: str | None = None,
        parent: Controller | None = None,
        owner: Agent | None = None,
        id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            e=e,
            k=k,
            preferences=preferences,
            ufun=ufun,
            name=name,
            parent=parent,
            owner=owner,
            id=id,
            **kwargs,
        )


class TimeDependentAgentHardliner(TimeDependentAgent):
    """
    Hardliner (never concede) negotiation strategy.

    A Hardliner agent always offers its best possible bid and never concedes.

    Uses e=0 by default, which means no concession at all.

    Args:
        e: Concession exponent (default 0.0 for Hardliner behavior).
        k: Initial concession constant (default 0.0).
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
        e: float = 0.0,
        k: float = 0.0,
        preferences: BaseUtilityFunction | None = None,
        ufun: BaseUtilityFunction | None = None,
        name: str | None = None,
        parent: Controller | None = None,
        owner: Agent | None = None,
        id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            e=e,
            k=k,
            preferences=preferences,
            ufun=ufun,
            name=name,
            parent=parent,
            owner=owner,
            id=id,
            **kwargs,
        )
