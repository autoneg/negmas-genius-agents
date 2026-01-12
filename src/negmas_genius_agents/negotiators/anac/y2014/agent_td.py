"""AgentTD from ANAC 2014."""

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

__all__ = ["AgentTD"]


class AgentTD(SAONegotiator):
    """
    AgentTD (Time-Dependent Agent) from ANAC 2014.

    AgentTD implements a classic time-dependent negotiation strategy based on
    the foundational work by Faratin et al. The agent's behavior is primarily
    driven by time pressure, with configurable concession curves.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.AgentTD.AgentTD

    **Offering Strategy:**
        Uses the generalized time-dependent function:
        target(t) = min_util + (max_util - min_util) * (1 - t^(1/e))

        The exponent 'e' controls the concession behavior:
        - e < 1: Boulware behavior (concedes slowly, then rapidly at deadline)
        - e = 1: Linear concession over time
        - e > 1: Conceder behavior (concedes quickly early, slows later)

        Bid selection includes exploration early (random from candidates)
        and exploitation later (focus on best candidates).

    **Acceptance Strategy:**
        Time-pressure-aware acceptance with three conditions:
        1. Accept if offer utility meets or exceeds target utility
        2. Accept if offer utility is at least as good as next planned bid
        3. Accept near deadline (t > deadline_acceptance_time) if utility exceeds minimum threshold
        This ensures deals are reached even under aggressive Boulware settings.

    **Opponent Modeling:**
        Minimal opponent modeling for computational efficiency. Tracks only
        the best opponent bid received (highest utility for self) for
        potential use in late-game decision making. The strategy relies
        primarily on time-based concession rather than opponent adaptation.

    Args:
        e: Concession exponent (default 0.2). Values < 1 give Boulware
           behavior, > 1 give Conceder behavior, = 1 gives linear.
        min_utility: Minimum acceptable utility threshold (default 0.6).
        exploration_end_time: Time threshold ending exploration phase (default 0.5).
        deadline_acceptance_time: Time threshold for near-deadline acceptance (default 0.99).
        exploration_divisor: Divisor for selecting from top candidates during exploration (default 2).
        exploitation_divisor: Divisor for selecting from top candidates during exploitation (default 4).
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
        min_utility: float = 0.6,
        exploration_end_time: float = 0.5,
        deadline_acceptance_time: float = 0.99,
        exploration_divisor: int = 2,
        exploitation_divisor: int = 4,
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
        self._min_utility = min_utility
        self._exploration_end_time = exploration_end_time
        self._deadline_acceptance_time = deadline_acceptance_time
        self._exploration_divisor = exploration_divisor
        self._exploitation_divisor = exploitation_divisor
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._best_opponent_utility: float = 0.0
        self._best_opponent_bid: Outcome | None = None

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._best_opponent_utility = 0.0
        self._best_opponent_bid = None

    def _compute_target_utility(self, time: float) -> float:
        """Compute target utility using time-dependent function."""
        # Generalized time-dependent function
        # f(t) = min + (max - min) * (1 - t^(1/e))
        # e < 1: Boulware (concedes slowly, then fast at end)
        # e > 1: Conceder (concedes fast initially)
        # e = 1: Linear

        if self._e == 0:
            # Pure hardball
            return self._max_utility

        factor = 1.0 - time ** (1.0 / self._e)
        target = self._min_utility + (self._max_utility - self._min_utility) * factor

        return max(self._min_utility, min(self._max_utility, target))

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid based on current target utility."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._compute_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Simple selection: random from top candidates
        if time < self._exploration_end_time:
            # Early: more exploration
            return random.choice(
                candidates[: max(1, len(candidates) // self._exploration_divisor)]
            ).bid
        else:
            # Late: focus on best
            return random.choice(
                candidates[: max(1, len(candidates) // self._exploitation_divisor)]
            ).bid

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Simple opponent tracking."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        return self._select_bid(time)

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
        self._update_opponent_model(offer)

        target = self._compute_target_utility(time)

        # Accept if above target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than our next offer
        next_bid = self._select_bid(time)
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid))
            if offer_utility >= next_utility:
                return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if reasonable
        if time > self._deadline_acceptance_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
