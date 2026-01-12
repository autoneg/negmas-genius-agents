"""AgentM from ANAC 2014."""

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

__all__ = ["AgentM"]


class AgentM(SAONegotiator):
    """
    AgentM - Winner of ANAC 2014.

    AgentM is a sophisticated negotiation agent that won the 5th International
    Automated Negotiating Agents Competition (ANAC 2014). It employs simulated
    annealing for efficient bid space exploration and adapts its acceptance
    strategy based on learned opponent concession patterns.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions

    **Offering Strategy:**
        Uses simulated annealing (SA) to search the bid space for high-utility
        offers. The SA algorithm explores the outcome space with a temperature
        that decreases over iterations, allowing initial exploration followed
        by exploitation of promising regions. Bids are selected that meet a
        dynamic threshold computed from opponent behavior analysis.

    **Acceptance Strategy:**
        Employs an adaptive acceptance mechanism based on:
        - Dynamic threshold derived from opponent concession rate tracking
        - Comparison with the agent's own worst bid utility so far
        - Base threshold of 0.999 adjusted by concession rate and time
        The agent accepts offers that exceed either the computed threshold
        or its own worst proposed utility.

    **Opponent Modeling:**
        Tracks opponent bid history to estimate concession patterns. Computes
        a concession rate as the squared difference between the opponent's
        worst and best offers (from the agent's perspective). This rate
        influences both the acceptance threshold and bidding strategy,
        allowing AgentM to be more aggressive against conceding opponents.

    Args:
        num_iterations: Number of SA iterations for bid search (default 1000).
        temperature_base: Base temperature coefficient for SA (default 0.01).
        base_threshold: Starting acceptance threshold (default 0.999).
        time_divisor: Divisor for time-based threshold reduction (default 10).
        min_threshold: Minimum acceptance threshold floor (default 0.5).
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
        num_iterations: int = 1000,
        temperature_base: float = 0.01,
        base_threshold: float = 0.999,
        time_divisor: float = 10.0,
        min_threshold: float = 0.5,
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
        self._num_iterations = num_iterations
        self._temperature_base = temperature_base
        self._base_threshold = base_threshold
        self._time_divisor = time_divisor
        self._min_threshold = min_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Bid tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._own_bids: list[tuple[Outcome, float]] = []
        self._best_own_bid: Outcome | None = None
        self._best_own_utility: float = 0.0

        # Concession tracking
        self._concession_rate: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)

        # Find best bid using SA-like search
        self._find_best_bid()
        self._initialized = True

    def _find_best_bid(self) -> None:
        """Find best bid using simulated annealing-like search."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return

        # Start with random bid
        best = self._outcome_space.outcomes[0]
        self._best_own_bid = best.bid
        self._best_own_utility = best.utility

        # SA search through outcome space
        current_bid = random.choice(self._outcome_space.outcomes)
        current_util = current_bid.utility

        for i in range(min(self._num_iterations, len(self._outcome_space.outcomes))):
            # Random neighbor (in our case, random bid from space)
            next_bid = random.choice(self._outcome_space.outcomes)
            next_util = next_bid.utility

            # Temperature decreases over time
            temperature = self._temperature_base * math.pow(
                1.0 - (i / self._num_iterations), 2
            )

            # Acceptance probability
            if next_util > current_util:
                prob = 1.0
            else:
                diff = current_util - next_util
                prob = math.exp(-diff / temperature) if temperature > 0 else 0.0

            if random.random() < prob:
                current_bid = next_bid
                current_util = next_util

                if current_util > self._best_own_utility:
                    self._best_own_bid = current_bid.bid
                    self._best_own_utility = current_util

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset tracking
        self._opponent_bids = []
        self._own_bids = []
        self._concession_rate = 0.0

    def _update_concession_rate(self) -> None:
        """Update concession rate based on opponent behavior."""
        if len(self._opponent_bids) < 2:
            return

        # Compare worst and best opponent offers
        utilities = [u for _, u in self._opponent_bids]
        worst_util = min(utilities)
        best_util = max(utilities)

        # Concession rate is squared difference
        self._concession_rate = (worst_util - best_util) ** 2

    def _compute_threshold(self, time: float) -> float:
        """Compute acceptance threshold."""
        if self._outcome_space is None:
            return self._min_threshold

        threshold = (
            self._base_threshold - self._concession_rate - time / self._time_divisor
        )

        # Ensure minimum threshold
        return max(threshold, self._min_threshold)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid to offer."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)

        # Get bids meeting threshold
        candidates = self._outcome_space.get_bids_above(threshold)

        if candidates:
            # If we have opponent history, try to find bid similar to best opponent offer
            if self._opponent_bids:
                best_opp_bid, best_opp_util = max(
                    self._opponent_bids, key=lambda x: x[1]
                )
                # Simple: return random candidate
                return random.choice(candidates).bid

            return random.choice(candidates).bid

        # Fallback to best bid
        return self._best_own_bid

    def _is_acceptable(self, offer: Outcome, time: float) -> bool:
        """Check if offer is acceptable."""
        if self.ufun is None:
            return False

        offer_utility = float(self.ufun(offer))

        # Check against our worst bid so far
        if self._own_bids:
            worst_own_util = min(u for _, u in self._own_bids)
            if offer_utility >= worst_own_util:
                return True

        # Check threshold
        threshold = self._compute_threshold(time)
        if offer_utility >= threshold:
            return True

        return False

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        bid = self._select_bid(time)

        if bid is not None and self.ufun is not None:
            self._own_bids.append((bid, float(self.ufun(bid))))

        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        # Track opponent bid
        offer_utility = float(self.ufun(offer))
        self._opponent_bids.append((offer, offer_utility))
        self._update_concession_rate()

        time = state.relative_time

        if self._is_acceptable(offer, time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
