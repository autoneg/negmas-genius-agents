"""AgentBuyog from ANAC 2015."""

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

__all__ = ["AgentBuyog"]


class AgentBuyog(SAONegotiator):
    """
    AgentBuyog negotiation agent from ANAC 2015.

    AgentBuyog uses a strategic bidding approach with Nash point estimation
    and opponent modeling to find mutually beneficial outcomes.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.agentBuyogV2.AgentBuyogMain

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Phase-based time-dependent concession with variable rate
        - Early phase (t<0.2): Stays firm at 95% of max utility
        - Middle phase (0.2<t<0.7): Gradual Boulware-like concession toward
          estimated Nash point
        - Late phase (t>0.7): Accelerated concession toward minimum acceptable
        - Selects bids balancing own utility (60%) and estimated opponent
          preference (40%)

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - AC_Next: Accepts if offer utility >= utility of our next planned bid
        - End-game (t>0.95): Accepts if offer >= estimated Nash utility - 0.1

    **Opponent Modeling:**
        - Frequency-based model tracking issue-value occurrences
        - Estimates Nash point based on best and average opponent utilities
        - Updates opponent preference estimate for bid selection

    Args:
        e: Concession exponent controlling concession speed (default 0.15)
        early_time_threshold: Time threshold for early phase (default 0.2)
        main_time_threshold: Time threshold for main/late phase transition (default 0.7)
        deadline_time_threshold: Time after which end-game acceptance triggers (default 0.95)
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
        e: float = 0.15,
        early_time_threshold: float = 0.2,
        main_time_threshold: float = 0.7,
        deadline_time_threshold: float = 0.95,
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
        self._early_time_threshold = early_time_threshold
        self._main_time_threshold = main_time_threshold
        self._deadline_time_threshold = deadline_time_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0

        # Opponent modeling
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._value_frequencies: dict[int, dict] = {}
        self._best_opponent_utility: float = 0.0
        self._estimated_nash_utility: float = 0.7

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = self._outcome_space.min_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._value_frequencies = {}
        self._best_opponent_utility = 0.0
        self._estimated_nash_utility = 0.7

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update opponent model with frequency analysis."""
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        for i, value in enumerate(bid):
            if i not in self._value_frequencies:
                self._value_frequencies[i] = {}
            self._value_frequencies[i][value] = (
                self._value_frequencies[i].get(value, 0) + 1
            )

        # Estimate Nash point based on best mutual offer
        if len(self._opponent_bids) > 5:
            avg_util = sum(u for _, u in self._opponent_bids[-10:]) / min(
                10, len(self._opponent_bids)
            )
            self._estimated_nash_utility = (avg_util + self._best_opponent_utility) / 2

    def _estimate_opponent_preference(self, bid: Outcome) -> float:
        """Estimate how much opponent prefers this bid."""
        if not self._value_frequencies:
            return 0.5

        score = 0.0
        for i, value in enumerate(bid):
            if i in self._value_frequencies:
                freq = self._value_frequencies[i].get(value, 0)
                max_freq = (
                    max(self._value_frequencies[i].values())
                    if self._value_frequencies[i]
                    else 1
                )
                score += freq / max_freq if max_freq > 0 else 0

        return score / len(bid) if len(bid) > 0 else 0.5

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold with variable concession."""
        # Phase-based concession
        if time < self._early_time_threshold:
            # Early: stay firm
            return self._max_utility * 0.95
        elif time < self._main_time_threshold:
            # Middle: gradual concession toward Nash
            progress = (time - self._early_time_threshold) / (
                self._main_time_threshold - self._early_time_threshold
            )
            f_t = math.pow(progress, 1 / self._e)
            target_min = max(self._estimated_nash_utility, 0.6)
            return (
                self._max_utility * 0.95
                - (self._max_utility * 0.95 - target_min) * f_t * 0.5
            )
        else:
            # Late: concede toward Nash point
            progress = (time - self._main_time_threshold) / (
                1.0 - self._main_time_threshold
            )
            target_min = max(self._estimated_nash_utility - 0.1, 0.5)
            base = (
                self._max_utility * 0.95
                - (self._max_utility * 0.95 - self._estimated_nash_utility) * 0.5
            )
            return base - (base - target_min) * progress

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid that balances own utility and opponent preference."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # If we have opponent model, find best mutual benefit
        if self._value_frequencies:
            best_bid = None
            best_score = -1.0

            for bd in candidates[:50]:  # Limit search
                opp_pref = self._estimate_opponent_preference(bd.bid)
                # Combined score: our utility + opponent preference
                score = bd.utility * 0.6 + opp_pref * 0.4
                if score > best_score:
                    best_score = score
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        return self._select_bid(state.relative_time)

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

        self._update_opponent_model(offer, offer_utility)

        threshold = self._compute_threshold(time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # AC_Next: accept if >= our next offer
        our_bid = self._select_bid(time)
        if our_bid is not None:
            our_utility = float(self.ufun(our_bid))
            if offer_utility >= our_utility:
                return ResponseType.ACCEPT_OFFER

        # End-game: accept if better than Nash estimate
        if (
            time > self._deadline_time_threshold
            and offer_utility >= self._estimated_nash_utility - 0.1
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
