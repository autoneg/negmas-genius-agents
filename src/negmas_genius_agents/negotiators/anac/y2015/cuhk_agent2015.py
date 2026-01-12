"""CUHKAgent2015 from ANAC 2015."""

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

__all__ = ["CUHKAgent2015"]


class CUHKAgent2015(SAONegotiator):
    """
    CUHKAgent2015 negotiation agent from ANAC 2015.

    CUHKAgent2015 from the Chinese University of Hong Kong uses sophisticated
    opponent modeling with Nash bargaining solution approximation.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.cuhkagent2015.CUHKAgent2015

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Three-phase quadratic concession:
          * Initial (t<0.2): High threshold at 95% of max utility
          * Main (0.2<t<0.8): Quadratic concession (t^2) toward estimated
            Nash point
          * End (t>0.8): 60% of remaining concession toward minimum
        - Nash-optimizing bid selection: maximizes product of own utility
          and estimated opponent preference
        - Searches top 50 candidates for best Nash product

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - AC_Next: Accepts if offer utility >= utility of our next bid
        - AC_Nash (t>0.9): Accepts if offer >= estimated Nash utility
        - End-game (t>0.98): Accepts any offer >= minimum utility

    **Opponent Modeling:**
        - Comprehensive frequency-based model tracking issue-values
        - Maintains running average and best opponent utility
        - Estimates Nash point as average of best and mean opponent utilities
        - Uses normalized frequencies to estimate opponent preference
          for bid selection

    Args:
        min_utility: Minimum acceptable utility (default 0.55)
        initial_time_threshold: Time threshold for initial phase (default 0.2)
        main_time_threshold: Time threshold for main/end phase transition (default 0.8)
        nash_accept_time_threshold: Time after which AC_Nash applies (default 0.9)
        deadline_time_threshold: Time after which end-game acceptance triggers (default 0.98)
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
        min_utility: float = 0.55,
        initial_time_threshold: float = 0.2,
        main_time_threshold: float = 0.8,
        nash_accept_time_threshold: float = 0.9,
        deadline_time_threshold: float = 0.98,
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
        self._min_utility = min_utility
        self._initial_time_threshold = initial_time_threshold
        self._main_time_threshold = main_time_threshold
        self._nash_accept_time_threshold = nash_accept_time_threshold
        self._deadline_time_threshold = deadline_time_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._estimated_nash: float = 0.7

        # Opponent modeling
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._value_frequencies: dict[int, dict] = {}
        self._best_opponent_utility: float = 0.0
        self._avg_opponent_utility: float = 0.0

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

        self._opponent_bids = []
        self._value_frequencies = {}
        self._best_opponent_utility = 0.0
        self._avg_opponent_utility = 0.0
        self._estimated_nash = 0.7

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update comprehensive opponent model."""
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Update average
        total = sum(u for _, u in self._opponent_bids)
        self._avg_opponent_utility = total / len(self._opponent_bids)

        # Update Nash estimate
        if len(self._opponent_bids) >= 3:
            self._estimated_nash = (
                self._best_opponent_utility + self._avg_opponent_utility
            ) / 2

        # Track value frequencies for opponent preference estimation
        for i, value in enumerate(bid):
            if i not in self._value_frequencies:
                self._value_frequencies[i] = {}
            self._value_frequencies[i][value] = (
                self._value_frequencies[i].get(value, 0) + 1
            )

    def _estimate_opponent_preference(self, bid: Outcome) -> float:
        """Estimate opponent's preference for a bid."""
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
        """Compute threshold with CUHK strategy."""
        if time < self._initial_time_threshold:
            # Initial phase: high threshold
            return self._max_utility * 0.95
        elif time < self._main_time_threshold:
            # Main phase: concede toward Nash
            progress = (time - self._initial_time_threshold) / (
                self._main_time_threshold - self._initial_time_threshold
            )
            f_t = math.pow(progress, 2)  # Quadratic concession
            target = (
                self._max_utility * 0.95
                - (self._max_utility * 0.95 - self._estimated_nash) * f_t
            )
            return max(target, self._min_utility)
        else:
            # End phase: more aggressive concession
            progress = (time - self._main_time_threshold) / (
                1.0 - self._main_time_threshold
            )
            target = (
                self._estimated_nash
                - (self._estimated_nash - self._min_utility) * progress * 0.6
            )
            return max(target, self._min_utility)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid with Nash optimization."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Try to find Nash-like bid
        if self._value_frequencies and time > self._initial_time_threshold:
            best_bid = None
            best_nash_score = -1.0

            for bd in candidates[:50]:
                opp_pref = self._estimate_opponent_preference(bd.bid)
                # Nash product approximation
                nash_score = bd.utility * opp_pref
                if nash_score > best_nash_score:
                    best_nash_score = nash_score
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

        # AC_Time: accept if above threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # AC_Next: accept if >= our next offer
        our_bid = self._select_bid(time)
        if our_bid is not None:
            our_utility = float(self.ufun(our_bid))
            if offer_utility >= our_utility:
                return ResponseType.ACCEPT_OFFER

        # AC_Nash: accept if above Nash estimate
        if (
            time > self._nash_accept_time_threshold
            and offer_utility >= self._estimated_nash
        ):
            return ResponseType.ACCEPT_OFFER

        # End-game
        if time > self._deadline_time_threshold and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
