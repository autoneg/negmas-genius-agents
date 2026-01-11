"""ParsAgent from ANAC 2015."""

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

__all__ = ["ParsAgent"]


class ParsAgent(SAONegotiator):
    """
    ParsAgent negotiation agent from ANAC 2015 - 2nd place agent.

    ParsAgent is designed for multilateral negotiations with frequency-based
    opponent modeling and mutual issue exploitation.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.ParsAgent.ParsAgent

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Standard Boulware-like concession (e=0.15):
          target = max_util - (max_util - min_threshold) * t^(1/e)
        - Minimum threshold of 0.7 utility
        - Three-tier bid selection:
          1. First tries previously accepted bids above target
          2. Then tries mutual preference bids (using opponent model)
          3. Falls back to random selection from candidates
        - Mutual bid selection scores candidates by opponent value frequencies

    **Acceptance Strategy:**
        - Accepts if offer utility >= max(target, min_threshold of 0.7)

    **Opponent Modeling:**
        - Frequency-based model tracking issue-value occurrences
        - Builds preference profile from opponent bid history
        - Uses frequencies to find mutually beneficial bids
        - Tracks accepted bids for future reference

    Args:
        e: Concession exponent (default 0.15 without discount, 0.20 with)
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
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_threshold: float = 0.7

        # Opponent modeling
        self._opponent_value_freq: dict[int, dict] = {}  # issue -> {value: count}
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._accepted_bids: list[tuple[Outcome, float]] = []  # Bids opponent accepted

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

        self._opponent_value_freq = {}
        self._opponent_bids = []
        self._accepted_bids = []

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update frequency-based opponent model."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append((bid, utility))

        for i, value in enumerate(bid):
            if i not in self._opponent_value_freq:
                self._opponent_value_freq[i] = {}
            self._opponent_value_freq[i][value] = (
                self._opponent_value_freq[i].get(value, 0) + 1
            )

    def _get_target_utility(self, time: float) -> float:
        """Calculate target utility using time-dependent formula."""
        f_t = math.pow(time, 1 / self._e) if self._e > 0 else 0
        target = self._max_utility - (self._max_utility - self._min_threshold) * f_t
        return max(target, self._min_threshold)

    def _get_mutual_bid(self) -> Outcome | None:
        """Try to find a bid using mutual issue values."""
        if not self._opponent_value_freq or self._outcome_space is None:
            return None

        target = self._get_target_utility(0.5)  # Use mid-point target

        # For each candidate bid, score by opponent preference
        best_bid = None
        best_score = -1.0

        for bd in self._outcome_space.outcomes:
            if bd.utility < target:
                break

            # Score based on opponent value frequencies
            score = 0.0
            for i, value in enumerate(bd.bid):
                if i in self._opponent_value_freq:
                    freq = self._opponent_value_freq[i].get(value, 0)
                    max_freq = (
                        max(self._opponent_value_freq[i].values())
                        if self._opponent_value_freq[i]
                        else 1
                    )
                    score += freq / max_freq if max_freq > 0 else 0

            if score > best_score:
                best_score = score
                best_bid = bd.bid

        return best_bid

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid to offer."""
        if self._outcome_space is None:
            return None

        target = self._get_target_utility(time)

        # First try: check accepted bids
        for bid, util in self._accepted_bids:
            if util >= target:
                return bid

        # Second try: mutual preference bid
        if len(self._opponent_bids) > 5:
            mutual_bid = self._get_mutual_bid()
            if mutual_bid is not None and self.ufun is not None:
                if float(self.ufun(mutual_bid)) >= target:
                    return mutual_bid

        # Third try: standard bid selection
        candidates = self._outcome_space.get_bids_above(target)
        if candidates:
            return random.choice(candidates).bid

        if self._outcome_space.outcomes:
            return self._outcome_space.outcomes[0].bid
        return None

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

        self._update_opponent_model(offer)
        time = state.relative_time
        offer_utility = float(self.ufun(offer))
        target = self._get_target_utility(time)

        if offer_utility >= max(target, self._min_threshold):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
