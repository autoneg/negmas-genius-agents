"""AgentH from ANAC 2015."""

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

__all__ = ["AgentH"]


class AgentH(SAONegotiator):
    """
    AgentH negotiation agent from ANAC 2015.

    AgentH uses a hybrid approach combining adaptive concession with
    frequency-based opponent modeling for strategic bid selection.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.agenth.AgentH

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Time-dependent concession using Boulware-like formula
        - Adaptive concession rate: becomes firmer if opponent is conceding
          (e * 0.8), more flexible if opponent is tough (e * 1.2)
        - Hybrid bid selection balancing own utility (70%) and estimated
          opponent utility (30%)
        - Selects from candidates above computed threshold

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - AC_Next: Accepts if offer utility >= utility of our next hybrid bid

    **Opponent Modeling:**
        - Frequency-based model tracking issue-value occurrences
        - Tracks best opponent utility for comparison
        - Analyzes recent vs older offers to detect opponent concession
        - Estimates opponent preference based on normalized value frequencies

    Args:
        e: Concession exponent controlling concession speed (default 0.2)
        recent_window: Number of recent opponent bids used for concession detection (default 5)
        older_window: Number of older opponent bids used for concession detection (default 10)
        firm_multiplier: Multiplier applied to e when opponent is conceding (default 0.8)
        flexible_multiplier: Multiplier applied to e when opponent is tough (default 1.2)
        concession_factor: Fraction of utility range conceded over time (default 0.4)
        min_utility_margin: Margin above min utility used as acceptance floor (default 0.1)
        fallback_threshold_ratio: Ratio used when lowering threshold if no candidates (default 0.9)
        hybrid_search_count: Number of top candidates searched for hybrid bid (default 30)
        own_utility_weight: Weight of own utility in hybrid scoring (default 0.7)
        opponent_utility_weight: Weight of opponent utility in hybrid scoring (default 0.3)
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
        recent_window: int = 5,
        older_window: int = 10,
        firm_multiplier: float = 0.8,
        flexible_multiplier: float = 1.2,
        concession_factor: float = 0.4,
        min_utility_margin: float = 0.1,
        fallback_threshold_ratio: float = 0.9,
        hybrid_search_count: int = 30,
        own_utility_weight: float = 0.7,
        opponent_utility_weight: float = 0.3,
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
        self._recent_window = recent_window
        self._older_window = older_window
        self._firm_multiplier = firm_multiplier
        self._flexible_multiplier = flexible_multiplier
        self._concession_factor = concession_factor
        self._min_utility_margin = min_utility_margin
        self._fallback_threshold_ratio = fallback_threshold_ratio
        self._hybrid_search_count = hybrid_search_count
        self._own_utility_weight = own_utility_weight
        self._opponent_utility_weight = opponent_utility_weight
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0

        # Opponent modeling
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._value_freq: dict[int, dict] = {}
        self._best_opponent_utility: float = 0.0

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
        self._value_freq = {}
        self._best_opponent_utility = 0.0

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update frequency-based opponent model."""
        self._opponent_bids.append((bid, utility))
        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        for i, value in enumerate(bid):
            if i not in self._value_freq:
                self._value_freq[i] = {}
            self._value_freq[i][value] = self._value_freq[i].get(value, 0) + 1

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent's preference for a bid."""
        if not self._value_freq:
            return 0.5

        score = 0.0
        for i, value in enumerate(bid):
            if i in self._value_freq:
                freq = self._value_freq[i].get(value, 0)
                max_freq = (
                    max(self._value_freq[i].values()) if self._value_freq[i] else 1
                )
                score += freq / max_freq if max_freq > 0 else 0

        return score / len(bid) if len(bid) > 0 else 0.5

    def _compute_threshold(self, time: float) -> float:
        """Compute adaptive concession threshold."""
        # Adaptive e based on opponent concession
        if len(self._opponent_bids) > self._recent_window:
            recent = [u for _, u in self._opponent_bids[-self._recent_window :]]
            older = (
                [
                    u
                    for _, u in self._opponent_bids[
                        -self._older_window : -self._recent_window
                    ]
                ]
                if len(self._opponent_bids) > self._older_window
                else recent
            )
            if sum(recent) / len(recent) > sum(older) / len(older):
                # Opponent conceding, be firmer
                e = self._e * self._firm_multiplier
            else:
                e = self._e * self._flexible_multiplier
        else:
            e = self._e

        f_t = math.pow(time, 1 / e) if e > 0 else time
        target = self._max_utility - (
            self._max_utility - self._min_utility
        ) * self._concession_factor * f_t
        return max(target, self._min_utility + self._min_utility_margin)

    def _select_hybrid_bid(self, time: float) -> Outcome | None:
        """Select bid using hybrid approach."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(
                threshold * self._fallback_threshold_ratio
            )

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Hybrid: balance own utility and opponent preference
        if self._value_freq:
            best_bid = None
            best_score = -1.0

            for bd in candidates[: self._hybrid_search_count]:
                own_score = bd.utility
                opp_score = self._estimate_opponent_utility(bd.bid)
                # Hybrid score: weighted combination
                hybrid_score = (
                    own_score * self._own_utility_weight
                    + opp_score * self._opponent_utility_weight
                )
                if hybrid_score > best_score:
                    best_score = hybrid_score
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal using hybrid strategy."""
        if not self._initialized:
            self._initialize()

        return self._select_hybrid_bid(state.relative_time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond using AC_Next condition."""
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

        # Accept if meets threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # AC_Next: accept if >= our next offer
        our_bid = self._select_hybrid_bid(time)
        if our_bid is not None:
            our_utility = float(self.ufun(our_bid))
            if offer_utility >= our_utility:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER