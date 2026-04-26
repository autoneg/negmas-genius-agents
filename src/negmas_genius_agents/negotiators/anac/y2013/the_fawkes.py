"""TheFawkes from ANAC 2013."""

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

__all__ = ["TheFawkes"]


class TheFawkes(SAONegotiator):
    """
    TheFawkes from ANAC 2013 - The winning agent.

    TheFawkes won ANAC 2013 using a BOA (Bidding-Opponent modeling-Acceptance)
    framework. The key innovation is combining frequency-based opponent modeling
    with Pareto-efficient bid selection that maximizes estimated opponent utility
    while maintaining acceptable own utility.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        Original Genius class: ``agents.anac.y2013.TheFawkes.TheFawkes``

        ANAC 2013: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        Time-dependent concession using Boulware formula: threshold = max -
        (max - min) * (k + (1-k) * t^(1/e)). From candidates above threshold,
        selects the bid that maximizes estimated opponent utility (Pareto-
        efficient selection). This approach offers bids that are likely to
        be accepted by the opponent while still meeting our utility requirements.
        Early in negotiation (< 5 opponent offers), selects randomly.

    **Acceptance Strategy:**
        Implements AC_Next: accepts if offer utility >= utility of our next
        planned offer. This ensures we never reject an offer better than what
        we would propose. Also accepts if offer meets the current threshold.
        No explicit deadline handling - relies on threshold convergence.

    **Opponent Modeling:**
        Frequency-based model tracking how often each issue value appears in
        opponent offers. Estimates opponent utility for a bid by computing
        the average normalized frequency score: for each issue, the value's
        count divided by the max count for that issue, averaged across all
        issues. Higher frequency values are assumed to be preferred by opponent.

    Args:
        e: Concession exponent (default 0.2, Boulware-like)
        k: Initial utility threshold offset (default 0.0)
        early_game_offers: Number of opponent offers before using opponent model (default 5)
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
        early_game_offers: int = 5,
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
        self._early_game_offers = early_game_offers
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling - frequency counts per issue value
        self._issue_value_counts: dict[str, dict[str, int]] = {}
        self._total_opponent_offers: int = 0

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._last_own_bid: Outcome | None = None

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

        # Reset opponent model
        self._issue_value_counts = {}
        self._total_opponent_offers = 0
        self._last_own_bid = None

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update frequency-based opponent model."""
        if bid is None:
            return

        self._total_opponent_offers += 1

        # Count each issue value
        for i, value in enumerate(bid):
            issue_key = str(i)
            value_key = str(value)

            if issue_key not in self._issue_value_counts:
                self._issue_value_counts[issue_key] = {}

            if value_key not in self._issue_value_counts[issue_key]:
                self._issue_value_counts[issue_key][value_key] = 0

            self._issue_value_counts[issue_key][value_key] += 1

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent's utility for a bid based on frequency model."""
        if self._total_opponent_offers == 0 or bid is None:
            return 0.5

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            issue_key = str(i)
            value_key = str(value)

            if issue_key in self._issue_value_counts:
                counts = self._issue_value_counts[issue_key]
                if value_key in counts:
                    # Normalize by total offers for this issue
                    max_count = max(counts.values()) if counts else 1
                    total_score += counts[value_key] / max_count

        return total_score / num_issues if num_issues > 0 else 0.5

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold using time-dependent concession."""
        # Boulware-like formula: starts high, concedes slowly
        f_t = (
            self._k + (1 - self._k) * math.pow(time, 1 / self._e)
            if self._e != 0
            else self._k
        )

        # Threshold decreases from max to min over time
        threshold = self._max_utility - (self._max_utility - self._min_utility) * f_t
        return max(threshold, self._min_utility)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid that maximizes opponent utility while meeting our threshold."""
        if self._outcome_space is None:
            return None

        threshold = self._compute_threshold(time)

        # Get acceptable bids
        candidates = self._outcome_space.get_bids_above(threshold)
        if not candidates:
            if self._outcome_space.outcomes:
                return self._outcome_space.outcomes[0].bid
            return None

        # If we have opponent model, select bid that maximizes opponent utility
        if self._total_opponent_offers > self._early_game_offers:
            best_bid = None
            best_opponent_util = -1.0

            for bid_details in candidates:
                opponent_util = self._estimate_opponent_utility(bid_details.bid)
                if opponent_util > best_opponent_util:
                    best_opponent_util = opponent_util
                    best_bid = bid_details.bid

            if best_bid is not None:
                return best_bid

        # Otherwise random from candidates
        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        bid = self._select_bid(time)
        self._last_own_bid = bid
        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer using AC_Next acceptance condition."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        # Update opponent model
        self._update_opponent_model(offer)

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        time = state.relative_time

        # AC_Next: Accept if offer >= our next offer's utility
        our_bid = self._select_bid(time)
        if our_bid is not None:
            our_utility = float(self.ufun(our_bid))
            if offer_utility >= our_utility:
                return ResponseType.ACCEPT_OFFER

        # Also accept if offer meets threshold
        threshold = self._compute_threshold(time)
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
