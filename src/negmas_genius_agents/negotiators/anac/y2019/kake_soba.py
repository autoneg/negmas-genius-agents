"""KakeSoba from ANAC 2019."""

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

__all__ = ["KakeSoba"]


class KakeSoba(SAONegotiator):
    """
    KakeSoba from ANAC 2019 - 2nd place agent.

    KakeSoba achieved 2nd place in ANAC 2019 using a unique strategy
    that maintains a fixed high utility threshold while diversifying
    offers to explore opponent preferences. This conservative approach
    avoids poor deals while still enabling agreement through diversity.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    **Offering Strategy:**
        - Fixed utility threshold at 0.85 (no time-based concession!)
        - Tracks frequency of issue values across all offers made
        - For each candidate bid, computes "frequency imbalance" score:
          imbalance = sum(frequencies) / (num_issues * total_offers)
        - Selects bid with LOWEST imbalance (most diversifying)
        - Goal: systematically explore different value combinations
        - First offer is always the best available bid

    **Acceptance Strategy:**
        - Accepts offers meeting the fixed 0.85 threshold
        - Very near deadline (t >= 0.99): Accepts 95% of threshold (0.8075)
        - Extremely conservative - may fail negotiations if opponent
          cannot or will not meet the high threshold

    **Opponent Modeling:**
        - Self-modeling rather than opponent modeling
        - Tracks own offer frequencies to ensure diversity
        - No explicit opponent preference estimation
        - Diversity implicitly explores what opponent might accept
        - Simple but effective exploration strategy

    Args:
        min_utility: Fixed minimum utility threshold (default 0.85)
        deadline_threshold: Time threshold for deadline acceptance (default 0.99)
        deadline_ratio: Ratio of min_utility for deadline acceptance (default 0.95)
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
        min_utility: float = 0.85,
        deadline_threshold: float = 0.99,
        deadline_ratio: float = 0.95,
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
        self._deadline_threshold = deadline_threshold
        self._deadline_ratio = deadline_ratio
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Bid diversification tracking
        self._value_frequencies: dict[int, dict[str, int]] = {}
        self._total_offers_made: int = 0

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._value_frequencies = {}
        self._total_offers_made = 0

    def _record_offer(self, bid: Outcome) -> None:
        """Record bid value frequencies for diversification."""
        if bid is None:
            return

        self._total_offers_made += 1

        for i, value in enumerate(bid):
            if i not in self._value_frequencies:
                self._value_frequencies[i] = {}

            value_str = str(value)
            if value_str not in self._value_frequencies[i]:
                self._value_frequencies[i][value_str] = 0
            self._value_frequencies[i][value_str] += 1

    def _compute_frequency_imbalance(self, bid: Outcome) -> float:
        """
        Compute frequency imbalance score for a bid.

        Lower score = more balanced (better for diversification).
        """
        if bid is None or self._total_offers_made == 0:
            return 0.0

        total_freq = 0.0
        for i, value in enumerate(bid):
            value_str = str(value)
            if i in self._value_frequencies and value_str in self._value_frequencies[i]:
                freq = self._value_frequencies[i][value_str]
                total_freq += freq

        # Normalize by number of issues and total offers
        num_issues = len(bid)
        if num_issues > 0:
            return total_freq / (num_issues * self._total_offers_made)
        return 0.0

    def _select_diversified_bid(self) -> Outcome | None:
        """Select a bid that diversifies our offer history."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        # Get candidates above threshold
        candidates = self._outcome_space.get_bids_above(self._min_utility)

        if not candidates:
            return self._best_bid

        if len(candidates) == 1:
            bid = candidates[0].bid
            self._record_offer(bid)
            return bid

        # Select bid with lowest frequency imbalance (most diversifying)
        best_bid = candidates[0].bid
        best_imbalance = float("inf")

        for bd in candidates:
            imbalance = self._compute_frequency_imbalance(bd.bid)
            if imbalance < best_imbalance:
                best_imbalance = imbalance
                best_bid = bd.bid

        self._record_offer(best_bid)
        return best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        # First offer: best bid
        if self._total_offers_made == 0:
            self._record_offer(self._best_bid)
            return self._best_bid

        return self._select_diversified_bid()

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))

        # Accept if above fixed threshold
        if offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, slightly lower threshold
        time = state.relative_time
        if (
            time >= self._deadline_threshold
            and offer_utility >= self._min_utility * self._deadline_ratio
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
