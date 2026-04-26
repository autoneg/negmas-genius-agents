"""CaduceusDC16 from ANAC 2017."""

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

__all__ = ["CaduceusDC16"]


class CaduceusDC16(SAONegotiator):
    """
    CaduceusDC16 from ANAC 2017 - 2nd place agent.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    CaduceusDC16 is a variant of Caduceus (the 2016 winner) adapted for 2017.
    It uses a multi-strategy ensemble approach with weighted voting.

    **Offering Strategy:**
        Two-phase approach:
        - Early phase (0-83% of time): Offers only the best bid (conservative).
        - Later phase: Uses weighted voting among 5 sub-strategies with
          different concession rates. Each strategy proposes a bid, and
          roulette wheel selection based on weights determines the final offer.

    **Acceptance Strategy:**
        During the early phase, only accepts offers near maximum utility (>99%).
        In the later phase, combines votes from all sub-strategies using
        weighted voting. Accepts if the weighted accept score exceeds the
        reject score.

    **Opponent Modeling:**
        Tracks the last received offer and its utility. This information
        is used by each sub-strategy to evaluate whether to accept. The
        ensemble approach implicitly handles opponent adaptation through
        the diversity of sub-strategies.

    Args:
        percentage_best_bid: Fraction of time to offer best bid (default 0.83).
        reservation_value: Minimum acceptable utility (default 0.75).
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
        percentage_best_bid: float = 0.83,
        reservation_value: float = 0.75,
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
        self._percentage_best_bid = percentage_best_bid
        self._reservation_value = reservation_value
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Sub-agent weights (5 strategies with different weights)
        # Higher weight = more conservative strategy has more influence
        self._strategy_weights = [500.0, 10.0, 5.0, 3.0, 1.0]
        self._normalize_weights()

        # State
        self._last_received_bid: Outcome | None = None
        self._last_received_utility: float = 0.0
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

    def _normalize_weights(self) -> None:
        """Normalize strategy weights to sum to 1."""
        total = sum(self._strategy_weights)
        if total > 0:
            self._strategy_weights = [w / total for w in self._strategy_weights]

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
        self._last_received_bid = None
        self._last_received_utility = 0.0

    def _is_best_offer_time(self, time: float) -> bool:
        """Check if we should still be offering the best bid."""
        return time < self._percentage_best_bid

    def _get_strategy_threshold(self, strategy_idx: int, time: float) -> float:
        """Get threshold for a specific strategy."""
        # Each strategy has a different concession rate
        concession_rates = [0.1, 0.15, 0.2, 0.25, 0.3]
        rate = concession_rates[min(strategy_idx, len(concession_rates) - 1)]
        threshold = self._max_utility * (1 - rate * time)
        return max(threshold, self._reservation_value)

    def _get_strategy_decision(
        self, strategy_idx: int, time: float
    ) -> tuple[bool, Outcome | None]:
        """
        Get decision from a sub-strategy.

        Returns:
            Tuple of (should_accept, bid_to_offer)
        """
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return False, None

        threshold = self._get_strategy_threshold(strategy_idx, time)

        # Check acceptance
        should_accept = self._last_received_utility >= threshold

        # Select bid
        candidates = self._outcome_space.get_bids_above(threshold)
        if candidates:
            bid = random.choice(candidates).bid
        else:
            bid = self._best_bid

        return should_accept, bid

    def _weighted_vote(self, time: float) -> tuple[bool, Outcome | None]:
        """
        Combine decisions from all sub-strategies using weighted voting.

        Returns:
            Tuple of (should_accept, bid_to_offer)
        """
        accept_score = 0.0
        offer_score = 0.0
        offering_strategies: list[tuple[int, Outcome]] = []

        for i, weight in enumerate(self._strategy_weights):
            should_accept, bid = self._get_strategy_decision(i, time)

            if should_accept:
                accept_score += weight
            else:
                offer_score += weight
                if bid is not None:
                    offering_strategies.append((i, bid))

        # Decision based on weighted votes
        if accept_score > offer_score:
            return True, None

        # Select bid using roulette wheel
        if not offering_strategies:
            return False, self._best_bid

        # Build selection weights
        selection_weights: list[float] = []
        for idx, _ in offering_strategies:
            selection_weights.append(self._strategy_weights[idx])

        total = sum(selection_weights)
        if total > 0:
            selection_weights = [w / total for w in selection_weights]

        # Roulette selection
        r = random.random()
        cumulative = 0.0
        for i, (_, bid) in enumerate(offering_strategies):
            cumulative += selection_weights[i]
            if r <= cumulative:
                return False, bid

        return False, offering_strategies[-1][1]

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        # Early game: offer best bid
        if self._is_best_offer_time(time):
            return self._best_bid

        # Use weighted voting
        _, bid = self._weighted_vote(time)
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

        # Track opponent's offer
        self._last_received_bid = offer
        self._last_received_utility = float(self.ufun(offer))

        time = state.relative_time

        # During best-offer phase, only accept if offer is very good
        if self._is_best_offer_time(time):
            if self._last_received_utility >= self._max_utility * 0.99:
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

        # Use weighted voting
        should_accept, _ = self._weighted_vote(time)

        if should_accept:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
