"""Gin from ANAC 2017."""

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

__all__ = ["Gin"]


class Gin(SAONegotiator):
    """
    Gin from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This is a reimplementation of Gin from ANAC 2017.
    Original: agents.anac.y2017.gin.Gin

    Gin uses a smooth, gradual concession strategy inspired by
    the smoothness of gin (the drink).

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/nego/node/7

    **Offering Strategy:**
        Uses polynomial concession curve (time^smoothness) for smooth,
        predictable concession. Maintains bid history (last 10 bids) to
        promote diversity and avoid repetition. Selects from a narrow
        utility range around the threshold for precise targeting.

    **Acceptance Strategy:**
        Accepts offers above the polynomial threshold. Late-game (>90%)
        uses extrapolation to estimate expected future opponent offers
        and accepts if current offer is within 0.02 of that estimate.
        Very late (>98%) accepts any offer above minimum utility.

    **Opponent Modeling:**
        Tracks opponent utility history and uses linear extrapolation
        on recent offers (last 5) to estimate future offer quality.
        This prediction informs late-game acceptance decisions.

    Args:
        min_utility: Minimum acceptable utility (default 0.65).
        smoothness: Controls concession curve smoothness (default 2.0).
        late_game_threshold: Time threshold for late game acceleration (default 0.9).
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
        min_utility: float = 0.65,
        smoothness: float = 2.0,
        late_game_threshold: float = 0.9,
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
        self._smoothness = smoothness
        self._late_game_threshold = late_game_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

        # Bid history for diversity
        self._recent_bids: list[Outcome] = []
        self._max_history = 10

        # Opponent modeling
        self._opponent_utilities: list[float] = []
        self._best_opponent_utility: float = 0.0

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
        self._recent_bids = []
        self._opponent_utilities = []
        self._best_opponent_utility = 0.0

    def _update_opponent_model(self, offer: Outcome) -> None:
        """Track opponent behavior."""
        if self.ufun is None:
            return

        offer_utility = float(self.ufun(offer))
        self._opponent_utilities.append(offer_utility)
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)

    def _estimate_opponent_future_utility(self, time: float) -> float:
        """Estimate what utility opponent might offer in the future."""
        if len(self._opponent_utilities) < 2:
            return self._best_opponent_utility

        # Linear extrapolation based on recent trend
        recent = self._opponent_utilities[-5:]
        if len(recent) >= 2:
            avg_utility = sum(recent) / len(recent)
            trend = recent[-1] - recent[0]

            # Estimate future utility
            remaining_time = 1.0 - time
            future_estimate = avg_utility + trend * remaining_time

            return min(max(future_estimate, 0.0), 1.0)

        return self._best_opponent_utility

    def _calculate_threshold(self, time: float) -> float:
        """Calculate target utility using smooth polynomial concession."""
        # Smooth polynomial concession
        concession_rate = math.pow(time, self._smoothness)
        utility_range = self._max_utility - self._min_utility

        threshold = self._max_utility - concession_rate * utility_range

        # Late game adjustment
        if time > self._late_game_threshold:
            late_pressure = (time - self._late_game_threshold) / (
                1.0 - self._late_game_threshold
            )
            threshold -= 0.1 * late_pressure * late_pressure

        return max(threshold, self._min_utility)

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Select a diverse bid above threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        candidates = self._outcome_space.get_bids_in_range(threshold, threshold + 0.05)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold)

        if candidates:
            # Prefer bids not recently offered
            diverse_candidates = [
                c for c in candidates if c.bid not in self._recent_bids
            ]
            if diverse_candidates:
                selected = random.choice(diverse_candidates).bid
            else:
                selected = random.choice(candidates).bid

            # Update history
            self._recent_bids.append(selected)
            if len(self._recent_bids) > self._max_history:
                self._recent_bids.pop(0)

            return selected

        return self._best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        threshold = self._calculate_threshold(time)

        return self._select_bid(threshold)

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
        self._update_opponent_model(offer)

        threshold = self._calculate_threshold(time)
        offer_utility = float(self.ufun(offer))

        # Accept if above threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than expected future offers (near deadline)
        if time > self._late_game_threshold:
            expected_future = self._estimate_opponent_future_utility(time)
            if offer_utility >= expected_future - 0.02:
                return ResponseType.ACCEPT_OFFER

        # Near deadline, accept above minimum
        if time > 0.98 and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
