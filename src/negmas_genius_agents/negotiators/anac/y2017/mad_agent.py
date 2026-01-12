"""MadAgent from ANAC 2017."""

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

__all__ = ["MadAgent"]


class MadAgent(SAONegotiator):
    """
    MadAgent from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This is a reimplementation of MadAgent from ANAC 2017.
    Original: agents.anac.y2017.madagent.MadAgent

    MadAgent uses an unpredictable, "mad" negotiation strategy with
    random elements to confuse opponent modeling while maintaining
    reasonable utility.

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/nego/node/7

    **Offering Strategy:**
        Combines quadratic time-based concession with random oscillation
        (+-15% * madness). Occasionally (madness * 10% chance) becomes
        very stubborn (+0.1) or generous (-0.1). With madness * 15%
        probability, selects a completely random valid bid. Late-game
        (>85%) gradually transitions to rational behavior.

    **Acceptance Strategy:**
        Accepts offers above the oscillating threshold. With madness * 10%
        probability, accepts offers up to 0.1 below threshold if above
        minimum utility. Near deadline (>95%), becomes fully rational
        and accepts above minimum.

    **Opponent Modeling:**
        Minimal modeling - only tracks opponent's best offer utility.
        The "madness" strategy deliberately avoids sophisticated opponent
        modeling to prevent being exploited by adaptive opponents.

    Args:
        min_utility: Minimum acceptable utility (default 0.5).
        madness: Controls unpredictability level, 0-1 (default 0.3).
        late_game_threshold: Time threshold for normalizing behavior (default 0.85).
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
        min_utility: float = 0.5,
        madness: float = 0.3,
        late_game_threshold: float = 0.85,
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
        self._madness = min(max(madness, 0.0), 1.0)  # Clamp to [0, 1]
        self._late_game_threshold = late_game_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

        # Opponent tracking
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
        self._best_opponent_utility = 0.0

    def _calculate_threshold(self, time: float) -> float:
        """Calculate threshold with random oscillation."""
        # Base polynomial concession
        base_concession = math.pow(time, 2)
        utility_range = self._max_utility - self._min_utility
        base_threshold = self._max_utility - base_concession * utility_range

        # Add "madness" - random oscillation
        mad_factor = random.uniform(-self._madness, self._madness) * 0.15
        threshold = base_threshold + mad_factor

        # Occasionally be very stubborn or very generous
        if random.random() < self._madness * 0.1:
            if random.random() < 0.5:
                # Stubborn
                threshold = max(threshold + 0.1, self._max_utility * 0.9)
            else:
                # More generous
                threshold = threshold - 0.1

        # Late game normalization - reduce madness
        if time > self._late_game_threshold:
            late_factor = (time - self._late_game_threshold) / (
                1.0 - self._late_game_threshold
            )
            # Blend toward rational behavior
            rational_threshold = self._min_utility + (1 - late_factor) * 0.2
            threshold = threshold * (1 - late_factor * 0.5) + rational_threshold * (
                late_factor * 0.5
            )

        return max(min(threshold, self._max_utility), self._min_utility)

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Select a bid with occasional random choice."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        # Occasionally select a completely random bid (controlled chaos)
        if random.random() < self._madness * 0.15:
            valid_outcomes = [
                bd
                for bd in self._outcome_space.outcomes
                if bd.utility >= self._min_utility
            ]
            if valid_outcomes:
                return random.choice(valid_outcomes).bid

        # Normal selection
        candidates = self._outcome_space.get_bids_in_range(
            threshold - 0.05, threshold + 0.1
        )

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(self._min_utility)

        if candidates:
            return random.choice(candidates).bid

        return self._best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        threshold = self._calculate_threshold(time)

        return self._select_bid(threshold)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer with occasional unpredictability."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)

        time = state.relative_time
        threshold = self._calculate_threshold(time)

        # Normal acceptance check
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Mad acceptance - occasionally accept lower offers
        if random.random() < self._madness * 0.1:
            if offer_utility >= threshold - 0.1 and offer_utility >= self._min_utility:
                return ResponseType.ACCEPT_OFFER

        # Near deadline - be more rational
        if time > 0.95 and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
