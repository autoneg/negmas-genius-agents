"""GeneKing from ANAC 2017."""

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

__all__ = ["GeneKing"]


class GeneKing(SAONegotiator):
    """
    GeneKing from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This is a reimplementation of GeneKing from ANAC 2017.
    Original: agents.anac.y2017.geneking.GeneKing

    GeneKing uses a genetic algorithm-inspired approach for bid selection,
    maintaining a population of good bids for diverse exploration.

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/nego/node/7

    **Offering Strategy:**
        Maintains a population of top-utility bids initialized from the
        outcome space. Uses quadratic time-dependent decay (1 - t^2) for
        threshold calculation. Selects bids from the population that meet
        the current threshold, falling back to outcome space search if needed.

    **Acceptance Strategy:**
        Accepts offers above the adaptive threshold, which is adjusted based
        on opponent concession trends. Near deadline (>98% time), accepts
        any offer above minimum utility.

    **Opponent Modeling:**
        Tracks opponent's utility trend over recent offers. Positive trends
        (>0.1) indicate concession, triggering more patience (+0.02 threshold).
        Negative trends (<-0.05) indicate hardening, triggering faster
        concession (-0.03 threshold). Late-game (>95%) applies additional
        time pressure reduction.

    Args:
        min_utility: Minimum acceptable utility (default 0.6).
        population_size: Number of bids to maintain in population (default 10).
        late_game_threshold: Time threshold for late game pressure (default 0.95).
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
        min_utility: float = 0.6,
        population_size: int = 10,
        late_game_threshold: float = 0.95,
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
        self._population_size = population_size
        self._late_game_threshold = late_game_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

        # Population of good bids
        self._population: list[Outcome] = []

        # Opponent tracking
        self._opponent_utilities: list[float] = []
        self._best_opponent_utility: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space and population."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility

            # Initialize population with top bids
            top_bids = self._outcome_space.outcomes[: self._population_size * 2]
            self._population = [bd.bid for bd in top_bids]

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._opponent_utilities = []
        self._best_opponent_utility = 0.0

    def _update_opponent_model(self, offer: Outcome) -> None:
        """Track opponent's offers."""
        if self.ufun is None:
            return

        offer_utility = float(self.ufun(offer))
        self._opponent_utilities.append(offer_utility)
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)

    def _get_opponent_trend(self) -> float:
        """Calculate opponent's concession trend."""
        if len(self._opponent_utilities) < 3:
            return 0.0

        recent = self._opponent_utilities[-5:]
        if len(recent) >= 2:
            return recent[-1] - recent[0]
        return 0.0

    def _calculate_threshold(self, time: float) -> float:
        """Calculate acceptance threshold using adaptive decay."""
        # Exponential decay with opponent adaptation
        base_decay = 1 - math.pow(time, 2)
        utility_range = self._max_utility - self._min_utility

        threshold = self._min_utility + base_decay * utility_range

        # Adapt to opponent behavior
        opponent_trend = self._get_opponent_trend()
        if opponent_trend > 0.1:
            # Opponent is conceding, be more patient
            threshold += 0.02
        elif opponent_trend < -0.05:
            # Opponent is hardening, concede more
            threshold -= 0.03

        # Late game pressure
        if time > self._late_game_threshold:
            pressure = (time - self._late_game_threshold) / (
                1.0 - self._late_game_threshold
            )
            threshold -= 0.1 * pressure

        return max(min(threshold, self._max_utility), self._min_utility)

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Select a bid from the population or outcome space."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        # First try to select from population
        if self._population:
            valid_population = [
                bid
                for bid in self._population
                if self.ufun is not None and float(self.ufun(bid)) >= threshold
            ]
            if valid_population:
                return random.choice(valid_population)

        # Fall back to outcome space
        candidates = self._outcome_space.get_bids_above(threshold)

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
        threshold = self._calculate_threshold(time)
        offer_utility = float(self.ufun(offer))

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if above minimum
        if time > 0.98 and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
