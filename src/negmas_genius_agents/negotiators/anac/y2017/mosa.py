"""Mosa from ANAC 2017."""

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

__all__ = ["Mosa"]


class Mosa(SAONegotiator):
    """
    Mosa from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This is a reimplementation of Mosa from ANAC 2017.
    Original: agents.anac.y2017.mosateam.Mosa

    Mosa (Multi-Objective Simulated Annealing inspired) uses a cooling
    schedule similar to simulated annealing for concession control. It
    treats the negotiation as an optimization problem with a cooling schedule.

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/nego/node/7

    **Offering Strategy:**
        Uses exponential cooling schedule: T = T0 * cooling_rate^(time*100).
        Threshold is min_utility + temperature * utility_range. At high
        temperature, bid selection is more random (exploration); at low
        temperature, prefers higher-utility bids (exploitation). Search
        range width also depends on temperature.

    **Acceptance Strategy:**
        Accepts offers above the temperature-based threshold, adjusted to
        stay above (opponent_best - 0.05). At high temperature (>0.3),
        has 10% * temperature chance to accept slightly lower offers
        (exploration). Late-game (>90%) blends toward opponent's best
        or minimum utility. Very late (>98%) accepts above minimum.

    **Opponent Modeling:**
        Tracks opponent's best offer and utility history. The best offer
        serves as a floor for acceptance and influences late-game threshold
        calculation to ensure realistic expectations.

    Args:
        min_utility: Minimum acceptable utility (default 0.6).
        initial_temperature: Starting temperature (default 1.0).
        cooling_rate: Exponential decay rate (default 0.95).
        late_game_threshold: Time threshold for late game acceleration (default 0.9).
        cooling_time_scale: Multiplier on relative time in the exponential cooling
            schedule (default 100.0).
        min_temperature: Lower bound on the cooling temperature (default 0.01).
        opponent_best_offset: Tolerance below the opponent's best utility used as
            a threshold floor (default 0.05).
        base_range_width: Base width of the bid search range when temperature is
            zero (default 0.05).
        temperature_range_scale: Additional bid search range width per unit of
            temperature (default 0.1).
        high_temperature_threshold: Temperature above which bid selection is
            fully random among candidates (default 0.5).
        top_candidates_divisor: Divisor of the candidate list used to pick the
            top-scoring bids to randomize among at low temperature (default 3).
        exploration_temperature_threshold: Temperature above which exploratory
            acceptance of lower offers may occur (default 0.3).
        exploration_probability_scale: Multiplier on temperature giving the
            probability of exploratory acceptance (default 0.1).
        exploration_acceptance_tolerance: Tolerance below the threshold still
            accepted during exploratory acceptance (default 0.1).
        deadline_threshold: Relative time after which any offer above
            ``min_utility`` is accepted (default 0.98).
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
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        late_game_threshold: float = 0.9,
        cooling_time_scale: float = 100.0,
        min_temperature: float = 0.01,
        opponent_best_offset: float = 0.05,
        base_range_width: float = 0.05,
        temperature_range_scale: float = 0.1,
        high_temperature_threshold: float = 0.5,
        top_candidates_divisor: int = 3,
        exploration_temperature_threshold: float = 0.3,
        exploration_probability_scale: float = 0.1,
        exploration_acceptance_tolerance: float = 0.1,
        deadline_threshold: float = 0.98,
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
        self._initial_temperature = initial_temperature
        self._cooling_rate = cooling_rate
        self._late_game_threshold = late_game_threshold
        self._cooling_time_scale = cooling_time_scale
        self._min_temperature = min_temperature
        self._opponent_best_offset = opponent_best_offset
        self._base_range_width = base_range_width
        self._temperature_range_scale = temperature_range_scale
        self._high_temperature_threshold = high_temperature_threshold
        self._top_candidates_divisor = top_candidates_divisor
        self._exploration_temperature_threshold = exploration_temperature_threshold
        self._exploration_probability_scale = exploration_probability_scale
        self._exploration_acceptance_tolerance = exploration_acceptance_tolerance
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._temperature: float = initial_temperature

        # Opponent tracking
        self._best_opponent_utility: float = 0.0
        self._opponent_utilities: list[float] = []

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
        self._temperature = self._initial_temperature
        self._best_opponent_utility = 0.0
        self._opponent_utilities = []

    def _update_opponent_model(self, offer: Outcome) -> None:
        """Track opponent's offers."""
        if self.ufun is None:
            return

        offer_utility = float(self.ufun(offer))
        self._opponent_utilities.append(offer_utility)
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)

    def _update_temperature(self, time: float) -> None:
        """Update temperature based on time using cooling schedule."""
        # Exponential cooling
        self._temperature = self._initial_temperature * math.pow(
            self._cooling_rate, time * self._cooling_time_scale
        )

        # Ensure minimum temperature
        self._temperature = max(self._temperature, self._min_temperature)

    def _calculate_threshold(self, time: float) -> float:
        """Calculate threshold based on current temperature."""
        self._update_temperature(time)

        # Threshold is based on temperature and utility range
        utility_range = self._max_utility - self._min_utility
        threshold = self._min_utility + self._temperature * utility_range

        # Adjust based on opponent's best offer
        if self._best_opponent_utility > self._min_utility:
            # Don't accept less than what opponent has offered
            threshold = max(threshold, self._best_opponent_utility - self._opponent_best_offset)

        # Late game acceleration
        if time > self._late_game_threshold:
            late_factor = (time - self._late_game_threshold) / (
                1.0 - self._late_game_threshold
            )
            # Blend toward best opponent utility
            target = max(self._best_opponent_utility, self._min_utility)
            threshold = threshold * (1 - late_factor) + target * late_factor

        return max(threshold, self._min_utility)

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Select a bid using temperature-controlled randomness."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        # Width of search range depends on temperature
        range_width = self._base_range_width + self._temperature * self._temperature_range_scale

        candidates = self._outcome_space.get_bids_in_range(
            threshold, threshold + range_width
        )

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold)

        if candidates:
            # At high temperature, more random selection
            # At low temperature, prefer higher utility
            if self._temperature > self._high_temperature_threshold and len(candidates) > 1:
                return random.choice(candidates).bid
            else:
                # Prefer higher utility bids
                top_count = max(1, len(candidates) // self._top_candidates_divisor)
                return random.choice(candidates[:top_count]).bid

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

        # Accept if above threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # At high temperature, occasionally accept lower offers (exploration)
        if (
            self._temperature > self._exploration_temperature_threshold
            and random.random() < self._temperature * self._exploration_probability_scale
        ):
            if (
                offer_utility >= threshold - self._exploration_acceptance_tolerance
                and offer_utility >= self._min_utility
            ):
                return ResponseType.ACCEPT_OFFER

        # Near deadline
        if time > self._deadline_threshold and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
