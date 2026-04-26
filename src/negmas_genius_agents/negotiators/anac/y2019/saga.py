"""
SAGA from ANAC 2019 - 3rd place agent.

This module contains the Python reimplementation of SAGA
(Self-Adaptive Genetic Algorithm-based agent), which placed
3rd in ANAC 2019. SAGA uses genetic algorithm concepts for
bid exploration and selection.

References:
    Baarslag, T., Fujita, K., Gerding, E. H., Hindriks, K., Ito, T.,
    Jennings, N. R., ... & Williams, C. R. (2019). The Tenth International
    Automated Negotiating Agents Competition (ANAC 2019).
    In Proceedings of the International Joint Conference on Autonomous
    Agents and Multiagent Systems (AAMAS).

Original Genius class: agents.anac.y2019.saga.SAGA
"""

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

__all__ = ["SAGA"]


class SAGA(SAONegotiator):
    """
    SAGA from ANAC 2019 - 3rd place agent.

    SAGA (Self-Adaptive Genetic Algorithm) achieved 3rd place in ANAC 2019
    using genetic algorithm concepts for bid exploration. It maintains a
    population of candidate bids that evolves over the negotiation to
    balance self-interest and opponent satisfaction.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    **Offering Strategy:**
        - Maintains a population of candidate bids (default size 20)
        - Population evolves using selection, mutation, and immigration
        - Fitness = alpha * own_utility + (1-alpha) * opponent_utility
        - Alpha decreases over time: alpha = 1.0 - 0.5*t
          (early: focus on self, late: balance with opponent)
        - Selection: keep top 50% by fitness
        - Immigration: add high-utility bids from outcome space
        - Mutation: replace with similar-utility bids
        - Returns highest-fitness bid meeting current threshold

    **Acceptance Strategy:**
        - Quadratic concession threshold: t = initial - (initial - min) * t^2
        - Accepts offers meeting or exceeding the threshold
        - Near reservation value protection built in
        - Very near deadline (t >= 0.99): Accepts minimum threshold or
          98% of best received offer

    **Opponent Modeling:**
        - Frequency-based model tracking issue value occurrences
        - Estimates opponent utility as normalized frequency scores
        - Used in fitness calculation to evolve population toward
          mutually beneficial outcomes
        - Influences bid selection but not acceptance threshold

    Args:
        population_size: Size of the bid population (default 20)
        initial_threshold: Starting acceptance threshold (default 0.95)
        min_threshold: Minimum acceptance threshold (default 0.6)
        deadline_threshold: Time threshold for deadline acceptance (default 0.99)
        deadline_best_ratio: Ratio of best received utility for deadline acceptance (default 0.98)
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
        population_size: int = 20,
        initial_threshold: float = 0.95,
        min_threshold: float = 0.6,
        deadline_threshold: float = 0.99,
        deadline_best_ratio: float = 0.98,
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
        self._population_size = population_size
        self._initial_threshold = initial_threshold
        self._min_threshold = min_threshold
        self._deadline_threshold = deadline_threshold
        self._deadline_best_ratio = deadline_best_ratio

        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Population of candidate bids
        self._population: list[Outcome] = []

        # Opponent model: tracks value frequencies from opponent offers
        self._opponent_value_freq: dict[int, dict[str, int]] = {}
        self._opponent_offers_count: int = 0

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._current_threshold: float = initial_threshold
        self._best_received_utility: float = 0.0

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

        # Initialize population with diverse high-utility bids
        self._initialize_population()

        self._initialized = True

    def _initialize_population(self) -> None:
        """Initialize the population with diverse high-utility bids."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return

        self._population = []

        # Take top bids as initial population
        num_outcomes = len(self._outcome_space.outcomes)
        step = max(1, num_outcomes // (self._population_size * 2))

        for i in range(0, num_outcomes, step):
            if len(self._population) >= self._population_size:
                break
            self._population.append(self._outcome_space.outcomes[i].bid)

        # Fill remaining slots with random high-utility bids
        high_utility_bids = self._outcome_space.get_bids_above(0.7)
        while len(self._population) < self._population_size and high_utility_bids:
            bid = random.choice(high_utility_bids).bid
            if bid not in self._population:
                self._population.append(bid)

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_value_freq = {}
        self._opponent_offers_count = 0
        self._current_threshold = self._initial_threshold
        self._best_received_utility = 0.0

        # Re-initialize population
        self._initialize_population()

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent model based on received bid."""
        if bid is None:
            return

        self._opponent_offers_count += 1

        for i, value in enumerate(bid):
            if i not in self._opponent_value_freq:
                self._opponent_value_freq[i] = {}
            value_str = str(value)
            if value_str not in self._opponent_value_freq[i]:
                self._opponent_value_freq[i][value_str] = 0
            self._opponent_value_freq[i][value_str] += 1

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """
        Estimate opponent's utility for a bid based on value frequencies.

        Higher frequency values are assumed to be more preferred by opponent.
        """
        if bid is None or self._opponent_offers_count == 0:
            return 0.5  # Neutral estimate

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            value_str = str(value)
            if i in self._opponent_value_freq:
                freq = self._opponent_value_freq[i].get(value_str, 0)
                # Normalize by max frequency for this issue
                max_freq = (
                    max(self._opponent_value_freq[i].values())
                    if self._opponent_value_freq[i]
                    else 1
                )
                total_score += freq / max_freq if max_freq > 0 else 0

        return total_score / num_issues if num_issues > 0 else 0.5

    def _compute_fitness(self, bid: Outcome, time: float) -> float:
        """
        Compute fitness of a bid combining own and opponent utility.

        Early in negotiation: prioritize own utility
        Late in negotiation: balance own and opponent utility
        """
        if bid is None or self.ufun is None:
            return 0.0

        own_utility = float(self.ufun(bid))
        opponent_utility = self._estimate_opponent_utility(bid)

        # Selection pressure decreases over time
        # Early: alpha close to 1 (focus on own utility)
        # Late: alpha decreases (consider opponent more)
        alpha = 1.0 - 0.5 * time

        return alpha * own_utility + (1 - alpha) * opponent_utility

    def _update_threshold(self, time: float) -> None:
        """Update acceptance threshold based on time."""
        # Concession curve: starts high, decreases over time
        # Use a polynomial concession for smoother decline
        concession_rate = time**2  # Slow early, faster late

        self._current_threshold = (
            self._initial_threshold
            - (self._initial_threshold - self._min_threshold) * concession_rate
        )

        # Never go below reservation value if we have one
        reservation = self.reserved_value if self.reserved_value is not None else 0.0
        self._current_threshold = max(self._current_threshold, reservation)

    def _evolve_population(self, time: float) -> None:
        """
        Evolve the population using genetic algorithm concepts.

        - Selection: keep high-fitness individuals
        - Crossover: combine good bids
        - Mutation: explore nearby bids
        """
        if self._outcome_space is None or not self._population:
            return

        # Compute fitness for all individuals
        fitness_scores = [
            (bid, self._compute_fitness(bid, time)) for bid in self._population
        ]

        # Sort by fitness (descending)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # Keep top half (selection)
        survivors = [bid for bid, _ in fitness_scores[: self._population_size // 2]]

        # Generate new individuals
        new_population = list(survivors)

        # Add some high-utility bids from outcome space (immigration)
        threshold_for_new = max(0.7, self._current_threshold - 0.1)
        candidates = self._outcome_space.get_bids_above(threshold_for_new)

        # Limit iterations to avoid infinite loop when all candidates are duplicates
        max_iterations = self._population_size * 10
        iterations = 0

        while (
            len(new_population) < self._population_size and iterations < max_iterations
        ):
            iterations += 1

            if candidates and random.random() < 0.5:
                # Immigration: add a random good bid
                new_bid = random.choice(candidates).bid
            elif survivors:
                # Mutation: slightly modify a survivor
                # (in this simplified version, we just pick a similar bid)
                parent = random.choice(survivors)
                parent_utility = float(self.ufun(parent)) if self.ufun else 0.5
                nearby = self._outcome_space.get_bids_in_range(
                    parent_utility - 0.1, parent_utility + 0.05
                )
                new_bid = random.choice(nearby).bid if nearby else parent
            else:
                break

            if new_bid not in new_population:
                new_population.append(new_bid)

        self._population = new_population

    def _select_bid(self, time: float) -> Outcome | None:
        """Select the best bid from the population."""
        if not self._population:
            return self._best_bid

        # Evolve population
        self._evolve_population(time)

        # Find bid with best fitness that meets our threshold
        if self.ufun is None:
            return self._population[0] if self._population else None

        best_bid = None
        best_fitness = -float("inf")

        for bid in self._population:
            utility = float(self.ufun(bid))
            if utility >= self._current_threshold:
                fitness = self._compute_fitness(bid, time)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_bid = bid

        # Fallback to highest utility bid if none meets threshold
        if best_bid is None:
            best_bid = max(self._population, key=lambda b: float(self.ufun(b)))

        return best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        # First offer: best bid
        if self._opponent_offers_count == 0:
            return self._best_bid

        # Update threshold
        self._update_threshold(time)

        # Select bid from population
        return self._select_bid(time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
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

        # Track best received offer
        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility

        # Update threshold
        self._update_threshold(time)

        # Accept if above adaptive threshold
        if offer_utility >= self._current_threshold:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if better than what we've seen
        if time >= self._deadline_threshold:
            if offer_utility >= self._best_received_utility * self._deadline_best_ratio:
                return ResponseType.ACCEPT_OFFER
            # Accept anything above minimum threshold
            if offer_utility >= self._min_threshold:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
