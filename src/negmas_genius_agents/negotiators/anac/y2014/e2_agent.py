"""E2Agent from ANAC 2014."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["E2Agent"]


class E2Agent(SAONegotiator):
    """
    E2Agent (AnacSampleAgent) from ANAC 2014.

    E2Agent implements a straightforward negotiation strategy with clear
    exploration-exploitation phases. Its simplicity makes it computationally
    efficient while still achieving reasonable performance across domains.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.E2Agent.AnacSampleAgent

    **Offering Strategy:**
        Linear time-based concession with exploration-exploitation phases:
        - Target utility: max - (max - min) * time * 0.7
        - Early (t < 0.3): Full exploration - random selection from candidates
        - Middle (0.3 <= t < 0.7): Mixed - 30% random, 70% best candidate
        - Late (t >= 0.7): Exploitation - best candidates, occasional
          consideration of best opponent bid if above target (20% chance)

        This phase structure allows learning opponent behavior early while
        optimizing outcomes later in negotiation.

    **Acceptance Strategy:**
        Standard threshold-based with comparative backup:
        1. Accept if offer utility meets computed target threshold
        2. Accept if offer matches or exceeds next planned bid utility
        3. Near-deadline acceptance (t > 0.98) for offers above minimum
        The strategy ensures deals are reached while maintaining aspirations.

    **Opponent Modeling:**
        Minimal opponent tracking for efficiency:
        - Records (bid, utility) tuples from opponent offers
        - Tracks best opponent bid (highest utility for self)
        - Best bid considered for late-game offers (t > 0.9) when
          it meets the target threshold, potentially accelerating agreement
        - No complex frequency analysis - relies on simple best-offer tracking

    Args:
        min_utility: Minimum acceptable utility threshold (default 0.6).
        exploration_end_time: Time threshold ending exploration phase (default 0.3).
        exploitation_start_time: Time threshold starting exploitation phase (default 0.7).
        deadline_acceptance_time: Time threshold for near-deadline acceptance (default 0.98).
        concession_multiplier: Multiplier for linear concession (default 0.7).
        mid_phase_random_probability: Probability of random selection in mid phase (default 0.3).
        late_phase_time: Time threshold for considering best opponent bid (default 0.9).
        best_opponent_bid_probability: Probability of using best opponent bid late (default 0.2).
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
        exploration_end_time: float = 0.3,
        exploitation_start_time: float = 0.7,
        deadline_acceptance_time: float = 0.98,
        concession_multiplier: float = 0.7,
        mid_phase_random_probability: float = 0.3,
        late_phase_time: float = 0.9,
        best_opponent_bid_probability: float = 0.2,
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
        self._exploration_end_time = exploration_end_time
        self._exploitation_start_time = exploitation_start_time
        self._deadline_acceptance_time = deadline_acceptance_time
        self._concession_multiplier = concession_multiplier
        self._mid_phase_random_probability = mid_phase_random_probability
        self._late_phase_time = late_phase_time
        self._best_opponent_bid_probability = best_opponent_bid_probability
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._last_bid: Outcome | None = None

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._last_bid = None

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Track opponent bids."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

    def _compute_target_utility(self, time: float) -> float:
        """Compute target using linear concession."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return self._min_utility

        max_util = self._outcome_space.max_utility

        # Linear concession
        target = (
            max_util
            - (max_util - self._min_utility) * time * self._concession_multiplier
        )
        return max(self._min_utility, target)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid with exploration-exploitation balance."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._compute_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Early exploration: random from candidates
        if time < self._exploration_end_time:
            return random.choice(candidates).bid

        # Mid-game: mix of random and best
        if time < self._exploitation_start_time:
            if random.random() < self._mid_phase_random_probability:
                return random.choice(candidates).bid
            return candidates[0].bid

        # Late game: mostly best offers, occasionally best opponent
        if time > self._late_phase_time and self._best_opponent_bid is not None:
            if self._best_opponent_utility >= target:
                if random.random() < self._best_opponent_bid_probability:
                    return self._best_opponent_bid

        return candidates[0].bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        bid = self._select_bid(time)
        self._last_bid = bid
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

        time = state.relative_time
        offer_utility = float(self.ufun(offer))
        self._update_opponent_model(offer)

        target = self._compute_target_utility(time)

        # Accept if above target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than our next offer
        next_bid = self._select_bid(time)
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid))
            if offer_utility >= next_utility:
                return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if reasonable
        if time > self._deadline_acceptance_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
