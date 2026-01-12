"""KGAgent from ANAC 2014."""

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

__all__ = ["KGAgent"]


class KGAgent(SAONegotiator):
    """
    KGAgent from ANAC 2014.

    KGAgent (Knowledge-Guided Agent) employs adaptive strategy adjustment
    based on learned opponent behavior patterns. It estimates opponent
    concession rates and modifies its own concession accordingly, using
    a simplified Kalman-filter-like state estimation approach.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.KGAgent.KGAgent

    **Offering Strategy:**
        Adaptive target with opponent-responsive concession:
        - Base concession: time^2 acceleration (faster near deadline)
        - Adaptation based on opponent concession rate:
          * Opponent conceding (rate > 0.01): tougher stance
            (target += learning_rate * opponent_rate)
          * Opponent hardening (rate < -0.01): faster concession
            (target -= learning_rate * 0.5)
          * Stable opponent: normal time-based concession
        - Target = initial - base_concession + adaptation

        Bid selection mixes exploration and exploitation:
        - Early (t < 0.5): Random from top 50% of candidates
        - Late: Best candidate from those meeting target

    **Acceptance Strategy:**
        Target-based with flexibility conditions:
        1. Accept if offer utility meets current adaptive target
        2. Accept if offer matches or exceeds next planned bid utility
        3. Increased flexibility near deadline (t > 0.95) for offers
           above minimum threshold
        The adaptive target ensures responsiveness to opponent behavior.

    **Opponent Modeling:**
        Concession rate estimation from utility trajectory:
        - Tracks utilities of all opponent bids (from self's perspective)
        - Estimates concession rate from recent 5 bids:
          rate = (last_utility - first_utility) / window_size
        - Positive rate: opponent giving better offers (conceding)
        - Negative rate: opponent demanding more (hardening)
        - Best opponent bid tracked for potential reference

    Args:
        initial_target: Starting utility target (default 0.95).
        learning_rate: Rate of strategy adaptation (default 0.1).
        exploration_end_time: Time threshold ending exploration phase (default 0.5).
        flexibility_time: Time threshold for increased flexibility (default 0.95).
        positive_concession_threshold: Threshold for detecting positive concession (default 0.01).
        negative_concession_threshold: Threshold for detecting negative concession (default -0.01).
        hardening_adaptation_factor: Factor for adapting when opponent hardens (default 0.5).
        base_concession_multiplier: Multiplier for base concession (default 0.5).
        lowered_threshold_factor: Factor for lowering target when no candidates (default 0.95).
        exploration_candidates_divisor: Divisor for exploration candidates (default 2).
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
        initial_target: float = 0.95,
        learning_rate: float = 0.1,
        exploration_end_time: float = 0.5,
        flexibility_time: float = 0.95,
        positive_concession_threshold: float = 0.01,
        negative_concession_threshold: float = -0.01,
        hardening_adaptation_factor: float = 0.5,
        base_concession_multiplier: float = 0.5,
        lowered_threshold_factor: float = 0.95,
        exploration_candidates_divisor: int = 2,
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
        self._initial_target = initial_target
        self._learning_rate = learning_rate
        self._exploration_end_time = exploration_end_time
        self._flexibility_time = flexibility_time
        self._positive_concession_threshold = positive_concession_threshold
        self._negative_concession_threshold = negative_concession_threshold
        self._hardening_adaptation_factor = hardening_adaptation_factor
        self._base_concession_multiplier = base_concession_multiplier
        self._lowered_threshold_factor = lowered_threshold_factor
        self._exploration_candidates_divisor = exploration_candidates_divisor
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent tracking
        self._opponent_utilities: list[float] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_concession_rate: float = 0.0

        # State estimation
        self._estimated_opponent_target: float = 0.5
        self._current_target: float = initial_target
        self._min_utility: float = 0.5

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._min_utility = max(0.5, self._outcome_space.min_utility)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_utilities = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_concession_rate = 0.0
        self._estimated_opponent_target = 0.5
        self._current_target = self._initial_target

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent model and estimate concession rate."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_utilities.append(utility)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        # Estimate opponent concession rate
        if len(self._opponent_utilities) >= 3:
            recent = self._opponent_utilities[-5:]
            if len(recent) >= 2:
                # Positive = opponent is conceding (giving us better utility)
                self._opponent_concession_rate = (recent[-1] - recent[0]) / len(recent)

    def _update_target(self, time: float) -> None:
        """Update own target based on opponent behavior and time."""
        # Base time-dependent concession
        time_pressure = time**2  # Accelerating near deadline

        # Adapt based on opponent behavior
        if self._opponent_concession_rate > self._positive_concession_threshold:
            # Opponent is conceding, we can be tougher
            adaptation = -self._learning_rate * self._opponent_concession_rate
        elif self._opponent_concession_rate < self._negative_concession_threshold:
            # Opponent is hardening, we need to concede more
            adaptation = self._learning_rate * self._hardening_adaptation_factor
        else:
            # Opponent is stable, normal time concession
            adaptation = 0.0

        # Compute new target
        base_concession = (
            (self._initial_target - self._min_utility)
            * time_pressure
            * self._base_concession_multiplier
        )
        self._current_target = self._initial_target - base_concession + adaptation
        self._current_target = max(self._min_utility, min(1.0, self._current_target))

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid above current target."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        candidates = self._outcome_space.get_bids_above(self._current_target)

        if not candidates:
            # Lower target slightly and retry
            lowered = self._current_target * self._lowered_threshold_factor
            candidates = self._outcome_space.get_bids_above(lowered)
            if not candidates:
                return self._outcome_space.outcomes[0].bid

        # Mix of best and random from candidates
        if time < self._exploration_end_time and len(candidates) > 3:
            return random.choice(
                candidates[: len(candidates) // self._exploration_candidates_divisor]
            ).bid

        return candidates[0].bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        self._update_target(time)

        return self._select_bid(time)

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
        self._update_target(time)

        # Accept if above current target
        if offer_utility >= self._current_target:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than our next offer
        next_bid = self._select_bid(time)
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid))
            if offer_utility >= next_utility:
                return ResponseType.ACCEPT_OFFER

        # Near deadline, be flexible
        if time > self._flexibility_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
