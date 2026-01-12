"""Mercury from ANAC 2015."""

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

__all__ = ["Mercury"]


class Mercury(SAONegotiator):
    """
    Mercury negotiation agent from ANAC 2015.

    Mercury uses a fluid, quick-moving strategy with fast adaptation
    to opponent behavior and swift deal-making in the end-game.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.Mercury.Mercury

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Fluid concession with adaptive rate multiplier:
          threshold = max_util - (max_util - min_acceptable) * t^(1/(e*fluid_rate))
        - Minimum acceptable floor at 45% or min_utility + 0.1
        - Fluid rate adapts: 0.7 if opponent conceding (slows down),
          1.3 if opponent hardening (speeds up), 1.0 otherwise
        - Varied bid selection from candidates above threshold

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - AC_Next: Accepts if offer >= utility of our next bid
        - Swift end-game (t>0.93): Accepts if offer >= best opponent
          utility OR offer >= minimum acceptable

    **Opponent Modeling:**
        - Tracks opponent utilities to calculate trend (derivative)
        - Trend based on last 3 offers: (recent - older) / 2
        - Positive trend (>0.02): opponent conceding, slow down
        - Negative trend (<-0.02): opponent hardening, speed up
        - Updates fluid rate multiplier based on trend

    Args:
        e: Base concession exponent (default 0.25)
        deadline_time_threshold: Time after which swift end-game triggers (default 0.93)
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
        e: float = 0.25,
        deadline_time_threshold: float = 0.93,
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
        self._deadline_time_threshold = deadline_time_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._fluid_rate: float = 1.0  # Multiplier for concession

        # Opponent tracking
        self._opponent_utilities: list[float] = []
        self._best_opponent_utility: float = 0.0
        self._opponent_trend: float = 0.0  # Positive = conceding

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

        self._opponent_utilities = []
        self._best_opponent_utility = 0.0
        self._opponent_trend = 0.0
        self._fluid_rate = 1.0

    def _update_opponent_model(self, utility: float) -> None:
        """Track opponent and calculate trend."""
        self._opponent_utilities.append(utility)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Calculate trend (derivative of opponent utility)
        if len(self._opponent_utilities) >= 3:
            recent = self._opponent_utilities[-3:]
            self._opponent_trend = (recent[-1] - recent[0]) / 2

            # Adapt fluid rate based on opponent
            if self._opponent_trend > 0.02:
                # Opponent conceding, slow down
                self._fluid_rate = 0.7
            elif self._opponent_trend < -0.02:
                # Opponent hardening, speed up
                self._fluid_rate = 1.3
            else:
                self._fluid_rate = 1.0

    def _compute_threshold(self, time: float) -> float:
        """Compute threshold with fluid adaptation."""
        e = self._e * self._fluid_rate

        # Mercury formula: fast and fluid
        f_t = math.pow(time, 1 / e) if e > 0 else time
        min_acceptable = max(0.45, self._min_utility + 0.1)

        target = self._max_utility - (self._max_utility - min_acceptable) * f_t
        return max(target, min_acceptable)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid with fluid variety."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Mercury: fluid, varied selection
        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        return self._select_bid(state.relative_time)

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

        self._update_opponent_model(offer_utility)

        threshold = self._compute_threshold(time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # AC_Next
        our_bid = self._select_bid(time)
        if our_bid is not None:
            our_utility = float(self.ufun(our_bid))
            if offer_utility >= our_utility:
                return ResponseType.ACCEPT_OFFER

        # Swift end-game
        if time > self._deadline_time_threshold:
            min_acceptable = max(0.45, self._min_utility + 0.1)
            if offer_utility >= max(self._best_opponent_utility, min_acceptable):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
