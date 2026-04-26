"""AgentW from ANAC 2015."""

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

__all__ = ["AgentW"]


class AgentW(SAONegotiator):
    """
    AgentW negotiation agent from ANAC 2015.

    AgentW uses a patient "waiting" strategy that observes and classifies
    opponent behavior before adapting its concession pattern.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.AgentW.AgentW

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Waiting phase (first 5 offers): Stays at 95% utility while
          gathering information
        - Main phase (0.15<t<0.85): Boulware-like concession toward 55%
          with adaptive rate based on opponent type
        - End phase (t>0.85): Accelerated concession with 0.7 factor
        - In waiting phase, prefers top 10% of candidates

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - End-game (t>0.95): Accepts if offer >= best opponent utility
          OR offer >= min_utility + 0.1

    **Opponent Modeling:**
        - Classifies opponent after 5+ observations into three types:
          * "conceder": recent offers 5%+ better than earlier (e * 0.7)
          * "hardhead": recent offers 2%+ worse than earlier (e * 1.5)
          * "unknown": neutral behavior (base e)
        - Tracks best opponent utility and full utility history
        - Adapts concession exponent based on classification

    Args:
        e: Base concession exponent (default 0.15)
        main_phase_time_threshold: Time threshold for main phase (default 0.85)
        deadline_time_threshold: Time after which end-game acceptance triggers (default 0.95)
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
        e: float = 0.15,
        main_phase_time_threshold: float = 0.85,
        deadline_time_threshold: float = 0.95,
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
        self._main_phase_time_threshold = main_phase_time_threshold
        self._deadline_time_threshold = deadline_time_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._reservation_value: float = 0.0
        self._waiting_phase: bool = True

        # Opponent modeling
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._opponent_utilities: list[float] = []
        self._opponent_type: str = "unknown"  # "conceder", "hardhead", "unknown"
        self._best_opponent_utility: float = 0.0

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

        self._opponent_bids = []
        self._opponent_utilities = []
        self._opponent_type = "unknown"
        self._best_opponent_utility = 0.0
        self._waiting_phase = True

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent and classify their type."""
        self._opponent_bids.append((bid, utility))
        self._opponent_utilities.append(utility)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Classify opponent after enough observations
        if len(self._opponent_utilities) >= 5:
            self._waiting_phase = False

            # Analyze utility trend
            first_half = self._opponent_utilities[: len(self._opponent_utilities) // 2]
            second_half = self._opponent_utilities[len(self._opponent_utilities) // 2 :]

            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            if avg_second > avg_first + 0.05:
                self._opponent_type = "conceder"
            elif avg_second < avg_first - 0.02:
                self._opponent_type = "hardhead"
            else:
                self._opponent_type = "unknown"

    def _get_adaptive_e(self) -> float:
        """Get concession exponent based on opponent type."""
        if self._opponent_type == "conceder":
            return self._e * 0.7  # Be firmer against conceder
        elif self._opponent_type == "hardhead":
            return self._e * 1.5  # Concede more against hardhead
        return self._e

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold with waiting strategy."""
        e = self._get_adaptive_e()

        # Waiting phase: stay high
        if self._waiting_phase or time < 0.15:
            return self._max_utility * 0.95

        # Main negotiation phase
        if time < self._main_phase_time_threshold:
            progress = (time - 0.15) / (self._main_phase_time_threshold - 0.15)
            f_t = math.pow(progress, 1 / e)
            target = self._max_utility * 0.95 - (self._max_utility * 0.95 - 0.55) * f_t
            return max(target, self._reservation_value)

        # End phase
        progress = (time - self._main_phase_time_threshold) / (
            1.0 - self._main_phase_time_threshold
        )
        base = 0.55
        target = base - (base - self._min_utility - 0.1) * progress * 0.7
        return max(target, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid based on current phase."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # In waiting phase, prefer top bids
        if self._waiting_phase:
            top_n = max(1, len(candidates) // 10)
            return random.choice(candidates[:top_n]).bid

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

        self._update_opponent_model(offer, offer_utility)

        threshold = self._compute_threshold(time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # End-game: accept reasonable offers
        if time > self._deadline_time_threshold:
            if offer_utility >= max(
                self._best_opponent_utility, self._min_utility + 0.1
            ):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
