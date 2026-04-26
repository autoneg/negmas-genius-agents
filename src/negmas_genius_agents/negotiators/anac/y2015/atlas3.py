"""Atlas3 from ANAC 2015."""

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

__all__ = ["Atlas3"]


class Atlas3(SAONegotiator):
    """
    Atlas3 negotiation agent - Winner of ANAC 2015.

    Atlas3 was designed for multilateral negotiation but also excels in
    bilateral settings. It uses time-dependent concession with special
    end-game handling and tracks popular bids for final-phase proposals.
    This is a bilateral reimplementation of the original Java agent.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.Atlas3.Atlas3

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Three-phase Boulware concession (e=0.1):
          * Start phase (t<0.2): Very high threshold at 95% of max utility
          * Main phase (0.2<t<0.8): Gradual 30% concession of range
          * End phase (t>0.8): Quadratic accelerated concession of 50%
            of remaining range
        - Random bid search within acceptable utility range
        - Final phase: proposes popular bids (those opponent accepted)
          in descending utility order

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - Final phase acceptance: Accepts any offer above reservation
          value when fewer than 5 turns remain

    **Opponent Modeling:**
        - Tracks opponent bid history with utilities
        - Maintains list of "popular" bids (those accepted by opponent
          in multilateral settings - simplified for bilateral)
        - Monitors time between turns to detect final phase
        - No explicit preference estimation; uses bid history for
          end-game proposals

    Args:
        e: Concession exponent (default 0.1, Boulware-like)
        start_phase_time_threshold: Time threshold for start phase (default 0.2)
        main_phase_time_threshold: Time threshold for main/end phase transition (default 0.8)
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
        e: float = 0.1,
        start_phase_time_threshold: float = 0.2,
        main_phase_time_threshold: float = 0.8,
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
        self._start_phase_time_threshold = start_phase_time_threshold
        self._main_phase_time_threshold = main_phase_time_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State tracking
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._reservation_value: float = 0.0
        self._time_scale: float = 0.0
        self._last_time: float = 0.0

        # Bid histories
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._own_bids: list[tuple[Outcome, float]] = []
        self._popular_bids: list[tuple[Outcome, float]] = []  # Bids others accepted

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

        # Reset state
        self._opponent_bids = []
        self._own_bids = []
        self._popular_bids = []
        self._time_scale = 0.0
        self._last_time = 0.0

    def _update_time_scale(self, time: float) -> None:
        """Track time between turns for end-game planning."""
        if self._last_time > 0:
            self._time_scale = time - self._last_time
        self._last_time = time

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold using time-dependent formula."""
        # Atlas3 uses a Boulware-like formula with reservation value consideration
        if time < self._start_phase_time_threshold:
            # Start phase: very high threshold
            return self._max_utility * 0.95
        elif time < self._main_phase_time_threshold:
            # Main negotiation phase
            f_t = (
                math.pow(
                    (time - self._start_phase_time_threshold)
                    / (
                        self._main_phase_time_threshold
                        - self._start_phase_time_threshold
                    ),
                    1 / self._e,
                )
                if self._e != 0
                else 0
            )
            target = (
                self._max_utility - (self._max_utility - self._min_utility) * 0.3 * f_t
            )
            return max(target, self._reservation_value)
        else:
            # End phase: more aggressive concession
            f_t = math.pow(
                (time - self._main_phase_time_threshold)
                / (1.0 - self._main_phase_time_threshold),
                2,
            )  # Quadratic concession
            target = (
                self._max_utility * 0.7
                - (self._max_utility * 0.7 - self._min_utility) * 0.5 * f_t
            )
            return max(target, self._reservation_value)

    def _search_bid(self, threshold: float) -> Outcome | None:
        """Search for a bid meeting the threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        # Get bids above threshold
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            # Try lower threshold
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            # Return best bid
            return self._outcome_space.outcomes[0].bid

        # Random selection from candidates
        return random.choice(candidates).bid

    def _is_in_final_phase(self, time: float) -> bool:
        """Check if we're in the final proposal phase."""
        if self._time_scale <= 0:
            return False

        # Final phase if remaining time is less than a few turns
        remaining_turns = (1.0 - time) / self._time_scale
        return remaining_turns < 5

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        self._update_time_scale(time)

        # Check for final phase
        if self._is_in_final_phase(time) and self._popular_bids:
            # In final phase, try popular bids in reverse order (best first)
            for bid, util in sorted(self._popular_bids, key=lambda x: -x[1]):
                if util > self._reservation_value:
                    return bid

        threshold = self._compute_threshold(time)
        bid = self._search_bid(threshold)

        if bid is not None and self.ufun is not None:
            self._own_bids.append((bid, float(self.ufun(bid))))

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
        self._update_time_scale(time)

        offer_utility = float(self.ufun(offer))
        self._opponent_bids.append((offer, offer_utility))

        threshold = self._compute_threshold(time)

        # Accept if meets threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # In final phase, accept if above reservation
        if self._is_in_final_phase(time) and offer_utility > self._reservation_value:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
