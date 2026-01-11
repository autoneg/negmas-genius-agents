"""Group3 from ANAC 2017."""

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

__all__ = ["Group3"]


class Group3(SAONegotiator):
    """
    Group3 from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This is a reimplementation of Group3 from ANAC 2017.
    Original: agents.anac.y2017.group3.Group3

    Group3 uses a clear three-phase negotiation strategy with distinct
    behaviors in each phase.

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/nego/node/7

    **Offering Strategy:**
        Three-phase approach:
        - Phase 1 (0-40%): Hardball with fixed high threshold (0.95).
        - Phase 2 (40-80%): Linear decrease from Phase 1 threshold to
          (min_utility + 0.2).
        - Phase 3 (80-100%): Accelerated concession using power function
          (t^1.5), considering opponent's best offer.
        Bids are selected randomly from a range around the threshold.

    **Acceptance Strategy:**
        Accepts offers above the phase-dependent threshold. In the final
        phase, adjusts threshold toward opponent's best offer if it exceeds
        minimum utility. Near deadline (>95%), accepts any offer above
        minimum utility.

    **Opponent Modeling:**
        Tracks opponent's best offer utility and offer count. The best
        opponent utility is used in Phase 3 to inform threshold adjustments,
        ensuring we don't demand more than the best we've been offered.

    Args:
        min_utility: Minimum acceptable utility (default 0.55).
        phase1_threshold: Threshold during phase 1 (default 0.95).
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
        min_utility: float = 0.55,
        phase1_threshold: float = 0.95,
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
        self._phase1_threshold = phase1_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

        # Phase boundaries
        self._phase1_end = 0.4
        self._phase2_end = 0.8

        # Opponent tracking
        self._best_opponent_utility: float = 0.0
        self._opponent_offers_count: int = 0

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
        self._opponent_offers_count = 0

    def _update_opponent_model(self, offer: Outcome) -> None:
        """Track opponent's best offer."""
        if self.ufun is None:
            return

        offer_utility = float(self.ufun(offer))
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)
        self._opponent_offers_count += 1

    def _calculate_threshold(self, time: float) -> float:
        """Calculate threshold based on current phase."""
        if time < self._phase1_end:
            # Phase 1: Hardball
            return self._phase1_threshold

        elif time < self._phase2_end:
            # Phase 2: Exploration - linear decrease
            phase_progress = (time - self._phase1_end) / (
                self._phase2_end - self._phase1_end
            )
            phase2_end_threshold = self._min_utility + 0.2
            threshold = self._phase1_threshold - phase_progress * (
                self._phase1_threshold - phase2_end_threshold
            )
            return threshold

        else:
            # Phase 3: Agreement seeking - faster concession
            phase_progress = (time - self._phase2_end) / (1.0 - self._phase2_end)
            phase3_start = self._min_utility + 0.2

            # Use quadratic decay for faster concession
            threshold = phase3_start - math.pow(phase_progress, 1.5) * (
                phase3_start - self._min_utility
            )

            # Consider best opponent offer
            if self._best_opponent_utility > self._min_utility:
                threshold = min(threshold, self._best_opponent_utility + 0.05)

            return max(threshold, self._min_utility)

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Select a bid above threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        candidates = self._outcome_space.get_bids_in_range(threshold, threshold + 0.1)

        if not candidates:
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

        # In final phase, accept if above minimum
        if time > 0.95 and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
