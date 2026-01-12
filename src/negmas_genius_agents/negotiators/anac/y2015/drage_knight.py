"""DrageKnight from ANAC 2015."""

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

__all__ = ["DrageKnight"]


class DrageKnight(SAONegotiator):
    """
    DrageKnight negotiation agent from ANAC 2015.

    DrageKnight uses a bold initial stance with adaptive concession that
    responds to opponent behavior patterns.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.DrageKnight.DrageKnight

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Three-phase Boulware concession (e=0.12):
          * Bold phase (t<0.4): Very firm, concedes only 20% of computed
            amount, stays near 85-100% utility
          * Strategic phase (0.4<t<0.8): Gradual 60% concession toward 55%
            minimum acceptable
          * Honor phase (t>0.8): Concedes 80% toward fair deal, considers
            best opponent utility + 5%
        - If opponent is conceding, becomes firmer (e * 0.6)
        - Bold phase prefers top 20% of candidates

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - Honorable end-game (t>0.95): Accepts if offer >= best opponent
          utility OR offer >= min_acceptable - 5%

    **Opponent Modeling:**
        - Tracks opponent bid history and best utility offered
        - Detects opponent concession by comparing last 3 offers
        - Uses concession detection to adjust firmness
        - Incorporates best opponent utility into end-game target

    Args:
        e: Concession exponent controlling concession speed (default 0.12)
        bold_time_threshold: Time threshold for bold phase (default 0.4)
        strategic_time_threshold: Time threshold for strategic phase (default 0.8)
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
        e: float = 0.12,
        bold_time_threshold: float = 0.4,
        strategic_time_threshold: float = 0.8,
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
        self._bold_time_threshold = bold_time_threshold
        self._strategic_time_threshold = strategic_time_threshold
        self._deadline_time_threshold = deadline_time_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._min_acceptable: float = 0.55

        # Opponent tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._best_opponent_utility: float = 0.0
        self._opponent_conceding: bool = False

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
        self._best_opponent_utility = 0.0
        self._opponent_conceding = False

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent and detect concession."""
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Detect if opponent is conceding
        if len(self._opponent_bids) >= 3:
            recent = [u for _, u in self._opponent_bids[-3:]]
            if recent[-1] > recent[0]:
                self._opponent_conceding = True
            else:
                self._opponent_conceding = False

    def _compute_threshold(self, time: float) -> float:
        """Compute threshold with DrageKnight strategy."""
        e = self._e

        # If opponent is conceding, stay firm
        if self._opponent_conceding:
            e *= 0.6

        if time < self._bold_time_threshold:
            # Bold phase: stay very high
            f_t = math.pow(time / self._bold_time_threshold, 1 / e)
            return self._max_utility - (self._max_utility - 0.85) * f_t * 0.2
        elif time < self._strategic_time_threshold:
            # Strategic phase: gradual concession
            progress = (time - self._bold_time_threshold) / (
                self._strategic_time_threshold - self._bold_time_threshold
            )
            f_t = math.pow(progress, 1 / e)
            return 0.85 - (0.85 - self._min_acceptable) * f_t * 0.6
        else:
            # Honor phase: consider fair dealing
            progress = (time - self._strategic_time_threshold) / (
                1.0 - self._strategic_time_threshold
            )
            target = max(self._best_opponent_utility + 0.05, self._min_acceptable)
            current = 0.85 - (0.85 - self._min_acceptable) * 0.6
            return current - (current - target) * progress * 0.8

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid based on dragon-knight principles."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Bold phase: prefer top bids
        if time < self._bold_time_threshold:
            top_n = max(1, len(candidates) // 5)
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

        # Honorable end-game: accept fair deals
        if time > self._deadline_time_threshold:
            if offer_utility >= max(
                self._best_opponent_utility, self._min_acceptable - 0.05
            ):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
