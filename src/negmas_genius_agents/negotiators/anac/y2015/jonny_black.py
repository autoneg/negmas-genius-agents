"""JonnyBlack from ANAC 2015."""

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

__all__ = ["JonnyBlack"]


class JonnyBlack(SAONegotiator):
    """
    JonnyBlack negotiation agent from ANAC 2015.

    JonnyBlack uses unpredictable concession behavior with opponent
    exploitation when weakness (rapid concession) is detected.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.JonnyBlack.JonnyBlack

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Three-phase concession with "mystery factor" noise (-8% to +8%):
          * Early (t<0.3): Firm at ~92% utility plus mystery factor
          * Middle (0.3<t<0.8): Boulware concession toward 55% with
            unpredictable variance
          * End (t>0.8): Deal-making mode toward 45% with 50% factor
        - If opponent is desperate (rapid 15%+ concession in 4 offers),
          becomes more aggressive (e * 0.5)
        - Random selection from candidates adds variety

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold (with
          mystery factor)
        - Last-minute (t>0.98): Accepts if offer >= 95% of best opponent
          utility OR offer >= min_utility + 0.1

    **Opponent Modeling:**
        - Tracks opponent bid history and best utility offered
        - Detects "desperation": 15%+ improvement over last 4 offers
        - Uses desperation detection to stay more aggressive
        - Mystery factor obscures true preference patterns

    Args:
        e: Base concession exponent (default 0.15)
        early_time_threshold: Time threshold for early phase (default 0.3)
        main_time_threshold: Time threshold for main/end phase transition (default 0.8)
        deadline_time_threshold: Time after which end-game acceptance triggers (default 0.98)
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
        early_time_threshold: float = 0.3,
        main_time_threshold: float = 0.8,
        deadline_time_threshold: float = 0.98,
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
        self._early_time_threshold = early_time_threshold
        self._main_time_threshold = main_time_threshold
        self._deadline_time_threshold = deadline_time_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._mystery_factor: float = 0.0

        # Opponent tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._best_opponent_utility: float = 0.0
        self._opponent_desperate: bool = False

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
        self._opponent_desperate = False
        self._mystery_factor = 0.0

    def _update_mystery(self) -> None:
        """Add unpredictability to behavior."""
        self._mystery_factor = random.uniform(-0.08, 0.08)

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent and detect desperation."""
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Detect desperation: opponent rapidly conceding
        if len(self._opponent_bids) >= 4:
            recent = [u for _, u in self._opponent_bids[-4:]]
            if recent[-1] - recent[0] > 0.15:
                self._opponent_desperate = True

    def _compute_threshold(self, time: float) -> float:
        """Compute threshold with mysterious behavior."""
        e = self._e

        # Exploit desperate opponent
        if self._opponent_desperate:
            e *= 0.5

        if time < self._early_time_threshold:
            # Early: firm but mysterious
            base = self._max_utility * 0.92
            return base + self._mystery_factor
        elif time < self._main_time_threshold:
            # Middle: unpredictable concession
            progress = (time - self._early_time_threshold) / (
                self._main_time_threshold - self._early_time_threshold
            )
            f_t = math.pow(progress, 1 / e)
            base = self._max_utility * 0.92 - (self._max_utility * 0.92 - 0.55) * f_t
            return max(base + self._mystery_factor, 0.5)
        else:
            # End: deal-making mode
            progress = (time - self._main_time_threshold) / (
                1.0 - self._main_time_threshold
            )
            base = 0.55 - (0.55 - 0.45) * progress * 0.5
            return max(base + self._mystery_factor, self._min_utility + 0.1)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid with strategic variance."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        self._update_mystery()

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.85)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Add variety to offers
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

        # Last-minute deal
        if time > self._deadline_time_threshold:
            if offer_utility >= max(
                self._best_opponent_utility * 0.95, self._min_utility + 0.1
            ):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
