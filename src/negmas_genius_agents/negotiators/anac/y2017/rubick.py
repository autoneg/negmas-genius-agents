"""Rubick from ANAC 2017."""

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

__all__ = ["Rubick"]


class Rubick(SAONegotiator):
    """
    Rubick from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This is a reimplementation of Rubick from ANAC 2017.
    Original: agents.anac.y2017.rubick.Rubick

    Rubick is named after the Dota 2 character known for stealing abilities.
    The agent learns from and adapts to opponent behavior.

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/nego/node/7

    **Offering Strategy:**
        Uses linear time-based decay (initial - 0.35*time) as base threshold,
        adapted by opponent concession estimate. Opponent concession (>0.1)
        adds patience (+0.05), opponent hardening (<-0.05) triggers faster
        concession (-0.05). Prefers bids slightly above threshold for
        exploration, selecting from the upper half of valid candidates.

    **Acceptance Strategy:**
        Accepts offers above the adaptive threshold. Late-game (>90%)
        applies additional time pressure (-0.1 * progress). Near deadline
        (>98%) accepts any offer above minimum utility.

    **Opponent Modeling:**
        Tracks opponent offers with timestamps. Estimates opponent's
        concession rate from utility change over time in recent offers
        (last 5). This estimate directly influences our concession
        adaptation, effectively "stealing" information about opponent's
        strategy.

    Args:
        min_utility: Minimum acceptable utility (default 0.6).
        initial_threshold: Starting acceptance threshold (default 0.95).
        late_game_threshold: Time threshold for late game pressure (default 0.9).
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
        initial_threshold: float = 0.95,
        late_game_threshold: float = 0.9,
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
        self._initial_threshold = initial_threshold
        self._late_game_threshold = late_game_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

        # Opponent modeling
        self._opponent_bids: list[tuple[float, float]] = []  # (time, utility)
        self._opponent_concession_estimate: float = 0.0
        self._best_opponent_utility: float = 0.0

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
        self._opponent_bids = []
        self._opponent_concession_estimate = 0.0
        self._best_opponent_utility = 0.0

    def _update_opponent_model(self, offer: Outcome, time: float) -> None:
        """Update opponent modeling based on received offer."""
        if self.ufun is None:
            return

        offer_utility = float(self.ufun(offer))
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)
        self._opponent_bids.append((time, offer_utility))

        # Estimate opponent's concession rate
        if len(self._opponent_bids) >= 3:
            recent = self._opponent_bids[-5:]
            if len(recent) >= 2:
                utility_change = recent[-1][1] - recent[0][1]
                time_change = recent[-1][0] - recent[0][0]
                if time_change > 0:
                    self._opponent_concession_estimate = utility_change / time_change

    def _calculate_threshold(self, time: float) -> float:
        """Calculate acceptance threshold based on time and opponent model."""
        # Base concession based on time
        base_threshold = self._initial_threshold - 0.35 * time

        # Adapt based on opponent's concession
        if self._opponent_concession_estimate > 0.1:
            # Opponent is conceding - we can be more patient
            adaptation = 0.05
        elif self._opponent_concession_estimate < -0.05:
            # Opponent is hardening - we need to concede more
            adaptation = -0.05
        else:
            adaptation = 0.0

        threshold = base_threshold + adaptation

        # Late game pressure
        if time > self._late_game_threshold:
            time_pressure = (time - self._late_game_threshold) / (
                1.0 - self._late_game_threshold
            )
            threshold = threshold - 0.1 * time_pressure

        return max(threshold, self._min_utility)

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Select a bid above the threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(self._min_utility)

        if candidates:
            # Prefer bids slightly above threshold for exploration
            if len(candidates) > 3:
                return random.choice(candidates[: len(candidates) // 2]).bid
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

        time = state.relative_time
        self._update_opponent_model(offer, time)

        threshold = self._calculate_threshold(time)
        offer_utility = float(self.ufun(offer))

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if better than our minimum
        if time > 0.98 and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
