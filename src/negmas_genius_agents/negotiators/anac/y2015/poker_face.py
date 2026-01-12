"""PokerFace from ANAC 2015."""

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

__all__ = ["PokerFace"]


class PokerFace(SAONegotiator):
    """
    PokerFace negotiation agent from ANAC 2015.

    PokerFace uses a bluffing-inspired strategy that hides true preferences
    by varying offers unpredictably while maintaining a consistent hidden
    acceptance threshold.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.pokerface.PokerFace

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Three-phase bluffing approach:
          * Bluffing (t<0.5): Displays ~98% demands with +/-5% noise
            and +/-10% random variance to confuse opponent modeling
          * Transition (0.5<t<0.8): Gradually reveals true position,
            moving from 95% toward true_threshold + 0.1
          * Reveal (t>0.8): Converges to true_threshold (70%)
        - Random bid selection to hide preference patterns
        - Bluff mode disables after t=0.6

    **Acceptance Strategy:**
        - Uses hidden "true threshold" (70%) for acceptance decisions
        - Acceptance threshold: true_threshold + 0.15 until t=0.7,
          then gradually decreases to true_threshold - 0.1 by t=1.0
        - Near deadline (t>0.98): Accepts if offer >= true_threshold - 0.1

    **Opponent Modeling:**
        - Minimal tracking: stores opponent bids but doesn't use for
          strategy adaptation
        - Strategy relies on hiding own preferences rather than
          exploiting opponent's

    Args:
        bluff_factor: How much to bluff (default 0.3)
        bluff_time_threshold: Time threshold for bluffing phase (default 0.5)
        transition_time_threshold: Time threshold for transition phase (default 0.8)
        acceptance_transition_time: Time when acceptance becomes more lenient (default 0.7)
        acceptance_reveal_time: Time when acceptance threshold starts decreasing (default 0.9)
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
        bluff_factor: float = 0.3,
        bluff_time_threshold: float = 0.5,
        transition_time_threshold: float = 0.8,
        acceptance_transition_time: float = 0.7,
        acceptance_reveal_time: float = 0.9,
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
        self._bluff_factor = bluff_factor
        self._bluff_time_threshold = bluff_time_threshold
        self._transition_time_threshold = transition_time_threshold
        self._acceptance_transition_time = acceptance_transition_time
        self._acceptance_reveal_time = acceptance_reveal_time
        self._deadline_time_threshold = deadline_time_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._true_threshold: float = 0.7  # Hidden true threshold

        # Tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._bluff_mode: bool = True

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
        self._bluff_mode = True

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent offers."""
        self._opponent_bids.append((bid, utility))

    def _compute_displayed_threshold(self, time: float) -> float:
        """Compute the threshold we display through offers (may include bluff)."""
        if time < self._bluff_time_threshold and self._bluff_mode:
            # Bluffing phase: display higher demands
            base = self._max_utility * 0.98
            noise = random.uniform(-0.05, 0.05)
            return min(self._max_utility, base + noise)
        elif time < self._transition_time_threshold:
            # Transition phase
            progress = (time - self._bluff_time_threshold) / (
                self._transition_time_threshold - self._bluff_time_threshold
            )
            bluff_threshold = self._max_utility * 0.95
            true_target = self._true_threshold + 0.1
            return bluff_threshold - (bluff_threshold - true_target) * progress
        else:
            # Reveal phase
            progress = (time - self._transition_time_threshold) / (
                1.0 - self._transition_time_threshold
            )
            return self._true_threshold + 0.1 * (1 - progress)

    def _compute_acceptance_threshold(self, time: float) -> float:
        """Compute actual acceptance threshold (the true one)."""
        # True threshold is more lenient than displayed
        if time < self._acceptance_transition_time:
            return self._true_threshold + 0.15
        elif time < self._acceptance_reveal_time:
            progress = (time - self._acceptance_transition_time) / (
                self._acceptance_reveal_time - self._acceptance_transition_time
            )
            return self._true_threshold + 0.15 * (1 - progress)
        else:
            progress = (time - self._acceptance_reveal_time) / (
                1.0 - self._acceptance_reveal_time
            )
            return self._true_threshold - 0.1 * progress

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid with poker-style variation."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_displayed_threshold(time)

        # Add unpredictability
        if self._bluff_mode and time < self._bluff_time_threshold:
            # Vary bids to confuse opponent modeling
            variance = random.uniform(-0.1, 0.1)
            threshold = max(
                self._true_threshold, min(self._max_utility, threshold + variance)
            )

        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold - 0.1)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Random selection to hide preference patterns
        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal with poker-style bluffing."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        # Decide whether to continue bluffing
        if time > self._bluff_time_threshold + 0.1:  # 0.6 by default
            self._bluff_mode = False

        return self._select_bid(time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer using true threshold."""
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

        # Use true threshold for acceptance (poker face - what we accept differs from what we show)
        threshold = self._compute_acceptance_threshold(time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, accept reasonable offers
        if (
            time > self._deadline_time_threshold
            and offer_utility >= self._true_threshold - 0.1
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
