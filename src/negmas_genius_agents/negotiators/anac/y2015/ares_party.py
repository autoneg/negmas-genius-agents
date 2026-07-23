"""AresParty from ANAC 2015."""

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

__all__ = ["AresParty"]


class AresParty(SAONegotiator):
    """
    AresParty negotiation agent from ANAC 2015.

    AresParty uses an aggressive Boulware strategy with minimal early
    concession and opponent weakness exploitation.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.AresParty.AresParty

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Very aggressive three-phase concession (e=0.08):
          * First half (t<0.5): Barely concedes, stays near 85-100% with
            30% of computed concession applied
          * Second half (0.5<t<0.9): Moderate concession toward 60%
            minimum with 50% factor
          * Final phase (t>0.9): Tactical retreat toward minimum with
            60% factor
        - If opponent is weakening (conceding), becomes even more
          aggressive (e * 0.5)
        - Prefers top 20% of candidates to maintain high demands

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - Near-deadline (t>0.98): Accepts if offer >= best opponent
          utility OR offer >= 60% minimum acceptable

    **Opponent Modeling:**
        - Tracks opponent bid history and best utility offered
        - Detects opponent "weakening" by comparing recent (last 5)
          vs earlier offer averages
        - Uses weakness detection to adjust aggressiveness

    Args:
        e: Concession exponent (default 0.08, very Boulware)
        first_half_time_threshold: Time threshold for first half phase (default 0.5)
        second_half_time_threshold: Time threshold for second half/final phase transition (default 0.9)
        deadline_time_threshold: Time after which end-game acceptance triggers (default 0.98)
        min_acceptable: Minimum acceptable utility target (default 0.6)
        weakening_multiplier: Multiplier applied to e when opponent is weakening (default 0.5)
        weakness_detection_min_bids: Minimum opponent bids to detect weakening (default 5)
        weakness_recent_window: Number of recent opponent bids for weakness detection (default 5)
        high_target: High utility target maintained in early phases (default 0.85)
        first_half_concession_factor: Fraction of concession applied in first half (default 0.3)
        second_half_concession_factor: Fraction of concession applied in second half (default 0.5)
        final_phase_min_margin: Margin above min utility in final phase (default 0.1)
        final_phase_concession_factor: Fraction of concession applied in final phase (default 0.6)
        fallback_threshold_ratio: Ratio used when lowering threshold if no candidates (default 0.9)
        top_fraction: Divisor for selecting top candidates (default 5)
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
        e: float = 0.08,
        first_half_time_threshold: float = 0.5,
        second_half_time_threshold: float = 0.9,
        deadline_time_threshold: float = 0.98,
        min_acceptable: float = 0.6,
        weakening_multiplier: float = 0.5,
        weakness_detection_min_bids: int = 5,
        weakness_recent_window: int = 5,
        high_target: float = 0.85,
        first_half_concession_factor: float = 0.3,
        second_half_concession_factor: float = 0.5,
        final_phase_min_margin: float = 0.1,
        final_phase_concession_factor: float = 0.6,
        fallback_threshold_ratio: float = 0.9,
        top_fraction: int = 5,
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
        self._first_half_time_threshold = first_half_time_threshold
        self._second_half_time_threshold = second_half_time_threshold
        self._deadline_time_threshold = deadline_time_threshold
        self._min_acceptable = min_acceptable
        self._weakening_multiplier = weakening_multiplier
        self._weakness_detection_min_bids = weakness_detection_min_bids
        self._weakness_recent_window = weakness_recent_window
        self._high_target = high_target
        self._first_half_concession_factor = first_half_concession_factor
        self._second_half_concession_factor = second_half_concession_factor
        self._final_phase_min_margin = final_phase_min_margin
        self._final_phase_concession_factor = final_phase_concession_factor
        self._fallback_threshold_ratio = fallback_threshold_ratio
        self._top_fraction = top_fraction
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._reservation_value: float = 0.0

        # Opponent tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._best_opponent_utility: float = 0.0
        self._opponent_weakening: bool = False

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
        self._opponent_weakening = False

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent behavior for weakness detection."""
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Detect opponent weakening (offering better deals over time)
        if len(self._opponent_bids) >= self._weakness_detection_min_bids:
            recent = [u for _, u in self._opponent_bids[-self._weakness_recent_window :]]
            earlier = [u for _, u in self._opponent_bids[: -self._weakness_recent_window]]
            if earlier and sum(recent) / len(recent) > sum(earlier) / len(earlier):
                self._opponent_weakening = True

    def _compute_threshold(self, time: float) -> float:
        """Compute threshold with aggressive Ares strategy."""
        e = self._e

        # If opponent is weakening, stay very firm
        if self._opponent_weakening:
            e *= self._weakening_multiplier

        if time < self._first_half_time_threshold:
            # First half: very aggressive, barely concede
            f_t = math.pow(time / self._first_half_time_threshold, 1 / e)
            return self._max_utility - (self._max_utility - self._high_target) * f_t * self._first_half_concession_factor
        elif time < self._second_half_time_threshold:
            # Second half: moderate concession
            progress = (time - self._first_half_time_threshold) / (
                self._second_half_time_threshold - self._first_half_time_threshold
            )
            f_t = math.pow(progress, 1 / e)
            return self._high_target - (self._high_target - self._min_acceptable) * f_t * self._second_half_concession_factor
        else:
            # Final phase: tactical retreat
            progress = (time - self._second_half_time_threshold) / (
                1.0 - self._second_half_time_threshold
            )
            return (
                self._min_acceptable
                - (self._min_acceptable - self._min_utility - self._final_phase_min_margin)
                * progress
                * self._final_phase_concession_factor
            )

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid with aggressive strategy."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(
                threshold * self._fallback_threshold_ratio
            )

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Ares prefers higher utility bids
        top_n = max(1, len(candidates) // self._top_fraction)
        return random.choice(candidates[:top_n]).bid

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

        # Very end: accept if better than expected
        if time > self._deadline_time_threshold:
            if offer_utility >= max(self._best_opponent_utility, self._min_acceptable):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER