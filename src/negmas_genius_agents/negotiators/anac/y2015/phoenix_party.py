"""PhoenixParty from ANAC 2015."""

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

__all__ = ["PhoenixParty"]


class PhoenixParty(SAONegotiator):
    """
    PhoenixParty negotiation agent from ANAC 2015.

    PhoenixParty uses a rebirth strategy with aggressive initial stance
    and adaptive phase transitions when negotiations stall.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.Phoenix.PhoenixParty

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Three-phase rebirth mechanism:
          * Phase 1 - Aggressive (t<0.6): Starts at initial_threshold (98%),
            concedes 15% by t=0.6, prefers top 20% of candidates
          * Phase 2 - Adaptive (0.6<t<0.8): Concedes toward
            max(best_opponent + 0.1, min_threshold + 0.2), varied selection
          * Phase 3 - Final (t>0.8): Concedes 70% toward min_threshold + 0.1
        - "Rebirth" triggers phase transition when stuck (3+ rounds of
          <5% variance in opponent offers)

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - Near deadline (t>0.98): Accepts if offer >= best opponent utility
          AND offer >= minimum threshold

    **Opponent Modeling:**
        - Tracks opponent utility pattern to detect stuck negotiation
        - Monitors best opponent utility for adaptive targeting
        - "Stuck counter" increments when opponent offers show <5% variance
          over 5 rounds
        - Rebirth decision based on stuck counter reaching 3

    Args:
        initial_threshold: Starting threshold (default 0.98)
        min_threshold: Minimum acceptable utility (default 0.5)
        phase1_time_threshold: Time threshold for phase 1 (aggressive) (default 0.3)
        phase2_time_threshold: Time threshold for phase 2 (adaptive) (default 0.6)
        phase3_time_threshold: Time threshold for phase 3 (final) (default 0.8)
        stuck_window: Number of recent opponent offers checked for stuck detection (default 5)
        stuck_variance_threshold: Variance below which a round counts as stuck (default 0.05)
        rebirth_stuck_count: Stuck rounds needed to trigger rebirth (default 3)
        phase1_concession: Utility conceded during phase 1 (default 0.15)
        opponent_best_bonus: Bonus above best opponent utility for adaptive target (default 0.1)
        adaptive_target_margin: Margin above min threshold for adaptive/final current target (default 0.2)
        final_phase_target_margin: Margin above min threshold for final phase target (default 0.1)
        fallback_threshold_ratio: Ratio used when lowering threshold if no candidates (default 0.9)
        phase1_top_fraction: Divisor for selecting top candidates in phase 1 (default 5)
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
        initial_threshold: float = 0.98,
        min_threshold: float = 0.5,
        phase1_time_threshold: float = 0.3,
        phase2_time_threshold: float = 0.6,
        phase3_time_threshold: float = 0.8,
        stuck_window: int = 5,
        stuck_variance_threshold: float = 0.05,
        rebirth_stuck_count: int = 3,
        phase1_concession: float = 0.15,
        opponent_best_bonus: float = 0.1,
        adaptive_target_margin: float = 0.2,
        final_phase_target_margin: float = 0.1,
        fallback_threshold_ratio: float = 0.9,
        phase1_top_fraction: int = 5,
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
        self._initial_threshold = initial_threshold
        self._min_threshold = min_threshold
        self._phase1_time_threshold = phase1_time_threshold
        self._phase2_time_threshold = phase2_time_threshold
        self._phase3_time_threshold = phase3_time_threshold
        self._stuck_window = stuck_window
        self._stuck_variance_threshold = stuck_variance_threshold
        self._rebirth_stuck_count = rebirth_stuck_count
        self._phase1_concession = phase1_concession
        self._opponent_best_bonus = opponent_best_bonus
        self._adaptive_target_margin = adaptive_target_margin
        self._final_phase_target_margin = final_phase_target_margin
        self._fallback_threshold_ratio = fallback_threshold_ratio
        self._phase1_top_fraction = phase1_top_fraction
        self._deadline_time_threshold = deadline_time_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._phase: int = 1  # Phoenix has phases
        self._stuck_counter: int = 0

        # Opponent tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._opponent_best_for_us: float = 0.0
        self._opponent_pattern: list[float] = []

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._phase = 1
        self._stuck_counter = 0
        self._opponent_bids = []
        self._opponent_best_for_us = 0.0
        self._opponent_pattern = []

    def _update_opponent_tracking(self, bid: Outcome, utility: float) -> None:
        """Track opponent behavior patterns."""
        self._opponent_bids.append((bid, utility))
        self._opponent_pattern.append(utility)

        if utility > self._opponent_best_for_us:
            self._opponent_best_for_us = utility

        # Check if stuck (no improvement in recent offers)
        if len(self._opponent_pattern) >= self._stuck_window:
            recent = self._opponent_pattern[-self._stuck_window :]
            if max(recent) - min(recent) < self._stuck_variance_threshold:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0

    def _should_rebirth(self) -> bool:
        """Check if we should rebirth (change strategy phase)."""
        return self._stuck_counter >= self._rebirth_stuck_count

    def _compute_threshold(self, time: float) -> float:
        """Compute threshold with phoenix rebirth mechanics."""
        # Phoenix phases affect concession
        if self._phase == 1:
            # Aggressive phase
            if time < self._phase1_time_threshold:
                return self._initial_threshold
            elif time < self._phase2_time_threshold:
                progress = (time - self._phase1_time_threshold) / (
                    self._phase2_time_threshold - self._phase1_time_threshold
                )
                return self._initial_threshold - self._phase1_concession * progress
            else:
                # Time to consider rebirth
                if self._should_rebirth():
                    self._phase = 2
                return self._initial_threshold - self._phase1_concession

        elif self._phase == 2:
            # Adaptive phase - concede based on opponent best
            target = max(
                self._opponent_best_for_us + self._opponent_best_bonus,
                self._min_threshold + self._adaptive_target_margin,
            )
            if time < self._phase3_time_threshold:
                progress = (time - self._phase2_time_threshold) / (
                    self._phase3_time_threshold - self._phase2_time_threshold
                )
                base = self._initial_threshold - self._phase1_concession
                return base - (base - target) * progress
            else:
                if self._should_rebirth():
                    self._phase = 3
                return target

        else:
            # Final phase - more aggressive concession
            progress = min(
                1.0,
                (time - self._phase3_time_threshold)
                / (1.0 - self._phase3_time_threshold),
            )
            target = self._min_threshold + self._final_phase_target_margin
            current = max(
                self._opponent_best_for_us, self._min_threshold + self._adaptive_target_margin
            )
            return current - (current - target) * progress

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid based on current phase."""
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

        # In later phases, prefer variety to avoid deadlock
        if self._phase >= 2:
            return random.choice(candidates).bid

        # Phase 1: prefer top bids
        top_n = max(1, len(candidates) // self._phase1_top_fraction)
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

        self._update_opponent_tracking(offer, offer_utility)

        threshold = self._compute_threshold(time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, accept best we've seen
        if (
            time > self._deadline_time_threshold
            and offer_utility >= self._opponent_best_for_us
        ):
            if offer_utility >= self._min_threshold:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER