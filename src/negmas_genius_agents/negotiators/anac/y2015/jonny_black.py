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
        mystery_range: Magnitude of random mystery factor applied to threshold (default 0.08)
        desperation_min_bids: Minimum opponent bids to detect desperation (default 4)
        desperation_window: Number of recent opponent bids checked for desperation (default 4)
        desperation_threshold: Utility improvement over window classifying opponent as desperate (default 0.15)
        desperate_multiplier: Multiplier applied to e when opponent is desperate (default 0.5)
        max_utility_ratio: Ratio of max utility used as firm starting threshold (default 0.92)
        main_phase_target: Utility target at end of main phase / base of end phase (default 0.55)
        middle_phase_floor: Floor utility in middle phase (default 0.5)
        end_phase_target: Utility target at end of end phase (default 0.45)
        end_phase_concession_factor: Fraction of concession applied in end phase (default 0.5)
        min_utility_margin: Margin above min utility used as end-game floor (default 0.1)
        fallback_threshold_ratio: Ratio used when lowering threshold if no candidates (default 0.85)
        endgame_opponent_ratio: Ratio of best opponent utility used in end-game acceptance (default 0.95)
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
        mystery_range: float = 0.08,
        desperation_min_bids: int = 4,
        desperation_window: int = 4,
        desperation_threshold: float = 0.15,
        desperate_multiplier: float = 0.5,
        max_utility_ratio: float = 0.92,
        main_phase_target: float = 0.55,
        middle_phase_floor: float = 0.5,
        end_phase_target: float = 0.45,
        end_phase_concession_factor: float = 0.5,
        min_utility_margin: float = 0.1,
        fallback_threshold_ratio: float = 0.85,
        endgame_opponent_ratio: float = 0.95,
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
        self._mystery_range = mystery_range
        self._desperation_min_bids = desperation_min_bids
        self._desperation_window = desperation_window
        self._desperation_threshold = desperation_threshold
        self._desperate_multiplier = desperate_multiplier
        self._max_utility_ratio = max_utility_ratio
        self._main_phase_target = main_phase_target
        self._middle_phase_floor = middle_phase_floor
        self._end_phase_target = end_phase_target
        self._end_phase_concession_factor = end_phase_concession_factor
        self._min_utility_margin = min_utility_margin
        self._fallback_threshold_ratio = fallback_threshold_ratio
        self._endgame_opponent_ratio = endgame_opponent_ratio
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
        self._mystery_factor = random.uniform(-self._mystery_range, self._mystery_range)

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent and detect desperation."""
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Detect desperation: opponent rapidly conceding
        if len(self._opponent_bids) >= self._desperation_min_bids:
            recent = [u for _, u in self._opponent_bids[-self._desperation_window :]]
            if recent[-1] - recent[0] > self._desperation_threshold:
                self._opponent_desperate = True

    def _compute_threshold(self, time: float) -> float:
        """Compute threshold with mysterious behavior."""
        e = self._e

        # Exploit desperate opponent
        if self._opponent_desperate:
            e *= self._desperate_multiplier

        if time < self._early_time_threshold:
            # Early: firm but mysterious
            base = self._max_utility * self._max_utility_ratio
            return base + self._mystery_factor
        elif time < self._main_time_threshold:
            # Middle: unpredictable concession
            progress = (time - self._early_time_threshold) / (
                self._main_time_threshold - self._early_time_threshold
            )
            f_t = math.pow(progress, 1 / e)
            base = (
                self._max_utility * self._max_utility_ratio
                - (self._max_utility * self._max_utility_ratio - self._main_phase_target)
                * f_t
            )
            return max(base + self._mystery_factor, self._middle_phase_floor)
        else:
            # End: deal-making mode
            progress = (time - self._main_time_threshold) / (
                1.0 - self._main_time_threshold
            )
            base = (
                self._main_phase_target
                - (self._main_phase_target - self._end_phase_target)
                * progress
                * self._end_phase_concession_factor
            )
            return max(base + self._mystery_factor, self._min_utility + self._min_utility_margin)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid with strategic variance."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        self._update_mystery()

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(
                threshold * self._fallback_threshold_ratio
            )

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
                self._best_opponent_utility * self._endgame_opponent_ratio,
                self._min_utility + self._min_utility_margin,
            ):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER