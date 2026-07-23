"""XianFaAgent from ANAC 2015."""

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

__all__ = ["XianFaAgent"]


class XianFaAgent(SAONegotiator):
    """
    XianFaAgent negotiation agent from ANAC 2015.

    XianFaAgent uses a constitutional approach with fixed rules in early
    negotiation and strategic reserves for the final phase.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.xianfa.XianFaAgent

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Four constitutional rules:
          * Rule 1 (t<0.2): Start high at 95% of max utility
          * Rule 2 (0.2<t<0.5): Slow Boulware concession of 15%
          * Rule 3 (0.5<t<0.8): Adapt to opponent, concede toward
            max(best_opponent + 0.05, constitutional_min)
          * Rule 4 (t>0.8): Final 50% concession toward constitutional minimum
        - Constitutional minimum of 60% utility
        - Reserve bids (10 bids in 50-65% range) used in final phase

    **Acceptance Strategy:**
        - Constitutional acceptance: accepts if offer >= threshold
        - Constitutional minimum is absolute until t=0.95
        - Near deadline (t>0.98): Accepts any offer >= constitutional minimum

    **Opponent Modeling:**
        - Tracks opponent bids and best utility offered
        - Uses best opponent utility to adjust Rule 3 target
        - No complex preference estimation

    Args:
        e: Concession exponent (default 0.2)
        rule1_time_threshold: Time threshold for Rule 1 - start high (default 0.2)
        rule2_time_threshold: Time threshold for Rule 2 - slow concession (default 0.5)
        rule3_time_threshold: Time threshold for Rule 3 - adapt to opponent (default 0.8)
        constitutional_min: Constitutional minimum utility (default 0.6)
        reserve_range: Utility range above mid used to build reserve bids (default 0.15)
        reserve_count: Number of reserve bids stored for end game (default 10)
        max_utility_ratio: Ratio of max utility used as starting threshold (default 0.95)
        rule2_concession: Utility conceded during Rule 2 (default 0.15)
        rule3_base_ratio: Ratio of max utility used as Rule 3 base (default 0.8)
        opponent_best_bonus: Bonus above best opponent utility for Rule 3/4 target (default 0.05)
        rule4_concession_factor: Fraction of concession applied in Rule 4 (default 0.5)
        final_phase_time_threshold: Time after which reserve bids are used (default 0.9)
        constitutional_min_absolute_time: Time until which constitutional min is absolute (default 0.95)
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
        e: float = 0.2,
        rule1_time_threshold: float = 0.2,
        rule2_time_threshold: float = 0.5,
        rule3_time_threshold: float = 0.8,
        constitutional_min: float = 0.6,
        reserve_range: float = 0.15,
        reserve_count: int = 10,
        max_utility_ratio: float = 0.95,
        rule2_concession: float = 0.15,
        rule3_base_ratio: float = 0.8,
        opponent_best_bonus: float = 0.05,
        rule4_concession_factor: float = 0.5,
        final_phase_time_threshold: float = 0.9,
        constitutional_min_absolute_time: float = 0.95,
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
        self._rule1_time_threshold = rule1_time_threshold
        self._rule2_time_threshold = rule2_time_threshold
        self._rule3_time_threshold = rule3_time_threshold
        self._constitutional_min = constitutional_min
        self._reserve_range = reserve_range
        self._reserve_count = reserve_count
        self._max_utility_ratio = max_utility_ratio
        self._rule2_concession = rule2_concession
        self._rule3_base_ratio = rule3_base_ratio
        self._opponent_best_bonus = opponent_best_bonus
        self._rule4_concession_factor = rule4_concession_factor
        self._final_phase_time_threshold = final_phase_time_threshold
        self._constitutional_min_absolute_time = constitutional_min_absolute_time
        self._deadline_time_threshold = deadline_time_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0

        # Tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._best_opponent_utility: float = 0.0
        self._reserve_bids: list[Outcome] = []

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

            # Build reserve bids - good bids saved for end game
            mid_util = (self._max_utility + self._min_utility) / 2
            reserves = self._outcome_space.get_bids_in_range(mid_util, mid_util + self._reserve_range)
            self._reserve_bids = [bd.bid for bd in reserves[: self._reserve_count]]

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._best_opponent_utility = 0.0

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent offers."""
        self._opponent_bids.append((bid, utility))
        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

    def _compute_threshold(self, time: float) -> float:
        """Compute threshold following constitutional rules."""
        # Constitutional rules by phase
        if time < self._rule1_time_threshold:
            # Rule 1: Start high
            return self._max_utility * self._max_utility_ratio
        elif time < self._rule2_time_threshold:
            # Rule 2: Slow concession
            progress = (time - self._rule1_time_threshold) / (
                self._rule2_time_threshold - self._rule1_time_threshold
            )
            f_t = math.pow(progress, 1 / self._e)
            return self._max_utility * self._max_utility_ratio - self._rule2_concession * f_t
        elif time < self._rule3_time_threshold:
            # Rule 3: Adapt to opponent
            base = self._max_utility * self._rule3_base_ratio
            target = max(
                self._best_opponent_utility + self._opponent_best_bonus,
                self._constitutional_min,
            )
            progress = (time - self._rule2_time_threshold) / (
                self._rule3_time_threshold - self._rule2_time_threshold
            )
            return base - (base - target) * progress
        else:
            # Rule 4: Final concession with constitutional minimum
            progress = (time - self._rule3_time_threshold) / (
                1.0 - self._rule3_time_threshold
            )
            current = max(
                self._best_opponent_utility + self._opponent_best_bonus,
                self._constitutional_min,
            )
            target = self._constitutional_min
            return current - (current - target) * progress * self._rule4_concession_factor

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid following constitutional approach."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)

        # In final phase, use reserves if available
        if time > self._final_phase_time_threshold and self._reserve_bids:
            for bid in self._reserve_bids:
                if self.ufun is not None:
                    util = float(self.ufun(bid))
                    if util >= threshold:
                        return bid

        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(self._constitutional_min)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

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

        # Constitutional acceptance
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Constitutional minimum is absolute
        if (
            offer_utility < self._constitutional_min
            and time < self._constitutional_min_absolute_time
        ):
            return ResponseType.REJECT_OFFER

        # Near deadline exceptions
        if (
            time > self._deadline_time_threshold
            and offer_utility >= self._constitutional_min
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER