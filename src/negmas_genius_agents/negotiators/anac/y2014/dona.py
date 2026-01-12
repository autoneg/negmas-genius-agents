"""DoNA from ANAC 2014."""

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

__all__ = ["DoNA"]


class DoNA(SAONegotiator):
    """
    DoNA (Deadline-oriented Negotiation Agent) - 2nd Place ANAC 2014.

    DoNA achieved second place in ANAC 2014 by adapting its strategy based
    on domain analysis and discount factor. It uses statistical sampling for
    large domains and priority-based decision making for efficient negotiation.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.DoNA.DoNA

    **Offering Strategy:**
        Discount-factor-aware concession strategy:
        - High discount (> 0.8): Last-moment strategy. Stay at best utility
          until t > 0.95, then rapidly concede through top 30% of outcome space.
        - Low discount (< 0.2): Fast concession. Linear progression through
          top 50% of outcomes over full negotiation time.
        - Medium discount: Scaled time approach. Time factor =
          1 / (((discount - 0.2) / 0.6) * 0.5 + 0.5), progress through top 40%.

        Domain analysis computes statistical thresholds:
        min_utility = min(2.4 * stdev + avg * 1.2, 0.95)
        Near deadline (t > 0.99), may offer opponent's best bid if acceptable.

    **Acceptance Strategy:**
        Priority-based decision hierarchy (EndNego > Accept > Counter):
        1. Accept if offer >= computed minimum threshold AND >= reservation
        2. Accept if offer >= next planned bid utility
        3. Accept near deadline (t > 0.99) if offer >= reservation value
        The priority structure ensures agreement when beneficial while
        maintaining minimum acceptable outcomes.

    **Opponent Modeling:**
        Time-weighted frequency analysis for preference learning:
        - Unique opponent bids tracked (no duplicates)
        - Frequencies weighted by time * 100 (heavy late emphasis)
        - Best opponent bid saved for potential late-game reciprocity
        - Model informs understanding of opponent flexibility
        - Best opponent offer may be returned near deadline as safe fallback

    Args:
        sample_size: Maximum bids to sample for domain analysis (default 100000).
        last_moment_time: Time threshold for last-moment rapid concession (default 0.95).
        deadline_time: Time threshold for deadline-based decisions (default 0.99).
        stdev_multiplier: Multiplier for standard deviation in min utility calculation (default 2.4).
        avg_multiplier: Multiplier for average utility in min utility calculation (default 1.2).
        max_utility_cap: Maximum cap for computed min utility threshold (default 0.95).
        time_weight_factor: Base multiplier for time-based frequency weighting (default 100).
        high_discount_threshold: Discount factor threshold for last-moment strategy (default 0.8).
        low_discount_threshold: Discount factor threshold for fast concession (default 0.2).
        high_discount_progress_ratio: Progress ratio for high discount rapid concession (default 0.3).
        low_discount_progress_ratio: Progress ratio for low discount concession (default 0.5).
        medium_discount_progress_ratio: Progress ratio for medium discount concession (default 0.4).
        medium_discount_time_base: Base value for medium discount time factor (default 0.5).
        medium_discount_time_scale: Scale factor for medium discount time calculation (default 0.5).
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
        sample_size: int = 100000,
        last_moment_time: float = 0.95,
        deadline_time: float = 0.99,
        stdev_multiplier: float = 2.4,
        avg_multiplier: float = 1.2,
        max_utility_cap: float = 0.95,
        time_weight_factor: float = 100.0,
        high_discount_threshold: float = 0.8,
        low_discount_threshold: float = 0.2,
        high_discount_progress_ratio: float = 0.3,
        low_discount_progress_ratio: float = 0.5,
        medium_discount_progress_ratio: float = 0.4,
        medium_discount_time_base: float = 0.5,
        medium_discount_time_scale: float = 0.5,
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
        self._sample_size = sample_size
        self._last_moment_time = last_moment_time
        self._deadline_time = deadline_time
        self._stdev_multiplier = stdev_multiplier
        self._avg_multiplier = avg_multiplier
        self._max_utility_cap = max_utility_cap
        self._time_weight_factor = time_weight_factor
        self._high_discount_threshold = high_discount_threshold
        self._low_discount_threshold = low_discount_threshold
        self._high_discount_progress_ratio = high_discount_progress_ratio
        self._low_discount_progress_ratio = low_discount_progress_ratio
        self._medium_discount_progress_ratio = medium_discount_progress_ratio
        self._medium_discount_time_base = medium_discount_time_base
        self._medium_discount_time_scale = medium_discount_time_scale
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Domain analysis
        self._avg_utility: float = 0.5
        self._stdev: float = 0.1
        self._min_utility: float = 0.5
        self._reservation_value: float = 0.0
        self._discount_factor: float = 1.0

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_value_weights: dict[int, dict] = {}

        # Bidding state
        self._offer_count: int = 0
        self._sorted_utilities: list[float] = []
        self._utility_to_bid: dict[float, Outcome] = {}

    def _initialize(self) -> None:
        """Initialize by sampling domain."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if not self._outcome_space.outcomes:
            return

        # Compute statistics from samples
        utilities = [
            bd.utility
            for bd in self._outcome_space.outcomes[
                : min(self._sample_size, len(self._outcome_space.outcomes))
            ]
        ]
        self._avg_utility = sum(utilities) / len(utilities) if utilities else 0.5
        variance = (
            sum((u - self._avg_utility) ** 2 for u in utilities) / len(utilities)
            if utilities
            else 0.01
        )
        self._stdev = math.sqrt(variance)

        # Compute minimum utility threshold
        self._min_utility = min(
            self._stdev_multiplier * self._stdev
            + self._avg_utility * self._avg_multiplier,
            self._max_utility_cap,
        )

        # Build sorted utility list
        for bd in self._outcome_space.outcomes:
            self._sorted_utilities.append(bd.utility)
            self._utility_to_bid[bd.utility] = bd.bid

        self._sorted_utilities.sort(reverse=True)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_value_weights = {}
        self._offer_count = 0

    def _update_opponent_model(self, bid: Outcome, time: float) -> None:
        """Update opponent model with weighted frequencies."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))

        # Unique bid tracking
        bid_tuple = tuple(bid)
        if bid_tuple not in [tuple(b) for b in self._opponent_bids]:
            self._opponent_bids.append(bid)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        # Weight increases with time (later bids matter more)
        weight = time * self._time_weight_factor

        # Update value weights
        for i, value in enumerate(bid):
            if i not in self._opponent_value_weights:
                self._opponent_value_weights[i] = {}
            self._opponent_value_weights[i][value] = (
                self._opponent_value_weights[i].get(value, 0) + weight
            )

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid using concession strategy."""
        if not self._sorted_utilities:
            return None

        # Calculate target utility based on time
        # High discount: wait until end
        # Low discount: concede faster
        if self._discount_factor > self._high_discount_threshold:
            # Last moment strategy - stay high until end
            if time < self._last_moment_time:
                target = self._sorted_utilities[0]  # Best bid
            else:
                # Rapid concession at end
                progress = (time - self._last_moment_time) / (
                    1.0 - self._last_moment_time
                )
                idx = int(
                    progress
                    * len(self._sorted_utilities)
                    * self._high_discount_progress_ratio
                )
                target = self._sorted_utilities[
                    min(idx, len(self._sorted_utilities) - 1)
                ]
        elif self._discount_factor < self._low_discount_threshold:
            # Fast concession
            progress = time
            idx = int(
                progress
                * len(self._sorted_utilities)
                * self._low_discount_progress_ratio
            )
            target = self._sorted_utilities[min(idx, len(self._sorted_utilities) - 1)]
        else:
            # Medium - scaled time
            discount_range = (
                self._high_discount_threshold - self._low_discount_threshold
            )
            time_factor = 1 / (
                (
                    (self._discount_factor - self._low_discount_threshold)
                    / discount_range
                )
                * self._medium_discount_time_scale
                + self._medium_discount_time_base
            )
            scaled_time = min(1.0, time * time_factor)
            idx = int(
                scaled_time
                * len(self._sorted_utilities)
                * self._medium_discount_progress_ratio
            )
            target = self._sorted_utilities[min(idx, len(self._sorted_utilities) - 1)]

        # Ensure above minimum
        target = max(target, self._min_utility)

        # Find bid near target
        if target in self._utility_to_bid:
            return self._utility_to_bid[target]

        # Otherwise find closest
        for util in self._sorted_utilities:
            if util <= target:
                return self._utility_to_bid.get(util)

        return (
            self._utility_to_bid.get(self._sorted_utilities[0])
            if self._sorted_utilities
            else None
        )

    def _choose_best_opponent_bid(self) -> Outcome | None:
        """Choose bid from opponent history that maximizes our utility."""
        return self._best_opponent_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        self._offer_count += 1

        # Consider offering opponent's best bid near deadline
        if time > self._deadline_time and self._best_opponent_bid is not None:
            if self._best_opponent_utility >= self._min_utility:
                return self._best_opponent_bid

        return self._select_bid(time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond using priority-based decision."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))
        self._update_opponent_model(offer, time)

        my_next_bid = self._select_bid(time)
        my_next_utility = (
            float(self.ufun(my_next_bid)) if my_next_bid and self.ufun else 0.0
        )

        # Priority-based decision
        # 1. Accept if offer exceeds our min threshold
        if (
            offer_utility >= self._min_utility
            and offer_utility >= self._reservation_value
        ):
            return ResponseType.ACCEPT_OFFER

        # 2. Accept if offer >= our next bid utility
        if offer_utility >= my_next_utility:
            return ResponseType.ACCEPT_OFFER

        # 3. Accept near deadline if reasonable
        if time > self._deadline_time and offer_utility >= self._reservation_value:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
