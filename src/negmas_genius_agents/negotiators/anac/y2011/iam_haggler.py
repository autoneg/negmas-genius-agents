"""IAMhaggler2011 from ANAC 2011."""

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

__all__ = ["IAMhaggler2011"]


class IAMhaggler2011(SAONegotiator):
    """
    IAMhaggler2011 from ANAC 2011 - 3rd place agent.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This implementation faithfully reproduces IAMhaggler2011's core strategies:

    - Time-dependent concession with adaptive rate based on opponent behavior
    - Opponent preference estimation using running averages (simplified from GP)
    - Nash-like bid selection that maximizes product of both utilities
    - Multi-criteria acceptance that adapts threshold near the deadline

    References:
        Original Genius class: ``agents.anac.y2011.IAMhaggler2011.IAMhaggler2011``

        ANAC 2011: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
    - Time-dependent concession: u(t) = max - (max - min) * t^(1/e)
    - Adaptive concession rate based on opponent behavior observation
    - Tougher stance when opponent is conceding (reduces e)
    - Faster concession when opponent appears tough (increases e)
    - First round always offers maximum utility bid

    **Acceptance Strategy:**
    - Accept if offer utility >= current target utility
    - Accept if offer utility >= utility of next planned bid
    - Near deadline (t > 0.98): Accept if above minimum utility
    - Very near deadline (t > 0.995): Accept if >= 95% of best received

    **Opponent Modeling:**
    - Tracks running averages of value selections per issue (recency-weighted)
    - Estimates opponent issue weights from selection consistency
    - Uses linear regression on received utilities to estimate concession rate
    - Estimates opponent reservation value from minimum received utility
    - Nash-like scoring: selects bids maximizing (own_util * opponent_util)

    Args:
        max_utility_target: Initial target utility (default 1.0)
        min_utility_target: Minimum acceptable utility (default 0.6)
        concession_rate: Base concession rate parameter (default 0.05)
        deadline_accept_time: Time threshold for deadline acceptance (default 0.98)
        final_accept_time: Time threshold for final acceptance (default 0.995)
        final_accept_factor: Factor for accepting best opponent bid at deadline (default 0.95)
        recency_weight_factor: Factor for recency weighting in opponent model (default 0.1)
        min_bids_for_estimation: Minimum bids before estimating concession (default 5)
        trend_window_size: Number of recent bids for trend analysis (default 10)
        reservation_adjustment: Adjustment for estimated opponent reservation (default 0.1)
        opponent_conceding_threshold: Threshold to detect opponent concession (default 0.01)
        concession_rate_reduction: Rate reduction when opponent concedes (default 0.5)
        opponent_tough_threshold: Threshold to detect tough opponent (default 0.001)
        opponent_tough_bids: Minimum bids to detect tough opponent (default 10)
        concession_rate_increase: Rate increase against tough opponent (default 0.02)
        min_concession_rate: Minimum concession rate (default 0.02)
        max_concession_rate: Maximum concession rate (default 0.2)
        best_opponent_utility_factor: Factor for best opponent utility target (default 0.95)
        bid_tolerance: Tolerance for bid selection around target (default 0.02)
        nash_tie_breaker: Tie-breaking bonus for own utility in Nash scoring (default 0.01)
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
        max_utility_target: float = 1.0,
        min_utility_target: float = 0.6,
        concession_rate: float = 0.05,
        deadline_accept_time: float = 0.98,
        final_accept_time: float = 0.995,
        final_accept_factor: float = 0.95,
        recency_weight_factor: float = 0.1,
        min_bids_for_estimation: int = 5,
        trend_window_size: int = 10,
        reservation_adjustment: float = 0.1,
        opponent_conceding_threshold: float = 0.01,
        concession_rate_reduction: float = 0.5,
        opponent_tough_threshold: float = 0.001,
        opponent_tough_bids: int = 10,
        concession_rate_increase: float = 0.02,
        min_concession_rate: float = 0.02,
        max_concession_rate: float = 0.2,
        best_opponent_utility_factor: float = 0.95,
        bid_tolerance: float = 0.02,
        nash_tie_breaker: float = 0.01,
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
        self._max_utility_target = max_utility_target
        self._min_utility_target = min_utility_target
        self._concession_rate = concession_rate
        self._deadline_accept_time = deadline_accept_time
        self._final_accept_time = final_accept_time
        self._final_accept_factor = final_accept_factor
        self._recency_weight_factor = recency_weight_factor
        self._min_bids_for_estimation = min_bids_for_estimation
        self._trend_window_size = trend_window_size
        self._reservation_adjustment = reservation_adjustment
        self._opponent_conceding_threshold = opponent_conceding_threshold
        self._concession_rate_reduction = concession_rate_reduction
        self._opponent_tough_threshold = opponent_tough_threshold
        self._opponent_tough_bids = opponent_tough_bids
        self._concession_rate_increase = concession_rate_increase
        self._min_concession_rate = min_concession_rate
        self._max_concession_rate = max_concession_rate
        self._best_opponent_utility_factor = best_opponent_utility_factor
        self._bid_tolerance = bid_tolerance
        self._nash_tie_breaker = nash_tie_breaker

        # Will be initialized when negotiation starts
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling - running averages
        self._opponent_bids: list[Outcome] = []
        self._opponent_utilities: list[float] = []  # Our utility for their bids
        self._estimated_opponent_reservation: float = 0.0
        self._opponent_concession_rate: float = 0.0

        # Issue-level opponent preference estimation
        self._opponent_value_sums: dict[str, dict] = {}
        self._opponent_value_counts: dict[str, dict] = {}
        self._opponent_issue_weights: dict[str, float] = {}

        # Tracking
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._my_last_bid: Outcome | None = None
        self._my_last_utility: float = 1.0

    def _initialize(self) -> None:
        """Initialize the outcome space and utility bounds."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)

        # Get reservation value if available
        reservation = getattr(self.ufun, "reserved_value", 0.0)
        if reservation is not None and reservation != float("-inf"):
            self._min_utility_target = max(self._min_utility_target, reservation)

        # Initialize opponent model with equal weights
        if self.nmi is not None:
            issues = self.nmi.issues
            n_issues = len(issues)
            weight = 1.0 / n_issues if n_issues > 0 else 1.0

            for issue in issues:
                self._opponent_issue_weights[issue.name] = weight
                self._opponent_value_sums[issue.name] = {}
                self._opponent_value_counts[issue.name] = {}

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

    def _update_opponent_model(self, bid: Outcome) -> None:
        """
        Update opponent model using running average of bid values.

        This is a simplified version of the GP approach - we track the
        average value selected for each issue to estimate preferences.

        Args:
            bid: The opponent's bid.
        """
        if bid is None or self.nmi is None:
            return

        self._opponent_bids.append(bid)

        # Track our utility for their bids
        if self.ufun is not None:
            utility = float(self.ufun(bid))
            self._opponent_utilities.append(utility)

            if utility > self._best_opponent_utility:
                self._best_opponent_utility = utility
                self._best_opponent_bid = bid

        # Update running averages for each issue value
        issues = self.nmi.issues
        for i, issue in enumerate(issues):
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)
            if val is not None:
                val_key = str(val)
                if val_key not in self._opponent_value_sums[issue.name]:
                    self._opponent_value_sums[issue.name][val_key] = 0.0
                    self._opponent_value_counts[issue.name][val_key] = 0

                # Use recency weighting - more recent bids count more
                recency_weight = 1.0 + self._recency_weight_factor * len(
                    self._opponent_bids
                )
                self._opponent_value_sums[issue.name][val_key] += recency_weight
                self._opponent_value_counts[issue.name][val_key] += 1

        # Update issue weights based on variance in selections
        self._update_issue_weights()

        # Estimate opponent concession rate
        self._estimate_opponent_concession()

    def _update_issue_weights(self) -> None:
        """
        Update estimated opponent issue weights based on selection consistency.

        Issues where the opponent consistently selects the same values are
        assumed to be more important to them.
        """
        if self.nmi is None or len(self._opponent_bids) < 3:
            return

        issues = self.nmi.issues
        consistency_scores: dict[str, float] = {}

        for issue in issues:
            counts = self._opponent_value_counts.get(issue.name, {})
            if not counts:
                consistency_scores[issue.name] = 1.0
                continue

            # Calculate consistency as max_count / total_count
            total = sum(counts.values())
            max_count = max(counts.values()) if counts else 0
            consistency = max_count / total if total > 0 else 0.5
            consistency_scores[issue.name] = consistency

        # Normalize to weights
        total_consistency = sum(consistency_scores.values())
        if total_consistency > 0:
            for issue in issues:
                self._opponent_issue_weights[issue.name] = (
                    consistency_scores[issue.name] / total_consistency
                )

    def _estimate_opponent_concession(self) -> None:
        """
        Estimate opponent's concession rate from their bid history.

        Uses the trend in utilities of opponent bids (from our perspective)
        to estimate how quickly they're conceding.
        """
        if len(self._opponent_utilities) < self._min_bids_for_estimation:
            return

        # Look at recent trend
        recent = self._opponent_utilities[-self._trend_window_size :]
        if len(recent) < 2:
            return

        # Simple linear regression to estimate trend
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator
            # Positive slope means they're conceding (giving us better utility)
            self._opponent_concession_rate = max(0, slope)

        # Estimate opponent reservation value
        # If they're conceding, their reservation is likely lower
        min_received = min(self._opponent_utilities) if self._opponent_utilities else 0
        self._estimated_opponent_reservation = max(
            0, min_received - self._reservation_adjustment
        )

    def _get_opponent_utility(self, bid: Outcome) -> float:
        """
        Estimate opponent's utility for a bid based on learned preferences.

        Args:
            bid: The outcome to evaluate.

        Returns:
            Estimated opponent utility in [0, 1].
        """
        if self.nmi is None or not self._opponent_value_counts:
            return 0.5

        issues = self.nmi.issues
        total_utility = 0.0

        for i, issue in enumerate(issues):
            weight = self._opponent_issue_weights.get(issue.name, 0.0)
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)

            if val is not None:
                val_key = str(val)
                sums = self._opponent_value_sums.get(issue.name, {})
                counts = self._opponent_value_counts.get(issue.name, {})

                if val_key in sums and val_key in counts and counts[val_key] > 0:
                    # Higher sum means more frequently/recently selected
                    max_sum = max(sums.values()) if sums else 1
                    value_preference = sums[val_key] / max_sum if max_sum > 0 else 0.5
                else:
                    # Unknown value - assume moderate preference
                    value_preference = 0.3

                total_utility += weight * value_preference

        return min(1.0, max(0.0, total_utility))

    def _get_target_utility(self, t: float) -> float:
        """
        Calculate target utility using time-dependent strategy with adaptation.

        The concession is adaptive based on:
        1. Base time-dependent concession
        2. Opponent's observed concession rate
        3. Best utility received from opponent

        Args:
            t: Normalized time in [0, 1].

        Returns:
            Target utility value.
        """
        # Base time-dependent concession (Boulware-like)
        # u(t) = max - (max - min) * t^(1/e)
        e = self._concession_rate

        # Adapt concession rate based on opponent behavior
        if self._opponent_concession_rate > self._opponent_conceding_threshold:
            # Opponent is conceding - we can be tougher
            e = max(
                self._min_concession_rate,
                e - self._opponent_concession_rate * self._concession_rate_reduction,
            )
        elif (
            len(self._opponent_bids) > self._opponent_tough_bids
            and self._opponent_concession_rate < self._opponent_tough_threshold
        ):
            # Opponent is tough - concede a bit faster
            e = min(self._max_concession_rate, e + self._concession_rate_increase)

        # Calculate base target
        if e > 0:
            concession = t ** (1.0 / e)
        else:
            concession = 0.0

        target = (
            self._max_utility_target
            - (self._max_utility_target - self._min_utility_target) * concession
        )

        # Don't go below reservation value
        target = max(target, self._min_utility_target)

        # Don't concede faster than we need to
        # If opponent is giving us good bids, we can stay high
        if self._best_opponent_utility > target:
            target = max(
                target, self._best_opponent_utility * self._best_opponent_utility_factor
            )

        return target

    def _select_bid(self, target_utility: float) -> Outcome | None:
        """
        Select a bid that balances own utility with opponent utility.

        Uses Nash-like product maximization among bids near target.

        Args:
            target_utility: The target utility to aim for.

        Returns:
            Selected outcome, or None if unavailable.
        """
        if self._outcome_space is None:
            return None

        tolerance = self._bid_tolerance

        # Get bids near target utility
        candidates = self._outcome_space.get_bids_in_range(
            target_utility - tolerance,
            min(1.0, target_utility + tolerance),
        )

        if not candidates:
            # Fall back to getting closest bid above target
            bid_details = self._outcome_space.get_bid_near_utility(target_utility)
            return bid_details.bid if bid_details else None

        if len(candidates) == 1:
            return candidates[0].bid

        # Score candidates by Nash product (our_util * opponent_util)
        best_bid = None
        best_score = -1.0

        for bd in candidates:
            our_util = bd.utility
            opp_util = self._get_opponent_utility(bd.bid)

            # Nash product with slight preference for our utility
            nash_score = our_util * opp_util
            # Add small bonus for our utility to break ties
            score = nash_score + self._nash_tie_breaker * our_util

            if score > best_score:
                best_score = score
                best_bid = bd.bid

        return best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """
        Generate a proposal using IAMhaggler strategy.

        Args:
            state: Current negotiation state.
            dest: Destination negotiator ID (ignored).

        Returns:
            Outcome to propose, or None.
        """
        if not self._initialized:
            self._initialize()

        t = state.relative_time

        # First round: offer best bid
        if state.step == 0:
            if self._outcome_space is not None:
                best_bids = self._outcome_space.get_bids_above(
                    self._max_utility_target - 0.001
                )
                if best_bids:
                    bid = best_bids[0].bid
                    self._my_last_bid = bid
                    self._my_last_utility = best_bids[0].utility
                    return bid
            return None

        # Calculate target utility
        target = self._get_target_utility(t)

        # Don't go below what we offered before (monotonic concession)
        if target > self._my_last_utility:
            target = self._my_last_utility

        # Select bid
        bid = self._select_bid(target)

        if bid is not None and self.ufun is not None:
            self._my_last_bid = bid
            self._my_last_utility = float(self.ufun(bid))

        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """
        Respond to an offer using IAMhaggler acceptance strategy.

        Accepts if:
        - Offer utility >= our target utility, OR
        - Offer utility >= utility of what we would offer next, OR
        - Near deadline and offer is above minimum

        Args:
            state: Current negotiation state.
            source: Source negotiator ID (ignored).

        Returns:
            ResponseType indicating acceptance or rejection.
        """
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        # Update opponent model
        self._update_opponent_model(offer)

        offer_utility = float(self.ufun(offer))
        t = state.relative_time

        # Calculate our target
        target = self._get_target_utility(t)

        # Accept if offer meets our target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if offer is better than what we would offer
        my_next_bid = self._select_bid(target)
        if my_next_bid is not None:
            my_next_utility = float(self.ufun(my_next_bid))
            if offer_utility >= my_next_utility:
                return ResponseType.ACCEPT_OFFER

        # Near deadline - accept if above minimum
        if t > self._deadline_accept_time and offer_utility >= self._min_utility_target:
            return ResponseType.ACCEPT_OFFER

        # Very near deadline - accept best we've seen
        if (
            t > self._final_accept_time
            and offer_utility >= self._best_opponent_utility * self._final_accept_factor
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
