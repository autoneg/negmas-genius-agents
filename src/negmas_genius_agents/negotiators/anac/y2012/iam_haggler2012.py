"""IAMhaggler2012 from ANAC 2012."""

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

__all__ = ["IAMhaggler2012"]


class IAMhaggler2012(SAONegotiator):
    """
    IAMhaggler2012 negotiation agent from ANAC 2012.

    IAMhaggler2012 is an improved version of IAMhaggler (from ANAC 2010/2011).
    It features enhanced opponent modeling using Gaussian Process-inspired
    approaches and Nash-product based bid selection.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        Original Genius class: ``agents.anac.y2012.IAMhaggler2012.IAMhaggler2012``

        ANAC 2012: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        Uses adaptive time-dependent Boulware concession with Nash optimization:

        - First round: Offers best available bid (maximum utility).
        - Subsequent rounds: Target utility computed as:
          target = max - (max - min) * t^(1/e)
          where e (concession rate) adapts based on opponent behavior.
        - If opponent is conceding (rate > 0.01): e reduced to 70% (tougher).
        - If opponent is tough (rate < 0.001 after 10 bids): e increased to
          150% (more flexible).
        - Maintains monotonic concession (never bids higher than previous).

        Bid selection uses Nash-product maximization:
        - Scores candidates by: own_utility * estimated_opponent_utility
        - Slight preference for own utility: score += 0.01 * own_utility
        - Selects bid with highest Nash score from candidates near target.

    **Acceptance Strategy:**
        Multi-criteria acceptance with end-game handling:

        - Accept if offer utility >= target utility.
        - Accept if offer >= utility of next bid we would propose.
        - Near deadline (t > 0.97): Accept if offer >= minimum target.
        - Very near deadline (t > 0.995): Accept if offer >= 95% of best
          opponent bid.

    **Opponent Modeling:**
        Frequency-based preference estimation with consistency weighting:

        - Tracks value selection frequency per issue with recency weighting
          (more recent bids count more).
        - Estimates issue weights from value consistency: issues with more
          consistent value selections are assumed more important.
        - Estimates opponent utility for bids using learned preferences.
        - Uses linear regression on recent utilities to estimate opponent
          concession rate and predict future behavior.
        - Estimates opponent reservation value from minimum received utility.

    Args:
        max_utility_target: Initial target utility (default 1.0).
        min_utility_target: Minimum acceptable utility (default 0.55).
        concession_rate: Base concession rate exponent (default 0.04).
            Lower values result in tougher (slower) concession.
        recency_weight_factor: Factor for recency weighting in opponent model
            (default 0.05).
        opponent_conceding_threshold: Threshold for detecting opponent concession
            (default 0.01).
        opponent_tough_threshold: Threshold for detecting tough opponent
            (default 0.001).
        min_opponent_bids_for_tough: Minimum bids to detect tough opponent
            (default 10).
        concession_rate_tough_factor: Factor to multiply concession rate when
            opponent is conceding (default 0.7).
        concession_rate_flexible_factor: Factor to multiply concession rate when
            opponent is tough (default 1.5).
        min_concession_rate: Minimum concession rate (default 0.02).
        max_concession_rate: Maximum concession rate (default 0.15).
        best_opponent_multiplier: Multiplier for best opponent utility in
            target calculation (default 0.95).
        bid_tolerance: Tolerance for bid selection range (default 0.025).
        nash_own_utility_bonus: Bonus for own utility in Nash score (default 0.01).
        near_deadline_time: Time threshold for accepting minimum target
            (default 0.97).
        final_deadline_time: Time threshold for accepting best opponent bid
            (default 0.995).
        final_accept_multiplier: Multiplier for best opponent utility at final
            deadline (default 0.95).
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
        min_utility_target: float = 0.55,
        concession_rate: float = 0.04,
        recency_weight_factor: float = 0.05,
        opponent_conceding_threshold: float = 0.01,
        opponent_tough_threshold: float = 0.001,
        min_opponent_bids_for_tough: int = 10,
        concession_rate_tough_factor: float = 0.7,
        concession_rate_flexible_factor: float = 1.5,
        min_concession_rate: float = 0.02,
        max_concession_rate: float = 0.15,
        best_opponent_multiplier: float = 0.95,
        bid_tolerance: float = 0.025,
        nash_own_utility_bonus: float = 0.01,
        near_deadline_time: float = 0.97,
        final_deadline_time: float = 0.995,
        final_accept_multiplier: float = 0.95,
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
        self._base_concession_rate = concession_rate
        self._concession_rate = concession_rate
        self._recency_weight_factor = recency_weight_factor
        self._opponent_conceding_threshold = opponent_conceding_threshold
        self._opponent_tough_threshold = opponent_tough_threshold
        self._min_opponent_bids_for_tough = min_opponent_bids_for_tough
        self._concession_rate_tough_factor = concession_rate_tough_factor
        self._concession_rate_flexible_factor = concession_rate_flexible_factor
        self._min_concession_rate = min_concession_rate
        self._max_concession_rate = max_concession_rate
        self._best_opponent_multiplier = best_opponent_multiplier
        self._bid_tolerance = bid_tolerance
        self._nash_own_utility_bonus = nash_own_utility_bonus
        self._near_deadline_time = near_deadline_time
        self._final_deadline_time = final_deadline_time
        self._final_accept_multiplier = final_accept_multiplier

        # Will be initialized when negotiation starts
        self._outcome_space: SortedOutcomeSpace | None = None
        self._max_utility: float = 1.0
        self._reservation_value: float = 0.0
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._opponent_utilities: list[float] = []  # Our utility for their bids
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._estimated_opponent_reservation: float = 0.0
        self._opponent_concession_rate: float = 0.0

        # Issue-level opponent preference estimation
        self._opponent_value_freq: dict[str, dict] = {}  # issue -> {value: frequency}
        self._opponent_issue_weights: dict[str, float] = {}

        # Own bidding state
        self._my_last_bid: Outcome | None = None
        self._my_last_utility: float = 1.0
        self._target_utility: float = 1.0

    def _initialize(self) -> None:
        """Initialize the outcome space and opponent model."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
        else:
            self._max_utility = 1.0

        # Get reservation value if available
        reservation = getattr(self.ufun, "reserved_value", 0.0)
        if reservation is not None and reservation != float("-inf"):
            self._reservation_value = max(0.0, reservation)
            self._min_utility_target = max(
                self._min_utility_target, self._reservation_value
            )

        # Initialize opponent model with equal weights
        if self.nmi is not None:
            issues = self.nmi.issues
            n_issues = len(issues)
            weight = 1.0 / n_issues if n_issues > 0 else 1.0

            for issue in issues:
                self._opponent_issue_weights[issue.name] = weight
                self._opponent_value_freq[issue.name] = {}

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_bids = []
        self._opponent_utilities = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._estimated_opponent_reservation = 0.0
        self._opponent_concession_rate = 0.0
        self._concession_rate = self._base_concession_rate
        self._my_last_bid = None
        self._my_last_utility = 1.0
        self._target_utility = self._max_utility

        # Reset opponent model
        if self.nmi is not None:
            for issue in self.nmi.issues:
                self._opponent_value_freq[issue.name] = {}

    def _update_opponent_model(self, bid: Outcome) -> None:
        """
        Update opponent model with a new bid.

        Uses frequency-based approach to estimate opponent preferences.
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

        # Update value frequencies with recency weighting
        issues = self.nmi.issues
        recency_weight = 1.0 + self._recency_weight_factor * len(self._opponent_bids)

        for i, issue in enumerate(issues):
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)
            if val is not None:
                val_key = str(val)
                if val_key not in self._opponent_value_freq[issue.name]:
                    self._opponent_value_freq[issue.name][val_key] = 0.0
                self._opponent_value_freq[issue.name][val_key] += recency_weight

        # Update issue weights based on selection consistency
        self._update_issue_weights()

        # Estimate opponent concession rate
        self._estimate_opponent_concession()

    def _update_issue_weights(self) -> None:
        """
        Update estimated opponent issue weights based on value consistency.

        Issues with more consistent value selections are assumed more important.
        """
        if self.nmi is None or len(self._opponent_bids) < 3:
            return

        issues = self.nmi.issues
        consistency_scores: dict[str, float] = {}

        for issue in issues:
            freqs = self._opponent_value_freq.get(issue.name, {})
            if not freqs:
                consistency_scores[issue.name] = 1.0
                continue

            # Calculate consistency as max_freq / total_freq
            total = sum(freqs.values())
            max_freq = max(freqs.values()) if freqs else 0
            consistency = max_freq / total if total > 0 else 0.5
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
        """
        if len(self._opponent_utilities) < 5:
            return

        # Look at recent trend in utilities we receive
        recent = self._opponent_utilities[-10:]
        if len(recent) < 2:
            return

        # Linear regression to estimate trend
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator
            # Positive slope means they're improving offers to us
            self._opponent_concession_rate = max(0, slope)

        # Estimate opponent reservation value
        min_received = min(self._opponent_utilities) if self._opponent_utilities else 0
        self._estimated_opponent_reservation = max(0, min_received - 0.1)

    def _get_opponent_utility(self, bid: Outcome) -> float:
        """
        Estimate opponent's utility for a bid based on learned preferences.
        """
        if self.nmi is None or not self._opponent_value_freq:
            return 0.5

        issues = self.nmi.issues
        total_utility = 0.0

        for i, issue in enumerate(issues):
            weight = self._opponent_issue_weights.get(issue.name, 0.0)
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)

            if val is not None:
                val_key = str(val)
                freqs = self._opponent_value_freq.get(issue.name, {})

                if val_key in freqs:
                    # Higher frequency means more preferred
                    max_freq = max(freqs.values()) if freqs else 1
                    value_preference = (
                        freqs[val_key] / max_freq if max_freq > 0 else 0.5
                    )
                else:
                    # Unknown value - assume low preference
                    value_preference = 0.2

                total_utility += weight * value_preference

        return min(1.0, max(0.0, total_utility))

    def _get_target_utility(self, t: float) -> float:
        """
        Calculate target utility using adaptive time-dependent strategy.
        """
        # Adapt concession rate based on opponent behavior
        e = self._concession_rate

        if self._opponent_concession_rate > self._opponent_conceding_threshold:
            # Opponent is conceding - we can be tougher
            e = max(
                self._min_concession_rate,
                self._base_concession_rate * self._concession_rate_tough_factor,
            )
        elif (
            len(self._opponent_bids) > self._min_opponent_bids_for_tough
            and self._opponent_concession_rate < self._opponent_tough_threshold
        ):
            # Opponent is tough - concede a bit faster
            e = min(
                self._max_concession_rate,
                self._base_concession_rate * self._concession_rate_flexible_factor,
            )

        # Boulware-like concession: u(t) = max - (max - min) * t^(1/e)
        if e > 0:
            concession = math.pow(t, 1.0 / e)
        else:
            concession = 0.0

        target = (
            self._max_utility_target
            - (self._max_utility_target - self._min_utility_target) * concession
        )

        # Don't go below reservation value
        target = max(target, self._min_utility_target)

        # If opponent is giving us good bids, stay high
        if self._best_opponent_utility > target:
            target = max(
                target, self._best_opponent_utility * self._best_opponent_multiplier
            )

        return target

    def _select_bid(self, target_utility: float) -> Outcome | None:
        """
        Select a bid using Nash-product maximization.

        Balances own utility with estimated opponent utility.
        """
        if self._outcome_space is None:
            return None

        # Get bids near target utility
        candidates = self._outcome_space.get_bids_in_range(
            target_utility - self._bid_tolerance,
            min(1.0, target_utility + self._bid_tolerance),
        )

        if not candidates:
            bid_details = self._outcome_space.get_bid_near_utility(target_utility)
            return bid_details.bid if bid_details else None

        if len(candidates) == 1:
            return candidates[0].bid

        # Score candidates by Nash product
        best_bid = None
        best_score = -1.0

        for bd in candidates:
            our_util = bd.utility
            opp_util = self._get_opponent_utility(bd.bid)

            # Nash product with slight preference for our utility
            nash_score = our_util * opp_util
            score = nash_score + self._nash_own_utility_bonus * our_util

            if score > best_score:
                best_score = score
                best_bid = bd.bid

        return best_bid

    def _accept_condition(self, offer: Outcome, time: float) -> bool:
        """
        Decide whether to accept an offer.
        """
        if self.ufun is None or offer is None:
            return False

        offer_utility = float(self.ufun(offer))

        # Accept if offer meets our target
        if offer_utility >= self._target_utility:
            return True

        # Accept if offer is better than what we would offer
        my_next_bid = self._select_bid(self._target_utility)
        if my_next_bid is not None:
            my_next_utility = float(self.ufun(my_next_bid))
            if offer_utility >= my_next_utility:
                return True

        # Near deadline - accept if above minimum
        if (
            time > self._near_deadline_time
            and offer_utility >= self._min_utility_target
        ):
            return True

        # Very near deadline - accept best we've seen
        if (
            time > self._final_deadline_time
            and offer_utility
            >= self._best_opponent_utility * self._final_accept_multiplier
        ):
            return True

        return False

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """
        Generate a proposal using IAMhaggler2012 strategy.

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
        self._target_utility = self._get_target_utility(t)

        # Don't go below what we offered before (monotonic concession)
        if self._target_utility > self._my_last_utility:
            self._target_utility = self._my_last_utility

        # Select bid
        bid = self._select_bid(self._target_utility)

        if bid is not None and self.ufun is not None:
            self._my_last_bid = bid
            self._my_last_utility = float(self.ufun(bid))

        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """
        Respond to an offer using IAMhaggler2012 acceptance strategy.

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

        t = state.relative_time

        # Calculate our target
        self._target_utility = self._get_target_utility(t)

        if self._accept_condition(offer, t):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
