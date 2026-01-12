"""HardHeaded from ANAC 2011."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = [
    "HardHeaded",
]


@dataclass
class BidEntry:
    """A bid with its utility value."""

    utility: float
    bid: Outcome


class HardHeaded(SAONegotiator):
    """
    HardHeaded from ANAC 2011 - 1st place (winning) agent.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This implementation faithfully reproduces HardHeaded's core strategies:

    - Very slow concession using a Boulware-like time-dependent function
    - Frequency-based opponent modeling that tracks issue preferences
    - Bid selection that maximizes estimated opponent utility while meeting targets
    - Acceptance strategy that compares offers to its own concession trajectory

    References:
        Original Genius class: ``agents.anac.y2011.HardHeaded.KLH``

        ANAC 2011: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
    - Uses time-dependent target: p(t) = min + (1 - Fa) * (max - min)
    - Concession factor Fa = Ka + (1 - Ka) * (t / step_point)^(1/e)
    - Very slow concession with default e=0.05 (Boulware-like)
    - Discount factor awareness: adjusts step_point based on discount
    - May offer opponent's best bid if it exceeds current target

    **Acceptance Strategy:**
    - Accepts if offer utility >= lowest utility we have offered
    - Accepts if offer utility >= utility of our next planned bid
    - Falls back to accepting if above minimum utility threshold

    **Opponent Modeling:**
    - Tracks frequency of values selected by opponent for each issue
    - Issues with unchanged values between bids get higher weight
    - Estimates opponent utility using weighted frequency analysis
    - Selects bids that maximize estimated opponent utility among candidates

    Args:
        ka: Initial concession constant (default 0.05)
        e: Concession exponent (default 0.05, very slow concession)
        min_utility: Minimum acceptable utility (default 0.585)
        learning_coef: Learning coefficient for opponent modeling (default 0.2)
        learning_value_addition: Value addition for learning (default 1)
        utility_tolerance: Tolerance for bid selection around target (default 0.01)
        top_selected_bids: Number of top bids to consider for opponent utility (default 4)
        ignore_discount_threshold: Discount factor threshold for standard concession (default 0.9)
        late_concession_exponent: Concession exponent after step point (default 30.0)
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
        ka: float = 0.05,
        e: float = 0.05,
        min_utility: float = 0.585,
        learning_coef: float = 0.2,
        learning_value_addition: int = 1,
        utility_tolerance: float = 0.01,
        top_selected_bids: int = 4,
        ignore_discount_threshold: float = 0.9,
        late_concession_exponent: float = 30.0,
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
        self._ka = ka
        self._e = e
        self._min_utility = min_utility

        # Will be initialized when negotiation starts
        self._outcome_space: SortedOutcomeSpace | None = None
        self._max_util: float = 1.0
        self._discount_factor: float = 1.0
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._opponent_issue_weights: dict[str, float] = {}
        self._opponent_value_counts: dict[str, dict] = {}

        # Bid tracking
        self._my_bids: list[BidEntry] = []
        self._lowest_offered_utility: float = 1.0
        self._opponent_best_bid: Outcome | None = None
        self._opponent_best_bid_utility: float = 0.0
        self._last_opponent_bid: Outcome | None = None

        # Constants from original (now configurable)
        self._learning_coef = learning_coef
        self._learning_value_addition = learning_value_addition
        self._utility_tolerance = utility_tolerance
        self._top_selected_bids = top_selected_bids
        self._ignore_discount_threshold = ignore_discount_threshold
        self._late_concession_exponent = late_concession_exponent

    def _initialize(self) -> None:
        """Initialize the outcome space and utility bounds."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        self._max_util = self._outcome_space.max_utility

        # Get reservation value
        reservation = getattr(self.ufun, "reserved_value", 0.0)
        if reservation is not None and reservation != float("-inf"):
            self._min_utility = max(self._min_utility, reservation)

        # Initialize opponent model with equal weights
        if self.nmi is not None:
            issues = self.nmi.issues
            n_issues = len(issues)
            weight = 1.0 / n_issues if n_issues > 0 else 1.0

            for issue in issues:
                self._opponent_issue_weights[issue.name] = weight
                self._opponent_value_counts[issue.name] = {}

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

    def _update_opponent_model(self) -> None:
        """
        Update opponent model based on their bidding behavior.

        This implements a frequency-based learning approach:
        - Issues that remain unchanged between consecutive bids get higher weight
        - Values that appear frequently are assumed to be more preferred
        """
        if len(self._opponent_bids) < 2:
            return

        if self.nmi is None:
            return

        last_bid = self._opponent_bids[-1]
        second_last_bid = self._opponent_bids[-2]

        issues = self.nmi.issues
        n_issues = len(issues)

        # Count unchanged issues
        unchanged_count = 0
        changed_issues: dict[str, bool] = {}

        for i, issue in enumerate(issues):
            last_val = (
                last_bid[i] if isinstance(last_bid, tuple) else last_bid.get(issue.name)
            )
            prev_val = (
                second_last_bid[i]
                if isinstance(second_last_bid, tuple)
                else second_last_bid.get(issue.name)
            )

            if last_val == prev_val:
                unchanged_count += 1
                changed_issues[issue.name] = False
            else:
                changed_issues[issue.name] = True

        # Update weights based on unchanged issues
        golden_value = self._learning_coef / n_issues
        total_sum = 1.0 + golden_value * unchanged_count
        max_weight = 1.0 - n_issues * golden_value / total_sum

        for issue in issues:
            current_weight = self._opponent_issue_weights[issue.name]
            if not changed_issues[issue.name] and current_weight < max_weight:
                new_weight = (current_weight + golden_value) / total_sum
            else:
                new_weight = current_weight / total_sum
            self._opponent_issue_weights[issue.name] = new_weight

        # Update value counts for frequency-based learning
        for i, issue in enumerate(issues):
            val = (
                last_bid[i] if isinstance(last_bid, tuple) else last_bid.get(issue.name)
            )
            if val is not None:
                val_key = str(val)
                if val_key not in self._opponent_value_counts[issue.name]:
                    self._opponent_value_counts[issue.name][val_key] = 0
                self._opponent_value_counts[issue.name][
                    val_key
                ] += self._learning_value_addition

    def _get_opponent_utility(self, bid: Outcome) -> float:
        """
        Estimate opponent's utility for a bid based on learned model.

        Args:
            bid: The outcome to evaluate.

        Returns:
            Estimated opponent utility in [0, 1].
        """
        if self.nmi is None or not self._opponent_value_counts:
            return 0.5

        issues = self.nmi.issues
        total_utility = 0.0
        total_weight = 0.0

        for i, issue in enumerate(issues):
            weight = self._opponent_issue_weights.get(issue.name, 0.0)
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)

            if val is not None:
                val_key = str(val)
                counts = self._opponent_value_counts.get(issue.name, {})
                value_count = counts.get(val_key, 1)
                max_count = max(counts.values()) if counts else 1

                # Normalize value utility
                value_utility = value_count / max_count if max_count > 0 else 0.5
                total_utility += weight * value_utility

            total_weight += weight

        return total_utility / total_weight if total_weight > 0 else 0.5

    def _get_target_utility(self, t: float) -> float:
        """
        Calculate target utility for current time.

        This implements the HardHeaded concession strategy which takes into
        account the discount factor.

        Args:
            t: Normalized time in [0, 1].

        Returns:
            Target utility value.
        """
        step_point = self._discount_factor

        if step_point >= self._ignore_discount_threshold:
            # Standard concession when discount is high (near 1)
            fa = (
                self._ka + (1 - self._ka) * (t / step_point) ** (1.0 / self._e)
                if self._e > 0
                else self._ka
            )
            return self._min_utility + (1 - fa) * (self._max_util - self._min_utility)

        elif t <= step_point:
            # Before step point: slower concession
            temp_e = self._e / step_point
            fa = (
                self._ka + (1 - self._ka) * (t / step_point) ** (1.0 / temp_e)
                if temp_e > 0
                else self._ka
            )
            temp_min = (
                self._min_utility + abs(self._max_util - self._min_utility) * step_point
            )
            return temp_min + (1 - fa) * (self._max_util - temp_min)

        else:
            # After step point: faster concession
            temp_e = self._late_concession_exponent
            fa = self._ka + (1 - self._ka) * ((t - step_point) / (1 - step_point)) ** (
                1.0 / temp_e
            )
            temp_max = (
                self._min_utility + abs(self._max_util - self._min_utility) * step_point
            )
            return self._min_utility + (1 - fa) * (temp_max - self._min_utility)

    def _select_bid(self, target_utility: float) -> Outcome | None:
        """
        Select a bid near the target utility, preferring bids good for opponent.

        Args:
            target_utility: The target utility to aim for.

        Returns:
            Selected outcome, or None if unavailable.
        """
        if self._outcome_space is None:
            return None

        # Get bids near target utility
        candidates = self._outcome_space.get_bids_in_range(
            target_utility - self._utility_tolerance,
            target_utility + self._utility_tolerance,
        )

        if not candidates:
            # Fall back to getting closest bid
            bid_details = self._outcome_space.get_bid_near_utility(target_utility)
            return bid_details.bid if bid_details else None

        # If we have candidates, select based on opponent utility
        if len(candidates) <= self._top_selected_bids:
            # Score by opponent utility and pick best
            best_bid = None
            best_opp_util = -1.0
            for bd in candidates:
                opp_util = self._get_opponent_utility(bd.bid)
                if opp_util > best_opp_util:
                    best_opp_util = opp_util
                    best_bid = bd.bid
            return best_bid

        # Select top N by opponent utility
        scored = [(bd.bid, self._get_opponent_utility(bd.bid)) for bd in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_bids = scored[: self._top_selected_bids]

        # Return the one with highest opponent utility
        return top_bids[0][0] if top_bids else None

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """
        Generate a proposal based on HardHeaded strategy.

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
                best = self._outcome_space.get_bids_above(self._max_util - 0.001)
                if best:
                    bid = best[0].bid
                    self._my_bids.append(BidEntry(utility=self._max_util, bid=bid))
                    return bid
            return None

        # Calculate target utility
        target = self._get_target_utility(t)

        # Check if opponent's best bid is better than our target
        if (
            self._opponent_best_bid is not None
            and self._opponent_best_bid_utility > target
        ):
            # Prefer opponent's best bid if it's above our target
            self._my_bids.append(
                BidEntry(
                    utility=self._opponent_best_bid_utility, bid=self._opponent_best_bid
                )
            )
            if self._opponent_best_bid_utility < self._lowest_offered_utility:
                self._lowest_offered_utility = self._opponent_best_bid_utility
            return self._opponent_best_bid

        # Get our last offered utility and don't go below it too quickly
        if self._my_bids:
            last_util = self._my_bids[-1].utility
            if target > last_util:
                target = last_util

        # Select bid
        bid = self._select_bid(target)

        if bid is not None and self.ufun is not None:
            bid_util = float(self.ufun(bid))
            self._my_bids.append(BidEntry(utility=bid_util, bid=bid))
            if bid_util < self._lowest_offered_utility:
                self._lowest_offered_utility = bid_util

        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """
        Respond to an offer using HardHeaded acceptance strategy.

        Accepts if:
        - Offer utility >= lowest utility we have offered, OR
        - Offer utility >= what we would offer next

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

        # Store opponent bid and update model
        self._last_opponent_bid = offer
        self._opponent_bids.append(offer)
        self._update_opponent_model()

        offer_utility = float(self.ufun(offer))

        # Track best opponent bid
        if (
            self._opponent_best_bid is None
            or offer_utility > self._opponent_best_bid_utility
        ):
            self._opponent_best_bid = offer
            self._opponent_best_bid_utility = offer_utility

        # Accept if offer is at least as good as what we've been offering
        if offer_utility >= self._lowest_offered_utility:
            return ResponseType.ACCEPT_OFFER

        # Get what we would offer next
        t = state.relative_time
        my_bid = self._select_bid(self._get_target_utility(t))

        if my_bid is None:
            # If we can't make a bid, accept if above minimum
            return (
                ResponseType.ACCEPT_OFFER
                if offer_utility >= self._min_utility
                else ResponseType.REJECT_OFFER
            )

        my_bid_utility = float(self.ufun(my_bid))

        # Accept if offer is at least as good as what we would offer
        if offer_utility >= my_bid_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
