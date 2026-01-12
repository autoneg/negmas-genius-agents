"""BRAMAgent from ANAC 2011."""

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

__all__ = ["BramAgent"]


class BramAgent(SAONegotiator):
    """
    BRAMAgent from ANAC 2011.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This implementation faithfully reproduces BRAMAgent's core strategies:

    - Boulware-like time-dependent concession function with configurable beta
    - Frequency-based opponent modeling tracking value selections per issue
    - Bid selection that prefers outcomes estimated to be good for the opponent
    - Acceptance that compares offers to both target utility and planned offers

    References:
        Original Genius class: ``agents.anac.y2011.BramAgent.BRAMAgent``

        ANAC 2011: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
    - Boulware-like concession: target(t) = max - (max - min) * t^(1/beta)
    - Higher beta = slower concession (default 0.1 is quite slow)
    - First round always offers maximum utility bid
    - Monotonic concession: never offers above previous offer's utility
    - Prefers bids estimated to be good for opponent among candidates

    **Acceptance Strategy:**
    - Accept if offer utility >= current target utility
    - Accept if offer utility >= utility of next planned bid
    - Near deadline (t > 0.95): Accept if above minimum utility threshold

    **Opponent Modeling:**
    - Tracks frequency of values selected per issue by opponent
    - Estimates issue weights from selection consistency
    - Higher frequency of a value indicates stronger preference
    - Scores candidate bids by estimated opponent utility
    - Selects bid with highest opponent utility among those meeting target

    Args:
        beta: Concession rate parameter (default 0.1, slow concession)
        min_utility: Minimum acceptable utility (default 0.6)
        bid_tolerance: Tolerance range for bid selection around target (default 0.02)
        deadline_acceptance_time: Time threshold for deadline acceptance (default 0.95)
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
        beta: float = 0.1,
        min_utility: float = 0.6,
        bid_tolerance: float = 0.02,
        deadline_acceptance_time: float = 0.95,
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
        self._beta = beta
        self._min_utility = min_utility
        self._bid_tolerance = bid_tolerance
        self._deadline_acceptance_time = deadline_acceptance_time

        # Will be initialized when negotiation starts
        self._outcome_space: SortedOutcomeSpace | None = None
        self._max_util: float = 1.0
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._opponent_issue_frequencies: dict[str, dict] = {}
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
                self._opponent_issue_frequencies[issue.name] = {}

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_bids = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._my_last_bid = None
        self._my_last_utility = 1.0

    def _update_opponent_model(self, bid: Outcome) -> None:
        """
        Update opponent model based on bid frequency analysis.

        Issues where the opponent consistently selects the same values
        are assumed to be more important to them.

        Args:
            bid: The opponent's bid.
        """
        if bid is None or self.nmi is None:
            return

        self._opponent_bids.append(bid)

        # Track our utility for their bids
        if self.ufun is not None:
            utility = float(self.ufun(bid))
            if utility > self._best_opponent_utility:
                self._best_opponent_utility = utility
                self._best_opponent_bid = bid

        # Update value frequencies per issue
        issues = self.nmi.issues
        for i, issue in enumerate(issues):
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)
            if val is not None:
                val_key = str(val)
                if val_key not in self._opponent_issue_frequencies[issue.name]:
                    self._opponent_issue_frequencies[issue.name][val_key] = 0
                self._opponent_issue_frequencies[issue.name][val_key] += 1

        # Update issue weights based on value consistency
        if len(self._opponent_bids) >= 3:
            self._update_issue_weights()

    def _update_issue_weights(self) -> None:
        """
        Update estimated opponent issue weights based on selection consistency.
        """
        if self.nmi is None:
            return

        issues = self.nmi.issues
        consistency_scores: dict[str, float] = {}

        for issue in issues:
            counts = self._opponent_issue_frequencies.get(issue.name, {})
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

    def _get_opponent_utility(self, bid: Outcome) -> float:
        """
        Estimate opponent's utility for a bid based on learned preferences.

        Args:
            bid: The outcome to evaluate.

        Returns:
            Estimated opponent utility in [0, 1].
        """
        if self.nmi is None or not self._opponent_issue_frequencies:
            return 0.5

        issues = self.nmi.issues
        total_utility = 0.0

        for i, issue in enumerate(issues):
            weight = self._opponent_issue_weights.get(issue.name, 0.0)
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)

            if val is not None:
                val_key = str(val)
                counts = self._opponent_issue_frequencies.get(issue.name, {})

                if val_key in counts and counts:
                    # Higher frequency means more preferred
                    max_count = max(counts.values())
                    value_preference = (
                        counts[val_key] / max_count if max_count > 0 else 0.5
                    )
                else:
                    # Unknown value - assume moderate preference
                    value_preference = 0.3

                total_utility += weight * value_preference

        return min(1.0, max(0.0, total_utility))

    def _get_target_utility(self, t: float) -> float:
        """
        Calculate target utility using time-dependent Boulware strategy.

        Args:
            t: Normalized time in [0, 1].

        Returns:
            Target utility value.
        """
        # Boulware-like concession: target(t) = max - (max - min) * t^(1/beta)
        if self._beta > 0:
            concession = t ** (1.0 / self._beta)
        else:
            concession = 0.0

        target = self._max_util - (self._max_util - self._min_utility) * concession

        # Don't go below minimum
        target = max(target, self._min_utility)

        return target

    def _select_bid(self, target_utility: float) -> Outcome | None:
        """
        Select a bid near the target utility that is good for opponent.

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
            # Fall back to getting closest bid
            bid_details = self._outcome_space.get_bid_near_utility(target_utility)
            return bid_details.bid if bid_details else None

        if len(candidates) == 1:
            return candidates[0].bid

        # Score candidates by estimated opponent utility
        best_bid = None
        best_opp_util = -1.0

        for bd in candidates:
            opp_util = self._get_opponent_utility(bd.bid)
            if opp_util > best_opp_util:
                best_opp_util = opp_util
                best_bid = bd.bid

        return best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """
        Generate a proposal using BramAgent strategy.

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
                best_bids = self._outcome_space.get_bids_above(self._max_util - 0.001)
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
        Respond to an offer using BramAgent acceptance strategy.

        Accepts if:
        - Offer utility >= our target utility, OR
        - Offer utility >= utility of what we would offer next

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
        if t > self._deadline_acceptance_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
