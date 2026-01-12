"""TheNegotiator from ANAC 2011."""

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

__all__ = ["TheNegotiator"]


class TheNegotiator(SAONegotiator):
    """
    TheNegotiator from ANAC 2011.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This implementation faithfully reproduces TheNegotiator's core strategies:

    - Four-phase negotiation with varying concession rates (early/middle/late/final)
    - Opponent behavior analysis to classify tough vs. cooperative opponents
    - Adaptive concession that speeds up against tough opponents
    - Nash-like bid selection maximizing product of both parties' utilities

    References:
        Original Genius class: ``agents.anac.y2011.TheNegotiator.TheNegotiator``

        ANAC 2011: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
    - Four-phase concession with varying rates:
      - Early (t < 0.2): rate * 0.5 (very slow)
      - Middle (0.2 <= t < 0.8): rate * 1.0 (normal)
      - Late (0.8 <= t < 0.95): rate * 2.0 (faster)
      - Final (t >= 0.95): rate * 5.0 (rapid)
    - Opponent-adaptive: 1.5x faster against tough, 0.7x against cooperative
    - Boulware curve: target = max - (max - min) * t^(1/rate)
    - Nash-like bid selection maximizing (own_util * opponent_util)

    **Acceptance Strategy:**
    - Accept if offer utility >= current target utility
    - Accept if offer utility >= utility of next planned bid
    - Late phase (t > 0.9): Accept if above minimum utility
    - Final phase (t > 0.98): Accept if >= 95% of best opponent bid

    **Opponent Modeling:**
    - Tracks value frequencies per issue from opponent bids
    - Uses linear regression on received utilities for concession rate
    - Classifies opponent as "tough" if avg utility < 0.5 and rate < 0.005
    - Adapts concession speed based on opponent classification
    - Estimates opponent utility from frequency-weighted value preferences

    Args:
        initial_target: Initial target utility (default 0.95)
        min_utility: Minimum acceptable utility (default 0.5)
        concession_rate: Base concession rate (default 0.1)
        early_phase_end: Time threshold for early phase end (default 0.2)
        middle_phase_end: Time threshold for middle phase end (default 0.8)
        late_phase_end: Time threshold for late phase end (default 0.95)
        late_accept_time: Time threshold for late acceptance (default 0.9)
        final_accept_time: Time threshold for final acceptance (default 0.98)
        tough_avg_utility: Average utility threshold for tough opponent (default 0.5)
        tough_concession_rate: Concession rate threshold for tough opponent (default 0.005)
        early_rate_multiplier: Concession rate multiplier for early phase (default 0.5)
        middle_rate_multiplier: Concession rate multiplier for middle phase (default 1.0)
        late_rate_multiplier: Concession rate multiplier for late phase (default 2.0)
        final_rate_multiplier: Concession rate multiplier for final phase (default 5.0)
        tough_opponent_multiplier: Concession rate multiplier against tough opponents (default 1.5)
        cooperative_opponent_multiplier: Concession rate multiplier against cooperative opponents (default 0.7)
        cooperative_threshold: Concession rate threshold to detect cooperative opponent (default 0.01)
        final_phase_best_bid_factor: Factor for best bid in final phase (default 1.05)
        bid_tolerance: Tolerance for bid selection around target (default 0.03)
        final_accept_factor: Factor for accepting best opponent bid in final phase (default 0.95)
        min_bids_for_analysis: Minimum bids for opponent behavior analysis (default 5)
        trend_window_size: Number of recent bids for trend analysis (default 10)
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
        initial_target: float = 0.95,
        min_utility: float = 0.5,
        concession_rate: float = 0.1,
        early_phase_end: float = 0.2,
        middle_phase_end: float = 0.8,
        late_phase_end: float = 0.95,
        late_accept_time: float = 0.9,
        final_accept_time: float = 0.98,
        tough_avg_utility: float = 0.5,
        tough_concession_rate: float = 0.005,
        early_rate_multiplier: float = 0.5,
        middle_rate_multiplier: float = 1.0,
        late_rate_multiplier: float = 2.0,
        final_rate_multiplier: float = 5.0,
        tough_opponent_multiplier: float = 1.5,
        cooperative_opponent_multiplier: float = 0.7,
        cooperative_threshold: float = 0.01,
        final_phase_best_bid_factor: float = 1.05,
        bid_tolerance: float = 0.03,
        final_accept_factor: float = 0.95,
        min_bids_for_analysis: int = 5,
        trend_window_size: int = 10,
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
        self._initial_target = initial_target
        self._min_utility = min_utility
        self._concession_rate = concession_rate
        self._early_phase_end = early_phase_end
        self._middle_phase_end = middle_phase_end
        self._late_phase_end = late_phase_end
        self._late_accept_time = late_accept_time
        self._final_accept_time = final_accept_time
        self._tough_avg_utility = tough_avg_utility
        self._tough_concession_rate = tough_concession_rate
        self._early_rate_multiplier = early_rate_multiplier
        self._middle_rate_multiplier = middle_rate_multiplier
        self._late_rate_multiplier = late_rate_multiplier
        self._final_rate_multiplier = final_rate_multiplier
        self._tough_opponent_multiplier = tough_opponent_multiplier
        self._cooperative_opponent_multiplier = cooperative_opponent_multiplier
        self._cooperative_threshold = cooperative_threshold
        self._final_phase_best_bid_factor = final_phase_best_bid_factor
        self._bid_tolerance = bid_tolerance
        self._final_accept_factor = final_accept_factor
        self._min_bids_for_analysis = min_bids_for_analysis
        self._trend_window_size = trend_window_size

        # Will be initialized when negotiation starts
        self._outcome_space: SortedOutcomeSpace | None = None
        self._max_util: float = 1.0
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._opponent_utilities: list[float] = []
        self._opponent_issue_frequencies: dict[str, dict] = {}
        self._opponent_issue_weights: dict[str, float] = {}

        # Tracking
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._my_last_bid: Outcome | None = None
        self._my_last_utility: float = 1.0
        self._current_target: float = 0.95

        # Strategy adaptation
        self._opponent_is_tough: bool = False
        self._opponent_concession_rate: float = 0.0

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

        self._current_target = self._initial_target
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
        self._my_last_bid = None
        self._my_last_utility = 1.0
        self._opponent_is_tough = False
        self._opponent_concession_rate = 0.0

    def _update_opponent_model(self, bid: Outcome) -> None:
        """
        Update opponent model based on bid analysis.

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

        # Update value frequencies per issue
        issues = self.nmi.issues
        for i, issue in enumerate(issues):
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)
            if val is not None:
                val_key = str(val)
                if val_key not in self._opponent_issue_frequencies[issue.name]:
                    self._opponent_issue_frequencies[issue.name][val_key] = 0
                self._opponent_issue_frequencies[issue.name][val_key] += 1

        # Analyze opponent behavior
        self._analyze_opponent_behavior()

    def _analyze_opponent_behavior(self) -> None:
        """Analyze opponent's concession behavior to adapt strategy."""
        if len(self._opponent_utilities) < self._min_bids_for_analysis:
            return

        # Calculate recent trend
        recent = self._opponent_utilities[-self._trend_window_size :]
        if len(recent) < 2:
            return

        # Simple linear regression for trend
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator
            self._opponent_concession_rate = max(0, slope)

        # Determine if opponent is tough
        avg_utility = sum(self._opponent_utilities) / len(self._opponent_utilities)
        self._opponent_is_tough = (
            avg_utility < self._tough_avg_utility
            and self._opponent_concession_rate < self._tough_concession_rate
        )

    def _get_opponent_utility(self, bid: Outcome) -> float:
        """
        Estimate opponent's utility for a bid.

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
                    max_count = max(counts.values())
                    value_preference = (
                        counts[val_key] / max_count if max_count > 0 else 0.5
                    )
                else:
                    value_preference = 0.3

                total_utility += weight * value_preference

        return min(1.0, max(0.0, total_utility))

    def _get_target_utility(self, t: float) -> float:
        """
        Calculate target utility with adaptive concession.

        The strategy adapts based on:
        - Current phase of negotiation
        - Opponent's behavior (tough vs cooperative)
        - Time pressure

        Args:
            t: Normalized time in [0, 1].

        Returns:
            Target utility value.
        """
        # Phase-dependent base concession rate
        if t < self._early_phase_end:
            # Early phase: very slow concession
            phase_rate = self._concession_rate * self._early_rate_multiplier
        elif t < self._middle_phase_end:
            # Middle phase: normal concession
            phase_rate = self._concession_rate * self._middle_rate_multiplier
        elif t < self._late_phase_end:
            # Late phase: faster concession
            phase_rate = self._concession_rate * self._late_rate_multiplier
        else:
            # Final phase: rapid concession
            phase_rate = self._concession_rate * self._final_rate_multiplier

        # Adapt based on opponent behavior
        if self._opponent_is_tough:
            # Against tough opponents, concede faster to reach agreement
            phase_rate *= self._tough_opponent_multiplier
        elif self._opponent_concession_rate > self._cooperative_threshold:
            # If opponent is conceding, we can stay tough
            phase_rate *= self._cooperative_opponent_multiplier

        # Calculate target using Boulware-like curve
        if phase_rate > 0:
            concession = t ** (1.0 / phase_rate)
        else:
            concession = 0.0

        target = self._max_util - (self._max_util - self._min_utility) * concession

        # Don't go below minimum
        target = max(target, self._min_utility)

        # In final phase, be willing to accept best opponent bid
        if t > self._late_phase_end and self._best_opponent_utility > self._min_utility:
            target = min(
                target, self._best_opponent_utility * self._final_phase_best_bid_factor
            )

        return target

    def _select_bid(self, target_utility: float) -> Outcome | None:
        """
        Select a bid near the target utility.

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
            bid_details = self._outcome_space.get_bid_near_utility(target_utility)
            return bid_details.bid if bid_details else None

        if len(candidates) == 1:
            return candidates[0].bid

        # Score candidates considering opponent preferences
        best_bid = None
        best_score = -1.0

        for bd in candidates:
            opp_util = self._get_opponent_utility(bd.bid)
            # Nash-like product
            score = bd.utility * opp_util
            if score > best_score:
                best_score = score
                best_bid = bd.bid

        return best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """
        Generate a proposal using TheNegotiator strategy.

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
        self._current_target = target

        # Ensure monotonic concession
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
        Respond to an offer using TheNegotiator acceptance strategy.

        Uses multiple acceptance criteria depending on phase:
        - Accept if offer >= target
        - Accept if offer >= what we would offer
        - Near deadline: accept if >= minimum
        - Very near deadline: accept best opponent bid

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

        # Late phase: accept if above minimum
        if t > self._late_accept_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        # Final phase: accept best offer we've seen
        if (
            t > self._final_accept_time
            and offer_utility >= self._best_opponent_utility * self._final_accept_factor
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
