"""
Utility classes for negmas-genius-agents.

This module provides helper classes used by the reimplemented Genius agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas.preferences import BaseUtilityFunction


@dataclass
class BidDetails:
    """
    A bid with its associated utility value.

    This is equivalent to Genius's BidDetails class.
    """

    bid: Outcome
    utility: float

    def __lt__(self, other: BidDetails) -> bool:
        return self.utility < other.utility

    def __le__(self, other: BidDetails) -> bool:
        return self.utility <= other.utility

    def __gt__(self, other: BidDetails) -> bool:
        return self.utility > other.utility

    def __ge__(self, other: BidDetails) -> bool:
        return self.utility >= other.utility


@dataclass
class SortedOutcomeSpace:
    """
    A sorted list of all possible outcomes with their utilities.

    This is equivalent to Genius's SortedOutcomeSpace class. It provides
    efficient lookup of bids by utility value.

    The outcomes are sorted in descending order by utility (best first).
    """

    ufun: BaseUtilityFunction
    _outcomes: list[BidDetails] = field(default_factory=list, init=False)
    _initialized: bool = field(default=False, init=False)

    def _initialize(self) -> None:
        """Generate and sort all outcomes by utility."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        # Get the outcome space from the utility function
        outcome_space = self.ufun.outcome_space
        if outcome_space is None:
            return

        # Generate all outcomes and compute utilities
        outcomes = []
        for outcome in outcome_space.enumerate():
            utility = float(self.ufun(outcome))
            outcomes.append(BidDetails(bid=outcome, utility=utility))

        # Sort by utility (descending - best first)
        outcomes.sort(key=lambda x: x.utility, reverse=True)
        self._outcomes = outcomes
        self._initialized = True

    @property
    def outcomes(self) -> list[BidDetails]:
        """Get all outcomes sorted by utility (descending)."""
        self._initialize()
        return self._outcomes

    @property
    def max_utility(self) -> float:
        """Get the maximum possible utility."""
        self._initialize()
        if not self._outcomes:
            return 1.0
        return self._outcomes[0].utility

    @property
    def min_utility(self) -> float:
        """Get the minimum possible utility."""
        self._initialize()
        if not self._outcomes:
            return 0.0
        return self._outcomes[-1].utility

    def get_bid_near_utility(self, target_utility: float) -> BidDetails | None:
        """
        Find the bid with utility closest to the target utility.

        Args:
            target_utility: The desired utility value.

        Returns:
            The BidDetails with utility closest to the target, or None if no bids.
        """
        self._initialize()
        if not self._outcomes:
            return None

        # Binary search for the closest utility
        left, right = 0, len(self._outcomes) - 1

        # Handle edge cases
        if target_utility >= self._outcomes[0].utility:
            return self._outcomes[0]
        if target_utility <= self._outcomes[-1].utility:
            return self._outcomes[-1]

        # Binary search (outcomes are sorted descending)
        while left < right:
            mid = (left + right) // 2
            if self._outcomes[mid].utility > target_utility:
                left = mid + 1
            else:
                right = mid

        # Check which of left or left-1 is closer
        if left > 0:
            diff_left = abs(self._outcomes[left].utility - target_utility)
            diff_prev = abs(self._outcomes[left - 1].utility - target_utility)
            if diff_prev < diff_left:
                return self._outcomes[left - 1]

        return self._outcomes[left]

    def get_bids_in_range(self, min_util: float, max_util: float) -> list[BidDetails]:
        """
        Get all bids with utility in the specified range.

        Args:
            min_util: Minimum utility (inclusive).
            max_util: Maximum utility (inclusive).

        Returns:
            List of BidDetails with utilities in [min_util, max_util].
        """
        self._initialize()
        return [bd for bd in self._outcomes if min_util <= bd.utility <= max_util]

    def get_bids_above(self, min_util: float) -> list[BidDetails]:
        """
        Get all bids with utility >= min_util.

        Args:
            min_util: Minimum utility threshold.

        Returns:
            List of BidDetails with utilities >= min_util.
        """
        self._initialize()
        # Since outcomes are sorted descending, we can stop early
        result = []
        for bd in self._outcomes:
            if bd.utility >= min_util:
                result.append(bd)
            else:
                break
        return result
