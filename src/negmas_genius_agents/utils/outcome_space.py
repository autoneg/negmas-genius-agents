"""
Utility classes for negmas-genius-agents.

This module provides helper classes used by the reimplemented Genius agents.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, overload

import numpy as np

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


class _BidDetailsView(Sequence):
    """
    A read-only sequence of `BidDetails` sorted by utility (descending).

    The bids and their utilities live in the ufun's inverter (see
    `SortedOutcomeSpace`); `BidDetails` wrappers are created on demand so that
    agents that only ever look at a handful of bids never pay for materializing
    the whole list.
    """

    __slots__ = ("_bids", "_utils", "_items")

    def __init__(self, bids: Sequence[Outcome], utils: Any):
        self._bids = bids
        self._utils = utils
        # Wrappers are created once per bid and then reused, so repeated
        # queries return the same `BidDetails` objects (as a plain list would).
        self._items: list[BidDetails | None] = [None] * len(bids)

    def __len__(self) -> int:
        return len(self._bids)

    def _item(self, index: int) -> BidDetails:
        item = self._items[index]
        if item is None:
            item = BidDetails(bid=self._bids[index], utility=float(self._utils[index]))
            self._items[index] = item
        return item

    def _materialize(self) -> list[BidDetails]:
        """Creates every wrapper (once) and returns them as a plain list."""
        items = self._items
        for i in range(len(items)):
            if items[i] is None:
                items[i] = BidDetails(bid=self._bids[i], utility=float(self._utils[i]))
        return items  # type: ignore[return-value]

    @overload
    def __getitem__(self, index: int) -> BidDetails: ...

    @overload
    def __getitem__(self, index: slice) -> list[BidDetails]: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._materialize()[index]
        if index < 0:
            index += len(self._bids)
        if not 0 <= index < len(self._bids):
            raise IndexError(index)
        return self._item(index)

    def __iter__(self) -> Iterator[BidDetails]:
        return iter(self._materialize())


@dataclass
class SortedOutcomeSpace:
    """
    A sorted list of all possible outcomes with their utilities.

    This is equivalent to Genius's SortedOutcomeSpace class. It provides
    efficient lookup of bids by utility value.

    The outcomes are sorted in descending order by utility (best first).

    Remarks:
        - The sorted outcomes are obtained from the utility function's own
          inverter (`BaseUtilityFunction.invert()`) rather than being
          enumerated and sorted here. NegMAS caches inverters on the ufun (and
          can load them from disk when a scenario was saved with them), so all
          negotiators sharing a ufun share one presorted list instead of each
          re-evaluating the whole outcome space.
        - Lookups (`get_bid_near_utility`, `get_bids_above`,
          `get_bids_in_range`) binary-search the inverter's sorted utilities
          instead of scanning all outcomes.
        - The inverter does not sort stably, so outcomes of exactly equal
          utility are re-ordered by their enumeration order (see
          `_break_ties`); the resulting list is identical to enumerating the
          space and sorting it here, which is what agents relying on "the
          first bid at or above X" depend on.
    """

    ufun: BaseUtilityFunction
    _outcomes: _BidDetailsView | list[BidDetails] = field(
        default_factory=list, init=False
    )
    # Utilities of `_outcomes` (descending). Kept as a numpy array so that
    # lookups can use binary search.
    _utils: Any = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)

    def _sorted_from_inverter(self) -> tuple[list[Outcome], Any] | None:
        """Gets (bids, utilities) sorted by utility descending from the inverter.

        Returns None if the ufun cannot be inverted, in which case the caller
        falls back to enumerating the outcome space directly.
        """
        invert = getattr(self.ufun, "invert", None)
        if invert is None:
            return None
        try:
            inv = invert()
        except Exception:
            return None
        # The sorted result is stashed on the inverter (not on the ufun, whose
        # attributes are watched for cache invalidation by negmas) so that all
        # negotiators sharing a ufun share this work too.
        cached = getattr(inv, "_genius_sorted", None)
        if cached is not None:
            return cached
        result = self._sorted_from(inv)
        if result is not None:
            try:
                inv._genius_sorted = result
            except Exception:
                pass
        return result

    def _sorted_from(self, inv: Any) -> tuple[list[Outcome], Any] | None:
        """Extracts (bids, utilities) sorted by utility descending from `inv`."""
        # Adaptive inverters wrap a concrete one; the wrapped inverter is the
        # one holding the presorted arrays.
        inner = getattr(inv, "delegate", None) or inv
        bids = getattr(inner, "outcomes", None)
        utils = getattr(inner, "utils", None)
        if not bids:
            return None
        if utils is not None and len(utils) == len(bids):
            utils = np.asarray(utils, dtype=float)
            last_rational = getattr(inner, "_last_rational", len(bids) - 1)
            if last_rational == len(bids) - 1:
                # The usual case: the inverter sorted everything ascending.
                order = slice(None, None, -1)
                return self._break_ties(list(bids)[order], utils[order])
            # `rational_only` inverters keep the irrational outcomes unsorted
            # at the end of the list, so sort here (still without
            # re-evaluating the ufun).
            order = np.argsort(-utils, kind="stable")
            return self._break_ties([bids[i] for i in order], utils[order])
        # No presorted arrays exposed: fall back to the ranked accessors
        # (rank 0 is the best outcome) if the inverter provides them.
        outcome_at = getattr(inv, "outcome_at", None)
        utility_at = getattr(inv, "utility_at", None)
        if outcome_at is None or utility_at is None:
            # Inverters that do not presort the whole space (e.g. sampling
            # based ones) cannot be used to build the sorted list.
            return None
        ranked, ranked_utils = [], []
        for i in range(len(bids)):
            bid, util = outcome_at(i), utility_at(i)
            if bid is None or util is None:
                return None
            ranked.append(bid)
            ranked_utils.append(float(util))
        return self._break_ties(ranked, np.asarray(ranked_utils, dtype=float))

    def _break_ties(self, bids: list[Outcome], utils: Any) -> tuple[list[Outcome], Any]:
        """Orders outcomes of exactly equal utility by their enumeration order.

        The inverter does not sort stably, so outcomes sharing a utility come
        out in an arbitrary order. Genius agents pick "the first bid at/above
        some utility", so the order within a tie decides which bid is offered.
        Re-establishing the enumeration order here keeps that choice identical
        to enumerating and sorting the space directly (and stable across runs)
        while still avoiding a full pass of utility evaluations.
        """
        if len(utils) < 2 or not np.any(np.diff(utils) == 0):
            return bids, utils
        outcome_space = self.ufun.outcome_space
        try:
            ranks = {o: i for i, o in enumerate(outcome_space.enumerate())}  # type: ignore[union-attr]
        except Exception:
            # Continuous (or otherwise non-enumerable) spaces: the inverter's
            # own order is all we have.
            return bids, utils
        if any(b not in ranks for b in bids):
            # The inverter sampled/discretized the space, so enumeration ranks
            # do not cover it.
            return bids, utils
        order = np.lexsort((np.asarray([ranks[b] for b in bids]), -utils))
        return [bids[i] for i in order], utils[order]

    def _initialize(self) -> None:
        """Get all outcomes sorted by utility (best first)."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        # Get the outcome space from the utility function
        outcome_space = self.ufun.outcome_space
        if outcome_space is None:
            return

        sorted_bids = self._sorted_from_inverter()
        if sorted_bids is not None:
            bids, utils = sorted_bids
            self._outcomes = _BidDetailsView(bids, utils)
            self._utils = utils
        else:
            # Fallback: generate all outcomes and compute utilities ourselves.
            outcomes = []
            for outcome in outcome_space.enumerate():
                utility = float(self.ufun(outcome))
                outcomes.append(BidDetails(bid=outcome, utility=utility))

            # Sort by utility (descending - best first)
            outcomes.sort(key=lambda x: x.utility, reverse=True)
            self._outcomes = outcomes
            self._utils = np.asarray([_.utility for _ in outcomes], dtype=float)
        self._initialized = True

    @property
    def outcomes(self) -> Sequence[BidDetails]:
        """Get all outcomes sorted by utility (descending)."""
        self._initialize()
        return self._outcomes

    @property
    def max_utility(self) -> float:
        """Get the maximum possible utility."""
        self._initialize()
        if self._utils is None or not len(self._utils):
            return 1.0
        return float(self._utils[0])

    @property
    def min_utility(self) -> float:
        """Get the minimum possible utility."""
        self._initialize()
        if self._utils is None or not len(self._utils):
            return 0.0
        return float(self._utils[-1])

    def _index_near_utility(self, target_utility: float) -> int:
        """Index (in descending order) of the bid closest to `target_utility`."""
        utils = self._utils
        n = len(utils)
        if target_utility >= utils[0]:
            return 0
        if target_utility <= utils[-1]:
            return n - 1
        # `utils` is descending, so search its ascending reverse and map back.
        # `left` is the first index (descending) whose utility <= target.
        left = n - int(np.searchsorted(utils[::-1], target_utility, side="right"))
        if left > 0 and abs(utils[left - 1] - target_utility) < abs(
            utils[left] - target_utility
        ):
            return left - 1
        return left

    def get_bid_near_utility(self, target_utility: float) -> BidDetails | None:
        """
        Find the bid with utility closest to the target utility.

        Args:
            target_utility: The desired utility value.

        Returns:
            The BidDetails with utility closest to the target, or None if no bids.
        """
        self._initialize()
        if self._utils is None or not len(self._utils):
            return None
        return self._outcomes[self._index_near_utility(target_utility)]

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
        utils = self._utils
        if utils is None or not len(utils):
            return []
        n = len(utils)
        rev = utils[::-1]
        # first index (descending) with utility <= max_util
        start = n - int(np.searchsorted(rev, max_util, side="right"))
        # one past the last index (descending) with utility >= min_util
        end = n - int(np.searchsorted(rev, min_util, side="left"))
        if end <= start:
            return []
        return list(self._outcomes[start:end])

    def get_bids_above(self, min_util: float) -> list[BidDetails]:
        """
        Get all bids with utility >= min_util.

        Args:
            min_util: Minimum utility threshold.

        Returns:
            List of BidDetails with utilities >= min_util.
        """
        self._initialize()
        utils = self._utils
        if utils is None or not len(utils):
            return []
        n = len(utils)
        end = n - int(np.searchsorted(utils[::-1], min_util, side="left"))
        return list(self._outcomes[:end])
