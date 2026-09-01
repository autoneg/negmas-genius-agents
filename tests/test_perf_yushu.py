"""Equivalence tests for the Yushu ``_get_next_bid`` performance fix.

The original implementation is copied verbatim below as an oracle. The test
asserts that the current implementation returns exactly the same bid (same
object, so ties are broken identically) over a large randomized space of
outcome spaces and agent states, including the awkward cases: empty outcome
spaces, single-outcome spaces, ties in utility, empty candidate sets, and all
candidates filtered out by the recent-bid window.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from negmas_genius_agents.negotiators.anac.y2010.yushu import Yushu


@dataclass
class _BD:
    bid: Any
    utility: float


class _Space:
    def __init__(self, outcomes):
        self.outcomes = outcomes


class _Stub:
    """Duck-typed stand-in carrying exactly the attributes ``_get_next_bid`` reads."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _original_get_next_bid(self, target: float, time: float):
    """Verbatim copy of the pre-fix ``Yushu._get_next_bid``."""
    if self._outcome_space is None:
        return None

    lower = self._bid_lower_factor * target
    upper = self._bid_upper_factor * target

    candidates: list = []
    for bd in self._outcome_space.outcomes:
        if lower <= bd.utility <= upper:
            candidates.append(bd.bid)
        elif bd.utility < lower:
            break

    if (
        self._best_ten_indices
        and self._estimate_rounds_left(time) > self._many_rounds_threshold
    ):
        best_opp_util = self._opponent_utilities[self._best_ten_indices[0]]
        for bd in self._outcome_space.outcomes:
            if bd.utility >= best_opp_util and bd.bid not in candidates:
                candidates.append(bd.bid)
            elif bd.utility < best_opp_util:
                break

    if not candidates:
        if self._outcome_space.outcomes:
            return self._outcome_space.outcomes[0].bid
        return None

    recent = set(
        tuple(b) if b else ()
        for b in self._my_history[-self._recent_bid_window :]
        if b is not None
    )
    filtered = [c for c in candidates if tuple(c) not in recent]

    if filtered and len(self._my_history) > self._random_selection_min_history:
        return random.choice(filtered)
    elif filtered:
        return min(
            filtered,
            key=lambda b: abs(float(self.ufun(b)) - target) if self.ufun else 0,
        )
    elif candidates:
        return candidates[0]

    return (
        self._outcome_space.outcomes[0].bid if self._outcome_space.outcomes else None
    )


def _make_case(rng: random.Random):
    """Build one random (stub, target, time) case."""
    n = rng.choice([0, 1, 1, 2, 5, 20, 60, 200])
    # A small value alphabet makes utility ties (and hence tie-breaking)
    # frequent, and a small bid alphabet makes duplicate bids possible.
    n_issues = rng.randint(1, 3)
    vals = ["a", "b", "c"]
    bids = []
    for _ in range(n):
        bids.append(tuple(rng.choice(vals) for _ in range(n_issues)))
    # utilities: descending, drawn from a coarse grid so ties abound
    utils = sorted(
        (rng.randrange(0, 11) / 10.0 for _ in range(n)), reverse=True
    )
    outcomes = [_BD(bid=b, utility=u) for b, u in zip(bids, utils)]

    n_opp = rng.randint(0, 6)
    opp_utils = [rng.randrange(0, 11) / 10.0 for _ in range(n_opp)]
    best_ten = rng.sample(range(n_opp), rng.randint(0, n_opp)) if n_opp else []

    hist_len = rng.choice([0, 1, 3, 8])
    my_history = [
        (bids[rng.randrange(n)] if (n and rng.random() < 0.8) else None)
        for _ in range(hist_len)
    ]

    rounds_left = rng.choice([0.0, 1.0, 5.0, 50.0])
    ufun = None
    if rng.random() < 0.8:
        table = {b: rng.random() for b in set(bids)}
        ufun = lambda b, _t=table: _t.get(b, 0.0)  # noqa: E731

    stub = _Stub(
        _outcome_space=_Space(outcomes) if rng.random() < 0.95 else None,
        _bid_lower_factor=rng.choice([0.7, 0.9, 1.0]),
        _bid_upper_factor=rng.choice([1.0, 1.1, 1.4]),
        _best_ten_indices=best_ten,
        _opponent_utilities=opp_utils,
        _many_rounds_threshold=rng.choice([0.0, 2.0, 10.0]),
        _estimate_rounds_left=lambda _t, _r=rounds_left: _r,
        _my_history=my_history,
        _recent_bid_window=rng.choice([1, 3, 10]),
        _random_selection_min_history=rng.choice([0, 2, 100]),
        ufun=ufun,
    )
    return stub, rng.choice([0.0, 0.3, 0.5, 0.8, 1.0]), rng.random()


def test_get_next_bid_matches_original():
    rng = random.Random(20260901)
    n_cases = 3000
    n_nonempty = 0
    for i in range(n_cases):
        stub, target, t = _make_case(rng)
        random.seed(i)
        expected = _original_get_next_bid(stub, target, t)
        random.seed(i)
        actual = Yushu._get_next_bid(stub, target, t)
        assert actual == expected, f"case {i}: {actual!r} != {expected!r}"
        if expected is not None:
            n_nonempty += 1
    # Sanity: the randomized space actually exercised real results.
    assert n_nonempty > n_cases // 2


def test_get_next_bid_matches_original_large_space():
    """Larger spaces, where the original's O(n^2) membership test bites."""
    rng = random.Random(7)
    for i in range(20):
        n = 4000
        bids = [(x,) for x in range(n)]
        utils = sorted((rng.randrange(0, 50) / 50.0 for _ in range(n)), reverse=True)
        outcomes = [_BD(bid=b, utility=u) for b, u in zip(bids, utils)]
        stub = _Stub(
            _outcome_space=_Space(outcomes),
            _bid_lower_factor=0.9,
            _bid_upper_factor=1.1,
            _best_ten_indices=[0],
            _opponent_utilities=[rng.randrange(0, 50) / 50.0],
            _many_rounds_threshold=1.0,
            _estimate_rounds_left=lambda _t: 100.0,
            _my_history=[bids[rng.randrange(n)] for _ in range(5)],
            _recent_bid_window=3,
            _random_selection_min_history=rng.choice([0, 100]),
            ufun=lambda b: b[0] / n,
        )
        target = rng.random()
        random.seed(i)
        expected = _original_get_next_bid(stub, target, 0.5)
        random.seed(i)
        actual = Yushu._get_next_bid(stub, target, 0.5)
        assert actual == expected
