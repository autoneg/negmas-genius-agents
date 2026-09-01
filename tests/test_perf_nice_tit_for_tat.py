"""Equivalence tests for the NiceTitForTat opponent-model performance fix.

Two things changed: `_get_opponent_utility` now reads a per-issue maximum
frequency that is maintained as the frequency tables are updated instead of
recomputing `max(counts.values())` for every issue of every bid, and
`_estimate_nash_utility` reuses its result while the opponent model is
unchanged. Both are checked against verbatim copies of the pre-fix code.

The driver below replays random sequences of opponent bids through the real
`_update_opponent_model` and, after every update, compares the current
implementations against the originals for every outcome in the space -- so the
comparison covers the empty model, single-bid models, ties in the frequency
tables, and values never seen by the opponent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism

from negmas_genius_agents import NiceTitForTat


def _original_get_opponent_utility(self, bid) -> float:
    """Verbatim copy of the pre-fix ``_get_opponent_utility``."""
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
                value_preference = counts[val_key] / max_count if max_count > 0 else 0.5
            else:
                value_preference = self._unknown_value_preference
            total_utility += weight * value_preference

    return min(1.0, max(0.0, total_utility))


def _original_estimate_nash_utility(self) -> float:
    """Verbatim copy of the pre-fix ``_estimate_nash_utility``."""
    if self._outcome_space is None or not self._outcome_space.outcomes:
        return 0.7

    best_score = -1.0
    best_util = 0.7
    for bd in self._outcome_space.outcomes:
        opp_util = _original_get_opponent_utility(self, bd.bid)
        score = bd.utility * opp_util
        if score > best_score:
            best_score = score
            best_util = bd.utility

    return best_util


def _make_agent(rng: random.Random, n_issues: int, n_values: int):
    """A NiceTitForTat wired to a real (tiny) domain and initialized."""
    issues = [
        make_issue(values=[f"v{j}" for j in range(n_values)], name=f"i{i}")
        for i in range(n_issues)
    ]
    os_ = make_os(issues)
    ufun = LinearAdditiveUtilityFunction(
        values=[
            {f"v{j}": rng.random() for j in range(n_values)} for _ in range(n_issues)
        ],
        weights=[rng.random() for _ in range(n_issues)],
        outcome_space=os_,
    )
    agent = NiceTitForTat(name="a", ufun=ufun)
    # An SAOMechanism gives the negotiator a real nmi (needed for .issues).
    m = SAOMechanism(outcome_space=os_, n_steps=10)
    m.add(agent)
    agent._initialize()
    return agent, list(os_.enumerate_or_sample())


def test_opponent_utility_and_nash_match_originals():
    rng = random.Random(20260901)
    checked_bids = 0
    checked_nash = 0
    for trial in range(150):
        n_issues = rng.randint(1, 3)
        n_values = rng.randint(1, 4)
        agent, outcomes = _make_agent(rng, n_issues, n_values)

        # Model with nothing observed yet.
        for bid in outcomes:
            assert agent._get_opponent_utility(bid) == _original_get_opponent_utility(
                agent, bid
            )
            checked_bids += 1
        assert agent._estimate_nash_utility() == _original_estimate_nash_utility(agent)
        checked_nash += 1

        # Feed a random opponent bid sequence, re-checking after every update.
        for k in range(rng.randint(1, 12)):
            agent._update_opponent_model(rng.choice(outcomes), k / 12.0)
            for bid in outcomes:
                assert agent._get_opponent_utility(
                    bid
                ) == _original_get_opponent_utility(agent, bid)
                checked_bids += 1
            # Called twice: the second call must hit the memo and still agree.
            expected = _original_estimate_nash_utility(agent)
            assert agent._estimate_nash_utility() == expected
            assert agent._estimate_nash_utility() == expected
            checked_nash += 2
            # The maintained maxima must equal the recomputed ones.
            for name, counts in agent._opponent_issue_frequencies.items():
                if counts:
                    assert agent._opponent_issue_max_count[name] == max(counts.values())

    assert checked_bids > 5000
    assert checked_nash > 500


def test_nash_memo_invalidated_by_model_updates():
    """A change in the opponent model must be visible to the next estimate."""
    rng = random.Random(3)
    agent, outcomes = _make_agent(rng, 3, 4)
    seen = set()
    for k in range(30):
        agent._update_opponent_model(rng.choice(outcomes), k / 30.0)
        seen.add(agent._estimate_nash_utility())
        assert agent._estimate_nash_utility() == _original_estimate_nash_utility(agent)
    # Sanity: the estimate really does move, so the memo is not just frozen.
    assert len(seen) > 1
