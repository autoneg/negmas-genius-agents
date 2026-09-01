"""Equivalence test for the y2016 MyAgent opponent-model performance fix.

`_estimate_opponent_utility` used to recompute `max(freq_map.values())` for
every issue of every candidate bid. That maximum is now maintained as the
frequency model is updated. The pre-fix function is copied verbatim below as
an oracle and compared against the current one after every model update, for
every outcome of a random domain, covering the empty model, a single observed
bid, ties in the frequency table and values the opponent never offered.
"""

from __future__ import annotations

import random

from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism

from negmas_genius_agents import MyAgent


def _original_estimate_opponent_utility(self, bid) -> float:
    """Verbatim copy of the pre-fix ``_estimate_opponent_utility``."""
    if not self._opponent_value_frequencies:
        return 0.5

    total_utility = 0.0

    for i, value in enumerate(bid):
        weight = self._opponent_issue_weights.get(i, 0.0)
        value_str = str(value)

        freq_map = self._opponent_value_frequencies.get(i, {})
        if not freq_map:
            total_utility += weight * 0.5
            continue

        # Value utility based on frequency
        value_count = freq_map.get(value_str, 0)
        max_count = max(freq_map.values()) if freq_map else 1

        value_utility = value_count / max_count if max_count > 0 else 0.5
        total_utility += weight * value_utility

    return total_utility


def _make_agent(rng: random.Random, n_issues: int, n_values: int):
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
    agent = MyAgent(name="a", ufun=ufun)
    m = SAOMechanism(outcome_space=os_, n_steps=10)
    m.add(agent)
    agent._initialize()
    return agent, list(os_.enumerate_or_sample())


def test_estimate_opponent_utility_matches_original():
    rng = random.Random(20260901)
    checked = 0
    for _ in range(150):
        n_issues = rng.randint(1, 3)
        n_values = rng.randint(1, 4)
        agent, outcomes = _make_agent(rng, n_issues, n_values)

        for bid in outcomes:
            assert agent._estimate_opponent_utility(
                bid
            ) == _original_estimate_opponent_utility(agent, bid)
            checked += 1

        for _ in range(rng.randint(1, 12)):
            agent._update_opponent_model(rng.choice(outcomes))
            for bid in outcomes:
                assert agent._estimate_opponent_utility(
                    bid
                ) == _original_estimate_opponent_utility(agent, bid)
                checked += 1
            for i, freq in agent._opponent_value_frequencies.items():
                if freq:
                    assert agent._opponent_value_frequency_max[i] == max(freq.values())

    assert checked > 5000


def test_estimate_opponent_utility_with_unseen_issue_values():
    """Values the opponent never offered must still score identically."""
    rng = random.Random(5)
    agent, outcomes = _make_agent(rng, 3, 4)
    # Feed only bids that share one fixed value on the first issue, so many
    # values of that issue are never observed.
    subset = [o for o in outcomes if o[0] == "v0"]
    for k in range(10):
        agent._update_opponent_model(subset[k % len(subset)])
    for bid in outcomes:
        assert agent._estimate_opponent_utility(
            bid
        ) == _original_estimate_opponent_utility(agent, bid)
