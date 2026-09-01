"""Equivalence test for the WhaleAgent opponent-model performance fix.

`_estimate_opponent_utility` used to recompute `max(freq[i].values())` for
every issue of every candidate bid. The maximum is now maintained as the
frequency model is updated. The pre-fix function is copied verbatim below as
an oracle and compared against the current one after every single model
update, for every outcome of a random domain -- so the comparison covers the
empty model, a single observed bid, ties in the frequency table, and values
the opponent never offered.
"""

from __future__ import annotations

import random

from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction

from negmas_genius_agents import WhaleAgent


def _original_estimate_opponent_utility(self, bid) -> float:
    """Verbatim copy of the pre-fix ``_estimate_opponent_utility``."""
    if not self._opponent_value_freq:
        return 0.5

    total_score = 0.0
    num_issues = len(bid)

    for i, value in enumerate(bid):
        if i in self._opponent_value_freq:
            freq = self._opponent_value_freq[i].get(value, 0)
            max_freq = max(self._opponent_value_freq[i].values())
            if max_freq > 0:
                total_score += freq / max_freq

    return total_score / num_issues if num_issues > 0 else 0.5


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
    agent = WhaleAgent(name="a", ufun=ufun)
    agent.on_negotiation_start(None)  # type: ignore[arg-type]
    return agent, list(os_.enumerate_or_sample())


def test_estimate_opponent_utility_matches_original():
    rng = random.Random(20260901)
    checked = 0
    for _ in range(150):
        n_issues = rng.randint(1, 3)
        n_values = rng.randint(1, 4)
        agent, outcomes = _make_agent(rng, n_issues, n_values)

        # Empty model.
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
            for i, freq in agent._opponent_value_freq.items():
                assert agent._opponent_value_freq_max[i] == max(freq.values())

    assert checked > 5000


def test_maxima_reset_between_negotiations():
    """A fresh negotiation must not inherit the previous one's maxima."""
    rng = random.Random(11)
    agent, outcomes = _make_agent(rng, 3, 4)
    for _ in range(20):
        agent._update_opponent_model(rng.choice(outcomes))
    agent.on_negotiation_start(None)  # type: ignore[arg-type]
    assert agent._opponent_value_freq_max == {}
    for bid in outcomes:
        assert agent._estimate_opponent_utility(
            bid
        ) == _original_estimate_opponent_utility(agent, bid)
    agent._update_opponent_model(outcomes[0])
    for bid in outcomes:
        assert agent._estimate_opponent_utility(
            bid
        ) == _original_estimate_opponent_utility(agent, bid)
