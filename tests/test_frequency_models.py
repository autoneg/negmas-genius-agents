"""Tests for the extra frequency-analysis opponent models
(negmas_genius_agents.models.frequency_extra)."""

from __future__ import annotations

import pytest

from negmas.outcomes import make_issue, make_os
from negmas.preferences.base_ufun import BaseUtilityFunction

from negmas_genius_agents.models.frequency_extra import (
    SmithFrequencyModel,
    CUHKFrequencyModelV2,
    NashFrequencyModel,
    AgentXFrequencyModel,
)

MODELS = [
    SmithFrequencyModel,
    CUHKFrequencyModelV2,
    NashFrequencyModel,
    AgentXFrequencyModel,
]


def _issues():
    return [
        make_issue(["a", "b", "c"], "color"),
        make_issue(["x", "y"], "size"),
        make_issue([str(i) for i in range(5)], "q"),
    ]


def _trace():
    """Opponent insists on ("a", "x", *), conceding only on 'q'."""
    return [
        ("a", "x", "0"),
        ("a", "x", "0"),
        ("a", "x", "1"),
        ("a", "y", "2"),
        ("a", "x", "0"),
        ("a", "x", "3"),
    ]


@pytest.mark.parametrize("cls", MODELS)
def test_learns_offered_outcome_is_favored(cls):
    issues = _issues()
    m = cls().initialize(issues)
    for offer in _trace():
        m.update(offer)
    consistently_offered = ("a", "x", "0")
    rarely_offered = ("c", "y", "4")
    assert m(consistently_offered) >= m(rarely_offered)


@pytest.mark.parametrize("cls", MODELS)
def test_is_a_ufun(cls):
    m = cls().initialize(_issues())
    assert isinstance(m, BaseUtilityFunction)
    assert isinstance(float(m(("a", "x", "0"))), float)


@pytest.mark.parametrize("cls", MODELS)
def test_bounded_and_finite_over_outcome_space(cls):
    issues = _issues()
    os_ = make_os(issues)
    m = cls().initialize(os_)
    for offer in _trace():
        m.update(offer)
    for outcome in os_.enumerate_or_sample():
        val = float(m(outcome))
        assert val == val  # not NaN
        assert val != float("inf") and val != float("-inf")
        assert 0.0 <= val <= 1.0 + 1e-9
