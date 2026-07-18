"""Tests for the Genius BOA opponent models (negmas_genius_agents.models)."""

from __future__ import annotations

import pytest

from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism

from negmas_genius_agents.models import (
    HardHeadedFrequencyModel,
    PerfectModel,
    WorstModel,
    OppositeModel,
    UniformModel,
    DefaultModel,
)


def _issues():
    return [
        make_issue(["a", "b", "c"], "color"),
        make_issue(["x", "y"], "size"),
        make_issue([str(i) for i in range(5)], "q"),
    ]


def test_hardheaded_learns_offered_outcome_is_best():
    issues = _issues()
    m = HardHeadedFrequencyModel().initialize(issues)
    # Opponent insists on ("a","x",*), conceding only on q -> a,x important.
    for offer in [("a", "x", "0"), ("a", "x", "0"), ("a", "x", "1"), ("a", "y", "2"), ("a", "x", "0")]:
        m.update(offer)
    # The frequently/consistently offered outcome should score at/near the top.
    assert m(("a", "x", "0")) >= m(("c", "y", "4"))
    assert 0.0 <= m(("c", "y", "4")) <= 1.0
    # Issue 'color' (never changed) should be weighed >= 'q' (always changing).
    assert m._weights["color"] >= m._weights["q"]


def test_hardheaded_is_a_ufun():
    from negmas.preferences.base_ufun import BaseUtilityFunction

    m = HardHeadedFrequencyModel().initialize(_issues())
    assert isinstance(m, BaseUtilityFunction)
    # Callable like a ufun.
    assert isinstance(float(m(("a", "x", "0"))), float)


def test_uniform_and_default_constants():
    issues = _issues()
    u = UniformModel().initialize(issues)
    d = DefaultModel().initialize(issues)
    assert u(("a", "x", "0")) == 0.5
    assert d(("a", "x", "0")) == 0.0


def test_opposite_is_zero_sum_of_own():
    issues = _issues()
    own = LinearAdditiveUtilityFunction.random(outcome_space=make_os(issues), reserved_value=0.0)
    m = OppositeModel(own_ufun=own).initialize(issues)
    off = ("a", "x", "0")
    assert abs(m.eval_normalized(off) - (1.0 - own.eval_normalized(off))) < 1e-9


def test_perfect_and_worst_are_oracles():
    issues = _issues()
    opp = LinearAdditiveUtilityFunction.random(outcome_space=make_os(issues), reserved_value=0.0)
    p = PerfectModel(opponent_ufun=opp).initialize(issues)
    w = WorstModel(opponent_ufun=opp).initialize(issues)
    off = ("b", "y", "3")
    assert abs(p.eval_normalized(off) - opp.eval_normalized(off)) < 1e-9
    assert abs(w.eval_normalized(off) - (1.0 - opp.eval_normalized(off))) < 1e-9


def test_models_are_registered():
    try:
        from negmas.registry import component_registry
    except ImportError:
        pytest.skip("negmas component registry unavailable")
    import negmas_genius_agents.models  # noqa: F401  (triggers registration)

    for cls in (HardHeadedFrequencyModel, OppositeModel, UniformModel, DefaultModel, PerfectModel, WorstModel):
        assert component_registry.is_registered(cls), f"{cls.__name__} not registered"


def test_hardheaded_drives_in_a_real_negotiation():
    """Feeding a live SAO trace through the model produces sane, bounded utilities."""
    import random as _random

    _random.seed(0)
    issues = _issues()
    os_ = make_os(issues)
    buyer = LinearAdditiveUtilityFunction.random(outcome_space=os_, reserved_value=0.0)
    seller = LinearAdditiveUtilityFunction.random(outcome_space=os_, reserved_value=0.0)
    from negmas.sao import AspirationNegotiator

    mech = SAOMechanism(issues=issues, n_steps=50)
    mech.add(AspirationNegotiator(name="b"), preferences=buyer)
    mech.add(AspirationNegotiator(name="s"), preferences=seller)
    model = HardHeadedFrequencyModel().initialize(os_)
    # Drive the model from the seller's offers as seen by the buyer.
    seller_id = mech.negotiators[1].id
    while not mech.state.ended:
        mech.step()
        # last offer from seller in the trace (entries are (step, negotiator_id, offer))
        for entry in reversed(mech.extended_trace):
            tid, offer = entry[-2], entry[-1]
            if tid == seller_id and offer is not None:
                model.update(offer)
                break
    # After learning, all utilities are finite and in [0, 1] (within float epsilon).
    for outcome in os_.enumerate_or_sample():
        val = float(model(outcome))
        assert -1e-9 <= val <= 1.0 + 1e-9
