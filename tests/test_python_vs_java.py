"""Behavioral comparison of the Python agent ports against the real Java Genius
agents, run through the ``negmas-genius`` bridge.

Each Python agent in this package is a reimplementation of a Java Genius agent.
This module lets us *observe* both on the same negotiation and check that the
Python port behaves in the same ballpark as the original Java implementation
(which runs under the hood via ``negmas.genius.GeniusNegotiator``).

Two levels of testing:

- :func:`test_python_agent_runs` (always runs) — every Python agent completes a
  negotiation cleanly against a fixed opponent, without crashing.
- :func:`test_python_vs_java_behavior` (only when the Genius bridge is running) —
  runs the Python port *and* the real Java agent on the same domain against the
  same opponent and asserts the two agree on whether a deal is reached and that
  the achieved utilities are in the same ballpark. Individual agents whose Java
  side errors inside the bridge (a handful of ANAC agents crash when run
  bilaterally with full information) are skipped, not failed.

Run only the bridge comparison with::

    pytest tests/test_python_vs_java.py -k behavior

These tests are approximate by design: the ports are AI-assisted and not
bit-for-bit identical, so tolerances are deliberately loose.
"""

from __future__ import annotations

import pytest

from negmas.sao import SAOMechanism

from negmas_genius_agents import TimeDependentAgentConceder

from tests.test_behavioral_comparison import (
    AGENT_MAPPING,
    PYTHON_CLASSES,
    create_test_domain,
)

try:
    from negmas.genius import GeniusNegotiator, genius_bridge_is_running

    BRIDGE_RUNNING = genius_bridge_is_running()
except Exception:  # pragma: no cover - genius optional
    GeniusNegotiator = None  # type: ignore
    BRIDGE_RUNNING = False

# Loose "same ballpark" tolerance on the agent's own final utility (range is [0, 1]).
UTIL_TOLERANCE = 0.34
N_STEPS = 100


def _run(agent, agent_ufun, opponent, opponent_ufun):
    """Run one negotiation and return (agreement_reached, agent_final_utility)."""
    mech = SAOMechanism(issues=agent_ufun.outcome_space.issues, n_steps=N_STEPS)
    mech.add(agent, preferences=agent_ufun)
    mech.add(opponent, preferences=opponent_ufun)
    state = mech.run()
    if state.agreement is None:
        return False, None
    return True, float(agent_ufun(state.agreement))


@pytest.mark.parametrize("name", sorted(PYTHON_CLASSES.keys()))
def test_python_agent_runs(name):
    """Every Python agent completes a negotiation vs a conceder without crashing."""
    _issues, buyer, seller = create_test_domain()
    agent = PYTHON_CLASSES[name](name="agent")
    reached, _util = _run(agent, buyer, TimeDependentAgentConceder(name="opp"), seller)
    assert isinstance(reached, bool)  # ran to completion without raising


@pytest.mark.skipif(not BRIDGE_RUNNING, reason="Genius bridge not running")
@pytest.mark.parametrize("name", sorted(AGENT_MAPPING.keys()))
def test_python_vs_java_behavior(name):
    """Python port and real Java agent behave in the same ballpark on one domain."""
    if name not in PYTHON_CLASSES:
        pytest.skip(f"no Python class for {name}")
    java_class = AGENT_MAPPING[name]
    _issues, buyer, seller = create_test_domain()

    # Python port vs a conceder.
    py_reached, py_util = _run(
        PYTHON_CLASSES[name](name="py"), buyer, TimeDependentAgentConceder(name="opp"), seller
    )

    # Real Java agent vs the same conceder on the same domain.
    try:
        java = GeniusNegotiator(java_class_name=java_class)
        jv_reached, jv_util = _run(
            java, buyer, TimeDependentAgentConceder(name="opp"), seller
        )
    except Exception as e:  # Java agent crashes inside the bridge for some ANAC agents
        pytest.skip(f"Java agent {java_class} errored in bridge: {e}")

    # Both should agree on whether a deal is reached.
    assert py_reached == jv_reached, (
        f"{name}: agreement mismatch (python={py_reached}, java={jv_reached})"
    )
    # When both reach a deal, the achieved utilities should be in the same ballpark.
    if py_reached and jv_reached and py_util is not None and jv_util is not None:
        assert abs(py_util - jv_util) <= UTIL_TOLERANCE, (
            f"{name}: utility gap too large (python={py_util:.2f}, java={jv_util:.2f})"
        )
