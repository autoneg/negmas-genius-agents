"""
Tests for ANAC 2019 negotiating agents.

Tests verify that all agents can:
1. Be instantiated correctly
2. Complete negotiations without errors
3. Handle various negotiation scenarios
"""

import pytest
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism

from negmas_genius_agents import get_agents
from negmas_genius_agents.negotiators.anac.y2019 import (
    AgentGG,
    AgentGP,
    AgentLarry,
    DandikAgent,
    EAgent,
    FSEGA2019,
    GaravelAgent,
    Gravity,
    HardDealer,
    KAgent,
    KakeSoba,
    MINF,
    SAGA,
    WinkyAgent,
)

# Get all 2019 agents
ANAC_2019_AGENTS = get_agents(group="anac2019")


@pytest.fixture
def simple_issues():
    """Create a simple negotiation domain with 3 discrete issues."""
    issues = [
        make_issue(values=["low", "medium", "high"], name="price"),
        make_issue(values=["1", "2", "3"], name="quantity"),
        make_issue(values=["fast", "normal", "slow"], name="delivery"),
    ]
    return issues


@pytest.fixture
def outcome_space(simple_issues):
    """Create outcome space from issues."""
    return make_os(simple_issues)


@pytest.fixture
def buyer_ufun(outcome_space):
    """Create a utility function for a buyer (prefers low price, high quantity)."""
    return LinearAdditiveUtilityFunction(
        values={
            "price": {"low": 1.0, "medium": 0.5, "high": 0.0},
            "quantity": {"1": 0.0, "2": 0.5, "3": 1.0},
            "delivery": {"fast": 1.0, "normal": 0.5, "slow": 0.0},
        },
        weights={"price": 0.5, "quantity": 0.3, "delivery": 0.2},
        outcome_space=outcome_space,
    )


@pytest.fixture
def seller_ufun(outcome_space):
    """Create a utility function for a seller (prefers high price, low quantity)."""
    return LinearAdditiveUtilityFunction(
        values={
            "price": {"low": 0.0, "medium": 0.5, "high": 1.0},
            "quantity": {"1": 1.0, "2": 0.5, "3": 0.0},
            "delivery": {"fast": 0.0, "normal": 0.5, "slow": 1.0},
        },
        weights={"price": 0.5, "quantity": 0.3, "delivery": 0.2},
        outcome_space=outcome_space,
    )


class TestANAC2019AgentInstantiation:
    """Test that all ANAC 2019 agents can be properly instantiated."""

    @pytest.mark.parametrize("agent_class", ANAC_2019_AGENTS)
    def test_instantiation(self, agent_class):
        """Test that each agent can be instantiated with defaults."""
        agent = agent_class(name=f"test_{agent_class.__name__}")
        assert agent is not None
        assert agent.name == f"test_{agent_class.__name__}"


class TestANAC2019AgentNegotiations:
    """Test basic negotiation functionality of ANAC 2019 agents."""

    @pytest.mark.parametrize("agent_class", ANAC_2019_AGENTS)
    def test_agent_completes_negotiation(
        self, agent_class, simple_issues, buyer_ufun, seller_ufun
    ):
        """Test that each agent can complete a negotiation."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = agent_class(name="buyer", ufun=buyer_ufun)
        seller = agent_class(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    @pytest.mark.parametrize(
        "buyer_class,seller_class",
        [
            (AgentGG, AgentGP),
            (AgentGP, AgentLarry),
            (AgentLarry, DandikAgent),
            (DandikAgent, EAgent),
            (EAgent, FSEGA2019),
            (FSEGA2019, GaravelAgent),
            (GaravelAgent, Gravity),
            (Gravity, HardDealer),
            (HardDealer, KAgent),
            (KAgent, KakeSoba),
            (KakeSoba, MINF),
            (MINF, SAGA),
            (SAGA, WinkyAgent),
            (WinkyAgent, AgentGG),
        ],
    )
    def test_cross_agent_negotiation(
        self, buyer_class, seller_class, simple_issues, buyer_ufun, seller_ufun
    ):
        """Test negotiations between different 2019 agents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = buyer_class(name="buyer", ufun=buyer_ufun)
        seller = seller_class(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended


class TestANAC2019EdgeCases:
    """Test edge cases for ANAC 2019 agents."""

    @pytest.mark.parametrize("agent_class", ANAC_2019_AGENTS)
    def test_single_step_negotiation(
        self, agent_class, simple_issues, buyer_ufun, seller_ufun
    ):
        """Test negotiation with minimal steps."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=1)

        buyer = agent_class(name="buyer", ufun=buyer_ufun)
        seller = agent_class(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.ended

    @pytest.mark.parametrize("agent_class", ANAC_2019_AGENTS)
    def test_many_steps_negotiation(
        self, agent_class, simple_issues, buyer_ufun, seller_ufun
    ):
        """Test negotiation with many steps."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=500)

        buyer = agent_class(name="buyer", ufun=buyer_ufun)
        seller = agent_class(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.ended
