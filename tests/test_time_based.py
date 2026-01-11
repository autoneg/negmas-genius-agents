"""
Tests for the time-dependent negotiating agents.

These tests verify that:
1. Agents can be instantiated and used in NegMAS mechanisms
2. Negotiations complete without errors
3. The concession behavior matches expected patterns
4. Agents can negotiate with each other and with NegMAS agents
"""

import pytest
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism, AspirationNegotiator

from negmas_genius_agents import (
    TimeDependentAgent,
    TimeDependentAgentBoulware,
    TimeDependentAgentConceder,
    TimeDependentAgentLinear,
    TimeDependentAgentHardliner,
)


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


class TestAgentInstantiation:
    """Test that agents can be properly instantiated."""

    def test_boulware_instantiation(self):
        """Test TimeDependentAgentBoulware can be created."""
        agent = TimeDependentAgentBoulware(name="test_boulware")
        assert agent is not None
        assert agent.e == 0.2

    def test_conceder_instantiation(self):
        """Test TimeDependentAgentConceder can be created."""
        agent = TimeDependentAgentConceder(name="test_conceder")
        assert agent is not None
        assert agent.e == 2.0

    def test_linear_instantiation(self):
        """Test TimeDependentAgentLinear can be created."""
        agent = TimeDependentAgentLinear(name="test_linear")
        assert agent is not None
        assert agent.e == 1.0

    def test_hardliner_instantiation(self):
        """Test TimeDependentAgentHardliner can be created."""
        agent = TimeDependentAgentHardliner(name="test_hardliner")
        assert agent is not None
        assert agent.e == 0.0

    def test_custom_e_value(self):
        """Test TimeDependentAgent with custom e value."""
        agent = TimeDependentAgent(e=0.5, name="test_custom")
        assert agent is not None
        assert agent.e == 0.5


class TestBasicNegotiation:
    """Test basic negotiation functionality."""

    def test_boulware_vs_conceder(self, simple_issues, buyer_ufun, seller_ufun):
        """Test a negotiation between Boulware and Conceder agents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = TimeDependentAgentBoulware(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentConceder(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    def test_linear_vs_linear(self, simple_issues, buyer_ufun, seller_ufun):
        """Test a negotiation between two TimeDependentAgentLinear instances."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = TimeDependentAgentLinear(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentLinear(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    def test_hardliner_vs_conceder(self, simple_issues, buyer_ufun, seller_ufun):
        """Test Hardliner vs Conceder - Conceder should concede."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = TimeDependentAgentHardliner(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentConceder(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended


class TestConcessionBehavior:
    """Test that concession patterns match expected behavior."""

    def test_boulware_concedes_slowly(self, buyer_ufun):
        """Test that Boulware agent's concession function is slow."""
        agent = TimeDependentAgentBoulware(name="test", ufun=buyer_ufun)

        # At t=0.5, Boulware should have conceded less than Linear
        f_boulware = agent.f(0.5)
        f_linear = 0.5  # Linear would be at 0.5

        assert f_boulware < f_linear, "Boulware should concede slower than Linear"

    def test_conceder_concedes_quickly(self, buyer_ufun):
        """Test that Conceder agent's concession function is fast."""
        agent = TimeDependentAgentConceder(name="test", ufun=buyer_ufun)

        # At t=0.5, Conceder should have conceded more than Linear
        f_conceder = agent.f(0.5)
        f_linear = 0.5  # Linear would be at 0.5

        assert f_conceder > f_linear, "Conceder should concede faster than Linear"

    def test_linear_concedes_linearly(self, buyer_ufun):
        """Test that Linear agent's concession function is linear."""
        agent = TimeDependentAgentLinear(name="test", ufun=buyer_ufun)

        # At t=0.5, Linear should be at approximately 0.5
        f_linear = agent.f(0.5)

        assert abs(f_linear - 0.5) < 0.01, "Linear should concede linearly"

    def test_hardliner_never_concedes(self, buyer_ufun):
        """Test that Hardliner agent never concedes."""
        agent = TimeDependentAgentHardliner(name="test", ufun=buyer_ufun)

        # At any time, Hardliner should remain at k=0
        assert agent.f(0.0) == 0.0
        assert agent.f(0.5) == 0.0
        assert agent.f(1.0) == 0.0


class TestAgreementQuality:
    """Test that agreements (when reached) are valid and reasonable."""

    def test_agreement_is_valid_outcome(self, simple_issues, buyer_ufun, seller_ufun):
        """If agreement is reached, it should be a valid outcome."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=200)

        buyer = TimeDependentAgentConceder(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentConceder(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        if state.agreement is not None:
            # Agreement should be a tuple with 3 values (one per issue)
            assert len(state.agreement) == 3
            # Each value should be valid for its issue
            price, quantity, delivery = state.agreement
            assert price in ["low", "medium", "high"]
            assert quantity in ["1", "2", "3"]
            assert delivery in ["fast", "normal", "slow"]

    def test_agreement_utilities_are_positive(
        self, simple_issues, buyer_ufun, seller_ufun
    ):
        """If agreement is reached, both parties should get non-negative utility."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=200)

        buyer = TimeDependentAgentConceder(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentConceder(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        if state.agreement is not None:
            buyer_util = buyer_ufun(state.agreement)
            seller_util = seller_ufun(state.agreement)
            assert buyer_util >= 0
            assert seller_util >= 0


class TestMixedNegotiations:
    """Test negotiations between Genius agents and NegMAS agents."""

    def test_boulware_vs_aspiration(self, simple_issues, buyer_ufun, seller_ufun):
        """Test Genius TimeDependentAgentBoulware vs NegMAS AspirationNegotiator."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = TimeDependentAgentBoulware(name="buyer", ufun=buyer_ufun)
        seller = AspirationNegotiator(name="seller")

        mechanism.add(buyer)
        mechanism.add(seller, preferences=seller_ufun)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    def test_conceder_vs_aspiration(self, simple_issues, buyer_ufun, seller_ufun):
        """Test Genius TimeDependentAgentConceder vs NegMAS AspirationNegotiator."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = TimeDependentAgentConceder(name="buyer", ufun=buyer_ufun)
        seller = AspirationNegotiator(name="seller")

        mechanism.add(buyer)
        mechanism.add(seller, preferences=seller_ufun)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended


class TestEdgeCases:
    """Test edge cases and potential issues."""

    def test_single_step_negotiation(self, simple_issues, buyer_ufun, seller_ufun):
        """Test negotiation with minimal steps."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=1)

        buyer = TimeDependentAgentBoulware(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentBoulware(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.ended

    def test_many_steps_negotiation(self, simple_issues, buyer_ufun, seller_ufun):
        """Test negotiation with many steps."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=1000)

        buyer = TimeDependentAgentConceder(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentConceder(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.ended


class TestAllAgentCombinations:
    """Test all combinations of agent types."""

    @pytest.mark.parametrize(
        "agent_a_class,agent_b_class",
        [
            (TimeDependentAgentBoulware, TimeDependentAgentBoulware),
            (TimeDependentAgentBoulware, TimeDependentAgentConceder),
            (TimeDependentAgentBoulware, TimeDependentAgentLinear),
            (TimeDependentAgentBoulware, TimeDependentAgentHardliner),
            (TimeDependentAgentConceder, TimeDependentAgentConceder),
            (TimeDependentAgentConceder, TimeDependentAgentLinear),
            (TimeDependentAgentConceder, TimeDependentAgentHardliner),
            (TimeDependentAgentLinear, TimeDependentAgentLinear),
            (TimeDependentAgentLinear, TimeDependentAgentHardliner),
            (TimeDependentAgentHardliner, TimeDependentAgentHardliner),
        ],
    )
    def test_agent_combination(
        self, agent_a_class, agent_b_class, simple_issues, buyer_ufun, seller_ufun
    ):
        """Test all combinations of agent types complete without errors."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=50)

        buyer = agent_a_class(name="buyer", ufun=buyer_ufun)
        seller = agent_b_class(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended
