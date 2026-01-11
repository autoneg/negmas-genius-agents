"""
Tests for ANAC 2010 negotiating agents.

These tests verify that:
1. AgentK can be instantiated and used in NegMAS mechanisms
2. Negotiations complete without errors
3. The agent behavior matches expected patterns
4. AgentK can negotiate with various opponent types
"""

import pytest
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism, AspirationNegotiator

from negmas_genius_agents import (
    AgentK,
    HardHeaded,
    TimeDependentAgentBoulware,
    TimeDependentAgentConceder,
    TimeDependentAgentLinear,
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


class TestAgentKInstantiation:
    """Test that AgentK can be properly instantiated."""

    def test_default_instantiation(self):
        """Test AgentK can be created with defaults."""
        agent = AgentK(name="test_agentk")
        assert agent is not None

    def test_custom_tremor(self):
        """Test AgentK with custom tremor parameter."""
        agent = AgentK(name="test_custom", tremor=1.0)
        assert agent is not None
        assert agent._tremor == 1.0


class TestAgentKBasicNegotiation:
    """Test basic negotiation functionality of AgentK."""

    def test_agentk_vs_conceder(self, simple_issues, buyer_ufun, seller_ufun):
        """Test a negotiation between AgentK and Conceder agents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = AgentK(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentConceder(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    def test_agentk_vs_boulware(self, simple_issues, buyer_ufun, seller_ufun):
        """Test a negotiation between AgentK and Boulware agents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = AgentK(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentBoulware(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    def test_agentk_vs_linear(self, simple_issues, buyer_ufun, seller_ufun):
        """Test a negotiation between AgentK and Linear agents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = AgentK(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentLinear(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    def test_agentk_vs_agentk(self, simple_issues, buyer_ufun, seller_ufun):
        """Test a negotiation between two AgentK instances."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = AgentK(name="buyer", ufun=buyer_ufun)
        seller = AgentK(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    def test_agentk_vs_hardheaded(self, simple_issues, buyer_ufun, seller_ufun):
        """Test AgentK vs HardHeaded (both ANAC winners)."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = AgentK(name="buyer", ufun=buyer_ufun)
        seller = HardHeaded(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    def test_agentk_vs_aspiration(self, simple_issues, buyer_ufun, seller_ufun):
        """Test AgentK vs NegMAS AspirationNegotiator."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = AgentK(name="buyer", ufun=buyer_ufun)
        seller = AspirationNegotiator(name="seller")

        mechanism.add(buyer)
        mechanism.add(seller, preferences=seller_ufun)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended


class TestAgentKStatistics:
    """Test AgentK's statistical tracking."""

    def test_statistics_updated(self, simple_issues, buyer_ufun, seller_ufun):
        """Test that statistics are updated during negotiation."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=50)

        buyer = AgentK(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentLinear(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        # After negotiation, buyer should have tracked opponent offers
        assert buyer._rounds > 0, "Should have tracked opponent offers"
        assert buyer._sum > 0, "Sum of utilities should be positive"

    def test_offered_bids_tracked(self, simple_issues, buyer_ufun, seller_ufun):
        """Test that offered bids are tracked."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = AgentK(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentConceder(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        # Should have tracked bids from opponent
        assert len(buyer._offered_bids) > 0, "Should have tracked opponent bids"


class TestAgentKAgreementQuality:
    """Test that AgentK agreements are valid and reasonable."""

    def test_agreement_is_valid_outcome(self, simple_issues, buyer_ufun, seller_ufun):
        """If agreement is reached, it should be a valid outcome."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=200)

        buyer = AgentK(name="buyer", ufun=buyer_ufun)
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


class TestAgentKEdgeCases:
    """Test edge cases for AgentK."""

    def test_single_step_negotiation(self, simple_issues, buyer_ufun, seller_ufun):
        """Test negotiation with minimal steps."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=1)

        buyer = AgentK(name="buyer", ufun=buyer_ufun)
        seller = AgentK(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.ended

    def test_many_steps_negotiation(self, simple_issues, buyer_ufun, seller_ufun):
        """Test negotiation with many steps."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=500)

        buyer = AgentK(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentConceder(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.ended


class TestAgentKVsAllAgents:
    """Test AgentK against all other agent types."""

    @pytest.mark.parametrize(
        "opponent_class",
        [
            TimeDependentAgentBoulware,
            TimeDependentAgentConceder,
            TimeDependentAgentLinear,
            HardHeaded,
            AgentK,
        ],
    )
    def test_agentk_as_buyer(
        self, opponent_class, simple_issues, buyer_ufun, seller_ufun
    ):
        """Test AgentK as buyer against various opponents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = AgentK(name="buyer", ufun=buyer_ufun)
        seller = opponent_class(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    @pytest.mark.parametrize(
        "opponent_class",
        [
            TimeDependentAgentBoulware,
            TimeDependentAgentConceder,
            TimeDependentAgentLinear,
            HardHeaded,
            AgentK,
        ],
    )
    def test_agentk_as_seller(
        self, opponent_class, simple_issues, buyer_ufun, seller_ufun
    ):
        """Test AgentK as seller against various opponents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = opponent_class(name="buyer", ufun=buyer_ufun)
        seller = AgentK(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended
