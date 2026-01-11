"""
Tests for ANAC 2011 negotiating agents.

These tests verify that:
1. HardHeaded agent can be instantiated and used in NegMAS mechanisms
2. Negotiations complete without errors
3. The agent behavior matches expected patterns
4. HardHeaded can negotiate with various opponent types
"""

import pytest
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism, AspirationNegotiator

from negmas_genius_agents import (
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


class TestHardHeadedInstantiation:
    """Test that HardHeaded agent can be properly instantiated."""

    def test_default_instantiation(self):
        """Test HardHeaded can be created with defaults."""
        agent = HardHeaded(name="test_hardheaded")
        assert agent is not None

    def test_custom_parameters(self):
        """Test HardHeaded with custom parameters."""
        agent = HardHeaded(
            name="test_custom",
            ka=0.1,
            e=0.1,
            min_utility=0.6,
        )
        assert agent is not None


class TestHardHeadedBasicNegotiation:
    """Test basic negotiation functionality of HardHeaded agent."""

    def test_hardheaded_vs_conceder(self, simple_issues, buyer_ufun, seller_ufun):
        """Test a negotiation between HardHeaded and Conceder agents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentConceder(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    def test_hardheaded_vs_boulware(self, simple_issues, buyer_ufun, seller_ufun):
        """Test a negotiation between HardHeaded and Boulware agents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentBoulware(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    def test_hardheaded_vs_linear(self, simple_issues, buyer_ufun, seller_ufun):
        """Test a negotiation between HardHeaded and Linear agents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentLinear(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    def test_hardheaded_vs_hardheaded(self, simple_issues, buyer_ufun, seller_ufun):
        """Test a negotiation between two HardHeaded agents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
        seller = HardHeaded(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended

    def test_hardheaded_vs_aspiration(self, simple_issues, buyer_ufun, seller_ufun):
        """Test HardHeaded vs NegMAS AspirationNegotiator."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
        seller = AspirationNegotiator(name="seller")

        mechanism.add(buyer)
        mechanism.add(seller, preferences=seller_ufun)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended


class TestHardHeadedConcession:
    """Test HardHeaded concession behavior."""

    def test_concession_function_starts_high(self, buyer_ufun):
        """Test that HardHeaded starts with high target utility."""
        agent = HardHeaded(name="test", ufun=buyer_ufun)
        agent._initialize()

        # At t=0, target should be near maximum
        target_t0 = agent._get_target_utility(0.0)
        assert target_t0 >= 0.9, "HardHeaded should start near max utility"

    def test_concession_function_slow(self, buyer_ufun):
        """Test that HardHeaded concedes slowly."""
        agent = HardHeaded(name="test", ufun=buyer_ufun)
        agent._initialize()

        # At t=0.5, HardHeaded should still have high target
        target_t05 = agent._get_target_utility(0.5)
        target_t0 = agent._get_target_utility(0.0)

        # Should have conceded less than 50% of the range
        concession = target_t0 - target_t05
        max_range = target_t0 - agent._min_utility
        assert concession < 0.5 * max_range, "HardHeaded should concede slowly"


class TestHardHeadedAgreementQuality:
    """Test that HardHeaded agreements (when reached) are valid and reasonable."""

    def test_agreement_is_valid_outcome(self, simple_issues, buyer_ufun, seller_ufun):
        """If agreement is reached, it should be a valid outcome."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=200)

        buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
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

    def test_hardheaded_gets_good_agreements(
        self, simple_issues, buyer_ufun, seller_ufun
    ):
        """HardHeaded should get good utility when negotiating with Conceder."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=200)

        buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentConceder(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        if state.agreement is not None:
            buyer_util = buyer_ufun(state.agreement)
            # HardHeaded should get reasonably good utility against a Conceder
            assert buyer_util >= 0.5, "HardHeaded should get good utility vs Conceder"


class TestHardHeadedOpponentModeling:
    """Test HardHeaded's opponent modeling functionality."""

    def test_opponent_model_updates(self, simple_issues, buyer_ufun, seller_ufun):
        """Test that opponent model is updated during negotiation."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=50)

        buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentLinear(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        # After negotiation, buyer should have recorded opponent bids
        assert len(buyer._opponent_bids) > 0, "Should have recorded opponent bids"

    def test_opponent_best_bid_tracked(self, simple_issues, buyer_ufun, seller_ufun):
        """Test that best opponent bid is tracked."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentConceder(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        # Should have tracked the best bid from opponent
        if len(buyer._opponent_bids) > 0:
            assert buyer._opponent_best_bid is not None, (
                "Should track best opponent bid"
            )
            assert buyer._opponent_best_bid_utility > 0, (
                "Best bid should have positive utility"
            )


class TestHardHeadedEdgeCases:
    """Test edge cases for HardHeaded agent."""

    def test_single_step_negotiation(self, simple_issues, buyer_ufun, seller_ufun):
        """Test negotiation with minimal steps."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=1)

        buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
        seller = HardHeaded(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.ended

    def test_many_steps_negotiation(self, simple_issues, buyer_ufun, seller_ufun):
        """Test negotiation with many steps."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=500)

        buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
        seller = TimeDependentAgentConceder(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.ended


class TestHardHeadedVsAllAgents:
    """Test HardHeaded against all other agent types."""

    @pytest.mark.parametrize(
        "opponent_class",
        [
            TimeDependentAgentBoulware,
            TimeDependentAgentConceder,
            TimeDependentAgentLinear,
            HardHeaded,
        ],
    )
    def test_hardheaded_as_buyer(
        self, opponent_class, simple_issues, buyer_ufun, seller_ufun
    ):
        """Test HardHeaded as buyer against various opponents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
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
        ],
    )
    def test_hardheaded_as_seller(
        self, opponent_class, simple_issues, buyer_ufun, seller_ufun
    ):
        """Test HardHeaded as seller against various opponents."""
        mechanism = SAOMechanism(issues=simple_issues, n_steps=100)

        buyer = opponent_class(name="buyer", ufun=buyer_ufun)
        seller = HardHeaded(name="seller", ufun=seller_ufun)

        mechanism.add(buyer)
        mechanism.add(seller)

        state = mechanism.run()

        assert state is not None
        assert state.started
        assert state.ended
