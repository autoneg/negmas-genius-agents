"""AgentGP from ANAC 2019."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["AgentGP"]


class AgentGP(SAONegotiator):
    """
    AgentGP from ANAC 2019 - Nash-based category 3rd place.

    AgentGP uses a Gaussian Process-inspired approach for opponent modeling
    and bid selection, treating the negotiation as a multi-armed bandit
    problem with uncertainty in opponent preferences.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        .. code-block:: bibtex

            @inproceedings{baarslag2019tenth,
                title={The Tenth International Automated Negotiating Agents
                       Competition (ANAC 2019)},
                author={Baarslag, Tim and Fujita, Katsuhide and Gerding,
                        Enrico H and Hindriks, Koen and Ito, Takayuki and
                        Jennings, Nicholas R and others},
                booktitle={Proceedings of the International Joint Conference
                           on Autonomous Agents and Multiagent Systems (AAMAS)},
                year={2019}
            }

        Original Genius class: ``agents.anac.y2019.agentgp.AgentGP``

    **Offering Strategy:**
        - Polynomial concession from initial target (0.95) to minimum (0.55)
        - Uses UCB (Upper Confidence Bound) style bid selection
        - UCB score = own_utility * opponent_mean + exploration_weight * sqrt(variance)
        - Exploration weight decreases linearly over time: w * (1 - t)
        - Early negotiation: high exploration (try diverse bids)
        - Late negotiation: exploitation (focus on expected value)

    **Acceptance Strategy:**
        - Accepts offers meeting the polynomial concession target
        - Near deadline (t >= 0.95): Accepts offers at 90% of target
        - Very near deadline (t >= 0.99): Accepts minimum target or 95% of
          best received offer

    **Opponent Modeling:**
        - Tracks frequency and recency of opponent's issue value choices
        - Models opponent preference as (mean, variance) tuple per issue value
        - Mean = normalized frequency (count / total_count)
        - Variance = 1 / (1 + count) - decreases with more observations
        - Higher variance indicates uncertainty, encouraging exploration

    Args:
        exploration_weight: Weight for exploration bonus (default 0.5)
        initial_target: Initial target utility (default 0.95)
        min_target: Minimum acceptable utility (default 0.55)
        near_deadline_time: Time threshold for near deadline (default 0.95)
        final_deadline_time: Time threshold for final deadline (default 0.99)
        near_deadline_ratio: Ratio of target for near deadline acceptance (default 0.9)
        final_best_ratio: Ratio of best received utility for final deadline acceptance (default 0.95)
        preferences: NegMAS preferences/utility function.
        ufun: Utility function (overrides preferences if given).
        name: Negotiator name.
        parent: Parent controller.
        owner: Agent that owns this negotiator.
        id: Unique identifier.
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(
        self,
        exploration_weight: float = 0.5,
        initial_target: float = 0.95,
        min_target: float = 0.55,
        near_deadline_time: float = 0.95,
        final_deadline_time: float = 0.99,
        near_deadline_ratio: float = 0.9,
        final_best_ratio: float = 0.95,
        preferences: BaseUtilityFunction | None = None,
        ufun: BaseUtilityFunction | None = None,
        name: str | None = None,
        parent: Controller | None = None,
        owner: Agent | None = None,
        id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            preferences=preferences,
            ufun=ufun,
            name=name,
            parent=parent,
            owner=owner,
            id=id,
            **kwargs,
        )
        self._exploration_weight = exploration_weight
        self._initial_target = initial_target
        self._min_target = min_target
        self._near_deadline_time = near_deadline_time
        self._final_deadline_time = final_deadline_time
        self._near_deadline_ratio = near_deadline_ratio
        self._final_best_ratio = final_best_ratio

        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling with mean and variance
        self._value_counts: dict[int, dict[str, int]] = {}
        self._value_sum: dict[int, dict[str, float]] = {}
        self._opponent_offers_count: int = 0

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._best_received_utility: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = self._outcome_space.min_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._value_counts = {}
        self._value_sum = {}
        self._opponent_offers_count = 0
        self._best_received_utility = 0.0

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent model with new observation."""
        if bid is None:
            return

        self._opponent_offers_count += 1

        # For each issue, track frequency and recency
        for i, value in enumerate(bid):
            if i not in self._value_counts:
                self._value_counts[i] = {}
                self._value_sum[i] = {}

            value_str = str(value)
            if value_str not in self._value_counts[i]:
                self._value_counts[i][value_str] = 0
                self._value_sum[i][value_str] = 0.0

            self._value_counts[i][value_str] += 1
            # Weight by recency (more recent = higher weight)
            self._value_sum[i][value_str] += self._opponent_offers_count

    def _estimate_opponent_utility(self, bid: Outcome) -> tuple[float, float]:
        """
        Estimate opponent utility with uncertainty.

        Returns (mean, variance) estimate.
        """
        if bid is None or self._opponent_offers_count == 0:
            return 0.5, 0.5  # High uncertainty

        total_mean = 0.0
        total_var = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            value_str = str(value)
            if i in self._value_counts:
                count = self._value_counts[i].get(value_str, 0)
                total_count = sum(self._value_counts[i].values())

                if total_count > 0:
                    # Mean is normalized frequency
                    mean = count / total_count
                    # Variance decreases with more observations
                    var = 1.0 / (1.0 + count)
                else:
                    mean = 0.5
                    var = 0.5
            else:
                mean = 0.5
                var = 0.5

            total_mean += mean
            total_var += var

        if num_issues > 0:
            return total_mean / num_issues, total_var / num_issues
        return 0.5, 0.5

    def _compute_ucb_score(self, bid: Outcome, time: float) -> float:
        """
        Compute UCB-style score for bid selection.

        Early: high exploration (consider uncertainty)
        Late: low exploration (focus on expected value)
        """
        if bid is None or self.ufun is None:
            return 0.0

        own_utility = float(self.ufun(bid))
        opp_mean, opp_var = self._estimate_opponent_utility(bid)

        # Exploration weight decreases over time
        exploration = self._exploration_weight * (1.0 - time)

        # UCB score: product of utilities + exploration bonus
        score = own_utility * opp_mean + exploration * math.sqrt(opp_var)

        return score

    def _get_target_utility(self, time: float) -> float:
        """Get target utility based on time."""
        # Polynomial concession
        target = self._initial_target - (self._initial_target - self._min_target) * (
            time**1.5
        )

        return max(target, self._min_target)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid using UCB-style selection."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return self._best_bid

        target = self._get_target_utility(time)

        # Get candidates above target
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            bid_detail = self._outcome_space.get_bid_near_utility(target)
            return bid_detail.bid if bid_detail else self._best_bid

        if len(candidates) == 1:
            return candidates[0].bid

        # Select bid with highest UCB score
        best_score = -float("inf")
        best_bid = candidates[0].bid

        for bd in candidates:
            score = self._compute_ucb_score(bd.bid, time)
            if score > best_score:
                best_score = score
                best_bid = bd.bid

        return best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        # First move: best bid
        if self._opponent_offers_count == 0:
            return self._best_bid

        time = state.relative_time
        return self._select_bid(time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        # Update opponent model
        self._update_opponent_model(offer)

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        time = state.relative_time

        # Track best received
        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility

        # Get target utility
        target = self._get_target_utility(time)

        # Accept if above target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, be more flexible
        if time >= self._near_deadline_time:
            # Accept if close to target
            if offer_utility >= target * self._near_deadline_ratio:
                return ResponseType.ACCEPT_OFFER

        if time >= self._final_deadline_time:
            # Accept anything reasonable
            if offer_utility >= self._min_target:
                return ResponseType.ACCEPT_OFFER
            if offer_utility >= self._best_received_utility * self._final_best_ratio:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
