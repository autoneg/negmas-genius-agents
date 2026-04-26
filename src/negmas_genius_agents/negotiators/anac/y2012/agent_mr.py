"""AgentMR from ANAC 2012."""

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

__all__ = ["AgentMR"]


class AgentMR(SAONegotiator):
    """
    AgentMR (Mixed Reality) negotiation agent from ANAC 2012.

    AgentMR uses a multi-phase negotiation strategy with risk-aware utility
    computation and opponent behavior prediction.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        Original Genius class: ``agents.anac.y2012.AgentMR.AgentMR``

        ANAC 2012: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        The agent operates in three distinct phases:

        - Exploration phase (t < exploration_ratio): Bids high with some
          randomness to learn about opponent preferences. Target utility
          decreases slowly: max - (max - min) * t * 0.5. Random bid selection
          from candidates within tolerance.
        - Exploitation phase (exploration_ratio <= t < 0.9): Uses Boulware-like
          concession with risk adjustment. Target: max - (max - min) * t^2.
          Selects from top-k candidates near target utility.
        - Compromise phase (t >= 0.9): Faster linear concession toward minimum
          acceptable utility. Prioritizes reaching agreement.

        If opponent's best bid meets target, offers that bid back to signal
        willingness to agree.

    **Acceptance Strategy:**
        Risk-aware acceptance using multiple criteria:

        - Accept if offer utility >= target utility.
        - Accept if offer >= predicted opponent next bid * (1 - time) and
          offer >= minimum utility.
        - Compromise phase: Accept if offer >= opponent's best - 0.02 or
          offer >= minimum utility.
        - Near deadline (t > 0.99): Accept any offer >= minimum utility.
        - Accept if offer >= utility of bid we would propose.

    **Opponent Modeling:**
        Tracks opponent behavior for prediction and adaptation:

        - Maintains history of opponent bids with utility statistics (mean,
          variance).
        - Tracks best opponent bid for potential reciprocation.
        - Uses linear regression on recent bids to predict next opponent bid
          utility.
        - Estimates agreement probability from opponent utility variance
          (higher variance = less predictable = higher risk).

    Args:
        exploration_ratio: Fraction of time spent in exploration (default 0.3).
        risk_aversion: Risk aversion coefficient (default 0.5). Higher values
            prefer lower but more certain outcomes.
        min_utility: Minimum acceptable utility (default 0.6).
        compromise_time: Time threshold for entering compromise phase (default 0.9).
        final_deadline_time: Time threshold for accepting any offer above
            minimum utility (default 0.99).
        stdev_multiplier: Multiplier for standard deviation in agreement
            probability estimation (default 2.0).
        exploration_concession_factor: Concession factor during exploration
            (default 0.5).
        compromise_start_utility: Starting utility multiplier for compromise
            phase (default 0.75).
        exploration_tolerance: Bid selection tolerance during exploration
            (default 0.05).
        exploitation_tolerance: Bid selection tolerance during exploitation
            (default 0.02).
        top_k_candidates: Number of top candidates to select from (default 5).
        acceptance_margin: Margin below opponent best for compromise acceptance
            (default 0.02).
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
        exploration_ratio: float = 0.3,
        risk_aversion: float = 0.5,
        min_utility: float = 0.6,
        compromise_time: float = 0.9,
        final_deadline_time: float = 0.99,
        stdev_multiplier: float = 2.0,
        exploration_concession_factor: float = 0.5,
        compromise_start_utility: float = 0.75,
        exploration_tolerance: float = 0.05,
        exploitation_tolerance: float = 0.02,
        top_k_candidates: int = 5,
        acceptance_margin: float = 0.02,
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
        self._exploration_ratio = exploration_ratio
        self._risk_aversion = risk_aversion
        self._min_utility = min_utility
        self._compromise_time = compromise_time
        self._final_deadline_time = final_deadline_time
        self._stdev_multiplier = stdev_multiplier
        self._exploration_concession_factor = exploration_concession_factor
        self._compromise_start_utility = compromise_start_utility
        self._exploration_tolerance = exploration_tolerance
        self._exploitation_tolerance = exploitation_tolerance
        self._top_k_candidates = top_k_candidates
        self._acceptance_margin = acceptance_margin

        # Will be initialized when negotiation starts
        self._outcome_space: SortedOutcomeSpace | None = None
        self._max_utility: float = 1.0
        self._reservation_value: float = 0.0
        self._initialized = False

        # Phase tracking
        self._current_phase: str = (
            "exploration"  # exploration, exploitation, compromise
        )

        # Opponent modeling
        self._opponent_bids: list[tuple[Outcome, float]] = []  # (bid, our_utility)
        self._opponent_best_bid: Outcome | None = None
        self._opponent_best_utility: float = 0.0
        self._opponent_avg_utility: float = 0.0
        self._opponent_utility_variance: float = 0.0
        self._predicted_opponent_next: float = 0.0

        # Own bidding state
        self._own_bids: list[tuple[Outcome, float]] = []
        self._last_bid: Outcome | None = None
        self._target_utility: float = 1.0

        # Risk assessment
        self._estimated_agreement_probability: float = 0.5
        self._risk_adjusted_target: float = 1.0

    def _initialize(self) -> None:
        """Initialize the outcome space and utility bounds."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
        else:
            self._max_utility = 1.0

        # Get reservation value if available
        reservation = getattr(self.ufun, "reserved_value", 0.0)
        if reservation is not None and reservation != float("-inf"):
            self._reservation_value = max(0.0, reservation)
            self._min_utility = max(self._min_utility, self._reservation_value)

        self._target_utility = self._max_utility
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._current_phase = "exploration"
        self._opponent_bids = []
        self._opponent_best_bid = None
        self._opponent_best_utility = 0.0
        self._opponent_avg_utility = 0.0
        self._opponent_utility_variance = 0.0
        self._predicted_opponent_next = 0.0
        self._own_bids = []
        self._last_bid = None
        self._target_utility = self._max_utility
        self._estimated_agreement_probability = 0.5
        self._risk_adjusted_target = self._max_utility

    def _update_phase(self, time: float) -> None:
        """Update the current negotiation phase based on time."""
        if time < self._exploration_ratio:
            self._current_phase = "exploration"
        elif time < self._compromise_time:
            self._current_phase = "exploitation"
        else:
            self._current_phase = "compromise"

    def _update_opponent_model(self, bid: Outcome) -> None:
        """
        Update opponent model with a new bid.

        Tracks statistics and predicts future opponent behavior.
        """
        if self.ufun is None or bid is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append((bid, utility))

        # Track best bid
        if utility > self._opponent_best_utility:
            self._opponent_best_utility = utility
            self._opponent_best_bid = bid

        # Update statistics
        utilities = [u for _, u in self._opponent_bids]
        n = len(utilities)
        self._opponent_avg_utility = sum(utilities) / n

        if n > 1:
            variance = sum((u - self._opponent_avg_utility) ** 2 for u in utilities) / n
            self._opponent_utility_variance = variance

        # Predict next opponent bid utility
        self._predict_opponent_next()

    def _predict_opponent_next(self) -> None:
        """
        Predict the utility of opponent's next bid.

        Uses linear extrapolation from recent trend.
        """
        if len(self._opponent_bids) < 3:
            self._predicted_opponent_next = self._opponent_avg_utility
            return

        # Use recent bids for prediction
        recent = [u for _, u in self._opponent_bids[-5:]]
        n = len(recent)

        if n < 2:
            self._predicted_opponent_next = self._opponent_avg_utility
            return

        # Linear regression
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator
            # Predict next value
            self._predicted_opponent_next = recent[-1] + slope
        else:
            self._predicted_opponent_next = self._opponent_avg_utility

        # Bound the prediction
        self._predicted_opponent_next = max(
            0.0, min(1.0, self._predicted_opponent_next)
        )

    def _compute_risk_adjusted_utility(self, base_utility: float, time: float) -> float:
        """
        Compute risk-adjusted target utility.

        Considers the risk of no agreement vs accepting lower utility.
        """
        # Estimate agreement probability based on opponent behavior
        if len(self._opponent_bids) > 5:
            # Higher variance = less predictable = higher risk
            stdev = (
                math.sqrt(self._opponent_utility_variance)
                if self._opponent_utility_variance > 0
                else 0
            )
            self._estimated_agreement_probability = max(
                0.2, 1.0 - stdev * self._stdev_multiplier
            )
        else:
            self._estimated_agreement_probability = 0.5

        # Time pressure increases risk of no agreement
        time_risk = 1.0 - math.pow(1.0 - time, 2)

        # Risk-adjusted utility considers expected value
        # With probability (1 - agreement_prob * (1 - time_risk)), we get nothing
        risk_factor = self._estimated_agreement_probability * (1.0 - time_risk)

        # Risk-averse agents prefer lower but more certain outcomes
        adjusted_min = self._min_utility + (base_utility - self._min_utility) * (
            1.0 - self._risk_aversion
        )

        # Blend based on risk factor
        risk_adjusted = base_utility * risk_factor + adjusted_min * (1.0 - risk_factor)

        return max(risk_adjusted, self._min_utility)

    def _compute_target_utility(self, time: float) -> float:
        """
        Compute target utility based on current phase and risk assessment.
        """
        self._update_phase(time)

        if self._current_phase == "exploration":
            # During exploration, bid high to learn about opponent
            base_target = (
                self._max_utility
                - (self._max_utility - self._min_utility)
                * time
                * self._exploration_concession_factor
            )
        elif self._current_phase == "exploitation":
            # During exploitation, use Boulware-like concession
            adjusted_time = (time - self._exploration_ratio) / (
                self._compromise_time - self._exploration_ratio
            )
            base_target = self._max_utility - (
                self._max_utility - self._min_utility
            ) * math.pow(adjusted_time, 2)
        else:
            # Compromise phase - concede faster
            adjusted_time = (time - self._compromise_time) / (
                1.0 - self._compromise_time
            )
            start_util = self._max_utility * self._compromise_start_utility
            base_target = start_util - (start_util - self._min_utility) * adjusted_time

        # Apply risk adjustment
        self._risk_adjusted_target = self._compute_risk_adjusted_utility(
            base_target, time
        )

        return self._risk_adjusted_target

    def _select_bid(self, time: float) -> Outcome | None:
        """
        Select a bid based on current phase and target utility.
        """
        if self._outcome_space is None:
            return None

        # Compute target
        self._target_utility = self._compute_target_utility(time)

        # In exploration phase, add some randomness
        if self._current_phase == "exploration":
            tolerance = self._exploration_tolerance
        else:
            tolerance = self._exploitation_tolerance

        # Check if opponent's best bid meets our target
        if (
            self._opponent_best_bid is not None
            and self._opponent_best_utility >= self._target_utility
        ):
            return self._opponent_best_bid

        # Get candidates near target utility
        candidates = self._outcome_space.get_bids_in_range(
            self._target_utility - tolerance,
            min(1.0, self._target_utility + tolerance),
        )

        if not candidates:
            # Fall back to closest bid above target
            candidates = self._outcome_space.get_bids_above(self._target_utility)
            if not candidates:
                if self._outcome_space.outcomes:
                    return self._outcome_space.outcomes[0].bid
                return None

        # Selection strategy based on phase
        if self._current_phase == "exploration":
            # Random selection to explore
            selected = random.choice(candidates)
        else:
            # Select from top candidates
            top_k = min(self._top_k_candidates, len(candidates))
            selected = random.choice(candidates[:top_k])

        return selected.bid

    def _accept_condition(self, offer: Outcome, time: float) -> bool:
        """
        Decide whether to accept an offer using risk-aware strategy.
        """
        if self.ufun is None or offer is None:
            return False

        offer_utility = float(self.ufun(offer))

        # Accept if offer meets our target
        if offer_utility >= self._target_utility:
            return True

        # Risk-based acceptance: accept if offer > risk-adjusted expected value
        expected_future = self._predicted_opponent_next * (1.0 - time)
        if offer_utility >= expected_future and offer_utility >= self._min_utility:
            return True

        # Phase-specific acceptance
        if self._current_phase == "compromise":
            # In compromise phase, be more flexible
            if offer_utility >= self._opponent_best_utility - self._acceptance_margin:
                return True
            if offer_utility >= self._min_utility:
                return True

        if time > self._final_deadline_time:
            # Very near deadline
            if offer_utility >= self._min_utility:
                return True

        # Accept if offer is at least as good as what we would offer
        my_bid = self._select_bid(time)
        if my_bid is not None:
            my_utility = float(self.ufun(my_bid))
            if offer_utility >= my_utility:
                return True

        return False

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """
        Generate a proposal.

        Args:
            state: Current negotiation state.
            dest: Destination negotiator ID (ignored).

        Returns:
            Outcome to propose, or None.
        """
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        bid = self._select_bid(time)

        if bid is not None and self.ufun is not None:
            self._last_bid = bid
            self._own_bids.append((bid, float(self.ufun(bid))))

        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """
        Respond to an offer.

        Args:
            state: Current negotiation state.
            source: Source negotiator ID (ignored).

        Returns:
            ResponseType indicating acceptance or rejection.
        """
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        # Update opponent model
        self._update_opponent_model(offer)

        time = state.relative_time

        # Update target utility
        self._target_utility = self._compute_target_utility(time)

        if self._accept_condition(offer, time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
