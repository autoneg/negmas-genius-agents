"""
IAMhaggler from ANAC 2010 - 4th place agent.

IAMhaggler (I AM haggler) was developed by the Southampton team (Colin R. Williams,
Valentin Robu, Enrico H. Gerding, Nicholas R. Jennings) and uses a Bayesian learning
approach to model the opponent and make strategic concessions.

This implementation includes:
1. Bayesian opponent preference modeling (issue weight + value estimation)
2. Time-dependent concession with adaptive rate based on opponent behavior
3. Nash-product bid selection for win-win outcomes
4. Multiple acceptance criteria including deadline awareness

References:
    Williams, C.R., Robu, V., Gerding, E.H., Jennings, N.R. (2010).
    "IAMhaggler: A Negotiation Agent for Complex Environments"
    New Trends in Agent-Based Complex Automated Negotiations, pp. 151-158.

    @inproceedings{williams2010iamhaggler,
      title={IAMhaggler: A negotiation agent for complex environments},
      author={Williams, Colin R and Robu, Valentin and Gerding, Enrico H and Jennings, Nicholas R},
      booktitle={New Trends in Agent-based Complex Automated Negotiations},
      pages={151--158},
      year={2010},
      publisher={Springer}
    }
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["IAMhaggler"]


class BayesianOpponentModel:
    """
    Bayesian opponent model for estimating opponent's preferences.

    Tracks issue weights and value preferences using frequency analysis
    with recency weighting. This allows estimating which issues matter
    most to the opponent and which values they prefer for each issue.
    """

    def __init__(self, issues: list[str], decay: float = 0.95):
        """
        Initialize the opponent model.

        Args:
            issues: List of issue names.
            decay: Decay factor for recency weighting (0-1).
        """
        self._issues = issues
        self._decay = decay

        # Track value frequencies for each issue: {issue: {value: weighted_count}}
        self._value_counts: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Track total weighted counts per issue
        self._issue_totals: dict[str, float] = defaultdict(float)

        # Track how consistently opponent chooses same values (for weight estimation)
        self._issue_consistency: dict[str, float] = defaultdict(float)
        self._last_values: dict[str, str | None] = {issue: None for issue in issues}

        # Number of observations
        self._num_observations = 0

    def update(self, bid: Outcome) -> None:
        """
        Update the opponent model with an observed bid.

        Args:
            bid: The opponent's bid (outcome dictionary).
        """
        self._num_observations += 1

        # Apply decay to existing counts
        for issue in self._issues:
            for value in self._value_counts[issue]:
                self._value_counts[issue][value] *= self._decay
            self._issue_totals[issue] *= self._decay

        # Update with new observation
        if isinstance(bid, dict):
            for issue in self._issues:
                if issue in bid:
                    value = str(bid[issue])

                    # Update value count
                    self._value_counts[issue][value] += 1.0
                    self._issue_totals[issue] += 1.0

                    # Update consistency score
                    if self._last_values[issue] is not None:
                        if value == self._last_values[issue]:
                            self._issue_consistency[issue] += 1.0
                        else:
                            self._issue_consistency[issue] *= 0.9

                    self._last_values[issue] = value

    def get_issue_weights(self) -> dict[str, float]:
        """
        Estimate opponent's issue weights based on selection consistency.

        Higher consistency in value selection indicates higher importance.

        Returns:
            Dictionary mapping issue names to estimated weights (sum to 1).
        """
        if self._num_observations < 2:
            # Uniform weights if not enough data
            return {issue: 1.0 / len(self._issues) for issue in self._issues}

        # Calculate consistency scores
        scores = {}
        for issue in self._issues:
            if self._issue_totals[issue] > 0:
                # Max frequency / total gives consistency measure
                max_count = max(self._value_counts[issue].values(), default=0)
                scores[issue] = max_count / self._issue_totals[issue]
            else:
                scores[issue] = 0.0

        # Normalize to weights
        total_score = sum(scores.values())
        if total_score > 0:
            return {issue: score / total_score for issue, score in scores.items()}
        else:
            return {issue: 1.0 / len(self._issues) for issue in self._issues}

    def get_value_preferences(self, issue: str) -> dict[str, float]:
        """
        Get estimated value preferences for an issue.

        Args:
            issue: The issue name.

        Returns:
            Dictionary mapping values to preference scores (normalized).
        """
        if issue not in self._value_counts or self._issue_totals[issue] == 0:
            return {}

        total = self._issue_totals[issue]
        return {
            value: count / total for value, count in self._value_counts[issue].items()
        }

    def estimate_opponent_utility(self, bid: Outcome) -> float:
        """
        Estimate the opponent's utility for a given bid.

        Uses issue weights and value preferences to compute a score.

        Args:
            bid: The bid to evaluate.

        Returns:
            Estimated utility in [0, 1].
        """
        if self._num_observations < 2 or not isinstance(bid, dict):
            return 0.5  # Neutral estimate if insufficient data

        weights = self.get_issue_weights()
        utility = 0.0

        for issue, weight in weights.items():
            if issue in bid:
                value = str(bid[issue])
                prefs = self.get_value_preferences(issue)
                # Use preference score if available, else assume neutral
                utility += weight * prefs.get(value, 0.5)

        return utility


class IAMhaggler(SAONegotiator):
    """
    IAMhaggler from ANAC 2010 - 4th place agent.

    IAMhaggler (I AM haggler) uses a Bayesian opponent modeling approach:

    **Offering Strategy:**
    - Time-dependent concession with polynomial curve: target(t) = max - (max-min) * t^(1/e)
    - Adaptive concession rate based on opponent behavior (slower if opponent conceding)
    - Nash-product bid selection: maximizes own_utility * estimated_opponent_utility

    **Acceptance Strategy:**
    - Accept if offer >= target utility at current time
    - Accept if near deadline (t > 0.95) and offer >= 95% of best opponent offer
    - Accept if very near deadline (t > 0.99) and offer >= reservation value

    **Opponent Modeling:**
    - Bayesian issue weight estimation from selection consistency
    - Value preference learning with recency-weighted frequency tracking
    - Opponent utility estimation for Nash-product bid selection

    Args:
        e: Concession exponent (default 2.0, Boulware-style)
        min_utility: Minimum acceptable utility floor (default 0.5)
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
        e: float = 2.0,
        min_utility: float = 0.5,
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
        self._e = e
        self._min_utility_param = min_utility

        self._outcome_space: SortedOutcomeSpace | None = None
        self._opponent_model: BayesianOpponentModel | None = None
        self._initialized = False

        # Opponent tracking
        self._opponent_offers: list[tuple[Outcome, float]] = []
        self._best_opponent_offer: Outcome | None = None
        self._best_opponent_utility: float = 0.0

        # Utility bounds
        self._min_utility: float = min_utility
        self._max_utility: float = 1.0

        # Adaptive concession rate
        self._effective_e: float = e

    def _initialize(self) -> None:
        """Initialize the outcome space and opponent model."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)

        # Set min utility based on reservation value
        reservation = getattr(self.ufun, "reserved_value", None)
        if reservation is not None and reservation != float("-inf"):
            self._min_utility = max(self._min_utility_param, float(reservation))
        else:
            self._min_utility = self._min_utility_param

        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility

        # Initialize opponent model with issue names
        if hasattr(self.nmi, "issues") and self.nmi.issues:
            issue_names = [issue.name for issue in self.nmi.issues]
            self._opponent_model = BayesianOpponentModel(issue_names)

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_offers = []
        self._best_opponent_offer = None
        self._best_opponent_utility = 0.0
        self._effective_e = self._e

    def _update_opponent_model(self, offer: Outcome, utility: float) -> None:
        """
        Update the opponent model with a received offer.

        Args:
            offer: The offer received from opponent.
            utility: Our utility for the offer.
        """
        self._opponent_offers.append((offer, utility))

        # Track best opponent offer
        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_offer = offer

        # Update Bayesian model
        if self._opponent_model is not None:
            self._opponent_model.update(offer)

        # Adapt concession rate based on opponent behavior
        if len(self._opponent_offers) >= 3:
            recent = [u for _, u in self._opponent_offers[-5:]]
            if len(recent) >= 2:
                trend = recent[-1] - recent[0]
                if trend > 0.02:
                    # Opponent is conceding, be more patient
                    self._effective_e = min(self._e * 1.5, 10.0)
                elif trend < -0.02:
                    # Opponent is hardening, concede more
                    self._effective_e = max(self._e * 0.7, 0.5)
                else:
                    # Neutral, use default
                    self._effective_e = self._e

    def _get_target_utility(self, time: float) -> float:
        """
        Calculate target utility using polynomial concession.

        Formula: target(t) = max - (max - min) * t^(1/e)

        Args:
            time: Normalized time [0, 1].

        Returns:
            Target utility value.
        """
        # Polynomial concession function
        concession = math.pow(time, 1.0 / self._effective_e)
        target = self._max_utility - concession * (
            self._max_utility - self._min_utility
        )
        return max(self._min_utility, min(self._max_utility, target))

    def _select_bid(self, time: float) -> Outcome | None:
        """
        Select a bid using Nash-product maximization.

        Finds bids near target utility and selects one that maximizes
        own_utility * estimated_opponent_utility.

        Args:
            time: Normalized time [0, 1].

        Returns:
            The selected bid, or None if no suitable bid found.
        """
        if self._outcome_space is None:
            return None

        target = self._get_target_utility(time)

        # Get bids near target utility
        candidates = self._outcome_space.get_bids_in_range(target - 0.05, target + 0.05)

        if not candidates:
            # Fallback to bids above minimum
            candidates = self._outcome_space.get_bids_above(self._min_utility)

        if not candidates:
            if self._outcome_space.outcomes:
                return self._outcome_space.outcomes[0].bid
            return None

        # If we have opponent model, use Nash product selection
        if self._opponent_model is not None and len(self._opponent_offers) >= 3:
            best_bid = None
            best_score = -1.0

            for bd in candidates[:50]:  # Limit search for efficiency
                own_util = bd.utility
                opp_util = self._opponent_model.estimate_opponent_utility(bd.bid)
                # Nash product + small bonus for own utility
                score = own_util * opp_util + 0.01 * own_util

                if score > best_score:
                    best_score = score
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        # Fallback: closest to target
        bid_details = self._outcome_space.get_bid_near_utility(target)
        if bid_details is not None:
            return bid_details.bid

        return candidates[0].bid if candidates else None

    def _should_accept(self, offer: Outcome, time: float) -> bool:
        """
        Decide whether to accept an offer.

        Acceptance criteria:
        1. Offer utility >= target utility
        2. Near deadline (t > 0.95): accept if >= 95% of best opponent offer
        3. Very near deadline (t > 0.99): accept if >= reservation value

        Args:
            offer: The offer to evaluate.
            time: Normalized time [0, 1].

        Returns:
            True if should accept, False otherwise.
        """
        if self.ufun is None:
            return False

        offer_utility = float(self.ufun(offer))
        target = self._get_target_utility(time)

        # Accept if meets target
        if offer_utility >= target:
            return True

        # Accept best opponent offer near deadline
        if time > 0.95 and offer_utility >= self._best_opponent_utility * 0.95:
            return True

        # Accept anything above minimum very close to deadline
        if time > 0.99 and offer_utility >= self._min_utility:
            return True

        return False

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """
        Generate a proposal using Nash-product bid selection.

        Args:
            state: Current negotiation state.
            dest: Destination negotiator ID (ignored).

        Returns:
            Outcome to propose, or None.
        """
        if not self._initialized:
            self._initialize()

        return self._select_bid(state.relative_time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """
        Respond to an offer using multi-criteria acceptance.

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

        offer_utility = float(self.ufun(offer))

        # Update opponent model
        self._update_opponent_model(offer, offer_utility)

        # Decide whether to accept
        if self._should_accept(offer, state.relative_time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
