"""AgentK2 from ANAC 2011."""

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

__all__ = ["AgentK2"]


class AgentK2(SAONegotiator):
    """
    AgentK2 from ANAC 2011.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This implementation faithfully reproduces AgentK2's core strategies:

    - Probabilistic acceptance based on statistical analysis of opponent offers
    - Running statistics (mean, variance, deviation) of received utilities
    - Adaptive target utility that responds to opponent concession patterns
    - Enhanced opponent modeling using frequency-based preference learning

    References:
        Original Genius class: ``agents.anac.y2011.AgentK2.Agent_K2``

        ANAC 2011: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
    - Maintains dynamic bid_target based on statistical analysis
    - First checks if any opponent bid exceeds current target (offers it randomly)
    - Otherwise searches for bids near bid_target using outcome space
    - Enhanced selection prefers bids good for opponent (30% weight)
    - Target adapts based on estimate_max derived from opponent statistics

    **Acceptance Strategy:**
    - Probabilistic acceptance using p = t^alpha/5 + util_eval + satisfy
    - util_eval = offered_utility - estimate_max (opponent capability)
    - satisfy = offered_utility - target (satisfaction level)
    - Near deadline boost: additional acceptance probability bonus after t=0.95
    - Minimum probability threshold of 0.1 required for consideration

    **Opponent Modeling:**
    - Tracks running statistics: sum, sum of squares, count of offers
    - Calculates mean, variance, and deviation of received utilities
    - estimate_max = mean + (1 - mean) * deviation predicts opponent ceiling
    - Frequency-based preference learning tracks values per issue
    - Issue weights derived from selection consistency (K2 enhancement)

    Args:
        tremor: Randomness factor for target calculation (default 2.0)
        deadline_boost_time: Time threshold after which acceptance probability gets boosted (default 0.95)
        deadline_boost_multiplier: Multiplier for deadline proximity bonus (default 2.0)
        min_accept_probability: Minimum probability threshold for consideration (default 0.1)
        bid_tolerance: Tolerance range for bid selection around target (default 0.02)
        own_utility_weight: Weight for own utility in bid scoring (default 0.7)
        opponent_utility_weight: Weight for opponent utility in bid scoring (default 0.3)
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
        tremor: float = 2.0,
        deadline_boost_time: float = 0.95,
        deadline_boost_multiplier: float = 2.0,
        min_accept_probability: float = 0.1,
        bid_tolerance: float = 0.02,
        own_utility_weight: float = 0.7,
        opponent_utility_weight: float = 0.3,
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
        self._tremor = tremor
        self._deadline_boost_time = deadline_boost_time
        self._deadline_boost_multiplier = deadline_boost_multiplier
        self._min_accept_probability = min_accept_probability
        self._bid_tolerance = bid_tolerance
        self._own_utility_weight = own_utility_weight
        self._opponent_utility_weight = opponent_utility_weight
        self._deadline_boost_time = deadline_boost_time
        self._deadline_boost_multiplier = deadline_boost_multiplier
        self._min_accept_probability = min_accept_probability
        self._bid_tolerance = bid_tolerance
        self._own_utility_weight = own_utility_weight
        self._opponent_utility_weight = opponent_utility_weight
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Statistics tracking
        self._offered_bids: dict[tuple, float] = {}  # bid -> utility mapping
        self._target: float = 1.0  # Target utility for accepting
        self._bid_target: float = 1.0  # Target utility for bidding
        self._sum: float = 0.0  # Sum of opponent offer utilities
        self._sum2: float = 0.0  # Sum of squared utilities
        self._rounds: int = 0  # Number of opponent offers received

        # Store last received offer
        self._last_offer: Outcome | None = None

        # Enhanced opponent modeling (new in K2)
        self._opponent_issue_frequencies: dict[str, dict] = {}
        self._opponent_issue_weights: dict[str, float] = {}
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space and utility bounds."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)

        # Initialize opponent model with equal weights
        if self.nmi is not None:
            issues = self.nmi.issues
            n_issues = len(issues)
            weight = 1.0 / n_issues if n_issues > 0 else 1.0

            for issue in issues:
                self._opponent_issue_weights[issue.name] = weight
                self._opponent_issue_frequencies[issue.name] = {}

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset statistics
        self._offered_bids = {}
        self._target = 1.0
        self._bid_target = 1.0
        self._sum = 0.0
        self._sum2 = 0.0
        self._rounds = 0
        self._last_offer = None
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0

    def _update_opponent_model(self, bid: Outcome) -> None:
        """
        Update enhanced opponent model based on bid frequency analysis.

        Args:
            bid: The opponent's bid.
        """
        if bid is None or self.nmi is None:
            return

        # Track our utility for their bids
        if self.ufun is not None:
            utility = float(self.ufun(bid))
            if utility > self._best_opponent_utility:
                self._best_opponent_utility = utility
                self._best_opponent_bid = bid

        # Update value frequencies per issue
        issues = self.nmi.issues
        for i, issue in enumerate(issues):
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)
            if val is not None:
                val_key = str(val)
                if val_key not in self._opponent_issue_frequencies[issue.name]:
                    self._opponent_issue_frequencies[issue.name][val_key] = 0
                self._opponent_issue_frequencies[issue.name][val_key] += 1

        # Update issue weights based on value consistency
        if self._rounds >= 3:
            self._update_issue_weights()

    def _update_issue_weights(self) -> None:
        """Update estimated opponent issue weights based on selection consistency."""
        if self.nmi is None:
            return

        issues = self.nmi.issues
        consistency_scores: dict[str, float] = {}

        for issue in issues:
            counts = self._opponent_issue_frequencies.get(issue.name, {})
            if not counts:
                consistency_scores[issue.name] = 1.0
                continue

            # Calculate consistency as max_count / total_count
            total = sum(counts.values())
            max_count = max(counts.values()) if counts else 0
            consistency = max_count / total if total > 0 else 0.5
            consistency_scores[issue.name] = consistency

        # Normalize to weights
        total_consistency = sum(consistency_scores.values())
        if total_consistency > 0:
            for issue in issues:
                self._opponent_issue_weights[issue.name] = (
                    consistency_scores[issue.name] / total_consistency
                )

    def _get_opponent_utility(self, bid: Outcome) -> float:
        """
        Estimate opponent's utility for a bid based on learned preferences.

        Args:
            bid: The outcome to evaluate.

        Returns:
            Estimated opponent utility in [0, 1].
        """
        if self.nmi is None or not self._opponent_issue_frequencies:
            return 0.5

        issues = self.nmi.issues
        total_utility = 0.0

        for i, issue in enumerate(issues):
            weight = self._opponent_issue_weights.get(issue.name, 0.0)
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)

            if val is not None:
                val_key = str(val)
                counts = self._opponent_issue_frequencies.get(issue.name, {})

                if val_key in counts and counts:
                    max_count = max(counts.values())
                    value_preference = (
                        counts[val_key] / max_count if max_count > 0 else 0.5
                    )
                else:
                    value_preference = 0.3

                total_utility += weight * value_preference

        return min(1.0, max(0.0, total_utility))

    def _accept_probability(self, offered_bid: Outcome, time: float) -> float:
        """
        Calculate the probability of accepting an offered bid.

        This implements AgentK2's statistical acceptance strategy with
        enhanced opponent modeling.

        Args:
            offered_bid: The bid offered by opponent.
            time: Normalized time [0, 1].

        Returns:
            Acceptance probability in [0, 1].
        """
        if self.ufun is None:
            return 0.0

        offered_utility = float(self.ufun(offered_bid))

        # Store the bid and its utility
        bid_key = tuple(offered_bid) if offered_bid else ()
        self._offered_bids[bid_key] = offered_utility

        # Update running statistics
        self._sum += offered_utility
        self._sum2 += offered_utility * offered_utility
        self._rounds += 1

        # Update opponent model (new in K2)
        self._update_opponent_model(offered_bid)

        # Calculate mean and variance
        mean = self._sum / self._rounds
        variance = (self._sum2 / self._rounds) - (mean * mean)

        # Calculate deviation (scaled by sqrt(12))
        deviation = math.sqrt(max(0, variance) * 12)
        if math.isnan(deviation):
            deviation = 0.0

        # Time transformation (cubic)
        t = time * time * time

        # Clamp utility to valid range
        if offered_utility > 1.0:
            offered_utility = 1.0

        # Estimate maximum attainable utility from opponent
        estimate_max = mean + ((1 - mean) * deviation)

        # Calculate alpha and beta (with tremor randomness)
        alpha = 1 + self._tremor + (10 * mean) - (2 * self._tremor * mean)
        beta = alpha + (random.random() * self._tremor) - (self._tremor / 2)

        # Calculate pre-target values
        pre_target = 1 - (math.pow(time, alpha) * (1 - estimate_max))
        pre_target2 = 1 - (math.pow(time, beta) * (1 - estimate_max))

        # Calculate ratio for target adjustment
        ratio = (deviation + 0.1) / (1 - pre_target) if (1 - pre_target) != 0 else 2.0
        if math.isnan(ratio) or ratio > 2.0:
            ratio = 2.0

        ratio2 = (
            (deviation + 0.1) / (1 - pre_target2) if (1 - pre_target2) != 0 else 2.0
        )
        if math.isnan(ratio2) or ratio2 > 2.0:
            ratio2 = 2.0

        # Update targets
        self._target = ratio * pre_target + 1 - ratio
        self._bid_target = ratio2 * pre_target2 + 1 - ratio2

        # Apply target adjustment based on estimate_max
        m = t * (-300) + 400

        if self._target > estimate_max:
            r = self._target - estimate_max
            f = 1 / (r * r) if r != 0 else m
            if f > m or math.isnan(f):
                f = m
            app = r * f / m
            self._target = self._target - app
        else:
            self._target = estimate_max

        if self._bid_target > estimate_max:
            r = self._bid_target - estimate_max
            f = 1 / (r * r) if r != 0 else m
            if f > m or math.isnan(f):
                f = m
            app = r * f / m
            self._bid_target = self._bid_target - app
        else:
            self._bid_target = estimate_max

        # Calculate acceptance probability
        utility_evaluation = offered_utility - estimate_max
        satisfy = offered_utility - self._target

        p = (math.pow(time, alpha) / 5) + utility_evaluation + satisfy

        # K2 enhancement: boost acceptance near deadline
        if time > self._deadline_boost_time:
            time_bonus = (
                time - self._deadline_boost_time
            ) * self._deadline_boost_multiplier  # Up to 0.1 bonus at t=1
            p += time_bonus

        if p < self._min_accept_probability:
            p = 0.0

        return max(0.0, min(1.0, p))

    def _select_bid(self) -> Outcome | None:
        """
        Select a bid to offer.

        AgentK2's bid selection strategy with enhanced opponent modeling:
        1. First, check if any previously offered bid exceeds target
        2. If so, randomly select one of those bids
        3. Otherwise, search for a bid meeting bid_target, preferring
           bids that are good for the opponent

        Returns:
            The selected bid, or None if no suitable bid found.
        """
        if self._outcome_space is None:
            return None

        # Find bids from opponent that exceed our target
        good_bids = [
            bid for bid, util in self._offered_bids.items() if util > self._target
        ]

        if good_bids:
            # Randomly select from good bids
            selected = random.choice(good_bids)
            return selected

        # Search for a bid meeting bid_target
        current_target = self._bid_target
        tolerance = self._bid_tolerance

        # Get candidates near target
        candidates = self._outcome_space.get_bids_in_range(
            current_target - tolerance,
            min(1.0, current_target + tolerance),
        )

        if candidates:
            # K2 enhancement: prefer bids good for opponent
            if len(candidates) > 1 and self._opponent_issue_frequencies:
                best_bid = None
                best_score = -1.0

                for bd in candidates:
                    opp_util = self._get_opponent_utility(bd.bid)
                    # Score combines our utility and opponent utility
                    score = (
                        bd.utility * self._own_utility_weight
                        + opp_util * self._opponent_utility_weight
                    )
                    if score > best_score:
                        best_score = score
                        best_bid = bd.bid

                if best_bid is not None:
                    return best_bid

            return candidates[0].bid

        # Try to find a bid near the target
        bid_details = self._outcome_space.get_bid_near_utility(current_target)
        if bid_details is not None:
            return bid_details.bid

        # Fallback: return best available bid
        if self._outcome_space.outcomes:
            return self._outcome_space.outcomes[0].bid

        return None

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

        return self._select_bid()

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """
        Respond to an offer using AgentK2's probabilistic acceptance.

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

        self._last_offer = offer

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time

        # Calculate acceptance probability
        p = self._accept_probability(offer, time)

        # Probabilistic acceptance
        if p > random.random():
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
