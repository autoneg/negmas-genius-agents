"""MetaAgent2012 from ANAC 2012."""

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

__all__ = ["MetaAgent2012"]


class MetaAgent2012(SAONegotiator):
    """
    MetaAgent2012 negotiation agent from ANAC 2012.

    MetaAgent2012 uses a meta-learning approach to combine multiple negotiation
    strategies (Boulware, Linear, Conceder) and dynamically adjusts weights
    based on domain characteristics and opponent behavior.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        Original Genius class: ``agents.anac.y2012.MetaAgent.MetaAgent``

        ANAC 2012: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        Blends three base strategies with adaptive weighting:

        - Boulware (slow concession): target = max - (max - min) * t^3
        - Linear (constant rate): target = max - (max - min) * t
        - Conceder (fast concession): target = max - (max - min) * t^0.3

        Final target is weighted blend: sum(weight_i * target_i)

        Initial weights based on domain analysis:
        - Small domains (<500 outcomes): Balanced (0.3/0.4/0.3)
        - Large domains with high utility variance (>0.2): Boulware-heavy
          (0.6/0.25/0.15)
        - Default: Slight Boulware bias (0.5/0.3/0.2)

        Weights adapt during negotiation based on opponent behavior.
        If opponent's best bid meets target, offers that bid back.

    **Acceptance Strategy:**
        Target-based acceptance with end-game handling:

        - Accept if offer utility >= blended target utility.
        - Near deadline (t > 0.98): Accept if offer >= opponent's best - 0.02.
        - Very near deadline (t > 0.995): Accept if offer >= reservation value.
        - Accept if offer >= utility of bid we would propose.

    **Opponent Modeling:**
        Monitors opponent concession to adjust strategy weights:

        - Tracks opponent bid history and best bid.
        - Estimates concession rate from recent bid utility trend.
        - If opponent conceding (rate > 0.01): Increase Boulware weight
          (be tougher).
        - If opponent hardening (rate < -0.01): Increase Conceder weight
          (be more flexible).
        - Time pressure (t > 0.8): Progressively increases Conceder weight.

        Strategy weights are updated at configurable intervals and normalized.

    Args:
        strategy_update_interval: How often to update strategy weights
            (default 0.1, i.e., every 10% of negotiation time).
        time_pressure_threshold: Time threshold for increasing Conceder weight
            (default 0.8).
        near_deadline_time: Time threshold for relaxed acceptance near deadline
            (default 0.98).
        final_deadline_time: Time threshold for accepting any offer above
            reservation value (default 0.995).
        small_domain_threshold: Outcome count threshold for small domain
            classification (default 500).
        high_variance_threshold: Utility variance threshold for high variance
            domain classification (default 0.2).
        small_domain_boulware: Boulware weight for small domains (default 0.3).
        small_domain_linear: Linear weight for small domains (default 0.4).
        small_domain_conceder: Conceder weight for small domains (default 0.3).
        high_variance_boulware: Boulware weight for high variance domains
            (default 0.6).
        high_variance_linear: Linear weight for high variance domains
            (default 0.25).
        high_variance_conceder: Conceder weight for high variance domains
            (default 0.15).
        default_boulware: Default Boulware weight (default 0.5).
        default_linear: Default Linear weight (default 0.3).
        default_conceder: Default Conceder weight (default 0.2).
        opponent_conceding_threshold: Threshold for detecting opponent concession
            (default 0.01).
        opponent_hardening_threshold: Threshold for detecting opponent hardening
            (default -0.01).
        weight_adjustment: Amount to adjust strategy weights (default 0.1).
        max_boulware_weight: Maximum Boulware weight (default 0.8).
        min_boulware_weight: Minimum Boulware weight (default 0.2).
        max_conceder_weight: Maximum Conceder weight (default 0.5).
        min_conceder_weight: Minimum Conceder weight (default 0.1).
        pressure_factor: Factor for time pressure adjustment (default 0.2).
        min_target_offset: Minimum target utility offset (default 0.1).
        boulware_exponent: Exponent for Boulware strategy (default 3).
        conceder_exponent: Exponent for Conceder strategy (default 0.3).
        bid_tolerance: Tolerance for bid selection range (default 0.02).
        top_k_candidates: Number of top candidates to select from (default 5).
        acceptance_margin: Margin below opponent best for near-deadline
            acceptance (default 0.02).
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
        strategy_update_interval: float = 0.1,
        time_pressure_threshold: float = 0.8,
        near_deadline_time: float = 0.98,
        final_deadline_time: float = 0.995,
        small_domain_threshold: int = 500,
        high_variance_threshold: float = 0.2,
        small_domain_boulware: float = 0.3,
        small_domain_linear: float = 0.4,
        small_domain_conceder: float = 0.3,
        high_variance_boulware: float = 0.6,
        high_variance_linear: float = 0.25,
        high_variance_conceder: float = 0.15,
        default_boulware: float = 0.5,
        default_linear: float = 0.3,
        default_conceder: float = 0.2,
        opponent_conceding_threshold: float = 0.01,
        opponent_hardening_threshold: float = -0.01,
        weight_adjustment: float = 0.1,
        max_boulware_weight: float = 0.8,
        min_boulware_weight: float = 0.2,
        max_conceder_weight: float = 0.5,
        min_conceder_weight: float = 0.1,
        pressure_factor: float = 0.2,
        min_target_offset: float = 0.1,
        boulware_exponent: float = 3.0,
        conceder_exponent: float = 0.3,
        bid_tolerance: float = 0.02,
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
        self._strategy_update_interval = strategy_update_interval
        self._time_pressure_threshold = time_pressure_threshold
        self._near_deadline_time = near_deadline_time
        self._final_deadline_time = final_deadline_time
        self._small_domain_threshold = small_domain_threshold
        self._high_variance_threshold = high_variance_threshold
        self._small_domain_boulware = small_domain_boulware
        self._small_domain_linear = small_domain_linear
        self._small_domain_conceder = small_domain_conceder
        self._high_variance_boulware = high_variance_boulware
        self._high_variance_linear = high_variance_linear
        self._high_variance_conceder = high_variance_conceder
        self._default_boulware = default_boulware
        self._default_linear = default_linear
        self._default_conceder = default_conceder
        self._opponent_conceding_threshold = opponent_conceding_threshold
        self._opponent_hardening_threshold = opponent_hardening_threshold
        self._weight_adjustment = weight_adjustment
        self._max_boulware_weight = max_boulware_weight
        self._min_boulware_weight = min_boulware_weight
        self._max_conceder_weight = max_conceder_weight
        self._min_conceder_weight = min_conceder_weight
        self._pressure_factor = pressure_factor
        self._min_target_offset = min_target_offset
        self._boulware_exponent = boulware_exponent
        self._conceder_exponent = conceder_exponent
        self._bid_tolerance = bid_tolerance
        self._top_k_candidates = top_k_candidates
        self._acceptance_margin = acceptance_margin

        # Will be initialized when negotiation starts
        self._outcome_space: SortedOutcomeSpace | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._reservation_value: float = 0.0
        self._initialized = False

        # Domain analysis
        self._domain_size: int = 0
        self._utility_mean: float = 0.5
        self._utility_stdev: float = 0.1

        # Strategy weights: Boulware, Linear, Conceder
        self._strategy_weights: dict[str, float] = {
            "boulware": 0.5,
            "linear": 0.3,
            "conceder": 0.2,
        }
        self._strategy_performance: dict[str, float] = {
            "boulware": 0.0,
            "linear": 0.0,
            "conceder": 0.0,
        }

        # Opponent modeling
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._opponent_best_bid: Outcome | None = None
        self._opponent_best_utility: float = 0.0
        self._opponent_concession_rate: float = 0.0

        # Own bidding state
        self._own_bids: list[tuple[Outcome, float]] = []
        self._last_bid: Outcome | None = None
        self._target_utility: float = 1.0
        self._last_strategy_update: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space and analyze domain."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if not self._outcome_space.outcomes:
            return

        self._max_utility = self._outcome_space.max_utility
        self._min_utility = self._outcome_space.min_utility

        # Get reservation value if available
        reservation = getattr(self.ufun, "reserved_value", 0.0)
        if reservation is not None and reservation != float("-inf"):
            self._reservation_value = max(0.0, reservation)

        # Analyze domain characteristics
        self._analyze_domain()

        # Initialize strategy weights based on domain
        self._initialize_strategy_weights()

        self._initialized = True

    def _analyze_domain(self) -> None:
        """Analyze domain characteristics for strategy selection."""
        if self._outcome_space is None:
            return

        outcomes = self._outcome_space.outcomes
        self._domain_size = len(outcomes)

        if not outcomes:
            return

        # Compute utility statistics
        utilities = [bd.utility for bd in outcomes]
        self._utility_mean = sum(utilities) / len(utilities)
        variance = sum((u - self._utility_mean) ** 2 for u in utilities) / len(
            utilities
        )
        self._utility_stdev = math.sqrt(variance) if variance > 0 else 0

    def _initialize_strategy_weights(self) -> None:
        """Initialize strategy weights based on domain characteristics."""
        # Small domains favor conceding strategies (fewer options)
        if self._domain_size < self._small_domain_threshold:
            self._strategy_weights = {
                "boulware": self._small_domain_boulware,
                "linear": self._small_domain_linear,
                "conceder": self._small_domain_conceder,
            }
        # Large domains with high variance favor Boulware
        elif self._utility_stdev > self._high_variance_threshold:
            self._strategy_weights = {
                "boulware": self._high_variance_boulware,
                "linear": self._high_variance_linear,
                "conceder": self._high_variance_conceder,
            }
        # Default: balanced with Boulware bias
        else:
            self._strategy_weights = {
                "boulware": self._default_boulware,
                "linear": self._default_linear,
                "conceder": self._default_conceder,
            }

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_bids = []
        self._opponent_best_bid = None
        self._opponent_best_utility = 0.0
        self._opponent_concession_rate = 0.0
        self._own_bids = []
        self._last_bid = None
        self._target_utility = self._max_utility
        self._last_strategy_update = 0.0
        self._strategy_performance = {
            "boulware": 0.0,
            "linear": 0.0,
            "conceder": 0.0,
        }

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent model with a new bid."""
        if self.ufun is None or bid is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append((bid, utility))

        # Track best bid
        if utility > self._opponent_best_utility:
            self._opponent_best_utility = utility
            self._opponent_best_bid = bid

        # Estimate opponent concession rate
        if len(self._opponent_bids) >= 5:
            recent = [u for _, u in self._opponent_bids[-5:]]
            # Positive rate means opponent is improving offers to us
            if len(recent) > 1:
                diff = recent[-1] - recent[0]
                self._opponent_concession_rate = diff / (len(recent) - 1)

    def _update_strategy_weights(self, time: float) -> None:
        """Update strategy weights based on performance and opponent behavior."""
        if time < self._last_strategy_update + self._strategy_update_interval:
            return

        self._last_strategy_update = time

        # Adapt based on opponent concession rate
        if self._opponent_concession_rate > self._opponent_conceding_threshold:
            # Opponent is conceding - favor tougher strategies
            self._strategy_weights["boulware"] = min(
                self._max_boulware_weight,
                self._strategy_weights["boulware"] + self._weight_adjustment,
            )
            self._strategy_weights["conceder"] = max(
                self._min_conceder_weight,
                self._strategy_weights["conceder"] - self._weight_adjustment,
            )
        elif self._opponent_concession_rate < self._opponent_hardening_threshold:
            # Opponent is getting tougher - consider conceding more
            self._strategy_weights["conceder"] = min(
                self._max_conceder_weight,
                self._strategy_weights["conceder"] + self._weight_adjustment,
            )
            self._strategy_weights["boulware"] = max(
                self._min_boulware_weight,
                self._strategy_weights["boulware"] - self._weight_adjustment,
            )

        # Time pressure: favor conceder strategies near deadline
        if time > self._time_pressure_threshold:
            pressure_factor = (time - self._time_pressure_threshold) / (
                1.0 - self._time_pressure_threshold
            )
            self._strategy_weights["conceder"] += (
                pressure_factor * self._pressure_factor
            )
            # Normalize
            total = sum(self._strategy_weights.values())
            for k in self._strategy_weights:
                self._strategy_weights[k] /= total

    def _boulware_target(self, time: float) -> float:
        """Compute target utility using Boulware strategy (slow concession)."""
        # u(t) = max - (max - min) * t^boulware_exponent
        min_target = max(
            self._reservation_value, self._min_utility + self._min_target_offset
        )
        concession = math.pow(time, self._boulware_exponent)
        return self._max_utility - (self._max_utility - min_target) * concession

    def _linear_target(self, time: float) -> float:
        """Compute target utility using Linear strategy."""
        # u(t) = max - (max - min) * t
        min_target = max(
            self._reservation_value, self._min_utility + self._min_target_offset
        )
        return self._max_utility - (self._max_utility - min_target) * time

    def _conceder_target(self, time: float) -> float:
        """Compute target utility using Conceder strategy (fast concession)."""
        # u(t) = max - (max - min) * t^conceder_exponent
        min_target = max(
            self._reservation_value, self._min_utility + self._min_target_offset
        )
        concession = math.pow(time, self._conceder_exponent)
        return self._max_utility - (self._max_utility - min_target) * concession

    def _compute_target_utility(self, time: float) -> float:
        """Compute blended target utility from all strategies."""
        # Get targets from each strategy
        boulware_target = self._boulware_target(time)
        linear_target = self._linear_target(time)
        conceder_target = self._conceder_target(time)

        # Blend based on weights
        target = (
            self._strategy_weights["boulware"] * boulware_target
            + self._strategy_weights["linear"] * linear_target
            + self._strategy_weights["conceder"] * conceder_target
        )

        return max(target, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid to offer."""
        if self._outcome_space is None:
            return None

        # Update strategy weights
        self._update_strategy_weights(time)

        # Compute target
        self._target_utility = self._compute_target_utility(time)

        # Check if opponent's best bid meets our target
        if (
            self._opponent_best_bid is not None
            and self._opponent_best_utility >= self._target_utility
        ):
            return self._opponent_best_bid

        # Get candidates near target utility
        candidates = self._outcome_space.get_bids_in_range(
            self._target_utility - self._bid_tolerance,
            min(1.0, self._target_utility + self._bid_tolerance),
        )

        if not candidates:
            # Fall back to closest bid above target
            candidates = self._outcome_space.get_bids_above(self._target_utility)
            if not candidates:
                if self._outcome_space.outcomes:
                    return self._outcome_space.outcomes[0].bid
                return None

        # Select randomly from candidates
        selected = random.choice(
            candidates[: min(self._top_k_candidates, len(candidates))]
        )
        return selected.bid

    def _accept_condition(self, offer: Outcome, time: float) -> bool:
        """Decide whether to accept an offer."""
        if self.ufun is None or offer is None:
            return False

        offer_utility = float(self.ufun(offer))

        # Accept if offer meets our target
        if offer_utility >= self._target_utility:
            return True

        # Near deadline strategies
        if time > self._near_deadline_time:
            if offer_utility >= self._opponent_best_utility - self._acceptance_margin:
                return True

        if time > self._final_deadline_time:
            if offer_utility >= self._reservation_value:
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
