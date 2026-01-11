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

    .. warning::
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
        if self._domain_size < 500:
            self._strategy_weights = {
                "boulware": 0.3,
                "linear": 0.4,
                "conceder": 0.3,
            }
        # Large domains with high variance favor Boulware
        elif self._utility_stdev > 0.2:
            self._strategy_weights = {
                "boulware": 0.6,
                "linear": 0.25,
                "conceder": 0.15,
            }
        # Default: balanced with Boulware bias
        else:
            self._strategy_weights = {
                "boulware": 0.5,
                "linear": 0.3,
                "conceder": 0.2,
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
        if self._opponent_concession_rate > 0.01:
            # Opponent is conceding - favor tougher strategies
            self._strategy_weights["boulware"] = min(
                0.8, self._strategy_weights["boulware"] + 0.1
            )
            self._strategy_weights["conceder"] = max(
                0.1, self._strategy_weights["conceder"] - 0.1
            )
        elif self._opponent_concession_rate < -0.01:
            # Opponent is getting tougher - consider conceding more
            self._strategy_weights["conceder"] = min(
                0.5, self._strategy_weights["conceder"] + 0.1
            )
            self._strategy_weights["boulware"] = max(
                0.2, self._strategy_weights["boulware"] - 0.1
            )

        # Time pressure: favor conceder strategies near deadline
        if time > 0.8:
            pressure_factor = (time - 0.8) / 0.2
            self._strategy_weights["conceder"] += pressure_factor * 0.2
            # Normalize
            total = sum(self._strategy_weights.values())
            for k in self._strategy_weights:
                self._strategy_weights[k] /= total

    def _boulware_target(self, time: float) -> float:
        """Compute target utility using Boulware strategy (slow concession)."""
        # u(t) = max - (max - min) * t^3
        min_target = max(self._reservation_value, self._min_utility + 0.1)
        concession = math.pow(time, 3)
        return self._max_utility - (self._max_utility - min_target) * concession

    def _linear_target(self, time: float) -> float:
        """Compute target utility using Linear strategy."""
        # u(t) = max - (max - min) * t
        min_target = max(self._reservation_value, self._min_utility + 0.1)
        return self._max_utility - (self._max_utility - min_target) * time

    def _conceder_target(self, time: float) -> float:
        """Compute target utility using Conceder strategy (fast concession)."""
        # u(t) = max - (max - min) * t^0.3
        min_target = max(self._reservation_value, self._min_utility + 0.1)
        concession = math.pow(time, 0.3)
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
        tolerance = 0.02
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

        # Select randomly from candidates
        selected = random.choice(candidates[: min(5, len(candidates))])
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
        if time > 0.98:
            if offer_utility >= self._opponent_best_utility - 0.02:
                return True

        if time > 0.995:
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
