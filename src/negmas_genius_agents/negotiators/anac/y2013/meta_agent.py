"""MetaAgent2013 from ANAC 2013."""

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

__all__ = ["MetaAgent2013"]


class MetaAgent2013(SAONegotiator):
    """
    MetaAgent2013 from ANAC 2013 - 2nd place agent.

    MetaAgent2013 is a meta-learning agent that selects negotiation strategies
    based on domain features. The original used CART decision trees and UCB-MAB
    for portfolio selection among strategies.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        Original Genius class: ``agents.anac.y2013.MetaAgent.MetaAgent``

        ANAC 2013: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        Selects between three strategies based on domain analysis:
        - "aggressive": Linear concession (1 - 0.3*t) for small domains (< 1000)
        - "adaptive": Quadratic concession towards opponent's first bid for
          high-variance domains (stdev > 0.2)
        - "cuhk": Conservative Boulware-like (t^3) for other domains
        Offers random bids above the computed threshold.

    **Acceptance Strategy:**
        Accepts offers above the strategy-dependent threshold. Emergency
        acceptance near deadline (> 0.99) for offers above reservation value.
        Does not implement AC_Next or other sophisticated acceptance conditions.

    **Opponent Modeling:**
        Records opponent's first bid utility and all subsequent bids. Uses
        opponent's first bid as baseline for adaptive strategy's target
        computation. Does not perform frequency-based opponent modeling.
        Strategy selection is based primarily on domain features (size and
        utility distribution) rather than opponent behavior.

    Args:
        prediction_factor: Weight for predictions vs actual performance (default 5)
        small_domain_threshold: Domain size threshold for aggressive strategy (default 1000)
        high_variance_threshold: Utility stdev threshold for adaptive strategy (default 0.2)
        aggressive_concession_rate: Concession rate for aggressive strategy (default 0.3)
        adaptive_baseline_multiplier: Baseline multiplier for adaptive strategy (default 0.7)
        adaptive_concession_exponent: Exponent for adaptive strategy concession (default 2)
        cuhk_target_floor: Target floor multiplier for cuhk strategy (default 0.7)
        cuhk_concession_exponent: Exponent for cuhk strategy concession (default 3)
        deadline_threshold: Time threshold for emergency acceptance (default 0.99)
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
        prediction_factor: int = 5,
        small_domain_threshold: int = 1000,
        high_variance_threshold: float = 0.2,
        aggressive_concession_rate: float = 0.3,
        adaptive_baseline_multiplier: float = 0.7,
        adaptive_concession_exponent: float = 2,
        cuhk_target_floor: float = 0.7,
        cuhk_concession_exponent: float = 3,
        deadline_threshold: float = 0.99,
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
        self._prediction_factor = prediction_factor
        self._small_domain_threshold = small_domain_threshold
        self._high_variance_threshold = high_variance_threshold
        self._aggressive_concession_rate = aggressive_concession_rate
        self._adaptive_baseline_multiplier = adaptive_baseline_multiplier
        self._adaptive_concession_exponent = adaptive_concession_exponent
        self._cuhk_target_floor = cuhk_target_floor
        self._cuhk_concession_exponent = cuhk_concession_exponent
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Domain features
        self._domain_size: int = 0
        self._avg_utility: float = 0.5
        self._utility_stdev: float = 0.1
        self._reservation_value: float = 0.0

        # Strategy selection
        self._selected_strategy: str = "cuhk"  # Default to CUHK-like strategy
        self._strategy_scores: dict[str, float] = {}

        # State
        self._opponent_first_bid_utility: float = 0.0
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._threshold: float = 1.0

    def _initialize(self) -> None:
        """Initialize and compute domain features."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if not self._outcome_space.outcomes:
            return

        # Compute domain features
        self._domain_size = len(self._outcome_space.outcomes)
        utilities = [bd.utility for bd in self._outcome_space.outcomes]
        self._avg_utility = sum(utilities) / len(utilities) if utilities else 0.5
        variance = (
            sum((u - self._avg_utility) ** 2 for u in utilities) / len(utilities)
            if utilities
            else 0.01
        )
        self._utility_stdev = math.sqrt(variance)

        # Select strategy based on domain
        self._select_strategy()
        self._initialized = True

    def _select_strategy(self) -> None:
        """Select best strategy based on domain features."""
        # Simplified strategy selection based on domain size and utility distribution
        if self._domain_size < self._small_domain_threshold:
            # Small domain: use more aggressive concession
            self._selected_strategy = "aggressive"
        elif self._utility_stdev > self._high_variance_threshold:
            # High variance: use adaptive strategy
            self._selected_strategy = "adaptive"
        else:
            # Default: CUHK-like conservative strategy
            self._selected_strategy = "cuhk"

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_first_bid_utility = 0.0
        self._opponent_bids = []
        self._threshold = 1.0

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold based on selected strategy."""
        max_util = self._outcome_space.max_utility if self._outcome_space else 1.0

        if self._selected_strategy == "aggressive":
            # More linear concession
            return max_util * (1 - self._aggressive_concession_rate * time)
        elif self._selected_strategy == "adaptive":
            # Adaptive based on opponent first bid
            baseline = (
                self._opponent_first_bid_utility
                if self._opponent_first_bid_utility > 0
                else max_util * self._adaptive_baseline_multiplier
            )
            return max_util - (max_util - baseline) * math.pow(
                time, self._adaptive_concession_exponent
            )
        else:  # cuhk
            # Conservative Boulware-like
            return max_util - (
                max_util - max_util * self._cuhk_target_floor
            ) * math.pow(time, self._cuhk_concession_exponent)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid based on threshold."""
        if self._outcome_space is None:
            return None

        self._threshold = self._compute_threshold(time)

        # Get bids above threshold
        candidates = self._outcome_space.get_bids_above(self._threshold)
        if not candidates:
            if self._outcome_space.outcomes:
                return self._outcome_space.outcomes[0].bid
            return None

        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        return self._select_bid(state.relative_time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        self._opponent_bids.append((offer, offer_utility))

        if self._opponent_first_bid_utility == 0.0:
            self._opponent_first_bid_utility = offer_utility

        time = state.relative_time
        self._threshold = self._compute_threshold(time)

        # Accept if above threshold
        if offer_utility >= self._threshold:
            return ResponseType.ACCEPT_OFFER

        # Accept near deadline if reasonable
        if time > self._deadline_threshold and offer_utility >= self._reservation_value:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
