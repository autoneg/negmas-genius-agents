"""CUHKAgent from ANAC 2012."""

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

__all__ = ["CUHKAgent"]


class CUHKAgent(SAONegotiator):
    """
    CUHKAgent negotiation agent - Winner of ANAC 2012.

    CUHKAgent (Chinese University of Hong Kong Agent) won the ANAC 2012
    competition using an adaptive time-based concession strategy with
    opponent behavior tracking and discount factor adaptation.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        Original Genius class: ``agents.anac.y2012.CUHKAgent.CUHKAgent``

        ANAC 2012: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        Uses a two-phase time-dependent concession with adaptive threshold:

        - Phase 1 (t <= concede_factor): Gradual concession using formula:
          threshold = max_util - (max_util - min_threshold) * (t/concede_factor)^alpha
          where alpha controls concession speed (higher = tougher).
        - Phase 2 (t > concede_factor): Discount-adjusted threshold:
          threshold = (max_util * discount) / discount^t

        The concede_factor is dynamically determined based on:
        - Discount factor > 0.75: beta=1.8, later concession start
        - Discount factor 0.5-0.75: beta=1.5
        - Discount factor < 0.5: beta=1.2, earlier concession

        If opponent's best bid meets threshold, offers that bid back.
        Otherwise selects randomly from candidates above threshold.

    **Acceptance Strategy:**
        Multi-condition acceptance with strategic end-game handling:

        - Accept if offer utility >= current threshold.
        - Near deadline (t > 0.99): Accept if offer > reservation value.
        - Accept if offer >= utility of bid we would propose.
        - Very near deadline (t > 0.9985): Accept if offer >= opponent's
          best utility - 0.01.

    **Opponent Modeling:**
        Tracks opponent behavior to adjust concession rate:

        - Maintains bid history with utility values.
        - Tracks best bid received from opponent.
        - Computes opponent concession degree from utility variance (higher
          variance suggests more concession).
        - Adjusts concede_factor based on opponent toughness: tougher
          opponents trigger faster concession.

    Args:
        alpha: Concession rate exponent (default 2.0). Higher values result
            in slower initial concession (tougher negotiation).
        min_concede_factor: Minimum time fraction before conceding starts
            (default 0.08).
        high_discount_threshold: Discount factor threshold for high beta (default 0.75)
        medium_discount_threshold: Discount factor threshold for medium beta (default 0.5)
        high_discount_beta: Beta value when discount > high_threshold (default 1.8)
        medium_discount_beta: Beta value when discount > medium_threshold (default 1.5)
        low_discount_beta: Beta value when discount <= medium_threshold (default 1.2)
        opponent_gamma: Exponent for opponent toughness adjustment (default 10)
        opponent_weight: Weight for opponent toughness adjustment (default 0.1)
        fallback_utility_factor: Fallback utility factor when discount is 0 (default 0.7)
        deadline_accept_time: Time threshold for deadline acceptance (default 0.99)
        final_deadline_time: Time threshold for final strategic acceptance (default 0.9985)
        final_accept_margin: Margin below opponent best for final acceptance (default 0.01)
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
        alpha: float = 2.0,
        min_concede_factor: float = 0.08,
        high_discount_threshold: float = 0.75,
        medium_discount_threshold: float = 0.5,
        high_discount_beta: float = 1.8,
        medium_discount_beta: float = 1.5,
        low_discount_beta: float = 1.2,
        opponent_gamma: float = 10.0,
        opponent_weight: float = 0.1,
        fallback_utility_factor: float = 0.7,
        deadline_accept_time: float = 0.99,
        final_deadline_time: float = 0.9985,
        final_accept_margin: float = 0.01,
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
        self._alpha = alpha
        self._min_concede_factor = min_concede_factor
        self._high_discount_threshold = high_discount_threshold
        self._medium_discount_threshold = medium_discount_threshold
        self._high_discount_beta = high_discount_beta
        self._medium_discount_beta = medium_discount_beta
        self._low_discount_beta = low_discount_beta
        self._opponent_gamma = opponent_gamma
        self._opponent_weight = opponent_weight
        self._fallback_utility_factor = fallback_utility_factor
        self._deadline_accept_time = deadline_accept_time
        self._final_deadline_time = final_deadline_time
        self._final_accept_margin = final_accept_margin
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Bid histories
        self._opponent_bids: list[tuple[Outcome, float]] = []  # (bid, utility) pairs
        self._own_bids: list[tuple[Outcome, float]] = []

        # State variables
        self._utility_threshold: float = 1.0
        self._max_utility: float = 1.0
        self._reservation_value: float = 0.0
        self._discounting_factor: float = 1.0
        self._concede_to_discounting_factor: float = 1.0
        self._concede_to_discounting_factor_original: float = 1.0

        # Opponent modeling
        self._opponent_best_bid: Outcome | None = None
        self._opponent_best_utility: float = 0.0
        self._opponent_sum_utility: float = 0.0
        self._opponent_count: int = 0

        # Strategy flags
        self._concede_to_opponent: bool = False
        self._tough_agent: bool = False

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

        self._utility_threshold = self._max_utility
        self._choose_concede_to_discounting_degree()
        self._initialized = True

    def _choose_concede_to_discounting_degree(self) -> None:
        """Determine concede-to-time degree based on discounting factor."""
        # Beta controls how much the agent concedes
        if self._discounting_factor > self._high_discount_threshold:
            beta = self._high_discount_beta
        elif self._discounting_factor > self._medium_discount_threshold:
            beta = self._medium_discount_beta
        else:
            beta = self._low_discount_beta

        alpha = math.pow(self._discounting_factor, beta)
        self._concede_to_discounting_factor = (
            self._min_concede_factor + (1 - self._min_concede_factor) * alpha
        )
        self._concede_to_discounting_factor_original = (
            self._concede_to_discounting_factor
        )

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_bids = []
        self._own_bids = []
        self._utility_threshold = self._max_utility
        self._opponent_best_bid = None
        self._opponent_best_utility = 0.0
        self._opponent_sum_utility = 0.0
        self._opponent_count = 0
        self._concede_to_opponent = False
        self._tough_agent = False

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent model with a new bid."""
        if self.ufun is None or bid is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append((bid, utility))
        self._opponent_sum_utility += utility
        self._opponent_count += 1

        # Track best bid from opponent
        if utility > self._opponent_best_utility:
            self._opponent_best_utility = utility
            self._opponent_best_bid = bid

    def _get_opponent_concession_degree(self) -> float:
        """Estimate opponent's concession degree."""
        if self._opponent_count < 2:
            return 0.0

        # Calculate variance in opponent utilities
        mean = self._opponent_sum_utility / self._opponent_count
        variance = 0.0
        for _, util in self._opponent_bids:
            variance += (util - mean) ** 2
        variance /= self._opponent_count

        # Higher variance suggests more concession
        return min(1.0, math.sqrt(variance) * 2)

    def _update_concede_degree(self) -> None:
        """Update concession degree based on opponent behavior."""
        gamma = self._opponent_gamma
        weight = self._opponent_weight
        opponent_toughness = self._get_opponent_concession_degree()

        self._concede_to_discounting_factor = (
            self._concede_to_discounting_factor_original
            + weight
            * (1 - self._concede_to_discounting_factor_original)
            * math.pow(opponent_toughness, gamma)
        )
        self._concede_to_discounting_factor = min(
            1.0, self._concede_to_discounting_factor
        )

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold for the given time."""
        max_util = self._max_utility

        if time <= self._concede_to_discounting_factor:
            # Gradual concession phase
            min_threshold = (
                (max_util * self._discounting_factor)
                / math.pow(
                    self._discounting_factor, self._concede_to_discounting_factor
                )
                if self._discounting_factor > 0
                else max_util * self._fallback_utility_factor
            )

            threshold = max_util - (max_util - min_threshold) * math.pow(
                time / self._concede_to_discounting_factor
                if self._concede_to_discounting_factor > 0
                else 1.0,
                self._alpha,
            )
        else:
            # Post-concession phase (approaching deadline)
            threshold = (
                (max_util * self._discounting_factor)
                / math.pow(self._discounting_factor, time)
                if self._discounting_factor > 0
                else max_util * self._fallback_utility_factor
            )

        return max(threshold, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid to offer."""
        if self._outcome_space is None:
            return None

        # Update threshold
        self._utility_threshold = self._compute_threshold(time)

        # First check if opponent's best bid meets our threshold
        if (
            self._opponent_best_bid is not None
            and self._opponent_best_utility >= self._utility_threshold
        ):
            return self._opponent_best_bid

        # Get bids in acceptable range
        candidates = self._outcome_space.get_bids_above(self._utility_threshold)

        if not candidates:
            # Fallback to best bid
            if self._outcome_space.outcomes:
                return self._outcome_space.outcomes[0].bid
            return None

        # Select randomly from candidates to give opponent more information
        selected = random.choice(candidates)
        return selected.bid

    def _accept_offer(self, offer: Outcome, time: float) -> bool:
        """Decide whether to accept an offer."""
        if self.ufun is None or offer is None:
            return False

        offer_utility = float(self.ufun(offer))

        # Accept if offer meets threshold
        if offer_utility >= self._utility_threshold:
            return True

        # Near deadline, accept if better than reservation value
        if (
            time > self._deadline_accept_time
            and offer_utility > self._reservation_value
        ):
            return True

        # Accept if this is as good as the best we can offer
        our_bid = self._select_bid(time)
        if our_bid is not None:
            our_utility = float(self.ufun(our_bid))
            if offer_utility >= our_utility:
                return True

        # Check for strategic acceptance near deadline
        if time > self._final_deadline_time and self._opponent_best_bid is not None:
            if offer_utility >= self._opponent_best_utility - self._final_accept_margin:
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

        # Update opponent model
        self._update_opponent_model(offer)
        self._update_concede_degree()

        time = state.relative_time

        # Update threshold for decision
        self._utility_threshold = self._compute_threshold(time)

        if self._accept_offer(offer, time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
