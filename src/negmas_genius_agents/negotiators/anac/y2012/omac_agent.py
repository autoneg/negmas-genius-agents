"""
OMACAgent - 3rd place agent from ANAC 2012.

This module implements the OMACAgent (Opponent Model-based Adaptive Concession)
negotiation agent which achieved third place in the Automated Negotiating Agents
Competition (ANAC) 2012. The agent uses opponent modeling to estimate reservation
values and adapts its concession strategy accordingly.

References:
    ANAC 2012 Competition Results and Agent Descriptions.
    Chen, S., & Weiss, G. (2012). OMAC: A Discrete Wavelet Transformation Based
    Negotiation Agent. ANAC 2012 Proceedings.
    Baarslag, T., Fujita, K., Gerding, E. H., Hindriks, K., Ito, T., Jennings, N. R.,
    Jonker, C., Kraus, S., Lin, R., Robu, V., & Williams, C. R. (2013).
    Evaluating Practical Negotiating Agents: Results and Analysis of the 2011
    International Competition. Artificial Intelligence, 198, 73-103.
"""

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

__all__ = ["OMACAgent"]


class OMACAgent(SAONegotiator):
    """
    OMACAgent negotiation agent - 3rd place at ANAC 2012.

    OMAC (Opponent Model-based Adaptive Concession) uses opponent modeling to
    estimate reservation values and dynamically adjusts its Boulware-like
    concession strategy based on opponent behavior.

    **Offering Strategy:**
        Uses adaptive Boulware concession with dynamic beta parameter:

        - Base formula: target = max - (max - min_target) * t^beta
        - min_target is the maximum of: minimum utility, reservation value,
          and estimated opponent reservation.
        - Beta (concession exponent) adapts based on opponent behavior:
          - Opponent conceding quickly (rate > 0.02): beta increases by 0.5
            (tougher, slower concession)
          - Opponent moderately flexible (0.005-0.02): beta stays at initial
          - Opponent tough (rate < 0.005): beta decreases by 0.3 (more flexible)
        - Discount factor adjustment: If discount < threshold, beta reduced
          by 0.5 to close deal faster.

        If opponent's best bid meets target, offers that bid back.
        Otherwise selects randomly from candidates near target utility.

    **Acceptance Strategy:**
        Target-based acceptance with strategic end-game handling:

        - Accept if offer utility >= target utility.
        - Near deadline (t > 0.98): Accept if offer >= opponent's best - 0.02.
        - Very near deadline (t > 0.995): Accept if offer >= minimum utility.
        - Accept if offer >= utility of bid we would propose.

    **Opponent Modeling:**
        Estimates opponent reservation value and concession behavior:

        - Tracks opponent bid history with utility values.
        - Uses linear regression on recent bids to estimate concession rate
          (positive slope = opponent improving offers = conceding).
        - Estimates opponent reservation value based on concession rate:
          - Fast concession: opponent may accept lower (best_utility - 0.1)
          - Slow concession: opponent wants high utility (best_utility + 0.05)
        - Updates model at configurable intervals (default every 5% of time).

    Args:
        beta: Initial concession exponent (default 1.5). Higher values result
            in slower concession (Boulware-like behavior).
        discount_threshold: Discount factor threshold below which agent
            concedes faster (default 0.9).
        min_utility: Minimum acceptable utility (default 0.65).
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
        beta: float = 1.5,
        discount_threshold: float = 0.9,
        min_utility: float = 0.65,
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
        self._initial_beta = beta
        self._beta = beta
        self._discount_threshold = discount_threshold
        self._min_utility = min_utility

        # Will be initialized when negotiation starts
        self._outcome_space: SortedOutcomeSpace | None = None
        self._max_utility: float = 1.0
        self._reservation_value: float = 0.0
        self._discount_factor: float = 1.0
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[tuple[Outcome, float]] = []  # (bid, our_utility)
        self._opponent_utilities: list[float] = []  # Our utility for their bids
        self._opponent_best_bid: Outcome | None = None
        self._opponent_best_utility: float = 0.0
        self._estimated_opponent_reservation: float = 0.3
        self._opponent_concession_rate: float = 0.0

        # Own bidding
        self._own_bids: list[tuple[Outcome, float]] = []
        self._last_bid: Outcome | None = None
        self._target_utility: float = 1.0

        # Time tracking
        self._time_of_last_update: float = 0.0
        self._update_interval: float = 0.05

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

        # Get reservation value
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
        self._beta = self._initial_beta
        self._opponent_bids = []
        self._opponent_utilities = []
        self._opponent_best_bid = None
        self._opponent_best_utility = 0.0
        self._estimated_opponent_reservation = 0.3
        self._opponent_concession_rate = 0.0
        self._own_bids = []
        self._last_bid = None
        self._target_utility = self._max_utility
        self._time_of_last_update = 0.0

    def _update_opponent_model(self, bid: Outcome) -> None:
        """
        Update opponent model with new bid.

        Tracks opponent's bidding history and estimates:
        - Opponent's best bid for us
        - Opponent's concession rate
        - Opponent's estimated reservation value
        """
        if self.ufun is None or bid is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append((bid, utility))
        self._opponent_utilities.append(utility)

        # Track best bid from opponent
        if utility > self._opponent_best_utility:
            self._opponent_best_utility = utility
            self._opponent_best_bid = bid

    def _estimate_opponent_reservation(self) -> None:
        """
        Estimate opponent's reservation value from their bidding pattern.

        Uses the trend in opponent's offers to estimate what minimum
        utility they might accept from us.
        """
        if len(self._opponent_utilities) < 3:
            return

        # Calculate trend in opponent's concessions
        recent_utilities = self._opponent_utilities[-10:]
        if len(recent_utilities) < 2:
            return

        # Simple linear regression to estimate trend
        n = len(recent_utilities)
        sum_x = sum(range(n))
        sum_y = sum(recent_utilities)
        sum_xy = sum(i * u for i, u in enumerate(recent_utilities))
        sum_x2 = sum(i * i for i in range(n))

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Positive slope means opponent is conceding (their offers are getting better for us)
        self._opponent_concession_rate = max(0.0, slope)

        # Estimate opponent's reservation based on their best offer and concession rate
        # If they're conceding quickly, they may accept lower; if slowly, they're tough
        if self._opponent_concession_rate > 0.01:
            # Opponent is conceding, estimate they might accept something reasonable
            self._estimated_opponent_reservation = max(
                0.2, self._opponent_best_utility - 0.1
            )
        else:
            # Opponent is tough, they probably want high utility
            self._estimated_opponent_reservation = max(
                0.4, self._opponent_best_utility + 0.05
            )

    def _update_beta(self, time: float) -> None:
        """
        Update concession exponent based on opponent behavior and time.

        Adjusts beta to be:
        - Higher (tougher) if opponent isn't conceding
        - Lower (more flexible) if opponent is conceding
        - Adjusted for discount factor considerations
        """
        if time < self._time_of_last_update + self._update_interval:
            return

        self._time_of_last_update = time
        self._estimate_opponent_reservation()

        # Base beta on opponent's concession rate
        if self._opponent_concession_rate > 0.02:
            # Opponent is conceding, we can be slightly tougher
            self._beta = min(3.0, self._initial_beta + 0.5)
        elif self._opponent_concession_rate > 0.005:
            # Opponent is moderately flexible
            self._beta = self._initial_beta
        else:
            # Opponent is tough, we need to be more flexible
            self._beta = max(0.5, self._initial_beta - 0.3)

        # Adjust for discount factor
        if self._discount_factor < self._discount_threshold:
            # High discounting, need to close deal faster
            self._beta = max(0.3, self._beta - 0.5)

    def _compute_target_utility(self, time: float) -> float:
        """
        Compute target utility for the current time.

        Uses Boulware-like formula: target = max - (max - min) * time^beta
        """
        # Determine minimum target based on opponent modeling
        min_target = max(
            self._min_utility,
            self._reservation_value,
            self._estimated_opponent_reservation,
        )

        # Ensure min_target doesn't exceed max_utility
        min_target = min(min_target, self._max_utility - 0.05)

        # Boulware concession curve
        concession = math.pow(time, self._beta)
        target = self._max_utility - (self._max_utility - min_target) * concession

        # Don't go below minimum
        return max(target, self._min_utility, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """
        Select a bid to offer.

        Strategy:
        1. First check if opponent's best bid meets our target
        2. Otherwise select from candidates near target utility
        """
        if self._outcome_space is None:
            return None

        # Update opponent model and beta
        self._update_beta(time)

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
            self._target_utility - tolerance, self._target_utility + tolerance
        )

        if not candidates:
            # Try to get closest bid above target
            candidates = self._outcome_space.get_bids_above(self._target_utility)
            if not candidates:
                # Fallback to best bid
                if self._outcome_space.outcomes:
                    return self._outcome_space.outcomes[0].bid
                return None

        # Select randomly from candidates to add variety
        if len(candidates) > 1:
            selected = random.choice(candidates[: min(5, len(candidates))])
        else:
            selected = candidates[0]

        return selected.bid

    def _accept_condition(self, offer: Outcome, time: float) -> bool:
        """
        Decide whether to accept an offer.

        Accepts if:
        1. Offer utility >= target utility
        2. Near deadline and offer >= opponent's best - small margin
        3. Very near deadline and offer >= minimum utility
        """
        if self.ufun is None or offer is None:
            return False

        offer_utility = float(self.ufun(offer))

        # Accept if offer meets our target
        if offer_utility >= self._target_utility:
            return True

        # Near deadline strategies
        if time > 0.98:
            # Accept if offer is close to opponent's best
            if offer_utility >= self._opponent_best_utility - 0.02:
                return True

        if time > 0.995:
            # Very near deadline, accept if above minimum
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
