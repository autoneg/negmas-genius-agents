"""AgentKF from ANAC 2013."""

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

__all__ = ["AgentKF"]


class AgentKF(SAONegotiator):
    """
    AgentKF from ANAC 2013.

    AgentKF is an extension of the AgentK family (AgentK, AgentK2) with improved
    features for ANAC 2013, including statistical opponent modeling and adaptive
    concession behavior.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        Original Genius class: ``agents.anac.y2013.AgentKF.AgentKF``

        ANAC 2013: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        Uses a dynamic target utility that adapts based on statistical analysis
        of opponent offers. The bid target is computed using the formula:
        target = ratio * pre_target + 1 - ratio, where pre_target incorporates
        time-dependent concession (t^alpha) and estimated maximum attainable
        utility. Selects bids near the target utility, with preference for
        bids previously offered by the opponent if they exceed the target.
        Includes a "tremor" randomness factor for controlled exploration.

    **Acceptance Strategy:**
        Probabilistic acceptance based on multiple factors: (1) utility of the
        offer relative to estimated maximum attainable, (2) whether the offer
        satisfies the dynamic target, and (3) time pressure (t^alpha). The
        acceptance probability p = (t^alpha)/5 + utility_evaluation + satisfy.
        More aggressive acceptance in late game (after threshold, default 0.9).
        Emergency acceptance near deadline for offers above 0.5 utility.

    **Opponent Modeling:**
        Statistical tracking of opponent offers including running sum, sum of
        squares, count, min, and max utilities. Computes mean and variance to
        estimate the maximum attainable utility from the opponent. Detects
        opponent concession patterns by monitoring if max received utility
        increases significantly. Uses deviation (sqrt(variance * 12)) to adjust
        the target and acceptance calculations.

    Args:
        tremor: Randomness factor for exploration (default 2.0)
        late_game_threshold: Time after which to be more aggressive (default 0.9)
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
        late_game_threshold: float = 0.9,
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
        self._late_game_threshold = late_game_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Statistics tracking for opponent modeling
        self._offered_bids: dict[tuple, float] = {}  # bid -> utility mapping
        self._target: float = 1.0  # Target utility for accepting
        self._bid_target: float = 1.0  # Target utility for bidding
        self._sum: float = 0.0  # Sum of opponent offer utilities
        self._sum2: float = 0.0  # Sum of squared utilities
        self._rounds: int = 0  # Number of opponent offers received

        # Additional tracking for improved opponent modeling
        self._max_received_utility: float = 0.0
        self._min_received_utility: float = 1.0
        self._opponent_concession_detected: bool = False
        self._last_offer: Outcome | None = None

    def _initialize(self) -> None:
        """Initialize the outcome space and utility bounds."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
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
        self._max_received_utility = 0.0
        self._min_received_utility = 1.0
        self._opponent_concession_detected = False
        self._last_offer = None

    def _accept_probability(self, offered_bid: Outcome, time: float) -> float:
        """
        Calculate the probability of accepting an offered bid.

        Implements AgentKF's statistical acceptance strategy with improvements:
        1. Track mean and variance of opponent offers
        2. Detect opponent concession patterns
        3. Estimate maximum attainable utility
        4. Compute dynamic target based on time and statistics
        5. Calculate acceptance probability

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

        # Track min/max received utilities
        if offered_utility > self._max_received_utility:
            # Detect if opponent is conceding
            if self._rounds > 5 and offered_utility > self._max_received_utility + 0.02:
                self._opponent_concession_detected = True
            self._max_received_utility = offered_utility
        if offered_utility < self._min_received_utility:
            self._min_received_utility = offered_utility

        # Update running statistics
        self._sum += offered_utility
        self._sum2 += offered_utility * offered_utility
        self._rounds += 1

        # Calculate mean and variance
        mean = self._sum / self._rounds
        variance = (self._sum2 / self._rounds) - (mean * mean)

        # Calculate deviation (scaled by sqrt(12) as in original AgentK)
        deviation = math.sqrt(max(0, variance) * 12)
        if math.isnan(deviation):
            deviation = 0.0

        # Time transformation (cubic for early game, more aggressive late game)
        if time < self._late_game_threshold:
            t = time * time * time
        else:
            # More aggressive in late game
            t = time * time

        # Clamp utility to valid range
        if offered_utility > 1.0:
            offered_utility = 1.0

        # Estimate maximum attainable utility from opponent
        # AgentKF uses a slightly more optimistic estimate
        estimate_max = mean + ((1 - mean) * deviation)

        # Also consider the actual max received
        estimate_max = max(estimate_max, self._max_received_utility * 0.98)

        # Calculate alpha (concession exponent) with tremor randomness
        # Higher alpha = more Boulware-like behavior
        alpha = 1 + self._tremor + (10 * mean) - (2 * self._tremor * mean)

        # If opponent is conceding, we can be tougher
        if self._opponent_concession_detected:
            alpha *= 1.2

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

        # Late game boost - more likely to accept good offers
        if time > self._late_game_threshold:
            late_factor = (time - self._late_game_threshold) / (
                1 - self._late_game_threshold
            )
            p += late_factor * 0.2

        if p < 0.1:
            p = 0.0

        return max(0.0, min(1.0, p))

    def _select_bid(self, time: float) -> Outcome | None:
        """
        Select a bid to offer.

        AgentKF's bid selection strategy:
        1. Check if any previously offered bid exceeds target
        2. If so, randomly select one of those bids
        3. Otherwise, search for a bid meeting bid_target
        4. In late game, may offer bids closer to opponent's best offer

        Args:
            time: Current normalized time.

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
            return random.choice(good_bids)

        # Adjust target in late game
        current_target = self._bid_target
        if time > self._late_game_threshold:
            # Gradually lower target towards max received
            late_progress = (time - self._late_game_threshold) / (
                1 - self._late_game_threshold
            )
            min_acceptable = max(self._max_received_utility * 0.95, 0.5)
            current_target = (
                current_target - late_progress * (current_target - min_acceptable) * 0.5
            )

        # Search for a bid meeting bid_target
        max_attempts = 50

        for attempt in range(max_attempts):
            bid_details = self._outcome_space.get_bid_near_utility(current_target)

            if bid_details is not None:
                bid_utility = float(self.ufun(bid_details.bid)) if self.ufun else 0.0
                if bid_utility >= current_target * 0.95:  # Allow slight tolerance
                    return bid_details.bid

            # Lower target slightly and retry
            if attempt > 0 and attempt % 10 == 0:
                current_target -= 0.02

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

        return self._select_bid(state.relative_time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """
        Respond to an offer using AgentKF's probabilistic acceptance.

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

        # Emergency acceptance near deadline
        if time > 0.995:
            offer_utility = float(self.ufun(offer))
            if offer_utility > 0.5:  # Accept anything reasonable
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
