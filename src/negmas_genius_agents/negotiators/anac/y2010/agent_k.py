"""AgentK from ANAC 2010."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["AgentK"]


class AgentK(SAONegotiator):
    """
    AgentK from ANAC 2010 - The winning agent (1st place).

    AgentK won the first ANAC competition with its innovative adaptive
    negotiation strategy based on statistical analysis of opponent behavior.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This implementation reproduces AgentK's core strategies:

    - Statistical tracking of opponent offers (mean, variance, deviation)
    - Dynamic target adjustment using estimated maximum attainable utility
    - Probabilistic acceptance with time-dependent thresholds
    - "Tremor" mechanism for controlled randomness in decision-making

    References:
        Original Genius class: ``agents.anac.y2010.AgentK.Agent_K``

        ANAC 2010 results and army descriptions in:
        Baarslag, T., et al. (2012). "The First Automated Negotiating Agents
        Competition (ANAC 2010)". New Trends in Agent-based Complex Automated
        Negotiations. Studies in Computational Intelligence, vol 383. Springer.

    **Offering Strategy:**
        - Random bid generation with adaptive target utility
        - First checks if opponent's previous offers exceed current target
        - Falls back to random search for bids meeting `bid_target`
        - Target decreases by 0.01 after every 500 failed search attempts

    **Acceptance Strategy:**
        - Probabilistic acceptance based on statistical analysis
        - Tracks mean and variance of opponent offers
        - Estimates maximum attainable utility: `estimateMax = mean + (1-mean)*deviation`
        - Calculates acceptance probability:
          `p = (time^alpha)/5 + (offer - estimateMax) + (offer - target)`
        - Uses "tremor" parameter for randomness in alpha calculation

    **Opponent Modeling:**
        - Running statistics: sum, sum of squares, round count
        - Mean and variance tracking of opponent utilities
        - Deviation estimate (scaled by sqrt(12) for uniform distribution)
        - `estimateMax` predicts best achievable utility from opponent

    Key formulas:
        - alpha = 1 + tremor + 10*mean - 2*tremor*mean
        - preTarget = 1 - time^alpha * (1 - estimateMax)
        - ratio = (deviation + 0.1) / (1 - preTarget)

    Args:
        tremor: Randomness factor (default 2.0)
        search_limit: Iterations before reducing bid target (default 500)
        bid_target_decrement: How much to reduce bid target each cycle (default 0.01)
        max_search_iterations: Safety limit for bid search (default 10000)
        min_accept_probability: Threshold below which acceptance is rejected (default 0.1)
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
        search_limit: int = 500,
        bid_target_decrement: float = 0.01,
        max_search_iterations: int = 10000,
        min_accept_probability: float = 0.1,
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
        self._search_limit = search_limit
        self._bid_target_decrement = bid_target_decrement
        self._max_search_iterations = max_search_iterations
        self._min_accept_probability = min_accept_probability
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

    def _initialize(self) -> None:
        """Initialize the agent."""
        if self._initialized:
            return

        if self.ufun is None:
            return

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

    def _accept_probability(self, offered_bid: Outcome, time: float) -> float:
        """
        Calculate the probability of accepting an offered bid.

        This implements AgentK's statistical acceptance strategy:
        1. Track mean and variance of opponent offers
        2. Estimate maximum attainable utility
        3. Compute dynamic target based on time and statistics
        4. Calculate acceptance probability

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

        # Calculate mean and variance
        mean = self._sum / self._rounds
        variance = (self._sum2 / self._rounds) - (mean * mean)

        # Calculate deviation (scaled by sqrt(12) as in original)
        # This scaling approximates the range for a uniform distribution
        deviation = math.sqrt(max(0, variance) * 12)
        if math.isnan(deviation):
            deviation = 0.0

        # Time transformation (cubic)
        t = time * time * time

        # Validate utility and time (matches Java validation)
        # Java throws exception for utility < 0 or > 1.05, and time outside [0,1]
        if offered_utility < 0 or offered_utility > 1.05:
            # In Java this throws; we clamp instead for robustness
            offered_utility = max(0.0, min(1.05, offered_utility))

        if t < 0 or t > 1:
            t = max(0.0, min(1.0, t))

        # Clamp utility to 1.0 for calculations (matches Java: if (offeredUtility > 1.) offeredUtility = 1)
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
        if p < self._min_accept_probability:
            p = 0.0

        return max(0.0, min(1.0, p))

    def _search_random_bid(self) -> Outcome | None:
        """
        Generate a random bid by randomly selecting values for each issue.

        This matches the Java searchBid() method which randomly samples
        the outcome space.

        Returns:
            A randomly generated bid, or None if no outcome space.
        """
        if self.ufun is None or self.ufun.outcome_space is None:
            return None

        return self.ufun.outcome_space.random_outcome()

    def _select_bid(self) -> Outcome | None:
        """
        Select a bid to offer.

        AgentK's bid selection strategy:
        1. First, check if any previously offered bid exceeds target
        2. If so, randomly select one of those bids
        3. Otherwise, randomly search for a bid meeting bid_target

        Returns:
            The selected bid, or None if no suitable bid found.
        """
        if self.ufun is None:
            return None

        # Find bids from opponent that exceed our target
        good_bids = [
            bid for bid, util in self._offered_bids.items() if util > self._target
        ]

        if good_bids:
            # Randomly select from good bids (matches Java: random200.nextDouble() * size)
            selected = random.choice(good_bids)
            return selected

        # Random search for a bid meeting bid_target (matches Java searchBid loop)
        loop = 0
        while True:
            if loop > self._search_limit:
                self._bid_target -= self._bid_target_decrement
                loop = 0

            next_bid = self._search_random_bid()
            if next_bid is not None:
                search_util = float(self.ufun(next_bid))
                if search_util >= self._bid_target:
                    return next_bid

            loop += 1

            # Safety limit to prevent infinite loops (Java relies on eventually finding a bid)
            if loop > self._max_search_iterations:
                break

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
        Respond to an offer using AgentK's probabilistic acceptance.

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
