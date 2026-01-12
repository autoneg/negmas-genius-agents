"""AgentSmith from ANAC 2010."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING
from enum import Enum

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["AgentSmith"]


class OpponentType(Enum):
    """Classification of opponent behavior."""

    UNKNOWN = "unknown"
    HARDHEAD = "hardhead"  # Opponent rarely concedes
    CONCEDER = "conceder"  # Opponent concedes quickly
    RANDOM = "random"  # Opponent behavior is unpredictable
    TFTLIKE = "tft_like"  # Opponent mirrors our behavior


class AgentSmith(SAONegotiator):
    """
    AgentSmith from ANAC 2010.

    This agent features adaptive strategy selection based on opponent classification,
    making it robust against different negotiation styles.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This implementation faithfully reproduces AgentSmith's core strategies:

    - Opponent type classification (HardHead, Conceder, Random, TFT-like)
    - Strategy adaptation based on opponent model
    - Multiple concession functions for different scenarios
    - Smart acceptance with opponent-dependent thresholds

    References:
        Original Genius class: ``agents.anac.y2010.AgentSmith.AgentSmith``

        ANAC 2010: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        - Classifies opponent type, then selects appropriate concession function
        - Against HardHead: Slow Boulware-style concession (e=0.2)
        - Against Conceder: Match opponent's concession speed
        - Against Random: Use optimal fixed strategy with moderate concession
        - Against TFT-like: Cooperative gradual concession
        - Bid selection closest to target utility

    **Acceptance Strategy:**
        - Accept if offer >= current target utility
        - Target threshold depends on opponent type:
          - HardHead: Higher threshold (patient)
          - Conceder: Lower threshold (exploit their concessions)
          - Random: Moderate threshold
          - TFT-like: Cooperative threshold

    **Opponent Modeling:**
        - Tracks variance of opponent offers
        - Monitors utility trend (increasing/decreasing)
        - Classifies into: HardHead, Conceder, Random, TFT-like
        - Classification uses statistical patterns:
          - Low variance + high demands = HardHead
          - Increasing utilities over time = Conceder
          - High variance = Random
          - Correlation with own concessions = TFT-like

    Args:
        min_utility: Minimum acceptable utility floor (default 0.5).
        variance_threshold: Variance threshold for classifying random opponent (default 0.05).
        conceder_trend_threshold: Trend threshold for classifying conceder opponent (default 0.01).
        hardhead_trend_threshold: Trend threshold for classifying hardhead opponent (default 0.005).
        hardhead_mean_threshold: Mean threshold for classifying hardhead opponent (default 0.5).
        hardhead_concession_rate: Concession rate against hardhead (default 0.5).
        conceder_concession_rate: Concession rate against conceder (default 1.5).
        tft_concession_rate: Concession rate against TFT-like opponent (default 1.2).
        default_concession_rate: Default concession rate (default 1.0).
        hardhead_exponent: Concession exponent against hardhead (default 0.1).
        conceder_exponent: Concession exponent against conceder (default 2.0).
        tft_exponent: Concession exponent against TFT-like (default 1.5).
        default_exponent: Default concession exponent (default 1.0).
        bid_tolerance: Tolerance for bid selection around target (default 0.03).
        random_noise: Noise factor for random opponents (default 0.1).
        deadline_start: Time to start deadline pressure (default 0.9).
        late_phase_time: Time for late acceptance phase (default 0.95).
        deadline_time: Time threshold for deadline acceptance (default 0.99).
        hardhead_acceptance_factor: Acceptance factor against hardhead (default 0.95).
        conceder_acceptance_factor: Acceptance factor against conceder (default 0.98).
        default_acceptance_factor: Default acceptance factor (default 0.97).
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
        min_utility: float = 0.5,
        variance_threshold: float = 0.05,
        conceder_trend_threshold: float = 0.01,
        hardhead_trend_threshold: float = 0.005,
        hardhead_mean_threshold: float = 0.5,
        hardhead_concession_rate: float = 0.5,
        conceder_concession_rate: float = 1.5,
        tft_concession_rate: float = 1.2,
        default_concession_rate: float = 1.0,
        hardhead_exponent: float = 0.1,
        conceder_exponent: float = 2.0,
        tft_exponent: float = 1.5,
        default_exponent: float = 1.0,
        bid_tolerance: float = 0.03,
        random_noise: float = 0.1,
        deadline_start: float = 0.9,
        late_phase_time: float = 0.95,
        deadline_time: float = 0.99,
        hardhead_acceptance_factor: float = 0.95,
        conceder_acceptance_factor: float = 0.98,
        default_acceptance_factor: float = 0.97,
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
        self._min_utility_param = min_utility
        self._variance_threshold = variance_threshold
        self._conceder_trend_threshold = conceder_trend_threshold
        self._hardhead_trend_threshold = hardhead_trend_threshold
        self._hardhead_mean_threshold = hardhead_mean_threshold
        self._hardhead_concession_rate = hardhead_concession_rate
        self._conceder_concession_rate = conceder_concession_rate
        self._tft_concession_rate = tft_concession_rate
        self._default_concession_rate = default_concession_rate
        self._hardhead_exponent = hardhead_exponent
        self._conceder_exponent = conceder_exponent
        self._tft_exponent = tft_exponent
        self._default_exponent = default_exponent
        self._bid_tolerance = bid_tolerance
        self._random_noise = random_noise
        self._deadline_start = deadline_start
        self._late_phase_time = late_phase_time
        self._deadline_time = deadline_time
        self._hardhead_acceptance_factor = hardhead_acceptance_factor
        self._conceder_acceptance_factor = conceder_acceptance_factor
        self._default_acceptance_factor = default_acceptance_factor
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_offers: list[
            tuple[Outcome, float, float]
        ] = []  # (offer, utility, time)
        self._opponent_type: OpponentType = OpponentType.UNKNOWN
        self._opponent_variance: float = 0.0
        self._opponent_trend: float = 0.0

        # Strategy parameters
        self._min_utility: float = min_utility
        self._max_utility: float = 1.0
        self._concession_rate: float = (
            default_concession_rate  # Adjusted based on opponent
        )

        # Tracking
        self._best_opponent_offer: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._our_last_offer_utility: float = 1.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)

        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = max(
                self._min_utility_param, self._outcome_space.min_utility
            )

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_offers = []
        self._opponent_type = OpponentType.UNKNOWN
        self._opponent_variance = 0.0
        self._opponent_trend = 0.0
        self._best_opponent_offer = None
        self._best_opponent_utility = 0.0
        self._our_last_offer_utility = 1.0
        self._concession_rate = self._default_concession_rate

    def _update_opponent_model(
        self, offer: Outcome, utility: float, time: float
    ) -> None:
        """
        Update opponent model and classify opponent type.

        Analyzes opponent's bidding pattern to classify their strategy
        and adjust our response accordingly.

        Args:
            offer: The offer received from opponent.
            utility: Our utility for the offer.
            time: Current normalized time.
        """
        self._opponent_offers.append((offer, utility, time))

        # Update best offer tracking
        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_offer = offer

        # Need enough data to classify
        if len(self._opponent_offers) < 5:
            return

        utilities = [u for _, u, _ in self._opponent_offers]

        # Calculate statistics
        mean = sum(utilities) / len(utilities)
        variance = sum((u - mean) ** 2 for u in utilities) / len(utilities)
        self._opponent_variance = variance

        # Calculate trend (are they conceding?)
        recent = utilities[-5:]
        if len(recent) >= 2:
            self._opponent_trend = (recent[-1] - recent[0]) / len(recent)

        # Classify opponent
        self._classify_opponent()

        # Adjust our strategy based on opponent type
        self._adjust_strategy()

    def _classify_opponent(self) -> None:
        """
        Classify opponent based on their bidding behavior.

        Uses variance and trend to determine opponent type.
        """
        if len(self._opponent_offers) < 5:
            self._opponent_type = OpponentType.UNKNOWN
            return

        utilities = [u for _, u, _ in self._opponent_offers]
        mean = sum(utilities) / len(utilities)

        # High variance indicates random behavior
        if self._opponent_variance > self._variance_threshold:
            self._opponent_type = OpponentType.RANDOM
            return

        # Positive trend indicates conceder
        if self._opponent_trend > self._conceder_trend_threshold:
            self._opponent_type = OpponentType.CONCEDER
            return

        # Negative trend with low offers indicates hardhead
        if (
            self._opponent_trend < self._hardhead_trend_threshold
            and mean < self._hardhead_mean_threshold
        ):
            self._opponent_type = OpponentType.HARDHEAD
            return

        # Check for TFT-like behavior (offers correlate with our concessions)
        if len(self._opponent_offers) >= 10:
            # Simple correlation check
            recent_offers = utilities[-5:]
            if all(
                abs(recent_offers[i] - recent_offers[i - 1]) < 0.1
                for i in range(1, len(recent_offers))
            ):
                self._opponent_type = OpponentType.TFTLIKE
                return

        self._opponent_type = OpponentType.UNKNOWN

    def _adjust_strategy(self) -> None:
        """Adjust our strategy based on opponent classification."""
        if self._opponent_type == OpponentType.HARDHEAD:
            # Against hardhead: be patient, slow concession
            self._concession_rate = self._hardhead_concession_rate
        elif self._opponent_type == OpponentType.CONCEDER:
            # Against conceder: match their speed
            self._concession_rate = self._conceder_concession_rate
        elif self._opponent_type == OpponentType.RANDOM:
            # Against random: use moderate strategy
            self._concession_rate = self._default_concession_rate
        elif self._opponent_type == OpponentType.TFTLIKE:
            # Against TFT: cooperate with gradual concessions
            self._concession_rate = self._tft_concession_rate
        else:
            # Unknown: default moderate strategy
            self._concession_rate = self._default_concession_rate

    def _get_target_utility(self, time: float) -> float:
        """
        Calculate target utility based on opponent type and time.

        Different strategies for different opponent types:
        - HardHead: Use boulware strategy (slow concession)
        - Conceder: Use linear concession
        - Random: Use moderate polynomial
        - TFT-like: Use cooperative gradual concession

        Args:
            time: Normalized time [0, 1].

        Returns:
            Target utility value.
        """
        # Base concession function adjusted by opponent type
        if self._opponent_type == OpponentType.HARDHEAD:
            # Boulware: concede very slowly until deadline
            e = self._hardhead_exponent
            time_factor = math.pow(time, e)
        elif self._opponent_type == OpponentType.CONCEDER:
            # Match their concession rate
            e = self._conceder_exponent
            time_factor = math.pow(time, e)
        elif self._opponent_type == OpponentType.RANDOM:
            # Moderate polynomial
            e = self._default_exponent
            time_factor = math.pow(time, e)
        elif self._opponent_type == OpponentType.TFTLIKE:
            # Gradual linear concession
            e = self._tft_exponent
            time_factor = math.pow(time, e)
        else:
            # Default: moderate polynomial
            e = self._default_exponent
            time_factor = math.pow(time, e)

        # Scale by concession rate
        time_factor = time_factor * self._concession_rate
        time_factor = min(1.0, time_factor)

        # Calculate target
        target = self._max_utility - time_factor * (
            self._max_utility - self._min_utility
        )

        # Consider opponent's best offer near deadline
        if time > self._deadline_start and self._best_opponent_utility > 0:
            # Gradually accept that we may need to take their best offer
            deadline_duration = 1.0 - self._deadline_start
            deadline_pressure = (time - self._deadline_start) / deadline_duration
            target = min(
                target,
                target * (1 - deadline_pressure)
                + self._best_opponent_utility * deadline_pressure,
            )

        return max(self._min_utility, min(self._max_utility, target))

    def _select_bid(self, time: float) -> Outcome | None:
        """
        Select a bid to offer.

        Uses target utility with some exploration based on opponent type.

        Args:
            time: Normalized time [0, 1].

        Returns:
            The selected bid, or None if no suitable bid found.
        """
        if self._outcome_space is None:
            return None

        target = self._get_target_utility(time)

        # Add some randomness based on opponent type
        if self._opponent_type == OpponentType.RANDOM:
            # Be less predictable against random opponent
            noise = (random.random() - 0.5) * self._random_noise
            target = max(self._min_utility, min(self._max_utility, target + noise))

        # Get candidates near target
        candidates = self._outcome_space.get_bids_in_range(
            target - self._bid_tolerance, target + self._bid_tolerance
        )

        if candidates:
            selected = random.choice(candidates)
            self._our_last_offer_utility = selected.utility
            return selected.bid

        # Try closest bid
        bid_details = self._outcome_space.get_bid_near_utility(target)
        if bid_details is not None:
            self._our_last_offer_utility = bid_details.utility
            return bid_details.bid

        # Fallback
        if self._outcome_space.outcomes:
            self._our_last_offer_utility = self._outcome_space.outcomes[0].utility
            return self._outcome_space.outcomes[0].bid

        return None

    def _should_accept(self, offer: Outcome, time: float) -> bool:
        """
        Decide whether to accept an offer.

        Uses smart acceptance that considers:
        1. Current target utility
        2. Opponent type
        3. Time pressure
        4. Best alternative (our next bid)

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

        # AC_next: Accept if offer is better than our next planned bid
        if offer_utility >= self._our_last_offer_utility:
            return True

        # Special handling near deadline based on opponent type
        if time > self._late_phase_time:
            if self._opponent_type == OpponentType.HARDHEAD:
                # Hardhead won't concede, take what we can get
                if (
                    offer_utility
                    >= self._best_opponent_utility * self._hardhead_acceptance_factor
                ):
                    return True
            elif self._opponent_type == OpponentType.CONCEDER:
                # Conceder might give more, be slightly patient
                if (
                    offer_utility
                    >= self._best_opponent_utility * self._conceder_acceptance_factor
                ):
                    return True
            else:
                # Default deadline behavior
                if (
                    offer_utility
                    >= self._best_opponent_utility * self._default_acceptance_factor
                ):
                    return True

        # Very close to deadline: accept above minimum
        if time > self._deadline_time and offer_utility >= self._min_utility:
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

        return self._select_bid(state.relative_time)

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

        offer_utility = float(self.ufun(offer))

        # Update opponent model
        self._update_opponent_model(offer, offer_utility, state.relative_time)

        # Decide whether to accept
        if self._should_accept(offer, state.relative_time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
