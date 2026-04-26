"""GAgent from ANAC 2013."""

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

__all__ = ["GAgent"]


class GAgent(SAONegotiator):
    """
    GAgent from ANAC 2013.

    GAgent (originally AgentI) is a general purpose adaptive negotiator that
    aims to find mutually beneficial agreements through Nash-optimal bid
    selection and adaptive concession strategies.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        Original Genius class: ``agents.anac.y2013.GAgent.AgentI``

        ANAC 2013: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        Selects bids that balance own utility with estimated opponent utility
        using a Nash-like approach. For each candidate bid above the current
        threshold, computes a Nash score using the geometric mean:
        score = sqrt(own_utility * opponent_utility). Selects randomly from
        the top quarter of candidates by Nash score. Early in negotiation
        (< 10 opponent offers), returns random high-utility bids.

    **Acceptance Strategy:**
        Accepts offers above the adaptive threshold, which starts at
        initial_threshold (default 0.95) and concedes based on time^2.
        The concession rate adapts to opponent behavior: slower (0.7x) if
        opponent is conceding, faster (1.3x) if opponent is tough. Also
        accepts if the offer is better than what we would propose next.
        Emergency acceptance near deadline (> 0.99) for offers above
        min_threshold.

    **Opponent Modeling:**
        Frequency-based model tracking issue value counts from opponent offers.
        Estimates opponent utility for any bid by computing the average
        normalized frequency score across all issues. Detects opponent
        concession by comparing average utility of recent offers (last 5)
        vs early offers (first 5). Tracks best opponent bid for reference.

    Args:
        initial_threshold: Starting utility threshold (default 0.95)
        min_threshold: Minimum acceptable utility (default 0.6)
        concession_rate: Base rate of concession per time unit (default 0.1)
        time_pressure_threshold: Time after which to apply time pressure (default 0.95)
        emergency_acceptance_threshold: Time after which to accept anything above min_threshold (default 0.99)
        opponent_conceding_threshold: Threshold for detecting opponent concession (default 0.02)
        opponent_tough_threshold: Threshold for detecting tough opponent (default -0.02)
        concession_slow_factor: Factor to slow concession when opponent concedes (default 0.7)
        concession_fast_factor: Factor to speed concession when opponent is tough (default 1.3)
        time_pressure_multiplier: Multiplier for time pressure adjustment (default 0.3)
        early_game_offers: Number of opponent offers before using Nash scoring (default 10)
        top_k_divisor: Divisor for selecting top candidates (default 4)
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
        initial_threshold: float = 0.95,
        min_threshold: float = 0.6,
        concession_rate: float = 0.1,
        time_pressure_threshold: float = 0.95,
        emergency_acceptance_threshold: float = 0.99,
        opponent_conceding_threshold: float = 0.02,
        opponent_tough_threshold: float = -0.02,
        concession_slow_factor: float = 0.7,
        concession_fast_factor: float = 1.3,
        time_pressure_multiplier: float = 0.3,
        early_game_offers: int = 10,
        top_k_divisor: int = 4,
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
        self._initial_threshold = initial_threshold
        self._min_threshold = min_threshold
        self._concession_rate = concession_rate
        self._time_pressure_threshold = time_pressure_threshold
        self._emergency_acceptance_threshold = emergency_acceptance_threshold
        self._opponent_conceding_threshold = opponent_conceding_threshold
        self._opponent_tough_threshold = opponent_tough_threshold
        self._concession_slow_factor = concession_slow_factor
        self._concession_fast_factor = concession_fast_factor
        self._time_pressure_multiplier = time_pressure_multiplier
        self._early_game_offers = early_game_offers
        self._top_k_divisor = top_k_divisor
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_offers: list[tuple[Outcome, float]] = []
        self._issue_value_counts: dict[str, dict[str, int]] = {}
        self._total_opponent_offers: int = 0

        # State tracking
        self._current_threshold: float = initial_threshold
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_concession_estimate: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
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

        # Reset state
        self._opponent_offers = []
        self._issue_value_counts = {}
        self._total_opponent_offers = 0
        self._current_threshold = self._initial_threshold
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_concession_estimate = 0.0

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update opponent model based on received bid."""
        if bid is None:
            return

        self._opponent_offers.append((bid, utility))
        self._total_opponent_offers += 1

        # Track best opponent bid
        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        # Update issue value frequency counts
        for i, value in enumerate(bid):
            issue_key = str(i)
            value_key = str(value)

            if issue_key not in self._issue_value_counts:
                self._issue_value_counts[issue_key] = {}

            if value_key not in self._issue_value_counts[issue_key]:
                self._issue_value_counts[issue_key][value_key] = 0

            self._issue_value_counts[issue_key][value_key] += 1

        # Estimate opponent concession rate
        if len(self._opponent_offers) >= 5:
            recent_utils = [u for _, u in self._opponent_offers[-5:]]
            early_utils = [u for _, u in self._opponent_offers[:5]]
            if early_utils and recent_utils:
                self._opponent_concession_estimate = sum(recent_utils) / len(
                    recent_utils
                ) - sum(early_utils) / len(early_utils)

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent's utility for a bid based on frequency model."""
        if self._total_opponent_offers == 0 or bid is None:
            return 0.5

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            issue_key = str(i)
            value_key = str(value)

            if issue_key in self._issue_value_counts:
                counts = self._issue_value_counts[issue_key]
                if value_key in counts:
                    max_count = max(counts.values()) if counts else 1
                    total_score += counts[value_key] / max_count

        return total_score / num_issues if num_issues > 0 else 0.5

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold with adaptive concession."""
        # Base time-dependent concession
        base_concession = self._concession_rate * math.pow(time, 2)

        # Adjust based on opponent behavior
        if self._opponent_concession_estimate > self._opponent_conceding_threshold:
            # Opponent is conceding - stay tougher
            adjusted_rate = base_concession * self._concession_slow_factor
        elif self._opponent_concession_estimate < self._opponent_tough_threshold:
            # Opponent is getting tougher - concede more
            adjusted_rate = base_concession * self._concession_fast_factor
        else:
            adjusted_rate = base_concession

        # Calculate threshold
        threshold = self._initial_threshold - adjusted_rate * (
            self._initial_threshold - self._min_threshold
        )

        # Apply time pressure near deadline
        if time > self._time_pressure_threshold:
            time_pressure = (time - self._time_pressure_threshold) / (
                1 - self._time_pressure_threshold
            )
            threshold = (
                threshold
                - time_pressure
                * (threshold - self._min_threshold)
                * self._time_pressure_multiplier
            )

        self._current_threshold = max(threshold, self._min_threshold)
        return self._current_threshold

    def _compute_nash_score(self, own_utility: float, opponent_utility: float) -> float:
        """Compute Nash-like score using geometric mean."""
        if own_utility <= 0 or opponent_utility <= 0:
            return 0.0
        return math.sqrt(own_utility * opponent_utility)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid that balances own utility with estimated opponent utility."""
        if self._outcome_space is None:
            return None

        threshold = self._compute_threshold(time)

        # Get bids above our threshold
        candidates = self._outcome_space.get_bids_above(threshold)
        if not candidates:
            if self._outcome_space.outcomes:
                return self._outcome_space.outcomes[0].bid
            return None

        # Early game: just return high utility bids
        if self._total_opponent_offers < self._early_game_offers:
            return random.choice(candidates).bid

        # Score candidates by Nash product
        scored_candidates = []
        for bd in candidates:
            own_util = bd.utility
            opp_util = self._estimate_opponent_utility(bd.bid)
            nash_score = self._compute_nash_score(own_util, opp_util)
            scored_candidates.append((bd.bid, nash_score, own_util))

        if not scored_candidates:
            return random.choice(candidates).bid

        # Sort by Nash score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Select from top candidates with some randomness
        top_k = max(1, len(scored_candidates) // self._top_k_divisor)
        return random.choice(scored_candidates[:top_k])[0]

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

        # Update opponent model
        self._update_opponent_model(offer, offer_utility)

        time = state.relative_time
        threshold = self._compute_threshold(time)

        # Accept if above threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than what we would offer
        our_bid = self._select_bid(time)
        if our_bid is not None:
            our_utility = float(self.ufun(our_bid))
            if offer_utility >= our_utility:
                return ResponseType.ACCEPT_OFFER

        # Emergency acceptance near deadline
        if (
            time > self._emergency_acceptance_threshold
            and offer_utility >= self._min_threshold
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
