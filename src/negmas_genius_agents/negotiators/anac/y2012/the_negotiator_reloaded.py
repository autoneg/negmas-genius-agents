"""TheNegotiatorReloaded from ANAC 2012."""

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

__all__ = ["TheNegotiatorReloaded"]


class TheNegotiatorReloaded(SAONegotiator):
    """
    TheNegotiatorReloaded negotiation agent from ANAC 2012.

    TheNegotiatorReloaded is an improved version of TheNegotiator from ANAC 2011.
    It features adaptive Boulware-like concession with opponent toughness estimation.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        Original Genius class: ``agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded``

        ANAC 2012: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        Uses adaptive time-dependent Boulware concession:

        - Base formula: target = max - (max - min) * t^(1/e)
          where e is the concession factor.
        - Concession factor adapts based on opponent toughness:
          - Tough opponent (toughness > 0.7): e reduced to 50% (slower concession)
          - Flexible opponent (toughness < 0.3): e increased to 150% (we can
            be tougher)
        - Near deadline (t > 0.9): Additional adjustment based on time pressure,
          reducing target by up to 50% of utility range.

        If opponent's best bid meets target, offers that bid back.
        Otherwise selects randomly from top-k candidates near target.

    **Acceptance Strategy:**
        Target-based acceptance with progressive deadline handling:

        - Accept if offer utility >= target utility.
        - Near deadline (t > 0.95): Accept if offer >= opponent's best - 0.02.
        - Very near deadline (t > 0.99): Accept if offer >= minimum utility.
        - Accept if offer >= utility of bid we would propose.

    **Opponent Modeling:**
        Estimates opponent toughness from bidding behavior:

        - Tracks opponent bid history with utility statistics.
        - Tracks best bid received from opponent.
        - Computes average utility of opponent bids.
        - Detects concession: consecutive improving bids (3+ bids trending up).
        - Estimates toughness from utility variance:
          - Low variance = tough opponent (not conceding)
          - High variance = flexible opponent
          - Formula: toughness = 1.0 - stdev * 5 (bounded 0.1-0.9)

    Args:
        concession_factor: Controls base concession speed (default 0.02).
            Lower values result in slower (tougher) concession.
        min_utility: Minimum acceptable utility (default 0.65).
        flexibility_time: Time threshold for increasing flexibility near
            deadline (default 0.9).
        near_deadline_time: Time threshold for relaxed acceptance near deadline
            (default 0.95).
        final_deadline_time: Time threshold for accepting any offer above
            minimum utility (default 0.99).
        min_toughness: Minimum opponent toughness estimate (default 0.1).
        max_toughness: Maximum opponent toughness estimate (default 0.9).
        toughness_stdev_multiplier: Multiplier for standard deviation in
            toughness calculation (default 5.0).
        tough_opponent_threshold: Threshold for tough opponent detection
            (default 0.7).
        flexible_opponent_threshold: Threshold for flexible opponent detection
            (default 0.3).
        tough_opponent_factor: Factor applied to concession when opponent is
            tough (default 0.5).
        flexible_opponent_factor: Factor applied to concession when opponent is
            flexible (default 1.5).
        min_concession_factor: Minimum concession factor (default 0.01).
        max_concession_factor: Maximum concession factor (default 0.1).
        flexibility_adjustment: Adjustment factor for flexibility near deadline
            (default 0.5).
        bid_tolerance: Tolerance for bid selection range (default 0.03).
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
        concession_factor: float = 0.02,
        min_utility: float = 0.65,
        flexibility_time: float = 0.9,
        near_deadline_time: float = 0.95,
        final_deadline_time: float = 0.99,
        min_toughness: float = 0.1,
        max_toughness: float = 0.9,
        toughness_stdev_multiplier: float = 5.0,
        tough_opponent_threshold: float = 0.7,
        flexible_opponent_threshold: float = 0.3,
        tough_opponent_factor: float = 0.5,
        flexible_opponent_factor: float = 1.5,
        min_concession_factor: float = 0.01,
        max_concession_factor: float = 0.1,
        flexibility_adjustment: float = 0.5,
        bid_tolerance: float = 0.03,
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
        self._concession_factor = concession_factor
        self._min_utility = min_utility
        self._flexibility_time = flexibility_time
        self._near_deadline_time = near_deadline_time
        self._final_deadline_time = final_deadline_time
        self._min_toughness = min_toughness
        self._max_toughness = max_toughness
        self._toughness_stdev_multiplier = toughness_stdev_multiplier
        self._tough_opponent_threshold = tough_opponent_threshold
        self._flexible_opponent_threshold = flexible_opponent_threshold
        self._tough_opponent_factor = tough_opponent_factor
        self._flexible_opponent_factor = flexible_opponent_factor
        self._min_concession_factor = min_concession_factor
        self._max_concession_factor = max_concession_factor
        self._flexibility_adjustment = flexibility_adjustment
        self._bid_tolerance = bid_tolerance
        self._top_k_candidates = top_k_candidates
        self._acceptance_margin = acceptance_margin

        # Will be initialized when negotiation starts
        self._outcome_space: SortedOutcomeSpace | None = None
        self._max_utility: float = 1.0
        self._reservation_value: float = 0.0
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[tuple[Outcome, float]] = []  # (bid, our_utility)
        self._opponent_best_bid: Outcome | None = None
        self._opponent_best_utility: float = 0.0
        self._opponent_avg_utility: float = 0.0
        self._opponent_concession_detected: bool = False

        # Own bidding state
        self._own_bids: list[tuple[Outcome, float]] = []
        self._last_bid: Outcome | None = None
        self._target_utility: float = 1.0

        # Adaptation parameters
        self._time_pressure: float = 0.0
        self._opponent_toughness: float = 0.5

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

        # Get reservation value if available
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
        self._opponent_bids = []
        self._opponent_best_bid = None
        self._opponent_best_utility = 0.0
        self._opponent_avg_utility = 0.0
        self._opponent_concession_detected = False
        self._own_bids = []
        self._last_bid = None
        self._target_utility = self._max_utility
        self._time_pressure = 0.0
        self._opponent_toughness = 0.5

    def _update_opponent_model(self, bid: Outcome) -> None:
        """
        Update opponent model with a new bid.

        Tracks opponent's bidding pattern and estimates their toughness.
        """
        if self.ufun is None or bid is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append((bid, utility))

        # Track best bid from opponent
        if utility > self._opponent_best_utility:
            self._opponent_best_utility = utility
            self._opponent_best_bid = bid

        # Update average utility
        total = sum(u for _, u in self._opponent_bids)
        self._opponent_avg_utility = total / len(self._opponent_bids)

        # Detect concession (opponent improving offers to us)
        if len(self._opponent_bids) >= 3:
            recent_utils = [u for _, u in self._opponent_bids[-3:]]
            if all(
                recent_utils[i] <= recent_utils[i + 1]
                for i in range(len(recent_utils) - 1)
            ):
                self._opponent_concession_detected = True

        # Estimate opponent toughness
        self._estimate_opponent_toughness()

    def _estimate_opponent_toughness(self) -> None:
        """
        Estimate how tough the opponent is based on their bidding pattern.

        Toughness is measured as how little they concede over time.
        """
        if len(self._opponent_bids) < 5:
            return

        # Look at the variance in opponent utilities
        utilities = [u for _, u in self._opponent_bids]
        mean = sum(utilities) / len(utilities)
        variance = sum((u - mean) ** 2 for u in utilities) / len(utilities)

        # Low variance = tough opponent (not conceding)
        # High variance = flexible opponent
        stdev = math.sqrt(variance) if variance > 0 else 0

        # Normalize toughness: high stdev = low toughness
        self._opponent_toughness = max(
            self._min_toughness,
            min(self._max_toughness, 1.0 - stdev * self._toughness_stdev_multiplier),
        )

    def _compute_target_utility(self, time: float) -> float:
        """
        Compute target utility for the current time.

        Uses an adaptive concession function that responds to:
        - Time pressure
        - Opponent toughness
        """
        # Base Boulware-like concession: slow at start, faster at end
        # u(t) = max - (max - min) * t^(1/e)
        e = self._concession_factor

        # Adapt concession rate based on opponent toughness
        if self._opponent_toughness > self._tough_opponent_threshold:
            # Tough opponent - concede slower
            e = max(self._min_concession_factor, e * self._tough_opponent_factor)
        elif self._opponent_toughness < self._flexible_opponent_threshold:
            # Flexible opponent - we can be tougher
            e = min(self._max_concession_factor, e * self._flexible_opponent_factor)

        # Calculate concession
        if e > 0:
            concession = math.pow(time, 1.0 / e)
        else:
            concession = 0.0

        target = (
            self._max_utility - (self._max_utility - self._min_utility) * concession
        )

        # Apply time pressure adjustment
        self._time_pressure = time * time  # Quadratic time pressure
        if time > self._flexibility_time:
            # Near deadline, increase flexibility
            adjustment = (
                (time - self._flexibility_time)
                / (1.0 - self._flexibility_time)
                * self._flexibility_adjustment
                * (self._max_utility - self._min_utility)
            )
            target = max(self._min_utility, target - adjustment)

        return max(target, self._min_utility, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """
        Select a bid to offer.

        Strategy:
        1. If opponent's best bid meets our target, offer it back
        2. Otherwise, select from candidates near target utility
        """
        if self._outcome_space is None:
            return None

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
                # Last resort: best bid
                if self._outcome_space.outcomes:
                    return self._outcome_space.outcomes[0].bid
                return None

        # Select randomly from top candidates
        if len(candidates) > 1:
            top_k = min(self._top_k_candidates, len(candidates))
            selected = random.choice(candidates[:top_k])
        else:
            selected = candidates[0]

        return selected.bid

    def _accept_condition(self, offer: Outcome, time: float) -> bool:
        """
        Decide whether to accept an offer.

        Accepts if:
        1. Offer utility >= target utility
        2. Near deadline and offer >= opponent's best - margin
        3. Very near deadline and offer >= minimum
        """
        if self.ufun is None or offer is None:
            return False

        offer_utility = float(self.ufun(offer))

        # Accept if offer meets our target
        if offer_utility >= self._target_utility:
            return True

        # Near deadline strategies
        if time > self._near_deadline_time:
            # Accept if close to opponent's best
            if offer_utility >= self._opponent_best_utility - self._acceptance_margin:
                return True

        if time > self._final_deadline_time:
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
