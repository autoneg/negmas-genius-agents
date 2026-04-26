"""TMFAgent from ANAC 2013."""

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

__all__ = ["TMFAgent"]


class TMFAgent(SAONegotiator):
    """
    TMFAgent from ANAC 2013 - 3rd place agent.

    TMFAgent (The Mischief of Fortune) combines adaptive time-dependent
    concession with frequency-based opponent modeling. It balances
    exploitation of opponent preferences with exploration of the Pareto
    frontier.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        Original Genius class: ``agents.anac.y2013.TMFAgent.TMFAgent``

        ANAC 2013: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        Adaptive time-dependent concession: threshold = max - (max - min) *
        t^(1/adaptive_e). The exponent adapts based on opponent behavior:
        tougher (lower e) if opponent is conceding well, more conceding
        (higher e) if opponent is tough. Uses exploration vs exploitation
        trade-off (exploration_rate): explores by picking randomly from
        top third of candidates by opponent utility, exploits by picking
        the best for opponent (near-Pareto).

    **Acceptance Strategy:**
        Dynamic threshold that adapts near deadline: reduces by up to 15%
        in final 5% of negotiation. Also accepts if opponent has given
        good offers (>= 95% of threshold) and current offer is near best
        received (>= 98%). Late game (> 0.9): accepts if offer >= our next
        proposal. Emergency acceptance (> 0.99) above min_utility_threshold.

    **Opponent Modeling:**
        Frequency-based model similar to TheFawkes, tracking issue value
        counts. Additionally tracks opponent concession rate by comparing
        average utility of recent 10 offers vs early offers. Uses concession
        rate to adapt the concession exponent: stay tough if opponent
        concedes, concede more if opponent is tough.

    Args:
        e: Base concession exponent (default 0.15, more Boulware than TheFawkes)
        min_utility_threshold: Minimum acceptable utility (default 0.65)
        exploration_rate: How much to explore near-Pareto bids (default 0.3)
        deadline_threshold: Time after which to apply deadline factor (default 0.95)
        emergency_threshold: Time after which to accept anything above min_utility (default 0.99)
        late_game_threshold: Time after which to compare with our next proposal (default 0.9)
        opponent_conceding_threshold: Threshold for detecting good opponent concession (default 0.05)
        opponent_tough_threshold: Threshold for detecting tough opponent (default 0.01)
        min_adaptive_e: Minimum adaptive concession exponent (default 0.08)
        adaptive_e_reduction: Reduction to e when opponent concedes (default 0.05)
        max_adaptive_e: Maximum adaptive concession exponent (default 0.5)
        adaptive_e_time_factor: Factor for adaptive e increase with time (default 0.1)
        early_game_offers: Number of opponent offers before using opponent model (default 5)
        top_k_divisor: Divisor for selecting top candidates (default 3)
        deadline_factor_max: Maximum reduction factor near deadline (default 0.15)
        good_opponent_utility_threshold: Multiplier for threshold to detect good opponent offers (default 0.95)
        good_opponent_acceptance_multiplier: Multiplier for opponent best utility in acceptance (default 0.98)
        late_game_acceptance_multiplier: Multiplier for opponent best utility in late game acceptance (default 0.98)
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
        e: float = 0.15,
        min_utility_threshold: float = 0.65,
        exploration_rate: float = 0.3,
        deadline_threshold: float = 0.95,
        emergency_threshold: float = 0.99,
        late_game_threshold: float = 0.9,
        opponent_conceding_threshold: float = 0.05,
        opponent_tough_threshold: float = 0.01,
        min_adaptive_e: float = 0.08,
        adaptive_e_reduction: float = 0.05,
        max_adaptive_e: float = 0.5,
        adaptive_e_time_factor: float = 0.1,
        early_game_offers: int = 5,
        top_k_divisor: int = 3,
        deadline_factor_max: float = 0.15,
        good_opponent_utility_threshold: float = 0.95,
        good_opponent_acceptance_multiplier: float = 0.98,
        late_game_acceptance_multiplier: float = 0.98,
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
        self._e = e
        self._min_utility_threshold = min_utility_threshold
        self._exploration_rate = exploration_rate
        self._deadline_threshold = deadline_threshold
        self._emergency_threshold = emergency_threshold
        self._late_game_threshold = late_game_threshold
        self._opponent_conceding_threshold = opponent_conceding_threshold
        self._opponent_tough_threshold = opponent_tough_threshold
        self._min_adaptive_e = min_adaptive_e
        self._adaptive_e_reduction = adaptive_e_reduction
        self._max_adaptive_e = max_adaptive_e
        self._adaptive_e_time_factor = adaptive_e_time_factor
        self._early_game_offers = early_game_offers
        self._top_k_divisor = top_k_divisor
        self._deadline_factor_max = deadline_factor_max
        self._good_opponent_utility_threshold = good_opponent_utility_threshold
        self._good_opponent_acceptance_multiplier = good_opponent_acceptance_multiplier
        self._late_game_acceptance_multiplier = late_game_acceptance_multiplier
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling - frequency counts per issue value
        self._issue_value_counts: dict[str, dict[str, int]] = {}
        self._total_opponent_offers: int = 0

        # Opponent behavior tracking for adaptive concession
        self._opponent_utilities: list[float] = []
        self._opponent_concession_rate: float = 0.0

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._adaptive_e: float = e

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = self._outcome_space.min_utility
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset opponent model
        self._issue_value_counts = {}
        self._total_opponent_offers = 0
        self._opponent_utilities = []
        self._opponent_concession_rate = 0.0
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._adaptive_e = self._e

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update frequency-based opponent model and track concession."""
        if bid is None:
            return

        self._total_opponent_offers += 1

        # Track utility for concession estimation
        if self.ufun is not None:
            bid_utility = float(self.ufun(bid))
            self._opponent_utilities.append(bid_utility)

            # Track best opponent bid
            if bid_utility > self._best_opponent_utility:
                self._best_opponent_utility = bid_utility
                self._best_opponent_bid = bid

            # Estimate opponent concession rate (how fast they're improving our utility)
            if len(self._opponent_utilities) >= 10:
                recent = self._opponent_utilities[-10:]
                early = (
                    self._opponent_utilities[:10]
                    if len(self._opponent_utilities) >= 20
                    else self._opponent_utilities[: len(self._opponent_utilities) // 2]
                )
                if early:
                    recent_avg = sum(recent) / len(recent)
                    early_avg = sum(early) / len(early)
                    self._opponent_concession_rate = max(0, recent_avg - early_avg)

        # Count each issue value for frequency model
        for i, value in enumerate(bid):
            issue_key = str(i)
            value_key = str(value)

            if issue_key not in self._issue_value_counts:
                self._issue_value_counts[issue_key] = {}

            if value_key not in self._issue_value_counts[issue_key]:
                self._issue_value_counts[issue_key][value_key] = 0

            self._issue_value_counts[issue_key][value_key] += 1

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
                    # Normalize by max count for this issue
                    max_count = max(counts.values()) if counts else 1
                    total_score += counts[value_key] / max_count

        return total_score / num_issues if num_issues > 0 else 0.5

    def _update_adaptive_concession(self, time: float) -> None:
        """Adapt concession rate based on opponent behavior."""
        # If opponent is conceding (giving us better offers), we can be tougher
        # If opponent is tough, we may need to concede more
        if self._opponent_concession_rate > self._opponent_conceding_threshold:
            # Opponent is conceding well - stay tough (lower e = more Boulware)
            self._adaptive_e = max(
                self._min_adaptive_e, self._e - self._adaptive_e_reduction
            )
        elif (
            self._opponent_concession_rate < self._opponent_tough_threshold
            and time > 0.3
        ):
            # Opponent is tough - be slightly more conceding
            self._adaptive_e = min(
                self._max_adaptive_e, self._e + self._adaptive_e_time_factor * time
            )
        else:
            # Normal behavior
            self._adaptive_e = self._e

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold using adaptive time-dependent concession."""
        self._update_adaptive_concession(time)

        # Time-dependent formula with adaptive exponent
        if self._adaptive_e != 0:
            f_t = math.pow(time, 1 / self._adaptive_e)
        else:
            f_t = 0.0

        # Calculate threshold that decreases over time
        threshold = (
            self._max_utility - (self._max_utility - self._min_utility_threshold) * f_t
        )

        # Ensure we don't go below minimum threshold
        return max(threshold, self._min_utility_threshold)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid exploring near-Pareto frontier using opponent model."""
        if self._outcome_space is None:
            return None

        threshold = self._compute_threshold(time)

        # Get acceptable bids (above our threshold)
        candidates = self._outcome_space.get_bids_above(threshold)
        if not candidates:
            if self._outcome_space.outcomes:
                return self._outcome_space.outcomes[0].bid
            return None

        # If we have enough opponent data, use opponent model for Pareto exploration
        if self._total_opponent_offers > self._early_game_offers:
            # Score candidates by opponent utility estimate
            scored_candidates = [
                (bd, self._estimate_opponent_utility(bd.bid)) for bd in candidates
            ]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Exploration vs exploitation
            if random.random() < self._exploration_rate:
                # Explore: pick randomly from top candidates
                top_k = max(1, len(scored_candidates) // self._top_k_divisor)
                return random.choice(scored_candidates[:top_k])[0].bid
            else:
                # Exploit: pick best for opponent (near-Pareto)
                return scored_candidates[0][0].bid

        # Early negotiation: random from acceptable bids
        return random.choice(candidates).bid

    def _compute_acceptance_threshold(self, time: float) -> float:
        """Compute dynamic acceptance threshold."""
        base_threshold = self._compute_threshold(time)

        # Near deadline, be more flexible
        if time > self._deadline_threshold:
            deadline_factor = (
                1
                - (time - self._deadline_threshold)
                / (1 - self._deadline_threshold)
                * self._deadline_factor_max
            )  # Up to deadline_factor_max reduction
            return base_threshold * deadline_factor

        # If opponent has given us good offers, accept good ones
        if (
            self._best_opponent_utility
            > base_threshold * self._good_opponent_utility_threshold
        ):
            return min(
                base_threshold,
                self._best_opponent_utility * self._good_opponent_acceptance_multiplier,
            )

        return base_threshold

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        return self._select_bid(time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer with dynamic acceptance threshold."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        # Update opponent model
        self._update_opponent_model(offer)

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        time = state.relative_time

        # Compute acceptance threshold
        threshold = self._compute_acceptance_threshold(time)

        # Accept if offer meets threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Near deadline: accept anything above minimum
        if (
            time > self._emergency_threshold
            and offer_utility >= self._min_utility_threshold
        ):
            return ResponseType.ACCEPT_OFFER

        # Accept if this is the best offer and we're running low on time
        if (
            time > self._late_game_threshold
            and offer_utility
            >= self._best_opponent_utility * self._late_game_acceptance_multiplier
        ):
            our_bid = self._select_bid(time)
            if our_bid is not None:
                our_utility = float(self.ufun(our_bid))
                if offer_utility >= our_utility:
                    return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
