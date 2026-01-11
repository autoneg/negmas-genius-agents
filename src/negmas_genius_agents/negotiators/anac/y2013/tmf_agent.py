"""
TMFAgent from ANAC 2013 - 3rd place agent.

This module implements TMFAgent (The Mischief of Fortune), the 3rd place
agent at ANAC 2013. TMFAgent combines adaptive time-dependent concession
with frequency-based opponent modeling for near-Pareto bid exploration.

References:
    - Baarslag, T., et al. (2013). "Evaluating Practical Negotiating Agents:
      Results and Analysis of the 2013 International Competition."
      Artificial Intelligence, 198, 73-103.
    - Original Java implementation: agents.anac.y2013.TMFAgent.TMFAgent
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

__all__ = ["TMFAgent"]


class TMFAgent(SAONegotiator):
    """
    TMFAgent from ANAC 2013 - 3rd place agent.

    TMFAgent (The Mischief of Fortune) combines adaptive time-dependent
    concession with frequency-based opponent modeling. It balances
    exploitation of opponent preferences with exploration of the Pareto
    frontier.

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
        if self._opponent_concession_rate > 0.05:
            # Opponent is conceding well - stay tough (lower e = more Boulware)
            self._adaptive_e = max(0.08, self._e - 0.05)
        elif self._opponent_concession_rate < 0.01 and time > 0.3:
            # Opponent is tough - be slightly more conceding
            self._adaptive_e = min(0.5, self._e + 0.1 * time)
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
        if self._total_opponent_offers > 5:
            # Score candidates by opponent utility estimate
            scored_candidates = [
                (bd, self._estimate_opponent_utility(bd.bid)) for bd in candidates
            ]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Exploration vs exploitation
            if random.random() < self._exploration_rate:
                # Explore: pick randomly from top candidates
                top_k = max(1, len(scored_candidates) // 3)
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
        if time > 0.95:
            deadline_factor = 1 - (time - 0.95) / 0.05 * 0.15  # Up to 15% reduction
            return base_threshold * deadline_factor

        # If opponent has given us good offers, accept good ones
        if self._best_opponent_utility > base_threshold * 0.95:
            return min(base_threshold, self._best_opponent_utility * 0.98)

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
        if time > 0.99 and offer_utility >= self._min_utility_threshold:
            return ResponseType.ACCEPT_OFFER

        # Accept if this is the best offer and we're running low on time
        if time > 0.9 and offer_utility >= self._best_opponent_utility * 0.98:
            our_bid = self._select_bid(time)
            if our_bid is not None:
                our_utility = float(self.ufun(our_bid))
                if offer_utility >= our_utility:
                    return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
