"""AgentSmith2016 from ANAC 2016."""

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

__all__ = ["AgentSmith2016"]


class AgentSmith2016(SAONegotiator):
    """
    AgentSmith2016 negotiation agent from ANAC 2016.

    AgentSmith2016 is an evolved version of the original AgentSmith from
    ANAC 2010 with enhanced opponent modeling, adaptive concession strategies,
    and Nash-based bid selection.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2016.agentsmith2016.AgentSmith2016

    References:
        .. code-block:: bibtex

            @inproceedings{fujita2016anac,
                title={The Sixth Automated Negotiating Agents Competition (ANAC 2016)},
                author={Fujita, Katsuhide and others},
                booktitle={Proceedings of the International Joint Conference on
                    Artificial Intelligence (IJCAI)},
                year={2016}
            }

        ANAC 2016 Competition: https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
    Uses a multi-phase adaptive concession strategy:

    - Phase 1 (t < 0.1): High aspiration at 97% of max utility
    - Phase 2 (0.1 <= t < 0.85): Boulware concession from 97% to 72%,
      with exponent adjusted based on detected opponent concession:
      - Opponent conceding: exponent increases (more flexible)
      - Opponent hardening: exponent decreases (more firm)
    - Phase 3 (t >= 0.85): End-game concession to reservation value

    Bid selection uses Nash product scoring: 0.6 * own_utility + 0.4 *
    (own_utility * estimated_opponent_utility) to balance self-interest
    with agreement probability.

    **Acceptance Strategy:**
    Accepts an offer if:

    - Offer utility meets or exceeds current target threshold
    - Offer utility is at least as good as the agent's last offered bid
    - Near deadline (t >= 0.95): accepts if above reservation value

    **Opponent Modeling:**
    Comprehensive opponent preference estimation:

    - Tracks value frequencies for each issue across all opponent bids
    - Updates issue weights based on consistency (unchanged values
      indicate higher importance)
    - Estimates opponent utility as weighted sum of normalized frequencies
    - Computes opponent concession rate from moving averages to adapt
      concession exponent

    Args:
        e: Base concession exponent (default 0.15)
        min_utility: Minimum acceptable utility threshold (default 0.6)
        phase1_end: End time for phase 1 (default 0.1)
        phase2_end: End time for phase 2 (default 0.85)
        early_time: Time threshold for early phase best-bid offering (default 0.03)
        deadline_time: Time threshold for deadline acceptance (default 0.95)
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
        min_utility: float = 0.6,
        phase1_end: float = 0.1,
        phase2_end: float = 0.85,
        early_time: float = 0.03,
        deadline_time: float = 0.95,
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
        self._min_utility = min_utility
        self._phase1_end = phase1_end
        self._phase2_end = phase2_end
        self._early_time = early_time
        self._deadline_time = deadline_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._reservation_value: float = min_utility

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._opponent_utilities: list[float] = []
        self._opponent_value_frequencies: dict[int, dict[str, int]] = {}
        self._opponent_issue_weights: dict[int, float] = {}
        self._opponent_concession: float = 0.0

        # Tracking
        self._best_received_utility: float = 0.0
        self._best_received_bid: Outcome | None = None
        self._last_offered_utility: float = 1.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility

        # Initialize opponent model
        if self.nmi is not None:
            n_issues = len(self.nmi.issues)
            for i in range(n_issues):
                self._opponent_issue_weights[i] = 1.0 / n_issues
                self._opponent_value_frequencies[i] = {}

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_bids = []
        self._opponent_utilities = []
        self._opponent_concession = 0.0
        self._best_received_utility = 0.0
        self._best_received_bid = None
        self._last_offered_utility = self._max_utility

        if self.nmi is not None:
            n_issues = len(self.nmi.issues)
            for i in range(n_issues):
                self._opponent_issue_weights[i] = 1.0 / n_issues
                self._opponent_value_frequencies[i] = {}

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update opponent model."""
        if bid is None:
            return

        self._opponent_bids.append(bid)
        self._opponent_utilities.append(utility)

        # Track best
        if utility > self._best_received_utility:
            self._best_received_utility = utility
            self._best_received_bid = bid

        # Track value frequencies
        for i, value in enumerate(bid):
            if i not in self._opponent_value_frequencies:
                self._opponent_value_frequencies[i] = {}

            value_str = str(value)
            if value_str not in self._opponent_value_frequencies[i]:
                self._opponent_value_frequencies[i][value_str] = 0
            self._opponent_value_frequencies[i][value_str] += 1

        # Update weights and concession
        if len(self._opponent_bids) >= 2:
            self._update_issue_weights()
            self._update_concession_estimate()

    def _update_issue_weights(self) -> None:
        """Update opponent issue weights."""
        if len(self._opponent_bids) < 2:
            return

        last_bid = self._opponent_bids[-1]
        prev_bid = self._opponent_bids[-2]

        for i in range(len(last_bid)):
            if last_bid[i] == prev_bid[i]:
                self._opponent_issue_weights[i] += 0.1

        total = sum(self._opponent_issue_weights.values())
        if total > 0:
            for i in self._opponent_issue_weights:
                self._opponent_issue_weights[i] /= total

    def _update_concession_estimate(self) -> None:
        """Estimate opponent's concession rate."""
        if len(self._opponent_utilities) < 5:
            return

        recent = self._opponent_utilities[-3:]
        earlier = (
            self._opponent_utilities[-6:-3]
            if len(self._opponent_utilities) >= 6
            else self._opponent_utilities[:3]
        )

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        # Positive = opponent giving better offers
        self._opponent_concession = recent_avg - earlier_avg

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent's utility for a bid."""
        if not self._opponent_value_frequencies:
            return 0.5

        total = 0.0
        for i, value in enumerate(bid):
            weight = self._opponent_issue_weights.get(i, 0.0)
            value_str = str(value)

            freq_map = self._opponent_value_frequencies.get(i, {})
            if not freq_map:
                total += weight * 0.5
                continue

            count = freq_map.get(value_str, 0)
            max_count = max(freq_map.values()) if freq_map else 1
            total += weight * (count / max_count if max_count > 0 else 0.5)

        return total

    def _get_target_utility(self, time: float) -> float:
        """Adaptive target utility with opponent awareness."""
        # Adjust exponent based on opponent behavior
        adjusted_e = self._e
        if self._opponent_concession > 0.01:
            # Opponent conceding, we can be slightly more flexible
            adjusted_e = min(0.25, self._e * 1.3)
        elif self._opponent_concession < -0.01:
            # Opponent hardening, be more firm
            adjusted_e = max(0.08, self._e * 0.7)

        if time < self._phase1_end:
            return self._max_utility * 0.97
        elif time < self._phase2_end:
            phase_time = (time - self._phase1_end) / (
                self._phase2_end - self._phase1_end
            )
            f_t = math.pow(phase_time, 1 / adjusted_e) if adjusted_e > 0 else phase_time
            start = self._max_utility * 0.97
            end = self._max_utility * 0.72
            return start - (start - end) * f_t
        else:
            # End-game
            phase_time = (time - self._phase2_end) / (1.0 - self._phase2_end)
            start = self._max_utility * 0.72
            end = self._reservation_value
            return max(start - (start - end) * phase_time, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid using Nash-based approach."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            for bd in self._outcome_space.outcomes:
                if bd.utility >= self._reservation_value:
                    candidates = [bd]
                    break

        if not candidates:
            return self._best_bid

        # Nash product scoring
        scored: list[tuple[Outcome, float]] = []
        for bd in candidates:
            opp_util = self._estimate_opponent_utility(bd.bid)
            # Weight toward own utility
            score = 0.6 * bd.utility + 0.4 * (bd.utility * opp_util)
            scored.append((bd.bid, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        top_n = min(5, len(scored))
        return random.choice(scored[:top_n])[0]

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        if time < self._early_time:
            return self._best_bid

        bid = self._select_bid(time)

        if bid is not None and self.ufun is not None:
            self._last_offered_utility = float(self.ufun(bid))

        return bid

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
        self._update_opponent_model(offer, offer_utility)

        time = state.relative_time
        target = self._get_target_utility(time)

        # Accept if meets target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than what we would offer
        if offer_utility >= self._last_offered_utility:
            return ResponseType.ACCEPT_OFFER

        # Near deadline
        if time >= self._deadline_time and offer_utility >= self._reservation_value:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
