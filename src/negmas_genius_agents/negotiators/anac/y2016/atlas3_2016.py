"""Atlas32016 from ANAC 2016."""

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

__all__ = ["Atlas32016"]


class Atlas32016(SAONegotiator):
    """
    Atlas32016 negotiation agent from ANAC 2016.

    Atlas32016 is an updated version of Atlas3 from ANAC 2015 adapted for the
    2016 competition with improved opponent modeling, multi-phase concession
    strategy, and end-game optimization.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2016.atlas3.Atlas32016

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
    Uses a four-phase time-dependent concession approach:

    - Phase 1 (t < 0.2): Conservative at 95% of max utility
    - Phase 2 (0.2 <= t < 0.7): Boulware concession from 95% to 75%
      using configurable exponent
    - Phase 3 (0.7 <= t < 0.9): Moderate linear concession toward
      reservation value + 10%
    - Phase 4 (t >= 0.9): Final concession to reservation value

    Bids are selected using Nash product scoring (own_util * opponent_util)
    to maximize mutual benefit. In the final phase with few remaining turns,
    may offer the best received bid if above reservation.

    **Acceptance Strategy:**
    Accepts an offer if:

    - Offer utility meets or exceeds current phase threshold
    - In final phase (estimated < 5 turns remaining): accepts if above
      reservation value

    Time tracking estimates remaining turns based on observed time
    between negotiations.

    **Opponent Modeling:**
    Frequency-based opponent preference estimation:

    - Tracks value frequencies for each issue from opponent bids
    - Updates issue weights based on value consistency between
      consecutive bids (unchanged = more important)
    - Estimates opponent utility using weighted normalized frequencies
    - Maintains history of opponent bids with their utilities

    Args:
        e: Concession exponent for Boulware curve (default 0.15)
        phase1_end: End time for phase 1 (default 0.2)
        phase2_end: End time for phase 2 (default 0.7)
        phase3_end: End time for phase 3 (default 0.9)
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
        phase1_end: float = 0.2,
        phase2_end: float = 0.7,
        phase3_end: float = 0.9,
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
        self._phase1_end = phase1_end
        self._phase2_end = phase2_end
        self._phase3_end = phase3_end
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._reservation_value: float = 0.6

        # Time tracking
        self._time_scale: float = 0.0
        self._last_time: float = 0.0

        # Opponent tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._own_bids: list[tuple[Outcome, float]] = []
        self._best_received_utility: float = 0.0
        self._best_received_bid: Outcome | None = None

        # Opponent model
        self._opponent_value_frequencies: dict[int, dict[str, int]] = {}
        self._opponent_issue_weights: dict[int, float] = {}

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
            self._min_utility = self._outcome_space.min_utility

        # Set reservation based on domain
        self._reservation_value = max(
            0.6, self._min_utility + 0.4 * (self._max_utility - self._min_utility)
        )

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
        self._own_bids = []
        self._time_scale = 0.0
        self._last_time = 0.0
        self._best_received_utility = 0.0
        self._best_received_bid = None

    def _update_time_scale(self, time: float) -> None:
        """Track time between turns."""
        if self._last_time > 0:
            self._time_scale = time - self._last_time
        self._last_time = time

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent model from received bid."""
        if bid is None:
            return

        for i, value in enumerate(bid):
            if i not in self._opponent_value_frequencies:
                self._opponent_value_frequencies[i] = {}

            value_str = str(value)
            if value_str not in self._opponent_value_frequencies[i]:
                self._opponent_value_frequencies[i][value_str] = 0
            self._opponent_value_frequencies[i][value_str] += 1

        # Update issue weights
        if len(self._opponent_bids) >= 2:
            self._update_issue_weights()

    def _update_issue_weights(self) -> None:
        """Update opponent issue weights."""
        if len(self._opponent_bids) < 2:
            return

        last_bid = self._opponent_bids[-1][0]
        prev_bid = self._opponent_bids[-2][0]

        for i in range(len(last_bid)):
            if last_bid[i] == prev_bid[i]:
                self._opponent_issue_weights[i] += 0.1

        total = sum(self._opponent_issue_weights.values())
        if total > 0:
            for i in self._opponent_issue_weights:
                self._opponent_issue_weights[i] /= total

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

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold using multi-phase approach."""
        if time < self._phase1_end:
            # Phase 1: High threshold
            return self._max_utility * 0.95
        elif time < self._phase2_end:
            # Phase 2: Boulware concession
            phase_time = (time - self._phase1_end) / (
                self._phase2_end - self._phase1_end
            )
            f_t = math.pow(phase_time, 1 / self._e) if self._e > 0 else phase_time
            start = self._max_utility * 0.95
            end = self._max_utility * 0.75
            return start - (start - end) * f_t
        elif time < self._phase3_end:
            # Phase 3: Moderate concession
            phase_time = (time - self._phase2_end) / (
                self._phase3_end - self._phase2_end
            )
            start = self._max_utility * 0.75
            end = self._reservation_value + 0.1
            return start - (start - end) * phase_time
        else:
            # Phase 4: End-game
            phase_time = (time - self._phase3_end) / (1.0 - self._phase3_end)
            start = self._reservation_value + 0.1
            end = self._reservation_value
            return max(start - (start - end) * phase_time, self._reservation_value)

    def _is_in_final_phase(self, time: float) -> bool:
        """Check if in final negotiation phase."""
        if self._time_scale <= 0:
            return False
        remaining_turns = (1.0 - time) / self._time_scale
        return remaining_turns < 5

    def _search_bid(self, threshold: float, time: float) -> Outcome | None:
        """Search for a bid meeting the threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            return self._best_bid

        # Score by Nash product
        scored: list[tuple[Outcome, float]] = []
        for bd in candidates:
            opp_util = self._estimate_opponent_utility(bd.bid)
            score = bd.utility * opp_util
            scored.append((bd.bid, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        top_n = min(5, len(scored))
        return random.choice(scored[:top_n])[0]

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        self._update_time_scale(time)

        # Final phase: offer best received if acceptable
        if self._is_in_final_phase(time) and self._best_received_bid is not None:
            if self._best_received_utility >= self._reservation_value:
                return self._best_received_bid

        threshold = self._compute_threshold(time)
        bid = self._search_bid(threshold, time)

        if bid is not None and self.ufun is not None:
            self._own_bids.append((bid, float(self.ufun(bid))))

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

        time = state.relative_time
        self._update_time_scale(time)

        offer_utility = float(self.ufun(offer))
        self._opponent_bids.append((offer, offer_utility))
        self._update_opponent_model(offer)

        # Track best received
        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility
            self._best_received_bid = offer

        threshold = self._compute_threshold(time)

        # Accept if meets threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Final phase acceptance
        if self._is_in_final_phase(time) and offer_utility > self._reservation_value:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
