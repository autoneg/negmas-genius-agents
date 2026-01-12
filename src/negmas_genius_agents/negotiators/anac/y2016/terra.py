"""Terra from ANAC 2016."""

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

__all__ = ["Terra"]


class Terra(SAONegotiator):
    """
    Terra negotiation agent from ANAC 2016.

    Terra is a grounded and stable negotiation agent that maintains firm
    positions while adapting to opponent behavior through careful observation.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2016.terra.Terra

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
    Three-phase firm-to-flexible concession:

    - Phase 1 (t < 0.15): Very firm at 95% of max utility
    - Phase 2 (0.15 <= t < 0.85): Boulware concession from 95% to 75%
      using firmness parameter (default 0.15, lower = more firm)
    - Phase 3 (t >= 0.85): Accelerated linear concession to reservation

    Bid selection uses Nash product scoring (own_util * opponent_util)
    to balance self-interest with cooperation. Selects randomly from
    top-5 candidates for variety.

    **Acceptance Strategy:**
    Multi-criteria acceptance:

    - Accepts if offer utility meets or exceeds current phase threshold
    - Accepts if offer utility >= agent's last offered bid utility
    - Near deadline (t >= 0.95): accepts if above reservation value

    **Opponent Modeling:**
    Frequency-based preference estimation with issue weighting:

    - Tracks value frequencies for each issue from opponent bids
    - Updates issue weights based on value consistency between
      consecutive bids (unchanged values get +0.08 weight, normalized)
    - Estimates opponent utility as weighted sum of normalized frequencies
    - Tracks best received bid and utility for reference

    Args:
        firmness: Firmness parameter - concession exponent (default 0.15,
            lower = more firm with slower concession)
        min_utility: Minimum acceptable utility threshold (default 0.6)
        phase1_end: End time for phase 1 firm phase (default 0.15)
        phase2_end: End time for phase 2 Boulware phase (default 0.85)
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
        firmness: float = 0.15,
        min_utility: float = 0.6,
        phase1_end: float = 0.15,
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
        self._firmness = firmness
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
        self._opponent_value_frequencies = {}
        self._best_received_utility = 0.0
        self._best_received_bid = None
        self._last_offered_utility = self._max_utility

        if self.nmi is not None:
            n_issues = len(self.nmi.issues)
            for i in range(n_issues):
                self._opponent_issue_weights[i] = 1.0 / n_issues
                self._opponent_value_frequencies[i] = {}

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update opponent model based on received bid."""
        if bid is None:
            return

        self._opponent_bids.append(bid)
        self._opponent_utilities.append(utility)

        # Track best received
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

        # Update issue weights based on consistency
        if len(self._opponent_bids) >= 2:
            self._update_issue_weights()

    def _update_issue_weights(self) -> None:
        """Update opponent issue weights."""
        if len(self._opponent_bids) < 2:
            return

        last_bid = self._opponent_bids[-1]
        prev_bid = self._opponent_bids[-2]

        # Issues that stay consistent are more important to opponent
        for i in range(len(last_bid)):
            if last_bid[i] == prev_bid[i]:
                self._opponent_issue_weights[i] += 0.08

        # Normalize
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

    def _get_target_utility(self, time: float) -> float:
        """Calculate target utility - firm early, flexible late."""
        if time < self._phase1_end:
            # Very firm in early phase
            return self._max_utility * 0.95
        elif time < self._phase2_end:
            # Gradual Boulware concession
            phase_time = (time - self._phase1_end) / (
                self._phase2_end - self._phase1_end
            )
            f_t = (
                math.pow(phase_time, 1 / self._firmness)
                if self._firmness > 0
                else phase_time
            )
            start = self._max_utility * 0.95
            end = self._max_utility * 0.75
            return start - (start - end) * f_t
        else:
            # End-game: accelerated concession
            phase_time = (time - self._phase2_end) / (1.0 - self._phase2_end)
            start = self._max_utility * 0.75
            end = self._reservation_value
            return max(start - (start - end) * phase_time, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid considering both parties' interests."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            # Relax and find any acceptable bid
            for bd in self._outcome_space.outcomes:
                if bd.utility >= self._reservation_value:
                    candidates = [bd]
                    break

        if not candidates:
            return self._best_bid

        # Score by Nash product
        scored: list[tuple[Outcome, float]] = []
        for bd in candidates:
            opp_util = self._estimate_opponent_utility(bd.bid)
            nash_score = bd.utility * opp_util
            scored.append((bd.bid, nash_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Select from top with randomness
        top_n = min(5, len(scored))
        return random.choice(scored[:top_n])[0]

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        # Very early: best bid
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
