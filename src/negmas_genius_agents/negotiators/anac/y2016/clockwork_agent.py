"""ClockworkAgent from ANAC 2016."""

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

__all__ = ["ClockworkAgent"]


class ClockworkAgent(SAONegotiator):
    """
    ClockworkAgent negotiation agent from ANAC 2016.

    ClockworkAgent is a precision-timed negotiation agent that operates with
    clockwork-like regularity in its concession and bidding patterns using
    discrete negotiation phases.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2016.clockworkagent.ClockworkAgent

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
    Phase-based concession with pre-computed thresholds:

    - Negotiation divided into N phases (default 5)
    - Each phase has a fixed utility threshold computed at initialization
    - Thresholds decrease from 95% of max utility to reservation value
    - Decrease follows sqrt curve (faster early, slower late)
    - Current phase determined by: phase = floor(time * num_phases)

    Bids selected using Nash-like scoring: own_util * (0.5 + 0.5 * opponent_util)
    to favor mutually beneficial outcomes while maintaining self-interest.

    **Acceptance Strategy:**
    Phase-threshold based acceptance:

    - Accepts if offer utility meets or exceeds current phase threshold
    - Final phase fallback (t >= 0.95): accepts any offer above
      reservation value

    **Opponent Modeling:**
    Frequency-based opponent preference estimation:

    - Tracks value frequencies for each issue from opponent bids
    - Estimates opponent utility as average normalized frequency
      across all issues (equal issue weights)
    - Used in Nash-like bid scoring to select mutually acceptable bids
    - Maintains best received bid and utility for reference

    The clockwork approach provides predictability and systematic behavior
    while still adapting bid selection to opponent preferences.

    Args:
        phases: Number of discrete negotiation phases (default 5)
        min_utility: Minimum acceptable utility threshold (default 0.6)
        early_time: Time threshold for early phase best-bid offering (default 0.02)
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
        phases: int = 5,
        min_utility: float = 0.6,
        early_time: float = 0.02,
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
        self._phases = phases
        self._min_utility = min_utility
        self._early_time = early_time
        self._deadline_time = deadline_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._reservation_value: float = min_utility

        # Phase management
        self._phase_thresholds: list[float] = []
        self._current_phase: int = 0

        # Tracking
        self._opponent_utilities: list[float] = []
        self._best_received_utility: float = 0.0
        self._best_received_bid: Outcome | None = None

        # Opponent model (simple)
        self._opponent_value_frequencies: dict[int, dict[str, int]] = {}

    def _initialize(self) -> None:
        """Initialize the outcome space and phase thresholds."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility

        # Create phase thresholds (decreasing)
        self._phase_thresholds = []
        start = self._max_utility * 0.95
        end = self._reservation_value
        for i in range(self._phases):
            progress = i / (self._phases - 1) if self._phases > 1 else 0
            # Boulware-like decrease
            threshold = start - (start - end) * math.pow(progress, 0.5)
            self._phase_thresholds.append(threshold)

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._current_phase = 0
        self._opponent_utilities = []
        self._best_received_utility = 0.0
        self._best_received_bid = None
        self._opponent_value_frequencies = {}

    def _get_current_phase(self, time: float) -> int:
        """Determine current phase based on time."""
        phase = int(time * self._phases)
        return min(phase, self._phases - 1)

    def _get_threshold(self, time: float) -> float:
        """Get threshold for current phase."""
        phase = self._get_current_phase(time)
        self._current_phase = phase

        if not self._phase_thresholds:
            return self._max_utility * 0.8

        return self._phase_thresholds[phase]

    def _update_opponent(self, bid: Outcome, utility: float) -> None:
        """Update opponent tracking."""
        self._opponent_utilities.append(utility)

        if utility > self._best_received_utility:
            self._best_received_utility = utility
            self._best_received_bid = bid

        # Track frequencies
        for i, value in enumerate(bid):
            if i not in self._opponent_value_frequencies:
                self._opponent_value_frequencies[i] = {}
            value_str = str(value)
            if value_str not in self._opponent_value_frequencies[i]:
                self._opponent_value_frequencies[i][value_str] = 0
            self._opponent_value_frequencies[i][value_str] += 1

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Simple opponent utility estimation."""
        if not self._opponent_value_frequencies:
            return 0.5

        n_issues = len(bid)
        total = 0.0

        for i, value in enumerate(bid):
            value_str = str(value)
            freq_map = self._opponent_value_frequencies.get(i, {})

            if not freq_map:
                total += 0.5 / n_issues
                continue

            count = freq_map.get(value_str, 0)
            max_count = max(freq_map.values()) if freq_map else 1
            total += (count / max_count if max_count > 0 else 0.5) / n_issues

        return total

    def _select_bid(self, time: float) -> Outcome | None:
        """Systematic bid selection."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._get_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            # Relax threshold
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            return self._best_bid

        # Score by Nash-like product
        scored: list[tuple[Outcome, float]] = []
        for bd in candidates:
            opp_util = self._estimate_opponent_utility(bd.bid)
            score = bd.utility * (0.5 + 0.5 * opp_util)
            scored.append((bd.bid, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Select from top with regularity (clockwork-like)
        n = min(4, len(scored))
        return random.choice(scored[:n])[0]

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal on schedule."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        if time < self._early_time:
            return self._best_bid

        return self._select_bid(time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond with precision timing."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        self._update_opponent(offer, offer_utility)

        threshold = self._get_threshold(time)

        # Accept if meets phase threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Final phase acceptance
        if time >= self._deadline_time:
            if offer_utility >= self._reservation_value:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
