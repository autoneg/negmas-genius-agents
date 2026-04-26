"""Ngent from ANAC 2016."""

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

__all__ = ["Ngent"]


class Ngent(SAONegotiator):
    """
    Ngent negotiation agent from ANAC 2016.

    Ngent uses a "gentle" concession strategy with careful opponent analysis
    for balanced negotiation outcomes.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2016.ngent.Ngent

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
    Gentle Boulware-style concession:

    - Very early (t < 0.05): Offers maximum utility bid
    - Main phase: Concession using formula:
      target = max_util - (max_util - reservation) * t^(1/gentleness) * 0.8
    - Lower gentleness parameter = slower concession (default 0.25)
    - The 0.8 factor ensures concession doesn't reach full range

    Bids selected with weighted scoring: 70% own utility + 30%
    estimated opponent utility, favoring mutually beneficial outcomes.

    **Acceptance Strategy:**
    Threshold-based with deadline fallbacks:

    - Accepts if offer utility meets or exceeds gentle threshold
    - Near deadline (t >= 0.95): accepts if above reservation value
    - Very near deadline (t >= 0.99): accepts if best received offer
      is above reservation value

    **Opponent Modeling:**
    Frequency-based preference estimation:

    - Tracks value frequencies for each issue from opponent bids
    - Estimates opponent utility as average normalized frequency
      across all issues (equal issue weights)
    - Maintains running average of opponent offer utilities
    - Tracks best received bid and utility for deadline decisions

    Args:
        min_utility: Minimum acceptable utility threshold (default 0.6)
        gentleness: Concession gentleness parameter (default 0.25,
            lower = gentler/slower concession)
        early_phase_end: End time for early max-utility phase (default 0.05)
        early_time: Time threshold for early phase best-bid offering (default 0.02)
        deadline_time: Time threshold for deadline acceptance (default 0.95)
        critical_time: Time threshold for critical deadline acceptance (default 0.99)
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
        min_utility: float = 0.6,
        gentleness: float = 0.25,
        early_phase_end: float = 0.05,
        early_time: float = 0.02,
        deadline_time: float = 0.95,
        critical_time: float = 0.99,
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
        self._min_utility = min_utility
        self._gentleness = gentleness
        self._early_phase_end = early_phase_end
        self._early_time = early_time
        self._deadline_time = deadline_time
        self._critical_time = critical_time
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

        # Tracking
        self._best_received_utility: float = 0.0
        self._best_received_bid: Outcome | None = None
        self._opponent_avg_utility: float = 0.0

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
        self._opponent_avg_utility = 0.0

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update opponent model."""
        if bid is None:
            return

        self._opponent_bids.append(bid)
        self._opponent_utilities.append(utility)

        # Track best received
        if utility > self._best_received_utility:
            self._best_received_utility = utility
            self._best_received_bid = bid

        # Update average
        self._opponent_avg_utility = sum(self._opponent_utilities) / len(
            self._opponent_utilities
        )

        # Track value frequencies
        for i, value in enumerate(bid):
            if i not in self._opponent_value_frequencies:
                self._opponent_value_frequencies[i] = {}

            value_str = str(value)
            if value_str not in self._opponent_value_frequencies[i]:
                self._opponent_value_frequencies[i][value_str] = 0
            self._opponent_value_frequencies[i][value_str] += 1

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent's utility for a bid."""
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

    def _get_target_utility(self, time: float) -> float:
        """Calculate target utility with gentle concession."""
        if time < self._early_phase_end:
            return self._max_utility

        # Gentle concession formula
        # Lower gentleness = slower concession
        f_t = math.pow(time, 1 / self._gentleness) if self._gentleness > 0 else time
        concession_range = self._max_utility - self._reservation_value
        target = self._max_utility - concession_range * f_t * 0.8

        # Don't go below reservation
        return max(target, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid above target utility."""
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

        # Score by estimated opponent utility (prefer mutually good bids)
        scored: list[tuple[Outcome, float]] = []
        for bd in candidates:
            opp_util = self._estimate_opponent_utility(bd.bid)
            # Combine own utility with opponent utility estimate
            score = 0.7 * bd.utility + 0.3 * opp_util
            scored.append((bd.bid, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Pick from top candidates
        top_n = min(5, len(scored))
        return random.choice(scored[:top_n])[0]

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        # Early game
        if time < self._early_time:
            return self._best_bid

        return self._select_bid(time)

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

        # Near deadline: accept if above reservation
        if time >= self._deadline_time and offer_utility >= self._reservation_value:
            return ResponseType.ACCEPT_OFFER

        # Very near deadline: accept best received
        if (
            time >= self._critical_time
            and self._best_received_utility >= self._reservation_value
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
