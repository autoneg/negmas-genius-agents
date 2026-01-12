"""ParsCat from ANAC 2016."""

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

__all__ = ["ParsCat"]


class ParsCat(SAONegotiator):
    """
    ParsCat negotiation agent from ANAC 2016.

    ParsCat uses time-dependent concession strategy with opponent modeling and
    Nash product-based bid selection for win-win negotiation outcomes.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2016.parscat.ParsCat

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
    Boulware concession with domain-adaptive reservation:

    - Early phase (t < 0.1): Conservative at 98% of max utility
    - Main phase: Standard Boulware formula:
      target = max_util - (max_util - reservation) * t^(1/e)
    - Reservation value computed from domain analysis:
      max(min_utility, 0.4 * mean_util + 0.6 * min_utility)

    Bid selection uses Nash product scoring (own_util * opponent_util)
    to maximize joint utility. Selects randomly from top-5 candidates
    for unpredictability.

    **Acceptance Strategy:**
    Threshold-based acceptance:

    - Accepts if offer utility meets or exceeds Boulware target threshold
    - Near deadline (t >= 0.95): accepts if above reservation value

    **Opponent Modeling:**
    Frequency-based preference estimation with issue weighting:

    - Tracks value frequencies for each issue from opponent bids
    - Updates issue weights based on value consistency between
      consecutive bids (unchanged values get +0.1 weight, normalized)
    - Estimates opponent utility as weighted sum of normalized frequencies
    - Tracks best received bid and utility for reference

    Args:
        e: Concession exponent for Boulware curve (default 0.2)
        min_utility: Minimum acceptable utility threshold (default 0.65)
        early_phase_end: End time for early conservative phase (default 0.1)
        early_time: Time threshold for early phase best-bid offering (default 0.05)
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
        e: float = 0.2,
        min_utility: float = 0.65,
        early_phase_end: float = 0.1,
        early_time: float = 0.05,
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
        self._early_phase_end = early_phase_end
        self._early_time = early_time
        self._deadline_time = deadline_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_domain_utility: float = 0.0
        self._reservation_value: float = min_utility

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._opponent_issue_weights: dict[int, float] = {}
        self._opponent_value_frequencies: dict[int, dict[str, int]] = {}

        # Tracking
        self._last_received_utility: float = 0.0
        self._best_received_utility: float = 0.0
        self._best_received_bid: Outcome | None = None

    def _initialize(self) -> None:
        """Initialize the outcome space and domain analysis."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility
            self._min_domain_utility = self._outcome_space.min_utility

        # Initialize opponent model
        if self.nmi is not None:
            n_issues = len(self.nmi.issues)
            for i in range(n_issues):
                self._opponent_issue_weights[i] = 1.0 / n_issues
                self._opponent_value_frequencies[i] = {}

        # Set reservation value based on domain
        self._compute_reservation_value()

        self._initialized = True

    def _compute_reservation_value(self) -> None:
        """Compute reservation value based on domain analysis."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return

        utilities = [bd.utility for bd in self._outcome_space.outcomes]
        if not utilities:
            return

        mean_util = sum(utilities) / len(utilities)
        # Reservation: weighted average between mean and min_utility
        self._reservation_value = max(
            self._min_utility, 0.4 * mean_util + 0.6 * self._min_utility
        )

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_bids = []
        self._last_received_utility = 0.0
        self._best_received_utility = 0.0
        self._best_received_bid = None

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent model based on received bid."""
        if bid is None:
            return

        self._opponent_bids.append(bid)

        # Track value frequencies for each issue
        for i, value in enumerate(bid):
            if i not in self._opponent_value_frequencies:
                self._opponent_value_frequencies[i] = {}

            value_str = str(value)
            if value_str not in self._opponent_value_frequencies[i]:
                self._opponent_value_frequencies[i][value_str] = 0
            self._opponent_value_frequencies[i][value_str] += 1

        # Update issue weights
        self._update_issue_weights()

    def _update_issue_weights(self) -> None:
        """Update estimated opponent issue weights based on consistency."""
        if len(self._opponent_bids) < 2:
            return

        last_bid = self._opponent_bids[-1]
        prev_bid = self._opponent_bids[-2]

        # Issues that don't change are more important to opponent
        for i in range(len(last_bid)):
            if last_bid[i] == prev_bid[i]:
                self._opponent_issue_weights[i] += 0.1

        # Normalize
        total = sum(self._opponent_issue_weights.values())
        if total > 0:
            for i in self._opponent_issue_weights:
                self._opponent_issue_weights[i] /= total

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent's utility for a bid."""
        if not self._opponent_value_frequencies:
            return 0.5

        total_utility = 0.0

        for i, value in enumerate(bid):
            weight = self._opponent_issue_weights.get(i, 0.0)
            value_str = str(value)

            freq_map = self._opponent_value_frequencies.get(i, {})
            if not freq_map:
                total_utility += weight * 0.5
                continue

            value_count = freq_map.get(value_str, 0)
            max_count = max(freq_map.values()) if freq_map else 1

            value_utility = value_count / max_count if max_count > 0 else 0.5
            total_utility += weight * value_utility

        return total_utility

    def _get_target_utility(self, time: float) -> float:
        """Calculate target utility using Boulware concession."""
        if time < self._early_phase_end:
            return self._max_utility * 0.98

        # Boulware concession formula
        f_t = math.pow(time, 1 / self._e) if self._e > 0 else time
        target = self._max_utility - (self._max_utility - self._reservation_value) * f_t

        return max(target, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid considering Nash product with opponent utility."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)

        # Get candidates above target
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            for bd in self._outcome_space.outcomes:
                if bd.utility >= self._reservation_value:
                    candidates = [bd]
                    break

        if not candidates:
            return self._best_bid

        # Score by Nash product (own_util * opp_util)
        scored_bids: list[tuple[Outcome, float]] = []

        for bd in candidates:
            own_util = bd.utility
            opp_util = self._estimate_opponent_utility(bd.bid)
            nash_score = own_util * opp_util
            scored_bids.append((bd.bid, nash_score))

        scored_bids.sort(key=lambda x: x[1], reverse=True)

        # Pick from top candidates with randomness
        top_n = min(5, len(scored_bids))
        return random.choice(scored_bids[:top_n])[0]

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        # Early game: offer best bid
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

        # Update opponent model
        self._update_opponent_model(offer)

        offer_utility = float(self.ufun(offer))
        self._last_received_utility = offer_utility

        # Track best received offer
        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility
            self._best_received_bid = offer

        time = state.relative_time
        target = self._get_target_utility(time)

        # Accept if exceeds target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Near deadline: accept if above reservation
        if time >= self._deadline_time and offer_utility >= self._reservation_value:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
