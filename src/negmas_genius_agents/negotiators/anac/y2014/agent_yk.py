"""AgentYK from ANAC 2014."""

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

__all__ = ["AgentYK"]


class AgentYK(SAONegotiator):
    """
    AgentYK from ANAC 2014.

    AgentYK implements a phase-based negotiation strategy that transitions
    through distinct behavioral phases: hardball, concession, and final
    acceptance. The agent learns opponent preferences to improve bid selection.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.AgentYK.AgentYK

    **Offering Strategy:**
        Three-phase approach based on negotiation time:
        1. Hardball phase (t < hardball_time): Maintains maximum utility target,
           making only top-tier offers without concession.
        2. Concession phase (hardball_time <= t < concession_end_time): Gradual concession
           using exponential curve: concession = (normalized_time^speed) * range.
           Target decreases at concession_factor of full potential concession.
        3. Final phase (t >= concession_end_time): Rapid concession with linear decay,
           preparing for deal closure while maintaining minimum threshold.

        After hardball phase, opponent model influences bid selection by
        favoring bids estimated to satisfy opponent preferences.

    **Acceptance Strategy:**
        Phase-aware acceptance with fallback conditions:
        1. Accept if offer utility meets phase-appropriate target
        2. Accept if offer matches or exceeds next planned bid utility
        3. Final phase acceptance (t > final_acceptance_time) for any offer above minimum
        The strategy becomes progressively more flexible as deadline approaches.

    **Opponent Modeling:**
        Frequency-based preference learning activated after hardball phase:
        - Tracks value frequencies per issue from opponent bid history
        - Normalizes frequencies to estimate relative value preferences
        - Combines own utility with opponent utility estimate (opponent_weight)
          for bid scoring when selecting offers
        - Best opponent bid tracked for potential late-game reference

    Args:
        hardball_time: Duration of initial hardball phase (default 0.3).
        concession_speed: Exponent for concession curve steepness (default 2.0).
        concession_end_time: Time threshold for ending concession phase (default 0.95).
        final_acceptance_time: Time threshold for final phase acceptance (default 0.99).
        min_utility_floor: Floor value for minimum utility (default 0.5).
        concession_factor: Factor applied to concession amount (default 0.6).
        final_phase_factor: Factor for final phase linear decay (default 0.5).
        opponent_weight: Weight for opponent utility in bid scoring (default 0.3).
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
        hardball_time: float = 0.3,
        concession_speed: float = 2.0,
        concession_end_time: float = 0.95,
        final_acceptance_time: float = 0.99,
        min_utility_floor: float = 0.5,
        concession_factor: float = 0.6,
        final_phase_factor: float = 0.5,
        opponent_weight: float = 0.3,
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
        self._hardball_time = hardball_time
        self._concession_speed = concession_speed
        self._concession_end_time = concession_end_time
        self._final_acceptance_time = final_acceptance_time
        self._min_utility_floor = min_utility_floor
        self._concession_factor = concession_factor
        self._final_phase_factor = final_phase_factor
        self._opponent_weight = opponent_weight
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_value_freq: dict[int, dict] = {}

        # State
        self._min_utility: float = min_utility_floor
        self._max_utility: float = 1.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = max(
                self._min_utility_floor, self._outcome_space.min_utility
            )
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_value_freq = {}

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent model."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append(bid)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        # Update value frequencies
        for i, value in enumerate(bid):
            if i not in self._opponent_value_freq:
                self._opponent_value_freq[i] = {}
            self._opponent_value_freq[i][value] = (
                self._opponent_value_freq[i].get(value, 0) + 1
            )

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility using frequency model."""
        if not self._opponent_value_freq:
            return 0.5

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            if i in self._opponent_value_freq:
                freq = self._opponent_value_freq[i].get(value, 0)
                max_freq = max(self._opponent_value_freq[i].values())
                if max_freq > 0:
                    total_score += freq / max_freq

        return total_score / num_issues if num_issues > 0 else 0.5

    def _compute_target_utility(self, time: float) -> float:
        """Compute target utility based on negotiation phase."""
        # Hardball phase
        if time < self._hardball_time:
            return self._max_utility

        # Concession phase
        if time < self._concession_end_time:
            # Normalize time to concession phase
            phase_time = (time - self._hardball_time) / (
                self._concession_end_time - self._hardball_time
            )
            # Exponential concession curve
            concession = (phase_time**self._concession_speed) * (
                self._max_utility - self._min_utility
            )
            target = self._max_utility - concession * self._concession_factor
            return max(self._min_utility, target)

        # Final phase - rapid concession
        final_time = (time - self._concession_end_time) / (
            1.0 - self._concession_end_time
        )
        target = self._min_utility + (self._max_utility - self._min_utility) * (
            1 - final_time * self._final_phase_factor
        )
        return max(self._min_utility, target)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid considering opponent preferences."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._compute_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Use opponent model if available
        if self._opponent_value_freq and time > self._hardball_time:
            best_bid = None
            best_score = -1.0

            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                # Favor bids that opponent might like
                score = bd.utility + self._opponent_weight * opp_util
                if score > best_score:
                    best_score = score
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        # Default: best utility from candidates
        return candidates[0].bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
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

        time = state.relative_time
        offer_utility = float(self.ufun(offer))
        self._update_opponent_model(offer)

        target = self._compute_target_utility(time)

        # Accept if above target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than our next offer
        next_bid = self._select_bid(time)
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid))
            if offer_utility >= next_utility:
                return ResponseType.ACCEPT_OFFER

        # Final phase - accept anything reasonable
        if time > self._final_acceptance_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
