"""BraveCat from ANAC 2014."""

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

__all__ = ["BraveCat"]


class BraveCat(SAONegotiator):
    """
    BraveCat from ANAC 2014.

    BraveCat implements a negotiation strategy based on the BOA (Bidding
    strategy, Opponent model, Acceptance condition) framework with combined
    acceptance conditions. It was designed to perform well across diverse
    negotiation scenarios.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.BraveCat.BraveCat

    **Offering Strategy:**
        Sigmoid-based concession curve with opponent utility maximization:
        - Three-phase concession pattern:
          * Early (t < 0.2): Stay at maximum utility
          * Middle (0.2 <= t < 0.8): Gradual concession = t^1.5 * range * 0.3
          * Late (t >= 0.8): Faster concession = sqrt(t) * remaining * 0.5
        - Bid selection prioritizes estimated opponent utility among
          candidates meeting threshold, promoting efficient agreements

    **Acceptance Strategy:**
        Combined acceptance using weighted voting (AC_combi + AC_next):
        - AC_combi: Accept if offer utility >= computed threshold
        - AC_next: Accept if offer utility >= utility of next planned bid
        - Combined decision: Both true -> accept; one true -> use alpha weight
          (if alpha * AC_combi + (1-alpha) * AC_next >= 0.5, accept)
        - Emergency acceptance at t > 0.99 for offers above minimum
        This combination balances aspirational and practical acceptance.

    **Opponent Modeling:**
        Time-weighted frequency analysis for opponent preference estimation:
        - Frequencies weighted by (1 + time * 2), emphasizing recent bids
        - Total weight tracked for normalization across negotiation
        - Utility estimated as normalized weighted frequency score
        - Best opponent bid tracked for potential reciprocal consideration
        - Model drives bid selection to maximize opponent satisfaction
          while meeting own threshold requirements

    Args:
        acceptance_alpha: Weight for AC_combi vs AC_next voting (default 0.8).
        early_phase_end: Time threshold ending early high-utility phase (default 0.2).
        late_phase_start: Time threshold starting faster concession phase (default 0.8).
        emergency_time: Time threshold for emergency acceptance (default 0.99).
        middle_phase_exponent: Exponent for middle phase concession (default 1.5).
        middle_phase_concession_factor: Concession factor for middle phase (default 0.3).
        late_phase_exponent: Exponent for late phase concession (default 0.5).
        late_phase_concession_factor: Concession factor for late phase (default 0.5).
        top_candidates_divisor: Divisor for selecting top candidates (default 3).
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
        acceptance_alpha: float = 0.8,
        early_phase_end: float = 0.2,
        late_phase_start: float = 0.8,
        emergency_time: float = 0.99,
        middle_phase_exponent: float = 1.5,
        middle_phase_concession_factor: float = 0.3,
        late_phase_exponent: float = 0.5,
        late_phase_concession_factor: float = 0.5,
        top_candidates_divisor: int = 3,
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
        self._acceptance_alpha = acceptance_alpha
        self._early_phase_end = early_phase_end
        self._late_phase_start = late_phase_start
        self._emergency_time = emergency_time
        self._middle_phase_exponent = middle_phase_exponent
        self._middle_phase_concession_factor = middle_phase_concession_factor
        self._late_phase_exponent = late_phase_exponent
        self._late_phase_concession_factor = late_phase_concession_factor
        self._top_candidates_divisor = top_candidates_divisor
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_value_weights: dict[int, dict] = {}
        self._total_weight: float = 0.0

        # State
        self._min_utility: float = 0.5
        self._max_utility: float = 1.0
        self._last_bid: Outcome | None = None

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = max(0.5, self._outcome_space.min_utility)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_value_weights = {}
        self._total_weight = 0.0
        self._last_bid = None

    def _update_opponent_model(self, bid: Outcome, time: float) -> None:
        """Update opponent model with time-weighted frequencies."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        # Time-weighted frequency update
        weight = 1.0 + time * 2.0
        self._total_weight += weight

        for i, value in enumerate(bid):
            if i not in self._opponent_value_weights:
                self._opponent_value_weights[i] = {}
            self._opponent_value_weights[i][value] = (
                self._opponent_value_weights[i].get(value, 0) + weight
            )

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility."""
        if not self._opponent_value_weights or self._total_weight == 0:
            return 0.5

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            if i in self._opponent_value_weights:
                value_weight = self._opponent_value_weights[i].get(value, 0)
                max_weight = max(self._opponent_value_weights[i].values())
                if max_weight > 0:
                    total_score += value_weight / max_weight

        return total_score / num_issues if num_issues > 0 else 0.5

    def _compute_target_utility(self, time: float) -> float:
        """Compute adaptive target utility."""
        # BraveCat uses a sigmoid-like curve
        if time < self._early_phase_end:
            # Stay high early
            return self._max_utility
        elif time < self._late_phase_start:
            # Gradual concession
            phase_time = (time - self._early_phase_end) / (
                self._late_phase_start - self._early_phase_end
            )
            concession = (
                phase_time**self._middle_phase_exponent
                * (self._max_utility - self._min_utility)
                * self._middle_phase_concession_factor
            )
            return self._max_utility - concession
        else:
            # Faster concession near end
            phase_time = (time - self._late_phase_start) / (
                1.0 - self._late_phase_start
            )
            base = (
                self._max_utility
                - (self._max_utility - self._min_utility)
                * self._middle_phase_concession_factor
            )
            extra = (
                phase_time**self._late_phase_exponent
                * (base - self._min_utility)
                * self._late_phase_concession_factor
            )
            return max(self._min_utility, base - extra)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid maximizing opponent utility while meeting target."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._compute_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Select bid that maximizes estimated opponent utility
        if self._opponent_value_weights:
            best_bid = None
            best_opp_util = -1.0

            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                if opp_util > best_opp_util:
                    best_opp_util = opp_util
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        # No model yet, return random from top
        return random.choice(
            candidates[: max(1, len(candidates) // self._top_candidates_divisor)]
        ).bid

    def _ac_combi(self, offer_utility: float, time: float) -> bool:
        """Combined acceptance condition."""
        target = self._compute_target_utility(time)
        return offer_utility >= target

    def _ac_next(self, offer_utility: float, time: float) -> bool:
        """Accept if offer >= our next bid."""
        next_bid = self._select_bid(time)
        if next_bid is None or self.ufun is None:
            return False
        next_utility = float(self.ufun(next_bid))
        return offer_utility >= next_utility

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        bid = self._select_bid(time)
        self._last_bid = bid
        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond using combined acceptance strategies."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))
        self._update_opponent_model(offer, time)

        # Combined acceptance: weighted vote
        ac_combi_result = self._ac_combi(offer_utility, time)
        ac_next_result = self._ac_next(offer_utility, time)

        if ac_combi_result and ac_next_result:
            return ResponseType.ACCEPT_OFFER

        if ac_combi_result or ac_next_result:
            # Use alpha to decide
            score = self._acceptance_alpha if ac_combi_result else 0.0
            score += (1 - self._acceptance_alpha) if ac_next_result else 0.0
            if score >= 0.5:
                return ResponseType.ACCEPT_OFFER

        # Emergency acceptance near deadline
        if time > self._emergency_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
