"""ParsAgent3 (ShahAgent) from ANAC 2017."""

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

__all__ = ["ParsAgent3", "ShahAgent"]


class ParsAgent3(SAONegotiator):
    """
    ParsAgent3 (ShahAgent) from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This is a reimplementation of ParsAgent3 from ANAC 2017.
    Original: agents.anac.y2017.parsagent3.ShahAgent

    ParsAgent3 is the 2017 version of the successful ParsAgent series that
    competed in ANAC 2015 (ParsAgent) and ANAC 2016 (ParsCat). The series
    is known for its adaptive opponent modeling and Nash product optimization.

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/nego/node/7

    **Offering Strategy:**
        Three-phase approach:
        - Phase 1 (0-30%): Near-optimal bids (max_utility * (1 - 0.1*time)).
        - Phase 2 (30-85%): Gradual concession from 97% to 75% of max utility.
        - Phase 3 (85-100%): Faster concession toward minimum utility.
        Bids are selected to maximize Nash product (our_utility * opponent_utility),
        using frequency analysis to estimate opponent preferences.

    **Acceptance Strategy:**
        Accepts offers above the phase-dependent target utility. Near deadline
        (>95%) accepts above minimum. Late-game (>90%) accepts offers close
        to opponent's best if above minimum utility.

    **Opponent Modeling:**
        Uses frequency analysis of opponent bid values to estimate opponent
        utility for any bid. Counts how often each issue value appears in
        opponent's history and scores bids based on overlap. This enables
        Nash product optimization without knowing opponent's actual utility
        function.

    Args:
        min_utility: Minimum acceptable utility (default 0.65).
        phase1_end: End of phase 1 (high target) (default 0.3).
        phase2_end: End of phase 2 (gradual concession) (default 0.85).
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
        min_utility: float = 0.65,
        phase1_end: float = 0.3,
        phase2_end: float = 0.85,
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
        self._phase1_end = phase1_end
        self._phase2_end = phase2_end
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._opponent_utilities: list[float] = []
        self._best_opponent_utility: float = 0.0

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
        self._opponent_bids = []
        self._opponent_utilities = []
        self._best_opponent_utility = 0.0

    def _update_opponent_model(self, offer: Outcome) -> None:
        """Update opponent modeling based on received offer."""
        if self.ufun is None:
            return

        self._opponent_bids.append(offer)
        offer_utility = float(self.ufun(offer))
        self._opponent_utilities.append(offer_utility)
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """
        Estimate opponent's utility for a bid based on frequency analysis.

        Uses frequency of issue values in opponent's bids to estimate
        which values the opponent prefers.
        """
        if not self._opponent_bids or not isinstance(bid, tuple):
            return 0.5  # Default estimate

        # Count frequency of each value at each position
        value_frequencies: dict[int, dict[any, int]] = {}

        for opp_bid in self._opponent_bids:
            if not isinstance(opp_bid, tuple):
                continue
            for i, val in enumerate(opp_bid):
                if i not in value_frequencies:
                    value_frequencies[i] = {}
                value_frequencies[i][val] = value_frequencies[i].get(val, 0) + 1

        # Score the bid based on how often its values appear in opponent bids
        total_score = 0.0
        num_issues = len(bid)

        for i, val in enumerate(bid):
            if i in value_frequencies:
                freq = value_frequencies[i].get(val, 0)
                total_freq = sum(value_frequencies[i].values())
                if total_freq > 0:
                    total_score += freq / total_freq

        return total_score / num_issues if num_issues > 0 else 0.5

    def _calculate_target_utility(self, time: float) -> float:
        """Calculate target utility based on negotiation phase."""
        if time < self._phase1_end:
            # Phase 1: High target
            return self._max_utility * (1 - 0.1 * time)
        elif time < self._phase2_end:
            # Phase 2: Gradual concession
            progress = (time - self._phase1_end) / (self._phase2_end - self._phase1_end)
            start_util = self._max_utility * 0.97
            end_util = self._max_utility * 0.75
            return start_util - progress * (start_util - end_util)
        else:
            # Phase 3: Faster concession
            progress = (time - self._phase2_end) / (1.0 - self._phase2_end)
            start_util = self._max_utility * 0.75
            end_util = self._min_utility
            return start_util - progress * (start_util - end_util)

    def _select_bid(self, target_utility: float) -> Outcome | None:
        """Select bid that maximizes Nash product with opponent."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        candidates = self._outcome_space.get_bids_above(
            max(target_utility - 0.1, self._min_utility)
        )

        if not candidates:
            candidates = self._outcome_space.get_bids_above(self._min_utility)

        if not candidates:
            return self._best_bid

        # Select bid that maximizes Nash product
        best_score = -1.0
        best_candidates: list[Outcome] = []

        for candidate in candidates:
            our_utility = candidate.utility
            opp_utility = self._estimate_opponent_utility(candidate.bid)

            # Nash product
            nash_score = our_utility * opp_utility

            if nash_score > best_score:
                best_score = nash_score
                best_candidates = [candidate.bid]
            elif abs(nash_score - best_score) < 0.01:
                best_candidates.append(candidate.bid)

        if best_candidates:
            return random.choice(best_candidates)

        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        target = self._calculate_target_utility(time)

        return self._select_bid(target)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        self._update_opponent_model(offer)

        time = state.relative_time
        target = self._calculate_target_utility(time)
        offer_utility = float(self.ufun(offer))

        # Accept if above target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if above minimum and near deadline
        if time > 0.95 and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        # Accept if best opponent offer and we're late
        if time > 0.9 and offer_utility >= self._best_opponent_utility - 0.01:
            if offer_utility >= self._min_utility:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


# Alias matching original Genius class name
ShahAgent = ParsAgent3
