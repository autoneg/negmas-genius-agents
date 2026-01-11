"""Farma2017 (Farma17) from ANAC 2017."""

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

__all__ = ["Farma2017", "Farma17"]


class Farma2017(SAONegotiator):
    """
    Farma17 from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This is a reimplementation of Farma17 from ANAC 2017.
    Original: agents.anac.y2017.farma.Farma17

    Farma is a series of agents that competed in multiple ANAC years,
    including ANAC 2016 with Farma (original version). Farma17 uses a
    hybrid strategy combining time-dependent concession with opponent
    behavior analysis and Nash-inspired bid selection.

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/nego/node/7

    **Offering Strategy:**
        Uses exponential concession curve (time^(1/e)) for smooth, gradual
        concession that starts slow and accelerates. Bids are scored by
        combining own utility (70%) with estimated opponent preference (30%)
        to approximate Nash equilibrium. Maintains bid history to avoid
        repetitive offers.

    **Acceptance Strategy:**
        Accepts offers above the current exponential threshold, adjusted
        slightly upward when opponent average utility is high (>0.6).
        Late-game (>95% time) accepts the best opponent offer if it
        exceeds minimum utility.

    **Opponent Modeling:**
        Uses frequency analysis of opponent's recent bids to estimate their
        preferences. Calculates opponent preference for each bid based on
        overlap with values appearing in opponent's offer history. Tracks
        windowed average (last 5 offers) to detect cooperation patterns.

    Args:
        min_utility: Minimum acceptable utility (default 0.5).
        e: Concession exponent controlling curve shape (default 0.2).
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
        min_utility: float = 0.5,
        e: float = 0.2,
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
        self._e = e
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._opponent_utilities: list[float] = []
        self._best_opponent_utility: float = 0.0

        # Bid history
        self._our_bids: set[tuple] = set()

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
        self._our_bids = set()

    def _update_opponent_model(self, offer: Outcome) -> None:
        """Update opponent modeling."""
        if self.ufun is None:
            return

        self._opponent_bids.append(offer)
        offer_utility = float(self.ufun(offer))
        self._opponent_utilities.append(offer_utility)
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)

    def _get_opponent_window_avg(self, window: int = 5) -> float:
        """Get average utility of opponent's recent offers."""
        if not self._opponent_utilities:
            return 0.5
        recent = self._opponent_utilities[-window:]
        return sum(recent) / len(recent)

    def _calculate_threshold(self, time: float) -> float:
        """Calculate acceptance threshold using exponential concession."""
        # Exponential concession: starts slow, accelerates
        factor = math.pow(time, 1.0 / self._e) if self._e > 0 else time

        utility_range = self._max_utility - self._min_utility
        threshold = self._max_utility - factor * utility_range

        # Adjust based on opponent's average
        opp_avg = self._get_opponent_window_avg()
        if opp_avg > 0.6:
            # Opponent offers are good, be slightly more patient
            threshold = min(threshold + 0.02, self._max_utility * 0.98)

        return max(threshold, self._min_utility)

    def _estimate_opponent_preference(self, bid: Outcome) -> float:
        """Estimate opponent's preference for a bid using frequency analysis."""
        if not self._opponent_bids or not isinstance(bid, tuple):
            return 0.5

        # Calculate overlap with opponent's preferred values
        matches = 0
        total = 0

        for opp_bid in self._opponent_bids[-10:]:  # Recent bids
            if not isinstance(opp_bid, tuple):
                continue
            for i, val in enumerate(bid):
                if i < len(opp_bid) and val == opp_bid[i]:
                    matches += 1
                total += 1

        return matches / total if total > 0 else 0.5

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Select a bid above threshold, preferring those likely to be accepted."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(self._min_utility)

        if not candidates:
            return self._best_bid

        # Score candidates by estimated opponent preference
        scored: list[tuple[float, Outcome]] = []
        for candidate in candidates:
            bid = candidate.bid
            # Avoid repetition
            if isinstance(bid, tuple) and bid in self._our_bids:
                continue

            opp_pref = self._estimate_opponent_preference(bid)
            # Balance our utility with opponent's estimated preference
            score = 0.7 * candidate.utility + 0.3 * opp_pref
            scored.append((score, bid))

        if scored:
            # Sort by score and pick from top candidates
            scored.sort(reverse=True)
            top_candidates = scored[: max(1, len(scored) // 3)]
            _, selected = random.choice(top_candidates)

            if isinstance(selected, tuple):
                self._our_bids.add(selected)
            return selected

        bid = random.choice(candidates).bid
        if isinstance(bid, tuple):
            self._our_bids.add(bid)
        return bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        threshold = self._calculate_threshold(time)

        return self._select_bid(threshold)

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
        threshold = self._calculate_threshold(time)
        offer_utility = float(self.ufun(offer))

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Late game: accept if it's the best opponent offer
        if time > 0.95 and offer_utility >= self._best_opponent_utility - 0.01:
            if offer_utility >= self._min_utility:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


# Alias matching original Genius class name
Farma17 = Farma2017
