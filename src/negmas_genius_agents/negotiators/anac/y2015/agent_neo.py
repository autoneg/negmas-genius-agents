"""AgentNeo from ANAC 2015."""

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

__all__ = ["AgentNeo"]


class AgentNeo(SAONegotiator):
    """
    AgentNeo negotiation agent from ANAC 2015.

    AgentNeo uses adaptive opponent modeling with Nash-seeking bid selection
    to find mutually beneficial outcomes.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.AgentNeo.AgentNeo

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Three-phase Boulware-like concession:
          * Early (t<0.2): Very firm at 95% of max utility
          * Main (0.2<t<0.8): Gradual concession toward 60% utility
          * End (t>0.8): Accelerated concession toward minimum
        - Adaptive rate: concedes faster (e * 0.8) if opponent is conceding
        - Nash-seeking bid selection balancing own utility (70%) and
          estimated opponent utility (30%) after t=0.3

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - AC_Next: Accepts if offer utility >= utility of our next bid
        - End-game (t>0.95): Accepts if offer >= best opponent utility
          AND offer > reservation value

    **Opponent Modeling:**
        - Frequency-based model tracking issue-value occurrences
        - Detects opponent concession by comparing recent vs earlier offers
        - Estimates opponent utility based on normalized value frequencies
        - Updates concession rate based on detected opponent behavior

    Args:
        e: Concession exponent controlling concession speed (default 0.2)
        early_time_threshold: Time threshold for early phase (default 0.2)
        main_time_threshold: Time threshold for main/end phase transition (default 0.8)
        deadline_time_threshold: Time after which end-game acceptance triggers (default 0.95)
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
        early_time_threshold: float = 0.2,
        main_time_threshold: float = 0.8,
        deadline_time_threshold: float = 0.95,
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
        self._early_time_threshold = early_time_threshold
        self._main_time_threshold = main_time_threshold
        self._deadline_time_threshold = deadline_time_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._reservation_value: float = 0.0

        # Opponent modeling
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._value_frequencies: dict[int, dict] = {}
        self._best_opponent_utility: float = 0.0
        self._opponent_concession_detected: bool = False

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = self._outcome_space.min_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._value_frequencies = {}
        self._best_opponent_utility = 0.0
        self._opponent_concession_detected = False

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update opponent model with frequency analysis."""
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Check for opponent concession
        if len(self._opponent_bids) >= 3:
            recent = [u for _, u in self._opponent_bids[-5:]]
            if len(recent) >= 3 and recent[-1] > recent[0]:
                self._opponent_concession_detected = True

        # Track value frequencies
        for i, value in enumerate(bid):
            if i not in self._value_frequencies:
                self._value_frequencies[i] = {}
            self._value_frequencies[i][value] = (
                self._value_frequencies[i].get(value, 0) + 1
            )

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent's utility for a bid based on frequencies."""
        if not self._value_frequencies:
            return 0.5

        score = 0.0
        for i, value in enumerate(bid):
            if i in self._value_frequencies:
                freq = self._value_frequencies[i].get(value, 0)
                total = sum(self._value_frequencies[i].values())
                score += freq / total if total > 0 else 0

        return score / len(bid) if len(bid) > 0 else 0.5

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold with Neo's adaptive strategy."""
        # Neo uses Boulware-like concession
        e = self._e

        # Adjust based on opponent behavior
        if self._opponent_concession_detected:
            e *= 0.8  # Be more aggressive if opponent is conceding

        if time < self._early_time_threshold:
            # Early phase: very firm
            return self._max_utility * 0.95
        elif time < self._main_time_threshold:
            # Main phase: gradual concession
            progress = (time - self._early_time_threshold) / (
                self._main_time_threshold - self._early_time_threshold
            )
            f_t = math.pow(progress, 1 / e)
            target = self._max_utility * 0.95 - (self._max_utility * 0.95 - 0.6) * f_t
            return max(target, self._reservation_value)
        else:
            # End phase: accelerated concession
            progress = (time - self._main_time_threshold) / (
                1.0 - self._main_time_threshold
            )
            base = 0.6
            target = base - (base - self._min_utility - 0.1) * progress * 0.5
            return max(target, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid balancing own utility and opponent preference."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # If we have opponent model, try to find mutually beneficial bid
        if self._value_frequencies and time > 0.3:
            best_bid = None
            best_score = -1.0

            for bd in candidates[:50]:
                opp_util = self._estimate_opponent_utility(bd.bid)
                # Score combines our utility with opponent's estimated utility
                score = bd.utility * 0.7 + opp_util * 0.3
                if score > best_score:
                    best_score = score
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        return self._select_bid(state.relative_time)

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

        self._update_opponent_model(offer, offer_utility)

        threshold = self._compute_threshold(time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # AC_Next: accept if >= our next offer
        our_bid = self._select_bid(time)
        if our_bid is not None:
            our_utility = float(self.ufun(our_bid))
            if offer_utility >= our_utility:
                return ResponseType.ACCEPT_OFFER

        # End-game: accept if reasonable
        if (
            time > self._deadline_time_threshold
            and offer_utility >= self._best_opponent_utility
        ):
            if offer_utility > self._reservation_value:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
