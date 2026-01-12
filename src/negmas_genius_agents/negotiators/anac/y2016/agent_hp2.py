"""AgentHP2 from ANAC 2016."""

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

__all__ = ["AgentHP2"]


class AgentHP2(SAONegotiator):
    """
    AgentHP2 negotiation agent from ANAC 2016.

    AgentHP2 is an evolved version of AgentHP from ANAC 2015, featuring
    optimized bid caching, multi-phase time-dependent concession, and
    opponent trend analysis for adaptive negotiation behavior.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2016.agenthp2.AgentHP2

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
    Uses a multi-phase time-dependent concession approach:

    - Phase 1 (t < 0.1): Maintains high aspiration at 98% of max utility
    - Phase 2 (0.1 <= t < 0.8): Boulware concession from 98% to 78% of max
      utility using configurable exponent parameter
    - Phase 3 (t >= 0.8): End-game concession toward reservation value,
      with rate adapted based on detected opponent concession trend

    Bids are selected from candidates above the current threshold with
    randomization among top options for unpredictability. Smart caching
    reduces computational overhead by reusing candidates when threshold
    changes are minimal.

    **Acceptance Strategy:**
    Accepts an offer if:

    - Offer utility meets or exceeds the current phase threshold
    - In end-game (t > 0.95): accepts if offer is above reservation value
      AND at least 99% of the best offer received from opponent

    **Opponent Modeling:**
    Tracks opponent bid utilities over time to detect concession trends:

    - Computes moving average of recent vs earlier opponent offers
    - Positive trend (opponent conceding) allows agent to maintain firmer
      stance in end-game
    - Trend information used to adjust Phase 3 concession target

    Args:
        e: Concession exponent for Boulware curve (default 0.12)
        min_utility: Minimum acceptable utility threshold (default 0.65)
        phase1_end: End time for phase 1 (default 0.1)
        phase2_end: End time for phase 2 (default 0.8)
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
        e: float = 0.12,
        min_utility: float = 0.65,
        phase1_end: float = 0.1,
        phase2_end: float = 0.8,
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
        self._e = e
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
        self._domain_min_utility: float = 0.0
        self._reservation_value: float = min_utility

        # Caching
        self._last_threshold: float = 1.0
        self._cached_candidates: list = []
        self._cache_valid: bool = False

        # Opponent tracking
        self._opponent_utilities: list[float] = []
        self._best_opponent_utility: float = 0.0
        self._best_opponent_bid: Outcome | None = None
        self._opponent_trend: float = 0.0

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
            self._domain_min_utility = self._outcome_space.min_utility

        # Set reservation value based on domain
        self._reservation_value = max(
            self._min_utility,
            self._domain_min_utility
            + 0.5 * (self._max_utility - self._domain_min_utility),
        )

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._last_threshold = 1.0
        self._cached_candidates = []
        self._cache_valid = False
        self._opponent_utilities = []
        self._best_opponent_utility = 0.0
        self._best_opponent_bid = None
        self._opponent_trend = 0.0

    def _update_opponent(self, utility: float, bid: Outcome) -> None:
        """Track opponent behavior with trend analysis."""
        self._opponent_utilities.append(utility)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        # Compute trend (positive = opponent conceding)
        if len(self._opponent_utilities) >= 5:
            recent = self._opponent_utilities[-3:]
            earlier = (
                self._opponent_utilities[-6:-3]
                if len(self._opponent_utilities) >= 6
                else self._opponent_utilities[:3]
            )
            recent_avg = sum(recent) / len(recent)
            earlier_avg = sum(earlier) / len(earlier)
            self._opponent_trend = recent_avg - earlier_avg

    def _compute_threshold(self, time: float) -> float:
        """Multi-phase threshold computation."""
        if time < self._phase1_end:
            # Phase 1: High aspiration
            return self._max_utility * 0.98
        elif time < self._phase2_end:
            # Phase 2: Boulware concession
            phase_time = (time - self._phase1_end) / (
                self._phase2_end - self._phase1_end
            )
            f_t = math.pow(phase_time, 1 / self._e) if self._e > 0 else phase_time
            start = self._max_utility * 0.98
            end = self._max_utility * 0.78
            return start - (start - end) * f_t
        else:
            # Phase 3: End-game with adaptation based on opponent
            phase_time = (time - self._phase2_end) / (1.0 - self._phase2_end)
            start = self._max_utility * 0.78

            # If opponent is conceding, we can be more patient
            if self._opponent_trend > 0.02:
                end = self._reservation_value + 0.1
            else:
                end = self._reservation_value

            return max(start - (start - end) * phase_time, self._reservation_value)

    def _get_candidates(self, threshold: float) -> list:
        """Get candidates with smart caching."""
        if self._outcome_space is None:
            return []

        # Use cache if threshold is close enough
        if self._cache_valid and abs(threshold - self._last_threshold) < 0.03:
            return self._cached_candidates

        candidates = self._outcome_space.get_bids_above(threshold)
        self._last_threshold = threshold
        self._cached_candidates = candidates
        self._cache_valid = True
        return candidates

    def _select_bid(self, time: float) -> Outcome | None:
        """Efficient bid selection with caching."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._get_candidates(threshold)

        if not candidates:
            # Relax threshold
            candidates = self._get_candidates(threshold * 0.9)

        if not candidates:
            return self._best_bid

        # Select from top candidates with some randomness
        n = min(8, len(candidates))
        return random.choice(candidates[:n]).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal efficiently."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        # Very early: best bid
        if time < self._early_time:
            return self._best_bid

        return self._select_bid(time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Fast response evaluation with trend awareness."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        self._update_opponent(offer_utility, offer)

        threshold = self._compute_threshold(time)

        # Accept if meets threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # End-game acceptance logic
        if time > self._deadline_time:
            # Accept if above reservation and at least as good as best received
            if offer_utility >= self._reservation_value:
                if offer_utility >= self._best_opponent_utility * 0.99:
                    return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
