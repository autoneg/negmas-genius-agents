"""AgentLight from ANAC 2016."""

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

__all__ = ["AgentLight"]


class AgentLight(SAONegotiator):
    """
    AgentLight negotiation agent from ANAC 2016.

    AgentLight is a lightweight and efficient negotiation agent designed for
    minimal computational overhead while maintaining effective bidding and
    acceptance strategies.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2016.agentlight.AgentLight

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
    Employs a standard Boulware time-dependent tactic:

    - Target utility computed as: max_util - (max_util - reservation) * t^(1/e)
    - Low exponent (e=0.2) creates conceding behavior that starts slow
      and accelerates toward deadline
    - Bids selected randomly from candidates above threshold
    - Falls back to relaxed threshold (90%) if no candidates found

    Early negotiation (t < 0.02) always offers the best available bid.

    **Acceptance Strategy:**
    Simple threshold-based acceptance:

    - Accepts if offer utility meets or exceeds current Boulware threshold
    - End-game fallback (t > 0.95): accepts any offer above reservation value

    **Opponent Modeling:**
    Minimal opponent modeling for efficiency:

    - Tracks only the best utility received from opponent
    - No frequency analysis or preference estimation
    - Decisions based purely on own utility function

    Args:
        e: Concession exponent for Boulware curve (default 0.2)
        min_utility: Minimum acceptable utility threshold (default 0.55)
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
        e: float = 0.2,
        min_utility: float = 0.55,
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
        self._early_time = early_time
        self._deadline_time = deadline_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Minimal state
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._reservation_value: float = min_utility

        # Simple tracking
        self._best_received_utility: float = 0.0
        self._last_threshold: float = 1.0

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

        self._best_received_utility = 0.0
        self._last_threshold = 1.0

    def _get_threshold(self, time: float) -> float:
        """Simple Boulware threshold."""
        f_t = math.pow(time, 1 / self._e) if self._e > 0 else time
        threshold = (
            self._max_utility - (self._max_utility - self._reservation_value) * f_t
        )
        return max(threshold, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Fast bid selection."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._get_threshold(time)
        self._last_threshold = threshold

        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            return self._best_bid

        # Simple random selection from top
        n = min(5, len(candidates))
        return random.choice(candidates[:n]).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal quickly."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        if time < self._early_time:
            return self._best_bid

        return self._select_bid(time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Quick response evaluation."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        # Track best received
        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility

        threshold = self._get_threshold(time)

        # Accept if meets threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # End-game
        if time > self._deadline_time and offer_utility >= self._reservation_value:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
