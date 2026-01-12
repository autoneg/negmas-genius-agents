"""HardDealer from ANAC 2019."""

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

__all__ = ["HardDealer"]


class HardDealer(SAONegotiator):
    """
    HardDealer from ANAC 2019.

    HardDealer implements an aggressive hardball negotiation strategy
    that maintains very high utility demands for most of the negotiation,
    only conceding significantly in the final moments before deadline.
    This strategy aims to extract maximum value from cooperative opponents.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        .. code-block:: bibtex

            @inproceedings{baarslag2019tenth,
                title={The Tenth International Automated Negotiating Agents
                       Competition (ANAC 2019)},
                author={Baarslag, Tim and Fujita, Katsuhide and Gerding,
                        Enrico H and Hindriks, Koen and Ito, Takayuki and
                        Jennings, Nicholas R and others},
                booktitle={Proceedings of the International Joint Conference
                           on Autonomous Agents and Multiagent Systems (AAMAS)},
                year={2019}
            }

        Original Genius class: ``agents.anac.y2019.harddealer.HardDealer``

    **Offering Strategy:**
        - Three-phase threshold system:
          - t < 0.9: Fixed high threshold (0.9)
          - 0.9 <= t < 0.95: Quick drop to midpoint (50% of range)
          - t >= 0.95: Final drop to deadline threshold (0.6)
        - Randomly selects from bids at or above current threshold
        - First offer is always the best available bid
        - No opponent modeling for bid selection - purely threshold-based

    **Acceptance Strategy:**
        - Accepts offers meeting or exceeding the current threshold
        - Very near deadline (t >= 0.99): Accepts offers above deadline
          threshold even if below current target
        - Binary decision based on threshold comparison

    **Opponent Modeling:**
        - Minimal modeling - only tracks offer count and best received
        - Does not adapt strategy based on opponent behavior
        - Relies on time pressure to force opponent concession
        - High-risk strategy that may fail against equally tough opponents

    Args:
        high_threshold: Utility threshold for most of negotiation (default 0.9)
        deadline_threshold: Lower threshold near deadline (default 0.6)
        phase1_time: Time threshold for phase 1 (default 0.9)
        phase2_time: Time threshold for phase 2 (default 0.95)
        final_accept_time: Time threshold for deadline acceptance (default 0.99)
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
        high_threshold: float = 0.9,
        deadline_threshold: float = 0.6,
        phase1_time: float = 0.9,
        phase2_time: float = 0.95,
        final_accept_time: float = 0.99,
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
        self._high_threshold = high_threshold
        self._deadline_threshold = deadline_threshold
        self._phase1_time = phase1_time
        self._phase2_time = phase2_time
        self._final_accept_time = final_accept_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._opponent_offers_count: int = 0
        self._best_received_utility: float = 0.0

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
        self._opponent_offers_count = 0
        self._best_received_utility = 0.0

    def _get_target_utility(self, time: float) -> float:
        """Get target utility - high until near deadline."""
        if time < self._phase1_time:
            return self._high_threshold
        elif time < self._phase2_time:
            # Quick drop
            progress = (time - self._phase1_time) / (
                self._phase2_time - self._phase1_time
            )
            return (
                self._high_threshold
                - (self._high_threshold - self._deadline_threshold) * progress * 0.5
            )
        else:
            # Final drop
            progress = (time - self._phase2_time) / (1.0 - self._phase2_time)
            mid = (
                self._high_threshold
                - (self._high_threshold - self._deadline_threshold) * 0.5
            )
            return mid - (mid - self._deadline_threshold) * progress

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid at or above target utility."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return self._best_bid

        target = self._get_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            bid_detail = self._outcome_space.get_bid_near_utility(target)
            return bid_detail.bid if bid_detail else self._best_bid

        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        if self._opponent_offers_count == 0:
            return self._best_bid

        time = state.relative_time
        return self._select_bid(time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        self._opponent_offers_count += 1

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        time = state.relative_time

        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility

        target = self._get_target_utility(time)

        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Very near deadline, accept reasonable offers
        if (
            time >= self._final_accept_time
            and offer_utility >= self._deadline_threshold
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
