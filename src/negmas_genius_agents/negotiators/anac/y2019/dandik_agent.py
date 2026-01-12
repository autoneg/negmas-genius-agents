"""DandikAgent from ANAC 2019."""

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

__all__ = ["DandikAgent"]


class DandikAgent(SAONegotiator):
    """
    DandikAgent from ANAC 2019.

    DandikAgent uses a Boulware-style concession strategy that maintains
    high demands early in the negotiation and concedes more rapidly as
    the deadline approaches. This strategy aims to extract maximum value
    while avoiding failed negotiations.

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

        Original Genius class: ``agents.anac.y2019.dandikagent.dandikAgent``

    **Offering Strategy:**
        - Boulware concession curve: target = max - (max - min) * t^(1/e)
        - With e=0.2, concedes slowly early and rapidly near deadline
        - Searches for bids within +/- 0.05 of target utility
        - After 5 opponent offers, selects bids maximizing estimated
          opponent utility from candidates near target
        - Before sufficient data, randomly selects from candidate bids

    **Acceptance Strategy:**
        - Accepts offers meeting or exceeding the Boulware target
        - Near deadline (t >= 0.98): Accepts offers above minimum utility
        - Very near deadline (t >= 0.99): Accepts 95% of best received offer
        - Tracks best received utility for fallback acceptance

    **Opponent Modeling:**
        - Frequency-based model tracking issue value occurrences
        - Estimates opponent utility as sum of normalized frequencies
        - For each value: score = frequency / max_frequency_for_issue
        - Used to select opponent-friendly bids from near-target candidates

    Args:
        e: Concession exponent (default 0.2, Boulware-like)
        min_utility: Minimum acceptable utility (default 0.55)
        near_deadline_time: Time threshold for near deadline (default 0.98)
        final_deadline_time: Time threshold for final deadline (default 0.99)
        final_best_ratio: Ratio of best received utility for final deadline acceptance (default 0.95)
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
        near_deadline_time: float = 0.98,
        final_deadline_time: float = 0.99,
        final_best_ratio: float = 0.95,
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
        self._near_deadline_time = near_deadline_time
        self._final_deadline_time = final_deadline_time
        self._final_best_ratio = final_best_ratio
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_value_freq: dict[int, dict[str, int]] = {}
        self._opponent_offers_count: int = 0

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
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
        self._opponent_value_freq = {}
        self._opponent_offers_count = 0
        self._best_received_utility = 0.0

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update frequency-based opponent model."""
        if bid is None:
            return

        self._opponent_offers_count += 1

        for i, value in enumerate(bid):
            if i not in self._opponent_value_freq:
                self._opponent_value_freq[i] = {}
            value_str = str(value)
            if value_str not in self._opponent_value_freq[i]:
                self._opponent_value_freq[i][value_str] = 0
            self._opponent_value_freq[i][value_str] += 1

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility based on frequency model."""
        if bid is None or self._opponent_offers_count == 0:
            return 0.5

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            value_str = str(value)
            if i in self._opponent_value_freq:
                freq = self._opponent_value_freq[i].get(value_str, 0)
                max_freq = (
                    max(self._opponent_value_freq[i].values())
                    if self._opponent_value_freq[i]
                    else 1
                )
                total_score += freq / max_freq if max_freq > 0 else 0

        return total_score / num_issues if num_issues > 0 else 0.5

    def _get_target_utility(self, time: float) -> float:
        """Get target utility based on Boulware curve."""
        # Boulware: slow concession early, faster late
        target = self._max_utility - (self._max_utility - self._min_utility) * (
            time ** (1.0 / self._e)
        )
        return max(target, self._min_utility)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid based on target and opponent model."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return self._best_bid

        target = self._get_target_utility(time)
        candidates = self._outcome_space.get_bids_in_range(target - 0.05, target + 0.05)

        if not candidates:
            bid_detail = self._outcome_space.get_bid_near_utility(target)
            return bid_detail.bid if bid_detail else self._best_bid

        if len(candidates) == 1:
            return candidates[0].bid

        # Select bid with highest opponent utility estimate
        if self._opponent_offers_count >= 5:
            best_opp_util = -1.0
            best_bid = candidates[0].bid
            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                if opp_util > best_opp_util:
                    best_opp_util = opp_util
                    best_bid = bd.bid
            return best_bid

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

        self._update_opponent_model(offer)

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        time = state.relative_time

        if offer_utility > self._best_received_utility:
            self._best_received_utility = offer_utility

        target = self._get_target_utility(time)

        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        if time >= self._near_deadline_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        if (
            time >= self._final_deadline_time
            and offer_utility >= self._best_received_utility * self._final_best_ratio
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
