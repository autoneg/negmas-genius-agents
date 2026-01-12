"""FSEGA2019 from ANAC 2019."""

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

__all__ = ["FSEGA2019"]


class FSEGA2019(SAONegotiator):
    """
    FSEGA2019 from ANAC 2019 - Nash-based category 2nd place.

    FSEGA2019 is an evolution of the FSEGA agent family with enhanced
    opponent modeling and adaptive concession. The agent monitors
    opponent concession patterns and adjusts its own strategy to
    reach mutually beneficial outcomes.

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

            @inproceedings{luca2010agentfsega,
                title={Agent FSEGA - A Negotiating Agent from the West
                       University of Timisoara},
                author={Luca, Liviu and Ciortea, Ecaterina Madalina and
                        Szilagyi, Attila},
                booktitle={ANAC 2010},
                year={2010}
            }

        Original Genius class: ``agents.anac.y2019.fsega2019.agent.FSEGA2019``

    **Offering Strategy:**
        - Quadratic base concession: target = initial - (initial - min) * t^2
        - Adaptive adjustment based on opponent concession rate:
          - If opponent conceding (delta > 0.05): +0.05 adjustment (stay firm)
          - If opponent hardening (delta < -0.05): -0.05 adjustment (concede more)
        - Searches for bids within +/- 0.05 of adjusted target
        - After 5 offers, selects bids maximizing estimated opponent utility
        - Before sufficient data, randomly selects from candidates

    **Acceptance Strategy:**
        - Accepts offers meeting or exceeding the adaptive target
        - Near deadline (t >= 0.98): Accepts offers above minimum target
        - Very near deadline (t >= 0.99): Accepts 98% of best received offer
        - Tracks history of opponent offers for comparison

    **Opponent Modeling:**
        - Frequency-based model for issue value preferences
        - Tracks utility (for self) of all opponent offers
        - Estimates opponent concession by comparing early vs recent offers:
          concession = average(last_3_offers) - average(first_3_offers)
        - Positive concession means opponent is giving better offers over time
        - Used to adjust own concession rate adaptively

    Args:
        initial_target: Initial target utility (default 0.95)
        min_target: Minimum acceptable utility (default 0.6)
        concession_rate: Base concession rate (default 0.1)
        near_deadline_time: Time threshold for near deadline (default 0.98)
        final_deadline_time: Time threshold for final deadline (default 0.99)
        final_best_ratio: Ratio of best received utility for final deadline acceptance (default 0.98)
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
        initial_target: float = 0.95,
        min_target: float = 0.6,
        concession_rate: float = 0.1,
        near_deadline_time: float = 0.98,
        final_deadline_time: float = 0.99,
        final_best_ratio: float = 0.98,
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
        self._initial_target = initial_target
        self._min_target = min_target
        self._concession_rate = concession_rate
        self._near_deadline_time = near_deadline_time
        self._final_deadline_time = final_deadline_time
        self._final_best_ratio = final_best_ratio

        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_value_freq: dict[int, dict[str, int]] = {}
        self._opponent_offers: list[float] = []

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._current_target: float = initial_target

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
            self._min_utility = self._outcome_space.min_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._opponent_value_freq = {}
        self._opponent_offers = []
        self._current_target = self._initial_target

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update frequency-based opponent model."""
        if bid is None:
            return

        # Track utility of opponent offers (for us)
        if self.ufun is not None:
            util = float(self.ufun(bid))
            self._opponent_offers.append(util)

        for i, value in enumerate(bid):
            if i not in self._opponent_value_freq:
                self._opponent_value_freq[i] = {}
            value_str = str(value)
            if value_str not in self._opponent_value_freq[i]:
                self._opponent_value_freq[i][value_str] = 0
            self._opponent_value_freq[i][value_str] += 1

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility based on frequency model."""
        if bid is None or not self._opponent_offers:
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

    def _estimate_opponent_concession(self) -> float:
        """Estimate how much the opponent is conceding."""
        if len(self._opponent_offers) < 3:
            return 0.0

        # Compare recent offers to early offers
        early_avg = sum(self._opponent_offers[:3]) / 3
        recent_avg = sum(self._opponent_offers[-3:]) / 3

        # Positive means opponent is giving us better offers
        return recent_avg - early_avg

    def _get_target_utility(self, time: float) -> float:
        """Get adaptive target utility based on time and opponent behavior."""
        # Base concession curve
        base_target = self._initial_target - (
            self._initial_target - self._min_target
        ) * (time**2)

        # Adjust based on opponent concession
        opp_concession = self._estimate_opponent_concession()

        if opp_concession > 0.05:
            # Opponent is conceding, we can stay firm
            adjustment = 0.05
        elif opp_concession < -0.05:
            # Opponent is hardening, we need to concede more
            adjustment = -0.05
        else:
            adjustment = 0.0

        target = base_target + adjustment
        return max(min(target, self._max_utility), self._min_target)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid based on target utility and opponent model."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return self._best_bid

        target = self._get_target_utility(time)

        # Get candidates near target
        candidates = self._outcome_space.get_bids_in_range(target - 0.05, target + 0.05)

        if not candidates:
            bid_detail = self._outcome_space.get_bid_near_utility(target)
            return bid_detail.bid if bid_detail else self._best_bid

        if len(candidates) == 1:
            return candidates[0].bid

        # Select bid with highest estimated opponent utility
        if len(self._opponent_offers) >= 5:
            best_opp_util = -1.0
            best_bid = candidates[0].bid
            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                if opp_util > best_opp_util:
                    best_opp_util = opp_util
                    best_bid = bd.bid
            return best_bid
        else:
            return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        # First move: best bid
        if not self._opponent_offers:
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

        # Update opponent model
        self._update_opponent_model(offer)

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        time = state.relative_time

        # Get target utility
        target = self._get_target_utility(time)

        # Accept if above target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Near deadline
        if time >= self._near_deadline_time:
            # Accept anything above minimum
            if offer_utility >= self._min_target:
                return ResponseType.ACCEPT_OFFER

        if time >= self._final_deadline_time:
            # Accept best offer we've seen
            if (
                self._opponent_offers
                and offer_utility >= max(self._opponent_offers) * self._final_best_ratio
            ):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
