"""GroupY from ANAC 2018.

This module implements GroupY, a Genius agent with a three-phase
strategy: an opening phase that repeatedly offers its best bid (only
accepting an opponent offer that matches the maximum utility), a middle
phase that offers bids chosen to maximize an aggregate frequency-based
opponent-model score above a time-dependent utility threshold, and a
final phase (very close to the deadline) that accepts more generous
offers and falls back to a fixed target bid.

References:
    - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
    - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
      Competition (ANAC 2018)." IJCAI 2019.
    - Genius framework: https://ii.tudelft.nl/genius/
    - Original package: agents.anac.y2018.groupy.GroupY
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["GroupY"]


class GroupY(SAONegotiator):
    """
    GroupY from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    GroupY opens with its maximum-utility bid for the first few turns
    (only accepting if the opponent already matches that maximum), then
    switches to offering the bid within a time-dependent utility range
    that maximizes an aggregate frequency-based opponent-model score,
    accepting offers that already beat that target. Near the deadline
    it relaxes to a fixed acceptance/offer threshold.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.groupy.GroupY

    **Offering Strategy:**
        First 3 turns: always offer the maximum-utility bid.
        Middle phase (t < 0.99): compute a time-dependent utility
        threshold (``_time_deal``) and offer the bid, among those above
        the threshold, that maximizes the summed expected utility under
        each opponent's frequency-based model (``_get_super_offer``).
        Deadline phase (t >= 0.99): offer the bid nearest utility 0.7.

    **Acceptance Strategy:**
        First 3 turns: accept only if the opponent's bid already matches
        the maximum utility.
        Middle phase: accept if the opponent's last bid utility is at
        least the current target-bid utility (and above the reservation
        value).
        Deadline phase: accept if opponent's last bid utility >= 0.65.

    **Opponent Modeling:**
        Per-opponent frequency table over (issue, value) pairs, weighted
        by a "time effect" factor (bids seen later in the negotiation
        count more) used to estimate the expected utility of a candidate
        bid to that opponent.

    Args:
        reservation_value: Minimum acceptable utility (default 0.0; uses
            ``ufun.reserved_value`` if available at negotiation start).
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
        reservation_value: float = 0.0,
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
        self._reservation_value_arg = reservation_value
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        self._max_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._reservation_value: float = 0.0

        # Per-opponent frequency model: {issue_index: {value: weight}}
        self._opponent_models: dict[str, dict[int, dict]] = {}

        self._n_actions = 0
        self._last_received_bid: Outcome | None = None
        self._last_received_utility: float = 0.0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        if self._initialized:
            return
        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        outcomes = self._outcome_space.outcomes
        if outcomes:
            self._max_bid = outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = self._outcome_space.min_utility

        reserved = getattr(self.ufun, "reserved_value", None)
        self._reservation_value = (
            reserved if reserved is not None else self._reservation_value_arg
        )

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._initialize()
        self._n_actions = 0
        self._last_received_bid = None
        self._last_received_utility = 0.0
        self._opponent_models = {}

    # ------------------------------------------------------------------
    # Opponent modeling
    # ------------------------------------------------------------------

    def _time_effect(self, time: float) -> float:
        return math.pow(max(1 - time, 0.0), math.e)

    def _update_model(self, source: str, bid: Outcome, time: float, normalize: float = 1.0) -> None:
        model = self._opponent_models.setdefault(source, {})
        weight = self._time_effect(time) / normalize if normalize else 0.0
        for i, value in enumerate(bid):
            issue_map = model.setdefault(i, {})
            issue_map[value] = issue_map.get(value, 0.0) + weight

    def _expected_utility(self, bid: Outcome, model: dict[int, dict]) -> float:
        total = 0.0
        for i, value in enumerate(bid):
            issue_map = model.get(i, {})
            total += issue_map.get(value, 0.0)
        return total

    # ------------------------------------------------------------------
    # Bidding strategy
    # ------------------------------------------------------------------

    def _time_deal(self, time: float) -> float:
        rem_time_ratio = 1 - time
        if rem_time_ratio > 0.5:
            min_utility = 0.85 + self._min_utility
        elif rem_time_ratio > 0.2:
            min_utility = 0.65 + self._min_utility
        else:
            min_utility = 0.45 + self._min_utility

        return min_utility + (self._max_utility - min_utility) * math.pow(
            max(rem_time_ratio, 0.0), 1 / math.e
        )

    def _get_super_offer(self, time: float) -> Outcome | None:
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return self._max_bid

        threshold = self._time_deal(time)
        candidates = self._outcome_space.get_bids_above(threshold)
        if not candidates:
            return self._max_bid

        best_bid = candidates[0].bid
        best_util = 0.0
        for bd in candidates:
            util = sum(
                self._expected_utility(bd.bid, model)
                for model in self._opponent_models.values()
            )
            if util > best_util:
                best_bid = bd.bid
                best_util = util
        return best_bid

    # ------------------------------------------------------------------
    # SAONegotiator interface
    # ------------------------------------------------------------------

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        if not self._initialized:
            self._initialize()

        self._n_actions += 1
        time = state.relative_time

        if self._n_actions < 4:
            return self._max_bid

        if time < 0.99:
            return self._get_super_offer(time)

        if self._outcome_space is not None and self._outcome_space.outcomes:
            near = self._outcome_space.get_bid_near_utility(0.7)
            return near.bid if near else self._max_bid
        return self._max_bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        self._last_received_bid = offer
        self._last_received_utility = float(self.ufun(offer))

        src = source or "opponent"
        time = state.relative_time
        self._update_model(src, offer, time, normalize=1.0)

        if self._n_actions < 4:
            if self._last_received_utility >= self._max_utility - 1e-9:
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

        if time < 0.99:
            target_bid = self._get_super_offer(time)
            target_util = float(self.ufun(target_bid)) if target_bid is not None else 1.0
            if (
                self._last_received_utility >= target_util
                and self._last_received_utility > self._reservation_value
            ):
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

        if self._last_received_utility >= 0.65:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER
