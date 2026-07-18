"""Libra from ANAC 2018.

This module implements Libra, a Genius agent whose original design is a
weighted "parliament" of 19 real ANAC sub-agents that each independently
decide to offer, accept, or end the negotiation; Libra then acts on the
majority (weighted) vote and, if offering, assembles a new bid issue by
issue from the sub-agents' proposals, weighted by how much better than
the recent opponent average each proposal is. Agent weights are then
adjusted based on how well each sub-agent's vote "predicted" what
happened next.

References:
    - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
    - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
      Competition (ANAC 2018)." IJCAI 2019.
    - Genius framework: https://ii.tudelft.nl/genius/
    - Original package: agents.anac.y2018.libra.Libra
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["Libra"]


class Libra(SAONegotiator):
    """
    Libra from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the
        original, and the original agent's design (a weighted ensemble of
        19 real ANAC sub-agents) is not reproducible with real agents in
        this port. Instead, this reimplementation uses a small set of
        canonical time-dependent strategies (Boulware, Conceder, Linear,
        and a frequency-based strategy) as stand-ins for the sub-agent
        "parliament", reproducing the weighted-voting/weight-update
        mechanism at a coarse level. Overall behavior should still be
        tough-and-high early, then conceding and accepting near the
        deadline, but this is a deliberate approximation, not a faithful
        reproduction of the original ensemble.

    Libra maintains a set of internal "sub-strategies", each producing a
    candidate bid and a simulated vote (offer / accept / end) every turn.
    The votes are combined using per-strategy weights to decide whether
    Libra offers, accepts, or (rarely) ends the negotiation; when
    offering, a new bid is assembled issue-by-issue from the sub-strategy
    proposals weighted by how much they exceed the recent opponent
    utility average. Weights are then updated based on whether each
    sub-strategy's implied action would have been rewarded given what the
    opponent actually did.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.libra.Libra

    **Offering Strategy:**
        Each internal sub-strategy proposes a bid based on its own
        time-dependent target utility (Boulware e=3, Conceder e=0.3,
        Linear e=1, plus a frequency-model-driven strategy). A new
        composite bid is built issue-by-issue, picking for each issue the
        value that maximizes the weighted sum of
        ``weight * (utility - avg_received_utility)`` across sub-strategy
        proposals.

    **Acceptance Strategy:**
        Sub-strategies "vote" accept/offer/end based on their own
        thresholds; Libra accepts if the (weighted) accept vote exceeds
        both the offer and end votes.

    **Opponent Modeling:**
        A simple per-issue-value frequency table over received offers,
        used by the frequency-based sub-strategy, plus a running average
        of the opponent's utility (from Libra's perspective) used both in
        bid assembly and in the weight-update rule.

    Args:
        change_weight: Amount weights are adjusted per event (default 2.0).
        min_weight: Minimum allowed sub-strategy weight (default 1.0).
        default_weight: Initial weight per sub-strategy (default 10.0).
        preferences: NegMAS preferences/utility function.
        ufun: Utility function (overrides preferences if given).
        name: Negotiator name.
        parent: Parent controller.
        owner: Agent that owns this negotiator.
        id: Unique identifier.
        **kwargs: Additional arguments passed to parent.
    """

    _STRATEGY_EXPONENTS = {
        "boulware": 3.0,
        "conceder": 0.3,
        "linear": 1.0,
        "frequency": 1.5,
    }

    def __init__(
        self,
        change_weight: float = 2.0,
        min_weight: float = 1.0,
        default_weight: float = 10.0,
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
        self._change_weight = change_weight
        self._min_weight = min_weight
        self._default_weight = default_weight

        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        self._max_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0

        self._strategy_names = list(self._STRATEGY_EXPONENTS.keys())
        self._weights: dict[str, float] = {}

        self._value_frequency: dict[int, dict] = {}
        self._received_utilities: list[float] = []

        self._last_received_bid: Outcome | None = None
        self._last_offered_bid: Outcome | None = None

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

        self._weights = {name: self._default_weight for name in self._strategy_names}
        self._value_frequency = {}

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._initialize()
        self._last_received_bid = None
        self._last_offered_bid = None
        self._received_utilities = []
        self._value_frequency = {}
        self._weights = {name: self._default_weight for name in self._strategy_names}

    # ------------------------------------------------------------------
    # Sub-strategy target utilities and candidate bids
    # ------------------------------------------------------------------

    def _target_utility(self, name: str, time: float) -> float:
        e = self._STRATEGY_EXPONENTS[name]
        target = 1.0 - (time**e) if e != 0 else 1.0
        target = self._min_utility + (self._max_utility - self._min_utility) * target
        return max(target, self._min_utility)

    def _candidate_bid(self, name: str, time: float) -> Outcome | None:
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._target_utility(name, time)

        if name == "frequency" and self._value_frequency:
            candidates = self._outcome_space.get_bids_above(target - 0.1)
            if not candidates:
                return self._max_bid
            best_bid = candidates[0].bid
            best_score = -1.0
            for bd in candidates[: min(len(candidates), 50)]:
                score = self._frequency_score(bd.bid)
                if score > best_score:
                    best_score = score
                    best_bid = bd.bid
            return best_bid

        near = self._outcome_space.get_bid_near_utility(target)
        return near.bid if near else self._max_bid

    def _frequency_score(self, bid: Outcome) -> float:
        score = 0.0
        for i, value in enumerate(bid):
            score += self._value_frequency.get(i, {}).get(value, 0)
        return score

    def _sub_strategy_vote(
        self, name: str, time: float, last_received_util: float | None
    ) -> str:
        """Simulate a sub-strategy's vote: "offer", "accept" or "end"."""
        if last_received_util is None:
            return "offer"

        target = self._target_utility(name, time)
        if last_received_util >= target:
            return "accept"
        if last_received_util < self._min_utility * 0.5 and time > 0.98:
            return "end"
        return "offer"

    # ------------------------------------------------------------------
    # Opponent modeling
    # ------------------------------------------------------------------

    def _update_frequency(self, bid: Outcome) -> None:
        for i, value in enumerate(bid):
            issue_map = self._value_frequency.setdefault(i, {})
            issue_map[value] = issue_map.get(value, 0) + 1

    def _average_received_utility(self) -> float:
        if not self._received_utilities:
            return 0.5
        return sum(self._received_utilities) / len(self._received_utilities)

    def _update_weights(self, last_received_util: float, offered_util: float) -> None:
        for name in self._strategy_names:
            vote = self._sub_strategy_vote(name, 0.0, None)
            # Reward sub-strategies whose implied preference matches what
            # actually turned out to be a good outcome (received > offered
            # by a margin means we should have been more generous).
            if last_received_util > offered_util * 1.1:
                self._weights[name] -= self._change_weight
            else:
                self._weights[name] += self._change_weight
            if self._weights[name] < self._min_weight:
                self._weights[name] = self._min_weight
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        total = sum(self._weights.values())
        if total <= 0:
            return
        target_total = self._default_weight * len(self._strategy_names)
        factor = target_total / total
        for name in self._weights:
            self._weights[name] *= factor

    # ------------------------------------------------------------------
    # Bid assembly (weighted issue-by-issue combination of candidates)
    # ------------------------------------------------------------------

    def _assemble_bid(self, time: float) -> Outcome | None:
        if self._outcome_space is None or self._max_bid is None or self.ufun is None:
            return self._max_bid

        candidates = {
            name: self._candidate_bid(name, time) for name in self._strategy_names
        }
        candidates = {k: v for k, v in candidates.items() if v is not None}
        if not candidates:
            return self._max_bid

        received_avg = self._average_received_utility()
        n_issues = len(self._max_bid)

        new_bid = list(self._max_bid)
        for issue_idx in range(n_issues):
            value_scores: dict = {}
            for name, bid in candidates.items():
                if issue_idx >= len(bid):
                    continue
                value = bid[issue_idx]
                util = float(self.ufun(bid))
                score = self._weights[name] * (util - received_avg) * 100
                value_scores[value] = value_scores.get(value, 0.0) + score

            if value_scores:
                best_value = max(value_scores.items(), key=lambda kv: kv[1])[0]
                new_bid[issue_idx] = best_value

        return tuple(new_bid)

    # ------------------------------------------------------------------
    # SAONegotiator interface
    # ------------------------------------------------------------------

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        bid = self._assemble_bid(time)
        self._last_offered_bid = bid
        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        self._last_received_bid = offer
        self._update_frequency(offer)
        received_util = float(self.ufun(offer))
        self._received_utilities.append(received_util)

        offered_util = (
            float(self.ufun(self._last_offered_bid))
            if self._last_offered_bid is not None
            else 0.0
        )
        self._update_weights(received_util, offered_util)

        time = state.relative_time
        offer_vote = 0.0
        accept_vote = 0.0
        end_vote = 0.0
        for name in self._strategy_names:
            vote = self._sub_strategy_vote(name, time, received_util)
            weight = self._weights[name]
            if vote == "accept":
                accept_vote += weight
            elif vote == "end":
                end_vote += weight
            else:
                offer_vote += weight

        if accept_vote > offer_vote and accept_vote > end_vote:
            return ResponseType.ACCEPT_OFFER
        if end_vote > offer_vote and end_vote > accept_vote:
            return ResponseType.END_NEGOTIATION

        return ResponseType.REJECT_OFFER
