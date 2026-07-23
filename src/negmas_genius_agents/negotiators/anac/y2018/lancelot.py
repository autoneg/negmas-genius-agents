"""Lancelot from ANAC 2018.

This module implements Lancelot, a Genius agent that uses a parabolic
time-dependent acceptance/offering threshold combined with a simple
running-average opponent evaluation and a frequency-based bid-repair
routine that swaps in the opponent's frequently-offered values whenever
doing so does not reduce the agent's own utility.

References:
    - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
    - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
      Competition (ANAC 2018)." IJCAI 2019.
    - Genius framework: https://ii.tudelft.nl/genius/
    - Original package: agents.anac.y2018.lancelot.Lancelot
"""

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

__all__ = ["Lancelot"]


class Lancelot(SAONegotiator):
    """
    Lancelot from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Lancelot combines a parabolic acceptance/offering utility threshold
    (centered on an "opponent evaluation" value) with a simple bid-table
    based repair heuristic: it starts from a random bid above the current
    threshold and swaps in the opponent's most frequently offered values
    issue by issue whenever the swap does not decrease its own utility.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.lancelot.Lancelot

    **Offering Strategy:**
        Before t=0.2: offers a random bid above utility ``1 - t - 0.1``.
        Between t=0.2 and t=0.98: offers a random bid above a parabolic
        threshold (see ``_get_util_threshold``), then attempts to repair
        each issue with the opponent's most frequent value if it does not
        reduce utility. After t=0.98: offers a random bid above a fixed
        0.90 threshold, with the same repair step.

    **Acceptance Strategy:**
        Accepts if the offer's utility exceeds the same parabolic threshold
        used for offering, which is parameterized by an "opponent
        evaluation" value: the running mean of the agent's own utility for
        bids the opponent has sent, plus one standard deviation.

    **Opponent Modeling:**
        Tracks, per issue, how often each value has been offered by the
        opponent (used only for the bid-repair heuristic). Separately
        tracks the running mean/std of the agent's own utility for
        opponent bids as an "opponent evaluation" signal that shifts the
        acceptance/offering threshold.

    Args:
        opening_phase_threshold: Relative time before which the opening
            offering strategy applies (default 0.2).
        opening_threshold_offset: Utility offset subtracted from ``1 - time``
            to set the opening-phase minimum utility (default 0.1).
        late_phase_threshold: Relative time before which the parabolic
            threshold applies and at/after which the fixed offer threshold
            applies (default 0.98).
        sep_point: Separation point between the two parabolic branches of the
            threshold curve (default 0.7).
        deadline_phase_threshold: Relative time at/after which the deadline
            branch of the parabolic threshold applies (default 0.99).
        deadline_threshold_factor: Factor multiplying ``max_util * time`` in
            the deadline branch of the threshold curve (default 0.75).
        offer_threshold: Fixed minimum utility used for offering in the late
            phase (default 0.90).
        max_util_backoff: Utility subtracted from the maximum utility when it
            is the only value available, to widen the candidate search
            (default 0.1).
        repair_random_range: Upper bound (inclusive) of the random integer
            used to decide whether a bid issue is repaired (default 4).
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
        opening_phase_threshold: float = 0.2,
        opening_threshold_offset: float = 0.1,
        late_phase_threshold: float = 0.98,
        sep_point: float = 0.7,
        deadline_phase_threshold: float = 0.99,
        deadline_threshold_factor: float = 0.75,
        offer_threshold: float = 0.90,
        max_util_backoff: float = 0.1,
        repair_random_range: int = 4,
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
        self._opening_phase_threshold = opening_phase_threshold
        self._opening_threshold_offset = opening_threshold_offset
        self._late_phase_threshold = late_phase_threshold
        self._sep_point = sep_point
        self._deadline_phase_threshold = deadline_phase_threshold
        self._deadline_threshold_factor = deadline_threshold_factor
        self._offer_threshold = offer_threshold
        self._max_util_backoff = max_util_backoff
        self._repair_random_range = repair_random_range
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        self._max_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._n_issues: int = 0

        # Opponent bid-value frequency table: issue index -> {value: count}
        self._bid_table: list[dict] = []

        # Opponent utility evaluation running stats
        self._opponent_util_sum: float = 0.0
        self._opponent_util_count: int = 0
        self._opponent_util_values: list[float] = []
        self._opponent_eval: float = 0.0

        self._last_received_bid: Outcome | None = None

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
            self._n_issues = len(self._max_bid)
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = self._outcome_space.min_utility
        self._bid_table = [dict() for _ in range(self._n_issues)]

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._initialize()
        self._last_received_bid = None
        self._opponent_util_sum = 0.0
        self._opponent_util_count = 0
        self._opponent_util_values = []
        self._opponent_eval = 0.0

    # ------------------------------------------------------------------
    # Opponent modeling
    # ------------------------------------------------------------------

    def _update_bid_table(self, bid: Outcome) -> None:
        for i, value in enumerate(bid):
            if i >= len(self._bid_table):
                continue
            self._bid_table[i][value] = self._bid_table[i].get(value, 0) + 1

    def _evaluate_opponent(self, bid: Outcome) -> float:
        """Update and return the running "opponent evaluation" signal.

        This mirrors the Java agent's use of the agent's *own* utility of
        the bids the opponent sends as a signal of how generous the
        opponent is being (average + standard deviation).
        """
        if self.ufun is None:
            return 0.0

        my_util = float(self.ufun(bid))
        self._opponent_util_count += 1
        self._opponent_util_sum += my_util
        self._opponent_util_values.append(my_util)

        avg = self._opponent_util_sum / self._opponent_util_count
        variance = sum((v - avg) ** 2 for v in self._opponent_util_values) / (
            self._opponent_util_count
        )
        std_dev = math.sqrt(variance)
        return avg + std_dev

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------

    def _get_util_threshold(self, time: float, opponent_value: float) -> float:
        """Parabolic threshold centered near ``opponent_value``."""
        max_util = self._max_utility
        sep_point = self._sep_point
        if time < sep_point:
            threshold = (max_util - opponent_value) / (sep_point**2) * (
                time - sep_point
            ) ** 2 + opponent_value
        elif time < self._deadline_phase_threshold:
            threshold = (opponent_value - max_util) / ((1 - sep_point) ** 2) * (
                time - 1
            ) ** 2 + max_util
        else:
            threshold = max_util * self._deadline_threshold_factor * time
        return threshold

    def _get_util_threshold_for_offer(self) -> float:
        return self._offer_threshold

    # ------------------------------------------------------------------
    # Bidding strategy
    # ------------------------------------------------------------------

    def _get_random_bid_above(self, min_util: float) -> Outcome | None:
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None
        min_util = min(min_util, self._max_utility)
        if min_util == self._max_utility:
            min_util -= self._max_util_backoff
        candidates = self._outcome_space.get_bids_above(min_util)
        if not candidates:
            return self._max_bid
        return random.choice(candidates).bid

    def _offer_positive_bid(self, min_util: float) -> Outcome | None:
        """Random bid above ``min_util``, repaired with frequent opponent values."""
        offer_bid = self._get_random_bid_above(min_util)
        if offer_bid is None or self.ufun is None:
            return offer_bid

        offer_bid = list(offer_bid)
        for i in range(min(self._n_issues, len(offer_bid))):
            if random.randint(0, self._repair_random_range) % 2 != 0:
                continue
            table = self._bid_table[i] if i < len(self._bid_table) else {}
            if not table:
                continue
            # Sort values by descending frequency (most frequent first).
            sorted_values = sorted(table.items(), key=lambda kv: kv[1], reverse=True)
            origin_util = float(self.ufun(tuple(offer_bid)))
            for value, _count in sorted_values:
                candidate = list(offer_bid)
                candidate[i] = value
                candidate_util = float(self.ufun(tuple(candidate)))
                if candidate_util - origin_util > 0:
                    offer_bid = candidate
                    break
        return tuple(offer_bid)

    # ------------------------------------------------------------------
    # SAONegotiator interface
    # ------------------------------------------------------------------

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        if time < self._opening_phase_threshold:
            return self._get_random_bid_above(1 - time - self._opening_threshold_offset)
        elif time < self._late_phase_threshold:
            min_util = self._get_util_threshold(time, self._opponent_eval)
            return self._offer_positive_bid(min_util)
        else:
            min_util = self._get_util_threshold_for_offer()
            return self._offer_positive_bid(min_util)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        self._last_received_bid = offer
        self._update_bid_table(offer)
        self._opponent_eval = self._evaluate_opponent(offer)

        time = state.relative_time
        util = float(self.ufun(offer))
        threshold = self._get_util_threshold(time, self._opponent_eval)

        if util > threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
