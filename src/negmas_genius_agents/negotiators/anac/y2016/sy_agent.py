"""SYAgent from ANAC 2016."""

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

__all__ = ["SYAgent"]


class SYAgent(SAONegotiator):
    """
    SYAgent negotiation agent from ANAC 2016.

    SYAgent uses a fixed, purely time-based acceptance/offering threshold
    curve together with a frequency-based "bid improvement" step that nudges
    a target bid towards the most frequently observed opponent values.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

        In the original Java implementation the threshold formula is guarded by
        several branches that depend on the reservation value and discount
        factor of the domain, but the final lines of ``getThreshold`` in the
        Java code unconditionally overwrite ``threshold`` with a purely
        time-based formula, making those branches unreachable. This port
        reproduces the effective (reachable) behavior: a threshold curve that
        starts at 1.0, dips to about 0.805 around relative time 0.6 and then
        rises again towards about 0.975 as the deadline approaches.

    Original Java class: agents.anac.y2016.syagent.SYAgent

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
    The threshold at time ``t`` is computed as:

    - If ``t <= 0.6``: ``threshold = 1.0 + ln(1 - t) / 4.7``
    - Else: ``threshold = min_threshold - ln(1.6 - t) / 3.0`` where
      ``min_threshold = 1.0 + ln(0.4) / 4.7 ~= 0.805``

    A bid is selected by finding the outcome with the lowest utility that is
    still above (or equal to) the threshold (mimicking a "concede down from
    the maximum utility bid until the threshold is reached" search), then
    "improved" by trying, issue by issue (in random order), to replace the
    issue's value with the most frequently observed opponent value as long
    as doing so does not lower the bid's utility.

    **Acceptance Strategy:**
    Accepts an offer whenever its utility is greater than or equal to the
    current threshold.

    **Opponent Modeling:**
    Tracks, for every issue, how often each value has been seen in offers
    made by the opponent (frequency counting), used only for the bid
    "improvement" step described above.

    Args:
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
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False
        self._max_bid: Outcome | None = None

        # Opponent frequency model: issue index -> value -> count
        self._value_frequency: dict[int, dict[str, int]] = {}

    def _initialize(self) -> None:
        """Initialize the outcome space and frequency tables."""
        if self._initialized:
            return
        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_bid = self._outcome_space.outcomes[0].bid

        if self.nmi is not None:
            for i in range(len(self.nmi.issues)):
                self._value_frequency[i] = {}

        self._initialized = True

    def _get_threshold(self, time: float) -> float:
        """Compute the (purely time-based) threshold."""
        min_threshold = 1.0 + math.log(0.4) / 4.7
        if time <= 0.6:
            # Guard against log(0) at t == 1.0 (never reached here since t<=0.6)
            return 1.0 + math.log(max(1.0 - time, 1e-9)) / 4.7
        return min_threshold - math.log(max(1.6 - time, 1e-9)) / 3.0

    def _update_frequency(self, bid: Outcome) -> None:
        """Update the opponent value-frequency table with a received bid."""
        if bid is None:
            return
        for i, value in enumerate(bid):
            table = self._value_frequency.setdefault(i, {})
            value_str = str(value)
            table[value_str] = table.get(value_str, 0) + 1

    def _most_frequent_value(self, issue_index: int, bid: Outcome) -> object:
        """Return the most frequently observed value for an issue, if any."""
        table = self._value_frequency.get(issue_index, {})
        if not table:
            return bid[issue_index]

        # Find the value(s) with maximum frequency (shuffle for tie-breaking)
        items = list(table.items())
        random.shuffle(items)
        best_value_str, best_count = items[0]
        for value_str, count in items[1:]:
            if count > best_count:
                best_count = count
                best_value_str = value_str

        if self.nmi is None:
            return bid[issue_index]

        issue = self.nmi.issues[issue_index]
        for v in issue.all if hasattr(issue, "all") else []:
            if str(v) == best_value_str:
                return v
        return bid[issue_index]

    def _improve_bid(self, bid: Outcome) -> Outcome:
        """Try replacing each issue's value with the most frequent opponent value."""
        if self.ufun is None or self.nmi is None:
            return bid

        current = list(bid)
        current_utility = float(self.ufun(tuple(current)))

        issue_order = list(range(len(current)))
        random.shuffle(issue_order)

        for i in issue_order:
            candidate = list(current)
            candidate[i] = self._most_frequent_value(i, tuple(current))
            candidate_tuple = tuple(candidate)
            candidate_utility = float(self.ufun(candidate_tuple))
            if candidate_utility >= current_utility:
                current = candidate
                current_utility = candidate_utility

        return tuple(current)

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Find the bid with lowest utility above the threshold, then improve it."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return self._max_bid

        candidates = self._outcome_space.get_bids_above(threshold)
        if not candidates:
            bid = self._max_bid
        else:
            # Candidates are sorted descending by utility; take the smallest
            # one that still satisfies the threshold (last of the list).
            bid = candidates[-1].bid

        if bid is None:
            return self._max_bid

        return self._improve_bid(bid)

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        threshold = self._get_threshold(state.relative_time)
        return self._select_bid(threshold)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        self._update_frequency(offer)

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        threshold = self._get_threshold(state.relative_time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
