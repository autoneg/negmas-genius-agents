"""
PodAgent from ANAC 2019.

This module contains the Python reimplementation of PodAgent
(``Group1_BOA``), a BOA-framework agent built from a custom acceptance
strategy (``Group1_AS``), offering strategy (``Group1_BS``) and opponent
model (``Group1_OM``).

References:
    Baarslag, T., Fujita, K., Gerding, E. H., Hindriks, K., Ito, T.,
    Jennings, N. R., ... & Williams, C. R. (2019). The Tenth International
    Automated Negotiating Agents Competition (ANAC 2019).
    In Proceedings of the International Joint Conference on Autonomous
    Agents and Multiagent Systems (AAMAS).

Original Genius class: agents.anac.y2019.podagent.Group1_BOA
(combining Group1_AS, Group1_BS and Group1_OM)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["PodAgent"]


class PodAgent(SAONegotiator):
    """
    PodAgent from ANAC 2019.

    PodAgent is built with the Genius BOA framework, combining a custom
    offering strategy (``Group1_BS``), acceptance strategy (``Group1_AS``)
    and a frequency-based opponent model (``Group1_OM``). It concedes on a
    ``1 - t^3`` curve scaled by an estimate of the opponent's "friendliness"
    (whether the opponent's last few offers are trending toward higher
    utility for us), and only accepts based on comparing the opponent's
    offer to the bid it is about to make next (rather than a fixed
    threshold).

    Note:
        This is an AI-generated reimplementation based on the original Java
        code from the Genius framework. It may not behave identically to
        the original.

    **Offering Strategy (Group1_BS):**
        - Target utility follows ``target = target - (delta_undiscounted) *
          friendliness`` where the undiscounted target follows
          ``1 - t**3`` and friendliness is estimated from the opponent's
          recent sentiment (increasing utility trend => friendlier => more
          concession).
        - Every ``timeLimitForStep`` (starts at 0.1, shrinks by 10% each
          step) recomputes the set of candidate bids in
          ``[target_utility, 1]``.
        - From the candidate set, offers the bid maximizing the estimated
          opponent utility (from the frequency opponent model).
        - Extreme panic (last ~1.67% of time): offers the opponent's best
          received bid.
        - Panic (last 5% of time): if the opponent appears "hard headed"
          (few distinct bids offered), stays near utility 1 instead of
          conceding further.

    **Acceptance Strategy (Group1_AS):**
        - ACnext: accept if the opponent's last offer utility is >= the
          utility of the bid we are about to offer next.
        - Otherwise, in panic mode against a hard-headed opponent, accept
          if the offer utility is >= 0.95.

    **Opponent Modeling (Group1_OM):**
        - Simplified frequency model: tracks how often each issue value
          appears in the opponent's bids and derives an opponent utility
          estimate from value frequencies (normalized per issue), similar
          to the HardHeaded-style frequency model.
        - "Hard-headed" opponent detection: opponent is considered
          hard-headed if it has offered 3 or fewer distinct bids so far.

    Args:
        panic_threshold: Fraction of remaining time to trigger panic mode
            (default 0.05).
        extreme_panic_threshold: Fraction of remaining time to trigger
            extreme panic mode (default 0.05/3).
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
        panic_threshold: float = 0.05,
        extreme_panic_threshold: float = 0.05 / 3,
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
        self._panic_threshold = panic_threshold
        self._extreme_panic_threshold = extreme_panic_threshold

        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Offering strategy state (Group1_BS)
        self._target_utility: float = 1.0
        self._undiscounted_target: float = 1.0
        self._time_limit_for_step: float = 0.1
        self._last_step_time: float = 0.0
        self._panic_mode: bool = False
        self._extreme_panic_mode: bool = False
        self._hard_headed_locked: bool = False
        self._candidates: list = []
        self._next_bid: Outcome | None = None
        self._best_bid: Outcome | None = None

        # Opponent model state (Group1_OM)
        self._value_frequencies: dict[int, dict[str, int]] = {}
        self._opponent_history: list[Outcome] = []
        self._last_step_bids_average: float = 0.0

    # -- setup -----------------------------------------------------------

    def _initialize(self) -> None:
        if self._initialized:
            return
        if self.ufun is None:
            return
        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._initialize()
        self._target_utility = 1.0
        self._undiscounted_target = 1.0
        self._time_limit_for_step = 0.1
        self._last_step_time = 0.0
        self._panic_mode = False
        self._extreme_panic_mode = False
        self._hard_headed_locked = False
        self._candidates = []
        self._next_bid = self._best_bid
        self._value_frequencies = {}
        self._opponent_history = []
        self._last_step_bids_average = 0.0

    # -- opponent model ----------------------------------------------------

    def _update_opponent_model(self, bid: Outcome) -> None:
        if bid is None:
            return
        self._opponent_history.append(bid)
        for i, value in enumerate(bid):
            freqs = self._value_frequencies.setdefault(i, {})
            value_str = str(value)
            freqs[value_str] = freqs.get(value_str, 0) + 1

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility from normalized value frequencies."""
        if bid is None or not self._value_frequencies:
            return 0.5

        num_issues = len(bid)
        total = 0.0
        for i, value in enumerate(bid):
            freqs = self._value_frequencies.get(i)
            if not freqs:
                continue
            max_freq = max(freqs.values())
            total += freqs.get(str(value), 0) / max_freq if max_freq > 0 else 0.0
        return total / num_issues if num_issues > 0 else 0.5

    def _is_hard_headed(self) -> bool:
        """Opponent is hard-headed if it has offered <= 3 distinct bids."""
        seen: set = set()
        for bid in self._opponent_history:
            seen.add(bid)
            if len(seen) > 3:
                return False
        return True

    def _opponent_sentiment(self, time: float) -> float:
        """Estimate how much the opponent's recent offers are improving for us."""
        if self.ufun is None or not self._opponent_history:
            return 0.5

        if self._last_step_bids_average == 0.0:
            self._last_step_bids_average = float(self.ufun(self._opponent_history[-1]))
            return 0.5

        recent = [
            float(self.ufun(b))
            for b in self._opponent_history[-max(1, len(self._opponent_history) // 4):]
        ]
        new_average = sum(recent) / len(recent) if recent else 0.0
        sentiment = new_average - self._last_step_bids_average
        self._last_step_bids_average = new_average

        if sentiment < 0:
            return sentiment - 0.5
        elif sentiment > 0:
            return sentiment + 0.5
        return 0.0

    # -- offering strategy (Group1_BS) -------------------------------------

    def _set_panic_mode_if_necessary(self, time: float) -> None:
        if not self._extreme_panic_mode and 1 - time < self._extreme_panic_threshold:
            self._extreme_panic_mode = True
        if not self._panic_mode and 1 - time < self._panic_threshold:
            self._panic_mode = True
            self._hard_headed_locked = self._is_hard_headed()

    def _current_concession_target(self, time: float, reservation: float) -> float:
        friendliness = 1.0 if self._panic_mode else self._opponent_sentiment(time)
        time_discount = self._undiscounted_target - (1 - time**3)
        discount = time_discount * friendliness
        self._undiscounted_target = 1 - time**3
        new_utility = self._target_utility - discount
        if new_utility < reservation:
            return reservation
        return max(0.0, min(1.0, new_utility))

    def _set_new_step(self, time: float) -> None:
        if self._time_limit_for_step > 0.01:
            self._time_limit_for_step *= 0.9

        reservation = self.reserved_value if self.reserved_value is not None else 0.0
        if not self._hard_headed_locked:
            self._target_utility = self._current_concession_target(time, reservation)
        else:
            self._target_utility = 1.0

        if self._outcome_space is not None:
            self._candidates = self._outcome_space.get_bids_above(self._target_utility)
        self._last_step_time = time

    def _determine_next_bid(self, time: float) -> Outcome | None:
        if self._outcome_space is None or self._outcome_space.outcomes is None:
            return self._best_bid

        self._set_panic_mode_if_necessary(time)

        if self._extreme_panic_mode:
            if self._opponent_history:
                return max(self._opponent_history, key=lambda b: float(self.ufun(b)))
            return self._best_bid

        if (
            not self._candidates
            or time >= self._time_limit_for_step + self._last_step_time
            or self._panic_mode
        ):
            self._set_new_step(time)

        if not self._candidates:
            return self._best_bid

        best_bid = None
        best_opponent_util = 0.0
        for bd in self._candidates:
            opp_util = self._estimate_opponent_utility(bd.bid)
            if opp_util > best_opponent_util:
                best_bid = bd.bid
                best_opponent_util = opp_util

        if best_bid is None:
            best_bid = self._candidates[0].bid
        return best_bid

    # -- SAONegotiator interface --------------------------------------------

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        if not self._initialized:
            self._initialize()

        if self._next_bid is None:
            self._next_bid = self._best_bid

        bid_to_offer = self._next_bid
        self._next_bid = self._determine_next_bid(state.relative_time)
        return bid_to_offer

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        self._update_opponent_model(offer)

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))

        # Pre-compute next bid so ACnext can compare against it.
        if self._next_bid is None:
            self._next_bid = self._determine_next_bid(state.relative_time)
        next_bid_utility = (
            float(self.ufun(self._next_bid)) if self._next_bid is not None else 1.0
        )

        if offer_utility >= next_bid_utility:
            return ResponseType.ACCEPT_OFFER

        if not self._panic_mode:
            return ResponseType.REJECT_OFFER

        if self._is_hard_headed() and offer_utility >= 0.95:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
