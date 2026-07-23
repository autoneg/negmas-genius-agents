"""AgentX from ANAC 2015."""

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

__all__ = ["AgentX"]


class AgentX(SAONegotiator):
    """
    AgentX negotiation agent from ANAC 2015.

    AgentX uses an exploratory strategy in early phases before transitioning
    to Nash-seeking bid selection based on learned opponent preferences.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.AgentX.AgentX

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Three-phase approach with exploration:
          * Exploration (t<0.25): Cycles through top 10 candidates with
            small random variance (90-98% utility)
          * Main (0.25<t<0.75): Gradual concession toward 55% with
            adaptive rate based on opponent generosity
          * End (t>0.75): More aggressive concession toward minimum
        - Adaptive rate: firmer (e * 0.8) if avg opponent utility > 60%,
          more flexible (e * 1.2) if < 30%
        - Nash-seeking selection balancing own utility (60%) and opponent
          fit (40%)

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - AC_Next: Accepts if offer utility >= utility of our next bid
        - End-game (t>0.95): Accepts if offer >= 95% of best opponent
          utility OR offer >= min_utility + 0.1

    **Opponent Modeling:**
        - Frequency-based model tracking issue-value preferences
        - Maintains running average of opponent utility offers
        - Estimates opponent fit based on normalized value frequencies
        - Uses exploration phase to gather opponent information

    Args:
        e: Concession exponent controlling concession speed (default 0.2)
        exploration_time_threshold: Time threshold for exploration phase (default 0.25)
        main_time_threshold: Time threshold for main/end phase transition (default 0.75)
        deadline_time_threshold: Time after which end-game acceptance triggers (default 0.95)
        opponent_generous_threshold: Avg opponent utility above which agent stays firmer (default 0.6)
        opponent_tough_threshold: Avg opponent utility below which agent concedes more (default 0.3)
        firm_multiplier: Multiplier applied to e when opponent is generous (default 0.8)
        flexible_multiplier: Multiplier applied to e when opponent is tough (default 1.2)
        exploration_base: Base utility ratio during exploration phase (default 0.90)
        exploration_variance: Random variance added to exploration threshold (default 0.08)
        main_phase_max_ratio: Utility ratio at start of main phase (default 0.95)
        main_phase_target: Utility target at end of main phase (default 0.55)
        main_phase_concession_factor: Fraction of concession applied in main phase (default 0.7)
        min_utility_margin: Margin above min utility used as acceptance floor (default 0.1)
        end_phase_base: Utility base at start of end phase (default 0.55)
        end_phase_min_margin: Margin above min utility in end phase (default 0.05)
        end_phase_concession_factor: Fraction of concession applied in end phase (default 0.8)
        fallback_threshold_ratio: Ratio used when lowering threshold if no candidates (default 0.85)
        exploration_candidate_count: Number of top candidates cycled in exploration (default 10)
        nash_search_count: Number of top candidates searched for Nash-like bid (default 40)
        own_utility_weight: Weight of own utility in Nash-like scoring (default 0.6)
        opponent_fit_weight: Weight of opponent fit in Nash-like scoring (default 0.4)
        endgame_opponent_ratio: Ratio of best opponent utility used in end-game acceptance (default 0.95)
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
        exploration_time_threshold: float = 0.25,
        main_time_threshold: float = 0.75,
        deadline_time_threshold: float = 0.95,
        opponent_generous_threshold: float = 0.6,
        opponent_tough_threshold: float = 0.3,
        firm_multiplier: float = 0.8,
        flexible_multiplier: float = 1.2,
        exploration_base: float = 0.90,
        exploration_variance: float = 0.08,
        main_phase_max_ratio: float = 0.95,
        main_phase_target: float = 0.55,
        main_phase_concession_factor: float = 0.7,
        min_utility_margin: float = 0.1,
        end_phase_base: float = 0.55,
        end_phase_min_margin: float = 0.05,
        end_phase_concession_factor: float = 0.8,
        fallback_threshold_ratio: float = 0.85,
        exploration_candidate_count: int = 10,
        nash_search_count: int = 40,
        own_utility_weight: float = 0.6,
        opponent_fit_weight: float = 0.4,
        endgame_opponent_ratio: float = 0.95,
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
        self._exploration_time_threshold = exploration_time_threshold
        self._main_time_threshold = main_time_threshold
        self._deadline_time_threshold = deadline_time_threshold
        self._opponent_generous_threshold = opponent_generous_threshold
        self._opponent_tough_threshold = opponent_tough_threshold
        self._firm_multiplier = firm_multiplier
        self._flexible_multiplier = flexible_multiplier
        self._exploration_base = exploration_base
        self._exploration_variance = exploration_variance
        self._main_phase_max_ratio = main_phase_max_ratio
        self._main_phase_target = main_phase_target
        self._main_phase_concession_factor = main_phase_concession_factor
        self._min_utility_margin = min_utility_margin
        self._end_phase_base = end_phase_base
        self._end_phase_min_margin = end_phase_min_margin
        self._end_phase_concession_factor = end_phase_concession_factor
        self._fallback_threshold_ratio = fallback_threshold_ratio
        self._exploration_candidate_count = exploration_candidate_count
        self._nash_search_count = nash_search_count
        self._own_utility_weight = own_utility_weight
        self._opponent_fit_weight = opponent_fit_weight
        self._endgame_opponent_ratio = endgame_opponent_ratio
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._exploration_index: int = 0

        # Opponent modeling
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._value_preferences: dict[int, dict] = {}
        self._best_opponent_utility: float = 0.0
        self._avg_opponent_utility: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = self._outcome_space.min_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._value_preferences = {}
        self._best_opponent_utility = 0.0
        self._avg_opponent_utility = 0.0
        self._exploration_index = 0

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update opponent model with preference tracking."""
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Update average
        total = sum(u for _, u in self._opponent_bids)
        self._avg_opponent_utility = total / len(self._opponent_bids)

        # Track value preferences
        for i, value in enumerate(bid):
            if i not in self._value_preferences:
                self._value_preferences[i] = {}
            self._value_preferences[i][value] = (
                self._value_preferences[i].get(value, 0) + 1
            )

    def _estimate_opponent_fit(self, bid: Outcome) -> float:
        """Estimate how well bid fits opponent preferences."""
        if not self._value_preferences:
            return 0.5

        score = 0.0
        num_issues = 0

        for i, value in enumerate(bid):
            if i in self._value_preferences:
                freq = self._value_preferences[i].get(value, 0)
                max_freq = (
                    max(self._value_preferences[i].values())
                    if self._value_preferences[i]
                    else 1
                )
                score += freq / max_freq if max_freq > 0 else 0
                num_issues += 1

        return score / num_issues if num_issues > 0 else 0.5

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold with X-factor exploration."""
        # Adjust e based on opponent behavior
        e = self._e

        if self._avg_opponent_utility > self._opponent_generous_threshold:
            e *= self._firm_multiplier  # Opponent offering good deals, stay firm
        elif self._avg_opponent_utility < self._opponent_tough_threshold:
            e *= self._flexible_multiplier  # Opponent tough, concede more

        if time < self._exploration_time_threshold:
            # Exploration phase: stay high but vary
            return self._max_utility * (
                self._exploration_base + random.uniform(0, self._exploration_variance)
            )
        elif time < self._main_time_threshold:
            # Main phase: gradual concession
            progress = (time - self._exploration_time_threshold) / (
                self._main_time_threshold - self._exploration_time_threshold
            )
            f_t = math.pow(progress, 1 / e)
            target = (
                self._max_utility * self._main_phase_max_ratio
                - (
                    self._max_utility * self._main_phase_max_ratio
                    - self._main_phase_target
                )
                * f_t
                * self._main_phase_concession_factor
            )
            return max(target, self._min_utility + self._min_utility_margin)
        else:
            # End phase: more aggressive
            progress = (time - self._main_time_threshold) / (
                1.0 - self._main_time_threshold
            )
            base = self._end_phase_base
            target = (
                base
                - (base - self._min_utility - self._end_phase_min_margin)
                * progress
                * self._end_phase_concession_factor
            )
            return max(target, self._min_utility + self._end_phase_min_margin)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid with X-factor exploration."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(
                threshold * self._fallback_threshold_ratio
            )

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Early: explore different high-utility bids
        if time < self._exploration_time_threshold:
            self._exploration_index = (self._exploration_index + 1) % min(
                len(candidates), self._exploration_candidate_count
            )
            return candidates[self._exploration_index].bid

        # Later: find Nash-like bids
        if self._value_preferences:
            best_bid = None
            best_score = -1.0

            for bd in candidates[: self._nash_search_count]:
                opponent_fit = self._estimate_opponent_fit(bd.bid)
                score = (
                    bd.utility * self._own_utility_weight
                    + opponent_fit * self._opponent_fit_weight
                )
                if score > best_score:
                    best_score = score
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        return self._select_bid(state.relative_time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        self._update_opponent_model(offer, offer_utility)

        threshold = self._compute_threshold(time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # AC_Next: accept if >= our next offer
        our_bid = self._select_bid(time)
        if our_bid is not None:
            our_utility = float(self.ufun(our_bid))
            if offer_utility >= our_utility:
                return ResponseType.ACCEPT_OFFER

        # End-game acceptance
        if time > self._deadline_time_threshold:
            if offer_utility >= max(
                self._best_opponent_utility * self._endgame_opponent_ratio,
                self._min_utility + self._min_utility_margin,
            ):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER