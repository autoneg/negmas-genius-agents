"""AgentQuest from ANAC 2014."""

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

__all__ = ["AgentQuest"]


class AgentQuest(SAONegotiator):
    """
    AgentQuest from ANAC 2014.

    AgentQuest employs a quest-based metaphor for negotiation, setting
    sequential utility targets ("quests") that guide its bidding behavior.
    The agent adapts its goals based on time pressure and opponent behavior.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.AgentQuest.AgentQuest

    **Offering Strategy:**
        Maintains a dynamic "quest goal" that represents the current utility
        target. Bids are selected from candidates that meet this goal, with
        preference given to offers estimated to satisfy the opponent. The
        quest goal starts high and decays over time using an aspiration-based
        function: decay = time^(aspiration_decay) * utility_range. The decay
        rate adapts based on opponent behavior trends.

    **Acceptance Strategy:**
        Multi-stage acceptance criteria:
        1. Accept if offer utility meets or exceeds the current quest goal
        2. Accept if offer utility is at least as good as the next planned bid
        3. Accept near deadline (t > deadline_time) if utility exceeds minimum threshold
        This layered approach balances aspiration with practical deal-making.

    **Opponent Modeling:**
        Uses frequency analysis to build a preference model of the opponent.
        Tracks how often the opponent proposes each value for each issue,
        building a frequency map. This model estimates opponent utility for
        candidate bids, enabling the agent to select offers that balance
        self-interest with opponent satisfaction. Recent opponent utilities
        are analyzed to detect concession trends and adjust quest decay rate.

    Args:
        initial_aspiration: Starting aspiration level (default 0.95).
        aspiration_decay: Exponent controlling decay speed (default 0.5).
        min_utility_floor: Floor value for minimum utility (default 0.5).
        positive_trend_threshold: Threshold for detecting positive opponent trend (default 0.05).
        negative_trend_threshold: Threshold for detecting negative opponent trend (default -0.05).
        positive_trend_decay_factor: Decay multiplier when opponent conceding (default 0.8).
        negative_trend_decay_factor: Decay multiplier when opponent hardening (default 1.2).
        decay_multiplier: Multiplier applied to base decay for quest goal (default 0.6).
        lowered_threshold_factor: Factor to lower threshold when no candidates (default 0.95).
        own_utility_weight: Weight for own utility in bid scoring (default 0.7).
        opponent_utility_weight: Weight for opponent utility in bid scoring (default 0.3).
        top_candidates_divisor: Divisor to select top candidates (default 3).
        deadline_time: Time threshold for near-deadline acceptance (default 0.98).
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
        initial_aspiration: float = 0.95,
        aspiration_decay: float = 0.5,
        min_utility_floor: float = 0.5,
        positive_trend_threshold: float = 0.05,
        negative_trend_threshold: float = -0.05,
        positive_trend_decay_factor: float = 0.8,
        negative_trend_decay_factor: float = 1.2,
        decay_multiplier: float = 0.6,
        lowered_threshold_factor: float = 0.95,
        own_utility_weight: float = 0.7,
        opponent_utility_weight: float = 0.3,
        top_candidates_divisor: int = 3,
        deadline_time: float = 0.98,
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
        self._initial_aspiration = initial_aspiration
        self._aspiration_decay = aspiration_decay
        self._min_utility_floor = min_utility_floor
        self._positive_trend_threshold = positive_trend_threshold
        self._negative_trend_threshold = negative_trend_threshold
        self._positive_trend_decay_factor = positive_trend_decay_factor
        self._negative_trend_decay_factor = negative_trend_decay_factor
        self._decay_multiplier = decay_multiplier
        self._lowered_threshold_factor = lowered_threshold_factor
        self._own_utility_weight = own_utility_weight
        self._opponent_utility_weight = opponent_utility_weight
        self._top_candidates_divisor = top_candidates_divisor
        self._deadline_time = deadline_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Quest tracking
        self._current_quest_goal: float = initial_aspiration
        self._quest_history: list[float] = []

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_value_freq: dict[int, dict] = {}

        # State
        self._min_utility: float = min_utility_floor
        self._max_utility: float = 1.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = max(
                self._min_utility_floor, self._outcome_space.min_utility
            )
        self._current_quest_goal = self._max_utility
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._current_quest_goal = self._initial_aspiration
        self._quest_history = []
        self._opponent_bids = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_value_freq = {}

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent model with frequency analysis."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append(bid)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        # Update value frequencies
        for i, value in enumerate(bid):
            if i not in self._opponent_value_freq:
                self._opponent_value_freq[i] = {}
            self._opponent_value_freq[i][value] = (
                self._opponent_value_freq[i].get(value, 0) + 1
            )

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility using frequency model."""
        if not self._opponent_value_freq:
            return 0.5

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            if i in self._opponent_value_freq:
                freq = self._opponent_value_freq[i].get(value, 0)
                max_freq = max(self._opponent_value_freq[i].values())
                if max_freq > 0:
                    total_score += freq / max_freq

        return total_score / num_issues if num_issues > 0 else 0.5

    def _update_quest_goal(self, time: float) -> None:
        """Update the current quest goal based on time and progress."""
        # Aspiration-based decay
        base_decay = (time**self._aspiration_decay) * (
            self._max_utility - self._min_utility
        )

        # Adjust based on opponent behavior
        if self._opponent_bids and len(self._opponent_bids) > 3:
            # If opponent is improving offers, slow decay
            recent_utilities = [
                float(self.ufun(b)) for b in self._opponent_bids[-5:] if self.ufun
            ]
            if len(recent_utilities) >= 2:
                trend = recent_utilities[-1] - recent_utilities[0]
                if trend > self._positive_trend_threshold:
                    base_decay *= (
                        self._positive_trend_decay_factor
                    )  # Opponent conceding, slow our decay
                elif trend < self._negative_trend_threshold:
                    base_decay *= (
                        self._negative_trend_decay_factor
                    )  # Opponent hardening, speed up decay

        self._current_quest_goal = (
            self._max_utility - base_decay * self._decay_multiplier
        )
        self._current_quest_goal = max(self._min_utility, self._current_quest_goal)
        self._quest_history.append(self._current_quest_goal)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid that meets quest goal and considers opponent."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        self._update_quest_goal(time)
        candidates = self._outcome_space.get_bids_above(self._current_quest_goal)

        if not candidates:
            # Lower threshold and try again
            lowered = self._current_quest_goal * self._lowered_threshold_factor
            candidates = self._outcome_space.get_bids_above(lowered)
            if not candidates:
                return self._outcome_space.outcomes[0].bid

        # Use opponent model if available
        if self._opponent_value_freq:
            best_bid = None
            best_score = -1.0

            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                # Balance own utility with opponent satisfaction
                score = (
                    self._own_utility_weight * bd.utility
                    + self._opponent_utility_weight * opp_util
                )
                if score > best_score:
                    best_score = score
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        # No model, return random from top candidates
        return random.choice(
            candidates[: max(1, len(candidates) // self._top_candidates_divisor)]
        ).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        return self._select_bid(time)

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
        self._update_opponent_model(offer)
        self._update_quest_goal(time)

        # Accept if meets quest goal
        if offer_utility >= self._current_quest_goal:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than our next bid
        next_bid = self._select_bid(time)
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid))
            if offer_utility >= next_utility:
                return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if reasonable
        if time > self._deadline_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
