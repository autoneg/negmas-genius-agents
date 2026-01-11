"""
SlavaAgent from ANAC 2013.

This module implements SlavaAgent, a concession-based negotiation agent that
combines opponent modeling with adaptive concession strategies to find
mutually beneficial agreements.

References:
    - Baarslag, T., et al. (2013). "Evaluating Practical Negotiating Agents:
      Results and Analysis of the 2013 International Competition."
      Artificial Intelligence, 198, 73-103.
    - Original Java implementation: agents.anac.y2013.SlavaAgent.SlavaAgent
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

__all__ = ["SlavaAgent"]


class SlavaAgent(SAONegotiator):
    """
    SlavaAgent from ANAC 2013.

    SlavaAgent is a concession-based negotiation agent that combines opponent
    modeling with adaptive concession strategies to find mutually beneficial
    agreements while protecting its own interests.

    **Offering Strategy:**
        Computes a target utility using time-dependent concession:
        target = max - (max - min) * t^(1/concession_speed). The concession
        rate adapts: slows down (0.7x) if opponent is conceding, speeds up
        (1.2x) if opponent is tough. Scores candidate bids using weighted
        combination: (1 - opponent_weight) * own_util + opponent_weight *
        estimated_opponent_util. Selects from top third by score.

    **Acceptance Strategy:**
        Accepts if offer utility >= target. Also accepts if offer >= utility
        of our next proposal. Late game (> 0.95): accepts offers near opponent's
        best (>= 99%). Emergency acceptance (> 0.99) for offers above min_target.

    **Opponent Modeling:**
        Frequency-based model tracking issue value counts from opponent offers.
        Estimates opponent utility by computing average normalized frequency
        across issues. Detects opponent concession by comparing average utility
        of recent 5 offers vs early offers. Tracks opponent's best bid and
        uses it for late-game acceptance decisions.

    Args:
        initial_target: Initial target utility (default 0.95)
        min_target: Minimum target utility (default 0.6)
        concession_speed: How fast to concede over time (default 0.15)
        opponent_weight: Weight given to opponent utility in bid selection (default 0.3)
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
        concession_speed: float = 0.15,
        opponent_weight: float = 0.3,
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
        self._concession_speed = concession_speed
        self._opponent_weight = opponent_weight
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._issue_value_counts: dict[str, dict[str, int]] = {}
        self._total_opponent_offers: int = 0
        self._opponent_utilities: list[float] = []
        self._opponent_best_utility: float = 0.0
        self._opponent_best_bid: Outcome | None = None

        # State
        self._max_utility: float = 1.0
        self._current_target: float = initial_target
        self._opponent_conceding: bool = False

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._issue_value_counts = {}
        self._total_opponent_offers = 0
        self._opponent_utilities = []
        self._opponent_best_utility = 0.0
        self._opponent_best_bid = None
        self._current_target = self._initial_target
        self._opponent_conceding = False

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update opponent model based on received bid."""
        if bid is None:
            return

        self._total_opponent_offers += 1
        self._opponent_utilities.append(utility)

        # Track best opponent bid
        if utility > self._opponent_best_utility:
            self._opponent_best_utility = utility
            self._opponent_best_bid = bid

        # Update issue value frequency
        for i, value in enumerate(bid):
            issue_key = str(i)
            value_key = str(value)

            if issue_key not in self._issue_value_counts:
                self._issue_value_counts[issue_key] = {}

            if value_key not in self._issue_value_counts[issue_key]:
                self._issue_value_counts[issue_key][value_key] = 0

            self._issue_value_counts[issue_key][value_key] += 1

        # Detect if opponent is conceding
        if len(self._opponent_utilities) >= 5:
            recent = self._opponent_utilities[-5:]
            early = (
                self._opponent_utilities[:5]
                if len(self._opponent_utilities) >= 10
                else self._opponent_utilities[: len(self._opponent_utilities) // 2]
            )
            if early and recent:
                self._opponent_conceding = (
                    sum(recent) / len(recent) > sum(early) / len(early) + 0.02
                )

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent's utility based on frequency model."""
        if self._total_opponent_offers == 0 or bid is None:
            return 0.5

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            issue_key = str(i)
            value_key = str(value)

            if issue_key in self._issue_value_counts:
                counts = self._issue_value_counts[issue_key]
                if value_key in counts:
                    max_count = max(counts.values()) if counts else 1
                    total_score += counts[value_key] / max_count

        return total_score / num_issues if num_issues > 0 else 0.5

    def _compute_target(self, time: float) -> float:
        """Compute target utility with adaptive concession."""
        # Base concession formula (time-dependent)
        if self._concession_speed != 0:
            f_t = math.pow(time, 1 / self._concession_speed)
        else:
            f_t = 0.0

        # Adjust concession based on opponent behavior
        if self._opponent_conceding:
            # Opponent is conceding - slow down our concession
            f_t *= 0.7
        elif self._total_opponent_offers > 10 and not self._opponent_conceding:
            # Opponent is tough - speed up slightly
            f_t *= 1.2
            f_t = min(f_t, 1.0)

        # Calculate target
        target = self._max_utility - (self._max_utility - self._min_target) * f_t

        # Time pressure adjustment
        if time > 0.9:
            time_pressure = (time - 0.9) / 0.1
            target = target - time_pressure * (target - self._min_target) * 0.3

        self._current_target = max(target, self._min_target)
        return self._current_target

    def _compute_bid_score(self, own_utility: float, opponent_utility: float) -> float:
        """Compute weighted bid score balancing own and opponent utility."""
        return (
            1 - self._opponent_weight
        ) * own_utility + self._opponent_weight * opponent_utility

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid considering both own and opponent utility."""
        if self._outcome_space is None:
            return None

        target = self._compute_target(time)

        # Get candidates above target
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            if self._outcome_space.outcomes:
                return self._outcome_space.outcomes[0].bid
            return None

        # Early game: just return high utility bids
        if self._total_opponent_offers < 5:
            return random.choice(candidates).bid

        # Score candidates considering opponent utility
        scored_candidates = []
        for bd in candidates:
            opp_util = self._estimate_opponent_utility(bd.bid)
            score = self._compute_bid_score(bd.utility, opp_util)
            scored_candidates.append((bd.bid, score, bd.utility))

        if not scored_candidates:
            return random.choice(candidates).bid

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Select from top candidates
        top_k = max(1, len(scored_candidates) // 3)
        return random.choice(scored_candidates[:top_k])[0]

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

        offer_utility = float(self.ufun(offer))
        time = state.relative_time

        # Update opponent model
        self._update_opponent_model(offer, offer_utility)

        # Compute current target
        target = self._compute_target(time)

        # Accept if above target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than what we'd propose next
        our_bid = self._select_bid(time)
        if our_bid is not None:
            our_utility = float(self.ufun(our_bid))
            if offer_utility >= our_utility:
                return ResponseType.ACCEPT_OFFER

        # Accept opponent's best offer near deadline
        if time > 0.95:
            if offer_utility >= self._opponent_best_utility * 0.99:
                return ResponseType.ACCEPT_OFFER

        # Emergency acceptance
        if time > 0.99 and offer_utility >= self._min_target:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
