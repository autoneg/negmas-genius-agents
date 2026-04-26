"""WhaleAgent from ANAC 2014."""

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

__all__ = ["WhaleAgent"]


class WhaleAgent(SAONegotiator):
    """
    WhaleAgent from ANAC 2014.

    WhaleAgent implements a classic Boulware-style time-dependent strategy
    enhanced with opponent modeling for Nash-product-based bid selection.
    The name reflects its patient, "whale-like" approach of waiting before
    making concessions.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.AgentWhale.WhaleAgent

    **Offering Strategy:**
        Boulware concession curve with Nash-product optimization:
        - Target utility: min + (max - min) * (1 - t^e)
        - Default e=0.2 creates Boulware behavior: slow initial concession,
          accelerating sharply near deadline
        - Bids selected from candidates above target using Nash product:
          score = own_utility * estimated_opponent_utility
          maximizing joint welfare promotes Pareto-efficient agreements

        When no opponent model exists, selects randomly from top 25% of
        candidates to maintain high utility early in negotiation.

    **Acceptance Strategy:**
        Target-based with adaptive flexibility:
        1. Accept if offer utility meets Boulware-computed target
        2. Accept if offer matches or exceeds next planned bid utility
        3. Near-deadline flexibility (t > 0.98) for offers above minimum
        The Boulware curve means acceptance threshold stays high until
        late in negotiation, then drops rapidly.

    **Opponent Modeling:**
        Frequency-based preference analysis:
        - Tracks value frequencies per issue from opponent bid history
        - Estimates opponent utility as normalized frequency score
          (how often opponent proposes each value)
        - Nash product scoring: own_util * opponent_util
        - Best opponent bid tracked for reference
        - Model enables finding mutually beneficial outcomes within
          the agent's acceptable utility range

    Args:
        e: Boulware exponent (default 0.2). Lower values create more
           patient (Boulware) behavior; higher values more eager (Conceder).
        min_utility_floor: Minimum acceptable utility threshold (default 0.6).
        deadline_acceptance_time: Time threshold for near-deadline acceptance (default 0.98).
        top_candidates_divisor: Divisor for selecting top candidates (default 4).
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
        min_utility_floor: float = 0.6,
        deadline_acceptance_time: float = 0.98,
        top_candidates_divisor: int = 4,
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
        self._min_utility_floor = min_utility_floor
        self._deadline_acceptance_time = deadline_acceptance_time
        self._top_candidates_divisor = top_candidates_divisor
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_value_freq: dict[int, dict] = {}

        # Bidding state
        self._reservation_value: float = 0.0
        self._min_util: float = min_utility_floor

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._min_util = self._outcome_space.min_utility
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

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

    def _compute_target_utility(self, time: float) -> float:
        """Compute target utility using Boulware curve."""
        max_util = 1.0
        if self._outcome_space and self._outcome_space.outcomes:
            max_util = self._outcome_space.max_utility

        # Boulware concession curve
        target = self._min_util + (max_util - self._min_util) * (1 - time**self._e)
        return max(self._min_util, target)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid using Nash product approach."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._compute_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            # Fallback to best bid
            return self._outcome_space.outcomes[0].bid

        # If we have opponent model, maximize Nash product
        if self._opponent_value_freq:
            best_bid = None
            best_product = -1.0

            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                nash_product = bd.utility * opp_util
                if nash_product > best_product:
                    best_product = nash_product
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        # No opponent model, return random high utility bid
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

        # Accept if above target utility
        target = self._compute_target_utility(time)
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if offer >= our next bid utility
        next_bid = self._select_bid(time)
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid))
            if offer_utility >= next_utility:
                return ResponseType.ACCEPT_OFFER

        # Near deadline, be more flexible
        if time > self._deadline_acceptance_time and offer_utility >= self._min_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
