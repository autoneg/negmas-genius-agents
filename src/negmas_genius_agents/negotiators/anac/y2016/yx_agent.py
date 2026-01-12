"""YXAgent from ANAC 2016."""

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

__all__ = ["YXAgent"]


class YXAgent(SAONegotiator):
    """
    YXAgent negotiation agent from ANAC 2016 - 2nd place.

    YXAgent achieved 2nd place in ANAC 2016. The agent uses a conservative
    threshold-based strategy with opponent hardness estimation based on
    value variance analysis.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2016.yxagent.YXAgent

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
    Conservative threshold-based bidding:

    - Initial threshold: max(min_threshold, 1.0 - num_opponents * 0.1)
    - Time adjustment: base threshold * (1 - 0.1 * time)
    - Hardness adjustment: subtracts factor based on hardest opponent
    - Final threshold capped at minimum threshold (default 0.7)

    Early negotiation (t < 0.1) always offers the best available bid.
    Bids selected randomly from all candidates above adjusted threshold.

    **Acceptance Strategy:**
    Threshold-based acceptance:

    - Accepts if offer utility meets or exceeds adjusted threshold
    - Very near deadline (t >= 0.98): accepts if above minimum threshold

    **Opponent Modeling:**
    Hardness estimation via value variance analysis:

    - Tracks value frequencies for each issue per opponent
    - Computes variance of value frequencies for each issue
    - Low variance = opponent is consistent/stubborn (hard)
    - High variance = opponent varies bids (easier)
    - Hardness computed as: 1 / (1 + average_variance)
    - Hardest opponent's hardness used to adjust acceptance threshold
    - Higher hardness triggers slight threshold reduction for flexibility

    The strategy maintains high utility requirements while adapting
    slightly based on opponent negotiation difficulty.

    Args:
        min_threshold: Minimum acceptable utility threshold (default 0.7)
        early_time: Time threshold for early phase best-bid offering (default 0.1)
        deadline_time: Time threshold for deadline acceptance (default 0.98)
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
        min_threshold: float = 0.7,
        early_time: float = 0.1,
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
        self._min_threshold = min_threshold
        self._early_time = early_time
        self._deadline_time = deadline_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_value_frequencies: dict[str, dict[int, dict[str, int]]] = {}
        self._opponent_hardness: dict[str, float] = {}
        self._num_opponents: int = 1

        # State
        self._threshold: float = 0.9
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

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

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Estimate number of opponents
        if self.nmi is not None:
            n_negotiators = self.nmi.n_negotiators
            self._num_opponents = max(1, n_negotiators - 1)
        else:
            self._num_opponents = 1

        # Calculate initial threshold
        self._threshold = max(self._min_threshold, 1.0 - (self._num_opponents * 0.1))

        # Reset state
        self._opponent_value_frequencies = {}
        self._opponent_hardness = {}

    def _update_opponent_model(self, source: str | None, bid: Outcome) -> None:
        """Update opponent model based on received bid."""
        if bid is None or source is None:
            return

        if source not in self._opponent_value_frequencies:
            self._opponent_value_frequencies[source] = {}

        # Track value frequencies for each issue
        for i, value in enumerate(bid):
            if i not in self._opponent_value_frequencies[source]:
                self._opponent_value_frequencies[source][i] = {}

            value_str = str(value)
            if value_str not in self._opponent_value_frequencies[source][i]:
                self._opponent_value_frequencies[source][i][value_str] = 0
            self._opponent_value_frequencies[source][i][value_str] += 1

        # Compute hardness as average variance across issues
        self._compute_opponent_hardness(source)

    def _compute_opponent_hardness(self, source: str) -> None:
        """
        Compute opponent hardness based on value frequency variance.

        Higher variance means opponent varies their bids more (easier to negotiate with).
        Lower variance means opponent is more consistent/stubborn (harder).
        """
        if source not in self._opponent_value_frequencies:
            return

        total_variance = 0.0
        num_issues = 0

        for issue_idx, freq_map in self._opponent_value_frequencies[source].items():
            if not freq_map:
                continue

            values = list(freq_map.values())
            n = len(values)
            if n <= 1:
                continue

            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            total_variance += variance
            num_issues += 1

        # Hardness is inverse of variance (low variance = hard opponent)
        if num_issues > 0:
            avg_variance = total_variance / num_issues
            # Invert: high variance -> low hardness
            self._opponent_hardness[source] = 1.0 / (1.0 + avg_variance)
        else:
            self._opponent_hardness[source] = 0.5

    def _get_hardest_opponent_factor(self) -> float:
        """Get adjustment factor based on hardest opponent."""
        if not self._opponent_hardness:
            return 0.0

        # Find hardest opponent (highest hardness value)
        max_hardness = max(self._opponent_hardness.values())
        # Harder opponents mean we should be slightly more flexible
        return max_hardness * 0.05

    def _get_adjusted_threshold(self, time: float) -> float:
        """Get threshold adjusted by time and opponent hardness."""
        # Base threshold with slight time concession
        base = self._threshold * (1 - 0.1 * time)

        # Adjust based on hardest opponent
        hardness_adjustment = self._get_hardest_opponent_factor()

        adjusted = base - hardness_adjustment
        return max(adjusted, self._min_threshold)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid above the threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._get_adjusted_threshold(time)

        # Get candidates above threshold
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            # Fallback: lower threshold until we find candidates
            for bd in self._outcome_space.outcomes:
                if bd.utility >= self._min_threshold:
                    candidates = [bd]
                    break

        if candidates:
            return random.choice(candidates).bid

        return self._best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        # Early game: offer best bid
        if time < self._early_time:
            return self._best_bid

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

        # Update opponent model
        self._update_opponent_model(source, offer)

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        threshold = self._get_adjusted_threshold(time)

        # Accept if offer exceeds threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if above minimum threshold
        if time >= self._deadline_time and offer_utility >= self._min_threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
