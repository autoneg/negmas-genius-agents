"""
IAMcrazyHaggler from ANAC 2010.

IAMcrazyHaggler (I AM crazy Haggler) was developed by the Southampton team
(Colin R. Williams, Valentin Robu, Enrico H. Gerding, Nicholas R. Jennings)
as part of the IAM agent family. It uses a deliberately "irrational" random
bidding strategy combined with stubborn acceptance.

This implementation faithfully reproduces IAMcrazyHaggler's core strategies:
- Random high-utility bid selection (appears "crazy" to opponents)
- Boulware-style acceptance (very reluctant to accept)
- No strategic concession - maintains tough stance throughout
- Deadline concession to avoid negotiation breakdown

References:
    Williams, C.R., Robu, V., Gerding, E.H., Jennings, N.R. (2011).
    "Iamhaggler: A Negotiation Agent for Complex Environments"
    New Trends in Agent-based Complex Automated Negotiations, pp. 151-158.

    @inproceedings{williams2011iamhaggler,
      title={Iamhaggler: A Negotiation Agent for Complex Environments},
      author={Williams, Colin R and Robu, Valentin and Gerding, Enrico H
              and Jennings, Nicholas R},
      booktitle={New Trends in Agent-based Complex Automated Negotiations},
      series={Studies in Computational Intelligence},
      volume={383},
      pages={151--158},
      year={2011},
      publisher={Springer}
    }
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

__all__ = ["IAMcrazyHaggler"]


class IAMcrazyHaggler(SAONegotiator):
    """
    IAMcrazyHaggler from ANAC 2010.

    IAMcrazyHaggler uses random high-utility bidding with stubborn acceptance.

    **Offering Strategy:**
    - Random bid selection from outcomes with utility >= bid_threshold (0.9)
    - No time-dependent concession - maintains high demands throughout
    - Appears "crazy" due to random jumping between high-utility bids
    - Does not respond to opponent's behavior in bid selection

    **Acceptance Strategy:**
    - Boulware-like: very reluctant to accept
    - Normal phase (t < 0.95): Accept only if utility >= acceptance_threshold (0.9)
    - Deadline phase (t >= 0.95): Accept if utility >= deadline_threshold (0.7)
    - This prevents negotiation breakdown while maximizing utility

    **Opponent Modeling:**
    - None - IAMcrazyHaggler deliberately ignores opponent behavior
    - The "crazy" appearance is a strategy to confuse opponents
    - Works well against conceding opponents who fear breakdown

    This agent exploits risk-averse opponents who concede to avoid
    breakdown, but performs poorly against similarly stubborn agents.

    Args:
        acceptance_threshold: Minimum utility to accept (default 0.9).
        bid_threshold: Minimum utility for random bid selection (default 0.9).
        deadline_threshold: Acceptance threshold near deadline (default 0.7).
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
        acceptance_threshold: float = 0.9,
        bid_threshold: float = 0.9,
        deadline_threshold: float = 0.7,
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
        self._acceptance_threshold = acceptance_threshold
        self._bid_threshold = bid_threshold
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # High utility bids cache
        self._high_utility_bids: list[Outcome] = []

        # Track opponent for deadline behavior
        self._best_opponent_utility: float = 0.0
        self._best_opponent_offer: Outcome | None = None

    def _initialize(self) -> None:
        """Initialize the outcome space and cache high-utility bids."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)

        # Cache all high-utility bids for random selection
        if self._outcome_space.outcomes:
            max_util = self._outcome_space.max_utility
            # Use relative threshold if max utility is less than our threshold
            effective_threshold = min(self._bid_threshold, max_util * 0.9)

            high_bids = self._outcome_space.get_bids_above(effective_threshold)
            self._high_utility_bids = [bd.bid for bd in high_bids]

            # Fallback: ensure we have at least some bids
            if not self._high_utility_bids:
                # Take top 10% of bids
                n_top = max(1, len(self._outcome_space.outcomes) // 10)
                self._high_utility_bids = [
                    bd.bid for bd in self._outcome_space.outcomes[:n_top]
                ]

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset tracking
        self._best_opponent_utility = 0.0
        self._best_opponent_offer = None

    def _select_random_bid(self) -> Outcome | None:
        """
        Select a random bid from high-utility bids.

        This is the "crazy" part - randomly jumping between
        high-utility bids with no strategic pattern.

        Returns:
            A randomly selected high-utility bid, or None.
        """
        if not self._high_utility_bids:
            # Fallback to best bid
            if self._outcome_space and self._outcome_space.outcomes:
                return self._outcome_space.outcomes[0].bid
            return None

        return random.choice(self._high_utility_bids)

    def _get_acceptance_threshold(self, time: float) -> float:
        """
        Get current acceptance threshold based on time.

        Maintains very high threshold until close to deadline,
        then drops to avoid negotiation breakdown.

        Args:
            time: Normalized time [0, 1].

        Returns:
            Current acceptance threshold.
        """
        if time < 0.95:
            # Stay stubborn most of the time
            return self._acceptance_threshold
        elif time < 0.99:
            # Gradually lower threshold near deadline
            progress = (time - 0.95) / 0.04  # 0 to 1 in time 0.95-0.99
            return self._acceptance_threshold - progress * (
                self._acceptance_threshold - self._deadline_threshold
            )
        else:
            # Very close to deadline: use deadline threshold
            return self._deadline_threshold

    def _should_accept(self, offer: Outcome, time: float) -> bool:
        """
        Decide whether to accept an offer.

        Boulware-like acceptance: only accepts very high utility offers.

        Args:
            offer: The offer to evaluate.
            time: Normalized time [0, 1].

        Returns:
            True if should accept, False otherwise.
        """
        if self.ufun is None:
            return False

        offer_utility = float(self.ufun(offer))

        # Update best opponent offer tracking
        if offer_utility > self._best_opponent_utility:
            self._best_opponent_utility = offer_utility
            self._best_opponent_offer = offer

        # Get current threshold
        threshold = self._get_acceptance_threshold(time)

        # Accept if meets threshold
        if offer_utility >= threshold:
            return True

        # Very close to deadline: consider best opponent offer
        if time > 0.995 and offer_utility >= self._best_opponent_utility * 0.99:
            return True

        return False

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """
        Generate a proposal by randomly selecting from high-utility bids.

        Args:
            state: Current negotiation state.
            dest: Destination negotiator ID (ignored).

        Returns:
            Randomly selected high-utility outcome.
        """
        if not self._initialized:
            self._initialize()

        return self._select_random_bid()

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """
        Respond to an offer using boulware-like acceptance.

        Args:
            state: Current negotiation state.
            source: Source negotiator ID (ignored).

        Returns:
            ResponseType indicating acceptance or rejection.
        """
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self._should_accept(offer, state.relative_time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
