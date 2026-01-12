"""AgentLG from ANAC 2012."""

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

__all__ = ["AgentLG"]


class AgentLG(SAONegotiator):
    """
    AgentLG negotiation agent - 2nd place at ANAC 2012.

    AgentLG is a "mostly stubborn but compromises in the end" negotiation agent
    that uses a phased bidding approach with conditional concession based on
    opponent behavior.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        Original Genius class: ``agents.anac.y2012.AgentLG.AgentLG``

        ANAC 2012: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        The agent uses a phased time-dependent bidding strategy:

        - Early phase (t < 0.6): Bids from top 25% of utility space, cycling
          through filtered bids sorted by utility.
        - Compromise phase (0.6 <= t < 0.9): Gradually expands bid pool if
          opponent shows concession (improves offers). Expands every 0.05 time
          units if opponent concedes.
        - Panic phase (0.9 <= t < 0.9995): Expands pool more aggressively every
          0.008 time units, targeting bids above midpoint utility.
        - End phase (t >= 0.9995): Offers opponent's best bid (best offer
          received from opponent).

        Bid selection cycles through available pool with periodic pool expansion.

    **Acceptance Strategy:**
        Multiple acceptance conditions based on time and offer quality:

        - Accept if offer utility >= 99% of last own bid utility.
        - Near deadline (t > 0.999): Accept if offer >= 90% of last own bid.
        - Late game (t > 0.9): Accept if offer >= midpoint between own best
          and opponent's first offer.

    **Opponent Modeling:**
        Uses frequency-based value preference modeling:

        - Tracks value selection frequency per issue to estimate preferences.
        - Monitors opponent's best bid (highest utility for self).
        - Detects opponent concession by comparing max utility improvements.
        - Conditional concession: only expands bid pool if opponent concedes.

    Args:
        initial_filter_fraction: Initial bid filtering fraction (default 0.75).
            Controls what portion of utility range is considered for bidding.
        early_phase_time: Time threshold for early phase bidding (default 0.6).
        panic_phase_time: Time threshold for panic phase bidding (default 0.9).
        late_accept_time: Time threshold for late-game acceptance strategy
            (default 0.9).
        max_bids: Maximum number of bids to consider (default 160000).
        early_expansion_interval: Time interval for bid pool expansion in early
            phase (default 0.1).
        middle_expansion_interval: Time interval for bid pool expansion in middle
            phase (default 0.05).
        panic_expansion_interval: Time interval for bid pool expansion in panic
            phase (default 0.008).
        end_phase_time: Time threshold for offering opponent's best bid
            (default 0.9995).
        accept_ratio: Acceptance ratio relative to own last bid (default 0.99).
        near_deadline_time: Time threshold for near-deadline acceptance
            (default 0.999).
        near_deadline_accept_ratio: Acceptance ratio near deadline (default 0.9).
        fallback_utility_ratio: Fallback utility ratio when no opponent offer
            received (default 0.7).
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
        initial_filter_fraction: float = 0.75,
        early_phase_time: float = 0.6,
        panic_phase_time: float = 0.9,
        late_accept_time: float = 0.9,
        max_bids: int = 160000,
        early_expansion_interval: float = 0.1,
        middle_expansion_interval: float = 0.05,
        panic_expansion_interval: float = 0.008,
        end_phase_time: float = 0.9995,
        accept_ratio: float = 0.99,
        near_deadline_time: float = 0.999,
        near_deadline_accept_ratio: float = 0.9,
        fallback_utility_ratio: float = 0.7,
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
        self._filter_fraction = initial_filter_fraction
        self._early_phase_time = early_phase_time
        self._panic_phase_time = panic_phase_time
        self._late_accept_time = late_accept_time
        self._max_bids = max_bids
        self._early_expansion_interval = early_expansion_interval
        self._middle_expansion_interval = middle_expansion_interval
        self._panic_expansion_interval = panic_expansion_interval
        self._end_phase_time = end_phase_time
        self._accept_ratio = accept_ratio
        self._near_deadline_time = near_deadline_time
        self._near_deadline_accept_ratio = near_deadline_accept_ratio
        self._fallback_utility_ratio = fallback_utility_ratio
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Bid tracking
        self._all_bids: list = []  # Sorted list of acceptable bids
        self._num_possible_bids: int = 0  # Current pool size
        self._bid_index: int = 0  # Current position in pool

        # Opponent modeling
        self._opponent_value_counts: dict[int, dict] = {}  # issue -> {value: count}
        self._max_opponent_bid: Outcome | None = None
        self._max_opponent_utility: float = 0.0
        self._previous_max_utility: float = 0.0

        # Own tracking
        self._my_last_bid: Outcome | None = None
        self._my_last_utility: float = 1.0
        self._last_concession_time: float = 0.0
        self._my_best_utility: float = 1.0
        self._opp_first_utility: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space and filtered bids."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if not self._outcome_space.outcomes:
            return

        self._my_best_utility = self._outcome_space.max_utility

        # Filter bids based on fraction
        self._filter_bids()
        self._initialized = True

    def _filter_bids(self) -> None:
        """Filter bids to keep only acceptable ones."""
        if self._outcome_space is None:
            return

        opp_best = self._opp_first_utility if self._opp_first_utility > 0 else 0.0
        down_bond = (
            self._my_best_utility
            - (self._my_best_utility - opp_best) * self._filter_fraction
        )

        self._all_bids = [
            bd for bd in self._outcome_space.outcomes if bd.utility >= down_bond
        ]

        # Limit to reasonable size
        while len(self._all_bids) > self._max_bids:
            self._filter_fraction *= 0.85
            down_bond = (
                self._my_best_utility
                - (self._my_best_utility - opp_best) * self._filter_fraction
            )
            self._all_bids = [
                bd for bd in self._outcome_space.outcomes if bd.utility >= down_bond
            ]

        self._num_possible_bids = max(1, len(self._all_bids) // 4)  # Start with top 25%
        self._bid_index = 0

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._bid_index = 0
        self._opponent_value_counts = {}
        self._max_opponent_bid = None
        self._max_opponent_utility = 0.0
        self._previous_max_utility = 0.0
        self._my_last_bid = None
        self._my_last_utility = 1.0
        self._last_concession_time = 0.0

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update frequency-based opponent model."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))

        # Track best opponent bid
        if utility > self._max_opponent_utility:
            self._max_opponent_utility = utility
            self._max_opponent_bid = bid

        if self._opp_first_utility == 0.0:
            self._opp_first_utility = utility

        # Update value frequencies
        for i, value in enumerate(bid):
            if i not in self._opponent_value_counts:
                self._opponent_value_counts[i] = {}
            self._opponent_value_counts[i][value] = (
                self._opponent_value_counts[i].get(value, 0) + 1
            )

    def _expand_bid_pool(self, time: float) -> None:
        """Expand bid pool based on time and opponent concession."""
        if not self._all_bids:
            return

        # Check if opponent conceded
        opponent_conceded = self._max_opponent_utility > self._previous_max_utility
        self._previous_max_utility = self._max_opponent_utility

        # Time-based expansion
        if time < self._early_phase_time:
            # Early phase: expand every early_expansion_interval time
            if time - self._last_concession_time > self._early_expansion_interval:
                self._num_possible_bids = min(
                    len(self._all_bids),
                    self._num_possible_bids + len(self._all_bids) // 10,
                )
                self._last_concession_time = time
        elif time < self._panic_phase_time:
            # Middle phase: expand every middle_expansion_interval if opponent conceded
            if (
                time - self._last_concession_time > self._middle_expansion_interval
                and opponent_conceded
            ):
                self._num_possible_bids = min(
                    len(self._all_bids),
                    self._num_possible_bids + len(self._all_bids) // 20,
                )
                self._last_concession_time = time
        else:
            # Panic phase: expand more aggressively
            if time - self._last_concession_time > self._panic_expansion_interval:
                floor = (self._my_best_utility + self._opp_first_utility) / 2
                for i, bd in enumerate(self._all_bids):
                    if bd.utility < floor:
                        self._num_possible_bids = min(i + 1, len(self._all_bids))
                        break
                else:
                    self._num_possible_bids = len(self._all_bids)
                self._last_concession_time = time

    def _select_bid(self, time: float) -> Outcome | None:
        """Select next bid to offer."""
        if not self._all_bids:
            return None

        # Very end: offer opponent's best bid
        if time > self._end_phase_time and self._max_opponent_bid is not None:
            return self._max_opponent_bid

        self._expand_bid_pool(time)

        # Cycle through pool
        if self._bid_index >= self._num_possible_bids:
            self._bid_index = 0

        bid = self._all_bids[self._bid_index].bid
        self._bid_index += 1
        return bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        bid = self._select_bid(time)

        if bid is not None and self.ufun is not None:
            self._my_last_bid = bid
            self._my_last_utility = float(self.ufun(bid))

        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        self._update_opponent_model(offer)
        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        # Multiple acceptance conditions
        # 1. Offer is nearly as good as our last bid
        if offer_utility >= self._my_last_utility * self._accept_ratio:
            return ResponseType.ACCEPT_OFFER

        # 2. Near deadline with near_deadline_accept_ratio of our bid value
        if (
            time > self._near_deadline_time
            and offer_utility
            >= self._my_last_utility * self._near_deadline_accept_ratio
        ):
            return ResponseType.ACCEPT_OFFER

        # 3. Offer meets minimum threshold
        min_threshold = (
            (self._my_best_utility + self._opp_first_utility) / 2
            if self._opp_first_utility > 0
            else self._my_best_utility * self._fallback_utility_ratio
        )
        if time > self._late_accept_time and offer_utility >= min_threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
