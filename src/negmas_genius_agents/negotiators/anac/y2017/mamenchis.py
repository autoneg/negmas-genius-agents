"""Mamenchis from ANAC 2017."""

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

__all__ = ["Mamenchis"]


class Mamenchis(SAONegotiator):
    """
    Mamenchis from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This is a reimplementation of Mamenchis from ANAC 2017.
    Original: agents.anac.y2017.mamenchis.Mamenchis

    Named after the Mamenchisaurus dinosaur, known for its extremely long neck,
    symbolizing the agent's patient, long-reaching negotiation strategy.
    High patience parameter results in very slow early concession.

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/nego/node/7

    **Offering Strategy:**
        Uses power function (time^patience) for threshold decay, where
        high patience (default 3.0) results in very slow initial concession.
        This keeps demands high early in the negotiation, only conceding
        significantly as time runs out. Bids selected from a range around
        the threshold.

    **Acceptance Strategy:**
        Accepts offers above the patient threshold. Adapts based on opponent
        behavior: good opponent concession (>0.2 rate) increases patience
        (+0.02), while opponent hardening (<0 rate after 50% time) triggers
        faster concession (-0.05). Late-game (>95%) considers accepting
        near opponent's best offer.

    **Opponent Modeling:**
        Tracks opponent utility history with timestamps. Calculates
        concession rate from recent offers (last 5). Uses this rate
        to decide whether to maintain patience or accelerate concession.

    Args:
        min_utility: Minimum acceptable utility (default 0.6).
        patience: Controls concession patience, higher = slower (default 3.0).
        late_game_threshold: Time threshold for late game adjustment (default 0.95).
        min_opponent_data: Minimum opponent offers required before adapting
            the threshold to the opponent concession rate (default 3).
        opponent_window: Number of recent opponent offers used to compute the
            concession rate (default 5).
        good_concession_threshold: Opponent concession rate above which the
            opponent is considered to be conceding well (default 0.2).
        patience_boost: Threshold increase applied when the opponent is
            conceding well (default 0.02).
        hardening_time_threshold: Relative time after which a hardening opponent
            triggers faster concession (default 0.5).
        concession_boost: Threshold decrease applied when the opponent is
            hardening (default 0.05).
        late_opponent_offset: Offset above the opponent's best utility used as
            a late-game threshold cap (default 0.02).
        bid_range_width: Width of the utility range sampled around the threshold
            when selecting a bid (default 0.1).
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
        min_utility: float = 0.6,
        patience: float = 3.0,
        late_game_threshold: float = 0.95,
        min_opponent_data: int = 3,
        opponent_window: int = 5,
        good_concession_threshold: float = 0.2,
        patience_boost: float = 0.02,
        hardening_time_threshold: float = 0.5,
        concession_boost: float = 0.05,
        late_opponent_offset: float = 0.02,
        bid_range_width: float = 0.1,
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
        self._min_utility = min_utility
        self._patience = patience
        self._late_game_threshold = late_game_threshold
        self._min_opponent_data = min_opponent_data
        self._opponent_window = opponent_window
        self._good_concession_threshold = good_concession_threshold
        self._patience_boost = patience_boost
        self._hardening_time_threshold = hardening_time_threshold
        self._concession_boost = concession_boost
        self._late_opponent_offset = late_opponent_offset
        self._bid_range_width = bid_range_width
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0

        # Opponent modeling
        self._opponent_utilities: list[tuple[float, float]] = []  # (time, utility)
        self._opponent_concession_rate: float = 0.0
        self._best_opponent_utility: float = 0.0

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
        self._opponent_utilities = []
        self._opponent_concession_rate = 0.0
        self._best_opponent_utility = 0.0

    def _update_opponent_model(self, offer: Outcome, time: float) -> None:
        """Update opponent modeling."""
        if self.ufun is None:
            return

        offer_utility = float(self.ufun(offer))
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)
        self._opponent_utilities.append((time, offer_utility))

        # Calculate opponent's concession rate
        if len(self._opponent_utilities) >= self._min_opponent_data:
            recent = self._opponent_utilities[-self._opponent_window :]
            utility_change = recent[-1][1] - recent[0][1]
            time_change = recent[-1][0] - recent[0][0]
            if time_change > 0:
                self._opponent_concession_rate = utility_change / time_change

    def _calculate_threshold(self, time: float) -> float:
        """Calculate acceptance threshold with patient concession."""
        # Use power function for patient concession (high patience = slower concession)
        concession = math.pow(time, self._patience)

        utility_range = self._max_utility - self._min_utility
        threshold = self._max_utility - concession * utility_range

        # Adapt to opponent's concession rate
        if self._opponent_concession_rate > self._good_concession_threshold:
            # Opponent is conceding well, be more patient
            threshold = min(threshold + self._patience_boost, self._max_utility)
        elif self._opponent_concession_rate < 0 and time > self._hardening_time_threshold:
            # Opponent is hardening, need to concede more
            threshold = max(threshold - self._concession_boost, self._min_utility)

        # Late game adjustment
        if time > self._late_game_threshold:
            threshold = min(threshold, self._best_opponent_utility + self._late_opponent_offset)
            threshold = max(threshold, self._min_utility)

        return max(threshold, self._min_utility)

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Select a bid above the threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        candidates = self._outcome_space.get_bids_in_range(
            threshold, threshold + self._bid_range_width
        )

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold)

        if candidates:
            return random.choice(candidates).bid

        return self._best_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        threshold = self._calculate_threshold(time)

        return self._select_bid(threshold)

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
        self._update_opponent_model(offer, time)

        threshold = self._calculate_threshold(time)
        offer_utility = float(self.ufun(offer))

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
