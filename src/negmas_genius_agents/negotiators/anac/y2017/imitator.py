"""Imitator from ANAC 2017."""

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

__all__ = ["Imitator"]


class Imitator(SAONegotiator):
    """
    Imitator from ANAC 2017.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This is a reimplementation of Imitator from ANAC 2017.
    Original: agents.anac.y2017.limitator.Imitator

    Imitator uses a tit-for-tat inspired strategy that mirrors the
    opponent's concession behavior.

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/nego/node/7

    **Offering Strategy:**
        Adjusts concession rate to match opponent's behavior:
        - Fast opponent concession (>0.1): Concede slower (-0.02 per update).
        - Slow opponent concession (>0): Match rate (-0.01 per update).
        - Opponent hardening (<0): Stay patient but prepare (-0.005).
        Falls back to quadratic time-based decay when insufficient data.
        Bids selected from a narrow range around the threshold.

    **Acceptance Strategy:**
        Accepts offers above the adaptive threshold. Late-game (>90%)
        blends threshold toward opponent's best offer. Very late (>98%)
        accepts any offer above minimum utility.

    **Opponent Modeling:**
        Calculates opponent's concession rate from utility change over
        time change in recent offers (last 5). Uses first opponent utility
        as reference point. The concession rate directly determines our
        response behavior through the imitation mechanism.

    Args:
        min_utility: Minimum acceptable utility (default 0.55).
        initial_threshold: Starting threshold (default 0.95).
        late_game_threshold: Time threshold for late game acceleration (default 0.9).
        min_opponent_data: Minimum opponent offers required before imitating
            the opponent's concession rate (default 3).
        opponent_window: Number of recent opponent offers used to compute the
            concession rate (default 5).
        decay_exponent: Exponent of the time-decay used for the fallback
            time-based threshold (default 2.0).
        fast_concession_threshold: Opponent concession rate above which the
            opponent is considered to be conceding fast (default 0.1).
        fast_concession_decrement: Per-update threshold decrement applied when
            the opponent concedes fast (default 0.02).
        slow_concession_decrement: Per-update threshold decrement applied when
            the opponent concedes slowly (default 0.01).
        hardening_decrement: Per-update threshold decrement applied when the
            opponent is hardening (default 0.005).
        hardening_offset: Offset above the time-based threshold used while the
            opponent is hardening (default 0.03).
        stable_decrement: Per-update threshold decrement applied when the
            opponent is stable (default 0.008).
        late_opponent_offset: Offset above the opponent's best utility used in
            the late-game blend (default 0.05).
        bid_range_width: Width of the utility range sampled around the threshold
            when selecting a bid (default 0.05).
        deadline_threshold: Relative time after which any offer above
            ``min_utility`` is accepted (default 0.98).
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
        min_utility: float = 0.55,
        initial_threshold: float = 0.95,
        late_game_threshold: float = 0.9,
        min_opponent_data: int = 3,
        opponent_window: int = 5,
        decay_exponent: float = 2.0,
        fast_concession_threshold: float = 0.1,
        fast_concession_decrement: float = 0.02,
        slow_concession_decrement: float = 0.01,
        hardening_decrement: float = 0.005,
        hardening_offset: float = 0.03,
        stable_decrement: float = 0.008,
        late_opponent_offset: float = 0.05,
        bid_range_width: float = 0.05,
        deadline_threshold: float = 0.98,
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
        self._initial_threshold = initial_threshold
        self._late_game_threshold = late_game_threshold
        self._min_opponent_data = min_opponent_data
        self._opponent_window = opponent_window
        self._decay_exponent = decay_exponent
        self._fast_concession_threshold = fast_concession_threshold
        self._fast_concession_decrement = fast_concession_decrement
        self._slow_concession_decrement = slow_concession_decrement
        self._hardening_decrement = hardening_decrement
        self._hardening_offset = hardening_offset
        self._stable_decrement = stable_decrement
        self._late_opponent_offset = late_opponent_offset
        self._bid_range_width = bid_range_width
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._current_threshold: float = initial_threshold

        # Opponent modeling
        self._opponent_utilities: list[tuple[float, float]] = []  # (time, utility)
        self._opponent_concession_rate: float = 0.0
        self._best_opponent_utility: float = 0.0
        self._first_opponent_utility: float | None = None

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
        self._current_threshold = self._initial_threshold
        self._opponent_utilities = []
        self._opponent_concession_rate = 0.0
        self._best_opponent_utility = 0.0
        self._first_opponent_utility = None

    def _update_opponent_model(self, offer: Outcome, time: float) -> None:
        """Track opponent behavior and compute concession rate."""
        if self.ufun is None:
            return

        offer_utility = float(self.ufun(offer))
        self._opponent_utilities.append((time, offer_utility))
        self._best_opponent_utility = max(self._best_opponent_utility, offer_utility)

        if self._first_opponent_utility is None:
            self._first_opponent_utility = offer_utility

        # Calculate opponent's concession rate
        if len(self._opponent_utilities) >= self._min_opponent_data:
            recent = self._opponent_utilities[-self._opponent_window :]
            utility_change = recent[-1][1] - recent[0][1]
            time_change = recent[-1][0] - recent[0][0]
            if time_change > 0:
                self._opponent_concession_rate = utility_change / time_change

    def _calculate_threshold(self, time: float) -> float:
        """Calculate threshold by imitating opponent's concession."""
        # Default time-based decay
        time_based_threshold = self._initial_threshold - math.pow(
            time, self._decay_exponent
        ) * (self._initial_threshold - self._min_utility)

        # If we have enough data, imitate opponent's concession rate
        if len(self._opponent_utilities) >= self._min_opponent_data:
            # Our concession should mirror opponent's
            if self._opponent_concession_rate > self._fast_concession_threshold:
                # Opponent is conceding fast, we can be more patient
                self._current_threshold = max(
                    self._current_threshold - self._fast_concession_decrement,
                    time_based_threshold,
                )
            elif self._opponent_concession_rate > 0:
                # Opponent is conceding slowly, match rate
                self._current_threshold = max(
                    self._current_threshold - self._slow_concession_decrement,
                    time_based_threshold,
                )
            elif self._opponent_concession_rate < 0:
                # Opponent is hardening, be patient but prepare for late concession
                self._current_threshold = max(
                    self._current_threshold - self._hardening_decrement,
                    time_based_threshold + self._hardening_offset,
                )
            else:
                # Opponent is stable, slight concession
                self._current_threshold = max(
                    self._current_threshold - self._stable_decrement,
                    time_based_threshold,
                )
        else:
            # Not enough data, use time-based threshold
            self._current_threshold = time_based_threshold

        # Late game acceleration
        if time > self._late_game_threshold:
            late_factor = (time - self._late_game_threshold) / (
                1.0 - self._late_game_threshold
            )
            self._current_threshold = min(
                self._current_threshold,
                self._best_opponent_utility
                + self._late_opponent_offset * (1 - late_factor),
            )

        return max(self._current_threshold, self._min_utility)

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Select a bid above threshold."""
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

        # Near deadline
        if time > self._deadline_threshold and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
