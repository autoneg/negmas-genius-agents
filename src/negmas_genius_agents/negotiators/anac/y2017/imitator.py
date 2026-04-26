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
        if len(self._opponent_utilities) >= 3:
            recent = self._opponent_utilities[-5:]
            utility_change = recent[-1][1] - recent[0][1]
            time_change = recent[-1][0] - recent[0][0]
            if time_change > 0:
                self._opponent_concession_rate = utility_change / time_change

    def _calculate_threshold(self, time: float) -> float:
        """Calculate threshold by imitating opponent's concession."""
        # Default time-based decay
        time_based_threshold = self._initial_threshold - math.pow(time, 2) * (
            self._initial_threshold - self._min_utility
        )

        # If we have enough data, imitate opponent's concession rate
        if len(self._opponent_utilities) >= 3:
            # Our concession should mirror opponent's
            if self._opponent_concession_rate > 0.1:
                # Opponent is conceding fast, we can be more patient
                self._current_threshold = max(
                    self._current_threshold - 0.02, time_based_threshold
                )
            elif self._opponent_concession_rate > 0:
                # Opponent is conceding slowly, match rate
                self._current_threshold = max(
                    self._current_threshold - 0.01, time_based_threshold
                )
            elif self._opponent_concession_rate < 0:
                # Opponent is hardening, be patient but prepare for late concession
                self._current_threshold = max(
                    self._current_threshold - 0.005, time_based_threshold + 0.03
                )
            else:
                # Opponent is stable, slight concession
                self._current_threshold = max(
                    self._current_threshold - 0.008, time_based_threshold
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
                self._best_opponent_utility + 0.05 * (1 - late_factor),
            )

        return max(self._current_threshold, self._min_utility)

    def _select_bid(self, threshold: float) -> Outcome | None:
        """Select a bid above threshold."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        candidates = self._outcome_space.get_bids_in_range(threshold, threshold + 0.05)

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
        if time > 0.98 and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
