"""InoxAgent from ANAC 2013."""

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

__all__ = ["InoxAgent"]


class InoxAgent(SAONegotiator):
    """
    InoxAgent from ANAC 2013.

    InoxAgent is a robust negotiation agent designed to be "inoxidable"
    (rustproof) - maintaining strategic integrity under various opponent
    behaviors while protecting its reservation value. The name "Inox" refers to
    stainless steel, symbolizing the agent's resistance to exploitation.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        Original Genius class: ``agents.anac.y2013.InoxAgent.InoxAgent``

        ANAC 2013: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        Boulware-style time-dependent concession with two phases. Early game
        (before late_game_start): uses formula threshold = max - (max - min) *
        (t/late_game_start)^(1/e) * 0.5, conceding only half the range. Late
        game: accelerates concession using (late_progress)^2 from where early
        game ended. Selects bids from the lower third of candidates above
        threshold (closer to threshold = more efficient).

    **Acceptance Strategy:**
        Never accepts below min_utility (reservation value protection). Accepts
        if offer utility >= current threshold. Also accepts if offer >= utility
        of our next proposal. Late game (> 0.95): accepts offers near opponent's
        best (>= 98%). Very late (> 0.995): accepts anything above min_utility.

    **Opponent Modeling:**
        Simple tracking model that records all opponent offer utilities and
        identifies the best opponent bid. Does not estimate opponent preferences
        but uses opponent history for acceptance decisions (accepting near
        opponent's best offer in late game). Focuses on robustness rather than
        sophisticated opponent modeling.

    Args:
        e: Concession exponent controlling Boulware behavior (default 0.1)
        min_utility: Minimum acceptable utility / reservation value (default 0.65)
        late_game_start: Time point to start accelerated concession (default 0.85)
        late_acceptance_threshold: Time after which to accept near opponent's best (default 0.95)
        emergency_acceptance_threshold: Time after which to accept anything above min_utility (default 0.995)
        early_concession_factor: Concession multiplier in early game (default 0.5)
        late_concession_exponent: Exponent for late game concession (default 2)
        bid_selection_divisor: Divisor for selecting from lower portion of candidates (default 3)
        opponent_best_multiplier: Multiplier for opponent's best utility in acceptance (default 0.98)
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
        e: float = 0.1,
        min_utility: float = 0.65,
        late_game_start: float = 0.85,
        late_acceptance_threshold: float = 0.95,
        emergency_acceptance_threshold: float = 0.995,
        early_concession_factor: float = 0.5,
        late_concession_exponent: float = 2,
        bid_selection_divisor: int = 3,
        opponent_best_multiplier: float = 0.98,
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
        self._min_utility = min_utility
        self._late_game_start = late_game_start
        self._late_acceptance_threshold = late_acceptance_threshold
        self._emergency_acceptance_threshold = emergency_acceptance_threshold
        self._early_concession_factor = early_concession_factor
        self._late_concession_exponent = late_concession_exponent
        self._bid_selection_divisor = bid_selection_divisor
        self._opponent_best_multiplier = opponent_best_multiplier
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._opponent_best_bid: Outcome | None = None
        self._opponent_best_utility: float = 0.0
        self._opponent_utilities: list[float] = []
        self._last_proposed_utility: float = 1.0

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
        self._opponent_best_bid = None
        self._opponent_best_utility = 0.0
        self._opponent_utilities = []
        self._last_proposed_utility = self._max_utility

    def _compute_threshold(self, time: float) -> float:
        """Compute utility threshold using Boulware-style concession."""
        # Early game: stay close to max utility
        if time < self._late_game_start:
            # Standard Boulware formula
            if self._e != 0:
                f_t = math.pow(time / self._late_game_start, 1 / self._e)
            else:
                f_t = 0.0

            threshold = self._max_utility - (
                (self._max_utility - self._min_utility)
                * f_t
                * self._early_concession_factor
            )
        else:
            # Late game: accelerate concession
            late_progress = (time - self._late_game_start) / (1 - self._late_game_start)

            # Start from where early game left off
            early_end_threshold = self._max_utility - (
                (self._max_utility - self._min_utility) * self._early_concession_factor
            )

            # Concede towards minimum
            threshold = early_end_threshold - (
                (early_end_threshold - self._min_utility)
                * math.pow(late_progress, self._late_concession_exponent)
            )

        return max(threshold, self._min_utility)

    def _update_opponent_tracking(self, offer: Outcome, utility: float) -> None:
        """Track opponent offers."""
        self._opponent_utilities.append(utility)

        if utility > self._opponent_best_utility:
            self._opponent_best_utility = utility
            self._opponent_best_bid = offer

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid above the current threshold."""
        if self._outcome_space is None:
            return None

        threshold = self._compute_threshold(time)

        # Get bids above threshold
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            # If no candidates above threshold, get best available
            if self._outcome_space.outcomes:
                return self._outcome_space.outcomes[0].bid
            return None

        # Prefer bids close to threshold (more efficient)
        # Sort by utility ascending (so closest to threshold is first)
        candidates_sorted = sorted(candidates, key=lambda x: x.utility)

        # Select from lower portion of candidates (closer to threshold)
        select_range = max(1, len(candidates_sorted) // self._bid_selection_divisor)
        selected = random.choice(candidates_sorted[:select_range])

        self._last_proposed_utility = selected.utility
        return selected.bid

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

        # Update opponent tracking
        self._update_opponent_tracking(offer, offer_utility)

        # Compute current threshold
        threshold = self._compute_threshold(time)

        # Never accept below minimum utility
        if offer_utility < self._min_utility:
            return ResponseType.REJECT_OFFER

        # Accept if above threshold
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than what we'd propose
        if offer_utility >= self._last_proposed_utility:
            return ResponseType.ACCEPT_OFFER

        # Late game: accept opponent's best if reasonable
        if time > self._late_acceptance_threshold:
            if (
                self._opponent_best_bid is not None
                and offer_utility
                >= self._opponent_best_utility * self._opponent_best_multiplier
                and offer_utility >= self._min_utility
            ):
                return ResponseType.ACCEPT_OFFER

        # Very late: accept anything above minimum
        if (
            time > self._emergency_acceptance_threshold
            and offer_utility >= self._min_utility
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
