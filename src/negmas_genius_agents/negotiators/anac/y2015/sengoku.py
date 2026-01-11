"""
SENGOKU from ANAC 2015.

SENGOKU (Warring States) uses a battle-inspired strategy:
1. Strategic territorial defense of high-value outcomes
2. Tactical concession in stages
3. Alliance-seeking through mutual benefit
4. Victory-oriented end-game

Original: agents.anac.y2015.SENGOKU.SENGOKU

References:
    - https://ii.tudelft.nl/negotiation/node/12 (ANAC 2015)
    - Genius negotiation framework: https://ii.tudelft.nl/genius/
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

__all__ = ["SENGOKU"]


class SENGOKU(SAONegotiator):
    """
    SENGOKU from ANAC 2015.

    SENGOKU (Warring States) uses a battle-inspired strategy:
    1. Strategic territorial defense of high-value outcomes
    2. Tactical concession in stages
    3. Alliance-seeking through mutual benefit
    4. Victory-oriented end-game

    Key features:
    - Territory defense (high utility protection)
    - Staged tactical concession
    - Opponent alliance assessment
    - Victory-focused decisions

    Args:
        e: Concession exponent (default 0.1, very Boulware)
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
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._territory: float = 0.85  # Defended utility territory
        self._battle_phase: int = 1

        # Opponent tracking
        self._opponent_bids: list[tuple[Outcome, float]] = []
        self._best_opponent_utility: float = 0.0
        self._opponent_alliance_score: float = 0.0  # How cooperative

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = self._outcome_space.min_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._best_opponent_utility = 0.0
        self._opponent_alliance_score = 0.0
        self._battle_phase = 1
        self._territory = 0.85

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent and assess alliance potential."""
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Assess alliance: is opponent offering good deals?
        if len(self._opponent_bids) >= 3:
            recent_utils = [u for _, u in self._opponent_bids[-5:]]
            avg_recent = sum(recent_utils) / len(recent_utils)

            if avg_recent > 0.5:
                self._opponent_alliance_score = min(
                    1.0, self._opponent_alliance_score + 0.1
                )
            else:
                self._opponent_alliance_score = max(
                    0.0, self._opponent_alliance_score - 0.05
                )

    def _update_battle_phase(self, time: float) -> None:
        """Progress through battle phases."""
        if time < 0.4:
            self._battle_phase = 1  # Defense
        elif time < 0.7:
            self._battle_phase = 2  # Tactical
        elif time < 0.9:
            self._battle_phase = 3  # Alliance
        else:
            self._battle_phase = 4  # Victory

    def _compute_threshold(self, time: float) -> float:
        """Compute threshold based on battle phase."""
        e = self._e

        # Alliance bonus: concede more for cooperative opponent
        if self._opponent_alliance_score > 0.5:
            e *= 1.2

        self._update_battle_phase(time)

        if self._battle_phase == 1:
            # Defense phase: protect territory
            return self._territory
        elif self._battle_phase == 2:
            # Tactical phase: gradual concession
            progress = (time - 0.4) / 0.3
            f_t = math.pow(progress, 1 / e)
            return self._territory - (self._territory - 0.65) * f_t
        elif self._battle_phase == 3:
            # Alliance phase: seek mutual benefit
            progress = (time - 0.7) / 0.2
            target = max(0.55, self._best_opponent_utility + 0.1)
            return 0.65 - (0.65 - target) * progress
        else:
            # Victory phase: seal the deal
            progress = (time - 0.9) / 0.1
            current = max(0.55, self._best_opponent_utility + 0.1)
            target = max(0.45, self._min_utility + 0.1)
            return current - (current - target) * progress * 0.7

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid based on battle strategy."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(threshold * 0.9)

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Defense phase: top bids only
        if self._battle_phase == 1:
            top_n = max(1, len(candidates) // 5)
            return random.choice(candidates[:top_n]).bid

        return random.choice(candidates).bid

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

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        self._update_opponent_model(offer, offer_utility)

        threshold = self._compute_threshold(time)

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        # Victory phase: accept if better than expected
        if self._battle_phase == 4:
            if offer_utility >= max(
                self._best_opponent_utility, self._min_utility + 0.1
            ):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
