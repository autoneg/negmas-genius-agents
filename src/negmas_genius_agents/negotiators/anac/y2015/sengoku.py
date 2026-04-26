"""SENGOKU from ANAC 2015."""

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
    SENGOKU negotiation agent from ANAC 2015.

    SENGOKU (Warring States) uses a battle-inspired strategy with territorial
    defense, tactical concession, and alliance-seeking behavior.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2015.SENGOKU.SENGOKU

    References:
        ANAC 2015 competition:
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
        - Four-phase battle strategy (e=0.1, very Boulware):
          * Defense (t<0.4): Defends territory at 85% utility, prefers
            top 20% of candidates
          * Tactical (0.4<t<0.7): Boulware concession toward 65%
          * Alliance (0.7<t<0.9): Concedes toward max(55%, best_opponent + 0.1)
          * Victory (t>0.9): 70% concession toward max(45%, min_util + 0.1)
        - Alliance bonus: concedes faster (e * 1.2) for cooperative opponents
          (alliance score > 0.5)

    **Acceptance Strategy:**
        - AC_Time: Accepts if offer utility >= computed threshold
        - Victory phase: Accepts if offer >= best opponent utility
          OR offer >= min_utility + 0.1

    **Opponent Modeling:**
        - Tracks opponent bids and best utility
        - Calculates "alliance score" based on average recent offer quality:
          * Increases (+0.1) if avg recent offers > 50%
          * Decreases (-0.05) otherwise
        - Uses alliance score to adjust concession rate

    Args:
        e: Concession exponent (default 0.1, very Boulware)
        defense_time_threshold: Time threshold for defense phase (default 0.4)
        tactical_time_threshold: Time threshold for tactical phase (default 0.7)
        alliance_time_threshold: Time threshold for alliance phase (default 0.9)
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
        defense_time_threshold: float = 0.4,
        tactical_time_threshold: float = 0.7,
        alliance_time_threshold: float = 0.9,
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
        self._defense_time_threshold = defense_time_threshold
        self._tactical_time_threshold = tactical_time_threshold
        self._alliance_time_threshold = alliance_time_threshold
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
        if time < self._defense_time_threshold:
            self._battle_phase = 1  # Defense
        elif time < self._tactical_time_threshold:
            self._battle_phase = 2  # Tactical
        elif time < self._alliance_time_threshold:
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
            progress = (time - self._defense_time_threshold) / (
                self._tactical_time_threshold - self._defense_time_threshold
            )
            f_t = math.pow(progress, 1 / e)
            return self._territory - (self._territory - 0.65) * f_t
        elif self._battle_phase == 3:
            # Alliance phase: seek mutual benefit
            progress = (time - self._tactical_time_threshold) / (
                self._alliance_time_threshold - self._tactical_time_threshold
            )
            target = max(0.55, self._best_opponent_utility + 0.1)
            return 0.65 - (0.65 - target) * progress
        else:
            # Victory phase: seal the deal
            progress = (time - self._alliance_time_threshold) / (
                1.0 - self._alliance_time_threshold
            )
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
