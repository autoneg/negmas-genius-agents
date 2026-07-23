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
        territory: Defended utility territory in defense phase (default 0.85)
        alliance_min_bids: Minimum opponent bids to assess alliance (default 3)
        alliance_recent_window: Number of recent opponent bids for alliance assessment (default 5)
        alliance_threshold: Alliance score/quality threshold for alliance behavior (default 0.5)
        alliance_increment: Increment to alliance score for cooperative offers (default 0.1)
        alliance_decrement: Decrement to alliance score for uncooperative offers (default 0.05)
        alliance_bonus_multiplier: Multiplier applied to e for cooperative opponents (default 1.2)
        tactical_target: Utility target in tactical/alliance phases (default 0.65)
        alliance_target_floor: Floor utility target in alliance/victory phases (default 0.55)
        opponent_best_bonus: Bonus above best opponent utility for alliance/victory target (default 0.1)
        victory_target_floor: Floor utility target in victory phase (default 0.45)
        victory_min_utility_margin: Margin above min utility for victory target (default 0.1)
        victory_concession_factor: Fraction of concession applied in victory phase (default 0.7)
        fallback_threshold_ratio: Ratio used when lowering threshold if no candidates (default 0.9)
        defense_top_fraction: Divisor for selecting top candidates in defense phase (default 5)
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
        territory: float = 0.85,
        alliance_min_bids: int = 3,
        alliance_recent_window: int = 5,
        alliance_threshold: float = 0.5,
        alliance_increment: float = 0.1,
        alliance_decrement: float = 0.05,
        alliance_bonus_multiplier: float = 1.2,
        tactical_target: float = 0.65,
        alliance_target_floor: float = 0.55,
        opponent_best_bonus: float = 0.1,
        victory_target_floor: float = 0.45,
        victory_min_utility_margin: float = 0.1,
        victory_concession_factor: float = 0.7,
        fallback_threshold_ratio: float = 0.9,
        defense_top_fraction: int = 5,
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
        self._territory = territory
        self._alliance_min_bids = alliance_min_bids
        self._alliance_recent_window = alliance_recent_window
        self._alliance_threshold = alliance_threshold
        self._alliance_increment = alliance_increment
        self._alliance_decrement = alliance_decrement
        self._alliance_bonus_multiplier = alliance_bonus_multiplier
        self._tactical_target = tactical_target
        self._alliance_target_floor = alliance_target_floor
        self._opponent_best_bonus = opponent_best_bonus
        self._victory_target_floor = victory_target_floor
        self._victory_min_utility_margin = victory_min_utility_margin
        self._victory_concession_factor = victory_concession_factor
        self._fallback_threshold_ratio = fallback_threshold_ratio
        self._defense_top_fraction = defense_top_fraction
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
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

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Track opponent and assess alliance potential."""
        self._opponent_bids.append((bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        # Assess alliance: is opponent offering good deals?
        if len(self._opponent_bids) >= self._alliance_min_bids:
            recent_utils = [u for _, u in self._opponent_bids[-self._alliance_recent_window :]]
            avg_recent = sum(recent_utils) / len(recent_utils)

            if avg_recent > self._alliance_threshold:
                self._opponent_alliance_score = min(
                    1.0, self._opponent_alliance_score + self._alliance_increment
                )
            else:
                self._opponent_alliance_score = max(
                    0.0, self._opponent_alliance_score - self._alliance_decrement
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
        if self._opponent_alliance_score > self._alliance_threshold:
            e *= self._alliance_bonus_multiplier

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
            return self._territory - (self._territory - self._tactical_target) * f_t
        elif self._battle_phase == 3:
            # Alliance phase: seek mutual benefit
            progress = (time - self._tactical_time_threshold) / (
                self._alliance_time_threshold - self._tactical_time_threshold
            )
            target = max(
                self._alliance_target_floor,
                self._best_opponent_utility + self._opponent_best_bonus,
            )
            return self._tactical_target - (self._tactical_target - target) * progress
        else:
            # Victory phase: seal the deal
            progress = (time - self._alliance_time_threshold) / (
                1.0 - self._alliance_time_threshold
            )
            current = max(
                self._alliance_target_floor,
                self._best_opponent_utility + self._opponent_best_bonus,
            )
            target = max(
                self._victory_target_floor, self._min_utility + self._victory_min_utility_margin
            )
            return current - (current - target) * progress * self._victory_concession_factor

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid based on battle strategy."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._compute_threshold(time)
        candidates = self._outcome_space.get_bids_above(threshold)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(
                threshold * self._fallback_threshold_ratio
            )

        if not candidates:
            return self._outcome_space.outcomes[0].bid

        # Defense phase: top bids only
        if self._battle_phase == 1:
            top_n = max(1, len(candidates) // self._defense_top_fraction)
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
                self._best_opponent_utility, self._min_utility + self._victory_min_utility_margin
            ):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER