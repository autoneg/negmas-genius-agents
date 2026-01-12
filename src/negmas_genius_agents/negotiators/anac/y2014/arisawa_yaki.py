"""ArisawaYaki from ANAC 2014."""

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

__all__ = ["ArisawaYaki"]


class ArisawaYaki(SAONegotiator):
    """
    ArisawaYaki from ANAC 2014.

    ArisawaYaki employs a unique wave-based oscillation strategy, alternating
    between tougher and more conciliatory phases. This creates uncertainty
    for opponents while maintaining overall concession progress toward deadline.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.ArisawaYaki.ArisawaYaki

    **Offering Strategy:**
        Combines linear base concession with sinusoidal oscillation:
        - Base: linear decay from max to 50% of range over full negotiation
        - Wave: sin(2 * pi * time / period) * amplitude added to base
        - Wave dampening near deadline (t > 0.8): wave *= (1-t) / 0.2

        The oscillation creates periods where the agent appears to harden
        (wave trough) and periods of apparent concession (wave peak),
        potentially extracting concessions from reactive opponents.
        Bids are selected from candidates above the wave-adjusted target,
        with opponent model used to favor mutually beneficial options.

    **Acceptance Strategy:**
        Standard threshold-based acceptance with safety nets:
        1. Accept if offer utility meets wave-adjusted target threshold
        2. Accept if offer matches or exceeds next planned bid utility
        3. Near-deadline acceptance (t > 0.98) for offers above minimum
        The wave pattern affects acceptance threshold, creating windows
        of higher and lower acceptance probability.

    **Opponent Modeling:**
        Frequency-based opponent preference estimation:
        - Tracks value frequencies for each issue from opponent bids
        - Estimates opponent utility as normalized frequency score
        - Uses 70/30 split (own/opponent) for candidate bid scoring
        - Best opponent bid tracked for potential reciprocal offers

    Args:
        wave_period: Time period for one complete wave cycle (default 0.2).
        wave_amplitude: Amplitude of utility oscillation (default 0.1).
        wave_dampen_time: Time threshold when wave dampening begins (default 0.8).
        deadline_acceptance_time: Time threshold for near-deadline acceptance (default 0.98).
        concession_multiplier: Multiplier for base linear concession (default 0.5).
        lowered_threshold_factor: Factor for lowering target when no candidates (default 0.95).
        own_utility_weight: Weight for own utility in bid scoring (default 0.7).
        opponent_utility_weight: Weight for opponent utility in bid scoring (default 0.3).
        top_candidates_divisor: Divisor for selecting top candidates (default 3).
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
        wave_period: float = 0.2,
        wave_amplitude: float = 0.1,
        wave_dampen_time: float = 0.8,
        deadline_acceptance_time: float = 0.98,
        concession_multiplier: float = 0.5,
        lowered_threshold_factor: float = 0.95,
        own_utility_weight: float = 0.7,
        opponent_utility_weight: float = 0.3,
        top_candidates_divisor: int = 3,
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
        self._wave_period = wave_period
        self._wave_amplitude = wave_amplitude
        self._wave_dampen_time = wave_dampen_time
        self._deadline_acceptance_time = deadline_acceptance_time
        self._concession_multiplier = concession_multiplier
        self._lowered_threshold_factor = lowered_threshold_factor
        self._own_utility_weight = own_utility_weight
        self._opponent_utility_weight = opponent_utility_weight
        self._top_candidates_divisor = top_candidates_divisor
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_value_freq: dict[int, dict] = {}

        # State
        self._min_utility: float = 0.5
        self._max_utility: float = 1.0
        self._base_concession: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = max(0.5, self._outcome_space.min_utility)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_bids = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_value_freq = {}
        self._base_concession = 0.0

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent model."""
        if bid is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_bids.append(bid)

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        for i, value in enumerate(bid):
            if i not in self._opponent_value_freq:
                self._opponent_value_freq[i] = {}
            self._opponent_value_freq[i][value] = (
                self._opponent_value_freq[i].get(value, 0) + 1
            )

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent utility using frequency model."""
        if not self._opponent_value_freq:
            return 0.5

        total_score = 0.0
        num_issues = len(bid)

        for i, value in enumerate(bid):
            if i in self._opponent_value_freq:
                freq = self._opponent_value_freq[i].get(value, 0)
                max_freq = max(self._opponent_value_freq[i].values())
                if max_freq > 0:
                    total_score += freq / max_freq

        return total_score / num_issues if num_issues > 0 else 0.5

    def _compute_wave_factor(self, time: float) -> float:
        """Compute wave oscillation factor."""
        # Sinusoidal wave that oscillates around base concession
        wave_count = time / self._wave_period
        wave = math.sin(2 * math.pi * wave_count) * self._wave_amplitude
        return wave

    def _compute_target_utility(self, time: float) -> float:
        """Compute target utility with wave oscillation."""
        # Base linear concession
        base_target = (
            self._max_utility
            - (self._max_utility - self._min_utility)
            * time
            * self._concession_multiplier
        )

        # Add wave component
        wave = self._compute_wave_factor(time)

        # Dampen wave near deadline
        if time > self._wave_dampen_time:
            wave *= (1 - time) / (1.0 - self._wave_dampen_time)

        target = base_target + wave
        return max(self._min_utility, min(self._max_utility, target))

    def _select_bid(self, time: float) -> Outcome | None:
        """Select bid considering wave pattern and opponent preferences."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._compute_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            lowered = target * self._lowered_threshold_factor
            candidates = self._outcome_space.get_bids_above(lowered)
            if not candidates:
                return self._outcome_space.outcomes[0].bid

        # Use opponent model if available
        if self._opponent_value_freq:
            best_bid = None
            best_score = -1.0

            for bd in candidates:
                opp_util = self._estimate_opponent_utility(bd.bid)
                score = (
                    self._own_utility_weight * bd.utility
                    + self._opponent_utility_weight * opp_util
                )
                if score > best_score:
                    best_score = score
                    best_bid = bd.bid

            if best_bid is not None:
                return best_bid

        return random.choice(
            candidates[: max(1, len(candidates) // self._top_candidates_divisor)]
        ).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
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

        time = state.relative_time
        offer_utility = float(self.ufun(offer))
        self._update_opponent_model(offer)

        target = self._compute_target_utility(time)

        # Accept if above target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than our next offer
        next_bid = self._select_bid(time)
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid))
            if offer_utility >= next_utility:
                return ResponseType.ACCEPT_OFFER

        # Near deadline
        if time > self._deadline_acceptance_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
