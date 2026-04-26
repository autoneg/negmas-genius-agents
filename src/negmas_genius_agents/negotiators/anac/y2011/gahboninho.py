"""Gahboninho from ANAC 2011."""

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

__all__ = ["Gahboninho"]


class Gahboninho(SAONegotiator):
    """
    Gahboninho from ANAC 2011 - 2nd place agent.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    The name "Gahboninho" means "bully" in Hebrew, reflecting its aggressive
    negotiation strategy that exploits cooperative opponents while remaining
    robust against tough ones.

    This implementation reproduces Gahboninho's core strategies:

    - Three-phase negotiation: early profiling, main exploitation, and panic concession
    - Noise-based opponent classification (lower noise = more cooperative opponent)
    - Exploitation mechanism that becomes more selfish against nicer opponents
    - Rapid concession in final moments with willingness to offer opponent's best bid

    References:
        Original Genius class: ``agents.anac.y2011.Gahboninho.Gahboninho``

        Ben Adar, M., Sofy, N., Elimelech, A. (2013). Gahboninho: Strategy for
        Balancing Pressure and Compromise in Automated Negotiation. In: Ito, T.,
        Zhang, M., Robu, V., Matsuo, T. (eds) Complex Automated Negotiations:
        Theories, Models, and Software Competitions. Studies in Computational
        Intelligence, vol 435. Springer, Berlin, Heidelberg.
        https://doi.org/10.1007/978-3-642-30737-9_13

        .. code-block:: bibtex

            @incollection{benadar2013gahboninho,
                title={Gahboninho: Strategy for Balancing Pressure and Compromise
                       in Automated Negotiation},
                author={Ben Adar, Mai and Sofy, Nadav and Elimelech, Avshalom},
                booktitle={Complex Automated Negotiations: Theories, Models, and
                           Software Competitions},
                pages={205--208},
                year={2013},
                publisher={Springer},
                doi={10.1007/978-3-642-30737-9_13}
            }

    **Offering Strategy:**
    - Early phase (first 40 actions, t < 0.15): Gradual decrease from 1.0 to 0.925
    - Main phase: Target based on noise estimation and time-dependent bounds
    - Rate-limited concession: max decrease of 0.0009 per step normally
    - Panic mode (t > 0.985): Rapid concession with increased rate limits
    - Frenzy mode (t > 0.9996): Offers opponent's best bid directly

    **Acceptance Strategy:**
    - Early phase: Accept if offer utility > 0.95
    - Normal: Accept if offer utility >= compromising_factor * target
    - Compromising factor decreases through phases (0.95 -> 0.90)

    **Opponent Modeling:**
    - Tracks value frequencies per issue to build opponent profile
    - Maintains noise estimate measuring opponent "toughness" (0=nice, 1=tough)
    - Updates noise every 20 bids based on best utility improvement
    - Lower noise (nicer opponent) triggers more selfish behavior
    - Tracks best opponent bid utility to adjust phase thresholds

    Args:
        initial_noise: Starting estimate of opponent toughness (0=nice, 1=tough)
        compromising_factor: Multiplier for acceptance threshold
        early_phase_actions: Number of actions in early phase (default 40)
        early_phase_time: Time threshold for early phase (default 0.15)
        early_phase_min_util: Minimum utility at end of early phase (default 0.925)
        noise_update_interval: Number of bids between noise updates (default 20)
        noise_decrease_max: Maximum noise decrease rate (default 0.015)
        noise_decrease_min: Minimum noise decrease rate (default 0.003)
        noise_decrease_base: Base noise decrease rate (default 0.01)
        noise_domain_ref: Reference domain size for noise calculation (default 400)
        phase1_time: Time threshold for phase 1 (default 0.85)
        phase1_min_util: Minimum utility factor for phase 1 (default 0.9125)
        phase2_time: Time threshold for phase 2 (default 0.92)
        phase2_min_util: Minimum utility factor for phase 2 (default 0.84)
        phase3_time: Time threshold for phase 3 (default 0.94)
        phase3_min_util: Minimum utility factor for phase 3 (default 0.775)
        phase4_time: Time threshold for phase 4 (default 0.985)
        phase4_min_util: Minimum utility factor for phase 4 (default 0.7)
        panic_time: Time threshold for panic mode (default 0.9996)
        panic_min_util: Minimum utility factor for panic mode (default 0.5)
        base_rate_limit: Base maximum decrease per step (default 0.0009)
        panic_rate_limit_base: Base rate limit in panic mode (default 0.001)
        panic_rate_limit_factor: Rate limit factor in panic mode (default 0.01)
        panic_rate_limit_divisor: Divisor for panic rate limit (default 0.015)
        bid_selection_factor: Factor for bid selection threshold (default 0.95)
        early_accept_threshold: Early phase acceptance threshold (default 0.95)
        min_target_utility: Minimum target utility floor (default 0.5)
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
        initial_noise: float = 0.4,
        compromising_factor: float = 0.95,
        early_phase_actions: int = 40,
        early_phase_time: float = 0.15,
        early_phase_min_util: float = 0.925,
        noise_update_interval: int = 20,
        noise_decrease_max: float = 0.015,
        noise_decrease_min: float = 0.003,
        noise_decrease_base: float = 0.01,
        noise_domain_ref: int = 400,
        phase1_time: float = 0.85,
        phase1_min_util: float = 0.9125,
        phase2_time: float = 0.92,
        phase2_min_util: float = 0.84,
        phase3_time: float = 0.94,
        phase3_min_util: float = 0.775,
        phase4_time: float = 0.985,
        phase4_min_util: float = 0.7,
        panic_time: float = 0.9996,
        panic_min_util: float = 0.5,
        base_rate_limit: float = 0.0009,
        panic_rate_limit_base: float = 0.001,
        panic_rate_limit_factor: float = 0.01,
        panic_rate_limit_divisor: float = 0.015,
        bid_selection_factor: float = 0.95,
        early_accept_threshold: float = 0.95,
        min_target_utility: float = 0.5,
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
        self._initial_noise = initial_noise
        self._compromising_factor = compromising_factor
        self._early_phase_actions = early_phase_actions
        self._early_phase_time = early_phase_time
        self._early_phase_min_util = early_phase_min_util
        self._noise_update_interval = noise_update_interval
        self._noise_decrease_max = noise_decrease_max
        self._noise_decrease_min = noise_decrease_min
        self._noise_decrease_base = noise_decrease_base
        self._noise_domain_ref = noise_domain_ref
        self._phase1_time = phase1_time
        self._phase1_min_util = phase1_min_util
        self._phase2_time = phase2_time
        self._phase2_min_util = phase2_min_util
        self._phase3_time = phase3_time
        self._phase3_min_util = phase3_min_util
        self._phase4_time = phase4_time
        self._phase4_min_util = phase4_min_util
        self._panic_time = panic_time
        self._panic_min_util = panic_min_util
        self._base_rate_limit = base_rate_limit
        self._panic_rate_limit_base = panic_rate_limit_base
        self._panic_rate_limit_factor = panic_rate_limit_factor
        self._panic_rate_limit_divisor = panic_rate_limit_divisor
        self._bid_selection_factor = bid_selection_factor
        self._early_accept_threshold = early_accept_threshold
        self._min_target_utility = min_target_utility
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._noise: float = initial_noise
        self._first_actions: int = 0
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._opponent_bid_count: int = 0
        self._last_offer_utility: float = 1.0
        self._in_frenzy: bool = False

        # Opponent modeling
        self._opponent_issue_frequencies: dict[int, dict] = {}
        self._previous_similarity: float = 0.0

    def _initialize(self) -> None:
        """Initialize the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._noise = self._initial_noise
        self._first_actions = 0
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._opponent_bid_count = 0
        self._last_offer_utility = 1.0
        self._in_frenzy = False
        self._opponent_issue_frequencies = {}
        self._previous_similarity = 0.0

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update frequency-based opponent model."""
        if bid is None:
            return

        self._opponent_bid_count += 1
        bid_utility = float(self.ufun(bid)) if self.ufun else 0.0

        if bid_utility > self._best_opponent_utility:
            self._best_opponent_utility = bid_utility
            self._best_opponent_bid = bid

        # Track value frequencies per issue
        for i, value in enumerate(bid):
            if i not in self._opponent_issue_frequencies:
                self._opponent_issue_frequencies[i] = {}
            freq_dict = self._opponent_issue_frequencies[i]
            freq_dict[value] = freq_dict.get(value, 0) + 1

    def _update_noise(self, time: float) -> None:
        """Update noise estimate based on opponent behavior."""
        if (
            self._opponent_bid_count < self._noise_update_interval
            or self._opponent_bid_count % self._noise_update_interval != 0
        ):
            return

        # Estimate domain size for noise decrease rate
        domain_size = len(self._outcome_space.outcomes) if self._outcome_space else 1000
        noise_decrease_rate = min(
            self._noise_decrease_max,
            max(
                self._noise_decrease_min,
                self._noise_decrease_base * domain_size / self._noise_domain_ref,
            ),
        )

        # Check if opponent is conceding
        if self._best_opponent_utility > self._previous_similarity:
            # Opponent gave us something better - they're nice
            self._noise = max(0, self._noise - noise_decrease_rate)
        else:
            # Opponent being tough
            self._noise = min(1, self._noise + noise_decrease_rate)

        self._previous_similarity = self._best_opponent_utility

    def _get_target_utility(self, time: float) -> float:
        """Calculate next recommended offer utility."""
        # Discount factor (assume 1.0 if not available)
        df = 1.0

        # Early phase: gradual decrease from 1.0 to early_phase_min_util
        if (
            self._first_actions < self._early_phase_actions
            and time < self._early_phase_time
        ):
            util_decrease = (
                1.0 - self._early_phase_min_util
            ) / self._early_phase_actions
            return self._early_phase_min_util + util_decrease * (
                self._early_phase_actions - self._first_actions
            )

        # Main phase calculation
        min_util = math.pow(df, 2 * time)
        max_util = min_util * (1 + 6 * self._noise * math.pow(df, 5))

        # Phase-dependent adjustments
        if time < self._phase1_time * df:
            min_util *= max(self._best_opponent_utility, self._phase1_min_util)
            self._compromising_factor = 0.95
        elif time <= self._phase2_time * df:
            min_util *= max(self._best_opponent_utility, self._phase2_min_util)
            self._compromising_factor = 0.94
        elif time <= self._phase3_time * df:
            min_util *= max(self._best_opponent_utility, self._phase3_min_util)
            self._compromising_factor = 0.93
        elif time <= self._phase4_time:
            min_util *= max(self._best_opponent_utility, self._phase4_min_util)
            self._compromising_factor = 0.91
        elif time <= self._panic_time:
            # Rapid concession allowed
            min_util *= max(self._best_opponent_utility, self._panic_min_util)
            self._compromising_factor = 0.90
        else:
            # Frenzy mode
            self._in_frenzy = True

        # Calculate target with smoothing
        target = min(1.0, max_util - (max_util - min_util) * time)

        # Rate limit changes
        max_decrease = self._base_rate_limit
        if time > self._phase4_time:
            max_decrease = (
                self._panic_rate_limit_base
                + self._panic_rate_limit_factor
                * (time - self._phase4_time)
                / self._panic_rate_limit_divisor
            )

        if self._last_offer_utility - target > max_decrease:
            target = self._last_offer_utility - max_decrease

        return max(self._min_target_utility, target)

    def _select_bid(self, target: float) -> Outcome | None:
        """Select a bid near the target utility."""
        if self._outcome_space is None:
            return None

        # In frenzy mode, offer best opponent bid
        if self._in_frenzy and self._best_opponent_bid is not None:
            return self._best_opponent_bid

        # Find bid near target
        for bd in self._outcome_space.outcomes:
            if bd.utility >= target:
                return bd.bid
            if bd.utility < target * self._bid_selection_factor:
                break

        # Fallback to best
        if self._outcome_space.outcomes:
            return self._outcome_space.outcomes[0].bid
        return None

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        self._first_actions += 1
        self._update_noise(time)

        target = self._get_target_utility(time)
        bid = self._select_bid(target)

        if bid is not None and self.ufun:
            self._last_offer_utility = float(self.ufun(bid))

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

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        self._update_opponent_model(offer)
        self._update_noise(time)

        # Early phase acceptance
        if (
            self._first_actions < self._early_phase_actions
            and offer_utility > self._early_accept_threshold
        ):
            return ResponseType.ACCEPT_OFFER

        # Normal acceptance
        target = self._get_target_utility(time)
        threshold = self._compromising_factor * target

        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
