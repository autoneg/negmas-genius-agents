"""MaxOops from ANAC 2016."""

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

__all__ = ["MaxOops"]


class MaxOops(SAONegotiator):
    """
    MaxOops negotiation agent from ANAC 2016.

    MaxOops is an aggressive negotiation agent that aims to maximize utility
    while featuring recovery mechanisms for adapting to difficult negotiation
    situations.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2016.maxoops.MaxOops

    References:
        .. code-block:: bibtex

            @inproceedings{fujita2016anac,
                title={The Sixth Automated Negotiating Agents Competition (ANAC 2016)},
                author={Fujita, Katsuhide and others},
                booktitle={Proceedings of the International Joint Conference on
                    Artificial Intelligence (IJCAI)},
                year={2016}
            }

        ANAC 2016 Competition: https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
    Mode-dependent aggressive strategy with recovery:

    Normal aggressive mode (three phases):
    - Phase 1 (t < 0.2): Very aggressive at 98% of max utility
    - Phase 2 (0.2 <= t < 0.8): Boulware concession using low exponent
      (default 0.1) from 98% to 75%
    - Phase 3 (t >= 0.8): Linear concession to reservation value

    Recovery mode (activated when opponent is not conceding and best
    offers are poor):
    - More flexible threshold based on best received offer
    - Bid selection prefers bids closer to target (ascending sort)
    - Enables reaching agreement in difficult negotiations

    Recovery mode triggers when: opponent utilities not improving over
    10 bids AND best received utility < 0.7

    **Acceptance Strategy:**
    Mode-aware acceptance:

    - Accepts if offer utility meets or exceeds current target threshold
    - In recovery mode: accepts if offer meets recovery threshold
      (98% of best received or reservation value, whichever is higher)
    - End-game (t >= 0.95): accepts any offer above reservation value

    **Opponent Modeling:**
    Trend analysis for recovery detection:

    - Tracks opponent bid utilities over time
    - Compares recent (last 5) vs earlier (previous 5) average utilities
    - Detects stagnation when recent <= earlier and best offer < 0.7
    - Triggers recovery mode to adapt strategy
    - Maintains best received bid for recovery threshold calculation

    Args:
        aggression: Aggression level - concession exponent (default 0.1,
            lower = more aggressive with slower concession)
        min_utility: Minimum acceptable utility threshold (default 0.5)
        phase1_end: End time for phase 1 aggressive phase (default 0.2)
        phase2_end: End time for phase 2 Boulware phase (default 0.8)
        recovery_phase_end: End time for recovery phase flexibility (default 0.9)
        early_time: Time threshold for early phase best-bid offering (default 0.05)
        deadline_time: Time threshold for deadline acceptance (default 0.95)
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
        aggression: float = 0.1,
        min_utility: float = 0.5,
        phase1_end: float = 0.2,
        phase2_end: float = 0.8,
        recovery_phase_end: float = 0.9,
        early_time: float = 0.05,
        deadline_time: float = 0.95,
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
        self._aggression = aggression
        self._min_utility = min_utility
        self._phase1_end = phase1_end
        self._phase2_end = phase2_end
        self._recovery_phase_end = recovery_phase_end
        self._early_time = early_time
        self._deadline_time = deadline_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._reservation_value: float = min_utility

        # Tracking
        self._opponent_utilities: list[float] = []
        self._best_received_utility: float = 0.0
        self._best_received_bid: Outcome | None = None
        self._own_bid_utilities: list[float] = []

        # Recovery state
        self._in_recovery_mode: bool = False
        self._recovery_threshold: float = 0.0

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

        # Reset state
        self._opponent_utilities = []
        self._best_received_utility = 0.0
        self._best_received_bid = None
        self._own_bid_utilities = []
        self._in_recovery_mode = False
        self._recovery_threshold = 0.0

    def _update_opponent(self, utility: float, bid: Outcome) -> None:
        """Track opponent and check if recovery needed."""
        self._opponent_utilities.append(utility)

        if utility > self._best_received_utility:
            self._best_received_utility = utility
            self._best_received_bid = bid

        # Check if we need recovery (opponent not conceding and we're behind)
        if len(self._opponent_utilities) >= 10:
            recent = self._opponent_utilities[-5:]
            earlier = self._opponent_utilities[-10:-5]
            recent_avg = sum(recent) / len(recent)
            earlier_avg = sum(earlier) / len(earlier)

            # If opponent is not conceding and best offers are poor
            if recent_avg <= earlier_avg and self._best_received_utility < 0.7:
                self._in_recovery_mode = True
                self._recovery_threshold = max(
                    self._reservation_value, self._best_received_utility * 0.98
                )

    def _get_target_utility(self, time: float) -> float:
        """Aggressive target with recovery adjustment."""
        if self._in_recovery_mode:
            # In recovery: be more flexible
            if time < self._recovery_phase_end:
                return max(self._recovery_threshold, self._reservation_value + 0.1)
            else:
                return self._reservation_value

        # Normal aggressive mode
        if time < self._phase1_end:
            return self._max_utility * 0.98
        elif time < self._phase2_end:
            phase_time = (time - self._phase1_end) / (
                self._phase2_end - self._phase1_end
            )
            f_t = (
                math.pow(phase_time, 1 / self._aggression)
                if self._aggression > 0
                else phase_time
            )
            start = self._max_utility * 0.98
            end = self._max_utility * 0.75
            return start - (start - end) * f_t
        else:
            # End-game
            phase_time = (time - self._phase2_end) / (1.0 - self._phase2_end)
            start = self._max_utility * 0.75
            end = self._reservation_value
            return max(start - (start - end) * phase_time, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Aggressive bid selection."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            candidates = self._outcome_space.get_bids_above(target * 0.9)

        if not candidates:
            return self._best_bid

        # In recovery mode, prefer bids closer to target
        if self._in_recovery_mode:
            # Sort by utility ascending (closer to target)
            candidates_sorted = sorted(candidates, key=lambda x: x.utility)
            n = min(3, len(candidates_sorted))
            bid = random.choice(candidates_sorted[:n]).bid
        else:
            # Normal: pick from top
            n = min(5, len(candidates))
            bid = random.choice(candidates[:n]).bid

        if self.ufun is not None:
            self._own_bid_utilities.append(float(self.ufun(bid)))

        return bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate an aggressive proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        if time < self._early_time:
            return self._best_bid

        return self._select_bid(time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond with recovery awareness."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        self._update_opponent(offer_utility, offer)

        target = self._get_target_utility(time)

        # Accept if meets target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Recovery mode: be more accepting
        if self._in_recovery_mode and offer_utility >= self._recovery_threshold:
            return ResponseType.ACCEPT_OFFER

        # End-game acceptance
        if time >= self._deadline_time and offer_utility >= self._reservation_value:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
