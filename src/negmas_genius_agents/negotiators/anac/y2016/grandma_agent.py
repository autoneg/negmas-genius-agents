"""GrandmaAgent from ANAC 2016."""

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

__all__ = ["GrandmaAgent"]


class GrandmaAgent(SAONegotiator):
    """
    GrandmaAgent negotiation agent from ANAC 2016.

    GrandmaAgent is a patient and conservative negotiation agent that takes
    time to understand opponent behavior before making significant concessions.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2016.grandma.GrandmaAgent

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
    Two-phase patience-based concession:

    - Patience phase (t < patience, default 0.85): Minimal concession,
      only 10% of the way toward reservation value regardless of time
      progression within this phase
    - End-game phase (t >= patience): Quadratic concession acceleration
      from patience-phase end point to reservation value

    End-game target adjusted based on opponent behavior:
    - Opponent conceding: targets reservation + 5%
    - Opponent not conceding: targets reservation + 10%

    Bids selected with 60% own utility + 40% estimated opponent utility
    weighting, favoring stable, mutually acceptable outcomes.

    **Acceptance Strategy:**
    Conservative acceptance with multiple criteria:

    - Accepts if offer utility meets or exceeds current target threshold
    - Accepts if offer utility >= agent's last offered bid utility
    - Near deadline (t >= 0.95): accepts any offer above reservation value

    **Opponent Modeling:**
    Behavior pattern detection with preference estimation:

    - Tracks value frequencies for each issue from opponent bids
    - Estimates opponent utility based on normalized value frequencies
    - Detects opponent concession pattern by comparing recent vs
      earlier bid utilities (moving average comparison)
    - Concession detection affects end-game target adjustment

    The patient approach allows gathering information about opponent
    preferences before committing to significant concessions.

    Args:
        patience: Patience factor - fraction of time before accelerated
            concession begins (default 0.85, higher = more patient)
        min_utility: Minimum acceptable utility threshold (default 0.65)
        early_time: Time threshold for early phase best-bid offering (default 0.02)
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
        patience: float = 0.85,
        min_utility: float = 0.65,
        early_time: float = 0.02,
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
        self._patience = patience
        self._min_utility = min_utility
        self._early_time = early_time
        self._deadline_time = deadline_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._reservation_value: float = min_utility

        # Opponent tracking
        self._opponent_bids: list[Outcome] = []
        self._opponent_utilities: list[float] = []
        self._opponent_value_frequencies: dict[int, dict[str, int]] = {}
        self._opponent_conceding: bool = False

        # Tracking
        self._best_received_utility: float = 0.0
        self._best_received_bid: Outcome | None = None
        self._last_offered_utility: float = 1.0

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
        self._opponent_bids = []
        self._opponent_utilities = []
        self._opponent_value_frequencies = {}
        self._opponent_conceding = False
        self._best_received_utility = 0.0
        self._best_received_bid = None
        self._last_offered_utility = self._max_utility

    def _update_opponent_model(self, bid: Outcome, utility: float) -> None:
        """Update opponent model based on received bid."""
        if bid is None:
            return

        self._opponent_bids.append(bid)
        self._opponent_utilities.append(utility)

        # Track best received
        if utility > self._best_received_utility:
            self._best_received_utility = utility
            self._best_received_bid = bid

        # Track value frequencies
        for i, value in enumerate(bid):
            if i not in self._opponent_value_frequencies:
                self._opponent_value_frequencies[i] = {}

            value_str = str(value)
            if value_str not in self._opponent_value_frequencies[i]:
                self._opponent_value_frequencies[i][value_str] = 0
            self._opponent_value_frequencies[i][value_str] += 1

        # Detect if opponent is conceding
        if len(self._opponent_utilities) >= 5:
            recent = self._opponent_utilities[-3:]
            earlier = (
                self._opponent_utilities[-6:-3]
                if len(self._opponent_utilities) >= 6
                else self._opponent_utilities[:3]
            )
            self._opponent_conceding = sum(recent) / len(recent) > sum(earlier) / len(
                earlier
            )

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent's utility for a bid."""
        if not self._opponent_value_frequencies:
            return 0.5

        n_issues = len(bid)
        total = 0.0

        for i, value in enumerate(bid):
            value_str = str(value)
            freq_map = self._opponent_value_frequencies.get(i, {})

            if not freq_map:
                total += 0.5 / n_issues
                continue

            count = freq_map.get(value_str, 0)
            max_count = max(freq_map.values()) if freq_map else 1
            total += (count / max_count if max_count > 0 else 0.5) / n_issues

        return total

    def _get_target_utility(self, time: float) -> float:
        """Calculate target utility with patient concession."""
        # Patient phase: very slow concession
        if time < self._patience:
            # Minimal concession during patience phase
            progress = time / self._patience
            target = self._max_utility - 0.1 * progress * (
                self._max_utility - self._reservation_value
            )
            return max(target, self._reservation_value)

        # End-game phase: accelerated concession
        progress = (time - self._patience) / (1.0 - self._patience)
        start = self._max_utility - 0.1 * (self._max_utility - self._reservation_value)

        # Faster concession if opponent is also conceding
        if self._opponent_conceding:
            end = self._reservation_value + 0.05
        else:
            end = self._reservation_value + 0.1

        # Quadratic concession in end-game
        f_t = progress**2
        target = start - (start - end) * f_t

        return max(target, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid above target utility."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            for bd in self._outcome_space.outcomes:
                if bd.utility >= self._reservation_value:
                    candidates = [bd]
                    break

        if not candidates:
            return self._best_bid

        # Score considering opponent utility
        scored: list[tuple[Outcome, float]] = []
        for bd in candidates:
            opp_util = self._estimate_opponent_utility(bd.bid)
            # Grandma prefers stable, mutually acceptable outcomes
            score = 0.6 * bd.utility + 0.4 * opp_util
            scored.append((bd.bid, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Pick from top candidates
        top_n = min(3, len(scored))
        selected = random.choice(scored[:top_n])
        return selected[0]

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        # Very early: best bid
        if time < self._early_time:
            return self._best_bid

        bid = self._select_bid(time)

        if bid is not None and self.ufun is not None:
            self._last_offered_utility = float(self.ufun(bid))

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

        offer_utility = float(self.ufun(offer))
        self._update_opponent_model(offer, offer_utility)

        time = state.relative_time
        target = self._get_target_utility(time)

        # Accept if meets target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than what we would offer
        if offer_utility >= self._last_offered_utility:
            return ResponseType.ACCEPT_OFFER

        # Near deadline
        if time >= self._deadline_time and offer_utility >= self._reservation_value:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
