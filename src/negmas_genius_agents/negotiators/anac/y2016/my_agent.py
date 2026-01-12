"""MyAgent from ANAC 2016."""

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

__all__ = ["MyAgent"]


class MyAgent(SAONegotiator):
    """
    MyAgent (Rubick) negotiation agent from ANAC 2016 - 3rd place.

    MyAgent (also known as Rubick) achieved 3rd place in ANAC 2016. The agent
    uses Nash equilibrium estimation combined with multi-phase time-dependent
    concession and comprehensive opponent modeling.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    Original Java class: agents.anac.y2016.myagent.MyAgent

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
    Three-phase concession toward estimated Nash point:

    - Phase 1 (t < 0.2): Conservative, linear from max to 95% of max utility
    - Phase 2 (0.2 <= t < 0.8): Boulware concession (beta=0.3) from 95%
      toward Nash utility or reservation value (whichever is higher)
    - Phase 3 (t >= 0.8): Linear concession from Nash toward reservation,
      with adjustment if opponent concession detected

    Bid selection uses time-varying weighted scoring:
    - Own weight decreases from 1.0 to 0.7 over time
    - Opponent weight increases from 0.0 to 0.3 over time
    - Gradually shifts from self-interest to cooperation

    Nash point estimated by maximizing product of own utility and
    estimated opponent utility across all outcomes.

    **Acceptance Strategy:**
    Multi-criteria acceptance with Nash reference:

    - Accepts if offer utility meets or exceeds target threshold
    - Accepts if offer utility >= agent's last offered bid utility
    - Near deadline (t >= 0.95): accepts if above reservation value
    - Very near deadline (t >= 0.99): accepts best received if reasonable

    Never accepts below computed reservation value.

    **Opponent Modeling:**
    Comprehensive frequency-based preference estimation:

    - Tracks value frequencies for each issue from opponent bids
    - Updates issue weights based on consistency (unchanged values
      get +0.1 weight, then normalized)
    - Estimates opponent utility as weighted sum of normalized frequencies
    - Detects opponent concession when recent offers improve utility
    - Periodically re-estimates Nash point (every 5 opponent bids)
    - Dynamic reservation value computed from domain utility distribution

    Args:
        min_utility: Minimum acceptable utility threshold (default 0.6)
        concession_rate: Base concession rate parameter (default 0.1)
        phase1_end: End time for phase 1 (default 0.2)
        phase2_end: End time for phase 2 (default 0.8)
        deadline_time: Time threshold for deadline acceptance (default 0.95)
        critical_time: Time threshold for critical deadline acceptance (default 0.99)
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
        min_utility: float = 0.6,
        concession_rate: float = 0.1,
        phase1_end: float = 0.2,
        phase2_end: float = 0.8,
        deadline_time: float = 0.95,
        critical_time: float = 0.99,
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
        self._concession_rate = concession_rate
        self._phase1_end = phase1_end
        self._phase2_end = phase2_end
        self._deadline_time = deadline_time
        self._critical_time = critical_time
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # State
        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._reservation_value: float = min_utility

        # Opponent modeling
        self._opponent_bids: list[Outcome] = []
        self._opponent_utilities: list[float] = []  # Our utility for their bids
        self._opponent_issue_weights: dict[int, float] = {}
        self._opponent_value_frequencies: dict[int, dict[str, int]] = {}

        # Nash estimation
        self._nash_point: Outcome | None = None
        self._nash_utility: float = 0.0
        self._estimated_opponent_nash_utility: float = 0.0

        # Tracking
        self._last_offered_utility: float = 1.0
        self._opponent_best_bid: Outcome | None = None
        self._opponent_best_bid_utility: float = 0.0
        self._opponent_concession_detected: bool = False

    def _initialize(self) -> None:
        """Initialize the outcome space and compute initial Nash estimate."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility

        # Initialize opponent model with equal weights
        if self.nmi is not None:
            n_issues = len(self.nmi.issues)
            for i in range(n_issues):
                self._opponent_issue_weights[i] = 1.0 / n_issues
                self._opponent_value_frequencies[i] = {}

        # Set initial reservation value based on domain
        self._compute_reservation_value()

        # Initial Nash point estimate (will be refined with opponent data)
        self._estimate_nash_point()

        self._initialized = True

    def _compute_reservation_value(self) -> None:
        """Compute reservation value based on domain analysis."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return

        # Analyze utility distribution
        utilities = [bd.utility for bd in self._outcome_space.outcomes]
        if not utilities:
            return

        mean_util = sum(utilities) / len(utilities)
        min_util = min(utilities)

        # Reservation value: between mean and min, biased toward mean
        self._reservation_value = max(
            self._min_utility, min_util + 0.7 * (mean_util - min_util)
        )

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_bids = []
        self._opponent_utilities = []
        self._last_offered_utility = self._max_utility
        self._opponent_best_bid = None
        self._opponent_best_bid_utility = 0.0
        self._opponent_concession_detected = False

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent model based on received bid."""
        if bid is None or self.ufun is None:
            return

        self._opponent_bids.append(bid)
        self._opponent_utilities.append(float(self.ufun(bid)))

        # Track value frequencies for each issue
        for i, value in enumerate(bid):
            if i not in self._opponent_value_frequencies:
                self._opponent_value_frequencies[i] = {}

            value_str = str(value)
            if value_str not in self._opponent_value_frequencies[i]:
                self._opponent_value_frequencies[i][value_str] = 0
            self._opponent_value_frequencies[i][value_str] += 1

        # Update issue weights based on consistency
        if len(self._opponent_bids) >= 2:
            self._update_issue_weights()

        # Detect opponent concession
        if len(self._opponent_utilities) >= 3:
            recent = self._opponent_utilities[-3:]
            if recent[-1] > recent[0]:  # Opponent giving us better utility
                self._opponent_concession_detected = True

        # Re-estimate Nash point periodically
        if len(self._opponent_bids) % 5 == 0:
            self._estimate_nash_point()

    def _update_issue_weights(self) -> None:
        """Update estimated opponent issue weights."""
        if len(self._opponent_bids) < 2:
            return

        last_bid = self._opponent_bids[-1]
        prev_bid = self._opponent_bids[-2]

        # Issues that don't change are likely more important to opponent
        total_unchanged = 0
        unchanged_issues: list[int] = []

        for i in range(len(last_bid)):
            if last_bid[i] == prev_bid[i]:
                total_unchanged += 1
                unchanged_issues.append(i)

        if total_unchanged == 0:
            return

        # Increase weight of unchanged issues
        learning_rate = 0.1
        for i in unchanged_issues:
            self._opponent_issue_weights[i] += learning_rate

        # Normalize weights
        total = sum(self._opponent_issue_weights.values())
        if total > 0:
            for i in self._opponent_issue_weights:
                self._opponent_issue_weights[i] /= total

    def _estimate_opponent_utility(self, bid: Outcome) -> float:
        """Estimate opponent's utility for a bid."""
        if not self._opponent_value_frequencies:
            return 0.5

        total_utility = 0.0

        for i, value in enumerate(bid):
            weight = self._opponent_issue_weights.get(i, 0.0)
            value_str = str(value)

            freq_map = self._opponent_value_frequencies.get(i, {})
            if not freq_map:
                total_utility += weight * 0.5
                continue

            # Value utility based on frequency
            value_count = freq_map.get(value_str, 0)
            max_count = max(freq_map.values()) if freq_map else 1

            value_utility = value_count / max_count if max_count > 0 else 0.5
            total_utility += weight * value_utility

        return total_utility

    def _estimate_nash_point(self) -> None:
        """Estimate the Nash equilibrium point."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return

        best_nash_product = -1.0
        best_nash_bid = None
        best_own_util = 0.0
        best_opp_util = 0.0

        # Search through outcomes for Nash point
        for bd in self._outcome_space.outcomes:
            own_util = bd.utility
            opp_util = self._estimate_opponent_utility(bd.bid)

            # Nash product: maximize product of utilities
            nash_product = own_util * opp_util

            if nash_product > best_nash_product:
                best_nash_product = nash_product
                best_nash_bid = bd.bid
                best_own_util = own_util
                best_opp_util = opp_util

        if best_nash_bid is not None:
            self._nash_point = best_nash_bid
            self._nash_utility = best_own_util
            self._estimated_opponent_nash_utility = best_opp_util

    def _get_target_utility(self, time: float) -> float:
        """
        Calculate target utility based on time and negotiation phase.

        Phase 1 (t < phase1_end): Stay near max utility
        Phase 2 (phase1_end <= t < phase2_end): Gradual concession toward Nash
        Phase 3 (t >= phase2_end): More aggressive concession
        """
        if time < self._phase1_end:
            # Phase 1: Conservative
            # Linear concession from max to 0.95 * max
            phase_progress = time / self._phase1_end
            target = self._max_utility - phase_progress * 0.05 * self._max_utility
        elif time < self._phase2_end:
            # Phase 2: Gradual concession toward Nash point
            phase_progress = (time - self._phase1_end) / (
                self._phase2_end - self._phase1_end
            )

            # Start from end of phase 1
            phase1_end = self._max_utility * 0.95

            # Target Nash utility or reservation value (whichever is higher)
            nash_target = max(self._nash_utility, self._reservation_value)

            # Boulware-like concession (slow at first, faster later)
            beta = 0.3  # Concession shape parameter
            concession = phase_progress ** (1.0 / beta)

            target = phase1_end - concession * (phase1_end - nash_target)
        else:
            # Phase 3: More aggressive concession
            phase_progress = (time - self._phase2_end) / (1.0 - self._phase2_end)

            # Start from Nash utility
            phase2_end = max(self._nash_utility, self._reservation_value)

            # End at reservation value
            target = phase2_end - phase_progress * (
                phase2_end - self._reservation_value
            )

            # If opponent is conceding, be less aggressive
            if self._opponent_concession_detected:
                target = max(target, phase2_end * 0.95)

        return max(target, self._reservation_value)

    def _select_bid(self, time: float) -> Outcome | None:
        """Select a bid balancing own utility and opponent satisfaction."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        target = self._get_target_utility(time)

        # Get candidates above target
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            # Fallback: get best bid above reservation
            for bd in self._outcome_space.outcomes:
                if bd.utility >= self._reservation_value:
                    candidates = [bd]
                    break

        if not candidates:
            return self._best_bid

        # Score candidates by balance of own utility and opponent utility
        scored_bids: list[tuple[Outcome, float]] = []

        for bd in candidates:
            own_util = bd.utility
            opp_util = self._estimate_opponent_utility(bd.bid)

            # Combined score: weighted sum
            # Early: favor own utility; later: balance more
            own_weight = 1.0 - 0.3 * time
            opp_weight = 0.3 * time

            score = own_weight * own_util + opp_weight * opp_util
            scored_bids.append((bd.bid, score))

        # Sort by score descending
        scored_bids.sort(key=lambda x: x[1], reverse=True)

        # Pick from top candidates with some randomness
        top_n = min(3, len(scored_bids))
        selected = random.choice(scored_bids[:top_n])

        return selected[0]

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time

        # First few rounds: offer best bid
        if state.step < 3:
            self._last_offered_utility = self._max_utility
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

        # Update opponent model
        self._update_opponent_model(offer)

        offer_utility = float(self.ufun(offer))
        time = state.relative_time

        # Track best opponent bid
        if (
            self._opponent_best_bid is None
            or offer_utility > self._opponent_best_bid_utility
        ):
            self._opponent_best_bid = offer
            self._opponent_best_bid_utility = offer_utility

        # Never accept below reservation value
        if offer_utility < self._reservation_value:
            return ResponseType.REJECT_OFFER

        # Get our target utility
        target = self._get_target_utility(time)

        # Accept if offer exceeds our target
        if offer_utility >= target:
            return ResponseType.ACCEPT_OFFER

        # Accept if offer is better than what we would offer
        if offer_utility >= self._last_offered_utility:
            return ResponseType.ACCEPT_OFFER

        # Near deadline: accept if above reservation and decent
        if time >= self._deadline_time:
            if offer_utility >= self._reservation_value:
                return ResponseType.ACCEPT_OFFER

        # Very near deadline: accept best received offer if reasonable
        if time >= self._critical_time:
            if self._opponent_best_bid_utility >= self._reservation_value:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
