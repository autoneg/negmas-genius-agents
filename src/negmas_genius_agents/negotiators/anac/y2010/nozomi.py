"""Nozomi from ANAC 2010."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["Nozomi"]


class ThreatBasedOpponentModel:
    """
    Opponent model for Nozomi's threat-based negotiation strategy.

    This model tracks:
    1. Opponent's concession rate over time
    2. Best offer received from opponent
    3. Value preferences (for predicting acceptable bids)

    The key metric is whether the opponent is conceding. If they are not,
    Nozomi responds with a "threat" - refusing to concede as well.
    """

    def __init__(
        self, issues: list[str], decay: float = 0.9, concession_threshold: float = 0.02
    ):
        """
        Initialize the opponent model.

        Args:
            issues: List of issue names.
            decay: Decay factor for recency weighting.
            concession_threshold: Threshold for considering opponent as conceding.
        """
        self._issues = issues
        self._decay = decay
        self._concession_threshold = concession_threshold

        # Track all opponent offers with timestamps
        self._offers: list[tuple[float, float, Outcome]] = []  # (time, utility, bid)

        # Track value frequencies for opponent preference estimation
        self._value_counts: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._issue_totals: dict[str, float] = defaultdict(float)

        # Key metrics
        self._best_utility: float = 0.0
        self._best_offer: Outcome | None = None
        self._concession_rate: float = 0.0
        self._is_opponent_conceding: bool = False

    def update(self, time: float, utility: float, bid: Outcome) -> None:
        """
        Update the model with an opponent offer.

        Args:
            time: Normalized time [0, 1].
            utility: Our utility for the opponent's offer.
            bid: The opponent's bid.
        """
        self._offers.append((time, utility, bid))

        # Track best offer
        if utility > self._best_utility:
            self._best_utility = utility
            self._best_offer = bid

        # Update value frequencies (with decay)
        bid_dict = dict(bid) if hasattr(bid, "items") or isinstance(bid, dict) else {}
        if bid_dict:
            for issue in self._issues:
                for val in self._value_counts[issue]:
                    self._value_counts[issue][val] *= self._decay
                self._issue_totals[issue] *= self._decay

                if issue in bid_dict:
                    val_str = str(bid_dict[issue])
                    self._value_counts[issue][val_str] += 1.0
                    self._issue_totals[issue] += 1.0

        # Calculate concession rate
        self._update_concession_rate()

    def _update_concession_rate(self) -> None:
        """
        Calculate opponent's concession rate.

        Compares average utility of first third vs last third of offers.
        Positive rate means opponent is conceding (offering more utility to us).
        """
        if len(self._offers) < 3:
            self._concession_rate = 0.0
            self._is_opponent_conceding = False
            return

        n = len(self._offers)
        third = max(1, n // 3)

        # Average utility in first third
        first_avg = sum(u for _, u, _ in self._offers[:third]) / third

        # Average utility in last third
        last_avg = sum(u for _, u, _ in self._offers[-third:]) / third

        # Positive = opponent is giving us more utility over time
        self._concession_rate = last_avg - first_avg

        # Threshold for "conceding" - opponent must show meaningful improvement
        self._is_opponent_conceding = self._concession_rate > self._concession_threshold

    @property
    def best_utility(self) -> float:
        """Best utility received from opponent."""
        return self._best_utility

    @property
    def best_offer(self) -> Outcome | None:
        """Best offer received from opponent."""
        return self._best_offer

    @property
    def concession_rate(self) -> float:
        """Opponent's concession rate (positive = conceding)."""
        return self._concession_rate

    @property
    def is_conceding(self) -> bool:
        """Whether opponent is meaningfully conceding."""
        return self._is_opponent_conceding

    def get_issue_weights(self) -> dict[str, float]:
        """
        Estimate opponent's issue weights.

        Returns:
            Dictionary mapping issues to estimated weights.
        """
        if not self._issues:
            return {}

        # Use consistency of value selection as weight proxy
        weights = {}
        for issue in self._issues:
            if self._issue_totals[issue] > 0:
                max_count = max(self._value_counts[issue].values(), default=0)
                weights[issue] = max_count / self._issue_totals[issue]
            else:
                weights[issue] = 1.0 / len(self._issues)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def estimate_opponent_utility(self, bid: Outcome) -> float:
        """
        Estimate opponent's utility for a bid.

        Args:
            bid: The bid to evaluate.

        Returns:
            Estimated utility in [0, 1].
        """
        if len(self._offers) < 3:
            return 0.5

        # Convert bid to dict for access
        bid_dict = dict(bid) if hasattr(bid, "items") or isinstance(bid, dict) else {}
        if not bid_dict:
            return 0.5

        weights = self.get_issue_weights()
        utility = 0.0

        for issue, weight in weights.items():
            if issue in bid_dict:
                value = str(bid_dict[issue])
                if self._issue_totals[issue] > 0:
                    pref = (
                        self._value_counts[issue].get(value, 0)
                        / self._issue_totals[issue]
                    )
                    utility += weight * pref
                else:
                    utility += weight * 0.5

        return utility


class Nozomi(SAONegotiator):
    """
    Nozomi from ANAC 2010 - 3rd place agent.

    This agent uses a threat-based negotiation strategy with adaptive concession
    based on opponent modeling. The key insight is that negotiation is a strategic
    interaction where revealing too much willingness to concede can be exploited.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This implementation includes:

    1. Threat-based concession: mirrors opponent's concession behavior
    2. Bayesian opponent preference modeling for predicting opponent reactions
    3. Time-pressure aware acceptance with multiple criteria
    4. Weighted random bid selection biased toward higher utility

    References:
        Original Genius class: ``agents.anac.y2010.Nozomi.Nozomi``

        ANAC 2010: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        - Time-dependent concession with cubic curve: target(t) = max - (max-min) * t^3
        - Threat mechanism: If opponent doesn't concede, Nozomi slows concession by 70%
        - Weighted random bid selection among candidates, biased toward higher utility
        - Near deadline (t > 0.95): considers opponent's best offer as fallback

    **Acceptance Strategy:**
        - Accept if offer utility >= current target utility
        - Accept if very near deadline (t >= 0.99) and offer >= minimum utility
        - Multiple acceptance windows based on time pressure

    **Opponent Modeling:**
        - Tracks opponent's concession rate (first third vs last third of offers)
        - Monitors best offer received for deadline strategy
        - Value frequency analysis for predicting opponent preferences
        - "Threat response": mirrors non-conceding behavior

    Args:
        initial_target: Initial target utility (default 0.95).
        min_utility: Minimum acceptable utility floor (default 0.65).
        threat_factor: Concession slowdown when opponent doesn't concede (default 0.3).
        concession_exponent: Exponent for concession curve (default 3).
        deadline_start: Time to start deadline pressure (default 0.95).
        deadline_buffer: Buffer above opponent's best offer near deadline (default 0.05).
        final_deadline_time: Time threshold for final deadline acceptance (default 0.99).
        concession_threshold: Threshold for considering opponent as conceding (default 0.02).
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
        initial_target: float = 0.95,
        min_utility: float = 0.65,
        threat_factor: float = 0.3,
        concession_exponent: int = 3,
        deadline_start: float = 0.95,
        deadline_buffer: float = 0.05,
        final_deadline_time: float = 0.99,
        concession_threshold: float = 0.02,
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
        self._initial_target = initial_target
        self._min_utility_param = min_utility
        self._threat_factor = threat_factor
        self._concession_exponent = concession_exponent
        self._deadline_start = deadline_start
        self._deadline_buffer = deadline_buffer
        self._final_deadline_time = final_deadline_time
        self._concession_threshold = concession_threshold

        # Will be initialized later
        self._outcome_space: SortedOutcomeSpace | None = None
        self._opponent_model: ThreatBasedOpponentModel | None = None
        self._initialized = False

        # Working variables
        self._target: float = initial_target
        self._max_utility: float = 1.0
        self._min_utility: float = min_utility
        self._best_bid: Outcome | None = None

    def _initialize(self) -> None:
        """Initialize outcome space and opponent model."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)

        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility

        # Set min utility based on reservation value
        reservation = getattr(self.ufun, "reserved_value", None)
        if reservation is not None and reservation != float("-inf"):
            self._min_utility = max(self._min_utility_param, float(reservation))
        else:
            self._min_utility = self._min_utility_param

        # Initialize opponent model
        if hasattr(self.nmi, "issues") and self.nmi.issues:
            issue_names = [issue.name for issue in self.nmi.issues]
            self._opponent_model = ThreatBasedOpponentModel(
                issue_names, concession_threshold=self._concession_threshold
            )

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts. Initializes state."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset working state
        self._target = self._initial_target
        if self._opponent_model is not None:
            # Reinitialize opponent model for new negotiation
            if hasattr(self.nmi, "issues") and self.nmi.issues:
                issue_names = [issue.name for issue in self.nmi.issues]
                self._opponent_model = ThreatBasedOpponentModel(
                    issue_names, concession_threshold=self._concession_threshold
                )

    def _calculate_target(self, time: float) -> float:
        """
        Calculate target utility using threat-based concession.

        Uses cubic concession curve modulated by opponent behavior:
        - If opponent is conceding: normal concession rate
        - If opponent is NOT conceding: concession slowed by threat_factor

        Args:
            time: Normalized time [0, 1].

        Returns:
            Target utility value.
        """
        # Cubic concession curve
        time_factor = math.pow(time, self._concession_exponent)

        # Determine threat factor based on opponent behavior
        if self._opponent_model is not None and self._opponent_model.is_conceding:
            # Opponent is conceding, use normal concession rate
            effective_threat = 1.0
        else:
            # Opponent is NOT conceding, slow down our concession (threat)
            effective_threat = self._threat_factor

        # Calculate concession amount
        concession = (
            (self._max_utility - self._min_utility) * time_factor * effective_threat
        )

        target = self._max_utility - concession

        # Near deadline adjustment: consider opponent's best offer
        if time > self._deadline_start and self._opponent_model is not None:
            best_opp = self._opponent_model.best_utility
            if best_opp > 0:
                # Be willing to accept slightly above opponent's best offer
                deadline_target = best_opp + self._deadline_buffer
                target = min(target, deadline_target)

        return max(target, self._min_utility)

    def _select_bid(self, target: float) -> Outcome | None:
        """
        Select a bid using weighted random selection.

        Selects from bids above target utility, with weights proportional
        to utility (biased toward better bids).

        Args:
            target: Target utility threshold.

        Returns:
            Selected bid, or best bid as fallback.
        """
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        # Get candidate bids above target
        candidates = self._outcome_space.get_bids_above(target)

        if not candidates:
            # Fallback to bids above minimum
            candidates = self._outcome_space.get_bids_above(self._min_utility)

        if not candidates:
            return self._best_bid

        if len(candidates) == 1:
            return candidates[0].bid

        # Weighted random selection (bias toward higher utility)
        weights = [bd.utility for bd in candidates]
        total = sum(weights)

        if total == 0:
            return random.choice(candidates).bid

        r = random.random() * total
        cumulative = 0.0
        for bd in candidates:
            cumulative += bd.utility
            if r <= cumulative:
                return bd.bid

        return candidates[0].bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """
        Generate a proposal using threat-based strategy.

        Args:
            state: Current negotiation state.
            dest: Destination negotiator ID (ignored).

        Returns:
            Outcome to propose, or None.
        """
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        self._target = self._calculate_target(time)

        return self._select_bid(self._target)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """
        Respond to an offer using threat-based acceptance criteria.

        Args:
            state: Current negotiation state.
            source: Source negotiator ID (ignored).

        Returns:
            ResponseType indicating acceptance or rejection.
        """
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        # Update opponent model
        if self._opponent_model is not None:
            self._opponent_model.update(time, offer_utility, offer)

        # Calculate current target
        self._target = self._calculate_target(time)

        # Accept if meets target
        if offer_utility >= self._target:
            return ResponseType.ACCEPT_OFFER

        # Accept anything above minimum very close to deadline
        if time >= self._final_deadline_time and offer_utility >= self._min_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
