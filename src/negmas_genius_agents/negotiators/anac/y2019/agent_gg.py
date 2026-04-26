"""AgentGG from ANAC 2019."""

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

__all__ = ["AgentGG"]


class AgentGG(SAONegotiator):
    """
    AgentGG from ANAC 2019 - The winning agent.

    AgentGG won ANAC 2019 using an importance-based bidding strategy that
    focuses on understanding which issue values contribute most to utility,
    rather than just evaluating complete bids. This allows for more nuanced
    bid generation and opponent modeling.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        .. code-block:: bibtex

            @inproceedings{baarslag2019tenth,
                title={The Tenth International Automated Negotiating Agents
                       Competition (ANAC 2019)},
                author={Baarslag, Tim and Fujita, Katsuhide and Gerding,
                        Enrico H and Hindriks, Koen and Ito, Takayuki and
                        Jennings, Nicholas R and others},
                booktitle={Proceedings of the International Joint Conference
                           on Autonomous Agents and Multiagent Systems (AAMAS)},
                year={2019}
            }

        Original Genius class: ``agents.anac.y2019.agentgg.AgentGG``

    **Offering Strategy:**
        - Computes "importance" values for each issue value based on average
          utility contribution across all outcomes containing that value
        - Uses multi-phase time-dependent thresholds based on importance ratios
        - Early phase (t < 0.2): Random exploration within threshold range
        - Middle phase (0.2 < t < 0.9): Select bids maximizing opponent importance
        - Late phase (t > 0.9): Aggressive concession toward Nash estimate
        - Threshold adjusts based on estimated Nash point, not fixed schedule

    **Acceptance Strategy:**
        - Accepts offers whose importance ratio exceeds current threshold
        - Importance ratio = (bid_importance - min) / (max - min)
        - Near deadline (t > 0.999): Accepts offers above reservation ratio
        - Threshold decreases in phases aligned with offering strategy

    **Opponent Modeling:**
        - Builds opponent importance map by tracking frequency of issue values
          in opponent's offers
        - Estimates Nash point from opponent's best offers during initial phase
          (first 0.3% of negotiation after initial exchange)
        - Uses opponent importance to select bids that opponent might prefer
          while meeting own threshold requirements

    Args:
        reservation_ratio: Base reservation value ratio (default 0.0)
        deadline_threshold: Time threshold for final deadline acceptance (default 0.999)
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
        reservation_ratio: float = 0.0,
        deadline_threshold: float = 0.999,
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
        self._reservation_ratio = reservation_ratio
        self._deadline_threshold = deadline_threshold
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Importance maps
        self._own_importance: dict[int, dict[str, float]] = {}
        self._opponent_importance: dict[int, dict[str, float]] = {}

        # Thresholds
        self._offer_lower_ratio: float = 1.0
        self._offer_higher_ratio: float = 1.1

        # State
        self._max_importance: float = 1.0
        self._min_importance: float = 0.0
        self._max_importance_bid: Outcome | None = None
        self._estimated_nash_point: float = 0.8
        self._max_opponent_bid_importance: float = 0.0
        self._last_received_bid: Outcome | None = None
        self._opponent_offers: int = 0
        self._offer_randomly: bool = True

        # Nash estimation state
        self._initial_time_pass: bool = False
        self._start_time: float = 0.0
        self._nash_estimated: bool = False

    def _initialize(self) -> None:
        """Initialize the outcome space and importance maps."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)

        # Build importance map based on utility contributions
        self._build_importance_map()

        self._initialized = True

    def _build_importance_map(self) -> None:
        """Build importance map for each issue value."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return

        # Collect all unique values per issue
        issue_values: dict[int, set[str]] = {}
        for bd in self._outcome_space.outcomes:
            for i, value in enumerate(bd.bid):
                if i not in issue_values:
                    issue_values[i] = set()
                issue_values[i].add(str(value))

        # Compute average utility for each value
        value_utilities: dict[int, dict[str, list[float]]] = {}
        for bd in self._outcome_space.outcomes:
            for i, value in enumerate(bd.bid):
                if i not in value_utilities:
                    value_utilities[i] = {}
                value_str = str(value)
                if value_str not in value_utilities[i]:
                    value_utilities[i][value_str] = []
                value_utilities[i][value_str].append(bd.utility)

        # Compute importance as average utility contribution
        for i, values in value_utilities.items():
            self._own_importance[i] = {}
            for value_str, utilities in values.items():
                avg_utility = sum(utilities) / len(utilities)
                self._own_importance[i][value_str] = avg_utility

            # Sort by importance (descending)
            sorted_values = sorted(
                self._own_importance[i].items(), key=lambda x: x[1], reverse=True
            )
            self._own_importance[i] = dict(sorted_values)

        # Find max and min importance bids
        self._find_max_min_importance_bids()

    def _find_max_min_importance_bids(self) -> None:
        """Find bids with maximum and minimum importance."""
        if not self._own_importance:
            return

        # Max importance bid: take best value for each issue
        max_bid_values: dict[int, str] = {}
        min_bid_values: dict[int, str] = {}
        for i, values in self._own_importance.items():
            if values:
                sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
                max_bid_values[i] = sorted_values[0][0]
                min_bid_values[i] = sorted_values[-1][0]

        # Find matching outcomes
        if self._outcome_space:
            for bd in self._outcome_space.outcomes:
                bid_vals = {i: str(v) for i, v in enumerate(bd.bid)}
                if bid_vals == max_bid_values:
                    self._max_importance_bid = bd.bid
                    self._max_importance = self._compute_importance(bd.bid)

            # Also compute min importance
            for bd in reversed(self._outcome_space.outcomes):
                bid_vals = {i: str(v) for i, v in enumerate(bd.bid)}
                if bid_vals == min_bid_values:
                    self._min_importance = self._compute_importance(bd.bid)
                    break

        # Fallback: use best utility bid
        if self._max_importance_bid is None and self._outcome_space:
            self._max_importance_bid = self._outcome_space.outcomes[0].bid
            self._max_importance = self._compute_importance(self._max_importance_bid)
            if self._outcome_space.outcomes:
                self._min_importance = self._compute_importance(
                    self._outcome_space.outcomes[-1].bid
                )

    def _compute_importance(self, bid: Outcome) -> float:
        """Compute total importance of a bid."""
        if bid is None:
            return 0.0

        total = 0.0
        for i, value in enumerate(bid):
            value_str = str(value)
            if i in self._own_importance and value_str in self._own_importance[i]:
                total += self._own_importance[i][value_str]
        return total

    def _compute_opponent_importance(self, bid: Outcome) -> float:
        """Compute estimated opponent importance of a bid."""
        if bid is None or not self._opponent_importance:
            return 0.0

        total = 0.0
        for i, value in enumerate(bid):
            value_str = str(value)
            if (
                i in self._opponent_importance
                and value_str in self._opponent_importance[i]
            ):
                total += self._opponent_importance[i][value_str]
        return total

    def _update_opponent_model(self, bid: Outcome) -> None:
        """Update opponent importance model based on received bid."""
        if bid is None:
            return

        for i, value in enumerate(bid):
            if i not in self._opponent_importance:
                self._opponent_importance[i] = {}
            value_str = str(value)
            if value_str not in self._opponent_importance[i]:
                self._opponent_importance[i][value_str] = 0.0
            self._opponent_importance[i][value_str] += 1.0

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset state
        self._opponent_importance = {}
        self._max_opponent_bid_importance = 0.0
        self._last_received_bid = None
        self._opponent_offers = 0
        self._offer_lower_ratio = 1.0
        self._offer_higher_ratio = 1.1
        self._offer_randomly = True
        self._initial_time_pass = False
        self._nash_estimated = False

    def _estimate_nash_point(self, time: float) -> None:
        """Estimate Nash point from opponent's early offers."""
        if self._nash_estimated:
            return

        if self._last_received_bid is not None:
            bid_importance = self._compute_importance(self._last_received_bid)
            if bid_importance > self._max_opponent_bid_importance:
                self._max_opponent_bid_importance = bid_importance

        if self._initial_time_pass:
            if time - self._start_time > 0.003:  # 3/1000 of negotiation
                imp_range = self._max_importance - self._min_importance
                if imp_range > 0:
                    max_ratio = (
                        self._max_opponent_bid_importance - self._min_importance
                    ) / imp_range
                else:
                    max_ratio = 0.5
                # Estimate Nash between opponent's best for us and 1.0
                self._estimated_nash_point = (1 - max_ratio) / 1.7 + max_ratio
                self._nash_estimated = True
        else:
            # Start timing after first different bid
            if self._opponent_offers > 1:
                self._initial_time_pass = True
                self._start_time = time

    def _get_threshold(self, time: float) -> None:
        """Update thresholds based on time."""
        if time < 0.01:
            self._offer_lower_ratio = 0.9999
        elif time < 0.02:
            self._offer_lower_ratio = 0.99
        elif time < 0.2:
            self._offer_lower_ratio = 0.99 - 0.5 * (time - 0.02)
        elif time < 0.5:
            self._offer_randomly = False
            p2 = 0.3 * (1 - self._estimated_nash_point) + self._estimated_nash_point
            self._offer_lower_ratio = 0.9 - (0.9 - p2) / 0.3 * (time - 0.2)
        elif time < 0.9:
            p1 = 0.3 * (1 - self._estimated_nash_point) + self._estimated_nash_point
            p2 = 0.15 * (1 - self._estimated_nash_point) + self._estimated_nash_point
            self._offer_lower_ratio = p1 - (p1 - p2) / 0.4 * (time - 0.5)
        elif time < 0.98:
            p1 = 0.15 * (1 - self._estimated_nash_point) + self._estimated_nash_point
            p2 = 0.05 * (1 - self._estimated_nash_point) + self._estimated_nash_point
            possible_ratio = p1 - (p1 - p2) / 0.08 * (time - 0.9)
            self._offer_lower_ratio = max(possible_ratio, self._reservation_ratio + 0.3)
        elif time < 0.995:
            p1 = 0.05 * (1 - self._estimated_nash_point) + self._estimated_nash_point
            p2 = 0.0 * (1 - self._estimated_nash_point) + self._estimated_nash_point
            possible_ratio = p1 - (p1 - p2) / 0.015 * (time - 0.98)
            self._offer_lower_ratio = max(
                possible_ratio, self._reservation_ratio + 0.25
            )
        elif time < 0.999:
            p1 = 0.0 * (1 - self._estimated_nash_point) + self._estimated_nash_point
            p2 = -0.35 * (1 - self._estimated_nash_point) + self._estimated_nash_point
            possible_ratio = p1 - (p1 - p2) / 0.004 * (time - 0.995)
            self._offer_lower_ratio = max(
                possible_ratio, self._reservation_ratio + 0.25
            )
        else:
            possible_ratio = (
                -0.4 * (1 - self._estimated_nash_point) + self._estimated_nash_point
            )
            self._offer_lower_ratio = max(possible_ratio, self._reservation_ratio + 0.2)

        self._offer_higher_ratio = self._offer_lower_ratio + 0.1

    def _get_importance_ratio(self, bid: Outcome) -> float:
        """Get importance ratio for a bid (0 to 1)."""
        imp_range = self._max_importance - self._min_importance
        if imp_range <= 0:
            return 0.5
        return (self._compute_importance(bid) - self._min_importance) / imp_range

    def _select_bid(self) -> Outcome | None:
        """Select a bid within the threshold range."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        lower_threshold = (
            self._offer_lower_ratio * (self._max_importance - self._min_importance)
            + self._min_importance
        )
        upper_threshold = (
            self._offer_higher_ratio * (self._max_importance - self._min_importance)
            + self._min_importance
        )

        # Search for bids in range
        candidates: list[tuple[Outcome, float]] = []
        for bd in self._outcome_space.outcomes:
            bid_imp = self._compute_importance(bd.bid)
            if lower_threshold <= bid_imp <= upper_threshold:
                if self._offer_randomly:
                    return bd.bid
                opp_imp = self._compute_opponent_importance(bd.bid)
                candidates.append((bd.bid, opp_imp))

        if candidates:
            # Select bid with highest opponent importance
            best_bid = max(candidates, key=lambda x: x[1])
            return best_bid[0]

        # Fallback: find any bid above threshold
        for bd in self._outcome_space.outcomes:
            if self._compute_importance(bd.bid) >= lower_threshold:
                return bd.bid

        return self._max_importance_bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        # First offer: best bid
        if self._opponent_offers == 0:
            return self._max_importance_bid

        time = state.relative_time
        self._estimate_nash_point(time)
        self._get_threshold(time)

        return self._select_bid()

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        # Update opponent model
        self._last_received_bid = offer
        self._opponent_offers += 1
        self._update_opponent_model(offer)

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        time = state.relative_time

        # Update thresholds
        self._estimate_nash_point(time)
        self._get_threshold(time)

        # Compute acceptance threshold based on importance ratio
        offer_importance_ratio = self._get_importance_ratio(offer)

        # Accept if above threshold
        if offer_importance_ratio >= self._offer_lower_ratio:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, be more flexible
        if time >= self._deadline_threshold:
            reservation = self._reservation_ratio
            if offer_utility >= reservation:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
