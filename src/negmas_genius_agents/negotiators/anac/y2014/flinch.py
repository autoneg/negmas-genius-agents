"""Flinch from ANAC 2014."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import BidDetails, SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["Flinch"]


class Flinch(SAONegotiator):
    """
    Flinch from ANAC 2014.

    Flinch computes a time-dependent acceptance threshold shaped by an
    estimate of the best achievable utility and a "kink" time (``t_lambda``)
    derived from the domain's discount factor, and proposes bids that
    balance our own utility against a kernel-regression estimate of the
    opponent's utility for that bid (learned from the opponent's offer
    history using a cubic kernel over an issue-wise bid distance metric).

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. The original agent searches for candidate
        bids using a genetic algorithm over the full bid space; this port
        instead enumerates/filters the discrete outcome space directly
        (via :class:`SortedOutcomeSpace`), which is equivalent in effect for
        the small-to-medium discrete domains this port targets but skips the
        GA machinery itself. Behaviour is intended to be in the same
        ballpark, not bit-for-bit identical.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.Flinch.Flinch

    **Offering Strategy:**
        1. Filter outcomes to those with utility above
           ``acceptance_threshold * f`` (starting ``f=1.0``, backing off by
           squaring ``f`` repeatedly if no candidates are found).
        2. Normalize both our utility and a kernel-estimated opponent
           utility for each candidate to [0, 1].
        3. Score each candidate as
           ``score = my_util * (1 - Kop) + opp_util * Kop`` where ``Kop``
           grows over time (``0.5 + 0.4 * sqrt(t)``), gradually shifting
           weight toward opponent-friendly bids as the deadline nears.
        4. If any previously received opponent bid is better for us than the
           chosen candidate, re-offer that bid instead (mirrors Flinch's
           final fallback to the best-seen opponent bid).

    **Acceptance Strategy:**
        A smooth two-phase acceptance threshold curve parameterized by
        ``t_lambda`` (a discount-factor-dependent "kink" time):
        - Before ``t_lambda``: threshold interpolates from
          ``max(r, M*p_max)`` down to the midpoint using a
          ``(t/t_lambda)^(1/beta)`` curve.
        - After ``t_lambda``: threshold continues from the midpoint toward
          ``max(r, M*p_min)`` (capped by the best opponent utility seen so
          far) using a ``((t - t_lambda)/(1 - t_lambda))^(1/beta)`` curve.
        Accept whenever the offer's utility exceeds this threshold; end the
        negotiation early only if the reservation value alone would already
        exceed the threshold (mirrored here as a plain rejection since the
        NegMAS API has no explicit "end negotiation" action from
        ``respond``).

    **Opponent Modeling:**
        Nadaraya-Watson-style kernel regression over the opponent's offer
        history: distance between two bids is the average per-issue
        normalized difference (0/1 for discrete mismatches, normalized
        absolute difference for numeric issues). A bandwidth ``h`` is
        redrawn each bidding round (``0.3 + 0.7 * U(0,1)``) and combined
        with a cubic kernel ``(1 - x^2)^3`` for ``|x| <= 1``. Estimation
        only activates once at least ``min_train_data`` opponent offers
        have been observed; before that, all candidates are treated as
        equally good for the opponent (utility 1.0), matching the original.

    Args:
        t_lambda_min: Minimum value for the concession "kink" time (default 0.1).
        t_lambda_max: Maximum value for the concession "kink" time (default 0.8).
        beta: Shape exponent for the threshold curve (default 2.0).
        p_max: Fraction of estimated max utility used as an early ceiling (default 0.95).
        p_min: Fraction of estimated max utility used as a late floor (default 0.84).
        min_train_data: Minimum opponent offers needed before kernel opponent
            utility estimation activates (default 20).
        kop_base: Base weight given to opponent utility in bid scoring (default 0.5).
        kop_range: Additional weight added to ``kop_base`` as time approaches
            the deadline (default 0.4).
        backoff_factor: Multiplicative squaring factor used to relax the
            acceptance threshold when no candidates are found (default 0.98).
        backoff_epsilon: Lower bound on the backoff factor below which the
            candidate search stops relaxing the threshold (default 1e-6).
        bandwidth_base: Base value of the kernel-regression bandwidth drawn
            each bidding round (default 0.3).
        bandwidth_range: Range added to ``bandwidth_base`` when drawing the
            kernel-regression bandwidth (default 0.7).
        kop_time_exponent: Exponent applied to time when growing the opponent
            weight in bid scoring (default 0.5).
        discount_high_threshold: Discount factor above which the t_lambda
            interpolation uses the high-discount exponent (default 0.75).
        discount_mid_threshold: Discount factor above which the t_lambda
            interpolation uses the mid-discount exponent (default 0.5).
        discount_high_beta: Exponent applied to the discount factor when
            computing t_lambda for high-discount domains (default 2.5).
        discount_mid_beta: Exponent applied to the discount factor when
            computing t_lambda for mid-discount domains (default 2.0).
        discount_low_beta: Exponent applied to the discount factor when
            computing t_lambda for low-discount domains (default 1.5).
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
        t_lambda_min: float = 0.1,
        t_lambda_max: float = 0.8,
        beta: float = 2.0,
        p_max: float = 0.95,
        p_min: float = 0.84,
        min_train_data: int = 20,
        kop_base: float = 0.5,
        kop_range: float = 0.4,
        backoff_factor: float = 0.98,
        backoff_epsilon: float = 1e-6,
        bandwidth_base: float = 0.3,
        bandwidth_range: float = 0.7,
        kop_time_exponent: float = 0.5,
        discount_high_threshold: float = 0.75,
        discount_mid_threshold: float = 0.5,
        discount_high_beta: float = 2.5,
        discount_mid_beta: float = 2.0,
        discount_low_beta: float = 1.5,
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
        self._t_lambda_min = t_lambda_min
        self._t_lambda_max = t_lambda_max
        self._beta = beta
        self._p_max = p_max
        self._p_min = p_min
        self._min_train_data = min_train_data
        self._kop_base = kop_base
        self._kop_range = kop_range
        self._backoff_factor = backoff_factor
        self._backoff_epsilon = backoff_epsilon
        self._bandwidth_base = bandwidth_base
        self._bandwidth_range = bandwidth_range
        self._kop_time_exponent = kop_time_exponent
        self._discount_high_threshold = discount_high_threshold
        self._discount_mid_threshold = discount_mid_threshold
        self._discount_high_beta = discount_high_beta
        self._discount_mid_beta = discount_mid_beta
        self._discount_low_beta = discount_low_beta

        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        self._estimated_max_utility: float = 1.0
        self._t_lambda: float = 0.5
        self._accept_threshold: float = 0.0

        self._opponent_history: list[BidDetails] = []
        self._best_opponent_bid: BidDetails | None = None

        self._n_issues = 0

    def _initialize(self) -> None:
        """Initialize the outcome space and lambda parameter."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._estimated_max_utility = self._outcome_space.max_utility

        outcome_space = self.ufun.outcome_space
        if outcome_space is not None:
            self._n_issues = len(outcome_space.issues)

        discount = float(getattr(self.ufun, "discount_factor", None) or 1.0)
        if discount > self._discount_high_threshold:
            beta = self._discount_high_beta
        elif discount > self._discount_mid_threshold:
            beta = self._discount_mid_beta
        else:
            beta = self._discount_low_beta
        self._t_lambda = self._t_lambda_min + (
            self._t_lambda_max - self._t_lambda_min
        ) * (discount**beta)

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._opponent_history = []
        self._best_opponent_bid = None

    def _reservation_value(self) -> float:
        r = float(getattr(self.ufun, "reserved_value", 0.0) or 0.0)
        if r == float("-inf"):
            return 0.0
        return r

    def _accept_threshold_at(self, time: float) -> float:
        """Compute Flinch's two-phase acceptance threshold curve."""
        r = self._reservation_value()
        m = self._estimated_max_utility

        left = max(r, m * self._p_max)
        right = max(r, m * self._p_min)
        middle = 0.5 * left + 0.5 * right

        if time <= self._t_lambda:
            frac = 0.0 if self._t_lambda <= 0 else time / self._t_lambda
            ret = left + (middle - left) * (max(frac, 0.0) ** (1.0 / self._beta))
        else:
            if self._best_opponent_bid is not None:
                right = min(right, self._best_opponent_bid.utility)
            denom = 1.0 - self._t_lambda
            frac = 0.0 if denom <= 0 else (time - self._t_lambda) / denom
            ret = middle + (right - middle) * (max(frac, 0.0) ** (1.0 / self._beta))

        if ret != ret or ret in (float("inf"), float("-inf")):  # NaN/inf guard
            ret = left
        return ret

    def _distance(self, b1: Outcome, b2: Outcome) -> float:
        """Average normalized per-issue distance between two bids."""
        if self.ufun is None or self.ufun.outcome_space is None or not b1 or not b2:
            return 1.0

        issues = self.ufun.outcome_space.issues
        total = 0.0
        n = len(issues)
        if n == 0:
            return 0.0

        for i, issue in enumerate(issues):
            v1, v2 = b1[i], b2[i]
            if issue.is_continuous():
                span = (issue.max_value - issue.min_value) or 1.0
                total += abs(v1 - v2) / span
            else:
                total += 0.0 if v1 == v2 else 1.0

        return total / n

    def _kernel_function(self, x: float) -> float:
        if -1.0 <= x <= 1.0:
            return (1 - x * x) ** 3
        return 0.0

    def _estimated_opponent_utility(self, bid: Outcome, h: float) -> float:
        """Kernel-regression estimate of the opponent's utility for `bid`."""
        if len(self._opponent_history) < self._min_train_data:
            return 1.0

        total = 0.0
        denom = 0.0
        for bd in self._opponent_history:
            distance = self._distance(bd.bid, bid) / h if h > 0 else 0.0
            total += self._kernel_function(distance)
            denom += 1.0

        return total / denom if denom > 0 else 1.0

    def _choose_next_bid(self, time: float) -> Outcome | None:
        """Select the next bid to offer using the score-based heuristic."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        threshold = self._accept_threshold
        f = 1.0
        candidates = self._outcome_space.get_bids_above(threshold * f)
        while not candidates and f > self._backoff_epsilon:
            f *= self._backoff_factor
            candidates = self._outcome_space.get_bids_above(threshold * f)
        if not candidates:
            candidates = self._outcome_space.outcomes

        h = self._bandwidth_base + self._bandwidth_range * random.random()
        kop = self._kop_base + self._kop_range * (
            max(time, 0.0) ** self._kop_time_exponent
        )

        my_utils = [
            c.utility / self._estimated_max_utility
            if self._estimated_max_utility > 0
            else 0.0
            for c in candidates
        ]
        max_my = max(my_utils) if my_utils else 0.0
        if max_my > 0:
            my_utils = [u / max_my for u in my_utils]

        op_utils = [self._estimated_opponent_utility(c.bid, h) for c in candidates]
        max_op = max(op_utils) if op_utils else 0.0
        if max_op > 0:
            op_utils = [u / max_op for u in op_utils]

        best_idx = 0
        best_score = -1.0
        for i in range(len(candidates)):
            score = my_utils[i] * (1 - kop) + op_utils[i] * kop
            if score > best_score:
                best_score = score
                best_idx = i

        result = candidates[best_idx]

        # Prefer any previously seen opponent bid if it is better for us.
        for bd in self._opponent_history:
            if bd.utility > result.utility:
                result = bd

        return result.bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        time = state.relative_time
        self._accept_threshold = self._accept_threshold_at(time)
        return self._choose_next_bid(time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond based on the two-phase acceptance threshold."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        bd = BidDetails(bid=offer, utility=offer_utility)
        self._opponent_history.append(bd)
        if self._best_opponent_bid is None or offer_utility > self._best_opponent_bid.utility:
            self._best_opponent_bid = bd

        self._accept_threshold = self._accept_threshold_at(time)

        if offer_utility > self._accept_threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
