"""
BetaOne - reimplementation of ``agents.anac.y2018.beta_one.Group2``.

This module implements BetaOne, a Python port of the Java Genius agent
``agents.anac.y2018.beta_one.Group2`` that competed in the Automated
Negotiating Agents Competition (ANAC) 2018. ``BetaOne`` is the canonical name
for this agent in ``negmas.genius.ginfo.GENIUS_INFO``.

References:
    - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
    - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
      Competition (ANAC 2018)." IJCAI 2019.
    - Genius framework: https://ii.tudelft.nl/genius/
    - Original Java class: agents.anac.y2018.beta_one.Group2
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["BetaOne"]


def _lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate between ``a`` and ``b``, clamping ``t`` to [0, 1]."""
    if t < 0:
        t = 0.0
    elif t > 1:
        t = 1.0
    return a + (b - a) * t


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


class _SimpleRegression:
    """
    Minimal ordinary-least-squares slope tracker.

    Equivalent to the ``slope`` computed by Apache Commons Math's
    ``SimpleRegression`` used by the original Java agent to track how an
    opponent's offered utility (to us) evolves over negotiation time.
    """

    def __init__(self) -> None:
        self._xs: list[float] = []
        self._ys: list[float] = []

    def add_data(self, x: float, y: float) -> None:
        self._xs.append(x)
        self._ys.append(y)

    def slope(self) -> float:
        n = len(self._xs)
        if n < 2:
            return 0.0
        mean_x = sum(self._xs) / n
        mean_y = sum(self._ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(self._xs, self._ys))
        den = sum((x - mean_x) ** 2 for x in self._xs)
        if den == 0:
            return 0.0
        return num / den


class BetaOne(SAONegotiator):
    """
    BetaOne, a Python port of the ANAC 2018 Genius agent
    ``agents.anac.y2018.beta_one.Group2``.

    Note:
        This is an AI-generated reimplementation based on the original Java
        code from the Genius framework. It may not behave identically to the
        original. See the "Simplifications" section below for known
        deviations.

    The original Java agent (``Group2``, extending a small ``GroupNegotiator``
    / ``AntiAnalysis`` framework) computes a "fair" reference utility level by
    building an *anti*-utility function (its own utility function with each
    issue's evaluator values reversed in rank) and finding the
    Kalai-Smorodinsky point of the bid space spanned by (own utility,
    anti-utility). It then concedes from full self-interest down to a
    "selfish point" interpolated between that fair reference point and the
    maximum utility, tightens or freezes concession if the opponent appears
    to be exploiting it (via a linear-regression "betrayal" detector on the
    opponent's offered utility over time), and offers uniformly random bids
    within the resulting acceptable utility band.

    **Offering Strategy:**
        Maintains a per-opponent acceptable utility range (a narrow "box")
        that is recomputed after every incoming offer:

        - First offer: the best possible bid.
        - Early phase (``t < OBSERVE_DURATION = 0.75``, time already adjusted
          for discounting): the box linearly shrinks from
          ``[max_utility, max_utility]`` toward ``[selfish_point,
          selfish_point + box_size]``.
        - Late phase (``t >= OBSERVE_DURATION``) while the opponent has not
          "betrayed" us: the box continues to concede linearly from
          ``selfish_point`` toward the fair reference utility
          (``anti_kalai_util``), narrowing toward it.
        - If the opponent has "betrayed" us (see below) or ``t > 1``: the box
          freezes/reverts to ``[selfish_point, max_utility]`` -- concession
          stops.

        Within the current box, a bid is picked uniformly at random among all
        outcomes whose utility falls in that range (falling back to the bid
        nearest the box's midpoint if none is found).

    **Acceptance Strategy:**
        Accepts any offer whose utility is greater than or equal to the lower
        bound of the current acceptable box (mirrors ``isAcceptable`` in the
        original, which only checks the lower bound of the box).

    **Opponent Modeling / Betrayal Detection:**
        Tracks a simple linear regression of the opponent's offered utility
        (to us) against negotiation time. If the sum of the opponent's
        estimated slope and our own "selfish slope" (how fast we are
        supposed to be conceding) drops below ``-SLOPE_TOLERANCE``, the
        opponent is considered to have "betrayed" us (i.e., it is not
        reciprocating our concessions), and further concession is frozen.

    **Simplifications (deviations from the original Java agent):**
        1. ``initializeHistory`` in the original uses Genius's persistent
           cross-session storage (``StandardInfoList``) of utilities received
           in *previous* negotiation sessions on the same domain to bias the
           ``selfishRatio`` between ``MIN_SELFISH_RATIO`` and
           ``MAX_SELFISH_RATIO``. NegMAS's ``SAONegotiator`` has no
           equivalent standard persistent-storage mechanism across separate
           mechanism runs, so this port always uses the midpoint
           ``(MIN_SELFISH_RATIO + MAX_SELFISH_RATIO) / 2`` as the original
           does before any history is available.
        2. The original's ``AntiAnalysis`` builds a full per-issue "anti"
           (rank-reversed) utility function and computes the exact
           Kalai-Smorodinsky point of the resulting 2D bid space (own utility
           vs. anti-utility) using Genius's ``BidSpace`` geometry routines.
           Reproducing that geometry (Pareto frontier + Kalai-Smorodinsky
           solution over an issue-level rank-reversed utility function) is
           out of scope for this port. Instead we approximate the resulting
           reference utility directly as the midpoint of our achievable
           utility range: ``anti_kalai_util = min_utility + 0.5 *
           (max_utility - min_utility)``. This is a deliberate, simple
           stand-in rather than an exact derivation, but it is not an
           arbitrary choice either: it matches exactly the *fallback* value
           (``0.5``) the original Java code uses whenever an issue's
           evaluator is not a discrete evaluator, and -- since the KS point
           of a roughly-antisymmetric preference-reversal space is expected
           to lie near the middle of the utility range -- it is a reasonable
           qualitative stand-in for fully-discrete domains too. The actual
           Java ``antiKalaiUtil`` for a given discrete domain may differ
           from 0.5 (shifting the resulting ``selfish_point``), which is a
           known source of quantitative (not qualitative) divergence from
           the original agent.
        3. Regression / betrayal detection is only tracked against a single
           opponent id (the negotiation ``source``), matching the common
           bilateral ``SAOMechanism`` setting; the original's per-``AgentID``
           map generalizes to multilateral settings which are out of scope
           here.

    Args:
        observe_duration: Fraction of (discount-adjusted) time during which
            the acceptable box shrinks from full self-interest to the
            selfish point (default 0.75, matches Java ``OBSERVE_DURATION``).
        box_size: Width of the acceptable utility box, in normalized
            utility-range units (default 0.05, matches Java ``BOX_SIZE``).
        slope_tolerance: Threshold below which the sum of estimated slopes
            triggers betrayal detection (default 0.035, matches Java
            ``SLOPE_TOLERANCE``).
        min_selfish_ratio: Lower bound for interpolating the selfish point
            between the fair reference utility and full self-interest
            (default 0.15, matches Java ``MIN_SELFISH_RATIO``).
        max_selfish_ratio: Upper bound for interpolating the selfish point
            (default 0.30, matches Java ``MAX_SELFISH_RATIO``).
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
        observe_duration: float = 0.75,
        box_size: float = 0.05,
        slope_tolerance: float = 0.035,
        min_selfish_ratio: float = 0.15,
        max_selfish_ratio: float = 0.30,
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
        self._observe_duration = observe_duration
        self._box_size = box_size
        self._slope_tolerance = slope_tolerance
        self._min_selfish_ratio = min_selfish_ratio
        self._max_selfish_ratio = max_selfish_ratio

        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        self._best_bid: Outcome | None = None
        self._max_utility: float = 1.0
        self._min_utility: float = 0.0
        self._anti_kalai_util: float = 0.5
        self._selfish_point: float = 0.9
        self._discount_factor: float = 1.0

        self._last_received_offer: Outcome | None = None
        self._regressions: dict[str, _SimpleRegression] = {}
        self._acceptable_ranges: dict[str, tuple[float, float]] = {}
        self._betrayed: dict[str, bool] = {}

    def _initialize(self) -> None:
        """Initialize the outcome space and the "anti-analysis" reference point."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._best_bid = self._outcome_space.outcomes[0].bid
            self._max_utility = self._outcome_space.max_utility
            self._min_utility = self._outcome_space.min_utility

        util_range = self._max_utility - self._min_utility
        # Approximate Kalai-Smorodinsky point of the (utility, anti-utility)
        # space as the midpoint of the achievable utility range -- see the
        # "Simplifications" note in the class docstring.
        self._anti_kalai_util = self._min_utility + 0.5 * util_range

        selfish_ratio = (self._min_selfish_ratio + self._max_selfish_ratio) / 2.0
        self._selfish_point = _lerp(
            self._anti_kalai_util, self._max_utility, selfish_ratio
        )

        self._discount_factor = getattr(self.ufun, "discount_factor", 1.0) or 1.0

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()
        self._last_received_offer = None
        self._regressions = {}
        self._acceptable_ranges = {}
        self._betrayed = {}

    def _regression_for(self, source: str) -> _SimpleRegression:
        if source not in self._regressions:
            self._regressions[source] = _SimpleRegression()
        return self._regressions[source]

    def _selfish_slope(self) -> float:
        """Estimate of our own concession slope (always <= 0)."""
        discount = self._discount_factor if self._discount_factor > 1e-9 else 1e-9
        util_range = self._max_utility - self._min_utility
        box_size = self._box_size * util_range
        d_util = (self._selfish_point + box_size / 2.0) - self._max_utility
        return d_util / discount

    def _has_betrayed(self, source: str) -> bool:
        my_slope = self._selfish_slope()
        opp_slope = self._regression_for(source).slope()
        util_range = self._max_utility - self._min_utility
        tolerance = self._slope_tolerance * util_range
        return (opp_slope + my_slope) < -tolerance

    def _get_box(self, time: float, betrayed: bool) -> tuple[float, float]:
        """Compute the acceptable utility box at a given (discount-adjusted) time.

        Mirrors the Java ``AntiAnalysis.getBox`` branch order exactly: the
        early-phase (``time < observe_duration``) descending box always
        applies regardless of ``betrayed`` -- betrayal only freezes
        concession once we are past the observation window.
        """
        util_range = self._max_utility - self._min_utility
        box_size = self._box_size * util_range

        if time > 1:
            return (self._selfish_point, self._max_utility)

        if time < self._observe_duration:
            t = time / self._observe_duration if self._observe_duration > 0 else 1.0
            lower = _lerp(self._max_utility, self._selfish_point, t)
            upper = _clamp(lower + box_size, lower, self._max_utility)
            return (lower, upper)

        if not betrayed:
            denom = 1.0 - self._observe_duration
            t = (time - self._observe_duration) / denom if denom > 0 else 1.0
            lower = _lerp(
                self._selfish_point, self._anti_kalai_util - box_size / 2.0, t
            )
            upper = _lerp(
                self._selfish_point + box_size,
                self._anti_kalai_util + box_size / 2.0,
                t,
            )
            return (lower, upper)

        return (self._selfish_point, self._max_utility)

    def _select_bid(self, source: str) -> Outcome | None:
        """Select a bid within the current acceptable range for ``source``."""
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return self._best_bid

        lower, upper = self._acceptable_ranges.get(
            source, (self._selfish_point, self._max_utility)
        )
        if lower > upper:
            lower, upper = upper, lower

        candidates = self._outcome_space.get_bids_in_range(lower, upper)
        if not candidates:
            bid_details = self._outcome_space.get_bid_near_utility((lower + upper) / 2.0)
            return bid_details.bid if bid_details else self._best_bid

        return random.choice(candidates).bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        if self._last_received_offer is None:
            return self._best_bid

        source = dest or "opponent"
        return self._select_bid(source)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        opponent_id = source or "opponent"
        self._last_received_offer = offer
        offer_utility = float(self.ufun(offer))

        raw_time = state.relative_time
        self._regression_for(opponent_id).add_data(raw_time, offer_utility)

        betrayed = self._has_betrayed(opponent_id)
        self._betrayed[opponent_id] = betrayed

        discount = self._discount_factor if self._discount_factor > 1e-9 else 1e-9
        adjusted_time = raw_time / discount

        acceptable_range = self._get_box(adjusted_time, betrayed)
        self._acceptable_ranges[opponent_id] = acceptable_range

        lower_bound = min(acceptable_range)
        if offer_utility >= lower_bound:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
