"""NiceTitForTat from ANAC 2011."""

from __future__ import annotations

from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["NiceTitForTat"]


class NiceTitForTat(SAONegotiator):
    """
    NiceTitForTat from ANAC 2011 (Tim Baarslag).

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

        FIDELITY NOTE: The original agent estimates the opponent's utility
        function with a full Bayesian opponent model (``BayesianOpponentModelScalable``)
        and computes an exact Nash bargaining point over the estimated bid space
        (``BidSpace.getNash()``). This is a heavy, domain-size dependent
        computation that is impractical to reproduce exactly in a lightweight
        Python port. This implementation APPROXIMATES the opponent model with a
        simple frequency-based estimator (tracking how often the opponent picks
        each value per issue) and approximates the Nash point by scanning our
        own outcome space for the bid that maximizes the product of our utility
        and the estimated opponent utility. The core, distinguishing mechanic of
        this agent -- conceding in direct proportion to how much the opponent
        has conceded, relative to an estimated Nash point -- is reproduced
        faithfully. This is NOT a Boulware/time-based concession agent: time is
        only used for a small end-game bonus and a domain-size/discount based
        speed-up, not to drive the concession curve itself.

    References:
        Original Genius class: ``agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat``

        ANAC 2011: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
    - Opening bid: our maximum-utility outcome.
    - Estimate ``myNashUtility``: our utility at the (estimated) Nash point,
      scaled by a multiplier ``1.4 - 0.6 * gap`` (gap = 1 - utility of the
      opponent's first bids), clamped to ``[0.5, 1.0]``.
    - Estimate the opponent's concession: how far the best bid they have
      offered (in our utility) has moved from their first bids, relative to
      the distance between their first bids and our Nash utility.
      ``opponent_concede_factor = clip((max_opp_util - min_first_bids) /
      (nash_utility - min_first_bids), 0, 1)``.
    - Tit-for-tat: we concede by the same factor:
      ``my_concession = opponent_concede_factor * (1 - nash_utility)``,
      giving ``target = 1 - my_concession``.
    - A small end-game/discount "bonus" nudges the target down slightly
      toward the Nash point as time runs out or when there is a discount
      factor, mirroring the original's ``getBonus()``.
    - We never offer something worse for us than the best bid the opponent
      has made so far: if their best bid dominates our planned bid, we
      offer their best bid instead ("makeAppropriate").
    - Among candidate bids near the target utility, we prefer ones that are
      estimated to be good for the opponent (frequency-model based).

    **Acceptance Strategy:**
    - Accept if the opponent's offer utility (to us) is >= the utility of
      the bid we are about to offer.
    - After t > 0.98, use a simple expected-value-of-waiting heuristic:
      accept if the offer is better than the (probability-weighted) average
      of recent better offers we might still see.

    **Opponent Modeling:**
    - Frequency-based: for each issue, count how often each value has been
      offered by the opponent; issues with more consistent value choices are
      assumed more important, and estimated opponent utility of a bid is a
      weighted sum of value-frequency-derived preferences.

    Args:
        first_bids_time_window: fraction of time used to determine the
            opponent's starting point (default 0.01, following the Java agent).
        bid_tolerance: tolerance range used when searching for bids near the
            target utility (default 0.03).
        late_accept_time: time after which the expected-value-of-waiting
            acceptance heuristic is used (default 0.98).
        nash_multiplier_base: base of the Nash-utility scaling multiplier
            ``base - gap_coeff * gap`` (default 1.4).
        nash_multiplier_gap_coeff: coefficient of the gap in the Nash-utility
            scaling multiplier (default 0.6).
        nash_utility_floor: lower clamp on the estimated Nash utility
            (default 0.5).
        discount_bonus_base: base of the discount bonus ``base - factor * df``
            (default 0.5).
        discount_bonus_factor: coefficient of the discount factor in the
            discount bonus (default 0.4).
        big_domain_threshold: outcome count above which a domain is treated
            as "big" (default 10000).
        big_domain_min_time: time after which the end-game time bonus starts
            on big domains (default 0.85).
        small_domain_min_time: time after which the end-game time bonus starts
            on small domains (default 0.91).
        time_bonus_multiplier: slope of the end-game time bonus
            ``min(1, mult * (t - min_time))`` (default 20.0).
        nash_update_cutoff: time after which the Nash estimate stops being
            updated (default 0.99).
        big_domain_nash_update_time: on big domains, time after which the Nash
            estimate stops being updated (default 0.5).
        concession_denom_epsilon: denominator epsilon below which the
            opponent concession factor defaults to full concession
            (default 1e-9).
        late_accept_window_epsilon: floor on the look-back window used by the
            expected-value-of-waiting acceptance heuristic (default 1e-6).
        unknown_value_preference: estimated opponent preference for unseen
            values (default 0.3).
        best_bid_epsilon: tolerance below max utility used to fetch the
            opening best bid (default 0.001).
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
        first_bids_time_window: float = 0.01,
        bid_tolerance: float = 0.03,
        late_accept_time: float = 0.98,
        nash_multiplier_base: float = 1.4,
        nash_multiplier_gap_coeff: float = 0.6,
        nash_utility_floor: float = 0.5,
        discount_bonus_base: float = 0.5,
        discount_bonus_factor: float = 0.4,
        big_domain_threshold: int = 10000,
        big_domain_min_time: float = 0.85,
        small_domain_min_time: float = 0.91,
        time_bonus_multiplier: float = 20.0,
        nash_update_cutoff: float = 0.99,
        big_domain_nash_update_time: float = 0.5,
        concession_denom_epsilon: float = 1e-9,
        late_accept_window_epsilon: float = 1e-6,
        unknown_value_preference: float = 0.3,
        best_bid_epsilon: float = 0.001,
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
        self._first_bids_time_window = first_bids_time_window
        self._bid_tolerance = bid_tolerance
        self._late_accept_time = late_accept_time
        self._nash_multiplier_base = nash_multiplier_base
        self._nash_multiplier_gap_coeff = nash_multiplier_gap_coeff
        self._nash_utility_floor = nash_utility_floor
        self._discount_bonus_base = discount_bonus_base
        self._discount_bonus_factor = discount_bonus_factor
        self._big_domain_threshold = big_domain_threshold
        self._big_domain_min_time = big_domain_min_time
        self._small_domain_min_time = small_domain_min_time
        self._time_bonus_multiplier = time_bonus_multiplier
        self._nash_update_cutoff = nash_update_cutoff
        self._big_domain_nash_update_time = big_domain_nash_update_time
        self._concession_denom_epsilon = concession_denom_epsilon
        self._late_accept_window_epsilon = late_accept_window_epsilon
        self._unknown_value_preference = unknown_value_preference
        self._best_bid_epsilon = best_bid_epsilon

        # Will be initialized when negotiation starts
        self._outcome_space: SortedOutcomeSpace | None = None
        self._max_util: float = 1.0
        self._initialized = False

        # Opponent bid history: list of (time, bid, utility_to_us)
        self._opponent_history: list[tuple[float, Outcome, float]] = []

        # Frequency-based opponent model
        self._opponent_issue_frequencies: dict[str, dict] = {}
        self._opponent_issue_weights: dict[str, float] = {}

        # Tracking
        self._best_opponent_bid: Outcome | None = None
        self._best_opponent_utility: float = 0.0
        self._my_last_bid: Outcome | None = None
        self._my_last_utility: float = 1.0
        self._my_nash_utility: float = 0.7
        self._initial_gap: float = 0.0

    def _initialize(self) -> None:
        if self._initialized:
            return
        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        self._max_util = self._outcome_space.max_utility

        if self.nmi is not None:
            issues = self.nmi.issues
            n_issues = len(issues)
            weight = 1.0 / n_issues if n_issues > 0 else 1.0
            for issue in issues:
                self._opponent_issue_weights[issue.name] = weight
                self._opponent_issue_frequencies[issue.name] = {}

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._initialize()
        self._opponent_history = []
        self._best_opponent_bid = None
        self._best_opponent_utility = 0.0
        self._my_last_bid = None
        self._my_last_utility = 1.0
        self._my_nash_utility = 0.7
        self._initial_gap = 0.0

    def _update_opponent_model(self, bid: Outcome, t: float) -> None:
        if bid is None or self.nmi is None or self.ufun is None:
            return

        utility = float(self.ufun(bid))
        self._opponent_history.append((t, bid, utility))

        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility
            self._best_opponent_bid = bid

        issues = self.nmi.issues
        for i, issue in enumerate(issues):
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)
            if val is not None:
                val_key = str(val)
                freqs = self._opponent_issue_frequencies[issue.name]
                freqs[val_key] = freqs.get(val_key, 0) + 1

        if len(self._opponent_history) >= 3:
            self._update_issue_weights()

    def _update_issue_weights(self) -> None:
        if self.nmi is None:
            return
        issues = self.nmi.issues
        consistency_scores: dict[str, float] = {}
        for issue in issues:
            counts = self._opponent_issue_frequencies.get(issue.name, {})
            if not counts:
                consistency_scores[issue.name] = 1.0
                continue
            total = sum(counts.values())
            max_count = max(counts.values()) if counts else 0
            consistency_scores[issue.name] = max_count / total if total > 0 else 0.5

        total_consistency = sum(consistency_scores.values())
        if total_consistency > 0:
            for issue in issues:
                self._opponent_issue_weights[issue.name] = (
                    consistency_scores[issue.name] / total_consistency
                )

    def _get_opponent_utility(self, bid: Outcome) -> float:
        if self.nmi is None or not self._opponent_issue_frequencies:
            return 0.5

        issues = self.nmi.issues
        total_utility = 0.0
        for i, issue in enumerate(issues):
            weight = self._opponent_issue_weights.get(issue.name, 0.0)
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)
            if val is not None:
                val_key = str(val)
                counts = self._opponent_issue_frequencies.get(issue.name, {})
                if val_key in counts and counts:
                    max_count = max(counts.values())
                    value_preference = counts[val_key] / max_count if max_count > 0 else 0.5
                else:
                    value_preference = self._unknown_value_preference
                total_utility += weight * value_preference

        return min(1.0, max(0.0, total_utility))

    def _get_min_utility_of_opponent_first_bids(self) -> float:
        """
        Utility (to us) of the opponent's opening bids, used as a reference
        point for whether the opponent is conceding.
        """
        first_window = [
            u for (t, _, u) in self._opponent_history if t <= self._first_bids_time_window
        ]
        if not first_window:
            if self._opponent_history:
                return self._opponent_history[0][2]
            return 0.0
        return min(first_window)

    def _estimate_nash_utility(self) -> float:
        """
        Approximate the Nash bargaining point by scanning our outcome space
        for the bid maximizing (our_utility * estimated_opponent_utility),
        then taking our utility of that bid. This mirrors the role of
        ``BidSpace.getNash()`` in the original, without an exact Bayesian
        opponent model.
        """
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return 0.7

        best_score = -1.0
        best_util = 0.7
        # Only consider a reasonable prefix (already sorted by our utility,
        # descending) to keep this cheap on larger domains.
        for bd in self._outcome_space.outcomes:
            opp_util = self._get_opponent_utility(bd.bid)
            score = bd.utility * opp_util
            if score > best_score:
                best_score = score
                best_util = bd.utility

        return best_util

    def _update_my_nash_utility(self) -> None:
        min_first_bids = self._get_min_utility_of_opponent_first_bids()
        self._initial_gap = 1 - min_first_bids

        nash_estimate = self._estimate_nash_utility()
        nash_multiplier = max(0.0, self._nash_multiplier_base - self._nash_multiplier_gap_coeff * self._initial_gap)
        nash_utility = nash_estimate * nash_multiplier

        nash_utility = min(1.0, nash_utility)
        nash_utility = max(self._nash_utility_floor, nash_utility)

        self._my_nash_utility = nash_utility

    def _get_bonus(self, t: float) -> float:
        """Small bonus nudging our target toward the Nash point near the deadline."""
        discount_factor = 1.0
        if self.ufun is not None:
            discount_factor = getattr(self.ufun, "discount_factor", None) or 1.0
        discount_bonus = self._discount_bonus_base - self._discount_bonus_factor * discount_factor

        domain_size = 0
        if self._outcome_space is not None:
            domain_size = len(self._outcome_space.outcomes)
        is_big_domain = domain_size > self._big_domain_threshold

        min_time = self._big_domain_min_time if is_big_domain else self._small_domain_min_time
        time_bonus = 0.0
        if t > min_time:
            time_bonus = min(1.0, self._time_bonus_multiplier * (t - min_time))

        bonus = max(discount_bonus, time_bonus)
        return min(1.0, max(0.0, bonus))

    def _select_bid(self, target_utility: float) -> Outcome | None:
        if self._outcome_space is None:
            return None

        tolerance = self._bid_tolerance
        candidates = self._outcome_space.get_bids_in_range(
            max(0.0, target_utility - tolerance), min(1.0, target_utility + tolerance)
        )
        if not candidates:
            bid_details = self._outcome_space.get_bid_near_utility(target_utility)
            return bid_details.bid if bid_details else None
        if len(candidates) == 1:
            return candidates[0].bid

        best_bid = None
        best_opp_util = -1.0
        for bd in candidates:
            opp_util = self._get_opponent_utility(bd.bid)
            if opp_util > best_opp_util:
                best_opp_util = opp_util
                best_bid = bd.bid
        return best_bid

    def _make_appropriate(self, planned_bid: Outcome | None) -> Outcome | None:
        """
        Prevent overshooting: if the opponent's best bid so far is better for
        us than our planned bid, offer that instead.
        """
        if planned_bid is None or self.ufun is None:
            return planned_bid
        if self._best_opponent_bid is None:
            return planned_bid

        planned_util = float(self.ufun(planned_bid))
        if self._best_opponent_utility >= planned_util:
            return self._best_opponent_bid
        return planned_bid

    def _plan_next_bid(self, state: SAOState) -> Outcome | None:
        """Compute (without side effects other than the Nash estimate) the bid
        we would offer next, given the current state. Used both to generate
        our proposal and to evaluate whether an incoming offer is at least as
        good as what we would offer ourselves."""
        t = state.relative_time

        if self._my_last_bid is None:
            if self._outcome_space is not None:
                best_bids = self._outcome_space.get_bids_above(self._max_util - self._best_bid_epsilon)
                if best_bids:
                    return best_bids[0].bid
            return None

        # Update opponent model / Nash estimate if we still have time to.
        can_update = t <= self._nash_update_cutoff
        if self._outcome_space is not None and len(self._outcome_space.outcomes) > self._big_domain_threshold:
            can_update = can_update and t <= self._big_domain_nash_update_time
        if can_update:
            self._update_my_nash_utility()

        min_first_bids = self._get_min_utility_of_opponent_first_bids()
        nash = self._my_nash_utility

        opponent_concession = self._best_opponent_utility - min_first_bids
        denom = nash - min_first_bids
        if denom > self._concession_denom_epsilon:
            opponent_concede_factor = min(1.0, max(0.0, opponent_concession / denom))
        else:
            opponent_concede_factor = 1.0

        my_concession = opponent_concede_factor * (1 - nash)
        target = 1 - my_concession

        gap_to_nash = max(0.0, target - nash)
        bonus = self._get_bonus(t)
        target -= bonus * gap_to_nash

        bid = self._select_bid(target)
        bid = self._make_appropriate(bid)
        return bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        if not self._initialized:
            self._initialize()

        bid = self._plan_next_bid(state)

        if bid is not None and self.ufun is not None:
            self._my_last_bid = bid
            self._my_last_utility = float(self.ufun(bid))

        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        t = state.relative_time
        self._update_opponent_model(offer, t)

        offer_utility = float(self.ufun(offer))

        my_next_bid = self._plan_next_bid(state)
        if my_next_bid is not None:
            my_next_utility = float(self.ufun(my_next_bid))
            if offer_utility >= my_next_utility:
                return ResponseType.ACCEPT_OFFER

        if t < self._late_accept_time:
            return ResponseType.REJECT_OFFER

        # Expected-value-of-waiting heuristic near the deadline.
        time_left = 1 - t
        window = max(time_left, self._late_accept_window_epsilon)
        recent_better = [
            u for (bt, _, u) in self._opponent_history if bt >= t - window and u >= offer_utility
        ]
        n = len(recent_better)
        p = min(1.0, time_left / window)
        p_all_miss = (1 - p) ** n if n > 0 else 1.0
        p_at_least_one_hit = 1 - p_all_miss
        avg = sum(recent_better) / n if n > 0 else 0.0
        expected_value_of_waiting = p_at_least_one_hit * avg

        if offer_utility > expected_value_of_waiting:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER
