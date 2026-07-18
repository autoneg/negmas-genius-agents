"""ValueModelAgent from ANAC 2011."""

from __future__ import annotations

from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["ValueModelAgent"]


class ValueModelAgent(SAONegotiator):
    """
    ValueModelAgent from ANAC 2011 (Asaf, Dror and Gal).

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

        FIDELITY NOTE: The original agent uses a fairly elaborate temporal
        difference reinforcement-learning model (``ValueModeler`` /
        ``OpponentModeler``) that estimates, per issue value, how much
        utility the opponent has given up by offering it, using running
        estimates of standard deviation and reliability to split credit
        across issues. That machinery is impractical to reproduce exactly in
        a lightweight Python port. This implementation APPROXIMATES the
        opponent utility model with a simple per-issue value-frequency
        estimator (as used in the other ports in this package) and
        approximates the "concession so far" / "concession granted" signals
        with (a) our own utility of the opponent's best bid relative to
        their first bid, and (b) the estimated opponent utility of their own
        current bid relative to their first bid. The core, distinguishing
        behavior -- a slowly-decreasing acceptance threshold that only
        concedes when the opponent also appears to be conceding, a hard
        floor (``lowestAcceptable``), and a multi-stage end-game acceptance
        ladder that becomes progressively more lenient -- is reproduced.

    References:
        Original Genius class: ``agents.anac.y2011.ValueModelAgent.ValueModelAgent``

        ANAC 2011: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
    - First bid: our maximum-utility outcome.
    - While we have few observations of the opponent (first few rounds), we
      keep offering high-utility bids from the top of our outcome space.
    - Afterward we offer the highest-estimated-opponent-utility bid among our
      outcomes with utility >= our current acceptance threshold
      (``lowestApproved``) that we have not already offered ("bestScan").
    - As the deadline approaches (``t > 0.9``), we progressively relax
      towards the opponent's best bid so far, in a small number of discrete
      end-game stages, only if the opponent hasn't been conceding either.
    - Very close to the deadline (``t > 0.995``), if the opponent's best bid
      gave us at least 0.55 utility, we offer that bid (or accept anything
      close to it); otherwise we keep scanning for the best available bid.

    **Acceptance Strategy:**
    - Accept if the opponent's offer utility (to us) exceeds our current
      threshold ``lowestApproved`` (initially 0.98, floor 0.7).
    - The threshold concedes slowly: it only decreases when both (a) our
      utility for the opponent's improvement and (b) the estimated utility
      the opponent has given up both indicate real concession; the new
      threshold never drops below the hard floor or below the opponent's
      best offer (plus a small margin).
    - In the end-game ladder (``t`` in ``(0.9, 0.995]``), the threshold is
      pulled part-way toward the opponent's best bid, with acceptance
      possible once the offer is within a small margin of the (relaxed)
      threshold.

    **Opponent Modeling:**
    - Frequency-based: for each issue, count how often each value has been
      offered by the opponent; used both to estimate the opponent's utility
      for a bid and to construct opponent-friendly candidate bids.

    Args:
        initial_threshold: initial acceptance threshold, "lowestApproved"
            (default 0.98).
        lowest_acceptable: hard floor on the acceptance threshold,
            "lowestAcceptable" (default 0.7).
        endgame_start_time: time at which the end-game relaxation ladder
            begins (default 0.9).
        final_stage_time: time after which we fall back to the opponent's
            best bid if it is decent (default 0.995).
        final_stage_min_utility: minimum utility of opponent's best bid to
            consider offering/accepting it in the final stage (default 0.55).
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
        initial_threshold: float = 0.98,
        lowest_acceptable: float = 0.7,
        endgame_start_time: float = 0.9,
        final_stage_time: float = 0.995,
        final_stage_min_utility: float = 0.55,
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
        self._initial_threshold = initial_threshold
        self._lowest_acceptable = lowest_acceptable
        self._endgame_start_time = endgame_start_time
        self._final_stage_time = final_stage_time
        self._final_stage_min_utility = final_stage_min_utility

        self._outcome_space: SortedOutcomeSpace | None = None
        self._max_util: float = 1.0
        self._initialized = False

        self._opponent_issue_frequencies: dict[str, dict] = {}
        self._opponent_bid_count = 0
        self._opponent_start_bid_utility: float | None = None
        self._opponent_max_bid_utility: float = 0.0
        self._opponent_max_bid: Outcome | None = None

        self._lowest_approved: float = initial_threshold
        self._offered_bids: set = set()

    def _initialize(self) -> None:
        if self._initialized:
            return
        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        self._max_util = self._outcome_space.max_utility

        if self.nmi is not None:
            for issue in self.nmi.issues:
                self._opponent_issue_frequencies[issue.name] = {}

        self._lowest_approved = min(self._initial_threshold, self._max_util)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._initialize()
        self._opponent_bid_count = 0
        self._opponent_start_bid_utility = None
        self._opponent_max_bid_utility = 0.0
        self._opponent_max_bid = None
        self._lowest_approved = min(self._initial_threshold, self._max_util)
        self._offered_bids = set()

    def _update_opponent_model(self, bid: Outcome) -> float:
        """Update the frequency model and tracking stats; returns our utility
        of the received bid."""
        if bid is None or self.nmi is None or self.ufun is None:
            return 0.0

        utility = float(self.ufun(bid))
        self._opponent_bid_count += 1
        if self._opponent_start_bid_utility is None:
            self._opponent_start_bid_utility = utility
        if utility > self._opponent_max_bid_utility:
            self._opponent_max_bid_utility = utility
            self._opponent_max_bid = bid

        issues = self.nmi.issues
        for i, issue in enumerate(issues):
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)
            if val is not None:
                val_key = str(val)
                freqs = self._opponent_issue_frequencies[issue.name]
                freqs[val_key] = freqs.get(val_key, 0) + 1

        return utility

    def _get_opponent_utility(self, bid: Outcome) -> float:
        if self.nmi is None or not self._opponent_issue_frequencies:
            return 0.5
        issues = self.nmi.issues
        total = 0.0
        n = 0
        for i, issue in enumerate(issues):
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)
            if val is None:
                continue
            counts = self._opponent_issue_frequencies.get(issue.name, {})
            issue_total = sum(counts.values())
            if issue_total > 0:
                total += counts.get(str(val), 0) / issue_total
            else:
                total += 0.3
            n += 1
        return total / n if n > 0 else 0.5

    def _update_threshold(self) -> None:
        """
        Slowly concede our acceptance threshold, but only in proportion to
        how much the opponent appears to have conceded (approximating the
        original's dual concession-tracking logic), and rate-limited per
        round (mirroring the original's ``minConcessionMaker`` smoothing,
        which only ever nudges the threshold by small steps per time
        segment instead of jumping straight to the "ideal" concession).
        """
        if self._opponent_start_bid_utility is None:
            return
        # Mirror the original's "bidCount >= 4" gate before any concession
        # is considered at all.
        if self._opponent_bid_count < 4:
            return

        # How much our utility for the opponent's bids has improved.
        concession_ours = self._opponent_max_bid_utility - self._opponent_start_bid_utility
        # How much utility the opponent appears to have given up, estimated
        # via the frequency model applied to their best bid.
        opp_util_of_best = (
            self._get_opponent_utility(self._opponent_max_bid)
            if self._opponent_max_bid is not None
            else 1.0
        )
        concession_theirs = 1 - opp_util_of_best

        min_concession = max(0.0, min(concession_ours, concession_theirs))
        # Never require a concession from ourselves larger than what would be
        # needed to just match the opponent's best offer.
        min_concession = min(min_concession, 1 - self._opponent_max_bid_utility)

        candidate = 1 - min_concession
        candidate = max(candidate, self._lowest_acceptable)
        candidate = max(candidate, self._opponent_max_bid_utility + 0.001)

        # Rate-limit the concession: never drop the threshold by more than a
        # small step per round, so we stay tough unless the opponent keeps
        # conceding steadily over many rounds (approximating the original's
        # per-time-segment damping).
        max_step = 0.02
        candidate = max(candidate, self._lowest_approved - max_step)

        # Threshold is monotonically non-increasing.
        if candidate < self._lowest_approved:
            self._lowest_approved = candidate

    def _best_scan(self) -> Outcome | None:
        """Pick the highest-estimated-opponent-utility bid we haven't offered
        yet, among those above our current threshold."""
        if self._outcome_space is None:
            return None

        candidates = self._outcome_space.get_bids_above(self._lowest_approved)
        best_bid = None
        best_opp_util = -1.0
        for bd in candidates:
            key = bd.bid if not isinstance(bd.bid, dict) else tuple(sorted(bd.bid.items()))
            if key in self._offered_bids:
                continue
            opp_util = self._get_opponent_utility(bd.bid)
            if opp_util > best_opp_util:
                best_opp_util = opp_util
                best_bid = bd.bid

        if best_bid is None and candidates:
            best_bid = candidates[0].bid

        return best_bid

    def _mark_offered(self, bid: Outcome) -> None:
        key = bid if not isinstance(bid, dict) else tuple(sorted(bid.items()))
        self._offered_bids.add(key)

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        if not self._initialized:
            self._initialize()

        if self._outcome_space is None:
            return None

        t = state.relative_time

        if self._opponent_bid_count == 0:
            best_bids = self._outcome_space.get_bids_above(self._max_util - 0.001)
            bid = best_bids[0].bid if best_bids else None
            if bid is not None:
                self._mark_offered(bid)
            return bid

        # Final stage: fall back to opponent's best bid if it is decent.
        if t > self._final_stage_time:
            if self._opponent_max_bid_utility > self._final_stage_min_utility:
                bid = self._opponent_max_bid
            else:
                bid = self._best_scan()
            if bid is not None:
                self._mark_offered(bid)
            return bid

        # End-game relaxation ladder: pull the threshold part-way toward the
        # opponent's best bid the closer we get to the deadline.
        if t > self._endgame_start_time:
            progress = min(1.0, (t - self._endgame_start_time) / (
                self._final_stage_time - self._endgame_start_time
            ))
            concession_left = self._lowest_approved - self._opponent_max_bid_utility
            relaxed = self._lowest_approved - progress * max(0.0, concession_left) * 0.8
            relaxed = max(relaxed, self._opponent_max_bid_utility)
            self._lowest_approved = min(self._lowest_approved, max(relaxed, self._lowest_acceptable))
        else:
            self._update_threshold()

        bid = self._best_scan()
        if bid is not None:
            self._mark_offered(bid)
        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        offer_utility = self._update_opponent_model(offer)
        t = state.relative_time

        if t > self._final_stage_time:
            if self._opponent_max_bid_utility > self._final_stage_min_utility:
                if offer_utility >= self._opponent_max_bid_utility * 0.99:
                    return ResponseType.ACCEPT_OFFER
                return ResponseType.REJECT_OFFER

        if offer_utility >= self._lowest_approved - 0.01:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
