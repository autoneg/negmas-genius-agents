"""BRAMAgent2 from ANAC 2012."""

from __future__ import annotations

from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["BRAMAgent2"]


class BRAMAgent2(SAONegotiator):
    """
    BRAMAgent2 from ANAC 2012.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

        FIDELITY NOTE: The original agent pre-generates a large array of random
        bids and walks through it with random up/down jumps while limiting how
        often any single bid may be re-offered (``FREQUENCY_OF_PROPOSAL``) --
        this is a mechanism to deal with huge domains where full enumeration is
        infeasible. On the small domains this package targets, we instead
        enumerate the outcome space directly via ``SortedOutcomeSpace`` and
        pick from the candidates near/above the threshold, which is behaviorally
        equivalent for small/medium domains. The core, distinguishing mechanic --
        a 4-step time-based acceptance threshold (computed as a percentage
        between the best and worst bid we have offered) plus opponent-value-
        frequency-based bid construction -- is reproduced faithfully.

    References:
        Original Genius class: ``agents.anac.y2012.BRAMAgent2.BRAMAgent2``

        ANAC 2012: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
    - First bid: our maximum-utility outcome.
    - For the first ``OPPONENT_ARRAY_SIZE`` (10) opponent offers, we keep
      offering the maximum-utility bid while gathering opponent value
      frequency statistics.
    - Afterward, we build several candidate bids by sampling, per issue,
      the value the opponent has offered most often (a frequency-model
      "opponent-friendly" bid), and offer the highest-utility candidate that
      is still above our current threshold; if none qualifies, we fall back
      to our best bid so far.

    **Acceptance Strategy:**
    - The acceptance threshold moves through four flexibility levels based on
      relative time, computed as ``max_util - (max_util - min_util) * flex``
      where ``min_util``/``max_util`` are the utilities of the worst/best bid
      we have offered so far and ``flex`` is 0.07 (t < 1/3), 0.15 (t < 5/6),
      0.3 (t < 0.972), else 0.6.
    - Accept if the offer's utility is >= the current threshold, OR if it is
      >= the utility of the bid we would offer next.

    **Opponent Modeling:**
    - Frequency-based: for each issue, count how often each value has been
      offered by the opponent (mirrors the original's per-issue statistics
      structures for discrete/real/integer issues), and construct candidate
      bids using the most frequently offered value per issue.

    Args:
        opponent_history_size: number of opponent bids to gather before
            switching from "max-utility" offers to opponent-modeled offers
            (default 10, matching ``OPPONENT_ARRAY_SIZE``).
        flex_1: flexibility fraction for t < 1/3 (default 0.07).
        flex_2: flexibility fraction for t < 5/6 (default 0.15).
        flex_3: flexibility fraction for t < 0.972 (default 0.3).
        flex_4: flexibility fraction otherwise (default 0.6).
        n_candidates: number of opponent-modeled candidate bids to sample per
            round (default 10, matching the Java loop bound).
        best_bid_epsilon: epsilon tolerance used when searching for the
            maximum-utility bid (default 1e-9).
        phase_1_cutoff: relative-time cutoff below which ``flex_1`` applies
            (default 60.0/180.0, i.e. 1/3 of the negotiation).
        phase_2_cutoff: relative-time cutoff below which ``flex_2`` applies
            (default 150.0/180.0, i.e. 5/6 of the negotiation).
        phase_3_cutoff: relative-time cutoff below which ``flex_3`` applies
            (default 175.0/180.0, i.e. ~0.972 of the negotiation).
        max_checked: maximum number of outcomes to score when building an
            opponent-modeled bid (default 200).
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
        opponent_history_size: int = 10,
        flex_1: float = 0.07,
        flex_2: float = 0.15,
        flex_3: float = 0.3,
        flex_4: float = 0.6,
        n_candidates: int = 10,
        best_bid_epsilon: float = 1e-9,
        phase_1_cutoff: float = 60.0 / 180.0,
        phase_2_cutoff: float = 150.0 / 180.0,
        phase_3_cutoff: float = 175.0 / 180.0,
        max_checked: int = 200,
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
        self._opponent_history_size = opponent_history_size
        self._flex_1 = flex_1
        self._flex_2 = flex_2
        self._flex_3 = flex_3
        self._flex_4 = flex_4
        self._n_candidates = n_candidates
        self._best_bid_epsilon = best_bid_epsilon
        self._phase_1_cutoff = phase_1_cutoff
        self._phase_2_cutoff = phase_2_cutoff
        self._phase_3_cutoff = phase_3_cutoff
        self._max_checked = max_checked

        self._outcome_space: SortedOutcomeSpace | None = None
        self._max_util: float = 1.0
        self._best_bid: Outcome | None = None
        self._initialized = False

        self._opponent_bids: list[Outcome] = []
        self._opponent_issue_frequencies: dict[str, dict] = {}

        self._our_bid_utilities: list[float] = []
        self._threshold: float = 1.0
        self._random_seed_state = 100  # deterministic-ish sampling helper

    def _initialize(self) -> None:
        if self._initialized:
            return
        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        self._max_util = self._outcome_space.max_utility
        best_bids = self._outcome_space.get_bids_above(self._max_util - self._best_bid_epsilon)
        self._best_bid = best_bids[0].bid if best_bids else None

        if self.nmi is not None:
            for issue in self.nmi.issues:
                self._opponent_issue_frequencies[issue.name] = {}

        self._our_bid_utilities = [self._max_util]
        self._threshold = self._max_util
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._initialize()
        self._opponent_bids = []
        self._our_bid_utilities = [self._max_util]
        self._threshold = self._max_util

    def _update_opponent_model(self, bid: Outcome) -> None:
        if bid is None or self.nmi is None:
            return
        self._opponent_bids.append(bid)
        issues = self.nmi.issues
        for i, issue in enumerate(issues):
            val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)
            if val is not None:
                val_key = str(val)
                freqs = self._opponent_issue_frequencies[issue.name]
                freqs[val_key] = freqs.get(val_key, 0) + 1

    def _get_new_threshold(self, t: float) -> float:
        min_util = self._our_bid_utilities[-1] if self._our_bid_utilities else self._max_util
        max_util = self._max_util

        if t < self._phase_1_cutoff:
            flex = self._flex_1
        elif t < self._phase_2_cutoff:
            flex = self._flex_2
        elif t < self._phase_3_cutoff:
            flex = self._flex_3
        else:
            flex = self._flex_4

        return max_util - (max_util - min_util) * flex

    def _build_opponent_modeled_bid(self) -> Outcome | None:
        """
        Construct a bid using, per issue, the value the opponent has offered
        most frequently -- mirroring the intent of the Java agent's
        ``createBidByOpponentModeling`` (adapted here to enumerate all
        possible outcomes and pick the best-scoring one for small domains
        instead of constructing arbitrary partial bids that may not be valid
        outcomes).
        """
        if self._outcome_space is None or self.nmi is None:
            return None

        issues = self.nmi.issues

        def opponent_score(bid: Outcome) -> float:
            score = 0.0
            for i, issue in enumerate(issues):
                val = bid[i] if isinstance(bid, tuple) else bid.get(issue.name)
                if val is None:
                    continue
                counts = self._opponent_issue_frequencies.get(issue.name, {})
                total = sum(counts.values())
                if total > 0:
                    score += counts.get(str(val), 0) / total
            return score

        best_bid = None
        best_score = -1.0
        best_util = 0.0
        checked = 0
        for bd in self._outcome_space.outcomes:
            if bd.utility < self._threshold:
                break
            score = opponent_score(bd.bid)
            if score > best_score:
                best_score = score
                best_bid = bd.bid
                best_util = bd.utility
            checked += 1
            if checked >= self._max_checked:  # keep this cheap; small domains only need a few
                break

        if best_bid is None:
            return None
        return best_bid

    def _select_bid_to_offer(self, t: float) -> Outcome | None:
        if len(self._opponent_bids) < self._opponent_history_size:
            return self._best_bid

        bid = self._build_opponent_modeled_bid()
        if bid is None:
            return self._best_bid
        return bid

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        if not self._initialized:
            self._initialize()

        t = state.relative_time
        self._threshold = self._get_new_threshold(t)

        if not self._opponent_bids:
            bid = self._best_bid
        else:
            bid = self._select_bid_to_offer(t)

        if bid is not None and self.ufun is not None:
            self._our_bid_utilities.append(float(self.ufun(bid)))

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
        self._threshold = self._get_new_threshold(t)

        offer_utility = float(self.ufun(offer))

        if offer_utility >= self._threshold:
            self._update_opponent_model(offer)
            return ResponseType.ACCEPT_OFFER

        next_bid = self._select_bid_to_offer(t) if self._opponent_bids else self._best_bid
        if next_bid is not None:
            next_utility = float(self.ufun(next_bid))
            if offer_utility >= next_utility:
                self._update_opponent_model(offer)
                return ResponseType.ACCEPT_OFFER

        self._update_opponent_model(offer)
        return ResponseType.REJECT_OFFER
