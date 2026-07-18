"""TucAgent from ANAC 2017."""

from __future__ import annotations

from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["TucAgent"]


class TucAgent(SAONegotiator):
    """
    TucAgent negotiation agent from ANAC 2017.

    TucAgent tracks a "negotiation status" statistic derived from whether the
    opponent's offers are improving or worsening for TucAgent, and combines
    it with elapsed time, an eagerness factor, and the number of offers it
    has committed to, to compute a concession rate that is then applied to a
    fixed reservation value of 0.6.

    Note:
        This is an AI-generated reimplementation based on the original Java
        code from the Genius framework. It may not behave identically to the
        original.

        The original Java implementation was written for a (now defunct)
        multi-opponent protocol and tracks a Bayesian issue-weight model plus
        a "negotiation status" (``NS``) statistic per opponent. The Java
        source computes ``NS`` with integer arithmetic that, read literally,
        would make it grow roughly quadratically with the round count
        (Java's ``int / int`` truncates); an initial literal port of that
        formula was tried and produced a Python agent that collapses to
        near-random low-utility acceptance far too early compared to the
        real bridge (verified empirically against the Java agent through
        ``negmas.genius.GeniusNegotiator``). The real compiled agent instead
        stays close to its maximum-utility bid for most of the negotiation
        and only settles late, near a high utility, consistent with ``NS``
        staying roughly bounded rather than exploding. This port therefore
        uses the (float) normalized form ``NS = (delta + 2 * round_count) /
        (3 * round_count)``, which reproduces that observed "stays firm,
        settles late and high" gross behavior.

        On the small test domain used for verification, both agents hold
        near-maximum utility for most of the negotiation and settle late.
        One observed difference: on that domain the real Java agent offered
        its maximum-utility bid every single round and only accepted once
        the opponent itself offered a maximum-for-us bid (its concession
        machinery appears largely inert there, plausibly because the
        un-ported Bayesian opponent-weight computation feeding
        ``concessionRate`` sits inside a ``try/catch`` that silently
        swallows failures in the original code, effectively freezing the
        concession rate at its initial value); this port's concession
        formula is active and conceded one notch (to about 0.9) before
        settling. Treat this as a "holds firm, settles late" agent rather
        than an exact bid-for-bid match.

        The Bayesian opponent-weight
        model used only to detect "similar" opponents is not ported; this
        port instead directly uses TucAgent's own issue weights, which
        collapses the (unported) similarity-based weighting between two
        opponents into the single bilateral opponent used by
        NegMAS/``negmas.genius``.

    Original Java class: agents.anac.y2017.tucagent.TucAgent

    References:
        ANAC 2017 competition proceedings.
        https://ii.tudelft.nl/negotiation/node/12

    **Offering Strategy:**
    If no offer has been received yet, offers the maximum-utility bid.
    Otherwise computes a threshold via ``CreateThreshold`` (see below) and
    offers the bid with the lowest utility that is still at or above that
    threshold (falling back to the maximum-utility bid if none exists).

    ``CreateThreshold()``:

    - If ``time >= 0.95``: threshold = best utility ever received from the
      opponent.
    - Else: ``threshold = max_utility + (0.6 - max_utility) * concession_rate``.

    ``concession_rate``:

    - ``time <= 0.2``: 0.0
    - ``time >= 0.9``: 0.5
    - otherwise: ``my_total_weight * self_factor + 0.11`` where
      ``self_factor = 0.25 * (1/num_committed_offers + NS + time + eagerness)``
      and ``NS`` is the negotiation-status statistic described above.

    **Acceptance Strategy:**
    Accepts immediately if the offer's utility exceeds 0.95. Otherwise
    accepts if the offer's utility is at or above the current threshold
    (``CreateThreshold()``).

    **Opponent Modeling:**
    Maintains the best utility (for TucAgent) ever offered by the opponent
    and a simple "negotiation status" counter that increases faster when
    the opponent's offers are not improving for TucAgent, driving the
    concession rate described above.

    Args:
        reservation_value: Fixed internal reservation value used by the
            threshold formula (default 0.6, matching the Java agent's
            hardcoded value rather than the domain's actual reserved value).
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
        reservation_value: float = 0.6,
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
        self._reservation_value = reservation_value

        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False
        self._max_bid: Outcome | None = None
        self._max_bid_utility: float = 1.0
        self._my_total_weight: float = 0.0

        # State tracked across the negotiation.
        self._last_received_bid: Outcome | None = None
        self._last_received_utility: float | None = None
        self._best_opponent_utility: float = 0.0
        self._round_count: int = 0
        self._delta: int = 0
        self._num_committed_offers: int = 1
        self._eagerness: float = 0.5

    def _initialize(self) -> None:
        """Initialize the outcome space and per-issue total weight estimate."""
        if self._initialized:
            return
        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        if self._outcome_space.outcomes:
            self._max_bid = self._outcome_space.outcomes[0].bid
            self._max_bid_utility = self._outcome_space.outcomes[0].utility

        weights = list(getattr(self.ufun, "weights", []) or [])
        total = 0.0
        for w in weights:
            total = (total + float(w)) / 2.0
        self._my_total_weight = total

        self._initialized = True

    def _process_opponent_bid(self, bid: Outcome) -> None:
        """Update tracked statistics from a newly received opponent bid."""
        if self.ufun is None or bid is None:
            return

        utility = float(self.ufun(bid))
        if utility > self._best_opponent_utility:
            self._best_opponent_utility = utility

        self._round_count += 1

        if self._last_received_utility is not None:
            if utility < self._last_received_utility:
                self._delta += 1
            else:
                self._delta += 3

        self._last_received_bid = bid
        self._last_received_utility = utility

    def _negotiation_status(self) -> float:
        """
        Compute the negotiation-status statistic.

        NS = (delta + 2 * round_count) / (3 * round_count), which stays
        (roughly) bounded in a sensible range as rounds progress.
        """
        if self._round_count == 0:
            return 0.0
        return (self._delta + 2 * self._round_count) / (3 * self._round_count)

    def _concession_rate(self, time: float) -> float:
        """Compute the concession rate used by ``_create_threshold``."""
        if time <= 0.2:
            return 0.0
        if time >= 0.9:
            return 0.5

        ns = self._negotiation_status()
        self_factor = 0.25 * (
            (1.0 / self._num_committed_offers) + ns + time + self._eagerness
        )
        return self._my_total_weight * self_factor + 0.11

    def _create_threshold(self, time: float) -> float:
        """Compute the current acceptance/offering threshold."""
        if time >= 0.95:
            return self._best_opponent_utility

        concession_rate = self._concession_rate(time)
        return self._max_bid_utility + (
            self._reservation_value - self._max_bid_utility
        ) * concession_rate

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        if self._last_received_bid is None:
            return self._max_bid

        self._num_committed_offers += 1
        threshold = self._create_threshold(state.relative_time)

        if self._outcome_space is None or not self._outcome_space.outcomes:
            return self._max_bid

        candidates = self._outcome_space.get_bids_above(threshold)
        if not candidates:
            return self._max_bid
        # Smallest utility bid still satisfying the threshold (ceiling entry).
        return candidates[-1].bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER

        self._process_opponent_bid(offer)

        offer_utility = float(self.ufun(offer))
        if offer_utility > 0.95:
            return ResponseType.ACCEPT_OFFER

        threshold = self._create_threshold(state.relative_time)
        if offer_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
