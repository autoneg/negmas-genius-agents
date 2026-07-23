"""SimpaticoAgent from ANAC 2014."""

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

__all__ = ["SimpaticoAgent"]


class SimpaticoAgent(SAONegotiator):
    """
    SimpaticoAgent from ANAC 2014 (original Java name: Simpatico).

    Simpatico bids randomly above a minimum utility threshold at first, then
    performs a local (hill-climbing) neighbourhood search around the
    opponent's last bid to find bids that are good for us but close to what
    the opponent proposed. It tracks whether the opponent seems cooperative
    (offers utility above a threshold often enough) and relaxes its minimum
    acceptable utility over time accordingly, and keeps a ranked history of
    the opponent's best offers to fall back on near the deadline.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    References:
        - ANAC 2014 Competition Results and Agent Descriptions
        - Original Genius implementation: agents.anac.y2014.SimpaticoAgent.Simpatico

    **Offering Strategy:**
        - First offer: a random bid above ``initial_min_bid_utility``.
        - Subsequent offers: perform a local neighbourhood search around the
          opponent's last bid (randomly perturbing a fraction of the issues,
          recursively, up to a search depth) looking for a bid at least as
          good as ``min_bid_utility`` for us. If none is found, fall back to
          another random bid above the (possibly relaxed) minimum utility,
          occasionally itself locally searched.
        - Near the deadline (estimated round-trip time based cutoff), offer
          back one of the best bids ever received from the opponent in turn
          (round-robin over the top 10 saved opponent bids).

    **Acceptance Strategy:**
        - Straight accept if close to the deadline and the offer utility is
          reasonably high (>= 0.75).
        - Accept if offer utility >= a discounted acceptance threshold that
          starts at 0.9 and decays with ``discount_factor ** time`` (loosens
          further to 0.85 after t >= 0.9).
        - Accept if offer utility >= the utility of the bid we were about to
          propose ourselves.

    **Opponent Modeling:**
        Simple cooperation classification: an opponent is "cooperative" if
        the fraction of its offers exceeding ``min_cooperative_utility``
        exceeds ``cooperation_threshold``. When cooperative and past the
        halfway point in time, the agent relaxes its minimum acceptable
        utility (allowing faster agreement); otherwise it holds firm. The
        best 10 opponent bids (by our utility) are cached for late-game
        reciprocation.

    Args:
        initial_min_bid_utility: Starting minimum utility for our own random
            bids (default 0.9).
        acceptance_bid_utility: Initial acceptance utility baseline before
            discounting (default 0.85).
        search_depth: Depth of the recursive local neighbourhood search
            (default 3).
        search_depth_for_own_bids: Search depth used when locally improving
            our own random bids (default 2).
        fraction_issues_to_change: Fraction of issues perturbed per search
            step (default 0.5).
        random_search_ratio: Probability of locally searching around a
            freshly generated random bid (default 0.3).
        cooperation_threshold: Fraction of "good" opponent offers needed to
            call the opponent cooperative (default 0.5).
        min_cooperative_utility: Utility threshold defining a "good" offer
            for cooperation tracking (default 0.5).
        best_offers_cache_size: Number of top opponent offers cached for
            late-game reciprocation (default 10).
        max_random_tries: Maximum attempts to find a random bid meeting the
            utility floor before giving up (default 2000).
        deadline_reciprocation_time: Relative-time threshold near the
            deadline at which the agent starts offering back the opponent's
            best remembered bids (default 0.95).
        emergency_accept_time: Relative time threshold for the straight
            "good enough and time is short" acceptance rule (default 0.99).
        emergency_accept_utility: Utility threshold paired with
            ``emergency_accept_time`` (default 0.75).
        tolerance_increment_interval: Number of random-bid tries before
            relaxing the utility floor by one step (default 200).
        tolerance_increment_step: Amount added to the random-bid tolerance
            each time the floor is relaxed (default 0.01).
        perturb_step_fraction: Fraction of an issue's span used as the step
            when perturbing continuous issue values (default 0.05).
        max_issues_to_change: Cap on the number of issues perturbed per
            neighbourhood-search step (default 5).
        relaxation_time: Relative time after which the minimum bid utility is
            relaxed for cooperative opponents (default 0.5).
        relaxed_min_bid_factor: Factor applied to the discounted utility when
            relaxing the minimum bid utility (default 0.88).
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
        initial_min_bid_utility: float = 0.9,
        acceptance_bid_utility: float = 0.85,
        search_depth: int = 3,
        search_depth_for_own_bids: int = 2,
        fraction_issues_to_change: float = 0.5,
        random_search_ratio: float = 0.3,
        cooperation_threshold: float = 0.5,
        min_cooperative_utility: float = 0.5,
        best_offers_cache_size: int = 10,
        max_random_tries: int = 2000,
        deadline_reciprocation_time: float = 0.95,
        emergency_accept_time: float = 0.99,
        emergency_accept_utility: float = 0.75,
        tolerance_increment_interval: int = 200,
        tolerance_increment_step: float = 0.01,
        perturb_step_fraction: float = 0.05,
        max_issues_to_change: int = 5,
        relaxation_time: float = 0.5,
        relaxed_min_bid_factor: float = 0.88,
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
        self._initial_min_bid_utility = initial_min_bid_utility
        self._min_bid_utility = initial_min_bid_utility
        self._acceptance_bid_utility = acceptance_bid_utility
        self._search_depth = search_depth
        self._search_depth_for_own_bids = search_depth_for_own_bids
        self._fraction_issues_to_change = fraction_issues_to_change
        self._random_search_ratio = random_search_ratio
        self._cooperation_threshold = cooperation_threshold
        self._min_cooperative_utility = min_cooperative_utility
        self._best_offers_cache_size = best_offers_cache_size
        self._max_random_tries = max_random_tries
        self._deadline_reciprocation_time = deadline_reciprocation_time
        self._emergency_accept_time = emergency_accept_time
        self._emergency_accept_utility = emergency_accept_utility
        self._tolerance_increment_interval = tolerance_increment_interval
        self._tolerance_increment_step = tolerance_increment_step
        self._perturb_step_fraction = perturb_step_fraction
        self._max_issues_to_change = max_issues_to_change
        self._relaxation_time = relaxation_time
        self._relaxed_min_bid_factor = relaxed_min_bid_factor

        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        self._best_opponent_bid: BidDetails | None = None
        self._best_opponent_offers: list[BidDetails] = []
        self._bid_to_submit_back = 0

        self._opponent_offer_count = 0
        self._n_cooperative_utilities = 0
        self._opponent_is_cooperative = True

        self._n_issues = 0
        self._issue_names: list[str] = []

    def _initialize(self) -> None:
        """Initialize by inspecting the outcome space."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        outcome_space = self.ufun.outcome_space
        if outcome_space is not None:
            self._issue_names = [i.name for i in outcome_space.issues]
            self._n_issues = len(self._issue_names)
        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        self._min_bid_utility = self._initial_min_bid_utility
        self._best_opponent_bid = None
        self._best_opponent_offers = []
        self._bid_to_submit_back = 0
        self._opponent_offer_count = 0
        self._n_cooperative_utilities = 0
        self._opponent_is_cooperative = True

    def _random_bid(self) -> Outcome | None:
        """Draw a random outcome, retrying to approximately meet our floor."""
        outcome_space = self.ufun.outcome_space if self.ufun else None
        if outcome_space is None:
            return None

        tolerance = 0.0
        tries = 0
        bid = None
        for _ in range(self._max_random_tries):
            bid = outcome_space.random_outcome()
            tries += 1
            if bid is None:
                continue
            if float(self.ufun(bid)) >= self._min_bid_utility - tolerance:
                break
            if tries > self._tolerance_increment_interval:
                tolerance += self._tolerance_increment_step
                tries = 0

        if bid is None and self._outcome_space and self._outcome_space.outcomes:
            bid = self._outcome_space.outcomes[0].bid

        if bid is not None and random.random() < self._random_search_ratio:
            searched = self._search_neighbourhood(
                BidDetails(bid=bid, utility=float(self.ufun(bid))),
                self._search_depth_for_own_bids,
            )
            bid = searched.bid

        return bid

    def _perturb_value(self, outcome: list, issue_index: int, delta_steps: int):
        """Return a new outcome list with one issue value shifted."""
        if self.ufun is None or self.ufun.outcome_space is None:
            return None

        issue = self.ufun.outcome_space.issues[issue_index]
        new_outcome = list(outcome)

        if issue.is_continuous():
            lo, hi = issue.min_value, issue.max_value
            span = hi - lo
            step = span * self._perturb_step_fraction * delta_steps
            new_value = new_outcome[issue_index] + step
            if new_value < lo or new_value > hi:
                return None
            new_outcome[issue_index] = new_value
            return tuple(new_outcome)

        # discrete/integer issue: shift by position in the value list
        values = list(issue.all)
        try:
            current_index = values.index(new_outcome[issue_index])
        except ValueError:
            return None
        new_index = current_index + delta_steps
        if new_index < 0 or new_index >= len(values):
            return None
        new_outcome[issue_index] = values[new_index]
        return tuple(new_outcome)

    def _search_neighbourhood(
        self, initial: BidDetails, depth: int
    ) -> BidDetails:
        """Recursive local search around `initial`, matching the Java hill-climb."""
        if depth <= 0 or self.ufun is None or self._n_issues == 0:
            return initial

        best = initial
        n_change = max(
            1,
            min(
                self._max_issues_to_change,
                int(self._n_issues * self._fraction_issues_to_change),
            ),
        )
        issue_indices = list(range(self._n_issues))
        random.shuffle(issue_indices)
        chosen = issue_indices[:n_change]

        for idx in chosen:
            for delta in (1, -1):
                candidate_outcome = self._perturb_value(
                    list(initial.bid), idx, delta
                )
                if candidate_outcome is None:
                    continue
                utility = float(self.ufun(candidate_outcome))
                candidate = BidDetails(bid=candidate_outcome, utility=utility)
                if candidate.utility > best.utility:
                    best = candidate
                deeper = self._search_neighbourhood(candidate, depth - 1)
                if deeper.utility > best.utility:
                    best = deeper

        return best

    def _save_best_opponent_bid(self, bid: BidDetails) -> None:
        """Keep a ranked cache of the opponent's best offers."""
        insert_pos = 0
        for existing in self._best_opponent_offers:
            if existing.utility > bid.utility:
                insert_pos += 1
        self._best_opponent_offers.insert(insert_pos, bid)
        if len(self._best_opponent_offers) > self._best_offers_cache_size:
            self._best_opponent_offers = self._best_opponent_offers[
                : self._best_offers_cache_size
            ]

    def _update_opponent_cooperation(self, offer_utility: float) -> None:
        self._opponent_offer_count += 1
        if offer_utility > self._min_cooperative_utility:
            self._n_cooperative_utilities += 1
        if self._opponent_offer_count > 0:
            ratio = self._n_cooperative_utilities / self._opponent_offer_count
            self._opponent_is_cooperative = ratio > self._cooperation_threshold

    def _update_min_bid_utility(self, time: float) -> None:
        discount = float(getattr(self.ufun, "discount_factor", None) or 1.0)
        if time >= self._relaxation_time and self._opponent_is_cooperative:
            self._min_bid_utility = self._relaxed_min_bid_factor * (discount**time)

    def _acceptance_threshold(self, time: float) -> float:
        discount = float(getattr(self.ufun, "discount_factor", None) or 1.0)
        return self._initial_min_bid_utility * (discount**time)

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Generate a proposal."""
        if not self._initialized:
            self._initialize()

        if self.ufun is None:
            return None

        offer = state.current_offer
        time = state.relative_time

        if offer is None:
            return self._random_bid()

        opponent_bid = BidDetails(bid=offer, utility=float(self.ufun(offer)))

        # Late-game: reciprocate with the best remembered opponent bid.
        if time >= self._deadline_reciprocation_time and self._best_opponent_offers:
            if self._bid_to_submit_back >= len(self._best_opponent_offers):
                self._bid_to_submit_back = 0
            chosen = self._best_opponent_offers[self._bid_to_submit_back]
            self._bid_to_submit_back += 1
            return chosen.bid

        best_in_vicinity = self._search_neighbourhood(opponent_bid, self._search_depth)
        if best_in_vicinity.utility < self._min_bid_utility:
            candidate = self._random_bid()
            if candidate is None and self._best_opponent_bid is not None:
                return self._best_opponent_bid.bid
            return candidate

        return best_in_vicinity.bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond following Simpatico's layered acceptance rules."""
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER

        time = state.relative_time
        offer_utility = float(self.ufun(offer))

        if time >= self._emergency_accept_time and offer_utility >= self._emergency_accept_utility:
            return ResponseType.ACCEPT_OFFER

        opponent_bid = BidDetails(bid=offer, utility=offer_utility)
        self._save_best_opponent_bid(opponent_bid)
        if (
            self._best_opponent_bid is None
            or offer_utility > self._best_opponent_bid.utility
        ):
            self._best_opponent_bid = opponent_bid

        self._update_opponent_cooperation(offer_utility)
        self._update_min_bid_utility(time)

        acceptance_threshold = self._acceptance_threshold(time)
        if offer_utility >= acceptance_threshold:
            return ResponseType.ACCEPT_OFFER

        # Look ahead at what we would offer without triggering the
        # late-game reciprocation counter's side effect (that path is only
        # exercised from `propose` itself when we actually send an offer).
        best_in_vicinity = self._search_neighbourhood(opponent_bid, self._search_depth)
        next_utility = best_in_vicinity.utility
        if offer_utility >= next_utility:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
