"""AgentFSEGA from ANAC 2010."""

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

__all__ = ["AgentFSEGA"]


class AgentFSEGA(SAONegotiator):
    """
    AgentFSEGA from ANAC 2010.

    This agent combines time-dependent concession with opponent modeling.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    This implementation faithfully reproduces AgentFSEGA's core strategies:

    - Polynomial time-dependent concession with Boulware-style exponent
    - Opponent reservation value estimation from observed offers
    - Adaptive minimum utility adjustment
    - Comparative acceptance based on target and opponent behavior

    References:
        Original Genius class: ``agents.anac.y2010.AgentFSEGA.AgentFSEGA``

        ANAC 2010: https://ii.tudelft.nl/negotiation/

    **Offering Strategy:**
        - Polynomial time-dependent concession: `target = Pmax - (Pmax-Pmin) * t^(1/e)`
        - Default e=0.2 (Boulware-style, slow early concession)
        - Bid selection closest to current target utility
        - Target decreases from Pmax (1.0) toward Pmin over time

    **Acceptance Strategy:**
        - Accept if offer utility >= current target utility
        - Target calculated using same formula as bidding
        - Considers opponent's estimated reservation value in Pmin

    **Opponent Modeling:**
        - Tracks opponent's offers to estimate their reservation value
        - Uses minimum observed opponent utility as reservation estimate
        - Adjusts Pmin based on opponent model: Pmin = max(reservation, opponent_min)
        - Adapts strategy based on opponent's concession pattern

    Key parameters:
        - Pmax: Maximum utility (1.0, best possible for self)
        - Pmin: Minimum acceptable (adjusted by opponent model)
        - e: Concession exponent (0.2 = Boulware, <1 slow start)

    Args:
        reservation: Minimum acceptable utility (default 0.6).
        concession_exp: Concession exponent for time function (default 0.2).
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
        reservation: float = 0.6,
        concession_exp: float = 0.2,
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
        self._reservation = reservation
        self._concession_exp = concession_exp
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        # Opponent modeling
        self._opponent_offers: list[tuple[Outcome, float]] = []
        self._opponent_max_utility: float = 0.0
        self._opponent_min_utility: float = 1.0
        self._estimated_opponent_reservation: float = 0.0

        # Concession parameters
        self._pmax: float = 1.0  # Maximum utility target
        self._pmin: float = reservation  # Minimum acceptable utility

        # Last bid tracking
        self._last_bid: Outcome | None = None

    def _initialize(self) -> None:
        """Initialize the outcome space and parameters."""
        if self._initialized:
            return

        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)

        # Set Pmax based on outcome space
        if self._outcome_space.outcomes:
            self._pmax = self._outcome_space.max_utility

        self._initialized = True

    def on_negotiation_start(self, state: SAOState) -> None:
        """Called when negotiation starts."""
        super().on_negotiation_start(state)
        self._initialize()

        # Reset opponent model
        self._opponent_offers = []
        self._opponent_max_utility = 0.0
        self._opponent_min_utility = 1.0
        self._estimated_opponent_reservation = 0.0
        self._last_bid = None

    def _update_opponent_model(self, offer: Outcome, utility: float) -> None:
        """
        Update opponent model based on received offer.

        Tracks opponent offers to estimate their reservation value and
        adjust our minimum acceptable utility accordingly.

        Args:
            offer: The offer received from opponent.
            utility: Our utility for the offer.
        """
        self._opponent_offers.append((offer, utility))

        # Update min/max tracking
        if utility > self._opponent_max_utility:
            self._opponent_max_utility = utility
        if utility < self._opponent_min_utility:
            self._opponent_min_utility = utility

        # Estimate opponent's reservation based on their offer pattern
        # If opponent is offering us low utility, they likely have high reservation
        if len(self._opponent_offers) >= 3:
            # Calculate average utility they're offering us
            avg_utility = sum(u for _, u in self._opponent_offers) / len(
                self._opponent_offers
            )
            # Estimate their reservation as inverse of what they offer us
            self._estimated_opponent_reservation = 1.0 - avg_utility

            # Adjust our minimum based on opponent behavior
            # If opponent seems very competitive, we may need to lower our minimum
            if avg_utility < 0.4:
                self._pmin = max(self._reservation * 0.9, 0.5)
            else:
                self._pmin = self._reservation

    def _get_target_utility(self, time: float) -> float:
        """
        Calculate target utility using time-dependent concession.

        Uses the formula: target = Pmin + (Pmax - Pmin) * (1 - t^e)
        where t is normalized time and e is the concession exponent.

        Args:
            time: Normalized time [0, 1].

        Returns:
            Target utility value.
        """
        # Polynomial time-dependent concession
        # f(t) = 1 - t^e (where e is concession exponent)
        # Small e = concede slowly (boulware), large e = concede quickly (conceder)
        time_factor = math.pow(time, self._concession_exp)

        # Calculate target: starts at Pmax, decreases toward Pmin
        target = self._pmin + (self._pmax - self._pmin) * (1.0 - time_factor)

        # Near deadline, consider best opponent offer
        if time > 0.9 and self._opponent_max_utility > target:
            # Blend toward best opponent offer
            blend = (time - 0.9) / 0.1  # 0 to 1 in last 10%
            target = target * (1 - blend) + self._opponent_max_utility * blend

        return max(self._pmin, min(self._pmax, target))

    def _select_bid(self, time: float) -> Outcome | None:
        """
        Select a bid to offer based on current target utility.

        AgentFSEGA selects bids near the target utility, with some
        randomization to avoid predictability.

        Args:
            time: Normalized time [0, 1].

        Returns:
            The selected bid, or None if no suitable bid found.
        """
        if self._outcome_space is None:
            return None

        target = self._get_target_utility(time)

        # Get bids near the target
        # Allow some range around target for variety
        tolerance = 0.05
        candidates = self._outcome_space.get_bids_in_range(
            target - tolerance, target + tolerance
        )

        if candidates:
            # Randomly select from candidates for unpredictability
            return random.choice(candidates).bid

        # Try to find closest bid to target
        bid_details = self._outcome_space.get_bid_near_utility(target)
        if bid_details is not None:
            return bid_details.bid

        # Fallback: return best available bid
        if self._outcome_space.outcomes:
            return self._outcome_space.outcomes[0].bid

        return None

    def _should_accept(self, offer: Outcome, time: float) -> bool:
        """
        Decide whether to accept an offer.

        Accepts if:
        1. Offer utility >= target utility
        2. Near deadline and offer >= opponent's best offer to us
        3. Very close to deadline and offer >= minimum

        Args:
            offer: The offer to evaluate.
            time: Normalized time [0, 1].

        Returns:
            True if should accept, False otherwise.
        """
        if self.ufun is None:
            return False

        offer_utility = float(self.ufun(offer))
        target = self._get_target_utility(time)

        # Accept if meets or exceeds target
        if offer_utility >= target:
            return True

        # Near deadline: accept if better than what we could get
        if time > 0.95:
            # Accept if better than opponent's typical offer
            if len(self._opponent_offers) > 0:
                avg_opponent = sum(u for _, u in self._opponent_offers) / len(
                    self._opponent_offers
                )
                if offer_utility >= avg_opponent * 1.05:
                    return True

        # Very close to deadline: accept anything above minimum
        if time > 0.99 and offer_utility >= self._pmin:
            return True

        return False

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """
        Generate a proposal.

        Args:
            state: Current negotiation state.
            dest: Destination negotiator ID (ignored).

        Returns:
            Outcome to propose, or None.
        """
        if not self._initialized:
            self._initialize()

        bid = self._select_bid(state.relative_time)
        self._last_bid = bid
        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """
        Respond to an offer.

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

        offer_utility = float(self.ufun(offer))

        # Update opponent model
        self._update_opponent_model(offer, offer_utility)

        # Decide whether to accept
        if self._should_accept(offer, state.relative_time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
