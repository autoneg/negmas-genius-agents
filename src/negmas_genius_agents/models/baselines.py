"""Baseline and oracle opponent models from the Genius BOA framework.

These are the trivial reference models used mainly as benchmarks (survey §6, on
measuring opponent-model quality) and as fall-backs:

- :class:`PerfectModel` / :class:`WorstModel` are *oracles*: they read the
  opponent's true utility function (only available in analysis/testing).
- :class:`OppositeModel` assumes a zero-sum negotiation using the agent's own
  utility function (no opponent information needed).
- :class:`UniformModel` / :class:`DefaultModel` are constant no-learning models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define

from .base import GeniusOpponentModel

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas.preferences import BaseUtilityFunction
    from negmas import Value

__all__ = [
    "PerfectModel",
    "WorstModel",
    "OppositeModel",
    "UniformModel",
    "DefaultModel",
]


@define
class UniformModel(GeniusOpponentModel):
    """Constant model that assigns every outcome the same utility ``0.5``.

    Port of ``negotiator.boaframework.opponentmodel.UniformModel`` — "a simple
    baseline opponent model which always returns the same preference". Learns
    nothing.
    """

    def eval(self, offer: Outcome) -> Value:
        return 0.5

    def eval_normalized(
        self, offer: Outcome | None, above_reserve: bool = True, expected_limits: bool = True
    ) -> Value:
        return 0.0 if offer is None else 0.5


@define
class DefaultModel(GeniusOpponentModel):
    """No-op model that assigns every outcome utility ``0.0``.

    Port of ``negotiator.boaframework.opponentmodel.DefaultModel`` — the neutral
    default used when no real opponent model is desired. Learns nothing.
    """

    def eval(self, offer: Outcome) -> Value:
        return 0.0

    def eval_normalized(
        self, offer: Outcome | None, above_reserve: bool = True, expected_limits: bool = True
    ) -> Value:
        return 0.0


@define
class OppositeModel(GeniusOpponentModel):
    """Zero-sum model: opponent utility is ``1 - own_utility``.

    Port of ``negotiator.boaframework.opponentmodel.OppositeModel``, which
    returns ``1 - u_self(bid)`` using the agent's *own* utility space. It needs
    no opponent information and is exact when the negotiation really is zero-sum.
    (Equivalent in spirit to :class:`negmas.gb.components.models.ufun.ZeroSumModel`.)

    Args:
        own_ufun: The agent's own utility function. If not given, it is taken
            from the negotiator when the model is attached as a component.
    """

    own_ufun: "BaseUtilityFunction | None" = None

    def on_preferences_changed(self, changes):
        negotiator = getattr(self, "negotiator", None)
        if self.own_ufun is None and negotiator is not None:
            self.own_ufun = getattr(negotiator, "ufun", None)
        self._update_private_info()

    def eval(self, offer: Outcome) -> Value:
        if offer is None or self.own_ufun is None:
            return 0.5
        return 1.0 - float(self.own_ufun.eval_normalized(offer))

    def eval_normalized(
        self, offer: Outcome | None, above_reserve: bool = True, expected_limits: bool = True
    ) -> Value:
        if offer is None or self.own_ufun is None:
            return 0.0
        return 1.0 - float(self.own_ufun.eval_normalized(offer, above_reserve, expected_limits))


@define
class PerfectModel(GeniusOpponentModel):
    """Oracle model that returns the opponent's *true* utility.

    Port of ``negotiator.boaframework.opponentmodel.PerfectModel``. In Genius it
    reads the opponent's actual profile from the session; here the opponent's
    true utility function must be supplied (only possible in analysis/testing).

    Args:
        opponent_ufun: The opponent's true utility function.
    """

    opponent_ufun: "BaseUtilityFunction | None" = None

    def on_preferences_changed(self, changes):
        self._update_private_info()

    def eval(self, offer: Outcome) -> Value:
        if offer is None or self.opponent_ufun is None:
            return 0.5
        return float(self.opponent_ufun.eval_normalized(offer))

    def eval_normalized(
        self, offer: Outcome | None, above_reserve: bool = True, expected_limits: bool = True
    ) -> Value:
        if offer is None or self.opponent_ufun is None:
            return 0.0
        return float(self.opponent_ufun.eval_normalized(offer, above_reserve, expected_limits))


@define
class WorstModel(PerfectModel):
    """Oracle model that returns ``1 - u_opponent(bid)`` (the opponent's worst view).

    Port of ``negotiator.boaframework.opponentmodel.WorstModel``. Like
    :class:`PerfectModel` it reads the opponent's true utility function, but
    inverts it — a deliberately pessimistic reference model.
    """

    def eval(self, offer: Outcome) -> Value:
        if offer is None or self.opponent_ufun is None:
            return 0.5
        return 1.0 - float(self.opponent_ufun.eval_normalized(offer))

    def eval_normalized(
        self, offer: Outcome | None, above_reserve: bool = True, expected_limits: bool = True
    ) -> Value:
        if offer is None or self.opponent_ufun is None:
            return 0.0
        return 1.0 - float(self.opponent_ufun.eval_normalized(offer, above_reserve, expected_limits))
