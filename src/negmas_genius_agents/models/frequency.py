"""Frequency-analysis opponent models from the Genius BOA framework.

Frequency models are the most widely used preference-profile learning technique
among ANAC agents (survey §5.3.4 / §5.3.1). They assume the opponent uses a
linear-additive utility function and estimate, from the opponent's offer
history, (a) per-issue *value* preferences — values offered more often are
assumed more preferred — and (b) *issue weights* — issues whose value the
opponent keeps constant while conceding elsewhere are assumed more important.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from .base import GeniusOpponentModel, bucket_value

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas import Value

__all__ = ["HardHeadedFrequencyModel"]


@define
class HardHeadedFrequencyModel(GeniusOpponentModel):
    """The HardHeaded Frequency Model (the canonical ANAC frequency model).

    Port of ``negotiator.boaframework.opponentmodel.HardHeadedFrequencyModel``.
    Introduced with the BOA framework and used (in variants) by many ANAC
    agents. It models the opponent's utility as a linear-additive function whose
    issue weights and per-value evaluations are learned online:

    - **Issue weights.** All issues start with equal weight ``1/n``. Each time a
      new opponent bid arrives, it is compared with the previous one; every issue
      whose value did *not* change gets ``goldenValue = l / n`` added to its
      weight (capped so no single issue dominates), after which the weights are
      renormalised to sum to 1. The intuition: an opponent concedes on
      unimportant issues first, so issues it keeps fixed are important.
    - **Value evaluations.** Every value present in the latest bid has a constant
      ``learn_value_addition`` added to its score; the estimated utility of an
      outcome is the weighted sum of per-issue value scores, each normalised by
      the maximum score seen for that issue.

    Args:
        learn_coef: The learning coefficient ``l`` (Genius default ``0.2``) — how
            fast issue weights are learned (trade-off between speed and accuracy).
        learn_value_addition: Constant added to a value's score each time it is
            offered (Genius default ``1``).

    Note:
        AI-assisted reimplementation of the Java Genius model; may not behave
        identically in every case.
    """

    learn_coef: float = 0.2
    learn_value_addition: int = 1
    _weights: dict = field(init=False, factory=dict)  # issue name -> weight
    _evals: dict = field(init=False, factory=dict)  # issue name -> {bucketed value: score}
    _golden: float = field(init=False, default=0.0)
    _last: object = field(init=False, default=None)

    def _setup(self) -> None:
        n = len(self._issues)
        if n == 0:
            return
        self._golden = self.learn_coef / n
        self._weights = {}
        self._evals = {}
        self._last = None
        for issue, dissue in zip(self._issues, self._discrete_issues):
            self._weights[issue.name] = 1.0 / n
            self._evals[issue.name] = {v: 1 for v in dissue.all}

    def update(self, offer: Outcome | None) -> None:
        """Update issue weights and value scores from a new opponent bid.

        Mirrors ``HardHeadedFrequencyModel.updateModel``: weights are only
        re-estimated once at least two opponent bids have been seen (so a
        difference can be computed).
        """
        if offer is None or not self._issues:
            return
        if self._last is not None:
            # 1 = value changed between the last two bids, 0 = unchanged.
            diff = {}
            for issue, dissue, prev, cur in zip(
                self._issues, self._discrete_issues, self._last, offer
            ):
                pb = bucket_value(issue, dissue, prev)
                cb = bucket_value(issue, dissue, cur)
                diff[issue.name] = 0 if pb == cb else 1
            n_unchanged = sum(1 for v in diff.values() if v == 0)
            total_sum = 1.0 + self._golden * n_unchanged
            max_weight = 1.0 - len(self._issues) * self._golden / total_sum
            for issue in self._issues:
                w = self._weights[issue.name]
                if diff[issue.name] == 0 and w < max_weight:
                    self._weights[issue.name] = (w + self._golden) / total_sum
                else:
                    self._weights[issue.name] = w / total_sum
            # Add a constant to the score of every value in the latest bid.
            for issue, dissue, value in zip(
                self._issues, self._discrete_issues, offer
            ):
                b = bucket_value(issue, dissue, value)
                ev = self._evals[issue.name]
                ev[b] = ev.get(b, 0) + self.learn_value_addition
        self._last = offer

    def eval(self, offer: Outcome) -> Value:
        """Estimated opponent utility: weighted sum of normalised value scores."""
        if offer is None or not self._issues:
            return 0.0
        u = 0.0
        for issue, dissue, value in zip(self._issues, self._discrete_issues, offer):
            ev = self._evals.get(issue.name, {})
            mx = max(ev.values()) if ev else 0
            b = bucket_value(issue, dissue, value)
            score = (ev.get(b, 0) / mx) if mx > 0 else 0.0
            u += self._weights.get(issue.name, 0.0) * score
        return u

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """The model already returns weighted, per-issue-normalised values in ``[0, 1]``."""
        if offer is None:
            return 0.0
        return float(self.eval(offer))
