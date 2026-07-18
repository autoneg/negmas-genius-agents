"""More frequency-analysis opponent models from the Genius BOA framework.

Like :class:`~negmas_genius_agents.models.frequency.HardHeadedFrequencyModel`,
the models here (survey §5.3.4 / §5.3.1) assume the opponent uses a
linear-additive utility function and estimate per-issue value preferences and
issue weights from the frequency with which the opponent offers particular
values. Each class below ports one Java Genius BOA opponent model:

- :class:`SmithFrequencyModel` ports ``opponentmodel.SmithFrequencyModel``
  (backed by ``opponentmodel.agentsmith.SmithModel``/``IssueModel``).
- :class:`CUHKFrequencyModelV2` ports ``opponentmodel.CUHKFrequencyModelV2``.
- :class:`NashFrequencyModel` ports ``opponentmodel.NashFrequencyModel``
  (backed by the ``opponentmodel.nash`` package).
- :class:`AgentXFrequencyModel` ports ``opponentmodel.AgentXFrequencyModel``
  (backed by the ``opponentmodel.agentx`` package).

Note:
    AI-assisted reimplementations of the Java Genius models; may not behave
    identically in every case. Because
    :class:`~negmas_genius_agents.models.base.GeniusOpponentModel` already
    discretizes every issue (via ``bucket_value``), the numerical-triangle
    interpolation used by the Java ``NashFrequencyModel`` for real/integer
    issues has no analog here: all issues are treated as discrete, exactly as
    :class:`~negmas_genius_agents.models.frequency.HardHeadedFrequencyModel`
    does. ``AgentXFrequencyModel`` also drops the (wall-clock) time-dependent
    "bids per second" / stubbornness correction factor used by the Java
    ``DiscreteIssueProcessor``, since :meth:`update` here (like every other
    model in this package) is not given a negotiation-time argument; a
    constant correction factor of ``1`` is used instead, which preserves the
    qualitative "issues that change more often get less weight" behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from .base import GeniusOpponentModel, bucket_value

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas import Value

__all__ = [
    "SmithFrequencyModel",
    "CUHKFrequencyModelV2",
    "NashFrequencyModel",
    "AgentXFrequencyModel",
]


@define
class SmithFrequencyModel(GeniusOpponentModel):
    """The Smith Frequency Model.

    Port of ``negotiator.boaframework.opponentmodel.SmithFrequencyModel``,
    backed by ``opponentmodel.agentsmith.SmithModel``/``IssueModel``. For
    every issue it keeps the multiset of values the opponent has offered so
    far:

    - **Value evaluation** of a value ``v`` for an issue is the fraction of
      the opponent's past offers on that issue that equalled ``v`` (Java
      ``IssueModel.getDiscreteUtility``) — the opponent's *most recently
      common* choice is not privileged over an old one, unlike HardHeaded's
      additive scores.
    - **Issue weight** is the highest such fraction observed for any single
      value of the issue (Java ``IssueModel.getDiscreteWeight``): an issue on
      which the opponent keeps repeating the *same* value is assumed
      important, regardless of *which* value it is. Weights are renormalised
      across issues to sum to 1 (Java ``SmithModel.getWeights``).

    Real/integer issues in the Java original use a deviation-based weight and
    a distance-based value utility instead (``IssueModel.getRealWeight``` /
    ``getRealUtility``); since this port discretizes every issue up front
    (see module docstring), only the discrete path is implemented.

    Note:
        AI-assisted reimplementation of the Java Genius model; may not behave
        identically in every case.
    """

    _counts: dict = field(init=False, factory=dict)  # issue name -> {value: count}
    _n: int = field(init=False, default=0)  # number of bids seen

    def _setup(self) -> None:
        self._counts = {issue.name: {} for issue in self._issues}
        self._n = 0

    def update(self, offer: Outcome | None) -> None:
        if offer is None or not self._issues:
            return
        for issue, dissue, value in zip(self._issues, self._discrete_issues, offer):
            b = bucket_value(issue, dissue, value)
            counts = self._counts[issue.name]
            counts[b] = counts.get(b, 0) + 1
        self._n += 1

    def eval(self, offer: Outcome) -> Value:
        if offer is None or not self._issues or self._n == 0:
            return 0.0
        max_counts = {}
        for issue in self._issues:
            counts = self._counts.get(issue.name, {})
            max_counts[issue.name] = max(counts.values()) if counts else 0
        total = sum(max_counts.values())
        if total <= 0:
            return 0.0
        u = 0.0
        for issue, dissue, value in zip(self._issues, self._discrete_issues, offer):
            weight = max_counts[issue.name] / total
            counts = self._counts.get(issue.name, {})
            b = bucket_value(issue, dissue, value)
            local_util = counts.get(b, 0) / self._n
            u += weight * local_util
        return u

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        if offer is None:
            return 0.0
        return float(self.eval(offer))


@define
class CUHKFrequencyModelV2(GeniusOpponentModel):
    """The (optimized) CUHKAgent frequency model, ported to BOA.

    Port of ``negotiator.boaframework.opponentmodel.CUHKFrequencyModelV2``
    (the ANAC 2012 CUHKAgent opponent model, BOA-adapted by Mark Hendrikx).
    For each issue it counts how often each value has been offered. The
    estimated utility of a bid is the sum, over issues, of the count of the
    offered value for that issue, divided by the sum over issues of the
    *highest* count observed for any value of that issue (the "maximum
    possible total" the Java class tracks incrementally; here it is derived
    at evaluation time, which is equivalent since every unit increase of a
    per-issue maximum count contributes exactly 1 to that running total).

    This model does not use issue weights in its utility estimate (all issues
    are implicitly weighted by their raw counts, not normalised weights); the
    Java ``getWeight`` — provided for callers that want an explicit weight
    — is uniform (``1/n``) and is not used in :meth:`eval`.

    Note:
        The Java class stops updating after 100 unique bids (a fixed-size
        cache optimisation with no algorithmic effect); this port updates on
        every offer.

        AI-assisted reimplementation of the Java Genius model; may not
        behave identically in every case.
    """

    _counts: dict = field(init=False, factory=dict)  # issue name -> {value: count}

    def _setup(self) -> None:
        self._counts = {}
        for issue, dissue in zip(self._issues, self._discrete_issues):
            self._counts[issue.name] = {v: 0 for v in dissue.all}

    def update(self, offer: Outcome | None) -> None:
        if offer is None or not self._issues:
            return
        for issue, dissue, value in zip(self._issues, self._discrete_issues, offer):
            b = bucket_value(issue, dissue, value)
            counts = self._counts.setdefault(issue.name, {})
            counts[b] = counts.get(b, 0) + 1

    def eval(self, offer: Outcome) -> Value:
        if offer is None or not self._issues:
            return 0.0
        max_total = 0
        for issue in self._issues:
            counts = self._counts.get(issue.name, {})
            if counts:
                max_total += max(counts.values())
        if max_total <= 0:
            return 0.0
        total_bid_value = 0
        for issue, dissue, value in zip(self._issues, self._discrete_issues, offer):
            counts = self._counts.get(issue.name, {})
            b = bucket_value(issue, dissue, value)
            total_bid_value += counts.get(b, 0)
        if total_bid_value == 0:
            return 0.0
        return total_bid_value / max_total

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        if offer is None:
            return 0.0
        return float(self.eval(offer))


@define
class NashFrequencyModel(GeniusOpponentModel):
    """The Nash Frequency Model.

    Port of ``negotiator.boaframework.opponentmodel.NashFrequencyModel``
    (backed by the ``opponentmodel.nash`` package's ``IssueEvaluation*``
    classes). For each issue it keeps a frequency count per value:

    - **Value weight** of a value is its count divided by the highest count
      seen for any value of that issue (``IssueEvaluationDiscrete.
      getNormalizedValueWeight`` — 1 for the most-offered value(s)).
    - **Issue weight** (unnormalised) is the *percentage of the highest
      frequency*: ``highest_count / total_count`` for that issue
      (``IssueEvaluationDiscrete.getPercentageOfHighestFrequency``) — an
      issue on which the opponent keeps repeating one value scores close to
      1, one where offers are spread across many values scores low. Issue
      weights are then renormalised across issues to sum to 1 (falling back
      to a uniform split if every issue's unnormalised weight is 0, i.e.
      before any bid has been seen).
    - The estimated utility is ``sum(normalized_issue_weight * normalized_
      value_weight)``, clamped to ``[0, 1]`` to remove round-off error, as in
      the Java ``getBidEvaluation``.

    The Java class also implements a distinct algorithm for real/integer
    issues (a triangular utility function fit around the earliest-offered
    values, weighted by favouring early bids exponentially, with issue weight
    derived from the offered values' standard deviation relative to the
    issue's range). Since this port discretizes every issue up front (see
    module docstring), only the discrete-issue path is implemented; all
    issues use the discrete frequency algorithm above.

    Note:
        AI-assisted reimplementation of the Java Genius model; may not behave
        identically in every case.
    """

    _counts: dict = field(init=False, factory=dict)  # issue name -> {value: count}
    _ready: bool = field(init=False, default=False)

    def _setup(self) -> None:
        self._counts = {}
        for issue, dissue in zip(self._issues, self._discrete_issues):
            self._counts[issue.name] = {v: 0 for v in dissue.all}
        self._ready = False

    def update(self, offer: Outcome | None) -> None:
        if offer is None or not self._issues:
            return
        for issue, dissue, value in zip(self._issues, self._discrete_issues, offer):
            b = bucket_value(issue, dissue, value)
            counts = self._counts.setdefault(issue.name, {})
            counts[b] = counts.get(b, 0) + 1
        self._ready = True

    def _issue_weights(self) -> dict:
        """Normalised issue weights (percentage-of-highest-frequency, renormalised)."""
        unnorm = {}
        for issue in self._issues:
            counts = self._counts.get(issue.name, {})
            total = sum(counts.values())
            highest = max(counts.values()) if counts else 0
            unnorm[issue.name] = (highest / total) if total > 0 else 0.0
        total_weight = sum(unnorm.values())
        if total_weight <= 0:
            n = len(self._issues) or 1
            return {issue.name: 1.0 / n for issue in self._issues}
        return {name: w / total_weight for name, w in unnorm.items()}

    def eval(self, offer: Outcome) -> Value:
        if offer is None or not self._issues or not self._ready:
            return 0.0
        weights = self._issue_weights()
        u = 0.0
        for issue, dissue, value in zip(self._issues, self._discrete_issues, offer):
            counts = self._counts.get(issue.name, {})
            highest = max(counts.values()) if counts else 0
            b = bucket_value(issue, dissue, value)
            value_weight = (counts.get(b, 0) / highest) if highest > 0 else 0.0
            u += weights.get(issue.name, 0.0) * value_weight
        return min(1.0, max(0.0, u))

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        if offer is None:
            return 0.0
        return float(self.eval(offer))


@define
class AgentXFrequencyModel(GeniusOpponentModel):
    """The AgentX Frequency Model.

    Port of ``negotiator.boaframework.opponentmodel.AgentXFrequencyModel``
    (backed by the ``opponentmodel.agentx`` package's ``DiscreteIssueProcessor``
    / ``DiscreteValueProcessor``). Unlike the other frequency models, issue
    weights here are driven by how often an issue's value *changes* between
    consecutive opponent bids rather than by how concentrated its value
    counts are:

    - **Issue weight.** Every issue starts with a ``changeList`` entry of 1.
      Whenever the opponent's value for an issue differs between two
      consecutive bids, ``changeList[i] += corr / changeList[i]`` (Java
      ``DiscreteIssueProcessor.adaptWeightsByBid``); the issue's weight is
      then ``(1 / changeList[i])`` renormalised across issues so weights sum
      to 1 — issues that change often accumulate a large ``changeList``
      value and so end up with low weight. The Java ``corr`` factor depends
      on real negotiation time (bids-per-second and a "stubbornness"
      estimate derived from repeated bids); since :meth:`update` here is not
      given a time argument, a constant ``corr = 1`` is used instead, which
      preserves the qualitative behaviour (more changes -> lower weight) but
      not the exact magnitudes.
    - **Value rank.** Each issue keeps values ranked 1..n_values (1 = least
      preferred). Every time a value is offered, its "bid count" is
      incremented and it is bubbled up past any higher-ranked value with a
      lower bid count (Java ``DiscreteValueProcessor.changeRankByBid`` /
      ``increaseRank``), so values offered more often drift towards rank
      ``n_values``. The normalized value weight is ``rank / n_values``.
      Values never offered keep their initial (arbitrary, domain-order)
      rank, so their scores are not meaningful until they are observed —
      faithful to the Java original.
    - The estimated utility is the weighted sum, over issues, of
      ``issue_weight * normalized_value_rank``.

    Note:
        AI-assisted reimplementation of the Java Genius model; may not behave
        identically in every case.
    """

    corr: float = 1.0
    _change: dict = field(init=False, factory=dict)  # issue name -> changeList value
    _ranks: dict = field(init=False, factory=dict)  # issue name -> {value: rank}
    _rank_to_value: dict = field(init=False, factory=dict)  # issue name -> {rank: value}
    _bid_counts: dict = field(init=False, factory=dict)  # issue name -> {value: count}
    _n_values: dict = field(init=False, factory=dict)  # issue name -> n
    _last: object = field(init=False, default=None)

    def _setup(self) -> None:
        self._change = {issue.name: 1.0 for issue in self._issues}
        self._ranks = {}
        self._rank_to_value = {}
        self._bid_counts = {}
        self._n_values = {}
        self._last = None
        for issue, dissue in zip(self._issues, self._discrete_issues):
            values = list(dissue.all)
            n = len(values)
            self._n_values[issue.name] = n
            self._ranks[issue.name] = {v: i + 1 for i, v in enumerate(values)}
            self._rank_to_value[issue.name] = {i + 1: v for i, v in enumerate(values)}
            self._bid_counts[issue.name] = {v: 0 for v in values}

    def _increase_rank(self, issue_name: str, value) -> None:
        ranks = self._ranks[issue_name]
        rank_to_value = self._rank_to_value[issue_name]
        n = self._n_values[issue_name]
        old_rank = ranks[value]
        if old_rank == n:
            return
        new_rank = old_rank + 1
        other_value = rank_to_value[new_rank]
        ranks[value] = new_rank
        ranks[other_value] = old_rank
        rank_to_value[new_rank] = value
        rank_to_value[old_rank] = other_value

    def _add_bid_for_value(self, issue_name: str, value) -> None:
        counts = self._bid_counts[issue_name]
        counts[value] = counts.get(value, 0) + 1
        # Bubble the value's rank up past any lower-count, higher-ranked value.
        n = self._n_values[issue_name]
        ranks = self._ranks[issue_name]
        rank_to_value = self._rank_to_value[issue_name]
        new_rank = ranks[value] + 1
        bids = counts[value]
        rank_change = 0
        while new_rank <= n:
            other_value = rank_to_value[new_rank]
            if bids > counts.get(other_value, 0):
                new_rank += 1
                rank_change += 1
            else:
                break
        for _ in range(rank_change):
            self._increase_rank(issue_name, value)

    def update(self, offer: Outcome | None) -> None:
        if offer is None or not self._issues:
            return
        if self._last is not None:
            for issue, dissue, prev, cur in zip(
                self._issues, self._discrete_issues, self._last, offer
            ):
                pb = bucket_value(issue, dissue, prev)
                cb = bucket_value(issue, dissue, cur)
                if pb != cb:
                    change = self._change[issue.name]
                    self._change[issue.name] = change + self.corr / change
        for issue, dissue, value in zip(self._issues, self._discrete_issues, offer):
            b = bucket_value(issue, dissue, value)
            self._add_bid_for_value(issue.name, b)
        self._last = offer

    def _issue_weights(self) -> dict:
        inverse_total = sum(1.0 / self._change[issue.name] for issue in self._issues)
        if inverse_total <= 0:
            n = len(self._issues) or 1
            return {issue.name: 1.0 / n for issue in self._issues}
        return {
            issue.name: (1.0 / self._change[issue.name]) / inverse_total
            for issue in self._issues
        }

    def eval(self, offer: Outcome) -> Value:
        if offer is None or not self._issues:
            return 0.0
        weights = self._issue_weights()
        u = 0.0
        for issue, dissue, value in zip(self._issues, self._discrete_issues, offer):
            b = bucket_value(issue, dissue, value)
            n = self._n_values.get(issue.name, 0)
            if n <= 0:
                continue
            rank = self._ranks.get(issue.name, {}).get(b)
            if rank is None:
                continue
            u += weights.get(issue.name, 0.0) * (rank / n)
        return u

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        if offer is None:
            return 0.0
        return float(self.eval(offer))
