"""Base class and helpers for Genius BOA opponent models.

The Genius BOA framework (Baarslag et al., "Decoupling negotiating agents to
explore the space of negotiation strategies") splits a negotiator into a
**B**idding strategy, an **O**pponent model, and an **A**cceptance strategy.
This package ports the Genius *opponent models* (``negotiator.boaframework.
opponentmodel.*`` in the Java source) to NegMAS.

Every model here is a :class:`negmas.gb.components.models.ufun.UFunModel`
subclass — i.e. a full stand-in for a :class:`~negmas.preferences.BaseUtilityFunction`
that estimates the *opponent's* utility of an outcome. In the taxonomy of
Baarslag, Hendrikx, Hindriks & Jonker, *Learning about the opponent in automated
bilateral negotiation: a comprehensive survey of opponent modeling techniques*
(JAAMAS 30:849–898, 2016), all Genius BOA opponent models learn the opponent's
**preference profile** (survey §5.3).

Each model supports two usage modes:

- **Standalone** (for use inside a ported agent, or for testing): construct it,
  call :meth:`GeniusOpponentModel.initialize` with the negotiation issues (or an
  outcome space), feed opponent offers with :meth:`GeniusOpponentModel.update`,
  and query the estimated opponent utility with ``model(offer)`` / ``model.eval(offer)``.
- **Component** (when attached to a NegMAS component-based negotiator): the model
  sets itself up in ``on_preferences_changed`` and learns from ``on_partner_proposal``
  / ``before_responding`` automatically.

Note:
    These are AI-assisted Python reimplementations of the original Java Genius
    models and may not behave identically in every case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from attrs import define, field

from negmas.preferences.base_ufun import BaseUtilityFunction

from negmas.gb.components.models.ufun import UFunModel

if TYPE_CHECKING:
    from negmas.outcomes import Issue, Outcome
    from negmas import PreferencesChange, Value

__all__ = ["GeniusOpponentModel", "discretize_issues", "bucket_value"]


def discretize_issues(issues: Iterable[Issue], levels: int = 10) -> list:
    """Return a discrete counterpart of each issue via ``issue.to_discrete``.

    Already-discrete issues are returned unchanged; continuous issues are
    discretized to at most ``levels`` grid values so that frequency counts can
    be kept over a finite value set. Genius domains are discrete, so this is a
    no-op there, but it lets the models degrade gracefully on continuous issues.

    Args:
        issues: The negotiation issues.
        levels: Maximum number of grid values for continuous issues.

    Returns:
        A list of (possibly discretized) issues, parallel to ``issues``.
    """
    out = []
    for issue in issues:
        try:
            out.append(issue.to_discrete(n=levels))
        except Exception:  # pragma: no cover - defensive
            out.append(issue)
    return out


def bucket_value(original_issue: Issue, discrete_issue: Issue, value):
    """Map an offered ``value`` to the discretized value it falls into.

    For already-discrete issues the value is returned unchanged. For continuous
    issues it is snapped to the nearest grid value of ``discrete_issue``.

    Args:
        original_issue: The original (possibly continuous) issue.
        discrete_issue: Its discretized counterpart (from :func:`discretize_issues`).
        value: The offered value.

    Returns:
        The bucketed (hashable, discrete) value.
    """
    if getattr(original_issue, "is_discrete", lambda: False)():
        return value
    best = None
    best_d = None
    for d in discrete_issue.all:
        try:
            dd = abs(float(d) - float(value))
        except (TypeError, ValueError):
            if d == value:
                return d
            continue
        if best_d is None or dd < best_d:
            best_d = dd
            best = d
    return best if best is not None else value


@define
class GeniusOpponentModel(UFunModel):
    """Base class for Genius BOA opponent models ported as NegMAS ufun models.

    Subclasses implement:

    - :meth:`_setup` — initialise the internal model from ``self._issues`` (called
      by :meth:`initialize` and by :meth:`on_preferences_changed`).
    - :meth:`update` — learn from a single observed opponent offer.
    - :meth:`eval` — estimate the opponent's utility of an outcome.

    The base handles the NegMAS component plumbing (defensive
    ``__attrs_post_init__`` so the model is a full ``BaseUtilityFunction`` even on
    negmas versions whose base ``UFunModel`` omits it, the ``outcome_space``
    property, and wiring ``on_partner_proposal`` / ``before_responding`` to
    :meth:`update`).
    """

    _issues: list = field(init=False, factory=list)
    _discrete_issues: list = field(init=False, factory=list)

    # -- NegMAS ufun plumbing -------------------------------------------------
    def __attrs_post_init__(self):
        """Ensure full ``BaseUtilityFunction`` state exists.

        ``@define`` subclasses skip ``BaseUtilityFunction.__init__``. Newer negmas
        provides a ``UFunModel.__attrs_post_init__`` that sets this up; older
        negmas does not, so we fall back to initialising it ourselves.
        """
        parent = getattr(super(), "__attrs_post_init__", None)
        if parent is not None:
            parent()
        else:  # pragma: no cover - depends on installed negmas version
            BaseUtilityFunction.__init__(self, reserved_value=0.0)

    @property
    def outcome_space(self):  # type: ignore[override]
        """The outcome space the model is defined over (the negotiator's)."""
        negotiator = getattr(self, "negotiator", None)
        if negotiator is not None and getattr(negotiator, "ufun", None) is not None:
            return negotiator.ufun.outcome_space
        return getattr(self, "_outcome_space", None)

    @outcome_space.setter
    def outcome_space(self, value) -> None:
        self._outcome_space = value

    # -- standalone API -------------------------------------------------------
    def initialize(self, issues_or_space, levels: int = 10) -> GeniusOpponentModel:
        """Initialise the model from issues or an outcome space (standalone use).

        Args:
            issues_or_space: A list of :class:`~negmas.outcomes.Issue`, or an
                :class:`~negmas.outcomes.OutcomeSpace` with an ``issues`` attribute.
            levels: Discretization granularity for continuous issues.

        Returns:
            ``self`` (so calls can be chained).
        """
        issues = getattr(issues_or_space, "issues", issues_or_space)
        self._issues = list(issues) if issues is not None else []
        self._discrete_issues = discretize_issues(self._issues, levels)
        if getattr(issues_or_space, "issues", None) is not None:
            self._outcome_space = issues_or_space
        self._setup()
        return self

    def _setup(self) -> None:
        """Initialise internal learning structures from ``self._issues``.

        Subclasses override this. The default is a no-op.
        """

    def update(self, offer: Outcome | None) -> None:
        """Learn from a single observed opponent offer.

        Subclasses override this. The default is a no-op.

        Args:
            offer: The opponent's offer (``None`` is ignored).
        """

    # -- component callbacks --------------------------------------------------
    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Set up the model from the negotiator's outcome space (component use)."""
        negotiator = getattr(self, "negotiator", None)
        if negotiator is None or getattr(negotiator, "ufun", None) is None:
            return
        os_ = negotiator.ufun.outcome_space
        if os_ is not None:
            self.initialize(os_)
        self._update_private_info()

    def before_responding(self, state, offer: Outcome | None, source: str | None = None):
        """Learn from the offer the negotiator is about to respond to."""
        if offer is not None:
            self.update(offer)

    def on_partner_proposal(self, state, partner_id: str, offer: Outcome) -> None:
        """Learn from a partner proposal (only called with ``enable_callbacks``)."""
        if offer is not None:
            self.update(offer)

    # -- ufun API -------------------------------------------------------------
    def eval(self, offer: Outcome) -> Value:
        """Estimate the opponent's utility of ``offer`` (subclasses override)."""
        raise NotImplementedError()
