"""Genius BOA opponent models as NegMAS ufun models.

Python reimplementations of the opponent models from the Genius ``negotiator.
boaframework.opponentmodel`` package, exposed as
:class:`negmas.gb.components.models.ufun.UFunModel` subclasses (full stand-ins
for a utility function that estimate the *opponent's* utility of an outcome).

All Genius BOA opponent models learn the opponent's **preference profile**
(§5.3 of Baarslag, Hendrikx, Hindriks & Jonker, *Learning about the opponent in
automated bilateral negotiation*, JAAMAS 30:849–898, 2016). They are grouped by
learning technique:

- **Frequency analysis** (§5.3.4): :class:`HardHeadedFrequencyModel`.
- **Baselines / oracles** (§6): :class:`OppositeModel`, :class:`UniformModel`,
  :class:`DefaultModel`, :class:`PerfectModel`, :class:`WorstModel`.

Importing this package registers every model with ``negmas``'s
``component_registry`` (``component_type="model"``).

Example:
    >>> from negmas.outcomes import make_issue
    >>> from negmas_genius_agents.models import HardHeadedFrequencyModel
    >>> issues = [make_issue(["a", "b"], "x"), make_issue(["p", "q", "r"], "y")]
    >>> model = HardHeadedFrequencyModel().initialize(issues)
    >>> for offer in [("a", "p"), ("a", "p"), ("a", "q")]:
    ...     model.update(offer)
    >>> round(model(("a", "p")), 3) >= round(model(("b", "r")), 3)
    True
"""

from __future__ import annotations

from negmas_genius_agents.models.base import GeniusOpponentModel
from negmas_genius_agents.models.frequency import HardHeadedFrequencyModel
from negmas_genius_agents.models.baselines import (
    PerfectModel,
    WorstModel,
    OppositeModel,
    UniformModel,
    DefaultModel,
)

# Trigger registration with the negmas component registry (best-effort).
from negmas_genius_agents.models import registry_init as _registry_init  # noqa: E402,F401

__all__ = [
    "GeniusOpponentModel",
    "HardHeadedFrequencyModel",
    "PerfectModel",
    "WorstModel",
    "OppositeModel",
    "UniformModel",
    "DefaultModel",
]
