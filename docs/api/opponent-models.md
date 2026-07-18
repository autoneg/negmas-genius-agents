# Opponent Models API Reference

Python reimplementations of the **Genius BOA opponent models** (from the Java
`negotiator.boaframework.opponentmodel` package), exposed as NegMAS
[`UFunModel`](https://negmas.readthedocs.io/) subclasses — full stand-ins for a
utility function that estimate the **opponent's** utility of an outcome.

All Genius BOA opponent models learn the opponent's **preference profile** (§5.3 of
Baarslag, Hendrikx, Hindriks & Jonker, *Learning about the opponent in automated
bilateral negotiation: a comprehensive survey of opponent modeling techniques*,
JAAMAS 30:849–898, 2016).

## Usage

Each model can be used **standalone** (feed it observed opponent offers and query the
estimated opponent utility) or attached to a NegMAS component-based negotiator (it then
learns automatically from `on_partner_proposal`).

```python
from negmas.outcomes import make_issue
from negmas_genius_agents import HardHeadedFrequencyModel

issues = [make_issue(["a", "b", "c"], "color"), make_issue(["x", "y"], "size")]
model = HardHeadedFrequencyModel().initialize(issues)
for offer in [("a", "x"), ("a", "x"), ("a", "y")]:
    model.update(offer)          # learn from the opponent's offers
u = model(("a", "x"))            # estimated opponent utility (a full ufun call)
```

Importing the package registers every model with NegMAS's `component_registry`
(`component_type="model"`), so they are discoverable alongside the built-in models.

## Model catalogue

| Model | Technique (survey) | Description |
|-------|--------------------|-------------|
| `HardHeadedFrequencyModel` | §5.3.4 frequency analysis | Canonical ANAC frequency model: learns issue weights from which issues the opponent keeps fixed while conceding, and value scores from offer counts. |
| `SmithFrequencyModel` | §5.3.4 frequency analysis | AgentSmith's frequency model: issue weight from the concentration of per-value counts; value score from offer fraction. |
| `CUHKFrequencyModelV2` | §5.3.4 frequency analysis | CUHKAgent's frequency model: sum of per-issue value counts normalised by the sum of each issue's max count. |
| `NashFrequencyModel` | §5.3.4 frequency analysis | Frequency model with per-issue "percentage of highest frequency" weighting; utility clamped to [0, 1]. |
| `AgentXFrequencyModel` | §5.3.4 frequency analysis | AgentX's model: issue weight ∝ 1/change-count (issues that change more matter less); value score is an offer-driven rank. |
| `OppositeModel` | baseline (zero-sum) | Assumes `u_opponent = 1 − u_self`; needs no opponent information. |
| `UniformModel` | baseline | Constant `0.5` for every outcome. |
| `DefaultModel` | baseline | Constant `0.0` for every outcome. |
| `PerfectModel` | oracle (§6) | Returns the opponent's **true** utility (analysis/testing only). |
| `WorstModel` | oracle (§6) | Returns `1 − u_opponent(bid)` (analysis/testing only). |

## API

::: negmas_genius_agents.models
