# Genius Completion Audit — Missing Agents & Opponent Models

_Audit date: 2026-07-18. Canonical agent list from `negmas.genius.ginfo.GENIUS_INFO`;
Java source from `../negmas-genius-bridge/src/main/java/`; opponent-model taxonomy from
Baarslag, Hendrikx, Hindriks & Jonker, "Learning about the opponent in automated bilateral
negotiation: a comprehensive survey of opponent modeling techniques", JAAMAS 30:849–898 (2016)._

This document is the single source of truth for the completion effort. Items are checked
off as they are implemented, verified against the Java bridge, and committed.

---

## Part 1 — Missing ANAC Agents

Before this effort: **124** agents (ANAC 2010–2019) + 5 time-dependent base agents.
Comparison of the canonical `GENIUS_INFO` class list against the Python package found the
following **missing** competition agents (non-competition helper packages such as 2014
`kGA_gent` and 2015 `TUDMixedStrategyAgent` are intentionally excluded — they are absent
from `GENIUS_INFO`).

| Year | Agent | Java class | Status |
|------|-------|-----------|--------|
| 2011 | NiceTitForTat | `agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat` | ⬜ |
| 2011 | ValueModelAgent | `agents.anac.y2011.ValueModelAgent.ValueModelAgent` | ⬜ |
| 2012 | BRAMAgent2 | `agents.anac.y2012.BRAMAgent2.BRAMAgent2` | ⬜ |
| 2014 | Flinch | `agents.anac.y2014.Flinch.Flinch` | ⬜ |
| 2014 | SimpaticoAgent | `agents.anac.y2014.SimpaticoAgent.Simpatico` | ⬜ |
| 2014 | Sobut | `agents.anac.y2014.Sobut.Sobut` | ⬜ |
| 2016 | ParsAgent2 | `agents.anac.y2016.pars2.ParsAgent2` | ⬜ |
| 2016 | SYAgent | `agents.anac.y2016.syagent.SYAgent` | ⬜ |
| 2017 | TucAgent | `agents.anac.y2017.tucagent.TucAgent` | ⬜ |
| 2018 | BetaOne | `agents.anac.y2018.beta_one.Group2` | ⬜ (relocate mislabeled 2017 file) |
| 2018 | GroupY | `agents.anac.y2018.groupy.GroupY` | ⬜ |
| 2018 | Lancelot | `agents.anac.y2018.lancelot.Lancelot` | ⬜ |
| 2018 | Libra | `agents.anac.y2018.libra.Libra` | ⬜ |
| 2018 | SMACAgent | `agents.anac.y2018.smac_agent.SMAC_Agent` | ⬜ |
| 2019 | PodAgent | `agents.anac.y2019.podagent.Group1_BOA` | ⬜ |
| 2019 | SACRA | `agents.anac.y2019.sacra.SACRA` | ⬜ |
| 2019 | SolverAgent | `agents.anac.y2019.solveragent.SolverAgent` | ⬜ |
| 2019 | TheNewDeal | `agents.anac.y2019.thenewdeal.TheNewDeal` | ⬜ |

**Naming rule:** cross-year name clashes are disambiguated by appending the year to the
newer agent (matching the existing convention `AgentSmith2016`, `Farma2017`). Relevant
clashes: 2018 `BetaOne` (note the current `y2017/beta_one.py` is mislabeled — the real
`BetaOne`/`Group2` is a 2018 agent).

---

## Part 2 — Opponent Models (BOA framework)

Genius BOA opponent models live in
`negotiator/boaframework/opponentmodel/*.java` (plus `agents/bayesianopponentmodel/`).
They are being ported as `negmas` `UFunModel` subclasses (`GBComponent` +
`BaseUtilityFunction`) in a new `negmas_genius_agents/models/` package and registered with
`negmas.registry.component_registry` (`component_type="model"`).

All Genius BOA opponent models estimate the opponent's **preference profile** (Table 2 §5.3
in the survey). They divide into two families by learning technique:

### 2a. Frequency-analysis models (§5.3.4 — heuristics; §5.3.1 — issue preference order)
| Model | Java class | Notes | Status |
|-------|-----------|-------|--------|
| HardHeadedFrequencyModel | `...opponentmodel.HardHeadedFrequencyModel` | Most-used ANAC freq. model (weights from concession + value counts) | ⬜ |
| SmithFrequencyModel | `...opponentmodel.SmithFrequencyModel` | AgentSmith frequency model | ⬜ |
| SmithFrequencyModelV2 | `...opponentmodel.SmithFrequencyModelV2` | | ⬜ |
| CUHKFrequencyModelV2 | `...opponentmodel.CUHKFrequencyModelV2` | | ⬜ |
| NashFrequencyModel | `...opponentmodel.NashFrequencyModel` | | ⬜ |
| AgentXFrequencyModel | `...opponentmodel.AgentXFrequencyModel` | | ⬜ |
| AgentLGModel | `...opponentmodel.AgentLGModel` | value-statistics based | ⬜ |

### 2b. Bayesian-learning models (§5.3.2 — classifying the negotiation trace)
| Model | Java class | Notes | Status |
|-------|-----------|-------|--------|
| BayesianModel | `...opponentmodel.BayesianModel` | Hindriks–Tykhonov Bayesian | ⬜ |
| ScalableBayesianModel | `...opponentmodel.ScalableBayesianModel` | scalable variant | ⬜ |
| IAMhagglerBayesianModel | `...opponentmodel.IAMhagglerBayesianModel` | | ⬜ |
| FSEGABayesianModel | `...opponentmodel.FSEGABayesianModel` | | ⬜ |

### 2c. Oracle / baseline models (perfect or trivial — useful for benchmarking, per survey §6)
| Model | Java class | Notes | Status |
|-------|-----------|-------|--------|
| PerfectModel | `...opponentmodel.PerfectModel` | knows the true opponent ufun (oracle) | ⬜ |
| WorstModel | `...opponentmodel.WorstModel` | negated ufun (≈ `ZeroSumModel` in negmas) | ⬜ |
| OppositeModel | `...opponentmodel.OppositeModel` | | ⬜ |
| UniformModel | `...opponentmodel.UniformModel` | constant utility | ⬜ |
| DefaultModel | `...opponentmodel.DefaultModel` | no-op default | ⬜ |

### 2d. Agent-specific models
| Model | Java class | Status |
|-------|-----------|--------|
| TheFawkes_OM | `...opponentmodel.TheFawkes_OM` | ⬜ |
| InoxAgent_OM | `...opponentmodel.InoxAgent_OM` | ⬜ |
| SmithModel (agentsmith) | `...opponentmodel.agentsmith.SmithModel` | ⬜ |

---

## Part 3 — negmas Table-2 model-type coverage

The user asked that the **types** of opponent models available in `negmas` cover the four
opponent attributes of the survey's Table 2. Audit of `negmas/src/negmas/models/`:

| Table 2 attribute | negmas support | Action |
|-------------------|----------------|--------|
| §5.1 Acceptance strategy | `models/acceptance.py` — `DiscreteAcceptanceModel`, `AdaptiveDiscreteAcceptanceModel` | ✅ present |
| §5.2 Deadline | — none — | ➕ add base type + simple example to negmas |
| §5.3 Preference profile | `models/preferences.py::OpponentUtilityFunction`; `sao/gb components UFunModel` family | ✅ present |
| §5.4 Bidding strategy | `models/strategy.py` is **empty**; `models/future.py::FutureUtilityRegressor` partial | ➕ add base type + simple example to negmas |

Deadline and bidding-strategy model base classes (with a simple worked example each) are to
be added to `negmas` itself (not to this package), per the survey taxonomy.

---

## Verification

The Java Genius bridge is available in this environment
(`negmas.genius.genius_bridge_is_running()` → `True`). Each port is checked for
behavioral consistency against the real Java implementation (via `GeniusNegotiator`) using
the harness in `tests/test_behavioral_comparison.py` before being marked ✅.
