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
| 2011 | NiceTitForTat | `agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat` | ✅ |
| 2011 | ValueModelAgent | `agents.anac.y2011.ValueModelAgent.ValueModelAgent` | ✅ |
| 2012 | BRAMAgent2 | `agents.anac.y2012.BRAMAgent2.BRAMAgent2` | ✅ |
| 2014 | Flinch | `agents.anac.y2014.Flinch.Flinch` | ✅ |
| 2014 | SimpaticoAgent | `agents.anac.y2014.SimpaticoAgent.Simpatico` | ✅ |
| 2014 | Sobut | `agents.anac.y2014.Sobut.Sobut` | ✅ |
| 2016 | ParsAgent2 | `agents.anac.y2016.pars2.ParsAgent2` | ✅ |
| 2016 | SYAgent | `agents.anac.y2016.syagent.SYAgent` | ✅ |
| 2017 | TucAgent | `agents.anac.y2017.tucagent.TucAgent` | ✅ |
| 2018 | BetaOne2018 | `agents.anac.y2018.beta_one.Group2` | ✅ |
| 2018 | GroupY | `agents.anac.y2018.groupy.GroupY` | ✅ |
| 2018 | Lancelot | `agents.anac.y2018.lancelot.Lancelot` | ✅ |
| 2018 | Libra | `agents.anac.y2018.libra.Libra` | ✅ |
| 2018 | SMACAgent | `agents.anac.y2018.smac_agent.SMAC_Agent` | ✅ |
| 2019 | PodAgent | `agents.anac.y2019.podagent.Group1_BOA` | ✅ |
| 2019 | SACRA | `agents.anac.y2019.sacra.SACRA` | ✅ |
| 2019 | SolverAgent | `agents.anac.y2019.solveragent.SolverAgent` | ✅ |
| 2019 | TheNewDeal | `agents.anac.y2019.thenewdeal.TheNewDeal` | ✅ |

**Status:** ✅ **All 18 done and committed** (per-year commits, each verified against the
real Java agent via the bridge).

**Naming rule:** cross-year name clashes are disambiguated by appending the year to the
newer agent (matching the existing convention `AgentSmith2016`, `Farma2017`). The real
`BetaOne` is a **2018** agent (`agents.anac.y2018.beta_one.Group2`); it is added as
`BetaOne2018` because the existing `y2017/beta_one.py` already occupies the plain name —
that 2017 file is **misattributed** (there is no `agents.anac.y2017.*.BetaOne` in Genius /
`GENIUS_INFO`) and is flagged for follow-up cleanup.

---

## Part 2 — Opponent Models (BOA framework) — live in **negmas**, not here

Opponent models (and all BOA acceptance/offering components) belong in
`negmas.gb.components.genius`, which **already transpiles the full Genius BOA set**:

- **17 opponent models** (`negmas.gb.components.genius.models`): `GHardHeadedFrequencyModel`,
  `GSmithFrequencyModel`, `GCUHKFrequencyModel`, `GNashFrequencyModel`,
  `GAgentXFrequencyModel`, `GAgentLGModel` (frequency, §5.3.4); `GBayesianModel`,
  `GScalableBayesianModel`, `GFSEGABayesianModel`, `GIAMhagglerBayesianModel` (Bayesian,
  §5.3.2); `GDefaultModel`, `GUniformModel`, `GOppositeModel`, `GWorstModel`,
  `GPerfectModel` (baselines/oracles, §6); `GTheFawkesModel`, `GInoxAgentModel`
  (agent-specific).
- **48 acceptance policies** (`...genius.acceptance`) and **31 offering policies**
  (`...genius.offering`), all registered via negmas `component_registry`.

These modules were refactored from three huge single files into folder packages
(`models/`, `acceptance/`, `offering/`) — a verified-lossless split (class inventory
17/48/31 preserved exactly, no duplicates, all 199 genius tests pass). Import paths are
unchanged (`from negmas.gb.components.genius.models import GHardHeadedFrequencyModel`).

**This package therefore keeps only agents.** The short-lived
`negmas_genius_agents/models/` port (HardHeaded/Smith/CUHKv2/Nash/AgentX + baselines) has
been removed as a duplicate of the negmas set.

### negmas Table-2 model-type coverage additions
Two model *types* were missing from `negmas.models` and were added there (see Part 3).

---

## Part 3 — negmas Table-2 model-type coverage

The user asked that the **types** of opponent models available in `negmas` cover the four
opponent attributes of the survey's Table 2. Audit of `negmas/src/negmas/models/`:

| Table 2 attribute | negmas support | Action |
|-------------------|----------------|--------|
| §5.1 Acceptance strategy | `models/acceptance.py` — `DiscreteAcceptanceModel`, `AdaptiveDiscreteAcceptanceModel` | ✅ present |
| §5.2 Deadline | `models/deadline.py` — `DeadlineModel`, `ConcessionExtrapolatingDeadlineModel` | ✅ **added to negmas** |
| §5.3 Preference profile | `models/preferences.py::OpponentUtilityFunction`; `sao/gb components UFunModel` family | ✅ present |
| §5.4 Bidding/offering strategy | `models/strategy.py` — `OpponentOfferingModel`, `TimeSeriesOfferingModel` | ✅ **added to negmas** |

Done in negmas (commit `models: add Table-2 deadline (5.2) and offering-strategy (5.4)
model types`): abstract base + one simple worked example for each of the two missing
attribute types, and `negmas.models.__init__` now re-exports all four attribute families.
(Class names use `Offering` rather than `Bidding` to match negmas naming conventions.)

---

## Verification

The Java Genius bridge is available in this environment
(`negmas.genius.genius_bridge_is_running()` → `True`). Each port is checked for
behavioral consistency against the real Java implementation (via `GeniusNegotiator`).

`tests/test_python_vs_java.py` formalizes this: for every agent it runs the Python port
**and** the real Java agent on the same domain against the same opponent and asserts they
agree on whether a deal is reached and that achieved utilities are in the same ballpark
(agents whose Java side crashes in the bridge are skipped, not failed). A bridge-optional
`test_python_agent_runs` check (all 141 agents) always runs. Fixed two pre-existing wrong
Java-class mappings while wiring this up (`WhaleAgent`, `TaxiBox`).
