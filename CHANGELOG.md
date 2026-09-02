# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-09-02

### Added

- 18 previously missing ANAC agents, bringing the library to **141 ANAC agents**
  (2010-2019) plus 5 basic time-dependent negotiators:
  - **2011**: `NiceTitForTat`, `ValueModelAgent`
  - **2012**: `BRAMAgent2`
  - **2014**: `Flinch`, `SimpaticoAgent`, `Sobut`
  - **2016**: `ParsAgent2`, `SYAgent`
  - **2017**: `TucAgent`
  - **2018**: `BetaOne2018`, `GroupY`, `Lancelot`, `Libra`, `SMACAgent`
  - **2019**: `PodAgent`, `SACRA`, `SolverAgent`, `TheNewDeal`
- All new agents are wired into the package exports, the agent registry and the
  documentation catalog.
- Opt-in Python-vs-Java behavioral fidelity tests that run the original Genius
  agents through the NegMAS genius bridge and compare offer traces
  (enabled with `pytest --run-java`).

### Changed

- **Minimum `negmas` version is now 0.16.0** (was 0.15.2).
- Every ANAC agent (2010-2019) now exposes its previously hard-coded magic
  constants as `__init__` hyperparameters, so concession curves, acceptance
  thresholds and opponent-model parameters can be tuned without subclassing.
- Performance: `SortedOutcomeSpace` is now served from the utility function's
  cached inverse instead of re-sorting the outcome space.
- Performance: `MyAgent`, `WhaleAgent` and `NiceTitForTat` maintain per-issue
  maximum frequencies incrementally instead of recomputing them each round;
  `NiceTitForTat` also memoizes its Nash-point scan.
- Performance: `Yushu` uses a set for candidate-membership tests when choosing
  its next bid.

### Fixed

- `BetaOne` attribution corrected: the ANAC 2018 entry named BetaOne is
  `beta_one.Group2`, which is now implemented as `BetaOne2018`.

## [0.2.1] - 2026-01-17

### Fixed

- Updated the agent registry API for compatibility with newer `negmas` releases.

## [0.2.0] - 2026-01-13

### Added

- Parameterized all ANAC agents (2010-2019) with configurable magic numbers.

### Changed

- The package version is centralized in `pyproject.toml` and read at runtime via
  `importlib.metadata`.
- README lists the complete set of implemented agents.

### Fixed

- Infinite loop in the SAGA agent's `_evolve_population` method.
- Docstring warnings so the API reference renders correctly in mkdocs.

## [0.1.0] - 2026-01-11

### Added

- Initial release: Python reimplementations of 124 ANAC Genius agents
  (2010-2019) for NegMAS, plus basic time-dependent negotiators, an agent
  registry, and documentation.

[0.3.0]: https://github.com/autoneg/negmas-genius-agents/releases/tag/v0.3.0
[0.2.1]: https://github.com/autoneg/negmas-genius-agents/releases/tag/v0.2.1
[0.2.0]: https://github.com/autoneg/negmas-genius-agents/releases/tag/v0.2.0
[0.1.0]: https://github.com/autoneg/negmas-genius-agents/releases/tag/v0.1.0
