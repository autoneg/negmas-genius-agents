# Available Agents

This page lists all 124 negotiating agents available in the library, organized by ANAC competition year. Each agent is a Python reimplementation of its original Java Genius counterpart.

!!! warning "AI-Generated Implementations"
    All agents in this library are AI-generated reimplementations based on the original
    Java source code from the Genius framework. While we strive for behavioral fidelity,
    they may not behave identically to the originals in all cases.

## Quick Reference

| Group | Winner | Total Agents |
|-------|--------|--------------|
| basic | TimeDependentAgent | 5 |
| anac2010 | AgentK | 7 |
| anac2011 | HardHeaded | 6 |
| anac2012 | CUHKAgent | 7 |
| anac2013 | TheFawkes | 7 |
| anac2014 | AgentM | 15 |
| anac2015 | Atlas3 | 22 |
| anac2016 | Caduceus | 14 |
| anac2017 | PonPokoAgent | 17 |
| anac2018 | AgreeableAgent2018 | 15 |
| anac2019 | AgentGG | 14 |

---

## Basic Agents

The `basic` group contains fundamental time-dependent agents that serve as baselines and building blocks.

```python
from negmas_genius_agents import get_agents

# Get all basic agents
basic_agents = get_agents(group="basic")
```

| Agent | Parameter | Strategy Summary |
|-------|-----------|------------------|
| `TimeDependentAgent` | e=custom | Base class with formula: `target(t) = Pmin + (Pmax-Pmin) * (1 - t^(1/e))` |
| `TimeDependentAgentBoulware` | e=0.2 | Tough negotiator; concedes slowly, mainly near deadline |
| `TimeDependentAgentConceder` | e=2.0 | Cooperative; concedes quickly early in negotiation |
| `TimeDependentAgentLinear` | e=1.0 | Constant concession rate throughout |
| `TimeDependentAgentHardliner` | e=0 | Never concedes; always offers best bid for self |

---

## ANAC 2010

The first ANAC competition featured 7 agents.

| Agent | Rank | Strategy Summary |
|-------|------|------------------|
| `AgentK` | **1st** | Statistical opponent modeling with adaptive target; tracks mean/variance of offers |
| `Yushu` | 2nd | Time-dependent concession with best-10 opponent bid tracking |
| `Nozomi` | 3rd | Threat-based concession; mirrors opponent's concession behavior |
| `IAMhaggler` | 4th | Bayesian opponent model with Nash-product bid selection |
| `AgentFSEGA` | - | Boulware concession with opponent reservation value estimation |
| `AgentSmith` | - | Multi-strategy; classifies opponent as HardHead/Conceder/Random/TFT |
| `IAMcrazyHaggler` | - | Random high-utility bidding with stubborn acceptance (exploits conceders) |

## ANAC 2011

| Agent | Rank | Strategy Summary |
|-------|------|------------------|
| `HardHeaded` | **1st** | Boulware concession with frequency-based opponent weight estimation |
| `Gahboninho` | 2nd | Three-phase strategy (early/main/panic) with opponent cooperativeness tracking |
| `IAMhaggler2011` | 3rd | Adaptive concession rate based on opponent behavior trends |
| `AgentK2` | - | Enhanced AgentK with improved statistical acceptance |
| `BramAgent` | - | Boulware with issue frequency tracking |
| `TheNegotiator` | - | Four-phase strategy with linear regression opponent prediction |

## ANAC 2012

| Agent | Rank | Strategy Summary |
|-------|------|------------------|
| `CUHKAgent` | **1st** | Two-phase concession with variance-based opponent toughness detection |
| `AgentLG` | 2nd | Multi-phase (early/compromise/panic/end) with value preference learning |
| `OMACAgent` | 3rd | Adaptive Boulware with opponent reservation value estimation |
| `TheNegotiatorReloaded` | - | Improved TheNegotiator with toughness-based adaptation |
| `MetaAgent2012` | - | Blends Boulware/Linear/Conceder based on opponent concession rate |
| `IAMhaggler2012` | - | Nash-product maximization with consistency-weighted opponent model |
| `AgentMR` | - | Three-phase (exploration/exploitation/compromise) with risk awareness |

## ANAC 2013

| Agent | Rank | Strategy Summary |
|-------|------|------------------|
| `TheFawkes` | **1st** | AC_Next acceptance with adaptive concession and opponent classification |
| `MetaAgent2013` | 2nd | Strategy portfolio with domain-based selection |
| `TMFAgent` | 3rd | Time-management focused with exploration/exploitation balance |
| `AgentKF` | - | Kalman filter-inspired statistical tracking |
| `GAgent` | - | Nash-optimal bid selection with frequency analysis |
| `InoxAgent` | - | Robust Boulware (e=0.05) with minimal opponent modeling |
| `SlavaAgent` | - | Weighted bid scoring with issue importance estimation |

## ANAC 2014

| Agent | Rank | Strategy Summary |
|-------|------|------------------|
| `AgentM` | **1st** | Simulated annealing bid search with adaptive acceptance |
| `DoNA` | 2nd | Deadline-oriented with domain analysis and opponent sensitivity |
| `Gangster` | 3rd | Multi-strategy gang voting (randomizes between tactics) |
| `WhaleAgent` | - | Boulware curve with Nash product bid selection |
| `TUDelftGroup2` | - | Polynomial concession with issue weighting |
| `E2Agent` | - | Simple exploration-exploitation balance |
| `KGAgent` | - | Knowledge-guided adaptive threshold |
| `AgentYK` | - | Three-phase: hardball, concession, final flexibility |
| `BraveCat` | - | BOA framework with combined acceptance criteria |
| `Atlas` | - | Precursor to Atlas3 with Pareto estimation |
| `Aster` | - | Multi-criteria star-pattern bid selection |
| `ArisawaYaki` | - | Wave-based oscillation in concession |
| `AgentTD` | - | Classic time-dependent (Boulware/Conceder switch) |
| `AgentTRP` | - | Trade-off, Risk, and Pressure balancing |
| `AgentQuest` | - | Quest-based goal setting with adaptive thresholds |

## ANAC 2015

Atlas3 was the winning agent and became highly influential in subsequent years.

| Agent | Rank | Strategy Summary |
|-------|------|------------------|
| `Atlas3` | **1st** | Three-phase Boulware with popular bid tracking for end-game |
| `ParsAgent` | 2nd | Nash product optimization with frequency-based opponent model |
| `RandomDance` | 3rd | Randomized exploration with adaptive acceptance |
| `AgentBuyog` | - | Three-phase (tough/moderate/flexible) with deadline handling |
| `AgentH` | - | Hybrid time-dependent with opponent tracking |
| `AgentHP` | - | High-performance polynomial concession |
| `AgentNeo` | - | "Chosen one" strategy with opponent hardness response |
| `AgentW` | - | Weighted multi-criteria bid selection |
| `AgentX` | - | Experimental adaptive concession |
| `AresParty` | - | War-themed aggressive then cooperative |
| `CUHKAgent2015` | - | Updated CUHK strategy with improved opponent model |
| `DrageKnight` | - | Knight-themed defensive strategy |
| `Y2015Group2` | - | Team-based development, balanced approach |
| `JonnyBlack` | - | Dark-themed hardball strategy |
| `Kawaii` | - | "Cute" adaptive strategy (actually quite aggressive) |
| `MeanBot` | - | Mean/average-based decision making |
| `Mercury` | - | Fast concession with deadline awareness |
| `PNegotiator` | - | Probabilistic acceptance with exploration |
| `PhoenixParty` | - | Rising from rejection with adaptive recovery |
| `PokerFace` | - | Bluffing-inspired concealment of intentions |
| `SENGOKU` | - | Japanese warfare-inspired multi-phase |
| `XianFaAgent` | - | Chinese strategy with wisdom-based decisions |

## ANAC 2016

| Agent | Rank | Strategy Summary |
|-------|------|------------------|
| `Caduceus` | **1st** | Meta-strategy voting with multi-agent ensemble |
| `YXAgent` | 2nd | Opponent hardness estimation with adaptive response |
| `MyAgent` | 3rd | Rubick-based with Nash equilibrium estimation |
| `AgentHP2` | - | Multi-phase concession with trend detection |
| `AgentLight` | - | Lightweight Boulware with minimal overhead |
| `AgentSmith2016` | - | Updated Smith with Nash-based selection |
| `Atlas32016` | - | Atlas3 with four-phase refinement |
| `ClockworkAgent` | - | Precision timing with phase-based strategy |
| `Farma` | - | Frequency-based opponent model, adaptive concession |
| `GrandmaAgent` | - | Patient conservative with late concession |
| `MaxOops` | - | Aggressive with recovery mechanisms |
| `Ngent` | - | Gentle concession strategy |
| `ParsCat` | - | Persian cat - Nash product with issue analysis |
| `Terra` | - | Earth-themed firm-to-flexible three-phase |

## ANAC 2017

PonPokoAgent won with a surprisingly simple but effective randomized approach.

| Agent | Rank | Strategy Summary |
|-------|------|------------------|
| `PonPokoAgent` | **1st** | Randomized threshold patterns; no opponent modeling (simplicity wins!) |
| `CaduceusDC16` | 2nd | Multi-strategy ensemble with weighted voting |
| `BetaOne` | 3rd | Bayesian opponent modeling with three-phase strategy |
| `AgentF` | - | Linear concession with simple opponent tracking |
| `AgentKN` | - | Sigmoid concession curve with adaptive threshold |
| `Farma2017` | - | Exponential concession with Nash-inspired selection |
| `GeneKing` | - | Genetic algorithm-inspired population bidding |
| `Gin` | - | Smooth polynomial with future offer estimation |
| `Group3` | - | Three-phase: hardball, exploration, concession |
| `Imitator` | - | Tit-for-tat inspired concession mirroring |
| `MadAgent` | - | Unpredictable "madness" with random elements |
| `Mamenchis` | - | Patient dinosaur - high patience parameter |
| `Mosa` | - | Simulated annealing-inspired cooling schedule |
| `ParsAgent3` | - | Third generation Pars with Nash optimization |
| `Rubick` | - | Adaptive concession based on opponent behavior |
| `SimpleAgent2017` | - | Baseline linear strategy for benchmarking |
| `TaxiBox` | - | "Fare meter" accumulated concession |

## ANAC 2018

| Agent | Rank | Strategy Summary |
|-------|------|------------------|
| `AgreeableAgent2018` | **1st** | Cooperative focus with adaptive compromise |
| `MengWan` | 2nd | Chinese university agent with trend-based modeling |
| `Seto` | 3rd | Threshold-based with multi-phase response |
| `Agent33` | - | Triple-three structure (3 phases x 3 criteria) |
| `AgentHerb` | - | Herbal patience - slow and steady |
| `AgentNP1` | - | Nash product variant with issue weighting |
| `AteamAgent` | - | Team A's collaborative approach |
| `ConDAgent` | - | Condition-based with state machine transitions |
| `ExpRubick` | - | Experimental Rubick with enhanced adaptation |
| `FullAgent` | - | Complete implementation of multiple strategies |
| `IQSun2018` | - | Intelligence quotient-based decision making |
| `PonPokoRampage` | - | Aggressive PonPoko variant |
| `Shiboy` | - | Japanese style with honor-based acceptance |
| `Sontag` | - | Literary-inspired balanced strategy |
| `Yeela` | - | Israeli agent with adaptive concession |

## ANAC 2019

| Agent | Rank | Strategy Summary |
|-------|------|------------------|
| `AgentGG` | **1st** | Issue importance-based bidding with Nash estimation |
| `KakeSoba` | 2nd | Fixed threshold with bid diversification |
| `SAGA` | 3rd | Genetic algorithm with population evolution |
| `AgentGP` | 3rd (Nash) | UCB-style exploration/exploitation balance |
| `AgentLarry` | - | Simple linear concession baseline |
| `DandikAgent` | - | Boulware-style slow concession |
| `EAgent` | - | Exponential decay concession curve |
| `FSEGA2019` | 2nd (Nash) | Adaptive concession based on opponent behavior |
| `GaravelAgent` | - | Tit-for-tat inspired adaptation |
| `Gravity` | - | Gravitational (accelerating) concession |
| `HardDealer` | - | Aggressive hardball tactics |
| `KAgent` | - | AgentK-inspired adaptive concession |
| `MINF` | - | Minimal information for robustness |
| `WinkyAgent` | 1st (Nash) | Nash product maximization strategy |

---

## Usage Example

```python
from negmas import SAOMechanism, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as U

from negmas_genius_agents import AgentK, HardHeaded, Atlas3

# Create negotiation scenario
issues = [make_issue("price", (0, 100)), make_issue("quality", (0, 10))]
mechanism = SAOMechanism(issues=issues, n_steps=100)

# Create agents with utility functions
ufun1 = U.random(issues=issues)
ufun2 = U.random(issues=issues)

# Use any ANAC agent
agent1 = AgentK(ufun=ufun1)  # ANAC 2010 winner
agent2 = Atlas3(ufun=ufun2)  # ANAC 2015 winner

# Run negotiation
mechanism.add(agent1)
mechanism.add(agent2)
result = mechanism.run()

print(f"Agreement: {result.agreement}")
print(f"Agent1 utility: {ufun1(result.agreement)}")
print(f"Agent2 utility: {ufun2(result.agreement)}")
```

## See Also

- [API Reference](../api/agents.md) - Detailed API documentation for all agents
- [Time-Dependent Agents](../api/time-dependent.md) - Base strategy implementations
- [Quickstart Guide](../getting-started/quickstart.md) - Getting started tutorial
