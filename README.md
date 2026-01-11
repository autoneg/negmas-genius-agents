# negmas-genius-agents

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/negmas-genius-agents.svg)](https://pypi.python.org/pypi/negmas-genius-agents)
[![PyPI - Version](https://img.shields.io/pypi/v/negmas-genius-agents.svg)](https://pypi.python.org/pypi/negmas-genius-agents)
[![PyPI - Status](https://img.shields.io/pypi/status/negmas-genius-agents.svg)](https://pypi.python.org/pypi/negmas-genius-agents)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/negmas-genius-agents.svg)](https://pypi.python.org/pypi/negmas-genius-agents)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Python reimplementations of **124 [Genius](http://ii.tudelft.nl/genius/) negotiating agents** from ANAC competitions 2010-2019 for use with the [NegMAS](https://github.com/yasserfarouk/negmas) framework.

---

> **IMPORTANT NOTICE: AI-ASSISTED IMPLEMENTATION**
>
> The agents in this package were reimplemented from Java to Python with the assistance of AI (Large Language Models). While efforts have been made to faithfully reproduce the original agent behaviors, **these implementations may not behave identically to the original Genius agents in all cases**. 
>
> If you require guaranteed behavioral equivalence with the original Java implementations, please use the [GeniusNegotiator](https://negmas.readthedocs.io/en/latest/api/negmas.genius.GeniusNegotiator.html) wrapper in NegMAS, which runs the actual Java agents via a bridge.
>
> Bug reports and contributions to improve behavioral fidelity are welcome.

---

## Genius Attribution

This package provides Python reimplementations of agents originally developed for **[Genius](http://ii.tudelft.nl/genius/)** (General Environment for Negotiation with Intelligent multi-purpose Usage Simulation) — a Java-based automated negotiation framework developed at TU Delft.

- **Original Source:** [http://ii.tudelft.nl/genius/](http://ii.tudelft.nl/genius/)
- **Original License:** GPL-3.0
- **Original Version:** 10.4

Genius has been the standard platform for the [ANAC (Automated Negotiating Agents Competition)](http://ii.tudelft.nl/ANAC/) since 2010. The agents in this package have been reimplemented in Python to provide native NegMAS compatibility without requiring Java.

If you use this package, please cite Genius:

```bibtex
@article{lin2014genius,
  title={Genius: An integrated environment for supporting the design of generic automated negotiators},
  author={Lin, Raz and Kraus, Sarit and Baarslag, Tim and Tykhonov, Dmytro and Hindriks, Koen and Jonker, Catholijn M},
  journal={Computational Intelligence},
  volume={30},
  number={1},
  pages={48--70},
  year={2014},
  publisher={Wiley Online Library}
}
```

---

## Features

- **124 ANAC agents** from competitions 2010-2019
- **5 basic time-dependent agents** (Boulware, Conceder, Linear, Hardliner)
- **Pure Python** - No Java dependency required
- **Seamless integration** with NegMAS mechanisms and tournaments
- **Full compatibility** with NegMAS utility functions and outcome spaces
- **Easy agent retrieval** with `get_agents()` function

## Installation

```bash
pip install negmas-genius-agents
```

Or with uv:

```bash
uv add negmas-genius-agents
```

### Requirements

- Python >= 3.10
- NegMAS >= 0.10.0

### Development Installation

```bash
git clone https://github.com/autoneg/negmas-genius-agents.git
cd negmas-genius-agents
uv sync --dev
```

## Quick Start

```python
from negmas import SAOMechanism, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as U

from negmas_genius_agents import Atlas3, AgentK, get_agents

# Create negotiation scenario
issues = [make_issue("price", (0, 100)), make_issue("quality", (0, 10))]
mechanism = SAOMechanism(issues=issues, n_steps=100)

# Create agents with utility functions
ufun1 = U.random(issues=issues)
ufun2 = U.random(issues=issues)

# Use ANAC competition winners
agent1 = Atlas3(ufun=ufun1)   # ANAC 2015 winner
agent2 = AgentK(ufun=ufun2)   # ANAC 2010 winner

# Run negotiation
mechanism.add(agent1)
mechanism.add(agent2)
result = mechanism.run()

print(f"Agreement: {result.agreement}")
print(f"Agent1 utility: {ufun1(result.agreement)}")
print(f"Agent2 utility: {ufun2(result.agreement)}")
```

## Available Agents

### Summary

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

### Using `get_agents()`

```python
from negmas_genius_agents import get_agents

# Get all agents (129 total)
all_agents = get_agents()

# Get all ANAC winners
winners = get_agents(category="winners")

# Get agents from a specific year
agents_2015 = get_agents(group="anac2015")

# Get finalists (top 3) from multiple years
early_finalists = get_agents(group=["anac2010", "anac2011"], category="finalists")

# Get basic time-dependent agents
basic = get_agents(group="basic")
```

### Basic Time-Dependent Agents

| Agent | Parameter | Strategy |
|-------|-----------|----------|
| `TimeDependentAgent` | e=custom | Base class with configurable concession |
| `TimeDependentAgentBoulware` | e=0.2 | Tough negotiator; concedes slowly |
| `TimeDependentAgentConceder` | e=2.0 | Cooperative; concedes quickly |
| `TimeDependentAgentLinear` | e=1.0 | Constant concession rate |
| `TimeDependentAgentHardliner` | e=0 | Never concedes |

### ANAC 2010 Agents

| Agent | Rank | Strategy |
|-------|------|----------|
| `AgentK` | **1st** | Statistical opponent modeling with adaptive target |
| `Yushu` | 2nd | Time-dependent concession with best-10 tracking |
| `Nozomi` | 3rd | Threat-based concession |
| `IAMhaggler` | 4th | Bayesian opponent model |
| `AgentFSEGA` | - | Boulware with reservation estimation |
| `AgentSmith` | - | Multi-strategy opponent classification |
| `IAMcrazyHaggler` | - | Random high-utility bidding |

### ANAC 2011 Agents

| Agent | Rank | Strategy |
|-------|------|----------|
| `HardHeaded` | **1st** | Frequency-based opponent weight estimation |
| `Gahboninho` | 2nd | Three-phase strategy with cooperativeness tracking |
| `IAMhaggler2011` | 3rd | Adaptive concession rate |
| `AgentK2` | - | Enhanced AgentK |
| `BramAgent` | - | Boulware with issue frequency tracking |
| `TheNegotiator` | - | Four-phase with linear regression |

### ANAC 2012 Agents

| Agent | Rank | Strategy |
|-------|------|----------|
| `CUHKAgent` | **1st** | Two-phase with variance-based toughness detection |
| `AgentLG` | 2nd | Multi-phase with value preference learning |
| `OMACAgent` | 3rd | Adaptive Boulware |
| `TheNegotiatorReloaded` | - | Improved TheNegotiator |
| `MetaAgent2012` | - | Blends Boulware/Linear/Conceder |
| `IAMhaggler2012` | - | Nash-product maximization |
| `AgentMR` | - | Three-phase with risk awareness |

### ANAC 2013 Agents

| Agent | Rank | Strategy |
|-------|------|----------|
| `TheFawkes` | **1st** | AC_Next acceptance with opponent classification |
| `MetaAgent2013` | 2nd | Strategy portfolio |
| `TMFAgent` | 3rd | Time-management focused |
| `AgentKF` | - | Kalman filter-inspired tracking |
| `GAgent` | - | Nash-optimal bid selection |
| `InoxAgent` | - | Robust Boulware |
| `SlavaAgent` | - | Weighted bid scoring |

### ANAC 2014 Agents

| Agent | Rank | Strategy |
|-------|------|----------|
| `AgentM` | **1st** | Simulated annealing bid search |
| `DoNA` | 2nd | Deadline-oriented with domain analysis |
| `Gangster` | 3rd | Multi-strategy gang voting |
| `WhaleAgent` | - | Boulware with Nash product |
| `TUDelftGroup2` | - | Polynomial concession |
| `E2Agent` | - | Exploration-exploitation balance |
| `KGAgent` | - | Knowledge-guided threshold |
| `AgentYK` | - | Three-phase strategy |
| `BraveCat` | - | BOA framework |
| `Atlas` | - | Precursor to Atlas3 |
| `Aster` | - | Multi-criteria selection |
| `ArisawaYaki` | - | Wave-based oscillation |
| `AgentTD` | - | Classic time-dependent |
| `AgentTRP` | - | Trade-off, Risk, Pressure |
| `AgentQuest` | - | Quest-based goal setting |

### ANAC 2015 Agents

| Agent | Rank | Strategy |
|-------|------|----------|
| `Atlas3` | **1st** | Three-phase Boulware with popular bid tracking |
| `ParsAgent` | 2nd | Nash product optimization |
| `RandomDance` | 3rd | Randomized exploration |
| `AgentBuyog` | - | Three-phase deadline handling |
| `AgentH` | - | Hybrid time-dependent |
| `AgentHP` | - | High-performance polynomial |
| `AgentNeo` | - | Opponent hardness response |
| `AgentW` | - | Weighted multi-criteria |
| `AgentX` | - | Experimental adaptive |
| `AresParty` | - | Aggressive then cooperative |
| `CUHKAgent2015` | - | Updated CUHK strategy |
| `DrageKnight` | - | Defensive strategy |
| `Y2015Group2` | - | Balanced approach |
| `JonnyBlack` | - | Hardball strategy |
| `Kawaii` | - | Adaptive (actually aggressive) |
| `MeanBot` | - | Mean-based decisions |
| `Mercury` | - | Fast concession |
| `PNegotiator` | - | Probabilistic acceptance |
| `PhoenixParty` | - | Adaptive recovery |
| `PokerFace` | - | Bluffing-inspired |
| `SENGOKU` | - | Multi-phase warfare |
| `XianFaAgent` | - | Wisdom-based decisions |

### ANAC 2016 Agents

| Agent | Rank | Strategy |
|-------|------|----------|
| `Caduceus` | **1st** | Meta-strategy voting ensemble |
| `YXAgent` | 2nd | Opponent hardness estimation |
| `ParsCat` | 3rd | Nash product with issue analysis |
| `AgentHP2` | - | Multi-phase with trend detection |
| `AgentLight` | - | Lightweight Boulware |
| `AgentSmith2016` | - | Updated Smith with Nash |
| `Atlas32016` | - | Atlas3 with four-phase refinement |
| `ClockworkAgent` | - | Precision timing |
| `Farma` | - | Frequency-based opponent model |
| `GrandmaAgent` | - | Patient conservative |
| `MaxOops` | - | Aggressive with recovery |
| `MyAgent` | - | Rubick-based with Nash |
| `Ngent` | - | Gentle concession |
| `Terra` | - | Firm-to-flexible three-phase |

### ANAC 2017 Agents

| Agent | Rank | Strategy |
|-------|------|----------|
| `PonPokoAgent` | **1st** | Randomized threshold patterns (no opponent modeling!) |
| `CaduceusDC16` | 2nd | Multi-strategy ensemble |
| `BetaOne` | 3rd | Bayesian with three-phase |
| `AgentF` | - | Linear with opponent tracking |
| `AgentKN` | - | Sigmoid concession curve |
| `Farma2017` | - | Exponential with Nash |
| `GeneKing` | - | Genetic algorithm-inspired |
| `Gin` | - | Smooth polynomial |
| `Group3` | - | Three-phase strategy |
| `Imitator` | - | Tit-for-tat mirroring |
| `MadAgent` | - | Unpredictable randomness |
| `Mamenchis` | - | High patience |
| `Mosa` | - | Simulated annealing cooling |
| `ParsAgent3` | - | Third generation Pars |
| `Rubick` | - | Adaptive concession |
| `SimpleAgent2017` | - | Baseline linear |
| `TaxiBox` | - | Accumulated concession |

### ANAC 2018 Agents

| Agent | Rank | Strategy |
|-------|------|----------|
| `AgreeableAgent2018` | **1st** | Cooperative with adaptive compromise |
| `MengWan` | 2nd | Trend-based modeling |
| `Seto` | 3rd | Threshold-based multi-phase |
| `Agent33` | - | Triple-three structure |
| `AgentHerb` | - | Slow and steady |
| `AgentNP1` | - | Nash product variant |
| `AteamAgent` | - | Collaborative approach |
| `ConDAgent` | - | State machine transitions |
| `ExpRubick` | - | Enhanced Rubick |
| `FullAgent` | - | Multiple strategies |
| `IQSun2018` | - | IQ-based decisions |
| `PonPokoRampage` | - | Aggressive PonPoko |
| `Shiboy` | - | Honor-based acceptance |
| `Sontag` | - | Balanced strategy |
| `Yeela` | - | Adaptive concession |

### ANAC 2019 Agents

| Agent | Rank | Strategy |
|-------|------|----------|
| `AgentGG` | **1st** | Issue importance with Nash estimation |
| `KakeSoba` | 2nd | Fixed threshold with diversification |
| `SAGA` | 3rd | Genetic algorithm evolution |
| `AgentGP` | 3rd (Nash) | UCB exploration/exploitation |
| `AgentLarry` | - | Simple linear baseline |
| `DandikAgent` | - | Boulware slow concession |
| `EAgent` | - | Exponential decay |
| `FSEGA2019` | 2nd (Nash) | Adaptive concession |
| `GaravelAgent` | - | Tit-for-tat adaptation |
| `Gravity` | - | Accelerating concession |
| `HardDealer` | - | Aggressive hardball |
| `KAgent` | - | AgentK-inspired |
| `MINF` | - | Minimal information |
| `WinkyAgent` | 1st (Nash) | Nash product maximization |

## Mixing with NegMAS Agents

Genius agents can negotiate with native NegMAS agents:

```python
from negmas.sao import AspirationNegotiator
from negmas_genius_agents import HardHeaded

mechanism = SAOMechanism(issues=issues, n_steps=100)
mechanism.add(HardHeaded(name="genius_agent"), preferences=ufun1)
mechanism.add(AspirationNegotiator(name="negmas_agent"), preferences=ufun2)

state = mechanism.run()
```

## Running Tournaments

```python
from negmas.sao import SAOMechanism
from negmas_genius_agents import get_agents

# Get all ANAC winners
winners = get_agents(category="winners")

# Run round-robin tournament
results = []
for i, AgentA in enumerate(winners):
    for AgentB in winners[i+1:]:
        mechanism = SAOMechanism(issues=issues, n_steps=100)
        mechanism.add(AgentA(name="A"), preferences=ufun1)
        mechanism.add(AgentB(name="B"), preferences=ufun2)
        state = mechanism.run()
        results.append({
            "agent_a": AgentA.__name__,
            "agent_b": AgentB.__name__,
            "agreement": state.agreement is not None,
        })
```

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Building Documentation

```bash
uv run mkdocs serve
```

## License

AGPL-3.0 License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [NegMAS](https://github.com/yasserfarouk/negmas) - Negotiation Managed by Situated Agents
- [Genius](http://ii.tudelft.nl/genius/) - General Environment for Negotiation with Intelligent multi-purpose Usage Simulation
- [ANAC](http://ii.tudelft.nl/ANAC/) - Automated Negotiating Agents Competition

## Citation

If you use this library in your research, please cite this package, NegMAS, and Genius:

```bibtex
@software{negmas_genius_agents,
  title = {negmas-genius-agents: Python Reimplementations of Genius Negotiating Agents},
  author = {Mohammad, Yasser},
  year = {2024},
  url = {https://github.com/autoneg/negmas-genius-agents}
}

@inproceedings{mohammad2022negmas,
  title={NegMAS: A Platform for Automated Negotiations},
  author={Mohammad, Yasser and Nakadai, Shinji and Greenwald, Amy},
  booktitle={Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems},
  pages={1845--1847},
  year={2022}
}

@article{lin2014genius,
  title={Genius: An integrated environment for supporting the design of generic automated negotiators},
  author={Lin, Raz and Kraus, Sarit and Baarslag, Tim and Tykhonov, Dmytro and Hindriks, Koen and Jonker, Catholijn M},
  journal={Computational Intelligence},
  volume={30},
  number={1},
  pages={48--70},
  year={2014},
  publisher={Wiley Online Library}
}
```

## Related Projects

- [negmas-negolog](https://github.com/autoneg/negmas-negolog) - NegMAS wrappers for NegoLog agents
- [negmas](https://github.com/yasserfarouk/negmas) - The NegMAS negotiation framework
