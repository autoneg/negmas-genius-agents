# negmas-genius-agents

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/negmas-genius-agents.svg)](https://pypi.python.org/pypi/negmas-genius-agents)
[![PyPI - Version](https://img.shields.io/pypi/v/negmas-genius-agents.svg)](https://pypi.python.org/pypi/negmas-genius-agents)
[![PyPI - Status](https://img.shields.io/pypi/status/negmas-genius-agents.svg)](https://pypi.python.org/pypi/negmas-genius-agents)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/negmas-genius-agents.svg)](https://pypi.python.org/pypi/negmas-genius-agents)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Python reimplementations of [Genius](http://ii.tudelft.nl/genius/) negotiating agents for use with the [NegMAS](https://github.com/yasserfarouk/negmas) framework.

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

- **Pure Python** - No Java dependency required
- **ANAC Competition Agents** from 2010-2019 reimplemented as native NegMAS negotiators
- **Seamless integration** with NegMAS mechanisms and tournaments
- **Full compatibility** with NegMAS utility functions and outcome spaces
- **Faithful reimplementations** - Behavior matches the original Java agents

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
git clone https://github.com/yasserfarouk/negmas-genius-agents.git
cd negmas-genius-agents
uv sync --dev
```

## Quick Start

```python
from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism

from negmas_genius_agents import BoulwareAgent, ConcederAgent

# Define negotiation issues
issues = [
    make_issue(values=["low", "medium", "high"], name="price"),
    make_issue(values=["1", "2", "3"], name="quantity"),
]

# Create utility functions
buyer_ufun = LinearAdditiveUtilityFunction(
    values={
        "price": {"low": 1.0, "medium": 0.5, "high": 0.0},
        "quantity": {"1": 0.0, "2": 0.5, "3": 1.0},
    },
    weights={"price": 0.6, "quantity": 0.4},
)

seller_ufun = LinearAdditiveUtilityFunction(
    values={
        "price": {"low": 0.0, "medium": 0.5, "high": 1.0},
        "quantity": {"1": 1.0, "2": 0.5, "3": 0.0},
    },
    weights={"price": 0.6, "quantity": 0.4},
)

# Create mechanism and add agents
mechanism = SAOMechanism(issues=issues, n_steps=100)
mechanism.add(BoulwareAgent(name="buyer"), preferences=buyer_ufun)
mechanism.add(ConcederAgent(name="seller"), preferences=seller_ufun)

# Run negotiation
state = mechanism.run()

if state.agreement:
    print(f"Agreement reached: {state.agreement}")
    print(f"Buyer utility: {buyer_ufun(state.agreement):.3f}")
    print(f"Seller utility: {seller_ufun(state.agreement):.3f}")
else:
    print("No agreement reached")
```

## Available Agents

### Time-Based Concession Agents

| Agent | Description |
|-------|-------------|
| `BoulwareAgent` | Concedes slowly (e=0.2), tough negotiator |
| `ConcederAgent` | Concedes quickly (e=2.0), cooperative negotiator |
| `LinearAgent` | Linear concession over time (e=1.0) |
| `HardlinerAgent` | Never concedes (e=0), always offers maximum utility |

### ANAC 2011 Agents

| Agent | Description |
|-------|-------------|
| `HardHeaded` | **Winner** - Frequency-based opponent modeling |
| `Gahboninho` | **Runner-up** - Adaptive strategy |
| `NiceTitForTat` | Tit-for-tat strategy aiming for Nash point |

### ANAC 2012 Agents

| Agent | Description |
|-------|-------------|
| `CUHKAgent` | **Winner** - Chinese University of Hong Kong |
| `AgentLG` | Competition agent |

*(More agents will be added in future releases)*

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
from negmas_genius_agents import (
    BoulwareAgent, ConcederAgent, LinearAgent, HardlinerAgent
)

agents = [BoulwareAgent, ConcederAgent, LinearAgent, HardlinerAgent]

# Run round-robin tournament
results = []
for i, AgentA in enumerate(agents):
    for AgentB in agents[i+1:]:
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

## Architecture

```
negmas-genius-agents/
├── src/negmas_genius_agents/
│   ├── __init__.py           # Package exports
│   ├── base.py               # Base negotiator classes
│   ├── time_based.py         # Time-dependent agents
│   └── utils/                # Utility classes
│       ├── outcome_space.py  # Sorted outcome space
│       └── opponent_model.py # Opponent modeling
└── tests/
    └── test_agents.py        # Agent tests
```

## How It Works

These agents are **pure Python reimplementations** of the original Java Genius agents. The reimplementation process involved:

1. **Analyzing Java Source**: Understanding the original agent algorithms from Genius 10.4
2. **Python Translation**: Reimplementing the logic using NegMAS primitives
3. **Behavior Validation**: Ensuring the Python agents behave equivalently to their Java counterparts

Key components:
- **SortedOutcomeSpace**: Efficient bid lookup by utility value
- **Time-Dependent Strategy**: The classic `f(t) = k + (1-k) * t^(1/e)` concession function
- **Opponent Models**: Frequency-based and Bayesian opponent modeling

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
  url = {https://github.com/yasserfarouk/negmas-genius-agents}
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

- [negmas-negolog](https://github.com/yasserfarouk/negmas-negolog) - NegMAS wrappers for NegoLog agents
- [negmas](https://github.com/yasserfarouk/negmas) - The NegMAS negotiation framework
