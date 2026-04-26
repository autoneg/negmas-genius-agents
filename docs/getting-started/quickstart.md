# Quick Start

This guide will help you get started with `negmas-genius-agents` quickly.

## Basic Usage

### Running a Simple Negotiation

```python
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism

from negmas_genius_agents import HardHeaded, AgentK

# Define the negotiation domain
issues = [
    make_issue(values=["low", "medium", "high"], name="price"),
    make_issue(values=["1", "2", "3"], name="quantity"),
    make_issue(values=["fast", "normal", "slow"], name="delivery"),
]
outcome_space = make_os(issues)

# Create utility functions
buyer_ufun = LinearAdditiveUtilityFunction(
    values={
        "price": {"low": 1.0, "medium": 0.5, "high": 0.0},
        "quantity": {"1": 0.0, "2": 0.5, "3": 1.0},
        "delivery": {"fast": 1.0, "normal": 0.5, "slow": 0.0},
    },
    weights={"price": 0.5, "quantity": 0.3, "delivery": 0.2},
    outcome_space=outcome_space,
)

seller_ufun = LinearAdditiveUtilityFunction(
    values={
        "price": {"low": 0.0, "medium": 0.5, "high": 1.0},
        "quantity": {"1": 1.0, "2": 0.5, "3": 0.0},
        "delivery": {"fast": 0.0, "normal": 0.5, "slow": 1.0},
    },
    weights={"price": 0.5, "quantity": 0.3, "delivery": 0.2},
    outcome_space=outcome_space,
)

# Create negotiation mechanism
mechanism = SAOMechanism(issues=issues, n_steps=100)

# Add negotiating agents
buyer = HardHeaded(name="buyer", ufun=buyer_ufun)
seller = AgentK(name="seller", ufun=seller_ufun)

mechanism.add(buyer)
mechanism.add(seller)

# Run negotiation
state = mechanism.run()

# Check results
if state.agreement:
    print(f"Agreement reached: {state.agreement}")
    print(f"Buyer utility: {buyer_ufun(state.agreement)}")
    print(f"Seller utility: {seller_ufun(state.agreement)}")
else:
    print("No agreement reached")
```

## Discovering Agents

### List All Agents

```python
from negmas_genius_agents import get_agents

# Get all available agents
all_agents = get_agents()
print(f"Total agents available: {len(all_agents)}")
```

### Get Agents by Year

```python
from negmas_genius_agents import get_agents

# Get agents from ANAC 2011
anac2011 = get_agents(group="anac2011")
for agent_class in anac2011:
    print(f"  - {agent_class.__name__}")
```

### Import Specific Agents

```python
# Import winners
from negmas_genius_agents import (
    AgentK,          # 2010 winner
    HardHeaded,      # 2011 winner
    CUHKAgent,       # 2012 winner
    TheFawkes,       # 2013 winner
    AgentM,          # 2014 winner
    Atlas3,          # 2015 winner
    Caduceus,        # 2016 winner
    PonPokoAgent,    # 2017 winner
    AgreeableAgent2018,  # 2018 winner
    AgentGG,         # 2019 winner
)

# Import from specific years
from negmas_genius_agents.negotiators.anac.y2011 import (
    HardHeaded,
    Gahboninho,
    AgentK2,
)
```

## Using Time-Dependent Agents

The library also includes classic time-dependent negotiation strategies:

```python
from negmas_genius_agents import (
    BoulwareNegotiator,
    ConcederNegotiator,
    LinearNegotiator,
    HardlinerNegotiator,
)

# Boulware: Concedes slowly, stays firm until deadline
boulware = BoulwareNegotiator(name="boulware", ufun=my_ufun)

# Conceder: Concedes quickly
conceder = ConcederNegotiator(name="conceder", ufun=my_ufun)

# Linear: Concedes at constant rate
linear = LinearNegotiator(name="linear", ufun=my_ufun)

# Hardliner: Never concedes
hardliner = HardlinerNegotiator(name="hardliner", ufun=my_ufun)
```

## Next Steps

- See [Available Agents](../user-guide/agents.md) for the complete list of agents
- Learn about [Running Negotiations](../user-guide/negotiations.md) in more detail
- Check out [Running Tournaments](../user-guide/tournaments.md) to compare agents
