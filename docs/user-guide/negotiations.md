# Running Negotiations

This guide covers how to run negotiations using agents from this library.

## Basic Negotiation Setup

```python
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism

from negmas_genius_agents import HardHeaded, CUHKAgent

# 1. Define issues
issues = [
    make_issue(values=10, name="price"),  # Integer issue 0-9
    make_issue(values=["A", "B", "C"], name="quality"),  # Categorical
]
outcome_space = make_os(issues)

# 2. Create utility functions
ufun1 = LinearAdditiveUtilityFunction(
    values={
        "price": lambda x: x / 9,  # Higher price = better for seller
        "quality": {"A": 0.0, "B": 0.5, "C": 1.0},
    },
    outcome_space=outcome_space,
)

ufun2 = LinearAdditiveUtilityFunction(
    values={
        "price": lambda x: 1 - x / 9,  # Lower price = better for buyer
        "quality": {"A": 1.0, "B": 0.5, "C": 0.0},
    },
    outcome_space=outcome_space,
)

# 3. Create mechanism
mechanism = SAOMechanism(
    issues=issues,
    n_steps=100,  # Maximum rounds
)

# 4. Add negotiators
mechanism.add(HardHeaded(name="seller", ufun=ufun1))
mechanism.add(CUHKAgent(name="buyer", ufun=ufun2))

# 5. Run
state = mechanism.run()
```

## Analyzing Results

```python
# Check if agreement was reached
if state.agreement:
    print(f"Agreement: {state.agreement}")
    
    # Get utilities
    seller_utility = ufun1(state.agreement)
    buyer_utility = ufun2(state.agreement)
    
    print(f"Seller utility: {seller_utility:.3f}")
    print(f"Buyer utility: {buyer_utility:.3f}")
    print(f"Social welfare: {seller_utility + buyer_utility:.3f}")
else:
    print("No agreement reached")

# Get negotiation statistics
print(f"Number of rounds: {state.step}")
print(f"Relative time: {state.relative_time:.2f}")
```

## Comparing Agents

Run multiple negotiations to compare agent performance:

```python
from negmas_genius_agents import get_agents, HardHeaded

# Get all 2011 agents
agents_2011 = get_agents(group="anac2011")

results = {}
for agent_class in agents_2011:
    # Run 10 negotiations
    wins = 0
    total_utility = 0
    
    for _ in range(10):
        mechanism = SAOMechanism(issues=issues, n_steps=100)
        mechanism.add(agent_class(name="test", ufun=ufun1))
        mechanism.add(HardHeaded(name="opponent", ufun=ufun2))
        
        state = mechanism.run()
        if state.agreement:
            utility = ufun1(state.agreement)
            total_utility += utility
            if utility > 0.5:
                wins += 1
    
    results[agent_class.__name__] = {
        "wins": wins,
        "avg_utility": total_utility / 10
    }

# Print results
for name, data in sorted(results.items(), key=lambda x: -x[1]["avg_utility"]):
    print(f"{name}: {data['avg_utility']:.3f} avg utility, {data['wins']}/10 wins")
```

## Negotiation Parameters

### Controlling Time

```python
# By number of steps
mechanism = SAOMechanism(issues=issues, n_steps=100)

# By real time (seconds)
mechanism = SAOMechanism(issues=issues, time_limit=60.0)

# Both constraints
mechanism = SAOMechanism(issues=issues, n_steps=100, time_limit=60.0)
```

### Hidden Time

Some agents perform differently when they don't know the deadline:

```python
mechanism = SAOMechanism(
    issues=issues,
    n_steps=100,
    hidden_time_limit=True  # Agents don't know when negotiation ends
)
```

## Multi-Party Negotiations

Many agents from 2015-2018 support multi-party negotiations:

```python
from negmas_genius_agents import Atlas3, ParsAgent, Caduceus

mechanism = SAOMechanism(issues=issues, n_steps=100)

mechanism.add(Atlas3(name="agent1", ufun=ufun1))
mechanism.add(ParsAgent(name="agent2", ufun=ufun2))
mechanism.add(Caduceus(name="agent3", ufun=ufun3))

state = mechanism.run()
```
