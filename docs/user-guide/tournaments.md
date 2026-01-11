# Running Tournaments

NegMAS supports running tournaments to compare negotiating agents systematically.

## Basic Tournament

```python
from negmas.tournaments import tournament
from negmas_genius_agents import get_agents

# Get all ANAC 2011 agents
agents = get_agents(group="anac2011")

# Run a tournament
results = tournament(
    competitors=agents,
    n_steps=100,
    n_repetitions=10,
)

# Print results
print(results.scores)
```

## Custom Tournament Setup

```python
from negmas.tournaments import SAOTournament
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.outcomes import make_issue, make_os

from negmas_genius_agents import (
    HardHeaded,
    AgentK,
    CUHKAgent,
    TheFawkes,
    AgentM,
)

# Define scenarios
issues = [
    make_issue(values=10, name="price"),
    make_issue(values=5, name="quantity"),
    make_issue(values=3, name="delivery"),
]

# Create tournament
tournament = SAOTournament(
    competitors=[
        HardHeaded,
        AgentK,
        CUHKAgent,
        TheFawkes,
        AgentM,
    ],
    issues=issues,
    n_steps=100,
    n_repetitions=20,
    randomize_ufuns=True,  # Generate random utility functions
)

# Run tournament
results = tournament.run()

# Analyze results
print("Tournament Results:")
print("=" * 50)
for name, score in sorted(
    results.scores.items(),
    key=lambda x: -x[1]
):
    print(f"{name:30s}: {score:.3f}")
```

## Analyzing Tournament Results

```python
# Get detailed statistics
stats = results.stats

# Win rates
print("\nWin Rates:")
for agent, rate in stats["win_rate"].items():
    print(f"  {agent}: {rate:.1%}")

# Average utilities
print("\nAverage Utilities:")
for agent, util in stats["avg_utility"].items():
    print(f"  {agent}: {util:.3f}")

# Agreement rates
print("\nAgreement Rates:")
for agent, rate in stats["agreement_rate"].items():
    print(f"  {agent}: {rate:.1%}")
```

## Year vs Year Tournament

Compare agents from different ANAC years:

```python
from negmas_genius_agents import get_agents

# Get winners from each year
winners = {
    "2010": "AgentK",
    "2011": "HardHeaded", 
    "2012": "CUHKAgent",
    "2013": "TheFawkes",
    "2014": "AgentM",
    "2015": "Atlas3",
    "2016": "Caduceus",
    "2017": "PonPokoAgent",
    "2018": "AgreeableAgent2018",
    "2019": "AgentGG",
}

# Import winners
from negmas_genius_agents import (
    AgentK, HardHeaded, CUHKAgent, TheFawkes, AgentM,
    Atlas3, Caduceus, PonPokoAgent, AgreeableAgent2018, AgentGG
)

competitors = [
    AgentK, HardHeaded, CUHKAgent, TheFawkes, AgentM,
    Atlas3, Caduceus, PonPokoAgent, AgreeableAgent2018, AgentGG
]

# Run winner-takes-all tournament
results = tournament(
    competitors=competitors,
    n_steps=100,
    n_repetitions=50,
)

print("ANAC Winners Face-Off:")
for i, (name, score) in enumerate(
    sorted(results.scores.items(), key=lambda x: -x[1])
):
    year = [y for y, n in winners.items() if n == name][0]
    print(f"{i+1}. {name} ({year}): {score:.3f}")
```

## Exporting Results

```python
import pandas as pd

# Convert to DataFrame
df = pd.DataFrame(results.detailed_scores)

# Save to CSV
df.to_csv("tournament_results.csv", index=False)

# Save to Excel
df.to_excel("tournament_results.xlsx", index=False)
```
