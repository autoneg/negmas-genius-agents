# Installation

## Using pip

The simplest way to install `negmas-genius-agents` is using pip:

```bash
pip install negmas-genius-agents
```

## Using uv

If you're using [uv](https://github.com/astral-sh/uv):

```bash
uv add negmas-genius-agents
```

## From Source

To install from source:

```bash
git clone https://github.com/yasserfarouk/negmas-genius-agents.git
cd negmas-genius-agents
pip install -e .
```

## Dependencies

The package requires:

- Python >= 3.10
- negmas >= 0.10.0
- numpy >= 1.24.0

## Verify Installation

After installation, verify everything works:

```python
from negmas_genius_agents import get_agents

# List all available agents
all_agents = get_agents()
print(f"Found {len(all_agents)} agents")

# List agents from a specific year
anac2011 = get_agents(group="anac2011")
print(f"ANAC 2011 agents: {[a.__name__ for a in anac2011]}")
```
