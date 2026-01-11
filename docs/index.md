# negmas-genius-agents

Python reimplementations of Genius negotiating agents for NegMAS.

This library provides Python implementations of classic negotiating agents from the [Genius](http://ii.tudelft.nl/genius/) framework, designed to work seamlessly with [NegMAS](https://negmas.readthedocs.io/).

## Features

- **100+ negotiating agents** from ANAC competitions (2010-2019)
- **Pure Python** - No Java dependencies required
- **NegMAS integration** - Works directly with NegMAS negotiations and tournaments
- **Well-tested** - Comprehensive test suite with 656 tests

## Quick Start

```python
from negmas.sao import SAOMechanism
from negmas.preferences import LinearAdditiveUtilityFunction

# Import agents
from negmas_genius_agents import HardHeaded, AgentK, CUHKAgent

# Create a negotiation
mechanism = SAOMechanism(issues=my_issues, n_steps=100)

# Add agents with utility functions
mechanism.add(HardHeaded(ufun=buyer_ufun))
mechanism.add(AgentK(ufun=seller_ufun))

# Run the negotiation
result = mechanism.run()
```

## Available Agents

The library includes agents from all ANAC competitions:

| Year | Winner | Notable Agents |
|------|--------|----------------|
| 2010 | AgentK | Yushu, Nozomi, IAMhaggler |
| 2011 | HardHeaded | Gahboninho, AgentK2 |
| 2012 | CUHKAgent | AgentLG, OMACAgent |
| 2013 | TheFawkes | MetaAgent2013, TMFAgent |
| 2014 | AgentM | DoNA, Gangster |
| 2015 | Atlas3 | ParsAgent, RandomDance |
| 2016 | Caduceus | YXAgent, ParsCat |
| 2017 | PonPokoAgent | CaduceusDC16, BetaOne |
| 2018 | AgreeableAgent2018 | MengWan, Seto |
| 2019 | AgentGG | KakeSoba, SAGA |

## Installation

```bash
pip install negmas-genius-agents
```

## License

MIT License

## References

- [Genius Framework](http://ii.tudelft.nl/genius/)
- [ANAC Competition](http://ii.tudelft.nl/negotiation/)
- [NegMAS](https://negmas.readthedocs.io/)
