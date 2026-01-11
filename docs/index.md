# negmas-genius-agents

Python reimplementations of Genius negotiating agents for NegMAS.

This library provides Python implementations of classic negotiating agents from the [Genius](http://ii.tudelft.nl/genius/) framework, designed to work seamlessly with [NegMAS](https://negmas.readthedocs.io/).

---

## Important Notice: AI-Assisted Implementation

!!! warning "AI-Assisted Code Translation"

    The agents in this package were translated from Java to Python with the assistance of
    AI (Large Language Models). While significant effort has been made to faithfully
    reproduce the original agent behaviors, **these implementations may not behave
    identically to the original Java Genius agents in all cases**.

    If you require guaranteed behavioral equivalence with the original Java implementations,
    please use the `GeniusNegotiator` wrapper in NegMAS, which runs the actual Java agents
    via a bridge.

---

## Attribution & Citations

This library builds upon two foundational projects in automated negotiation research.
If you use this library in your research, please cite both:

### Genius Framework

The original agents were developed for the **Genius** negotiation platform. Please cite:

> Lin, R., Kraus, S., Baarslag, T., Tykhonov, D., Hindriks, K., & Jonker, C. M. (2014).
> **Genius: An Integrated Environment for Supporting the Design of Generic Automated Negotiators.**
> *Computational Intelligence*, 30(1), 48-70.
> [DOI: 10.1111/j.1467-8640.2012.00463.x](https://doi.org/10.1111/j.1467-8640.2012.00463.x)

```bibtex
@article{lin2014genius,
  title={Genius: An Integrated Environment for Supporting the Design of Generic Automated Negotiators},
  author={Lin, Raz and Kraus, Sarit and Baarslag, Tim and Tykhonov, Dmytro and Hindriks, Koen and Jonker, Catholijn M},
  journal={Computational Intelligence},
  volume={30},
  number={1},
  pages={48--70},
  year={2014},
  publisher={Wiley},
  doi={10.1111/j.1467-8640.2012.00463.x}
}
```

### NegMAS Platform

This library is designed to work with **NegMAS** (Negotiation Multi-Agent System). Please cite:

> Mohammad, Y., Nakadai, S., & Greenwald, A. (2021).
> **NegMAS: A Platform for Automated Negotiations.**
> In *PRIMA 2020: Principles and Practice of Multi-Agent Systems* (pp. 343-351). Springer.
> [DOI: 10.1007/978-3-030-69322-0_23](https://doi.org/10.1007/978-3-030-69322-0_23)

```bibtex
@inproceedings{mohammad2021negmas,
  title={NegMAS: A Platform for Automated Negotiations},
  author={Mohammad, Yasser and Nakadai, Shinji and Greenwald, Amy},
  booktitle={PRIMA 2020: Principles and Practice of Multi-Agent Systems},
  pages={343--351},
  year={2021},
  publisher={Springer},
  doi={10.1007/978-3-030-69322-0_23}
}
```

---

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

## Links & Resources

- **Genius Framework**: <http://ii.tudelft.nl/genius/>
- **ANAC Competition**: <http://ii.tudelft.nl/negotiation/>
- **NegMAS Documentation**: <https://negmas.readthedocs.io/>
- **NegMAS GitHub**: <https://github.com/yasserfarouk/negmas>
- **This Project**: <https://github.com/yasserfarouk/negmas-genius-agents>
