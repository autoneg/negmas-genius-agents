# negmas-genius-agents Implementation Progress

This file tracks the progress of reimplementing Genius negotiating agents in Python.

## Project Structure

```
src/negmas_genius_agents/
├── __init__.py                    # Main exports
├── negotiators/                   # Matches agents/ in Genius
│   ├── __init__.py
│   ├── time_dependent.py          # TimeDependentAgent, Boulware, Conceder, Linear
│   ├── anac/
│   │   ├── __init__.py
│   │   ├── y2010/
│   │   │   ├── __init__.py
│   │   │   ├── agent_k.py
│   │   │   └── ...
│   │   ├── y2011/
│   │   │   ├── __init__.py
│   │   │   ├── hard_headed.py     # HardHeaded (1st Place)
│   │   │   └── ...
│   │   └── ... (y2012-y2019)
│   └── bayesianopponentmodel/     # If needed by other agents
└── utils/
    ├── __init__.py
    └── outcome_space.py           # SortedOutcomeSpace, BidDetails
```

## Naming Conventions

- **Module names**: Python snake_case (e.g., `hard_headed.py`, `agent_k.py`)
- **Class names**: Match Genius names as closely as possible (e.g., `HardHeadedNegotiator`, `AgentK`)
- **Aliases**: Provide Agent suffix aliases (e.g., `HardHeadedAgent = HardHeadedNegotiator`)

## ANAC Competition Results & Implementation Status

Data sourced from `negmas.genius.ginfo.GENIUS_INFO`

### ANAC 2010 (Bilateral, Linear, Discounting) - ✅ COMPLETE
| Rank | Agent | Python Module | Status |
|------|-------|---------------|--------|
| 1st | AgentK | agent_k.py | ✅ DONE |
| 2nd | Yushu | yushu.py | ✅ DONE |
| 3rd | Nozomi | nozomi.py | ✅ DONE |
| 4th | IAMhaggler | iam_haggler.py | ✅ DONE |
| - | AgentFSEGA | agent_fsega.py | ✅ DONE |
| - | AgentSmith | agent_smith.py | ✅ DONE |
| - | IAMcrazyHaggler | iam_crazy_haggler.py | ✅ DONE |

### ANAC 2011 (Bilateral, Linear, Discounting) - ✅ COMPLETE
| Rank | Agent | Python Module | Status |
|------|-------|---------------|--------|
| 1st | HardHeaded | hard_headed.py | ✅ DONE |
| 2nd | Gahboninho | gahboninho.py | ✅ DONE |
| 3rd | IAMhaggler2011 | iam_haggler.py | ✅ DONE |
| F | AgentK2 | agent_k2.py | ✅ DONE |
| F | BramAgent | bram_agent.py | ✅ DONE |
| F | TheNegotiator | the_negotiator.py | ✅ DONE |

### ANAC 2012 (Bilateral, Linear, Discounting) - ✅ COMPLETE
| Rank | Agent | Python Module | Status |
|------|-------|---------------|--------|
| 1st | CUHKAgent | cuhk_agent.py | ✅ DONE |
| 2nd | AgentLG | agent_lg.py | ✅ DONE |
| 3rd | OMACAgent | omac_agent.py | ✅ DONE |
| F | TheNegotiatorReloaded | the_negotiator_reloaded.py | ✅ DONE |
| F | MetaAgent2012 | meta_agent.py | ✅ DONE |
| F | IAMhaggler2012 | iam_haggler2012.py | ✅ DONE |
| F | AgentMR | agent_mr.py | ✅ DONE |

### ANAC 2013 (Bilateral, Linear) - ✅ COMPLETE
| Rank | Agent | Python Module | Status |
|------|-------|---------------|--------|
| 1st | TheFawkes | the_fawkes.py | ✅ DONE |
| 2nd | MetaAgent2013 | meta_agent.py | ✅ DONE |
| 3rd | TMFAgent | tmf_agent.py | ✅ DONE |
| F | AgentKF | agent_kf.py | ✅ DONE |
| F | GAgent | g_agent.py | ✅ DONE |
| F | InoxAgent | inox_agent.py | ✅ DONE |
| F | SlavaAgent | slava_agent.py | ✅ DONE |

### ANAC 2014 (Bilateral, Non-Linear) - ✅ TOP 3 COMPLETE
| Rank | Agent | Python Module | Status |
|------|-------|---------------|--------|
| 1st | AgentM | agent_m.py | ✅ DONE |
| 2nd | DoNA | dona.py | ✅ DONE |
| 3rd | Gangster | gangster.py | ✅ DONE |
| F | WhaleAgent | whale_agent.py | ⬚ TODO |
| F | TUDelftGroup2 | tu_delft_group2.py | ⬚ TODO |
| F | E2Agent | e2_agent.py | ⬚ TODO |
| F | KGAgent | kg_agent.py | ⬚ TODO |
| F | AgentYK | agent_yk.py | ⬚ TODO |
| F | BraveCat | brave_cat.py | ⬚ TODO |

### ANAC 2015 (Multilateral, Linear) - ✅ TOP 3 COMPLETE
| Rank | Agent | Python Module | Status |
|------|-------|---------------|--------|
| 1st | Atlas3 | atlas3.py | ✅ DONE |
| 2nd | ParsAgent | pars_agent.py | ✅ DONE |
| 3rd | RandomDance | random_dance.py | ✅ DONE |
| F | Kawaii | kawaii.py | ⬚ TODO |
| F | AgentBuyog | agent_buyog.py | ⬚ TODO |
| F | PhoenixParty | phoenix_party.py | ⬚ TODO |

### ANAC 2016 (Multilateral, Linear) - ✅ TOP 3 COMPLETE
| Rank | Agent | Python Module | Status |
|------|-------|---------------|--------|
| 1st | Caduceus | caduceus.py | ✅ DONE |
| 2nd | YXAgent | yx_agent.py | ✅ DONE |
| 3rd | MyAgent | my_agent.py | ✅ DONE |
| F | ParsCat | pars_cat.py | ⬚ TODO |
| F | Farma | farma.py | ⬚ TODO |

### ANAC 2017 (Multilateral, Linear) - ✅ TOP 3 COMPLETE
| Rank | Agent | Python Module | Status |
|------|-------|---------------|--------|
| 1st | PonPokoAgent | ponpoko_agent.py | ✅ DONE |
| 2nd | CaduceusDC16 | caduceus_dc16.py | ✅ DONE |
| 3rd | BetaOne | beta_one.py | ✅ DONE |
| F | Rubick | rubick.py | ⬚ TODO |
| F | ParsAgent3 | pars_agent3.py | ⬚ TODO |

### ANAC 2018 (Multilateral, Linear) - ✅ TOP 3 COMPLETE
| Rank | Agent | Python Module | Status |
|------|-------|---------------|--------|
| 1st | AgreeableAgent2018 | agreeable_agent.py | ✅ DONE |
| 2nd | MengWan | meng_wan.py | ✅ DONE |
| 3rd | Seto | seto.py | ✅ DONE |
| F | IQSun2018 | iq_sun.py | ⬚ TODO |
| F | PonPokoRampage | pon_poko_rampage.py | ⬚ TODO |

### ANAC 2019 (Bilateral, Linear) - ✅ TOP 3 COMPLETE
| Rank | Agent | Python Module | Status |
|------|-------|---------------|--------|
| 1st | AgentGG | agent_gg.py | ✅ DONE |
| 2nd | KakeSoba | kake_soba.py | ✅ DONE |
| 3rd | SAGA | saga.py | ✅ DONE |
| F | WinkyAgent | winky_agent.py | ⬚ TODO |
| F | FSEGA2019 | fsega2019.py | ⬚ TODO |

## Root-Level Agents (non-ANAC)
| Agent | Status | Notes |
|-------|--------|-------|
| TimeDependentAgent | ✅ DONE | Base class |
| TimeDependentAgentBoulware | ✅ DONE | BoulwareNegotiator |
| TimeDependentAgentConceder | ✅ DONE | ConcederNegotiator |
| TimeDependentAgentLinear | ✅ DONE | LinearNegotiator |
| TimeDependentAgentHardliner | ✅ DONE | HardlinerNegotiator |
| ABMPAgent | ⬚ TODO | |
| SimpleAgent | ⬚ TODO | |
| DecUtilAgent | ⬚ TODO | |
| OptimalBidder | ⬚ TODO | |

## Supporting Modules
| Module | Status | Notes |
|--------|--------|-------|
| utils/outcome_space.py | ✅ DONE | SortedOutcomeSpace, BidDetails |
| bayesianopponentmodel/ | ⬚ TODO | Needed by NiceTitForTat, IAMhaggler, etc. |

## Implementation Statistics

### Current Totals
- **ANAC Winners (2010-2019)**: 10/10 ✅ COMPLETE
- **ANAC 2nd Place (2010-2019)**: 10/10 ✅ COMPLETE
- **ANAC 3rd Place (2010-2019)**: 10/10 ✅ COMPLETE
- **Additional Finalists**: ~10 more implemented
- **Root Agents**: 5/9 implemented

### Total Agents Implemented: ~45 agents

| Year | Agents Implemented |
|------|-------------------|
| 2010 | 7 (All) |
| 2011 | 6 (Top 3 + 3 Finalists) |
| 2012 | 7 (Top 3 + 4 Finalists) |
| 2013 | 7 (Top 3 + 4 Finalists) |
| 2014 | 3 (Top 3) |
| 2015 | 3 (Top 3) |
| 2016 | 3 (Top 3) |
| 2017 | 3 (Top 3) |
| 2018 | 3 (Top 3) |
| 2019 | 3 (Top 3) |
| **Total** | **45** |

## Priority Order for Future Implementation

### Phase 4: Remaining ANAC 2014 Finalists (~6 agents)
- WhaleAgent, TUDelftGroup2, E2Agent, KGAgent, AgentYK, BraveCat

### Phase 5: Remaining ANAC 2015-2019 Finalists (~20 agents)
- Various finalists from each year

### Phase 6: Non-ANAC Agents (~4 agents)
- ABMPAgent, SimpleAgent, DecUtilAgent, OptimalBidder

## Session Log

### Session 1 (Jan 10, 2026)
- Created project structure with pyproject.toml, README.md
- Implemented utils/outcome_space.py (SortedOutcomeSpace, BidDetails)
- Implemented TimeDependentNegotiator and variants (Boulware, Conceder, Linear, Hardliner)
- Implemented HardHeaded agent (ANAC 2011 Winner)
- Created test suites for time-based and ANAC 2011 agents
- Restructured to match Genius folder layout (negotiators/anac/y20XX/)
- Created this progress tracking file with accurate ANAC results from negmas.genius.ginfo
- Implemented AgentK (ANAC 2010 Winner)
- Fixed AgentK bug (_sorted_outcomes -> outcomes in _select_bid method)
- Implemented CUHKAgent (2012), TheFawkes (2013), AgentM (2014), Atlas3 (2015)
- Implemented Caduceus (2016), PonPokoAgent (2017), AgreeableAgent2018 (2018), AgentGG (2019)
- **Phase 1 Complete**: All 10 ANAC Winners (2010-2019) implemented and tested

### Session 2 (Jan 10, 2026) - Continued
- Implemented Phase 2 (2nd place agents) and Phase 3 (3rd place agents) for all years
- Implemented additional finalists for 2010-2013:
  - 2010: IAMhaggler, AgentFSEGA, AgentSmith, IAMcrazyHaggler
  - 2011: AgentK2, BramAgent, TheNegotiator
  - 2012: TheNegotiatorReloaded, MetaAgent2012, IAMhaggler2012, AgentMR
  - 2013: AgentKF, GAgent, InoxAgent, SlavaAgent
- Updated all __init__.py files to export new agents
- All 74 tests passing

### Current State
- **Phases 1-3 Complete**: All top 3 agents from every year (2010-2019) implemented
- **Early Years Complete**: All known ANAC 2010-2013 agents implemented
- Project compiles, all imports work, 74 tests passing
- Ready for Phase 4: Remaining finalists from 2014-2019

## Notes

### Common Patterns in Genius Agents
1. **Time-dependent concession**: Most agents use some form of time-based concession
2. **Opponent modeling**: Many use frequency-based or Bayesian opponent models
3. **Bid selection**: Select bids that maximize opponent utility while meeting target
4. **Acceptance strategy**: Accept if offer >= what we would propose next

### Dependencies Between Agents
- NiceTitForTat depends on BayesianOpponentModel
- IAMhaggler series depends on BayesianOpponentModel
- MetaAgent wraps other agents
- Some agents reuse components (e.g., SouthamptonAgent code in IAMhaggler)
- ParsCat2 (2017) reuses ParsCat (2016) code

### Testing Strategy
- Each agent should have basic negotiation tests
- Test against time-dependent agents and other ANAC agents
- Verify concession behavior matches original
- Test edge cases (single step, many steps, etc.)

### Year-Specific Notes
- 2010-2013: Bilateral only
- 2014: First year with non-linear utility spaces
- 2015-2018: Multilateral negotiations
- 2019: Back to bilateral
