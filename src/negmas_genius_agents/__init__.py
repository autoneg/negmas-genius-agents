"""
negmas-genius-agents: Python reimplementations of Genius negotiating agents.

This package provides Python reimplementations of negotiating agents originally
developed for the Genius negotiation framework (Java). These agents can be used
directly as NegMAS SAONegotiator instances.

IMPORTANT: AI-ASSISTED IMPLEMENTATION
-------------------------------------
The agents in this package were reimplemented from Java to Python with the
assistance of AI (Large Language Models). While efforts have been made to
faithfully reproduce the original agent behaviors, these implementations may
not behave identically to the original Genius agents in all cases.

If you require guaranteed behavioral equivalence with the original Java
implementations, please use the GeniusNegotiator wrapper in NegMAS, which
runs the actual Java agents via a bridge.

Example usage:
    >>> from negmas.outcomes import make_issue
    >>> from negmas.preferences import LinearAdditiveUtilityFunction
    >>> from negmas.sao import SAOMechanism
    >>> from negmas_genius_agents import TimeDependentAgentBoulware, TimeDependentAgentConceder
    >>>
    >>> issues = [
    ...     make_issue(values=["low", "medium", "high"], name="price"),
    ...     make_issue(values=["1", "2", "3"], name="quantity"),
    ... ]
    >>> mechanism = SAOMechanism(issues=issues, n_steps=100)
    >>> mechanism.add(TimeDependentAgentBoulware(name="buyer"), preferences=buyer_ufun)
    >>> mechanism.add(TimeDependentAgentConceder(name="seller"), preferences=seller_ufun)
    >>> result = mechanism.run()
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

# Basic time-dependent agents
from negmas_genius_agents.negotiators.time_dependent import (
    TimeDependentAgent,
    TimeDependentAgentBoulware,
    TimeDependentAgentConceder,
    TimeDependentAgentLinear,
    TimeDependentAgentHardliner,
)

# ANAC 2010 agents (7 agents)
from negmas_genius_agents.negotiators.anac.y2010 import (
    AgentK,
    Yushu,
    Nozomi,
    IAMhaggler,
    AgentFSEGA,
    AgentSmith,
    IAMcrazyHaggler,
)

# ANAC 2011 agents (6 agents)
from negmas_genius_agents.negotiators.anac.y2011 import (
    HardHeaded,
    Gahboninho,
    IAMhaggler2011,
    AgentK2,
    BramAgent,
    TheNegotiator,
)

# ANAC 2012 agents (7 agents)
from negmas_genius_agents.negotiators.anac.y2012 import (
    CUHKAgent,
    AgentLG,
    OMACAgent,
    TheNegotiatorReloaded,
    MetaAgent2012,
    IAMhaggler2012,
    AgentMR,
)

# ANAC 2013 agents (7 agents)
from negmas_genius_agents.negotiators.anac.y2013 import (
    TheFawkes,
    MetaAgent2013,
    TMFAgent,
    AgentKF,
    GAgent,
    InoxAgent,
    SlavaAgent,
)

# ANAC 2014 agents (15 agents)
from negmas_genius_agents.negotiators.anac.y2014 import (
    AgentM,
    DoNA,
    Gangster,
    WhaleAgent,
    TUDelftGroup2,
    E2Agent,
    KGAgent,
    AgentYK,
    BraveCat,
    AgentQuest,
    AgentTD,
    AgentTRP,
    ArisawaYaki,
    Aster,
    Atlas,
)

# ANAC 2015 agents (22 agents)
from negmas_genius_agents.negotiators.anac.y2015 import (
    Atlas3,
    ParsAgent,
    RandomDance,
    AgentBuyog,
    AgentH,
    AgentHP,
    AgentNeo,
    AgentW,
    AgentX,
    AresParty,
    CUHKAgent2015,
    DrageKnight,
    Y2015Group2,
    JonnyBlack,
    Kawaii,
    MeanBot,
    Mercury,
    PNegotiator,
    PhoenixParty,
    PokerFace,
    SENGOKU,
    XianFaAgent,
)

# ANAC 2016 agents (14 agents)
from negmas_genius_agents.negotiators.anac.y2016 import (
    Caduceus,
    YXAgent,
    ParsCat,
    AgentHP2,
    AgentLight,
    AgentSmith2016,
    Atlas32016,
    ClockworkAgent,
    Farma,
    GrandmaAgent,
    MaxOops,
    MyAgent,
    Ngent,
    Terra,
)

# ANAC 2017 agents (17 agents)
from negmas_genius_agents.negotiators.anac.y2017 import (
    PonPokoAgent,
    CaduceusDC16,
    BetaOne,
    AgentF,
    AgentKN,
    Farma2017,
    GeneKing,
    Gin,
    Group3,
    Imitator,
    MadAgent,
    Mamenchis,
    Mosa,
    ParsAgent3,
    Rubick,
    SimpleAgent2017,
    TaxiBox,
)

# ANAC 2018 agents (15 agents)
from negmas_genius_agents.negotiators.anac.y2018 import (
    AgreeableAgent2018,
    MengWan,
    Seto,
    Agent33,
    AgentHerb,
    AgentNP1,
    AteamAgent,
    ConDAgent,
    ExpRubick,
    FullAgent,
    IQSun2018,
    PonPokoRampage,
    Shiboy,
    Sontag,
    Yeela,
)

# ANAC 2019 agents (14 agents)
from negmas_genius_agents.negotiators.anac.y2019 import (
    AgentGG,
    KakeSoba,
    SAGA,
    AgentGP,
    AgentLarry,
    DandikAgent,
    EAgent,
    FSEGA2019,
    GaravelAgent,
    Gravity,
    HardDealer,
    KAgent,
    MINF,
    WinkyAgent,
)

from negmas_genius_agents.utils import (
    BidDetails,
    SortedOutcomeSpace,
)

__all__ = [
    # Time-dependent agents (basic)
    "TimeDependentAgent",
    "TimeDependentAgentBoulware",
    "TimeDependentAgentConceder",
    "TimeDependentAgentLinear",
    "TimeDependentAgentHardliner",
    # ANAC 2010 agents
    "AgentK",
    "Yushu",
    "Nozomi",
    "IAMhaggler",
    "AgentFSEGA",
    "AgentSmith",
    "IAMcrazyHaggler",
    # ANAC 2011 agents
    "HardHeaded",
    "Gahboninho",
    "IAMhaggler2011",
    "AgentK2",
    "BramAgent",
    "TheNegotiator",
    # ANAC 2012 agents
    "CUHKAgent",
    "AgentLG",
    "OMACAgent",
    "TheNegotiatorReloaded",
    "MetaAgent2012",
    "IAMhaggler2012",
    "AgentMR",
    # ANAC 2013 agents
    "TheFawkes",
    "MetaAgent2013",
    "TMFAgent",
    "AgentKF",
    "GAgent",
    "InoxAgent",
    "SlavaAgent",
    # ANAC 2014 agents
    "AgentM",
    "DoNA",
    "Gangster",
    "WhaleAgent",
    "TUDelftGroup2",
    "E2Agent",
    "KGAgent",
    "AgentYK",
    "BraveCat",
    "AgentQuest",
    "AgentTD",
    "AgentTRP",
    "ArisawaYaki",
    "Aster",
    "Atlas",
    # ANAC 2015 agents
    "Atlas3",
    "ParsAgent",
    "RandomDance",
    "AgentBuyog",
    "AgentH",
    "AgentHP",
    "AgentNeo",
    "AgentW",
    "AgentX",
    "AresParty",
    "CUHKAgent2015",
    "DrageKnight",
    "Y2015Group2",
    "JonnyBlack",
    "Kawaii",
    "MeanBot",
    "Mercury",
    "PNegotiator",
    "PhoenixParty",
    "PokerFace",
    "SENGOKU",
    "XianFaAgent",
    # ANAC 2016 agents
    "Caduceus",
    "YXAgent",
    "ParsCat",
    "AgentHP2",
    "AgentLight",
    "AgentSmith2016",
    "Atlas32016",
    "ClockworkAgent",
    "Farma",
    "GrandmaAgent",
    "MaxOops",
    "MyAgent",
    "Ngent",
    "Terra",
    # ANAC 2017 agents
    "PonPokoAgent",
    "CaduceusDC16",
    "BetaOne",
    "AgentF",
    "AgentKN",
    "Farma2017",
    "GeneKing",
    "Gin",
    "Group3",
    "Imitator",
    "MadAgent",
    "Mamenchis",
    "Mosa",
    "ParsAgent3",
    "Rubick",
    "SimpleAgent2017",
    "TaxiBox",
    # ANAC 2018 agents
    "AgreeableAgent2018",
    "MengWan",
    "Seto",
    "Agent33",
    "AgentHerb",
    "AgentNP1",
    "AteamAgent",
    "ConDAgent",
    "ExpRubick",
    "FullAgent",
    "IQSun2018",
    "PonPokoRampage",
    "Shiboy",
    "Sontag",
    "Yeela",
    # ANAC 2019 agents
    "AgentGG",
    "KakeSoba",
    "SAGA",
    "AgentGP",
    "AgentLarry",
    "DandikAgent",
    "EAgent",
    "FSEGA2019",
    "GaravelAgent",
    "Gravity",
    "HardDealer",
    "KAgent",
    "MINF",
    "WinkyAgent",
    # Utilities
    "BidDetails",
    "SortedOutcomeSpace",
    # Functions
    "get_agents",
]

from importlib.metadata import version as _get_version

__version__ = _get_version("negmas-genius-agents")


# Registry of all implemented agents organized by group and category
_AGENT_REGISTRY: dict[str, dict[str, list[type]]] = {
    "basic": {
        "winners": [TimeDependentAgent],
        "finalists": [TimeDependentAgent],
        "all": [
            TimeDependentAgent,
            TimeDependentAgentBoulware,
            TimeDependentAgentConceder,
            TimeDependentAgentLinear,
            TimeDependentAgentHardliner,
        ],
    },
    "anac2010": {
        "winners": [AgentK],
        "finalists": [AgentK, Yushu, Nozomi],
        "all": [
            AgentK,
            Yushu,
            Nozomi,
            IAMhaggler,
            AgentFSEGA,
            AgentSmith,
            IAMcrazyHaggler,
        ],
    },
    "anac2011": {
        "winners": [HardHeaded],
        "finalists": [HardHeaded, Gahboninho, IAMhaggler2011],
        "all": [
            HardHeaded,
            Gahboninho,
            IAMhaggler2011,
            AgentK2,
            BramAgent,
            TheNegotiator,
        ],
    },
    "anac2012": {
        "winners": [CUHKAgent],
        "finalists": [CUHKAgent, AgentLG, OMACAgent],
        "all": [
            CUHKAgent,
            AgentLG,
            OMACAgent,
            TheNegotiatorReloaded,
            MetaAgent2012,
            IAMhaggler2012,
            AgentMR,
        ],
    },
    "anac2013": {
        "winners": [TheFawkes],
        "finalists": [TheFawkes, MetaAgent2013, TMFAgent],
        "all": [
            TheFawkes,
            MetaAgent2013,
            TMFAgent,
            AgentKF,
            GAgent,
            InoxAgent,
            SlavaAgent,
        ],
    },
    "anac2014": {
        "winners": [AgentM],
        "finalists": [AgentM, DoNA, Gangster],
        "all": [
            AgentM,
            DoNA,
            Gangster,
            WhaleAgent,
            TUDelftGroup2,
            E2Agent,
            KGAgent,
            AgentYK,
            BraveCat,
            AgentQuest,
            AgentTD,
            AgentTRP,
            ArisawaYaki,
            Aster,
            Atlas,
        ],
    },
    "anac2015": {
        "winners": [Atlas3],
        "finalists": [Atlas3, ParsAgent, RandomDance],
        "all": [
            Atlas3,
            ParsAgent,
            RandomDance,
            AgentBuyog,
            AgentH,
            AgentHP,
            AgentNeo,
            AgentW,
            AgentX,
            AresParty,
            CUHKAgent2015,
            DrageKnight,
            Y2015Group2,
            JonnyBlack,
            Kawaii,
            MeanBot,
            Mercury,
            PNegotiator,
            PhoenixParty,
            PokerFace,
            SENGOKU,
            XianFaAgent,
        ],
    },
    "anac2016": {
        "winners": [Caduceus],
        "finalists": [Caduceus, YXAgent, ParsCat],
        "all": [
            Caduceus,
            YXAgent,
            ParsCat,
            AgentHP2,
            AgentLight,
            AgentSmith2016,
            Atlas32016,
            ClockworkAgent,
            Farma,
            GrandmaAgent,
            MaxOops,
            MyAgent,
            Ngent,
            Terra,
        ],
    },
    "anac2017": {
        "winners": [PonPokoAgent],
        "finalists": [PonPokoAgent, CaduceusDC16, BetaOne],
        "all": [
            PonPokoAgent,
            CaduceusDC16,
            BetaOne,
            AgentF,
            AgentKN,
            Farma2017,
            GeneKing,
            Gin,
            Group3,
            Imitator,
            MadAgent,
            Mamenchis,
            Mosa,
            ParsAgent3,
            Rubick,
            SimpleAgent2017,
            TaxiBox,
        ],
    },
    "anac2018": {
        "winners": [AgreeableAgent2018],
        "finalists": [AgreeableAgent2018, MengWan, Seto],
        "all": [
            AgreeableAgent2018,
            MengWan,
            Seto,
            Agent33,
            AgentHerb,
            AgentNP1,
            AteamAgent,
            ConDAgent,
            ExpRubick,
            FullAgent,
            IQSun2018,
            PonPokoRampage,
            Shiboy,
            Sontag,
            Yeela,
        ],
    },
    "anac2019": {
        "winners": [AgentGG],
        "finalists": [AgentGG, KakeSoba, SAGA],
        "all": [
            AgentGG,
            KakeSoba,
            SAGA,
            AgentGP,
            AgentLarry,
            DandikAgent,
            EAgent,
            FSEGA2019,
            GaravelAgent,
            Gravity,
            HardDealer,
            KAgent,
            MINF,
            WinkyAgent,
        ],
    },
}


def get_agents(
    group: str | list[str] | None = None,
    category: Literal["winners", "finalists", "all"] = "all",
) -> list[type]:
    """
    Get agent classes by group and category.

    Args:
        group: Group(s) to get agents from. Can be a single group (e.g., "anac2010"),
               a list of groups (e.g., ["anac2010", "anac2011"]), or None for all groups.
               Valid groups:
               - "basic": Basic time-dependent agents
               - "anac2010" through "anac2019": ANAC competition agents by year
        category: Category of agents to return:
                  - "winners": Only competition winners (1st place)
                  - "finalists": Top finishers (top 3)
                  - "all": All implemented agents from that group

    Returns:
        List of agent classes matching the criteria.

    Examples:
        >>> # Get all winners from all ANAC competitions
        >>> winners = get_agents(category="winners")

        >>> # Get all agents from 2011
        >>> agents_2011 = get_agents(group="anac2011")

        >>> # Get winners from 2010 and 2011
        >>> early_winners = get_agents(group=["anac2010", "anac2011"], category="winners")

        >>> # Get all finalists across all years
        >>> all_finalists = get_agents(category="finalists")

        >>> # Get basic time-dependent agents
        >>> basic = get_agents(group="basic")

        >>> # Get all ANAC agents (excluding basic)
        >>> anac_groups = [f"anac{year}" for year in range(2010, 2020)]
        >>> all_anac = get_agents(group=anac_groups)
    """
    if group is None:
        groups = list(_AGENT_REGISTRY.keys())
    elif isinstance(group, str):
        groups = [group]
    else:
        groups = group

    result: list[type] = []
    for g in groups:
        if g not in _AGENT_REGISTRY:
            raise ValueError(
                f"Group '{g}' not supported. Valid groups: {list(_AGENT_REGISTRY.keys())}"
            )
        agents = _AGENT_REGISTRY[g].get(category, [])
        for agent in agents:
            if agent not in result:
                result.append(agent)

    return result


# Register all negotiators with negmas registry (if available)
from . import registry_init as _registry_init  # noqa: F401
