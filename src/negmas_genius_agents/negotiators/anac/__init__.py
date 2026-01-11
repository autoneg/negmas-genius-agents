"""
ANAC (Automated Negotiating Agents Competition) agents.

This module contains Python reimplementations of agents from ANAC competitions
(2010-2019 and beyond).
"""

from negmas_genius_agents.negotiators.anac.y2010 import AgentK, Yushu, Nozomi
from negmas_genius_agents.negotiators.anac.y2011 import (
    HardHeaded,
    Gahboninho,
    IAMhaggler2011,
)
from negmas_genius_agents.negotiators.anac.y2012 import CUHKAgent, AgentLG, OMACAgent
from negmas_genius_agents.negotiators.anac.y2013 import (
    TheFawkes,
    MetaAgent2013,
    TMFAgent,
)
from negmas_genius_agents.negotiators.anac.y2014 import AgentM, DoNA, Gangster
from negmas_genius_agents.negotiators.anac.y2015 import Atlas3, ParsAgent, RandomDance
from negmas_genius_agents.negotiators.anac.y2016 import Caduceus, YXAgent, MyAgent
from negmas_genius_agents.negotiators.anac.y2017 import (
    PonPokoAgent,
    CaduceusDC16,
    BetaOne,
)
from negmas_genius_agents.negotiators.anac.y2018 import (
    AgreeableAgent2018,
    MengWan,
    Seto,
)
from negmas_genius_agents.negotiators.anac.y2019 import AgentGG, KakeSoba, SAGA

__all__ = [
    # ANAC 2010
    "AgentK",
    "Yushu",
    "Nozomi",
    # ANAC 2011
    "HardHeaded",
    "Gahboninho",
    "IAMhaggler2011",
    # ANAC 2012
    "CUHKAgent",
    "AgentLG",
    "OMACAgent",
    # ANAC 2013
    "TheFawkes",
    "MetaAgent2013",
    "TMFAgent",
    # ANAC 2014
    "AgentM",
    "DoNA",
    "Gangster",
    # ANAC 2015
    "Atlas3",
    "ParsAgent",
    "RandomDance",
    # ANAC 2016
    "Caduceus",
    "YXAgent",
    "MyAgent",
    # ANAC 2017
    "PonPokoAgent",
    "CaduceusDC16",
    "BetaOne",
    # ANAC 2018
    "AgreeableAgent2018",
    "MengWan",
    "Seto",
    # ANAC 2019
    "AgentGG",
    "KakeSoba",
    "SAGA",
]
