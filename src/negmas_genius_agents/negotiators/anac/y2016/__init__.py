"""
ANAC 2016 negotiating agents reimplemented from Genius.

This module contains Python reimplementations of agents that competed
in the Automated Negotiating Agents Competition (ANAC) 2016.

1st place: Caduceus
2nd place: YXAgent
3rd place: ParsCat

References:
    - https://ii.tudelft.nl/negotiation/node/12 (ANAC 2016)
    - Genius negotiation framework: https://ii.tudelft.nl/genius/
"""

from negmas_genius_agents.negotiators.anac.y2016.agent_hp2 import AgentHP2
from negmas_genius_agents.negotiators.anac.y2016.agent_light import AgentLight
from negmas_genius_agents.negotiators.anac.y2016.agent_smith2016 import AgentSmith2016
from negmas_genius_agents.negotiators.anac.y2016.atlas3_2016 import Atlas32016
from negmas_genius_agents.negotiators.anac.y2016.caduceus import Caduceus
from negmas_genius_agents.negotiators.anac.y2016.clockwork_agent import ClockworkAgent
from negmas_genius_agents.negotiators.anac.y2016.farma import Farma
from negmas_genius_agents.negotiators.anac.y2016.grandma_agent import GrandmaAgent
from negmas_genius_agents.negotiators.anac.y2016.max_oops import MaxOops
from negmas_genius_agents.negotiators.anac.y2016.my_agent import MyAgent
from negmas_genius_agents.negotiators.anac.y2016.ngent import Ngent
from negmas_genius_agents.negotiators.anac.y2016.pars_cat import ParsCat
from negmas_genius_agents.negotiators.anac.y2016.terra import Terra
from negmas_genius_agents.negotiators.anac.y2016.yx_agent import YXAgent

__all__ = [
    # Top 3
    "Caduceus",
    "YXAgent",
    "ParsCat",
    # Other agents (alphabetical)
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
]
