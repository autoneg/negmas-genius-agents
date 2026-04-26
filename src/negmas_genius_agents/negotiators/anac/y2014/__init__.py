"""
ANAC 2014 negotiating agents reimplemented from Genius.

This module contains Python reimplementations of agents that competed
in the Automated Negotiating Agents Competition (ANAC) 2014.

1st place: AgentM
2nd place: DoNA
3rd place: Gangster

References:
    - https://ii.tudelft.nl/negotiation/node/12 (ANAC 2014)
    - Genius negotiation framework: https://ii.tudelft.nl/genius/
"""

from negmas_genius_agents.negotiators.anac.y2014.agent_m import AgentM
from negmas_genius_agents.negotiators.anac.y2014.agent_quest import AgentQuest
from negmas_genius_agents.negotiators.anac.y2014.agent_td import AgentTD
from negmas_genius_agents.negotiators.anac.y2014.agent_trp import AgentTRP
from negmas_genius_agents.negotiators.anac.y2014.agent_yk import AgentYK
from negmas_genius_agents.negotiators.anac.y2014.arisawa_yaki import ArisawaYaki
from negmas_genius_agents.negotiators.anac.y2014.aster import Aster
from negmas_genius_agents.negotiators.anac.y2014.atlas import Atlas
from negmas_genius_agents.negotiators.anac.y2014.brave_cat import BraveCat
from negmas_genius_agents.negotiators.anac.y2014.dona import DoNA
from negmas_genius_agents.negotiators.anac.y2014.e2_agent import E2Agent
from negmas_genius_agents.negotiators.anac.y2014.gangster import Gangster
from negmas_genius_agents.negotiators.anac.y2014.kg_agent import KGAgent
from negmas_genius_agents.negotiators.anac.y2014.tu_delft_group2 import TUDelftGroup2
from negmas_genius_agents.negotiators.anac.y2014.whale_agent import WhaleAgent

__all__ = [
    # Top 3
    "AgentM",
    "DoNA",
    "Gangster",
    # Finalists
    "WhaleAgent",
    "TUDelftGroup2",
    "E2Agent",
    "KGAgent",
    "AgentYK",
    "BraveCat",
    # Other agents
    "AgentQuest",
    "AgentTD",
    "AgentTRP",
    "ArisawaYaki",
    "Aster",
    "Atlas",
]
