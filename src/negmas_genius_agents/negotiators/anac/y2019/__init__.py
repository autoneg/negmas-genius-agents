"""
ANAC 2019 negotiating agents reimplemented from Genius.

This module contains Python reimplementations of agents that competed
in the Automated Negotiating Agents Competition (ANAC) 2019.

1st place: AgentGG
2nd place: KakeSoba
3rd place: SAGA

References:
    - https://ii.tudelft.nl/negotiation/node/12 (ANAC 2019)
    - Genius negotiation framework: https://ii.tudelft.nl/genius/
"""

from negmas_genius_agents.negotiators.anac.y2019.agent_gg import AgentGG
from negmas_genius_agents.negotiators.anac.y2019.agent_gp import AgentGP
from negmas_genius_agents.negotiators.anac.y2019.agent_larry import AgentLarry
from negmas_genius_agents.negotiators.anac.y2019.dandik_agent import DandikAgent
from negmas_genius_agents.negotiators.anac.y2019.e_agent import EAgent
from negmas_genius_agents.negotiators.anac.y2019.fsega2019 import FSEGA2019
from negmas_genius_agents.negotiators.anac.y2019.garavel_agent import GaravelAgent
from negmas_genius_agents.negotiators.anac.y2019.gravity import Gravity
from negmas_genius_agents.negotiators.anac.y2019.hard_dealer import HardDealer
from negmas_genius_agents.negotiators.anac.y2019.k_agent import KAgent
from negmas_genius_agents.negotiators.anac.y2019.kake_soba import KakeSoba
from negmas_genius_agents.negotiators.anac.y2019.minf import MINF
from negmas_genius_agents.negotiators.anac.y2019.saga import SAGA
from negmas_genius_agents.negotiators.anac.y2019.winky_agent import WinkyAgent

__all__ = [
    # Top 3
    "AgentGG",
    "KakeSoba",
    "SAGA",
    # Other agents (alphabetical)
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
]
