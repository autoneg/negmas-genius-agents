"""
ANAC 2018 negotiating agents reimplemented from Genius.

This module contains Python reimplementations of agents that competed
in the Automated Negotiating Agents Competition (ANAC) 2018.

1st place: AgreeableAgent2018
2nd place: MengWan
3rd place: Seto

References:
    - https://ii.tudelft.nl/negotiation/node/12 (ANAC 2018)
    - Genius negotiation framework: https://ii.tudelft.nl/genius/
"""

from negmas_genius_agents.negotiators.anac.y2018.agent_herb import AgentHerb
from negmas_genius_agents.negotiators.anac.y2018.agent_np1 import AgentNP1
from negmas_genius_agents.negotiators.anac.y2018.agent33 import Agent33
from negmas_genius_agents.negotiators.anac.y2018.agreeable_agent import (
    AgreeableAgent2018,
)
from negmas_genius_agents.negotiators.anac.y2018.ateam_agent import AteamAgent
from negmas_genius_agents.negotiators.anac.y2018.cond_agent import ConDAgent
from negmas_genius_agents.negotiators.anac.y2018.exp_rubick import ExpRubick
from negmas_genius_agents.negotiators.anac.y2018.full_agent import FullAgent
from negmas_genius_agents.negotiators.anac.y2018.iq_sun import IQSun2018
from negmas_genius_agents.negotiators.anac.y2018.meng_wan import MengWan
from negmas_genius_agents.negotiators.anac.y2018.pon_poko_rampage import PonPokoRampage
from negmas_genius_agents.negotiators.anac.y2018.seto import Seto
from negmas_genius_agents.negotiators.anac.y2018.shiboy import Shiboy
from negmas_genius_agents.negotiators.anac.y2018.sontag import Sontag
from negmas_genius_agents.negotiators.anac.y2018.yeela import Yeela

__all__ = [
    # Top 3
    "AgreeableAgent2018",
    "MengWan",
    "Seto",
    # Other agents (alphabetical)
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
]
