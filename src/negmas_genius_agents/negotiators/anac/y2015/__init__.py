"""
ANAC 2015 negotiating agents reimplemented from Genius.

This module contains Python reimplementations of agents that competed
in the Automated Negotiating Agents Competition (ANAC) 2015.

1st place: Atlas3
2nd place: ParsAgent
3rd place: RandomDance

Note: ANAC 2015 was a multilateral negotiation competition, but this
implementation works in bilateral settings as well.

References:
    - https://ii.tudelft.nl/negotiation/node/12 (ANAC 2015)
    - Genius negotiation framework: https://ii.tudelft.nl/genius/
"""

from negmas_genius_agents.negotiators.anac.y2015.agent_buyog import AgentBuyog
from negmas_genius_agents.negotiators.anac.y2015.agent_h import AgentH
from negmas_genius_agents.negotiators.anac.y2015.agent_hp import AgentHP
from negmas_genius_agents.negotiators.anac.y2015.agent_neo import AgentNeo
from negmas_genius_agents.negotiators.anac.y2015.agent_w import AgentW
from negmas_genius_agents.negotiators.anac.y2015.agent_x import AgentX
from negmas_genius_agents.negotiators.anac.y2015.ares_party import AresParty
from negmas_genius_agents.negotiators.anac.y2015.atlas3 import Atlas3
from negmas_genius_agents.negotiators.anac.y2015.cuhk_agent2015 import CUHKAgent2015
from negmas_genius_agents.negotiators.anac.y2015.drage_knight import DrageKnight
from negmas_genius_agents.negotiators.anac.y2015.group2 import Y2015Group2
from negmas_genius_agents.negotiators.anac.y2015.jonny_black import JonnyBlack
from negmas_genius_agents.negotiators.anac.y2015.kawaii import Kawaii
from negmas_genius_agents.negotiators.anac.y2015.mean_bot import MeanBot
from negmas_genius_agents.negotiators.anac.y2015.mercury import Mercury
from negmas_genius_agents.negotiators.anac.y2015.p_negotiator import PNegotiator
from negmas_genius_agents.negotiators.anac.y2015.pars_agent import ParsAgent
from negmas_genius_agents.negotiators.anac.y2015.phoenix_party import PhoenixParty
from negmas_genius_agents.negotiators.anac.y2015.poker_face import PokerFace
from negmas_genius_agents.negotiators.anac.y2015.random_dance import RandomDance
from negmas_genius_agents.negotiators.anac.y2015.sengoku import SENGOKU
from negmas_genius_agents.negotiators.anac.y2015.xian_fa_agent import XianFaAgent

__all__ = [
    # Top 3
    "Atlas3",
    "ParsAgent",
    "RandomDance",
    # Other agents (alphabetical)
    "AgentBuyog",
    "AgentH",
    "AgentHP",
    "AgentNeo",
    "AgentW",
    "AgentX",
    "AresParty",
    "CUHKAgent2015",
    "DrageKnight",
    "JonnyBlack",
    "Kawaii",
    "MeanBot",
    "Mercury",
    "PhoenixParty",
    "PNegotiator",
    "PokerFace",
    "SENGOKU",
    "XianFaAgent",
    "Y2015Group2",
]
