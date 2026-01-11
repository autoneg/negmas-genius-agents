"""
ANAC 2017 negotiating agents reimplemented from Genius.

This module contains Python reimplementations of agents that competed
in the Automated Negotiating Agents Competition (ANAC) 2017.

1st Place: PonPokoAgent
2nd Place: CaduceusDC16
3rd Place: BetaOne

References:
    - https://ii.tudelft.nl/negotiation/node/12 (ANAC 2017)
    - Genius negotiation framework: https://ii.tudelft.nl/genius/
"""

from negmas_genius_agents.negotiators.anac.y2017.agent_f import AgentF
from negmas_genius_agents.negotiators.anac.y2017.agent_kn import AgentKN
from negmas_genius_agents.negotiators.anac.y2017.beta_one import BetaOne
from negmas_genius_agents.negotiators.anac.y2017.caduceus_dc16 import CaduceusDC16
from negmas_genius_agents.negotiators.anac.y2017.farma2017 import Farma2017
from negmas_genius_agents.negotiators.anac.y2017.gene_king import GeneKing
from negmas_genius_agents.negotiators.anac.y2017.gin import Gin
from negmas_genius_agents.negotiators.anac.y2017.group3 import Group3
from negmas_genius_agents.negotiators.anac.y2017.imitator import Imitator
from negmas_genius_agents.negotiators.anac.y2017.mad_agent import MadAgent
from negmas_genius_agents.negotiators.anac.y2017.mamenchis import Mamenchis
from negmas_genius_agents.negotiators.anac.y2017.mosa import Mosa
from negmas_genius_agents.negotiators.anac.y2017.pars_agent3 import ParsAgent3
from negmas_genius_agents.negotiators.anac.y2017.ponpoko_agent import PonPokoAgent
from negmas_genius_agents.negotiators.anac.y2017.rubick import Rubick
from negmas_genius_agents.negotiators.anac.y2017.simple_agent import SimpleAgent2017
from negmas_genius_agents.negotiators.anac.y2017.taxi_box import TaxiBox

__all__ = [
    # Top 3
    "PonPokoAgent",
    "CaduceusDC16",
    "BetaOne",
    # Other agents (alphabetical)
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
]
