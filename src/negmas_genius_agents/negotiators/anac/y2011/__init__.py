"""
ANAC 2011 negotiating agents reimplemented from Genius.

This module contains Python reimplementations of agents that competed
in the Automated Negotiating Agents Competition (ANAC) 2011.

1st place: HardHeaded (KLH)
2nd place: Gahboninho
3rd place: IAMhaggler2011

Additional agents:
- AgentK2: Extension of AgentK from 2010
- BramAgent: Adaptive negotiating agent
- TheNegotiator: Time-dependent negotiator with opponent modeling

References:
    - https://ii.tudelft.nl/negotiation/node/12 (ANAC 2011)
    - Genius negotiation framework: https://ii.tudelft.nl/genius/
"""

from negmas_genius_agents.negotiators.anac.y2011.hard_headed import HardHeaded
from negmas_genius_agents.negotiators.anac.y2011.gahboninho import Gahboninho
from negmas_genius_agents.negotiators.anac.y2011.iam_haggler import IAMhaggler2011
from negmas_genius_agents.negotiators.anac.y2011.agent_k2 import AgentK2
from negmas_genius_agents.negotiators.anac.y2011.bram_agent import BramAgent
from negmas_genius_agents.negotiators.anac.y2011.the_negotiator import TheNegotiator

__all__ = [
    # Top 3 agents
    "HardHeaded",
    "Gahboninho",
    "IAMhaggler2011",
    # Additional agents
    "AgentK2",
    "BramAgent",
    "TheNegotiator",
]
