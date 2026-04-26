"""
ANAC 2012 negotiating agents reimplemented from Genius.

This module contains Python reimplementations of agents that competed
in the Automated Negotiating Agents Competition (ANAC) 2012.

1st place: CUHKAgent
2nd place: AgentLG
3rd place: OMACAgent

Other notable agents:
- TheNegotiatorReloaded: Improved version of TheNegotiator with better adaptation
- MetaAgent2012: Meta-learning agent that combines multiple strategies
- IAMhaggler2012: Improved version of IAMhaggler with better opponent modeling
- AgentMR: Mixed-reality agent with adaptive bidding

References:
    - https://ii.tudelft.nl/negotiation/node/12 (ANAC 2012)
    - Genius negotiation framework: https://ii.tudelft.nl/genius/
"""

from negmas_genius_agents.negotiators.anac.y2012.cuhk_agent import CUHKAgent
from negmas_genius_agents.negotiators.anac.y2012.agent_lg import AgentLG
from negmas_genius_agents.negotiators.anac.y2012.omac_agent import OMACAgent
from negmas_genius_agents.negotiators.anac.y2012.the_negotiator_reloaded import (
    TheNegotiatorReloaded,
)
from negmas_genius_agents.negotiators.anac.y2012.meta_agent import MetaAgent2012
from negmas_genius_agents.negotiators.anac.y2012.iam_haggler2012 import IAMhaggler2012
from negmas_genius_agents.negotiators.anac.y2012.agent_mr import AgentMR

__all__ = [
    "CUHKAgent",
    "AgentLG",
    "OMACAgent",
    "TheNegotiatorReloaded",
    "MetaAgent2012",
    "IAMhaggler2012",
    "AgentMR",
]
