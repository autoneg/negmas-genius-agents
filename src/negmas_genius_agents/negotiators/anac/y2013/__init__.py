"""
ANAC 2013 negotiating agents reimplemented from Genius.

This module contains Python reimplementations of agents that competed
in the Automated Negotiating Agents Competition (ANAC) 2013.

1st place: TheFawkes
2nd place: MetaAgent2013
3rd place: TMFAgent

Additional agents:
- AgentKF: Extension of AgentK series
- GAgent: General purpose adaptive negotiator
- InoxAgent: Robust agent with time-dependent strategies
- SlavaAgent: Concession-based agent with opponent modeling

References:
    - https://ii.tudelft.nl/negotiation/node/12 (ANAC 2013)
    - Genius negotiation framework: https://ii.tudelft.nl/genius/
"""

from negmas_genius_agents.negotiators.anac.y2013.the_fawkes import TheFawkes
from negmas_genius_agents.negotiators.anac.y2013.meta_agent import MetaAgent2013
from negmas_genius_agents.negotiators.anac.y2013.tmf_agent import TMFAgent
from negmas_genius_agents.negotiators.anac.y2013.agent_kf import AgentKF
from negmas_genius_agents.negotiators.anac.y2013.g_agent import GAgent
from negmas_genius_agents.negotiators.anac.y2013.inox_agent import InoxAgent
from negmas_genius_agents.negotiators.anac.y2013.slava_agent import SlavaAgent

__all__ = [
    # Top 3 agents
    "TheFawkes",
    "MetaAgent2013",
    "TMFAgent",
    # Additional agents
    "AgentKF",
    "GAgent",
    "InoxAgent",
    "SlavaAgent",
]
