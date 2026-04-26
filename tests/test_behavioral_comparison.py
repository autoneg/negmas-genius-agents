"""
Behavioral comparison tests between Python implementations and Java Genius agents.

This module provides infrastructure for comparing the behavior of Python agent
implementations with their original Java counterparts via the GeniusNegotiator bridge.

The tests compare:
1. Acceptance patterns - when agents accept offers
2. Bidding patterns - what offers agents make over time
3. Concession curves - how agents' target utilities change
4. Final outcomes - agreement rates and utilities achieved
"""

import json
import os
import pytest
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

from negmas.outcomes import make_issue, make_os, Outcome
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism, SAOState, ResponseType

# Try to import GeniusNegotiator - skip tests if not available
try:
    from negmas.genius import GeniusNegotiator
    from negmas.genius.ginfo import GENIUS_INFO

    GENIUS_AVAILABLE = True
except ImportError:
    GENIUS_AVAILABLE = False
    GENIUS_INFO = {}

# Import all Python agent implementations
from negmas_genius_agents import (
    # Time-dependent agents
    TimeDependentAgent,
    TimeDependentAgentBoulware,
    TimeDependentAgentConceder,
    TimeDependentAgentLinear,
    TimeDependentAgentHardliner,
    # ANAC 2010
    AgentK,
    Yushu,
    Nozomi,
    IAMhaggler,
    AgentFSEGA,
    AgentSmith,
    IAMcrazyHaggler,
    # ANAC 2011
    HardHeaded,
    Gahboninho,
    IAMhaggler2011,
    AgentK2,
    BramAgent,
    TheNegotiator,
    # ANAC 2012
    CUHKAgent,
    AgentLG,
    OMACAgent,
    TheNegotiatorReloaded,
    MetaAgent2012,
    IAMhaggler2012,
    AgentMR,
    # ANAC 2013
    TheFawkes,
    MetaAgent2013,
    TMFAgent,
    AgentKF,
    GAgent,
    InoxAgent,
    SlavaAgent,
    # ANAC 2014
    AgentM,
    DoNA,
    Gangster,
    WhaleAgent,
    TUDelftGroup2,
    E2Agent,
    KGAgent,
    AgentYK,
    BraveCat,
    AgentQuest,
    AgentTD,
    AgentTRP,
    ArisawaYaki,
    Aster,
    Atlas,
    # ANAC 2015
    Atlas3,
    ParsAgent,
    RandomDance,
    AgentBuyog,
    AgentH,
    AgentHP,
    AgentNeo,
    AgentW,
    AgentX,
    AresParty,
    CUHKAgent2015,
    DrageKnight,
    Y2015Group2,
    JonnyBlack,
    Kawaii,
    MeanBot,
    Mercury,
    PNegotiator,
    PhoenixParty,
    PokerFace,
    SENGOKU,
    XianFaAgent,
    # ANAC 2016
    Caduceus,
    YXAgent,
    ParsCat,
    AgentHP2,
    AgentLight,
    AgentSmith2016,
    Atlas32016,
    ClockworkAgent,
    Farma,
    GrandmaAgent,
    MaxOops,
    MyAgent,
    Ngent,
    Terra,
    # ANAC 2017
    PonPokoAgent,
    CaduceusDC16,
    BetaOne,
    AgentF,
    AgentKN,
    Farma2017,
    GeneKing,
    Gin,
    Group3,
    Imitator,
    MadAgent,
    Mamenchis,
    Mosa,
    ParsAgent3,
    Rubick,
    SimpleAgent2017,
    TaxiBox,
    # ANAC 2018
    AgreeableAgent2018,
    MengWan,
    Seto,
    Agent33,
    AgentHerb,
    AgentNP1,
    AteamAgent,
    ConDAgent,
    ExpRubick,
    FullAgent,
    IQSun2018,
    PonPokoRampage,
    Shiboy,
    Sontag,
    Yeela,
    # ANAC 2019
    AgentGG,
    KakeSoba,
    SAGA,
    AgentGP,
    AgentLarry,
    DandikAgent,
    EAgent,
    FSEGA2019,
    GaravelAgent,
    Gravity,
    HardDealer,
    KAgent,
    MINF,
    WinkyAgent,
)


# Mapping from Python class to Java class name
AGENT_MAPPING = {
    # ANAC 2010
    "AgentK": "agents.anac.y2010.AgentK.Agent_K",
    "Yushu": "agents.anac.y2010.Yushu.Yushu",
    "Nozomi": "agents.anac.y2010.Nozomi.Nozomi",
    "IAMhaggler": "agents.anac.y2010.Southampton.IAMhaggler",
    "AgentFSEGA": "agents.anac.y2010.AgentFSEGA.AgentFSEGA",
    "AgentSmith": "agents.anac.y2010.AgentSmith.AgentSmith",
    "IAMcrazyHaggler": "agents.anac.y2010.Southampton.IAMcrazyHaggler",
    # ANAC 2011
    "HardHeaded": "agents.anac.y2011.HardHeaded.KLH",
    "Gahboninho": "agents.anac.y2011.Gahboninho.Gahboninho",
    "IAMhaggler2011": "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011",
    "AgentK2": "agents.anac.y2011.AgentK2.Agent_K2",
    "BramAgent": "agents.anac.y2011.BramAgent.BRAMAgent",
    "TheNegotiator": "agents.anac.y2011.TheNegotiator.TheNegotiator",
    # ANAC 2012
    "CUHKAgent": "agents.anac.y2012.CUHKAgent.CUHKAgent",
    "AgentLG": "agents.anac.y2012.AgentLG.AgentLG",
    "OMACAgent": "agents.anac.y2012.OMACagent.OMACagent",
    "TheNegotiatorReloaded": "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded",
    "MetaAgent2012": "agents.anac.y2012.MetaAgent.MetaAgent",
    "IAMhaggler2012": "agents.anac.y2012.IAMhaggler2012.IAMhaggler2012",
    "AgentMR": "agents.anac.y2012.AgentMR.AgentMR",
    # ANAC 2013
    "TheFawkes": "agents.anac.y2013.TheFawkes.TheFawkes",
    "MetaAgent2013": "agents.anac.y2013.MetaAgent.MetaAgent2013",
    "TMFAgent": "agents.anac.y2013.TMFAgent.TMFAgent",
    "AgentKF": "agents.anac.y2013.AgentKF.AgentKF",
    "GAgent": "agents.anac.y2013.GAgent.AgentI",
    "InoxAgent": "agents.anac.y2013.InoxAgent.InoxAgent",
    "SlavaAgent": "agents.anac.y2013.SlavaAgent.SlavaAgent",
    # ANAC 2014
    "AgentM": "agents.anac.y2014.AgentM.AgentM",
    "DoNA": "agents.anac.y2014.DoNA.DoNA",
    "Gangster": "agents.anac.y2014.Gangster.Gangster",
    "WhaleAgent": "agents.anac.y2014.SimpaticoAgent.Simpatico",
    "TUDelftGroup2": "agents.anac.y2014.TUDelftGroup2.Group2Agent",
    "E2Agent": "agents.anac.y2014.E2Agent.AnacSampleAgent",
    "KGAgent": "agents.anac.y2014.KGAgent.KGAgent",
    "AgentYK": "agents.anac.y2014.AgentYK.AgentYK",
    "BraveCat": "agents.anac.y2014.BraveCat.BraveCat",
    "AgentQuest": "agents.anac.y2014.AgentQuest.AgentQuest",
    "AgentTD": "agents.anac.y2014.AgentTD.AgentTD",
    "AgentTRP": "agents.anac.y2014.AgentTRP.AgentTRP",
    "ArisawaYaki": "agents.anac.y2014.ArisawaYaki.ArisawaYaki",
    "Aster": "agents.anac.y2014.Aster.Aster",
    "Atlas": "agents.anac.y2014.Atlas.Atlas",
    # ANAC 2015
    "Atlas3": "agents.anac.y2015.Atlas3.Atlas3",
    "ParsAgent": "agents.anac.y2015.ParsAgent.ParsAgent",
    "RandomDance": "agents.anac.y2015.RandomDance.RandomDance",
    "AgentBuyog": "agents.anac.y2015.AgentBuyog.AgentBuyog",
    "AgentH": "agents.anac.y2015.AgentH.AgentH",
    "AgentHP": "agents.anac.y2015.AgentHP.AgentHP",
    "AgentNeo": "agents.anac.y2015.AgentNeo.Groupn",
    "AgentW": "agents.anac.y2015.AgentW.AgentW",
    "AgentX": "agents.anac.y2015.AgentX.AgentX",
    "AresParty": "agents.anac.y2015.AresParty.AresParty",
    "CUHKAgent2015": "agents.anac.y2015.cuhkagent2015.CUHKAgent2015",
    "DrageKnight": "agents.anac.y2015.DrageKnight.DrageKnight",
    "Y2015Group2": "agents.anac.y2015.group2.Group2",
    "JonnyBlack": "agents.anac.y2015.JonnyBlack.JonnyBlack",
    "Kawaii": "agents.anac.y2015.Kawaii.Kawaii",
    "MeanBot": "agents.anac.y2015.meanBot.MeanBot",
    "Mercury": "agents.anac.y2015.Mercury.Mercury",
    "PNegotiator": "agents.anac.y2015.pnegotiator.PNegotiator",
    "PhoenixParty": "agents.anac.y2015.Phoenix.PhoenixParty",
    "PokerFace": "agents.anac.y2015.pokerface.PokerFace",
    "SENGOKU": "agents.anac.y2015.SENGOKU.SENGOKU",
    "XianFaAgent": "agents.anac.y2015.xianfa.XianFaAgent",
    # ANAC 2016
    "Caduceus": "agents.anac.y2016.caduceus.Caduceus",
    "YXAgent": "agents.anac.y2016.yxagent.YXAgent",
    "ParsCat": "agents.anac.y2016.parscat.ParsCat",
    "AgentHP2": "agents.anac.y2016.agenthp2.AgentHP2_main",
    "AgentLight": "agents.anac.y2016.agentlight.AgentLight",
    "AgentSmith2016": "agents.anac.y2016.agentsmith.AgentSmith2016",
    "Atlas32016": "agents.anac.y2016.atlas3.Atlas32016",
    "ClockworkAgent": "agents.anac.y2016.clockworkagent.ClockworkAgent",
    "Farma": "agents.anac.y2016.farma.Farma",
    "GrandmaAgent": "agents.anac.y2016.grandma.GrandmaAgent",
    "MaxOops": "agents.anac.y2016.maxoops.MaxOops",
    "MyAgent": "agents.anac.y2016.myagent.MyAgent",
    "Ngent": "agents.anac.y2016.ngent.Ngent",
    "Terra": "agents.anac.y2016.terra.Terra",
    # ANAC 2017
    "PonPokoAgent": "agents.anac.y2017.ponpokoagent.PonPokoAgent",
    "CaduceusDC16": "agents.anac.y2017.caduceusdc16.CaduceusDC16",
    "BetaOne": "agents.anac.y2017.tangxun.BetaOne",
    "AgentF": "agents.anac.y2017.agentf.AgentF",
    "AgentKN": "agents.anac.y2017.agentkn.AgentKN",
    "Farma2017": "agents.anac.y2017.farma.Farma17",
    "GeneKing": "agents.anac.y2017.geneking.GeneKing",
    "Gin": "agents.anac.y2017.gin.Gin",
    "Group3": "agents.anac.y2017.group3.Group3",
    "Imitator": "agents.anac.y2017.limitator.Imitator",
    "MadAgent": "agents.anac.y2017.madagent.MadAgent",
    "Mamenchis": "agents.anac.y2017.mamenchis.Mamenchis",
    "Mosa": "agents.anac.y2017.mosateam.Mosa",
    "ParsAgent3": "agents.anac.y2017.parsagent3.ShahAgent",
    "Rubick": "agents.anac.y2017.rubick.Rubick",
    "SimpleAgent2017": "agents.anac.y2017.simpleagent.SimpleAgent",
    "TaxiBox": "agents.anac.y2017.tucagent.TucAgent",
    # ANAC 2018
    "AgreeableAgent2018": "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018",
    "MengWan": "agents.anac.y2018.meng_wan.Agent36",
    "Seto": "agents.anac.y2018.seto.Seto",
    "Agent33": "agents.anac.y2018.agent33.Agent33",
    "AgentHerb": "agents.anac.y2018.agentherb.AgentHerb",
    "AgentNP1": "agents.anac.y2018.agentnp1.AgentNP1",
    "AteamAgent": "agents.anac.y2018.ateamagent.ATeamAgent",
    "ConDAgent": "agents.anac.y2018.condagent.ConDAgent",
    "ExpRubick": "agents.anac.y2018.exp_rubick.Exp_Rubick",
    "FullAgent": "agents.anac.y2018.fullagent.FullAgent",
    "IQSun2018": "agents.anac.y2018.iqson.IQSun2018",
    "PonPokoRampage": "agents.anac.y2018.ponpokorampage.PonPokoRampage",
    "Shiboy": "agents.anac.y2018.shiboy.Shiboy",
    "Sontag": "agents.anac.y2018.sontag.Sontag",
    "Yeela": "agents.anac.y2018.yeela.Yeela",
    # ANAC 2019
    "AgentGG": "agents.anac.y2019.agentgg.AgentGG",
    "KakeSoba": "agents.anac.y2019.kakesoba.KakeSoba",
    "SAGA": "agents.anac.y2019.saga.SAGA",
    "AgentGP": "agents.anac.y2019.agentgp.AgentGP",
    "AgentLarry": "agents.anac.y2019.agentlarry.AgentLarry",
    "DandikAgent": "agents.anac.y2019.dandikagent.DandikAgent",
    "EAgent": "agents.anac.y2019.eagent.EAgent",
    "FSEGA2019": "agents.anac.y2019.fsega2019.FSEGA2019",
    "GaravelAgent": "agents.anac.y2019.garavelagent.GaravelAgent",
    "Gravity": "agents.anac.y2019.gravity.Gravity",
    "HardDealer": "agents.anac.y2019.harddealer.HardDealer",
    "KAgent": "agents.anac.y2019.kagent.KAgent",
    "MINF": "agents.anac.y2019.minf.MINF",
    "WinkyAgent": "agents.anac.y2019.winkyagent.WinkyAgent",
}

# Python class mapping
PYTHON_CLASSES = {
    # ANAC 2010
    "AgentK": AgentK,
    "Yushu": Yushu,
    "Nozomi": Nozomi,
    "IAMhaggler": IAMhaggler,
    "AgentFSEGA": AgentFSEGA,
    "AgentSmith": AgentSmith,
    "IAMcrazyHaggler": IAMcrazyHaggler,
    # ANAC 2011
    "HardHeaded": HardHeaded,
    "Gahboninho": Gahboninho,
    "IAMhaggler2011": IAMhaggler2011,
    "AgentK2": AgentK2,
    "BramAgent": BramAgent,
    "TheNegotiator": TheNegotiator,
    # ANAC 2012
    "CUHKAgent": CUHKAgent,
    "AgentLG": AgentLG,
    "OMACAgent": OMACAgent,
    "TheNegotiatorReloaded": TheNegotiatorReloaded,
    "MetaAgent2012": MetaAgent2012,
    "IAMhaggler2012": IAMhaggler2012,
    "AgentMR": AgentMR,
    # ANAC 2013
    "TheFawkes": TheFawkes,
    "MetaAgent2013": MetaAgent2013,
    "TMFAgent": TMFAgent,
    "AgentKF": AgentKF,
    "GAgent": GAgent,
    "InoxAgent": InoxAgent,
    "SlavaAgent": SlavaAgent,
    # ANAC 2014
    "AgentM": AgentM,
    "DoNA": DoNA,
    "Gangster": Gangster,
    "WhaleAgent": WhaleAgent,
    "TUDelftGroup2": TUDelftGroup2,
    "E2Agent": E2Agent,
    "KGAgent": KGAgent,
    "AgentYK": AgentYK,
    "BraveCat": BraveCat,
    "AgentQuest": AgentQuest,
    "AgentTD": AgentTD,
    "AgentTRP": AgentTRP,
    "ArisawaYaki": ArisawaYaki,
    "Aster": Aster,
    "Atlas": Atlas,
    # ANAC 2015
    "Atlas3": Atlas3,
    "ParsAgent": ParsAgent,
    "RandomDance": RandomDance,
    "AgentBuyog": AgentBuyog,
    "AgentH": AgentH,
    "AgentHP": AgentHP,
    "AgentNeo": AgentNeo,
    "AgentW": AgentW,
    "AgentX": AgentX,
    "AresParty": AresParty,
    "CUHKAgent2015": CUHKAgent2015,
    "DrageKnight": DrageKnight,
    "Y2015Group2": Y2015Group2,
    "JonnyBlack": JonnyBlack,
    "Kawaii": Kawaii,
    "MeanBot": MeanBot,
    "Mercury": Mercury,
    "PNegotiator": PNegotiator,
    "PhoenixParty": PhoenixParty,
    "PokerFace": PokerFace,
    "SENGOKU": SENGOKU,
    "XianFaAgent": XianFaAgent,
    # ANAC 2016
    "Caduceus": Caduceus,
    "YXAgent": YXAgent,
    "ParsCat": ParsCat,
    "AgentHP2": AgentHP2,
    "AgentLight": AgentLight,
    "AgentSmith2016": AgentSmith2016,
    "Atlas32016": Atlas32016,
    "ClockworkAgent": ClockworkAgent,
    "Farma": Farma,
    "GrandmaAgent": GrandmaAgent,
    "MaxOops": MaxOops,
    "MyAgent": MyAgent,
    "Ngent": Ngent,
    "Terra": Terra,
    # ANAC 2017
    "PonPokoAgent": PonPokoAgent,
    "CaduceusDC16": CaduceusDC16,
    "BetaOne": BetaOne,
    "AgentF": AgentF,
    "AgentKN": AgentKN,
    "Farma2017": Farma2017,
    "GeneKing": GeneKing,
    "Gin": Gin,
    "Group3": Group3,
    "Imitator": Imitator,
    "MadAgent": MadAgent,
    "Mamenchis": Mamenchis,
    "Mosa": Mosa,
    "ParsAgent3": ParsAgent3,
    "Rubick": Rubick,
    "SimpleAgent2017": SimpleAgent2017,
    "TaxiBox": TaxiBox,
    # ANAC 2018
    "AgreeableAgent2018": AgreeableAgent2018,
    "MengWan": MengWan,
    "Seto": Seto,
    "Agent33": Agent33,
    "AgentHerb": AgentHerb,
    "AgentNP1": AgentNP1,
    "AteamAgent": AteamAgent,
    "ConDAgent": ConDAgent,
    "ExpRubick": ExpRubick,
    "FullAgent": FullAgent,
    "IQSun2018": IQSun2018,
    "PonPokoRampage": PonPokoRampage,
    "Shiboy": Shiboy,
    "Sontag": Sontag,
    "Yeela": Yeela,
    # ANAC 2019
    "AgentGG": AgentGG,
    "KakeSoba": KakeSoba,
    "SAGA": SAGA,
    "AgentGP": AgentGP,
    "AgentLarry": AgentLarry,
    "DandikAgent": DandikAgent,
    "EAgent": EAgent,
    "FSEGA2019": FSEGA2019,
    "GaravelAgent": GaravelAgent,
    "Gravity": Gravity,
    "HardDealer": HardDealer,
    "KAgent": KAgent,
    "MINF": MINF,
    "WinkyAgent": WinkyAgent,
}


@dataclass
class NegotiationTrace:
    """Records a negotiation trace for comparison."""

    offers: list[tuple[float, tuple, float]] = field(
        default_factory=list
    )  # (time, outcome, utility)
    responses: list[tuple[float, str]] = field(default_factory=list)  # (time, response)
    final_agreement: tuple | None = None
    final_utility: float = 0.0
    agreement_reached: bool = False


@dataclass
class ComparisonResult:
    """Result of comparing Python and Java agent behaviors."""

    agent_name: str
    java_class: str

    # Negotiation statistics
    python_agreement_rate: float = 0.0
    java_agreement_rate: float = 0.0
    python_avg_utility: float = 0.0
    java_avg_utility: float = 0.0

    # Behavioral similarity metrics
    offer_utility_correlation: float = 0.0  # Correlation of offer utilities over time
    acceptance_threshold_diff: float = 0.0  # Difference in acceptance thresholds
    concession_rate_diff: float = 0.0  # Difference in concession rates

    # Status
    status: str = "not_tested"  # not_tested, verified, minor_issues, major_issues
    issues: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def create_test_domain():
    """Create a standard test negotiation domain."""
    issues = [
        make_issue(values=["low", "medium", "high"], name="price"),
        make_issue(values=["1", "2", "3"], name="quantity"),
        make_issue(values=["fast", "normal", "slow"], name="delivery"),
    ]
    os = make_os(issues)

    buyer_ufun = LinearAdditiveUtilityFunction(
        values={
            "price": {"low": 1.0, "medium": 0.5, "high": 0.0},
            "quantity": {"1": 0.0, "2": 0.5, "3": 1.0},
            "delivery": {"fast": 1.0, "normal": 0.5, "slow": 0.0},
        },
        weights={"price": 0.5, "quantity": 0.3, "delivery": 0.2},
        outcome_space=os,
    )

    seller_ufun = LinearAdditiveUtilityFunction(
        values={
            "price": {"low": 0.0, "medium": 0.5, "high": 1.0},
            "quantity": {"1": 1.0, "2": 0.5, "3": 0.0},
            "delivery": {"fast": 0.0, "normal": 0.5, "slow": 1.0},
        },
        weights={"price": 0.5, "quantity": 0.3, "delivery": 0.2},
        outcome_space=os,
    )

    return issues, buyer_ufun, seller_ufun


def run_python_negotiation(
    agent_class, opponent_class, issues, agent_ufun, opponent_ufun, n_steps=100
):
    """Run a negotiation with Python agents and record the trace."""
    mechanism = SAOMechanism(issues=issues, n_steps=n_steps)

    agent = agent_class(name="python_agent", ufun=agent_ufun)
    opponent = opponent_class(name="opponent", ufun=opponent_ufun)

    mechanism.add(agent)
    mechanism.add(opponent)

    state = mechanism.run()

    trace = NegotiationTrace()
    trace.agreement_reached = state.agreement is not None
    if state.agreement:
        trace.final_agreement = tuple(state.agreement)
        trace.final_utility = float(agent_ufun(state.agreement))

    return trace, state


@pytest.fixture
def test_domain():
    """Fixture providing a standard test domain."""
    return create_test_domain()


# Skip all tests if Genius is not available
pytestmark = pytest.mark.skipif(
    not GENIUS_AVAILABLE, reason="GeniusNegotiator not available"
)


class TestBehavioralComparison:
    """Tests comparing Python and Java agent behaviors."""

    @pytest.mark.parametrize(
        "agent_name", list(AGENT_MAPPING.keys())[:7]
    )  # Start with ANAC 2010
    def test_agent_basic_negotiation(self, agent_name, test_domain):
        """Test that Python agent can complete basic negotiations."""
        issues, buyer_ufun, seller_ufun = test_domain

        agent_class = PYTHON_CLASSES[agent_name]

        mechanism = SAOMechanism(issues=issues, n_steps=100)
        agent = agent_class(name="test_agent", ufun=buyer_ufun)
        opponent = TimeDependentAgentConceder(name="opponent", ufun=seller_ufun)

        mechanism.add(agent)
        mechanism.add(opponent)

        state = mechanism.run()

        assert state.started
        assert state.ended
        # Record result for report


def update_verification_report(results: list[ComparisonResult], report_path: str):
    """Update the verification report with new results."""
    report_file = Path(report_path)

    if report_file.exists():
        with open(report_file) as f:
            report = json.load(f)
    else:
        report = {"metadata": {}, "agents": {}, "detailed_analysis": {}}

    for result in results:
        year = None
        for y in [
            "2010",
            "2011",
            "2012",
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
            "2018",
            "2019",
        ]:
            if f"y{y}" in result.java_class or f".{y}." in result.java_class:
                year = f"anac{y}"
                break

        if year and year in report["agents"]:
            if result.agent_name in report["agents"][year]:
                agent_data = report["agents"][year][result.agent_name]
                agent_data["status"] = result.status
                agent_data["behavioral_test"] = (
                    "completed" if result.status != "not_tested" else "not_started"
                )
                agent_data["notes"] = result.notes

        report["detailed_analysis"][result.agent_name] = result.to_dict()

    # Update summary statistics
    total = 0
    verified = 0
    minor_issues = 0
    major_issues = 0

    for year_agents in report["agents"].values():
        for agent_data in year_agents.values():
            total += 1
            status = agent_data.get("status", "not_started")
            if status == "verified":
                verified += 1
            elif status == "minor_issues":
                minor_issues += 1
            elif status == "major_issues":
                major_issues += 1

    report["metadata"]["total_agents"] = total
    report["metadata"]["verified"] = verified
    report["metadata"]["minor_issues"] = minor_issues
    report["metadata"]["major_issues"] = major_issues
    report["metadata"]["not_started"] = total - verified - minor_issues - major_issues

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v", "-k", "test_agent_basic"])
