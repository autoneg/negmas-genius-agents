"""Initialization module that registers all negotiators with the negmas registry.

This module is automatically imported when negmas_genius_agents is imported,
ensuring all negotiators are registered in the negmas registries if available.

The registration is done with the following tags:
- "genius-translated": Indicates these are Python translations of Java Genius agents
- "ai-generated": Indicates these implementations were created with AI assistance
- "anac": Indicates these are ANAC competition agents
- "anac-YYYY": Year-specific tag (e.g., "anac-2010", "anac-2019")

Note: Registration is skipped gracefully if the negmas registry module is not available.
"""

from __future__ import annotations

__all__: list[str] = []

# Flag to track if registration has been attempted
_registration_attempted = False


def _register_all() -> bool:
    """Register all negotiators with the negmas registry.

    Returns:
        True if registration was successful, False if registry is not available.
    """
    global _registration_attempted

    if _registration_attempted:
        return True

    _registration_attempted = True

    try:
        from negmas import negotiator_registry
    except ImportError:
        # negmas registry not available, skip registration silently
        return False

    # Common tags for all agents in this package
    base_tags = {"genius-translated", "ai-generated"}

    # Register basic time-dependent agents
    try:
        from negmas_genius_agents.negotiators.time_dependent import (
            TimeDependentAgent,
            TimeDependentAgentBoulware,
            TimeDependentAgentConceder,
            TimeDependentAgentLinear,
            TimeDependentAgentHardliner,
        )

        basic_agents = [
            TimeDependentAgent,
            TimeDependentAgentBoulware,
            TimeDependentAgentConceder,
            TimeDependentAgentLinear,
            TimeDependentAgentHardliner,
        ]
        for cls in basic_agents:
            if not negotiator_registry.is_registered(cls):
                negotiator_registry.register(
                    cls,
                    short_name=cls.__name__,
                    tags=base_tags | {"time-dependent"},
                    bilateral_only=True,
                )
    except ImportError:
        pass

    # Register ANAC 2010 agents
    try:
        from negmas_genius_agents.negotiators.anac.y2010 import (
            AgentK,
            Yushu,
            Nozomi,
            IAMhaggler,
            AgentFSEGA,
            AgentSmith,
            IAMcrazyHaggler,
        )

        anac_2010_agents = [
            AgentK,
            Yushu,
            Nozomi,
            IAMhaggler,
            AgentFSEGA,
            AgentSmith,
            IAMcrazyHaggler,
        ]
        for cls in anac_2010_agents:
            if not negotiator_registry.is_registered(cls):
                negotiator_registry.register(
                    cls,
                    short_name=cls.__name__,
                    tags=base_tags | {"anac", "anac-2010"},
                    anac_year=2010,
                    bilateral_only=True,
                )
    except ImportError:
        pass

    # Register ANAC 2011 agents
    try:
        from negmas_genius_agents.negotiators.anac.y2011 import (
            HardHeaded,
            Gahboninho,
            IAMhaggler2011,
            AgentK2,
            BramAgent,
            TheNegotiator,
        )

        anac_2011_agents = [
            HardHeaded,
            Gahboninho,
            IAMhaggler2011,
            AgentK2,
            BramAgent,
            TheNegotiator,
        ]
        for cls in anac_2011_agents:
            if not negotiator_registry.is_registered(cls):
                negotiator_registry.register(
                    cls,
                    short_name=cls.__name__,
                    tags=base_tags | {"anac", "anac-2011"},
                    anac_year=2011,
                    bilateral_only=True,
                )
    except ImportError:
        pass

    # Register ANAC 2012 agents
    try:
        from negmas_genius_agents.negotiators.anac.y2012 import (
            CUHKAgent,
            AgentLG,
            OMACAgent,
            TheNegotiatorReloaded,
            MetaAgent2012,
            IAMhaggler2012,
            AgentMR,
        )

        anac_2012_agents = [
            CUHKAgent,
            AgentLG,
            OMACAgent,
            TheNegotiatorReloaded,
            MetaAgent2012,
            IAMhaggler2012,
            AgentMR,
        ]
        for cls in anac_2012_agents:
            if not negotiator_registry.is_registered(cls):
                negotiator_registry.register(
                    cls,
                    short_name=cls.__name__,
                    tags=base_tags | {"anac", "anac-2012"},
                    anac_year=2012,
                    bilateral_only=True,
                )
    except ImportError:
        pass

    # Register ANAC 2013 agents
    try:
        from negmas_genius_agents.negotiators.anac.y2013 import (
            TheFawkes,
            MetaAgent2013,
            TMFAgent,
            AgentKF,
            GAgent,
            InoxAgent,
            SlavaAgent,
        )

        anac_2013_agents = [
            TheFawkes,
            MetaAgent2013,
            TMFAgent,
            AgentKF,
            GAgent,
            InoxAgent,
            SlavaAgent,
        ]
        for cls in anac_2013_agents:
            if not negotiator_registry.is_registered(cls):
                negotiator_registry.register(
                    cls,
                    short_name=cls.__name__,
                    tags=base_tags | {"anac", "anac-2013"},
                    anac_year=2013,
                    bilateral_only=True,
                )
    except ImportError:
        pass

    # Register ANAC 2014 agents
    try:
        from negmas_genius_agents.negotiators.anac.y2014 import (
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
        )

        anac_2014_agents = [
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
        ]
        for cls in anac_2014_agents:
            if not negotiator_registry.is_registered(cls):
                negotiator_registry.register(
                    cls,
                    short_name=cls.__name__,
                    tags=base_tags | {"anac", "anac-2014"},
                    anac_year=2014,
                    bilateral_only=True,
                )
    except ImportError:
        pass

    # Register ANAC 2015 agents
    try:
        from negmas_genius_agents.negotiators.anac.y2015 import (
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
        )

        anac_2015_agents = [
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
        ]
        for cls in anac_2015_agents:
            if not negotiator_registry.is_registered(cls):
                negotiator_registry.register(
                    cls,
                    short_name=cls.__name__,
                    tags=base_tags | {"anac", "anac-2015"},
                    anac_year=2015,
                    bilateral_only=True,
                )
    except ImportError:
        pass

    # Register ANAC 2016 agents
    try:
        from negmas_genius_agents.negotiators.anac.y2016 import (
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
        )

        anac_2016_agents = [
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
        ]
        for cls in anac_2016_agents:
            if not negotiator_registry.is_registered(cls):
                negotiator_registry.register(
                    cls,
                    short_name=cls.__name__,
                    tags=base_tags | {"anac", "anac-2016"},
                    anac_year=2016,
                    bilateral_only=True,
                )
    except ImportError:
        pass

    # Register ANAC 2017 agents
    try:
        from negmas_genius_agents.negotiators.anac.y2017 import (
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
        )

        anac_2017_agents = [
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
        ]
        for cls in anac_2017_agents:
            if not negotiator_registry.is_registered(cls):
                negotiator_registry.register(
                    cls,
                    short_name=cls.__name__,
                    tags=base_tags | {"anac", "anac-2017"},
                    anac_year=2017,
                    bilateral_only=True,
                )
    except ImportError:
        pass

    # Register ANAC 2018 agents
    try:
        from negmas_genius_agents.negotiators.anac.y2018 import (
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
        )

        anac_2018_agents = [
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
        ]
        for cls in anac_2018_agents:
            if not negotiator_registry.is_registered(cls):
                negotiator_registry.register(
                    cls,
                    short_name=cls.__name__,
                    tags=base_tags | {"anac", "anac-2018"},
                    anac_year=2018,
                    bilateral_only=True,
                )
    except ImportError:
        pass

    # Register ANAC 2019 agents
    try:
        from negmas_genius_agents.negotiators.anac.y2019 import (
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

        anac_2019_agents = [
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
        ]
        for cls in anac_2019_agents:
            if not negotiator_registry.is_registered(cls):
                negotiator_registry.register(
                    cls,
                    short_name=cls.__name__,
                    tags=base_tags | {"anac", "anac-2019"},
                    anac_year=2019,
                    bilateral_only=True,
                )
    except ImportError:
        pass

    return True


# Auto-register when this module is imported
_register_all()
