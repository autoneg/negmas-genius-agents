"""Register all Genius opponent models with the negmas component registry.

Imported automatically when ``negmas_genius_agents.models`` is imported.
Registration is skipped gracefully if the negmas registry is unavailable.

Models are registered with ``component_type="model"`` and tags reflecting their
classification in the opponent-modeling survey of Baarslag, Hendrikx, Hindriks &
Jonker (JAAMAS 30:849–898, 2016): all Genius BOA opponent models learn the
opponent's **preference profile** (§5.3); ``frequency`` models use §5.3.4
frequency analysis, ``bayesian`` models use §5.3.2 trace classification, and the
``oracle``/``baseline`` models are the reference models of §6.
"""

from __future__ import annotations

__all__: list[str] = []

_registration_attempted = False


def _register_all() -> bool:
    """Register all Genius opponent models. Returns True on success."""
    global _registration_attempted
    if _registration_attempted:
        return True
    _registration_attempted = True

    try:
        from negmas.registry import component_registry
    except ImportError:
        return False

    from negmas_genius_agents.models.frequency import HardHeadedFrequencyModel
    from negmas_genius_agents.models.frequency_extra import (
        SmithFrequencyModel,
        CUHKFrequencyModelV2,
        NashFrequencyModel,
        AgentXFrequencyModel,
    )
    from negmas_genius_agents.models.baselines import (
        PerfectModel,
        WorstModel,
        OppositeModel,
        UniformModel,
        DefaultModel,
    )

    base_tags = {"genius-translated", "ai-generated", "model"}
    freq_tags = {"preference-profile", "frequency", "learning"}
    models = [
        (HardHeadedFrequencyModel, freq_tags),
        (SmithFrequencyModel, freq_tags),
        (CUHKFrequencyModelV2, freq_tags),
        (NashFrequencyModel, freq_tags),
        (AgentXFrequencyModel, freq_tags),
        (OppositeModel, {"preference-profile", "zero-sum", "baseline"}),
        (UniformModel, {"baseline"}),
        (DefaultModel, {"baseline"}),
        (PerfectModel, {"oracle", "testing"}),
        (WorstModel, {"oracle", "testing"}),
    ]
    for cls, extra in models:
        if component_registry.is_registered(cls):
            continue
        try:
            component_registry.register(
                cls,
                short_name=cls.__name__,
                source="genius-agents",
                component_type="model",
                tags=base_tags | extra,
            )
        except TypeError:
            # Older registry API without source/component_type kwargs.
            component_registry.register(cls, short_name=cls.__name__, tags=base_tags | extra)
    return True


_register_all()
