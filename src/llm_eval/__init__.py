# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
LLM Evaluation Scaffolding for VGDL Games.

Subpackages:
    shared/              -- Config, harness, formatters, wrappers (used by both pipelines)
    generative_gameplay/ -- GameplayAgent, run.py (LLM plays games)
    human_replay/        -- ReplayAgent, run_replay.py (replay human data for feature extraction)

Usage:
    from src.llm_eval.generative_gameplay.agent import GameplayAgent
    from src.llm_eval.shared.config import Config
    from src.llm_eval.human_replay.replay_agent import ReplayAgent
"""

from importlib import import_module

__all__ = [
    "EventLogger",
    "ObservationFormatter",
    "Harness",
    "ResponseParser",
    "GameplayAgent",
    "run_game",
]


_LAZY_EXPORTS = {
    "EventLogger": ("src.llm_eval.shared.event_logger", "EventLogger"),
    "ObservationFormatter": (
        "src.llm_eval.shared.observation_formatter",
        "ObservationFormatter",
    ),
    "Harness": ("src.llm_eval.shared.harness", "Harness"),
    "ResponseParser": ("src.llm_eval.shared.response_parser", "ResponseParser"),
    "GameplayAgent": (
        "src.llm_eval.generative_gameplay.agent",
        "GameplayAgent",
    ),
    "run_game": ("src.llm_eval.generative_gameplay.agent", "run_game"),
}


def __getattr__(name: str):
    """Load gameplay dependencies only when a public facade is requested.

    This keeps lightweight analysis modules importable without installing the
    pygame/gym runtime, while preserving the package's existing public API.
    """
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
