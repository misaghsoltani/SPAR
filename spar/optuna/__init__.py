"""Public entrypoints for SPAR's Optuna integration."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.optuna.types import TemplateValue
    from spar.utils.config_utils.config_schema import (
        OptunaAnalyzeSPARConfig,
        OptunaReplaySPARConfig,
        OptunaStudySPARConfig,
    )
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession


def run_study(
    env: ABCEnvironment[ABCState], cfg: OptunaStudySPARConfig, tracking: WandbTrackingSession | None = None
) -> dict[str, TemplateValue]:
    """Run an Optuna study against SPAR workflows."""
    study_module: ModuleType = import_module("spar.optuna.study")
    return study_module.run_study(env, cfg, tracking)


def analyze_study(
    env: ABCEnvironment[ABCState], cfg: OptunaAnalyzeSPARConfig, tracking: WandbTrackingSession | None = None
) -> dict[str, TemplateValue]:
    """Analyze an existing Optuna study."""
    study_module: ModuleType = import_module("spar.optuna.study")
    return study_module.analyze_study(env, cfg, tracking)


def replay_trials(
    env: ABCEnvironment[ABCState], cfg: OptunaReplaySPARConfig, tracking: WandbTrackingSession | None = None
) -> dict[str, TemplateValue]:
    """Replay selected Optuna trials."""
    study_module: ModuleType = import_module("spar.optuna.study")
    return study_module.replay_trials(env, cfg, tracking)


__all__: list[str] = ["analyze_study", "replay_trials", "run_study"]
