"""Runtime contracts for SPAR's Optuna integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from spar.utils.config_utils.config_schema import (
    AnalysisConfig,
    ConstraintSpec,
    MetricSpec,
    OptunaConfig,
    ParameterSpec,
    PrunerConfig,
    ReplayConfig,
    SamplerConfig,
    SPARConfig,
    StorageConfig,
    StudyConfig,
    WorkflowStep,
)

from .types import ScalarMetric

if TYPE_CHECKING:
    from pathlib import Path

    from .types import PathValue, ReporterPayload, SampledValue, TemplateValue


@dataclass(slots=True)
class ObjectiveResult:
    """Normalized outcome of one stage execution for Optuna."""

    objective: float | tuple[float, ...] | None = None
    metrics: dict[str, ScalarMetric] = field(default_factory=dict)
    constraints: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    intermediates: list[dict[str, ScalarMetric]] = field(default_factory=list)
    status: str = "completed"


@dataclass(slots=True)
class StepExecutionResult:
    """Runtime information captured for a workflow step."""

    step: WorkflowStep
    config: SPARConfig
    raw_result: dict[str, PathValue]
    objective_result: ObjectiveResult
    resolved_config_path: str | None = None


@dataclass(slots=True)
class TrialContext:
    """Shared context threaded across workflow steps for one Optuna trial."""

    study_name: str
    trial_number: int
    trial_dir: Path
    env_name: str
    sampled_parameters: dict[str, SampledValue] = field(default_factory=dict)
    sampled_values_by_step: dict[str, dict[str, SampledValue]] = field(default_factory=dict)
    step_results: dict[str, StepExecutionResult] = field(default_factory=dict)
    resolved_config_paths: dict[str, str] = field(default_factory=dict)

    def template_context(self) -> dict[str, TemplateValue]:
        """Return a nested context object for string templating."""
        return {
            "study_name": self.study_name,
            "trial_number": self.trial_number,
            "trial_dir": str(self.trial_dir),
            "env_name": self.env_name,
            "params": self.sampled_parameters,
            "sampled": self.sampled_values_by_step,
            "steps": {
                step_name: {
                    "artifacts": result.objective_result.artifacts,
                    "metrics": result.objective_result.metrics,
                    "constraints": result.objective_result.constraints,
                    "resolved_config_path": result.resolved_config_path,
                }
                for step_name, result in self.step_results.items()
            },
        }


class StageReporter(Protocol):
    """Minimal progress reporter interface used by training stages."""

    def __call__(self, payload: ReporterPayload) -> None:
        """Report sparse checkpoint metrics for pruning and study metadata."""
        ...


class StageAdapter(Protocol):
    """Protocol implemented by Optuna adapters for SPAR stages."""

    def extract(
        self, *, step: WorkflowStep, config: SPARConfig, stage_result: dict[str, PathValue], trial_context: TrialContext
    ) -> ObjectiveResult:
        """Normalize one stage result into Optuna metrics, artifacts, and constraints."""
        ...

    def default_metric(self, *, step: WorkflowStep, config: SPARConfig, result: ObjectiveResult) -> MetricSpec:
        """Return the stage's default optimization metric when none is configured."""
        ...


__all__: list[str] = [
    "AnalysisConfig",
    "ConstraintSpec",
    "MetricSpec",
    "ObjectiveResult",
    "OptunaConfig",
    "ParameterSpec",
    "PrunerConfig",
    "ReplayConfig",
    "SPARConfig",
    "SamplerConfig",
    "ScalarMetric",
    "StageAdapter",
    "StageReporter",
    "StepExecutionResult",
    "StorageConfig",
    "StudyConfig",
    "TrialContext",
    "WorkflowStep",
]
