"""Stage adapters that normalize SPAR stage outputs for Optuna."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

from spar.utils.config_utils.config_schema import MetricSpec

from .contracts import ObjectiveResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spar.utils.config_utils.config_schema import SPARConfig, WorkflowStep

    from .contracts import TrialContext
    from .types import PathValue, ScalarMetric


def _collect_scalar_metrics(
    mapping: dict[str, PathValue] | dict[int, PathValue] | dict[str | int, PathValue], *, prefix: str = ""
) -> dict[str, ScalarMetric]:
    out: dict[str, ScalarMetric] = {}
    for key, value in mapping.items():
        name: str = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_collect_scalar_metrics(value, prefix=name))
        elif isinstance(value, bool | int | float | str) or value is None:
            out[name] = value
    return out


def _phase_metrics_mapping(value: PathValue) -> dict[int, dict[str, PathValue]]:
    """Return a phase-indexed metrics mapping when the runtime value matches that shape."""
    if not isinstance(value, dict):
        return {}
    phases: dict[int, dict[str, PathValue]] = {}
    for phase_key, phase_metrics in value.items():
        if isinstance(phase_key, int) and isinstance(phase_metrics, dict):
            phases[phase_key] = {str(metric_key): metric_value for metric_key, metric_value in phase_metrics.items()}
    return phases


def _string_key_mapping(value: PathValue) -> dict[str, PathValue]:
    """Return a string-key mapping view when the runtime value is mapping-like."""
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _final_metric(metrics_by_phase: Mapping[int, Mapping[str, PathValue]], key: str) -> float | None:
    if not metrics_by_phase:
        return None
    phase_key: int = max(metrics_by_phase)
    phase_metrics: Mapping[str, PathValue] = metrics_by_phase[phase_key]
    value: PathValue | None = phase_metrics.get(key)
    if isinstance(value, list):
        if not value:
            return None
        last_value = value[-1]
        return float(last_value) if isinstance(last_value, bool | int | float) else None
    if isinstance(value, bool | int | float):
        return float(value)
    return None


def _phase_intermediates(
    metrics_by_phase: Mapping[int, Mapping[str, PathValue]], metric_key: str
) -> list[dict[str, ScalarMetric]]:
    intermediates: list[dict[str, ScalarMetric]] = []
    for phase_index, phase_metrics in sorted(metrics_by_phase.items()):
        value: float | None = _final_metric({phase_index: phase_metrics}, metric_key)
        if value is None:
            continue
        iteration_raw: PathValue | None = phase_metrics.get("current_itr")
        intermediates.append({
            "phase_index": phase_index,
            "iteration": iteration_raw if isinstance(iteration_raw, bool | int | float | str) else None,
            "metric_name": metric_key,
            "value": value,
        })
    return intermediates


def _config_to_mapping(config: SPARConfig) -> dict[str, PathValue]:
    mapped = asdict(config) if is_dataclass(config) else vars(config)
    return dict(mapped.items())


def _collect_path_artifacts(data: PathValue | Mapping[str, PathValue], *, prefix: str = "") -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            name: str = f"{prefix}.{key}" if prefix else str(key)
            artifacts.update(_collect_path_artifacts(value, prefix=name))
        return artifacts
    if isinstance(data, list):
        for index, value in enumerate(data):
            artifacts.update(_collect_path_artifacts(value, prefix=f"{prefix}[{index}]"))
        return artifacts
    if isinstance(data, str) and prefix.endswith(("_path", "_dir", ".results", ".output_dir")):
        artifacts[prefix] = data
    return artifacts


def _default_test_metric(metrics: dict[str, float | int | bool | str | None]) -> MetricSpec:
    for candidate in ("eq_bit_min_mean", "cosine_similarity_mean", "val_loss", "reconstruction_mse_mean"):
        if candidate in metrics:
            goal: str = "minimize" if "loss" in candidate or "error" in candidate or "mse" in candidate else "maximize"
            return MetricSpec(name=candidate, goal=goal)
    first_numeric: str | None = next(
        (key for key, value in metrics.items() if isinstance(value, bool | int | float)), None
    )
    if first_numeric is None:
        raise ValueError("No scalar metrics available to select a default Optuna objective")
    return MetricSpec(name=first_numeric, goal="maximize")


class GenericStageAdapter:
    """Fallback adapter used for stages without special handling."""

    def extract(
        self, *, step: WorkflowStep, config: SPARConfig, stage_result: dict[str, PathValue], trial_context: TrialContext
    ) -> ObjectiveResult:
        """Collect scalar metrics directly from dictionary-like stage outputs."""
        del step, trial_context
        metrics: dict[str, ScalarMetric] = _collect_scalar_metrics(stage_result)
        return ObjectiveResult(metrics=metrics, artifacts=_collect_path_artifacts(_config_to_mapping(config)))

    def default_metric(self, *, step: WorkflowStep, config: SPARConfig, result: ObjectiveResult) -> MetricSpec:
        """Infer a sensible default metric from the emitted scalar metrics."""
        del step, config
        return _default_test_metric(result.metrics)


class WorldModelAdapter(GenericStageAdapter):
    """Adapter for world-model training stages."""

    def extract(
        self, *, step: WorkflowStep, config: SPARConfig, stage_result: dict[str, PathValue], trial_context: TrialContext
    ) -> ObjectiveResult:
        """Extract validation and training losses from phased trainer output."""
        del step, trial_context
        metrics_raw: PathValue = stage_result.get("metrics", {})
        metrics_by_phase: dict[int, dict[str, PathValue]] = _phase_metrics_mapping(metrics_raw)
        metrics: dict[str, ScalarMetric] = {
            "val_loss": _final_metric(metrics_by_phase, "val_loss"),
            "train_loss": _final_metric(metrics_by_phase, "train_loss"),
        }
        return ObjectiveResult(
            objective=metrics["val_loss"] if isinstance(metrics["val_loss"], int | float) else None,
            metrics=metrics,
            artifacts=_collect_path_artifacts(_config_to_mapping(config)),
            intermediates=_phase_intermediates(metrics_by_phase, "val_loss"),
        )

    def default_metric(self, *, step: WorkflowStep, config: SPARConfig, result: ObjectiveResult) -> MetricSpec:
        """Use final validation loss as the default world-model objective."""
        del step, config, result
        return MetricSpec(name="val_loss", goal="minimize")


class AlignmentTrainAdapter(WorldModelAdapter):
    """Adapter for alignment-model training stages."""


class HeuristicAdapter(GenericStageAdapter):
    """Adapter for DQN heuristic training."""

    def extract(
        self, *, step: WorkflowStep, config: SPARConfig, stage_result: dict[str, PathValue], trial_context: TrialContext
    ) -> ObjectiveResult:
        """Extract scalar solve-rate metrics and model artifact paths."""
        del step, trial_context
        raw_metrics: PathValue = stage_result.get("metrics", {})
        metrics: dict[str, ScalarMetric] = (
            {
                str(key): value
                for key, value in raw_metrics.items()
                if isinstance(value, bool | int | float | str) or value is None
            }
            if isinstance(raw_metrics, dict)
            else {}
        )
        artifacts: dict[str, str] = {}
        artifacts_raw: PathValue = stage_result.get("artifacts", {})
        if isinstance(artifacts_raw, dict):
            artifacts.update({str(key): value for key, value in artifacts_raw.items() if isinstance(value, str)})
        artifacts.update(_collect_path_artifacts(_config_to_mapping(config)))
        return ObjectiveResult(metrics=metrics, artifacts=artifacts)

    def default_metric(self, *, step: WorkflowStep, config: SPARConfig, result: ObjectiveResult) -> MetricSpec:
        """Prefer held-out solve rate when available, else best-so-far solve rate."""
        del step, config
        metric_name: str = "per_solved_test" if "per_solved_test" in result.metrics else "per_solved_best"
        return MetricSpec(name=metric_name, goal="maximize")


class TestModelAdapter(GenericStageAdapter):
    """Adapter for model testing stages."""

    def extract(
        self, *, step: WorkflowStep, config: SPARConfig, stage_result: dict[str, PathValue], trial_context: TrialContext
    ) -> ObjectiveResult:
        """Extract overall evaluation metrics from model-testing stages."""
        del step, trial_context
        overall_raw: PathValue = stage_result.get("overall_metrics", {})
        overall: dict[str, PathValue] = _string_key_mapping(overall_raw)
        metrics: dict[str, ScalarMetric] = {
            key: value for key, value in overall.items() if isinstance(value, bool | int | float | str) or value is None
        }
        artifacts: dict[str, str] = _collect_path_artifacts(_config_to_mapping(config))
        return ObjectiveResult(metrics=metrics, artifacts=artifacts)

    def default_metric(self, *, step: WorkflowStep, config: SPARConfig, result: ObjectiveResult) -> MetricSpec:
        """Use the best available tester summary metric as the objective."""
        del step, config
        return _default_test_metric(result.metrics)


class SearchAdapter(GenericStageAdapter):
    """Adapter for search stages backed by ``results.json`` summaries."""

    def extract(
        self, *, step: WorkflowStep, config: SPARConfig, stage_result: dict[str, PathValue], trial_context: TrialContext
    ) -> ObjectiveResult:
        """Extract scalar search-summary metrics from saved or in-memory results."""
        del step, trial_context
        raw_results: dict[str, PathValue] = dict(stage_result) if stage_result else {}
        if not raw_results:
            results_dir: str | None = getattr(getattr(config, "search", None), "results_dir", None)
            if results_dir:
                results_path: Path = Path(results_dir) / "results.json"
                if results_path.exists():
                    loaded_results = orjson.loads(results_path.read_bytes())
                    if not isinstance(loaded_results, dict):
                        raise TypeError(
                            f"Expected search results payload to be a mapping, got {type(loaded_results).__name__}"
                        )
                    raw_results = {str(key): value for key, value in loaded_results.items()}
        overall: dict[str, PathValue] = {}
        summary: PathValue = raw_results.get("summary", {})
        if isinstance(summary, dict):
            overall_raw: PathValue = summary.get("overall", {})
            overall = _string_key_mapping(overall_raw)
        metrics: dict[str, ScalarMetric] = {
            key: value for key, value in overall.items() if isinstance(value, bool | int | float | str) or value is None
        }
        artifacts: dict[str, str] = _collect_path_artifacts(_config_to_mapping(config))
        search_cfg: dict[str, PathValue] | None = getattr(config, "search", None)
        results_dir = getattr(search_cfg, "results_dir", None)
        if isinstance(results_dir, str):
            artifacts["search.results_json"] = str(Path(results_dir) / "results.json")
        return ObjectiveResult(metrics=metrics, artifacts=artifacts)

    def default_metric(self, *, step: WorkflowStep, config: SPARConfig, result: ObjectiveResult) -> MetricSpec:
        """Prefer explicit search success rate when present."""
        del step, config
        if "search_success_rate" in result.metrics:
            return MetricSpec(name="search_success_rate", goal="maximize")
        return _default_test_metric(result.metrics)


def get_stage_adapter(stage_name: str) -> GenericStageAdapter:
    """Return the adapter associated with a runtime stage name."""
    if stage_name == "train_world_model":
        return WorldModelAdapter()
    if stage_name == "train_alignment_model":
        return AlignmentTrainAdapter()
    if stage_name == "train_heuristic":
        return HeuristicAdapter()
    if stage_name == "test_model":
        return TestModelAdapter()
    if stage_name in {"search_qstar", "search_gbfs", "search_ucs"}:
        return SearchAdapter()
    return GenericStageAdapter()
