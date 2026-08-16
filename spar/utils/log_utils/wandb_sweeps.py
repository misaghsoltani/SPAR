"""Build, launch, and summarize W&B hyperparameter sweeps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import is_dataclass
import logging
import operator
from pathlib import Path
import textwrap
from typing import TYPE_CHECKING, Protocol, TypedDict
import warnings

from spar.utils.log_utils.wandb_logger import configure_sweep

if TYPE_CHECKING:
    from collections.abc import Iterable, MutableMapping
    from logging import Logger
    from typing import Literal, TypeAlias

    from omegaconf import DictConfig

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.utils.config_utils.config_schema import CreateSweepSPARConfig, SweepConfig, WandbConfig
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession


UnsupportedFieldAttributeWarning: type[Warning] | None
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning as _UnsupportedFieldAttributeWarningType
except ImportError:
    UnsupportedFieldAttributeWarning = None
else:
    UnsupportedFieldAttributeWarning = _UnsupportedFieldAttributeWarningType

with warnings.catch_warnings():
    if UnsupportedFieldAttributeWarning is not None:
        warnings.simplefilter("ignore", UnsupportedFieldAttributeWarning)
    warnings.filterwarnings(
        "ignore",
        message=r"The 'repr' attribute with value .* was provided to the `Field\(\)` function, which has no effect",
    )
    import wandb


log: Logger = logging.getLogger(__name__)

JSONType: TypeAlias = str | int | float | bool | list["JSONType"] | dict[str, "JSONType"] | None
ParamSpec: TypeAlias = dict[str, JSONType]
MetricType: TypeAlias = dict[str, JSONType] | list[dict[str, JSONType]]
SweepValue: TypeAlias = JSONType | MetricType | Mapping[str, ParamSpec]
ConfigValue: TypeAlias = str | int | float | bool | list["ConfigValue"] | dict[str, "ConfigValue"] | None
MetricScalar: TypeAlias = str | int | float | bool | None
if TYPE_CHECKING:
    ConfigMappingValue: TypeAlias = ConfigValue | DictConfig | CreateSweepSPARConfig | SweepConfig | WandbConfig
    ConfigInput: TypeAlias = DictConfig | Mapping[str, ConfigMappingValue] | CreateSweepSPARConfig
else:
    ConfigMappingValue: TypeAlias = ConfigValue
    ConfigInput: TypeAlias = Mapping[str, ConfigMappingValue]


class SweepRunLike(Protocol):
    """Structural run interface required by sweep helpers."""

    summary: Mapping[str, MetricScalar]
    state: str

    def logged_artifacts(self) -> Iterable[wandb.Artifact]:
        """Return artifacts logged for this run."""
        raise NotImplementedError


class SweepLike(Protocol):
    """Structural sweep interface required by sweep helpers."""

    runs: Iterable[SweepRunLike]


class DistributionSpec(TypedDict, total=False):
    """Specification for a continuous distribution parameter.

    Fields:
        distribution: Name of the distribution (e.g. 'log_uniform').
        min: Minimum value for the distribution.
        max: Maximum value for the distribution.
    """

    distribution: Literal["log_uniform", "uniform", "qlog_uniform"]
    min: float
    max: float


class ValuesSpec(TypedDict):
    """Specification for a discrete set of allowed values.

    Fields:
        values: Allowed values for the parameter (ints, floats or strings).
    """

    values: list[int | float | str]


_HEURISTIC_DEFAULT_PARAMS: dict[str, ParamSpec] = {
    "train.learning_rate": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-2},
    "train.weight_decay": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-3},
    "train.batch_size": {"values": [64, 128, 256, 512]},
    "model.heuristic.hidden_size": {"values": [64, 128, 256, 512]},
    "model.heuristic.num_layers": {"values": [2, 3, 4, 5]},
    "train.optimizer": {"values": ["adam", "sgd", "adamw"]},
}
_WORLD_MODEL_DEFAULT_PARAMS: dict[str, ParamSpec] = {
    "train.learning_rate": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-2},
    "train.weight_decay": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-3},
    "train.batch_size": {"values": [32, 64, 128, 256]},
    "model.world_model.latent_size": {"values": [64, 128, 256]},
    "model.world_model.hidden_size": {"values": [128, 256, 512]},
    "model.world_model.activation": {"values": ["relu", "leaky_relu", "elu", "gelu"]},
}


def _cfg_to_mapping(cfg: ConfigInput | ConfigMappingValue | None) -> Mapping[str, ConfigMappingValue]:
    """Convert supported config fragments into a mapping view."""
    if isinstance(cfg, Mapping):
        return cfg
    if is_dataclass(cfg):
        dataclass_values: dict[str, ConfigMappingValue] = dict(vars(cfg))
        return dataclass_values
    return {}


def _cfg_str(mapping: Mapping[str, ConfigMappingValue], key: str, default: str) -> str:
    value: ConfigMappingValue = mapping.get(key)
    return value if isinstance(value, str) else default


def _cfg_optional_str(mapping: Mapping[str, ConfigMappingValue], key: str) -> str | None:
    value: ConfigMappingValue = mapping.get(key)
    return value if isinstance(value, str) else None


def _cfg_int(mapping: Mapping[str, ConfigMappingValue], key: str, default: int) -> int:
    value: ConfigMappingValue = mapping.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _metric_scalar_to_float(value: MetricScalar) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _score_sweep_runs(runs: Iterable[SweepRunLike], metric_name: str) -> list[tuple[float, SweepRunLike]]:
    scored_runs: list[tuple[float, SweepRunLike]] = []
    for run in runs:
        metric_value: float | None = _metric_scalar_to_float(run.summary.get(metric_name))
        if metric_value is not None:
            scored_runs.append((metric_value, run))

    return scored_runs


def _artifacts_from_scored_runs(
    scored_runs: list[tuple[float, SweepRunLike]], top_k: int, artifact_type: str
) -> list[wandb.Artifact]:
    artifacts: list[wandb.Artifact] = []
    for _, run in scored_runs[:top_k]:
        run_artifacts: list[wandb.Artifact] = [
            a for a in run.logged_artifacts() if getattr(a, "type", None) == artifact_type
        ]
        artifacts.extend(run_artifacts)

    return artifacts[:top_k]


def _add_ensemble_references(ensemble: wandb.Artifact, artifacts: list[wandb.Artifact]) -> None:
    for i, artifact in enumerate(artifacts):
        src_candidate: str | None = (
            getattr(artifact, "source_qualified_name", None)
            or getattr(artifact, "id", None)
            or getattr(artifact, "name", None)
        )
        if isinstance(src_candidate, str):
            ensemble.add_reference(src_candidate, name=f"model_{i}")


def _sweep_config_with_run_limits(
    config: dict[str, JSONType], max_concurrent_runs: int, auto_stop_criteria: dict[str, JSONType] | None
) -> dict[str, JSONType]:
    launch_config: dict[str, JSONType] = config.copy()
    launch_config["controller"] = {"type": "cloud", "max_concurrent_runs": max_concurrent_runs}

    if auto_stop_criteria:
        launch_config["early_terminate"] = auto_stop_criteria

    return launch_config


def _get_sweep_runs(project: str, entity: str | None, sweep_id: str) -> list[SweepRunLike]:
    api: wandb.Api = wandb.Api()
    sweep_path: str = f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}"
    sweep: SweepLike = api.sweep(sweep_path)
    return list(sweep.runs)


def _metric_values(runs: Iterable[SweepRunLike], metric_name: str) -> list[float]:
    metrics: list[float] = []
    for run in runs:
        metric_value: float | None = _metric_scalar_to_float(run.summary.get(metric_name))
        if metric_value is not None:
            metrics.append(metric_value)

    return metrics


def _sweep_metric_summary(
    runs: list[SweepRunLike], completed_runs: list[SweepRunLike], metrics: list[float]
) -> dict[str, JSONType]:
    return {
        "total_runs": len(runs),
        "completed_runs": len(completed_runs),
        "best_metric": min(metrics),
        "worst_metric": max(metrics),
        "mean_metric": sum(metrics) / len(metrics),
        "improvement_rate": (max(metrics) - min(metrics)) / max(metrics) if max(metrics) > 0 else 0,
    }


def _add_recent_convergence_fields(summary: dict[str, JSONType], metrics: list[float]) -> None:
    if len(metrics) < 5:
        return

    recent_metrics: list[float] = metrics[-5:]
    improvement: float = (max(recent_metrics) - min(recent_metrics)) / max(recent_metrics)
    summary["recent_improvement"] = improvement
    summary["converged"] = improvement < 0.01


def _summarize_sweep_metrics(project: str, entity: str | None, sweep_id: str) -> dict[str, JSONType]:
    if not hasattr(wandb, "Api"):
        log.warning("W&B API not available")
        return {}

    runs: list[SweepRunLike] = _get_sweep_runs(project, entity, sweep_id)
    completed_runs: list[SweepRunLike] = [r for r in runs if r.state == "finished"]

    if not completed_runs:
        return {"status": "no_completed_runs", "total_runs": len(runs)}

    metrics: list[float] = _metric_values(completed_runs, "val_loss")
    if not metrics:
        return {"status": "no_metrics", "completed_runs": len(completed_runs)}

    summary: dict[str, JSONType] = _sweep_metric_summary(runs, completed_runs, metrics)
    _add_recent_convergence_fields(summary, metrics)
    return summary


def build_sweep_config(
    *, method: str, metric: MetricType, parameters: Mapping[str, ParamSpec], base_cfg: Mapping[str, JSONType] | None
) -> dict[str, JSONType]:
    """Merge params + base_cfg into a JSON-compatible sweep configuration.

    All values returned are plain dict/list/primitive JSON-compatible types so
    they can be passed directly to W&B APIs.
    """
    # Build a concrete dict for parameters (values are JSON-compatible)
    merged_parameters: MutableMapping[str, JSONType]
    if base_cfg:
        merged_parameters = dict(base_cfg)
        # Copy parameter mappings into plain dictionaries.
        merged_parameters.update({k: dict(v) for k, v in parameters.items()})
    else:
        # dict(parameters) yields a mapping of key -> ParamSpec (which is JSON-compatible)
        merged_parameters = dict(parameters)

    # Copy metric mappings and lists into JSON-compatible containers.
    metric_value: JSONType
    if isinstance(metric, list):
        # metric is a list of mapping-like dicts -> convert each to a plain dict
        metric_json: list[JSONType] = [dict(m) for m in metric]
        metric_value = metric_json
    else:
        # metric is a single mapping -> convert to a plain dict
        metric_value = dict(metric)

    return {"method": method, "metric": metric_value, "parameters": merged_parameters}


def create_sweep_config_for_world_model(
    *,
    method: str = "bayes",
    metric: Mapping[str, JSONType] | None = None,
    parameters: Mapping[str, ParamSpec] | None = None,
    base_cfg: Mapping[str, JSONType] | None = None,
) -> dict[str, JSONType]:
    """Create a sweep configuration for world model training (wrapper)."""
    return build_sweep_config(
        method=method,
        metric=dict(metric or {"name": "val_loss", "goal": "minimize"}),
        parameters=parameters or _WORLD_MODEL_DEFAULT_PARAMS,
        base_cfg=base_cfg,
    )


def create_sweep_config_for_heuristic(
    *,
    method: str = "bayes",
    metric: Mapping[str, JSONType] | None = None,
    parameters: Mapping[str, ParamSpec] | None = None,
    base_cfg: Mapping[str, JSONType] | None = None,
) -> dict[str, JSONType]:
    """Create a sweep configuration for heuristic training.

    Args:
        method: Optimization method (bayes, grid, random)
        metric: Metric configuration dict with 'name' and 'goal'
        parameters: Parameter space configuration
        base_cfg: Base configuration to merge with parameters

    Returns:
        Dictionary containing the sweep configuration
    """
    return build_sweep_config(
        method=method,
        metric=dict(metric or {"name": "val_loss", "goal": "minimize"}),
        parameters=parameters or _HEURISTIC_DEFAULT_PARAMS,
        base_cfg=base_cfg,
    )


def create_sweep_agent_script(
    sweep_id: str, train_script_path: str = "spar/cli.py", output_path: str | None = None, count: int = 10
) -> str:
    """Create an agent script for running a W&B sweep.

    Args:
        sweep_id: W&B sweep ID
        train_script_path: Path to the training script
        output_path: Path to save the agent script. If None, return the script content.
        count: Number of runs for the agent to execute

    Returns:
        Path to the agent script if output_path is provided, else the script content
    """
    script_template = f'''
"""Auto-generated W&B sweep agent script for SPAR."""

from pathlib import Path
import subprocess
import sys

import wandb

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _train_fn() -> None:
    cfg = wandb.config
    cmd = ["python", "{train_script_path}"]
    # Add configuration parameters as command line arguments
    for k, v in cfg.items():
        if k != "wandb" and not k.startswith("_"):
            prefix = "+" if "." not in k else ""
            cmd.append(f"{{prefix}}{{k}}={{v}}")

    proc = subprocess.Popen( cmd, text=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.stdout
    for ln in proc.stdout:
        print(ln, end="")

    proc.wait()
    if proc.returncode:
        raise RuntimeError(f"Training failed (exit={{proc.returncode}})")


if __name__ == "__main__":
    wandb.login()
    wandb.agent("{sweep_id}", function=_train_fn, count={count})

'''

    script = textwrap.dedent(
        script_template.format(train_script_path=train_script_path, sweep_id=sweep_id, count=count)
    )

    # Save the script if output path is provided
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(script, encoding="utf-8")
        out.chmod(0o755)
        return str(out)

    return script


def launch_sweep_from_config(
    cfg: ConfigInput,
    *,
    method: str = "bayes",
    sweep_type: str = "heuristic",
    metric_name: str = "val_loss",
    metric_goal: str = "minimize",
    count: int = 10,
    output_script_path: str | None = None,
) -> str:
    """Launch a W&B sweep from a Hydra config.

    Args:
        cfg: Hydra configuration
        method: Sweep method (bayes, grid, random)
        sweep_type: Type of sweep (heuristic, world_model)
        metric_name: Name of the metric to optimize
        metric_goal: Goal for the metric (minimize, maximize)
        count: Number of runs to execute
        output_script_path: Path to save the agent script

    Returns:
        Sweep ID if successful, empty string otherwise
    """
    metric_name = metric_name.strip()
    if not metric_name:
        log.error("Sweep metric_name cannot be empty.")
        return ""
    if metric_goal not in {"minimize", "maximize"}:
        log.error(f"Sweep metric_goal must be one of {{'minimize', 'maximize'}}, got: {metric_goal}")
        return ""

    metric: dict[str, str] = {"name": metric_name, "goal": metric_goal}
    cfg_map: Mapping[str, ConfigMappingValue] = _cfg_to_mapping(cfg)
    wb_cfg_map: Mapping[str, ConfigMappingValue] = _cfg_to_mapping(cfg_map.get("wandb", {}))
    project: str = _cfg_str(wb_cfg_map, "project", "spar")
    entity: str | None = _cfg_optional_str(wb_cfg_map, "entity")
    sweep_cfg: dict[str, JSONType]
    if sweep_type.lower() == "heuristic":
        sweep_cfg = create_sweep_config_for_heuristic(method=method, metric=metric)
    elif sweep_type.lower() == "world_model":
        sweep_cfg = create_sweep_config_for_world_model(method=method, metric=metric)
    else:
        log.error(f"Unknown sweep type: {sweep_type}")
        return ""

    sweep_id: str = configure_sweep(sweep_cfg, project_name=project, entity_name=entity)
    if not sweep_id:
        log.error("Failed to create W&B sweep")
        return ""

    # Create the agent script if requested
    if output_script_path:
        script: str = create_sweep_agent_script(
            sweep_id=sweep_id, train_script_path="spar/cli.py", output_path=output_script_path, count=count
        )
        log.info(f"Sweep agent script written to {script}")
        log.info(f"Run with: python {script}")

    return sweep_id


def create_wandb_sweep(cfg: ConfigInput) -> str:
    """Create a W&B sweep for hyperparameter optimization.

    Args:
        cfg: Configuration object with sweep settings.

    Returns:
        The created sweep ID, or an empty string when creation fails.

    """
    cfg_map: Mapping[str, ConfigMappingValue] = _cfg_to_mapping(cfg)
    sweep_cfg_raw: ConfigMappingValue = cfg_map.get("sweep")
    if sweep_cfg_raw is None:
        print("Error: Sweep configuration not found in the config.")
        return ""

    # Extract sweep configuration
    sweep_cfg: Mapping[str, ConfigMappingValue] = _cfg_to_mapping(sweep_cfg_raw)
    sweep_id: str = launch_sweep_from_config(
        cfg=cfg,
        method=_cfg_str(sweep_cfg, "method", "bayes"),
        sweep_type=_cfg_str(sweep_cfg, "type", "heuristic"),
        metric_name=_cfg_str(sweep_cfg, "metric_name", "val_loss"),
        metric_goal=_cfg_str(sweep_cfg, "metric_goal", "minimize"),
        count=_cfg_int(sweep_cfg, "count", 10),
        output_script_path=None,  # we create the script ourselves
    )

    if bool(sweep_cfg.get("create_agent_script", True)) and sweep_id:
        script_dir: Path = Path.cwd() / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path: Path = script_dir / f"run_sweep_{Path(sweep_id).name}.py"

        create_sweep_agent_script(
            sweep_id=sweep_id, output_path=str(script_path), count=_cfg_int(sweep_cfg, "count", 10)
        )
        print(f"W&B sweep created with ID: {sweep_id}")
        print(f"Helper script written to {script_path} - run with:")
        print(f"   python {script_path}")

    return sweep_id


def run_create_sweep(
    _env: ABCEnvironment[ABCState], cfg: ConfigInput, tracking: WandbTrackingSession | None = None
) -> str:
    """Stage entrypoint for create_sweep.

    Tracking session is accepted for stage API compatibility. Sweep creation uses
    explicit project/entity from config and falls back to active run context in
    configure_sweep when available.
    """
    _ = tracking
    return create_wandb_sweep(cfg)


def create_multi_metric_sweep_config_with_search_controls(
    *,
    method: str = "bayes",
    metrics: list[dict[str, JSONType]] | None = None,
    parameters: Mapping[str, ParamSpec] | None = None,
    early_terminate: dict[str, JSONType] | None = None,
    scheduler: dict[str, JSONType] | None = None,
    search_space: dict[str, JSONType] | None = None,
) -> dict[str, JSONType]:
    """Create a sweep config with multiple metrics and search controls.

    Args:
        method: Optimization method (bayes, grid, random, hyperband)
        metrics: List of metrics for multi-objective optimization
        parameters: Parameter space configuration
        early_terminate: Early termination configuration
        scheduler: Scheduler configuration.
        search_space: Search-space definition.

    Returns:
        Sweep configuration containing the supplied metrics and controls.
    """
    # Start with a JSON-compatible parameters dict
    params_json: dict[str, JSONType] = {key: dict(value) for key, value in (parameters or {}).items()}

    config: dict[str, JSONType] = {"method": method, "parameters": params_json}

    # Multi-objective optimization (wandb>=0.17)
    if metrics:
        # single or multi-objective metric(s) -> produce JSON-compatible value
        metrics_json: list[JSONType] = [dict(m) for m in metrics]
        config["metric"] = metrics_json[0] if len(metrics_json) == 1 else metrics_json
    else:
        config["metric"] = {"name": "val_loss", "goal": "minimize"}

    # Early termination (wandb>=0.16)
    if early_terminate:
        config["early_terminate"] = dict(early_terminate)

    # Scheduler configuration
    if scheduler:
        config["scheduler"] = dict(scheduler)

    # Search-space configuration
    if search_space:
        config["search_space"] = dict(search_space)

    return config


def create_multi_objective_sweep(
    objectives: list[dict[str, JSONType]],
    parameters: Mapping[str, ParamSpec],
    method: str = "bayes",
    weights: list[float] | None = None,
) -> dict[str, JSONType]:
    """Create a multi-objective optimization sweep with wandb>=0.17.

    Args:
        objectives: List of objective dicts with 'name' and 'goal'
        parameters: Parameter space configuration
        method: Optimization method
        weights: Optional weights for objectives

    Returns:
        Multi-objective sweep configuration
    """
    params_json: dict[str, JSONType] = {key: dict(value) for key, value in parameters.items()}

    # Copy the JSON-compatible objective mappings into a mutable list.
    config: dict[str, JSONType] = {"method": method, "metric": [dict(o) for o in objectives], "parameters": params_json}

    # Add objective weights if provided
    if weights and len(weights) == len(objectives):
        metric_weights: list[JSONType] = list(weights)
        config["metric_weights"] = metric_weights

    return config


def create_registry_linked_sweep(
    sweep_config: dict[str, JSONType], registry_name: str, artifact_type: str = "model", link_best_models: bool = True
) -> dict[str, JSONType]:
    """Create a sweep with Registry integration for wandb>=0.19.

    Args:
        sweep_config: Base sweep configuration
        registry_name: Name of the model registry
        artifact_type: Type of artifacts to track
        link_best_models: Whether to link the top-ranked models after the sweep.

    Returns:
        Sweep configuration with registry settings.
    """
    registry_config: dict[str, JSONType] = sweep_config.copy()

    # Add registry configuration
    registry_config["registry"] = {
        "name": registry_name,
        "artifact_type": artifact_type,
        "auto_link_best": link_best_models,
    }

    return registry_config


class SweepArtifactManager:
    """Collect and combine artifacts produced by W&B sweep runs."""

    def __init__(self, sweep_id: str, project: str | None = None, entity: str | None = None) -> None:
        """Initialize the sweep artifact manager.

        Args:
            sweep_id: W&B sweep ID
            project: W&B project name
            entity: W&B entity name
        """
        self.sweep_id: str = sweep_id
        self.project: str | None = project
        self.entity: str | None = entity
        self._api: wandb.Api | None = None

        try:
            if hasattr(wandb, "Api"):
                self._api = wandb.Api()
            else:
                log.warning("wandb.Api not available in this version")
        except Exception:
            log.warning("Failed to initialize W&B API client")

    def collect_best_artifacts(
        self, metric_name: str = "val_loss", minimize: bool = True, top_k: int = 5, artifact_type: str = "model"
    ) -> list[wandb.Artifact]:
        """Collect the best artifacts from sweep runs.

        Args:
            metric_name: Metric to use for ranking
            minimize: Whether to minimize the metric
            top_k: Number of top artifacts to collect
            artifact_type: Type of artifacts to collect

        Returns:
            List of best artifact objects
        """
        if not self._api:
            log.warning("W&B API not available")
            return []

        try:
            # Get sweep runs
            sweep: SweepLike = self._api.sweep(f"{self.entity}/{self.project}/{self.sweep_id}")
            scored_runs: list[tuple[float, SweepRunLike]] = _score_sweep_runs(sweep.runs, metric_name)
            scored_runs.sort(key=operator.itemgetter(0), reverse=not minimize)
            return _artifacts_from_scored_runs(scored_runs, top_k, artifact_type)

        except Exception:
            log.exception("Failed to collect best artifacts")
            return []

    @staticmethod
    def create_ensemble_artifact(
        artifacts: list[wandb.Artifact],
        ensemble_name: str = "sweep_ensemble",
        metadata: dict[str, JSONType] | None = None,
    ) -> wandb.Artifact | None:
        """Create an ensemble artifact from multiple model artifacts.

        Args:
            artifacts: List of model artifacts to ensemble
            ensemble_name: Name for the ensemble artifact
            metadata: Optional metadata for the ensemble

        Returns:
            Ensemble artifact object or None
        """
        if not artifacts:
            return None

        try:
            # Create ensemble artifact
            ensemble = wandb.Artifact(name=ensemble_name, type="ensemble", metadata=metadata or {})
            _add_ensemble_references(ensemble, artifacts)
            if wandb.run:
                wandb.run.log_artifact(ensemble)
        except Exception:
            log.exception("Failed to create ensemble artifact")
            return None
        else:
            return ensemble


def launch_sweep_with_run_limits(
    config: dict[str, JSONType],
    project: str,
    entity: str | None = None,
    max_concurrent_runs: int = 10,
    auto_stop_criteria: dict[str, JSONType] | None = None,
) -> str:
    """Launch a sweep with concurrency and early-stop limits.

    Args:
        config: Sweep configuration.
        project: W&B project name.
        entity: W&B entity name.
        max_concurrent_runs: Maximum number of concurrent runs.
        auto_stop_criteria: Early-termination criteria.

    Returns:
        Sweep ID if successful, or an empty string on failure.
    """
    try:
        launch_config: dict[str, JSONType] = _sweep_config_with_run_limits(
            config, max_concurrent_runs, auto_stop_criteria
        )
        sweep_id: str = configure_sweep(launch_config, project_name=project, entity_name=entity)

        if sweep_id:
            log.info(f"Sweep created with run limits: {sweep_id}")
    except Exception:
        log.exception("Failed to launch sweep with run limits")
        return ""
    else:
        return sweep_id


def summarize_sweep_metrics(
    sweep_id: str, project: str, entity: str | None = None, check_interval: int = 300
) -> dict[str, JSONType]:
    """Read completed sweep runs and summarize their validation loss.

    Args:
        sweep_id: W&B sweep ID
        project: W&B project name
        entity: W&B entity name
        check_interval: Retained for callers that schedule repeated checks.

    Returns:
        Run counts and validation-loss summary fields.
    """
    _ = check_interval
    try:
        summary: dict[str, JSONType] = _summarize_sweep_metrics(project, entity, sweep_id)
    except Exception:
        log.exception("Failed to summarize sweep metrics")
        return {"status": "error"}
    else:
        return summary
