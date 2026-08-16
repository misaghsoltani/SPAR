"""Hydra composition and workflow execution for SPAR Optuna trials."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from importlib import import_module
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import numpy as np
from omegaconf import OmegaConf, SCMode
import torch
from torch import Tensor, nn

from spar.utils.config_utils.config_validation import validate_config
from spar.utils.config_utils.misc import register_omega_conf_resolvers
from spar.utils.import_utils.stage_importer import run_stage

from .adapters import get_stage_adapter
from .contracts import StepExecutionResult
from .path_utils import render_template_string, render_templates, set_path_value, set_path_value_if_present
from .space import build_grid_search_space, collect_step_parameter_specs, sample_step_parameters

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import _GeneratorContextManager
    from logging import Logger
    from types import ModuleType
    from typing import TypeAlias

    from omegaconf.dictconfig import DictConfig
    from optuna.trial import Trial as OptunaTrial

    from spar.environments.abstracts import ABCEnvironment
    from spar.environments.abstracts.state import ABCState
    from spar.utils.config_utils.config_schema import (
        OptunaAnalyzeSPARConfig,
        OptunaReplaySPARConfig,
        OptunaRuntimeConfig,
        OptunaStudySPARConfig,
        SPARConfig,
        WorkflowStep,
    )
    from spar.utils.import_utils.stage_importer import StageReturn
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession

    from .adapters import GenericStageAdapter
    from .compat import ParameterSpec
    from .contracts import ObjectiveResult, StageReporter, TrialContext
    from .types import PathValue, SampledValue, StageValue

    RootOptunaConfig: TypeAlias = OptunaStudySPARConfig | OptunaAnalyzeSPARConfig | OptunaReplaySPARConfig


logger: Logger = getLogger(__name__)
_CONFIG_DIR: Path = Path(__file__).resolve().parents[1] / "configs"


def default_objective_step_name(steps: list[WorkflowStep]) -> str:
    """Return the default step used when metrics/params omit an explicit step."""
    for step in steps:
        if step.role == "objective":
            return step.name
    if not steps:
        raise ValueError("Optuna workflow requires at least one step")
    return steps[-1].name


@contextmanager
def _compose_context() -> Generator[None, None, None]:
    """Context manager for Hydra composition of SPAR configs."""
    import_module("spar.utils.config_utils.config_store")
    register_omega_conf_resolvers()
    if GlobalHydra.instance().is_initialized():
        yield
        return
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        yield


def _compose_step_cfg(
    root_cfg: RootOptunaConfig,
    *,
    step: WorkflowStep,
    trial_context: TrialContext,
    extra_base_overrides: list[str] | None = None,
) -> dict[str, PathValue]:
    """Compose a workflow step config with all overrides applied."""
    env_name: str = step.env_name or root_cfg.env.name
    overrides: list[str] = [f"env={env_name}", f"stage={step.stage}"]
    if step.experiment:
        overrides.append(f"+experiment={step.experiment}")

    rendered_base_overrides: list[str] = [
        render_template_string(override, trial_context.template_context())
        for override in root_cfg.optuna.study.base_overrides
    ]
    rendered_step_overrides: list[str] = [
        render_template_string(override, trial_context.template_context()) for override in step.overrides
    ]
    rendered_extra_base: list[str] = [
        render_template_string(override, trial_context.template_context()) for override in (extra_base_overrides or [])
    ]
    overrides.extend(rendered_base_overrides)
    overrides.extend(rendered_extra_base)
    overrides.extend(rendered_step_overrides)

    with _compose_context():
        composed_cfg: DictConfig = compose(config_name="config", overrides=overrides, return_hydra_config=False)

    raw_cfg = OmegaConf.to_container(composed_cfg, resolve=False, structured_config_mode=SCMode.DICT)
    if not isinstance(raw_cfg, dict):
        raise TypeError(
            f"Expected composed config for step {step.name!r} to be a mapping, got {type(raw_cfg).__name__}"
        )
    return {str(key): value for key, value in raw_cfg.items()}


def collect_workflow_grid_search_space(
    root_cfg: RootOptunaConfig, *, trial_context: TrialContext
) -> dict[str, list[SampledValue]]:
    """Collect the exact finite grid used by a workflow.

    Args:
        root_cfg: Root Optuna stage configuration.
        trial_context: Static context used to compose workflow configurations.

    Returns:
        Parameter names mapped to finite candidate values.

    Raises:
        ValueError: If the workflow is empty, dynamic, conditional, or non-finite.
    """
    workflow_steps: list[WorkflowStep] = list(root_cfg.optuna.study.workflow)
    if not workflow_steps:
        raise ValueError("optuna.study.workflow must contain at least one workflow step")

    default_step_name = default_objective_step_name(workflow_steps)
    parameter_specs_by_step: dict[str, list[ParameterSpec]] = {}
    for step in workflow_steps:
        try:
            raw_step_cfg = _compose_step_cfg(root_cfg, step=step, trial_context=trial_context)
        except Exception as exc:
            raise ValueError(
                f"Grid sampler requires a statically composable workflow. "
                f"Step {step.name!r} could not be composed before trial execution: {exc}"
            ) from exc
        parameter_specs, conversion_warnings = collect_step_parameter_specs(
            root_cfg.optuna, step=step, raw_step_cfg=raw_step_cfg, default_step_name=default_step_name
        )
        if conversion_warnings:
            details = " | ".join(conversion_warnings)
            raise ValueError(
                f"Grid sampler could not convert the configured search space exactly for step {step.name!r}: {details}"
            )
        parameter_specs_by_step[step.name] = parameter_specs

    return build_grid_search_space(
        parameter_specs_by_step, default_step_name=default_step_name, multi_step=len(workflow_steps) > 1
    )


def _path_mapping(value: Mapping[str, PathValue] | dict[str, PathValue], *, label: str) -> dict[str, PathValue]:
    """Require a string-key mapping and normalize nested mapping keys to strings."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected {label} to be a mapping, got {type(value).__name__}")
    return {key: _normalize_path_value(item) for key, item in value.items()}


def _normalize_path_value(value: PathValue) -> PathValue:
    """Normalize nested config values to use concrete dict/list containers."""
    if isinstance(value, dict):
        return {str(key): _normalize_path_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_path_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_path_value(item) for item in value)
    return value


def _preferred_device() -> str:
    """Return the preferred device for SPAR trial execution."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _apply_device_policy(raw_cfg: dict[str, PathValue], runtime_cfg: OptunaRuntimeConfig) -> None:
    """Apply the configured device policy to a trial config."""
    if runtime_cfg.device_policy == "preserve":
        return
    target_device: str = (
        runtime_cfg.device if runtime_cfg.device_policy == "explicit" and runtime_cfg.device else _preferred_device()
    )
    for path in ("train.device", "test.device", "device"):
        set_path_value_if_present(raw_cfg, path, target_device)


def _apply_trial_output_defaults(raw_cfg: dict[str, PathValue], *, step_name: str, trial_dir: Path) -> None:
    step_dir: Path = trial_dir / step_name
    output_overrides: dict[str, str] = {
        "save_dir": str(step_dir),
        "save_paths.model_dir": str(step_dir / "model"),
        "save_paths.images_dir": str(step_dir / "images"),
        "save_paths.plots_dir": str(step_dir / "plots"),
        "save_paths.metrics_dir": str(step_dir / "metrics"),
        "search.results_dir": str(step_dir / "results"),
        "data.save_dir": str(step_dir / "data"),
        "search_data.save_dir": str(step_dir / "search_data"),
        "train.dqn.save_dir": str(step_dir / "heuristic"),
        "paths.output_dir": str(step_dir / "artifacts"),
        "wandb.dir": str(step_dir / "wandb"),
        "hydra.run.dir": str(step_dir / "hydra"),
    }
    for path, value in output_overrides.items():
        set_path_value_if_present(raw_cfg, path, value)


def _apply_runtime_policy(
    raw_cfg: dict[str, PathValue],
    *,
    step_name: str,
    trial_dir: Path,
    runtime_cfg: OptunaRuntimeConfig,
    disable_wandb: bool,
) -> None:
    _apply_trial_output_defaults(raw_cfg, step_name=step_name, trial_dir=trial_dir)
    _apply_device_policy(raw_cfg, runtime_cfg)
    if disable_wandb:
        set_path_value_if_present(raw_cfg, "wandb.mode", "disabled")


def _save_resolved_config(validated_cfg: SPARConfig, *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict: dict[str, PathValue] = _path_mapping(asdict(validated_cfg), label="validated config")
    output_path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg_dict), resolve=True), encoding="utf-8")


def _normalize_stage_sequence(values: list[StageValue] | tuple[StageValue, ...]) -> list[PathValue]:
    """Normalize a sequence of runtime values into Optuna-safe values."""
    normalized: list[PathValue] = []
    for item in values:
        keep, normalized_value = _normalize_stage_value(item)
        if keep:
            normalized.append(normalized_value)
    return normalized


def _normalize_stage_value(value: StageValue) -> tuple[bool, PathValue]:
    """Convert stage output values into a serializable subset for Optuna bookkeeping."""
    if isinstance(value, bool | int | float | str) or value is None:
        return True, value
    if isinstance(value, Path):
        return True, str(value)
    if isinstance(value, bytes):
        return True, value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, bool | int | float | str) or scalar is None:
            return True, scalar
        return False, None
    if isinstance(value, np.ndarray):
        return True, _normalize_stage_sequence(value.tolist())
    if isinstance(value, Tensor):
        return True, _normalize_stage_sequence(value.detach().cpu().tolist())
    if isinstance(value, nn.Module):
        return False, None
    if isinstance(value, dict):
        normalized: dict[str | int, PathValue] = {}
        for key, item in value.items():
            keep, normalized_item = _normalize_stage_value(item)
            if keep:
                normalized[key] = normalized_item
        return True, normalized
    if isinstance(value, list):
        return True, _normalize_stage_sequence(value)
    if isinstance(value, tuple):
        return True, tuple(_normalize_stage_sequence(value))
    return False, None


def _normalize_stage_result(stage_result: StageValue) -> dict[str, PathValue]:
    """Drop non-serializable runtime objects from a stage result payload."""
    if not isinstance(stage_result, dict):
        return {}
    normalized_result: dict[str, PathValue] = {}
    for key, value in stage_result.items():
        keep, normalized_value = _normalize_stage_value(value)
        if keep:
            normalized_result[str(key)] = normalized_value
    return normalized_result


def _run_validated_stage(
    validated_cfg: SPARConfig, *, reporter: StageReporter | None
) -> tuple[dict[str, PathValue], WandbTrackingSession | None]:
    get_environment: Callable[[str], ABCEnvironment[ABCState]] = import_module("spar.utils.env_utils").get_environment
    wandb_logger_module: ModuleType = import_module("spar.utils.log_utils.wandb_logger")
    managed_tracking_session: Callable[..., _GeneratorContextManager[WandbTrackingSession, None, None]] = (
        wandb_logger_module.managed_tracking_session
    )
    define_default_metrics: Callable[[WandbTrackingSession], None] = wandb_logger_module.define_default_metrics

    env: ABCEnvironment[ABCState] = get_environment(validated_cfg.env.name)
    with managed_tracking_session(validated_cfg, stage=validated_cfg.stage, rank=0) as tracking:
        if tracking.enabled:
            define_default_metrics(tracking)
        stage_result: StageReturn = run_stage(
            validated_cfg.stage, env, validated_cfg, tracking=tracking, reporter=reporter
        )
        return _normalize_stage_result(stage_result), tracking


def run_workflow(
    root_cfg: RootOptunaConfig,
    *,
    trial_context: TrialContext,
    trial: OptunaTrial | None = None,
    sampled_values_override_by_step: Mapping[str, Mapping[str, SampledValue]] | None = None,
    reporters_by_step: Mapping[str, StageReporter] | None = None,
    extra_base_overrides: list[str] | None = None,
    disable_wandb: bool | None = None,
) -> TrialContext:
    """Compose, validate, and execute the configured workflow for one trial/replay."""
    workflow_steps: list[WorkflowStep] = list(root_cfg.optuna.study.workflow)
    if not workflow_steps:
        raise ValueError("optuna.study.workflow must contain at least one workflow step")

    default_step_name: str = default_objective_step_name(workflow_steps)
    multi_step: bool = len(workflow_steps) > 1
    sampled_values_by_step: dict[str, dict[str, SampledValue]] = {
        step_name: dict(values) for step_name, values in (sampled_values_override_by_step or {}).items()
    }
    base_cfgs_by_step: dict[str, dict[str, PathValue]] = {}
    reporters_by_step = reporters_by_step or {}
    disable_wandb_flag: bool = root_cfg.optuna.runtime.disable_wandb if disable_wandb is None else disable_wandb

    for step in workflow_steps:
        raw_base_cfg: dict[str, PathValue] = _compose_step_cfg(
            root_cfg, step=step, trial_context=trial_context, extra_base_overrides=extra_base_overrides
        )
        base_cfgs_by_step[step.name] = raw_base_cfg
        parameter_specs: list[ParameterSpec]
        warnings: list[str]
        parameter_specs, warnings = collect_step_parameter_specs(
            root_cfg.optuna, step=step, raw_step_cfg=raw_base_cfg, default_step_name=default_step_name
        )
        for warning in warnings:
            logger.warning(warning)

        step_sampled_values: dict[str, SampledValue]
        trial_named_values: dict[str, SampledValue] = {}
        if step.name in sampled_values_by_step:
            step_sampled_values = dict(sampled_values_by_step[step.name])
        elif trial is None:
            step_sampled_values = {}
        else:
            step_sampled_values, trial_named_values = sample_step_parameters(
                trial,
                step=step,
                parameter_specs=parameter_specs,
                default_step_name=default_step_name,
                sampled_values_by_step=sampled_values_by_step,
                base_cfgs_by_step=base_cfgs_by_step,
                multi_step=multi_step,
            )
            sampled_values_by_step[step.name] = dict(step_sampled_values)
            trial_context.sampled_parameters.update(trial_named_values)

        trial_context.sampled_values_by_step[step.name] = dict(step_sampled_values)

        raw_trial_cfg: dict[str, PathValue] = deepcopy(raw_base_cfg)
        for path, value in step_sampled_values.items():
            set_path_value(raw_trial_cfg, path, value)

        _apply_runtime_policy(
            raw_trial_cfg,
            step_name=step.name,
            trial_dir=trial_context.trial_dir,
            runtime_cfg=root_cfg.optuna.runtime,
            disable_wandb=disable_wandb_flag,
        )
        rendered_cfg_value: PathValue = render_templates(raw_trial_cfg, trial_context.template_context())
        if not isinstance(rendered_cfg_value, Mapping):
            raise TypeError(f"Expected rendered config for {step.name!r} to be a mapping")
        rendered_cfg: dict[str, PathValue] = _path_mapping(
            {str(key): value for key, value in rendered_cfg_value.items()}, label=f"rendered config for {step.name}"
        )
        cfg_for_validation: DictConfig = OmegaConf.create(rendered_cfg)
        stage_name = str(cfg_for_validation.get("stage"))
        validated_cfg: SPARConfig = validate_config(cfg_for_validation, stage=stage_name)

        resolved_config_path: Path | None = None
        if root_cfg.optuna.runtime.copy_resolved_configs:
            resolved_config_path = trial_context.trial_dir / step.name / "resolved_config.yaml"
            _save_resolved_config(validated_cfg, output_path=resolved_config_path)

        stage_result, _tracking = _run_validated_stage(validated_cfg, reporter=reporters_by_step.get(step.name))
        adapter: GenericStageAdapter = get_stage_adapter(validated_cfg.stage)
        objective_result: ObjectiveResult = adapter.extract(
            step=step, config=validated_cfg, stage_result=stage_result, trial_context=trial_context
        )
        step_result = StepExecutionResult(
            step=step,
            config=validated_cfg,
            raw_result=stage_result,
            objective_result=objective_result,
            resolved_config_path=str(resolved_config_path) if resolved_config_path is not None else None,
        )
        trial_context.step_results[step.name] = step_result
        if resolved_config_path is not None:
            trial_context.resolved_config_paths[step.name] = str(resolved_config_path)

    return trial_context
