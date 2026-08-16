"""High-level Optuna orchestration for SPAR."""

from __future__ import annotations

from dataclasses import asdict
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

from optuna.exceptions import TrialPruned
from optuna.importance import get_param_importances
from optuna.samplers import GridSampler
from optuna.study.study import create_study, load_study
from optuna.trial import TrialState
import orjson

from spar.utils.config_utils.config_validation import ConfigValidationError

from .builders import build_pruner, build_sampler, build_storage
from .contracts import ConstraintSpec, MetricSpec, TrialContext
from .workflow import collect_workflow_grid_search_space, default_objective_step_name, run_workflow

if TYPE_CHECKING:
    from logging import Logger

    from optuna.study.study import Study
    from optuna.trial import FrozenTrial, Trial as OptunaTrial

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.utils.config_utils.config_schema import (
        OptunaAnalyzeSPARConfig,
        OptunaReplaySPARConfig,
        OptunaStudySPARConfig,
        StudyConfig,
        WorkflowStep,
    )
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession

    from .builders import BasePruner, BaseSampler, BaseStorage
    from .contracts import ObjectiveResult
    from .types import ReporterPayload, ReporterValue, SampledValue, ScalarMetric, TemplateValue


logger: Logger = getLogger(__name__)


class RecoverableTrialError(RuntimeError):
    """Marker exception used to let Optuna continue after recoverable trial failures."""


def _is_training_stage(stage_name: str) -> bool:
    return stage_name.startswith("train_")


def _infer_default_metric(step: WorkflowStep) -> MetricSpec:
    stage_name: str = step.stage
    if stage_name in {"train_env_disc", "train_env_cont", "train_world_model"}:
        return MetricSpec(name="val_loss", step=step.name, goal="minimize")
    if stage_name in {"train_alignment_disc", "train_alignment_cont", "train_alignment_model"}:
        return MetricSpec(name="val_loss", step=step.name, goal="minimize")
    if stage_name == "train_heuristic":
        return MetricSpec(name="per_solved_test", step=step.name, goal="maximize")
    if stage_name == "test_model_cont":
        return MetricSpec(name="cosine_similarity_mean", step=step.name, goal="maximize")
    if stage_name in {"test_model_disc", "test_model", "test_model_combined"}:
        return MetricSpec(name="eq_bit_min_mean", step=step.name, goal="maximize")
    if stage_name in {"search_qstar", "search_gbfs", "search_ucs"}:
        return MetricSpec(name="search_success_rate", step=step.name, goal="maximize")
    return MetricSpec(name="val_loss", step=step.name, goal="minimize")


def _resolve_objective_specs(
    cfg: OptunaStudySPARConfig | OptunaAnalyzeSPARConfig | OptunaReplaySPARConfig,
) -> list[MetricSpec]:
    study_cfg: StudyConfig = cfg.optuna.study
    default_step: str | None = default_objective_step_name(list(study_cfg.workflow)) if study_cfg.workflow else None
    if study_cfg.objectives:
        return [
            MetricSpec(name=spec.name, step=spec.step or default_step, goal=spec.goal) for spec in study_cfg.objectives
        ]
    if study_cfg.objective is not None:
        return [
            MetricSpec(
                name=study_cfg.objective.name,
                step=study_cfg.objective.step or default_step,
                goal=study_cfg.objective.goal,
            )
        ]

    inferred_specs: list[MetricSpec] = []
    for step in study_cfg.workflow:
        if step.objective is not None:
            inferred_specs.append(
                MetricSpec(name=step.objective.name, step=step.objective.step or step.name, goal=step.objective.goal)
            )
        elif step.role == "objective":
            inferred_specs.append(_infer_default_metric(step))

    if not inferred_specs and study_cfg.workflow:
        inferred_specs.append(_infer_default_metric(study_cfg.workflow[-1]))
    return inferred_specs


def _resolve_constraint_specs(
    cfg: OptunaStudySPARConfig | OptunaAnalyzeSPARConfig | OptunaReplaySPARConfig,
) -> list[ConstraintSpec]:
    default_step: str | None = (
        default_objective_step_name(list(cfg.optuna.study.workflow)) if cfg.optuna.study.workflow else None
    )
    constraint_specs: list[ConstraintSpec] = [
        ConstraintSpec(name=spec.name, step=spec.step or default_step, operator=spec.operator, threshold=spec.threshold)
        for spec in cfg.optuna.study.constraints
    ]
    for step in cfg.optuna.study.workflow:
        constraint_specs.extend(
            ConstraintSpec(
                name=spec.name, step=spec.step or step.name, operator=spec.operator, threshold=spec.threshold
            )
            for spec in step.constraints
        )
    return constraint_specs


def _study_name(cfg: OptunaStudySPARConfig | OptunaAnalyzeSPARConfig | OptunaReplaySPARConfig) -> str:
    return cfg.optuna.study.study_name


def _study_dir(cfg: OptunaStudySPARConfig | OptunaAnalyzeSPARConfig | OptunaReplaySPARConfig) -> Path:
    return Path(cfg.optuna.runtime.output_root) / _study_name(cfg)


def _trial_dir(cfg: OptunaStudySPARConfig, *, trial_number: int) -> Path:
    return Path(
        cfg.optuna.runtime.trial_dir_template.format(
            output_root=cfg.optuna.runtime.output_root, study_name=_study_name(cfg), trial_number=trial_number
        )
    )


def _replay_trial_dir(cfg: OptunaReplaySPARConfig, *, trial_number: int) -> Path:
    return _study_dir(cfg) / "replay" / f"trial_{trial_number:05d}"


def _metric_value(trial_context: TrialContext, spec: MetricSpec) -> float:
    if spec.step is None:
        raise ValueError(f"Metric spec {spec.name!r} is missing a step reference")
    step_result: ObjectiveResult = trial_context.step_results[spec.step].objective_result
    if spec.name in step_result.metrics:
        metric_value: ScalarMetric = step_result.metrics[spec.name]
        if isinstance(metric_value, bool | int | float):
            return float(metric_value)
    if step_result.objective is not None and spec.name == "objective":
        objective_value: float | tuple[float, ...] = step_result.objective
        if isinstance(objective_value, tuple):
            raise ValueError("Tuple objective requires explicit metric names")
        return objective_value
    raise ValueError(f"Metric {spec.name!r} was not produced by workflow step {spec.step!r}")


def _constraint_violation(metric_value: float, spec: ConstraintSpec) -> float:
    if spec.operator in {"<=", "<"}:
        return metric_value - spec.threshold
    if spec.operator in {">=", ">"}:
        return spec.threshold - metric_value
    return abs(metric_value - spec.threshold)


def _completed_trials(study: Study) -> list[FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def _sort_trials_for_replay(trials: list[FrozenTrial], *, objective_spec: MetricSpec) -> list[FrozenTrial]:
    reverse: bool = objective_spec.goal == "maximize"

    def _trial_value(trial: FrozenTrial) -> float:
        return trial.value if trial.value is not None else float("-inf")

    return sorted(trials, key=_trial_value, reverse=reverse)


class OptunaTrialReporter:
    """Low-overhead Optuna reporter used at phase/checkpoint boundaries."""

    def __init__(self, trial: OptunaTrial) -> None:
        self._trial: OptunaTrial = trial
        self._report_count: int = 0

    def __call__(self, payload: ReporterPayload) -> None:
        """Report sparse phase-level progress into Optuna and trigger pruning when appropriate."""
        raw_value: ReporterValue = payload.get("primary")
        if not isinstance(raw_value, bool | int | float):
            return
        step_index_raw: ReporterValue = payload.get("iteration")
        step_index: int
        step_index = step_index_raw if isinstance(step_index_raw, int) else self._report_count
        self._trial.report(float(raw_value), step=step_index)
        self._report_count += 1
        if self._trial.should_prune():
            raise TrialPruned(f"Trial pruned at reported step {step_index}")


def _reporters_by_step(
    cfg: OptunaStudySPARConfig, *, trial: OptunaTrial, objective_specs: list[MetricSpec]
) -> dict[str, OptunaTrialReporter]:
    if cfg.optuna.pruner.kind == "none":
        return {}

    if len(objective_specs) != 1:
        return {}

    objective_spec: MetricSpec = objective_specs[0]
    if objective_spec.step is None:
        return {}

    step: WorkflowStep | None = next(
        (workflow_step for workflow_step in cfg.optuna.study.workflow if workflow_step.name == objective_spec.step),
        None,
    )
    if step is None or not _is_training_stage(step.stage):
        return {}
    return {objective_spec.step: OptunaTrialReporter(trial)}


def _recoverable(exc: Exception) -> bool:
    if isinstance(exc, (ConfigValidationError, FileNotFoundError, MemoryError)):
        return True
    if isinstance(exc, RuntimeError):
        message: str = str(exc).lower()
        return "out of memory" in message or "mps backend out of memory" in message
    return False


def _trial_summary_payload(
    *, trial_context: TrialContext, objective_specs: list[MetricSpec], constraints_vector: list[float]
) -> dict[str, TemplateValue]:
    objective_specs_payload: list[TemplateValue] = [asdict(spec) for spec in objective_specs]
    constraints_payload: list[TemplateValue] = list(constraints_vector)
    return {
        "study_name": trial_context.study_name,
        "trial_number": trial_context.trial_number,
        "trial_dir": str(trial_context.trial_dir),
        "sampled_parameters": trial_context.sampled_parameters,
        "sampled_values_by_step": trial_context.sampled_values_by_step,
        "objective_specs": objective_specs_payload,
        "constraints_vector": constraints_payload,
        "step_results": {
            step_name: {
                "metrics": result.objective_result.metrics,
                "constraints": result.objective_result.constraints,
                "artifacts": result.objective_result.artifacts,
                "status": result.objective_result.status,
                "resolved_config_path": result.resolved_config_path,
            }
            for step_name, result in trial_context.step_results.items()
        },
    }


def _write_json(output_path: Path, payload: dict[str, TemplateValue]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def _load_study(cfg: OptunaStudySPARConfig | OptunaAnalyzeSPARConfig | OptunaReplaySPARConfig) -> Study:
    study_dir: Path = _study_dir(cfg)
    storage: BaseStorage | str = build_storage(cfg.optuna.storage, study_dir=study_dir, study_name=_study_name(cfg))
    return load_study(study_name=_study_name(cfg), storage=storage)


def _analysis_payload(
    cfg: OptunaStudySPARConfig | OptunaAnalyzeSPARConfig | OptunaReplaySPARConfig, study: Study
) -> dict[str, TemplateValue]:
    completed_trials: list[FrozenTrial] = _completed_trials(study)
    payload: dict[str, TemplateValue] = {
        "study_name": study.study_name,
        "num_trials": len(study.trials),
        "num_completed_trials": len(completed_trials),
        "directions": [direction.name.lower() for direction in study.directions],
        "best_trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "values": list(trial.values) if trial.values is not None else None,
                "params": trial.params,
                "user_attrs": dict(trial.user_attrs),
            }
            for trial in (
                study.best_trials if len(study.directions) > 1 else ([study.best_trial] if completed_trials else [])
            )
        ],
    }
    if cfg.optuna.analysis.compute_param_importances and len(study.directions) == 1 and len(completed_trials) >= 2:
        try:
            importances: dict[str, float] = get_param_importances(study)
        except Exception as exc:
            payload["param_importances_error"] = str(exc)
        else:
            payload["param_importances"] = importances
    return payload


def analyze_study(
    _env: ABCEnvironment[ABCState], cfg: OptunaAnalyzeSPARConfig, tracking: WandbTrackingSession | None = None
) -> dict[str, TemplateValue]:
    """Analyze an existing Optuna study and export summary artifacts."""
    del tracking
    study: Study = _load_study(cfg)
    analysis_dir: Path = (
        Path(cfg.optuna.analysis.output_dir) if cfg.optuna.analysis.output_dir else _study_dir(cfg) / "analysis"
    )
    analysis: dict[str, TemplateValue] = _analysis_payload(cfg, study)
    _write_json(analysis_dir / "analysis.json", analysis)
    if cfg.optuna.analysis.export_csv:
        try:
            study.trials_dataframe().to_csv(analysis_dir / "trials.csv", index=False)
        except ImportError:
            logger.warning("Skipping Optuna CSV export because pandas is not installed")
    return analysis


def _run_one_trial(
    trial: OptunaTrial,
    *,
    cfg: OptunaStudySPARConfig,
    objective_specs: list[MetricSpec],
    constraint_specs: list[ConstraintSpec],
) -> float | tuple[float, ...]:
    trial_context = TrialContext(
        study_name=_study_name(cfg),
        trial_number=trial.number,
        trial_dir=_trial_dir(cfg, trial_number=trial.number),
        env_name=cfg.env.name,
    )
    reporters: dict[str, OptunaTrialReporter] = _reporters_by_step(cfg, trial=trial, objective_specs=objective_specs)
    executed_context: TrialContext = run_workflow(
        cfg,
        trial_context=trial_context,
        trial=trial,
        reporters_by_step=reporters,
        disable_wandb=cfg.optuna.runtime.disable_wandb,
    )

    objective_values: list[float] = [_metric_value(executed_context, spec) for spec in objective_specs]
    constraint_values: list[float] = [
        _constraint_violation(
            _metric_value(executed_context, MetricSpec(name=spec.name, step=spec.step, goal="maximize")), spec
        )
        for spec in constraint_specs
    ]

    trial.set_user_attr("sampled_values_by_step", executed_context.sampled_values_by_step)
    trial.set_user_attr("resolved_config_paths", executed_context.resolved_config_paths)
    trial.set_user_attr("constraints_vector", constraint_values)
    trial.set_user_attr(
        "objective_metrics",
        {f"{spec.step}:{spec.name}": value for spec, value in zip(objective_specs, objective_values, strict=False)},
    )
    summary_payload: dict[str, TemplateValue] = _trial_summary_payload(
        trial_context=executed_context, objective_specs=objective_specs, constraints_vector=constraint_values
    )
    _write_json(trial_context.trial_dir / "trial_summary.json", summary_payload)

    if len(objective_values) == 1:
        return objective_values[0]
    return tuple(objective_values)


def run_study(
    _env: ABCEnvironment[ABCState], cfg: OptunaStudySPARConfig, tracking: WandbTrackingSession | None = None
) -> dict[str, TemplateValue]:
    """Run a configured Optuna study against SPAR workflows."""
    del tracking
    study_dir: Path = _study_dir(cfg)
    study_dir.mkdir(parents=True, exist_ok=True)

    objective_specs: list[MetricSpec] = _resolve_objective_specs(cfg)
    if not objective_specs:
        raise ValueError("Optuna study requires at least one objective metric")
    constraint_specs: list[ConstraintSpec] = _resolve_constraint_specs(cfg)
    directions: list[str] = [spec.goal for spec in objective_specs]

    storage: BaseStorage | str = build_storage(cfg.optuna.storage, study_dir=study_dir, study_name=_study_name(cfg))
    grid_search_space: dict[str, list[SampledValue]] | None = None
    if cfg.optuna.sampler.kind == "grid":
        grid_context = TrialContext(
            study_name=_study_name(cfg),
            trial_number=0,
            trial_dir=_trial_dir(cfg, trial_number=0),
            env_name=cfg.env.name,
        )
        grid_search_space = collect_workflow_grid_search_space(cfg, trial_context=grid_context)
    sampler: BaseSampler = build_sampler(
        cfg.optuna.sampler, constraints_enabled=bool(constraint_specs), grid_search_space=grid_search_space
    )
    pruner: BasePruner = build_pruner(cfg.optuna.pruner)

    study: Study = (
        create_study(
            study_name=_study_name(cfg),
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction=directions[0],
            load_if_exists=cfg.optuna.study.load_if_exists,
        )
        if len(directions) == 1
        else create_study(
            study_name=_study_name(cfg),
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            directions=directions,
            load_if_exists=cfg.optuna.study.load_if_exists,
        )
    )

    def objective(trial: OptunaTrial) -> float | tuple[float, ...]:
        try:
            return _run_one_trial(trial, cfg=cfg, objective_specs=objective_specs, constraint_specs=constraint_specs)
        except TrialPruned:
            raise
        except Exception as exc:
            if _recoverable(exc):
                trial.set_user_attr("failure", str(exc))
                if cfg.optuna.study.recoverable_error_action == "prune":
                    raise TrialPruned(str(exc)) from exc
                if cfg.optuna.study.recoverable_error_action == "raise":
                    raise
                raise RecoverableTrialError(str(exc)) from exc
            if cfg.optuna.runtime.abort_on_unexpected_errors:
                raise
            raise RecoverableTrialError(str(exc)) from exc

    catch: tuple[type[Exception], ...] = (RecoverableTrialError,) if cfg.optuna.study.catch_exceptions else ()
    trials_to_run: int = cfg.optuna.study.n_trials
    if cfg.optuna.study.load_if_exists:
        budgeted_states: set[TrialState] = {TrialState.COMPLETE, TrialState.PRUNED}
        existing_trials: int = sum(1 for trial in study.trials if trial.state in budgeted_states)
        failed_trials: int = sum(1 for trial in study.trials if trial.state == TrialState.FAIL)
        unfinished_trials: int = len(study.trials) - existing_trials - failed_trials
        trials_to_run = max(0, trials_to_run - existing_trials)
        if trials_to_run == 0:
            logger.info(
                "Study %s already has %d completed/pruned trials. Skipping optimization because configured n_trials=%d",
                study.study_name,
                existing_trials,
                cfg.optuna.study.n_trials,
            )

        elif failed_trials or unfinished_trials:
            logger.info(
                "Study %s has %d failed and %d unfinished trial(s). "
                "they will not consume the configured n_trials=%d target",
                study.study_name,
                failed_trials,
                unfinished_trials,
                cfg.optuna.study.n_trials,
            )

    if isinstance(sampler, GridSampler) and sampler.is_exhausted(study):
        trials_to_run = 0
        logger.info("Study %s has exhausted its configured grid. Skipping optimization", study.study_name)

    if trials_to_run > 0:
        study.optimize(
            objective,
            n_trials=trials_to_run,
            timeout=cfg.optuna.study.timeout_sec,
            n_jobs=cfg.optuna.study.n_jobs,
            gc_after_trial=cfg.optuna.study.gc_after_trial,
            show_progress_bar=cfg.optuna.study.show_progress_bar,
            catch=catch,
        )

    summary: dict[str, TemplateValue] = _analysis_payload(cfg, study)
    _write_json(study_dir / "study_summary.json", summary)
    if cfg.optuna.analysis.export_csv:
        try:
            study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)
        except ImportError:
            logger.warning("Skipping Optuna CSV export because pandas is not installed")
    return summary


def replay_trials(
    _env: ABCEnvironment[ABCState], cfg: OptunaReplaySPARConfig, tracking: WandbTrackingSession | None = None
) -> dict[str, TemplateValue]:
    """Replay selected Optuna trials using stored sampled parameter values."""
    del tracking
    study: Study = _load_study(cfg)
    objective_specs: list[MetricSpec] = _resolve_objective_specs(cfg)
    if not objective_specs:
        raise ValueError("Replay requires at least one objective metric")

    selected_trials: list[FrozenTrial]
    if cfg.optuna.replay.trial_numbers:
        selected_trials = [trial for trial in study.trials if trial.number in set(cfg.optuna.replay.trial_numbers)]
    else:
        selected_trials = _sort_trials_for_replay(_completed_trials(study), objective_spec=objective_specs[0])[
            : cfg.optuna.replay.top_k
        ]

    replay_results: list[dict[str, TemplateValue]] = []
    for trial in selected_trials:
        sampled_values_by_step = trial.user_attrs.get("sampled_values_by_step", {})
        if not isinstance(sampled_values_by_step, dict):
            raise TypeError(f"Stored sampled_values_by_step is missing or invalid for trial {trial.number}")
        sampled_values_by_step_typed: dict[str, dict[str, SampledValue]] = {}
        for step_name, values in sampled_values_by_step.items():
            if not isinstance(step_name, str) or not isinstance(values, dict):
                raise TypeError(f"Stored sampled_values_by_step has invalid step payload for trial {trial.number}")
            typed_values: dict[str, SampledValue] = {}
            for path, value in values.items():
                if not isinstance(path, str):
                    raise TypeError(f"Stored sampled_values_by_step has a non-string path for trial {trial.number}")
                if not isinstance(value, bool | int | float | str) and value is not None:
                    raise TypeError(
                        f"Stored sampled_values_by_step has an unsupported sampled value type for trial {trial.number}"
                    )
                typed_values[path] = value
            sampled_values_by_step_typed[step_name] = typed_values
        trial_context = TrialContext(
            study_name=_study_name(cfg),
            trial_number=trial.number,
            trial_dir=_replay_trial_dir(cfg, trial_number=trial.number),
            env_name=cfg.env.name,
            sampled_parameters=dict(trial.params),
        )
        executed_context: TrialContext = run_workflow(
            cfg,
            trial_context=trial_context,
            trial=None,
            sampled_values_override_by_step=sampled_values_by_step_typed,
            extra_base_overrides=list(cfg.optuna.replay.overrides),
            disable_wandb=not cfg.optuna.replay.enable_wandb,
        )
        replay_payload: dict[str, TemplateValue] = {
            "trial_number": trial.number,
            "trial_dir": str(trial_context.trial_dir),
            "step_results": {
                step_name: {
                    "metrics": result.objective_result.metrics,
                    "artifacts": result.objective_result.artifacts,
                    "resolved_config_path": result.resolved_config_path,
                }
                for step_name, result in executed_context.step_results.items()
            },
        }
        replay_results.append(replay_payload)
        _write_json(trial_context.trial_dir / "replay_summary.json", replay_payload)

    replayed_trials_payload: list[TemplateValue] = list(replay_results)
    summary: dict[str, TemplateValue] = {"study_name": _study_name(cfg), "replayed_trials": replayed_trials_payload}
    output_dir: Path = (
        Path(cfg.optuna.replay.output_dir) if cfg.optuna.replay.output_dir else _study_dir(cfg) / "replay"
    )
    _write_json(output_dir / "replay_index.json", summary)
    return summary
