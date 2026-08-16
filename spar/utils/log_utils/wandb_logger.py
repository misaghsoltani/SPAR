"""Manage W&B tracking sessions and the payloads logged through them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Sized
import contextlib
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from functools import cache
import hashlib
from logging import getLogger
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import traceback
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf
from torch import Tensor, save as torch_save
import wandb

from spar.utils.config_utils.config_schema import ConfigScalar, HydraDict, HydraList, WandbConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable
    from contextvars import Token
    from io import BytesIO
    from logging import Logger
    from typing import Literal, TextIO, TypeAlias

    from torch import nn
    from wandb import Api, Artifact, Histogram, Image, Table, Video
    from wandb.apis.public.artifacts import Artifacts
    from wandb.apis.public.registries.registry import Registry
    from wandb.sdk.data_types.image import ImageDataOrPathType
    from wandb.sdk.data_types.table import ColumnKey, InputRow
    from wandb.sdk.wandb_run import Run

    from spar.optuna.types import PathValue, StageValue
    from spar.utils.config_utils.config_schema import SPARConfig


RunConfigPayload: TypeAlias = HydraDict | HydraList | str | None
MetricValue: TypeAlias = int | float | bool | str | Tensor | None
MetricPayload: TypeAlias = Mapping[str, MetricValue]
MetricDictPayload: TypeAlias = dict[str, MetricValue]
TableCell: TypeAlias = ConfigScalar | HydraDict | HydraList
ImageKwargValue: TypeAlias = ConfigScalar
HistogramValues: TypeAlias = Sequence[int | float] | Tensor | NDArray[np.float32 | np.float64 | np.int32 | np.int64]
SweepConfigPayload: TypeAlias = HydraDict
ArtifactMetadata: TypeAlias = HydraDict
StageTablePayload: TypeAlias = tuple[str, list[str], list[list[TableCell]]]

_STAGE_TABLE_MAX_ROWS: int = 256
_SEARCH_TABLE_COLUMNS: list[str] = [
    "index",
    "sequence_index",
    "start_variant",
    "goal_variant",
    "solved_by_search",
    "solved_by_env",
    "solve_category",
    "path_cost",
    "num_moves",
    "num_nodes_generated",
    "num_iterations",
    "elapsed_sec",
    "nodes_per_sec",
]


class RenderableEnvironment(Protocol):
    """Environment-like object that can produce an image render."""

    def render(self) -> ImageDataOrPathType | NDArray[np.uint8 | np.float32 | np.float64 | np.int32 | np.int64] | None:
        """Render the current environment state."""
        ...


logger: Logger = getLogger(__name__)

_ACTIVE_TRACKING_SESSION: ContextVar[WandbTrackingSession | None] = ContextVar("wandb_tracking_session", default=None)


@dataclass(slots=True)
class WandbTrackingSession:
    """Explicit run/session state for W&B logging."""

    enabled: bool
    mode: str
    rank: int
    is_primary: bool
    distributed_mode: str = "rank_zero"
    step_metric: str = "global_step"
    run: Run | None = None
    profile: str = "training"
    media_log_every_n_steps: int = 0
    table_log_every_n_steps: int = 0
    histogram_log_every_n_steps: int = 0
    max_metrics_per_log: int = 512
    max_metric_key_length: int = 128
    last_step: int = -1

    @property
    def run_id(self) -> str | None:
        """Underlying W&B run id when a run is active."""
        return None if self.run is None else self.run.id


@cache
def _git_hash() -> str | None:
    """Get Git commit hash."""
    git_executable: str | None = shutil.which("git")
    if git_executable is None:
        return None

    try:
        return (
            subprocess.check_output([git_executable, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _resolve_wandb_cfg(cfg: SPARConfig | WandbConfig) -> WandbConfig:

    return cfg if isinstance(cfg, WandbConfig) else cfg.wandb


def _resolve_stage_name(cfg: SPARConfig | WandbConfig, stage: str | None) -> str:
    if stage is not None:
        return stage

    return str(getattr(cfg, "stage", "unknown"))


def _to_run_config(cfg: SPARConfig | WandbConfig, wb_cfg: WandbConfig, rank: int) -> RunConfigPayload:
    run_cfg: RunConfigPayload = None
    if not wb_cfg.log_config or rank != 0:
        return run_cfg

    if OmegaConf.is_config(cfg):
        container_cfg = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(container_cfg, dict):
            run_cfg = {str(k): v for k, v in container_cfg.items()}
        elif isinstance(container_cfg, (list, str)):
            run_cfg = container_cfg
    elif is_dataclass(cfg):
        run_cfg = asdict(cfg)

    if run_cfg and isinstance(run_cfg, dict):
        if wb_cfg.config_exclude_keys:
            for key in wb_cfg.config_exclude_keys:
                run_cfg.pop(key, None)

        if wb_cfg.config_include_keys:
            filtered_cfg: HydraDict = {}
            for key in wb_cfg.config_include_keys:
                if key in run_cfg:
                    filtered_cfg[key] = run_cfg[key]
            run_cfg = filtered_cfg

    return run_cfg


def _deterministic_run_id(wb_cfg: WandbConfig, stage: str, rank: int) -> str:
    """Generate a stable run id for resume-friendly workflows.

    Args:
        wb_cfg: W&B configuration for the run.
        stage: Workflow stage name.
        rank: Distributed process rank.

    Returns:
        Stable hexadecimal run identifier.
    """
    rank_identity: str = str(rank) if _get_distributed_logging_mode(wb_cfg) == "per_rank" else ""
    identity_parts: tuple[str, ...] = (
        wb_cfg.project or "",
        wb_cfg.entity or "",
        wb_cfg.group or "",
        wb_cfg.job_type or "",
        stage,
        os.environ.get("SLURM_JOB_ID", ""),
        os.environ.get("HOSTNAME", ""),
        rank_identity,
    )
    digest = hashlib.sha1("|".join(identity_parts).encode("utf-8"), usedforsecurity=False).hexdigest()

    return digest[:16]


def _get_distributed_logging_mode(wb_cfg: WandbConfig) -> str:

    return wb_cfg.distributed_logging


def _is_session_enabled(wb_cfg: WandbConfig, rank: int) -> tuple[bool, bool]:
    if wb_cfg.mode == "disabled":
        return False, rank == 0

    distributed_mode: str = _get_distributed_logging_mode(wb_cfg)
    if distributed_mode == "rank_zero" and rank != 0:
        return False, False

    return True, rank == 0


def _parse_gpu_device_ids(value: ConfigScalar | HydraDict | HydraList) -> list[int]:
    """Parse configured GPU identifiers without accepting ambiguous values.

    Args:
        value: Scalar or Hydra container containing GPU identifiers.

    Returns:
        Non-negative GPU identifiers in their configured order.
    """
    if value is None or isinstance(value, bool | float) or type(value) is dict:
        return []
    if isinstance(value, int):
        return [value] if value >= 0 else []
    if isinstance(value, str):
        identifiers: list[int] = []
        for token in value.split(","):
            normalized: str = token.strip()
            if normalized.isdigit():
                identifiers.append(int(normalized))
        return identifiers

    identifiers = []
    for item in value:
        identifiers.extend(_parse_gpu_device_ids(item))
    return list(dict.fromkeys(identifiers))


def _profile_settings(wb_cfg: WandbConfig, rank: int) -> HydraDict:
    """Build low-overhead settings scoped to the current execution.

    Args:
        wb_cfg: W&B configuration for the run.
        rank: Distributed process rank.

    Returns:
        Settings dictionary ready for ``wandb.Settings``.
    """
    profile: str = wb_cfg.profile
    settings: HydraDict = dict(wb_cfg.settings_dict)

    if profile in {"training", "metrics-only"}:
        settings.setdefault("console", "off")
        settings.setdefault("disable_git", True)
        settings.setdefault("disable_code", True)

    settings.setdefault("x_disable_stats", False)
    settings.setdefault("x_stats_pid", os.getpid())
    settings.setdefault("x_stats_track_process_tree", True)
    execution_dir: Path = Path(wb_cfg.dir).resolve() if wb_cfg.dir is not None else Path.cwd().resolve()
    settings.setdefault("x_stats_disk_paths", [str(execution_dir)])

    distributed_mode: str = _get_distributed_logging_mode(wb_cfg)
    if distributed_mode == "shared":
        settings["mode"] = "shared"

    configured_gpu_ids: ConfigScalar | HydraDict | HydraList = settings.get("x_stats_gpu_device_ids")
    if configured_gpu_ids is None:
        configured_gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids: list[int] = _parse_gpu_device_ids(configured_gpu_ids)
    if gpu_ids and distributed_mode in {"shared", "per_rank"}:
        local_rank_raw: str = os.environ.get("LOCAL_RANK", str(rank))
        local_rank: int = int(local_rank_raw) if local_rank_raw.isdigit() else rank
        gpu_ids = [gpu_ids[local_rank % len(gpu_ids)]]
    if gpu_ids:
        gpu_id_values: HydraList = list(gpu_ids)
        settings["x_stats_gpu_device_ids"] = gpu_id_values
    else:
        settings.pop("x_stats_gpu_device_ids", None)

    return settings


def _resume_params(wb_cfg: WandbConfig, run_id: str | None) -> tuple[str | None, bool | str | None]:
    resume_strategy: str = wb_cfg.resume_strategy

    if wb_cfg.resume is not None:
        return run_id, wb_cfg.resume

    if resume_strategy == "never":
        return run_id, "never"

    if resume_strategy == "must":
        return run_id, "must"

    # deterministic_allow default
    if run_id is None:
        return run_id, None

    return run_id, "allow"


def _new_disabled_session(wb_cfg: WandbConfig, rank: int, is_primary: bool) -> WandbTrackingSession:

    return WandbTrackingSession(
        enabled=False,
        mode=wb_cfg.mode,
        rank=rank,
        is_primary=is_primary,
        distributed_mode=_get_distributed_logging_mode(wb_cfg),
        profile=wb_cfg.profile,
        media_log_every_n_steps=wb_cfg.media_log_every_n_steps,
        table_log_every_n_steps=wb_cfg.table_log_every_n_steps,
        histogram_log_every_n_steps=wb_cfg.histogram_log_every_n_steps,
        max_metrics_per_log=wb_cfg.max_metrics_per_log,
        max_metric_key_length=wb_cfg.max_metric_key_length,
        step_metric=wb_cfg.step_metric,
    )


def init_tracking_session(
    cfg: SPARConfig | WandbConfig, *, stage: str | None = None, rank: int = 0
) -> WandbTrackingSession:
    """Initialize a tracking session and return explicit run ownership state."""
    wb_cfg: WandbConfig = _resolve_wandb_cfg(cfg)
    stage_name: str = _resolve_stage_name(cfg, stage)

    enabled: bool
    is_primary: bool
    enabled, is_primary = _is_session_enabled(wb_cfg, rank)
    if not enabled:
        return _new_disabled_session(wb_cfg, rank, is_primary)

    run_cfg: RunConfigPayload = _to_run_config(cfg, wb_cfg, rank)

    tags: list[str] = list(wb_cfg.tags) if wb_cfg.tags is not None else []
    if commit := _git_hash():
        tags.append(f"git:{commit[:7]}")

    distributed_mode: str = _get_distributed_logging_mode(wb_cfg)
    if distributed_mode != "rank_zero":
        tags.append(f"rank_{rank}")

    requested_run_id: str | None = wb_cfg.id
    if requested_run_id is None and wb_cfg.resume_strategy == "deterministic_allow":
        requested_run_id = _deterministic_run_id(wb_cfg, stage_name, rank)

    resolved_run_id, resume_value = _resume_params(wb_cfg, requested_run_id)
    settings_dict: HydraDict = _profile_settings(wb_cfg, rank)

    if distributed_mode == "shared":
        settings_dict["x_primary"] = rank == 0
        settings_dict["x_label"] = f"rank_{rank}"
        settings_dict["x_update_finish_state"] = rank == 0

    wandb_settings = vars(wandb)["Settings"](**settings_dict) if settings_dict else None

    params = {
        "project": wb_cfg.project,
        "entity": wb_cfg.entity,
        "dir": wb_cfg.dir,
        "id": resolved_run_id,
        "name": wb_cfg.name,
        "notes": wb_cfg.notes,
        "tags": tags or None,
        "config": run_cfg,
        "config_exclude_keys": wb_cfg.config_exclude_keys,
        "config_include_keys": wb_cfg.config_include_keys,
        "allow_val_change": wb_cfg.allow_val_change,
        "group": wb_cfg.group,
        "job_type": wb_cfg.job_type,
        "mode": wb_cfg.mode,
        "force": wb_cfg.force,
        "anonymous": wb_cfg.anonymous,
        "reinit": wb_cfg.reinit,
        "resume": resume_value,
        "resume_from": wb_cfg.resume_from,
        "fork_from": wb_cfg.fork_from,
        "save_code": wb_cfg.save_code,
        "sync_tensorboard": wb_cfg.sync_tensorboard,
        "monitor_gym": wb_cfg.monitor_gym,
        "settings": wandb_settings,
    }

    try:
        filtered_params = {k: v for k, v in params.items() if v is not None}
        if logger.isEnabledFor(10):  # DEBUG
            logger.debug(f"Initializing W&B with params: {filtered_params}")

        wandb_init = vars(wandb)["init"]
        run: Run = wandb_init(**filtered_params)
    except Exception as e:
        if isinstance(e, TypeError) and "object is not subscriptable" in str(e):
            return _new_disabled_session(wb_cfg, rank, is_primary)

        tb_lines: list[str] = traceback.format_exception(type(e), e, e.__traceback__)
        if len(tb_lines) <= 2:
            brief_tb: str = "".join(tb_lines).strip()
        else:
            brief_tb = f"{tb_lines[1].strip()}\n(... snip {len(tb_lines) - 2} lines ...)\n{tb_lines[-1].strip()}"
        logger.exception(f"Failed to initialize W&B: {brief_tb}")

        return _new_disabled_session(wb_cfg, rank, is_primary)

    session = WandbTrackingSession(
        enabled=True,
        mode=wb_cfg.mode,
        rank=rank,
        is_primary=is_primary,
        distributed_mode=distributed_mode,
        step_metric=wb_cfg.step_metric,
        run=run,
        profile=wb_cfg.profile,
        media_log_every_n_steps=wb_cfg.media_log_every_n_steps,
        table_log_every_n_steps=wb_cfg.table_log_every_n_steps,
        histogram_log_every_n_steps=wb_cfg.histogram_log_every_n_steps,
        max_metrics_per_log=wb_cfg.max_metrics_per_log,
        max_metric_key_length=wb_cfg.max_metric_key_length,
    )

    logger.info(f"W&B run initialized: name={run.name} id={run.id} rank={rank}")

    return session


@contextlib.contextmanager
def managed_tracking_session(
    cfg: SPARConfig | WandbConfig, *, stage: str | None = None, rank: int = 0
) -> Generator[WandbTrackingSession, None, None]:
    """Context manager that owns full init/finish lifecycle.

    Yields:
        WandbTrackingSession: Active tracking session scoped to the context.
    """
    session: WandbTrackingSession = init_tracking_session(cfg, stage=stage, rank=rank)
    token: Token[WandbTrackingSession | None] = _ACTIVE_TRACKING_SESSION.set(session)
    exit_code: int = 0

    try:
        yield session
    except Exception:
        exit_code = 1
        raise
    finally:
        _ACTIVE_TRACKING_SESSION.reset(token)
        finish_tracking_session(session, exit_code=exit_code)


def get_active_tracking_session() -> WandbTrackingSession | None:
    """Return current tracking session set by managed context."""
    return _ACTIVE_TRACKING_SESSION.get()


def set_active_tracking_session(session: WandbTrackingSession | None) -> Token[WandbTrackingSession | None]:
    """Set the active session token for nested run contexts."""
    return _ACTIVE_TRACKING_SESSION.set(session)


def reset_active_tracking_session(token: Token[WandbTrackingSession | None]) -> None:
    """Reset contextvar token returned by set_active_tracking_session."""
    _ACTIVE_TRACKING_SESSION.reset(token)


def setup_wandb_service() -> None:
    """Setup W&B service for improved reliability in distributed jobs."""
    try:
        if hasattr(wandb, "setup") and callable(wandb.setup):
            wandb.setup()
            logger.info("W&B service setup completed")
        else:
            logger.info("W&B service setup not available in this version")
    except Exception as e:
        logger.warning(f"Failed to setup W&B service: {e}")


def define_default_metrics(
    session: WandbTrackingSession, *, train_prefix: str = "train", val_prefix: str = "validation"
) -> None:
    """Define default metric axes for monotonic step alignment."""
    if not session.enabled or session.run is None:
        return

    try:
        session.run.define_metric(session.step_metric)
        session.run.define_metric(f"{train_prefix}/*", step_metric=session.step_metric)
        session.run.define_metric(f"{val_prefix}/*", step_metric=session.step_metric)
        session.run.define_metric("stage/*")
        session.run.define_metric("result/*")
        session.run.define_metric("search/*")
        session.run.define_metric("evaluation/*")
        session.run.define_metric("optuna/*")
    except Exception:
        logger.exception("Failed to define W&B metrics")


def finish_tracking_session(session: WandbTrackingSession | None = None, *, exit_code: int = 0) -> None:
    """Finish an explicit tracking session."""
    target: WandbTrackingSession | None = session or get_active_tracking_session()
    if target is None or not target.enabled or target.run is None:
        return

    try:
        target.run.finish(exit_code=exit_code)
        logger.info(f"W&B run finished: id={target.run.id} exit_code={exit_code}")
    except Exception:
        logger.exception("Failed to finish W&B run")


def finish_distributed_run(rank: int = 0, session: WandbTrackingSession | None = None) -> None:
    """Compatibility wrapper to finish run state in distributed entrypoints."""
    _ = rank
    finish_tracking_session(session)


def _resolve_session_and_payload(
    session_or_payload: WandbTrackingSession | MetricPayload, payload: MetricPayload | None
) -> tuple[WandbTrackingSession | None, MetricDictPayload]:
    if isinstance(session_or_payload, WandbTrackingSession):
        return session_or_payload, dict(payload) if payload is not None else {}

    return get_active_tracking_session(), dict(session_or_payload)


def _guard_metrics(session: WandbTrackingSession, metrics: MetricPayload) -> MetricDictPayload:
    if not metrics:
        return {}

    guarded: MetricDictPayload = {}
    for key, value in metrics.items():
        if len(key) > session.max_metric_key_length:
            raise ValueError(
                f"W&B metric key length {len(key)} exceeds configured maximum {session.max_metric_key_length}: {key!r}"
            )
        guarded[key] = value

    return guarded


def _metric_batches(session: WandbTrackingSession, metrics: MetricDictPayload) -> list[MetricDictPayload]:
    """Split metrics into bounded payloads without dropping entries.

    Args:
        session: Active W&B tracking session.
        metrics: Validated metric mapping.

    Returns:
        Ordered payloads containing every metric exactly once.
    """
    batch_size: int = max(1, session.max_metrics_per_log)
    if len(metrics) <= batch_size:
        return [metrics]

    items: list[tuple[str, MetricValue]] = list(metrics.items())
    return [dict(items[start : start + batch_size]) for start in range(0, len(items), batch_size)]


def log_metrics(
    session_or_metrics: WandbTrackingSession | MetricPayload,
    metrics: MetricPayload | None = None,
    *,
    step: int | None = None,
    commit: bool = True,
) -> None:
    """Log scalar metrics with explicit session ownership and optional commit batching.

    New API:
        log_metrics(session, metrics, step=..., commit=...)

    Backward-compatible API:
        log_metrics(metrics, step=...)
    """
    session: WandbTrackingSession | None
    payload: MetricDictPayload
    session, payload = _resolve_session_and_payload(session_or_metrics, metrics)
    if session is None or not session.enabled or session.run is None:
        return

    if step is not None:
        if step < session.last_step:
            logger.debug(f"Skipping non-monotonic W&B step={step} < last_step={session.last_step}")

            return

        payload.setdefault(session.step_metric, step)
        session.last_step = step

    guarded: MetricDictPayload = _guard_metrics(session, payload)
    if not guarded:
        return

    try:
        batches: list[MetricDictPayload] = _metric_batches(session, guarded)
        for index, batch in enumerate(batches):
            batch_commit: bool = commit and index == len(batches) - 1
            session.run.log(batch, step=step, commit=batch_commit)
    except Exception:
        logger.exception("Failed to log metrics")


def log_metrics_batched(session: WandbTrackingSession, batches: Iterable[MetricPayload], *, step: int | None) -> None:
    """Log multiple metric payloads in a single logical step using commit batching."""
    payloads: list[MetricPayload] = [batch for batch in batches if batch]
    if not payloads:
        return

    for batch in payloads[:-1]:
        log_metrics(session, batch, step=step, commit=False)
    log_metrics(session, payloads[-1], step=step, commit=True)


def _collect_stage_metrics(value: StageValue, prefix: str, output: MetricDictPayload) -> None:
    """Collect bounded scalar metrics from a stage result.

    Args:
        value: Stage result node to inspect.
        prefix: Metric prefix for the current node.
        output: Destination metric mapping.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text == "artifacts":
                continue
            child_prefix = f"{prefix}/{key_text}" if prefix else key_text
            _collect_stage_metrics(item, child_prefix, output)
        return
    if isinstance(value, (list, tuple)):
        last_item: StageValue | None = value[-1] if value else None
        if isinstance(last_item, (int, float, bool)) and all(isinstance(item, (int, float, bool)) for item in value):
            output[f"{prefix}/count"] = len(value)
            output[f"{prefix}/last"] = last_item
        return
    if isinstance(value, Tensor):
        if value.numel() == 1:
            output[prefix] = value.detach().item()
        return
    if isinstance(value, (int, float, bool, str)) or value is None:
        output[prefix] = value


def _stage_table_cell(value: StageValue) -> TableCell:
    """Normalize a stage result value for a bounded W&B table.

    Args:
        value: Candidate table cell value.

    Returns:
        A W&B-compatible scalar value, or ``None`` for nested payloads.
    """
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return None


def _collect_search_outcome(result: StageValue, output: MetricDictPayload) -> list[StageTablePayload] | None:
    """Collect aggregate metrics and a bounded table from a search result.

    Args:
        result: Stage result to inspect.
        output: Destination metric mapping.

    Returns:
        Search table payloads when the result matches the search contract, otherwise ``None``.
    """
    if not isinstance(result, Mapping):
        return None
    logs = result.get("logs")
    summary = result.get("summary")
    if not isinstance(logs, list) or not isinstance(summary, Mapping):
        return None
    overall = summary.get("overall")
    if not isinstance(overall, Mapping):
        return None

    for key, value in overall.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            output[f"search/{key}"] = value
    output["search/outcomes_total"] = len(logs)

    rows: list[list[TableCell]] = []
    for entry in logs[:_STAGE_TABLE_MAX_ROWS]:
        if not isinstance(entry, Mapping):
            continue
        rows.append([_stage_table_cell(entry.get(column)) for column in _SEARCH_TABLE_COLUMNS])
    output["search/outcomes_table_rows"] = len(rows)
    return [("search/outcomes", list(_SEARCH_TABLE_COLUMNS), rows)] if rows else []


def _collect_evaluation_outcome(result: StageValue, output: MetricDictPayload) -> list[StageTablePayload] | None:
    """Collect model-evaluation metrics and a bounded variation table.

    Args:
        result: Stage result to inspect.
        output: Destination metric mapping.

    Returns:
        Evaluation table payloads when the result matches the evaluator contract, otherwise ``None``.
    """
    if not isinstance(result, Mapping):
        return None
    overall_metrics: StageValue | None = result.get("overall_metrics")
    variation_metrics: StageValue | None = result.get("variation_metrics")
    episode_metrics: StageValue | None = result.get("episode_metrics")
    if not isinstance(overall_metrics, Mapping) or not isinstance(variation_metrics, Mapping):
        return None
    if not isinstance(episode_metrics, list):
        return None

    for key, value in overall_metrics.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            output[f"evaluation/{key}"] = value
    for key in ("total_episodes", "total_batches"):
        value = result.get(key)
        if isinstance(value, (int, float, bool)):
            output[f"evaluation/{key}"] = value
    output["evaluation/variations_total"] = len(variation_metrics)
    output["evaluation/episode_rows_total"] = len(episode_metrics)

    metric_names: set[str] = set()
    for metrics in variation_metrics.values():
        if isinstance(metrics, Mapping):
            metric_names.update(str(key) for key in metrics)
    ordered_metrics: list[str] = sorted(metric_names)[:32]
    columns: list[str] = ["variation", *ordered_metrics]
    rows: list[list[TableCell]] = []
    variation_keys = sorted(variation_metrics, key=str)[:_STAGE_TABLE_MAX_ROWS]
    for variation in variation_keys:
        metrics = variation_metrics[variation]
        if not isinstance(metrics, Mapping):
            continue
        rows.append([str(variation), *[_stage_table_cell(metrics.get(metric_name)) for metric_name in ordered_metrics]])
    output["evaluation/variation_table_rows"] = len(rows)
    return [("evaluation/variations", columns, rows)] if rows else []


def _collect_optuna_outcome(
    stage: str, result: StageValue, output: MetricDictPayload
) -> list[StageTablePayload] | None:
    """Collect Optuna summary metrics and bounded best-trial tables.

    Args:
        stage: Executed stage name.
        result: Stage result to inspect.
        output: Destination metric mapping.

    Returns:
        Optuna table payloads for Optuna stages, otherwise ``None``.
    """
    if not stage.startswith("optuna_") or not isinstance(result, Mapping):
        return None
    for key, value in result.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            output[f"optuna/{key}"] = value

    best_trials = result.get("best_trials")
    if not isinstance(best_trials, list):
        return []
    columns: list[str] = ["number", "value", "values"]
    rows: list[list[TableCell]] = []
    for trial in best_trials[:_STAGE_TABLE_MAX_ROWS]:
        if not isinstance(trial, Mapping):
            continue
        rows.append([_stage_table_cell(trial.get(column)) for column in columns])
    output["optuna/best_trial_table_rows"] = len(rows)
    return [("optuna/best_trials", columns, rows)] if rows else []


def _nested_config_value(config: Mapping[str, PathValue] | None, *keys: str) -> PathValue | None:
    """Return a nested configuration value without resolving or copying the config.

    Args:
        config: Plain configuration mapping.
        *keys: Mapping keys to traverse. At least one key is required to produce a value.

    Returns:
        The nested value when every key exists, otherwise ``None``.
    """
    if config is None or not keys:
        return None

    current: PathValue | None = config.get(keys[0])
    for key in keys[1:]:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _report_files(directory_value: PathValue | None) -> list[Path]:
    """List immediate JSON and CSV report files from an output directory.

    Args:
        directory_value: Configured report directory.

    Returns:
        Existing report files in deterministic path order.
    """
    if not isinstance(directory_value, str):
        return []
    directory = Path(directory_value)
    if not directory.is_dir():
        return []
    try:
        return sorted(
            (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".csv", ".json"}),
            key=str,
        )
    except OSError:
        logger.exception("Failed to enumerate stage report files in %s", directory)
        return []


def _stage_artifact_files(stage: str, config: Mapping[str, PathValue] | None) -> list[Path]:
    """Resolve bounded, authoritative result files for a completed stage.

    Args:
        stage: Executed stage name.
        config: Plain stage configuration mapping.

    Returns:
        Existing report files suitable for one W&B artifact.
    """
    if config is None:
        return []
    if stage in {"search_qstar", "search_ucs", "search_gbfs", "ucs", "gbfs", "run_ucs_search", "run_gbfs_search"}:
        results_dir = _nested_config_value(config, "search", "results_dir")
        if isinstance(results_dir, str):
            results_file = Path(results_dir) / "results.json"
            return [results_file] if results_file.is_file() else []
        return []
    if stage == "test_model":
        return _report_files(_nested_config_value(config, "save_paths", "metrics_dir"))
    if not stage.startswith("optuna_"):
        return []

    study_name = _nested_config_value(config, "optuna", "study", "study_name")
    output_root = _nested_config_value(config, "optuna", "runtime", "output_root")
    if not isinstance(study_name, str) or not isinstance(output_root, str):
        return []
    study_dir = Path(output_root) / study_name
    if stage == "optuna_study":
        return [path for path in (study_dir / "study_summary.json", study_dir / "trials.csv") if path.is_file()]
    if stage == "optuna_analyze":
        analysis_dir = _nested_config_value(config, "optuna", "analysis", "output_dir")
        directory = Path(analysis_dir) if isinstance(analysis_dir, str) and analysis_dir else study_dir / "analysis"
        return [path for path in (directory / "analysis.json", directory / "trials.csv") if path.is_file()]
    replay_dir = _nested_config_value(config, "optuna", "replay", "output_dir")
    directory = Path(replay_dir) if isinstance(replay_dir, str) and replay_dir else study_dir / "replay"
    replay_index = directory / "replay_index.json"
    return [replay_index] if replay_index.is_file() else []


def log_stage_outcome(
    session: WandbTrackingSession,
    stage: str,
    result: StageValue,
    *,
    elapsed_seconds: float,
    succeeded: bool,
    error_type: str | None = None,
    config: Mapping[str, PathValue] | None = None,
) -> None:
    """Log one stage outcome with no work when tracking is disabled.

    Args:
        session: Explicit W&B tracking session.
        stage: Executed stage name.
        result: Stage result returned by the handler.
        elapsed_seconds: Stage execution time in seconds.
        succeeded: Whether the handler completed without raising an exception.
        error_type: Exception class name for a failed stage.
        config: Plain stage configuration used to resolve authoritative result files.
    """
    if not session.enabled or session.run is None:
        return
    if session.distributed_mode == "shared" and not session.is_primary:
        return

    base_payload: MetricDictPayload = {
        "stage/name": stage,
        "stage/elapsed_seconds": elapsed_seconds,
        "stage/succeeded": succeeded,
    }
    if error_type is not None:
        base_payload["stage/error_type"] = error_type
    payload: MetricDictPayload = dict(base_payload)
    tables: list[StageTablePayload] | None = _collect_search_outcome(result, payload)
    if tables is None:
        tables = _collect_evaluation_outcome(result, payload)
    if tables is None:
        tables = _collect_optuna_outcome(stage, result, payload)
    if tables is None:
        tables = []
        _collect_stage_metrics(result, "result", payload)

    for table_name, columns, rows in tables:
        log_table(session, table_name, columns, rows, commit=False)
    try:
        log_metrics(session, payload)
    except ValueError as error:
        logger.warning("Stage result metrics were omitted from W&B: %s", error)
        fallback_payload: MetricDictPayload = dict(base_payload)
        fallback_payload["stage/result_metrics_omitted"] = len(payload) - len(base_payload)
        log_metrics(session, fallback_payload)

    if succeeded:
        artifact_files = _stage_artifact_files(stage, config)
        if artifact_files:
            log_files_artifact(
                session,
                name=f"{stage}-results",
                artifact_type="results",
                files=artifact_files,
                metadata={"stage": stage, "elapsed_seconds": elapsed_seconds},
            )


def _should_log_payload(step: int | None, every_n_steps: int) -> bool:
    if every_n_steps == 0:
        return True

    if every_n_steps < 0:
        return False

    if step is None:
        return True

    return step % every_n_steps == 0


def log_table(
    session: WandbTrackingSession,
    table_name: str,
    columns: Sequence[ColumnKey],
    data: Sequence[InputRow],
    step: int | None = None,
    *,
    commit: bool = True,
) -> bool:
    """Log a table when cadence settings permit it.

    Args:
        session: W&B tracking session that owns the run.
        table_name: Name used for the logged table.
        columns: Column names in display order.
        data: Table rows in display order.
        step: Optional training or evaluation step.
        commit: Whether this log entry advances the W&B step.

    Returns:
        ``True`` when the table is logged, otherwise ``False``.
    """
    if not session.enabled or session.run is None:
        return False

    if not _should_log_payload(step, session.table_log_every_n_steps):
        return False

    try:
        table: Table = wandb.Table(columns=list(columns), data=list(data))
        session.run.log({table_name: table}, step=step, commit=commit)
    except Exception:
        logger.exception("Failed to log table")
        return False
    return True


def log_image(
    session: WandbTrackingSession,
    image_name: str,
    image_data: ImageDataOrPathType,
    caption: str | None = None,
    step: int | None = None,
    **kwargs: ImageKwargValue,
) -> None:
    """Log image payload with cadence gating."""
    if not session.enabled or session.run is None:
        return

    if not _should_log_payload(step, session.media_log_every_n_steps):
        return

    try:
        wandb_image = vars(wandb)["Image"]
        image: Image = wandb_image(image_data, caption=caption, **kwargs)
        session.run.log({image_name: image}, step=step)
    except Exception:
        logger.exception("Failed to log image")


def log_histogram(
    session: WandbTrackingSession, histogram_name: str, values: HistogramValues, step: int | None = None
) -> None:
    """Log histogram payload with cadence gating."""
    if not session.enabled or session.run is None:
        return

    if not _should_log_payload(step, session.histogram_log_every_n_steps):
        return

    try:
        wandb_histogram = vars(wandb)["Histogram"]
        session.run.log({histogram_name: wandb_histogram(values)}, step=step)
    except Exception:
        logger.exception("Failed to log histogram")


def _collect_model_weight_histograms(model: nn.Module, prefix: str) -> dict[str, Histogram]:
    histograms: dict[str, Histogram] = {}
    wandb_histogram = vars(wandb)["Histogram"]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        histograms[f"{prefix}/{name}"] = wandb_histogram(param.data.detach().cpu().numpy().tolist())

    return histograms


def log_model_weights_histograms(
    session: WandbTrackingSession, model: nn.Module, prefix: str = "weights", step: int | None = None
) -> None:
    """Log model parameter histograms."""
    if not session.enabled or session.run is None:
        return

    if not _should_log_payload(step, session.histogram_log_every_n_steps):
        return

    try:
        histograms: dict[str, Histogram] = _collect_model_weight_histograms(model, prefix)
        if histograms:
            session.run.log(histograms, step=step)
    except Exception:
        logger.exception("Failed to log model weight histograms")


def watch_model(
    session: WandbTrackingSession,
    models: nn.Module | list[nn.Module],
    criterion: nn.Module | Callable[..., Tensor] | None = None,
    watch_log: Literal["gradients", "parameters", "all"] | None = "gradients",
    log_freq: int = 1000,
    idx: int | None = None,
    log_graph: bool = False,
) -> None:
    """Watch model(s) via wandb.watch."""
    if not session.enabled or session.run is None:
        return

    try:
        models_list: list[nn.Module] = models if isinstance(models, list) else [models]
        for i, model in enumerate(models_list):
            wandb.watch(
                model,
                criterion=criterion,
                log=watch_log,
                log_freq=log_freq,
                idx=idx if idx is not None else i,
                log_graph=log_graph,
            )
    except Exception:
        logger.exception("Failed to watch model(s)")


def unwatch_model(session: WandbTrackingSession, models: nn.Module | list[nn.Module] | None = None) -> None:
    """Remove model hooks created by wandb.watch."""
    if not session.enabled:
        return

    try:
        wandb.unwatch(models)
    except Exception:
        logger.exception("Failed to unwatch model(s)")


def log_video(
    session: WandbTrackingSession,
    video_name: str,
    video_frames: str | Path | NDArray[np.uint8 | np.float32 | np.float64 | np.int32 | np.int64] | TextIO | BytesIO,
    fps: int = 4,
    format: Literal["gif", "mp4", "webm", "ogg"] = "gif",
    step: int | None = None,
) -> None:
    """Log video payload with cadence gating."""
    if not session.enabled or session.run is None:
        return

    if not _should_log_payload(step, session.media_log_every_n_steps):
        return

    try:
        video: Video = wandb.Video(video_frames, fps=fps, format=format)
        session.run.log({video_name: video}, step=step)

        if isinstance(video_frames, Sized):
            logger.debug(f"Logged video '{video_name}' with {len(video_frames)} frames")
    except Exception:
        logger.exception("Failed to log video")


def log_environment_render(
    session: WandbTrackingSession, env_name: str, env: RenderableEnvironment, step: int | None = None
) -> None:
    """Log environment render output as image."""
    if not session.enabled or session.run is None:
        return

    try:
        if hasattr(env, "render"):
            render_output = env.render()
            if render_output is not None:
                log_image(session, f"{env_name}_render", render_output, step=step)
    except Exception:
        logger.exception("Failed to render environment")


def log_files_artifact(
    session: WandbTrackingSession,
    *,
    name: str,
    artifact_type: str,
    files: Sequence[Path],
    metadata: ArtifactMetadata | None = None,
) -> Artifact | None:
    """Log existing report files as one versioned W&B artifact.

    Args:
        session: Explicit W&B tracking session.
        name: Artifact collection name.
        artifact_type: Artifact type such as ``results``.
        files: Existing files to include.
        metadata: Optional artifact metadata.

    Returns:
        The logged artifact, or ``None`` when disabled or no files exist.
    """
    if not session.enabled or session.run is None:
        return None
    existing_files: list[Path] = [path for path in files if path.is_file()]
    if not existing_files:
        return None

    normalized_name = "".join(character if character.isalnum() or character in "-_." else "-" for character in name)
    try:
        artifact: Artifact = wandb.Artifact(normalized_name, type=artifact_type, metadata=metadata)
        for path in existing_files:
            artifact.add_file(str(path), name=path.name)
        session.run.log_artifact(artifact)
    except Exception:
        logger.exception("Failed to log files artifact")
        return None
    return artifact


def log_model_artifact(
    session: WandbTrackingSession,
    model: nn.Module,
    name: str,
    artifact_type: str = "model",
    checkpoint_path: str | None = None,
    metadata: ArtifactMetadata | None = None,
) -> Artifact | None:
    """Log a model artifact from a checkpoint path or in-memory module."""
    if not session.enabled or session.run is None:
        return None

    def prepare_artifact() -> Artifact:
        artifact: Artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)

        if checkpoint_path:
            artifact.add_file(checkpoint_path)
        else:
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                torch_save(model.state_dict(), tmp_path)
                artifact.add_file(tmp_path, name="model.pth")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        return artifact

    try:
        artifact: Artifact = prepare_artifact()
        session.run.log_artifact(artifact)
    except Exception:
        logger.exception("Failed to log model artifact")

        return None

    else:
        return artifact


def link_model_to_registry(
    session: WandbTrackingSession, model_artifact: Artifact, registry_name: str, version_name: str = "latest"
) -> Artifact | None:
    """Link an artifact into W&B registry from active run context."""
    if not session.enabled or session.run is None:
        return None

    try:
        model_artifact.link(f"{session.run.entity}/{registry_name}", aliases=[version_name])
    except Exception:
        logger.exception("Failed to link artifact to registry")

        return None

    else:
        return model_artifact


def create_registry(registry_name: str, description: str = "") -> Registry | None:
    """Create a model registry via public API."""
    try:
        api: Api = wandb.Api()

        return api.create_registry(registry_name, description=description, visibility="organization")

    except Exception:
        logger.exception("Failed to create registry")

        return None


def search_registry(query: str, registry_name: str | None = None) -> Artifacts | list[Artifact | None]:
    """Search registry artifacts."""
    try:
        api: Api = wandb.Api()

        return api.artifact_type(registry_name).collection(query).artifacts() if registry_name else []

    except Exception:
        logger.exception("Failed to search registry")

        return []


def get_api_client() -> Api | None:
    """Get W&B API client."""
    try:
        return wandb.Api()

    except Exception:
        logger.exception("Failed to get W&B API client")

        return None


def configure_sweep(
    session_or_sweep_config: WandbTrackingSession | SweepConfigPayload,
    sweep_config: SweepConfigPayload | None = None,
    *,
    project_name: str | None = None,
    entity_name: str | None = None,
) -> str:
    """Configure a sweep with explicit or active session context."""
    session: WandbTrackingSession | None
    config: SweepConfigPayload

    if isinstance(session_or_sweep_config, WandbTrackingSession):
        session = session_or_sweep_config
        config = sweep_config or {}
    else:
        session = get_active_tracking_session()
        config = session_or_sweep_config

    if not config:
        return ""

    if session is not None and session.run is not None:
        if not project_name:
            project_name = session.run.project
        if not entity_name:
            entity_name = session.run.entity

    try:
        sweep_id: str = wandb.sweep(config, project=project_name, entity=entity_name)
    except Exception:
        logger.exception("Failed to configure W&B sweep")

        return ""

    else:
        logger.info(f"Created W&B sweep: {sweep_id}")

        return sweep_id


def finish_run(session: WandbTrackingSession | None = None) -> None:
    """Compatibility wrapper for run finalization."""
    finish_tracking_session(session)


# ---------------------------------------------------------------------------
# Backward-compatible wrappers
# ---------------------------------------------------------------------------


def init_wandb(cfg: SPARConfig | WandbConfig, rank: int = 0) -> Run | None:
    """Compatibility wrapper around init_tracking_session.

    This sets the active session token to support legacy global-style utilities.
    """
    session: WandbTrackingSession = init_tracking_session(cfg=cfg, stage=_resolve_stage_name(cfg, None), rank=rank)
    _ACTIVE_TRACKING_SESSION.set(session)

    return session.run
