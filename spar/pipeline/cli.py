"""Command-line interface for the SPAR framework."""

from __future__ import annotations

from importlib import import_module
from logging import getLogger
import os
from time import perf_counter
from typing import TYPE_CHECKING

import hydra

from spar.utils.config_utils.hydra_help_patch import patch_hydra_get_help_lazy_fields
from spar.utils.config_utils.misc import register_omega_conf_resolvers

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from logging import Logger
    from types import ModuleType
    from typing import Literal, Protocol

    from omegaconf import DictConfig
    from rich.align import Align
    from rich.panel import Panel
    from rich.table import Table

    from spar.environments import ABCEnvironment, ABCState
    from spar.utils.config_utils.config_schema import SPARConfig
    from spar.utils.config_utils.config_validation import ConfigValidationError
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession

    class _ValidateConfigFn(Protocol):
        def __call__(self, cfg: DictConfig, stage: str) -> SPARConfig: ...

    class _SaveResolvedConfigFn(Protocol):
        def __call__(self, cfg: DictConfig) -> None: ...

    class _GetEnvironmentFn(Protocol):
        def __call__(self, env_name: str) -> ABCEnvironment[ABCState]: ...

    class _RunStageFn(Protocol):
        def __call__(
            self,
            stage: str,
            env: ABCEnvironment[ABCState],
            cfg: SPARConfig,
            tracking: WandbTrackingSession | None = None,
        ) -> None: ...

    class _LogSystemStatsFn(Protocol):
        def __call__(
            self,
            description: str = "System Info",
            *,
            include_system_summary: bool = True,
            include_process_stats: bool = True,
            include_resource_allocation: bool = True,
            include_gpu_stats: bool = True,
            extra_packages: list[str] | None = None,
        ) -> None: ...

    class _DefineDefaultMetricsFn(Protocol):
        def __call__(self, session: WandbTrackingSession) -> None: ...

    class _ManagedTrackingSessionFn(Protocol):
        def __call__(
            self, cfg: SPARConfig, *, stage: str | None = None, rank: int = 0
        ) -> AbstractContextManager[WandbTrackingSession]: ...


logger: Logger = getLogger(__name__)

# ConfigStore registration happens via module import side effect.
import_module("spar.utils.config_utils.config_store")
patch_hydra_get_help_lazy_fields()
register_omega_conf_resolvers()


def _persist_resolved_config(save_resolved_config: _SaveResolvedConfigFn, cfg: DictConfig) -> None:
    """Persist resolved Hydra config when possible."""
    try:
        save_resolved_config(cfg)
    except Exception:
        logger.debug("Could not persist resolved config to Hydra outputs (non-fatal)")


def _format_elapsed_time(elapsed_time: float) -> str:
    """Format elapsed stage time as DD-HH:MM:SS-ms."""
    return (
        f"{int(elapsed_time // 86400):02d}-"
        f"{int(elapsed_time % 86400 // 3600):02d}:"
        f"{int(elapsed_time % 3600 // 60):02d}:"
        f"{int(elapsed_time % 60):02d}-"
        f"{int((elapsed_time % 1) * 1000):03d}"
    )


def _run_stage_with_timing(
    stage: str,
    env: ABCEnvironment[ABCState],
    cfg: SPARConfig,
    tracking: WandbTrackingSession | None,
    log_system_stats: _LogSystemStatsFn,
    run_stage: _RunStageFn,
) -> str:
    """Run a configured stage and return its formatted elapsed time."""
    log_system_stats(description=f"SPAR CLI Execution - {stage}")
    start_time: float = perf_counter()

    run_stage(stage, env, cfg, tracking)

    elapsed_time: float = perf_counter() - start_time
    return _format_elapsed_time(elapsed_time)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run the Hydra CLI with lazy stage imports and Pydantic validation."""
    align_cls: type[Align] = import_module("rich.align").Align
    panel_cls: type[Panel] = import_module("rich.panel").Panel
    table_cls: type[Table] = import_module("rich.table").Table

    config_validation_module: ModuleType = import_module("spar.utils.config_utils.config_validation")
    config_validation_error_type: type[ConfigValidationError] = config_validation_module.ConfigValidationError
    validate_config: _ValidateConfigFn = config_validation_module.validate_config
    save_resolved_config: _SaveResolvedConfigFn = import_module("spar.utils.config_utils.misc").save_resolved_config
    get_environment: _GetEnvironmentFn = import_module("spar.utils.env_utils").get_environment
    run_stage: _RunStageFn = import_module("spar.utils.import_utils.stage_importer").run_stage
    log_system_stats: _LogSystemStatsFn = import_module("spar.utils.log_utils.system_stats_logger").log_system_stats
    wandb_logger_module: ModuleType = import_module("spar.utils.log_utils.wandb_logger")
    define_default_metrics: _DefineDefaultMetricsFn = wandb_logger_module.define_default_metrics
    managed_tracking_session: _ManagedTrackingSessionFn = wandb_logger_module.managed_tracking_session

    # Extract stage early for validation
    stage: str = cfg.get("stage", "default")
    env_name: str = cfg.get("env", {}).get("name", "unknown")

    # Summary table of high-level settings
    table: Table = table_cls.grid(padding=(0, 1))
    table.add_column(justify="left", style="bold cyan")
    table.add_column(style="bright_white")
    summary_items: tuple[tuple[Literal["Stage", "Environment"], str], ...] = (
        ("Stage", stage),
        ("Environment", env_name),
    )
    for name, value in summary_items:
        table.add_row(name, value)
    logger.info(panel_cls(table, title="[bold blue]Stage Summary[/bold blue]", border_style="blue", width=120))

    # Validate configuration using Pydantic
    validated_cfg: SPARConfig
    try:
        # Persist resolved config YAML into the Hydra run directory for reproducibility
        _persist_resolved_config(save_resolved_config, cfg)
        validated_cfg = validate_config(cfg, stage)
        logger.info("[bold green dim]Configuration validation passed[/bold green dim]\n")

    except Exception as error:
        if isinstance(error, config_validation_error_type):
            logger.exception("Configuration validation failed")
            validation_errors: list[str] = getattr(error, "validation_errors", [])
            if validation_errors:
                logger.exception("Validation errors:")
                for validation_error in validation_errors:
                    logger.exception(f"  - {validation_error}")

        raise

    rank: int = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    with managed_tracking_session(validated_cfg, stage=stage, rank=rank) as tracking:
        if tracking.enabled:
            define_default_metrics(tracking)

        env: ABCEnvironment[ABCState] = get_environment(env_name)

        try:
            # Run the specified stage
            elapsed_time_str: str = _run_stage_with_timing(
                stage, env, validated_cfg, tracking, log_system_stats, run_stage
            )

            logger.info(
                panel_cls(
                    align_cls.center(
                        f"[bold green]Stage '{stage}' completed successfully in {elapsed_time_str}[/bold green]"
                    ),
                    border_style="green",
                    padding=(1, 2),
                    width=120,
                )
            )

        except Exception:
            logger.exception(
                panel_cls(
                    f"[bold red]Execution failed[/bold red]\n\n[dim]Stage: {stage}[/dim]",
                    border_style="red",
                    padding=(1, 2),
                    width=120,
                )
            )
            raise


if __name__ == "__main__":
    main()
