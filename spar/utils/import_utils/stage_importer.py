from __future__ import annotations

from dataclasses import asdict
from importlib import import_module
import inspect
from logging import DEBUG, getLogger
import sys
from time import perf_counter
import traceback
from typing import TYPE_CHECKING

from omegaconf import OmegaConf
from rich.panel import Panel

from .lazy_importer import LazyImporter

if TYPE_CHECKING:
    from collections.abc import Callable
    from inspect import Parameter, Signature
    from logging import Logger
    from types import MappingProxyType
    from typing import TypeAlias

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.optuna.types import PathValue, ReporterPayload, StageValue
    from spar.utils.config_utils.config_schema import SPARConfig
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession

    from .lazy_importer import CacheInfo, CompatibilityInfo, DirectoryLister, ImportMapping

    StageReturn: TypeAlias = StageValue
    StageReporter: TypeAlias = Callable[[ReporterPayload], None]
    StageRuntimeArg: TypeAlias = WandbTrackingSession | StageReporter | None
    StageCallable: TypeAlias = Callable[..., StageReturn]
    CLIImportInfo: TypeAlias = dict[
        str,
        LazyImporter[StageCallable]
        | list[str]
        | dict[str, str]
        | dict[str, tuple[str, str]]
        | CacheInfo
        | CompatibilityInfo
        | bool,
    ]


logger: Logger = getLogger(__name__)


__all__: list[str] = [
    "STAGE_FUNCTIONS",
    "STAGE_IMPORT_MAPPING",
    "CLIImportInfo",
    "StageCallable",
    "clear_cli_cache",
    "cli_lazy_importer",
    "create_stage_function",
    "get_cli_import_info",
    "run_stage",
    "validate_cli_imports",
]

# Lazy imports mapping for CLI stage functions
STAGE_IMPORT_MAPPING: ImportMapping = {
    "generate_data": ("spar.data.generator", "generate_data"),
    "generate_search_data": ("spar.data.search_data_generator", "generate_search_data"),
    "create_sweep": ("spar.utils.log_utils.wandb_sweeps", "run_create_sweep"),
    "train_world_model": ("spar.training", "train_world_model"),
    "train_alignment_model": ("spar.training", "train_alignment_model"),
    "test_model": ("spar.testing.model_tester.test_runner", "run_test"),
    "encode_offline_data": ("spar.data.encoders", "run_encode_offline_data"),
    "train_heuristic": ("spar.training.dqn_trainer.training_runner", "train_heuristic"),
    "qstar_run_search": ("spar.search.qstar", "run_search"),
    "ucs_run_search": ("spar.search.ucs", "run_search"),
    "gbfs_run_search": ("spar.search.gbfs", "run_search"),
    "visualize_unsolved_qstar": ("spar.scripts.visualize_unsolved_qstar", "visualize"),
    "bitwise_eq_report": ("spar.scripts.bitwise_eq_report", "run_bitwise_eq_report"),
    "alignment_encoder_match_report": (
        "spar.scripts.alignment_encoder_match_report",
        "run_alignment_encoder_match_report",
    ),
    "qstar_results_to_latex": ("spar.scripts.qstar_results_to_latex", "export_qstar_results_to_latex"),
    "run_plotter": ("spar.scripts.plot_test_results", "run_mse_plotter"),
    "run_mse_plotter": ("spar.scripts.plot_test_results", "run_mse_plotter"),
    "process_image": ("spar.testing.image_processor", "process_image"),
    "optuna_run_study": ("spar.optuna", "run_study"),
    "optuna_analyze_study": ("spar.optuna", "analyze_study"),
    "optuna_replay_trials": ("spar.optuna", "replay_trials"),
}

# Create lazy importer
cli_lazy_importer: LazyImporter[StageCallable] = LazyImporter(
    imports=STAGE_IMPORT_MAPPING,
    module_name=__name__,
    cache_enabled=True,
    thread_safe=False,
    debug_mode=False,
    validate_imports=True,
)


def create_stage_function(func_name: str) -> StageCallable:
    """Create a stage function wrapper that uses the lazy importer.

    Args:
        func_name: The name of the function to import and wrap.

    Returns:
        A callable that takes an environment and configuration, and calls the imported function.
    """

    def stage_wrapper(
        env: ABCEnvironment[ABCState],
        cfg: SPARConfig,
        tracking: StageRuntimeArg = None,
        reporter: StageRuntimeArg = None,
    ) -> StageReturn:
        """Wrapper that lazily imports and calls the stage function."""
        stage_func: StageCallable = cli_lazy_importer.get_attr(func_name)
        signature: Signature = inspect.signature(stage_func)
        parameters: MappingProxyType[str, Parameter] = signature.parameters
        kwargs: dict[str, StageRuntimeArg] = {}

        if "tracking" in parameters:
            kwargs["tracking"] = tracking
        if "reporter" in parameters:
            kwargs["reporter"] = reporter
        if kwargs:
            return stage_func(env, cfg, **kwargs)

        if len(parameters) >= 3:
            if len(parameters) >= 4:
                return stage_func(env, cfg, tracking, reporter)
            return stage_func(env, cfg, tracking)

        return stage_func(env, cfg)

    stage_wrapper.__name__ = stage_wrapper.__qualname__ = func_name
    return stage_wrapper


# Mapping of CLI stages to their handlers
STAGE_FUNCTIONS: dict[str, StageCallable] = {
    "gen_data": create_stage_function("generate_data"),
    "gen_search_data": create_stage_function("generate_search_data"),
    "create_sweep": create_stage_function("create_sweep"),
    "train_world_model": create_stage_function("train_world_model"),
    "train_alignment_model": create_stage_function("train_alignment_model"),
    "test_model": create_stage_function("test_model"),
    "encode_offline": create_stage_function("encode_offline_data"),
    "encode_offline_data": create_stage_function("encode_offline_data"),
    "train_heuristic": create_stage_function("train_heuristic"),
    "train_heur": create_stage_function("train_heuristic"),
    "train_heur_model": create_stage_function("train_heuristic"),
    "search_qstar": create_stage_function("qstar_run_search"),
    "search_ucs": create_stage_function("ucs_run_search"),
    "ucs": create_stage_function("ucs_run_search"),
    "run_ucs_search": create_stage_function("ucs_run_search"),
    "search_gbfs": create_stage_function("gbfs_run_search"),
    "gbfs": create_stage_function("gbfs_run_search"),
    "run_gbfs_search": create_stage_function("gbfs_run_search"),
    "visualize_unsolved_qstar": create_stage_function("visualize_unsolved_qstar"),
    "bitwise_eq_report": create_stage_function("bitwise_eq_report"),
    "alignment_encoder_match_report": create_stage_function("alignment_encoder_match_report"),
    "qstar_results_to_latex": create_stage_function("qstar_results_to_latex"),
    "plotter": create_stage_function("run_plotter"),
    "mse_plotter": create_stage_function("run_mse_plotter"),
    "process_image": create_stage_function("process_image"),
    "optuna_study": create_stage_function("optuna_run_study"),
    "optuna_analyze": create_stage_function("optuna_analyze_study"),
    "optuna_replay": create_stage_function("optuna_replay_trials"),
}


def run_stage(
    stage: str,
    env: ABCEnvironment[ABCState],
    cfg: SPARConfig,
    tracking: StageRuntimeArg = None,
    reporter: StageRuntimeArg = None,
) -> StageReturn:
    """Call the handler for stage."""
    if stage not in STAGE_FUNCTIONS:
        # Get available stages for better error reporting
        available_stages: list[str] = sorted(STAGE_FUNCTIONS.keys())

        logger.info(
            Panel(
                f"[bold red]Error: Unknown stage '{stage}'[/bold red]\n\n"
                f"[bold cyan]Available stages:[/bold cyan]\n"
                f"{', '.join(available_stages)}",
                border_style="red",
                padding=(1, 2),
                width=120,
            )
        )
        sys.exit(1)

    if logger.isEnabledFor(DEBUG):
        cache_info: CacheInfo = cli_lazy_importer.get_cache_info()
        logger.debug(f"Lazy importer cache info: {cache_info}")
        # Display compatibility information in debug mode
        compat_info: CompatibilityInfo = cli_lazy_importer.get_import_compatibility_info()
        logger.debug(
            "Lazy importer compatibility: "
            f"type_checking={compat_info['supports_type_checking']}, "
            f"star_import={compat_info['supports_star_import']}, "
            f"cached={compat_info['cached_imports']}/{compat_info['total_imports']}"
        )

        # Use dataclasses.asdict for Pydantic dataclasses
        cfg_dict: dict[str, PathValue] = asdict(cfg)

        # Convert validated config to YAML for display
        cfg_yaml: str = OmegaConf.to_yaml(OmegaConf.create(cfg_dict), resolve=True).strip()

        logger.debug(
            Panel(
                f"[bold blue]Running stage: {stage}[/bold blue]\n\n"
                f"[bold cyan]Stage Configuration:[/bold cyan]\n"
                f"[dim]{cfg_yaml}[/dim]",
                title="[bold]Configuration Details[/bold]",
                border_style="blue dim",
                padding=(1, 2),
                width=120,
            )
        )

    # Validate the stage function can be imported
    stage_func: StageCallable = STAGE_FUNCTIONS[stage]
    try:
        # Attempt to import the stage function to reveal any errors
        cli_lazy_importer.get_attr(getattr(stage_func, "__name__", type(stage_func).__name__))

    except Exception:
        # Display detailed traceback for import errors
        tb: str = traceback.format_exc()
        logger.exception(
            Panel(
                f"[bold red]Import Error: Stage '{stage}' function cannot be imported[/bold red]\n\n"
                f"[yellow]Error details:[/yellow]\n{tb}",
                border_style="red",
                padding=(1, 2),
                width=120,
            )
        )
        sys.exit(1)

    tracking_enabled: bool = tracking is not None and not callable(tracking) and tracking.enabled
    start_time: float = perf_counter() if tracking_enabled else 0.0
    try:
        result: StageReturn = stage_func(env, cfg, tracking, reporter)
    except Exception as error:
        if tracking_enabled and tracking is not None and not callable(tracking):
            log_stage_outcome = import_module("spar.utils.log_utils.wandb_logger").log_stage_outcome
            log_stage_outcome(
                tracking,
                stage,
                None,
                elapsed_seconds=perf_counter() - start_time,
                succeeded=False,
                error_type=type(error).__name__,
            )
        raise
    if tracking_enabled and tracking is not None and not callable(tracking):
        log_stage_outcome = import_module("spar.utils.log_utils.wandb_logger").log_stage_outcome
        outcome_config: dict[str, PathValue] = asdict(cfg)
        log_stage_outcome(
            tracking, stage, result, elapsed_seconds=perf_counter() - start_time, succeeded=True, config=outcome_config
        )

    # Log final cache statistics
    if logger.isEnabledFor(level=DEBUG):
        final_cache_info: CacheInfo = cli_lazy_importer.get_cache_info()
        cached_attrs: int = final_cache_info["cached_attributes"]
        total_attrs: int = final_cache_info["total_attributes"]
        cache_ratio: float = final_cache_info["cache_hit_ratio"]
        logger.debug(f"Lazy import cache usage: {cached_attrs}/{total_attrs} ({cache_ratio:.1%})")
    return result


def get_cli_import_info() -> CLIImportInfo:
    """Return the CLI importer mappings, cache state, and compatibility fields.

    Returns:
        Structured importer diagnostics.
    """
    return {
        "importer": cli_lazy_importer,
        "available_stages": sorted(STAGE_FUNCTIONS.keys()),
        "available_imports": cli_lazy_importer.get_available_attributes(),
        "cache_info": cli_lazy_importer.get_cache_info(),
        "compatibility_info": cli_lazy_importer.get_import_compatibility_info(),
        "import_mappings": cli_lazy_importer.get_module_attributes(),
        "type_stubs": cli_lazy_importer.get_type_stubs(),
        "supports_star_import": cli_lazy_importer.supports_star_import(),
    }


def validate_cli_imports() -> dict[str, bool]:
    """Attempt to import every registered CLI stage.

    Returns:
        Mapping from stage name to import success.
    """
    import_validation: dict[str, bool] = cli_lazy_importer.validate_all_imports()
    if logger.isEnabledFor(DEBUG):
        failed_imports: list[str] = [name for name, success in import_validation.items() if not success]
        if failed_imports:
            logger.warning(f"Some imports may fail: {failed_imports}")
    return import_validation


def clear_cli_cache() -> None:
    """Remove all cached CLI stage attributes."""
    cli_lazy_importer.clear_cache()


__getattr__: Callable[[str], StageCallable] = cli_lazy_importer.get_attr
__dir__: DirectoryLister = cli_lazy_importer.get_dir
