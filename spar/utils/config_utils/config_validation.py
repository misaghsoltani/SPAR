"""Validate composed SPAR configurations with stage-specific Pydantic schemas."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from omegaconf import OmegaConf, SCMode
from pydantic import TypeAdapter, ValidationError

from spar.utils.import_utils.lazy_importer import LazyImporter

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from .config_schema import SPARConfig

# Configuration class imports mapping for lazy loading
_CONFIG_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseSPARConfig": ("spar.utils.config_utils.config_schema", "BaseSPARConfig"),
    "CreateSweepSPARConfig": ("spar.utils.config_utils.config_schema", "CreateSweepSPARConfig"),
    "EncodeOfflineDataSPARConfig": ("spar.utils.config_utils.config_schema", "EncodeOfflineDataSPARConfig"),
    "GenDataSPARConfig": ("spar.utils.config_utils.config_schema", "GenDataSPARConfig"),
    "GenSearchDataSPARConfig": ("spar.utils.config_utils.config_schema", "GenSearchDataSPARConfig"),
    "PlotterSPARConfig": ("spar.utils.config_utils.config_schema", "PlotterSPARConfig"),
    "MSEPlotterSPARConfig": ("spar.utils.config_utils.config_schema", "MSEPlotterSPARConfig"),
    "OptunaAnalyzeSPARConfig": ("spar.utils.config_utils.config_schema", "OptunaAnalyzeSPARConfig"),
    "OptunaReplaySPARConfig": ("spar.utils.config_utils.config_schema", "OptunaReplaySPARConfig"),
    "OptunaStudySPARConfig": ("spar.utils.config_utils.config_schema", "OptunaStudySPARConfig"),
    "SearchGBFSSPARConfig": ("spar.utils.config_utils.config_schema", "SearchGBFSSPARConfig"),
    "SearchQStarSPARConfig": ("spar.utils.config_utils.config_schema", "SearchQStarSPARConfig"),
    "SearchUCSSPARConfig": ("spar.utils.config_utils.config_schema", "SearchUCSSPARConfig"),
    "TestModelContSPARConfig": ("spar.utils.config_utils.config_schema", "TestModelContSPARConfig"),
    "TestModelDiscSPARConfig": ("spar.utils.config_utils.config_schema", "TestModelDiscSPARConfig"),
    "TestModelSPARConfig": ("spar.utils.config_utils.config_schema", "TestModelSPARConfig"),
    "TrainAlignmentContSPARConfig": ("spar.utils.config_utils.config_schema", "TrainAlignmentContSPARConfig"),
    "TrainAlignmentDiscSPARConfig": ("spar.utils.config_utils.config_schema", "TrainAlignmentDiscSPARConfig"),
    "TrainAlignmentModelSPARConfig": ("spar.utils.config_utils.config_schema", "TrainAlignmentModelSPARConfig"),
    "TrainEnvContSPARConfig": ("spar.utils.config_utils.config_schema", "TrainEnvContSPARConfig"),
    "TrainEnvDiscSPARConfig": ("spar.utils.config_utils.config_schema", "TrainEnvDiscSPARConfig"),
    "TrainEnvModelSPARConfig": ("spar.utils.config_utils.config_schema", "TrainEnvModelSPARConfig"),
    "TrainHeuristicSPARConfig": ("spar.utils.config_utils.config_schema", "TrainHeuristicSPARConfig"),
    "VisualizeUnsolvedQStarSPARConfig": ("spar.utils.config_utils.config_schema", "VisualizeUnsolvedQStarSPARConfig"),
    "BitwiseEqReportSPARConfig": ("spar.utils.config_utils.config_schema", "BitwiseEqReportSPARConfig"),
    "AlignmentEncoderMatchReportSPARConfig": (
        "spar.utils.config_utils.config_schema",
        "AlignmentEncoderMatchReportSPARConfig",
    ),
    "ProcessImageSPARConfig": ("spar.utils.config_utils.config_schema", "ProcessImageSPARConfig"),
    "QStarResultsToLatexSPARConfig": ("spar.utils.config_utils.config_schema", "QStarResultsToLatexSPARConfig"),
}

_config_lazy_importer: LazyImporter[type[SPARConfig]] = LazyImporter(
    imports=_CONFIG_IMPORTS, module_name=__name__, cache_enabled=True, thread_safe=False, debug_mode=False
)

# Stage to configuration class mapping
_STAGE_CONFIG_MAP: dict[str, str] = {
    "gen_data": "GenDataSPARConfig",
    "gen_search_data": "GenSearchDataSPARConfig",
    "create_sweep": "CreateSweepSPARConfig",
    "encode_offline_data": "EncodeOfflineDataSPARConfig",
    "plotter": "PlotterSPARConfig",
    "train_world_model": "TrainEnvModelSPARConfig",
    "train_env_disc": "TrainEnvDiscSPARConfig",
    "train_env_cont": "TrainEnvContSPARConfig",
    "train_alignment_model": "TrainAlignmentModelSPARConfig",
    "train_alignment_disc": "TrainAlignmentDiscSPARConfig",
    "train_alignment_cont": "TrainAlignmentContSPARConfig",
    "train_heuristic": "TrainHeuristicSPARConfig",
    "search_gbfs": "SearchGBFSSPARConfig",
    "search_qstar": "SearchQStarSPARConfig",
    "search_ucs": "SearchUCSSPARConfig",
    "visualize_unsolved_qstar": "VisualizeUnsolvedQStarSPARConfig",
    "bitwise_eq_report": "BitwiseEqReportSPARConfig",
    "alignment_encoder_match_report": "AlignmentEncoderMatchReportSPARConfig",
    "qstar_results_to_latex": "QStarResultsToLatexSPARConfig",
    "test_model": "TestModelSPARConfig",
    "test_model_disc": "TestModelDiscSPARConfig",
    "test_model_cont": "TestModelContSPARConfig",
    "process_image": "ProcessImageSPARConfig",
    "mse_plotter": "MSEPlotterSPARConfig",
    "optuna_study": "OptunaStudySPARConfig",
    "optuna_analyze": "OptunaAnalyzeSPARConfig",
    "optuna_replay": "OptunaReplaySPARConfig",
}

# Cache TypeAdapters so validators are built once per stage.
_TYPE_ADAPTERS: dict[str, TypeAdapter[SPARConfig]] = {}
_STAGE_CONFIG_CLASSES: dict[str, type[SPARConfig]] = {}
# Module-level factory alias to keep adapter creation centralized and cheap to call.
_TYPE_ADAPTER_FACTORY: type[TypeAdapter[SPARConfig]] = TypeAdapter


def _resolve_config_class(stage: str) -> type[SPARConfig]:
    """Resolve and cache the stage-specific config class."""
    cached_config_class: type[SPARConfig] | None = _STAGE_CONFIG_CLASSES.get(stage)
    if cached_config_class is not None:
        return cached_config_class

    config_class_name: str | None = _STAGE_CONFIG_MAP.get(stage)
    if config_class_name is None:
        raise ValueError(f"Unknown stage: {stage}")

    config_class: type[SPARConfig] = _config_lazy_importer.get_attr(name=config_class_name)
    _STAGE_CONFIG_CLASSES[stage] = config_class

    return config_class


def get_type_adapter(stage: str) -> TypeAdapter[SPARConfig]:
    """Get or create TypeAdapter for the given stage.

    Args:
        stage: The stage name for which to get the TypeAdapter
    """
    cached_type_adapter: TypeAdapter[SPARConfig] | None = _TYPE_ADAPTERS.get(stage)
    if cached_type_adapter is not None:
        return cached_type_adapter

    config_class: type[SPARConfig] = _resolve_config_class(stage=stage)
    type_adapter: TypeAdapter[SPARConfig] = _TYPE_ADAPTER_FACTORY(config_class)
    _TYPE_ADAPTERS[stage] = type_adapter

    return type_adapter


def warm_type_adapter_cache(stages: list[str] | None = None) -> None:
    """Warm TypeAdapters for selected stages (or all stages when omitted)."""
    stages_to_warm: list[str] = stages if stages is not None else list(_STAGE_CONFIG_MAP)
    for stage in stages_to_warm:
        get_type_adapter(stage=stage)


def _format_validation_errors(error: ValidationError) -> list[str]:
    """Create compact error messages from Pydantic validation errors."""
    return [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in error.errors()]


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails.

    Attributes:
        message: A human-readable error message.
        validation_errors: List of specific validation error messages, if any.
    """

    def __init__(self, message: str, validation_errors: list[str] | None = None) -> None:
        """Initialize the ConfigValidationError.

        Args:
            message: A human-readable error message.
            validation_errors: Optional list of specific validation error messages.
        """
        super().__init__(message)
        self.validation_errors: list[str] = validation_errors or []


def validate_config(cfg: DictConfig, stage: str) -> SPARConfig:
    """Validate a Hydra config for a known stage.

    OmegaConf produces a dictionary without instantiating structured nodes, and
    the stage-specific :class:`TypeAdapter` is reused from a module cache.

    Args:
        cfg: The Hydra DictConfig to validate.
        stage: The stage name for lookup.

    Returns:
        Validated Pydantic configuration object.

    Raises:
        ConfigValidationError: If validation fails.
    """
    try:
        # Convert OmegaConf to dict, resolving interpolations but avoiding deep copy
        # SCMode.DICT skips instantiation for performance improvement
        cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True, structured_config_mode=SCMode.DICT)

        # Use cached TypeAdapter for validation
        # Pass strict=False to allow dict to dataclass conversion while retaining field-level strictness
        type_adapter: TypeAdapter[SPARConfig] = get_type_adapter(stage=stage)
        validated_config: SPARConfig = type_adapter.validate_python(cfg_dict, strict=False)

    except ValidationError as e:
        raise ConfigValidationError(
            f"Configuration validation failed for stage '{stage}'", validation_errors=_format_validation_errors(e)
        ) from e

    except Exception as e:
        raise ConfigValidationError(f"Unexpected error during validation: {e!s}") from e
    else:
        return validated_config


def validate_config_json(json_data: str | bytes | bytearray, stage: str) -> SPARConfig:
    """Validate JSON text or bytes with Pydantic's JSON parser.

    Args:
        json_data: JSON payload containing configuration
        stage: The stage name for validation

    Returns:
        Validated Pydantic configuration object

    Raises:
        ConfigValidationError: If validation fails
    """
    try:
        # Use TypeAdapter for direct JSON validation
        type_adapter: TypeAdapter[SPARConfig] = get_type_adapter(stage=stage)
        validated_config: SPARConfig = type_adapter.validate_json(json_data, strict=True)

    except ValidationError as e:
        raise ConfigValidationError(
            f"Configuration validation failed for stage '{stage}'", validation_errors=_format_validation_errors(e)
        ) from e
    except Exception as e:
        raise ConfigValidationError(f"Unexpected error during validation: {e!s}") from e
    else:
        return validated_config


def validate_config_from_file(json_file_path: str, stage: str) -> SPARConfig:
    """Validate configuration from a JSON file using byte-level ingestion.

    This avoids Python-side JSON deserialization by passing raw file bytes
    directly to ``TypeAdapter.validate_json``.

    Args:
        json_file_path: Path to JSON file containing configuration
        stage: The stage name for validation

    Returns:
        Validated Pydantic configuration object

    Raises:
        ConfigValidationError: If file reading or validation fails.
    """
    try:
        json_content: bytes = pathlib.Path(json_file_path).read_bytes()

        return validate_config_json(json_data=json_content, stage=stage)

    except FileNotFoundError as e:
        raise ConfigValidationError(f"Configuration file not found: {json_file_path}") from e
    except Exception as e:
        raise ConfigValidationError(f"Error reading configuration file: {e!s}") from e
