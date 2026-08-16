"""Dataclass schemas for SPAR configuration validation and type checking."""

from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Protocol, TypeAlias

from hydra.conf import HydraConf, JobConf, RunDir
import numpy as np
from numpy.typing import NDArray
from omegaconf import MISSING
from pydantic import ConfigDict, Field, field_validator, model_serializer
from pydantic.dataclasses import dataclass

pydantic_config: ConfigDict = ConfigDict(
    # Each dataclass declares ``frozen=True`` because subclasses inherit this setting.
    validate_assignment=False,  # No per-field runtime validation - configs rarely mutate after CLI/Hydra composition
    # Safety nets
    extra="forbid",  # Prevent accidental typos and eliminates runtime merges
    strict=True,  # Strict validation avoids costly coercions
    use_enum_values=True,  # Use enum values in serialization
    arbitrary_types_allowed=True,  # Allow complex types from Hydra
    validate_default=False,  # Skip validation of default values for Hydra compatibility
    str_strip_whitespace=True,  # Strip whitespace from strings
    revalidate_instances="never",  # Don't revalidate instances for performance
    loc_by_alias=False,  # Use field names in error messages
    regex_engine="rust-regex",
    # populate_by_name=True  # extra lookup per field when not using aliases
)


StageLiteral: TypeAlias = Annotated[
    str,
    Field(
        pattern="^(gen_data|gen_search_data|create_sweep|encode_offline_data|train_world_model|"
        "train_heuristic|search_gbfs|search_qstar|search_ucs|"
        "train_alignment_model|test_model|process_image|plotter|mse_plotter|default|visualize_unsolved_qstar|"
        "bitwise_eq_report|alignment_encoder_match_report|qstar_results_to_latex|optuna_study|"
        "optuna_analyze|optuna_replay)$"
    ),
]

ModelTypeLiteral: TypeAlias = Annotated[str, Field(pattern="^(discrete|continuous)$")]
TestModelTypeLiteral: TypeAlias = Annotated[str, Field(pattern="^(discrete|continuous|combined)$")]
WandbModeLiteral: TypeAlias = Annotated[str, Field(pattern="^(online|offline|disabled)$", description="WandB mode")]
LogLevelLiteral: TypeAlias = Annotated[
    str, Field(pattern="^(INFO|WARNING|ERROR|DEBUG)$", description="Log level for W&B")
]

DTypeLiteral: TypeAlias = Annotated[
    str, Field(pattern="^(float16|float32|float64|int8|int32|int64)$", description="Data type for tensors/models")
]
ConfigScalar: TypeAlias = str | int | float | bool | None
if TYPE_CHECKING:
    from collections.abc import Callable

    HydraValue: TypeAlias = ConfigScalar | list["HydraValue"] | dict[str, "HydraValue"]
else:
    HydraValue: TypeAlias = Any
HydraDict: TypeAlias = dict[str, HydraValue]
HydraList: TypeAlias = list[HydraValue]
ModuleSpec: TypeAlias = dict[str, HydraValue]
ArchitectureSpec: TypeAlias = list[ModuleSpec]
DataArray: TypeAlias = NDArray[np.float32 | np.float64 | np.integer]
if TYPE_CHECKING:
    DataTransform: TypeAlias = Callable[[DataArray], DataArray]
else:
    DataTransform: TypeAlias = Any
CompileOptionValue: TypeAlias = str | int | bool
CompileOptions: TypeAlias = dict[str, CompileOptionValue]
DDPDeviceId: TypeAlias = int | str
WandbResumeValue: TypeAlias = bool | str | None
OptunaParameterValue: TypeAlias = str | int | float | bool | None
SchedulerParamValue: TypeAlias = int | float
SchedulerParams: TypeAlias = dict[str, SchedulerParamValue]


class PydanticSerializer(Protocol):
    """Serializer interface injected by pydantic dataclasses."""

    def to_python(
        self, value: PydanticSerializable, *, mode: str, exclude_none: bool, by_alias: bool, serialize_as_any: bool
    ) -> HydraDict:
        """Serialize a pydantic dataclass instance to Python values."""
        ...


class PydanticSerializable(Protocol):
    """Pydantic dataclass instances expose a shared serializer."""

    __pydantic_serializer__: ClassVar[PydanticSerializer]


# Selection priority for unsolved visualization rows. "none" preserves file order,
# "best" chooses the lowest-cost rows, and "worst" chooses the highest-cost rows.
SelectionPriorityLiteral: TypeAlias = Annotated[
    str, Field(pattern="^(best|worst|none)$", description="Ordering when selecting unsolved Q* rows")
]
OptunaDirectionLiteral: TypeAlias = Annotated[
    str, Field(pattern="^(minimize|maximize)$", description="Optimization direction for Optuna")
]
OptunaParameterKindLiteral: TypeAlias = Annotated[
    str, Field(pattern="^(float|int|categorical|bool|fixed)$", description="Optuna parameter type")
]
OptunaWorkflowRoleLiteral: TypeAlias = Annotated[
    str, Field(pattern="^(prepare|objective|evaluation)$", description="Workflow role for an Optuna step")
]
OptunaSamplerKindLiteral: TypeAlias = Annotated[
    str, Field(pattern="^(tpe|random|cmaes|nsga2|grid)$", description="Optuna sampler kind")
]
OptunaPrunerKindLiteral: TypeAlias = Annotated[
    str, Field(pattern="^(median|none|successive_halving|hyperband|percentile)$", description="Optuna pruner kind")
]
OptunaStorageKindLiteral: TypeAlias = Annotated[
    str, Field(pattern="^(journal|sqlite|rdb)$", description="Optuna storage backend kind")
]
OptunaFailureActionLiteral: TypeAlias = Annotated[
    str, Field(pattern="^(fail|prune|raise)$", description="How recoverable trial errors are handled")
]
OptunaDevicePolicyLiteral: TypeAlias = Annotated[
    str, Field(pattern="^(preserve|auto|explicit)$", description="How Optuna trials choose devices")
]
ConstraintOperatorLiteral: TypeAlias = Annotated[
    str, Field(pattern="^(<=|>=|<|>|==)$", description="Constraint comparison operator")
]


# Keep Enum classes for backward compatibility and constants access
class Stage(Enum):
    """Enumeration of workflow stages in the SPAR framework."""

    GEN_DATA = "gen_data"
    GEN_SEARCH_DATA = "gen_search_data"
    CREATE_SWEEP = "create_sweep"
    ENCODE_OFFLINE_DATA = "encode_offline_data"
    TRAIN_WORLD_MODEL = "train_world_model"
    TRAIN_HEURISTIC = "train_heuristic"
    SEARCH_GBFS = "search_gbfs"
    SEARCH_QSTAR = "search_qstar"
    SEARCH_UCS = "search_ucs"
    BITWISE_EQ_REPORT = "bitwise_eq_report"
    ALIGNMENT_ENCODER_MATCH_REPORT = "alignment_encoder_match_report"
    TRAIN_ALIGNMENT_MODEL = "train_alignment_model"
    TEST_MODEL = "test_model"
    PROCESS_IMAGE = "process_image"
    PLOTTER = "plotter"
    MSE_PLOTTER = "mse_plotter"
    OPTUNA_STUDY = "optuna_study"
    OPTUNA_ANALYZE = "optuna_analyze"
    OPTUNA_REPLAY = "optuna_replay"
    DEFAULT = "default"


class ModelType(Enum):
    """Enumeration of model types in the SPAR framework."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


@dataclass(slots=True, config=pydantic_config)
class CompilerCacheConfig:
    """Configuration for torch.compile cache persistence and portable artifacts.

    These options control durable on-disk caches used by Inductor/Triton and
    the optional portable artifact blob that can be saved/loaded across runs
    on compatible stacks to prewarm caches at service start.
    """

    # Enable modular on-disk caches
    enable_fx_graph_cache: bool = False
    enable_autograd_cache: bool = False
    cache_dir: str | None = None
    triton_cache_dir: str | None = None
    # Optional remote cache via Redis
    enable_remote_cache: bool = False
    redis_host: str | None = None
    redis_port: int | None = None
    # Portable mega-cache artifact
    portable_load_path: str | None = None
    portable_save_path: str | None = None
    portable_info_save_path: str | None = None


@dataclass(slots=True, config=pydantic_config)
class CompileConfig:
    """Configuration for PyTorch model compilation."""

    fullgraph: bool = False
    dynamic: bool | None = None
    backend: str = "inductor"
    mode: str | None = None
    options: CompileOptions | None = None
    disable: bool = True
    # Nested cache controls for torch.compile
    cache: CompilerCacheConfig = field(default_factory=CompilerCacheConfig)

    @field_validator("options", mode="before")
    @classmethod
    def _validate_options(cls, v: dict[str, ConfigScalar] | None) -> CompileOptions | None:
        """Validate torch compile options.

        Args:
            v: Raw options mapping supplied by OmegaConf or Pydantic.

        Returns:
            A compile options mapping containing only string, integer, and
            boolean values, or ``None`` when options are unset.

        Raises:
            TypeError: If an option value is not supported by the structured
                config schema.
        """
        if v is None:
            return None

        options: CompileOptions = {}
        for key, value in v.items():
            if not isinstance(value, (str, int, bool)):
                raise TypeError(f"Unsupported options value type at {key}: {type(value)}")
            options[key] = value

        return options


@dataclass(slots=True, config=pydantic_config)
class DDPArgsConfig:
    """Configuration for Distributed Data Parallel arguments."""

    device_ids: list[DDPDeviceId] | None = None
    output_device: DDPDeviceId | None = None
    dim: int = 0
    broadcast_buffers: bool = True
    init_sync: bool = True
    process_group: ConfigScalar | None = None
    bucket_cap_mb: int | None = None
    find_unused_parameters: bool = False
    check_reduction: bool = False
    gradient_as_bucket_view: bool = False
    static_graph: bool = False
    delay_all_reduce_named_params: HydraDict | None = None
    param_to_hook_all_reduce: HydraDict | None = None
    mixed_precision: HydraValue = None
    device_mesh: HydraValue = None


@dataclass(slots=True, config=pydantic_config)
class DDPConfig:
    """Configuration for Distributed Data Parallel."""

    enabled: bool = False
    world_size: int = 1
    num_nodes: int = 1
    args: DDPArgsConfig = field(default_factory=DDPArgsConfig)


@dataclass(slots=True, config=pydantic_config)
class WandbOshConfig:
    """Configuration for wandb-osh offline sync hooks."""

    enabled: bool = False
    command_dir: str | None = None
    trigger_every_seconds: int = 300
    trigger_on_phase_end: bool = True
    timeout_seconds: int = 300


@dataclass(slots=True, config=pydantic_config)
class WandbConfig:
    """Configuration for Weights & Biases logging.

    Fields map to ``wandb.init`` settings plus SPAR's distributed logging and
    media cadence controls.
    """

    # Core wandb.init() parameters
    project: str | None = "SPAR"
    entity: str | None = None
    dir: str | None = None  # Absolute path to directory where logs are stored
    id: str | None = None  # Unique identifier for this run
    name: str | None = None
    notes: str | None = None
    tags: list[str] | None = None

    # Configuration parameters
    log_config: bool = True  # Whether to log configuration to wandb
    config_exclude_keys: list[str] | None = None
    config_include_keys: list[str] | None = None
    allow_val_change: bool | None = None

    # Organization parameters
    group: str | None = None  # Group runs by environment or other criteria
    job_type: str | None = None  # Type of run (e.g., "train", "eval")

    # Mode and behavior parameters
    mode: Annotated[str, Field(pattern="^(online|offline|disabled)$", description="Main W&B sync mode")] = "offline"
    force: bool | None = None  # Whether W&B login is required
    anonymous: str | None = None  # "never", "allow", "must"

    # Resume and fork parameters
    reinit: WandbResumeValue = None  # Can be bool or string - Behavior when wandb.init() called with active run
    resume: WandbResumeValue = None  # Can be bool or string - "allow", "never", "must", "auto", True, False
    resume_from: str | None = None  # Resume from specific run: "{run_id}?_step={step}"
    fork_from: str | None = None  # Fork from specific run: "{run_id}?_step={step}"

    # Code and logging parameters
    save_code: bool | None = None
    sync_tensorboard: bool | None = None
    monitor_gym: bool | None = None

    # These will be converted to Settings object
    settings_dict: Annotated[
        HydraDict, Field(default_factory=dict, description="Dictionary of settings for W&B logging")
    ] = field(default_factory=dict)

    # Additional SPAR-specific parameters
    log_model: bool = True  # Log model checkpoints as artifacts
    log_freq: int = 100
    log_level: Annotated[str, Field(pattern="^(INFO|WARNING|ERROR|DEBUG)$", description="Log level for W&B")] = "INFO"
    profile: Annotated[
        str, Field(pattern="^(training|debug|metrics-only|custom)$", description="W&B capture profile")
    ] = "training"
    distributed_logging: Annotated[
        str, Field(pattern="^(rank_zero|shared|per_rank)$", description="Distributed logging policy")
    ] = "rank_zero"
    resume_strategy: Annotated[
        str, Field(pattern="^(deterministic_allow|never|must)$", description="Run resume policy")
    ] = "deterministic_allow"
    step_metric: str = "global_step"
    media_log_every_n_steps: int = 0
    table_log_every_n_steps: int = 0
    histogram_log_every_n_steps: int = 0
    max_metrics_per_log: int = 512
    max_metric_key_length: int = 128
    wandb_osh: WandbOshConfig = field(default_factory=WandbOshConfig)

    @field_validator("settings_dict", mode="after")
    @classmethod
    def validate_settings_dict(cls, v: HydraDict) -> HydraDict:
        """Validate and clean settings_dict parameter."""
        # Fast path: avoid allocating a new dict when already clean.
        if all(val is not None for val in v.values()):
            return v

        return {k: val for k, val in v.items() if val is not None}


@dataclass(slots=True, config=pydantic_config)
class EnvConfig:
    """Base configuration for environments."""

    name: str = MISSING
    description: str = ""  # Optional description of the environment

    # Parameters defined by the selected environment implementation.
    params: HydraDict = field(default_factory=dict)

    # Core environment properties
    num_actions_max: Annotated[int, Field(gt=0)] = MISSING  # Maximum number of actions for the environment
    fixed_actions: bool = True  # Whether the environment has fixed actions
    action_dim: Annotated[int, Field(ge=1)] = MISSING
    discrete_actions: bool = True
    num_actions: Annotated[int, Field(ge=0)] = 0  # Only used for discrete action spaces

    # Visualization settings
    visualization: HydraDict | None = field(default_factory=dict)

    # Dimension settings (for image-based environments)
    dim: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True, config=pydantic_config)
class ModelArchitectureConfig:
    """Base configuration for a single model component with type-safe parameter access."""

    architecture: ArchitectureSpec | None = None

    def get_param(
        self, name: str, default: ConfigScalar | HydraList | HydraDict = None
    ) -> ConfigScalar | HydraList | HydraDict:
        """Get a parameter value."""
        return getattr(self, name, default)

    def has_param(self, name: str) -> bool:
        """Check if a parameter exists."""
        return hasattr(self, name)


@dataclass(slots=True, config=pydantic_config)
class AlignmentModelConfig(ModelArchitectureConfig):
    """Type-safe configuration for alignment model components."""

    chan_in: int | None = None
    resnet_chan: int | None = None
    num_resnet_blocks: int | None = None
    fc_in: int | None = None
    fc_h_dim: int | None = None


@dataclass(slots=True, config=pydantic_config)
class EvalModelConfig(ModelArchitectureConfig):
    """Type-safe configuration for evaluation model components."""

    # Model paths
    encoder_path: str = ""
    transition_model_path: str = ""
    decoder_path: str = ""
    alignment_model_path: str = ""

    # Evaluation settings
    use_alignment_model: bool = True
    log_interval: int = 10
    save_interval: int = 50
    output_dir: str = "test_outputs"

    # Variation filtering
    include_variations: list[str] | None = None
    exclude_variations: list[str] | None = None
    use_variation_for_all_states: bool = False

    # Device and batch settings
    device: str = "cpu"
    batch_size: int = 32


@dataclass(slots=True, config=pydantic_config)
class EncoderConfig(ModelArchitectureConfig):
    """Type-safe configuration for encoder components."""

    # Future: Add common encoder-specific parameters here
    chan_in: int | None = None
    resnet_chan: int | None = None
    num_resnet_blocks: int | None = None
    fc_in: int | None = None
    fc_h_dim: int | None = None


@dataclass(slots=True, config=pydantic_config)
class DecoderConfig(ModelArchitectureConfig):
    """Type-safe configuration for decoder components."""

    # Future: Add common decoder-specific parameters here


@dataclass(slots=True, config=pydantic_config)
class EnvModelConfig(ModelArchitectureConfig):
    """Type-safe configuration for environment model components."""

    # Future: Add common env_model-specific parameters here


@dataclass(slots=True, config=pydantic_config)
class ModelComponentConfig:
    """Configuration for model architecture components."""

    encoder: EncoderConfig | None = field(default_factory=EncoderConfig)
    env_model: EnvModelConfig | None = field(default_factory=EnvModelConfig)
    decoder: DecoderConfig | None = field(default_factory=DecoderConfig)
    alignment_model: AlignmentModelConfig | None = field(default_factory=AlignmentModelConfig)


@dataclass(slots=True, config=pydantic_config)
class ModelConfig:
    """Base configuration for models."""

    chan_in: int = MISSING
    chan_out: int = MISSING
    enc_dim: int = MISSING
    enc_h: int = 0
    enc_w: int = 0
    chan_enc: int = 0
    discrete: ModelComponentConfig | None = field(default_factory=ModelComponentConfig)
    continuous: ModelComponentConfig | None = field(default_factory=ModelComponentConfig)
    dqn: ModelArchitectureConfig | None = field(default_factory=ModelArchitectureConfig)


@dataclass(slots=True, config=pydantic_config)
class TrainDataPathConfig:
    """Configuration for training data paths."""

    train_data: str = MISSING
    val_data: str = MISSING
    test_data: str = ""


@dataclass(slots=True, config=pydantic_config)
class TestDataPathConfig:
    """Configuration for testing data paths."""

    test_data: str = MISSING


@dataclass(slots=True, config=pydantic_config)
class TrainSavePathConfig:
    """Configuration for training save paths."""

    model_dir: str = MISSING


@dataclass(slots=True, config=pydantic_config)
class TestSavePathConfig:
    """Configuration for test save paths."""

    images_dir: str = MISSING
    plots_dir: str = MISSING
    metrics_dir: str = MISSING


@dataclass(slots=True, config=pydantic_config)
class PretrainedModelPathConfig:
    """Configuration for paths to pretrained models."""

    encoder_path: str | None = None
    transition_model_path: str | None = None
    decoder_path: str | None = None


@dataclass(slots=True, config=pydantic_config)
class PretrainedModelTestPathConfig(PretrainedModelPathConfig):
    """Configuration for paths to pretrained models."""

    alignment_model_path: str | None = None


@dataclass(slots=True, config=pydantic_config)
class DatasetConfig:
    """Configuration for a single dataset."""

    name: str = MISSING
    num_eps: Annotated[int, Field(gt=0, description="Number of episodes")] = MISSING
    num_steps: Annotated[int, Field(gt=0, description="Number of steps per episode")] = MISSING
    start_seed: Annotated[int | None, Field(ge=0, description="Starting seed for reproducibility")] = None
    num_seeds: Annotated[int | None, Field(gt=0, description="Number of seeds to use")] = None
    use_variations: bool | None = None
    variations_first_state_only: bool | None = None
    variation_sharing_group_size: Annotated[int | None, Field(gt=0, description="Group size for variation sharing")] = (
        None
    )
    effects: HydraDict | None = None
    file_name: str | None = None
    save_dir: str | None = None
    num_cpus: Annotated[int | None, Field(gt=0, description="Number of CPUs to use")] = None
    compression: bool | None = None
    batch_size_per_worker: Annotated[
        int | None, Field(ge=0, description="Batch size per worker (0 = auto-calculate)")
    ] = None


@dataclass(slots=True, config=pydantic_config)
class DataConfig:
    """Base configuration for data handling."""

    save_dir: str = "data/offline_data"
    num_cpus: Annotated[int, Field(gt=0, description="Number of CPUs for data processing")] = 1
    compression: bool = False
    batch_size: Annotated[int, Field(ge=0, description="Batch size for data processing (0 = auto-calculate)")] = 1
    batch_size_per_worker: Annotated[int, Field(ge=0, description="Batch size per worker (0 = auto-calculate)")] = 1
    shuffle: bool = True

    use_variations: bool = False
    variations_first_state_only: bool = False
    variation_sharing_group_size: Annotated[int, Field(gt=0, description="Group size for variation sharing")] = 1

    effects: HydraDict = field(default_factory=dict)
    effect_combinations: HydraDict = field(default_factory=dict)

    num_eps: Annotated[int, Field(gt=0, description="Number of episodes")] = 1
    num_steps: Annotated[int, Field(gt=0, description="Number of steps")] = 1
    start_seed: Annotated[int, Field(ge=0, description="Starting seed")] = 1
    num_seeds: Annotated[int, Field(gt=0, description="Number of seeds")] = 1
    file_name: str = "data"

    # New datasets structure
    datasets: list[DatasetConfig] = field(default_factory=list)


@dataclass(slots=True, config=pydantic_config)
class DataLoaderConfig:
    """Configuration for PyTorch DataLoader."""

    batch_size: Annotated[int, Field(gt=0, description="Batch size for training")] = 32
    num_batches_per_epoch: Annotated[int | None, Field(ge=1, description="Number of batches per epoch")] = None
    replacement: bool = True
    transform: DataTransform | None = None
    dtype: Annotated[str, Field(pattern="^(float16|float32|float64|int8|int32|int64)$", description="Data type")] = (
        "float32"
    )
    num_workers: Annotated[int, Field(ge=0, description="Number of worker processes")] = 0
    enable_memory_optimization: bool = False
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Annotated[int | None, Field(ge=1, description="Batches prefetched per worker")] = None
    pin_memory_device: str = ""
    infinite: bool = False
    base_only: bool = True
    variations_to_use: list[str] | None = None
    variations_to_ignore: list[str] | None = None


@dataclass(slots=True, config=pydantic_config)
class TestDataLoaderConfig(DataLoaderConfig):
    """Configuration for test-specific DataLoader settings."""

    batch_size: Annotated[int, Field(gt=0, description="Batch size for testing")] = 100
    transform: DataTransform | None = None
    dtype: Annotated[str, Field(pattern="^(float16|float32|float64|int8|int32|int64)$", description="Data type")] = (
        "float32"
    )
    num_workers: Annotated[int, Field(ge=0, description="Number of worker processes")] = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    pin_memory_device: str = ""
    use_encoded_targets: bool = True
    precompute_targets: bool = True
    specific_variation: str | None = None
    use_variation_for_all_states: bool = False
    enable_memory_optimization: bool = False
    variations_to_use: list[str] | None = None
    variations_to_ignore: list[str] | None = None


@dataclass(slots=True, config=pydantic_config)
class SchedulerConfig:
    """Configuration for learning rate schedulers."""

    type: str = MISSING  # e.g., "ExponentialLR", "StepLR"
    params: SchedulerParams = field(default_factory=dict)


@dataclass(slots=True, config=pydantic_config)
class TrainPhaseConfig:
    """Configuration for a single training phase."""

    max_itrs: int = MISSING
    lr: float = MISSING
    env_coeff: float | None = None


# @dataclass(slots=True , config=pydantic_config)
# class FSDPConfig:
#     """Configuration for Fully Sharded Data Parallel (FSDP)."""

#     import torch
#     from torch import nn
#     from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy

#     enabled: bool = True  # Enable FSDP for distributed training
#     mixed_precision: bool = True  # Enable mixed precision for bandwidth optimization
#     sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
#     auto_wrap_policy: str = "default"
#     wrap_policy_module_classes: Iterable[type[nn.Module]] = field(
#         default_factory=lambda: {
#             nn.Linear,
#             nn.Conv2d,
#             nn.ConvTranspose2d,
#             nn.BatchNorm2d,
#             nn.Flatten,
#             nn.Unflatten,
#             nn.Sequential,
#         }
#     )
#     sync_module_states: bool = True  # Synchronize initial module state across ranks.
#     param_dtype: str = "float16"  # Parameter dtype for mixed precision
#     reduce_dtype: str = "float16"  # Reduce dtype for mixed precision
#     buffer_dtype: str = "float16"  # Buffer dtype for mixed precision
#     cpu_offload: CPUOffload | None = None
#     backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE
#     ignored_modules: Iterable[torch.nn.Module] | None = None
#     param_init_fn: Callable[[nn.Module], None] | None = None
#     device_id: int | torch.device | None = None
#     forward_prefetch: bool = False
#     limit_all_gathers: bool = True
#     use_orig_params: bool = False
#     ignored_states: Iterable[torch.nn.Parameter] | Iterable[torch.nn.Module] | None = None


@dataclass(slots=True, config=pydantic_config)
class NCCLConfig:
    """Configuration for NCCL communication backend."""

    ib_hca: str = "mlx5"  # InfiniBand HCA setting
    p2p_level: str = "NVL"  # Peer-to-peer communication level
    net_gdr_level: str = "PHB"  # GPUDirect RDMA level
    tree_threshold: str = "0"  # Prefer ring algorithms for better bandwidth
    debug_level: str = "INFO"  # NCCL debug level
    timeout_seconds: int = 300  # NCCL operation timeout


@dataclass(slots=True, config=pydantic_config)
class MemoryConfig:
    """Memory limits and DataLoader transfer settings."""

    pin_memory: bool = True  # Enable pinned memory for GPU transfers
    shared_memory: bool = True  # Use shared memory for inter-process communication
    memory_mapping: bool = False  # Use memory mapping for large datasets
    persistent_workers: bool = True  # Keep DataLoader workers alive
    prefetch_factor: int = 2  # Prefetch factor for DataLoader
    max_workers: int = 4  # Maximum number of DataLoader workers


@dataclass(slots=True, config=pydantic_config)
class OptimizationConfig:
    """PyTorch compilation and training-step settings."""

    torch_compile: bool = True  # Enable torch.compile
    compile_mode: str = "reduce-overhead"  # Compilation mode
    compile_fullgraph: bool = True  # Enable fullgraph compilation
    cuda_graphs: bool = True  # Enable CUDA graphs for inference
    fused_optimizer: bool = True  # Use fused optimizer when available
    gradient_clipping: float = 1.0  # Gradient clipping norm
    gradient_accumulation_steps: int = 1  # Steps to accumulate gradients


@dataclass(slots=True, config=pydantic_config)
class DQNTrainConfig:
    """Configuration specific to DQN training with distributed support."""

    # Core DQN parameters
    lr: Annotated[float, Field(gt=0.0, description="Initial learning rate")] = 0.001
    lr_d: Annotated[float, Field(gt=0.0, le=1.0, description="Learning rate decay per iteration")] = 0.9999993
    max_itrs: Annotated[int, Field(gt=0, description="Maximum training iterations")] = 1000000
    batch_size: Annotated[int, Field(gt=0, description="Training batch size")] = 1000

    # Target network update parameters
    loss_thresh: Annotated[float, Field(gt=0.0, description="Loss threshold for target network update")] = 0.05
    update_itrs: list[int] | None = None  # Specific iterations to update, if None uses loss_thresh
    states_per_update: Annotated[int, Field(gt=0, description="States to train before checking update")] = 100000
    update_nnet_batch_size: Annotated[int, Field(gt=0, description="Batch size for network updates")] = 1000

    # GBFS and exploration parameters
    max_solve_steps: Annotated[int, Field(ge=1, description="Max GBFS steps for training states")] = 1
    eps_max: Annotated[float, Field(ge=0.0, le=1.0, description="Maximum epsilon for random exploration")] = 0.1
    per_eq_tol: Annotated[float, Field(ge=0.0, le=100.0, description="Percentage equality tolerance")] = 90.0

    # Data generation parameters
    start_steps: Annotated[int, Field(gt=0, description="Steps from offline to start states")] = 10
    goal_steps: Annotated[int, Field(gt=0, description="Steps from start to goal states")] = 10
    rb_itrs: Annotated[int, Field(ge=1, description="Replay buffer iterations")] = 1

    # Testing parameters
    num_test: Annotated[int, Field(gt=0, description="Number of test states")] = 1000

    # GPU/compute parameters
    single_gpu_training: bool = False  # Train on single GPU even with multiple available

    # Data paths
    train_data_path: str = MISSING  # Training data file path
    val_data_path: str = MISSING  # Validation data file path
    env_model_path: str = MISSING  # Environment model directory path

    # Save configuration
    nnet_name: str = MISSING  # Name for the neural network
    save_dir: str = "saved_heur_models"  # Base directory for saving models

    comm_mode: Annotated[str, Field(pattern="^(auto|all_gather|reduce_scatter)$", description="Communication mode")] = (
        "auto"
    )

    # Distributed training configurations
    # fsdp: FSDPConfig= field(default_factory=FSDPConfig)
    fsdp: HydraDict | None = None
    nccl: NCCLConfig = field(default_factory=NCCLConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)


@dataclass(slots=True, config=pydantic_config)
class TrainConfig:
    """Base configuration for training."""

    # data_paths: dict[str, str] | None = None
    # save_paths: dict[str, str] | None = None
    pretrained_model_paths: dict[str, str] | None = None
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    device: str = "cpu"  # Device for training (cpu, cuda, mps)
    test_steps: Annotated[int, Field(gt=0, description="Number of test steps")] = 30
    compile: CompileConfig = field(default_factory=CompileConfig)
    ddp: DDPConfig | None = field(default_factory=DDPConfig)
    print_interval: Annotated[int, Field(gt=0, description="Print interval during training")] = 100
    eval_interval: Annotated[int, Field(gt=0, description="Evaluation interval")] = 100
    checkpoint_interval: Annotated[int, Field(gt=0, description="Checkpoint save interval")] = 100
    batch_size: Annotated[int, Field(gt=0, description="Training batch size")] = 100
    distributed: bool = False
    world_size: Annotated[int, Field(ge=1, description="World size for distributed training")] = 1
    num_nodes: Annotated[int, Field(ge=1, description="Number of nodes for distributed training")] = 1
    optimizer: str = "adam"  # Optimizer type
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    phases: list[TrainPhaseConfig] | None = field(default_factory=list)
    dqn: DQNTrainConfig | None = None


@dataclass(slots=True, config=pydantic_config)
class BBoxConfig:
    """TypedDict for Matplotlib bbox configuration."""

    boxstyle: str = "round"
    pad: float = 0.7
    facecolor: str = "#E8F4FD"
    edgecolor: str = "#3498DB"
    alpha: float = 0.95
    linewidth: float = 1.0


@dataclass(slots=True, config=pydantic_config)
class SuptitleConfig:
    """Configuration for visualization suptitles (main titles).

    Control both text templates and style. Set a template to an empty string "" to hide the suptitle.
    """

    # Text templates
    # Step visualization (single step) template variables: {episode}, {step}, {variation}, {variation_display}
    step_template: str | None = "Episode {episode}, Step {step} - {variation_display}"
    # Summary visualization template variables: {variation}, {variation_display}, {summary_type},
    # {summary_type_label}, {metric_name}, {metric_name_display}, {metric_value_formatted}, {episode}
    summary_template: str | None = (
        "{variation_display} {summary_type_label} {metric_name_display} ({metric_value_formatted}) - Episode {episode}"
    )
    # Best/Worst single-step tracker template variables: {step}, {episode}, {metric_name},
    # {metric_name_display}, {metric_value_formatted}
    step_tracker_template: str | None = (
        "Step {step} | Episode {episode} | {metric_name_display}: {metric_value_formatted}"
    )

    # Hide behavior: when the resolved text is an empty string, the suptitle is not drawn
    hide_when_empty: bool = True

    # Style controls
    font_family: str | None = "serif"
    font_serif: list[str] | None = field(default_factory=lambda: ["Times New Roman", "Computer Modern", "DejaVu Serif"])
    font_size: int = 10
    font_weight: str = "bold"
    color: str = "#2C3E50"
    # Matplotlib bbox for suptitle
    # Matplotlib bbox style values are scalars.
    bbox_style: BBoxConfig = field(default_factory=BBoxConfig)
    # Gap between column titles and suptitle (figure fraction)
    gap: float = 0.15

    # Optional mapping to customize the label used for summary_type values.
    # Keys: "best", "worst", "selected". Values are strings, e.g.,
    # {"best": "Top", "worst": "Bottom", "selected": "Chosen"}
    summary_type_labels: dict[str, str] | None = None


@dataclass(slots=True, config=pydantic_config)
class TestConfig:
    """Configuration for testing."""

    dataloader: TestDataLoaderConfig = field(default_factory=TestDataLoaderConfig)
    device: str = "cpu"
    end_to_end: bool = False
    test_steps: int | None = None  # Use all steps if None
    compile: CompileConfig = field(default_factory=CompileConfig)
    use_alignment_model: bool = False
    precompute_targets: bool = False
    log_interval: int = 10
    save_interval: int = 50
    visualization_format: str = "png"  # Format for saving visualizations (e.g., "png", "jpg", "pdf")
    visualization_episode_index: int = 0  # Index of episode in batch to use for visualization (default: first episode)
    visualization_steps: list[int] | None = None
    top_k: Annotated[int, Field(ge=0, description="Top K episodes to visualize")] = 0
    apply_diff_highlighting: bool = False  # Whether to apply highlighting for differences in visualizations
    metrics_to_save: list[str] | None = None  # List of metrics to save in JSON format
    # per-model metric keys when running in combined tester mode
    metrics_to_save_discrete: list[str] | None = None
    metrics_to_save_continuous: list[str] | None = None
    # Column metric display control: controls which metric(s) to show on per-step
    # and summary visualization column titles. The first matching key present in the
    # available metrics is displayed. Later keys are fallbacks.
    # Examples: ["eq_bit", "reconstruction_mse"], ["cosine_similarity", "reconstruction_mse"].
    column_metric_priority: list[str] | None = None
    # Visualization labels (optional)
    # If provided, these control the row labels shown in the image grids.
    # row_labels: labels for the left-side (typically the Starting State / left block)
    # rightmost_col_row_labels: labels to display specifically for the right-most column
    row_labels: list[str] | None = None
    rightmost_col_row_labels: list[str] | None = None
    # Placement for the rightmost column row labels in per-step visualizations: "left" or "right"
    rightmost_col_row_labels_side: str = "right"
    # Title for the optional left-side single-cell panel that shows the variant/noisy environment
    variant_panel_title: str | None = "Noisy Environment"
    # Tester mode: 'separate' uses the two-row tester. 'combined' runs
    # discrete and continuous together and generates 3-row visualizations.
    tester_mode: str = "separate"

    # Optional per-model flags for combined mode default to global end_to_end when None.
    end_to_end_discrete: bool | None = None
    end_to_end_continuous: bool | None = None

    suptitle: SuptitleConfig = field(default_factory=SuptitleConfig)


@dataclass(slots=True, config=pydantic_config)
class SearchConfig:
    """Base configuration for search algorithms."""

    max_search_itrs: Annotated[int, Field(gt=0, description="Maximum search iterations")] = 5000
    vis_recon_mode: Annotated[
        str, Field(pattern="^(none|image|gif|both)$", description="Reconstruction visuals: none|image|gif|both")
    ] = "none"
    vis_env_mode: Annotated[
        str, Field(pattern="^(none|image|gif|both)$", description="Environment-rendered visuals: none|image|gif|both")
    ] = "none"
    vis_combined_mode: Annotated[
        str, Field(pattern="^(none|image|gif|both)$", description="Combined (env + recon) visuals: none|image|gif|both")
    ] = "none"

    # Include start/goal images and labels in combined output
    vis_combined_include_start_goal: bool = True

    # Titles/labels for combined rows (when applicable)
    vis_env_row_title: str = "Environment Rendered"
    vis_recon_row_title: str = "Reconstruction"
    vis_start_title: str = "Start"
    vis_goal_title: str = "Goal"

    # Control behavior on unsolved states (search did not find a goal)
    validate_on_unsolved: bool = False
    vis_on_unsolved: bool = False
    log_moves_on_unsolved: bool = False

    # Decoder path used for reconstructions in visualizations (when enabled)
    decoder_model_path: str | None = None
    # GIF frames per second
    save_vis_fps: Annotated[int, Field(gt=0, description="FPS for GIF visualization")] = 5

    # Visualization sizing controls
    # Individual (env-only / recon-only) visuals: choose original or custom size
    vis_individual_size_mode: Annotated[
        str, Field(pattern="^(original|custom)$", description="Size mode for individual visuals: original|custom")
    ] = "original"
    vis_individual_height: int | None = None
    vis_individual_width: int | None = None

    # Combined visuals row sizing (applies to env and recon rows)
    # If not set, rows auto-size to the larger of env/recon strips.
    vis_combined_row_height: int | None = None
    vis_combined_row_width: int | None = None

    # Q* Search parameters
    qstar_batch_size: Annotated[int, Field(gt=0, description="Batch size for Q* search")] = 100
    qstar_weight: Annotated[float, Field(ge=0.0, le=1.0, description="Weight for Q* search path cost")] = 0.6
    qstar_show_running_summary: bool = False

    # Optional YAML-backed TF32 controls. Defaults leave PyTorch CUDA TF32 and
    # float32 matmul precision settings unchanged.
    # - allow_tf32: enables TF32 on supported CUDA backends when True.
    # - float32_matmul_precision: when set to 'highest'|'high'|'medium' will call
    #   torch.set_float32_matmul_precision(value). Use null to leave unchanged.
    allow_tf32: bool = False
    float32_matmul_precision: Annotated[
        str | None,
        Field(
            pattern="^(highest|high|medium)$",
            description="Precision for torch.set_float32_matmul_precision. null leaves PyTorch unchanged",
        ),
    ] = None

    # Model paths for Q* search
    heuristic_model_path: Annotated[str, Field(description="Path to heuristic model")] = ""
    env_model_path: Annotated[str, Field(description="Path to environment model")] = ""
    alignment_model_path: Annotated[str, Field(description="Path to alignment model")] = ""
    encoder_model_path: Annotated[
        str | None, Field(description="Path to encoder model when encoder_mode requires it")
    ] = None

    # Input/output paths for Q* search
    # Single image pair mode
    state_path: str | None = "outputs/image_processing_cube3/resized_input_matplotlib.png"
    goal_state_path: str | None = "spar/search_test/goal.png"
    # Folder mode (runs search for all images)
    # If only `state_dir` is provided, the single `goal_state_path` is used for all.
    # If both `state_dir` and `goal_state_dir` are provided, images are paired by
    # lexicographic order of file paths (after filtering by `images_glob`).
    state_dir: str | None = None
    goal_state_dir: str | None = None
    images_glob: str = "*.png"  # Glob applied within directories. Use "*.*" to allow every file.
    results_dir: str = "spar/search_test/results"

    # HDF5 pairs dataset (overrides single-image or folder modes when provided)
    pairs_file: str | None = None
    # Back-compat single selection (treated as include=[...])
    pairs_start_variant: str | None = None  # e.g., "base" or a variation name
    pairs_goal_variant: str | None = None  # e.g., "base" or a variation name
    # New: independent include/exclude lists for start/goal sides
    pairs_start_include: list[str] | None = None  # None -> include all available (including 'base')
    pairs_start_exclude: list[str] | None = None
    pairs_goal_include: list[str] | None = None  # None -> include all available (including 'base')
    pairs_goal_exclude: list[str] | None = None

    # Neural network batch size
    nnet_batch_size: int | None = None

    # Tolerance parameters
    per_eq_tol: Annotated[
        float,
        Field(ge=0.0, le=100.0, description="Percent of latent state elements that need to be equal to declare equal"),
    ] = 100.0

    # Encoder selection policy
    # - "variant_aware": use base encoder for 'base' variants and alignment model otherwise
    # - "align_only": use alignment model for both start and goal regardless of variant
    # - "encoder_only": use base encoder for both start and goal (requires encoder_model_path)
    encoder_mode: Annotated[
        str, Field(pattern="^(variant_aware|align_only|encoder_only)$", description="Encoder selection policy")
    ] = "variant_aware"

    # Misc parameters
    pair_indices: list[int] | None = None
    start_idx: int = 0
    verbose: bool = False

    # GBFS Search parameters
    gbfs_search_itrs: Annotated[int, Field(gt=0, description="Search iterations for GBFS")] = 100


@dataclass(slots=True, config=pydantic_config)
class VisualizationConfig:
    """Configuration for visualization settings."""

    num_train_trajs_viz: int = 8
    num_train_steps_viz: int = 2
    num_val_trajs_viz: int = 8
    num_val_steps_viz: int = 2


@dataclass(slots=True, config=pydantic_config)
class SweepConfig:
    """Configuration for hyperparameter sweeps."""

    type: str = "heuristic"
    method: str = "bayes"
    metric_name: str = "val_loss"
    metric_goal: str = "minimize"
    count: int = 10
    create_agent_script: bool = True
    parameters: HydraDict = field(default_factory=dict)


@dataclass(slots=True, config=pydantic_config)
class MetricSpec:
    """Metric definition used for optimization, replay, and analysis."""

    name: str = MISSING
    step: str | None = None
    goal: OptunaDirectionLiteral = "maximize"


@dataclass(slots=True, config=pydantic_config)
class ConstraintSpec:
    """Constraint applied to a metric emitted by a workflow step."""

    name: str = MISSING
    step: str | None = None
    operator: ConstraintOperatorLiteral = "<="
    threshold: float = 0.0


@dataclass(slots=True, config=pydantic_config)
class ParameterSpec:
    """Explicit Optuna search-space parameter definition."""

    path: str = MISSING
    name: str | None = None
    step: str | None = None
    kind: OptunaParameterKindLiteral = "float"
    low: float | int | None = None
    high: float | int | None = None
    choices: list[OptunaParameterValue] | None = None
    value: OptunaParameterValue = None
    log: bool = False
    step_size: float | int | None = None
    when_all: dict[str, OptunaParameterValue] = field(default_factory=dict)


@dataclass(slots=True, config=pydantic_config)
class WorkflowStep:
    """One stage execution inside an Optuna trial workflow."""

    name: str = MISSING
    role: OptunaWorkflowRoleLiteral = "objective"
    stage: str = MISSING
    experiment: str | None = None
    env_name: str | None = None
    overrides: list[str] = field(default_factory=list)
    scan_paths: list[str] = field(default_factory=list)
    objective: MetricSpec | None = None
    constraints: list[ConstraintSpec] = field(default_factory=list)


@dataclass(slots=True, config=pydantic_config)
class StudyConfig:
    """Top-level study settings for Optuna execution."""

    study_name: str = "spar_study"
    n_trials: Annotated[int, Field(gt=0, description="Maximum number of Optuna trials")] = 10
    timeout_sec: Annotated[int | None, Field(gt=0, description="Optional Optuna timeout in seconds")] = None
    n_jobs: Annotated[int, Field(ge=1, description="Number of Optuna worker jobs")] = 1
    seed: int | None = None
    load_if_exists: bool = True
    gc_after_trial: bool = False
    show_progress_bar: bool = False
    catch_exceptions: bool = True
    recoverable_error_action: OptunaFailureActionLiteral = "fail"
    objective: MetricSpec | None = None
    objectives: list[MetricSpec] = field(default_factory=list)
    constraints: list[ConstraintSpec] = field(default_factory=list)
    base_overrides: list[str] = field(default_factory=list)
    workflow: list[WorkflowStep] = field(default_factory=list)


@dataclass(slots=True, config=pydantic_config)
class StorageConfig:
    """Storage backend settings for Optuna studies."""

    kind: OptunaStorageKindLiteral = "journal"
    path: str | None = None
    url: str | None = None
    heartbeat_interval: Annotated[int | None, Field(gt=0)] = None
    grace_period: Annotated[int | None, Field(gt=0)] = None
    engine_kwargs: HydraDict = field(default_factory=dict)


@dataclass(slots=True, config=pydantic_config)
class SamplerConfig:
    """Sampler settings for Optuna."""

    kind: OptunaSamplerKindLiteral = "tpe"
    seed: int | None = None
    n_startup_trials: Annotated[int, Field(ge=0)] = 10
    multivariate: bool = False
    group: bool = False
    constant_liar: bool = False


@dataclass(slots=True, config=pydantic_config)
class PrunerConfig:
    """Pruner settings for Optuna."""

    kind: OptunaPrunerKindLiteral = "median"
    n_startup_trials: Annotated[int, Field(ge=0)] = 5
    n_warmup_steps: Annotated[int, Field(ge=0)] = 0
    interval_steps: Annotated[int, Field(ge=1)] = 1
    n_min_trials: Annotated[int, Field(ge=1)] = 1
    percentile: Annotated[float, Field(gt=0.0, lt=100.0)] = 50.0
    min_resource: int | str = 1
    reduction_factor: Annotated[int, Field(ge=2)] = 4


@dataclass(slots=True, config=pydantic_config)
class OptunaRuntimeConfig:
    """Runtime settings that affect only Optuna execution."""

    output_root: str = "outputs/optuna"
    trial_dir_template: str = "{output_root}/{study_name}/trial_{trial_number:05d}"
    device_policy: OptunaDevicePolicyLiteral = "preserve"
    device: str | None = None
    disable_wandb: bool = True
    copy_resolved_configs: bool = True
    fail_on_missing_artifacts: bool = True
    prune_on_invalid_config: bool = True
    abort_on_unexpected_errors: bool = True


@dataclass(slots=True, config=pydantic_config)
class AnalysisConfig:
    """Analysis/export settings for Optuna studies."""

    top_k: Annotated[int, Field(ge=1)] = 5
    export_csv: bool = True
    compute_param_importances: bool = True
    include_failed_trials: bool = True
    output_dir: str | None = None


@dataclass(slots=True, config=pydantic_config)
class ReplayConfig:
    """Replay/verification settings for selected Optuna trials."""

    top_k: Annotated[int, Field(ge=1)] = 3
    trial_numbers: list[int] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    enable_wandb: bool = False
    output_dir: str | None = None


@dataclass(slots=True, config=pydantic_config)
class OptunaConfig:
    """Top-level Optuna configuration block."""

    study: StudyConfig = field(default_factory=StudyConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    pruner: PrunerConfig = field(default_factory=PrunerConfig)
    runtime: OptunaRuntimeConfig = field(default_factory=OptunaRuntimeConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    parameters: list[ParameterSpec] = field(default_factory=list)
    scan_paths: list[str] = field(default_factory=list)
    import_sweep_parameters: bool = False
    sweep_parameters: HydraDict = field(default_factory=dict)


@dataclass(slots=True, config=pydantic_config)
class HydraJobConfig:
    """Configuration for Hydra job settings."""

    name: str = "${env.name}_${stage}"
    chdir: bool = False


@dataclass(slots=True, config=pydantic_config)
class HydraRunConfig:
    """Configuration for Hydra run settings."""

    dir: str = "${save_dir}/${env.name}/${stage}/${now:%Y-%m-%d_%H-%M-%S}"


@dataclass(slots=True, config=pydantic_config)
class HydraConfig(HydraConf):
    """Configuration for Hydra framework."""

    job: JobConf = field(default_factory=JobConf)
    run: RunDir = field(default_factory=RunDir)


# Base configuration class with common fields
@dataclass(slots=True, config=pydantic_config)
class BaseSPARConfig:
    """Base configuration containing fields common to all stages."""

    # Core fields from config.yaml
    save_dir: str = "outputs"
    debug: bool = False

    # Stage identification
    stage: StageLiteral = MISSING  # Field(default_factory=lambda: Stage.DEFAULT.value)
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)

    __pydantic_serializer__: ClassVar[PydanticSerializer]

    @model_serializer()  # default mode='plain'
    def serialize_model(self) -> HydraDict:
        """Custom serializer for fast config dumping to WandB/logs."""
        return self.__pydantic_serializer__.to_python(
            self, mode="json", exclude_none=False, by_alias=False, serialize_as_any=True
        )


# Stage-specific configuration classes
@dataclass(slots=True, config=pydantic_config)
class GenDataSPARConfig(BaseSPARConfig):
    """Configuration for data generation stage (gen_data)."""

    stage: StageLiteral = Stage.GEN_DATA.value
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True, config=pydantic_config)
class SearchPairsDatasetConfig:
    """Configuration for a single start/goal pairs dataset."""

    name: str = MISSING
    num_pairs: Annotated[int, Field(gt=0, description="Number of start/goal pairs")] = MISSING
    reverse_goal: bool = False
    goal_num_steps: int | None = None
    goal_seeds: list[int] | None = None

    # Start states level control (if environment supports level_seeds)
    start_seed: Annotated[int | None, Field(ge=0, description="Starting seed for reproducibility")] = None
    num_seeds: Annotated[int | None, Field(gt=0, description="Number of seeds to use")] = None

    # Variations applied at save time (allowed: "none" | "start" | "goal" | "both")
    # Use a plain string here because OmegaConf does not accept `typing.Literal` in
    # structured dataclass fields when converting to a DictConfig.
    apply_variations_to: str | None = "none"
    effects: HydraDict | None = None

    # Output
    file_name: str | None = None
    save_dir: str | None = None
    compression: bool | str | None = None  # false|true|'gzip'|'lzf'|'szip'|'none'


@dataclass(slots=True, config=pydantic_config)
class SearchPairsDataConfig:
    """Top-level configuration for search pairs data generation."""

    save_dir: str = "data/search_pairs"
    datasets: list[SearchPairsDatasetConfig] = field(default_factory=list)
    # Optional global effects configuration shared by all datasets unless overridden
    effects: HydraDict | None = None


@dataclass(slots=True, config=pydantic_config)
class GenSearchDataSPARConfig(BaseSPARConfig):
    """Configuration for search data generation stage (gen_search_data)."""

    stage: StageLiteral = Stage.GEN_SEARCH_DATA.value
    search_data: SearchPairsDataConfig | None = field(default_factory=SearchPairsDataConfig)
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True, config=pydantic_config)
class CreateSweepSPARConfig(BaseSPARConfig):
    """Configuration for sweep creation stage (create_sweep)."""

    stage: StageLiteral = Stage.CREATE_SWEEP.value
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True, config=pydantic_config)
class EncodeOfflineDataSPARConfig(BaseSPARConfig):
    """Configuration for encoding offline data stage (encode_offline_data)."""

    stage: StageLiteral = Stage.ENCODE_OFFLINE_DATA.value
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    input_path: str = MISSING
    output_path: str = MISSING
    batch_size: Annotated[int, Field(gt=0, description="Number of states encoded in one model batch")] = 100


@dataclass(slots=True, config=pydantic_config)
class TrainEnvModelSPARConfig(BaseSPARConfig):
    """Configuration for environment model training stage (train_world_model)."""

    stage: StageLiteral = Stage.TRAIN_WORLD_MODEL.value
    world_model_type: ModelTypeLiteral = ModelType.DISCRETE.value
    end_to_end: bool = False
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data_paths: TrainDataPathConfig = field(default_factory=TrainDataPathConfig)
    save_paths: TrainSavePathConfig = field(default_factory=TrainSavePathConfig)


@dataclass(slots=True, config=pydantic_config)
class TrainEnvDiscSPARConfig(TrainEnvModelSPARConfig):
    """Configuration for discrete environment model training stage (train_env_disc)."""

    world_model_type: ModelTypeLiteral = ModelType.DISCRETE.value
    end_to_end: bool = False


@dataclass(slots=True, config=pydantic_config)
class TrainEnvContSPARConfig(TrainEnvModelSPARConfig):
    """Configuration for continuous environment model training stage (train_env_cont)."""

    world_model_type: ModelTypeLiteral = ModelType.CONTINUOUS.value
    end_to_end: bool = True


@dataclass(slots=True, config=pydantic_config)
class TrainAlignmentModelSPARConfig(BaseSPARConfig):
    """Configuration for alignment model training stage (train_alignment_model)."""

    stage: StageLiteral = Stage.TRAIN_ALIGNMENT_MODEL.value
    alignment_model_type: ModelTypeLiteral = ModelType.DISCRETE.value
    end_to_end: bool = False
    freeze_pretrained_models: bool = True
    precompute_targets: bool = True
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data_paths: TrainDataPathConfig = field(default_factory=TrainDataPathConfig)
    save_paths: TrainSavePathConfig = field(default_factory=TrainSavePathConfig)
    pretrained_model_paths: PretrainedModelPathConfig = field(default_factory=PretrainedModelPathConfig)


@dataclass(slots=True, config=pydantic_config)
class TrainAlignmentDiscSPARConfig(TrainAlignmentModelSPARConfig):
    """Configuration for discrete alignment model training stage (train_alignment_disc)."""

    stage: StageLiteral = Stage.TRAIN_ALIGNMENT_MODEL.value
    alignment_model_type: ModelTypeLiteral = ModelType.DISCRETE.value
    end_to_end: bool = False
    freeze_pretrained_models: bool = True
    precompute_targets: bool = True


@dataclass(slots=True, config=pydantic_config)
class TrainAlignmentContSPARConfig(TrainAlignmentModelSPARConfig):
    """Configuration for continuous alignment model training stage (train_alignment_cont)."""

    stage: StageLiteral = Stage.TRAIN_ALIGNMENT_MODEL.value
    alignment_model_type: ModelTypeLiteral = ModelType.CONTINUOUS.value
    end_to_end: bool = False
    freeze_pretrained_models: bool = True
    precompute_targets: bool = True


@dataclass(slots=True, config=pydantic_config)
class TestModelSPARConfig(BaseSPARConfig):
    """Configuration for model testing stage (test_model)."""

    stage: StageLiteral = Stage.TEST_MODEL.value
    test_model_type: TestModelTypeLiteral = ModelType.DISCRETE.value
    data_paths: TestDataPathConfig = field(default_factory=TestDataPathConfig)
    save_paths: TestSavePathConfig = field(default_factory=TestSavePathConfig)
    pretrained_model_paths: PretrainedModelTestPathConfig = field(default_factory=PretrainedModelTestPathConfig)
    # Optional: separate pretrained paths to enable the combined tester which
    # requires both discrete and continuous models at the same time.
    pretrained_model_paths_discrete: PretrainedModelTestPathConfig | None = None
    pretrained_model_paths_continuous: PretrainedModelTestPathConfig | None = None
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    test: TestConfig = field(default_factory=TestConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True, config=pydantic_config)
class TestModelDiscSPARConfig(TestModelSPARConfig):
    """Configuration for discrete model testing stage (test_model_disc)."""

    stage: StageLiteral = Stage.TEST_MODEL.value
    test_model_type: ModelTypeLiteral = ModelType.DISCRETE.value


@dataclass(slots=True, config=pydantic_config)
class TestModelContSPARConfig(TestModelSPARConfig):
    """Configuration for continuous model testing stage (test_model_cont)."""

    stage: StageLiteral = Stage.TEST_MODEL.value
    test_model_type: ModelTypeLiteral = ModelType.CONTINUOUS.value


@dataclass(slots=True, config=pydantic_config)
class TrainHeuristicSPARConfig(BaseSPARConfig):
    """Configuration for heuristic training stage (train_heuristic)."""

    stage: StageLiteral = Stage.TRAIN_HEURISTIC.value
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    pretrained_model_paths: PretrainedModelPathConfig = field(default_factory=PretrainedModelPathConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True, config=pydantic_config)
class SearchGBFSSPARConfig(BaseSPARConfig):
    """Configuration for Greedy Best-First Search stage (search_gbfs)."""

    stage: StageLiteral = Stage.SEARCH_GBFS.value
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True, config=pydantic_config)
class SearchQStarSPARConfig(BaseSPARConfig):
    """Configuration for Q* Search stage (search_qstar)."""

    stage: StageLiteral = Stage.SEARCH_QSTAR.value
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True, config=pydantic_config)
class VisualizeUnsolvedQStarSPARConfig(BaseSPARConfig):
    """Configuration for the visualize_unsolved_qstar stage."""

    stage: StageLiteral = "visualize_unsolved_qstar"
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    alignment_model_type: str = "discrete"
    # Paths and input selection (one of pairs_file, state_dir+goal_dir, or single files)
    results: str | None = None
    pairs_file: str | None = None
    state_dir: str | None = None
    goal_dir: str | None = None
    single_state: str | None = None
    single_goal: str | None = None
    outdir: str = "./failures"
    max: int = 10
    # Cap visualizations per start or goal variant. None or a nonpositive value disables the cap.
    max_per_var: int | None = None
    # Ordering for unsolved rows: "worst" (highest cost), "best" (lowest cost), "none" (file order)
    prioritize: SelectionPriorityLiteral = "none"
    dpi: int = 150
    figsize: str = "10x6"
    # Multiplier applied to matplotlib default title/font sizes for this visualization
    font_scale: float = 1.25
    # Border thickness (in points) to draw around the final image. 0.0 -> no border.
    # Typical values: 0.0, 0.5, 1.0, 2.0
    image_border: float = 0.0
    # Padding (in points) between the visualized content and the outer border.
    # 0.0 -> no extra padding. Typical values: 0.0, 6.0, 12.0
    image_border_padding: float = 6.0
    # Output image format for visualizations. Allowed: 'png', 'pdf', 'jpeg'
    output_format: str = "png"
    overlay_moves: bool = False
    # Optional model checkpoint paths
    alignment_model_path: str | None = None
    decoder_model_path: str | None = None
    encoder_model_path: str | None = None
    device: str = "cpu"
    verbose: bool = False


@dataclass(slots=True, config=pydantic_config)
class BitwiseEqReportSPARConfig(BaseSPARConfig):
    """Configuration for reporting bitwise equality across all test cases."""

    stage: StageLiteral = Stage.BITWISE_EQ_REPORT.value
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    results: str = MISSING
    pairs_file: str = MISSING
    alignment_model_path: str = MISSING
    encoder_model_path: str = MISSING
    alignment_model_type: ModelTypeLiteral = ModelType.DISCRETE.value
    device: str = "cpu"
    verbose: bool = False
    batch_size: int = 32
    # Variant filtering: first select variants to include, then drop any to ignore.
    # "all" or None means include every variant found in the pairs file.
    vars_to_include: list[str] | None = None
    # "all" ignores every variant except the base. None ignores nothing.
    vars_to_ignore: list[str] | None = None


@dataclass(slots=True, config=pydantic_config)
class AlignmentEncoderMatchReportSPARConfig(BaseSPARConfig):
    """Configuration for bitwise agreement between the alignment model and encoder without logs."""

    stage: StageLiteral = Stage.ALIGNMENT_ENCODER_MATCH_REPORT.value
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    pairs_file: str = MISSING
    alignment_model_path: str = MISSING
    encoder_model_path: str = MISSING
    alignment_model_type: ModelTypeLiteral = ModelType.DISCRETE.value
    device: str = "cpu"
    verbose: bool = False
    batch_size: int = 32
    # Variant filtering: select variants to include, then drop any to ignore.
    vars_to_include: list[str] | None = None
    vars_to_ignore: list[str] | None = None


@dataclass(slots=True, config=pydantic_config)
class QStarResultsToLatexSPARConfig(BaseSPARConfig):
    """Configuration for exporting Q* results to LaTeX tables."""

    stage: StageLiteral = "qstar_results_to_latex"
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Input/result paths
    results: str | None = None
    output_tex: str | None = None
    per_variant_output_dir: str | None = None

    # Presentation
    domain_name: str = "Rubik's Cube"
    include_per_variant_tables: bool = True

    # Whether per-variant tables are saved to separate files (True) or combined into a single file (False)
    separate_per_variant_files: bool = True

    # New option: if True, write a single combined LaTeX table containing all per-variant rows.
    # If None, the older `separate_per_variant_files` setting is used for backward compatibility.
    per_variant_combined_table: bool | None = None

    # Optional manual variant -> category mapping (variant name -> clean|augmented|real)
    variant_category_overrides: dict[str, str] | None = None
    # Optional manual mapping for variant display names in LaTeX tables
    # Keys are the variant names as they appear in results.json (case preserved).
    # If provided, the value will be used in tables. If not provided, the code
    # will make the variant name human-readable (replace underscores/hyphens and title-case).
    variant_name_mapping: dict[str, str] | None = None
    # Controls table row height scaling in LaTeX via \renewcommand{\arraystretch}{...}.
    # A float where 1.0 is the default LaTeX row height. Use >1.0 to increase, <1.0 to decrease.
    cell_height: float = 1.0

    # LaTeX table style: 'booktabs' (default, no vertical rules, uses \toprule/\midrule/\bottomrule)
    # or 'classic' (legacy style with vertical bars and \hline rules)
    latex_table_style: str = "booktabs"


@dataclass(slots=True, config=pydantic_config)
class SearchUCSSPARConfig(BaseSPARConfig):
    """Configuration for Uniform Cost Search stage (search_ucs)."""

    stage: StageLiteral = Stage.SEARCH_UCS.value
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


# Shared search-stage configuration type. The Q*, GBFS, and UCS stages share
# the same field layout (env, model, search, wandb). Each stage selects its
# algorithm entry point and reads the fields relevant to that algorithm.
SearchStageConfig: TypeAlias = SearchQStarSPARConfig | SearchGBFSSPARConfig | SearchUCSSPARConfig


@dataclass(slots=True, config=pydantic_config)
class AxisConfig:
    """Configuration for plot axis properties."""

    # Axis limits (None for auto-scaling)
    min_value: float | None = None
    max_value: float | None = None

    # Tick configuration
    tick_spacing: str = "auto"  # "auto", "linear", "log", "custom"
    tick_count: int | None = None  # For auto/linear spacing
    custom_ticks: list[float] | None = None  # For custom spacing

    # Formatting
    label: str | None = None
    use_scientific_notation: bool = False
    scientific_notation_threshold: float = 1000.0
    decimal_places: int = 3

    # Scale
    scale: str = "linear"  # "linear", "log", "symlog"
    symlog_threshold: float = 1.0  # For symlog scale


@dataclass(slots=True, config=pydantic_config)
class LegendConfig:
    """Configuration for plot legend."""

    show_legend: bool = True
    position: str = "best"  # "best", "upper_right", "upper_left", etc.
    framealpha: float = 0.9
    fontsize: str | int = "small"  # size names or int
    border_width: float = 0.5
    shadow: bool = False


@dataclass(slots=True, config=pydantic_config)
class GridConfig:
    """Configuration for plot grid."""

    show_grid: bool = True
    which: str = "major"  # "major", "minor", "both"
    alpha: float = 0.3
    linewidth: float = 0.5
    linestyle: str = "-"  # "-", "--", "-.", ":"
    color: str = "gray"


@dataclass(slots=True, config=pydantic_config)
class StatisticsDisplayConfig:
    """Configuration for displaying statistics on plots."""

    show_statistics: bool = True
    position: str = "top_left"  # "top_left", "top_right", "bottom_left", "bottom_right", "outside"
    statistics_to_show: list[str] = field(
        default_factory=lambda: ["min", "max", "mean"]  # "min", "max", "mean", "std", "median", "final_value"
    )
    background_alpha: float = 0.8
    background_color: str = "lightblue"
    text_color: str = "black"
    fontsize: int = 8


@dataclass(slots=True, config=pydantic_config)
class ColorSchemeConfig:
    """Configuration for plot color schemes."""

    # Primary colors for variants
    variant_colors: list[str] = field(
        default_factory=lambda: [
            "#feb55a",  # Discrete (with latent state)
            "#57bec8",  # Continuous (with latent state)
            "#175c86",  # Continuous (predicting next state)
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )

    # Special episode colors
    best_episode_color: str = "#2e7d32"  # Green
    worst_episode_color: str = "#d32f2f"  # Red

    # Line properties
    line_alpha: float = 0.8
    fill_alpha: float = 0.3
    smoothed_alpha: float = 0.6

    # Uncertainty visualization
    uncertainty_alpha: float = 0.2
    uncertainty_colors: list[str] = field(default_factory=lambda: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])


@dataclass(slots=True, config=pydantic_config)
class AnnotationConfig:
    """Configuration for plot annotations."""

    # Vertical lines for marking specific time points
    vertical_lines: list[HydraDict] = field(default_factory=list)
    # Format: [{x: float, label: str, color: str, style: str}]

    # Horizontal lines for marking thresholds
    horizontal_lines: list[HydraDict] = field(default_factory=list)
    # Format: [{y: float, label: str, color: str, style: str}]

    # Text annotations
    text_annotations: list[HydraDict] = field(default_factory=list)
    # Format: [{x: float, y: float, text: str, fontsize: int}]

    # Default annotation styling
    default_line_color: str = "red"
    default_line_style: str = "--"  # "-", "--", "-.", ":"
    default_line_alpha: float = 0.7
    default_text_fontsize: int = 9


@dataclass(slots=True, config=pydantic_config)
class PlotterConfig:
    """Configuration for MSE inputs, statistics, layout, and export."""

    # Input/Output configuration
    input_file: str = "reconstruction_mse_data.json"
    output_directory: str = "mse_plots"

    # Figure properties
    style: str = "nature_journal"
    figsize: tuple[float, float] = (3.45, 2.3)  # Single column width for Nature
    dpi: int = 300
    export_formats: list[str] = field(default_factory=lambda: ["pdf", "png"])
    png_transparent: bool = False

    # Axis configuration
    x_axis: AxisConfig = field(default_factory=lambda: AxisConfig(label="Step"))
    y_axis: AxisConfig = field(
        default_factory=lambda: AxisConfig(label="Reconstruction MSE", scale="log", use_scientific_notation=True)
    )

    # Visual elements
    legend: LegendConfig = field(default_factory=LegendConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    colors: ColorSchemeConfig = field(default_factory=ColorSchemeConfig)
    annotations: AnnotationConfig = field(default_factory=AnnotationConfig)
    statistics_display: StatisticsDisplayConfig = field(default_factory=StatisticsDisplayConfig)

    # Statistical processing
    smoothing_method: str = "exponential"  # "exponential", "savgol", "none"
    smoothing_window_ratio: float = 0.01  # Fraction of time_steps
    uncertainty_method: str = "quantiles"  # "std", "quantiles", "minmax"
    quantiles: tuple[float, float] = (0.1, 0.9)

    # Large-series processing
    max_points_before_decimation: int = 10000
    streaming_threshold: int = 1000000  # Use streaming for T > 1M

    # Plot content configuration
    show_minmax: bool = True
    show_raw_mean: bool = True
    show_smoothed: bool = True
    show_uncertainty: bool = True

    # Plot types to generate
    create_individual_plots: bool = True  # Per-variant plots (mean, best, worst)
    create_comparison_plot: bool = True  # Multi-variant comparison

    # Title configuration (None for auto-generated titles)
    title_template: str | None = None  # Template like "MSE Results - {variant}"
    comparison_title: str | None = None  # Custom title for comparison plot


@dataclass(slots=True, config=pydantic_config)
class PlotterSPARConfig(BaseSPARConfig):
    """Configuration for plotter stage."""

    stage: StageLiteral = Stage.PLOTTER.value
    plotter: PlotterConfig = field(default_factory=PlotterConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True, config=pydantic_config)
class MSEPlotterConfig(PlotterConfig):
    """Configuration for MSE plotter using the mse_plotter utility."""

    # Input/Output configuration
    input_directory: str | None = None  # If directory given, all files with file_pattern will be used
    # Separate model-type directories take precedence when supplied.
    discrete_input_directory: str | None = None
    continuous_input_directory: str | None = None
    # Optional: an additional source directory (e.g., continuous next-state predictions)
    extra_input_directory: str | None = None
    file_pattern: str = "*.json"
    discrete_file_pattern: str = "*discrete*"  # Pattern to identify discrete model files
    continuous_file_pattern: str = "*continuous*"  # Pattern to identify continuous model files
    # The extra source uses file_pattern when this pattern is None.
    extra_file_pattern: str | None = None
    input_files: list[str] = field(default_factory=list)
    output_directory: str = "mse_plots"
    source_names: list[str] = field(default_factory=list)
    # Optional custom label for the extra source (used when source_names doesn't provide one)
    extra_source_name: str | None = None
    metric_keys_to_plot: list[str] = field(default_factory=lambda: ["reconstruction_mse"])

    # Plot generation modes
    plot_mode: str = "directory"  # Options: "directory", "files"
    create_individual_plots: bool = True  # Create individual episode plots (mean, best, worst)
    create_individual_model_plots: bool = True  # Create plots for each model type separately
    create_model_comparison_plots: bool = True  # Create comparison plots between model types
    create_variant_comparison_plots: bool = True  # Create comparison plots between variants (same model type)
    # Optional: Aggregated mean across all variants present (per model type and D vs C)
    create_cross_variant_mean_plots: bool = False

    # Figure properties
    style: str = "none"  # One of the 15 styles from mse_plotter
    figsize: tuple[float, float] = (3.45, 2.3)  # Single column width for Nature journal
    dpi: int = 300
    export_formats: list[str] = field(default_factory=lambda: ["png"])
    png_transparent: bool = False

    # Axis configuration
    x_axis: AxisConfig = field(
        default_factory=lambda: AxisConfig(label="Step", tick_count=8, scale="linear", decimal_places=0)
    )
    y_axis: AxisConfig = field(
        default_factory=lambda: AxisConfig(
            label="Reconstruction MSE",
            tick_count=6,
            use_scientific_notation=True,
            scientific_notation_threshold=1000.0,
            decimal_places=3,
            scale="log",
            symlog_threshold=1.0,
        )
    )

    # Legend configuration
    legend: LegendConfig = field(
        default_factory=lambda: LegendConfig(
            show_legend=True, position="best", framealpha=0.9, fontsize="small", border_width=0.5, shadow=False
        )
    )

    # Grid configuration
    grid: GridConfig = field(
        default_factory=lambda: GridConfig(
            show_grid=True, which="major", alpha=0.3, linewidth=0.5, linestyle="-", color="gray"
        )
    )

    # Color scheme configuration
    colors: ColorSchemeConfig = field(
        default_factory=lambda: ColorSchemeConfig(
            variant_colors=[
                "#feb55a",  # Discrete (with latent state)
                "#57bec8",  # Continuous (with latent state)
                "#175c86",  # Continuous (predicting next state)
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
            ],
            best_episode_color="#2e7d32",  # Green for best performance
            worst_episode_color="#d32f2f",  # Red for worst performance
            line_alpha=0.8,
            fill_alpha=0.3,
            smoothed_alpha=0.6,
            uncertainty_alpha=0.2,
            uncertainty_colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        )
    )

    # Statistics display configuration
    statistics_display: StatisticsDisplayConfig = field(
        default_factory=lambda: StatisticsDisplayConfig(
            show_statistics=True,
            position="top_left",
            statistics_to_show=["min", "max", "mean"],
            background_alpha=0.8,
            background_color="lightblue",
            text_color="black",
            fontsize=8,
        )
    )

    # Annotation configuration
    annotations: AnnotationConfig = field(
        default_factory=lambda: AnnotationConfig(
            vertical_lines=[],
            horizontal_lines=[],
            text_annotations=[],
            default_line_color="red",
            default_line_style="--",
            default_line_alpha=0.7,
            default_text_fontsize=9,
        )
    )

    # Statistical processing parameters
    smoothing_method: str = "exponential"  # Options: "exponential", "savgol", "none"
    smoothing_window_ratio: float = 0.01  # Fraction of time_steps for smoothing window
    uncertainty_method: str = "quantiles"  # Options: "std", "quantiles", "minmax"
    quantiles: tuple[float, float] = (0.1, 0.9)  # Lower and upper quantiles for uncertainty bands

    # Large-input processing
    max_points_before_decimation: int = 10000
    streaming_threshold: int = 1000000  # Use streaming computation for very large datasets

    # Plot content configuration
    show_minmax: bool = True
    show_raw_mean: bool = True
    show_smoothed: bool = True
    show_uncertainty: bool = True

    # Legend detail overrides use plotter defaults when None.
    legend_show_raw_mean: bool | None = None  # Include a raw-mean legend sample
    legend_show_minmax: bool | None = None  # Include a min/max legend sample
    legend_show_band: bool | None = None  # Include the shaded "range" legend sample

    # Legend label templates and naming
    legend_smoothed_label_template: str = "{source} (smoothed mean)"
    legend_raw_label_template: str = "{source} (mean)"
    legend_minmax_label: str = "min / max"
    legend_band_label_quantiles_template: str = "{percent}% range"
    legend_band_label_std: str = "+/- std range"

    # Optional renaming maps for display names
    model_type_labels: dict[str, str] | None = None  # e.g., {"Discrete": "D", "Continuous": "C"}
    variant_label_overrides: dict[str, str] | None = None  # e.g., {"base": "Baseline"}

    # Title customization
    title_template: str | None = None  # Template like "MSE Results - {variant}" (null for auto-generated)
    comparison_title: str | None = None  # Custom title for comparison plot (null for default)


@dataclass(slots=True, config=pydantic_config)
class MSEPlotterSPARConfig(BaseSPARConfig):
    """Configuration for MSE plotter stage using the mse_plotter utility."""

    stage: StageLiteral = Stage.MSE_PLOTTER.value
    plotter: MSEPlotterConfig = field(default_factory=MSEPlotterConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True, config=pydantic_config)
class ProcessImagePathConfig:
    """Configuration for image processing paths."""

    input_image: str = MISSING
    input_image2: str | None = None  # Optional second image for concatenation
    output_dir: str = "outputs/image_processing"

    # Explicit configurable output file paths (include filename + extension)
    # If these are relative paths they are interpreted relative to the repo working dir.
    processed_image: str = "${paths.output_dir}/resized_input.png"
    reconstruction_image: str = "${paths.output_dir}/reconstruction.png"
    encoding_info: str = "${paths.output_dir}/discrete_encoding_info.txt"
    final_viz: str = "${paths.output_dir}/final_viz.png"


@dataclass(slots=True, config=pydantic_config)
class ProcessImageConfig:
    """Configuration for image processing parameters."""

    target_height: int = 32
    target_width: int = 32
    concatenate: bool = False  # Whether to concatenate two images
    quality_interpolation: str = "LANCZOS"  # NEAREST, LINEAR, CUBIC, LANCZOS
    processing_method: str = "opencv"  # opencv, matplotlib

    # Visualization titles and labels
    main_title: str = "SPAR Alignment Model's Result on Real-World Images"
    original_title: str = "Original Image"
    processed_title: str = "Resized Image"
    reconstructed_title: str = "Reconstructed Image"
    subtitle_template: str = (
        "Original Dimensions: {orig_w}x{orig_h}x{orig_c} | "
        "Resized Dimensions: {proc_w}x{proc_h}x{proc_c} | "
        "Reconstruction Dimensions: {recon_w}x{recon_h}x{recon_c}"
    )


@dataclass(slots=True, config=pydantic_config)
class ProcessImageSPARConfig(BaseSPARConfig):
    """Configuration for image processing stage (process_image)."""

    stage: StageLiteral = Stage.PROCESS_IMAGE.value
    alignment_model_type: ModelTypeLiteral = ModelType.DISCRETE.value
    device: str = "cpu"

    # Paths configuration
    paths: ProcessImagePathConfig = field(default_factory=ProcessImagePathConfig)
    pretrained_model_paths: PretrainedModelTestPathConfig = field(default_factory=PretrainedModelTestPathConfig)

    # Processing configuration
    processing: ProcessImageConfig = field(default_factory=ProcessImageConfig)

    # Model configurations
    env: EnvConfig = MISSING  # field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True, config=pydantic_config)
class OptunaStudySPARConfig(BaseSPARConfig):
    """Configuration for running Optuna studies."""

    stage: StageLiteral = Stage.OPTUNA_STUDY.value
    env: EnvConfig = MISSING
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True, config=pydantic_config)
class OptunaAnalyzeSPARConfig(BaseSPARConfig):
    """Configuration for analyzing Optuna studies."""

    stage: StageLiteral = Stage.OPTUNA_ANALYZE.value
    env: EnvConfig = MISSING
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True, config=pydantic_config)
class OptunaReplaySPARConfig(BaseSPARConfig):
    """Configuration for replaying Optuna trials."""

    stage: StageLiteral = Stage.OPTUNA_REPLAY.value
    env: EnvConfig = MISSING
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


# Union type for all possible configurations
SPARConfig: TypeAlias = (
    GenDataSPARConfig
    | GenSearchDataSPARConfig
    | CreateSweepSPARConfig
    | EncodeOfflineDataSPARConfig
    | TrainEnvModelSPARConfig
    | TrainEnvDiscSPARConfig
    | TrainEnvContSPARConfig
    | TrainAlignmentModelSPARConfig
    | TrainAlignmentDiscSPARConfig
    | TrainAlignmentContSPARConfig
    | TrainHeuristicSPARConfig
    | SearchGBFSSPARConfig
    | SearchQStarSPARConfig
    | SearchUCSSPARConfig
    | PlotterSPARConfig
    | MSEPlotterSPARConfig
    | TestModelSPARConfig
    | TestModelDiscSPARConfig
    | TestModelContSPARConfig
    | ProcessImageSPARConfig
    | VisualizeUnsolvedQStarSPARConfig
    | BitwiseEqReportSPARConfig
    | AlignmentEncoderMatchReportSPARConfig
    | QStarResultsToLatexSPARConfig
    | OptunaStudySPARConfig
    | OptunaAnalyzeSPARConfig
    | OptunaReplaySPARConfig
)
