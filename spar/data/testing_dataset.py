"""Testing Dataset for testing World Models or Alignment Model in SPAR."""

from __future__ import annotations

import concurrent.futures
import gc
from logging import getLogger
import os
import pathlib
from typing import TYPE_CHECKING, TypedDict

import h5py
import numpy as np
from numpy.typing import NDArray
import psutil
import torch
from torch.utils.data import IterableDataset

from spar.utils.data_utils.hdf5_common import open_hdf5_for_read, read_float_dataset, read_int64_dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from concurrent.futures import Future
    from logging import Logger
    from typing import TypeAlias

    from psutil._ntuples import svmem
    from torch import Tensor, nn
    from typing_extensions import NotRequired


logger: Logger = getLogger(__name__)


FloatArray: TypeAlias = NDArray[np.float32 | np.float64]
Int64Array: TypeAlias = NDArray[np.int64]


def _ensure_float_array(array: NDArray[np.generic]) -> FloatArray:
    """Normalize an array into float32/float64 without changing numeric semantics."""
    if array.dtype == np.float64:
        return np.asarray(array, dtype=np.float64)
    # if array.dtype == np.float16:
    #     return np.asarray(array, dtype=np.float16)
    return np.asarray(array, dtype=np.float32)


def _extract_encoded(out: Tensor | tuple[Tensor, ...] | list[Tensor]) -> Tensor:
    """Normalize encoder/module return values into a single Tensor.

    Accepts a Tensor or a tuple/list where the encoded tensor is at index 0 or 1.
    Returns a torch.Tensor.
    """
    if isinstance(out, (tuple, list)):
        it: Iterator[Tensor] = iter(out)
        first: Tensor | None = next(it, None)
        second: Tensor | None = next(it, None)
        chosen: Tensor | None = second if second is not None else first
        if chosen is None:
            raise ValueError("Encoder returned empty tuple/list")

        return chosen

    return out


class EpisodeMeta(TypedDict):
    """Per-episode metadata containing the episode index."""

    episode_index: int


class BatchResult(TypedDict):
    """The structured output of a dataset batch."""

    variation_name: str
    episode_indices: Tensor
    states: Tensor
    actions: Tensor
    target_states: Tensor
    encoded_target_states: NotRequired[Tensor]


class EpisodeChunkData(TypedDict):
    """Chunked data loaded in parallel for a range of episodes."""

    base_states: dict[int, FloatArray]
    actions: dict[int, Int64Array]
    variation_states: dict[str, dict[int, FloatArray]]


class BaseEpisodeData(TypedDict):
    """Typed structure for legacy base episode data."""

    base_states: FloatArray
    actions: Int64Array


class BatchResultBase(TypedDict):
    """Batch result without optional fields."""

    variation_name: str
    episode_indices: Tensor
    states: Tensor
    actions: Tensor
    target_states: Tensor


class VariationInfo(TypedDict):
    """Summary of variations, counts, and configuration flags for the dataset."""

    variations: list[str]
    episodes_per_variation: dict[str, int]
    tracking_status: dict[str, int]
    batch_size: int
    use_encoded_targets: bool
    precompute_targets: bool
    specific_variation: str | None
    use_variation_for_all_states: bool


def get_gpu_memory_info() -> tuple[float, float, float]:
    """Get GPU memory usage information.

    Returns:
        Tuple of (allocated_gb, free_gb, total_gb)
    """
    if torch.cuda.is_available():
        allocated: float = torch.cuda.memory_allocated() / (1024**3)
        total: float = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free: float = total - allocated

        return allocated, free, total

    return 0.0, 0.0, 0.0


def get_system_memory_info() -> tuple[float, float]:
    """Get system RAM memory information.

    Returns:
        Tuple of (used_gb, total_gb)
    """
    memory: svmem = psutil.virtual_memory()
    used_gb: float = (memory.total - memory.available) / (1024**3)
    total_gb: float = memory.total / (1024**3)

    return used_gb, total_gb


def calculate_memory_safety_factors(device: torch.device) -> dict[str, float]:
    """Calculate memory safety factors based on available resources.

    Args:
        device: Target device for computations

    Returns:
        Dictionary with safety factors and limits
    """
    factors: dict[str, float] = {
        "memory_safety_factor": 0.8,  # Default to 80% usage
        "encoding_memory_factor": 0.6,  # More conservative for encoding
        "cache_memory_factor": 0.1,  # Use 10% for caching
        "precompute_threshold_gb": 4.0,  # Default threshold
        "max_chunk_memory_gb": 2.0,  # Default chunk limit
    }

    if device.type == "cuda" and torch.cuda.is_available():
        _, _, total = get_gpu_memory_info()

        # Adjust factors based on available GPU memory
        if total >= 24.0:  # GPU with 24GB+ of memory
            factors["memory_safety_factor"] = 0.85
            factors["encoding_memory_factor"] = 0.7
            factors["cache_memory_factor"] = 0.15
            factors["precompute_threshold_gb"] = 8.0
            factors["max_chunk_memory_gb"] = 4.0
        elif total >= 12.0:  # GPU with 12-24GB of memory
            factors["memory_safety_factor"] = 0.8
            factors["encoding_memory_factor"] = 0.6
            factors["cache_memory_factor"] = 0.12
            factors["precompute_threshold_gb"] = 6.0
            factors["max_chunk_memory_gb"] = 3.0
        elif total >= 8.0:  # GPU with 8-12GB of memory
            factors["memory_safety_factor"] = 0.75
            factors["encoding_memory_factor"] = 0.5
            factors["cache_memory_factor"] = 0.1
            factors["precompute_threshold_gb"] = 4.0
            factors["max_chunk_memory_gb"] = 2.0
        else:  # Low memory GPU (<8GB)
            factors["memory_safety_factor"] = 0.7
            factors["encoding_memory_factor"] = 0.4
            factors["cache_memory_factor"] = 0.05
            factors["precompute_threshold_gb"] = 2.0
            factors["max_chunk_memory_gb"] = 1.0
    else:
        # CPU mode - use system memory info
        used_ram, total_ram = get_system_memory_info()
        available_ram: float = total_ram - used_ram

        if available_ram >= 32.0:  # System with 32GB+ of RAM
            factors["memory_safety_factor"] = 0.9
            factors["encoding_memory_factor"] = 0.8
            factors["cache_memory_factor"] = 0.2
            factors["precompute_threshold_gb"] = 16.0
            factors["max_chunk_memory_gb"] = 8.0
        elif available_ram >= 16.0:  # System with 16-32GB of RAM
            factors["memory_safety_factor"] = 0.8
            factors["encoding_memory_factor"] = 0.7
            factors["cache_memory_factor"] = 0.15
            factors["precompute_threshold_gb"] = 8.0
            factors["max_chunk_memory_gb"] = 4.0
        else:  # System with <16GB of RAM
            factors["memory_safety_factor"] = 0.7
            factors["encoding_memory_factor"] = 0.5
            factors["cache_memory_factor"] = 0.1
            factors["precompute_threshold_gb"] = 4.0
            factors["max_chunk_memory_gb"] = 2.0

    return factors


def cleanup_gpu_memory() -> None:
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def estimate_tensor_memory_gb(shape: tuple[int, ...], dtype: torch.dtype) -> float:
    """Estimate memory usage of a tensor in GB."""
    with torch.inference_mode():
        element_size: int = torch.tensor([], dtype=dtype, requires_grad=False).element_size()

    total_elements: np.int64 = np.prod(shape)

    return float((total_elements * element_size) / (1024**3))


def adaptive_batch_size(
    base_batch_size: int,
    tensor_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    safety_factors: dict[str, float] | None = None,
    operation_type: str = "batching",
) -> int:
    """Calculate adaptive batch size based on available memory.

    Args:
        base_batch_size: User-requested batch size
        tensor_shape: Shape of individual tensor
        dtype: PyTorch dtype
        device: Target device
        safety_factors: Memory safety factors (calculated if None)
        operation_type: Memory budget to use ("batching", "encoding", or "caching").

    Returns:
        Requested batch size capped by the estimated memory limit.
    """
    if safety_factors is None:
        safety_factors = calculate_memory_safety_factors(device)

    # Select appropriate safety factor based on operation type
    memory_factor: float
    if operation_type == "encoding":
        memory_factor = safety_factors["encoding_memory_factor"]
    elif operation_type == "caching":
        memory_factor = safety_factors["cache_memory_factor"]
    else:
        memory_factor = safety_factors["memory_safety_factor"]

    available_memory: float
    if device.type == "cuda" and torch.cuda.is_available():
        free: float
        _, free, _ = get_gpu_memory_info()
        available_memory = free * memory_factor
    else:
        used_ram, total_ram = get_system_memory_info()
        available_memory = (total_ram - used_ram) * memory_factor

    per_item_memory: float = estimate_tensor_memory_gb(tensor_shape, dtype)
    if per_item_memory == 0:
        return base_batch_size

    # Calculate the estimated upper bound.
    max_items: int = int(available_memory / per_item_memory)

    # Do not exceed either the request or the estimated upper bound.
    bounded_batch_size: int = min(base_batch_size, max(1, max_items))

    # Log if we had to reduce
    if bounded_batch_size < base_batch_size:
        logger.warning(
            f"Adaptive batch sizing: reduced {operation_type} batch from {base_batch_size} "
            f"to {bounded_batch_size} due to memory constraints"
        )

    return bounded_batch_size


def calculate_memory_bounded_parameters(
    base_batch_size: int,
    sample_tensor_shape: tuple[int, ...] | None,
    dtype: torch.dtype,
    device: torch.device,
    user_chunk_size: int | None = None,
    user_encoding_batch_size: int | None = None,
    user_cache_size: int | None = None,
    user_max_workers: int | None = None,
) -> dict[str, int]:
    """Derive loader parameters from memory limits and caller-supplied values.

    Args:
        base_batch_size: User-requested batch size
        sample_tensor_shape: Shape of sample tensor for estimation
        dtype: PyTorch dtype
        device: Target device
        user_chunk_size: User-provided chunk size, or None to derive it.
        user_encoding_batch_size: User-provided encoding batch size, or None to derive it.
        user_cache_size: User-provided cache size, or None to derive it.
        user_max_workers: User-provided worker limit, or None to derive it.

    Returns:
        Loader parameters bounded by the estimated memory budget.
    """
    safety_factors: dict[str, float] = calculate_memory_safety_factors(device)

    # Bound the requested batch size by the available memory estimate.
    bounded_batch_size: int
    if sample_tensor_shape is not None:
        bounded_batch_size = adaptive_batch_size(base_batch_size, sample_tensor_shape, dtype, device, safety_factors)
    else:
        bounded_batch_size = base_batch_size

    # Calculate chunk size based on available memory and I/O
    if user_chunk_size is None:
        chunk_size: int
        if device.type == "cuda":
            _, _, total = get_gpu_memory_info()
            # Scale chunks with device memory, subject to fixed lower and upper bounds.
            if total >= 16.0:
                chunk_size = min(200, max(50, bounded_batch_size * 4))
            elif total >= 8.0:
                chunk_size = min(150, max(32, bounded_batch_size * 3))
            else:
                chunk_size = min(100, max(16, bounded_batch_size * 2))
        else:
            used_ram, total_ram = get_system_memory_info()
            available_ram: float = total_ram - used_ram
            if available_ram >= 16.0:
                chunk_size = min(300, max(100, bounded_batch_size * 5))
            else:
                chunk_size = min(200, max(50, bounded_batch_size * 3))
    else:
        chunk_size = user_chunk_size

    # Calculate encoding batch size
    if user_encoding_batch_size is None:
        if sample_tensor_shape is not None:
            encoding_batch_size: int = adaptive_batch_size(
                bounded_batch_size, sample_tensor_shape, dtype, device, safety_factors, operation_type="encoding"
            )
        else:
            # Use half the loader batch when no tensor shape is available.
            encoding_batch_size = max(1, bounded_batch_size // 2)
    else:
        encoding_batch_size = user_encoding_batch_size

    # Calculate cache size based on available memory
    if user_cache_size is None:
        cache_size: int
        if device.type == "cuda":
            _, _, total = get_gpu_memory_info()
            if total >= 16.0:
                cache_size = min(128, max(32, bounded_batch_size * 8))
            elif total >= 8.0:
                cache_size = min(96, max(24, bounded_batch_size * 6))
            else:
                cache_size = min(64, max(16, bounded_batch_size * 4))
        else:
            used_ram, total_ram = get_system_memory_info()
            available_ram = total_ram - used_ram
            if available_ram >= 16.0:
                cache_size = min(256, max(64, bounded_batch_size * 12))
            else:
                cache_size = min(128, max(32, bounded_batch_size * 8))
    else:
        cache_size = user_cache_size

    # Calculate max workers based on system capabilities
    if user_max_workers is None:
        max_workers: int
        try:
            cpu_count: int = os.cpu_count() or 4
            # Conservative threading for I/O bound operations
            max_workers = min(6, max(1, cpu_count // 2))
        except Exception:
            max_workers = 2
    else:
        max_workers = user_max_workers

    return {
        "batch_size": bounded_batch_size,
        "chunk_size": chunk_size,
        "encoding_batch_size": encoding_batch_size,
        "cache_size": cache_size,
        "max_workers": max_workers,
    }


class TestingDataset(IterableDataset[BatchResult]):
    """IterableDataset for testing trained models with episode-based batching."""

    def __init__(
        self,
        file_path: str,
        *,
        batch_size: int,
        transform: Callable[[Tensor], Tensor] | None = None,
        dtype: torch.dtype = torch.float32,
        encoder: nn.Module | None = None,
        use_encoded_targets: bool = False,
        precompute_targets: bool = False,
        device: str | torch.device = "cpu",
        specific_variation: str | None = None,
        use_variation_for_all_states: bool = False,
        enable_memory_optimization: bool = False,
        max_workers: int | None = None,
        prefetch_factor: int = 2,
        chunk_size: int | None = None,
        encoding_batch_size: int | None = None,
        use_memory_mapping: bool = True,
        cache_size: int | None = None,
        variations_to_use: list[str] | None = None,
        variations_to_ignore: list[str] | None = None,
    ) -> None:
        """Initialize the testing dataset.

        Args:
            file_path: Path to HDF5 data file containing variation data.
            batch_size: Number of episodes per batch.
            transform: Optional transform function to apply to states.
            dtype: PyTorch dtype for tensor conversion.
            encoder: Optional pretrained encoder to encode base states.
            use_encoded_targets: If True, return both encoded and original target states.
            precompute_targets: If True, pre-encode all target states during initialization.
            device: Device to use for encoder computations.
            specific_variation: If provided, only return batches of this variation type.
            use_variation_for_all_states: If True, use variation for all states in episode.
                Otherwise, only for first state of episode.
            enable_memory_optimization: If True, derive unset loader sizes from memory
                estimates and reduce batches that exceed those estimates. When False,
                use the caller-supplied values without adjustment.
            max_workers: Number of parallel workers for data loading (auto-calculated if None).
            prefetch_factor: Number of batches queued by each worker.
            chunk_size: Size of chunks for HDF5 operations (auto-calculated if None).
            encoding_batch_size: Batch size for encoding operations (auto-calculated if None).
            use_memory_mapping: Whether to map HDF5 file pages into the process address space.
            cache_size: Size of LRU cache for frequently accessed data (auto-calculated if None).
            variations_to_use: Specific variation names to include (for HDF5 loading).
                If None, all variations are included. If provided, only these variations
                will be loaded from the HDF5 file.
            variations_to_ignore: Variation names to exclude (for HDF5 loading).
                If None, no variations are excluded. If provided, these variations
                will be skipped during loading.
        """
        self.file_path: str = file_path
        self.transform: Callable[[Tensor], Tensor] | None = transform
        self.dtype: torch.dtype = dtype
        self.encoder: nn.Module | None = encoder
        self.use_encoded_targets: bool = use_encoded_targets
        self.precompute_targets: bool = precompute_targets
        self.specific_variation: str | None = specific_variation
        self.use_variation_for_all_states: bool = use_variation_for_all_states
        self.prefetch_factor: int = prefetch_factor
        self.use_memory_mapping: bool = use_memory_mapping
        self.enable_memory_optimization: bool = enable_memory_optimization
        self.variations_to_use: list[str] | None = variations_to_use
        self.variations_to_ignore: list[str] | None = variations_to_ignore

        # Device setup
        if isinstance(device, str):
            device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.device: torch.device = device

        # Validate encoder requirement
        if self.use_encoded_targets and self.encoder is None:
            raise ValueError("encoder must be provided when use_encoded_targets=True")

        # Calculate memory safety factors when parameter adjustment is enabled.
        self.safety_factors: dict[str, float] = (
            calculate_memory_safety_factors(self.device) if self.enable_memory_optimization else {}
        )

        # Store caller values for later adjustment or direct use.
        self.user_batch_size: int = batch_size
        self.user_max_workers: int | None = max_workers
        self.user_chunk_size: int | None = chunk_size
        self.user_encoding_batch_size: int | None = encoding_batch_size
        self.user_cache_size: int | None = cache_size

        # Initialize parameters for data loading
        self.episodes: dict[str, list[EpisodeMeta]] = {}  # variation_name -> list of episode metadata
        self.variation_names: list[str] = []
        self.variation_tracking: dict[str, int] = {}  # Track returned batches per variation
        self.total_episodes_per_variation: dict[str, int] = {}

        # Shared data storage
        self.base_states: dict[int, FloatArray] = {}  # episode_idx -> base_states (shared across variations)
        self.encoded_base_states: dict[int, FloatArray] = {}  # episode_idx -> encoded_base_states (if precomputed)
        self.episode_actions: dict[int, Int64Array] = {}  # episode_idx -> actions (shared across variations)
        self.variation_states: dict[str, dict[int, FloatArray]] = {}  # var_name -> {episode_idx -> var_states}

        # Caching for frequently accessed data
        self._episode_cache: dict[tuple[int, str, bool], FloatArray] = {}
        self._batch_cache: dict[tuple[tuple[int, ...], str, bool], BatchResult] = {}
        self._cache_access_order: list[tuple[tuple[int, ...], str, bool]] = []  # Track access order for LRU eviction
        self._cache_hits = 0
        self._cache_misses = 0

        # Disable target precomputation when the estimated free memory is below the configured threshold.
        if self.enable_memory_optimization:
            if self.device.type == "cuda" and torch.cuda.is_available():
                _, free, _ = get_gpu_memory_info()
                precompute_threshold: float = self.safety_factors["precompute_threshold_gb"]
                if self.precompute_targets and free < precompute_threshold:
                    logger.warning(
                        f"Limited GPU memory ({free:.2f}GB free, threshold: {precompute_threshold:.2f}GB). "
                        "Disabling precomputation to prevent OOM."
                    )
                    self.precompute_targets = False
            else:
                used_ram, total_ram = get_system_memory_info()
                available_ram: float = total_ram - used_ram
                precompute_threshold = self.safety_factors["precompute_threshold_gb"]
                if self.precompute_targets and available_ram < precompute_threshold:
                    logger.warning(
                        f"Limited system memory ({available_ram:.2f}GB available, "
                        f"threshold: {precompute_threshold:.2f}GB). "
                        "Disabling precomputation to prevent OOM."
                    )
                    self.precompute_targets = False

        # Use memory-derived parameters or the caller values.

        self.batch_size: int
        self.chunk_size: int
        self.encoding_batch_size: int
        self.cache_size: int
        self.max_workers: int

        if self.enable_memory_optimization:
            # These estimates are refined after a sample tensor is loaded.
            default_params: dict[str, int] = calculate_memory_bounded_parameters(
                base_batch_size=self.user_batch_size,
                sample_tensor_shape=None,  # No sample data available yet
                dtype=self.dtype,
                device=self.device,
                user_chunk_size=self.user_chunk_size,
                user_encoding_batch_size=self.user_encoding_batch_size,
                user_cache_size=self.user_cache_size,
                user_max_workers=self.user_max_workers,
            )

            # Set the initial parameters before sample data is available.
            self.batch_size = default_params["batch_size"]
            self.chunk_size = default_params["chunk_size"]
            self.encoding_batch_size = default_params["encoding_batch_size"]
            self.cache_size = default_params["cache_size"]
            self.max_workers = default_params["max_workers"]
        else:
            # Use user-provided parameters directly without modification
            self.batch_size = self.user_batch_size
            self.chunk_size = self.user_chunk_size or 100
            self.encoding_batch_size = self.user_encoding_batch_size or self.user_batch_size
            self.cache_size = self.user_cache_size or 64
            self.max_workers = self.user_max_workers or 2

            logger.info(
                f"Memory-based parameter adjustment disabled. Using caller values: "
                f"batch_size: {self.batch_size}, chunk_size: {self.chunk_size}, "
                f"encoding_batch_size: {self.encoding_batch_size}, cache_size: {self.cache_size}, "
                f"max_workers: {self.max_workers}"
            )

        # Load data
        self._load_from_hdf5()

        # Refine batch estimates with the loaded sample shape.
        if self.enable_memory_optimization:
            sample_tensor_shape: tuple[int, ...] | None = self._get_sample_tensor_shape()
            if sample_tensor_shape is not None:
                bounded_params: dict[str, int] = calculate_memory_bounded_parameters(
                    base_batch_size=self.user_batch_size,
                    sample_tensor_shape=sample_tensor_shape,
                    dtype=self.dtype,
                    device=self.device,
                    user_chunk_size=self.user_chunk_size,
                    user_encoding_batch_size=self.user_encoding_batch_size,
                    user_cache_size=self.user_cache_size,
                    user_max_workers=self.user_max_workers,
                )

                # Store previous values for logging
                prev_batch_size: int = self.batch_size
                prev_encoding_batch_size: int = self.encoding_batch_size

                # Update parameters that depend on sample shape.
                self.batch_size = bounded_params["batch_size"]
                self.encoding_batch_size = bounded_params["encoding_batch_size"]
                # Note: chunk_size and other parameters remain as initially calculated

                # Log any adjustments
                if self.batch_size != prev_batch_size:
                    logger.info(
                        f"Adjusted batch size from {prev_batch_size} to {self.batch_size} "
                        f"based on sample tensor shape {sample_tensor_shape}"
                    )
                if self.encoding_batch_size != prev_encoding_batch_size:
                    logger.info(
                        f"Adjusted encoding batch size from {prev_encoding_batch_size} to {self.encoding_batch_size} "
                        f"based on sample tensor shape {sample_tensor_shape}"
                    )

        # Initialize tracking
        for var_name in self.variation_names:
            self.variation_tracking[var_name] = 0

        # Log any batch reduction made by the memory estimate.
        if self.enable_memory_optimization and self.batch_size != self.user_batch_size:
            logger.info(
                f"Adaptive batch sizing: adjusted batch size from {self.user_batch_size} "
                f"to {self.batch_size} to stay within the estimated memory budget"
            )

        # Log dataset information
        logger.info(
            f"TestingDataset loaded: {sum(self.total_episodes_per_variation.values())} total episodes "
            f"across {len(self.variation_names)} variations"
        )

        # Log parameter information
        if self.enable_memory_optimization:
            logger.info(
                f"Memory-based parameter adjustment enabled. batch_size: {self.batch_size}, "
                f"chunk_size: {self.chunk_size}, encoding_batch_size: {self.encoding_batch_size}, "
                f"cache_size: {self.cache_size}, max_workers: {self.max_workers}"
            )

    def _load_from_hdf5(self) -> None:
        """Load data from HDF5 file."""
        if not pathlib.Path(self.file_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        # Derive HDF5 cache parameters from the configured memory policy.
        cache_bytes: int
        cache_slots: int
        cache_bytes, cache_slots = self._calculate_hdf5_cache_params()
        logger.debug(f"Using HDF5 cache: {cache_bytes / (1024**3):.2f}GB, {cache_slots} slots")

        # Apply the derived raw-data chunk cache settings.
        with open_hdf5_for_read(self.file_path, rdcc_nbytes=cache_bytes, rdcc_nslots=cache_slots, rdcc_w0=0.25) as f:
            # Read metadata
            variant_names_raw: list[bytes] = f.attrs.get("variant_names", [b"base"])
            available_variations: list[str] = [
                name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in variant_names_raw
            ]

            # Filter variations based on user preferences
            self.variation_names = self._filter_variations(available_variations)

            variant_type_attr: bytes = f.attrs.get("variant_type", b"full")
            variant_type: str = (
                variant_type_attr.decode("utf-8")
                if isinstance(variant_type_attr, bytes)
                else str(variant_type_attr)
                if variant_type_attr
                else "full"
            )

            logger.info(f"Dataset variant_type: {variant_type}, filtered variations: {self.variation_names}")

            # Initialize structures
            for var_name in self.variation_names:
                self.episodes[var_name] = []
                self.total_episodes_per_variation[var_name] = 0
                self.variation_states[var_name] = {}

            # Determine file structure and load accordingly
            episodes_group_unknown = f.get("episodes")
            if isinstance(episodes_group_unknown, h5py.Group):
                self._load_new_format(episodes_group_unknown)
            else:
                self._load_legacy_format(f)

    def _load_new_format(self, episodes_group: h5py.Group) -> None:
        """Parallel loading of data."""
        episode_names: list[str] = sorted(episodes_group.keys())
        numpy_dtype: type[np.float32 | np.float64] = np.float32 if self.dtype == torch.float32 else np.float64

        # Pre-allocate arrays
        total_episodes: int = len(episode_names)

        def load_episode_chunk(chunk_indices: list[int]) -> EpisodeChunkData:
            """Load a chunk of episodes."""
            chunk_data: EpisodeChunkData = {
                "base_states": {},
                "actions": {},
                "variation_states": {var: {} for var in self.variation_names},
            }

            for idx in chunk_indices:
                episode_name: str = episode_names[idx]
                grp_unknown = episodes_group.get(episode_name)
                if not isinstance(grp_unknown, h5py.Group):
                    continue

                episode_group: h5py.Group = grp_unknown

                # Load base states
                base_grp_unknown = episode_group.get("base")
                if not isinstance(base_grp_unknown, h5py.Group):
                    continue

                base_states_ds = base_grp_unknown.get("states")
                if not isinstance(base_states_ds, h5py.Dataset):
                    continue

                try:
                    base_states: FloatArray = read_float_dataset(base_states_ds, numpy_dtype)
                except (ValueError, TypeError):
                    base_states = _ensure_float_array(np.array(base_states_ds[:], dtype=numpy_dtype))

                if base_states.size == 0:
                    continue

                chunk_data["base_states"][idx] = base_states

                # Load actions
                actions_ds = episode_group.get("actions")
                if isinstance(actions_ds, h5py.Dataset):
                    actions_arr: Int64Array = read_int64_dataset(actions_ds)
                    if actions_arr.size > 0:
                        chunk_data["actions"][idx] = actions_arr

                # Load variation states
                for var_name in self.variation_names:
                    if var_name == "base":
                        continue

                    var_grp_unknown = episode_group.get(var_name)
                    if not isinstance(var_grp_unknown, h5py.Group):
                        continue

                    var_states_ds = var_grp_unknown.get("states")
                    if not isinstance(var_states_ds, h5py.Dataset):
                        continue

                    # Read the dataset before checking its element count.
                    try:
                        var_arr: FloatArray = read_float_dataset(var_states_ds, numpy_dtype)
                    except (ValueError, TypeError):
                        var_arr = _ensure_float_array(np.array(var_states_ds[:], dtype=numpy_dtype))
                    if var_arr.size > 0:
                        chunk_data["variation_states"][var_name][idx] = var_arr

            return chunk_data

        # Process episodes in parallel chunks
        chunk_indices: list[list[int]] = [
            list(range(i, min(i + self.chunk_size, total_episodes))) for i in range(0, total_episodes, self.chunk_size)
        ]

        # Use thread pool for I/O bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk: dict[Future[EpisodeChunkData], list[int]] = {
                executor.submit(load_episode_chunk, chunk): chunk for chunk in chunk_indices
            }

            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_out: EpisodeChunkData = future.result()

                # Merge chunk data
                self.base_states.update(chunk_out["base_states"])
                self.episode_actions.update(chunk_out["actions"])

                for var_name in self.variation_names:
                    if var_name not in self.variation_states:
                        self.variation_states[var_name] = {}
                    self.variation_states[var_name].update(chunk_out["variation_states"][var_name])

        # Pre-encode states if needed
        if self.use_encoded_targets and self.precompute_targets and self.encoder is not None:
            self._batch_encode_all_states()

        # Create episode metadata
        self._create_episode_metadata_vectorized()

    def _batch_encode_all_states(self) -> None:
        """Encode all base states."""
        if not self.base_states:
            return
        if self.encoder is None:
            return
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()

        # Get all episode indices
        all_episode_indices: list[int] = list(self.base_states.keys())

        # Get a sample state to estimate memory requirements
        sample_episode: int = all_episode_indices[0]
        sample_state: Tensor = self.base_states[sample_episode][0]
        state_shape: torch.Size = sample_state.shape

        # Determine batch size based on memory optimization setting
        if self.enable_memory_optimization:
            # Calculate memory-safe batch size
            initial_batch_size: int = self.encoding_batch_size

            logger.info(f"Starting encoding for {len(all_episode_indices)} episodes.")
            if self.device.type == "cuda" and torch.cuda.is_available():
                allocated, free, _ = get_gpu_memory_info()
                logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Free: {free:.2f}GB")
            else:
                used_ram, total_ram = get_system_memory_info()
                logger.info(f"System Memory - Used: {used_ram:.2f}GB, Total: {total_ram:.2f}GB")

            with torch.inference_mode():
                episode_idx = 0

                while episode_idx < len(all_episode_indices):
                    # Calculate adaptive batch size for current memory state
                    current_batch_size: int = adaptive_batch_size(
                        initial_batch_size,
                        state_shape,
                        self.dtype,
                        self.device,
                        self.safety_factors,
                        operation_type="encoding",
                    )

                    # Process one episode at a time to avoid memory accumulation
                    if episode_idx < len(all_episode_indices):
                        episode_id: int = all_episode_indices[episode_idx]
                        states = self.base_states[episode_id]

                        try:
                            # Process states in smaller chunks
                            encoded_states: FloatArray = self._encode_states_chunked(
                                states, chunk_size=current_batch_size
                            )
                            self.encoded_base_states[episode_id] = encoded_states
                            episode_idx += 1
                            self._log_episode_encoding_progress(episode_idx, len(all_episode_indices))

                        except torch.cuda.OutOfMemoryError:
                            logger.warning(
                                f"CUDA OOM during encoding episode {episode_id}. "
                                "Cleaning up and retrying with smaller chunks."
                            )
                            cleanup_gpu_memory()

                            # Fallback to conservative processing
                            try:
                                conservative_chunk_size = max(1, current_batch_size // 4)
                                encoded_states = self._encode_states_chunked(states, chunk_size=conservative_chunk_size)
                                self.encoded_base_states[episode_id] = encoded_states
                                episode_idx += 1
                            except torch.cuda.OutOfMemoryError:
                                logger.exception(f"Failed to encode episode {episode_id} even with reduced batch size.")
                                # Skip problematic episode and continue
                                episode_idx += 1
                                cleanup_gpu_memory()
        else:
            # When memory optimization is disabled, process all episodes with the fixed encoding batch size
            logger.info(f"Starting batch encoding for {len(all_episode_indices)} episodes")

            with torch.inference_mode():
                for episode_id in all_episode_indices:
                    states = self.base_states[episode_id]
                    try:
                        encoded_states = self._encode_states_chunked(states, chunk_size=self.encoding_batch_size)
                        self.encoded_base_states[episode_id] = encoded_states
                    except torch.cuda.OutOfMemoryError:
                        logger.exception(
                            f"CUDA OOM during encoding episode {episode_id} with fixed batch size "
                            f"{self.encoding_batch_size}. Memory optimization is disabled. "
                            "Consider enabling memory optimization or manually reducing the encoding_batch_size."
                        )
                        # Continue with the next episode
                        cleanup_gpu_memory()

                        continue

        logger.info("Batch encoding completed.")

    def _log_episode_encoding_progress(self, episode_idx: int, total_episodes: int) -> None:
        """Periodically clean memory and log encoding progress."""
        if episode_idx % 10 != 0:
            return

        cleanup_gpu_memory()
        if self.device.type == "cuda" and torch.cuda.is_available():
            allocated, free, _ = get_gpu_memory_info()
            logger.debug(
                f"Processed {episode_idx}/{total_episodes} episodes. "
                f"GPU Memory - Allocated: {allocated:.2f}GB, Free: {free:.2f}GB"
            )
        else:
            used_ram, total_ram = get_system_memory_info()
            logger.debug(
                f"Processed {episode_idx}/{total_episodes} episodes. "
                f"System Memory - Used: {used_ram:.2f}GB, Total: {total_ram:.2f}GB"
            )

    def _encode_chunk_tensor(self, encoder: nn.Module, chunk_tensor: Tensor) -> FloatArray:
        """Encode a chunk tensor and return it as a NumPy float array."""
        with torch.inference_mode():
            out: Tensor | tuple[Tensor, ...] | list[Tensor] = encoder(chunk_tensor)
            encoded_tensor: Tensor = _extract_encoded(out)
            return _ensure_float_array(encoded_tensor.detach().cpu().numpy())

    def _encode_states_chunked(self, states: FloatArray, chunk_size: int = 32) -> FloatArray:
        """Encode states in chunks."""
        encoder: nn.Module | None = self.encoder
        if encoder is None:
            return states

        encoded_chunks: list[FloatArray] = []

        for i in range(0, len(states), chunk_size):
            chunk: FloatArray = states[i : i + chunk_size]
            with torch.inference_mode():
                chunk_tensor: Tensor = torch.from_numpy(chunk).to(dtype=self.dtype, device=self.device).detach()

            try:
                encoded_chunk: FloatArray = self._encode_chunk_tensor(encoder, chunk_tensor)
                encoded_chunks.append(encoded_chunk)

                # Clean up intermediate tensors
                del chunk_tensor

            except torch.cuda.OutOfMemoryError:
                # Handle OOM differently based on memory optimization setting
                if self.enable_memory_optimization:
                    logger.warning(
                        f"OOM during chunk encoding. Reducing chunk size from {chunk_size} to {max(1, chunk_size // 2)}"
                    )
                    # Cleanup GPU memory
                    cleanup_gpu_memory()

                    # Retry with smaller chunk
                    if chunk_size > 1:
                        reduced_chunk_size: int = max(1, chunk_size // 2)

                        return self._encode_states_chunked(states, chunk_size=reduced_chunk_size)

                    raise
                # Memory optimization is disabled
                logger.exception(
                    f"CUDA OOM during chunk encoding with fixed chunk size {chunk_size}. "
                    "Memory optimization is disabled. Consider enabling memory optimization "
                    "or manually reducing the encoding_batch_size."
                )
                raise  # Re-raise the OOM error

        if encoded_chunks:
            concatenated: NDArray[np.generic] = np.concatenate(encoded_chunks, axis=0)
            return _ensure_float_array(concatenated)
        return states

    def _create_episode_metadata_vectorized(self) -> None:
        """Create episode metadata."""
        episode_indices: list[int]
        for var_name in self.variation_names:
            if var_name == "base":
                # For base variation, use base_states
                episode_indices = list(self.base_states.keys())
            else:
                # For other variations, use variation_states
                if var_name not in self.variation_states:
                    continue

                episode_indices = list(self.variation_states[var_name].keys())

            # Create metadata in batch
            metadata_list: list[EpisodeMeta] = [{"episode_index": idx} for idx in episode_indices]
            self.episodes[var_name].extend(metadata_list)
            self.total_episodes_per_variation[var_name] = len(episode_indices)

    def _load_legacy_format(self, f: h5py.Group) -> None:
        """Parallel loading for legacy format."""
        numpy_dtype: type[np.float32 | np.float64] = np.float32 if self.dtype == torch.float32 else np.float64

        # Load base data with threading
        if "base" in f:
            base_group_obj = f.get("base")
            if not isinstance(base_group_obj, h5py.Group):
                return
            base_group: h5py.Group = base_group_obj
            base_keys_list: list[str] = list(base_group.keys())
            # Keys in HDF5 are expected to be strings, filter by prefix
            episode_keys: list[str] = sorted([k for k in base_keys_list if k.startswith("episode_")])

            def load_base_chunk(chunk_keys: list[str]) -> dict[int, BaseEpisodeData]:
                chunk_data: dict[int, BaseEpisodeData] = {}
                for i, episode_key in enumerate(chunk_keys):
                    episode_group_obj = base_group.get(episode_key)
                    if not isinstance(episode_group_obj, h5py.Group):
                        continue

                    states_ds = episode_group_obj.get("states")
                    if not isinstance(states_ds, h5py.Dataset):
                        continue

                    try:
                        base_states_arr: NDArray[np.generic] = np.asarray(states_ds, dtype=numpy_dtype)
                    except ValueError:
                        continue

                    base_states: FloatArray = _ensure_float_array(base_states_arr)

                    if len(base_states) > 0:
                        actions_ds = episode_group_obj.get("actions")
                        chunk_data[i] = {
                            "base_states": base_states,
                            "actions": np.array(
                                actions_ds[:] if isinstance(actions_ds, h5py.Dataset) else [], dtype=np.int64
                            ),
                        }

                return chunk_data

            # Process in parallel
            chunks: list[list[str]] = [
                episode_keys[i : i + self.chunk_size] for i in range(0, len(episode_keys), self.chunk_size)
            ]

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures: list[Future[dict[int, BaseEpisodeData]]] = [
                    executor.submit(load_base_chunk, chunk) for chunk in chunks
                ]

                base_episode_idx = 0
                for future in concurrent.futures.as_completed(futures):
                    chunk_data: dict[int, BaseEpisodeData] = future.result()
                    for data in chunk_data.values():
                        self.base_states[base_episode_idx] = data["base_states"]
                        self.episode_actions[base_episode_idx] = data["actions"]
                        base_episode_idx += 1

        # Load variation data similarly with threading
        for var_name in self.variation_names:
            if var_name != "base" and var_name in f:
                vg = f.get(var_name)
                if isinstance(vg, h5py.Group):
                    self._load_variation_parallel(vg, var_name, numpy_dtype)

        # Pre-encode and create metadata
        if self.use_encoded_targets and self.precompute_targets and self.encoder is not None:
            self._batch_encode_all_states()
        self._create_episode_metadata_vectorized()

    def _load_variation_parallel(
        self, var_group: h5py.Group, var_name: str, numpy_dtype: type[np.float32 | np.float64]
    ) -> None:
        """Loading variation data."""
        episode_keys: list[str] = sorted([
            k for k in list(var_group.keys()) if isinstance(k, str) and k.startswith("episode_")
        ])

        def load_var_chunk(chunk_info: tuple[list[str], int]) -> dict[int, FloatArray]:
            chunk_keys: list[str]
            start_idx: int
            chunk_keys, start_idx = chunk_info
            chunk_data: dict[int, FloatArray] = {}
            for i, episode_key in enumerate(chunk_keys):
                episode_group_obj = var_group.get(episode_key)
                if not isinstance(episode_group_obj, h5py.Group):
                    continue

                episode_group: h5py.Group = episode_group_obj
                states_ds = episode_group.get("states")
                if not isinstance(states_ds, h5py.Dataset):
                    continue

                try:
                    var_states: FloatArray = _ensure_float_array(np.asarray(states_ds, dtype=numpy_dtype))
                except ValueError:
                    var_states = _ensure_float_array(np.array(states_ds[:], dtype=numpy_dtype))
                if len(var_states) > 0:
                    chunk_data[start_idx + i] = var_states

            return chunk_data

        chunks_with_idx: list[tuple[list[str], int]] = [
            (episode_keys[i : i + self.chunk_size], i) for i in range(0, len(episode_keys), self.chunk_size)
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: list[Future[dict[int, FloatArray]]] = [
                executor.submit(load_var_chunk, chunk_info) for chunk_info in chunks_with_idx
            ]

            for future in concurrent.futures.as_completed(futures):
                chunk_data: dict[int, FloatArray] = future.result()
                self.variation_states[var_name].update(chunk_data)

                # Create episode metadata
                for episode_idx in chunk_data:
                    episode_metadata: EpisodeMeta = {"episode_index": episode_idx}
                    self.episodes[var_name].append(episode_metadata)
                    self.total_episodes_per_variation[var_name] += 1

    def __iter__(self) -> Iterator[BatchResult]:
        """Iterate over batches of episodes."""
        if self.specific_variation is not None:
            return self._iterate_specific_variation()

        return self._iterate_all_variations()

    def _iterate_specific_variation(self) -> Generator[BatchResult, None, None]:
        """Iterate over batches of a specific variation type.

        Yields:
            Batch dictionaries for the selected variation.
        """
        if self.specific_variation not in self.episodes:
            logger.warning(f"Variation '{self.specific_variation}' not found in dataset")
            return
        assert self.specific_variation is not None
        episodes_list: list[EpisodeMeta] = self.episodes[self.specific_variation]
        start_idx: int = self.variation_tracking[self.specific_variation]

        for batch_start in range(start_idx, len(episodes_list), self.batch_size):
            batch_end: int = min(batch_start + self.batch_size, len(episodes_list))
            batch_episodes: list[EpisodeMeta] = episodes_list[batch_start:batch_end]

            # Update tracking
            self.variation_tracking[self.specific_variation] = batch_end

            yield self._create_batch(batch_episodes, self.specific_variation)

    def _iterate_all_variations(self) -> Generator[BatchResult, None, None]:
        """Iterate over batches of all variation types sequentially.

        Yields:
            Batch dictionaries across all configured variations.
        """
        for variation_name in self.variation_names:
            episodes_list: list[EpisodeMeta] = self.episodes[variation_name]
            start_idx: int = self.variation_tracking[variation_name]

            for batch_start in range(start_idx, len(episodes_list), self.batch_size):
                batch_end: int = min(batch_start + self.batch_size, len(episodes_list))
                batch_episodes: list[EpisodeMeta] = episodes_list[batch_start:batch_end]

                # Update tracking
                self.variation_tracking[variation_name] = batch_end

                yield self._create_batch(batch_episodes, variation_name)

    def _create_batch(self, batch_episodes: list[EpisodeMeta], variation_name: str) -> BatchResult:
        """Create batch from a list of episode metadata.

        Args:
            batch_episodes: List of episode metadata dictionaries.
            variation_name: Name of the variation type for this batch.

        Returns:
            Dictionary containing batch data with keys:
            - 'variation_name': Name of the variation
            - 'episode_indices': Episode indices tensor
            - 'states': States with variation for first state, base for others (or all variation if enabled)
            - 'actions': Action sequences
            - 'target_states': Always base states (original)
            - 'encoded_target_states': Encoded base states (if use_encoded_targets=True)
        """
        # Memory constraints may require dynamic sub-batching.
        if self.enable_memory_optimization:
            return self._create_batch_with_dynamic_splitting(batch_episodes, variation_name)

        return self._create_single_batch(batch_episodes, variation_name)

    def _create_batch_with_dynamic_splitting(
        self, batch_episodes: list[EpisodeMeta], variation_name: str
    ) -> BatchResult:
        """Split a batch when its estimated memory exceeds the available budget.

        The split size is based on the first episode's tensor shapes.
        """
        # For single episode, process directly
        if len(batch_episodes) == 1:
            return self._create_single_batch(batch_episodes, variation_name)

        # Calculate the largest estimated batch that fits the memory budget.
        sub_batch_size: int = self._calculate_memory_bounded_batch_size(batch_episodes)

        # If we can process the entire batch, do so directly
        if sub_batch_size >= len(batch_episodes):
            return self._create_single_batch(batch_episodes, variation_name)

        # Split into sub-batches and process them
        logger.info(
            f"Dynamic batching: splitting {len(batch_episodes)} episodes into sub-batches of size {sub_batch_size} "
            f"for variation '{variation_name}'"
        )

        # Process first sub-batch and collect all results
        sub_batches: list[list[EpisodeMeta]] = [
            batch_episodes[i : i + sub_batch_size] for i in range(0, len(batch_episodes), sub_batch_size)
        ]

        # Process first sub-batch as the main result
        main_result: BatchResult = self._create_single_batch(sub_batches[0], variation_name)

        # Process remaining sub-batches and concatenate results
        if len(sub_batches) > 1:
            for sub_batch in sub_batches[1:]:
                sub_result: BatchResult = self._create_single_batch(sub_batch, variation_name)
                main_result = self._concatenate_batch_results(main_result, sub_result)

        return main_result

    def _calculate_memory_bounded_batch_size(self, batch_episodes: list[EpisodeMeta]) -> int:
        """Estimate the number of episodes that fit in available memory.

        If memory-based adjustment is disabled, return the full batch size.
        """
        if not self.enable_memory_optimization:
            return len(batch_episodes)

        # Get sample episode for memory estimation
        sample_episode_idx: int = batch_episodes[0]["episode_index"]
        sample_states: FloatArray = self.base_states[sample_episode_idx]
        state_shape: tuple[int, ...] = sample_states.shape

        # Calculate memory per episode (including states, targets, actions, and optional encoded targets)
        memory_per_episode: float = (
            estimate_tensor_memory_gb((1, *state_shape), self.dtype) * 3
        )  # states, targets, actions

        if self.use_encoded_targets and sample_episode_idx in self.encoded_base_states:
            encoded_shape = self.encoded_base_states[sample_episode_idx].shape
            memory_per_episode += estimate_tensor_memory_gb((1, *encoded_shape), self.dtype)

        # Get available memory
        if self.device.type == "cuda" and torch.cuda.is_available():
            _, free, _ = get_gpu_memory_info()
            available_memory: float = free * self.safety_factors["memory_safety_factor"]
        else:
            used_ram, total_ram = get_system_memory_info()
            available_memory = (total_ram - used_ram) * self.safety_factors["memory_safety_factor"]

        # Divide the available budget by the estimated per-episode footprint.
        if memory_per_episode <= 0:
            return len(batch_episodes)  # Fallback if estimation fails

        bounded_size: int = max(1, int(available_memory / memory_per_episode))

        # Log memory optimization info
        if bounded_size < len(batch_episodes):
            logger.debug(
                f"Memory limit: reducing batch from {len(batch_episodes)} to {bounded_size} episodes "
                f"(per-episode: {memory_per_episode:.3f}GB, available: {available_memory:.2f}GB)"
            )

        return bounded_size

    @staticmethod
    def _concatenate_batch_results(result1: BatchResult, result2: BatchResult) -> BatchResult:
        """Concatenate two batch results into a single result."""
        concatenated: BatchResult = {
            "variation_name": result1["variation_name"],
            "episode_indices": torch.cat([result1["episode_indices"], result2["episode_indices"]], dim=0).detach(),
            "states": torch.cat([result1["states"], result2["states"]], dim=0).detach(),
            "actions": torch.cat([result1["actions"], result2["actions"]], dim=0).detach(),
            "target_states": torch.cat([result1["target_states"], result2["target_states"]], dim=0).detach(),
        }

        # Handle optional encoded targets
        if "encoded_target_states" in result1 and "encoded_target_states" in result2:
            concatenated["encoded_target_states"] = torch.cat(
                [result1["encoded_target_states"], result2["encoded_target_states"]], dim=0
            ).detach()
        elif "encoded_target_states" in result1:
            concatenated["encoded_target_states"] = result1["encoded_target_states"].detach()
        elif "encoded_target_states" in result2:
            concatenated["encoded_target_states"] = result2["encoded_target_states"].detach()

        return concatenated

    def _create_single_batch(self, batch_episodes: list[EpisodeMeta], variation_name: str) -> BatchResult:
        """Create a single batch without memory splitting."""
        # Use batch cache if available
        cache_key: tuple[tuple[int, ...], str, bool] = (
            tuple(ep["episode_index"] for ep in batch_episodes),
            variation_name,
            self.use_variation_for_all_states,
        )
        if cache_key in self._batch_cache:
            self._cache_hits += 1
            # Update LRU order
            if cache_key in self._cache_access_order:
                self._cache_access_order.remove(cache_key)

            self._cache_access_order.append(cache_key)

            cached_result: BatchResult = self._batch_cache[cache_key]
            # Return deep copy to avoid accidental modifications
            with torch.inference_mode():
                cloned: BatchResult = {
                    "variation_name": cached_result["variation_name"],
                    "episode_indices": cached_result["episode_indices"].clone().detach(),
                    "states": cached_result["states"].clone().detach(),
                    "actions": cached_result["actions"].clone().detach(),
                    "target_states": cached_result["target_states"].clone().detach(),
                }
                if "encoded_target_states" in cached_result:
                    cloned["encoded_target_states"] = cached_result["encoded_target_states"].clone().detach()

                return cloned

        self._cache_misses += 1

        # Pre-allocate arrays
        batch_size: int = len(batch_episodes)
        episode_indices: NDArray[np.int64] = np.array([ep["episode_index"] for ep in batch_episodes], dtype=np.int64)

        # Get sample shapes for pre-allocation
        sample_episode_idx: int = batch_episodes[0]["episode_index"]
        sample_states: FloatArray = self.base_states[sample_episode_idx]
        sample_actions: NDArray[np.int64] = self.episode_actions[sample_episode_idx]

        state_shape: tuple[int, ...] = sample_states.shape
        action_shape: tuple[int, ...] = sample_actions.shape

        # Pre-allocate output arrays with correct dtype
        numpy_dtype: type[np.float32 | np.float64] = np.float32 if self.dtype == torch.float32 else np.float64

        batch_states_array: NDArray[np.float32 | np.float64] = np.empty((batch_size, *state_shape), dtype=numpy_dtype)
        batch_actions_array: NDArray[np.int64] = np.empty((batch_size, *action_shape), dtype=np.int64)
        batch_target_states_array: NDArray[np.float32 | np.float64] = np.empty(
            (batch_size, *state_shape), dtype=numpy_dtype
        )

        # Fill arrays
        for i, episode_meta in enumerate(batch_episodes):
            episode_idx: int = episode_meta["episode_index"]

            # Get base states (always used for targets)
            base_states = self.base_states[episode_idx]
            batch_target_states_array[i] = base_states

            # Construct episode states with caching
            episode_states: FloatArray = self._construct_episode_states(episode_idx, variation_name, base_states)
            batch_states_array[i] = episode_states

            # Add actions
            batch_actions_array[i] = self.episode_actions[episode_idx]

        # Convert to tensors without gradients
        with torch.inference_mode():
            batch_states: Tensor = torch.from_numpy(batch_states_array).to(dtype=self.dtype).detach()
            batch_actions: Tensor = torch.from_numpy(batch_actions_array).to(dtype=torch.long).detach()
            batch_target_states: Tensor = torch.from_numpy(batch_target_states_array).to(dtype=self.dtype).detach()
            episode_indices_tensor: Tensor = torch.from_numpy(episode_indices).detach()

        # Apply transforms if needed
        if self.transform is not None:
            batch_states = self._apply_transform(batch_states)
            batch_target_states = self._apply_transform(batch_target_states)

        result: BatchResult = {
            "variation_name": variation_name,
            "episode_indices": episode_indices_tensor,
            "states": batch_states,
            "actions": batch_actions,
            "target_states": batch_target_states,
        }

        # Handle encoded targets
        if self.use_encoded_targets:
            if self.precompute_targets:
                # Use pre-encoded states
                encoded_batch: NDArray[np.float32 | np.float64] = np.empty(
                    (batch_size, *self.encoded_base_states[sample_episode_idx].shape), dtype=numpy_dtype
                )
                for i, episode_meta in enumerate(batch_episodes):
                    episode_idx = episode_meta["episode_index"]
                    encoded_batch[i] = self.encoded_base_states[episode_idx]

                with torch.inference_mode():
                    batch_encoded_targets: Tensor = torch.from_numpy(encoded_batch).to(dtype=self.dtype).detach()
                    if self.transform is not None:
                        batch_encoded_targets = self._apply_transform(batch_encoded_targets)

                result["encoded_target_states"] = batch_encoded_targets

            elif self.encoder is not None:
                # Encode on the fly (no gradients)
                with torch.inference_mode():
                    result["encoded_target_states"] = self._encode_batch_states(batch_target_states)

        # Cache result with LRU eviction policy
        self._add_to_cache(cache_key, result)

        return result

    def _add_to_cache(self, cache_key: tuple[tuple[int, ...], str, bool], result: BatchResult) -> None:
        """Add result to cache with LRU eviction policy."""
        # If cache is at capacity, remove least recently used item
        cache_at_capacity: bool = len(self._batch_cache) >= self.cache_size
        key_not_in_cache: bool = cache_key not in self._batch_cache
        if cache_at_capacity and key_not_in_cache and self._cache_access_order:
            # Remove least recently used item
            lru_key: tuple[tuple[int, ...], str, bool] = self._cache_access_order.pop(0)
            if lru_key in self._batch_cache:
                # Explicitly delete tensors to free memory
                cached_batch: BatchResult = self._batch_cache[lru_key]
                for value in cached_batch.values():
                    if torch.is_tensor(value):
                        del value

                del self._batch_cache[lru_key]

        # Add to cache
        with torch.inference_mode():
            cloned_dict: BatchResult = {
                "variation_name": result["variation_name"],
                "episode_indices": result["episode_indices"].clone().detach(),
                "states": result["states"].clone().detach(),
                "actions": result["actions"].clone().detach(),
                "target_states": result["target_states"].clone().detach(),
            }
            if "encoded_target_states" in result:
                cloned_dict["encoded_target_states"] = result["encoded_target_states"].clone().detach()
            self._batch_cache[cache_key] = cloned_dict

        # Update access order
        if cache_key in self._cache_access_order:
            self._cache_access_order.remove(cache_key)

        self._cache_access_order.append(cache_key)

    def _apply_transform(self, tensor: Tensor) -> Tensor:
        """Apply transform.

        Args:
            tensor: Input tensor to transform.

        Returns:
            Transformed tensor.
        """
        if self.transform is None:
            return tensor.detach()

        try:
            with torch.inference_mode():
                return self.transform(tensor).detach()

        except (TypeError, ValueError):
            # Fallback to per-episode transform
            transformed_list: list[Tensor] = []
            with torch.inference_mode():
                transformed_list.extend(self.transform(tensor[i : i + 1]).detach() for i in range(tensor.shape[0]))

            return torch.cat(transformed_list, dim=0).detach()

    def _construct_episode_states(self, episode_idx: int, variation_name: str, base_states: FloatArray) -> FloatArray:
        """Construct episode state.

        Uses cache for frequently accessed combinations
        """
        cache_key: tuple[int, str, bool] = (episode_idx, variation_name, self.use_variation_for_all_states)
        if cache_key in self._episode_cache:
            return self._episode_cache[cache_key]

        result: FloatArray
        if variation_name == "base":
            # For base variation, always use base_states
            result = base_states

        elif variation_name not in self.variation_states or episode_idx not in self.variation_states[variation_name]:
            result = base_states

        else:
            var_states = self.variation_states[variation_name][episode_idx]

            if self.use_variation_for_all_states:
                # When use_variation_for_all_states=True, prioritize variation states
                # Only fall back to base_states if variation is completely missing or empty
                if len(var_states) == 0:
                    result = base_states
                elif len(var_states) >= len(base_states):
                    result = var_states
                else:
                    # If variation has fewer states than base, pad with base states
                    # Prefer the variation image when one is available.
                    episode_states: FloatArray = np.empty_like(base_states)
                    episode_states[: len(var_states)] = var_states
                    episode_states[len(var_states) :] = base_states[len(var_states) :]
                    result = episode_states

            elif len(var_states) > 0 and len(base_states) > 0:
                if len(base_states) == 1:
                    result = var_states[0:1]

                else:
                    # Use memory view instead of copy when possible
                    episode_states = np.empty_like(base_states)
                    episode_states[0] = var_states[0]
                    episode_states[1:] = base_states[1:]
                    result = episode_states

            else:
                result = base_states

        # Cache result if cache isn't too large
        if len(self._episode_cache) < self.cache_size:
            self._episode_cache[cache_key] = result

        return result

    def _encode_batch_chunk_tensor(self, chunk: Tensor) -> Tensor:
        """Encode one reshaped batch chunk and return it on CPU."""
        encoder: nn.Module | None = self.encoder
        if encoder is None:
            return chunk.cpu().detach()

        out: Tensor | tuple[Tensor, ...] | list[Tensor] = encoder(chunk)
        return _extract_encoded(out).detach().cpu()

    def _encode_batch_chunk_on_cpu(self, chunk: Tensor) -> Tensor:
        """Retry one chunk on CPU after a CUDA OOM."""
        encoder: nn.Module | None = self.encoder
        if encoder is None:
            return chunk.cpu().detach()

        chunk_cpu: Tensor = chunk.cpu().detach()
        encoder_cpu: nn.Module = encoder.cpu()
        out: Tensor | tuple[Tensor, ...] | list[Tensor] = encoder_cpu(chunk_cpu)
        encoded_chunk: Tensor = _extract_encoded(out)
        self.encoder = encoder_cpu.to(self.device)
        return encoded_chunk.detach()

    def _encode_batch_states(self, batch_states: Tensor) -> Tensor:
        """Encode batch of states."""
        if self.encoder is None:
            return batch_states

        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()

        with torch.inference_mode():
            # Reshape encoding
            original_shape: torch.Size = batch_states.shape
            reshaped_states: Tensor = batch_states.reshape(-1, *batch_states.shape[2:])

            # Recompute the chunk cap from the current memory estimate.
            adaptive_chunk_size: int = adaptive_batch_size(
                self.encoding_batch_size,
                reshaped_states.shape[1:],  # Shape per item
                self.dtype,
                self.device,
                self.safety_factors,
                operation_type="encoding",
            )
            chunk_size: int = min(adaptive_chunk_size, len(reshaped_states))
            encoded_chunks: list[Tensor] = []

            for i in range(0, len(reshaped_states), chunk_size):
                chunk: Tensor = reshaped_states[i : i + chunk_size].to(device=self.device).detach()

                try:
                    encoded_chunk: Tensor = self._encode_batch_chunk_tensor(chunk)
                    encoded_chunks.append(encoded_chunk)

                except torch.cuda.OutOfMemoryError:
                    logger.warning("OOM in batch encoding, using CPU fallback")
                    encoded_chunk = self._encode_batch_chunk_on_cpu(chunk)
                    encoded_chunks.append(encoded_chunk.detach())

            # Concatenate and reshape back
            encoded_batch: Tensor
            if encoded_chunks:
                encoded_reshaped: Tensor = torch.cat(encoded_chunks, dim=0).detach()
                encoded_batch = encoded_reshaped.reshape(
                    original_shape[0], original_shape[1], *encoded_reshaped.shape[1:]
                ).detach()

            else:
                encoded_batch = batch_states.detach()

        return encoded_batch.to(dtype=self.dtype).detach()

    def get_variation_info(self) -> VariationInfo:
        """Get information about available variations and their episode counts."""
        return {
            "variations": self.variation_names,
            "episodes_per_variation": self.total_episodes_per_variation.copy(),
            "tracking_status": self.variation_tracking.copy(),
            "batch_size": self.batch_size,
            "use_encoded_targets": self.use_encoded_targets,
            "precompute_targets": self.precompute_targets,
            "specific_variation": self.specific_variation,
            "use_variation_for_all_states": self.use_variation_for_all_states,
        }

    def reset_tracking(self, variation: str | None = None) -> None:
        """Reset tracking for a specific variation or all variations.

        Args:
            variation: Variation name to reset, or None to reset all.
        """
        if variation is not None:
            if variation in self.variation_tracking:
                self.variation_tracking[variation] = 0
                logger.info(f"TestingDataset reset tracking for variation: {variation}")

            else:
                logger.warning(f"Variation '{variation}' not found")

        else:
            for var_name in self.variation_tracking:
                self.variation_tracking[var_name] = 0

            logger.info("TestingDataset reset tracking for all variations")

        # Clear batch cache to prevent memory accumulation
        self._clear_batch_cache()

    def _clear_batch_cache(self) -> None:
        """Clear the batch cache to free memory."""
        cache_size_before: int = len(self._batch_cache)
        if cache_size_before > 0:
            # Drop cache entries before requesting garbage collection.
            for cache_key in list(self._batch_cache.keys()):
                cached_batch: BatchResult = self._batch_cache[cache_key]
                # Delete tensor references
                for value in cached_batch.values():
                    if torch.is_tensor(value):
                        del value
                del cached_batch

            self._batch_cache.clear()
            self._cache_access_order.clear()

            # Force garbage collection to free memory immediately
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.debug(f"Cleared batch cache: removed {cache_size_before} entries")

            # Reset cache statistics
            self._cache_hits = 0
            self._cache_misses = 0

    def has_remaining_batches(self, variation: str | None = None) -> bool:
        """Check if there are remaining batches for a variation or any variation.

        Args:
            variation: Variation name to check, or None to check all variations.

        Returns:
            True if there are remaining batches.
        """
        if variation is not None:
            if variation not in self.variation_tracking:
                return False

            episodes_returned: int = self.variation_tracking[variation]
            total_episodes: int = self.total_episodes_per_variation[variation]

            return episodes_returned < total_episodes

        return any(self.has_remaining_batches(var_name) for var_name in self.variation_names)

    def get_remaining_episodes(self, variation: str | None = None) -> int:
        """Get the number of remaining episodes for a variation or all variations.

        Args:
            variation: Variation name to check, or None to get total across all variations.

        Returns:
            Number of remaining episodes.
        """
        if variation is not None:
            if variation not in self.variation_tracking:
                return 0

            episodes_returned: int = self.variation_tracking[variation]
            total_episodes: int = self.total_episodes_per_variation[variation]

            return max(0, total_episodes - episodes_returned)

        return sum(self.get_remaining_episodes(var) for var in self.variation_names)

    def __len__(self) -> int:
        """Return the total number of batches across all variations."""
        total_batches = 0
        for var_name in self.variation_names:
            episodes_count: int = self.total_episodes_per_variation[var_name]
            batches_count: int = (episodes_count + self.batch_size - 1) // self.batch_size
            total_batches += batches_count

        return total_batches

    def set_variation(self, variation: str | None) -> None:
        """Change the specific variation to use.

        Args:
            variation: Variation name to switch to, or None for all variations.
        """
        if variation is not None and variation not in self.variation_names:
            logger.warning(f"Variation '{variation}' not found in dataset. Available: {self.variation_names}")
            return

        self.specific_variation = variation
        logger.info(f"TestingDataset switched to variation: {variation or 'all_variations'}")

    def set_variation_for_all_states(self, use_variation: bool) -> None:
        """Change whether to use variation for all states or just first state.

        Args:
            use_variation: If True, use variation for all states. If False, use variation only for first state.
        """
        self.use_variation_for_all_states = use_variation
        logger.debug(f"TestingDataset set use_variation_for_all_states to: {use_variation}")

    def _get_sample_tensor_shape(self) -> tuple[int, ...] | None:
        """Get sample tensor shape for memory estimation.

        Returns:
            Shape tuple of first available state tensor, or None if no data loaded
        """
        if self.base_states:
            sample_episode_idx: int = next(iter(self.base_states.keys()))
            sample_states: FloatArray = self.base_states[sample_episode_idx]

            if len(sample_states) > 0:
                return sample_states[0].shape

        return None

    def _calculate_hdf5_cache_params(self) -> tuple[int, int]:
        """Derive HDF5 cache parameters from available memory.

        Returns:
            Tuple of (cache_bytes, cache_slots)
        """

        def _next_prime_at_least(value: int) -> int:
            if value <= 2:
                return 2
            candidate = value if value % 2 == 1 else value + 1
            while True:
                limit = int(candidate**0.5)
                is_prime = True
                for divisor in range(3, limit + 1, 2):
                    if candidate % divisor == 0:
                        is_prime = False
                        break
                if is_prime:
                    return candidate
                candidate += 2

        cache_bytes: int
        cache_slots: int
        if not self.enable_memory_optimization:
            # Use fixed defaults when memory-based adjustment is disabled.
            cache_bytes = 128 * 1024**2  # 128MB
            cache_slots = 10007  # Conservative prime number for slots

            return cache_bytes, cache_slots

        if self.device.type == "cuda" and torch.cuda.is_available():
            _, _, total = get_gpu_memory_info()
            # Use a fraction of available GPU memory for HDF5 cache
            cache_factor = 0.05  # 5% of total GPU memory
            cache_bytes = int(total * cache_factor * (1024**3))
        else:
            _, total_ram = get_system_memory_info()
            # Use a fraction of total system memory for HDF5 cache
            cache_factor = 0.1  # 10% of total system memory
            cache_bytes = int(total_ram * cache_factor * (1024**3))

        # Clamp cache size between reasonable bounds
        cache_bytes = max(128 * 1024**2, min(cache_bytes, 4 * 1024**3))  # 128MB to 4GB

        # Calculate appropriate number of cache slots (prime number for better hashing)
        base_slots = min(50021, max(10007, cache_bytes // (64 * 1024)))  # 64KB per slot estimate
        cache_slots = _next_prime_at_least(base_slots)

        return cache_bytes, cache_slots

    def _filter_variations(self, available_variations: list[str]) -> list[str]:
        """Filter variations based on variations_to_use and variations_to_ignore parameters.

        Args:
            available_variations: List of all available variations in the dataset

        Returns:
            List of filtered variations to load
        """
        filtered_variations: list[str] = available_variations.copy()

        # First apply variations_to_use filter (if provided)
        if self.variations_to_use is not None and len(self.variations_to_use) > 0:
            # Build the set once before repeated membership checks.
            use_set: set[str] = set(self.variations_to_use)
            filtered_variations = [var for var in filtered_variations if var in use_set]

            # Log if any requested variations are not available
            missing_variations: set[str] = use_set - set(available_variations)
            if missing_variations:
                logger.warning(
                    f"Requested variations not found in dataset: {sorted(missing_variations)}. "
                    f"Available variations: {sorted(available_variations)}"
                )

        # Then apply variations_to_ignore filter (if provided)
        if self.variations_to_ignore is not None:
            # Build the set once before repeated membership checks.
            ignore_set: set[str] = set(self.variations_to_ignore)
            filtered_variations = [var for var in filtered_variations if var not in ignore_set]

            # Log ignored variations that were actually present
            ignored_present: set[str] = ignore_set.intersection(set(available_variations))
            if ignored_present:
                logger.info(f"Ignoring variations: {sorted(ignored_present)}")

        # Use the base variation when no named variation is present.
        if not filtered_variations:
            logger.warning("No variations remain after filtering. Defaulting to all available variations.")
            filtered_variations = available_variations

        logger.info(f"Loading {len(filtered_variations)} variations: {sorted(filtered_variations)}")

        return filtered_variations


class TestDataLoader:
    """Callable wrapper that switches variations on a TestingDataset."""

    def __init__(self, dataset: TestingDataset) -> None:
        """Initialize with a TestingDataset."""
        self.dataset: TestingDataset = dataset

    def __call__(
        self, variation: str | None = None, *, use_variation_for_all_states: bool | None = None
    ) -> TestingDataset:
        """Switch to a specific variation and optionally change state usage mode."""
        if variation is not None:
            self.dataset.set_variation(variation)

        if use_variation_for_all_states is not None:
            self.dataset.set_variation_for_all_states(use_variation_for_all_states)

        return self.dataset

    def __len__(self) -> int:
        """Return the total number of batches across all variations."""
        return len(self.dataset)

    def get_variation_info(self) -> VariationInfo:
        """Get information about available variations and their episode counts."""
        return self.dataset.get_variation_info()

    def reset_tracking(self, variation: str | None = None) -> None:
        """Reset tracking for a specific variation or all variations."""
        self.dataset.reset_tracking(variation)

    def has_remaining_batches(self, variation: str | None = None) -> bool:
        """Check if there are remaining batches for a variation or any variation."""
        return self.dataset.has_remaining_batches(variation)

    def get_remaining_episodes(self, variation: str | None = None) -> int:
        """Get the number of remaining episodes for a variation or all variations."""
        return self.dataset.get_remaining_episodes(variation)


def create_dataloader(
    file_path: str,
    batch_size: int = 32,
    *,
    transform: Callable[[Tensor], Tensor] | None = None,
    dtype: torch.dtype = torch.float32,
    encoder: nn.Module | None = None,
    use_encoded_targets: bool = False,
    precompute_targets: bool = False,
    device: str | torch.device = "cpu",
    specific_variation: str | None = None,
    use_variation_for_all_states: bool = False,
    enable_memory_optimization: bool = False,
    max_workers: int | None = None,
    prefetch_factor: int = 2,
    chunk_size: int | None = None,
    encoding_batch_size: int | None = None,
    use_memory_mapping: bool = True,
    cache_size: int | None = None,
    variations_to_use: list[str] | None = None,
    variations_to_ignore: list[str] | None = None,
) -> TestDataLoader:
    """Create a TestDataLoader from HDF5 file for testing trained models.

    Args:
        file_path: Path to HDF5 data file containing variation data.
        batch_size: Number of episodes per batch.
        transform: Optional transform function to apply to states.
        dtype: PyTorch dtype for tensor conversion.
        encoder: Optional pretrained encoder to encode base states.
        use_encoded_targets: If True, return both encoded and original target states.
        precompute_targets: If True, pre-encode all target states during initialization.
        device: Device to use for encoder computations.
        specific_variation: If provided, only return batches of this variation type.
        use_variation_for_all_states: If True, use variation for all states in episode.
        enable_memory_optimization: If True, derive unset loader sizes from memory
            estimates and reduce batches that exceed those estimates. When False,
            use the caller-supplied values without adjustment.
        max_workers: Number of parallel workers for data loading (auto-calculated if None).
        prefetch_factor: Number of batches queued by each worker.
        chunk_size: Size of chunks for HDF5 operations (auto-calculated if None).
        encoding_batch_size: Batch size for encoding operations (auto-calculated if None).
        use_memory_mapping: Whether to use memory mapping for file access.
        cache_size: Size of LRU cache for frequently accessed data (auto-calculated if None).
        variations_to_use: Specific variation names to include (for HDF5 loading).
            If None, all variations are included. If provided, only these variations
            will be loaded from the HDF5 file.
        variations_to_ignore: Variation names to exclude (for HDF5 loading).
            If None, no variations are excluded. If provided, these variations
            will be skipped during loading.

    Returns:
        A TestDataLoader instance that can be used to iterate over batches of episodes.

    When ``enable_memory_optimization`` is true, unset loader parameters are
    derived from available memory and batch sizes may be reduced after a sample
    tensor is loaded. When it is false, caller values are used unchanged.
    """
    dataset = TestingDataset(
        file_path=file_path,
        batch_size=batch_size,
        transform=transform,
        dtype=dtype,
        encoder=encoder,
        use_encoded_targets=use_encoded_targets,
        precompute_targets=precompute_targets,
        device=device,
        specific_variation=specific_variation,
        use_variation_for_all_states=use_variation_for_all_states,
        enable_memory_optimization=enable_memory_optimization,
        max_workers=max_workers,
        prefetch_factor=prefetch_factor,
        chunk_size=chunk_size,
        encoding_batch_size=encoding_batch_size,
        use_memory_mapping=use_memory_mapping,
        cache_size=cache_size,
        variations_to_use=variations_to_use,
        variations_to_ignore=variations_to_ignore,
    )

    return TestDataLoader(dataset)
