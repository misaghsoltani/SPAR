"""Generate offline state trajectories and their configured render variations."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from itertools import islice
from logging import getLogger
import math
import os
import pathlib
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

# Ray launches many renderer workers for this module. Keep math and image-library
# thread pools at one thread per process unless the caller explicitly overrides.
_THREADPOOL_ENV_DEFAULTS: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "OPENCV_FOR_THREADS_NUM": "1",
}
for _env_name, _env_value in _THREADPOOL_ENV_DEFAULTS.items():
    os.environ.setdefault(_env_name, _env_value)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Hashable, Sequence
    from logging import Logger
    from pathlib import Path
    from typing import TypeAlias

    import h5py
    import numpy as np
    from numpy.typing import NDArray
    import ray
    from ray import ObjectRef
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.rule import Rule
    from rich.table import Table

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.utils.config_utils.config_schema import DataConfig, GenDataSPARConfig
    from spar.utils.config_utils.samplers import Sampler, sampler_from_spec
    from spar.utils.data_utils.hdf5_common import (
        CompressionType,
        create_array_dataset,
        ensure_hdf5_path,
        get_chunk_shape_4d,
        normalize_compression_value,
        open_hdf5_for_write,
        write_utf8_string_attr,
        write_utf8_string_list_attr,
    )
    from spar.utils.env_utils import get_environment
    from spar.utils.env_utils.effects_core import (
        EffectConfigMap,
        EffectConfigValue,
        StagePipelines,
        build_stage_pipelines,
        freeze as freeze_effects_cfg,
    )
    from spar.utils.log_utils.console_logger import terminal_console as console
else:
    np = import_module("numpy")
    NDArray = import_module("numpy.typing").NDArray
    ray = import_module("ray")

    _rich_panel_module = import_module("rich.panel")
    Panel = _rich_panel_module.Panel
    _rich_progress_module = import_module("rich.progress")
    BarColumn = _rich_progress_module.BarColumn
    Progress = _rich_progress_module.Progress
    SpinnerColumn = _rich_progress_module.SpinnerColumn
    TextColumn = _rich_progress_module.TextColumn
    TimeElapsedColumn = _rich_progress_module.TimeElapsedColumn
    TimeRemainingColumn = _rich_progress_module.TimeRemainingColumn
    Rule = import_module("rich.rule").Rule
    Table = import_module("rich.table").Table

    _samplers_module = import_module("spar.utils.config_utils.samplers")
    Sampler = _samplers_module.Sampler
    sampler_from_spec = _samplers_module.sampler_from_spec
    _hdf5_common_module = import_module("spar.utils.data_utils.hdf5_common")
    create_array_dataset = _hdf5_common_module.create_array_dataset
    ensure_hdf5_path = _hdf5_common_module.ensure_hdf5_path
    get_chunk_shape_4d = _hdf5_common_module.get_chunk_shape_4d
    normalize_compression_value = _hdf5_common_module.normalize_compression_value
    open_hdf5_for_write = _hdf5_common_module.open_hdf5_for_write
    write_utf8_string_attr = _hdf5_common_module.write_utf8_string_attr
    write_utf8_string_list_attr = _hdf5_common_module.write_utf8_string_list_attr
    get_environment = import_module("spar.utils.env_utils").get_environment
    _effects_core_module = import_module("spar.utils.env_utils.effects_core")
    build_stage_pipelines = _effects_core_module.build_stage_pipelines
    freeze_effects_cfg = _effects_core_module.freeze
    console = import_module("spar.utils.log_utils.console_logger").terminal_console


ImageArray: TypeAlias = NDArray[np.uint8 | np.float32]
StateVariation: TypeAlias = list[tuple[str, ImageArray]]
EpisodeResult: TypeAlias = ImageArray | tuple[ImageArray, list[StateVariation]]

logger: Logger = getLogger(__name__)

TARGET_BATCHES_PER_WORKER: int = 8
MIN_BATCH_SIZE_PER_WORKER: int = 32
HEAVY_VARIATION_MIN_BATCH_SIZE_PER_WORKER: int = 8
MAX_BATCH_SIZE_PER_WORKER: int = 2048


@runtime_checkable
class BaseImagePostRenderer(Protocol):
    """Render post-stage variations from an existing base image."""

    def render_post_variation_from_base(self, base_image: ImageArray, effects: StagePipelines) -> ImageArray:
        """Apply a post-stage pipeline while preserving renderer side effects.

        Args:
            base_image: Existing channels-first base image.
            effects: Pipeline containing only post-render effects.

        Returns:
            The varied image in the same layout as ``base_image``.
        """
        ...


def ensure_ray_initialized() -> None:
    """Initialize Ray lazily for data generation callers."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {key: os.environ[key] for key in _THREADPOOL_ENV_DEFAULTS if key in os.environ}},
        )


def _create_actions_dataset(
    group: h5py.Group, name: str, actions: NDArray[np.int32], compression: CompressionType
) -> h5py.Dataset:
    """Create an actions dataset with append-friendly chunking."""
    chunk_shape: tuple[int] = (min(2048, max(1, len(actions))),)
    return create_array_dataset(group, name, actions, compression=compression, chunks=chunk_shape, shuffle=True)


def _count_enabled_effects(effects_config: EffectConfigMap | None) -> int:
    """Count configured variation pipelines that will produce rendered states."""
    if not effects_config:
        return 0

    count: int = 0
    for effect_cfg in effects_config.values():
        if isinstance(effect_cfg, Mapping) and effect_cfg.get("enabled", True) is False:
            continue
        count += 1
    return count


def _estimate_render_work_multiplier(
    *,
    num_episodes: int,
    total_states: int,
    use_variations: bool,
    effects_config: EffectConfigMap | None,
    first_state_only: bool,
) -> float:
    """Estimate render work per scheduled state, including base and variation renders."""
    total_states = max(1, total_states)
    if not use_variations:
        return 1.0

    effect_count = _count_enabled_effects(effects_config)
    if effect_count <= 0:
        return 1.0

    if first_state_only:
        varied_states = max(0, num_episodes)
        return max(1.0, 1.0 + (varied_states * effect_count / total_states))

    return float(1 + effect_count)


def calculate_worker_batch_size(
    num_episodes: int,
    num_steps_per_episode: int,
    num_workers: int,
    *,
    use_variations: bool = False,
    effects_config: EffectConfigMap | None = None,
    first_state_only: bool = False,
) -> int:
    """Calculate batch size for throughput.

    Uses a target of 8 batches per worker and scales the scheduled state
    batch down when each state expands into many variation renders.

    Args:
        num_episodes: Total number of episodes.
        num_steps_per_episode: Steps per episode.
        num_workers: Number of parallel workers.
        use_variations: Whether variations are enabled (effects_config will be inspected if True).
        effects_config: Effects configuration to estimate variation count (if use_variations is True).
        first_state_only: Whether only first states have variations, which reduces the effective work multiplier.

    Returns:
        Batch size in number of states (>=1).
    """
    total_states: int = max(1, num_episodes * (num_steps_per_episode + 1))
    work_multiplier = _estimate_render_work_multiplier(
        num_episodes=num_episodes,
        total_states=total_states,
        use_variations=use_variations,
        effects_config=effects_config,
        first_state_only=first_state_only,
    )
    return calculate_worker_batch_size_from_state_count(
        total_states=total_states, num_workers=num_workers, work_multiplier=work_multiplier
    )


def calculate_worker_batch_size_from_state_count(
    total_states: int, num_workers: int, *, work_multiplier: float = 1.0
) -> int:
    """Calculate batch size from total state count and worker count."""
    total_states = max(1, total_states)
    num_workers = max(1, num_workers)
    work_multiplier = max(1.0, work_multiplier)

    # Aim for a fixed number of batches per worker
    total_batches: int = num_workers * TARGET_BATCHES_PER_WORKER

    # Compute raw scheduled states per task. Heavy variation pipelines already
    # have enough work in small batches, so scale by estimated render work.
    batch_size: int = max(1, math.ceil(total_states / (total_batches * work_multiplier)))

    # Clamp to realistic bounds
    if total_states < MIN_BATCH_SIZE_PER_WORKER:
        return total_states

    min_batch_size = HEAVY_VARIATION_MIN_BATCH_SIZE_PER_WORKER if work_multiplier >= 8.0 else MIN_BATCH_SIZE_PER_WORKER
    batch_size = max(batch_size, min_batch_size)
    return min(batch_size, MAX_BATCH_SIZE_PER_WORKER)


@dataclass(slots=True)
class BatchProcessingResult:
    """Result from processing a batch of states."""

    batch_id: int
    start_state_idx: int
    end_state_idx: int
    base_images: ImageArray
    variations: list[list[tuple[str, ImageArray]]] | None
    error: str | None = None


@dataclass(slots=True)
class BatchWorkerConfig:
    """Configuration for batch worker processes."""

    env_name: str
    use_variations: bool
    effects_config: EffectConfigMap | None
    effects_has_sampler: bool
    effects_cache_key: Hashable | None
    first_state_only: bool
    variation_sharing_group_size: int


env_cache: dict[str, ABCEnvironment[ABCState]] = {}

# Cache compiled effect pipelines per worker-process to avoid rebuilding them
pipeline_cache: dict[Hashable, dict[str, StagePipelines]] = {}
MAX_PIPELINE_CACHE_ENTRIES: int = 32
OBJECT_STORE_TARGET_UTILIZATION: float = 0.40


def _cache_compiled_pipelines(cache_key: Hashable, compiled: dict[str, StagePipelines]) -> None:
    """Store compiled pipelines in a bounded per-worker cache."""
    if cache_key in pipeline_cache:
        return
    if len(pipeline_cache) >= MAX_PIPELINE_CACHE_ENTRIES:
        pipeline_cache.pop(next(iter(pipeline_cache)))
    pipeline_cache[cache_key] = compiled


def _estimate_batch_result_nbytes(result: BatchProcessingResult) -> int:
    """Estimate result payload size in bytes for memory-aware backpressure tuning."""
    total_bytes: int = int(result.base_images.nbytes)
    if result.variations is None:
        return total_bytes

    state_vars: list[tuple[str, ImageArray]]
    image: ImageArray
    for state_vars in result.variations:
        for _, image in state_vars:
            total_bytes += int(image.nbytes)
    return total_bytes


def _compute_safe_in_flight_cap(current_cap: int, object_store_bytes: int, batch_nbytes: int) -> int:
    """Compute a memory-safe pending-task cap from object-store capacity."""
    if current_cap <= 1 or object_store_bytes <= 0 or batch_nbytes <= 0:
        return max(1, current_cap)

    target_bytes: int = max(batch_nbytes, int(object_store_bytes * OBJECT_STORE_TARGET_UTILIZATION))
    return max(1, min(current_cap, target_bytes // batch_nbytes))


@dataclass(slots=True)
class EpisodeResultReconstructor:
    """Incrementally reconstruct per-episode outputs as batches complete."""

    episode_lengths: list[int]
    episode_offsets: list[int]
    episode_images: list[ImageArray] | None = None
    episode_variations: list[list[StateVariation]] | None = None

    @classmethod
    def from_state_trajectories(cls, state_trajs: list[list[ABCState]]) -> EpisodeResultReconstructor:
        """Create reconstruction state from episode trajectories."""
        episode_lengths: list[int] = [len(traj) for traj in state_trajs]
        episode_offsets: list[int] = [0]
        for length in episode_lengths:
            episode_offsets.append(episode_offsets[-1] + length)
        return cls(episode_lengths=episode_lengths, episode_offsets=episode_offsets)

    def ingest_batch(self, result: BatchProcessingResult) -> None:
        """Insert a single batch result into episode-aligned output buffers."""
        if self.episode_images is None:
            image_shape: tuple[int, ...] = result.base_images.shape[1:]
            image_dtype: np.dtype[np.float32 | np.uint8] = result.base_images.dtype
            self.episode_images = [
                np.empty((length, *image_shape), dtype=image_dtype) for length in self.episode_lengths
            ]

        if result.variations is not None and self.episode_variations is None:
            self.episode_variations = [[[] for _ in range(length)] for length in self.episode_lengths]
        elif result.variations is None and self.episode_variations is not None:
            raise RuntimeError(f"Batch {result.batch_id} omitted variations after variation buffers were initialized")

        assert self.episode_images is not None

        batch_start: int = result.start_state_idx
        batch_end: int = result.end_state_idx
        local_offset: int = 0

        while batch_start < batch_end:
            episode_idx: int = bisect_right(self.episode_offsets, batch_start) - 1
            if episode_idx < 0 or episode_idx >= len(self.episode_lengths):
                raise RuntimeError(f"Batch {result.batch_id} maps to out-of-range episode index {episode_idx}")

            episode_start: int = self.episode_offsets[episode_idx]
            in_episode_idx: int = batch_start - episode_start
            capacity: int = self.episode_lengths[episode_idx] - in_episode_idx
            take: int = min(capacity, batch_end - batch_start)
            if take <= 0:
                raise RuntimeError(f"Batch {result.batch_id} produced invalid reconstruction slice size {take}")

            target_images: ImageArray = self.episode_images[episode_idx]
            target_images[in_episode_idx : in_episode_idx + take] = result.base_images[
                local_offset : local_offset + take
            ]

            if result.variations is not None and self.episode_variations is not None:
                self.episode_variations[episode_idx][in_episode_idx : in_episode_idx + take] = result.variations[
                    local_offset : local_offset + take
                ]

            batch_start += take
            local_offset += take

    def finalize(self) -> list[EpisodeResult]:
        """Materialize final per-episode outputs after all batches are ingested."""
        if self.episode_images is None:
            return []
        if self.episode_variations is None:
            return list(self.episode_images)
        return [
            (base_imgs, variations)
            for base_imgs, variations in zip(self.episode_images, self.episode_variations, strict=True)
        ]


def contains_sampler(config_value: EffectConfigValue | EffectConfigMap) -> bool:
    """Recursively detect whether an effect configuration includes a sampler."""
    if isinstance(config_value, Sampler):
        return True
    if isinstance(config_value, str) and sampler_from_spec(config_value) is not None:
        return True
    if isinstance(config_value, Mapping):
        return any(contains_sampler(value) for value in config_value.values())
    if isinstance(config_value, (list, tuple, set)):
        return any(contains_sampler(value) for value in config_value)
    return False


def _process_state_batch_impl(
    batch_id: int,
    start_idx: int,
    config: BatchWorkerConfig,
    batch_states: Sequence[ABCState],
    first_state_flags: Sequence[bool] | None,
) -> BatchProcessingResult:
    """Process a slice of states [start_idx:end_idx) and return the rendered images plus any requested variations."""
    env: ABCEnvironment[ABCState] | None = env_cache.get(config.env_name)
    if env is None:
        env = get_environment(config.env_name)
        env_cache[config.env_name] = env
    assert env is not None  # Either from cache or just created
    env_state_to_real: Callable[..., ImageArray] = env.state_to_real

    # Fast-path: if variations are disabled or no effects_config provided, we can
    # skip fetching the (potentially large) metadata list and any pipeline work.
    variations_enabled = bool(config.use_variations and config.effects_config)

    states: list[ABCState] = batch_states if isinstance(batch_states, list) else list(batch_states)
    bsz: int = len(states)
    end_idx: int = start_idx + bsz

    # Render the base (no-effect) images
    base_images: ImageArray = env_state_to_real(states)  # (N, 3, H, 2W)

    # Build variations (if requested)
    batch_variations: list[list[tuple[str, ImageArray]]] | None = None

    if variations_enabled:
        batch_variations = [[] for _ in range(bsz)]  # keep output shape

        # Normalize effects config and determine if re-sampling is required
        effects_cfg: EffectConfigMap = config.effects_config or {}
        has_sampler: bool = config.effects_has_sampler

        # Try to reuse cached compiled pipelines when there are no Samplers
        cached_pipelines: dict[str, StagePipelines] | None = None
        cache_key: Hashable | None = config.effects_cache_key
        if not has_sampler and cache_key is not None:
            cached_pipelines = pipeline_cache.get(cache_key)
            if cached_pipelines is None:
                cached_pipelines = build_stage_pipelines(effects_cfg)
                assert cached_pipelines is not None  # build_stage_pipelines always returns a dict
                _cache_compiled_pipelines(cache_key, cached_pipelines)

        # FIRST-STATE-ONLY: only render variations for first states
        if config.first_state_only:
            first_indices: list[int] = [i for i, is_first in enumerate(first_state_flags or []) if is_first]
            if first_indices:
                first_states: list[ABCState] = [states[i] for i in first_indices]
                pipelines_dict: dict[str, StagePipelines] = cached_pipelines or build_stage_pipelines(effects_cfg)

                # Render once per pipeline for the subset of first_states
                first_vars: list[tuple[str, ImageArray]] = [
                    (name, env_state_to_real(first_states, effects=pipeline))
                    for name, pipeline in pipelines_dict.items()
                ]

                # Distribute results back into per-state slots
                for local_pos, batch_pos in enumerate(first_indices):
                    batch_variations[batch_pos] = [(var_name, var_imgs[local_pos]) for var_name, var_imgs in first_vars]

        # GROUP-SHARING: process states in small groups sharing variation parameters
        else:
            can_reuse_base_images: bool = (
                cached_pipelines is not None
                and config.variation_sharing_group_size == 1
                and isinstance(env, BaseImagePostRenderer)
                and all(pipeline.pre is None and pipeline.obj is None for pipeline in cached_pipelines.values())
            )
            if can_reuse_base_images:
                assert cached_pipelines is not None
                assert isinstance(env, BaseImagePostRenderer)
                for state_idx, base_image in enumerate(base_images):
                    batch_variations[state_idx] = [
                        (name, env.render_post_variation_from_base(base_image, pipeline))
                        for name, pipeline in cached_pipelines.items()
                    ]
            else:
                gsz: int = config.variation_sharing_group_size
                group_states: list[ABCState]
                for group_start in range(0, bsz, gsz):
                    group_end: int = min(group_start + gsz, bsz)
                    group_states = states[group_start:group_end]
                    pipelines_dict = cached_pipelines or build_stage_pipelines(effects_cfg)

                    group_vars: list[tuple[str, ImageArray]] = [
                        (name, env_state_to_real(group_states, effects=pipeline))
                        for name, pipeline in pipelines_dict.items()
                    ]

                    for i, state_idx in enumerate(range(group_start, group_end)):
                        batch_variations[state_idx] = [(var_name, var_imgs[i]) for var_name, var_imgs in group_vars]

    return BatchProcessingResult(
        batch_id=batch_id,
        start_state_idx=start_idx,
        end_state_idx=end_idx,
        base_images=base_images,
        variations=batch_variations,
    )


@ray.remote(num_cpus=1)
def process_state_batch(
    batch_id: int,
    start_idx: int,
    config: BatchWorkerConfig,
    batch_states: Sequence[ABCState],
    first_state_flags: Sequence[bool] | None,
) -> BatchProcessingResult:
    """Process a slice of states [start_idx:end_idx) and return the rendered images plus any requested variations."""
    try:
        return _process_state_batch_impl(batch_id, start_idx, config, batch_states, first_state_flags)
    except Exception:
        logger.exception(f"process_state_batch failed for batch_id={batch_id}")
        raise


def format_time_duration(seconds: float) -> str:
    """Format time duration in HH:MM:SS format.

    Args:
        seconds: Time duration in seconds.

    Returns:
        Formatted time duration string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class BatchProcessingManager:
    """Coordinate worker processes for state-based batch processing."""

    def __init__(
        self,
        env_name: str,
        use_variations: bool,
        effects_config: EffectConfigMap | None,
        *,
        first_state_only: bool,
        variation_sharing_group_size: int,
        num_workers: int,
        batch_size_per_worker: int,
    ) -> None:
        if num_workers <= 0:
            raise ValueError(f"num_workers must be > 0, got {num_workers}")
        if batch_size_per_worker <= 0:
            raise ValueError(f"batch_size_per_worker must be > 0, got {batch_size_per_worker}")
        if variation_sharing_group_size <= 0:
            raise ValueError(f"variation_sharing_group_size must be > 0, got {variation_sharing_group_size}")

        effective_effects_cfg: EffectConfigMap | None = effects_config if use_variations else None
        effects_has_sampler: bool = bool(effective_effects_cfg and contains_sampler(effective_effects_cfg))
        effects_cache_key: Hashable | None = None
        if effective_effects_cfg and not effects_has_sampler:
            effects_cache_key = freeze_effects_cfg(effective_effects_cfg)

        self.num_workers: int = num_workers
        self.batch_size_per_worker: int = batch_size_per_worker
        self.config_template: BatchWorkerConfig = BatchWorkerConfig(
            env_name=env_name,
            use_variations=use_variations,
            effects_config=effective_effects_cfg,
            effects_has_sampler=effects_has_sampler,
            effects_cache_key=effects_cache_key,
            first_state_only=first_state_only,
            variation_sharing_group_size=variation_sharing_group_size,
        )

    def process_episodes(
        self, state_trajs: list[list[ABCState]], action_trajs: list[list[int]]
    ) -> tuple[list[EpisodeResult], list[list[int]]]:
        """Process episodes with state-based batching."""
        total_eps: int = len(state_trajs)
        total_steps: int = sum(len(t) for t in state_trajs)
        if total_eps == 0:
            return [], action_trajs

        logger.info(
            f"Starting state-based processing {total_eps:,} episodes ({total_steps:,} states) "
            f"with {self.num_workers} Ray workers, batch size {self.batch_size_per_worker}..."
        )

        cfg: BatchWorkerConfig = self.config_template
        need_first_state_flags: bool = bool(cfg.first_state_only and cfg.use_variations and cfg.effects_config)

        total_states: int = total_steps
        if total_states == 0:
            logger.info("No states are available for rendering. Returning actions only.")
            return [], action_trajs
        cfg_ref: ObjectRef[BatchWorkerConfig] = ray.put(cfg)

        reconstructor: EpisodeResultReconstructor = EpisodeResultReconstructor.from_state_trajectories(state_trajs)

        n_batches: int = math.ceil(total_states / self.batch_size_per_worker)
        logger.info(f"Using {n_batches} batches of {self.batch_size_per_worker} states each")
        max_in_flight: int = max(1, min(self.num_workers, n_batches))
        logger.info(f"Ray backpressure cap set to {max_in_flight} in-flight batches")
        object_store_bytes: int = ray.available_resources().get("object_store_memory", 0)

        state_stream: Generator[ABCState, None, None] | None = None
        state_stream_with_flags: Generator[tuple[ABCState, bool], None, None] | None = None
        if need_first_state_flags:
            state_stream_with_flags = (
                (state, state_idx == 0) for traj in state_trajs for state_idx, state in enumerate(traj)
            )
        else:
            state_stream = (state for traj in state_trajs for state in traj)

        # Progress setup
        step_threshold: float = total_states * 0.1
        next_milestone: float = step_threshold
        milestones_hit = 0

        def log_milestone(completed_states: int, total_states: int, start_time: float) -> None:
            """Log milestone progress if threshold is reached."""
            nonlocal milestones_hit, next_milestone
            if completed_states >= next_milestone and milestones_hit < 10:
                milestones_hit += 1
                pct: int = milestones_hit * 10
                elapsed: float = time.time() - start_time
                rate: float = completed_states / elapsed if elapsed > 0 else 0

                # Calculate estimated remaining time
                eta_str: str
                if completed_states > 0 and rate > 0:
                    remaining_states: int = total_states - completed_states
                    eta_seconds: float = remaining_states / rate
                    eta_str = format_time_duration(eta_seconds)
                else:
                    eta_str = "--:--:--"

                elapsed_str: str = format_time_duration(elapsed)
                logger.info(
                    f"Milestone: {pct}% • {completed_states:,}/{total_states:,} states • "
                    f"{rate:.0f} states/s • {elapsed_str} elapsed • {eta_str} remaining"
                )
                next_milestone += step_threshold

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("|"),
            TextColumn("{task.completed:>3}/{task.total}"),
            TextColumn("|"),
            TimeElapsedColumn(),
            TextColumn("|"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            batch_task: TaskID = progress.add_task(f"[cyan]Processing {n_batches} batches", total=n_batches)
            state_task: TaskID = progress.add_task("[yellow]States processed", total=total_states)

            pending: list[ObjectRef[BatchProcessingResult]] = []
            seen_batches: list[bool] = [False] * n_batches
            completed_states: int = 0
            next_batch_id: int = 0
            next_start_idx: int = 0
            t0: float = time.time()
            first_batch_size_bytes: int | None = None

            # Log 0% milestone
            logger.info(
                f"Milestone: 0% • 0/{total_states:,} states • 0 states/s • 00:00:00 elapsed • --:--:-- remaining"
            )

            def consume_result(res: BatchProcessingResult) -> None:
                """Validate and consume one completed batch result."""
                nonlocal completed_states, max_in_flight, first_batch_size_bytes

                if res.batch_id < 0 or res.batch_id >= n_batches:
                    raise RuntimeError(f"Received out-of-range batch_id={res.batch_id}, expected [0, {n_batches - 1}]")
                if seen_batches[res.batch_id]:
                    raise RuntimeError(f"Received duplicate batch result for batch_id={res.batch_id}")

                seen_batches[res.batch_id] = True
                reconstructor.ingest_batch(res)

                progress.advance(batch_task, 1)
                states_in_batch: int = res.end_state_idx - res.start_state_idx
                completed_states += states_in_batch
                progress.update(state_task, completed=completed_states)
                log_milestone(completed_states, total_states, t0)

                if first_batch_size_bytes is None:
                    first_batch_size_bytes = _estimate_batch_result_nbytes(res)
                    safe_cap: int = _compute_safe_in_flight_cap(
                        max_in_flight, object_store_bytes, first_batch_size_bytes
                    )
                    if safe_cap < max_in_flight:
                        logger.info(
                            f"Reducing in-flight cap from {max_in_flight} to {safe_cap} "
                            f"(estimated batch payload {first_batch_size_bytes / (1024**2):.2f} MB, "
                            f"object store {object_store_bytes / (1024**3):.2f} GB)"
                        )
                        max_in_flight = safe_cap

            def next_batch_payload() -> tuple[list[ABCState], list[bool] | None]:
                """Read and validate the next slice from the state stream."""
                if need_first_state_flags:
                    assert state_stream_with_flags is not None
                    chunk: list[tuple[ABCState, bool]] = list(
                        islice(state_stream_with_flags, self.batch_size_per_worker)
                    )
                    if not chunk:
                        raise RuntimeError("State stream exhausted before scheduling all batches")
                    return [state for state, _ in chunk], [is_first for _, is_first in chunk]

                assert state_stream is not None
                next_states: list[ABCState] = list(islice(state_stream, self.batch_size_per_worker))
                if not next_states:
                    raise RuntimeError("State stream exhausted before scheduling all batches")
                return next_states, None

            # Submit & consume with explicit backpressure.
            def run_batch_processing_loop() -> None:
                """Submit and consume Ray batches with explicit backpressure."""
                nonlocal next_batch_id, next_start_idx, pending
                while next_batch_id < n_batches or pending:
                    while next_batch_id < n_batches and len(pending) < max_in_flight:
                        batch_states, batch_first_flags = next_batch_payload()

                        pending.append(
                            process_state_batch.remote(
                                next_batch_id, next_start_idx, cfg_ref, batch_states, batch_first_flags
                            )
                        )
                        next_start_idx += len(batch_states)
                        next_batch_id += 1

                    if not pending:
                        break

                    done, pending = ray.wait(pending, num_returns=1)
                    consume_result(ray.get(done[0]))

            try:
                run_batch_processing_loop()
            except Exception:
                try:
                    for obj_ref in pending:
                        ray.cancel(obj_ref)
                except Exception:
                    logger.debug("Failed to cancel pending batch refs during error unwinding", exc_info=True)
                raise

            # Emit the final progress milestone.
            if completed_states == total_states and milestones_hit < 10:
                elapsed: float = time.time() - t0
                rate: float = completed_states / elapsed if elapsed > 0 else 0
                elapsed_str: str = format_time_duration(elapsed)
                logger.info(
                    f"Milestone: 100% • {completed_states:,}/{total_states:,} states • "
                    f"{rate:.0f} states/s • {elapsed_str} elapsed • 00:00:00 remaining"
                )

            # Final stats
            total_time: float = time.time() - t0
            eps_rate: float = total_eps / total_time if total_time > 0 else 0
            state_rate: float = total_states / total_time if total_time > 0 else 0
            logger.info(
                f"[bold green]✓[/bold green] Processing complete: {eps_rate:.1f} eps/s, {state_rate:.0f} states/s"
            )

        if next_batch_id != n_batches:
            raise RuntimeError(
                f"Scheduled {next_batch_id} batches but expected {n_batches}. This indicates state streaming drift"
            )

        missing_ids: list[int] = [i for i, seen in enumerate(seen_batches) if not seen]
        if missing_ids:
            raise RuntimeError(f"Missing batch results for ids: {missing_ids}")

        episode_results: list[EpisodeResult] = reconstructor.finalize()
        if len(episode_results) != total_eps:
            raise RuntimeError(
                f"Reconstructed {len(episode_results)} episodes but expected {total_eps}. Reconstruction drift detected"
            )
        return episode_results, action_trajs

    @staticmethod
    def _reconstruct_episodes(
        batch_results: list[BatchProcessingResult], state_trajs: list[list[ABCState]]
    ) -> list[EpisodeResult]:
        """Reconstruct episode-based results from state batch results."""
        if not batch_results:
            return []

        reconstructor: EpisodeResultReconstructor = EpisodeResultReconstructor.from_state_trajectories(state_trajs)
        for result in batch_results:
            reconstructor.ingest_batch(result)
        return reconstructor.finalize()


def generate_episodes(
    env: ABCEnvironment[ABCState], num_eps: int, num_steps: int, start_seed: int, num_levels: int
) -> tuple[list[list[ABCState]], list[list[int]]]:
    """Generate raw state and action trajectories."""
    logger.info(f"Generating {num_eps:,} episodes of {num_steps} steps each...")
    t0: float = time.time()
    state_trajs: list[list[ABCState]]
    action_trajs: list[list[int]]
    _, _, state_trajs, action_trajs = env.generate_episodes([num_steps] * num_eps, start_seed, num_levels)
    dt: float = time.time() - t0
    logger.info(f"[green]✓[/green] Generated {num_eps:,} episodes in {dt:.2f}s ({num_eps / dt:.2f} eps/s)")
    return state_trajs, action_trajs


def detect_variant_type(processed_episodes: list[EpisodeResult]) -> str:
    """Detect whether episodes use 'none', 'full' or 'first_state_only' variant pattern."""
    # If there are no episodes at all, default to "none"
    if not processed_episodes:
        return "none"

    # Find the first (base_imgs, variations_per_state) tuple where variations_per_state is non-empty.
    first_with_vars: EpisodeResult | None = next(
        (ep for ep in processed_episodes if isinstance(ep, tuple) and ep[1]), None
    )
    if first_with_vars is None:
        # No tuple had any variations_per_state => no variations used at all
        return "none"

    # Unpack
    variations_per_state: list[StateVariation]
    _, variations_per_state = first_with_vars
    return "full" if any(variations_per_state[1:]) else "first_state_only"


def extract_variant_names(processed_episodes: list[EpisodeResult]) -> list[str]:
    """Extract all unique variant names from processed episodes."""
    variant_names: list[str] = ["base"]
    seen_variant_names: set[str] = {"base"}

    variations_per_state: list[StateVariation]
    for episode in processed_episodes:
        if isinstance(episode, tuple):
            _, variations_per_state = episode
            for state_vars in variations_per_state:
                for var_name, _ in state_vars:
                    if var_name not in seen_variant_names:
                        seen_variant_names.add(var_name)
                        variant_names.append(var_name)

    return variant_names


def write_episode(
    h5_file: h5py.File,
    episode_idx: int,
    episode_data: EpisodeResult,
    actions: list[int],
    variant_names: list[str],
    variant_type: str,
    compression: CompressionType = "none",
) -> None:
    """Write a single episode to the file."""
    episode_group: h5py.Group = h5_file.create_group(f"episodes/episode_{episode_idx:07d}")
    episode_group.attrs["variant_type"] = variant_type

    # Extract base images and setup dimensions
    base_imgs: ImageArray = episode_data[0] if isinstance(episode_data, tuple) else episode_data
    time_steps: int = base_imgs.shape[0]

    # Validate action/state alignment
    if len(actions) != time_steps - 1:
        raise ValueError(
            f"Episode {episode_idx}: actions length {len(actions)} != states length - 1 ({time_steps - 1})"
        )

    # Write actions dataset
    actions_array: NDArray[np.int32] = np.array(actions, dtype=np.int32)
    _create_actions_dataset(episode_group, "actions", actions_array, compression)

    # Prepare variants data dictionary
    variants_data: dict[str, ImageArray] = {"base": base_imgs}

    # Process variations if present
    variations_per_state: list[StateVariation]
    if isinstance(episode_data, tuple):
        _, variations_per_state = episode_data
        for variant_name in variant_names:
            if variant_name != "base":
                imgs: list[ImageArray | None] = [
                    next((img for name, img in state_vars if name == variant_name), None)
                    for state_vars in variations_per_state
                ]
                valid_imgs: list[ImageArray] = [img for img in imgs if img is not None]
                if not valid_imgs:
                    logger.warning(f"Warning: No valid images found for variant '{variant_name}', skipping")
                else:
                    shapes: set[tuple[int, int]] = {img.shape for img in valid_imgs}
                    if len(shapes) == 1:
                        variants_data[variant_name] = np.stack(valid_imgs, axis=0)
                    else:
                        logger.info(
                            f"Warning: Skipping variant '{variant_name}' due to mismatched image shapes: {shapes}"
                        )

    # Write variant datasets
    variant_imgs: ImageArray
    for variant_name in variant_names:
        if variant_name in variants_data:
            variant_imgs = variants_data[variant_name]

            # Apply first_state_only optimization for non-base variants
            if variant_type == "first_state_only" and variant_name != "base":
                variant_imgs = variant_imgs[0:1]

            create_array_dataset(
                episode_group,
                f"{variant_name}/states",
                variant_imgs,
                compression=compression,
                chunks=get_chunk_shape_4d(variant_imgs),
            )


def _write_processed_episodes_file(
    tmp_path: Path,
    final_path: Path,
    processed_episodes: list[EpisodeResult],
    action_trajs: list[list[int]],
    variant_names: list[str],
    variant_type: str,
    compression: CompressionType,
) -> None:
    """Write processed episodes to a temporary HDF5 file and atomically replace the final path."""
    num_episodes: int = len(processed_episodes)
    with open_hdf5_for_write(str(tmp_path)) as h5_file:
        # Write root attributes
        h5_file.attrs["num_episodes"] = num_episodes
        write_utf8_string_list_attr(h5_file.attrs, "variant_names", variant_names)
        write_utf8_string_attr(h5_file.attrs, "variant_type", variant_type)

        # Create episodes group
        h5_file.create_group("episodes")

        # Progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Writing data"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("|"),
            TextColumn("{task.completed:>3}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            write_task: TaskID = progress.add_task("Writing episodes", total=num_episodes)

            # Write each episode
            for episode_idx, (episode_data, actions) in enumerate(zip(processed_episodes, action_trajs, strict=True)):
                write_episode(h5_file, episode_idx, episode_data, actions, variant_names, variant_type, compression)
                progress.advance(write_task, 1)

        h5_file.flush()

    pathlib.Path(tmp_path).replace(final_path)


def save_data(
    path: str,
    processed_episodes: list[EpisodeResult],
    action_trajs: list[list[int]],
    compression: CompressionType = "none",
    *,
    variant_type: str | None = None,
) -> None:
    """Save processed episodes and actions to file."""
    path = ensure_hdf5_path(path)
    final_path = pathlib.Path(path)
    final_path.parent.mkdir(exist_ok=True, parents=True)
    tmp_path: Path = final_path.with_name(f".{final_path.name}.tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    logger.info("Saving processed episodes to file...")
    logger.info("Processing episode structure...")

    # Detect variant configuration
    detected_variant_type: str = detect_variant_type(processed_episodes)
    if variant_type is None:
        variant_type = detected_variant_type
    elif variant_type != detected_variant_type:
        raise ValueError(
            f"Expected variant_type={variant_type}, but rendered data has variant_type={detected_variant_type}"
        )
    variant_names: list[str] = extract_variant_names(processed_episodes)

    start_time: float = time.time()

    try:
        _write_processed_episodes_file(
            tmp_path, final_path, processed_episodes, action_trajs, variant_names, variant_type, compression
        )
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug(f"Failed to remove temporary HDF5 file {tmp_path}", exc_info=True)
        raise

    elapsed: float = time.time() - start_time
    size_mb: float = final_path.stat().st_size / (1024**2)
    logger.info(f"[green]✓[/green] Saved data: {final_path} ({size_mb:.1f} MB) in {elapsed:.2f}s")


def process_episodes_batch(
    env_name: str,
    use_variations: bool,
    effects_config: EffectConfigMap | None,
    state_trajs: list[list[ABCState]],
    action_trajs: list[list[int]],
    *,
    first_state_only: bool = False,
    variation_sharing_group_size: int = 1,
    batch_size_per_worker: int = 128,
    num_workers: int | None = None,
) -> tuple[list[EpisodeResult], list[list[int]]]:
    """Convert state trajectories to images (and variations) using state-based batching.

    Args:
        env_name: Environment identifier.
        use_variations: Whether to apply variations.
        effects_config: Effects configuration dictionary for individual effects.
        state_trajs: List of state sequences per episode.
        action_trajs: Corresponding action sequences.
        first_state_only: If True, only vary first state each episode.
        variation_sharing_group_size: Number of states that share the same variation
            parameters. Default 1 means unique parameters per state.
        batch_size_per_worker: Number of states per batch (unified batching parameter).
        num_workers: Number of worker processes to use. If None, defaults to max(1, ray.cpu_count() - 1).

    Returns:
        Tuple of:
          - List of per-episode image arrays or (images, variations).
          - Original action_trajs unchanged.
    """
    ensure_ray_initialized()

    if num_workers is None:
        cpus: float = ray.available_resources().get("CPU", 1)
        num_workers = max(1, int(cpus) - 1)

    manager = BatchProcessingManager(
        env_name,
        use_variations,
        effects_config=effects_config,
        first_state_only=first_state_only,
        variation_sharing_group_size=variation_sharing_group_size,
        num_workers=num_workers,
        batch_size_per_worker=batch_size_per_worker,
    )
    return manager.process_episodes(state_trajs, action_trajs)


def generate_data(env: ABCEnvironment[ABCState], cfg: GenDataSPARConfig) -> None:
    """Run the full data-generation pipeline and save to file.

    Args:
        env: Environment instance.
        cfg: GenDataSPARConfig containing data generation configuration.
    """
    ensure_ray_initialized()

    data_cfg: DataConfig = cfg.data
    save_dir: str = data_cfg.save_dir
    pathlib.Path(save_dir).mkdir(exist_ok=True, parents=True)

    num_workers: int = (
        data_cfg.num_cpus
        if getattr(data_cfg, "num_cpus", 0) > 0
        else max(1, ray.available_resources().get("CPU", 1) - 1)
    )
    global_use_vars: bool = getattr(data_cfg, "use_variations", False)
    global_effects_config: EffectConfigMap | None = getattr(data_cfg, "effects", None)
    global_first_state: bool = getattr(data_cfg, "variations_first_state_only", False)
    global_batch_size: int = getattr(data_cfg, "batch_size_per_worker", 0)

    # Auto-calculate global batch size from total states/workers with bounded batch granularity.
    if global_batch_size == 0 and data_cfg.datasets:
        total_states_all: int = 0
        total_render_work_all = 0.0
        for ds in data_cfg.datasets:
            ds_num_eps: int = getattr(ds, "num_eps", 0)
            ds_num_steps: int = getattr(ds, "num_steps", 1)
            ds_total_states = max(1, ds_num_eps * (ds_num_steps + 1))
            ds_use_variations_override: bool | None = getattr(ds, "use_variations", None)
            ds_use_vars: bool = global_use_vars if ds_use_variations_override is None else ds_use_variations_override
            ds_effects_cfg: EffectConfigMap | None = (
                global_effects_config if getattr(ds, "effects", None) is None else ds.effects
            )
            ds_first_only_override: bool | None = getattr(ds, "variations_first_state_only", None)
            ds_first_only: bool = global_first_state if ds_first_only_override is None else ds_first_only_override
            ds_work_multiplier = _estimate_render_work_multiplier(
                num_episodes=ds_num_eps,
                total_states=ds_total_states,
                use_variations=ds_use_vars,
                effects_config=ds_effects_cfg,
                first_state_only=ds_first_only,
            )
            total_states_all += ds_total_states
            total_render_work_all += ds_total_states * ds_work_multiplier

        global_work_multiplier = total_render_work_all / max(1, total_states_all)
        global_batch_size = calculate_worker_batch_size_from_state_count(
            total_states=total_states_all, num_workers=num_workers, work_multiplier=global_work_multiplier
        )
        logger.info(
            f"Auto-calculated global batch_size_per_worker: {global_batch_size} "
            f"(render_work_multiplier={global_work_multiplier:.2f})"
        )
    elif global_batch_size == 0:
        # Fallback if not specified
        global_batch_size = 1
        logger.info(f"Using fallback batch_size_per_worker: {global_batch_size}")

    compression: CompressionType = normalize_compression_value(getattr(data_cfg, "compression_type", "none"))

    def _gen(
        num_eps: int,
        num_steps: int,
        seed: int,
        levels: int,
        file_name: str,
        *,
        use_variations: bool | None = None,
        effects_config: EffectConfigMap | None = None,
        first_state_only: bool | None = None,
        dataset_num_cpus: int | None = None,
        dataset_compression: bool | CompressionType | None = None,
        dataset_batch_size: int | None = None,
        dataset_save_dir: str | None = None,
        dataset_variation_sharing_group_size: int | None = None,
    ) -> None:
        """Generate dataset and save to file."""
        if num_eps <= 0:
            return

        use_vars: bool = global_use_vars if use_variations is None else use_variations
        effects_cfg: EffectConfigMap | None = global_effects_config if effects_config is None else effects_config
        first_only: bool = global_first_state if first_state_only is None else first_state_only
        dataset_total_states: int = max(1, num_eps * (num_steps + 1))
        dataset_work_multiplier = _estimate_render_work_multiplier(
            num_episodes=num_eps,
            total_states=dataset_total_states,
            use_variations=use_vars,
            effects_config=effects_cfg,
            first_state_only=first_only,
        )
        dataset_effect_count = _count_enabled_effects(effects_cfg) if use_vars else 0

        # Determine batch_size_per_worker with override precedence
        dataset_workers: int = dataset_num_cpus if dataset_num_cpus is not None else num_workers
        # Resolve batch_size_per_worker with proper precedence
        if dataset_batch_size is not None and dataset_batch_size != 0:
            # Use dataset-specific positive override
            dataset_batch_size_val: int = dataset_batch_size
        elif dataset_batch_size == 0:
            # Auto-calc per dataset
            dataset_batch_size_val = calculate_worker_batch_size(
                num_episodes=num_eps,
                num_steps_per_episode=num_steps,
                num_workers=dataset_workers,
                use_variations=use_vars,
                effects_config=effects_cfg,
                first_state_only=first_only,
            )
            logger.info(
                f"Dataset auto-calculated batch_size_per_worker: {dataset_batch_size_val} "
                f"(render_work_multiplier={dataset_work_multiplier:.2f})"
            )
        elif global_batch_size > 0:
            # Use global positive override
            dataset_batch_size_val = global_batch_size
        else:
            # Global zero and no dataset override
            dataset_batch_size_val = calculate_worker_batch_size(
                num_episodes=num_eps,
                num_steps_per_episode=num_steps,
                num_workers=dataset_workers,
                use_variations=use_vars,
                effects_config=effects_cfg,
                first_state_only=first_only,
            )
            logger.info(
                f"Global auto-calculated batch_size_per_worker: {dataset_batch_size_val} "
                f"(render_work_multiplier={dataset_work_multiplier:.2f})"
            )

        dataset_compression_val: CompressionType = normalize_compression_value(
            dataset_compression if dataset_compression is not None else compression
        )
        dataset_save_dir_val: str = dataset_save_dir if dataset_save_dir is not None else save_dir
        dataset_variation_sharing_group_size_val: int = (
            dataset_variation_sharing_group_size
            if dataset_variation_sharing_group_size is not None
            else getattr(data_cfg, "variation_sharing_group_size", 1)
        )

        # Log parameter override information
        override_info: list[str] = []
        if dataset_num_cpus is not None:
            override_info.append(f"num_cpus: {dataset_num_cpus} (override from {num_workers})")
        else:
            override_info.append(f"num_cpus: {num_workers} (global)")

        if dataset_batch_size is not None:
            batch_size_info: str = f"batch_size_per_worker: {dataset_batch_size_val}"
            batch_size_info += (
                f" (auto-calculated from {dataset_batch_size})"
                if dataset_batch_size == 0
                else f" (override from {global_batch_size})"
            )
            override_info.append(batch_size_info)
        else:
            batch_size_info = f"batch_size_per_worker: {dataset_batch_size_val}"
            batch_size_info += (
                " (auto-calculated global)"
                if global_batch_size != getattr(data_cfg, "batch_size_per_worker", 128)
                else " (global)"
            )
            override_info.append(batch_size_info)

        if dataset_compression is not None:
            override_info.append(f"compression: {dataset_compression_val} (override from {compression})")
        else:
            override_info.append(f"compression: {dataset_compression_val} (global)")

        if dataset_variation_sharing_group_size is not None:
            override_info.append(
                f"variation_sharing_group_size: {dataset_variation_sharing_group_size} "
                f"(override from {getattr(data_cfg, 'variation_sharing_group_size', 1)})"
            )
        else:
            override_info.append(
                f"variation_sharing_group_size: {getattr(data_cfg, 'variation_sharing_group_size', 1)} (global)"
            )

        if dataset_save_dir is not None:
            override_info.append(f"save_dir: {dataset_save_dir} (override)")
        else:
            override_info.append(f"save_dir: {save_dir} (global)")
        override_info.append(
            f"render_work_multiplier: {dataset_work_multiplier:.2f} ({dataset_effect_count} variation pipelines)"
        )

        logger.info("[bold cyan]Parameter Configuration:[/bold cyan]")
        for info in override_info:
            logger.info(f"  • {info}")

        logger.info(
            f"{'- ' * 5}\nEffects: {list(effects_cfg.keys()) if effects_cfg else 'None'}, "
            f"first_state_only={first_only}\n"
            f"Workers: {dataset_workers}, Batch size: {dataset_batch_size_val}, "
            f"Compression: {dataset_compression_val}\n{'- ' * 5}"
            if use_vars
            else f"{'- ' * 5}\nVariations: disabled\n"
            f"Workers: {dataset_workers}, Batch size: {dataset_batch_size_val}, "
            f"Compression: {dataset_compression_val}\n{'- ' * 5}"
        )

        # Generate episodes
        state_trajs: list[list[ABCState]]
        action_trajs: list[list[int]]
        state_trajs, action_trajs = generate_episodes(env, num_eps, num_steps, seed, levels)

        logger.info(f"Processing {len(state_trajs)} episodes into images...")
        t0: float = time.time()
        # Process states in groups sharing the same variation parameters
        imgs_list: list[EpisodeResult]
        acts: list[list[int]]
        imgs_list, acts = process_episodes_batch(
            env.get_env_name(),
            use_vars,
            effects_cfg,
            state_trajs,
            action_trajs,
            first_state_only=first_only,
            variation_sharing_group_size=dataset_variation_sharing_group_size_val,
            batch_size_per_worker=dataset_batch_size_val,
            num_workers=dataset_workers,
        )
        dt: float = time.time() - t0
        logger.info(f"[green]✓[/green] Processing complete in {dt:.2f}s ({len(state_trajs) / dt:.2f} eps/s)")

        # Save to file in dataset-specific directory
        pathlib.Path(dataset_save_dir_val).mkdir(exist_ok=True, parents=True)
        out_path: str = str(pathlib.Path(dataset_save_dir_val) / file_name)

        expected_variant_type = "none"
        if use_vars and dataset_effect_count > 0:
            expected_variant_type = "first_state_only" if first_only else "full"
        save_data(out_path, imgs_list, acts, compression=dataset_compression_val, variant_type=expected_variant_type)

    # Display configuration
    msg_str: str = f"Save directory: {pathlib.Path(save_dir).resolve()}\n"
    if global_first_state and global_use_vars:
        msg_str += "[dim]Note: variations will only be applied to the first state of each episode[/dim]\n"

    if not data_cfg.datasets:
        logger.info(
            Panel(
                msg_str,
                border_style="dim",
                padding=(1, 2),
                title="[bold dim]SPAR Data Generation Pipeline[/bold dim]",
                width=120,
            )
        )
        logger.info("[yellow]No datasets defined in configuration. Add datasets to data.datasets list.[/yellow]")
        return
    msg_str += f"Processing {len(data_cfg.datasets)} datasets."
    logger.info("\n")
    logger.info(
        Panel(
            msg_str,
            border_style="bright_blue",
            padding=(1, 2),
            title="[bright_blue]SPAR Data Generation Pipeline[bright_blue]",
            width=120,
        )
    )

    # Create summary table
    summary_table = Table(show_header=True, box=None, padding=(0, 1))
    summary_table.add_column("Dataset", style="bold cyan")
    summary_table.add_column("Episodes", style="bright_white", justify="right")
    summary_table.add_column("Steps/Episode", style="bright_white", justify="right")
    summary_table.add_column("Total Steps", style="dim", justify="right")

    total_episodes = 0
    total_steps = 0
    for ds in data_cfg.datasets:
        name: str = getattr(ds, "name", "unnamed")
        num_eps: int = getattr(ds, "num_eps", 0)
        num_steps: int = getattr(ds, "num_steps", 1)
        episode_total_steps: int = num_eps * num_steps
        total_episodes += num_eps
        total_steps += episode_total_steps

        summary_table.add_row(name, f"{num_eps:,}", f"{num_steps:,}", f"{episode_total_steps:,}")

    summary_table.add_row("", "", "", "", end_section=True)
    summary_table.add_row(
        "[bold]TOTAL[/bold]", f"[bold]{total_episodes:,}[/bold]", "[dim]-[/dim]", f"[bold]{total_steps:,}[/bold]"
    )

    logger.info(
        Panel(summary_table, title="[bold]Dataset Summary[/bold]", border_style="blue", padding=(1, 1), width=120)
    )

    # Process datasets
    for i, ds in enumerate(data_cfg.datasets, 1):
        if i > 1:
            logger.info("")
            logger.info(Rule())
            logger.info("")

        dataset_name: str = getattr(ds, "name", f"dataset_{i}")
        logger.info(f"[bold blue]Dataset {i}/{len(data_cfg.datasets)}: {dataset_name.upper()}[/bold blue]")
        _gen(
            num_eps=getattr(ds, "num_eps", 0),
            num_steps=getattr(ds, "num_steps", 1),  # Default to 1 if not specified
            seed=getattr(ds, "start_seed", 0),
            levels=getattr(ds, "num_seeds", 0),
            file_name=getattr(ds, "file_name", f"{dataset_name}_default"),
            use_variations=getattr(ds, "use_variations", None),
            effects_config=getattr(ds, "effects", None),
            first_state_only=getattr(ds, "variations_first_state_only", None),
            dataset_num_cpus=getattr(ds, "num_cpus", None),
            dataset_compression=getattr(ds, "compression", None),
            dataset_batch_size=getattr(ds, "batch_size_per_worker", None),
            dataset_save_dir=getattr(ds, "save_dir", None),
            dataset_variation_sharing_group_size=getattr(ds, "variation_sharing_group_size", None),
        )

    logger.info("\n[bold green]✓ Data generation complete.[/bold green]")
    logger.info(f"Files saved to: {pathlib.Path(save_dir).resolve()}")
