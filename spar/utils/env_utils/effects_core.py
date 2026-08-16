"""Register, configure, and apply staged effects to SPAR environment renders."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
import contextlib
import copy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, auto
import importlib
import inspect
from logging import getLogger
import operator
import pkgutil
import time
import tracemalloc
from typing import TYPE_CHECKING, Generic, ParamSpec, Protocol, TypeVar

from matplotlib.figure import Figure
import numpy as np

from spar.utils.config_utils.samplers import _DEFAULT_RNG, Sampler, sampler_from_spec

if TYPE_CHECKING:
    from collections.abc import Callable, MutableSequence
    from inspect import Signature
    from logging import Logger
    from types import ModuleType
    from typing import TypeAlias, TypeGuard

    from numpy.random import Generator
    from numpy.typing import NDArray

    from spar.utils.config_utils.samplers import SamplerScalar


logger: Logger = getLogger(__name__)

T = TypeVar("T")
TargetT = TypeVar("TargetT")
DecoratorParams = ParamSpec("DecoratorParams")
DecoratorReturn = TypeVar("DecoratorReturn")

ConfigAtom: TypeAlias = "Hashable | NDArray[np.generic]"
NestedEffectValue: TypeAlias = "Sequence[EffectValue] | set[EffectValue] | Mapping[str, EffectValue]"
EffectValue: TypeAlias = "ConfigAtom | NestedEffectValue | Mapping[str, EffectConfigValue]"
EffectConfigValue: TypeAlias = "EffectValue | Sampler[Hashable] | Mapping[str, EffectConfigValue]"
EffectParams: TypeAlias = "dict[str, EffectValue]"
EffectConfigMap: TypeAlias = "Mapping[str, EffectConfigValue]"

ImageArray: TypeAlias = "NDArray[np.float32]"
FigureType: TypeAlias = Figure
ParameterType: TypeAlias = "type | tuple[type, ...]"
FreezeInput: TypeAlias = "EffectValue | Sampler[Hashable]"
EffectSpec: TypeAlias = "str | tuple[str, EffectParams]"

# Parameter parsing constants
_ENABLE_KEY: str = "enabled"
RANGE_PREF, IRANGE_PREF, CHOICE_PREF = "_range_", "_irange_", "_choice_"


class EffectCategory(Enum):
    """Categories of visual effects for organizational purposes."""

    BACKGROUND = auto()
    LIGHTING = auto()
    COLOR = auto()
    GEOMETRY = auto()
    NOISE = auto()
    BLUR = auto()
    DISTORTION = auto()
    OCCLUSION = auto()
    WEATHER = auto()
    SENSOR = auto()
    MATERIAL = auto()


class EffectStage(Enum):
    """Stages in the rendering pipeline where effects can be applied."""

    PRE_RENDER = auto()  # Before rendering (background, figure setup)
    OBJECT_RENDER = auto()  # During object rendering (cube modifications)
    POST_RENDER = auto()  # After rendering (image processing)


@dataclass(frozen=True, slots=True)
class EffectMetadata:
    """Immutable metadata for effect registration and discovery."""

    name: str
    category: EffectCategory
    stage: EffectStage
    description: str
    parameters: dict[str, ParameterType] = field(default_factory=dict)
    default_values: dict[str, EffectValue] = field(default_factory=dict)
    performance_level: int = 1  # 1=fast, 2=medium, 3=slow
    requires_rng: bool = False
    is_destructive: bool = False  # Whether effect modifies input in-place


class EffectProtocol(Protocol):
    """Interface implemented by registered effect callables."""

    __effect_metadata__: EffectMetadata
    __name__: str

    def __call__(self, data: TargetT, /, **kwargs: EffectValue) -> TargetT:
        """Execute the effect with given parameters."""
        ...


@dataclass
class PerformanceMetrics:
    """Performance metrics for effect execution."""

    effect_name: str
    execution_time_ms: float
    memory_usage_mb: float
    throughput_fps: float


def _is_hashable(val: FreezeInput) -> TypeGuard[Hashable]:
    try:
        hash(val)
    except TypeError:
        return False
    else:
        return True


def _raise_type_error(message: str) -> None:
    """Raise a `TypeError` with a stable helper for linted call sites."""
    raise TypeError(message)


def _has_effect_metadata(func: Callable[..., DecoratorReturn]) -> TypeGuard[EffectProtocol]:
    """Check whether a callable has effect metadata attached.

    Args:
        func: Callable candidate.

    Returns:
        True when the callable exposes ``__effect_metadata__``.
    """
    return hasattr(func, "__effect_metadata__")


def _attach_effect_metadata(
    func: Callable[DecoratorParams, DecoratorReturn], metadata: EffectMetadata
) -> Callable[DecoratorParams, DecoratorReturn]:
    """Attach effect metadata to a callable and return the same callable."""
    metadata_attr: str = "__effect_metadata__"
    setattr(func, metadata_attr, metadata)
    return func


def freeze(obj: FreezeInput) -> Hashable:
    """Convert a supported value to a deterministic hashable representation.

    Args:
        obj: Value to convert.

    Returns:
        An immutable representation. Unknown unhashable values use ``repr``.
    """
    # Immutable built-ins
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj

    # Sequence types
    if isinstance(obj, (list, tuple)):
        # Return an immutable tuple of frozen elements (Hashable)
        return tuple(freeze(o) for o in obj)

    # Sets become sorted frozensets
    if isinstance(obj, set):
        # Sorting gives the frozen set a deterministic construction order.
        return frozenset(freeze(o) for o in sorted(obj, key=repr))

    # Mapping -> tuple of sorted key/value pairs
    if isinstance(obj, Mapping):
        # Convert mapping to a deterministic tuple of key/value pairs (Hashable)
        return tuple((k, freeze(v)) for k, v in sorted(obj.items(), key=operator.itemgetter(0)))

    # Dataclass instances (incl. all Sampler subclasses)
    if is_dataclass(obj) and not isinstance(obj, type):
        frozen_fields: tuple[tuple[str, Hashable], ...] = tuple(
            (f.name, freeze(getattr(obj, f.name))) for f in fields(obj)
        )
        # Use a tuple with the class qualname and frozen fields to represent dataclass (Hashable)
        return (obj.__class__.__qualname__, frozen_fields)

    # Return hashable values directly.
    if _is_hashable(obj):
        return obj

    # Stringify the remaining user objects deterministically.
    return repr(obj)


def _parse_effect_cfg(cfg: EffectConfigMap, rng: Generator | None = None) -> EffectParams:
    """Convert Sampler instances or OmegaConf stub-strings into concrete values.

    Args:
        cfg: Configuration mapping containing effect parameters
        rng: Random number generator for sampling, uses default if None

    Returns:
        Dictionary with parsed effect configuration parameters
    """
    out: EffectParams = {}
    local_rng: Generator = rng or _DEFAULT_RNG

    for name, val in cfg.items():
        # Skip the 'enabled' key as it's only used for pipeline building logic
        if name == _ENABLE_KEY:
            continue

        if isinstance(val, Sampler):
            out[name] = val.sample(local_rng)
        elif isinstance(val, str):
            sampler: Sampler[SamplerScalar] | None = sampler_from_spec(val)
            out[name] = sampler.sample(local_rng) if sampler is not None else val
        else:
            out[name] = val

    return out


def _is_effect_config_map(value: EffectConfigValue) -> TypeGuard[EffectConfigMap]:
    """Check whether a config value is a nested effect-config mapping.

    Args:
        value: Candidate config value.

    Returns:
        True if the value is a mapping of effect configuration values.
    """
    return isinstance(value, Mapping)


def _bind_effect(effect: EffectProtocol, params: EffectParams) -> Callable[[TargetT], TargetT]:
    """Bind effect parameters once and return a typed callable.

    Args:
        effect: Effect callable with metadata.
        params: Parameters to bind.

    Returns:
        A callable that accepts and returns the same target type.
    """
    if not params:

        def apply_without_params(data: TargetT) -> TargetT:
            return effect(data)

        return apply_without_params

    def apply_with_params(data: TargetT) -> TargetT:
        return effect(data, **params)

    return apply_with_params


class Pipeline(Generic[T]):
    """Compose effects and cache their parameter-bound callables."""

    __slots__: tuple[str, ...] = (
        "_compiled_func",
        "_is_compiled",
        "_metadata",
        "_original_config",
        "_steps",
        "_steps_hash",
    )

    def __init__(self, *, original_config: EffectConfigMap | None = None) -> None:
        """Initialize an empty pipeline.

        Args:
            original_config: Optional original configuration containing Sampler instances
                           for fresh parameter sampling support.
        """
        self._steps: list[tuple[EffectProtocol, EffectParams]] = []
        self._compiled_func: Callable[[T], T] | None = None
        self._steps_hash: int = 0
        self._metadata: list[EffectMetadata] = []
        self._is_compiled: bool = False
        self._original_config: EffectConfigMap | None = original_config

    def add(self, effect: EffectProtocol, **kwargs: EffectValue) -> Pipeline[T]:
        """Add an effect to the pipeline with specified parameters.

        Args:
            effect: The effect function to add
            **kwargs: Parameters to pass to the effect

        Returns:
            Self for fluent chaining
        """
        # Validate parameters against effect metadata
        metadata: EffectMetadata = effect.__effect_metadata__
        for param_name, param_value in kwargs.items():
            if param_name in metadata.parameters:
                expected_type: ParameterType = metadata.parameters[param_name]

                try:
                    if not isinstance(param_value, expected_type):
                        msg: str = (
                            f"Parameter '{param_name}' for effect '{metadata.name}' "
                            f"expected {expected_type}, got {type(param_value)}"
                        )
                        _raise_type_error(msg)

                except TypeError:
                    # Skip validation for types that can't be used with isinstance
                    continue

        self._steps.append((effect, kwargs))
        self._is_compiled = False  # Mark for recompilation
        return self

    def compile(self) -> Pipeline[T]:
        """Bind effect parameters and cache the composed callable.

        Returns:
            This pipeline with its composed callable cached.
        """
        effect: EffectProtocol
        params: EffectParams
        if not self._steps:
            self._compiled_func = lambda x: x
            self._metadata = []

        elif len(self._steps) == 1:
            effect, params = self._steps[0]
            self._compiled_func = _bind_effect(effect, params)
            self._metadata = [effect.__effect_metadata__]

        else:
            # Bind each effect's parameters once.
            bound_effects: list[Callable[[T], T]] = []
            metadata: list[EffectMetadata] = []
            for effect, params in self._steps:
                bound_effects.append(_bind_effect(effect, params))
                metadata.append(effect.__effect_metadata__)

            # Apply the bound callables in configured order.
            def compiled_pipeline(data: T) -> T:
                result: T = data
                for bound_effect in bound_effects:
                    result = bound_effect(result)

                return result

            self._compiled_func = compiled_pipeline
            self._metadata = metadata

        # Create deterministic hash for caching (tolerates nested lists/sets/dicts)
        frozen_steps: tuple[tuple[str, tuple[tuple[str, Hashable], ...]], ...] = tuple(
            (effect.__effect_metadata__.name, tuple((k, freeze(v)) for k, v in sorted(params.items())))
            for effect, params in self._steps
        )
        self._steps_hash = hash(frozen_steps)

        self._is_compiled = True
        return self

    def __call__(self, data: T) -> T:
        """Execute the pipeline on input data.

        Args:
            data: Input data to process

        Returns:
            Processed data
        """
        if not self._is_compiled:
            self.compile()

        if self._compiled_func is None:
            msg: str = "Pipeline compilation failed"
            raise RuntimeError(msg)

        return self._compiled_func(data)

    def __getstate__(
        self,
    ) -> tuple[
        list[tuple[EffectProtocol, EffectParams]],
        Callable[[T], T] | None,
        int,
        list[EffectMetadata],
        bool,
        EffectConfigMap | None,
    ]:
        """Support for multiprocessing pickling."""
        return (
            self._steps,
            self._compiled_func,
            self._steps_hash,
            self._metadata,
            self._is_compiled,
            self._original_config,
        )

    def __setstate__(
        self,
        state: tuple[
            list[tuple[EffectProtocol, EffectParams]],
            Callable[[T], T] | None,
            int,
            list[EffectMetadata],
            bool,
            EffectConfigMap | None,
        ],
    ) -> None:
        """Support for multiprocessing unpickling."""
        (
            self._steps,
            self._compiled_func,
            self._steps_hash,
            self._metadata,
            self._is_compiled,
            self._original_config,
        ) = state

    def clear(self) -> Pipeline[T]:
        """Clear all effects from the pipeline."""
        self._steps.clear()
        self._compiled_func = None
        self._metadata = []
        self._steps_hash = 0
        self._is_compiled = False
        self._original_config = None
        return self

    @property
    def effect_count(self) -> int:
        """Number of effects in the pipeline."""
        return len(self._steps)

    @property
    def is_compiled(self) -> bool:
        """Whether the pipeline currently has a compiled callable."""
        return self._is_compiled

    def get_metadata(self) -> list[EffectMetadata]:
        """Get metadata for all effects in the pipeline."""
        return self._metadata.copy() if self._is_compiled else [effect.__effect_metadata__ for effect, _ in self._steps]

    def apply_by_stage(self, target: TargetT, stage: EffectStage) -> TargetT:
        """Apply only effects that match the specified stage to the target object.

        Args:
            target: The object to apply effects to (figure, cube, image, etc.)
            stage: The pipeline stage to filter effects by

        Returns:
            The modified target object
        """
        if not self._steps:
            return target

        result: TargetT = target
        for effect, params in self._steps:
            if effect.__effect_metadata__.stage == stage:
                try:
                    result = effect(result, **params)
                except Exception:
                    # Skip and report full error details for incompatible effects
                    logger.exception(f"Skipping effect '{effect.__effect_metadata__.name}' in stage {stage.name}")
                    continue

        return result

    def apply_by_stage_with_fresh_params(
        self,
        target: TargetT,
        stage: EffectStage,
        *,
        effects_config: EffectConfigMap | None = None,
        rng: Generator | None = None,
    ) -> TargetT:
        """Apply effects for the specified stage with freshly sampled parameters.

        A new pipeline samples parameters independently for each call, allowing
        successive frames to use different values.

        Args:
            target: The object to apply effects to (figure, cube, image, etc.)
            stage: The pipeline stage to filter effects by
            effects_config: Configuration containing Sampler instances for fresh sampling.
                           If None, uses stored original_config or falls back to apply_by_stage.
            rng: Random number generator for sampling, uses default if None

        Returns:
            The modified target object
        """
        # Use provided config, stored config, or fall back to existing parameters
        config_to_use: EffectConfigMap | None = effects_config or self._original_config
        if not self._steps or config_to_use is None:
            return self.apply_by_stage(target, stage)

        local_rng: Generator = rng or _DEFAULT_RNG
        result: TargetT = target

        # Apply each effect with freshly sampled parameters
        for effect, _params in self._steps:
            if effect.__effect_metadata__.stage == stage:
                effect_name: str = effect.__effect_metadata__.name

                # Find the configuration for this effect in the provided config
                fresh_params: EffectParams = {}
                if effect_name in config_to_use:
                    effect_cfg: EffectConfigValue = config_to_use[effect_name]
                    if _is_effect_config_map(effect_cfg):
                        fresh_params = _parse_effect_cfg(effect_cfg, local_rng)

                try:
                    result = effect(result, **fresh_params)
                except Exception:
                    # Skip and report full error details for incompatible effects
                    logger.exception(f"Skipping effect '{effect_name}' in stage {stage.name} with fresh params")
                    continue

        return result

    def apply_individual_by_stage(
        self, target: TargetT, stage: EffectStage, *, copy_data: bool = True
    ) -> dict[str, TargetT]:
        """Apply each effect in the specified stage independently.

        Each effect receives its own (optionally deep-copied) target, preventing
        destructive effects from modifying the input of subsequent effects.

        Args:
            target: The input data to supply to every effect.
            stage: The pipeline stage whose effects will be applied.
            copy_data: If True, deep-copy target for each effect (default True).

        Returns:
            Mapping from each completed effect name to its output.
        """
        results: dict[str, TargetT] = {}
        for effect, params in self.get_effects_by_stage(stage):
            data_in: TargetT = copy.deepcopy(target) if copy_data else target
            try:
                results[effect.__effect_metadata__.name] = effect(data_in, **params)
            except Exception:
                logger.exception(f"Skipping effect '{effect.__effect_metadata__.name}' in stage {stage.name}")
        return results

    def get_effects_by_stage(self, stage: EffectStage) -> list[tuple[EffectProtocol, EffectParams]]:
        """Get all effects that match the specified stage.

        Args:
            stage: The pipeline stage to filter by

        Returns:
            List of (effect, parameters) tuples for the specified stage
        """
        return [(effect, params) for effect, params in self._steps if effect.__effect_metadata__.stage == stage]

    def get_stage_summary(self) -> dict[EffectStage, int]:
        """Get a summary of effects by stage.

        Returns:
            Mapping from each EffectStage to the count of effects in that stage.
        """
        summary: dict[EffectStage, int] = dict.fromkeys(EffectStage, 0)
        for effect, _ in self._steps:
            summary[effect.__effect_metadata__.stage] += 1
        return summary


@dataclass(frozen=True)
class StagePipelines:
    """Expose pre-render, object-render, and post-render pipelines by stage."""

    pre: Pipeline[FigureType] | None
    obj: Pipeline[FreezeInput] | None
    post: Pipeline[ImageArray] | None

    def apply_by_stage(self, target: TargetT, stage: EffectStage) -> TargetT:
        """Apply effects from the specified stage to the target object.

        Args:
            target: The object to apply effects to (figure, cube, image, etc.)
            stage: The pipeline stage to apply

        Returns:
            The modified target object
        """
        if stage == EffectStage.PRE_RENDER and self.pre is not None:
            try:
                return self.pre.apply_by_stage(target, stage)
            except Exception:
                logger.exception(f"Error in StagePipelines applying stage {stage.name}")
                return target

        if stage == EffectStage.OBJECT_RENDER and self.obj is not None:
            try:
                return self.obj.apply_by_stage(target, stage)
            except Exception:
                logger.exception(f"Error in StagePipelines applying stage {stage.name}")
                return target

        if stage == EffectStage.POST_RENDER and self.post is not None:
            try:
                return self.post.apply_by_stage(target, stage)
            except Exception:
                logger.exception(f"Error in StagePipelines applying stage {stage.name}")
                return target

        return target

    def apply_by_stage_with_fresh_params(
        self,
        target: TargetT,
        stage: EffectStage,
        *,
        effects_config: EffectConfigMap | None = None,
        rng: Generator | None = None,
    ) -> TargetT:
        """Apply effects from the specified stage with freshly sampled parameters.

        Each call samples its parameters independently.

        Args:
            target: The object to apply effects to (figure, cube, image, etc.)
            stage: The pipeline stage to apply
            effects_config: Configuration containing Sampler instances for fresh sampling.
                           If None, falls back to apply_by_stage with existing params.
            rng: Random number generator for sampling, uses default if None

        Returns:
            The modified target object
        """
        if stage == EffectStage.PRE_RENDER and self.pre is not None:
            try:
                return self.pre.apply_by_stage_with_fresh_params(target, stage, effects_config=effects_config, rng=rng)
            except Exception:
                logger.exception(f"Error in StagePipelines applying stage {stage.name} with fresh params")
                return target

        if stage == EffectStage.OBJECT_RENDER and self.obj is not None:
            try:
                return self.obj.apply_by_stage_with_fresh_params(target, stage, effects_config=effects_config, rng=rng)
            except Exception:
                logger.exception(f"Error in StagePipelines applying stage {stage.name} with fresh params")
                return target

        if stage == EffectStage.POST_RENDER and self.post is not None:
            try:
                return self.post.apply_by_stage_with_fresh_params(target, stage, effects_config=effects_config, rng=rng)
            except Exception:
                logger.exception(f"Error in StagePipelines applying stage {stage.name} with fresh params")
                return target

        return target

    def apply_individual_by_stage(
        self, target: TargetT, stage: EffectStage, *, copy_data: bool = True
    ) -> dict[str, TargetT]:
        """Apply each effect in the specified stage independently.

        Each effect receives its own (optionally deep-copied) target, preventing
        destructive effects from modifying the input of subsequent effects.

        Args:
            target: The input data to supply to every effect.
            stage: The pipeline stage whose effects will be applied.
            copy_data: If True, deep-copy target for each effect (default True).
        """
        if stage == EffectStage.PRE_RENDER and self.pre is not None:
            return self.pre.apply_individual_by_stage(target, stage, copy_data=copy_data)
        if stage == EffectStage.OBJECT_RENDER and self.obj is not None:
            return self.obj.apply_individual_by_stage(target, stage, copy_data=copy_data)
        if stage == EffectStage.POST_RENDER and self.post is not None:
            return self.post.apply_individual_by_stage(target, stage, copy_data=copy_data)
        return {}

    def apply_individual(self, target: TargetT, *, copy_data: bool = True) -> dict[EffectStage, dict[str, TargetT]]:
        """Apply each effect in all stages independently.

        Args:
            target: The input data to supply to every effect.
            copy_data: If True, deep-copy target for each effect (default True).

        Returns:
            Mapping from EffectStage to a dict mapping effect names to results.
        """
        return {
            stage: self.apply_individual_by_stage(target, stage, copy_data=copy_data)
            for stage in (EffectStage.PRE_RENDER, EffectStage.OBJECT_RENDER, EffectStage.POST_RENDER)
            if self.apply_individual_by_stage(target, stage, copy_data=copy_data)
        }

    def has_effects(self) -> bool:
        """Check if this pipeline has any effects in any stage."""
        return self.pre is not None or self.obj is not None or self.post is not None

    def get_stage_summary(self) -> dict[EffectStage, int]:
        """Get a summary of effects by stage."""
        summary: dict[EffectStage, int] = {
            EffectStage.PRE_RENDER: self.pre.effect_count if self.pre else 0,
            EffectStage.OBJECT_RENDER: self.obj.effect_count if self.obj else 0,
            EffectStage.POST_RENDER: self.post.effect_count if self.post else 0,
        }
        return summary

    # Pipeline Compatibility Interface

    def add(self, effect: EffectProtocol, **kwargs: EffectValue) -> StagePipelines:
        """Add an effect to the appropriate stage pipeline.

        Args:
            effect: The effect function to add
            **kwargs: Parameters to pass to the effect

        Returns:
            New StagePipelines instance with the effect added
        """
        stage: EffectStage = effect.__effect_metadata__.stage

        # Create new pipeline instances with the added effect
        new_pre: Pipeline[FigureType] | None = self.pre
        new_obj: Pipeline[FreezeInput] | None = self.obj
        new_post: Pipeline[ImageArray] | None = self.post

        if stage == EffectStage.PRE_RENDER:
            if new_pre is None:
                new_pre = Pipeline[FigureType]()
            new_pre = copy.deepcopy(new_pre)
            new_pre.add(effect, **kwargs)
        elif stage == EffectStage.OBJECT_RENDER:
            if new_obj is None:
                new_obj = Pipeline[FreezeInput]()
            new_obj = copy.deepcopy(new_obj)
            new_obj.add(effect, **kwargs)
        elif stage == EffectStage.POST_RENDER:
            if new_post is None:
                new_post = Pipeline[ImageArray]()
            new_post = copy.deepcopy(new_post)
            new_post.add(effect, **kwargs)

        # Since this is frozen dataclass, create new instance
        return StagePipelines(pre=new_pre, obj=new_obj, post=new_post)

    def compile(self) -> StagePipelines:
        """Compile each configured stage pipeline.

        Returns:
            New StagePipelines instance with all pipelines compiled
        """
        new_pre: Pipeline[FigureType] | None = copy.deepcopy(self.pre).compile() if self.pre else None
        new_obj: Pipeline[FreezeInput] | None = copy.deepcopy(self.obj).compile() if self.obj else None
        new_post: Pipeline[ImageArray] | None = copy.deepcopy(self.post).compile() if self.post else None

        return StagePipelines(pre=new_pre, obj=new_obj, post=new_post)

    def __call__(self, data: TargetT, stage: EffectStage | None = None) -> TargetT:
        """Execute the stage pipelines on input data.

        Args:
            data: Input data to process
            stage: Specific stage to apply (applies all stages if None)

        Returns:
            Processed data

        Note:
            If stage is specified, only that stage is applied.
            If stage is None, applies all stages in order: PRE_RENDER, OBJECT_RENDER, POST_RENDER.
        """
        if stage is not None:
            return self.apply_by_stage(data, stage)

        # Apply all stages in order
        result: TargetT = data
        result = self.apply_by_stage(result, EffectStage.PRE_RENDER)
        result = self.apply_by_stage(result, EffectStage.OBJECT_RENDER)
        return self.apply_by_stage(result, EffectStage.POST_RENDER)

    @staticmethod
    def create_empty() -> StagePipelines:
        """Create a new empty StagePipelines instance.

        Returns:
            New empty StagePipelines instance
        """
        return StagePipelines(pre=None, obj=None, post=None)

    def clear(self) -> StagePipelines:
        """Clear all effects from all stage pipelines.

        Returns:
            New empty StagePipelines instance
        """
        return self.create_empty()

    @property
    def effect_count(self) -> int:
        """Total number of effects across all stage pipelines."""
        count: int = 0
        if self.pre is not None:
            count += self.pre.effect_count
        if self.obj is not None:
            count += self.obj.effect_count
        if self.post is not None:
            count += self.post.effect_count
        return count

    def get_metadata(self) -> dict[EffectStage, list[EffectMetadata]]:
        """Get metadata for all effects organized by stage.

        Returns:
            Mapping from EffectStage to list of effect metadata for that stage
        """
        metadata: dict[EffectStage, list[EffectMetadata]] = {}

        if self.pre is not None:
            metadata[EffectStage.PRE_RENDER] = self.pre.get_metadata()
        if self.obj is not None:
            metadata[EffectStage.OBJECT_RENDER] = self.obj.get_metadata()
        if self.post is not None:
            metadata[EffectStage.POST_RENDER] = self.post.get_metadata()

        return metadata

    def get_all_metadata(self) -> list[EffectMetadata]:
        """Get metadata for all effects as a flat list (Pipeline-compatible).

        Returns:
            List of all effect metadata across all stages
        """
        all_metadata: list[EffectMetadata] = []

        if self.pre is not None:
            all_metadata.extend(self.pre.get_metadata())
        if self.obj is not None:
            all_metadata.extend(self.obj.get_metadata())
        if self.post is not None:
            all_metadata.extend(self.post.get_metadata())

        return all_metadata

    def get_effects_by_stage(self, stage: EffectStage) -> list[tuple[EffectProtocol, EffectParams]]:
        """Get all effects that match the specified stage.

        Args:
            stage: The pipeline stage to filter by

        Returns:
            List of (effect, parameters) tuples for the specified stage
        """
        pipeline: Pipeline[FigureType] | Pipeline[FreezeInput] | Pipeline[ImageArray] | None = None
        if stage == EffectStage.PRE_RENDER:
            pipeline = self.pre
        elif stage == EffectStage.OBJECT_RENDER:
            pipeline = self.obj
        elif stage == EffectStage.POST_RENDER:
            pipeline = self.post

        if pipeline is None:
            return []

        return pipeline.get_effects_by_stage(stage)

    def get_all_effects(self) -> list[tuple[EffectProtocol, EffectParams]]:
        """Get all effects across all stages as a flat list (Pipeline-compatible).

        Returns:
            List of all (effect, parameters) tuples across all stages
        """
        all_effects: list[tuple[EffectProtocol, EffectParams]] = []

        for stage in (EffectStage.PRE_RENDER, EffectStage.OBJECT_RENDER, EffectStage.POST_RENDER):
            all_effects.extend(self.get_effects_by_stage(stage))

        return all_effects

    @property
    def is_compiled(self) -> bool:
        """Check if all stage pipelines are compiled.

        Returns:
            True if all non-None stage pipelines are compiled, False otherwise
        """
        for pipeline in (self.pre, self.obj, self.post):
            if pipeline is not None and not pipeline.is_compiled:
                return False
        return True

    def __getstate__(
        self,
    ) -> tuple[Pipeline[FigureType] | None, Pipeline[FreezeInput] | None, Pipeline[ImageArray] | None]:
        """Support for multiprocessing pickling."""
        return (self.pre, self.obj, self.post)

    def __setstate__(
        self, state: tuple[Pipeline[FigureType] | None, Pipeline[FreezeInput] | None, Pipeline[ImageArray] | None]
    ) -> None:
        """Support for multiprocessing unpickling."""
        pre: Pipeline[FigureType] | None
        obj: Pipeline[FreezeInput] | None
        post: Pipeline[ImageArray] | None
        pre, obj, post = state
        # A frozen dataclass requires ``object.__setattr__`` during validation.
        object.__setattr__(self, "pre", pre)
        object.__setattr__(self, "obj", obj)
        object.__setattr__(self, "post", post)


class EffectRegistry:
    """Registry for effect discovery and management.

    Effects are indexed by name, category, and stage. The registry discovers
    ``*_effects`` modules on first access.
    """

    __slots__: tuple[str, ...] = (
        "_categories",
        "_discovery_cache",
        "_effects",
        "_effects_by_name",
        "_initialized",
        "_stages",
    )

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._effects: dict[str, EffectProtocol] = {}
        self._categories: dict[EffectCategory, set[str]] = {cat: set() for cat in EffectCategory}
        self._stages: dict[EffectStage, set[str]] = {stage: set() for stage in EffectStage}
        self._effects_by_name: dict[str, EffectProtocol] = {}
        self._discovery_cache: set[str] = set()
        self._initialized: bool = False

    def _initialize_if_needed(self) -> None:
        """Discover effect modules on the first registry lookup."""
        if not self._initialized:
            self.auto_discover_effects()
            self._initialized = True

    def register(self, effect: EffectProtocol) -> None:
        """Register an effect in the registry.

        Args:
            effect: Effect callable to register.
        """
        metadata: EffectMetadata = effect.__effect_metadata__
        name: str = metadata.name

        if name in self._effects:
            msg: str = f"Effect '{name}' is already registered"
            raise ValueError(msg)

        self._effects[name] = effect
        self._categories[metadata.category].add(name)
        self._stages[metadata.stage].add(name)
        self._effects_by_name[name] = effect

    def get(self, name: str) -> EffectProtocol:
        """Return a named effect, discovering modules on the first lookup.

        Args:
            name: Name of the effect.

        Returns:
            Registered effect callable.
        """
        # Check registered names before triggering discovery.
        if name in self._effects_by_name:
            return self._effects_by_name[name]

        # First lookup triggers registry discovery.
        self._initialize_if_needed()

        # Check again after discovery imports register their effects.
        if name in self._effects_by_name:
            return self._effects_by_name[name]

        # Retry discovery for callers that registered a new package after initialization.
        return self.load_effect_if_needed(name)

    def get_by_category(self, category: EffectCategory) -> list[EffectProtocol]:
        """Get all effects in a category.

        Args:
            category: The effect category

        Returns:
            List of effects in the category
        """
        return [self._effects_by_name[name] for name in self._categories[category]]

    def get_by_stage(self, stage: EffectStage) -> list[EffectProtocol]:
        """Get all effects for a pipeline stage.

        Args:
            stage: The pipeline stage

        Returns:
            List of effects for the stage
        """
        return [self._effects_by_name[name] for name in self._stages[stage]]

    def list_names(self) -> list[str]:
        """List all registered effect names."""
        return list(self._effects.keys())

    def get_metadata(self, name: str) -> EffectMetadata:
        """Get metadata for an effect.

        Args:
            name: Name of the effect

        Returns:
            Effect metadata
        """
        return self.get(name).__effect_metadata__

    def auto_discover_effects(self, package_name: str = "spar") -> None:
        """Import effect modules below a package once.

        Modules whose names end in ``_effects`` register their decorated
        callables when imported. Discovered package names are cached.

        Args:
            package_name: Root package to search.
        """
        # Skip packages that have already been scanned.
        if package_name in self._discovery_cache:
            return

        try:
            self._discover_effect_modules(package_name)
        except (ImportError, AttributeError):
            # Import the shared image effects directly if package discovery fails.
            with contextlib.suppress(ImportError):
                importlib.import_module(f"{package_name}.utils.env_utils.shared_image_effects")

    def _discover_effect_modules(self, package_name: str) -> None:
        """Discover and import effect modules for one package."""
        # Import the root package to get its path
        package: ModuleType = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            return

        # Walk the package without importing unrelated modules.
        package_paths: MutableSequence[str] = package.__path__

        # Discover all modules ending with '_effects'
        for _, modname, _ in pkgutil.walk_packages(
            package_paths,
            prefix=f"{package_name}.",
            onerror=lambda _err: None,  # Silently skip problematic modules
        ):
            # Only process modules that end with '_effects'
            if modname.endswith("_effects"):
                try:
                    importlib.import_module(modname)
                except (ImportError, AttributeError, ValueError):
                    # Skip modules that can't be imported (missing dependencies, etc.)
                    continue

        # Cache this package to avoid redundant discovery
        self._discovery_cache.add(package_name)

    def load_effect_if_needed(self, name: str) -> EffectProtocol:
        """Return a registered effect after one additional discovery pass.

        Args:
            name: Name of the effect to load.

        Returns:
            Registered effect callable.
        """
        # Return names registered since the initial discovery pass.
        if name in self._effects_by_name:
            return self._effects_by_name[name]

        # Scan any packages not yet present in the discovery cache.
        self.auto_discover_effects()

        if name in self._effects_by_name:
            return self._effects_by_name[name]

        raise KeyError(f"Effect '{name}' was not found after module discovery")

    ensure_effect_loaded = load_effect_if_needed

    @staticmethod
    def create_pipeline(
        *, original_config: EffectConfigMap | None = None, _type_hint: type[T] | None = None
    ) -> Pipeline[T]:
        """Create a new pipeline instance.

        Args:
            original_config: Optional original configuration containing Sampler instances
                           for fresh parameter sampling support.
        """
        del _type_hint
        return Pipeline(original_config=original_config)


_global_registry: EffectRegistry = EffectRegistry()


def register_effect(
    name: str | None = None,
    *,
    category: EffectCategory,
    stage: EffectStage,
    description: str = "",
    performance_level: int = 1,
    requires_rng: bool = False,
    is_destructive: bool = False,
) -> Callable[[Callable[DecoratorParams, DecoratorReturn]], Callable[DecoratorParams, DecoratorReturn]]:
    """Register an effect by attaching metadata and adding it to the global registry.

    The decorator reads parameter names and defaults from the function signature.

    Args:
        name: Unique name for the effect. If None, uses the function's own name.
        category: Effect category for organization.
        stage: Pipeline stage where the effect applies.
        description: Human-readable description of the effect.
        performance_level: Performance indicator (1=fast, 2=medium, 3=slow).
        requires_rng: Whether the effect uses random number generation.
        is_destructive: Whether the effect modifies input in-place.

    Returns:
        The original effect function with metadata attached.
    """

    def decorator(func: Callable[DecoratorParams, DecoratorReturn]) -> Callable[DecoratorParams, DecoratorReturn]:
        # Use function name as default effect name if none provided
        actual_name: str
        if name is not None:
            actual_name = name
        else:
            function_name: str | None = getattr(func, "__name__", None)
            actual_name = function_name if isinstance(function_name, str) else type(func).__name__

        # Extract parameter information from function signature
        sig: Signature = inspect.signature(func)
        parameters: dict[str, ParameterType] = {}
        default_values: dict[str, EffectValue] = {}

        for param_name, param in sig.parameters.items():
            if param_name in {"self", "cls"}:
                continue  # Skip self/cls parameters

            # Extract type annotation
            annotation = param.annotation
            param_type: ParameterType = (
                annotation
                if annotation is not inspect.Parameter.empty
                and (
                    isinstance(annotation, type)
                    or (
                        isinstance(annotation, tuple)
                        and annotation
                        and all(isinstance(member, type) for member in annotation)
                    )
                )
                else type(param)
            )
            parameters[param_name] = param_type

            # Extract default value
            if param.default != inspect.Parameter.empty:
                if isinstance(param.default, (Hashable, np.ndarray)):
                    default_values[param_name] = param.default
                else:
                    default_values[param_name] = repr(param.default)

        # Create metadata
        metadata: EffectMetadata = EffectMetadata(
            name=actual_name,
            category=category,
            stage=stage,
            description=description,
            parameters=parameters,
            default_values=default_values,
            performance_level=performance_level,
            requires_rng=requires_rng,
            is_destructive=is_destructive,
        )

        registered_func: Callable[DecoratorParams, DecoratorReturn] = _attach_effect_metadata(func, metadata)
        if not _has_effect_metadata(registered_func):
            msg: str = f"Effect '{actual_name}' is missing __effect_metadata__ after registration"
            raise TypeError(msg)

        # Register with global registry
        _global_registry.register(registered_func)

        # Return the original callable
        return func

    return decorator


def get_registry() -> EffectRegistry:
    """Get the global effect registry."""
    return _global_registry


def create_pipeline(
    *, original_config: EffectConfigMap | None = None, _type_hint: type[T] | None = None
) -> Pipeline[T]:
    """Create a new effect pipeline.

    Args:
        original_config: Optional original configuration containing Sampler instances
                       for fresh parameter sampling support.
    """
    return _global_registry.create_pipeline(original_config=original_config, _type_hint=_type_hint)


# PIPELINE FACTORIES


class EffectBuilder:
    """Build effect pipelines from registry entries."""

    def __init__(self) -> None:
        """Initialize the builder."""
        self.registry: EffectRegistry = get_registry()

    def create_lighting_pipeline(self) -> Pipeline[ImageArray]:
        """Create a pipeline with ambient and directional lighting.

        Returns:
            A configured lighting pipeline.
        """
        pipeline: Pipeline[ImageArray] = create_pipeline()
        pipeline.add(self.registry.get("ambient_light"), color="#ffd1a4", alpha=0.2)
        pipeline.add(self.registry.get("directional_light"), dark_factor=0.3, direction="left")
        return pipeline

    def create_noise_pipeline(self, level: str = "moderate") -> Pipeline[ImageArray]:
        """Create a pipeline with noise effects.

        Args:
            level: Noise level ("light", "moderate", "heavy")

        Returns:
            Configured noise pipeline
        """
        pipeline: Pipeline[ImageArray] = create_pipeline()
        noise_levels: dict[str, float] = {"light": 0.02, "moderate": 0.05, "heavy": 0.1}
        level_value: float = noise_levels.get(level, 0.05)

        pipeline.add(self.registry.get("gaussian_noise"), noise_level=level_value)
        pipeline.add(self.registry.get("salt_pepper_noise"), amount=level_value * 0.2)
        return pipeline

    def create_distortion_pipeline(self) -> Pipeline[ImageArray]:
        """Create a pipeline with rotation and zoom effects.

        Returns:
            A configured geometric distortion pipeline.
        """
        pipeline: Pipeline[ImageArray] = create_pipeline()
        pipeline.add(self.registry.get("rotate_image"), angle=5.0)
        pipeline.add(self.registry.get("zoom_effect"), factor=1.2)
        return pipeline


def apply_effect_sequence(image: ImageArray, effect_names: Sequence[str], **kwargs: EffectValue) -> ImageArray:
    """Apply multiple effects with default parameters.

    Args:
        image: Input image to process
        effect_names: List of effect names to apply
        **kwargs: Parameters to override for any effects

    Returns:
        Processed image

    Example:
        result = apply_effect_sequence(image, ["gaussian_noise", "motion_blur"], noise_level=0.1)
    """
    registry: EffectRegistry = get_registry()
    pipeline: Pipeline[ImageArray] = create_pipeline()

    for name in effect_names:
        effect: EffectProtocol = registry.get(name)
        effect_kwargs: EffectParams = {k: v for k, v in kwargs.items() if k in effect.__effect_metadata__.parameters}
        pipeline.add(effect, **effect_kwargs)

    compiled_pipeline: Pipeline[ImageArray] = pipeline.compile()
    return compiled_pipeline(image)


# MODULE INITIALIZATION


# Importing an effect module runs its registration decorators.


def get_effect(name: str) -> EffectProtocol:
    """Return a named effect, importing effect modules on the first lookup.

    Args:
        name: Name of the effect to retrieve.

    Returns:
        Registered effect callable.

    Example:
        # Get camera effect without importing cube3_effects module
        camera_effect = get_effect("camera_variation")
        pipeline = create_pipeline().add(camera_effect, yaw_range=(-30, 30))
    """
    return _global_registry.get(name)


def list_available_effects() -> list[str]:
    """Discover effect modules and return their registered names.

    Returns:
        Registered effect names.
    """
    # Trigger auto-discovery
    _global_registry.auto_discover_effects()
    return _global_registry.list_names()


def create_effect_pipeline(*effect_specs: EffectSpec, _type_hint: type[T] | None = None) -> Pipeline[T]:
    """Create a pipeline from effect specifications with automatic discovery.

    Args:
        effect_specs: Effect specifications - either strings (effect names) or
                     tuples of (effect_name, parameters)

    Returns:
        Pipeline with effect parameters bound before the first call.

    Example:
        # Named effects
        pipeline = create_effect_pipeline("gaussian_blur", "motion_blur")

        # Effects with parameters
        pipeline = create_effect_pipeline(
            ("camera_variation", {"yaw_range": (-45, 45)}),
            ("gaussian_blur", {"sigma": 1.5}),
            "cube_color_shift",
        )

        # Apply the configured effects
        result = pipeline(input_data)
    """
    pipeline: Pipeline[T] = create_pipeline(_type_hint=_type_hint)

    for spec in effect_specs:
        effect: EffectProtocol
        effect_name: str
        params: EffectParams
        if isinstance(spec, str):
            # Named effect
            effect = get_effect(spec)
            pipeline.add(effect)
        else:
            # Effect name with parameters (tuple[str, EffectParams])
            effect_name, params = spec
            effect = get_effect(effect_name)
            pipeline.add(effect, **params)

    return pipeline.compile()


def apply_effects_to_data(data: T, *effect_specs: EffectSpec) -> T:
    """Apply effects to data.

    Args:
        data: Input data to process
        effect_specs: Effect specifications

    Returns:
        Processed data
    """
    pipeline: Pipeline[T] = create_effect_pipeline(*effect_specs)
    return pipeline(data)


def create_batch_processor(*effect_specs: EffectSpec) -> Callable[[Sequence[FreezeInput]], list[FreezeInput]]:
    """Create a batch processor for multiprocessing.

    Args:
        effect_specs: Effect specifications

    Returns:
        Function that applies the compiled pipeline to each batch item.

    Example:
        >>> processor = create_batch_processor("gaussian_blur", ("cube_color_shift", {"r": 0.1}))
        >>> results = processor(batch_of_images)  # Batch processing
    """
    # Compile once and reuse the bound pipeline for each item.
    compiled_pipeline: Pipeline[FreezeInput] = create_effect_pipeline(*effect_specs)

    def batch_processor(data_batch: Sequence[FreezeInput]) -> list[FreezeInput]:
        """Process a batch of data."""
        return [compiled_pipeline(item) for item in data_batch]

    return batch_processor


def build_stage_pipelines(
    raw_cfg: EffectConfigMap, *, section: str | None = "effects", rng: Generator | None = None
) -> dict[str, StagePipelines]:
    """Expand configuration into stage-specific pipelines.

    Args:
        raw_cfg: Mapping with effect definitions and optional parameters.
        section: Top-level key in raw_cfg containing effects (default 'effects').
        rng: Random number generator for sampling parameters.

    Returns:
        A dict mapping pipeline names to StagePipelines(pre, obj, post).
    """
    if section is None or section not in raw_cfg:
        root: EffectConfigMap = raw_cfg
    else:
        section_cfg: EffectConfigValue = raw_cfg[section]
        if not _is_effect_config_map(section_cfg):
            return {}
        root = section_cfg

    out: dict[str, StagePipelines] = {}

    for name, cfg in root.items():
        if not _is_effect_config_map(cfg):
            continue  # Unsupported entry
        cfg_map: EffectConfigMap = cfg

        # Decide whether cfg itself is a leaf (single effect) or a combo:
        # Leaf iff every value (except 'enabled') is NOT a Mapping.
        # Otherwise, disabled leaves are ignored, disabled combos still inspected
        enabled_value: EffectConfigValue | bool = cfg_map.get(_ENABLE_KEY, True)
        is_enabled: bool = enabled_value if isinstance(enabled_value, bool) else True

        is_leaf: bool = (
            not any(_is_effect_config_map(v) for k, v in cfg_map.items() if k != _ENABLE_KEY) if is_enabled else False
        )

        specs: list[tuple[str, EffectParams]] = []

        # Case 1: single effect at the top level
        if is_leaf:
            specs.append((name, _parse_effect_cfg(cfg_map, rng)))

        # Case 2: combination
        else:
            if not is_enabled:
                continue  # whole combo disabled

            for sub_name, sub_cfg in cfg_map.items():
                if sub_name == _ENABLE_KEY or not _is_effect_config_map(sub_cfg):
                    continue

                sub_enabled_value: EffectConfigValue | bool = sub_cfg.get(_ENABLE_KEY, True)
                sub_enabled: bool = sub_enabled_value if isinstance(sub_enabled_value, bool) else True
                if sub_enabled:
                    specs.append((sub_name, _parse_effect_cfg(sub_cfg, rng)))

        if not specs:
            continue  # Nothing enabled -> Next entry

        # Partition by render stage and assemble StagePipelines
        by_stage: dict[EffectStage, list[tuple[str, EffectParams]]] = {
            EffectStage.PRE_RENDER: [],
            EffectStage.OBJECT_RENDER: [],
            EffectStage.POST_RENDER: [],
        }
        for eff_name, params in specs:
            stage: EffectStage = get_effect(eff_name).__effect_metadata__.stage
            by_stage[stage].append((eff_name, params))

        # Extract original configuration for POST_RENDER effects to enable fresh parameter sampling
        post_render_original_config: dict[str, EffectConfigValue] = {}
        if by_stage[EffectStage.POST_RENDER]:
            # Store original config for effects that will be used for POST_RENDER
            if is_leaf:
                # Single effect case
                if name in cfg_map:
                    post_render_original_config[name] = cfg_map
            else:
                # Combination case - collect configs for POST_RENDER effects
                for eff_name, _params in by_stage[EffectStage.POST_RENDER]:
                    if eff_name in cfg_map:
                        candidate_cfg: EffectConfigValue = cfg_map[eff_name]
                        if _is_effect_config_map(candidate_cfg):
                            post_render_original_config[eff_name] = candidate_cfg

        pre: Pipeline[FigureType] | None = (
            create_effect_pipeline(*by_stage[EffectStage.PRE_RENDER]) if by_stage[EffectStage.PRE_RENDER] else None
        )
        obj: Pipeline[FreezeInput] | None = (
            create_effect_pipeline(*by_stage[EffectStage.OBJECT_RENDER])
            if by_stage[EffectStage.OBJECT_RENDER]
            else None
        )
        post: Pipeline[ImageArray] | None = (
            create_effect_pipeline_with_config(by_stage[EffectStage.POST_RENDER], post_render_original_config)
            if by_stage[EffectStage.POST_RENDER]
            else None
        )

        out[name] = StagePipelines(pre, obj, post)

    return out


def create_effect_pipeline_with_config(
    effect_specs: list[tuple[str, EffectParams]], original_config: EffectConfigMap, _type_hint: type[T] | None = None
) -> Pipeline[T]:
    """Create a pipeline with original configuration for fresh parameter sampling.

    Args:
        effect_specs: List of (effect_name, parsed_params) tuples
        original_config: Original configuration containing Sampler instances

    Returns:
        Compiled pipeline with stored original configuration
    """
    pipeline: Pipeline[T] = create_pipeline(original_config=original_config, _type_hint=_type_hint)

    for effect_name, params in effect_specs:
        effect: EffectProtocol = get_effect(effect_name)
        pipeline.add(effect, **params)

    return pipeline.compile()


# PERFORMANCE UTILITIES


def benchmark_pipeline_performance(*effect_specs: EffectSpec, iterations: int = 1000) -> dict[str, float]:
    """Benchmark pipeline performance.

    Args:
        effect_specs: Effect specifications to benchmark
        iterations: Number of test iterations

    Returns:
        Performance metrics dictionary
    """
    # Create an array for the example pipeline.
    test_data: ImageArray = np.random.random((100, 100, 3)).astype(np.float32)

    # Compile pipeline
    start_compile: float = time.perf_counter()
    pipeline: Pipeline[ImageArray] = create_effect_pipeline(*effect_specs)
    compile_time: float = time.perf_counter() - start_compile

    # Warm up
    for _ in range(10):
        pipeline(test_data)

    # Benchmark execution
    # Collect execution times in a list for later conversion
    execution_times_list: list[float] = []
    for _ in range(iterations):
        start: float = time.perf_counter()
        pipeline(test_data)
        execution_times_list.append(time.perf_counter() - start)

    execution_times: NDArray[np.float64] = np.array(execution_times_list, dtype=np.float64)

    return {
        "compile_time_ms": compile_time * 1000,
        "avg_execution_time_ms": float(np.mean(execution_times) * 1000),
        "min_execution_time_ms": float(np.min(execution_times) * 1000),
        "max_execution_time_ms": float(np.max(execution_times) * 1000),
        "std_execution_time_ms": float(np.std(execution_times) * 1000),
        "throughput_ops_per_sec": float(1.0 / np.mean(execution_times)),
        "effect_count": pipeline.effect_count,
    }


class PerformanceProfiler:
    """Profiler for measuring effect performance."""

    def __init__(self) -> None:
        """Initialize the profiler."""
        self.metrics: list[PerformanceMetrics] = []

    def profile_effect(
        self, effect: EffectProtocol, test_data: FreezeInput, iterations: int = 100
    ) -> PerformanceMetrics:
        """Profile an effect's performance.

        Args:
            effect: Effect to profile
            test_data: Test data for profiling
            iterations: Number of iterations to run

        Returns:
            Performance metrics
        """
        # Warm up
        for _ in range(10):
            effect(test_data)

        # Profile
        tracemalloc.start()
        start_time: float = time.perf_counter()

        for _ in range(iterations):
            effect(test_data)

        end_time: float = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_time: float = (end_time - start_time) * 1000 / iterations  # ms per call
        memory_usage: float = peak / 1024 / 1024  # MB
        throughput: float = 1000 / execution_time  # FPS

        metrics: PerformanceMetrics = PerformanceMetrics(
            effect_name=effect.__effect_metadata__.name,
            execution_time_ms=execution_time,
            memory_usage_mb=memory_usage,
            throughput_fps=throughput,
        )

        self.metrics.append(metrics)
        return metrics

    def get_summary(self) -> str:
        """Get a performance summary."""
        if not self.metrics:
            return "No performance data available."

        summary: str = "Performance Summary:\n"
        summary += f"{'=' * 50}\n"

        def _execution_time_key(metric: PerformanceMetrics) -> float:
            return metric.execution_time_ms

        for metric in sorted(self.metrics, key=_execution_time_key):
            summary += (
                f"{metric.effect_name:20} | "
                f"{metric.execution_time_ms:6.2f}ms | "
                f"{metric.memory_usage_mb:6.2f}MB | "
                f"{metric.throughput_fps:6.1f}FPS\n"
            )
        return summary
