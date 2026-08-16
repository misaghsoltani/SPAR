"""Inspect environments, serialize states, and render dashboard previews.

Effect metadata and pipelines come from :mod:`spar.utils.env_utils.effects_core`.
The same helpers serve each environment registered with the dashboard.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping, Sequence
import copy
from dataclasses import dataclass
from functools import cache, lru_cache
import importlib
import inspect
import io
from itertools import starmap
import json
import operator
from pathlib import Path
import re
import sys
import types
from typing import TYPE_CHECKING, SupportsFloat, SupportsInt, TypedDict, get_args, get_origin, get_type_hints

from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from spar.environments.abstracts.state import ABCState
from spar.utils.env_utils.effects_core import EffectStage, Pipeline, StagePipelines, get_effect, get_registry
from spar.utils.env_utils.env_utils import get_environment_class, list_environment_names

from .rich_logger import get_rich_logger

if TYPE_CHECKING:
    from inspect import Parameter, Signature
    from types import ModuleType
    from typing import NotRequired, Required, TypeAlias

    from numpy.typing import NDArray

    from spar.environments.abstracts.environment import ABCEnvironment
    from spar.utils.env_utils.effects_core import (
        EffectMetadata,
        EffectProtocol,
        EffectRegistry,
        EffectValue,
        FigureType,
        FreezeInput,
        ImageArray,
    )

    from .rich_logger import RichLogger

DashboardPipeline: TypeAlias = "Pipeline[FigureType] | Pipeline[FreezeInput] | Pipeline[ImageArray]"


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = dict[str, "JSONValue"] | list["JSONValue"] | JSONScalar
EffectsStoreType: TypeAlias = dict[str, dict[str, "EffectEntry"]]
EffectSpecsMapping: TypeAlias = dict[str, list["EffectSpec"]]
RendererStateType: TypeAlias = dict[str, JSONValue]
StatePayloadType: TypeAlias = dict[str, JSONValue]
SelectionType: TypeAlias = dict[str, list["ActiveEffect"]]
RendererKwarg: TypeAlias = int | tuple[float, float]
StageTargetMapping: TypeAlias = dict[str, tuple[type, ...]]

_UNKNOWN_ANNOTATION = object()
_SHARED_IMAGE_EFFECT_MODULE = "spar.utils.env_utils.shared_image_effects"


rich_logger: RichLogger = get_rich_logger(__name__)

__all__: list[str] = [
    "build_stage_pipeline",
    "create_default_background_image",
    "decode_base64_image",
    "deserialize_state",
    "generate_start_state",
    "get_effect_specs_for_environment",
    "get_environment_default_renderer_settings",
    "image_to_data_uri",
    "list_environment_options",
    "normalize_renderer_settings",
    "print_startup_message",
    "render_environment_to_uri",
    "render_state_to_uri",
    "serialize_state",
]


class ParameterSpec(TypedDict, total=False):
    """Specification for a single effect parameter."""

    name: Required[str]
    label: Required[str]
    annotation: NotRequired[str]
    kind: NotRequired[str]
    placeholder: NotRequired[str]
    default: NotRequired[JSONValue]
    step: NotRequired[float]
    options: NotRequired[list[dict[str, JSONValue]]]


class EffectSpec(TypedDict, total=False):
    """Specification for a single effect."""

    name: Required[str]
    stage: NotRequired[str]
    category: Required[str]
    description: NotRequired[str]
    performance: NotRequired[str | int]
    requires_rng: NotRequired[bool]
    target_param: NotRequired[str | None]
    target_type: NotRequired[str]
    parameters: NotRequired[list[ParameterSpec]]


class EffectEntry(TypedDict):
    """State entry for a single effect."""

    enabled: bool
    params: dict[str, JSONValue]


class HistoryEntryType(TypedDict):
    """A single entry in the render history."""

    timestamp: str
    env: str
    state: StatePayloadType
    effects: EffectsStoreType
    renderer: RendererStateType
    image: str


class RestoreBufferType(TypedDict):
    """Buffer to hold a history entry for restoration on environment change."""

    env: str
    state: StatePayloadType
    effects: EffectsStoreType
    renderer: RendererStateType


class ActiveEffect(TypedDict):
    """An active effect ready to be applied during rendering."""

    name: str
    enabled: bool
    params: dict[str, JSONValue]


@dataclass(frozen=True)
class EnvironmentSpec:
    """Descriptor capturing environment-specific behaviour for the dashboard."""

    key: str
    label: str
    state_cls: type[ABCState]
    state_fields: tuple[StateFieldSpec, ...]
    object_stage_types: tuple[type, ...]
    stage_target_types: StageTargetMapping
    domain_module_prefix: str
    renderer_defaults: dict[str, JSONValue]


@dataclass(frozen=True)
class StateFieldSpec:
    """Constructor field descriptor used for generic state (de)serialization."""

    name: str
    annotation: object
    required: bool
    default: object = None


def _humanize_environment_label(env_name: str, class_name: str | None = None) -> str:
    """Build a readable environment label from canonical names."""
    source: str = class_name.removesuffix("Env") if class_name else env_name
    parts: list[str] = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", source)
    if not parts:
        return env_name.replace("_", " ").strip().title()

    return " ".join(part.upper() if part.isupper() else part.capitalize() for part in parts)


def _environment_effect_module_prefix(env_name: str) -> str:
    return f"spar.environments.{env_name}."


def _append_unique_type(targets: list[type], candidate: type | None) -> None:
    if candidate is None:
        return
    if candidate not in targets:
        targets.append(candidate)


def _known_annotation_type(name: str) -> type | None:
    stripped: str = name.strip().strip("\"'")
    if stripped in {"ArrayLike", "ImageArray", "NDArray", "ndarray"}:
        return np.ndarray
    if stripped in {"Figure", "FigureType"}:
        return Figure

    return None


def _lookup_type_in_module(module_name: str, type_name: str) -> object | None:
    """Resolve a named annotation from a module.

    Classes are returned directly. Parameterized typing constructs such as
    ``Literal`` aliases are also returned so the UI can surface their options.

    Args:
        module_name: Dotted module path to search.
        type_name: Attribute name of the annotation.

    Returns:
        The resolved annotation object, or None when unavailable.
    """
    try:
        module: ModuleType = sys.modules.get(module_name) or importlib.import_module(module_name)
    except (AttributeError, ImportError, TypeError, ValueError):
        return None

    value: str | type | None = getattr(module, type_name, None)
    if isinstance(value, type) or get_origin(value) is not None:
        return value
    return None


def _resolve_named_type(type_name: str, effect_module_name: str, env_name: str | None = None) -> object | None:
    known = _known_annotation_type(type_name)
    if known is not None:
        return known

    module_candidates: list[str] = [effect_module_name]
    package_name: str = effect_module_name.rsplit(".", 1)[0] if "." in effect_module_name else ""
    if package_name:
        module_candidates.append(package_name)
        if env_name:
            module_candidates.append(f"{package_name}.{env_name}")

    module_candidates.extend([
        "spar.environments",
        "spar.utils.env_utils",
        "spar.utils.env_utils.viz_utils",
        "spar.utils.env_utils.puzzlegen",
        "spar.utils.env_utils.puzzlegen.ice_slider",
        "matplotlib.figure",
    ])

    for module_name in module_candidates:
        resolved = _lookup_type_in_module(module_name, type_name)
        if resolved is not None:
            return resolved

    return None


def _resolve_annotation_for_effect(annotation: object, effect_module_name: str, env_name: str | None = None) -> object:
    if annotation is inspect.Parameter.empty:
        return _UNKNOWN_ANNOTATION
    if not isinstance(annotation, str):
        return annotation

    stripped: str = annotation.strip().strip("\"'")
    resolved = _resolve_named_type(stripped, effect_module_name, env_name)
    if resolved is not None:
        return resolved

    if any(token in stripped for token in ("ImageArray", "NDArray", "ArrayLike")):
        return np.ndarray
    if any(token in stripped for token in ("FigureType", "Figure")):
        return _known_annotation_type("FigureType") or annotation

    return annotation


def _effect_type_hints(effect: EffectProtocol, signature: Signature) -> dict[str, object]:
    try:
        hints: dict[str, object] = get_type_hints(effect, include_extras=True)
    except (AttributeError, NameError, TypeError):
        hints = {}

    effect_module_name: str = getattr(effect, "__module__", "")
    for name, param in signature.parameters.items():
        if name in hints:
            continue
        hints[name] = _resolve_annotation_for_effect(param.annotation, effect_module_name)

    return hints


def _infer_stage_target_types(env_name: str, state_cls: type[ABCState]) -> StageTargetMapping:
    """Infer target object types the environment can accept at each effect stage."""
    _effect_specs_by_stage()

    by_stage: dict[str, list[type]] = {stage.name: [] for stage in EffectStage}
    _append_unique_type(by_stage[EffectStage.OBJECT_RENDER.name], state_cls)
    _append_unique_type(by_stage[EffectStage.POST_RENDER.name], np.ndarray)

    domain_prefix: str = _environment_effect_module_prefix(env_name)
    for effect_name, module_name in _EFFECT_MODULES.items():
        if not module_name.startswith(domain_prefix):
            continue
        stage_name: str | None = _EFFECT_STAGES.get(effect_name)
        if stage_name is None:
            continue
        _append_unique_type(by_stage[stage_name], _EFFECT_TARGET_TYPES.get(effect_name))

    return {stage: tuple(targets) for stage, targets in by_stage.items()}


def _renderer_defaults_for_environment(env_name: str) -> dict[str, JSONValue]:
    defaults: dict[str, JSONValue] = {"dpi": 150}
    if env_name.startswith("cube"):
        defaults["size"] = 2.0

    return defaults


def _state_field_specs(state_cls: type[ABCState]) -> tuple[StateFieldSpec, ...]:
    """Infer constructor fields for a state class."""
    signature: inspect.Signature = inspect.signature(state_cls.__init__)
    try:
        hints: dict[str, object] = get_type_hints(state_cls.__init__, include_extras=True)
    except (AttributeError, NameError, TypeError):
        hints = {}

    fields: list[StateFieldSpec] = []
    name: str
    param: Parameter
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        annotation = hints.get(
            name, _UNKNOWN_ANNOTATION if param.annotation is inspect.Parameter.empty else param.annotation
        )
        required: bool = param.default is inspect.Parameter.empty
        default = None if required else param.default
        fields.append(StateFieldSpec(name=name, annotation=annotation, required=required, default=default))

    return tuple(fields)


def _generate_state_from_env(env: ABCEnvironment[ABCState]) -> ABCState:
    try:
        return env.generate_start_states(1)[0]

    except TypeError:
        return env.generate_start_states(1, level_seeds=None)[0]


def _int_from_value(value: object) -> int:
    if isinstance(value, (str, bytes, bytearray)):
        return int(value)
    if isinstance(value, SupportsInt):
        return int(value)

    raise TypeError(f"Cannot coerce {type(value).__name__} to int")


def _float_from_value(value: object) -> float:
    if isinstance(value, (str, bytes, bytearray)):
        return float(value)
    if isinstance(value, SupportsFloat):
        return float(value)
    if isinstance(value, SupportsInt):
        return float(int(value))

    raise TypeError(f"Cannot coerce {type(value).__name__} to float")


# ==================================================================================================
# Environment helpers
# ==================================================================================================


def list_environment_options() -> list[dict[str, str]]:
    """Return available environment options for UI selection."""
    options: list[dict[str, str]] = []
    for env_name in list_environment_names():
        label: str = _humanize_environment_label(env_name)
        try:
            env_class: type[ABCEnvironment[ABCState]] = get_environment_class(env_name)
            label = _humanize_environment_label(env_name, env_class.__name__)
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            rich_logger.warning(f"Could not load environment class for '{env_name}': {exc}")
        options.append({"label": label, "value": env_name})

    return options


@cache
def _get_environment_spec(env_name: str) -> EnvironmentSpec:
    key: str = env_name.strip().lower()
    if not key:
        raise ValueError("Environment name must be non-empty")

    env_class: type[ABCEnvironment[ABCState]] = get_environment_class(key)
    env: ABCEnvironment[ABCState] = _get_environment_instance(key)
    state: ABCState = _generate_state_from_env(env)
    state_cls: type[ABCState] = type(state)
    state_fields: tuple[StateFieldSpec, ...] = _state_field_specs(state_cls)
    if not state_fields:
        raise ValueError(f"Unable to infer state schema for environment '{key}'")

    stage_target_types: dict[str, tuple[type, ...]] = _infer_stage_target_types(key, state_cls)

    return EnvironmentSpec(
        key=key,
        label=_humanize_environment_label(key, env_class.__name__),
        state_cls=state_cls,
        state_fields=state_fields,
        object_stage_types=stage_target_types.get(EffectStage.OBJECT_RENDER.name, ()),
        stage_target_types=stage_target_types,
        domain_module_prefix=_environment_effect_module_prefix(key),
        renderer_defaults=_renderer_defaults_for_environment(key),
    )


@cache
def _get_environment_instance(env_name: str) -> ABCEnvironment[ABCState]:
    key: str = env_name.strip().lower()
    if not key:
        raise ValueError("Environment name must be non-empty")
    env_class: type[ABCEnvironment[ABCState]] = get_environment_class(key)
    rich_logger.debug(f"Creating environment instance for [cyan]{env_name}[/cyan]")

    return env_class()


def _unwrap_annotated(annotation: object) -> object:
    origin: type | None = get_origin(annotation)
    if origin is not None and getattr(origin, "__qualname__", "") == "Annotated":
        args = get_args(annotation)
        if args:
            return args[0]

    return annotation


def _split_union(annotation: object) -> tuple[bool, tuple[object, ...]]:
    ann = _unwrap_annotated(annotation)
    if isinstance(ann, str):
        text: str = ann.lower().replace(" ", "")
        has_none: bool = "|none" in text or "optional[" in text or "nonetype" in text
        if not has_none:
            return False, (ann,)

        if "ndarray" in text or "imagearray" in text:
            return True, (np.ndarray,)

        if "tuple" in text:
            return True, (tuple,)

        if "list" in text or "sequence" in text or "iterable" in text:
            return True, (list,)

        if "mapping" in text or "dict" in text:
            return True, (dict,)

        if "bool" in text:
            return True, (bool,)

        if "float" in text:
            return True, (float,)

        if "int" in text:
            return True, (int,)

        if "str" in text:
            return True, (str,)

        return True, (_UNKNOWN_ANNOTATION,)

    origin = get_origin(ann)
    if origin is not types.UnionType and str(origin) != "typing.Union":
        return False, (ann,)

    members: tuple[object, ...] = tuple(get_args(ann))
    concrete: tuple[object, ...] = tuple(
        member for member in members if member is not type(None) and member is not None
    )
    if not concrete:
        concrete = (_UNKNOWN_ANNOTATION,)

    return len(concrete) != len(members), concrete


def _is_literal(annotation: object) -> bool:
    origin = get_origin(_unwrap_annotated(annotation))

    return origin is not None and getattr(origin, "__qualname__", "") == "Literal"


def _json_parse_if_needed(value: object) -> object:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        return value


def _extract_numpy_dtype(annotation: object) -> type[np.generic] | None:
    if isinstance(annotation, type) and issubclass(annotation, np.generic):
        return annotation

    origin = get_origin(annotation)
    if origin is np.dtype:
        args = get_args(annotation)
        if args:
            inner = args[0]
            if isinstance(inner, type) and issubclass(inner, np.generic):
                return inner

    return None


def _numpy_dtype_from_annotation(annotation: object) -> type[np.generic] | None:
    ann = _unwrap_annotated(annotation)
    origin = get_origin(ann)
    if ann is np.ndarray or origin is np.ndarray:
        for arg in get_args(ann):
            dtype = _extract_numpy_dtype(arg)
            if dtype is not None:
                return dtype

    text = str(ann)
    if "bool" in text:
        return np.bool_

    if "uint8" in text:
        return np.uint8

    if "float" in text:
        return np.float32

    if "int" in text and "uint" not in text:
        return np.intp

    return None


def _is_integer_annotation(annotation: object) -> bool:
    ann = _unwrap_annotated(annotation)
    if isinstance(ann, str):
        text: str = ann.lower()
        if any(token in text for token in ("ndarray", "list", "tuple", "sequence", "iterable", "mapping", "dict")):
            return False

        return "int" in text and "bool" not in text and "float" not in text

    return ann is int or (isinstance(ann, type) and issubclass(ann, np.integer))


def _is_float_annotation(annotation: object) -> bool:
    ann = _unwrap_annotated(annotation)
    if isinstance(ann, str):
        text: str = ann.lower()
        if any(token in text for token in ("ndarray", "list", "tuple", "sequence", "iterable", "mapping", "dict")):
            return False

        return "float" in text

    return ann is float or (isinstance(ann, type) and issubclass(ann, np.floating))


def _is_bool_annotation(annotation: object) -> bool:
    ann = _unwrap_annotated(annotation)
    if isinstance(ann, str):
        text: str = ann.lower()
        if any(token in text for token in ("ndarray", "list", "tuple", "sequence", "iterable", "mapping", "dict")):
            return False

        return "bool" in text

    return ann is bool


def _literal_options(annotation: object) -> list[dict[str, JSONValue]] | None:
    ann = _unwrap_annotated(annotation)
    origin = get_origin(ann)
    if origin is None or getattr(origin, "__qualname__", "") != "Literal":
        return None

    args = get_args(ann)

    return [{"label": str(opt), "value": _to_json_serializable(opt)} for opt in args]


def _try_coerce_state_union_member(value: object, member: object) -> tuple[bool, object, Exception | None]:
    """Try coercing a union branch for state payloads."""
    try:
        return True, _coerce_state_value(value, member), None
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        return False, None, exc


def _coerce_state_value(value: object, annotation: object) -> object:
    ann = _unwrap_annotated(annotation)
    is_optional, union_members = _split_union(ann)
    if value is None and is_optional:
        return None

    if len(union_members) > 1:
        last_error: Exception | None = None
        member: object
        for member in union_members:
            success, resolved, err = _try_coerce_state_union_member(value, member)
            if success:
                return resolved
            last_error = err
        if last_error is not None:
            raise last_error

        return value

    if _is_literal(ann):
        return value

    if ann is _UNKNOWN_ANNOTATION or ann is inspect.Parameter.empty:
        return _json_parse_if_needed(value)

    if _is_bool_annotation(ann):
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True

            if lowered in {"false", "0", "no", "off"}:
                return False

        return bool(value)

    if ann is str:
        return str(value)

    if _is_integer_annotation(ann):
        if value is None:
            raise TypeError("Cannot coerce None to int")
        return _int_from_value(value)

    if _is_float_annotation(ann):
        if value is None:
            raise TypeError("Cannot coerce None to float")
        return _float_from_value(value)

    if isinstance(ann, str):
        text: str = ann.lower()
        if "ndarray" in text or "np.ndarray" in text:
            raw_arr = _json_parse_if_needed(value)
            dtype = _numpy_dtype_from_annotation(ann)

            return np.asarray(raw_arr, dtype=dtype if dtype is not None else None)

        if "tuple" in text:
            raw_tuple = _json_parse_if_needed(value)
            if isinstance(raw_tuple, Sequence) and not isinstance(raw_tuple, (str, bytes)):
                return tuple(raw_tuple)

        if "list" in text or "sequence" in text or "iterable" in text:
            raw_list = _json_parse_if_needed(value)
            if isinstance(raw_list, Sequence) and not isinstance(raw_list, (str, bytes)):
                return list(raw_list)

        if "dict" in text or "mapping" in text:
            raw_dict = _json_parse_if_needed(value)
            if isinstance(raw_dict, Mapping):
                return dict(raw_dict)

    origin = get_origin(ann)
    if ann is np.ndarray or origin is np.ndarray:
        raw = _json_parse_if_needed(value)
        dtype = _numpy_dtype_from_annotation(ann)

        return np.asarray(raw, dtype=dtype if dtype is not None else None)

    if origin in {dict, Mapping}:
        raw = _json_parse_if_needed(value)
        if not isinstance(raw, Mapping):
            raise TypeError(f"Expected mapping for annotation {ann}, got {type(raw)!r}")
        args = get_args(ann)
        key_ann = args[0] if len(args) >= 1 else _UNKNOWN_ANNOTATION
        val_ann = args[1] if len(args) >= 2 else _UNKNOWN_ANNOTATION

        return {_coerce_state_value(k, key_ann): _coerce_state_value(v, val_ann) for k, v in raw.items()}

    if origin in {list, tuple, Sequence, Iterable}:
        raw = _json_parse_if_needed(value)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise TypeError(f"Expected array-like value for annotation {ann}, got {type(raw)!r}")
        raw_items = list(raw)
        args = get_args(ann)

        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(_coerce_state_value(item, args[0]) for item in raw_items)

            if len(args) == len(raw_items) and len(args) > 0:
                return tuple(starmap(_coerce_state_value, zip(raw_items, args, strict=True)))
            elem_ann = args[0] if len(args) > 0 else _UNKNOWN_ANNOTATION

            return tuple(_coerce_state_value(item, elem_ann) for item in raw_items)

        elem_ann = args[0] if len(args) > 0 else _UNKNOWN_ANNOTATION

        return [_coerce_state_value(item, elem_ann) for item in raw_items]

    return _json_parse_if_needed(value)


def _serialize_state_payload(state: ABCState, fields: Sequence[StateFieldSpec]) -> dict[str, JSONValue]:
    payload: dict[str, JSONValue] = {}
    field: StateFieldSpec
    for field in fields:
        if not hasattr(state, field.name):
            if field.required:
                raise ValueError(f"State of type '{type(state).__name__}' is missing required field '{field.name}'")
            continue
        payload[field.name] = _to_json_serializable(getattr(state, field.name))

    return payload


def _construct_state(
    state_cls: type[ABCState], args: Sequence[object] = (), kwargs: Mapping[str, object] | None = None
) -> ABCState:
    state = type.__call__(state_cls, *args, **({} if kwargs is None else dict(kwargs)))
    if not isinstance(state, ABCState):
        raise TypeError(f"{state_cls.__name__} did not construct an ABCState")

    return state


def _deserialize_state_payload(
    state_cls: type[ABCState], fields: Sequence[StateFieldSpec], payload: Mapping[str, JSONValue]
) -> ABCState:
    kwargs: dict[str, object] = {}
    field: StateFieldSpec
    for field in fields:
        if field.name not in payload:
            if field.required:
                raise ValueError(f"Missing state field '{field.name}'")
            continue
        kwargs[field.name] = _coerce_state_value(payload[field.name], field.annotation)

    try:
        return _construct_state(state_cls, kwargs=kwargs)

    except TypeError as exc:
        args: list[object] = []
        for field in fields:
            if field.name in kwargs:
                args.append(kwargs[field.name])
            elif field.required:
                raise ValueError(f"Missing required field '{field.name}' for {state_cls.__name__}") from exc
            else:
                args.append(field.default)

        return _construct_state(state_cls, args=args)


def get_environment_default_renderer_settings(env_name: str) -> dict[str, JSONValue]:
    """Return a copy of renderer defaults for the environment."""
    spec: EnvironmentSpec = _get_environment_spec(env_name)

    return copy.deepcopy(spec.renderer_defaults)


def serialize_state(env_name: str, state: ABCState) -> dict[str, JSONValue]:
    """Serialize a state into a JSON-friendly structure."""
    spec: EnvironmentSpec = _get_environment_spec(env_name)

    return _serialize_state_payload(state, spec.state_fields)


def deserialize_state(env_name: str, payload: Mapping[str, JSONValue]) -> ABCState:
    """Deserialize a state payload produced by :func:`serialize_state`."""
    spec: EnvironmentSpec = _get_environment_spec(env_name)

    return _deserialize_state_payload(spec.state_cls, spec.state_fields, dict(payload))


def generate_start_state(env_name: str) -> dict[str, JSONValue]:
    """Generate a single start state for the requested environment."""
    env: ABCEnvironment[ABCState] = _get_environment_instance(env_name)
    state = _generate_state_from_env(env)

    return serialize_state(env_name, state)


def normalize_renderer_settings(
    env_name: str, overrides: Mapping[str, JSONValue] | None = None
) -> dict[str, RendererKwarg]:
    """Prepare renderer keyword arguments for a given environment."""
    defaults: dict[str, JSONValue] = get_environment_default_renderer_settings(env_name)
    merged: dict[str, JSONValue] = {**defaults}
    if overrides:
        merged.update({key: value for key, value in overrides.items() if value is not None})

    kwargs: dict[str, RendererKwarg] = {}
    if "dpi" in merged and merged["dpi"] is not None:
        kwargs["dpi"] = _int_from_value(merged["dpi"])

    size_value = merged.get("size")
    vals: list[float]
    if size_value is not None:
        if isinstance(size_value, Sequence) and not isinstance(size_value, (str, bytes)):
            vals = [_float_from_value(v) for v in size_value]
            if len(vals) == 2:
                kwargs["figsize"] = (vals[0], vals[1])
            elif len(vals) == 1:
                kwargs["figsize"] = (vals[0], vals[0])
        else:
            size_float = _float_from_value(size_value)
            kwargs["figsize"] = (size_float, size_float)

    if "figsize" in merged and "figsize" not in kwargs:
        fig_val = merged["figsize"]
        if isinstance(fig_val, Sequence) and not isinstance(fig_val, (str, bytes)):
            vals = [_float_from_value(v) for v in fig_val]
            if len(vals) == 2:
                kwargs["figsize"] = (vals[0], vals[1])
        else:
            size_float = _float_from_value(fig_val)
            kwargs["figsize"] = (size_float, size_float)

    return kwargs


# ==================================================================================================
# Effect metadata and UI descriptions
# ==================================================================================================


_EFFECT_PARAM_TYPES: dict[str, dict[str, object]] = {}
_EFFECT_DEFAULTS: dict[str, dict[str, EffectValue]] = {}
_EFFECT_PARAM_ORDER: dict[str, list[str]] = {}
_EFFECT_TARGET_TYPES: dict[str, type | None] = {}
_EFFECT_MODULES: dict[str, str] = {}
_EFFECT_STAGES: dict[str, str] = {}


@lru_cache(maxsize=1)
def _effect_specs_by_stage() -> EffectSpecsMapping:
    """Compute effect metadata grouped by stage for UI consumption."""
    registry: EffectRegistry = get_registry()
    registry.auto_discover_effects()

    specs: EffectSpecsMapping = {stage.name: [] for stage in EffectStage}

    for effect_name in sorted(registry.list_names()):
        effect: EffectProtocol = registry.get(effect_name)
        metadata: EffectMetadata = effect.__effect_metadata__
        effect_module_name = getattr(effect, "__module__", "")

        signature: Signature = inspect.signature(effect)
        parameters: list[tuple[str, Parameter]] = list(signature.parameters.items())
        annotations: dict[str, object] = _effect_type_hints(effect, signature)
        target_param: str | None = None
        if parameters:
            target_param = parameters[0][0]

        # Record parameter metadata excluding the implicit data parameter
        param_types: dict[str, object] = {}
        defaults: dict[str, EffectValue] = {}
        ordered_params: list[str] = []
        param_specs: list[ParameterSpec] = []

        for name, param in parameters[1:]:
            annotation: object = annotations.get(name, metadata.parameters.get(name, _UNKNOWN_ANNOTATION))
            default: EffectValue = (
                param.default if param.default is not inspect.Parameter.empty else metadata.default_values.get(name)
            )

            param_types[name] = annotation
            if default is not None:
                defaults[name] = default
            ordered_params.append(name)
            param_specs.append(_build_param_spec(name, annotation, default))

        # Persist raw metadata for pipeline construction
        _EFFECT_PARAM_TYPES[effect_name] = param_types
        _EFFECT_DEFAULTS[effect_name] = defaults
        _EFFECT_PARAM_ORDER[effect_name] = ordered_params
        _EFFECT_MODULES[effect_name] = effect_module_name
        _EFFECT_STAGES[effect_name] = metadata.stage.name
        target_annotation: object = (
            annotations.get(target_param, metadata.parameters.get(target_param, _UNKNOWN_ANNOTATION))
            if target_param
            else _UNKNOWN_ANNOTATION
        )
        _EFFECT_TARGET_TYPES[effect_name] = _resolve_annotation_type(target_annotation)

        specs[metadata.stage.name].append({
            "name": effect_name,
            "stage": metadata.stage.name,
            "category": metadata.category.name,
            "description": metadata.description,
            "performance": metadata.performance_level,
            "requires_rng": metadata.requires_rng,
            "target_param": target_param,
            "target_type": _format_annotation(target_annotation),
            "parameters": param_specs,
        })

    # Sort effects within each stage by category then name for consistent UI layout
    for stage_effects in specs.values():
        stage_effects.sort(key=operator.itemgetter("category", "name"))

    return specs


def _build_param_spec(name: str, annotation: object, default: object) -> ParameterSpec:
    """Create a JSON-friendly parameter description for the UI."""
    kind, options, step, placeholder = _infer_input_kind(annotation, default)
    spec: ParameterSpec = {
        "name": name,
        "label": name.replace("_", " ").title(),
        "kind": kind,
        "annotation": _format_annotation(annotation),
        "default": _to_json_serializable(default),
    }
    if options:
        spec["options"] = options
    if step is not None:
        spec["step"] = step
    if placeholder:
        spec["placeholder"] = placeholder

    return spec


def _infer_input_kind(
    annotation: object, default: object
) -> tuple[str, list[dict[str, JSONValue]] | None, float | None, str | None]:
    """Infer UI control type from parameter annotation and default."""
    ann = _unwrap_annotated(annotation)
    is_optional, union_members = _split_union(ann)
    if (default is None and is_optional and len(union_members) == 1) or len(union_members) == 1:
        ann = union_members[0]

    options = _literal_options(ann)
    if options:
        return "select", options, None, None

    member: object
    for member in union_members:
        options = _literal_options(member)
        if options:
            return "select", options, None, None

    # Booleans first (covers bool default)
    if _is_bool_annotation(ann) or isinstance(default, bool):
        return "switch", None, None, None

    # Distinguish ints/floats (bool already handled)
    if _is_integer_annotation(ann) or isinstance(default, (int, np.integer)):
        return "number", None, 1.0, None

    if _is_float_annotation(ann) or isinstance(default, (float, np.floating)):
        return "number", None, 0.05, None

    if ann is str or isinstance(default, str):
        if isinstance(default, str) and default.startswith("#") and len(default) in {4, 7}:
            return "color", None, None, None

        return "text", None, None, None

    if isinstance(ann, str):
        ann_text = ann.lower()
        if (
            "ndarray" in ann_text
            or "list" in ann_text
            or "tuple" in ann_text
            or "sequence" in ann_text
            or "mapping" in ann_text
            or "dict" in ann_text
        ):
            return "json", None, None, "Enter JSON value"

        if "str" in ann_text:
            return "text", None, None, None

    origin: type | None = get_origin(ann)

    # Arrays / sequences -> JSON editor
    if origin in {list, tuple, Sequence, Iterable} or isinstance(default, (list, tuple)):
        return "json", None, None, "Enter JSON array"

    # Mappings -> JSON editor
    if origin in {dict, Mapping} or isinstance(default, Mapping):
        return "json", None, None, "Enter JSON object"

    if ann is np.ndarray or origin is np.ndarray:
        return "json", None, None, "Enter JSON array"

    # Optional/union fallback: choose first informative member before defaulting to text.
    if len(union_members) > 1 or is_optional:
        for member in union_members:
            if member is type(None) or member is None or member is ann:
                continue
            kind, member_options, step, placeholder = _infer_input_kind(member, default)
            if kind != "text":
                return kind, member_options, step, placeholder

    # Fallback

    return "text", None, None, None


def _format_annotation(annotation: object) -> str:
    """Convert a type annotation into a readable string."""
    if annotation is _UNKNOWN_ANNOTATION or annotation is None:
        return "unknown"

    if isinstance(annotation, type):
        return annotation.__name__

    origin = get_origin(annotation)
    if origin:
        args: str = ", ".join(_format_annotation(arg) for arg in get_args(annotation))

        origin_name = getattr(origin, "__name__", str(origin))

        return f"{origin_name}[{args}]"

    return str(annotation)


def _resolve_annotation_type(annotation: object) -> type | None:
    """Extract a concrete type from annotation if possible."""
    ann = _unwrap_annotated(annotation)
    if isinstance(ann, type):
        return ann

    if isinstance(ann, str):
        stripped = ann.strip().strip("\"'")
        known = _known_annotation_type(stripped)
        if known is not None:
            return known
        if any(token in stripped for token in ("ImageArray", "NDArray", "ArrayLike")):
            return np.ndarray
        if any(token in stripped for token in ("FigureType", "Figure")):
            return _known_annotation_type("FigureType")

    _, members = _split_union(ann)
    member: object
    for member in members:
        if isinstance(member, type):
            return member

    origin: type | None = get_origin(ann)
    if isinstance(origin, type) and origin is not types.UnionType and str(origin) != "typing.Union":
        return origin

    return None


def _to_json_serializable(value: object) -> JSONValue:
    """Convert arbitrary Python objects into JSON-friendly structures."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_json_serializable(v) for v in value]

    if isinstance(value, Mapping):
        return {str(k): _to_json_serializable(v) for k, v in value.items()}

    return str(value)


def get_effect_specs_for_environment(env_name: str) -> EffectSpecsMapping:
    """Return effect specs filtered for an environment."""
    specs: EffectSpecsMapping = _effect_specs_by_stage()
    env_spec: EnvironmentSpec = _get_environment_spec(env_name)

    filtered: EffectSpecsMapping = {stage: [] for stage in specs}
    for stage, effects in specs.items():
        for effect in effects:
            if _effect_supported_in_env(effect["name"], stage, env_spec):
                filtered[stage].append(copy.deepcopy(effect))

    return filtered


def _effect_supported_in_env(effect_name: str, stage: str, env_spec: EnvironmentSpec) -> bool:
    module_name = _EFFECT_MODULES.get(effect_name, "")
    is_shared_image_effect = module_name == _SHARED_IMAGE_EFFECT_MODULE
    is_domain_specific = module_name.startswith(env_spec.domain_module_prefix)
    if not is_shared_image_effect and not is_domain_specific:
        return False

    expected: type | None = _EFFECT_TARGET_TYPES.get(effect_name)
    if expected is None:
        return True

    allowed_types: tuple[type, ...] = env_spec.stage_target_types.get(stage, ())
    if not allowed_types and is_domain_specific:
        return True

    return any(_types_compatible(expected, allowed) for allowed in allowed_types)


def _types_compatible(expected: type, allowed: type) -> bool:
    try:
        return issubclass(expected, allowed) or issubclass(allowed, expected)
    except TypeError:
        return False


# ==================================================================================================
# Pipeline construction
# ==================================================================================================


def build_stage_pipeline(selection: Mapping[str, Sequence[ActiveEffect]]) -> StagePipelines | None:
    """Build a StagePipelines instance from UI selection data.

    Args:
        selection: Mapping from stage name (``EffectStage.name``) to a sequence of
            effect descriptors. Each descriptor must provide ``name`` and may include
            ``enabled`` and ``params`` keys.

    Returns:
        A compiled :class:`StagePipelines` instance or ``None`` if no effects are enabled.
    """
    if not selection:
        return None

    _effect_specs_by_stage()  # Populate effect metadata caches.

    pre_pipeline: Pipeline[FigureType] | None = None
    obj_pipeline: Pipeline[FreezeInput] | None = None
    post_pipeline: Pipeline[ImageArray] | None = None

    for stage_name, effects in selection.items():
        try:
            stage_enum: EffectStage = EffectStage[stage_name]
        except KeyError:
            rich_logger.warning(f"Ignoring unknown stage '{stage_name}' in pipeline selection")
            continue

        for eff_entry in effects:
            if not eff_entry["enabled"]:
                continue

            effect_name: str = eff_entry["name"]
            if not effect_name:
                continue

            try:
                effect: EffectProtocol = get_effect(effect_name)
            except KeyError:
                rich_logger.effect_not_found(effect_name)
                continue

            param_types: dict[str, object] = _EFFECT_PARAM_TYPES[effect_name]
            order: list[str] = _EFFECT_PARAM_ORDER[effect_name]
            provided: dict[str, JSONValue] = eff_entry["params"]

            resolved_params: dict[str, EffectValue] = {}
            for param_name in order:
                raw_value: JSONValue = provided[param_name]
                try:
                    resolved = _coerce_param(raw_value, param_types[param_name])
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    # Interactive edits can leave a background path incomplete. For invalid paths,
                    # we silently fall back to the default generated background to avoid noisy stack traces.
                    if effect_name == "background_image" and param_name == "image":
                        rich_logger.debug(
                            f"Ignoring invalid background image input '{raw_value}': {exc}. "
                            "Falling back to default background."
                        )
                    else:
                        rich_logger.effect_creation_failed(effect_name, f"parameter '{param_name}': {exc}")
                    resolved = None

                if resolved is not None:
                    resolved_params[param_name] = resolved

            # Special-case defaults for known complex effects
            if effect_name == "background_image" and "image" not in resolved_params:
                resolved_params["image"] = create_default_background_image()

            if stage_enum is EffectStage.PRE_RENDER:
                if pre_pipeline is None:
                    pre_pipeline = Pipeline()
                pipeline: DashboardPipeline = pre_pipeline
            elif stage_enum is EffectStage.OBJECT_RENDER:
                if obj_pipeline is None:
                    obj_pipeline = Pipeline()
                pipeline = obj_pipeline
            else:
                if post_pipeline is None:
                    post_pipeline = Pipeline()
                pipeline = post_pipeline
            try:
                pipeline.add(effect, **resolved_params)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                rich_logger.effect_creation_failed(effect_name, str(exc))

    if pre_pipeline is None and obj_pipeline is None and post_pipeline is None:
        return None

    return StagePipelines(
        pre=pre_pipeline.compile() if (pre_pipeline is not None and pre_pipeline.effect_count > 0) else None,
        obj=obj_pipeline.compile() if (obj_pipeline is not None and obj_pipeline.effect_count > 0) else None,
        post=post_pipeline.compile() if (post_pipeline is not None and post_pipeline.effect_count > 0) else None,
    )


def _try_coerce_param_union_member(value: object, member: object) -> tuple[bool, EffectValue | None, Exception | None]:
    """Try coercing a single union branch for effect parameters."""
    try:
        return True, _coerce_param(value, member), None
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        return False, None, exc


def _coerce_param(value: object, annotation: object) -> EffectValue | None:
    """Convert UI-provided values into the types expected by an effect."""
    if value is None:
        return None

    if isinstance(value, str) and not value.strip():
        return None

    ann = _unwrap_annotated(annotation)
    is_optional, union_members = _split_union(ann)
    if is_optional and value is None:
        return None

    if len(union_members) == 1:
        ann = union_members[0]
    elif len(union_members) > 1:
        last_error: Exception | None = None
        member: object
        for member in union_members:
            if member is type(None) or member is None:
                continue
            success, resolved, err = _try_coerce_param_union_member(value, member)
            if success:
                return resolved
            last_error = err
        if last_error is not None:
            raise last_error

        return value

    # Literal choices: value already validated by select options.
    if _is_literal(ann):
        return value

    if isinstance(ann, str):
        ann_text: str = ann.lower().replace(" ", "")
        if "imagearray" in ann_text or "ndarray" in ann_text:
            return _coerce_to_ndarray(value)

    if _is_bool_annotation(ann) or isinstance(value, bool):
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            lowered: str = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True

            if lowered in {"false", "0", "no", "off"}:
                return False

        return bool(value)

    if _is_integer_annotation(ann) or (isinstance(value, (int, np.integer)) and not isinstance(value, bool)):
        return _int_from_value(value)

    if _is_float_annotation(ann) or isinstance(value, (float, np.floating)):
        return _float_from_value(value)

    if ann is str:
        return str(value)

    # Coerce array-like annotations after the scalar cases.
    origin: type | None = get_origin(ann)
    if ann is np.ndarray or origin is np.ndarray:
        return _coerce_to_ndarray(value)

    if origin in {list, tuple, Sequence, Iterable} or isinstance(value, (list, tuple)):
        parsed = _json_parse_if_needed(value)
        if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
            raise ValueError(f"expected array-like value, got {type(parsed)!r}")
        parsed_items = list(parsed)
        args = get_args(ann)
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(_coerce_param(item, args[0]) for item in parsed_items)

            if len(args) == len(parsed_items) and len(args) > 0:
                return tuple(starmap(_coerce_param, zip(parsed_items, args, strict=True)))
            elem_ann = args[0] if len(args) > 0 else _UNKNOWN_ANNOTATION

            return tuple(_coerce_param(item, elem_ann) for item in parsed_items)

        elem_ann = args[0] if len(args) > 0 else _UNKNOWN_ANNOTATION

        return [_coerce_param(item, elem_ann) for item in parsed_items]

    if origin in {dict, Mapping} or isinstance(value, Mapping):
        parsed = _json_parse_if_needed(value)
        if not isinstance(parsed, Mapping):
            raise ValueError(f"expected mapping value, got {type(parsed)!r}")
        args = get_args(ann)
        key_ann = args[0] if len(args) >= 1 else _UNKNOWN_ANNOTATION
        val_ann = args[1] if len(args) >= 2 else _UNKNOWN_ANNOTATION

        return {str(_coerce_param(k, key_ann)): _coerce_param(v, val_ann) for k, v in parsed.items()}

    if isinstance(value, str):
        return _json_parse_if_needed(value)

    return value


def _coerce_to_ndarray(value: object) -> NDArray[np.float32]:
    """Convert arbitrary input to a float32 numpy array with values in [0, 1]."""
    arr: NDArray[np.float32]
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32, copy=False)
    elif isinstance(value, list):
        arr = np.asarray(value, dtype=np.float32)
    elif isinstance(value, str):
        text: str = value.strip()
        if not text:
            raise ValueError("expected non-empty image input")
        if text.startswith("data:image"):
            arr = decode_base64_image(text)
        else:
            parsed: object = _json_parse_if_needed(text)
            if isinstance(parsed, str):
                path: Path = Path(parsed).expanduser()
                if not path.is_file():
                    raise ValueError(f"expected base64 image, JSON array, or existing image file path. Got '{parsed}'")
                with Image.open(path) as image:
                    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
            else:
                arr = np.asarray(parsed, dtype=np.float32)
    else:
        raise TypeError(f"Unsupported array parameter type: {type(value)!r}")

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.size > 0 and arr.max() > 1.0:
        arr /= 255.0

    return np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)


# ==================================================================================================
# Rendering helpers
# ==================================================================================================


def render_environment_to_uri(
    env_name: str,
    state_payload: Mapping[str, JSONValue],
    selection: Mapping[str, Sequence[ActiveEffect]],
    renderer_settings: Mapping[str, JSONValue] | None = None,
) -> str:
    """Render a state with optional effects and return a PNG data URI."""
    env: ABCEnvironment[ABCState] = _get_environment_instance(env_name)
    state: ABCState = deserialize_state(env_name, state_payload)

    return render_state_to_uri(env_name, env, state, selection, renderer_settings)


def render_state_to_uri(
    env_name: str,
    env: ABCEnvironment[ABCState],
    state: ABCState,
    selection: Mapping[str, Sequence[ActiveEffect]],
    renderer_settings: Mapping[str, JSONValue] | None = None,
) -> str:
    """Render a concrete environment/state pair with optional effects to PNG data URI."""
    effects: StagePipelines | None = build_stage_pipeline(selection)
    kwargs: dict[str, RendererKwarg] = normalize_renderer_settings(env_name, renderer_settings)
    try:
        raw: NDArray[np.float32] = env.state_to_real([state], effects=effects, **kwargs)
    except TypeError as exc:
        if kwargs and "unexpected keyword" in str(exc).lower():
            rich_logger.warning(
                f"Environment '{env_name}' rejected renderer kwargs ({sorted(kwargs.keys())}). Retrying with defaults."
            )
            raw = env.state_to_real([state], effects=effects)
        else:
            raise
    image: NDArray[np.float32] = _normalize_image(raw)

    return image_to_data_uri(image)


def _normalize_image(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize raw renderer output to HxWx3 float32 array in [0,1]."""
    arr: NDArray[np.float32] = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[2] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if arr.size > 0 and arr.max() > 1.0:
        arr /= 255.0
    # Upscale very low-resolution outputs for consistent dashboard preview quality.
    h: int = int(arr.shape[0])
    w: int = int(arr.shape[1])
    max_dim: int = max(h, w)
    min_preview_dim: int = 360
    if max_dim > 0 and max_dim < min_preview_dim:
        scale: int = int(np.ceil(min_preview_dim / max_dim))
        arr = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)

    return np.clip(arr, 0.0, 1.0)


# ==================================================================================================
# Image conversion utilities
# ==================================================================================================


_BASE64_BUFFER = io.BytesIO()


def image_to_data_uri(image: NDArray[np.float32]) -> str:
    """Convert an HxWx3 float32/uint8 image to a PNG data URI."""
    arr: NDArray[np.float32 | np.uint8] = image
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    _BASE64_BUFFER.seek(0)
    _BASE64_BUFFER.truncate()
    Image.fromarray(arr).save(_BASE64_BUFFER, format="PNG")
    data: str = base64.b64encode(_BASE64_BUFFER.getvalue()).decode("ascii")

    return f"data:image/png;base64,{data}"


def decode_base64_image(data_uri: str) -> NDArray[np.float32]:
    """Decode a base64 image data URI into a float32 NumPy array."""
    if "," in data_uri:
        _, payload = data_uri.split(",", 1)
    else:
        payload = data_uri
    image: Image.Image = Image.open(io.BytesIO(base64.b64decode(payload)))
    arr: NDArray[np.float32] = np.asarray(image).astype(np.float32, copy=False)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    return np.clip(arr / 255.0, 0.0, 1.0)


@lru_cache(maxsize=1)
def create_default_background_image() -> NDArray[np.float32]:
    """Generate a procedural 512x512 RGB background."""
    size = 512
    square = 32
    i_idx: NDArray[np.intp]
    j_idx: NDArray[np.intp]
    i_idx, j_idx = np.indices((size, size))
    checker: NDArray[np.bool_] = ((i_idx // square + j_idx // square) & 1) == 0

    r: NDArray[np.float32] = np.where(checker, 0.95, 0.80)
    g: NDArray[np.float32] = np.where(checker, 0.93, 0.82)
    b: NDArray[np.float32] = np.where(checker, 0.98, 0.90)

    radius: NDArray[np.float32] = np.sqrt(((2 * i_idx / size) - 1.0) ** 2 + ((2 * j_idx / size) - 1.0) ** 2)
    vignette: NDArray[np.float32] = np.clip(0.3 * (1.0 - radius), 0.0, 0.3)

    gradient_x: NDArray[np.float32] = (j_idx / size * 0.05).astype(np.float32, copy=False)
    gradient_y: NDArray[np.float32] = (i_idx / size * 0.08).astype(np.float32, copy=False)

    rgb: NDArray[np.float32] = np.stack((r - vignette + gradient_x, g - vignette, b - vignette + gradient_y), axis=-1)

    return np.clip(rgb.astype(np.float32, copy=False), 0.0, 1.0)


# ==================================================================================================
# Miscellaneous
# ==================================================================================================


def print_startup_message(host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    """Print a stylised startup banner."""
    rich_logger.startup_banner(host, port, debug)
