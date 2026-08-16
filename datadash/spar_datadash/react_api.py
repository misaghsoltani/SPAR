"""Flask API and static host for the React SPAR data dashboard."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from importlib import resources
import json
from pathlib import Path
import re
import threading
from time import monotonic, perf_counter
from typing import TYPE_CHECKING, TypedDict
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory

from spar.utils.env_utils.effects_core import EffectStage
from spar.utils.env_utils.env_utils import get_environment_class

from .rich_logger import get_rich_logger
from .sweep import (
    get_gen_data_effect_presets,
    list_env_config_files,
    parse_effect_presets_from_text,
    read_env_config_text,
)
from .utils import (
    deserialize_state,
    generate_start_state,
    get_effect_specs_for_environment,
    get_environment_default_renderer_settings,
    list_environment_options,
    render_environment_to_uri,
    render_state_to_uri,
    serialize_state,
)

if TYPE_CHECKING:
    from typing import TypeAlias

    from flask import Response

    from spar.environments.abstracts.environment import ABCEnvironment
    from spar.environments.abstracts.state import ABCState

    from .rich_logger import RichLogger
    from .utils import ActiveEffect, EffectSpecsMapping, ParameterSpec


logger: RichLogger = get_rich_logger(__name__)

STAGE_ORDER: tuple[str, ...] = tuple(
    stage.name for stage in (EffectStage.PRE_RENDER, EffectStage.OBJECT_RENDER, EffectStage.POST_RENDER)
)
_MAX_HISTORY_CACHE: int = 160
_MAX_SWEEP_CELLS: int = 96
_INTERACTIVE_MAX_SESSIONS: int = 64
_INTERACTIVE_SESSION_TTL_SEC: int = 60 * 30
_SWIPE_THRESHOLD_DEFAULT: float = 18.0

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonPayload: TypeAlias = Mapping[str, JsonValue]
JsonObject: TypeAlias = dict[str, JsonValue]
InteractiveEventPayload: TypeAlias = dict[str, str | int | float]


class EffectEntry(TypedDict):
    """Client-side effect toggle and parameter payload."""

    enabled: bool
    params: JsonObject


class RenderResult(TypedDict):
    """Serialized image render payload for non-interactive requests."""

    image: str
    render_ms: float
    cached: bool


class SweepCellResult(TypedDict):
    """One rendered cell in an effect-configuration sweep grid."""

    label: str
    image: str
    render_ms: float
    cached: bool


class InteractiveStartResult(TypedDict):
    """Initial payload for a newly created interactive session."""

    session_id: str
    env: str
    state: JsonObject
    action_count: int
    action_labels: list[str]
    interactive_bindings: JsonObject
    image: str
    render_ms: float


class InteractiveStepResult(TypedDict):
    """Payload returned after applying one interactive action."""

    session_id: str
    env: str
    state: JsonObject
    action_applied: int
    action_count: int
    action_labels: list[str]
    interactive_bindings: JsonObject
    image: str
    render_ms: float


class InteractiveEventResult(TypedDict, total=False):
    """Payload returned for higher-level interactive input events."""

    session_id: str
    env: str
    state: JsonObject
    action_count: int
    action_labels: list[str]
    interactive_bindings: JsonObject
    actions_applied: list[int]
    handled: bool
    image: str
    render_ms: float


@dataclass(slots=True)
class _CacheEntry:
    image: str
    render_ms: float


class _RenderCache:
    """A tiny lock-protected LRU cache for identical render requests."""

    def __init__(self, max_entries: int = _MAX_HISTORY_CACHE) -> None:
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock: threading.Lock = threading.Lock()
        self._max_entries: int = max_entries

    def get(self, key: str) -> _CacheEntry | None:
        with self._lock:
            entry: _CacheEntry | None = self._store.get(key)
            if entry is None:
                return None

            self._store.move_to_end(key)

            return entry

    def set(self, key: str, value: _CacheEntry) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)


@dataclass(slots=True)
class _InteractiveSession:
    session_id: str
    env_name: str
    env: ABCEnvironment[ABCState]
    state: ABCState
    touched_at: float
    pointer_anchors: dict[int, tuple[float, float]] = field(default_factory=dict, repr=False, compare=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class _InteractiveSessionStore:
    """Lock-protected store for active interactive sessions."""

    def __init__(
        self, max_sessions: int = _INTERACTIVE_MAX_SESSIONS, ttl_sec: int = _INTERACTIVE_SESSION_TTL_SEC
    ) -> None:
        self._store: OrderedDict[str, _InteractiveSession] = OrderedDict()
        self._lock: threading.Lock = threading.Lock()
        self._max_sessions: int = max_sessions
        self._ttl_sec: float = float(ttl_sec)

    def _prune_locked(self) -> None:
        now: float = monotonic()
        expired: list[str] = [sid for sid, sess in self._store.items() if (now - sess.touched_at) > self._ttl_sec]
        for sid in expired:
            self._store.pop(sid, None)
        while len(self._store) > self._max_sessions:
            self._store.popitem(last=False)

    def create(self, env_name: str, env: ABCEnvironment[ABCState], state: ABCState) -> _InteractiveSession:
        with self._lock:
            self._prune_locked()
            session_id: str = uuid4().hex
            session = _InteractiveSession(
                session_id=session_id, env_name=env_name, env=env, state=state, touched_at=monotonic()
            )
            self._store[session_id] = session
            self._store.move_to_end(session_id)
            self._prune_locked()

            return session

    def get(self, session_id: str) -> _InteractiveSession | None:
        with self._lock:
            self._prune_locked()
            session: _InteractiveSession | None = self._store.get(session_id)
            if session is None:
                return None

            session.touched_at = monotonic()
            self._store.move_to_end(session_id)

            return session

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)


def _generate_start_state_for_env_instance(env: ABCEnvironment[ABCState]) -> ABCState:
    try:
        return env.generate_start_states(1)[0]

    except TypeError:
        return env.generate_start_states(1, level_seeds=None)[0]


def _to_json_value(raw: object) -> JsonValue:
    if raw is None or isinstance(raw, (str, int, float, bool)):
        return raw

    if isinstance(raw, Mapping):
        return {str(key): _to_json_value(value) for key, value in raw.items()}

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [_to_json_value(value) for value in raw]

    return str(raw)


def _to_json_payload(raw: object) -> JsonObject:
    if not isinstance(raw, Mapping):
        return {}

    return {str(key): _to_json_value(value) for key, value in raw.items()}


def _json_mapping_or_empty(raw: JsonValue | None) -> JsonPayload:
    if isinstance(raw, Mapping):
        return raw

    return {}


def _resolve_action_labels(env: ABCEnvironment[ABCState]) -> tuple[int, list[str]]:
    count = int(getattr(env, "num_actions_max", 0) or 0)
    if count <= 0:
        return 0, []

    labels: list[str] = [f"Action {idx}" for idx in range(count)]
    moves = getattr(env, "moves", None)
    if isinstance(moves, Mapping):
        for idx in range(count):
            label = moves.get(idx)
            if label is not None:
                labels[idx] = str(label)

        return count, labels

    if isinstance(moves, Sequence) and not isinstance(moves, (str, bytes)):
        limit: int = min(count, len(moves))
        for idx in range(limit):
            labels[idx] = str(moves[idx])

    return count, labels


def _normalize_binding_key(raw: JsonValue) -> str:
    if not isinstance(raw, str):
        return ""

    key: str = raw.strip().lower()
    if key == " ":
        return "space"

    if key == "spacebar":
        return "space"

    return key


def _split_label_tokens(label: str) -> list[str]:

    return [token for token in re.split(r"[^a-z0-9]+", label.lower()) if token]


def _infer_direction_from_label(label: str) -> str | None:
    tokens: list[str] = _split_label_tokens(label)
    if not tokens:
        return None

    token_set: set[str] = set(tokens)
    if token_set.intersection({"noop", "wait", "stay", "idle", "none"}):
        return "noop"

    if token_set.intersection({"up", "north"}):
        return "up"

    if token_set.intersection({"down", "south"}):
        return "down"

    if token_set.intersection({"left", "west"}):
        return "left"

    if token_set.intersection({"right", "east"}):
        return "right"

    alpha_tokens: list[str] = [token for token in token_set if token.isalpha()]
    if len(alpha_tokens) != 1:
        return None

    short: str = alpha_tokens[0]
    if short in {"u", "n"}:
        return "up"

    if short in {"d", "s"}:
        return "down"

    if short in {"l", "w"}:
        return "left"

    if short in {"r", "e"}:
        return "right"

    return None


def _infer_default_directional_by_environment(env_name: str, action_count: int) -> dict[str, int]:
    if action_count < 4:
        return {}

    key: str = env_name.strip().lower()
    if "sokoban" in key:
        mapping: dict[str, int] = {"up": 0, "down": 1, "left": 2, "right": 3}
        if action_count > 4:
            mapping["noop"] = 4

        return mapping

    if "ice" in key or "digit" in key:
        mapping = {"up": 0, "right": 1, "left": 2, "down": 3}
        if action_count > 4:
            mapping["noop"] = 4

        return mapping

    mapping = {"up": 0, "right": 1, "down": 2, "left": 3}
    if action_count > 4:
        mapping["noop"] = 4

    return mapping


def _add_key_binding(bindings: dict[str, int], key: str, action: int) -> None:
    if action < 0:
        return
    normalized: str = _normalize_binding_key(key)
    if not normalized or normalized in bindings:
        return
    bindings[normalized] = action


def _bind_directional_keys(bindings: dict[str, int], directional: Mapping[str, int]) -> None:
    up: int | None = directional.get("up")
    down: int | None = directional.get("down")
    left: int | None = directional.get("left")
    right: int | None = directional.get("right")
    noop: int | None = directional.get("noop")

    if isinstance(up, int):
        _add_key_binding(bindings, "arrowup", up)
        _add_key_binding(bindings, "w", up)
        _add_key_binding(bindings, "k", up)
    if isinstance(down, int):
        _add_key_binding(bindings, "arrowdown", down)
        _add_key_binding(bindings, "s", down)
        _add_key_binding(bindings, "j", down)
    if isinstance(left, int):
        _add_key_binding(bindings, "arrowleft", left)
        _add_key_binding(bindings, "a", left)
        _add_key_binding(bindings, "h", left)
    if isinstance(right, int):
        _add_key_binding(bindings, "arrowright", right)
        _add_key_binding(bindings, "d", right)
        _add_key_binding(bindings, "l", right)
    if isinstance(noop, int):
        _add_key_binding(bindings, "space", noop)
        _add_key_binding(bindings, "enter", noop)
        _add_key_binding(bindings, "n", noop)


def _coerce_action_candidate(raw: JsonValue | None, action_count: int) -> int | None:
    if isinstance(raw, bool):
        return None

    value: int
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float):
        if not raw.is_integer():
            return None

        value = int(raw)
    elif isinstance(raw, str):
        text: str = raw.strip()
        if not text:
            return None

        try:
            value = int(text)
        except ValueError:
            return None

    else:
        return None

    if 0 <= value < action_count:
        return value

    return None


def _sanitize_action_map(raw: JsonValue | None, action_count: int) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}

    mapped: dict[str, int] = {}
    for key, value in raw.items():
        normalized_key: str = _normalize_binding_key(str(key))
        if not normalized_key:
            continue
        action: int | None = _coerce_action_candidate(value, action_count)
        if action is None:
            continue
        mapped[normalized_key] = action

    return mapped


def _merge_mappings(base: JsonPayload, override: JsonPayload) -> JsonObject:
    merged: JsonObject = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(existing, value)
            continue
        merged[key] = value

    return merged


def _resolve_env_binding_overrides(env: ABCEnvironment[ABCState]) -> JsonPayload:
    for attr in ("get_interactive_bindings", "interactive_bindings"):
        candidate = getattr(env, attr, None)
        if not callable(candidate):
            continue
        try:
            override = candidate()
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning(f"Ignoring '{attr}' interactive bindings due to error: {exc}")
            continue
        if isinstance(override, Mapping):
            return _to_json_payload(override)

    return {}


def _build_interactive_bindings(
    env_name: str, action_count: int, action_labels: Sequence[str], env_override: JsonPayload | None = None
) -> JsonObject:
    directional: dict[str, int] = {}
    key_to_action: dict[str, int] = {}

    for idx, label in enumerate(action_labels):
        _add_key_binding(key_to_action, str(idx), idx)
        _add_key_binding(key_to_action, f"digit{idx}", idx)
        _add_key_binding(key_to_action, f"numpad{idx}", idx)
        if idx < 9:
            _add_key_binding(key_to_action, str(idx + 1), idx)
            _add_key_binding(key_to_action, f"digit{idx + 1}", idx)
            _add_key_binding(key_to_action, f"numpad{idx + 1}", idx)

        normalized_label = _normalize_binding_key(label)
        if normalized_label:
            _add_key_binding(key_to_action, normalized_label, idx)

        label_tokens: list[str] = _split_label_tokens(label)
        for token in label_tokens:
            _add_key_binding(key_to_action, token, idx)
        if label_tokens:
            _add_key_binding(key_to_action, label_tokens[0][0], idx)

        direction: str | None = _infer_direction_from_label(label)
        if direction and direction not in directional:
            directional[direction] = idx

    fallback_directional: dict[str, int] = _infer_default_directional_by_environment(env_name, action_count)
    for key in ("up", "down", "left", "right", "noop"):
        if key not in directional and key in fallback_directional:
            directional[key] = fallback_directional[key]

    _bind_directional_keys(key_to_action, directional)

    button_to_action: dict[str, int] = {}
    event_to_action: dict[str, int] = {}
    for idx, label in enumerate(action_labels):
        token_set: set[str] = set(_split_label_tokens(label))
        if not token_set:
            continue
        click_hint = token_set.intersection({"click", "mouse", "button", "tap"})
        if click_hint:
            if token_set.intersection({"left", "primary", "lmb", "mouse1"}):
                button_to_action.setdefault("0", idx)
            if token_set.intersection({"middle", "aux", "mmb", "mouse2"}):
                button_to_action.setdefault("1", idx)
            if token_set.intersection({"right", "secondary", "rmb", "mouse3"}):
                button_to_action.setdefault("2", idx)
        if token_set.intersection({"double", "dbl"}):
            event_to_action.setdefault("dblclick", idx)
        if token_set.intersection({"context", "menu"}):
            event_to_action.setdefault("contextmenu", idx)

    if "0" not in button_to_action and "noop" in directional:
        button_to_action["0"] = directional["noop"]

    vertical: dict[str, int] = {}
    horizontal: dict[str, int] = {}
    if "up" in directional:
        vertical["negative"] = directional["up"]
    if "down" in directional:
        vertical["positive"] = directional["down"]
    if "left" in directional:
        horizontal["negative"] = directional["left"]
    if "right" in directional:
        horizontal["positive"] = directional["right"]

    base_bindings: JsonObject = _to_json_payload({
        "version": 1,
        "keyboard": {"enabled": action_count > 0, "events": ["keydown"], "key_to_action": key_to_action},
        "pointer": {
            "enabled": action_count > 0,
            "events": ["pointerdown", "pointerup", "click", "auxclick", "dblclick", "contextmenu"],
            "directional": directional,
            "button_to_action": button_to_action,
            "event_to_action": event_to_action,
            "swipe_threshold": _SWIPE_THRESHOLD_DEFAULT,
        },
        "wheel": {"enabled": action_count > 0, "events": ["wheel"], "vertical": vertical, "horizontal": horizontal},
    })

    merged = _merge_mappings(base_bindings, env_override) if env_override else base_bindings

    keyboard_cfg = _json_mapping_or_empty(merged.get("keyboard"))
    pointer_cfg = _json_mapping_or_empty(merged.get("pointer"))
    wheel_cfg = _json_mapping_or_empty(merged.get("wheel"))

    keyboard_events_raw = keyboard_cfg.get("events", ["keydown"])
    keyboard_events: list[str] = []
    if isinstance(keyboard_events_raw, Sequence) and not isinstance(keyboard_events_raw, (str, bytes)):
        for entry in keyboard_events_raw:
            normalized = _normalize_binding_key(str(entry))
            if normalized and normalized not in keyboard_events:
                keyboard_events.append(normalized)
    if not keyboard_events:
        keyboard_events = ["keydown"]

    pointer_events_raw = pointer_cfg.get("events", [])
    pointer_events: list[str] = []
    if isinstance(pointer_events_raw, Sequence) and not isinstance(pointer_events_raw, (str, bytes)):
        for entry in pointer_events_raw:
            normalized = _normalize_binding_key(str(entry))
            if normalized and normalized not in pointer_events:
                pointer_events.append(normalized)

    wheel_events_raw = wheel_cfg.get("events", ["wheel"])
    wheel_events: list[str] = []
    if isinstance(wheel_events_raw, Sequence) and not isinstance(wheel_events_raw, (str, bytes)):
        for entry in wheel_events_raw:
            normalized = _normalize_binding_key(str(entry))
            if normalized and normalized not in wheel_events:
                wheel_events.append(normalized)
    if not wheel_events:
        wheel_events = ["wheel"]

    directional_raw = pointer_cfg.get("directional", {})
    directional_map = _sanitize_action_map(directional_raw, action_count)
    sanitized_directional: dict[str, int] = {}
    for direction in ("up", "down", "left", "right", "noop"):
        if direction in directional_map:
            sanitized_directional[direction] = directional_map[direction]

    swipe_threshold_raw = pointer_cfg.get("swipe_threshold", _SWIPE_THRESHOLD_DEFAULT)
    swipe_threshold: float
    if isinstance(swipe_threshold_raw, bool):
        swipe_threshold = _SWIPE_THRESHOLD_DEFAULT
    elif isinstance(swipe_threshold_raw, (int, float)):
        swipe_threshold = float(swipe_threshold_raw)
    else:
        swipe_threshold = _SWIPE_THRESHOLD_DEFAULT
    if swipe_threshold < 0:
        swipe_threshold = _SWIPE_THRESHOLD_DEFAULT

    return _to_json_payload({
        "version": 1,
        "keyboard": {
            "enabled": bool(keyboard_cfg.get("enabled", action_count > 0)),
            "events": keyboard_events,
            "key_to_action": _sanitize_action_map(keyboard_cfg.get("key_to_action", {}), action_count),
        },
        "pointer": {
            "enabled": bool(pointer_cfg.get("enabled", action_count > 0)),
            "events": pointer_events,
            "directional": sanitized_directional,
            "button_to_action": _sanitize_action_map(pointer_cfg.get("button_to_action", {}), action_count),
            "event_to_action": _sanitize_action_map(pointer_cfg.get("event_to_action", {}), action_count),
            "swipe_threshold": swipe_threshold,
        },
        "wheel": {
            "enabled": bool(wheel_cfg.get("enabled", action_count > 0)),
            "events": wheel_events,
            "vertical": _sanitize_action_map(wheel_cfg.get("vertical", {}), action_count),
            "horizontal": _sanitize_action_map(wheel_cfg.get("horizontal", {}), action_count),
        },
    })


def _resolve_interactive_metadata(env_name: str, env: ABCEnvironment[ABCState]) -> tuple[int, list[str], JsonObject]:
    action_count, action_labels = _resolve_action_labels(env)
    overrides = _resolve_env_binding_overrides(env)
    bindings = _build_interactive_bindings(env_name, action_count, action_labels, overrides)

    return action_count, action_labels, bindings


def _extract_int(raw: JsonPayload, *keys: str) -> int | None:
    for key in keys:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool):
            return None

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            if value.is_integer():
                return int(value)

            return None

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None

            try:
                return int(text)

            except ValueError:
                return None

    return None


def _extract_float(raw: JsonPayload, *keys: str) -> float | None:
    for key in keys:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None

            try:
                return float(text)

            except ValueError:
                return None

    return None


def _coerce_interactive_event(payload: JsonPayload) -> InteractiveEventPayload:
    raw_event = payload.get("event")
    if not isinstance(raw_event, Mapping):
        raise TypeError("'event' must be an object")

    event_type = _normalize_binding_key(str(raw_event.get("type", "")))
    if not event_type:
        raise ValueError("'event.type' is required")

    kind_raw = _normalize_binding_key(str(raw_event.get("kind", "")))
    kind = kind_raw
    if kind not in {"keyboard", "pointer", "wheel"}:
        if event_type in {"keydown", "keyup", "keypress"}:
            kind = "keyboard"
        elif event_type in {
            "pointerdown",
            "pointerup",
            "pointercancel",
            "pointerleave",
            "pointermove",
            "click",
            "dblclick",
            "auxclick",
            "contextmenu",
            "mousedown",
            "mouseup",
            "mousemove",
        }:
            kind = "pointer"
        elif event_type in {"wheel", "mousewheel"}:
            kind = "wheel"
        else:
            raise ValueError(f"Unsupported interactive event type '{event_type}'")

    event: InteractiveEventPayload = {"kind": kind, "type": event_type}

    key = raw_event.get("key")
    if isinstance(key, str):
        event["key"] = key
    code = raw_event.get("code")
    if isinstance(code, str):
        event["code"] = code

    button: int | None = _extract_int(raw_event, "button")
    if button is not None:
        event["button"] = button
    pointer_id: int | None = _extract_int(raw_event, "pointer_id", "pointerId")
    if pointer_id is not None:
        event["pointer_id"] = pointer_id

    client_x: float | None = _extract_float(raw_event, "client_x", "clientX")
    if client_x is not None:
        event["client_x"] = client_x
    client_y: float | None = _extract_float(raw_event, "client_y", "clientY")
    if client_y is not None:
        event["client_y"] = client_y

    start_x: float | None = _extract_float(raw_event, "start_x", "startX")
    if start_x is not None:
        event["start_x"] = start_x
    start_y: float | None = _extract_float(raw_event, "start_y", "startY")
    if start_y is not None:
        event["start_y"] = start_y

    delta_x: float | None = _extract_float(raw_event, "delta_x", "deltaX")
    if delta_x is not None:
        event["delta_x"] = delta_x
    delta_y: float | None = _extract_float(raw_event, "delta_y", "deltaY")
    if delta_y is not None:
        event["delta_y"] = delta_y

    return event


def _resolve_pointer_button_action(
    pointer_cfg: JsonPayload, event_type: str, button: int | None, action_count: int
) -> int | None:
    event_map = pointer_cfg.get("event_to_action", {})
    button_map = pointer_cfg.get("button_to_action", {})
    directional = pointer_cfg.get("directional", {})
    action: int | None = None

    if isinstance(event_map, Mapping):
        action = _coerce_action_candidate(event_map.get(event_type), action_count)
        if action is None and button is not None:
            action = _coerce_action_candidate(event_map.get(f"{event_type}:{button}"), action_count)

    if action is None and isinstance(button_map, Mapping) and button is not None:
        action = _coerce_action_candidate(button_map.get(str(button)), action_count)

    if action is None and isinstance(directional, Mapping):
        action = _coerce_action_candidate(directional.get("noop"), action_count)

    return action


def _resolve_pointer_swipe_action(
    session: _InteractiveSession, pointer_cfg: JsonPayload, event: JsonPayload, action_count: int
) -> int | None:
    pointer_id: int | None = _extract_int(event, "pointer_id")
    button: int | None = _extract_int(event, "button")
    start: tuple[float, float] | None = (
        session.pointer_anchors.pop(pointer_id, None) if pointer_id is not None else None
    )

    delta_x: float | None = _extract_float(event, "delta_x")
    delta_y: float | None = _extract_float(event, "delta_y")
    if delta_x is None or delta_y is None:
        client_x: float | None = _extract_float(event, "client_x")
        client_y: float | None = _extract_float(event, "client_y")
        start_x: float | None = _extract_float(event, "start_x")
        start_y: float | None = _extract_float(event, "start_y")
        if start is not None:
            start_x = start[0]
            start_y = start[1]
        if client_x is not None and client_y is not None and start_x is not None and start_y is not None:
            delta_x = client_x - start_x
            delta_y = client_y - start_y

    if delta_x is None or delta_y is None:
        return _resolve_pointer_button_action(pointer_cfg, "pointerup", button, action_count)

    swipe_threshold_raw = pointer_cfg.get("swipe_threshold", _SWIPE_THRESHOLD_DEFAULT)
    swipe_threshold: float = (
        float(swipe_threshold_raw) if isinstance(swipe_threshold_raw, (int, float)) else _SWIPE_THRESHOLD_DEFAULT
    )
    if swipe_threshold < 0:
        swipe_threshold = _SWIPE_THRESHOLD_DEFAULT

    if abs(delta_x) < swipe_threshold and abs(delta_y) < swipe_threshold:
        return _resolve_pointer_button_action(pointer_cfg, "pointerup", button, action_count)

    directional = pointer_cfg.get("directional", {})
    if not isinstance(directional, Mapping):
        return None

    direction: str
    if abs(delta_x) >= abs(delta_y):
        direction = "right" if delta_x >= 0 else "left"
    else:
        direction = "down" if delta_y >= 0 else "up"

    return _coerce_action_candidate(directional.get(direction), action_count)


def _resolve_wheel_action(wheel_cfg: JsonPayload, event: JsonPayload, action_count: int) -> int | None:
    delta_x: float | None = _extract_float(event, "delta_x")
    delta_y: float | None = _extract_float(event, "delta_y")
    if delta_x is None:
        delta_x = 0.0
    if delta_y is None:
        delta_y = 0.0
    if not delta_x and not delta_y:
        return None

    vertical_raw = wheel_cfg.get("vertical", {})
    horizontal_raw = wheel_cfg.get("horizontal", {})
    vertical = vertical_raw if isinstance(vertical_raw, Mapping) else {}
    horizontal = horizontal_raw if isinstance(horizontal_raw, Mapping) else {}

    if abs(delta_y) >= abs(delta_x):
        key = "positive" if delta_y > 0 else "negative"

        return _coerce_action_candidate(vertical.get(key), action_count)

    key = "positive" if delta_x > 0 else "negative"

    return _coerce_action_candidate(horizontal.get(key), action_count)


def _resolve_actions_for_event(
    session: _InteractiveSession, event: JsonPayload, action_count: int, bindings: JsonPayload
) -> list[int]:
    if action_count <= 0:
        return []

    kind: str = _normalize_binding_key(str(event.get("kind", "")))
    event_type: str = _normalize_binding_key(str(event.get("type", "")))
    if not kind or not event_type:
        return []

    if kind == "keyboard":
        keyboard = _json_mapping_or_empty(bindings.get("keyboard"))
        if not bool(keyboard.get("enabled", True)):
            return []

        events_raw = keyboard.get("events", [])
        if isinstance(events_raw, Sequence) and not isinstance(events_raw, (str, bytes)):
            allowed_events = {_normalize_binding_key(str(item)) for item in events_raw}
            if allowed_events and event_type not in allowed_events:
                return []

        key_map = _json_mapping_or_empty(keyboard.get("key_to_action"))
        key: str = _normalize_binding_key(str(event.get("key", "")))
        code: str = _normalize_binding_key(str(event.get("code", "")))
        action: int | None = _coerce_action_candidate(key_map.get(key), action_count)
        if action is None:
            action = _coerce_action_candidate(key_map.get(code), action_count)

        return [action] if action is not None else []

    if kind == "pointer":
        pointer = _json_mapping_or_empty(bindings.get("pointer"))
        if not bool(pointer.get("enabled", True)):
            return []

        events_raw = pointer.get("events", [])
        if isinstance(events_raw, Sequence) and not isinstance(events_raw, (str, bytes)):
            allowed_events = {_normalize_binding_key(str(item)) for item in events_raw}
            if allowed_events and event_type not in allowed_events:
                return []

        pointer_id: int | None = _extract_int(event, "pointer_id")
        client_x: float | None = _extract_float(event, "client_x")
        client_y: float | None = _extract_float(event, "client_y")
        button: int | None = _extract_int(event, "button")

        if event_type in {"pointerdown", "mousedown"}:
            if pointer_id is not None and client_x is not None and client_y is not None:
                session.pointer_anchors[pointer_id] = (client_x, client_y)
            action = _resolve_pointer_button_action(pointer, event_type, button, action_count)

            return [action] if action is not None else []

        if event_type in {"pointercancel", "pointerleave", "mouseleave"}:
            if pointer_id is not None:
                session.pointer_anchors.pop(pointer_id, None)

            return []

        if event_type in {"pointerup", "mouseup"}:
            action = _resolve_pointer_swipe_action(session, pointer, event, action_count)

            return [action] if action is not None else []

        if event_type in {"click", "auxclick", "dblclick", "contextmenu"}:
            action = _resolve_pointer_button_action(pointer, event_type, button, action_count)

            return [action] if action is not None else []

        return []

    if kind == "wheel":
        wheel = _json_mapping_or_empty(bindings.get("wheel"))
        if not bool(wheel.get("enabled", True)):
            return []

        events_raw = wheel.get("events", [])
        if isinstance(events_raw, Sequence) and not isinstance(events_raw, (str, bytes)):
            allowed_events = {_normalize_binding_key(str(item)) for item in events_raw}
            if allowed_events and event_type not in allowed_events:
                return []

        action = _resolve_wheel_action(wheel, event, action_count)

        return [action] if action is not None else []

    return []


def _apply_action(session: _InteractiveSession, action: int) -> None:
    next_states, _ = session.env.next_state([session.state], [action])
    if not next_states:
        raise RuntimeError("Environment returned no next state")
    session.state = next_states[0]


def _render_interactive_state(
    session: _InteractiveSession, effects_store: JsonPayload, renderer: JsonPayload
) -> tuple[str, float]:
    effect_specs: EffectSpecsMapping = get_effect_specs_for_environment(session.env_name)
    selection: dict[str, list[ActiveEffect]] = _build_selection(effect_specs, effects_store)
    start: float = perf_counter()
    image: str = render_state_to_uri(session.env_name, session.env, session.state, selection, dict(renderer))
    render_ms: float = round((perf_counter() - start) * 1000.0, 2)

    return image, render_ms


def _coerce_session_id(payload: JsonPayload) -> str:
    raw = payload.get("session_id")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("'session_id' must be a non-empty string")

    return raw.strip()


def _coerce_action(payload: JsonPayload) -> int:
    raw = payload.get("action")
    if raw is None:
        raise ValueError("'action' is required")
    if isinstance(raw, bool):
        raise TypeError("'action' must be an integer")
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError("'action' must be an integer")
        value = int(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("'action' must be a non-empty integer string")
        value = int(text)
    else:
        raise TypeError("'action' must be an integer")

    return value


def _default_environment() -> str:
    options: list[dict[str, str]] = list_environment_options()

    return options[0]["value"] if options else ""


def _initial_effect_store(effect_specs: EffectSpecsMapping) -> dict[str, dict[str, EffectEntry]]:
    store: dict[str, dict[str, EffectEntry]] = {}
    for stage, effects in effect_specs.items():
        stage_entries: dict[str, EffectEntry] = {}
        for effect in effects:
            effect_name: str = effect["name"]
            params: Sequence[ParameterSpec] = effect.get("parameters", [])
            defaults: JsonObject = {}
            for param in params:
                param_name: str = param["name"]
                defaults[param_name] = param.get("default")
            stage_entries[effect_name] = {"enabled": False, "params": defaults}
        store[stage] = stage_entries

    return store


def _build_selection(effect_specs: EffectSpecsMapping, effects_store: JsonPayload) -> dict[str, list[ActiveEffect]]:
    selection: dict[str, list[ActiveEffect]] = {}
    for stage in STAGE_ORDER:
        stage_specs = effect_specs.get(stage, [])
        stage_state = _json_mapping_or_empty(effects_store.get(stage))
        active: list[ActiveEffect] = []
        for spec in stage_specs:
            effect_name = spec["name"]
            effect_state_raw = stage_state.get(effect_name, {})
            effect_state = _json_mapping_or_empty(effect_state_raw)
            enabled = bool(effect_state.get("enabled", False))
            if not enabled:
                continue
            params_raw = effect_state.get("params", {})
            params = dict(params_raw) if isinstance(params_raw, Mapping) else {}
            active.append({"name": effect_name, "enabled": True, "params": params})
        selection[stage] = active

    return selection


def _payload_key(payload: JsonPayload) -> str:
    raw: str = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    return hashlib.blake2b(raw.encode("utf-8"), digest_size=20).hexdigest()


def _frontend_dist_default() -> Path:
    """Resolve the packaged frontend bundle directory.

    Uses :mod:`importlib.resources` so the compiled single-page app is found both in
    editable installs and in wheels installed from PyPI, where the old source-tree
    sibling layout (``spar-datadash-react/dist``) no longer exists.

    Returns:
        Path: Directory expected to contain the built ``index.html`` and assets.
    """
    return Path(str(resources.files("spar_datadash").joinpath("_frontend")))


def _coerce_env(payload: JsonPayload) -> str:
    raw_env = payload.get("env")
    env_name: str = str(raw_env).strip().lower() if raw_env is not None else _default_environment()
    valid_envs: set[str] = {entry["value"] for entry in list_environment_options()}
    if env_name not in valid_envs:
        raise ValueError(f"Unsupported environment '{env_name}'")

    return env_name


def create_react_dashboard_app(frontend_dist: Path | None = None) -> Flask:
    """Create the Flask app that serves both API routes and built React assets."""
    resolved_dist: Path = (frontend_dist or _frontend_dist_default()).resolve()
    has_frontend_bundle: bool = (resolved_dist / "index.html").exists()
    render_cache = _RenderCache()
    interactive_sessions = _InteractiveSessionStore()

    app = Flask(__name__)

    def _render_response(payload: JsonPayload) -> tuple[Response, int] | Response:
        env_name: str = _coerce_env(payload)
        state_raw = payload.get("state")
        effects_raw = payload.get("effects", {})
        renderer_raw = payload.get("renderer", {})
        if not isinstance(state_raw, Mapping):
            return jsonify({"error": "'state' must be an object"}), 400

        if not isinstance(effects_raw, Mapping):
            return jsonify({"error": "'effects' must be an object"}), 400

        if not isinstance(renderer_raw, Mapping):
            return jsonify({"error": "'renderer' must be an object"}), 400

        state_payload = dict(state_raw)
        effects_store = dict(effects_raw)
        renderer = dict(renderer_raw)

        key_payload: JsonObject = {
            "env": env_name,
            "state": state_payload,
            "effects": effects_store,
            "renderer": renderer,
        }
        key: str = _payload_key(key_payload)
        cached: _CacheEntry | None = render_cache.get(key)
        if cached is not None:
            cached_result: RenderResult = {"image": cached.image, "render_ms": cached.render_ms, "cached": True}

            return jsonify(cached_result)

        effect_specs: EffectSpecsMapping = get_effect_specs_for_environment(env_name)
        selection = _build_selection(effect_specs, effects_store)

        start: float = perf_counter()
        image: str = render_environment_to_uri(env_name, state_payload, selection, renderer)
        render_ms: float = round((perf_counter() - start) * 1000.0, 2)

        render_cache.set(key, _CacheEntry(image=image, render_ms=render_ms))
        result: RenderResult = {"image": image, "render_ms": render_ms, "cached": False}

        return jsonify(result)

    def _sweep_presets_response(payload: JsonPayload) -> tuple[Response, int] | Response:
        env_name: str = _coerce_env(payload)
        presets = get_gen_data_effect_presets(env_name)

        return jsonify({"env": env_name, "presets": list(presets)})

    def _config_list_response(payload: JsonPayload) -> tuple[Response, int] | Response:
        env_name: str = _coerce_env(payload)
        files = list_env_config_files(env_name)

        return jsonify({"env": env_name, "files": list(files)})

    def _config_parse_response(payload: JsonPayload) -> tuple[Response, int] | Response:
        env_name: str = _coerce_env(payload)
        token_raw = payload.get("token")
        content_raw = payload.get("content")
        if token_raw is not None:
            if not isinstance(token_raw, str):
                return jsonify({"error": "'token' must be a string"}), 400

            text: str = read_env_config_text(env_name, token_raw)
            source: str = token_raw
        elif content_raw is not None:
            if not isinstance(content_raw, str):
                return jsonify({"error": "'content' must be a string"}), 400

            text = content_raw
            source = "upload"
        else:
            return jsonify({"error": "Provide either 'token' or 'content'"}), 400

        presets = parse_effect_presets_from_text(env_name, text)

        return jsonify({"env": env_name, "source": source, "presets": list(presets)})

    def _render_sweep_cell(
        env_name: str,
        state_payload: JsonPayload,
        renderer: JsonPayload,
        effect_specs: EffectSpecsMapping,
        cell: JsonPayload,
    ) -> SweepCellResult:
        label_raw = cell.get("label")
        label: str = str(label_raw) if isinstance(label_raw, (str, int, float)) else ""
        effects_raw = cell.get("effects", {})
        effects_store: JsonPayload = effects_raw if isinstance(effects_raw, Mapping) else {}

        key_payload: JsonObject = {
            "env": env_name,
            "state": dict(state_payload),
            "effects": dict(effects_store),
            "renderer": dict(renderer),
        }
        cache_key: str = _payload_key(key_payload)
        cached_entry: _CacheEntry | None = render_cache.get(cache_key)
        if cached_entry is not None:
            return {"label": label, "image": cached_entry.image, "render_ms": cached_entry.render_ms, "cached": True}

        selection = _build_selection(effect_specs, effects_store)
        start: float = perf_counter()
        image: str = render_environment_to_uri(env_name, state_payload, selection, dict(renderer))
        render_ms: float = round((perf_counter() - start) * 1000.0, 2)
        render_cache.set(cache_key, _CacheEntry(image=image, render_ms=render_ms))

        return {"label": label, "image": image, "render_ms": render_ms, "cached": False}

    def _sweep_render_response(payload: JsonPayload) -> tuple[Response, int] | Response:
        env_name = _coerce_env(payload)
        state_raw = payload.get("state")
        renderer_raw = payload.get("renderer", {})
        cells_raw = payload.get("cells")
        if not isinstance(state_raw, Mapping):
            return jsonify({"error": "'state' must be an object"}), 400

        if not isinstance(renderer_raw, Mapping):
            return jsonify({"error": "'renderer' must be an object"}), 400

        if not isinstance(cells_raw, Sequence) or isinstance(cells_raw, (str, bytes)):
            return jsonify({"error": "'cells' must be a list"}), 400

        if len(cells_raw) > _MAX_SWEEP_CELLS:
            return jsonify({"error": f"'cells' exceeds the maximum of {_MAX_SWEEP_CELLS}"}), 400

        effect_specs: EffectSpecsMapping = get_effect_specs_for_environment(env_name)
        results: list[SweepCellResult] = []
        for cell in cells_raw:
            if not isinstance(cell, Mapping):
                return jsonify({"error": "each sweep cell must be an object"}), 400
            results.append(_render_sweep_cell(env_name, state_raw, renderer_raw, effect_specs, cell))

        return jsonify({"env": env_name, "cells": results})

    def _interactive_start_response(payload: JsonPayload) -> tuple[Response, int] | Response:
        env_name: str = _coerce_env(payload)
        state_raw = payload.get("state")
        effects_raw = payload.get("effects", {})
        renderer_raw = payload.get("renderer", {})
        if state_raw is not None and not isinstance(state_raw, Mapping):
            return jsonify({"error": "'state' must be an object when provided"}), 400

        if not isinstance(effects_raw, Mapping):
            return jsonify({"error": "'effects' must be an object"}), 400

        if not isinstance(renderer_raw, Mapping):
            return jsonify({"error": "'renderer' must be an object"}), 400

        env_class: type[ABCEnvironment[ABCState]] = get_environment_class(env_name)
        env: ABCEnvironment[ABCState] = env_class()
        state: ABCState = (
            deserialize_state(env_name, dict(state_raw))
            if isinstance(state_raw, Mapping)
            else _generate_start_state_for_env_instance(env)
        )
        session: _InteractiveSession = interactive_sessions.create(env_name=env_name, env=env, state=state)

        action_count, action_labels, interactive_bindings = _resolve_interactive_metadata(env_name, env)
        image, render_ms = _render_interactive_state(session, dict(effects_raw), dict(renderer_raw))
        result: InteractiveStartResult = {
            "session_id": session.session_id,
            "env": env_name,
            "state": serialize_state(env_name, session.state),
            "action_count": action_count,
            "action_labels": action_labels,
            "interactive_bindings": interactive_bindings,
            "image": image,
            "render_ms": render_ms,
        }

        return jsonify(result)

    def _interactive_render_response(payload: JsonPayload) -> tuple[Response, int] | Response:
        session_id = _coerce_session_id(payload)
        effects_raw = payload.get("effects", {})
        renderer_raw = payload.get("renderer", {})
        if not isinstance(effects_raw, Mapping):
            return jsonify({"error": "'effects' must be an object"}), 400

        if not isinstance(renderer_raw, Mapping):
            return jsonify({"error": "'renderer' must be an object"}), 400

        session: _InteractiveSession | None = interactive_sessions.get(session_id)
        if session is None:
            return jsonify({"error": "Interactive session not found"}), 404

        with session.lock:
            image, render_ms = _render_interactive_state(session, dict(effects_raw), dict(renderer_raw))
            action_count, action_labels, interactive_bindings = _resolve_interactive_metadata(
                session.env_name, session.env
            )

            return jsonify({
                "session_id": session.session_id,
                "env": session.env_name,
                "state": serialize_state(session.env_name, session.state),
                "action_count": action_count,
                "action_labels": action_labels,
                "interactive_bindings": interactive_bindings,
                "image": image,
                "render_ms": render_ms,
            })

    def _interactive_step_response(payload: JsonPayload) -> tuple[Response, int] | Response:
        session_id = _coerce_session_id(payload)
        action = _coerce_action(payload)
        effects_raw = payload.get("effects", {})
        renderer_raw = payload.get("renderer", {})
        if not isinstance(effects_raw, Mapping):
            return jsonify({"error": "'effects' must be an object"}), 400

        if not isinstance(renderer_raw, Mapping):
            return jsonify({"error": "'renderer' must be an object"}), 400

        session: _InteractiveSession | None = interactive_sessions.get(session_id)
        if session is None:
            return jsonify({"error": "Interactive session not found"}), 404

        action_count, action_labels, interactive_bindings = _resolve_interactive_metadata(session.env_name, session.env)
        if action_count <= 0:
            return jsonify({"error": "Interactive stepping is not supported for this environment"}), 400

        if action < 0 or action >= action_count:
            return jsonify({"error": f"'action' must be in [0, {action_count - 1}]"}), 400

        with session.lock:
            _apply_action(session, action)
            image, render_ms = _render_interactive_state(session, dict(effects_raw), dict(renderer_raw))
            result: InteractiveStepResult = {
                "session_id": session.session_id,
                "env": session.env_name,
                "state": serialize_state(session.env_name, session.state),
                "action_applied": action,
                "action_count": action_count,
                "action_labels": action_labels,
                "interactive_bindings": interactive_bindings,
                "image": image,
                "render_ms": render_ms,
            }

            return jsonify(result)

    def _interactive_event_response(payload: JsonPayload) -> tuple[Response, int] | Response:
        session_id = _coerce_session_id(payload)
        event = _coerce_interactive_event(payload)
        effects_raw = payload.get("effects", {})
        renderer_raw = payload.get("renderer", {})
        if not isinstance(effects_raw, Mapping):
            return jsonify({"error": "'effects' must be an object"}), 400

        if not isinstance(renderer_raw, Mapping):
            return jsonify({"error": "'renderer' must be an object"}), 400

        session: _InteractiveSession | None = interactive_sessions.get(session_id)
        if session is None:
            return jsonify({"error": "Interactive session not found"}), 404

        with session.lock:
            action_count, action_labels, interactive_bindings = _resolve_interactive_metadata(
                session.env_name, session.env
            )
            actions = _resolve_actions_for_event(session, event, action_count, interactive_bindings)
            if not actions:
                skipped_result: InteractiveEventResult = {
                    "session_id": session.session_id,
                    "env": session.env_name,
                    "state": serialize_state(session.env_name, session.state),
                    "action_count": action_count,
                    "action_labels": action_labels,
                    "interactive_bindings": interactive_bindings,
                    "actions_applied": [],
                    "handled": False,
                }

                return jsonify(skipped_result)

            applied: list[int] = []
            for action in actions:
                _apply_action(session, action)
                applied.append(action)

            image, render_ms = _render_interactive_state(session, dict(effects_raw), dict(renderer_raw))
            result: InteractiveEventResult = {
                "session_id": session.session_id,
                "env": session.env_name,
                "state": serialize_state(session.env_name, session.state),
                "action_count": action_count,
                "action_labels": action_labels,
                "interactive_bindings": interactive_bindings,
                "actions_applied": applied,
                "handled": True,
                "image": image,
                "render_ms": render_ms,
            }

            return jsonify(result)

    def _interactive_reset_response(payload: JsonPayload) -> tuple[Response, int] | Response:
        session_id = _coerce_session_id(payload)
        effects_raw = payload.get("effects", {})
        renderer_raw = payload.get("renderer", {})
        if not isinstance(effects_raw, Mapping):
            return jsonify({"error": "'effects' must be an object"}), 400

        if not isinstance(renderer_raw, Mapping):
            return jsonify({"error": "'renderer' must be an object"}), 400

        session: _InteractiveSession | None = interactive_sessions.get(session_id)
        if session is None:
            return jsonify({"error": "Interactive session not found"}), 404

        with session.lock:
            session.state = _generate_start_state_for_env_instance(session.env)
            image, render_ms = _render_interactive_state(session, dict(effects_raw), dict(renderer_raw))
            action_count, action_labels, interactive_bindings = _resolve_interactive_metadata(
                session.env_name, session.env
            )

            return jsonify({
                "session_id": session.session_id,
                "env": session.env_name,
                "state": serialize_state(session.env_name, session.state),
                "action_count": action_count,
                "action_labels": action_labels,
                "interactive_bindings": interactive_bindings,
                "image": image,
                "render_ms": render_ms,
            })

    @app.get("/api/health")
    def api_health() -> Response:

        return jsonify({"status": "ok"})

    @app.get("/api/environments")
    def api_environments() -> Response:

        return jsonify({"environments": list_environment_options(), "default_env": _default_environment()})

    @app.post("/api/bootstrap")
    def api_bootstrap() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)
        try:
            env_name: str = _coerce_env(payload)
            effect_specs: EffectSpecsMapping = get_effect_specs_for_environment(env_name)

            return jsonify({
                "env": env_name,
                "effect_specs": effect_specs,
                "effects_store": _initial_effect_store(effect_specs),
                "renderer": get_environment_default_renderer_settings(env_name),
                "state": generate_start_state(env_name),
            })
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to bootstrap environment")

            return jsonify({"error": "Failed to bootstrap environment"}), 500

    @app.post("/api/randomize")
    def api_randomize() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)
        try:
            env_name: str = _coerce_env(payload)

            return jsonify({"env": env_name, "state": generate_start_state(env_name)})

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to randomize state")

            return jsonify({"error": "Failed to randomize state"}), 500

    @app.post("/api/render")
    def api_render() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)

        try:
            return _render_response(payload)

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to render image")

            return jsonify({"error": "Failed to render image"}), 500

    @app.post("/api/sweep/presets")
    def api_sweep_presets() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)

        try:
            return _sweep_presets_response(payload)

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to load sweep presets")

            return jsonify({"error": "Failed to load sweep presets"}), 500

    @app.post("/api/config/list")
    def api_config_list() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)

        try:
            return _config_list_response(payload)

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to list config files")

            return jsonify({"error": "Failed to list config files"}), 500

    @app.post("/api/config/parse")
    def api_config_parse() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)

        try:
            return _config_parse_response(payload)

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to parse config")

            return jsonify({"error": "Failed to parse config"}), 500

    @app.post("/api/sweep/render")
    def api_sweep_render() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)

        try:
            return _sweep_render_response(payload)

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to render sweep grid")

            return jsonify({"error": "Failed to render sweep grid"}), 500

    @app.post("/api/interactive/start")
    def api_interactive_start() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)

        try:
            return _interactive_start_response(payload)

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to start interactive session")

            return jsonify({"error": "Failed to start interactive session"}), 500

    @app.post("/api/interactive/render")
    def api_interactive_render() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)

        try:
            return _interactive_render_response(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to render interactive session")

            return jsonify({"error": "Failed to render interactive session"}), 500

    @app.post("/api/interactive/step")
    def api_interactive_step() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)

        try:
            return _interactive_step_response(payload)

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to step interactive session")

            return jsonify({"error": "Failed to step interactive session"}), 500

    @app.post("/api/interactive/event")
    def api_interactive_event() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)

        try:
            return _interactive_event_response(payload)

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to process interactive event")

            return jsonify({"error": "Failed to process interactive event"}), 500

    @app.post("/api/interactive/reset")
    def api_interactive_reset() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)

        try:
            return _interactive_reset_response(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to reset interactive session")

            return jsonify({"error": "Failed to reset interactive session"}), 500

    @app.post("/api/interactive/stop")
    def api_interactive_stop() -> tuple[Response, int] | Response:
        payload_raw = request.get_json(silent=True)
        payload = _to_json_payload(payload_raw)
        try:
            session_id: str = _coerce_session_id(payload)
            interactive_sessions.remove(session_id)

            return jsonify({"stopped": True})

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        except Exception:
            logger.exception("Failed to stop interactive session")

            return jsonify({"error": "Failed to stop interactive session"}), 500

    @app.get("/", defaults={"asset_path": ""})
    @app.get("/<path:asset_path>")
    def serve_frontend(asset_path: str) -> tuple[Response, int] | Response:
        if asset_path.startswith("api/"):
            return jsonify({"error": "Not found"}), 404

        if not has_frontend_bundle:
            return (
                jsonify({
                    "error": "Frontend build not found",
                    "hint": (
                        "Build the dashboard UI with "
                        "'pixi run -e datadashboard spar-datadash-react-build' (runs 'pnpm build') and restart."
                    ),
                    "expected_index": str(resolved_dist / "index.html"),
                }),
                503,
            )
        if asset_path:
            candidate: Path = resolved_dist / asset_path
            if candidate.is_file():
                return send_from_directory(resolved_dist, asset_path)

        return send_from_directory(resolved_dist, "index.html")

    for route_handler in (
        api_health,
        api_environments,
        api_bootstrap,
        api_randomize,
        api_render,
        api_sweep_presets,
        api_config_list,
        api_config_parse,
        api_sweep_render,
        api_interactive_start,
        api_interactive_render,
        api_interactive_step,
        api_interactive_event,
        api_interactive_reset,
        api_interactive_stop,
        serve_frontend,
    ):
        app.view_functions.setdefault(route_handler.__name__, route_handler)

    return app


def run_react_dashboard(
    host: str = "127.0.0.1", port: int = 8060, debug: bool = False, frontend_dist: Path | None = None
) -> None:
    """Run the Flask API/static server for the React dashboard."""
    app: Flask = create_react_dashboard_app(frontend_dist=frontend_dist)
    logger.info(f"Starting SPAR React dashboard API on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
