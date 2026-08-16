"""Path and templating helpers for SPAR Optuna workflows."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence
import re
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from .types import PathValue, TemplateValue


_PATH_TOKEN_RE: re.Pattern[str] = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")
_TEMPLATE_RE: re.Pattern[str] = re.compile(r"{{\s*([^}]+?)\s*}}")


class _MissingPathValue:
    """Sentinel for path lookups that distinguishes missing values from ``None``."""

    __slots__: tuple[str, ...] = ()


_MISSING: _MissingPathValue = _MissingPathValue()


def parse_path_tokens(path: str) -> list[str | int]:
    """Parse dotted/list-index config paths such as ``train.phases[0].lr``."""
    tokens: list[str | int] = []
    for match in _PATH_TOKEN_RE.finditer(path):
        key_token: str | None = match.group(1)
        index_token: str | None = match.group(2)
        if key_token is not None:
            tokens.append(key_token)
        elif index_token is not None:
            tokens.append(int(index_token))
    if not tokens:
        raise ValueError(f"Invalid config path: {path!r}")
    return tokens


def scoped_step_path(path: str, default_step: str | None) -> tuple[str | None, str]:
    """Split a ``step:path`` reference into ``(step, path)``."""
    if ":" in path:
        step_name, raw_path = path.split(":", 1)
        return step_name, raw_path
    return default_step, path


def _next_path_value(current: PathValue | Mapping[str, PathValue], token: str | int) -> tuple[bool, PathValue]:
    """Read a single path token from a nested mapping or sequence."""
    if isinstance(token, str) and isinstance(current, Mapping):
        if token in current:
            return True, current[token]
        return False, None
    if isinstance(token, int) and (type(current) is list or type(current) is tuple):
        if 0 <= token < len(current):
            return True, current[token]
        return False, None
    return False, None


@overload
def get_path_value(
    data: PathValue | Mapping[str, PathValue], path: str, *, default: _MissingPathValue
) -> PathValue | _MissingPathValue: ...


@overload
def get_path_value(
    data: PathValue | Mapping[str, PathValue], path: str, *, default: PathValue | None = None
) -> PathValue | None: ...


def get_path_value(
    data: PathValue | Mapping[str, PathValue], path: str, *, default: PathValue | _MissingPathValue | None = None
) -> PathValue | _MissingPathValue | None:
    """Read a nested value from mappings/lists/DictConfigs using SPAR path syntax."""
    current: PathValue = dict(data.items()) if isinstance(data, Mapping) else data
    for token in parse_path_tokens(path):
        found, next_value = _next_path_value(current, token)
        if not found:
            return default
        current = next_value
    return current


def set_path_value(data: PathValue | MutableMapping[str, PathValue], path: str, value: PathValue) -> None:
    """Set a nested value on mappings/lists/DictConfigs using SPAR path syntax."""
    tokens: list[str | int] = parse_path_tokens(path)
    current: PathValue | MutableMapping[str, PathValue] = data
    for token in tokens[:-1]:
        found, next_value = _next_path_value(current, token)
        if not found:
            raise KeyError(path)
        current = next_value

    leaf: str | int = tokens[-1]
    if isinstance(leaf, str) and isinstance(current, MutableMapping):
        current[leaf] = value
        return
    if isinstance(leaf, int) and isinstance(current, MutableSequence):
        current[leaf] = value
        return
    raise KeyError(path)


def set_path_value_if_present(data: PathValue | MutableMapping[str, PathValue], path: str, value: PathValue) -> bool:
    """Set a nested value only when the full path already exists."""
    if get_path_value(data, path, default=_MISSING) is _MISSING:
        return False
    try:
        set_path_value(data, path, value)
    except (KeyError, IndexError):
        return False
    return True


def sanitize_param_name(name: str) -> str:
    """Create a stable Optuna parameter name from a config path."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def resolve_context_value(context: Mapping[str, TemplateValue], expr: str) -> TemplateValue:
    """Resolve a dotted expression from a nested template context."""
    current: TemplateValue = context
    for token in expr.split("."):
        if isinstance(current, Mapping):
            if token not in current:
                raise KeyError(expr)
            current = current[token]
            continue
        if isinstance(current, list):
            current = current[int(token)]
            continue
        raise KeyError(expr)
    return current


def render_template_string(template: str, context: Mapping[str, TemplateValue]) -> str:
    """Render ``{{ dotted.path }}`` expressions inside a string."""

    def _replace(match: re.Match[str]) -> str:
        expr: str = match.group(1).strip()
        value: TemplateValue = resolve_context_value(context, expr)
        return str(value)

    return _TEMPLATE_RE.sub(_replace, template)


def render_templates(value: PathValue | Mapping[str, PathValue], context: Mapping[str, TemplateValue]) -> PathValue:
    """Recursively render template strings inside arbitrarily nested values."""
    if isinstance(value, str):
        return render_template_string(value, context) if "{{" in value else value
    if isinstance(value, list):
        return [render_templates(item, context) for item in value]
    if isinstance(value, tuple):
        return tuple(render_templates(item, context) for item in value)
    if isinstance(value, Mapping):
        return {key: render_templates(item, context) for key, item in value.items()}
    return value
