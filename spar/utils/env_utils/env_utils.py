"""Helpers for locating environment implementations without eager imports."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from logging import getLogger
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

from spar.environments.abstracts.environment import ABCEnvironment

if TYPE_CHECKING:
    from collections.abc import Mapping
    from logging import Logger
    from types import ModuleType
    from typing import TypeGuard

    from spar.environments import ABCState


logger: Logger = getLogger(__name__)

__all__: list[str] = ["get_environment", "get_environment_class", "get_environment_classes", "list_environment_names"]


class _EnvironmentSpec(NamedTuple):
    """Descriptor for an environment import target."""

    module: str
    attribute: str


def _is_env_class_candidate(alias: str, module_ref: str, attribute: str) -> bool:
    """Heuristically determine whether an import target is an environment class."""
    if alias != attribute:
        return False

    if not attribute or not attribute[0].isupper():
        return False

    if module_ref.startswith(".abstracts"):
        return False

    if attribute.startswith("ABC"):
        return False

    if attribute.endswith("State"):
        return False

    disallowed_tokens: tuple[str, ...] = ("Model", "Encoder", "Decoder", "DQN", "Triples")
    # Strip an "Env" suffix when present. Keep names such as Cube3 unchanged.

    return not any(token in attribute for token in disallowed_tokens)


def _is_environment_type(candidate: type) -> TypeGuard[type[ABCEnvironment[ABCState]]]:
    """Return whether a class object implements the SPAR environment contract.

    Args:
        candidate: Class object to validate.

    Returns:
        Whether `candidate` subclasses :class:`ABCEnvironment`.
    """
    return issubclass(candidate, ABCEnvironment)


def _discover_environment_specs() -> dict[str, _EnvironmentSpec]:
    """Build an environment mapping from ``spar.environments.IMPORTS``."""
    environments_pkg: ModuleType = import_module("spar.environments")
    imports_map = getattr(environments_pkg, "IMPORTS", {})
    if not isinstance(imports_map, dict):
        return {}

    candidates: dict[str, list[tuple[int, _EnvironmentSpec]]] = {}
    for alias, value in imports_map.items():
        if not isinstance(alias, str):
            continue
        if not isinstance(value, tuple) or len(value) != 2:
            continue
        module_ref, attribute = value
        if not isinstance(module_ref, str) or not isinstance(attribute, str):
            continue
        if not _is_env_class_candidate(alias, module_ref, attribute):
            continue

        module_root: str = module_ref.lstrip(".").split(".", 1)[0].strip().lower()
        if not module_root:
            continue

        module_name: str = f"{environments_pkg.__name__}{module_ref}" if module_ref.startswith(".") else module_ref
        spec = _EnvironmentSpec(module=module_name, attribute=attribute)
        candidates.setdefault(module_root, []).append((0 if attribute.endswith("Env") else 1, spec))

    resolved: dict[str, _EnvironmentSpec] = {}
    env_name: str
    options: list[tuple[int, _EnvironmentSpec]]
    for env_name, options in sorted(candidates.items()):

        def _environment_option_key(item: tuple[int, _EnvironmentSpec]) -> tuple[int, str]:
            return item[0], item[1].attribute

        options.sort(key=_environment_option_key)
        resolved[env_name] = options[0][1]

    return resolved


@lru_cache(maxsize=1)
def _environment_mapping() -> Mapping[str, _EnvironmentSpec]:

    return MappingProxyType(_discover_environment_specs())


_env_class_cache: dict[str, type[ABCEnvironment[ABCState]]] = {}


def _normalise_key(env_name: str) -> str:
    key: str = env_name.strip().lower()
    if not key:
        raise ValueError("Environment name must be a non-empty string")

    return key


def list_environment_names() -> list[str]:
    """Return known environment names without importing environment modules."""
    return sorted(_environment_mapping())


def get_environment_class(env_name: str) -> type[ABCEnvironment[ABCState]]:
    """Return the lazily imported environment class referenced by ``env_name``."""
    key: str = _normalise_key(env_name)

    cached: type[ABCEnvironment[ABCState]] | None = _env_class_cache.get(key)
    if cached is not None:
        return cached

    spec: _EnvironmentSpec
    available: str
    env_mapping = _environment_mapping()
    try:
        spec = env_mapping[key]
    except KeyError as exc:
        available = ", ".join(list_environment_names())
        raise ValueError(f"Environment '{env_name}' not found. Available environments: {available}") from exc

    module: ModuleType = import_module(spec.module)
    try:
        env_attr = getattr(module, spec.attribute)
    except AttributeError as exc:
        raise ImportError(
            f"Failed to import environment '{env_name}': '{module.__name__}' has no attribute '{spec.attribute}'."
        ) from exc

    if not isinstance(env_attr, type):
        raise TypeError(f"Expected '{spec.attribute}' from '{spec.module}' to be a class, got {type(env_attr)!r}")

    if not _is_environment_type(env_attr):
        raise TypeError(
            f"Expected '{spec.attribute}' from '{spec.module}' to subclass ABCEnvironment, got {env_attr!r}."
        )

    env_class: type[ABCEnvironment[ABCState]] = env_attr
    _env_class_cache[key] = env_class

    return env_class


def get_environment_classes() -> dict[str, type[ABCEnvironment[ABCState]]]:
    """Return all importable environment classes keyed by their canonical name."""
    classes: dict[str, type[ABCEnvironment[ABCState]]] = {}

    def _safe_get_environment_class(name: str) -> type[ABCEnvironment[ABCState]] | None:
        try:
            return get_environment_class(name)
        except ImportError as exc:
            logger.warning(f"Skipping environment '{name}': {exc}")
            return None

    for name in list_environment_names():
        env_cls = _safe_get_environment_class(name)
        if env_cls is not None:
            classes[name] = env_cls

    return classes


def get_environment(env_name: str) -> ABCEnvironment[ABCState]:
    """Instantiate and return the environment referenced by ``env_name``."""
    env_cls: type[ABCEnvironment[ABCState]] = get_environment_class(env_name)

    return env_cls()
