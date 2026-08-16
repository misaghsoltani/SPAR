"""Parse ``gen_data`` effect presets into sweepable parameter descriptors.

The data configs under ``spar/configs/data/<env>.yaml`` describe the effect
variations available to the ``gen_data`` stage. Each variation is
either a single effect (leaf) or a combination of sub-effects, and its numeric or
categorical parameters are expressed with the ``${range:}``, ``${irange:}`` and
``${choice:}`` resolvers. This module resolves those presets and flattens them
into descriptors that the dashboard can sample and export back to ``gen_data``.

Preset decomposition uses the same leaf, combination, and ``enabled`` rules as
:func:`spar.utils.env_utils.effects_core.build_stage_pipelines`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from importlib import resources
from typing import TYPE_CHECKING, Required, TypedDict

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from spar.utils.config_utils.misc import register_omega_conf_resolvers
from spar.utils.config_utils.samplers import ChoiceSampler, IntRangeSampler, RangeSampler, sampler_from_spec

from .rich_logger import get_rich_logger
from .utils import get_effect_specs_for_environment

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable
    from typing import TypeAlias

    from omegaconf import ListConfig
    from typing_extensions import TypeIs

    from .rich_logger import RichLogger
    from .utils import EffectSpecsMapping

    ConfigScalar: TypeAlias = str | int | float | bool | None
    ConfigLeaf: TypeAlias = "ConfigScalar | Sequence[ConfigTree]"
    ConfigTree: TypeAlias = "Mapping[str, ConfigTree] | Sequence[ConfigTree] | ConfigScalar"


logger: RichLogger = get_rich_logger(__name__)

_ENABLE_KEY: str = "enabled"


class ParamDescriptor(TypedDict, total=False):
    """A single sweepable effect parameter resolved from a ``gen_data`` preset."""

    name: Required[str]
    label: Required[str]
    kind: Required[str]  # "range" | "irange" | "choice" | "fixed"
    low: float
    high: float
    options: list[str | int | float | bool | None]
    value: str | int | float | bool | list[str | int | float | bool | None] | None


class PresetEffect(TypedDict):
    """An atomic effect within a preset, tagged with its render stage."""

    name: str
    stage: str
    params: list[ParamDescriptor]


class EffectPreset(TypedDict):
    """A named ``gen_data`` variation flattened into stage-aware effects."""

    name: str
    is_leaf: bool
    effects: list[PresetEffect]


class ConfigFileInfo(TypedDict):
    """A packaged config file that carries a usable ``effects`` section."""

    token: str
    label: str
    preset_count: int


def _data_config_path(env_name: str) -> Traversable | None:
    """Return the packaged data-config resource for an environment, if present.

    Args:
        env_name: Canonical environment name (e.g. ``"cube3"``).

    Returns:
        A traversable resource pointing at ``spar/configs/data/<env>.yaml`` when it
        exists, otherwise ``None``.
    """
    candidate = resources.files("spar").joinpath("configs", "data", f"{env_name}.yaml")

    return candidate if candidate.is_file() else None


def _is_config_mapping(value: ConfigTree) -> TypeIs[Mapping[str, ConfigTree]]:
    """Return whether a resolved config node is a nested mapping."""
    return isinstance(value, Mapping)


def _label_for(name: str) -> str:
    """Humanize a snake_case identifier for display."""
    return name.replace("_", " ").title()


def _describe_param(name: str, value: ConfigLeaf) -> ParamDescriptor:
    """Classify one resolved parameter value as a range, choice, or fixed value.

    Args:
        name: Parameter name.
        value: Fully resolved value. Sampler-typed parameters arrive as encoded
            specification strings produced by the SPAR resolvers. Sequence values
            (for example ``size_range: [0.5, 2]``) are preserved as fixed lists.

    Returns:
        A :class:`ParamDescriptor` describing how the parameter can be swept.
    """
    label: str = _label_for(name)
    if isinstance(value, str):
        sampler = sampler_from_spec(value)
        if isinstance(sampler, IntRangeSampler):
            return {
                "name": name,
                "label": label,
                "kind": "irange",
                "low": float(sampler.low),
                "high": float(sampler.high),
            }

        if isinstance(sampler, RangeSampler):
            return {
                "name": name,
                "label": label,
                "kind": "range",
                "low": float(sampler.low),
                "high": float(sampler.high),
            }

        if isinstance(sampler, ChoiceSampler):
            options: list[str | int | float | bool | None] = [
                _coerce_choice_option(option) for option in sampler.options
            ]

            return {"name": name, "label": label, "kind": "choice", "options": options}

    if isinstance(value, Sequence) and not isinstance(value, str):
        return {"name": name, "label": label, "kind": "fixed", "value": [_coerce_choice_option(item) for item in value]}

    return {"name": name, "label": label, "kind": "fixed", "value": value}


def _coerce_choice_option(option: ConfigTree) -> str | int | float | bool | None:
    """Narrow a decoded choice option to a JSON-serializable scalar."""
    if option is None or isinstance(option, (str, int, float, bool)):
        return option

    return str(option)


def _effect_params(cfg_map: Mapping[str, ConfigTree]) -> list[ParamDescriptor]:
    """Build parameter descriptors for one atomic effect, skipping nested nodes."""
    params: list[ParamDescriptor] = []
    for key, value in cfg_map.items():
        if key == _ENABLE_KEY or _is_config_mapping(value):
            continue
        params.append(_describe_param(key, value))

    return params


def _preset_effects(
    name: str, cfg_map: Mapping[str, ConfigTree], stage_by_effect: Mapping[str, str]
) -> tuple[bool, list[PresetEffect]]:
    """Flatten a preset into supported atomic effects, mirroring ``build_stage_pipelines``.

    Args:
        name: Preset (variation) name.
        cfg_map: Resolved mapping for the preset.
        stage_by_effect: Map from supported effect name to its render stage.

    Returns:
        A tuple ``(is_leaf, effects)`` where ``effects`` lists only atomic effects
        supported by the target environment.
    """
    enabled_value = cfg_map.get(_ENABLE_KEY, True)
    is_enabled: bool = enabled_value if isinstance(enabled_value, bool) else True
    is_leaf: bool = is_enabled and not any(
        _is_config_mapping(value) for key, value in cfg_map.items() if key != _ENABLE_KEY
    )

    effects: list[PresetEffect] = []
    if is_leaf:
        stage = stage_by_effect.get(name)
        if stage is not None:
            effects.append({"name": name, "stage": stage, "params": _effect_params(cfg_map)})

        return True, effects

    if not is_enabled:
        return False, []

    for sub_name, sub_value in cfg_map.items():
        if sub_name == _ENABLE_KEY or not _is_config_mapping(sub_value):
            continue
        sub_map: Mapping[str, ConfigTree] = sub_value
        sub_enabled_value = sub_map.get(_ENABLE_KEY, True)
        sub_enabled: bool = sub_enabled_value if isinstance(sub_enabled_value, bool) else True
        stage = stage_by_effect.get(sub_name)
        if sub_enabled and stage is not None:
            effects.append({"name": sub_name, "stage": stage, "params": _effect_params(sub_map)})

    return False, effects


def _stage_by_effect(env_key: str) -> dict[str, str]:
    """Map each supported effect name to its render stage for an environment."""
    specs: EffectSpecsMapping = get_effect_specs_for_environment(env_key)

    return {str(effect["name"]): stage for stage, effects in specs.items() for effect in effects}


def _resolve_effects_container(cfg: DictConfig | ListConfig) -> Mapping[str, ConfigTree] | None:
    """Resolve only the effects subtree of a config, never the whole document.

    Effect blocks appear either at the top level (packaged ``data`` configs) or under
    ``data`` (experiment configs). Resolving just this subtree avoids evaluating
    unrelated interpolations elsewhere in the document, such as filename templates in
    the ``datasets`` section.

    Args:
        cfg: A loaded OmegaConf config.

    Returns:
        The resolved effects mapping, or ``None`` when no effects section is present.
    """
    for path in ("data.effects", "effects"):
        node = OmegaConf.select(cfg, path)
        if node is None:
            continue
        resolved = OmegaConf.to_container(node, resolve=True)
        if isinstance(resolved, Mapping):
            return {str(key): value for key, value in resolved.items()}

    return None


def _flatten_effects_container(resolved: Mapping[str, ConfigTree], env_key: str) -> tuple[EffectPreset, ...]:
    """Flatten a resolved effects mapping into stage-aware presets for an environment."""
    stage_by_effect: dict[str, str] = _stage_by_effect(env_key)

    presets: list[EffectPreset] = []
    for preset_name, preset_cfg in resolved.items():
        if not isinstance(preset_cfg, Mapping):
            continue
        is_leaf, effects = _preset_effects(str(preset_name), preset_cfg, stage_by_effect)
        if effects:
            presets.append({"name": str(preset_name), "is_leaf": is_leaf, "effects": effects})

    return tuple(presets)


@cache
def get_gen_data_effect_presets(env_name: str) -> tuple[EffectPreset, ...]:
    """Return the ``gen_data`` effect presets available for an environment.

    Reads the packaged ``spar/configs/data/<env>.yaml`` data config, resolves its
    ``effects`` section, and flattens each preset into stage-aware atomic effects
    with sweepable parameter descriptors. Only effects the environment supports are
    included, so the result is consistent with the dashboard's effect discovery.

    Args:
        env_name: Canonical environment name.

    Returns:
        A tuple of :class:`EffectPreset` entries. Empty when the environment has no
        packaged data config or no supported presets.
    """
    key: str = env_name.strip().lower()
    config_path = _data_config_path(key)
    if config_path is None:
        return ()

    register_omega_conf_resolvers()
    loaded: DictConfig | ListConfig = OmegaConf.load(str(config_path))
    resolved = _resolve_effects_container(loaded)
    if resolved is None:
        return ()

    return _flatten_effects_container(resolved, key)


def parse_effect_presets_from_text(env_name: str, text: str) -> tuple[EffectPreset, ...]:
    """Parse effect presets from raw YAML config text for an environment.

    Accepts any config that carries an ``effects`` section, whether a packaged
    ``data`` config or an experiment config where effects live under ``data``. Only
    the effects subtree is resolved, and only effects the environment supports are
    kept, so the result stays consistent with the dashboard's effect discovery.

    Args:
        env_name: Canonical environment name.
        text: Raw YAML text of the config file.

    Returns:
        A tuple of :class:`EffectPreset` entries. Empty when the text carries no
        effects section or no supported presets.
    """
    key: str = env_name.strip().lower()
    register_omega_conf_resolvers()
    loaded = OmegaConf.create(text)
    if not isinstance(loaded, DictConfig):
        return ()

    resolved = _resolve_effects_container(loaded)
    if resolved is None:
        return ()

    return _flatten_effects_container(resolved, key)


def _configs_root() -> Traversable:
    """Return the packaged ``spar/configs`` resource directory."""
    return resources.files("spar").joinpath("configs")


def _config_label(token: str) -> str:
    """Humanize a config token into a display label."""
    name: str = token.rsplit("/", 1)[-1]
    stem: str = name.rsplit(".", 1)[0]
    group: str = "Data" if token.startswith("data/") else "Experiment"

    return f"{group} · {stem}"


def _candidate_config_tokens(env_key: str) -> list[str]:
    """List packaged config tokens that could carry effects for an environment."""
    root: Traversable = _configs_root()
    tokens: list[str] = []

    data_file: Traversable = root.joinpath("data", f"{env_key}.yaml")
    if data_file.is_file():
        tokens.append(f"data/{env_key}.yaml")

    experiment_dir: Traversable = root.joinpath("experiment", env_key)
    if experiment_dir.is_dir():
        tokens.extend(
            f"experiment/{env_key}/{entry.name}"
            for entry in sorted(experiment_dir.iterdir(), key=lambda item: item.name)
            if entry.is_file() and entry.name.endswith((".yaml", ".yml"))
        )

    return tokens


def _read_config_token_text(token: str) -> str:
    """Read the raw text of a packaged config identified by a ``/``-joined token."""
    node: Traversable = _configs_root()
    for part in token.split("/"):
        node = node.joinpath(part)

    return node.read_text(encoding="utf-8")


@cache
def list_env_config_files(env_name: str) -> tuple[ConfigFileInfo, ...]:
    """Return packaged config files that carry usable effect presets for an environment.

    Scans the packaged ``spar/configs/data/<env>.yaml`` and
    ``spar/configs/experiment/<env>/*.yaml`` configs and keeps only those that parse
    into at least one supported preset. The returned tokens form the allow-list that
    :func:`read_env_config_text` accepts.

    Args:
        env_name: Canonical environment name.

    Returns:
        A tuple of :class:`ConfigFileInfo` entries, ordered data config first then
        experiment configs by name.
    """
    key: str = env_name.strip().lower()
    infos: list[ConfigFileInfo] = []
    for token in _candidate_config_tokens(key):
        try:
            text: str = _read_config_token_text(token)
            presets = parse_effect_presets_from_text(key, text)
        except (OSError, OmegaConfBaseException, ValueError):
            logger.warning(f"Skipping unreadable config '{token}' for environment '{key}'")
            continue
        if presets:
            infos.append({"token": token, "label": _config_label(token), "preset_count": len(presets)})

    return tuple(infos)


def read_env_config_text(env_name: str, token: str) -> str:
    """Read a packaged config selected from the environment's available configs.

    Args:
        env_name: Canonical environment name.
        token: A ``/``-joined config token returned by :func:`list_env_config_files`.

    Returns:
        The raw YAML text of the requested config.

    Raises:
        ValueError: If the token is not available for the environment.
    """
    key: str = env_name.strip().lower()
    allowed: set[str] = {info["token"] for info in list_env_config_files(key)}
    if token not in allowed:
        raise ValueError(f"Config '{token}' is not an available config for environment '{key}'")

    return _read_config_token_text(token)
