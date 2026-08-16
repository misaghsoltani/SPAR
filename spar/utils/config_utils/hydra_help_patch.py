"""Hydra help rendering with deferred imports during CLI startup."""

from __future__ import annotations

from argparse import Action, ArgumentParser
import importlib
import os
import string
import sys
from threading import Lock
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from hydra._internal.hydra import Hydra

_PATCH_LOCK: Lock = Lock()
_PATCH_STATE: list[bool] = [False]
_PATCH_MARKER_ATTR: str = "__spar_lazy_help_patch__"
_HYDRA_GET_HELP_ATTR: str = "get_help"
_ARGPARSE_PATCH_STATE: list[bool] = [False]
_ARGPARSE_PATCH_MARKER_ATTR: str = "__spar_hydra_lazy_help_patch__"
_ARGPARSE_CHECK_HELP_ATTR: str = "_check_help"


def _patch_argparse_lazy_completion_help() -> bool:
    """Allow Hydra's lazy completion help value on Python 3.14 and later.

    Returns:
        True when the compatibility patch is active, otherwise False.
    """
    if sys.version_info.major * 100 + sys.version_info.minor < 314:
        return False

    if _ARGPARSE_PATCH_STATE[0]:
        return True

    current_check_help: Callable[[ArgumentParser, Action], None] = getattr(ArgumentParser, _ARGPARSE_CHECK_HELP_ATTR)
    if getattr(current_check_help, _ARGPARSE_PATCH_MARKER_ATTR, False):
        _ARGPARSE_PATCH_STATE[0] = True
        return True

    def check_help(parser: ArgumentParser, action: Action) -> None:
        help_value = vars(action).get("help")
        if help_value is not None and not isinstance(help_value, str):
            action.help = repr(help_value)
        current_check_help(parser, action)

    setattr(check_help, _ARGPARSE_PATCH_MARKER_ATTR, True)
    setattr(ArgumentParser, _ARGPARSE_CHECK_HELP_ATTR, check_help)
    _ARGPARSE_PATCH_STATE[0] = True
    return True


def _get_hydra_class() -> type[Hydra]:
    """Import the Hydra class when help rendering needs it."""
    hydra_internal_module: ModuleType = importlib.import_module("hydra._internal.hydra")
    return hydra_internal_module.Hydra


def _is_hydra_group(group_name: str) -> bool:
    return group_name.startswith("hydra/") or group_name == "hydra"


def _template_placeholders(template: string.Template) -> set[str]:
    """Extract placeholder names used by a ``string.Template``."""
    placeholders: set[str] = set()
    for match in template.pattern.finditer(template.template):
        if match.group("escaped") is not None:
            continue
        name: str | None = match.group("named") or match.group("braced")
        if name is not None:
            placeholders.add(name)
    return placeholders


def _get_help_lazy_fields(
    self: Hydra, help_cfg: DictConfig, cfg: DictConfig, args_parser: ArgumentParser, resolve: bool
) -> str:
    """Hydra.get_help replacement that computes only referenced template fields."""
    template: string.Template = string.Template(help_cfg.template)
    placeholders: set[str] = _template_placeholders(template)
    values: dict[str, str] = {}

    if "FLAGS_HELP" in placeholders:
        values["FLAGS_HELP"] = self.format_args_help(args_parser)
    if "HYDRA_CONFIG_GROUPS" in placeholders:
        values["HYDRA_CONFIG_GROUPS"] = self.format_config_groups(_is_hydra_group)
    if "APP_CONFIG_GROUPS" in placeholders:
        values["APP_CONFIG_GROUPS"] = self.format_config_groups(lambda group_name: not _is_hydra_group(group_name))
    if "CONFIG" in placeholders:
        values["CONFIG"] = OmegaConf.to_yaml(cfg, resolve=resolve)

    return template.substitute(values)


def patch_hydra_get_help_lazy_fields() -> bool:
    """Patch ``Hydra.get_help`` to avoid unnecessary config-group formatting work.

    Returns:
        bool: True when patched (or already patched), False when disabled via
        ``SPAR_DISABLE_HYDRA_HELP_PATCH=1``.
    """
    _patch_argparse_lazy_completion_help()

    if os.environ.get("SPAR_DISABLE_HYDRA_HELP_PATCH") == "1":
        return False

    if _PATCH_STATE[0]:
        return True

    with _PATCH_LOCK:
        hydra_class: type[Hydra] = _get_hydra_class()
        current_get_help = getattr(hydra_class, _HYDRA_GET_HELP_ATTR)
        if getattr(current_get_help, _PATCH_MARKER_ATTR, False):
            _PATCH_STATE[0] = True
            return True

        patched_get_help = _get_help_lazy_fields
        setattr(patched_get_help, _PATCH_MARKER_ATTR, True)
        setattr(hydra_class, _HYDRA_GET_HELP_ATTR, patched_get_help)
        _PATCH_STATE[0] = True
        return True
