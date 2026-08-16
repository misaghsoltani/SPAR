"""Puzzle generation environments."""

from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.import_utils import LazyImporter

from .registry import register_puzzle_environments

# Auto-register environments when the package is imported
register_puzzle_environments()

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute

    from .digit_jump import DigitJump
    from .ice_slider import IceSlider
    from .registry import make_digit_jump, make_ice_slider

__all__: list[str] = ["DigitJump", "IceSlider", "make_digit_jump", "make_ice_slider"]

# Lazy imports mapping
IMPORTS: dict[str, tuple[str, str]] = {
    "DigitJump": (".digit_jump", "DigitJump"),
    "IceSlider": (".ice_slider", "IceSlider"),
    "make_digit_jump": (".registry", "make_digit_jump"),
    "make_ice_slider": (".registry", "make_ice_slider"),
}

# Create lazy importer
lazy_importer: LazyImporter[LazyAttribute] = LazyImporter(
    imports=IMPORTS,
    module_name=__name__,
    cache_enabled=True,
    thread_safe=False,
    debug_mode=(os_environ.get("SPAR_DEBUG_LAZY_IMPORTS", "").lower() in {"true", "1", "yes"}),
    validate_imports=True,
)

__getattr__: AttributeGetter = lazy_importer.get_attr
__dir__: DirectoryLister = lazy_importer.get_dir
