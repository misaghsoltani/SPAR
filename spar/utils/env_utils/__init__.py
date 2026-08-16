"""Environment utilities for SPAR package."""

from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.import_utils.lazy_importer import LazyImporter

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute

    from .env_utils import get_environment
    from .puzzlegen import DigitJump, IceSlider, digit_jump, ice_slider
    from .viz_utils import InteractiveCube, Quaternion, euler_to_quaternion, project_points

__all__: list[str] = [
    "SOKOBAN_DATA_DIR",
    "DigitJump",
    "IceSlider",
    "InteractiveCube",
    "Quaternion",
    "digit_jump",
    "euler_to_quaternion",
    "get_environment",
    "ice_slider",
    "project_points",
]

IMPORTS: dict[str, tuple[str, str]] = {
    "get_environment": (".env_utils", "get_environment"),
    "DigitJump": (".puzzlegen.digit_jump", "DigitJump"),
    "IceSlider": (".puzzlegen.ice_slider", "IceSlider"),
    "InteractiveCube": (".viz_utils", "InteractiveCube"),
    "Quaternion": (".viz_utils", "Quaternion"),
    "euler_to_quaternion": (".viz_utils", "euler_to_quaternion"),
    "project_points": (".viz_utils", "project_points"),
}


lazy_importer: LazyImporter[LazyAttribute] = LazyImporter(
    imports=IMPORTS,
    module_name=__name__,
    cache_enabled=True,
    thread_safe=False,
    debug_mode=(os_environ.get("SPAR_DEBUG_LAZY_IMPORTS", "").lower() in {"true", "1", "yes"}),
    validate_imports=True,
)


lazy_importer.preload_attributes("get_environment")

SOKOBAN_DATA_DIR: str = "spar/utils/env_utils/sokoban_data"

__getattr__: AttributeGetter = lazy_importer.get_attr
__dir__: DirectoryLister = lazy_importer.get_dir
