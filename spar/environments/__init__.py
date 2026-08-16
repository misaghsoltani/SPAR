"""Environment implementations for the SPAR framework."""

from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.import_utils.lazy_importer import LazyImporter

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute

    from .abstracts.environment import ABCEnvironment
    from .abstracts.models import ABCDQN, ABCDecoder, ABCEncoder, ABCTransitionModelCont, ABCTransitionModelDisc
    from .abstracts.state import ABCState


__all__: list[str] = [
    "ABCDQN",
    "ABCDecoder",
    "ABCEncoder",
    "ABCEnvironment",
    "ABCState",
    "ABCTransitionModelCont",
    "ABCTransitionModelDisc",
]

# Lazy imports mapping
IMPORTS: dict[str, tuple[str, str]] = {
    "cube3": (".", "cube3"),
    "digitjump": (".", "digitjump"),
    "iceslider": (".", "iceslider"),
    "sokoban": (".", "sokoban"),
    "ABCDQN": (".abstracts.models", "ABCDQN"),
    "ABCDecoder": (".abstracts.models", "ABCDecoder"),
    "ABCEncoder": (".abstracts.models", "ABCEncoder"),
    "ABCTransitionModelCont": (".abstracts.models", "ABCTransitionModelCont"),
    "ABCTransitionModelDisc": (".abstracts.models", "ABCTransitionModelDisc"),
    "ABCEnvironment": (".abstracts.environment", "ABCEnvironment"),
    "ABCState": (".abstracts.state", "ABCState"),
    "Cube3": (".cube3", "Cube3"),
    "Cube3State": (".cube3", "Cube3State"),
    "Cube3Triples": (".cube3", "Cube3Triples"),
    "DigitJumpEnv": (".digitjump", "DigitJumpEnv"),
    "DigitJumpState": (".digitjump", "DigitJumpState"),
    "IceSliderEnv": (".iceslider", "IceSliderEnv"),
    "IceSliderState": (".iceslider", "IceSliderState"),
    "SokobanEnv": (".sokoban", "SokobanEnv"),
    "SokobanState": (".sokoban", "SokobanState"),
}


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
