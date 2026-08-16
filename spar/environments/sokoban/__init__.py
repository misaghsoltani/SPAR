"""SokobanEnv environment for the SPAR framework."""

from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.import_utils import LazyImporter

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute

    from .sokoban import SokobanEnv, SokobanState
    from .sokoban_nn import DecoderCont, DecoderDisc, EncoderCont, EncoderDisc, TransitionModelCont, TransitionModelDisc

__all__: list[str] = [
    "DecoderCont",
    "DecoderDisc",
    "EncoderCont",
    "EncoderDisc",
    "SokobanEnv",
    "SokobanState",
    "TransitionModelCont",
    "TransitionModelDisc",
]

IMPORTS: dict[str, tuple[str, str]] = {
    "SokobanEnv": (".sokoban", "SokobanEnv"),
    "SokobanState": (".sokoban", "SokobanState"),
    "EncoderDisc": (".sokoban_nn", "EncoderDisc"),
    "EncoderCont": (".sokoban_nn", "EncoderCont"),
    "DecoderDisc": (".sokoban_nn", "DecoderDisc"),
    "DecoderCont": (".sokoban_nn", "DecoderCont"),
    "TransitionModelDisc": (".sokoban_nn", "TransitionModelDisc"),
    "TransitionModelCont": (".sokoban_nn", "TransitionModelCont"),
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
