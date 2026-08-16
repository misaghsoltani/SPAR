"""Cube3 environment for the SPAR framework."""

from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.import_utils.lazy_importer import LazyImporter

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute

    from .cube3 import Cube3, Cube3State, Cube3Triples
    from .cube3_nn import DecoderCont, DecoderDisc, EncoderCont, EncoderDisc, TransitionModelCont, TransitionModelDisc

__all__: list[str] = [
    "Cube3",
    "Cube3State",
    "Cube3Triples",
    "DecoderCont",
    "DecoderDisc",
    "EncoderCont",
    "EncoderDisc",
    "TransitionModelCont",
    "TransitionModelDisc",
]


IMPORTS: dict[str, tuple[str, str]] = {
    "Cube3": (".cube3", "Cube3"),
    "Cube3State": (".cube3", "Cube3State"),
    "Cube3Triples": (".cube3", "Cube3Triples"),
    "EncoderDisc": (".cube3_nn", "EncoderDisc"),
    "EncoderCont": (".cube3_nn", "EncoderCont"),
    "DecoderDisc": (".cube3_nn", "DecoderDisc"),
    "DecoderCont": (".cube3_nn", "DecoderCont"),
    "TransitionModelDisc": (".cube3_nn", "TransitionModelDisc"),
    "TransitionModelCont": (".cube3_nn", "TransitionModelCont"),
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
