"""Abstract base classes for the SPAR framework."""

from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.import_utils.lazy_importer import LazyImporter

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute

    from .environment import ABCEnvironment
    from .models import (
        ABCDQN,
        ABCAlignmentModel,
        ABCDecoder,
        ABCEncoder,
        ABCTransitionModelCont,
        ABCTransitionModelDisc,
    )
    from .state import ABCState

__all__: list[str] = [
    "ABCDQN",
    "ABCAlignmentModel",
    "ABCDecoder",
    "ABCEncoder",
    "ABCEnvironment",
    "ABCState",
    "ABCTransitionModelCont",
    "ABCTransitionModelDisc",
]

# Lazy imports mapping
IMPORTS: dict[str, tuple[str, str]] = {
    "ABCEnvironment": (".environment", "ABCEnvironment"),
    "ABCState": (".state", "ABCState"),
    "ABCDQN": (".models", "ABCDQN"),
    "ABCEncoder": (".models", "ABCEncoder"),
    "ABCDecoder": (".models", "ABCDecoder"),
    "ABCTransitionModelCont": (".models", "ABCTransitionModelCont"),
    "ABCTransitionModelDisc": (".models", "ABCTransitionModelDisc"),
    "ABCAlignmentModel": (".models", "ABCAlignmentModel"),
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
