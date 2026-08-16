"""DigitJump environment for the SPAR framework."""

from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.import_utils.lazy_importer import LazyImporter

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute

    from .digit_jump import DigitJumpEnv, DigitJumpState
    from .digitjump_nn import (
        DecoderCont,
        DecoderDisc,
        EncoderCont,
        EncoderDisc,
        TransitionModelCont,
        TransitionModelDisc,
    )

__all__: list[str] = [
    "DecoderCont",
    "DecoderDisc",
    "DigitJumpEnv",
    "DigitJumpState",
    "EncoderCont",
    "EncoderDisc",
    "TransitionModelCont",
    "TransitionModelDisc",
]

IMPORTS: dict[str, tuple[str, str]] = {
    "DigitJumpEnv": (".digit_jump", "DigitJumpEnv"),
    "DigitJumpState": (".digit_jump", "DigitJumpState"),
    "EncoderDisc": (".digitjump_nn", "EncoderDisc"),
    "EncoderCont": (".digitjump_nn", "EncoderCont"),
    "DecoderDisc": (".digitjump_nn", "DecoderDisc"),
    "DecoderCont": (".digitjump_nn", "DecoderCont"),
    "TransitionModelDisc": (".digitjump_nn", "TransitionModelDisc"),
    "TransitionModelCont": (".digitjump_nn", "TransitionModelCont"),
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
