"""IceSlider environment for the SPAR framework."""

from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.import_utils.lazy_importer import LazyImporter

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute

    from .iceslider import IceSliderEnv, IceSliderState
    from .iceslider_nn import (
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
    "EncoderCont",
    "EncoderDisc",
    "IceSliderEnv",
    "IceSliderState",
    "TransitionModelCont",
    "TransitionModelDisc",
]

# Lazy imports mapping
IMPORTS: dict[str, tuple[str, str]] = {
    "IceSliderEnv": (".iceslider", "IceSliderEnv"),
    "IceSliderState": (".iceslider", "IceSliderState"),
    "EncoderDisc": (".iceslider_nn", "EncoderDisc"),
    "EncoderCont": (".iceslider_nn", "EncoderCont"),
    "DecoderDisc": (".iceslider_nn", "DecoderDisc"),
    "DecoderCont": (".iceslider_nn", "DecoderCont"),
    "TransitionModelDisc": (".iceslider_nn", "TransitionModelDisc"),
    "TransitionModelCont": (".iceslider_nn", "TransitionModelCont"),
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
