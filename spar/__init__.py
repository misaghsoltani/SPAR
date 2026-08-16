from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from .utils.import_utils import lazy_import_module

if TYPE_CHECKING:
    from . import (
        __main__,
        configs,
        data,
        environments,
        main,
        models,
        pipeline,
        scripts,
        search,
        testing,
        training,
        utils,
    )
    from .pipeline.cli import main as run
    from .utils.import_utils import AttributeGetter, DirectoryLister

    __all__: list[str] = [
        "__author__",
        "__main__",
        "__version__",
        "configs",
        "data",
        "environments",
        "main",
        "models",
        "pipeline",
        "run",
        "scripts",
        "search",
        "testing",
        "training",
        "utils",
    ]


__version__ = "0.1.0"
__author__ = "Misagh Soltani"

IMPORTS: dict[str, tuple[str, str]] = {
    "__main__": (".", "__main__"),
    "main": (".", "main"),
    "configs": (".", "configs"),
    "data": (".", "data"),
    "environments": (".", "environments"),
    "models": (".", "models"),
    "pipeline": (".", "pipeline"),
    "scripts": (".", "scripts"),
    "search": (".", "search"),
    "testing": (".", "testing"),
    "training": (".", "training"),
    "utils": (".", "utils"),
    "run": ("..pipeline.cli", "main"),
}

__getattr__: AttributeGetter
__dir__: DirectoryLister

__getattr__, __dir__, __all__ = lazy_import_module(
    imports=IMPORTS,
    module_name=__name__,
    cache_enabled=True,
    thread_safe=False,
    debug_mode=(os_environ.get("SPAR_DEBUG_LAZY_IMPORTS", "").lower() in {"true", "1", "yes"}),
    validate_imports=True,
)
