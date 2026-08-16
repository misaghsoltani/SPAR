"""Model classes for SPAR framework."""

from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.import_utils.lazy_importer import lazy_import_module

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister

    from .factory import ModelFactory


__all__: list[str] = ["ModelFactory"]

IMPORTS: dict[str, tuple[str, str]] = {"ModelFactory": ("spar.models.factory", "ModelFactory")}

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
