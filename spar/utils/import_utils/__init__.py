from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from .lazy_importer import LazyImporter

if TYPE_CHECKING:
    from .lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute, LazyImportError, lazy_import_module
    from .stage_importer import clear_cli_cache, get_cli_import_info, run_stage, validate_cli_imports


__all__: list[str] = [
    "AttributeGetter",
    "DirectoryLister",
    "LazyImportError",
    "LazyImporter",
    "clear_cli_cache",
    "get_cli_import_info",
    "lazy_import_module",
    "run_stage",
    "validate_cli_imports",
]

IMPORTS: dict[str, tuple[str, str]] = {
    "LazyImportError": (".lazy_importer", "LazyImportError"),
    "lazy_import_module": (".lazy_importer", "lazy_import_module"),
    "clear_cli_cache": (".stage_importer", "clear_cli_cache"),
    "get_cli_import_info": (".stage_importer", "get_cli_import_info"),
    "run_stage": (".stage_importer", "run_stage"),
    "validate_cli_imports": (".stage_importer", "validate_cli_imports"),
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
