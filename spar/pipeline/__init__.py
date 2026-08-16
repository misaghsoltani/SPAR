"""CLI entry point for SPAR framework."""

from __future__ import annotations

import contextlib
from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.config_utils.misc import register_omega_conf_resolvers
from spar.utils.import_utils.lazy_importer import LazyImporter

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute

    from .cli import main as run

__all__: list[str] = ["run"]

# Register OmegaConf resolvers early so Hydra help can use them before main() loads
with contextlib.suppress(Exception):  # safe to call multiple times
    register_omega_conf_resolvers()

# Lazy import mappings
IMPORTS: dict[str, tuple[str, str]] = {"run": ("spar.pipeline.cli", "main")}

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
