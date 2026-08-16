"""Training utilities for SPAR package."""

from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.import_utils.lazy_importer import LazyImporter

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute

    from .alignment_model_trainer.training_runner import train as train_alignment_model
    from .dqn_trainer.training_runner import train_heuristic
    from .world_model_trainer.training_runner import train as train_world_model

__all__: list[str] = ["train_alignment_model", "train_heuristic", "train_world_model"]

# Lazy imports mapping
IMPORTS: dict[str, tuple[str, str]] = {
    "train_world_model": ("spar.training.world_model_trainer.training_runner", "train"),
    "train_alignment_model": ("spar.training.alignment_model_trainer.training_runner", "train"),
    "train_heuristic": ("spar.training.dqn_trainer.training_runner", "train_heuristic"),
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
