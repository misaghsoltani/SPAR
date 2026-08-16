"""Data handling utilities for SPAR."""

from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.import_utils.lazy_importer import LazyImporter

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute

    from .alignment_dataset import (
        AlignmentDataset,
        AlignmentDatasetInfo,
        H5PyDataset,
        TransformProtocol,
        create_dataloader,
    )
    from .encoders import encode_offline_data
    from .generator import generate_data


__all__: list[str] = [
    "AlignmentDataset",
    "AlignmentDatasetInfo",
    "H5PyDataset",
    "TransformProtocol",
    "create_dataloader",
    "encode_offline_data",
    "generate_data",
]

# Lazy import mappings
IMPORTS: dict[str, tuple[str, str]] = {
    "encode_offline_data": ("spar.data.encoders", "encode_offline_data"),
    "generate_data": ("spar.data.generator", "generate_data"),
    "create_dataloader": ("spar.data.alignment_dataset", "create_dataloader"),
    "AlignmentDataset": ("spar.data.alignment_dataset", "AlignmentDataset"),
    "AlignmentDatasetInfo": ("spar.data.alignment_dataset", "AlignmentDatasetInfo"),
    "H5PyDataset": ("spar.data.alignment_dataset", "H5PyDataset"),
    "TransformProtocol": ("spar.data.alignment_dataset", "TransformProtocol"),
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
