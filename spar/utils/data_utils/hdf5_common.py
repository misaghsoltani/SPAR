"""Shared typed HDF5 I/O utilities for SPAR data modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import h5py
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from numpy.typing import NDArray


CompressionType: TypeAlias = Literal["none", "gzip", "szip", "lzf"]
ReadOpenStrategy: TypeAlias = Literal["swmr_locking", "locking_only", "plain"]

HDF5_RDCC_NBYTES_DEFAULT: int = 128 * 1024**2
HDF5_RDCC_NSLOTS_DEFAULT: int = 10_007
HDF5_RDCC_W0_DEFAULT: float = 0.25
_READ_OPEN_STRATEGY_CACHE: dict[str, ReadOpenStrategy] = {}


def ensure_hdf5_path(path: str) -> str:
    """Return ``path`` with a canonical HDF5 extension.

    Args:
        path: Candidate output path.

    Returns:
        ``path`` unchanged when it already ends with ``.h5`` or ``.hdf5``.
        Otherwise ``.h5`` is appended.
    """
    if path.endswith((".h5", ".hdf5")):
        return path
    return f"{path}.h5"


def open_hdf5_for_read(
    path: str,
    *,
    rdcc_nbytes: int = HDF5_RDCC_NBYTES_DEFAULT,
    rdcc_nslots: int = HDF5_RDCC_NSLOTS_DEFAULT,
    rdcc_w0: float = HDF5_RDCC_W0_DEFAULT,
) -> h5py.File:
    """Open an HDF5 file with a SWMR-first, compatibility-safe strategy.

    Args:
        path: File path to open.
        rdcc_nbytes: Raw data chunk cache size in bytes.
        rdcc_nslots: Number of raw data chunk cache hash slots.
        rdcc_w0: Raw data chunk cache eviction policy weight.

    Returns:
        Open ``h5py.File`` handle in read mode.
    """

    def _open_with_strategy(strategy: ReadOpenStrategy) -> h5py.File:
        if strategy == "swmr_locking":
            return h5py.File(
                path,
                "r",
                rdcc_nbytes=rdcc_nbytes,
                rdcc_nslots=rdcc_nslots,
                rdcc_w0=rdcc_w0,
                libver="latest",
                swmr=True,
                locking="best-effort",
            )
        if strategy == "locking_only":
            return h5py.File(
                path, "r", rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots, rdcc_w0=rdcc_w0, locking="best-effort"
            )
        return h5py.File(path, "r", rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots, rdcc_w0=rdcc_w0)

    cached_strategy: ReadOpenStrategy | None = _READ_OPEN_STRATEGY_CACHE.get(path)
    if cached_strategy is not None:
        try:
            cached_file: h5py.File = _open_with_strategy(cached_strategy)
        except (OSError, TypeError, ValueError):
            _READ_OPEN_STRATEGY_CACHE.pop(path, None)
        else:
            return cached_file

    last_error: OSError | TypeError | ValueError

    try:
        swmr_file: h5py.File = _open_with_strategy("swmr_locking")
    except (OSError, TypeError, ValueError) as exc:
        last_error = exc
    else:
        _READ_OPEN_STRATEGY_CACHE[path] = "swmr_locking"
        return swmr_file

    try:
        locking_file: h5py.File = _open_with_strategy("locking_only")
    except (OSError, TypeError, ValueError) as exc:
        last_error = exc
    else:
        _READ_OPEN_STRATEGY_CACHE[path] = "locking_only"
        return locking_file

    try:
        plain_file: h5py.File = _open_with_strategy("plain")
    except (OSError, TypeError, ValueError) as exc:
        last_error = exc
    else:
        _READ_OPEN_STRATEGY_CACHE[path] = "plain"
        return plain_file

    raise last_error


def open_hdf5_for_write(path: str) -> h5py.File:
    """Open an HDF5 file for deterministic write-heavy workflows.

    Args:
        path: File path to open.

    Returns:
        Open ``h5py.File`` handle in write mode.
    """
    try:
        return h5py.File(path, "w", libver="latest", track_order=True, locking="best-effort")
    except TypeError:
        return h5py.File(path, "w", libver="latest", track_order=True)


def normalize_compression_value(raw_value: str | bool | None) -> CompressionType:
    """Normalize a compression configuration value to supported literals.

    Args:
        raw_value: Compression config from user/config files.

    Returns:
        Normalized compression literal.

    Raises:
        ValueError: If the value is unsupported.
    """
    if raw_value is True:
        return "lzf"
    if raw_value is False or raw_value is None:
        return "none"
    if raw_value == "none":
        return "none"
    if raw_value == "gzip":
        return "gzip"
    if raw_value == "szip":
        return "szip"
    if raw_value == "lzf":
        return "lzf"

    raise ValueError(f"Unsupported compression value '{raw_value}'. Expected one of: none, gzip, szip, lzf.")


def decode_attr_text(value: np.generic | bytes | bytearray | str) -> str:
    """Decode a scalar HDF5 attribute value into UTF-8 text."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def write_utf8_string_attr(attrs: h5py.AttributeManager, name: str, value: str) -> None:
    """Write a UTF-8 text attribute with compatibility fallback.

    Args:
        attrs: Target HDF5 attribute manager.
        name: Attribute key.
        value: Attribute text value.
    """
    encoded: bytes = value.encode("utf-8")
    try:
        attrs[name] = encoded
    except (TypeError, ValueError):
        attrs[name] = value


def write_utf8_string_list_attr(attrs: h5py.AttributeManager, name: str, values: Sequence[str]) -> None:
    """Write a UTF-8 text-list attribute with compatibility fallback.

    Args:
        attrs: Target HDF5 attribute manager.
        name: Attribute key.
        values: Text values to store.
    """
    value_list: list[str] = list(values)
    encoded_values: list[bytes] = [item.encode("utf-8") for item in value_list]
    try:
        attrs[name] = encoded_values
    except (TypeError, ValueError):
        attrs[name] = value_list


def _decode_variant_names_array(raw_attr: h5py.Empty | NDArray[np.generic]) -> list[str]:
    """Decode variant names from a non-scalar HDF5 attribute value."""
    attr_array: NDArray[np.generic] = np.asarray(raw_attr)
    if attr_array.ndim == 0:
        scalar_value: np.generic | bytes | bytearray | str = attr_array.item()
        return [decode_attr_text(scalar_value)]

    return [decode_attr_text(name) for name in attr_array.reshape(-1)]


def read_variant_names_from_attrs(attrs: h5py.AttributeManager) -> list[str]:
    """Load and normalize ``variant_names`` from HDF5 attributes.

    Args:
        attrs: Attribute manager from an open HDF5 file.

    Returns:
        Decoded variant names, or an empty list when missing/unreadable.
    """
    raw_attr: h5py.Empty | bytes | bytearray | str | NDArray[np.generic] = attrs.get("variant_names", [])
    variant_names: list[str] = []

    if isinstance(raw_attr, (bytes, bytearray, str)):
        variant_names = [decode_attr_text(raw_attr)]
    else:
        try:
            variant_names = _decode_variant_names_array(raw_attr)
        except Exception:
            variant_names = []

    return [name for name in variant_names if name]


def read_variant_type_from_attrs(attrs: h5py.AttributeManager, *, default: str = "full") -> str:
    """Load and normalize ``variant_type`` from HDF5 attributes.

    Args:
        attrs: Attribute manager from an open HDF5 file.
        default: Fallback variant type if missing.

    Returns:
        Decoded variant type string.
    """
    raw_attr: h5py.Empty | bytes | str = attrs.get("variant_type", default.encode("utf-8"))
    if isinstance(raw_attr, bytes):
        decoded: str = raw_attr.decode("utf-8")
        return decoded or default
    if isinstance(raw_attr, str):
        return raw_attr or default
    return default


def read_float_dataset(dataset: h5py.Dataset, dtype: type[np.float32 | np.float64]) -> NDArray[np.float32 | np.float64]:
    """Read a dataset into a contiguous float buffer via ``read_direct``.

    Args:
        dataset: Source dataset.
        dtype: Destination float dtype.

    Returns:
        Contiguous array with the requested float dtype.
    """
    out: NDArray[np.float32 | np.float64] = np.empty(dataset.shape, dtype=dtype)
    dataset.read_direct(out)
    return out


def read_int64_dataset(dataset: h5py.Dataset) -> NDArray[np.int64]:
    """Read a dataset into a contiguous int64 buffer via ``read_direct``."""
    out: NDArray[np.int64] = np.empty(dataset.shape, dtype=np.int64)
    dataset.read_direct(out)
    return out


def get_chunk_shape_4d(
    data: NDArray[np.generic], *, target_chunk_bytes: int = 1 * 1024 * 1024
) -> tuple[int, int, int, int]:
    """Compute a chunk shape for 4D frame-major tensors near a target byte size.

    Args:
        data: Input 4D array with shape ``(N, C, H, W)``.
        target_chunk_bytes: Approximate target bytes per chunk.

    Returns:
        Chunk shape tuple ``(frames, C, H, W)``.

    Raises:
        ValueError: If ``data`` is not 4D.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data for chunk-shape calculation, got {data.ndim}D.")

    n, c, h, w = data.shape
    bytes_per_frame: int = int(data.dtype.itemsize) * c * h * w
    frames_per_chunk: int = max(1, min(n, target_chunk_bytes // max(1, bytes_per_frame)))
    return (frames_per_chunk, c, h, w)


def create_array_dataset(
    group: h5py.Group,
    name: str,
    data: NDArray[np.generic],
    *,
    compression: CompressionType,
    chunks: tuple[int, ...] | None = None,
    shuffle: bool = True,
) -> h5py.Dataset:
    """Create a dataset with consistent deterministic metadata settings.

    Args:
        group: Parent HDF5 group.
        name: Dataset name.
        data: NumPy array payload.
        compression: Compression mode.
        chunks: Optional chunk shape.
        shuffle: Whether to enable the shuffle filter when compressed.

    Returns:
        The created dataset.
    """
    if compression == "none":
        if chunks is None:
            return group.create_dataset(name, data=data, dtype=data.dtype, track_times=False)
        return group.create_dataset(name, data=data, dtype=data.dtype, chunks=chunks, track_times=False)

    if chunks is None:
        return group.create_dataset(
            name, data=data, dtype=data.dtype, compression=compression, shuffle=shuffle, track_times=False
        )
    return group.create_dataset(
        name, data=data, dtype=data.dtype, chunks=chunks, compression=compression, shuffle=shuffle, track_times=False
    )


def create_utf8_string_dataset(
    group: h5py.Group, name: str, data: Sequence[str], *, compression: CompressionType = "none"
) -> h5py.Dataset:
    """Create a UTF-8 string dataset with deterministic metadata.

    Args:
        group: Parent HDF5 group.
        name: Dataset name.
        data: Sequence of UTF-8 text entries.
        compression: Compression mode.

    Returns:
        The created dataset.
    """
    string_dtype: h5py.string_dtype = h5py.string_dtype(encoding="utf-8")
    text_data: list[str] = list(data)
    if compression == "none":
        return group.create_dataset(name, data=text_data, dtype=string_dtype, track_times=False)
    return group.create_dataset(name, data=text_data, dtype=string_dtype, compression=compression, track_times=False)
