"""Download, validate, and load CIFAR-10 image batches."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from logging import getLogger
import os
import pathlib
import pickle
import tarfile
import tempfile
import time
from typing import TYPE_CHECKING
import urllib.request

import filelock
import numpy as np

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path
    from tarfile import TarFile, TarInfo

    from filelock import BaseFileLock
    from numpy.typing import NDArray
    from typing_extensions import LiteralString


logger: Logger = getLogger(__name__)

# Enable CIFAR progress messages when requested by the caller.
VERBOSE: bool = os.environ.get("CIFAR_VERBOSE", "0") == "1"
MAIN_PROCESS_FILE: str | None = None


@dataclass(slots=True)
class _CifarImagesCache:
    images: list[NDArray[np.float64]] | None = None


_CIFAR10_IMAGES_CACHE = _CifarImagesCache()


def _extract_tar_members(tar: TarFile, destination: Path) -> None:
    """Extract archive members one by one.

    Args:
        tar: Open tar archive.
        destination: Destination directory.
    """
    member: TarInfo
    for member in tar:
        tar.extract(member, path=destination)


def _raise_download_failure() -> None:
    """Raise a deterministic download failure error."""
    raise FileNotFoundError("Failed to download CIFAR-10 dataset")


def _create_main_process_token(token_file: Path) -> bool:
    """Create the CIFAR main-process token for this process."""
    try:
        token_file.write_text(f"{os.getpid()}", encoding="utf-8")
    except Exception:
        return False
    else:
        return True


def _token_matches_current_process(token_file: Path) -> bool:
    """Return whether the existing token belongs to this process."""
    try:
        pid: str = token_file.read_text(encoding="utf-8").strip()
    except Exception:
        return False
    else:
        return pid == str(os.getpid())


def _is_main_process(data_dir: str) -> bool:
    """Determine whether this process should report CIFAR progress.

    The first process that creates the token file becomes the reporting process.

    Args:
        data_dir: Directory where CIFAR-10 data is stored.

    Returns:
        Whether this process should report progress.
    """
    token_file: Path = pathlib.Path(data_dir) / ".cifar_main_process"

    # Only one process can create the token file first.
    try:
        token_exists: bool = token_file.exists()
    except Exception:
        # Suppress progress when the token cannot be inspected.
        return False

    if not token_exists:
        return _create_main_process_token(token_file)
    return _token_matches_current_process(token_file)


def _log(message: str, level: str = "INFO") -> None:
    """Log a message only if this is the main process or VERBOSE is enabled.

    Args:
        message: Message to log
        level: Log level (INFO, SUCCESS, WARNING, ERROR)

    """
    if VERBOSE:
        logger.info(f"[{level}] {message}")


def _download_archive(tar_file: Path, is_main: bool) -> bool:
    """Download the CIFAR archive.

    Args:
        tar_file: Destination path for the downloaded archive.
        is_main: Whether this process should report progress.

    Returns:
        Whether the download completed.
    """
    cifar_url: LiteralString = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    start_time: float = time.time()
    last_reported_percent: int = -1

    def report_progress(count: int, block_size: int, total_size: int) -> None:
        """Report download progress."""
        nonlocal last_reported_percent
        if is_main:  # Only show progress if main process
            percent: float = count * block_size * 100.0 / total_size if total_size > 0 else 0
            current_milestone: int = int(percent // 10) * 10
            if current_milestone != last_reported_percent and current_milestone > 0:
                last_reported_percent = current_milestone
                elapsed: float = time.time() - start_time
                logger.info(f"CIFAR download progress: {current_milestone}% (Time: {elapsed:.2f}s)")

    try:
        urllib.request.urlretrieve(cifar_url, str(tar_file), reporthook=report_progress)
    except Exception as e:
        if is_main:
            logger.info(f"Error during CIFAR download: {e!s}")

        if tar_file.exists():
            with contextlib.suppress(Exception):
                tar_file.unlink()

        return False

    if is_main:
        elapsed: float = time.time() - start_time
        logger.info(f"CIFAR download complete in {elapsed:.2f}s")
    return True


def _extract_tar_file(tar_file: Path, data_dir_path: Path) -> None:
    """Extract a downloaded CIFAR archive."""
    with tarfile.open(tar_file, "r:gz") as tar:
        _extract_tar_members(tar, data_dir_path)


def _finalize_cifar_extraction(
    tar_file: Path, extracted_dir: Path, download_marker: Path, is_main: bool, extract_start: float
) -> None:
    """Log extraction success, remove the archive if possible, and write the completion marker."""
    if is_main:
        extract_time: float = time.time() - extract_start
        logger.info(f"CIFAR extraction complete in {extract_time:.2f}s")

    # Clean up tar file after successful extraction
    try:
        tar_file.unlink()
        if is_main:
            _log("Removed temporary archive file")
    except Exception as e:
        if is_main:
            _log(f"Note: Could not remove tar file: {e!s}", "WARNING")

    # Create marker file to indicate successful completion
    download_marker.write_text("download completed", encoding="utf-8")

    if is_main:
        logger.info(f"CIFAR-10 dataset successfully prepared at {extracted_dir}")


def _extract_archive(
    tar_file: Path, data_dir_path: Path, extracted_dir: Path, download_marker: Path, is_main: bool
) -> bool:
    """Extract the CIFAR archive and report whether extraction succeeded."""
    if is_main:
        logger.info("Extracting CIFAR-10 dataset...")

    extract_start: float = time.time()

    try:
        _extract_tar_file(tar_file, data_dir_path)
        _finalize_cifar_extraction(tar_file, extracted_dir, download_marker, is_main, extract_start)
    except Exception as e:
        if is_main:
            logger.info(f"Error during CIFAR extraction: {e!s}")
        return False
    else:
        return True


def _download_cifar10_with_lock(
    data_dir: str, data_dir_path: Path, tar_file: Path, extracted_dir: Path, lock_file: Path, download_marker: Path
) -> bool:
    """Download/extract CIFAR while holding the inter-process file lock."""
    # Long timeout to allow complete download
    lock: BaseFileLock = filelock.FileLock(str(lock_file), timeout=300)

    # Critical section - acquire lock
    with lock:
        is_main: bool = _is_main_process(data_dir)
        if is_main:
            _log("Secured exclusive access for CIFAR-10 download")

        # Double-check after acquiring the lock
        if extracted_dir.exists() and (extracted_dir / "data_batch_1").exists():
            if is_main:
                _log(f"CIFAR-10 dataset now available at {extracted_dir}")

            download_marker.write_text("download completed", encoding="utf-8")
            return True

        # Check if download is already in progress by another process
        if tar_file.exists() and tar_file.stat().st_size > 0:
            if is_main:
                _log(f"Found partial download at {tar_file}")

        else:
            if is_main:
                logger.info("Downloading CIFAR-10 dataset...")

            if not _download_archive(tar_file, is_main):
                return False

        # Extract the data
        if tar_file.exists():
            return _extract_archive(tar_file, data_dir_path, extracted_dir, download_marker, is_main)

        if is_main:
            logger.info(f"Error: Archive file {tar_file} not found for extraction")
        return False


def download_cifar10(data_dir: str) -> bool:
    """Download and extracts the CIFAR-10 dataset if it doesn't exist.

    Args:
        data_dir: Directory where CIFAR-10 data should be stored

    Returns:
        bool: True if successful, False otherwise

    """
    data_dir_path: Path = pathlib.Path(data_dir)
    # Create the dataset directory before downloading files.
    try:
        data_dir_path.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        if _is_main_process(data_dir):
            logger.info(f"Error creating data directory {data_dir}: {e!s}")
        return False

    tar_file: Path = data_dir_path / "cifar-10-python.tar.gz"
    extracted_dir: Path = data_dir_path / "cifar-10-batches-py"
    lock_file: Path = data_dir_path / "cifar10_download.lock"
    download_marker: Path = data_dir_path / ".download_complete"

    # Check if data is already extracted
    if extracted_dir.exists() and (extracted_dir / "data_batch_1").exists():
        if _is_main_process(data_dir):
            _log(f"CIFAR-10 dataset already available at {extracted_dir}")

        return True

    # Only the main process should log this
    if _is_main_process(data_dir):
        logger.info(f"CIFAR-10 dataset required at {extracted_dir}")

    # A file lock allows only one process to download the archive.
    try:
        return _download_cifar10_with_lock(data_dir, data_dir_path, tar_file, extracted_dir, lock_file, download_marker)

    except filelock.Timeout:
        # Only one process should show the waiting message
        if _is_main_process(data_dir):
            logger.info("Waiting for concurrent CIFAR download to complete...")

        # Wait and check if download completed while we were waiting
        for _ in range(10):
            time.sleep(5)  # Check every 5 seconds
            if extracted_dir.exists() and (extracted_dir / "data_batch_1").exists():
                if _is_main_process(data_dir):
                    _log(f"CIFAR-10 dataset now available at {extracted_dir}")

                return True

        if _is_main_process(data_dir):
            logger.info("Timed out waiting for CIFAR dataset download")

        return False

    except Exception as e:
        if _is_main_process(data_dir):
            logger.info(f"Error with CIFAR file lock: {e!s}")

        return False


def _resolve_cifar_data_dir() -> Path:
    """Return the preferred CIFAR data directory, falling back to system temp."""
    # Define a fixed, consistent data directory that will be used across processes
    project_root: Path = (pathlib.Path(__file__).parent / "../..").resolve()
    data_dir_path: Path = project_root / "data" / "cifar10"

    try:
        data_dir_path.mkdir(exist_ok=True, parents=True)
    except (OSError, PermissionError):
        # If we can't create in project directory, use a consistent system temp directory
        data_dir_path = pathlib.Path(tempfile.gettempdir()) / "spar_cifar10"
        data_dir_path.mkdir(exist_ok=True, parents=True)

    return data_dir_path


def _load_cifar10_images_from_dir(data_dir_path: Path) -> list[NDArray[np.float64]]:
    """Load CIFAR-10 image arrays from a prepared data directory."""
    # Path to extracted data
    extracted_dir: Path = data_dir_path / "cifar-10-batches-py"
    batch_file_path: Path = extracted_dir / "data_batch_1"

    # Only attempt download if needed
    downloaded: bool = batch_file_path.exists()

    if not downloaded:
        is_main: bool = _is_main_process(str(data_dir_path))
        if is_main:
            logger.info("CIFAR-10 dataset not found, downloading...")

        download_success: bool = download_cifar10(str(data_dir_path))

        if not download_success:
            _raise_download_failure()

    # Load the data batch - silent operation to reduce output noise
    with batch_file_path.open("rb") as f:
        data: dict[bytes, list[NDArray[np.uint8]]] = pickle.load(f, encoding="bytes")

    # Process CIFAR-10 data
    raw_images: list[NDArray[np.uint8]] = data[b"data"]
    # Reshape from flat (3072,) to (32, 32, 3) and normalize to [0, 1]
    images: list[NDArray[np.float64]] = [img.reshape(3, 32, 32).transpose(1, 2, 0) / 255.0 for img in raw_images]

    # Only one process should announce the successful loading
    if _is_main_process(str(data_dir_path)) and VERBOSE:
        logger.info(f"Loaded {len(images)} CIFAR-10 images")

    return images


def load_cifar10_images() -> list[NDArray[np.float64]]:
    """Load CIFAR-10 images for use as backgrounds.

    Returns:
        List of CIFAR-10 float32 images with pixel values in [0, 1].

    """
    if _CIFAR10_IMAGES_CACHE.images is not None:
        return _CIFAR10_IMAGES_CACHE.images

    data_dir_path: Path = pathlib.Path()
    try:
        data_dir_path = _resolve_cifar_data_dir()
        images: list[NDArray[np.float64]] = _load_cifar10_images_from_dir(data_dir_path)

    except Exception as e:
        if _is_main_process(str(data_dir_path)):
            logger.info(f"Error loading CIFAR-10 dataset: {e!s}")
            logger.info("Creating synthetic random images as fallback")

        images = []
        for _ in range(100):  # Generate 100 random "images"
            random_img: NDArray[np.float64] = np.random.rand(32, 32, 3)
            images.append(random_img)

        return images

    else:
        _CIFAR10_IMAGES_CACHE.images = images
        return images
