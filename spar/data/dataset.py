"""Dataset classes."""

from __future__ import annotations

from logging import getLogger
import pathlib
from typing import TYPE_CHECKING

import h5py
import numpy as np
from numpy import float32, float64, integer
import torch
from torch.utils.data import DataLoader, IterableDataset

from spar.utils.data_utils.hdf5_common import (
    open_hdf5_for_read,
    read_float_dataset,
    read_int64_dataset,
    read_variant_names_from_attrs,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from logging import Logger

    from numpy.random import Generator as NumpyGenerator
    from numpy.typing import NDArray
    from torch import Tensor


logger: Logger = getLogger(__name__)


class TransitionDataset(IterableDataset["dict[str, Tensor]"]):
    """Iterable dataset yielding full batches."""

    def __init__(
        self,
        states: NDArray[float32 | float64 | integer] | None = None,
        actions: NDArray[float32 | float64 | integer] | None = None,
        next_states: NDArray[float32 | float64 | integer] | None = None,
        file_path: str | None = None,
        *,
        batch_size: int,
        num_batches: int | None = None,
        replacement: bool = True,
        generator: torch.Generator | None = None,
        transform: Callable[[NDArray[float32 | float64 | integer]], NDArray[float32 | float64 | integer]] | None = None,
        dtype: torch.dtype = torch.float32,
        infinite: bool = False,
        base_only: bool = True,
        variations_to_use: list[str] | None = None,
        variations_to_ignore: list[str] | None = None,
    ) -> None:
        """Initialize the transition dataset.

        Args:
            states: State array of shape (num_transitions, *state_shape).
            actions: Action array of shape (num_transitions,).
            next_states: Next state array of shape (num_transitions, *state_shape).
            file_path: Path to HDF5 data file. Alternative to providing arrays directly.
            batch_size: Number of samples per batch.
            num_batches: Number of batches per epoch. If None, computed from dataset size.
                Ignored when infinite=True.
            replacement: Whether to sample with replacement.
            generator: Random number generator for reproducibility.
            transform: Optional transform function to apply to states.
            dtype: PyTorch dtype for tensor conversion.
            infinite: If True, the dataset will generate batches infinitely without stopping.
                Useful for continuous training without epoch boundaries.
            base_only: Whether to use only base images or include variations (for HDF5 loading).
            variations_to_use: Specific variation names to include (for HDF5 loading).
            variations_to_ignore: Variation names to exclude (for HDF5 loading).
        """
        self.batch_size: int = batch_size
        self.replacement: bool = replacement
        self.generator: torch.Generator | None = generator
        self.transform: (
            Callable[[NDArray[float32 | float64 | integer]], NDArray[float32 | float64 | integer]] | None
        ) = transform
        self.dtype: torch.dtype = dtype
        self.infinite: bool = infinite

        # HDF5 loading parameters
        self.base_only: bool = base_only
        self.variations_to_use: list[str] | None = variations_to_use
        self.variations_to_ignore: list[str] = variations_to_ignore or []

        # Initialize arrays
        self.states: NDArray[float32 | float64 | integer] | None = None
        self.actions: NDArray[float32 | float64 | integer] | None = None
        self.next_states: NDArray[float32 | float64 | integer] | None = None
        # Clean/target states (e.g., base variation) used for reconstruction targets
        self.target_states: NDArray[float32 | float64 | integer] | None = None
        self.target_next_states: NDArray[float32 | float64 | integer] | None = None
        self._size: int = 0

        # Load data
        if file_path is not None:
            self._load_from_hdf5(file_path)
        elif states is not None and actions is not None and next_states is not None:
            self._load_from_arrays(states, actions, next_states)
        else:
            raise ValueError("Either provide file_path or (states, actions, next_states)")

        # Compute number of batches (ignored if infinite)
        self.num_batches: int | None = None if infinite else num_batches or (self._size // self.batch_size)

        logger.info(f"Dataset loaded: {self._size} transitions")

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Iterate over dataset batches."""
        if self.infinite:
            # Infinite iteration mode - never stops generating batches
            return self._infinite_iter()

        # Finite iteration mode - generates num_batches batches
        return self._finite_iter()

    def _infinite_iter(self) -> Generator[dict[str, Tensor], None, None]:
        """Generate batches infinitely.

        Yields:
            Dictionary containing batched tensors for states, actions, and next states.
        """
        # Initialize random number generator
        seed: int | None = self.generator.initial_seed() if self.generator else None
        # precise generator type
        rng: NumpyGenerator = np.random.default_rng(seed)

        if self.replacement:
            # Infinite sampling with replacement
            while True:
                batch_indices: NDArray[np.intp] = rng.integers(0, self._size, size=self.batch_size, dtype=np.intp)
                yield self._get_batch(batch_indices)
        else:
            # Infinite sampling without replacement - reshuffle when exhausted
            current_perm: NDArray[np.intp] | None = None
            current_idx = 0

            while True:
                # Generate new permutation if needed
                if current_perm is None or current_idx + self.batch_size > len(current_perm):
                    if self.generator is not None:
                        current_perm = torch.randperm(self._size, generator=self.generator).numpy()
                    else:
                        current_perm = rng.permutation(self._size)
                    current_idx = 0

                # Get batch from current permutation
                batch_indices = current_perm[current_idx : current_idx + self.batch_size]
                current_idx += self.batch_size

                yield self._get_batch(batch_indices)

    def _finite_iter(self) -> Generator[dict[str, Tensor], None, None]:
        """Generate a finite number of batches.

        Yields:
            Dictionary containing batched tensors for states, actions, and next states.
        """
        # In finite mode, num_batches must be an int
        assert self.num_batches is not None
        num_batches: int = self.num_batches
        batch_indices: NDArray[np.intp]
        if self.replacement:
            # Pre-generate all indices
            seed: int | None = self.generator.initial_seed() if self.generator else None
            rng: NumpyGenerator = np.random.default_rng(seed)
            all_indices: NDArray[np.intp] = rng.integers(0, self._size, size=(num_batches, self.batch_size))

            for batch_indices in all_indices:
                yield self._get_batch(batch_indices)
        else:
            # Generate permutation once
            perm: NDArray[np.intp]
            if self.generator is not None:
                perm = torch.randperm(self._size, generator=self.generator).numpy()
            else:
                perm = np.random.permutation(self._size)

            # Generate batches from permutation
            for i in range(num_batches):
                start_idx: int = i * self.batch_size
                end_idx: int = start_idx + self.batch_size
                batch_indices = perm[start_idx:end_idx]
                yield self._get_batch(batch_indices)

    def _get_batch(self, indices: NDArray[np.intp]) -> dict[str, Tensor]:
        """Get a batch.

        Args:
            indices: Array of indices to fetch.

        Returns:
            Dictionary with batched tensors.
        """
        # Loading establishes non-None arrays for the iterator below.
        if self.states is None or self.actions is None or self.next_states is None:
            raise RuntimeError("Dataset arrays are not loaded.")
        # Slicing
        batch_states: NDArray[np.float32 | np.float64 | integer] = self.states[indices]
        batch_actions: NDArray[np.float32 | np.float64 | integer] = self.actions[indices]
        batch_next_states: NDArray[np.float32 | np.float64 | integer] = self.next_states[indices]
        # Clean targets (do not apply transforms)
        batch_target_states: NDArray[np.float32 | np.float64 | integer] = (
            self.target_states[indices] if self.target_states is not None else batch_states
        )
        batch_target_next_states: NDArray[np.float32 | np.float64 | integer] = (
            self.target_next_states[indices] if self.target_next_states is not None else batch_next_states
        )

        # Apply transform if provided
        if self.transform is not None:
            # Try vectorized transform first
            try:
                batch_states = self.transform(batch_states)
                batch_next_states = self.transform(batch_next_states)
            except (TypeError, ValueError):
                # Fallback to per-sample transform
                batch_states = np.stack([self.transform(s) for s in batch_states])
                batch_next_states = np.stack([self.transform(s) for s in batch_next_states])

        # Convert to tensors
        return {
            "states": torch.from_numpy(batch_states).to(dtype=self.dtype),
            "actions": torch.from_numpy(batch_actions).long(),
            "next_states": torch.from_numpy(batch_next_states).to(dtype=self.dtype),
            "target_states": torch.from_numpy(batch_target_states).to(dtype=self.dtype),
            "target_next_states": torch.from_numpy(batch_target_next_states).to(dtype=self.dtype),
        }

    def get_info(self) -> dict[str, int | tuple[int, ...] | np.dtype[np.generic] | torch.dtype | bool | None]:
        """Get dataset information."""
        if self._size == 0:
            return {"num_transitions": 0, "state_shape": None}

        state_shape: tuple[int, ...] | None = self.states.shape[1:] if self.states is not None else None
        state_dtype: np.dtype[np.generic] | None = self.states.dtype if self.states is not None else None
        action_dtype: np.dtype[np.generic] | None = self.actions.dtype if self.actions is not None else None

        return {
            "num_transitions": self._size,
            "state_shape": state_shape,
            "state_dtype": state_dtype,
            "action_dtype": action_dtype,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "infinite": self.infinite,
        }

    def __len__(self) -> int:
        """Return the number of batches per epoch.

        For infinite datasets, this returns a reasonable default for progress tracking.
        """
        if self.infinite:
            # Return a default number for progress tracking purposes
            return self._size // self.batch_size
        assert self.num_batches is not None
        return self.num_batches

    def _load_from_arrays(
        self,
        states: NDArray[float32 | float64 | integer],
        actions: NDArray[float32 | float64 | integer],
        next_states: NDArray[float32 | float64 | integer],
        *,
        target_states: NDArray[float32 | float64 | integer] | None = None,
        target_next_states: NDArray[float32 | float64 | integer] | None = None,
    ) -> None:
        """Load data from NumPy arrays."""
        # Store contiguous arrays so each slice is a view when possible.
        self.states = np.ascontiguousarray(states)
        self.actions = np.ascontiguousarray(actions)
        self.next_states = np.ascontiguousarray(next_states)
        # If explicit reconstruction targets are not provided, default to inputs
        self.target_states = np.ascontiguousarray(target_states) if target_states is not None else self.states
        self.target_next_states = (
            np.ascontiguousarray(target_next_states) if target_next_states is not None else self.next_states
        )

        # Validate input shapes
        assert self.states is not None
        assert self.actions is not None
        assert self.next_states is not None
        assert self.target_states is not None
        assert self.target_next_states is not None
        if not (
            self.states.shape[0]
            == self.actions.shape[0]
            == self.next_states.shape[0]
            == self.target_states.shape[0]
            == self.target_next_states.shape[0]
        ):
            raise ValueError("States, actions, next_states, and targets must have the same length")

        self._size = self.states.shape[0]

    @staticmethod
    def _scan_variant_names(file: h5py.File) -> list[str]:
        """Fallback: infer variant names by scanning the first episode group.

        We look for sub-groups that contain a 'states' dataset and are not 'actions'.
        """
        inferred: set[str] = set()
        episodes_obj = file.get("episodes")
        if not isinstance(episodes_obj, h5py.Group):
            return []
        for ep_key in sorted(episodes_obj):
            ep_obj: h5py.Group | h5py.Dataset | h5py.Datatype = episodes_obj[ep_key]
            if not isinstance(ep_obj, h5py.Group):
                continue
            for k in ep_obj:
                if k == "actions":
                    continue
                candidate = ep_obj.get(k)
                if isinstance(candidate, h5py.Group) and "states" in candidate:
                    # HDF5 keys are normalized as plain Python strings.
                    inferred.add(str(k))
            # If we already found some, we can stop after first informative episode
            if inferred:
                break
        return sorted(inferred)

    def _load_from_hdf5(self, file_path: str) -> None:
        """Load data from HDF5 file with vectorized operations."""
        if not pathlib.Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open_hdf5_for_read(file_path) as f:
            variant_names: list[str] = read_variant_names_from_attrs(f.attrs)

            if not variant_names:
                variant_names = self._scan_variant_names(f)
                if variant_names:
                    logger.info(
                        f"Inferred variant names {variant_names} for file '{file_path}' "
                        "(attribute missing or unreadable)"
                    )
                else:
                    logger.error(f"Could not determine variant names for file '{file_path}'. Dataset will be empty.")
                    # Leave this group empty. The remaining pipeline creates empty arrays.

            # Determine which variants to process
            variants_to_process: list[str] = self._get_variants_to_process(variant_names)

            use_float32: bool = self.dtype == torch.float32
            state_numpy_dtype: type[np.float32 | np.float64] = np.float32 if use_float32 else np.float64

            episodes_obj = f.get("episodes")
            if not isinstance(episodes_obj, h5py.Group):
                logger.error(f"Invalid or missing 'episodes' group in '{file_path}'. Dataset will be empty.")
                self.states = np.array([], dtype=state_numpy_dtype)
                self.actions = np.array([], dtype=np.int64)
                self.next_states = np.array([], dtype=state_numpy_dtype)
                self.target_states = np.array([], dtype=state_numpy_dtype)
                self.target_next_states = np.array([], dtype=state_numpy_dtype)
                self._size = 0
                return

            # Pass 1: index the (episode, variant) entries from shape metadata
            # only, so the output arrays can be preallocated once. This avoids
            # the previous accumulate-then-concatenate pattern, which held two
            # full copies of the dataset in memory at its peak.
            entries: list[tuple[str, str, int, int]] = []
            state_shape: tuple[int, ...] | None = None
            total_transitions: int = 0
            total_actions: int = 0
            for episode_name in sorted(episodes_obj.keys()):
                episode_obj = episodes_obj.get(episode_name)
                if not isinstance(episode_obj, h5py.Group):
                    continue

                actions_obj = episode_obj.get("actions")
                if not isinstance(actions_obj, h5py.Dataset):
                    continue
                num_actions: int = int(actions_obj.shape[0]) if actions_obj.ndim > 0 else 0
                if num_actions == 0:
                    continue

                for variant_name in variants_to_process:
                    variant_obj = episode_obj.get(variant_name)
                    if not isinstance(variant_obj, h5py.Group):
                        continue
                    states_obj = variant_obj.get("states")
                    if not isinstance(states_obj, h5py.Dataset):
                        continue
                    num_states: int = int(states_obj.shape[0])
                    if num_states <= 1:
                        continue

                    if state_shape is None:
                        state_shape = tuple(states_obj.shape[1:])
                    entries.append((episode_name, variant_name, num_states - 1, num_actions))
                    total_transitions += num_states - 1
                    total_actions += num_actions

            if not entries or state_shape is None:
                # Empty dataset
                self.states = np.array([], dtype=state_numpy_dtype)
                self.actions = np.array([], dtype=np.int64)
                self.next_states = np.array([], dtype=state_numpy_dtype)
                self.target_states = np.array([], dtype=state_numpy_dtype)
                self.target_next_states = np.array([], dtype=state_numpy_dtype)
                self._size = 0
                return

            # Preallocate the output arrays and fill them slice by slice.
            states = np.empty((total_transitions, *state_shape), dtype=state_numpy_dtype)
            next_states = np.empty_like(states)
            target_states = np.empty_like(states)
            target_next_states = np.empty_like(states)
            actions = np.empty(total_actions, dtype=np.int64)

            # Pass 2: read each episode once and write into the output slices.
            row: int = 0
            action_row: int = 0
            current_episode: str | None = None
            actions_ep: NDArray[np.int64] = np.empty(0, dtype=np.int64)
            base_variant_data: NDArray[np.float32 | np.float64] | None = None
            for episode_name, variant_name, num_transitions, num_actions in entries:
                episode_obj = episodes_obj.get(episode_name)
                if not isinstance(episode_obj, h5py.Group):
                    continue

                if episode_name != current_episode:
                    current_episode = episode_name
                    actions_obj = episode_obj.get("actions")
                    if not isinstance(actions_obj, h5py.Dataset):
                        continue
                    actions_ep = read_int64_dataset(actions_obj)

                    # Fetch base variation once for this episode (if available)
                    # to serve as clean target.
                    base_variant_data = None
                    base_obj = episode_obj.get("base")
                    if isinstance(base_obj, h5py.Group):
                        states_obj = base_obj.get("states")
                        if isinstance(states_obj, h5py.Dataset):
                            if use_float32:
                                base_variant_data = read_float_dataset(states_obj, np.float32)
                            else:
                                base_variant_data = read_float_dataset(states_obj, np.float64)

                variant_obj = episode_obj.get(variant_name)
                if not isinstance(variant_obj, h5py.Group):
                    continue
                states_obj = variant_obj.get("states")
                if not isinstance(states_obj, h5py.Dataset):
                    continue
                variant_data: NDArray[np.float32 | np.float64]
                if use_float32:
                    variant_data = read_float_dataset(states_obj, np.float32)
                else:
                    variant_data = read_float_dataset(states_obj, np.float64)

                end: int = row + num_transitions
                states[row:end] = variant_data[:-1]
                next_states[row:end] = variant_data[1:]

                # Determine clean targets: prefer base variation if available and shape-compatible
                if base_variant_data is not None and base_variant_data.shape[0] == variant_data.shape[0]:
                    target_states[row:end] = base_variant_data[:-1]
                    target_next_states[row:end] = base_variant_data[1:]
                else:
                    # Fallback to variant itself if base is missing or length-mismatched
                    if base_variant_data is not None and base_variant_data.shape[0] != variant_data.shape[0]:
                        logger.warning(
                            f"Base variant length ({base_variant_data.shape[0]}) != {variant_name} "
                            f"variant length ({variant_data.shape[0]}) in {file_path}/{episode_name}. "
                            "using variant as reconstruction target."
                        )
                    target_states[row:end] = variant_data[:-1]
                    target_next_states[row:end] = variant_data[1:]

                actions[action_row : action_row + num_actions] = actions_ep
                row = end
                action_row += num_actions

            self.states = states[:row] if row != total_transitions else states
            self.actions = actions[:action_row] if action_row != total_actions else actions
            self.next_states = next_states[:row] if row != total_transitions else next_states
            self.target_states = target_states[:row] if row != total_transitions else target_states
            self.target_next_states = target_next_states[:row] if row != total_transitions else target_next_states
            self._size = row

    def _get_variants_to_process(self, variant_names: list[str]) -> list[str]:
        """Determine which variants to process."""
        if self.base_only:
            return ["base"] if "base" in variant_names else variant_names[:1]

        variants: list[str] = variant_names.copy()

        if self.variations_to_use is not None:
            variants = [v for v in variants if v in self.variations_to_use]

        if self.variations_to_ignore:
            variants = [v for v in variants if v not in self.variations_to_ignore]

        if not variants:
            raise ValueError(f"No variants available after filtering: {variant_names}")

        return variants


def create_dataloader(
    states: NDArray[float32 | float64 | integer] | None = None,
    actions: NDArray[float32 | float64 | integer] | None = None,
    next_states: NDArray[float32 | float64 | integer] | None = None,
    file_path: str | None = None,
    batch_size: int = 32,
    *,
    num_batches_per_epoch: int | None = None,
    replacement: bool = True,
    generator: torch.Generator | None = None,
    transform: Callable[[NDArray[float32 | float64 | integer]], NDArray[float32 | float64 | integer]] | None = None,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    pin_memory_device: str = "",
    infinite: bool = False,
    base_only: bool = True,
    variations_to_use: list[str] | None = None,
    variations_to_ignore: list[str] | None = None,
) -> DataLoader[dict[str, Tensor]]:
    """Create a DataLoader from numpy arrays or HDF5 file.

    Args:
        states: State array of shape (num_transitions, *state_shape).
        actions: Action array of shape (num_transitions,).
        next_states: Next state array of shape (num_transitions, *state_shape).
        file_path: Path to HDF5 data file. Alternative to providing arrays directly.
        batch_size: Number of samples per batch.
        num_batches_per_epoch: Number of batches per epoch. Ignored if infinite=True.
        replacement: Whether to sample with replacement.
        generator: Random number generator for reproducibility.
        transform: Optional transform function to apply to states.
        dtype: PyTorch dtype for tensor conversion.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.
        persistent_workers: Whether to keep workers alive between epochs.
        prefetch_factor: Number of batches to prefetch per worker when workers are enabled.
        pin_memory_device: Device for pinned memory.
        infinite: If True, the DataLoader will generate batches infinitely.
        base_only: Whether to use only base images or include variations (for HDF5 loading).
        variations_to_use: Specific variation names to include (for HDF5 loading).
        variations_to_ignore: Variation names to exclude (for HDF5 loading).

    Returns:
        DataLoader using TransitionDataset.
    """
    batch_dataset: TransitionDataset = TransitionDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        file_path=file_path,
        batch_size=batch_size,
        num_batches=num_batches_per_epoch,
        replacement=replacement,
        generator=generator,
        transform=transform,
        dtype=dtype,
        infinite=infinite,
        base_only=base_only,
        variations_to_use=variations_to_use,
        variations_to_ignore=variations_to_ignore,
    )

    effective_persistent_workers: bool = persistent_workers if num_workers > 0 else False
    if num_workers > 0 and prefetch_factor is not None:
        dataloader: DataLoader[dict[str, Tensor]] = DataLoader(
            batch_dataset,
            batch_size=None,  # Batching handled by TransitionDataset
            shuffle=False,  # Shuffling handled by TransitionDataset
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=effective_persistent_workers,
            pin_memory_device=pin_memory_device,
            prefetch_factor=prefetch_factor,
        )
    else:
        dataloader = DataLoader(
            batch_dataset,
            batch_size=None,  # Batching handled by TransitionDataset
            shuffle=False,  # Shuffling handled by TransitionDataset
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=effective_persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    mode_str: str = "infinite" if infinite else f"finite ({batch_dataset.num_batches} batches)"
    logger.info(f"Created DataLoader: batch_size={batch_size}, mode={mode_str}")

    return dataloader
