"""Alignment dataset classes for loading variation data and mapping to base states."""

from __future__ import annotations

from logging import getLogger
import pathlib
from typing import TYPE_CHECKING, Protocol, TypedDict

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from spar.utils.data_utils.hdf5_common import (
    open_hdf5_for_read,
    read_float_dataset,
    read_int64_dataset,
    read_variant_names_from_attrs,
    read_variant_type_from_attrs,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from logging import Logger

    from numpy import floating as np_floating, integer as np_integer
    from numpy.random import Generator as np_Generator
    from numpy.typing import NDArray
    from torch import Generator as torch_Generator, Tensor
    from torch.nn import Module


__all__: list[str] = [
    "AlignmentDataset",
    "AlignmentDatasetInfo",
    "H5PyDataset",
    "TransformProtocol",
    "create_dataloader",
]

logger: Logger = getLogger(__name__)


# Type protocols for better type safety
class TransformProtocol(Protocol):
    """Protocol for transform functions that can handle both single arrays and batches.

    Transforms operate on floating-point state/images. They may accept a single
    example or a batch, but the element dtype is expected to be floating point.
    """

    def __call__(self, data: NDArray[np_floating]) -> NDArray[np_floating]:
        """Transform input data array."""
        ...


class H5PyDataset(Protocol):
    """Protocol for the array operations used from h5py datasets."""

    def __getitem__(self, key: int | slice) -> NDArray[np.generic]:
        """Get data by key/slice."""
        ...

    def __len__(self) -> int:
        """Return length of dataset."""
        ...

    @property
    def dtype(self) -> np.dtype:
        """Dataset dtype.

        Returns:
            np.dtype: Dataset dtype.
        """
        ...

    def astype(self, dtype: np.dtype, copy: bool = True) -> NDArray[np.generic]:
        """Convert dataset to specified dtype."""
        ...


class AlignmentDatasetInfo(TypedDict, total=False):
    """TypedDict for alignment dataset information."""

    num_pairs: int
    state_shape: tuple[int, ...] | None
    state_dtype: np.dtype | None
    batch_size: int
    num_batches: int | None
    infinite: bool
    precompute_targets: bool
    use_next_state_targets: bool
    encoded_targets_shape: tuple[int, ...]
    encoded_targets_dtype: np.dtype
    actions_shape: tuple[int, ...]
    actions_dtype: np.dtype


class AlignmentDataset(IterableDataset["dict[str, Tensor]"]):
    """Iterable dataset for alignment training with variations mapped to base states."""

    def __init__(
        self,
        file_path: str,
        *,
        batch_size: int,
        num_batches: int | None = None,
        replacement: bool = True,
        generator: torch_Generator | None = None,
        transform: TransformProtocol | None = None,
        dtype: torch.dtype = torch.float32,
        infinite: bool = False,
        variations_to_use: list[str] | None = None,
        variations_to_ignore: list[str] | None = None,
        encoder: Module | None = None,
        precompute_targets: bool = False,
        device: str | torch.device = "cpu",
        use_next_state_targets: bool = False,
    ) -> None:
        """Initialize the alignment dataset.

        Args:
            file_path: Path to HDF5 data file containing variation data.
            batch_size: Number of samples per batch.
            num_batches: Number of batches per epoch. If None, computed from dataset size.
                Ignored when infinite=True.
            replacement: Whether to sample with replacement.
            generator: Random number generator for reproducibility.
            transform: Optional transform function to apply to states.
            dtype: PyTorch dtype for tensor conversion.
            infinite: If True, the dataset will generate batches infinitely without stopping.
            variations_to_use: Specific variation names to include.
            variations_to_ignore: Variation names to exclude.
            encoder: Optional pretrained encoder to precompute targets. Required if precompute_targets=True.
            precompute_targets: If True, use encoder to precompute encoded targets for base states.
                Note: Encoded targets are not returned when use_next_state_targets=True.
            device: Device to use for encoder computations when precomputing targets.
            use_next_state_targets: If True, target states will be the base version of the NEXT state
                instead of the base version of the current state. This also returns actions.
                Raw base images are returned as targets, not encoded versions.
        """
        self.file_path: str = file_path
        self.batch_size: int = batch_size
        self.replacement: bool = replacement
        self.generator: torch_Generator | None = generator
        self.transform: TransformProtocol | None = transform
        self.dtype: torch.dtype = dtype
        self.infinite: bool = infinite
        self.variations_to_use: list[str] | None = variations_to_use
        self.variations_to_ignore: list[str] = variations_to_ignore or []
        self.use_next_state_targets: bool = use_next_state_targets

        # Call parent initializer for IterableDataset
        super().__init__()

        # Precomputing parameters
        self.encoder: Module | None = encoder
        self.precompute_targets: bool = precompute_targets
        if isinstance(device, str):
            device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.device: torch.device = device

        # Validate encoder requirement
        if self.precompute_targets and self.encoder is None:
            raise ValueError("encoder must be provided when precompute_targets=True")

        # Warn if precompute_targets is True but use_next_state_targets is also True
        if self.precompute_targets and self.use_next_state_targets:
            logger.warning(
                "precompute_targets=True has no effect when use_next_state_targets=True. "
                "Raw base images will be returned as targets instead of encoded versions."
            )

        # Initialize state arrays - using explicit types
        self.batch_states: NDArray[np_floating] = np.array([])
        self.batch_base_states: NDArray[np_floating] = np.array([])
        self.batch_encoded_targets: NDArray[np_floating] | None = None  # Precomputed encoded targets
        self.state_to_base_mapping: NDArray[np_integer] = (
            np.array([])
        )  # Maps variation state indices to base state indices
        self.batch_actions: NDArray[np_integer] | None = None  # Actions for next state targets mode
        self._size: int = 0

        # Load data
        self._load_from_hdf5()

        # Compute number of batches (ignored if infinite)
        self.num_batches: int | None
        if infinite:
            self.num_batches = None  # Infinite batches
        else:
            # Ceiling division includes a final partial batch.
            # The last batch might be smaller than batch_size
            self.num_batches = num_batches or ((self._size + self.batch_size - 1) // self.batch_size)

        targets_mode: str = "next_state" if self.use_next_state_targets else "current_state"
        logger.info(
            f"AlignmentDataset loaded: {self._size} state pairs, {self.num_batches} batches/epoch, "
            f"targets_mode={targets_mode}, infinite={infinite}"
        )

    def __iter__(self) -> Generator[dict[str, Tensor], None, None]:
        """Iterate over batches of data.

        Yields:
            Dict containing batched tensors for batch_states and batch_base_states.
        """
        if self.infinite:
            yield from self._infinite_iter()
        else:
            yield from self._finite_iter()

    def _infinite_iter(self) -> Generator[dict[str, Tensor], None, None]:
        """Generate batches infinitely.

        Yields:
            Batch dictionaries containing source and target state tensors.
        """
        seed: int | None = self.generator.initial_seed() if self.generator else None
        rng: np_Generator = np.random.default_rng(seed)

        if self.replacement:
            batch_indices: NDArray[np_integer]
            current_perm: NDArray[np_integer] | None = None
            while True:
                if self.generator is not None:
                    # Use torch generator for consistency
                    batch_indices = torch.randint(0, self._size, (self.batch_size,), generator=self.generator).numpy()
                else:
                    batch_indices = rng.integers(0, self._size, size=self.batch_size)
                yield self._get_batch(batch_indices)
        else:
            current_perm = None
            current_idx = 0

            while True:
                if current_perm is None or current_idx + self.batch_size > len(current_perm):
                    if self.generator is not None:
                        current_perm = torch.randperm(self._size, generator=self.generator).numpy()
                    else:
                        current_perm = rng.permutation(self._size)
                    current_idx = 0

                assert current_perm is not None
                batch_indices = current_perm[current_idx : current_idx + self.batch_size]
                current_idx += self.batch_size
                yield self._get_batch(batch_indices)

    def _finite_iter(self) -> Generator[dict[str, Tensor], None, None]:
        """Generate a finite number of batches.

        Yields:
            Batch dictionaries containing source and target state tensors.
        """
        # Finite iteration requires an explicit batch count.
        if self.num_batches is None:
            raise ValueError("num_batches must be set for finite iteration")

        batch_indices: NDArray[np_integer]
        if self.replacement:
            seed: int | None
            rng: np_Generator
            all_indices: NDArray[np_integer]
            if self.generator is not None:
                # Pre-generate all indices using torch generator for consistency
                all_indices = torch.randint(
                    0, self._size, (self.num_batches, self.batch_size), generator=self.generator
                ).numpy()
            else:
                seed = self.generator.initial_seed() if self.generator else None
                rng = np.random.default_rng(seed)
                all_indices = rng.integers(0, self._size, size=(self.num_batches, self.batch_size))

            for batch_indices in all_indices:
                yield self._get_batch(batch_indices)
        else:
            perm: NDArray[np_integer] | Tensor
            start_idx: int
            end_idx: int
            if self.generator is not None:
                perm = torch.randperm(self._size, generator=self.generator).numpy()
            else:
                perm = np.random.permutation(self._size)

            for i in range(self.num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(perm))  # Don't go beyond dataset size
                if start_idx >= len(perm):
                    break  # No more data
                batch_indices = perm[start_idx:end_idx]
                yield self._get_batch(batch_indices)

    def _get_batch(self, indices: NDArray[np_integer]) -> dict[str, Tensor]:
        """Get a batch of states and their corresponding base states.

        Args:
            indices: Array of indices to fetch.

        Returns:
            Dictionary with batched tensors for batch_states, batch_base_states, and optionally
            batch_encoded_targets (only when not using next state targets mode) and batch_actions.
        """
        # Get variation states and corresponding base states
        states_batch: NDArray[np.float32] = self.batch_states[indices]
        base_indices: NDArray[np_integer] = self.state_to_base_mapping[indices]
        base_states_batch: NDArray[np.float32] = self.batch_base_states[base_indices]

        # Apply transform if provided
        if self.transform is not None:
            try:
                states_batch = self.transform(states_batch)
                base_states_batch = self.transform(base_states_batch)
            except (TypeError, ValueError):
                # Fallback to per-sample transform
                states_batch = np.stack([self.transform(s) for s in states_batch])
                base_states_batch = np.stack([self.transform(s) for s in base_states_batch])

        # Prepare result dictionary
        result: dict[str, Tensor] = {
            "batch_states": torch.from_numpy(states_batch).to(dtype=self.dtype),
            "batch_base_states": torch.from_numpy(base_states_batch).to(dtype=self.dtype),
        }

        # Add actions if using next state targets mode
        if self.use_next_state_targets and self.batch_actions is not None:
            actions_batch: NDArray[np_integer] = self.batch_actions[indices]
            result["batch_actions"] = torch.from_numpy(actions_batch).to(dtype=torch.long)

        # Add precomputed encoded targets if available (but not when using next state targets mode)
        if self.precompute_targets and self.batch_encoded_targets is not None and not self.use_next_state_targets:
            encoded_targets_batch: NDArray[np.float32] = self.batch_encoded_targets[base_indices]
            result["batch_encoded_targets"] = torch.from_numpy(encoded_targets_batch).to(dtype=self.dtype)

        return result

    def get_info(self) -> AlignmentDatasetInfo:
        """Get dataset information."""
        if self._size == 0:
            return {"num_pairs": 0, "state_shape": None}

        info: AlignmentDatasetInfo = {
            "num_pairs": self._size,
            "state_shape": self.batch_states.shape[1:],
            "state_dtype": self.batch_states.dtype,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "infinite": self.infinite,
            "precompute_targets": self.precompute_targets,
            "use_next_state_targets": self.use_next_state_targets,
        }

        if self.precompute_targets and self.batch_encoded_targets is not None and not self.use_next_state_targets:
            info["encoded_targets_shape"] = tuple(self.batch_encoded_targets.shape[1:])
            info["encoded_targets_dtype"] = self.batch_encoded_targets.dtype

        if self.use_next_state_targets and self.batch_actions is not None:
            info["actions_shape"] = tuple(self.batch_actions.shape[1:]) if self.batch_actions.ndim > 1 else ()
            info["actions_dtype"] = self.batch_actions.dtype

        return info

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        if self.infinite:
            return self._size // self.batch_size

        # Finite datasets report their configured batch count.
        if self.num_batches is None:
            return self._size // self.batch_size

        return self.num_batches

    def _load_from_hdf5(self) -> None:
        """Load data from HDF5 file."""
        if not pathlib.Path(self.file_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        with open_hdf5_for_read(self.file_path) as f:
            # Read metadata
            variant_names: list[str] = read_variant_names_from_attrs(f.attrs)
            variant_type: str = read_variant_type_from_attrs(f.attrs)

            # Every alignment dataset requires the base variant.
            if "base" not in variant_names:
                raise ValueError(f"Base variant not found in dataset. Available variants: {variant_names}")

            # Determine which variants to process
            variants_to_process: list[str] = self._get_variants_to_process(variant_names)

            # Collect all states and mappings
            all_states: list[NDArray[np_floating]] = []
            all_mappings: list[NDArray[np_integer]] = []
            all_actions: list[NDArray[np_integer]] = []  # For next state targets mode

            episodes_group: h5py.Group | h5py.Dataset | h5py.Datatype = f["episodes"]
            if not isinstance(episodes_group, h5py.Group):
                raise TypeError(f"Expected 'episodes' to be a group, got {type(episodes_group)}")

            numpy_dtype: type[np_floating] = np.float32 if self.dtype == torch.float32 else np.float64

            if self.use_next_state_targets:
                # Next state targets mode: each state maps to next base state
                self._load_next_state_mode(
                    episodes_group,
                    variants_to_process,
                    variant_type,
                    numpy_dtype,
                    all_states,
                    all_mappings,
                    all_actions,
                )
            else:
                # Current state targets mode: each state maps to corresponding base state
                self._load_current_state_mode(
                    episodes_group, variants_to_process, variant_type, numpy_dtype, all_states, all_mappings
                )

            # Concatenate all data
            if all_states:
                self.batch_states = np.concatenate(all_states, axis=0)
                self.state_to_base_mapping = np.concatenate(all_mappings, axis=0)
                self._size = len(self.batch_states)

                if self.use_next_state_targets:
                    self.batch_actions = np.concatenate(all_actions, axis=0)

                # Keep the arrays contiguous so each slice is a view when possible.
                self.batch_states = np.ascontiguousarray(self.batch_states)
                self.state_to_base_mapping = np.ascontiguousarray(self.state_to_base_mapping)

                if self.use_next_state_targets:
                    self.batch_actions = np.ascontiguousarray(self.batch_actions)

                # Precompute encoded targets if requested
                if self.precompute_targets:
                    self._precompute_encoded_targets()
            else:
                # Empty dataset
                self.batch_states = np.array([], dtype=numpy_dtype)
                self.batch_base_states = np.array([], dtype=numpy_dtype)
                self.state_to_base_mapping = np.array([], dtype=np.int64)
                self.batch_encoded_targets = np.array([], dtype=numpy_dtype)
                if self.use_next_state_targets:
                    self.batch_actions = np.array([], dtype=np.int64)
                self._size = 0

    def _load_current_state_mode(
        self,
        episodes_group: h5py.Group,
        variants_to_process: list[str],
        variant_type: str,
        numpy_dtype: type[np_floating],
        all_states: list[NDArray[np_floating]],
        all_mappings: list[NDArray[np_integer]],
    ) -> None:
        """Load data for current state targets mode."""
        # Collect base states for one indexed read.
        base_states_list: list[NDArray[np_floating]] = []
        episode_base_offsets: dict[str, int] = {}
        total_base_size = 0

        for episode_name in sorted(episodes_group.keys()):
            episode_group: h5py.Group | h5py.Dataset | h5py.Datatype = episodes_group[episode_name]
            if not isinstance(episode_group, h5py.Group):
                continue

            if "base" in episode_group:
                base_group: h5py.Group | h5py.Dataset | h5py.Datatype = episode_group["base"]
                if isinstance(base_group, h5py.Group) and "states" in base_group:
                    states_dataset: h5py.Group | h5py.Dataset | h5py.Datatype = base_group["states"]
                    if isinstance(states_dataset, h5py.Dataset):
                        base_data: NDArray[np_floating] = read_float_dataset(states_dataset, numpy_dtype)
                        if len(base_data) > 0:
                            base_states_list.append(base_data)
                            episode_base_offsets[episode_name] = total_base_size
                            total_base_size += len(base_data)

        if not base_states_list:
            raise ValueError("No base states found in dataset")

        # Concatenate all base states
        self.batch_base_states = np.concatenate(base_states_list, axis=0)

        # Process variations and create mappings
        for episode_name in sorted(episodes_group.keys()):
            episode_group = episodes_group[episode_name]
            if not isinstance(episode_group, h5py.Group):
                continue

            if episode_name not in episode_base_offsets:
                continue
            base_offset: int = episode_base_offsets[episode_name]

            if "base" not in episode_group:
                continue

            base_group = episode_group["base"]
            if not isinstance(base_group, h5py.Group) or "states" not in base_group:
                continue

            states_dataset = base_group["states"]
            if not isinstance(states_dataset, h5py.Dataset):
                continue

            episode_base_data: NDArray[np_floating] = read_float_dataset(states_dataset, numpy_dtype)
            episode_base_size: int = len(episode_base_data)

            if episode_base_size == 0:
                continue

            # Process each variant for this episode
            for variant_name in variants_to_process:
                if variant_name not in episode_group:
                    continue

                variant_group: h5py.Group | h5py.Dataset | h5py.Datatype = episode_group[variant_name]
                if not isinstance(variant_group, h5py.Group) or "states" not in variant_group:
                    continue

                variant_states_dataset: h5py.Group | h5py.Dataset | h5py.Datatype = variant_group["states"]
                if not isinstance(variant_states_dataset, h5py.Dataset):
                    continue

                variant_data: NDArray[np_floating] = read_float_dataset(variant_states_dataset, numpy_dtype)
                if len(variant_data) == 0:
                    continue

                if variant_type == "first_state_only":
                    raise ValueError(f"Alignment dataset requires variations for all states in episode {episode_name}")

                # Pair only the common prefix of variant and base states.
                min_length: int = min(len(variant_data), episode_base_size)
                if min_length > 0:
                    all_states.append(variant_data[:min_length])

                    # Create mapping indices for this variant to corresponding base states
                    variant_to_base_indices = np.arange(base_offset, base_offset + min_length)
                    all_mappings.append(variant_to_base_indices)

    def _load_next_state_mode(
        self,
        episodes_group: h5py.Group,
        variants_to_process: list[str],
        variant_type: str,
        numpy_dtype: type[np_floating],
        all_states: list[NDArray[np_floating]],
        all_mappings: list[NDArray[np_integer]],
        all_actions: list[NDArray[np_integer]],
    ) -> None:
        """Load data for next state targets mode."""
        # Collect base states for one indexed read.
        base_states_list: list[NDArray[np_floating]] = []
        episode_base_offsets: dict[str, int] = {}
        total_base_size: int = 0

        # First pass: collect all base states
        for episode_name in sorted(episodes_group.keys()):
            episode_group: h5py.Group | h5py.Dataset | h5py.Datatype = episodes_group[episode_name]
            if not isinstance(episode_group, h5py.Group):
                continue

            if "base" in episode_group:
                base_group: h5py.Group | h5py.Dataset | h5py.Datatype = episode_group["base"]
                if isinstance(base_group, h5py.Group) and "states" in base_group:
                    states_dataset: h5py.Group | h5py.Dataset | h5py.Datatype = base_group["states"]
                    if isinstance(states_dataset, h5py.Dataset):
                        base_data: NDArray[np_floating] = read_float_dataset(states_dataset, numpy_dtype)
                        if len(base_data) > 0:
                            base_states_list.append(base_data)
                            episode_base_offsets[episode_name] = total_base_size
                            total_base_size += len(base_data)

        if not base_states_list:
            raise ValueError("No base states found in dataset")

        # Concatenate all base states
        self.batch_base_states = np.concatenate(base_states_list, axis=0)

        # Second pass: process variations and create mappings for next states
        for episode_name in sorted(episodes_group.keys()):
            episode_group = episodes_group[episode_name]
            if not isinstance(episode_group, h5py.Group):
                continue

            if episode_name not in episode_base_offsets:
                continue
            base_offset: int = episode_base_offsets[episode_name]

            if "base" not in episode_group:
                continue

            # Load actions for this episode
            if "actions" not in episode_group:
                continue

            actions_dataset: h5py.Group | h5py.Dataset | h5py.Datatype = episode_group["actions"]
            if not isinstance(actions_dataset, h5py.Dataset):
                continue

            actions: NDArray[np_integer] = read_int64_dataset(actions_dataset)
            if len(actions) == 0:
                continue

            base_group = episode_group["base"]
            if not isinstance(base_group, h5py.Group) or "states" not in base_group:
                continue

            states_dataset = base_group["states"]
            if not isinstance(states_dataset, h5py.Dataset):
                continue

            episode_base_data: NDArray[np_floating] = read_float_dataset(states_dataset, numpy_dtype)
            episode_base_size: int = len(episode_base_data)

            if episode_base_size <= 1:  # Need at least 2 states for state -> next_state pairs
                continue

            # Process each variant for this episode
            for variant_name in variants_to_process:
                if variant_name not in episode_group:
                    continue

                variant_group: h5py.Group | h5py.Dataset | h5py.Datatype = episode_group[variant_name]
                if not isinstance(variant_group, h5py.Group) or "states" not in variant_group:
                    continue

                variant_states_dataset: h5py.Group | h5py.Dataset | h5py.Datatype = variant_group["states"]
                if not isinstance(variant_states_dataset, h5py.Dataset):
                    continue

                variant_data: NDArray[np_floating] = read_float_dataset(variant_states_dataset, numpy_dtype)
                if len(variant_data) <= 1:
                    continue

                if variant_type == "first_state_only":
                    raise ValueError(
                        f"Next state targets mode requires variations for all states in episode {episode_name}"
                    )

                # For next state targets, we use state[t] -> base_state[t+1]
                # Map each state except the last to the following base state.
                max_transition_length = min(len(variant_data) - 1, len(actions), episode_base_size - 1)

                if max_transition_length > 0:
                    # Current states (exclude last state since it has no next state)
                    current_states = variant_data[:max_transition_length]
                    all_states.append(current_states)

                    # Actions corresponding to these transitions
                    episode_actions = actions[:max_transition_length].astype(np.int64)
                    all_actions.append(episode_actions)

                    # Mapping to next base states (base_states[1:])
                    next_base_indices = np.arange(base_offset + 1, base_offset + 1 + max_transition_length)
                    all_mappings.append(next_base_indices)

    def _precompute_encoded_targets(self) -> None:
        """Precompute encoded targets using the provided encoder."""
        if self.use_next_state_targets:
            logger.info("Skipping encoded targets precomputation in next state targets mode")
            return

        if self.encoder is None:
            logger.warning("No encoder provided for precomputing targets")
            return

        logger.info(
            f"Precomputing encoded targets using provided encoder (file={pathlib.Path(self.file_path).name})..."
        )

        # Move encoder to device and set to eval mode
        assert self.encoder is not None
        self.encoder = self.encoder.to(self.device)
        assert self.encoder is not None
        self.encoder.eval()

        # Determine dtype for encoded targets
        numpy_dtype: type[np_floating] = np.float32 if self.dtype == torch.float32 else np.float64

        with torch.inference_mode():
            # Convert base states to tensor
            base_states_tensor = torch.from_numpy(self.batch_base_states).to(dtype=self.dtype, device=self.device)

            # Encode in batches to avoid memory issues
            encoded_targets_list: list[NDArray[np_floating]] = []
            batch_size: int = min(1000, len(base_states_tensor))  # Use smaller batches for encoding

            for i in range(0, len(base_states_tensor), batch_size):
                batch_base_states = base_states_tensor[i : i + batch_size]
                # Encode base states
                assert self.encoder is not None
                encoded_batch: Tensor = self.encoder(batch_base_states)
                encoded_targets_list.append(encoded_batch.cpu().numpy().astype(numpy_dtype))

        # Concatenate all encoded targets
        self.batch_encoded_targets = np.concatenate(encoded_targets_list, axis=0)
        self.batch_encoded_targets = np.ascontiguousarray(self.batch_encoded_targets)

        assert self.batch_encoded_targets is not None
        logger.info(
            f"Encoded target cache written. file={pathlib.Path(self.file_path).name}, "
            f"shape={self.batch_encoded_targets.shape}"
        )

    def _get_variants_to_process(self, variant_names: list[str]) -> list[str]:
        """Determine which variants to process."""
        variants: list[str] = variant_names.copy()

        if self.variations_to_use is not None:
            variants = [v for v in variants if v in self.variations_to_use]

        if self.variations_to_ignore:
            variants = [v for v in variants if v not in self.variations_to_ignore]

        if not variants:
            raise ValueError(f"No variants available after filtering: {variant_names}")

        return variants


def create_dataloader(
    file_path: str,
    batch_size: int = 32,
    *,
    num_batches_per_epoch: int | None = None,
    replacement: bool = True,
    generator: torch_Generator | None = None,
    transform: TransformProtocol | None = None,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    pin_memory_device: str = "",
    infinite: bool = False,
    variations_to_use: list[str] | None = None,
    variations_to_ignore: list[str] | None = None,
    encoder: Module | None = None,
    precompute_targets: bool = False,
    device: str | torch.device = "cpu",
    use_next_state_targets: bool = False,
) -> DataLoader[dict[str, Tensor]]:
    """Create a DataLoader for alignment training from HDF5 file.

    Args:
        file_path: Path to HDF5 data file containing variation data.
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
        variations_to_use: Specific variation names to include.
        variations_to_ignore: Variation names to exclude.
        encoder: Optional pretrained encoder to precompute targets. Required if precompute_targets=True.
        precompute_targets: If True, use encoder to precompute encoded targets for base states.
            Note: Encoded targets are not returned when use_next_state_targets=True.
        device: Device to use for encoder computations when precomputing targets.
        use_next_state_targets: If True, target states will be the base version of the NEXT state
            instead of the base version of the current state. This also returns actions.
            Raw base images are returned as targets, not encoded versions.

    Returns:
        DataLoader using AlignmentDataset.
    """
    batch_dataset: AlignmentDataset = AlignmentDataset(
        file_path=file_path,
        batch_size=batch_size,
        num_batches=num_batches_per_epoch,
        replacement=replacement,
        generator=generator,
        transform=transform,
        dtype=dtype,
        infinite=infinite,
        variations_to_use=variations_to_use,
        variations_to_ignore=variations_to_ignore,
        encoder=encoder,
        precompute_targets=precompute_targets,
        device=device,
        use_next_state_targets=use_next_state_targets,
    )

    # Create DataLoader
    if num_workers > 0 and prefetch_factor is not None:
        dataloader: DataLoader[dict[str, Tensor]] = DataLoader(
            batch_dataset,
            batch_size=None,  # Batching handled by AlignmentDataset
            shuffle=False,  # Shuffling handled by AlignmentDataset
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            prefetch_factor=prefetch_factor,
        )
    else:
        dataloader = DataLoader(
            batch_dataset,
            batch_size=None,  # Batching handled by AlignmentDataset
            shuffle=False,  # Shuffling handled by AlignmentDataset
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            pin_memory_device=pin_memory_device,
        )

    mode_str: str = "infinite" if infinite else f"finite ({batch_dataset.num_batches} batches)"
    precompute_str: str = f", precomputed_targets={precompute_targets}" if precompute_targets else ""
    targets_mode: str = "next_state" if use_next_state_targets else "current_state"
    logger.info(
        "Created AlignmentDataLoader: "
        f"batch_size={batch_size}, mode={mode_str}, targets_mode={targets_mode}{precompute_str}"
    )

    return dataloader
