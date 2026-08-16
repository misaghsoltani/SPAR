"""Encode state datasets for goal-conditioned heuristic training."""

from __future__ import annotations

from logging import getLogger
import pathlib
import pickle
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch

from spar.utils.log_utils.wandb_logger import log_metrics

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import Logger
    from typing import TypeGuard

    from numpy.typing import NDArray
    from torch import Tensor, nn

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.optuna.types import ReporterPayload
    from spar.utils.config_utils.config_schema import EncodeOfflineDataSPARConfig
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession


logger: Logger = getLogger(__name__)
_ARRAY_TYPE: type[NDArray[np.float32]] = type(np.empty(0, dtype=np.float32))


@runtime_checkable
class StatesToTensorProtocol(Protocol):
    """Protocol for environments that implement `states_to_tensor`."""

    def states_to_tensor(self, states: list[ABCState]) -> Tensor:
        """Convert a list of states to a tensor representation."""
        ...


def _is_array_state_batch(states: list[ABCState] | list[NDArray[np.float32]]) -> TypeGuard[list[NDArray[np.float32]]]:
    return len(states) > 0 and isinstance(states[0], _ARRAY_TYPE)


def _is_abc_state_batch(states: list[ABCState] | list[NDArray[np.float32]]) -> TypeGuard[list[ABCState]]:
    return not _is_array_state_batch(states)


def encode_states(
    env: ABCEnvironment[ABCState],
    encoder_model: nn.Module,
    states: list[ABCState] | list[NDArray[np.float32]],
    batch_size: int = 100,
    device: torch.device | None = None,
) -> NDArray[np.float32]:
    """Encode states using the provided encoder model.

    Args:
        env: The environment instance.
        encoder_model: The encoder model to use.
        states: List of states to encode.
        batch_size: Batch size for encoding.
        device: The device to use for computation.

    Returns:
        Array of encoded states.

    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_model.to(device)
    encoder_model.eval()

    num_states: int = len(states)
    num_batches: int = (num_states + batch_size - 1) // batch_size

    encodings: list[NDArray[np.float32]] = []

    for batch_idx in range(num_batches):
        start_idx: int = batch_idx * batch_size
        end_idx: int = min((batch_idx + 1) * batch_size, num_states)

        batch_states: list[ABCState] | list[NDArray[np.float32]] = states[start_idx:end_idx]

        # Convert states to tensor representation
        state_tensors: Tensor
        if _is_array_state_batch(batch_states):
            batch_arr: NDArray[np.float32] = np.stack(batch_states, axis=0)
            # Flatten to (batch_size, -1)
            state_tensors = torch.tensor(batch_arr.reshape(batch_arr.shape[0], -1), dtype=torch.float32).to(device)
        elif _is_abc_state_batch(batch_states):
            if not isinstance(env, StatesToTensorProtocol):
                raise TypeError("Environment must implement states_to_tensor for ABCState inputs")
            state_tensors = env.states_to_tensor(batch_states).to(device)
        else:
            raise TypeError("States must be either NumPy arrays or ABCState instances")

        with torch.no_grad():
            # Get the discrete encodings (second output of encoder)
            enc_d: Tensor
            _, enc_d = encoder_model(state_tensors)
            encodings.append(enc_d.cpu().numpy())

    return np.concatenate(encodings, axis=0)


def encode_offline_data(
    env: ABCEnvironment[ABCState], input_path: str, output_path: str, batch_size: int = 100
) -> None:
    """Encode offline data for heuristic training.

    Args:
        env: The environment to use.
        input_path: Path to the raw data file.
        output_path: Path to save the encoded data.
        batch_size: Batch size for encoding.

    """
    # Load offline data
    logger.info(f"Loading offline data from {input_path}...")
    try:
        with pathlib.Path(input_path).open("rb") as f:
            state_trajs: list[list[ABCState]]
            action_trajs: list[list[int]]
            state_trajs, action_trajs = pickle.load(f)

    except Exception as e:
        # Preserve original exception context
        raise RuntimeError(f"Failed to load offline data: {e!s}") from e

    logger.info(f"Loaded {len(state_trajs)} episodes.")

    # Load the encoder model from the environment
    logger.info("Getting encoder from environment...")

    # Some environments expose a no-argument get_encoder helper outside the ABCEnvironment interface
    get_encoder: Callable[[], nn.Module] | None = getattr(env, "get_encoder", None)
    if get_encoder is None or not callable(get_encoder):
        raise RuntimeError("Failed to get encoder model from environment.")

    encoder_model: nn.Module = get_encoder()

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Process and encode each episode
    logger.info("Encoding states...")
    start_time: float = time.time()

    encoded_state_trajs: list[NDArray[np.float32]] = []

    display_steps: list[int] = list(np.linspace(1, len(state_trajs), 10, dtype=int))

    for idx, episode_states in enumerate(state_trajs):
        # Encode the states
        encoded_states: NDArray[np.float32] = encode_states(
            env, encoder_model, episode_states, batch_size=batch_size, device=device
        )

        encoded_state_trajs.append(encoded_states)

        # Display progress
        if (idx + 1) in display_steps:
            progress: float = 100.0 * (idx + 1) / len(state_trajs)
            elapsed: float = time.time() - start_time
            logger.info(f"{progress:.2f}% complete ({elapsed:.2f}s elapsed)")

    total_time: float = time.time() - start_time
    logger.info(f"Encoding complete. Total time: {total_time:.2f}s")

    # Save the encoded data
    logger.info(f"Saving encoded data to {output_path}...")
    save_start_time: float = time.time()

    save_dir: pathlib.Path = pathlib.Path(output_path).parent
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    with pathlib.Path(output_path).open("wb") as f:
        pickle.dump((encoded_state_trajs, action_trajs), f, protocol=-1)

    logger.info(f"Data saved. Write time: {time.time() - save_start_time:.2f}s")
    logger.info("Offline data encoding complete.")


def run_encode_offline_data(
    env: ABCEnvironment[ABCState],
    cfg: EncodeOfflineDataSPARConfig,
    tracking: WandbTrackingSession | None = None,
    reporter: Callable[[ReporterPayload], None] | None = None,
) -> None:
    """Run offline encoding from a validated stage configuration.

    Args:
        env: Environment used to construct and run the encoder.
        cfg: Validated offline-encoding stage configuration.
        tracking: Optional W&B session owned by the pipeline lifecycle.
        reporter: Optional sparse progress callback.
    """
    if reporter is not None:
        reporter({"phase": "encode", "primary": None, "metrics": {"status": "started"}})

    input_size_bytes: int = pathlib.Path(cfg.input_path).stat().st_size
    start_time: float = time.perf_counter()
    encode_offline_data(env, cfg.input_path, cfg.output_path, batch_size=cfg.batch_size)
    elapsed_seconds: float = time.perf_counter() - start_time
    output_size_bytes: int = pathlib.Path(cfg.output_path).stat().st_size

    if tracking is not None and tracking.enabled:
        output_mib: float = output_size_bytes / (1024**2)
        log_metrics(
            tracking,
            {
                "data/input_bytes": input_size_bytes,
                "data/output_bytes": output_size_bytes,
                "data/encoding_seconds": elapsed_seconds,
                "data/output_mib_per_second": output_mib / max(elapsed_seconds, 1e-12),
                "data/encoding_batch_size": cfg.batch_size,
            },
        )

    if reporter is not None:
        reporter({
            "phase": "encode",
            "primary": elapsed_seconds,
            "metrics": {
                "status": "completed",
                "input_bytes": input_size_bytes,
                "output_bytes": output_size_bytes,
                "elapsed_seconds": elapsed_seconds,
            },
        })
