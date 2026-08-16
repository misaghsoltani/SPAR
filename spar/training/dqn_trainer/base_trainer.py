"""Base trainer for DQN."""

from __future__ import annotations

from collections import OrderedDict
from logging import getLogger
import math
import os
import pathlib
import pickle
import re
import shutil
import time
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.checkpoint.state_dict_saver import save as save_distributed_state_dict
import torch.distributed.distributed_c10d as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.multiprocessing import Queue, get_context
from torch.utils.data import DataLoader, Dataset

from spar.utils.config_utils.config_schema import ModelArchitectureConfig, ModelConfig
from spar.utils.env_utils import get_environment
from spar.utils.pytorch_utils.nnet_utils import load_model

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from logging import Logger
    from multiprocessing.context import SpawnContext, SpawnProcess
    from typing import TypeAlias

    from torch.distributed import ProcessGroup
    from torch.distributed.rpc.api import RRef
    from torch.optim.optimizer import Optimizer

    from spar.environments.abstracts.environment import ABCEnvironment
    from spar.environments.abstracts.state import ABCState
    from spar.utils.config_utils.config_schema import (
        CompileConfig,
        DQNTrainConfig,
        MemoryConfig,
        NCCLConfig,
        OptimizationConfig,
    )


DataQueueType: TypeAlias = tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.intp], NDArray[np.float32]]
TimeQueueType: TypeAlias = OrderedDict[str, float]

logger: Logger = getLogger(__name__)

ModelT = TypeVar("ModelT", bound=nn.Module)
NestedT = TypeVar("NestedT")


def greedy_policy_rollout(
    dqn: nn.Module,
    env_model: nn.Module,
    states_start: NDArray[np.uint8] | NDArray[np.float32],
    states_goal: NDArray[np.uint8] | NDArray[np.float32],
    per_eq_tol: float,
    max_steps: int,
    device: torch.device,
) -> tuple[list[bool], None]:
    """Roll out the DQN greedy action policy for a fixed step budget.

    This is a batched policy evaluation, not a best-first search. Each state
    follows the action with minimum predicted cost at every step and the
    returned flags describe the states at the end of the rollout.

    Args:
        dqn: Network that predicts per-action costs for each state and goal.
        env_model: Learned transition model applied to the selected actions.
        states_start: Initial latent states.
        states_goal: Goal latent states paired with the initial states.
        per_eq_tol: Required percentage of equal latent elements.
        max_steps: Maximum number of greedy policy steps.
        device: Device used for model inference.

    Returns:
        Final solved flags and the retained empty auxiliary result.
    """
    # Disable training-only module behavior during rollout.
    dqn.eval()
    env_model.eval()

    # The rollout does not build gradients or mutate autograd state.
    with torch.inference_mode():
        # Convert inputs to tensors
        states: Tensor = torch.tensor(states_start, device=device, dtype=torch.float32)
        goals: Tensor = torch.tensor(states_goal, device=device, dtype=torch.float32)

        # Greedy policy rollout using the DQN action costs.
        is_solved: list[bool] = []
        current_states: Tensor = states.clone()

        for _ in range(max_steps):
            # Check if already solved
            solved_mask: Tensor = (100 * torch.mean(torch.eq(current_states, goals).float(), dim=1)) >= per_eq_tol

            if torch.all(solved_mask):
                break

            # Get Q-values and select best actions
            q_values: Tensor = dqn(current_states, goals)
            actions: Tensor = torch.argmin(q_values, dim=1)  # Minimize cost-to-go

            # Take actions in environment
            current_states = env_model(current_states, actions.float()).round()

        # Final solve check
        final_solved: Tensor = (100 * torch.mean(torch.eq(current_states, goals).float(), dim=1)) >= per_eq_tol
        is_solved = final_solved.cpu().tolist()

        return is_solved, None


def evaluate_greedy_policy(
    states_offline: NDArray[np.uint8] | NDArray[np.float32],
    num_test: int,
    dqn: nn.Module,
    env_model: nn.Module,
    num_actions: int,
    goal_steps: int,
    device: torch.device,
    max_steps: int,
    per_eq_tol: float,
) -> float:
    """Evaluate the greedy DQN policy on randomly generated state-goal pairs.

    Args:
        states_offline: Offline latent states used as rollout starts.
        num_test: Number of state-goal pairs to evaluate.
        dqn: Network that predicts per-action costs.
        env_model: Learned transition model used for random walks and rollouts.
        num_actions: Number of available actions.
        goal_steps: Maximum random-walk distance used to generate goals.
        device: Device used for model inference.
        max_steps: Maximum number of greedy policy steps.
        per_eq_tol: Required percentage of equal latent elements.

    Returns:
        Percentage of final rollout states that match their goals.
    """
    # Disable training-only module behavior during evaluation.
    dqn.eval()
    env_model.eval()

    # Evaluation does not build gradients or mutate autograd state.
    with torch.inference_mode():
        # Sample random test states
        samp_idxs: NDArray[np.intp] = np.random.randint(0, states_offline.shape[0], size=num_test)
        states_start: NDArray[np.float32] = states_offline[samp_idxs].astype(np.float32)

        # Generate random goal states using random walk
        goal_steps_samp: list[int] = list(np.random.randint(0, goal_steps + 1, size=num_test))
        states_goal: NDArray[np.float32] = random_walk(states_start, goal_steps_samp, num_actions, env_model, device)

        # Run the greedy policy evaluation.
        is_solved, _ = greedy_policy_rollout(dqn, env_model, states_start, states_goal, per_eq_tol, max_steps, device)

        # Calculate percentage solved
        percent_solved: float = 100.0 * sum(is_solved) / len(is_solved)
        return percent_solved


# ==============================================================================
# Utility Functions
# ==============================================================================


def copy_files(src_dir: str, dst_dir: str) -> None:
    """Copy all files from source directory to destination directory.

    Args:
        src_dir: Source directory path
        dst_dir: Destination directory path
    """
    if pathlib.Path(src_dir).exists():
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def flatten(data: list[list[NestedT]]) -> tuple[list[NestedT], list[int]]:
    """Flatten a nested list and return split indices for reconstruction.

    Args:
        data: Nested list to flatten.

    Returns:
        Tuple of flattened values and split indices.
    """
    num_each: list[int] = [len(x) for x in data]
    split_idxs: list[int] = list(np.cumsum(num_each)[:-1])
    data_flat: list[NestedT] = [item for sublist in data for item in sublist]
    return data_flat, split_idxs


def unflatten(data: list[NestedT], split_idxs: list[int]) -> list[list[NestedT]]:
    """Reconstruct nested list from flattened data and split indices.

    Args:
        data: Flattened data
        split_idxs: Split indices for reconstruction

    Returns:
        Reconstructed nested list
    """
    data_split: list[list[NestedT]] = []
    start_idx: int = 0

    for end_idx in split_idxs:
        data_split.append(data[start_idx:end_idx])
        start_idx = end_idx

    data_split.append(data[start_idx:])
    return data_split


def split_evenly(num_total: int, num_splits: int) -> list[int]:
    """Split a total number evenly across multiple groups.

    Args:
        num_total: Total number to split
        num_splits: Number of groups to split into

    Returns:
        List of counts per group
    """
    num_per: list[int] = [math.floor(num_total / num_splits) for _ in range(num_splits)]
    left_over: int = num_total % num_splits

    for idx in range(left_over):
        num_per[idx] += 1

    return num_per


# ==============================================================================
# Performance Profiling
# ==============================================================================


def record_time(times: dict[str, float], time_name: str, start_time: float, on_gpu: bool) -> None:
    """Record execution time with optional GPU synchronization.

    Args:
        times: Dictionary to store timing results
        time_name: Name/key for this timing
        start_time: Start time from time.time()
        on_gpu: Whether to synchronize GPU before measuring
    """
    if on_gpu:
        torch.cuda.synchronize()

    time_elapsed: float = time.time() - start_time
    if time_name in times:
        times[time_name] += time_elapsed
    else:
        times[time_name] = time_elapsed


def add_times(times: dict[str, float], times_to_add: dict[str, float]) -> None:
    """Add timing measurements from one dictionary to another.

    Args:
        times: Target dictionary to accumulate times
        times_to_add: Source dictionary with times to add
    """
    for key, value in times_to_add.items():
        times[key] += value


def get_time_str(times: dict[str, float]) -> str:
    """Format timing dictionary as a readable string.

    Args:
        times: Dictionary of timing measurements

    Returns:
        Formatted string representation
    """
    time_str_l: list[str] = []
    for key, val in times.items():
        time_str_i = f"{key}: {val:.2f}"
        time_str_l.append(time_str_i)
    return ", ".join(time_str_l)


# ==============================================================================
# Environment Model Functions
# ==============================================================================


def random_walk_traj(
    states_np_inp: NDArray[np.float32], num_steps: int, num_actions: int, env_model: nn.Module, device: torch.device
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Generate random walk trajectory from initial states.

    Args:
        states_np_inp: Initial states
        num_steps: Number of steps to take
        num_actions: Number of possible actions
        env_model: Environment model for state transitions
        device: Device for computation

    Returns:
        Tuple of (final_states, trajectory)
    """
    states_np: NDArray[np.float32] = states_np_inp.copy()
    states_traj_np: NDArray[np.float32] = np.zeros(
        (states_np.shape[0], num_steps + 1, states_np.shape[1]), dtype=np.float32
    )
    states_to_move: Tensor = torch.tensor(states_np, device=device).float().detach()

    for step_num in range(num_steps):
        states_traj_np[:, step_num, :] = states_to_move.cpu().data.numpy()

        actions_np: NDArray[np.intp] = np.asarray(
            np.random.randint(0, num_actions, size=states_np.shape[0]), dtype=np.intp
        )
        actions: Tensor = torch.tensor(actions_np, device=device).float().detach()

        states_to_move = env_model(states_to_move, actions).round().detach()

    states_np = states_to_move.cpu().data.numpy().astype(np.float32)
    states_traj_np[:, -1, :] = states_np

    return states_np, states_traj_np


def random_walk(
    states_np_inp: NDArray[np.float32],
    num_steps_l: list[int],
    num_actions: int,
    env_model: nn.Module,
    device: torch.device,
) -> NDArray[np.float32]:
    """Perform random walks with variable step counts.

    Args:
        states_np_inp: Initial states
        num_steps_l: List of step counts for each state
        num_actions: Number of possible actions
        env_model: Environment model for state transitions
        device: Device for computation

    Returns:
        Final states after random walks
    """
    # Initialize
    num_steps_max: int = max(num_steps_l)
    num_steps: NDArray[np.intp] = np.asarray(num_steps_l, dtype=np.intp)
    num_steps_curr: NDArray[np.intp] = np.asarray(num_steps_l, dtype=np.intp)
    states_np: NDArray[np.float32] = states_np_inp.copy()

    states_to_move: Tensor = torch.tensor(states_np, device=device).float().detach()

    for step_num in range(num_steps_max):
        # Get actions
        actions_np: NDArray[np.intp] = np.random.randint(0, num_actions, size=states_to_move.shape[0])
        actions: Tensor = torch.tensor(actions_np, device=device).float().detach()

        # Get next states
        states_to_move = env_model(states_to_move, actions).round().detach()

        # Record goal states
        end_step_mask: NDArray[np.bool_] = num_steps == (step_num + 1)
        end_step_mask_curr: NDArray[np.bool_] = num_steps_curr == (step_num + 1)
        end_step_mask_curr_tensor: Tensor = torch.from_numpy(end_step_mask_curr).to(device=states_to_move.device)
        states_np[end_step_mask] = states_to_move[end_step_mask_curr_tensor].to(torch.uint8).cpu().data.numpy()

        # Get only states that have not reached goal state
        move_mask_curr: NDArray[np.bool_] = num_steps_curr > (step_num + 1)
        move_mask_tensor: Tensor = torch.from_numpy(move_mask_curr).to(device=states_to_move.device)
        states_to_move = states_to_move[move_mask_tensor]
        num_steps_curr = num_steps_curr[move_mask_curr]

    return states_np


# ==============================================================================
# Device and Model Management
# ==============================================================================


def get_device() -> tuple[torch.device, list[int], bool]:
    """Get compute device information with GPU detection.

    Returns:
        Tuple of (device, device_list, is_gpu)
    """
    device: torch.device = torch.device("cpu")
    devices: list[int] = []
    on_gpu: bool = False

    if ("CUDA_VISIBLE_DEVICES" in os.environ) and torch.cuda.is_available():
        device = torch.device("cuda:0")
        devices = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        on_gpu = True

    return device, devices, on_gpu


def load_nnet(model_file: str, nnet: nn.Module, device: torch.device | None = None) -> nn.Module:
    """Load a neural network state dict from a checkpoint file.

    The checkpoint is mapped directly onto the target device, key prefixes
    added by DDP and FSDP wrappers are stripped, and loading uses
    ``weights_only=True`` so only tensor data is deserialized.

    Args:
        model_file: Path to model file
        nnet: Neural network module to load into
        device: Target device for model

    Returns:
        The network in eval mode on the target device
    """
    if device is None:
        device = torch.device("cpu")

    loaded_state = torch.load(model_file, map_location=device, weights_only=True)

    if not isinstance(loaded_state, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(loaded_state)}")

    raw_state: dict[str, Tensor]
    if "dqn_model" in loaded_state and isinstance(loaded_state["dqn_model"], dict):
        candidate_state = loaded_state["dqn_model"]
        raw_state = {k: v for k, v in candidate_state.items() if isinstance(v, Tensor)}
    else:
        raw_state = {k: v for k, v in loaded_state.items() if isinstance(v, Tensor)}

    # Remove module prefix from distributed training
    new_state_dict: OrderedDict[str, Tensor] = OrderedDict()
    for key, value in raw_state.items():
        clean_key: str = re.sub(r"^module\.", "", key)
        # Also handle FSDP prefix removal
        clean_key = re.sub(r"^_fsdp_wrapped_module\.", "", clean_key)
        new_state_dict[clean_key] = value

    nnet.load_state_dict(new_state_dict, strict=True)

    nnet.eval()

    # Non-blocking transfer is only valid for CUDA targets
    nnet.to(device, non_blocking=(device.type == "cuda"))

    return nnet


def get_heuristic_fn(
    nnet: nn.Module, device: torch.device, clip_zero: bool = False, batch_size: int | None = None
) -> Callable[[NDArray[np.float32], NDArray[np.float32]], NDArray[np.float32]]:
    """Create batched heuristic function from neural network.

    Args:
        nnet: Neural network for heuristic computation
        device: Computation device
        clip_zero: Whether to clip negative values to zero
        batch_size: Batch size for processing (None for full batch)

    Returns:
        Batched heuristic function
    """
    nnet.eval()

    def heuristic_fn(states_np: NDArray[np.float32], states_goal_np: NDArray[np.float32]) -> NDArray[np.float32]:
        cost_to_go_l: list[NDArray[np.float32]] = []
        num_states: int = states_np.shape[0]

        batch_size_inst: int = num_states
        if batch_size is not None:
            batch_size_inst = batch_size

        start_idx: int = 0
        while start_idx < num_states:
            # Get batch
            end_idx: int = min(start_idx + batch_size_inst, num_states)

            # Convert batch to tensor
            states_batch: Tensor = torch.tensor(states_np[start_idx:end_idx], device=device)
            states_goal_batch: Tensor = torch.tensor(states_goal_np[start_idx:end_idx], device=device)

            cost_to_go_batch: NDArray[np.float32] = nnet(states_batch, states_goal_batch).cpu().data.numpy()
            cost_to_go_l.append(cost_to_go_batch)

            start_idx = end_idx

        cost_to_go: NDArray[np.float32] = np.concatenate(cost_to_go_l, axis=0)
        assert cost_to_go.shape[0] == num_states, (
            f"Shape of cost_to_go is {cost_to_go.shape} num states is {num_states}"
        )

        if clip_zero:
            cost_to_go = np.maximum(cost_to_go, 0.0)

        return cost_to_go

    return heuristic_fn


def get_model_fn(
    nnet: nn.Module, device: torch.device, batch_size: int | None = None
) -> Callable[[NDArray[np.float32], NDArray[np.float32]], NDArray[np.uint8]]:
    """Create batched model function from neural network.

    Args:
        nnet: Neural network for model computation
        device: Computation device
        batch_size: Batch size for processing (None for full batch)

    Returns:
        Batched model function
    """
    nnet.eval()

    def model_fn(states_np: NDArray[np.float32], actions_np: NDArray[np.float32]) -> NDArray[np.uint8]:
        states_next_l: list[NDArray[np.uint8]] = []
        num_states: int = states_np.shape[0]

        batch_size_inst: int = num_states
        if batch_size is not None:
            batch_size_inst = batch_size

        start_idx: int = 0
        while start_idx < num_states:
            # Get batch
            end_idx: int = min(start_idx + batch_size_inst, num_states)

            states_batch_np: NDArray[np.float32] = states_np[start_idx:end_idx]
            actions_batch_np: NDArray[np.float32] = actions_np[start_idx:end_idx]

            # Get nnet output
            states_batch: Tensor = torch.tensor(states_batch_np, device=device).float()
            actions_batch: Tensor = torch.tensor(actions_batch_np, device=device).float()

            states_next_batch_np: NDArray[np.float32] = nnet(states_batch, actions_batch).cpu().data.numpy()
            states_next_l.append(states_next_batch_np.round().astype(np.uint8))

            start_idx = end_idx

        states_next_np: NDArray[np.uint8] = np.concatenate(states_next_l, axis=0)
        assert states_next_np.shape[0] == num_states

        return states_next_np

    return model_fn


def get_available_gpu_nums() -> list[int]:
    """Get list of available GPU numbers from environment.

    Returns:
        List of available GPU indices
    """
    gpu_nums: list[int] = []
    if ("CUDA_VISIBLE_DEVICES" in os.environ) and (len(os.environ["CUDA_VISIBLE_DEVICES"]) > 0):
        gpu_nums = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]

    return gpu_nums


def load_heuristic_fn(
    nnet_dir: str,
    device: torch.device,
    on_gpu: bool,
    nnet: nn.Module,
    *,
    clip_zero: bool = False,
    gpu_num: int = -1,
    batch_size: int | None = None,
) -> Callable[[NDArray[np.float32], NDArray[np.float32]], NDArray[np.float32]]:
    """Load heuristic function from saved model directory.

    Args:
        nnet_dir: Directory containing model files
        device: Computation device
        on_gpu: Whether GPU is being used
        nnet: Neural network architecture
        clip_zero: Whether to clip negative values
        gpu_num: Specific GPU number to use
        batch_size: Batch size for processing

    Returns:
        Loaded heuristic function
    """
    if (gpu_num >= 0) and on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    model_file: str = f"{nnet_dir}/model_state_dict.pt"

    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)
    return get_heuristic_fn(nnet, device, clip_zero=clip_zero, batch_size=batch_size)


def load_model_fn(
    model_file: str,
    device: torch.device,
    on_gpu: bool,
    nnet: nn.Module,
    *,
    gpu_num: int = -1,
    batch_size: int | None = None,
) -> Callable[[NDArray[np.float32], NDArray[np.float32]], NDArray[np.uint8]]:
    """Load model function from saved file.

    Args:
        model_file: Path to model file
        device: Computation device
        on_gpu: Whether GPU is being used
        nnet: Neural network architecture
        gpu_num: Specific GPU number to use
        batch_size: Batch size for processing

    Returns:
        Loaded model function
    """
    if (gpu_num >= 0) and on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)
    return get_model_fn(nnet, device, batch_size=batch_size)


class ZeroModel(nn.Module):
    """Zero-initialized Q-network model for cold start training.

    This model returns zero Q-values for all state-action pairs, useful when
    no pre-trained model is available.

    Attributes:
        num_actions_max: Maximum number of actions
        device: Computation device
    """

    def __init__(self, num_actions_max: int, device: torch.device) -> None:
        """Initialize zero model.

        Args:
            num_actions_max: Maximum number of actions
            device: Computation device
        """
        super().__init__()
        self.num_actions_max: int = num_actions_max
        self.device: torch.device = device

    def forward(self, states: Tensor, _: Tensor) -> Tensor:
        """Forward pass returning zero Q-values.

        Args:
            states: State tensor
            _: Goal states (unused)

        Returns:
            Zero Q-values tensor
        """
        return torch.zeros((states.shape[0], self.num_actions_max), device=self.device)


# ==============================================================================
# Q-Learning Core Functions with FSDP Support
# ==============================================================================


def sample_boltzmann(qvals: Tensor, temp: float) -> Tensor:
    """Sample actions from Boltzmann distribution based on Q-values.

    Args:
        qvals: Q-values tensor
        temp: Temperature parameter for sampling

    Returns:
        Sampled actions
    """
    exp_vals: Tensor = torch.exp((1.0 / temp) * (-qvals + qvals.min(dim=1, keepdim=True)[0]))
    probs: Tensor = exp_vals / torch.sum(exp_vals, dim=1, keepdim=True)
    actions: Tensor = torch.multinomial(probs, 1)[:, 0]
    return actions


def q_step(
    states: Tensor,
    states_goal: Tensor,
    per_eq_tol: float,
    env_model: nn.Module,
    dqn: nn.Module,
    dqn_targ: nn.Module,
    on_gpu: bool,
    times: TimeQueueType,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Perform a single Q-learning step with timing.

    Args:
        states: Current states
        states_goal: Goal states
        per_eq_tol: Percentage equality tolerance for solved detection
        env_model: Environment model for state transitions
        dqn: Current Q-network
        dqn_targ: Target Q-network
        on_gpu: Whether using GPU
        times: Timing dictionary

    Returns:
        Tuple of (next_states, actions, cost_to_go_backups, is_solved)
    """
    # Get action
    start_time: float = time.time()
    qvals: Tensor = dqn(states, states_goal).detach()
    record_time(times, "qvals", start_time, on_gpu)

    start_time = time.time()
    actions: Tensor = sample_boltzmann(qvals, 1 / 3.0)
    record_time(times, "samp_acts", start_time, on_gpu)

    # Check if solved
    start_time = time.time()
    is_solved: Tensor = (100 * torch.mean(torch.eq(states, states_goal).float(), dim=1)) >= per_eq_tol
    record_time(times, "is_solved", start_time, on_gpu)

    # Get next states
    start_time = time.time()
    states_next: Tensor = env_model(states, actions).round().detach()
    record_time(times, "env_model", start_time, on_gpu)

    # Min cost-to-go for next state
    start_time = time.time()
    ctg_acts_next: Tensor = torch.clamp(dqn_targ(states_next, states_goal).detach(), min=0)
    ctgs_next: Tensor = torch.min(ctg_acts_next, dim=1)[0]
    record_time(times, "ctgs", start_time, on_gpu)

    # Backup cost-to-go
    start_time = time.time()
    ctg_backups: Tensor = 1.0 + ctgs_next
    ctg_backups *= 1.0 - is_solved.float()
    record_time(times, "backup", start_time, on_gpu)

    return states_next, actions, ctg_backups, is_solved


def _compile_module_if_enabled(module: nn.Module, compile_cfg: CompileConfig | None) -> nn.Module:
    """Compile a module when compile is available and enabled.

    Args:
        module: Module to potentially compile.
        compile_cfg: Optional compilation config.

    Returns:
        Original module or compiled module when available.
    """
    if not hasattr(torch, "compile"):
        return module

    if compile_cfg is not None and compile_cfg.disable:
        return module

    torch_compile: Callable[..., nn.Module] = vars(torch)["compile"]
    if compile_cfg is None:
        compiled_candidate = torch_compile(module, mode="reduce-overhead", fullgraph=True)
    else:
        compiled_candidate = torch_compile(
            module,
            fullgraph=compile_cfg.fullgraph,
            dynamic=compile_cfg.dynamic,
            backend=compile_cfg.backend,
            mode=compile_cfg.mode,
            options=compile_cfg.options,
        )
    return compiled_candidate


def _queue_put_data(
    queue_ref: Queue[DataQueueType | None] | RRef[Queue[DataQueueType | None]], value: DataQueueType | None
) -> None:
    """Put a data payload into local or remote queue.

    Args:
        queue_ref: Local queue or remote queue reference.
        value: Data payload to enqueue.
    """
    if isinstance(queue_ref, Queue):
        queue_ref.put(value)
    else:
        queue_ref.rpc_async().put(value).wait()


def _queue_put_time(
    queue_ref: Queue[TimeQueueType | None] | RRef[Queue[TimeQueueType | None]], value: TimeQueueType
) -> None:
    """Put timing payload into local or remote queue.

    Args:
        queue_ref: Local queue or remote queue reference.
        value: Timing payload to enqueue.
    """
    if isinstance(queue_ref, Queue):
        queue_ref.put(value)
    else:
        queue_ref.rpc_async().put(value).wait()


def q_learning_runner(
    env_name: str,
    data_file: str,
    batch_size: int,
    num_batches: int,
    start_steps: int,
    goal_steps: int,
    per_eq_tol: float,
    max_steps: int,
    dqn_dir: str,
    *,
    dqn_targ_dir: str,
    gpu_num: int | None,
    device: torch.device,
    model_cfg: ModelConfig,
    pretrained_trans_model_path: str,
    compile_cfg: CompileConfig | None = None,
    data_queue: Queue[DataQueueType | None] | RRef[Queue[DataQueueType | None]] | None = None,
    time_queue: Queue[TimeQueueType | None] | RRef[Queue[TimeQueueType | None]] | None = None,
    use_dist: bool = False,
) -> tuple[DataQueueType, TimeQueueType] | None:
    """Generate Q-learning batches on one process or an FSDP worker.

    The loop uses:
    - Distributed training via RPC/FSDP
    - Device-local computation to minimize data movement
    - Boolean tensor conversion for NCCL, which cannot transport bool tensors

    Args:
        env_name: Environment name.
        data_file: Path to training data.
        batch_size: Batch size for training.
        num_batches: Number of batches to process.
        start_steps: Steps from offline states to start states.
        goal_steps: Steps from start to goal states.
        per_eq_tol: Percentage equality tolerance.
        max_steps: Maximum steps per episode.
        dqn_dir: Current DQN directory.
        dqn_targ_dir: Target DQN directory.
        gpu_num: GPU number to use (None for CPU).
        device: Computation device.
        model_cfg: Model architecture config.
        pretrained_trans_model_path: Path to pretrained transition model.
        compile_cfg: Optional compile config.
        data_queue: Optional queue for generated data.
        time_queue: Optional queue for timing information.
        use_dist: Whether to use distributed training.
    """
    times: TimeQueueType = OrderedDict([
        ("init", 0.0),
        ("gen", 0.0),
        ("qvals", 0.0),
        ("samp_acts", 0.0),
        ("is_solved", 0.0),
        ("env_model", 0.0),
        ("ctgs", 0.0),
        ("backup", 0.0),
        ("put", 0.0),
    ])
    start_time: float = time.time()

    env: ABCEnvironment[ABCState] = get_environment(env_name)
    num_actions: int = env.num_actions_max

    transition_model: nn.Module = load_model(
        model=env.get_env_model_disc(model_cfg), device=device, pretrained_path=pretrained_trans_model_path
    )

    if gpu_num is not None:
        if not use_dist:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
        else:
            device = torch.device(f"cuda:{gpu_num}")
            setup_nccl_environment()
    on_gpu: bool = device.type == "cuda"

    with pathlib.Path(data_file).open("rb") as file_handle:
        episodes_payload = pickle.load(file_handle)
    if not isinstance(episodes_payload, list) or len(episodes_payload) == 0:
        raise ValueError(f"Unexpected offline data format in '{data_file}'.")

    first_bucket: list[NDArray[np.uint8]] = episodes_payload[0]
    if not isinstance(first_bucket, list):
        raise TypeError(f"Unexpected offline data bucket type in '{data_file}'.")

    states_np: NDArray[np.uint8] = np.concatenate(first_bucket, axis=0).astype(np.uint8, copy=False)

    if use_dist and dist.is_initialized():
        shared_states: Tensor = create_shared_memory_tensor(torch.from_numpy(states_np))
        states_np = shared_states.numpy()

    transition_model.to(device)
    transition_model.eval()

    dqn_targ_file: str = f"{dqn_targ_dir}/model_state_dict.pt"
    if pathlib.Path(dqn_targ_file).is_file():
        dqn_targ: nn.Module = load_nnet(dqn_targ_file, env.get_dqn(model_cfg), device=device)
    else:
        dqn_targ = ZeroModel(num_actions, device)
    dqn_targ.to(device)
    dqn_targ.eval()
    dqn_targ = _compile_module_if_enabled(dqn_targ, compile_cfg)

    dqn_file: str = f"{dqn_dir}/model_state_dict.pt"
    if pathlib.Path(dqn_file).is_file():
        dqn: nn.Module = load_nnet(dqn_file, env.get_dqn(model_cfg), device=device)
    else:
        dqn = ZeroModel(num_actions, device)
    dqn.to(device)
    dqn.eval()
    dqn = _compile_module_if_enabled(dqn, compile_cfg)

    record_time(times, "init", start_time, on_gpu)

    all_states_start: list[NDArray[np.uint8]] = []
    all_states_goal: list[NDArray[np.uint8]] = []
    all_actions: list[NDArray[np.intp]] = []
    all_ctgs: list[NDArray[np.float32]] = []

    for _ in range(num_batches):
        start_time = time.time()
        sample_idxs: NDArray[np.intp] = np.asarray(
            np.random.randint(0, states_np.shape[0], size=batch_size), dtype=np.intp
        )

        start_steps_samp: list[int] = [start_steps] * batch_size
        goal_steps_samp: list[int] = list(np.random.randint(0, goal_steps + 1, size=batch_size))

        states_start_float: NDArray[np.float32] = random_walk(
            states_np[sample_idxs].astype(np.float32), start_steps_samp, num_actions, transition_model, device
        )
        states_goal_float: NDArray[np.float32] = random_walk(
            states_start_float, goal_steps_samp, num_actions, transition_model, device
        )

        states_start: Tensor = torch.tensor(states_start_float.astype(np.uint8), device=device).float()
        states_goal: Tensor = torch.tensor(states_goal_float.astype(np.uint8), device=device).float()
        record_time(times, "gen", start_time, on_gpu)

        for _ in range(max_steps):
            states_start_next, actions, ctgs, is_solved = q_step(
                states_start, states_goal, per_eq_tol, transition_model, dqn, dqn_targ, on_gpu, times
            )

            start_time = time.time()
            states_start_uint8: NDArray[np.uint8] = states_start.cpu().numpy().astype(np.uint8)
            states_goal_uint8: NDArray[np.uint8] = states_goal.cpu().numpy().astype(np.uint8)
            actions_np: NDArray[np.intp] = actions.cpu().numpy().astype(np.intp)
            ctgs_np: NDArray[np.float32] = ctgs.cpu().numpy().astype(np.float32)

            payload: DataQueueType = (states_start_uint8, states_goal_uint8, actions_np, ctgs_np)
            if data_queue is not None:
                if torch.cuda.is_available():
                    states_start_shared: NDArray[np.uint8] = create_shared_memory_tensor(
                        torch.from_numpy(states_start_uint8)
                    ).numpy()
                    states_goal_shared: NDArray[np.uint8] = create_shared_memory_tensor(
                        torch.from_numpy(states_goal_uint8)
                    ).numpy()
                    actions_shared: NDArray[np.intp] = create_shared_memory_tensor(torch.from_numpy(actions_np)).numpy()
                    ctgs_shared: NDArray[np.float32] = create_shared_memory_tensor(torch.from_numpy(ctgs_np)).numpy()
                    payload = (states_start_shared, states_goal_shared, actions_shared, ctgs_shared)
                _queue_put_data(data_queue, payload)
            else:
                all_states_start.append(states_start_uint8)
                all_states_goal.append(states_goal_uint8)
                all_actions.append(actions_np)
                all_ctgs.append(ctgs_np)

            record_time(times, "put", start_time, on_gpu)

            not_solved: Tensor = torch.logical_not(is_solved)
            states_start = states_start_next[not_solved]
            states_goal = states_goal[not_solved]
            if states_start.shape[0] == 0:
                break

        if data_queue is not None:
            _queue_put_data(data_queue, None)

    if data_queue is not None:
        if time_queue is not None:
            _queue_put_time(time_queue, times)
        return None

    if all_states_start:
        states_start_all: NDArray[np.uint8] = np.concatenate(all_states_start, axis=0)
        states_goal_all: NDArray[np.uint8] = np.concatenate(all_states_goal, axis=0)
        actions_all: NDArray[np.intp] = np.concatenate(all_actions, axis=0)
        ctgs_all: NDArray[np.float32] = np.concatenate(all_ctgs, axis=0)
    else:
        states_start_all = np.zeros((0, states_np.shape[1]), dtype=np.uint8)
        states_goal_all = np.zeros((0, states_np.shape[1]), dtype=np.uint8)
        actions_all = np.zeros((0,), dtype=np.intp)
        ctgs_all = np.zeros((0,), dtype=np.float32)

    return (states_start_all, states_goal_all, actions_all, ctgs_all), times


def q_update(
    env_name: str,
    data_file: str,
    batch_size: int,
    num_batches: int,
    start_steps: int,
    goal_steps: int,
    per_eq_tol: float,
    max_steps: int,
    *,
    env_model_dir: str,
    dqn_dir: str,
    dqn_targ_dir: str,
    model_cfg: ModelConfig,
    comm_mode: str = "auto",
    device: torch.device,
) -> tuple[DataQueueType, TimeQueueType]:
    """Update Q-network through multi-process data generation and collection.

    Each worker runs :func:`q_learning_runner` and returns generated training
    tuples to the parent process.

    Args:
        env_name: Environment name for the Q-learning task
        data_file: Path to offline training data
        batch_size: Number of samples per batch
        num_batches: Total number of batches to generate
        start_steps: Maximum steps from offline states to start states
        goal_steps: Maximum steps from start to goal states
        per_eq_tol: Percentage equality tolerance for solved detection
        max_steps: Maximum steps per episode
        env_model_dir: Directory containing environment model
        dqn_dir: Directory containing current DQN model
        dqn_targ_dir: Directory containing target DQN model
        model_cfg: Model architecture config for DQN and transition models.
        comm_mode: Communication mode for distributed training.
        device: PyTorch device for computation

    Returns:
        Tuple containing:
        - start_states: Generated start states (uint8)
        - goal_states: Generated goal states (uint8)
        - actions: Sampled actions (intp)
        - cost_to_go: Q-learning cost-to-go values (int_)
        - timing_info: Performance timing measurements
    """
    # ------------------------------------------------------------------
    # Fast path: keep tensors on device and
    # exchange them with NCCL collectives instead of CPU queues.
    # ------------------------------------------------------------------
    if dist.is_initialized() and comm_mode in {"auto", "all_gather", "reduce_scatter"}:
        # Generate data on the current rank - no extra processes, no queues.
        result: tuple[DataQueueType, TimeQueueType] | None = q_learning_runner(
            env_name=env_name,
            data_file=data_file,
            batch_size=batch_size,
            num_batches=num_batches,
            start_steps=start_steps,
            goal_steps=goal_steps,
            per_eq_tol=per_eq_tol,
            max_steps=max_steps,
            dqn_dir=dqn_dir,
            dqn_targ_dir=dqn_targ_dir,
            gpu_num=None,
            device=device,
            model_cfg=model_cfg,
            pretrained_trans_model_path=env_model_dir,
            data_queue=None,
            time_queue=None,
            use_dist=False,
        )
        if result is None:
            raise RuntimeError("q_learning_runner returned no data in direct-return mode.")

        payload: DataQueueType
        times: TimeQueueType
        s_start: NDArray[np.uint8]
        s_goal: NDArray[np.uint8]
        acts: NDArray[np.intp]
        ctgs: NDArray[np.float32]

        payload, times = result
        s_start, s_goal, acts, ctgs = payload

        # Convert all to tensors on the target device
        states_start_tensor: Tensor = torch.from_numpy(s_start).to(device)
        states_goal_tensor: Tensor = torch.from_numpy(s_goal).to(device)
        actions_tensor: Tensor = torch.from_numpy(acts).to(device)
        ctgs_tensor: Tensor = torch.from_numpy(ctgs).to(device)

        # Choose the communication op
        if comm_mode == "reduce_scatter":
            states_start_tensor = distributed_reduce_scatter(states_start_tensor)
            states_goal_tensor = distributed_reduce_scatter(states_goal_tensor)
            actions_tensor = distributed_reduce_scatter(actions_tensor)
            ctgs_tensor = distributed_reduce_scatter(ctgs_tensor)
        else:
            states_start_tensor = distributed_all_gather(states_start_tensor)
            states_goal_tensor = distributed_all_gather(states_goal_tensor)
            actions_tensor = distributed_all_gather(actions_tensor)
            ctgs_tensor = distributed_all_gather(ctgs_tensor)

        return (
            states_start_tensor.cpu().numpy().astype(np.uint8, copy=False),
            states_goal_tensor.cpu().numpy().astype(np.uint8, copy=False),
            actions_tensor.cpu().numpy().astype(np.intp, copy=False),
            ctgs_tensor.cpu().numpy().astype(np.float32, copy=False),
        ), times

    # get devices
    data_runner_devices_raw: list[int] = get_available_gpu_nums()
    data_runner_devices: list[int | None] = list(data_runner_devices_raw)

    if len(data_runner_devices) == 0:
        data_runner_devices = [None]

    num_procs: int = len(data_runner_devices)

    # start runners
    num_batches_l: list[int] = split_evenly(num_batches, num_procs)

    ctx: SpawnContext = get_context("spawn")
    procs: list[SpawnProcess] = []

    queue: Queue[DataQueueType | None] = ctx.Queue()
    time_queue: Queue[TimeQueueType] = ctx.Queue()

    for data_runner_idx, num_batches_idx in enumerate(num_batches_l):
        data_runner_device: int | None = data_runner_devices[data_runner_idx % len(data_runner_devices)]

        proc: SpawnProcess = ctx.Process(
            target=q_learning_runner,
            kwargs={
                "env_name": env_name,
                "data_file": data_file,
                "batch_size": batch_size,
                "num_batches": num_batches_idx,
                "start_steps": start_steps,
                "goal_steps": goal_steps,
                "per_eq_tol": per_eq_tol,
                "max_steps": max_steps,
                "dqn_dir": dqn_dir,
                "dqn_targ_dir": dqn_targ_dir,
                "gpu_num": data_runner_device,
                "device": device,
                "model_cfg": model_cfg,
                "pretrained_trans_model_path": env_model_dir,
                "data_queue": queue,
                "time_queue": time_queue,
                "use_dist": False,
            },
        )
        proc.daemon = True
        proc.start()

        procs.append(proc)

    # get data
    start_time: float = time.time()
    display_steps: list[int] = list(np.linspace(1, num_batches, 10, dtype=int))
    total_num_samples: int = batch_size * max_steps * num_batches

    states_start_np: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
    states_goal_np: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
    actions_np: NDArray[np.intp] = np.zeros(total_num_samples, dtype=np.intp)
    ctgs_np: NDArray[np.float32] = np.zeros(total_num_samples, dtype=np.float32)

    start_idx: int = 0
    batch_idx: int = 0
    while batch_idx < num_batches:
        q_res: DataQueueType | None = queue.get(block=True)
        if q_res is None:
            batch_idx += 1
            if batch_idx in display_steps:
                logger.info(f"{100 * batch_idx / num_batches:.2f}% ({time.time() - start_time:.2f})...")

        else:
            states_start_np_i, states_goal_np_i, actions_np_i, ctgs_np_i = q_res
            if states_start_np.shape[0] == 0:
                state_dim: int = states_start_np_i.shape[1]

                states_start_np = np.zeros((total_num_samples, state_dim), dtype=np.uint8)
                states_goal_np = np.zeros((total_num_samples, state_dim), dtype=np.uint8)

            end_idx: int = start_idx + states_start_np_i.shape[0]

            states_start_np[start_idx:end_idx] = states_start_np_i
            states_goal_np[start_idx:end_idx] = states_goal_np_i
            actions_np[start_idx:end_idx] = actions_np_i
            ctgs_np[start_idx:end_idx] = ctgs_np_i

            start_idx = end_idx

    states_start_np = states_start_np[:start_idx]
    states_goal_np = states_goal_np[:start_idx]
    actions_np = actions_np[:start_idx]
    ctgs_np = ctgs_np[:start_idx]

    # Apply distributed gathering if in distributed mode
    if dist.is_initialized():
        # Convert to tensors for distributed operations
        states_start_tensor = torch.from_numpy(states_start_np).to(device)
        states_goal_tensor = torch.from_numpy(states_goal_np).to(device)
        actions_tensor = torch.from_numpy(actions_np).to(device)
        ctgs_tensor = torch.from_numpy(ctgs_np).to(device)

        # NCCL collectives transport the solved mask as uint8.
        states_start_tensor = cast_bool_for_nccl(states_start_tensor)
        states_goal_tensor = cast_bool_for_nccl(states_goal_tensor)

        # Concatenate one generated batch from every rank.
        states_start_tensor = distributed_all_gather(states_start_tensor)
        states_goal_tensor = distributed_all_gather(states_goal_tensor)
        actions_tensor = distributed_all_gather(actions_tensor)
        ctgs_tensor = distributed_all_gather(ctgs_tensor)

        # Convert back to numpy, restoring boolean dtypes if needed
        states_start_np = uncast_bool_from_nccl(states_start_tensor, torch.uint8).cpu().numpy().astype(np.uint8)
        states_goal_np = uncast_bool_from_nccl(states_goal_tensor, torch.uint8).cpu().numpy().astype(np.uint8)
        actions_np = actions_tensor.cpu().numpy().astype(np.intp)
        ctgs_np = ctgs_tensor.cpu().numpy().astype(np.float32)

    logger.info(f"Generated {states_start_np.shape[0]:,} states\n")

    # get times
    times = time_queue.get()
    for _ in range(1, len(procs)):
        add_times(times, time_queue.get())

    for key, value in times.items():
        times[key] = value / len(procs)

    # join processes
    for proc in procs:
        proc.join()

    return (states_start_np, states_goal_np, actions_np, ctgs_np), times


def train_nnet(
    dqn: nn.Module,
    states_start_np: NDArray[np.float32],
    states_goal_np: NDArray[np.float32],
    actions_np: NDArray[np.int32],
    ctgs_np: NDArray[np.float32],
    batch_size: int,
    device: torch.device,
    on_gpu: bool,
    *,
    num_itrs: int,
    train_itr: int,
    lr: float,
    lr_d: float,
    optimizer: Optimizer,
    memory_cfg: MemoryConfig | None = None,
    optimization_cfg: OptimizationConfig | None = None,
    compile_cfg: CompileConfig | None = None,
    display: bool = True,
    use_dataloader: bool = True,
) -> float:
    """Train neural network using collected Q-learning data.

    The training loop uses:
    - FSDP for distributed parameter sharding
    - A DataLoader backed by shared-memory tensors, which avoids copying the
      dataset into each worker process
    - Gradient clipping and learning rate decay
    - Huber loss, which is less sensitive to outlier targets than MSE
    - Pinned host memory with non-blocking device transfers
    - Per-phase timing and loss tracking

    Args:
        dqn: DQN neural network to train
        states_start_np: Start state observations
        states_goal_np: Goal state observations
        actions_np: Actions taken
        ctgs_np: Cost-to-go targets
        batch_size: Training batch size
        device: PyTorch device for computation
        on_gpu: Whether using GPU acceleration
        num_itrs: Number of training iterations
        train_itr: Current training iteration
        lr: Base learning rate
        lr_d: Learning rate decay factor
        optimizer: Persistent optimizer instance reused across update rounds.
        memory_cfg: Memory configuration object
        optimization_cfg: Optimization controls (e.g., gradient clipping value).
        compile_cfg: Compilation configuration (accepted for API compatibility).
        display: Whether to display training progress
        use_dataloader: Whether to batch through a shared-memory DataLoader
            instead of sampling random index batches directly

    Returns:
        Final training loss value
    """
    dqn.train()
    num_exs: int = states_start_np.shape[0]
    assert batch_size <= num_exs, "Batch size should be less than or equal to number of train examples"
    _ = compile_cfg
    display_itrs: int = 100
    gradient_accumulation_steps: int = (
        optimization_cfg.gradient_accumulation_steps if optimization_cfg is not None else 1
    )
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least one")

    # Status tracking
    start_time_itr: float = time.time()
    times: OrderedDict[str, float] = OrderedDict([("fprop", 0.0), ("bprop", 0.0), ("itr", 0.0)])
    last_loss: float = np.inf

    dataloader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]] | None = None
    dataloader_iter: Iterator[tuple[Tensor, Tensor, Tensor, Tensor]] | None = None
    rand_batch_idxs: NDArray[np.intp] | None = None
    start_batch_idx: int = 0
    end_batch_idx: int = 0
    if use_dataloader:
        dataloader = create_shared_memory_dataloader(
            states_start=states_start_np,
            states_goal=states_goal_np,
            actions=actions_np,
            ctgs=ctgs_np,
            batch_size=batch_size,
            memory_cfg=memory_cfg,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        dataloader_iter = iter(dataloader)
    else:
        rand_batch_idxs = np.random.permutation(num_exs)
        end_batch_idx = start_batch_idx + batch_size

    for step_idx in range(num_itrs):
        current_itr: int = train_itr + step_idx
        accumulation_index: int = step_idx % gradient_accumulation_steps
        accumulation_window_start: int = step_idx - accumulation_index
        accumulation_window_size: int = min(gradient_accumulation_steps, num_itrs - accumulation_window_start)
        should_optimizer_step: bool = accumulation_index + 1 == accumulation_window_size
        if accumulation_index == 0:
            optimizer.zero_grad(set_to_none=True)
        lr_itr: float = lr * (lr_d**current_itr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_itr

        start_time: float = time.time()
        states_start_batch: Tensor
        states_goal_batch: Tensor
        actions_batch: Tensor
        ctgs_targ_batch: Tensor
        if use_dataloader:
            if dataloader is None:
                raise RuntimeError("Dataloader must be initialized when use_dataloader=True")
            try:
                assert dataloader_iter is not None
                batch_data: tuple[Tensor, Tensor, Tensor, Tensor] = next(dataloader_iter)

            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch_data = next(dataloader_iter)

            states_start_batch, states_goal_batch, actions_batch, ctgs_targ_batch = batch_data
            actions_batch = actions_batch.unsqueeze(1)
        else:
            if rand_batch_idxs is None:
                raise RuntimeError("Random batch indices must be initialized when use_dataloader=False")
            batch_idxs: NDArray[np.intp] = rand_batch_idxs[start_batch_idx:end_batch_idx]
            states_start_batch = torch.from_numpy(states_start_np[batch_idxs]).float()
            states_goal_batch = torch.from_numpy(states_goal_np[batch_idxs]).float()
            actions_batch = torch.from_numpy(actions_np[batch_idxs]).unsqueeze(1)
            ctgs_targ_batch = torch.from_numpy(ctgs_np[batch_idxs]).float()

        states_start: Tensor = transfer_tensor_to_device(states_start_batch, device, memory_cfg=memory_cfg)
        states_goal: Tensor = transfer_tensor_to_device(states_goal_batch, device, memory_cfg=memory_cfg)
        actions: Tensor = transfer_tensor_to_device(actions_batch, device, memory_cfg=memory_cfg)
        ctgs_targ: Tensor = transfer_tensor_to_device(ctgs_targ_batch, device, memory_cfg=memory_cfg)

        ctgs_nnet: Tensor = dqn(states_start, states_goal)
        ctgs_nnet_act: Tensor = ctgs_nnet.gather(1, actions)[:, 0]

        record_time(times, "fprop", start_time, on_gpu)

        # Backprop and step
        start_time = time.time()
        nnet_minus_targ: Tensor = ctgs_nnet_act - ctgs_targ
        squared_err: Tensor = torch.pow(nnet_minus_targ, 2)
        abs_err: Tensor = torch.abs(nnet_minus_targ)
        huber_err: Tensor = 0.5 * squared_err * (abs_err <= 1.0) + (abs_err - 0.5) * (abs_err > 1.0)

        loss: Tensor = (squared_err * (nnet_minus_targ >= 0) + huber_err * (nnet_minus_targ < 0)).mean()
        if gradient_accumulation_steps == 1:
            loss.backward()
        else:
            (loss / accumulation_window_size).backward()

        if should_optimizer_step:
            clip_norm: float = optimization_cfg.gradient_clipping if optimization_cfg is not None else 1.0

            if isinstance(dqn, FSDP) and hasattr(dqn, "clip_grad_norm_"):
                dqn.clip_grad_norm_(max_norm=clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=clip_norm)

            optimizer.step()

        last_loss = loss.item()

        # Distributed synchronization for consistent logging
        if dist.is_initialized():
            loss_tensor: Tensor = torch.tensor(last_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            last_loss = loss_tensor.item()

        record_time(times, "bprop", start_time, on_gpu)

        # Display progress
        if (current_itr % display_itrs == 0) and display:
            times["itr"] = time.time() - start_time_itr
            time_str: str = get_time_str(times)
            logger.info(
                f"Itr: {current_itr}, "
                f"lr: {lr_itr:.2E}, "
                f"loss: {loss.item():.2E}, "
                f"targ_ctg: {ctgs_targ.mean().item():.2f}, "
                f"nnet_ctg: {ctgs_nnet_act.mean().item():.2f}, "
                f"Times - {time_str}"
            )

            start_time_itr = time.time()
            for key in times:
                times[key] = 0.0

        if not use_dataloader:
            if rand_batch_idxs is None:
                raise RuntimeError("Random batch indices must be initialized when use_dataloader=False")
            start_batch_idx = end_batch_idx
            end_batch_idx = start_batch_idx + batch_size
            if end_batch_idx > rand_batch_idxs.shape[0]:
                rand_batch_idxs = np.random.permutation(num_exs)
                start_batch_idx = 0
                end_batch_idx = start_batch_idx + batch_size

    return last_loss


def load_data(
    env: ABCEnvironment[ABCState],
    env_model: nn.Module,
    states_offline_np: NDArray[np.float32],
    device: torch.device,
    dqn_cfg: DQNTrainConfig,
    curr_dir: str,
    model_cfg: ModelConfig | None = None,
) -> tuple[nn.Module, int, int, NDArray[np.float32], NDArray[np.float32], float]:
    """Load or initialize a DQN and generate its evaluation state pairs.

    Args:
        env: Environment instance
        env_model: Environment neural network model
        states_offline_np: Offline state data for sampling
        device: PyTorch device for computation
        dqn_cfg: DQN training configuration
        curr_dir: Current model directory path
        model_cfg: Model architecture configuration

    Returns:
        Tuple containing:
        - nnet: Loaded or initialized neural network
        - itr: Current iteration number
        - update_num: Current update number
        - states_start_t_np: Test start states
        - states_goal_t_np: Test goal states
        - per_solved_best: Best percentage solved so far
    """
    if model_cfg is None:
        model_cfg = ModelConfig(
            chan_in=6,  # Default channel count
            chan_out=6,
            enc_dim=400,  # Default encoding dimension
            dqn=ModelArchitectureConfig(),  # Default DQN architecture
        )

    nnet_file: str = f"{curr_dir}/model_state_dict.pt"
    if pathlib.Path(nnet_file).is_file():
        try:
            nnet: nn.Module = load_nnet(nnet_file, env.get_dqn(model_cfg))
        except Exception:
            # Fallback to ZeroModel if environment method fails
            nnet = ZeroModel(env.num_actions_max, torch.device("cpu"))

            itr: int
            update_num: int
            states_start_t_np: NDArray[np.float32]
            states_goal_t_np: NDArray[np.float32]
            per_solved_best: float

        with pathlib.Path(f"{curr_dir}/status.pkl").open("rb") as status_file:
            itr, update_num, states_start_t_np, states_goal_t_np, per_solved_best = pickle.load(status_file)

        logger.info(f"Loaded with itr: {itr}, update_num: {update_num}, per_solved_best: {per_solved_best}")

    else:
        try:
            nnet = env.get_dqn(model_cfg)

        except Exception:
            # Fallback to ZeroModel if environment method fails
            nnet = ZeroModel(env.num_actions_max, torch.device("cpu"))

        itr = 0
        update_num = 0
        per_solved_best = 0.0

        samp_idxs: NDArray[np.intp] = np.asarray(
            np.random.randint(0, states_offline_np.shape[0], size=dqn_cfg.num_test), dtype=np.intp
        )
        states_start_t_np = states_offline_np[samp_idxs]
        goal_steps_samp: list[int] = list(np.random.randint(0, dqn_cfg.goal_steps + 1, size=dqn_cfg.num_test))
        num_actions: int = env.num_actions_max
        states_goal_t_np = random_walk(states_start_t_np, goal_steps_samp, num_actions, env_model, device)

    return nnet, itr, update_num, states_start_t_np.copy(), states_goal_t_np.copy(), per_solved_best


# ==============================================================================
# Distributed and Memory Utilities
# ==============================================================================


def setup_distributed_training() -> tuple[int, int, int]:
    """Initialize the distributed process group from environment variables.

    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        # torchrun sets LOCAL_RANK, so use the NCCL backend for GPU communication
        dist.init_process_group(backend="nccl")
        rank: int = dist.get_rank()
        world_size: int = dist.get_world_size()
        local_rank: int = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Fallback for manual distributed setup
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def pin_tensor_memory(tensor: Tensor, memory_cfg: MemoryConfig | None = None, pin_memory: bool = False) -> Tensor:
    """Pin a CPU tensor's memory when pinning is enabled.

    Args:
        tensor: Input tensor
        memory_cfg: Memory configuration object
        pin_memory: Whether to pin memory for DMA transfers (fallback)

    Returns:
        The pinned tensor, or the input tensor unchanged when pinning
        is disabled or the tensor is not on the CPU
    """
    # Use config values if provided, otherwise fall back to function parameters
    if memory_cfg is not None:
        pin_memory = memory_cfg.pin_memory

    if pin_memory and tensor.device.type == "cpu":
        return tensor.pin_memory()
    return tensor


def transfer_tensor_to_device(
    tensor: Tensor, device: torch.device, memory_cfg: MemoryConfig | None = None, non_blocking: bool = True
) -> Tensor:
    """Transfer a tensor to the target device.

    The transfer is non-blocking when the source tensor is pinned and the
    target is a CUDA device, and synchronous otherwise.

    Args:
        tensor: Input tensor
        device: Target device
        memory_cfg: Memory configuration object
        non_blocking: Enable non-blocking transfer for pinned memory (fallback)

    Returns:
        Transferred tensor
    """
    # Use config values if provided, otherwise fall back to function parameters
    pin_memory: bool = memory_cfg.pin_memory if memory_cfg is not None else True

    if tensor.is_pinned() and device.type == "cuda" and pin_memory:
        return tensor.to(device, non_blocking=non_blocking)
    return tensor.to(device)


def distributed_barrier_sync() -> None:
    """Enter a process-group barrier when distributed execution is active."""
    if dist.is_initialized():
        dist.barrier()


def cast_bool_for_nccl(tensor: Tensor) -> Tensor:
    """Convert a Boolean tensor to uint8 for NCCL collectives.

    Args:
        tensor: Tensor passed to an NCCL collective.

    Returns:
        A uint8 tensor when the input is Boolean, otherwise the original tensor.
    """
    if tensor.dtype == torch.bool:
        return tensor.to(torch.uint8)
    return tensor


def uncast_bool_from_nccl(tensor: Tensor, original_dtype: torch.dtype) -> Tensor:
    """Restore a uint8 collective result to its original Boolean dtype.

    Args:
        tensor: Tensor returned by the collective.
        original_dtype: Dtype recorded before the collective.

    Returns:
        A Boolean tensor when the original dtype was Boolean, otherwise the input tensor.
    """
    if original_dtype == torch.bool and tensor.dtype == torch.uint8:
        return tensor.to(torch.bool)
    return tensor


def save_distributed_checkpoint(dqn: nn.Module | FSDP, optimizer: Optimizer, checkpoint_dir: str, epoch: int) -> None:
    """Save DQN training state with DCP when distributed execution is active.

    Args:
        dqn: DQN model to checkpoint.
        optimizer: Optimizer to checkpoint.
        checkpoint_dir: Parent directory for checkpoint data.
        epoch: Current epoch number.
    """
    pathlib.Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    checkpoint_path: pathlib.Path = pathlib.Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}"
    if dist.is_initialized():
        model_state_dict, optimizer_state_dict = get_state_dict(dqn, optimizer)
        save_distributed_state_dict(
            state_dict={"dqn_model": model_state_dict, "optimizer": optimizer_state_dict, "epoch": epoch},
            checkpoint_id=str(checkpoint_path),
            process_group=dist.group.WORLD,
        )
        return

    torch.save(
        {"dqn_model": dqn.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, f"{checkpoint_path}.pt"
    )


def create_shared_memory_tensor(tensor: Tensor, name: str | None = None) -> Tensor:
    """Move a tensor to CPU shared memory.

    Args:
        tensor: Tensor to share.
        name: Reserved name for callers that track shared buffers.

    Returns:
        CPU tensor backed by shared memory.
    """
    _ = name
    cpu_tensor: Tensor = tensor.cpu()
    shared_tensor: Tensor = cpu_tensor.share_memory_()
    return shared_tensor


def setup_nccl_environment(nccl_cfg: NCCLConfig | None = None) -> None:
    """Set NCCL environment variables before the process group starts.

    The defaults select the Mellanox InfiniBand HCA and raise the
    peer-to-peer and GPUDirect RDMA levels, which multi-node jobs need
    for direct GPU-to-NIC transfers.

    Args:
        nccl_cfg: NCCL environment settings.
    """
    if nccl_cfg is None:
        # Default values if no config provided
        nccl_env_vars: dict[str, str] = {
            "NCCL_IB_HCA": "mlx5",
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_NET_GDR_LEVEL": "PHB",
            "NCCL_TREE_THRESHOLD": "0",
            "NCCL_DEBUG": "INFO",
        }
    else:
        nccl_env_vars = {
            "NCCL_IB_HCA": nccl_cfg.ib_hca,
            "NCCL_P2P_LEVEL": nccl_cfg.p2p_level,
            "NCCL_NET_GDR_LEVEL": nccl_cfg.net_gdr_level,
            "NCCL_TREE_THRESHOLD": nccl_cfg.tree_threshold,
            "NCCL_DEBUG": nccl_cfg.debug_level,
        }

    for key, value in nccl_env_vars.items():
        if key not in os.environ:
            os.environ[key] = value


def distributed_reduce_scatter(tensor: Tensor, group: ProcessGroup | None = None) -> Tensor:
    """Reduce and scatter a tensor, converting Boolean data for NCCL.

    Args:
        tensor: Input tensor to scatter.
        group: Process group for communication.

    Returns:
        Portion assigned to the current rank.
    """
    if not dist.is_initialized():
        return tensor

    # Handle Boolean tensors for NCCL compatibility
    original_dtype: torch.dtype = tensor.dtype
    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.uint8)

    world_size: int = dist.get_world_size(group)
    input_list: list[Tensor] = list(tensor.chunk(world_size, dim=0))

    # Pad uneven chunks to the largest chunk size required by reduce_scatter.
    max_size: int = max(chunk.numel() for chunk in input_list)
    padded_list: list[Tensor] = []
    for chunk in input_list:
        if chunk.numel() < max_size:
            padding: Tensor = torch.zeros(max_size - chunk.numel(), dtype=tensor.dtype, device=tensor.device)
            padded_chunk: Tensor = torch.cat([chunk.flatten(), padding])
        else:
            padded_chunk = chunk.flatten()
        padded_list.append(padded_chunk)

    output: Tensor = torch.empty_like(padded_list[0])
    dist.reduce_scatter(output, padded_list, group=group)

    # Restore original dtype if needed
    if original_dtype == torch.bool:
        output = output.to(torch.bool)

    return output


def distributed_all_gather(tensor: Tensor, group: ProcessGroup | None = None) -> Tensor:
    """Gather one tensor from every rank, converting Boolean data for NCCL.

    Args:
        tensor: Input tensor to gather.
        group: Process group for communication.

    Returns:
        Rank-ordered concatenation of the gathered tensors.
    """
    if not dist.is_initialized():
        return tensor

    # Handle Boolean tensors for NCCL compatibility
    original_dtype: torch.dtype = tensor.dtype
    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.uint8)

    world_size: int = dist.get_world_size(group)
    gathered_list: list[Tensor] = [torch.empty_like(tensor) for _ in range(world_size)]

    dist.all_gather(gathered_list, tensor, group=group)

    # Concatenate results
    result: Tensor = torch.cat(gathered_list, dim=0)

    # Restore original dtype if needed
    if original_dtype == torch.bool:
        result = result.to(torch.bool)

    return result


def create_persistent_pinned_memory_pool(size_bytes: int) -> Tensor:
    """Create a persistent pool of pinned host memory.

    Pinning CPU pages is only useful together with
    ``.to(device, non_blocking=True)``, and re-pinning every batch is slow,
    so the pool is allocated and pinned once up front.

    Args:
        size_bytes: Size of the memory pool in bytes

    Returns:
        Pinned memory tensor pool
    """
    # Create a large CPU tensor and pin it once
    num_elements: int = size_bytes // 4  # Assuming float32
    pool: Tensor = torch.empty(num_elements, dtype=torch.float32, device="cpu")
    return pool.pin_memory()


# ==============================================================================
# Data Loading
# ==============================================================================


class SharedMemoryDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Dataset backed by shared-memory tensors.

    Multi-process DataLoader workers copy a regular in-memory dataset into
    every worker process. Placing the tensors in shared memory lets all
    workers read the same buffers instead of holding private copies.
    """

    def __init__(
        self,
        states_start: NDArray[np.float32],
        states_goal: NDArray[np.float32],
        actions: NDArray[np.int32],
        ctgs: NDArray[np.float32],
        use_shared_memory: bool = True,
    ) -> None:
        """Initialize the dataset, optionally moving tensors to shared memory.

        Args:
            states_start: Start states array
            states_goal: Goal states array
            actions: Actions array
            ctgs: Cost-to-go values array
            use_shared_memory: Whether to use shared memory for zero-copy access
        """
        super().__init__()
        self.length: int = len(states_start)

        if use_shared_memory:
            # Create shared memory tensors for zero-copy access across workers
            self.states_start: Tensor = create_shared_memory_tensor(torch.from_numpy(states_start))
            self.states_goal: Tensor = create_shared_memory_tensor(torch.from_numpy(states_goal))
            self.actions: Tensor = create_shared_memory_tensor(torch.from_numpy(actions))
            self.ctgs: Tensor = create_shared_memory_tensor(torch.from_numpy(ctgs))
        else:
            # Regular tensors
            self.states_start = torch.from_numpy(states_start)
            self.states_goal = torch.from_numpy(states_goal)
            self.actions = torch.from_numpy(actions)
            self.ctgs = torch.from_numpy(ctgs)

    def __len__(self) -> int:
        """Return dataset length."""
        return self.length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get item at index."""
        return (self.states_start[index], self.states_goal[index], self.actions[index], self.ctgs[index])


def create_shared_memory_dataloader(
    states_start: NDArray[np.float32],
    states_goal: NDArray[np.float32],
    actions: NDArray[np.int32],
    ctgs: NDArray[np.float32],
    batch_size: int,
    memory_cfg: MemoryConfig | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Create a DataLoader over a :class:`SharedMemoryDataset`.

    Args:
        states_start: Start states
        states_goal: Goal states
        actions: Actions taken
        ctgs: Cost-to-go values
        batch_size: Batch size
        memory_cfg: Memory configuration object
        num_workers: Number of worker processes, kept low because each
            worker adds memory overhead (fallback)
        pin_memory: Whether to pin memory (fallback)

    Returns:
        DataLoader instance
    """
    # Use config values if provided, otherwise fall back to function parameters
    if memory_cfg is not None:
        use_shared_memory: bool = memory_cfg.shared_memory
        safe_num_workers: int = min(memory_cfg.max_workers, num_workers or memory_cfg.max_workers)
        pin_memory = memory_cfg.pin_memory
        persistent_workers: bool = memory_cfg.persistent_workers
        prefetch_factor: int = memory_cfg.prefetch_factor
    else:
        use_shared_memory = num_workers > 0
        safe_num_workers = min(num_workers, 4)
        persistent_workers = safe_num_workers > 0
        prefetch_factor = 2

    dataset = SharedMemoryDataset(states_start, states_goal, actions, ctgs, use_shared_memory=use_shared_memory)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=safe_num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if safe_num_workers > 0 else None,
    )


def run_data_gen(
    env_name: str,
    train_data_path: str,
    batch_size: int,
    num_batches: int,
    start_steps: int,
    goal_steps: int,
    max_steps: int,
    *,
    per_eq_tol: float,
    transition_model_path: str,
    model_cfg: ModelConfig,
    dqn_curr_dir: str,
    dqn_targ_dir: str,
    dqn: nn.Module,
    device: torch.device,
    comm_mode: str = "auto",
) -> tuple[DataQueueType, TimeQueueType]:
    """Main function to run DQN heuristic training using config-based workflow.

    Args:
        env_name: Name of environment to train on
        train_data_path: Path to training data
        batch_size: Batch size for training
        num_batches: Number of batches to generate
        start_steps: Number of start steps for training
        goal_steps: Number of goal steps for training
        max_steps: Maximum steps for training
        per_eq_tol: Tolerance for per-equation convergence
        transition_model_path: Path to transition model directory
        model_cfg: Model architecture config for DQN and transition models
        dqn_curr_dir: Current DQN model directory
        dqn_targ_dir: Target DQN model directory
        dqn: DQN neural network to use for training
        device: PyTorch device for computation
        comm_mode: Communication mode for distributed training (default: "auto")
    """
    dqn.eval()
    distributed_barrier_sync()

    times: TimeQueueType
    (s_start, s_goal, acts, ctgs), times = q_update(
        env_name=env_name,
        data_file=train_data_path,
        batch_size=batch_size,
        num_batches=num_batches,
        start_steps=start_steps,
        goal_steps=goal_steps,
        per_eq_tol=per_eq_tol,
        max_steps=max_steps,
        env_model_dir=transition_model_path,
        dqn_dir=dqn_curr_dir,
        dqn_targ_dir=dqn_targ_dir,
        model_cfg=model_cfg,
        comm_mode=comm_mode,
        device=device,
    )

    distributed_barrier_sync()
    return (s_start, s_goal, acts, ctgs), times


def run_train(
    dqn: nn.Module,
    optimizer: Optimizer,
    states_start_np: NDArray[np.float32 | np.integer],
    states_goal_np: NDArray[np.float32 | np.integer],
    actions_np: NDArray[np.integer],
    ctgs: NDArray[np.float32 | np.integer],
    *,
    batch_size: int,
    device: torch.device,
    on_gpu: bool,
    num_itrs: int,
    train_itr: int,
    lr: float,
    lr_d: float,
    compile_cfg: CompileConfig,
    memory_cfg: MemoryConfig,
    optimization_cfg: OptimizationConfig | None = None,
    display: bool = True,
    use_dataloader: bool = True,
) -> float:
    """Main function to run DQN heuristic training using a persistent optimizer."""
    dqn.train()
    last_loss: float = train_nnet(
        dqn=dqn,
        states_start_np=states_start_np.astype(np.float32),
        states_goal_np=states_goal_np.astype(np.float32),
        actions_np=actions_np.astype(np.int32),
        ctgs_np=ctgs.astype(np.float32),
        batch_size=batch_size,
        device=device,
        on_gpu=on_gpu,
        num_itrs=num_itrs,
        train_itr=train_itr,
        lr=lr,
        lr_d=lr_d,
        optimizer=optimizer,
        memory_cfg=memory_cfg,
        optimization_cfg=optimization_cfg,
        compile_cfg=compile_cfg,
        display=display,
        use_dataloader=use_dataloader,
    )
    return last_loss
