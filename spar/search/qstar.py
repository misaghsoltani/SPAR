"""Q* search for SPAR."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from logging import DEBUG, getLogger
import pathlib
import pickle
import time
from typing import TYPE_CHECKING, Literal, TypedDict

import h5py
import numpy as np
from numpy import float32
import orjson
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
import torch

from spar.environments.abstracts import ABCState
from spar.environments.sokoban.sokoban import SokobanState
from spar.utils.log_utils.console_logger import terminal_console as console
from spar.utils.pytorch_utils.nnet_utils import load_model as load_model_ptu
from spar.utils.search_utils.misc_utils import flatten, unflatten
from spar.utils.search_utils.nnet_utils import HeuristicFnWrapper, ModelFnWrapper, apply_cuda_tf32_settings, get_device
from spar.utils.search_utils.solve_utils import is_valid_soln
from spar.utils.search_utils.summary_utils import QStarSummaryTracker, build_summary_panel
from spar.utils.search_utils.viz_utils import ImageHandler

if TYPE_CHECKING:
    from logging import Logger
    from typing import NoReturn, TypeAlias

    from h5py import Group as H5PyGroup
    from numpy import uint8
    from numpy.typing import NDArray
    from rich.progress import TaskID
    from torch import nn
    from typing_extensions import Self

    from spar.environments.abstracts import ABCEnvironment
    from spar.utils.config_utils.config_schema import ModelConfig, SearchStageConfig
    from spar.utils.search_utils.nnet_utils import CallableModelType
    from spar.utils.search_utils.summary_utils import (
        QStarLogEntry,
        QStarOutputSummary,
        QStarSummaryStats,
        SolveCategory,
    )
    from spar.utils.search_utils.viz_utils import QStarImageContext


logger: Logger = getLogger(__name__)

OpenSetElem: TypeAlias = tuple[float, int, "Node", int]
PairItem: TypeAlias = "tuple[int, NDArray[float32], NDArray[float32], str, str, ABCState | None, ABCState | None]"
VisMode: TypeAlias = Literal["none", "image", "gif", "both"]
IndividualSizeMode: TypeAlias = Literal["original", "custom"]
SearchMode: TypeAlias = Literal["qstar", "gbfs", "ucs"]
SearchOutputValue: TypeAlias = (
    float | int | bool | str | list["SearchOutputValue"] | dict[str, "SearchOutputValue"] | None
)
HeuristicFn: TypeAlias = "Callable[[NDArray[float32], NDArray[float32]], NDArray[float32]]"
ModelFn: TypeAlias = "Callable[[NDArray[float32], NDArray[float32]], NDArray[float32]]"
IsSolvedFn: TypeAlias = "Callable[[NDArray[float32], NDArray[float32]], NDArray[np.bool_]]"


def _as_vis_mode(value: str) -> VisMode:
    if value == "none":
        return "none"
    if value == "image":
        return "image"
    if value == "gif":
        return "gif"
    if value == "both":
        return "both"
    raise ValueError(f"Invalid visualization mode: {value!r}")


def _as_individual_size_mode(value: str) -> IndividualSizeMode:
    if value == "original":
        return "original"
    if value == "custom":
        return "custom"
    raise ValueError(f"Invalid individual size mode: {value!r}")


def _raise_invalid_pairs_schema(message: str) -> NoReturn:
    raise TypeError(message)


def _raise_invalid_pairs_data(message: str) -> NoReturn:
    raise ValueError(message)


def _state_pickle_integrity_errors(start_states_ds_chk: h5py.Dataset, goal_states_ds_chk: h5py.Dataset) -> list[str]:
    err_msgs: list[str] = []
    for ds_chk, side in ((start_states_ds_chk, "start"), (goal_states_ds_chk, "goal")):
        if ds_chk.shape[0] == 0:
            err_msgs.append(f"{side} states_pickle has zero length")
            break
        sample: NDArray[uint8] = ds_chk[0]
        buf: bytes = sample.tobytes() if hasattr(sample, "tobytes") else bytes(sample)
        # Protocol 2 and later pickle streams begin with 0x80.
        if len(buf) == 0 or buf[0] not in {0x80, 0x00}:
            err_msgs.append(f"{side} states_pickle appears malformed (empty or invalid header)")
            break
    return err_msgs


def inspect_h5_pair_inputs(
    h5f: h5py.File,
) -> tuple[int, h5py.Group | None, h5py.Group | None, h5py.Dataset, h5py.Dataset, list[str], list[str]]:
    """Validate and describe an HDF5 pairs dataset's start/goal groups and available variants.

    Args:
        h5f: Open HDF5 file containing ``pairs/start`` and ``pairs/goal`` groups.

    Returns:
        A tuple of (total_pairs, start_variations_group, goal_variations_group,
        start_base_images, goal_base_images, start_available_variants, goal_available_variants).
    """
    start_group_obj = h5f.get("pairs/start")
    goal_group_obj = h5f.get("pairs/goal")
    if not isinstance(start_group_obj, h5py.Group) or not isinstance(goal_group_obj, h5py.Group):
        _raise_invalid_pairs_schema(
            "Invalid search pairs HDF5: missing or invalid groups at pairs/start or pairs/goal."
        )
    start_group: H5PyGroup = start_group_obj
    goal_group = goal_group_obj

    start_base_obj = start_group.get("images")
    goal_base_obj = goal_group.get("images")
    if not isinstance(start_base_obj, h5py.Dataset) or not isinstance(goal_base_obj, h5py.Dataset):
        _raise_invalid_pairs_schema(
            "Invalid search pairs HDF5: missing or invalid datasets at pairs/start/images or pairs/goal/images."
        )
    start_base_ds = start_base_obj
    goal_base_ds = goal_base_obj

    total_pairs = start_base_ds.shape[0]
    if goal_base_ds.shape[0] != total_pairs:
        _raise_invalid_pairs_data("Start and goal base datasets have different lengths")

    start_vg_obj = start_group.get("variations")
    goal_vg_obj = goal_group.get("variations")
    start_vg: h5py.Group | None = start_vg_obj if isinstance(start_vg_obj, h5py.Group) else None
    goal_vg: h5py.Group | None = goal_vg_obj if isinstance(goal_vg_obj, h5py.Group) else None

    # Require base ABCState pickles for validation and env visuals when using HDF5 pairs
    start_states_obj = start_group.get("states_pickle")
    goal_states_obj = goal_group.get("states_pickle")
    start_states_ds_chk: h5py.Dataset | None = start_states_obj if isinstance(start_states_obj, h5py.Dataset) else None
    goal_states_ds_chk: h5py.Dataset | None = goal_states_obj if isinstance(goal_states_obj, h5py.Dataset) else None
    err_msgs: list[str] = []
    if start_states_ds_chk is None:
        err_msgs.append("missing pairs/start/states_pickle")
    if goal_states_ds_chk is None:
        err_msgs.append("missing pairs/goal/states_pickle")
    if not err_msgs and start_states_ds_chk is not None and goal_states_ds_chk is not None:
        # Minimal integrity test: try to read and unpickle the first element of both datasets
        try:
            err_msgs.extend(_state_pickle_integrity_errors(start_states_ds_chk, goal_states_ds_chk))
        except Exception as e:
            err_msgs.append(f"states_pickle integrity check failed: {e}")
    if err_msgs:
        _raise_invalid_pairs_data(
            f"Invalid search pairs HDF5 for environment validation and visuals: {'. '.join(err_msgs)}."
        )

    start_avail: list[str] = ["base"] + (list(start_vg.keys()) if isinstance(start_vg, h5py.Group) else [])
    goal_avail: list[str] = ["base"] + (list(goal_vg.keys()) if isinstance(goal_vg, h5py.Group) else [])
    return total_pairs, start_vg, goal_vg, start_base_ds, goal_base_ds, start_avail, goal_avail


def _deserialize_state(buf: bytes, *, side: str, pair_index: int) -> ABCState | None:
    try:
        loaded = pickle.loads(buf)
    except Exception as error:
        logger.debug(f"Failed to unpickle {side} ABCState at idx {pair_index}: {error}")
        return None
    if isinstance(loaded, ABCState):
        return loaded
    logger.debug(f"Decoded {side} payload at idx {pair_index} is not an ABCState instance")
    return None


class QStarResults(TypedDict):
    """JSON-serializable summary of a Q* run.

    The payload stores scalar summaries and log rows without raw state or path arrays.
    """

    logs: list[QStarLogEntry]
    summary: QStarOutputSummary


class Node:
    """Represents a node in the search tree.

    Attributes:
        state: The state representation as NDArray.
        path_cost: The cost of the path to this node.
        parent_move: The move that led to this node from its parent.
        parent: Reference to the parent node.
        _hash: Cached hash value for the node.
    """

    __slots__: tuple[str, ...] = ("_hash", "_state_bytes", "parent", "parent_move", "path_cost", "state")

    def __init__(self, state: NDArray[float32], path_cost: float, parent_move: int | None, parent: Node | None) -> None:
        """Initializes a Node.

        Args:
            state (NDArray[float32]): The state representation.
            path_cost (float): The cost of the path to this node.
            parent_move (int | None): The move that led to this node from its parent.
            parent (Node | None): Reference to the parent node.
        """
        self.state: NDArray[float32] = state
        self.path_cost: float = path_cost
        self.parent_move: int | None = parent_move
        self.parent: Node | None = parent

        self._hash: int | None = None
        self._state_bytes: bytes | None = None

    def _bytes_key(self) -> bytes:
        """Lazily compute and cache the bytes representation of the state for hashing/closed set.

        Returns:
            bytes: Immutable byte representation of the state.
        """
        b: bytes | None = self._state_bytes
        if b is None:
            b = self.state.tobytes()
            self._state_bytes = b
        return b

    def bytes_key(self) -> bytes:
        """Return stable bytes key for closed-set lookups."""
        return self._bytes_key()

    def __hash__(self) -> int:
        """Returns the hash of the node.

        Returns:
            The hash value.
        """
        if self._hash is None:
            # Derive hash from cached bytes to avoid repeated tobytes conversions
            self._hash = hash(self._bytes_key())
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Checks if two nodes are equal.

        Args:
            other: The other object to compare.

        Returns:
            True if nodes are equal, False otherwise.
        """
        if not isinstance(other, Node):
            return NotImplemented

        return np.array_equal(self.state, other.state)


class Instance:
    """Represents a search instance with open/closed sets.

    Attributes:
        state: The initial state as NDArray.
        state_goal: The goal state as NDArray.
        cost: The cost associated with this instance.
        open_set: Priority queue of open nodes.
        closed_set: Set of closed nodes.
        root_node: The root node of the search tree.
    """

    __slots__: tuple[str, ...] = (
        "closed_dict",
        "goal_nodes",
        "heappush_count",
        "num_nodes_generated",
        "open_set",
        "root_node",
        "state_goal",
        "weight",
    )

    def __init__(self, state: NDArray[float32], state_goal: NDArray[float32], cost: float, weight: float) -> None:
        """Initializes an Instance.

        Args:
            state (np.NDArray): The initial state.
            state_goal (np.NDArray): The goal state.
            cost (float): The initial cost.
            weight (float): The weight for path cost.
        """
        self.state_goal: NDArray[float32] = state_goal

        self.open_set: list[OpenSetElem] = []

        # CLOSED keyed by immutable bytes of state for faster equality checks and fewer collisions
        self.closed_dict: dict[bytes, float] = {}

        self.heappush_count: int = 0
        self.goal_nodes: list[Node] = []
        self.num_nodes_generated: int = 0

        self.root_node: Node = Node(state, 0.0, None, None)
        self.weight: float = weight

        self.push_to_open([self.root_node], [[-1]], [[cost]])

    def push_to_open(self, nodes: list[Node], moves: list[list[int]], costs: list[list[float]]) -> None:
        """Pushes nodes to the open set.

        Args:
            nodes (list[Node]): The nodes to push.
            moves (list[list[int]]): The moves associated with the nodes.
            costs (list[list[float]]): The costs associated with the nodes.
        """
        heappush_count: int = self.heappush_count
        new_entries: list[tuple[float, int, Node, int]] = []
        append: Callable[[tuple[float, int, Node, int]], None] = new_entries.append
        for node, moves_node, costs_node in zip(nodes, moves, costs, strict=False):
            for move, cost in zip(moves_node, costs_node, strict=False):
                append((cost, heappush_count, node, move))
                heappush_count += 1
        self.heappush_count = heappush_count

        # Every entry carries a strictly increasing counter, so entries are
        # totally ordered and the popped sequence is their sorted order. That
        # makes the pop order independent of how the heap invariant is
        # restored, so bulk heapify and per-entry pushes are interchangeable.
        open_set: list[tuple[float, int, Node, int]] = self.open_set
        if len(new_entries) >= len(open_set):
            open_set.extend(new_entries)
            heapify(open_set)
        else:
            for entry in new_entries:
                heappush(open_set, entry)

    def pop_from_open(self, num_nodes: int) -> tuple[list[Node], list[int]]:
        """Pops nodes from the open set.

        Args:
            num_nodes (int): The number of nodes to pop.

        Returns:
            tuple[list[Node], list[int]]: The popped nodes and their associated moves.
        """
        num_to_pop: int = min(num_nodes, len(self.open_set))
        if num_to_pop == 0:
            return [], []
        popped_elems: list[OpenSetElem] = [heappop(self.open_set) for _ in range(num_to_pop)]
        popped_nodes: list[Node] = [elem[2] for elem in popped_elems]
        moves: list[int] = [elem[3] for elem in popped_elems]

        return popped_nodes, moves

    def remove_in_closed(self, nodes: list[Node]) -> list[Node]:
        """Removes nodes that are in the closed set.

        Args:
            nodes (list[Node]): The nodes to check.

        Returns:
            list[Node]: The nodes not in the closed set.
        """
        nodes_not_in_closed: list[Node] = []
        closed_dict: dict[bytes, float] = self.closed_dict

        key: bytes
        prev: float | None
        for node in nodes:
            key = node.bytes_key()
            prev = closed_dict.get(key)
            if prev is None or prev > node.path_cost:
                nodes_not_in_closed.append(node)
                closed_dict[key] = node.path_cost

        return nodes_not_in_closed


def _broadcast_state_goals(
    instances: list[Instance], nodes_by_inst: list[list[Node]], *, include_weights: bool = False
) -> tuple[NDArray[float32], list[float]]:
    """Materialize goal tensors aligned with `nodes_by_inst` and optional weights."""
    total: int = sum(len(nodes) for nodes in nodes_by_inst)
    if total == 0:
        return np.empty((0,), dtype=float32), []

    goal_shape: tuple[int, ...] = instances[0].state_goal.shape
    goals: NDArray[float32] = np.empty((total, *goal_shape), dtype=float32)
    weights: list[float] = []

    offset: int = 0
    for instance, nodes in zip(instances, nodes_by_inst, strict=False):
        count: int = len(nodes)
        if count == 0:
            continue
        goals[offset : offset + count] = instance.state_goal
        if include_weights:
            weights.extend([instance.weight] * count)
        offset += count

    return goals, weights


@torch.inference_mode()
def pop_from_open(
    instances: list[Instance], batch_size: int, model_fn: CallableModelType, is_solved_fn: IsSolvedFn
) -> list[list[Node]]:
    """Pops nodes from the open sets of instances.

    Args:
        instances (list[Instance]): The instances to pop nodes from.
        batch_size (int): The batch size.
        model_fn (Callable): The model function.
        is_solved_fn (Callable): The function to check if a state is solved.

    Returns:
        list[list[Node]]: The popped nodes for each instance.
    """
    popped_nodes_by_inst: list[list[Node]] = []
    moves_by_inst: list[list[int]] = []
    for instance in instances:
        popped_nodes_inst, moves_inst = instance.pop_from_open(batch_size)
        popped_nodes_by_inst.append(popped_nodes_inst)
        moves_by_inst.append(moves_inst)

    # make moves
    popped_nodes_flat: list[Node]
    split_idxs: list[int]
    moves_flat_list: list[int]
    popped_nodes_flat, split_idxs = flatten(popped_nodes_by_inst)
    moves_flat_list, _ = flatten(moves_by_inst)

    if not popped_nodes_flat:
        return [[] for _ in instances]

    initial_layer: bool = moves_flat_list[0] == -1
    if initial_layer:
        assert all(len(pn) == 1 for pn in popped_nodes_by_inst), "Initial nodes expected to be singular per instance"
        assert all(mv == -1 for mv in moves_flat_list), "Initial layer moves must use sentinel -1"
        states_next_flat: NDArray[float32] = np.stack([n.state for n in popped_nodes_flat], axis=0)
        popped_nodes_next_flat: list[Node] = popped_nodes_flat
    else:
        states_flat: NDArray[float32] = np.stack([n.state for n in popped_nodes_flat], axis=0)
        actions: NDArray[float32] = np.asarray(moves_flat_list, dtype=float32)
        states_next_flat = np.ascontiguousarray(model_fn(states_flat, actions).round(), dtype=float32)

        parent_costs: NDArray[float32] = np.fromiter(
            (n.path_cost for n in popped_nodes_flat), count=len(popped_nodes_flat), dtype=float32
        )
        child_costs: NDArray[float32] = parent_costs + 1.0
        popped_nodes_next_flat = [
            Node(state, float(cost), move, parent)
            for state, cost, move, parent in zip(
                states_next_flat, child_costs, moves_flat_list, popped_nodes_flat, strict=False
            )
        ]

    state_goals_flat, _ = _broadcast_state_goals(instances, popped_nodes_by_inst)

    # solved?
    is_solved_flat: list[bool] = list(is_solved_fn(states_next_flat, state_goals_flat))
    is_solved_by_inst: list[list[bool]] = unflatten(is_solved_flat, split_idxs)
    popped_nodes_next_by_inst_all: list[list[Node]] = unflatten(popped_nodes_next_flat, split_idxs)

    # update per instance and filter solved nodes out of the OPEN additions
    popped_nodes_next_by_inst_filtered: list[list[Node]] = []
    kept_nodes: list[Node]
    for instance, next_nodes_inst, flags_inst in zip(
        instances, popped_nodes_next_by_inst_all, is_solved_by_inst, strict=False
    ):
        # Append solved nodes to instance.goal_nodes
        instance.goal_nodes.extend([n for n, ok in zip(next_nodes_inst, flags_inst, strict=False) if ok])
        # Filter out solved nodes from next_nodes to avoid adding them back to OPEN
        kept_nodes = [n for n, ok in zip(next_nodes_inst, flags_inst, strict=False) if not ok]
        popped_nodes_next_by_inst_filtered.append(kept_nodes)
        instance.num_nodes_generated += len(next_nodes_inst)

    return popped_nodes_next_by_inst_filtered


@torch.inference_mode()
def add_heuristic_and_cost(
    nodes: list[list[Node]],
    state_goals_flat: NDArray[float32],
    heuristic_fn: HeuristicFn | None,
    weights: list[float],
    num_actions_max: int,
) -> tuple[list[list[list[int]]], list[list[list[float]]], NDArray[float32], NDArray[float32]]:
    """Adds heuristic and cost to nodes.

    Args:
        nodes (list[list[Node]]): The nodes to add heuristic and cost to.
        state_goals_flat (np.NDArray): The flattened goal states.
        heuristic_fn (Callable): The heuristic function.
        weights (list[float]): The weights for the path costs.
        num_actions_max (int): The maximum number of actions.

    Returns:
        tuple[list[list[list[int]]], list[list[list[float]]], NDArray, NDArray]: The moves,
            costs, parent path costs, and heuristics.
    """
    nodes_flat: list[Node]
    split_idxs: list[int]
    nodes_flat, split_idxs = flatten(nodes)

    if len(nodes_flat) == 0:
        return [], [], np.zeros(0, dtype=float32), np.zeros(0, dtype=float32)

    # get heuristic
    states_flat: NDArray[float32] = np.stack([n.state for n in nodes_flat], axis=0)
    parent_costs: NDArray[float32] = np.fromiter(
        (n.path_cost for n in nodes_flat), count=len(nodes_flat), dtype=float32
    )

    # compute node cost
    heuristics_flat: NDArray[float32]

    # If performing Q* search
    if heuristic_fn is not None:
        heuristics_flat = heuristic_fn(states_flat, state_goals_flat).astype(float32, copy=False)

    # If performing Uniform Cost Search
    else:
        heuristics_flat = np.zeros((states_flat.shape[0], num_actions_max), dtype=float32)

    weights_arr: NDArray[float32] = np.fromiter(weights, count=len(weights), dtype=float32)
    path_cost_weighted: NDArray[float32] = (weights_arr * parent_costs)[:, None]  # (N,1)
    heuristics_min: NDArray[float32] = heuristics_flat.min(axis=1)
    np.add(heuristics_flat, path_cost_weighted, out=heuristics_flat)  # (N,A)

    # Reuse one immutable move template because later code does not mutate it.
    moves_template: list[int] = list(range(num_actions_max))
    moves_flat: list[list[int]] = [moves_template] * heuristics_flat.shape[0]
    costs_flat: list[list[float]] = heuristics_flat.tolist()

    moves: list[list[list[int]]] = unflatten(moves_flat, split_idxs)
    costs: list[list[list[float]]] = unflatten(costs_flat, split_idxs)

    # return the real parent costs (float) and per-node min heuristic for logging
    return moves, costs, parent_costs, heuristics_min


def add_to_open(
    instances: list[Instance], nodes: list[list[Node]], moves: list[list[list[int]]], costs: list[list[list[float]]]
) -> None:
    """Adds nodes to the open sets of instances.

    Args:
        instances (list[Instance]): The instances to add nodes to.
        nodes (list[list[Node]]): The nodes to add.
        moves (list[list[list[int]]]): The moves associated with the nodes.
        costs (list[list[list[float]]]): The costs associated with the nodes.
    """
    for instance, nodes_inst, moves_inst, costs_inst in zip(instances, nodes, moves, costs, strict=False):
        instance.push_to_open(nodes_inst, moves_inst, costs_inst)


def get_path(node: Node) -> tuple[list[NDArray[float32]], list[int], float]:
    """Gets the path from the root to the given node.

    Args:
        node (Node): The node to trace back from.

    Returns:
        The float32 path arrays, moves, and path cost.
    """
    path: list[NDArray[float32]] = []
    moves: list[int] = []

    parent_node: Node = node
    while parent_node.parent is not None:
        path.append(parent_node.state)
        if parent_node.parent_move is not None:
            moves.append(parent_node.parent_move)
        parent_node = parent_node.parent

    path.append(parent_node.state)

    path.reverse()
    moves.reverse()

    return path, moves, node.path_cost


class IsSolvedTolerance:
    """Callable predicate for equality percentage between states.

    Avoids inner function definitions by exposing a __call__ with captured tolerance.
    """

    __slots__: tuple[str, ...] = ("per_eq_tol",)

    def __init__(self, per_eq_tol: float) -> None:
        self.per_eq_tol: float = per_eq_tol

    def _cuda_mask(self, states: NDArray[float32], states_comp: NDArray[float32]) -> NDArray[np.bool_]:
        """Compute the equality mask on the current CUDA device."""
        xs: torch.Tensor = torch.from_numpy(states)
        ys: torch.Tensor = torch.from_numpy(states_comp)
        # Pin host for non_blocking and move to the current CUDA device
        with contextlib.suppress(Exception):
            xs = xs.pin_memory()
            ys = ys.pin_memory()
        xs = xs.to(torch.cuda.current_device(), non_blocking=True)
        ys = ys.to(torch.cuda.current_device(), non_blocking=True)
        eq: torch.Tensor = (xs == ys).float()
        # Reduce along axis=1 only to match numpy code
        perc: torch.Tensor = (100.0 * eq.mean(dim=1)).to(dtype=torch.float32)
        out: torch.Tensor = perc >= self.per_eq_tol
        return out.to("cpu").numpy().astype(np.bool_)

    def __call__(self, states: NDArray[float32], states_comp: NDArray[float32]) -> NDArray[np.bool_]:
        """Return mask where per-state equality percentage >= tolerance.

        Uses a CUDA-accelerated path when available. Keeps the same axis-reduction
        semantics as the numpy version (mean along axis=1).
        """
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                return self._cuda_mask(states, states_comp)
        perc_equal: NDArray[float32] = (100.0 * np.equal(states, states_comp).mean(axis=1)).astype(float32)
        return (perc_equal >= self.per_eq_tol).astype(np.bool_)


class QStarImag:
    """Q* search algorithm."""

    __slots__: tuple[str, ...] = ("instances", "last_node", "num_actions_max", "step_num", "timings", "weights")

    weights: list[float]
    step_num: int
    num_actions_max: int
    timings: dict[str, float]
    instances: list[Instance]
    last_node: Node | None

    @torch.inference_mode()
    def __init__(
        self,
        states: NDArray[float32],
        state_goals: NDArray[float32],
        heuristic_fn: CallableModelType | None,
        weights: list[float],
        num_actions_max: int,
    ) -> None:
        """Initializes a QStarImag instance.

        Args:
            states (np.NDArray): The initial states.
            state_goals (np.NDArray): The goal states.
            heuristic_fn (Callable): The heuristic function.
            weights (list[float]): The weights for the path costs.
            num_actions_max (int): The maximum number of actions.
        """
        self.weights: list[float] = weights
        self.step_num: int = 0
        self.num_actions_max: int = num_actions_max
        self.timings: dict[str, float] = {"pop": 0.0, "closed": 0.0, "heur": 0.0, "add": 0.0, "itr": 0.0}

        # compute starting costs
        # Heuristic values if performing Q* search, zero if performing Uniform Cost Search
        costs: NDArray[float32] = (
            heuristic_fn(states, state_goals).min(axis=1)
            if heuristic_fn is not None
            else np.zeros(len(states), dtype=float32)
        )

        # initialize instances
        self.instances: list[Instance] = [
            Instance(state, state_goal, cost, weights[i])
            for i, (state, state_goal, cost) in enumerate(zip(states, state_goals, costs, strict=False))
        ]

        self.last_node: Node | None = None

    # TODO: make separate is_solved_fn and is_same_fn

    @torch.inference_mode()
    def step(
        self,
        heuristic_fn: CallableModelType | None,
        model_fn: CallableModelType,
        is_solved_fn: IsSolvedFn,
        batch_size: int,
        verbose: bool = False,
    ) -> bool:
        """Performs a step in the Q* search.

        Args:
            heuristic_fn (Callable): The heuristic function.
            model_fn (Callable): The model function.
            is_solved_fn (Callable): The function to check if a state is solved.
            batch_size (int): The batch size.
            verbose (bool): Whether to print verbose output.

        Returns:
            bool: True if the search continues, False if no more nodes to expand.
        """
        start_time_itr: float = time.time()
        instances: list[Instance] = [
            instance for instance in self.instances if (len(instance.goal_nodes) == 0) and len(instance.open_set) > 0
        ]
        if len(instances) == 0:
            logger.info("Open set is empty. Returning the result ...")
            return False

        # Pop from open
        start_time: float = time.time()
        popped_nodes: list[list[Node]] = pop_from_open(instances, batch_size, model_fn, is_solved_fn)
        pop_time: float = time.time() - start_time

        # Check if popped nodes are in closed
        start_time = time.time()
        for inst_idx, instance in enumerate(instances):
            popped_nodes[inst_idx] = instance.remove_in_closed(popped_nodes[inst_idx])
        closed_time: float = time.time() - start_time

        if len(popped_nodes) > 0:
            popped: list[Node] = popped_nodes[-1]
            if len(popped) > 0:
                self.last_node = popped[-1]

        # Get heuristic of children
        start_time = time.time()
        state_goals_flat: NDArray[float32]
        weights_list: list[float]
        state_goals_flat, weights_list = _broadcast_state_goals(instances, popped_nodes, include_weights=True)
        moves: list[list[list[int]]]
        costs: list[list[list[float]]]
        path_costs: NDArray[float32]
        heuristics: NDArray[float32]
        moves, costs, path_costs, heuristics = add_heuristic_and_cost(
            popped_nodes, state_goals_flat, heuristic_fn, weights_list, self.num_actions_max
        )
        heur_time: float = time.time() - start_time

        # Add to open
        start_time = time.time()
        add_to_open(instances, popped_nodes, moves, costs)
        add_time: float = time.time() - start_time

        itr_time: float = time.time() - start_time_itr

        # Print to screen
        if verbose:
            if heuristics.shape[0] > 0:
                min_heur: float = float(np.min(heuristics))
                min_heur_pc: float = path_costs[np.argmin(heuristics)]
                max_heur: float = float(np.max(heuristics))
                max_heur_pc: float = path_costs[np.argmax(heuristics)]

                logger.info(
                    f"Itr: {self.step_num}, Added to OPEN - Min/Max Heur(PathCost): "
                    f"{min_heur:.2f}({min_heur_pc:.2f})/{max_heur:.2f}({max_heur_pc:.2f}) "
                )

            logger.info(
                f"\n---------------\n\n"
                f"Times - pop: {pop_time:.2f}, closed: {closed_time:.2f}, heur: {heur_time:.2f}, "
                f"add: {add_time:.2f}, itr: {itr_time:.2f}"
            )

        # Update timings
        self.timings["pop"] += pop_time
        self.timings["closed"] += closed_time
        self.timings["heur"] += heur_time
        self.timings["add"] += add_time
        self.timings["itr"] += itr_time

        self.step_num += 1

        return True

    def has_found_goal(self) -> list[bool]:
        """Checks if the goal has been found for each instance.

        Returns:
            List indicating if the goal has been found for each instance.
        """
        return [len(self.get_goal_nodes(idx)) > 0 for idx in range(len(self.instances))]

    def get_goal_nodes(self, inst_idx: int) -> list[Node]:
        """Gets the goal nodes for a given instance.

        Args:
            inst_idx: The index of the instance.

        Returns:
            The goal nodes.
        """
        return self.instances[inst_idx].goal_nodes

    def get_goal_node_smallest_path_cost(self, inst_idx: int) -> Node:
        """Gets the goal node with the smallest path cost for a given instance.

        Args:
            inst_idx: The index of the instance.

        Returns:
            The goal node with the smallest path cost.
        """
        goal_nodes: list[Node] = self.get_goal_nodes(inst_idx)
        if len(goal_nodes) == 0:
            # No goal nodes found for this instance. Fall back to the last expanded
            # node if available, otherwise return the root node to keep the
            # downstream callers safe.
            return self.last_node if self.last_node is not None else self.instances[inst_idx].root_node

        path_costs: list[float] = [node.path_cost for node in goal_nodes]
        return goal_nodes[np.argmin(path_costs)]

    def get_num_nodes_generated(self, inst_idx: int) -> int:
        """Gets the number of nodes generated for a given instance.

        Args:
            inst_idx: The index of the instance.

        Returns:
            The number of nodes generated.
        """
        return self.instances[inst_idx].num_nodes_generated


@dataclass(slots=True)
class GBFSResult:
    """Store the result of a batched GBFS rollout."""

    nodes: list[Node]
    solved: NDArray[np.bool_]
    num_steps: NDArray[np.intp]
    num_nodes_generated: int
    timings: dict[str, float]


def _sokoban_goal_states(goal_state: SokobanState) -> list[ABCState]:
    """Build goal states for every valid Sokoban agent position.

    Args:
        goal_state: Goal state whose box and wall positions are fixed.

    Returns:
        Goal states with the agent placed on every unblocked position.
    """
    blocked: NDArray[np.bool_] = np.logical_or(goal_state.walls, goal_state.boxes)
    blank_positions: NDArray[np.intp] = np.argwhere(~blocked).astype(np.intp)
    goal_states: list[ABCState] = [
        SokobanState(blank_position, goal_state.boxes, goal_state.walls) for blank_position in blank_positions
    ]
    return goal_states


def _node_path_cost(node: Node) -> float:
    """Return a node's accumulated path cost.

    Args:
        node: Search node to inspect.

    Returns:
        Accumulated path cost for the node.
    """
    return node.path_cost


@torch.inference_mode()
def run_gbfs(
    states: NDArray[float32],
    state_goals: NDArray[float32],
    heuristic_fn: HeuristicFn,
    model_fn: ModelFn,
    is_solved_fn: IsSolvedFn,
    max_steps: int,
) -> GBFSResult:
    """Run batched Greedy Best-First Search using action-value estimates.

    Args:
        states: Encoded start states.
        state_goals: Encoded goal states aligned with ``states``.
        heuristic_fn: Callable returning one action-value vector per state.
        model_fn: Callable applying one action to each state.
        is_solved_fn: Callable returning a solved mask for state pairs.
        max_steps: Maximum number of actions applied to each state.

    Returns:
        Final nodes, solved flags, per-state step counts, generated-node count,
        and timing totals.
    """
    current_states: NDArray[float32] = np.ascontiguousarray(states, dtype=float32).copy()
    nodes: list[Node] = [Node(np.array(state, dtype=float32, copy=True), 0.0, None, None) for state in current_states]
    num_steps: NDArray[np.intp] = np.zeros(current_states.shape[0], dtype=np.intp)
    solved: NDArray[np.bool_] = np.asarray(is_solved_fn(current_states, state_goals), dtype=np.bool_)
    timings: dict[str, float] = {"pop": 0.0, "closed": 0.0, "heur": 0.0, "add": 0.0, "itr": 0.0}
    num_nodes_generated: int = len(nodes)

    for _ in range(max_steps):
        active_indices: NDArray[np.intp] = np.flatnonzero(~solved).astype(np.intp, copy=False)
        if active_indices.size == 0:
            break

        iteration_start: float = time.perf_counter()
        active_states: NDArray[float32] = current_states[active_indices]
        active_goals: NDArray[float32] = state_goals[active_indices]

        heuristic_start: float = time.perf_counter()
        q_values: NDArray[float32] = heuristic_fn(active_states, active_goals)
        actions: NDArray[np.intp] = np.argmin(q_values, axis=1).astype(np.intp, copy=False)
        timings["heur"] += time.perf_counter() - heuristic_start

        transition_start: float = time.perf_counter()
        next_states: NDArray[float32] = np.ascontiguousarray(
            model_fn(active_states, actions.astype(float32, copy=False)).round(), dtype=float32
        )
        timings["add"] += time.perf_counter() - transition_start

        for active_index, next_state, action in zip(
            active_indices.tolist(), next_states, actions.tolist(), strict=True
        ):
            parent: Node = nodes[active_index]
            nodes[active_index] = Node(next_state, parent.path_cost + 1.0, int(action), parent)

        current_states[active_indices] = next_states
        num_steps[active_indices] += 1
        num_nodes_generated += int(active_indices.size)
        solved = np.asarray(is_solved_fn(current_states, state_goals), dtype=np.bool_)
        timings["itr"] += time.perf_counter() - iteration_start

    return GBFSResult(nodes, solved, num_steps, num_nodes_generated, timings)


class H5PairStreamer(Iterator[PairItem]):
    """Streams (index, start, goal, start_variant, goal_variant) from an HDF5 pairs file.

    The iteration order is base->base first for each index, then all selected variant combinations.
    """

    __slots__: tuple[str, ...] = (
        "curr_idx",
        "goal_base_ds",
        "goal_states_ds",
        "goal_variant_ds",
        "h5f",
        "i_g",
        "i_s",
        "sel_g",
        "sel_s",
        "start_base_ds",
        "start_idx",
        "start_states_ds",
        "start_variant_ds",
        "total_pairs",
    )

    def __init__(
        self,
        h5f: h5py.File,
        start_vg: h5py.Group | None,
        goal_vg: h5py.Group | None,
        start_base_ds: h5py.Dataset,
        goal_base_ds: h5py.Dataset,
        sel_s: list[str],
        sel_g: list[str],
        start_idx: int,
    ) -> None:
        self.h5f: h5py.File = h5f
        self.start_base_ds: h5py.Dataset = start_base_ds
        self.goal_base_ds: h5py.Dataset = goal_base_ds
        self.start_variant_ds: dict[str, h5py.Dataset] = self._resolve_variant_datasets(
            variation_group=start_vg, selected_variants=sel_s, side="start"
        )
        self.goal_variant_ds: dict[str, h5py.Dataset] = self._resolve_variant_datasets(
            variation_group=goal_vg, selected_variants=sel_g, side="goal"
        )
        # ABCState datasets
        start_obj = h5f.get("pairs/start/states_pickle")
        self.start_states_ds: h5py.Dataset | None = start_obj if isinstance(start_obj, h5py.Dataset) else None
        goal_obj = h5f.get("pairs/goal/states_pickle")
        self.goal_states_ds: h5py.Dataset | None = goal_obj if isinstance(goal_obj, h5py.Dataset) else None
        self.sel_s: list[str] = sel_s
        self.sel_g: list[str] = sel_g
        self.total_pairs: int = start_base_ds.shape[0]
        self.curr_idx: int = start_idx
        self.i_s: int = 0
        self.i_g: int = 0
        self.start_idx: int = start_idx

    @staticmethod
    def _resolve_variant_datasets(
        *, variation_group: h5py.Group | None, selected_variants: list[str], side: str
    ) -> dict[str, h5py.Dataset]:
        out: dict[str, h5py.Dataset] = {}
        for variant_name in selected_variants:
            if variant_name == "base":
                continue
            if variation_group is None:
                raise KeyError(f"Missing variations group for pairs/{side}, required variant '{variant_name}'.")
            var_obj = variation_group.get(variant_name)
            if not isinstance(var_obj, h5py.Group):
                raise KeyError(f"Variant '{variant_name}' not found under pairs/{side}/variations.")
            imgs_obj = var_obj.get("images")
            if not isinstance(imgs_obj, h5py.Dataset):
                raise KeyError(f"Missing images dataset under pairs/{side}/variations/{variant_name}.")
            out[variant_name] = imgs_obj
        return out

    def __iter__(self) -> Self:
        """Return self as iterator."""
        return self

    def __next__(self) -> PairItem:
        """Yield next (idx, start, goal, start_variant, goal_variant, base_start_state, base_goal_state)."""
        if self.curr_idx >= self.total_pairs:
            raise StopIteration

        s_name: str = self.sel_s[self.i_s]
        g_name: str = self.sel_g[self.i_g]
        idx_ret: int = self.curr_idx

        # Select the base or variant datasets requested by the caller.
        s_ds: h5py.Dataset
        g_ds: h5py.Dataset
        s_ds = self.start_base_ds if s_name == "base" else self.start_variant_ds[s_name]
        g_ds = self.goal_base_ds if g_name == "base" else self.goal_variant_ds[g_name]

        s_np: NDArray[float32] = np.array(s_ds[self.curr_idx : self.curr_idx + 1], dtype=float32, copy=False)
        g_np: NDArray[float32] = np.array(g_ds[self.curr_idx : self.curr_idx + 1], dtype=float32, copy=False)

        # Load base ABCStates (pickled) for environment validation
        base_start_state: ABCState | None = None
        base_goal_state: ABCState | None = None
        if self.start_states_ds is not None:
            raw: NDArray[uint8] = self.start_states_ds[self.curr_idx]
            buf: bytes = raw.tobytes() if hasattr(raw, "tobytes") else bytes(raw)
            base_start_state = _deserialize_state(buf, side="start", pair_index=self.curr_idx)
        if self.goal_states_ds is not None:
            rawg: NDArray[uint8] = self.goal_states_ds[self.curr_idx]
            bufg: bytes = rawg.tobytes() if hasattr(rawg, "tobytes") else bytes(rawg)
            base_goal_state = _deserialize_state(bufg, side="goal", pair_index=self.curr_idx)

        # Advance variant counters
        self.i_g += 1
        if self.i_g >= len(self.sel_g):
            self.i_g = 0
            self.i_s += 1
            if self.i_s >= len(self.sel_s):
                self.i_s = 0
                self.curr_idx += 1

        return idx_ret, s_np, g_np, s_name, g_name, base_start_state, base_goal_state

    def close(self) -> None:
        """Close the underlying HDF5 file if open."""
        with contextlib.suppress(Exception):
            self.h5f.close()


class DirPairStreamer(Iterator[PairItem]):
    """Streams pairs from folders one index at a time."""

    __slots__: tuple[str, ...] = ("curr_idx", "goal_nchw", "goal_paths", "start_idx", "start_paths", "total")

    def __init__(
        self, start_paths: list[str], goal_paths: list[str] | None, goal_img_path: str | None, start_idx: int
    ) -> None:
        self.start_paths: list[str] = start_paths
        self.goal_paths: list[str] | None = goal_paths
        self.curr_idx: int = start_idx
        self.total: int = len(start_paths)
        self.start_idx: int = start_idx
        self.goal_nchw: NDArray[float32]
        if goal_paths is None:
            if goal_img_path is None:
                raise ValueError("goal_img_path must be provided when goal_paths is None")
            goal_img: NDArray[float32] = ImageHandler.load_image(goal_img_path)
            self.goal_nchw = np.expand_dims(goal_img.transpose(2, 0, 1).astype(float32), axis=0)
        else:
            self.goal_nchw = np.empty((0,), dtype=float32)

    def __iter__(self) -> Self:
        """Return self as iterator."""
        return self

    def __next__(self) -> PairItem:
        """Yield next (idx, start, goal, start_variant, goal_variant, base_start_state, base_goal_state)."""
        if self.curr_idx >= self.total:
            raise StopIteration
        idx: int = self.curr_idx
        s_np: NDArray[float32] = ImageHandler.load_images_batch([self.start_paths[idx]])
        g_np: NDArray[float32] = (
            ImageHandler.load_images_batch([self.goal_paths[idx]]) if self.goal_paths is not None else self.goal_nchw
        )

        self.curr_idx += 1
        return idx, s_np, g_np, "base", "base", None, None

    @staticmethod
    def close() -> None:
        """No-op close for uniform interface."""
        return


class SinglePairStreamer(Iterator[PairItem]):
    """Streams a single (state, goal) pair once when start_idx==0."""

    __slots__: tuple[str, ...] = ("emitted", "goal", "start_idx", "state")

    def __init__(self, state: NDArray[float32], goal: NDArray[float32], start_idx: int) -> None:
        self.state: NDArray[float32] = state
        self.goal: NDArray[float32] = goal
        self.start_idx: int = start_idx
        self.emitted: bool = start_idx > 0

    def __iter__(self) -> Self:
        """Return self as iterator."""
        return self

    def __next__(self) -> PairItem:
        """Yield the single pair once when not emitted yet."""
        if self.emitted:
            raise StopIteration
        self.emitted = True
        return 0, self.state, self.goal, "base", "base", None, None

    @staticmethod
    def close() -> None:
        """No-op close for uniform interface."""
        return


def order_with_base_first(names: list[str]) -> list[str]:
    """Move 'base' to front if present, preserving others.

    Args:
        names: Variant names.

    Returns:
        Names with 'base' first when present.
    """
    if not names:
        return names
    names_sorted: list[str] = list(names)
    if "base" in names_sorted:
        idx: int = names_sorted.index("base")
        if idx != 0:
            base_name: str = names_sorted.pop(idx)
            names_sorted.insert(0, base_name)
    return names_sorted


def resolve_variant_selection(
    available: list[str], include: list[str] | None, exclude: list[str] | None, *, label: str
) -> list[str]:
    """Resolve include/exclude variant lists given available names.

    - When include is None: include all available.
    - Apply excludes if provided.
    - Validate names exist.
    """
    # Normalize available set
    avail_set: set[str] = set(available)
    selected: list[str]
    if include is None or len(include) == 0:
        selected = list(avail_set)
    else:
        missing: list[str] = [name for name in include if name not in avail_set]
        if missing:
            raise ValueError(f"Unknown {label} variant(s): {missing}. Available: {sorted(available)}")
        selected = list(include)

    if exclude:
        missing_ex: list[str] = [name for name in exclude if name not in avail_set]
        if missing_ex:
            raise ValueError(f"Unknown {label} variant(s) in exclude: {missing_ex}. Available: {sorted(available)}")
        selected = [name for name in selected if name not in set(exclude)]

    if len(selected) == 0:
        raise ValueError(f"After include/exclude filtering, no {label} variants remain")

    selected.sort()
    return selected


def _select_encoder_for_variant(
    variant: str, encoder_mode: str, align_encoder: nn.Module | None, base_encoder: nn.Module | None
) -> nn.Module:
    """Return the encoder to use for the given variant under the configured mode."""
    if encoder_mode == "align_only":
        if align_encoder is None:
            raise ValueError("Align encoder required when encoder_mode='align_only'")
        return align_encoder
    if encoder_mode == "encoder_only":
        if base_encoder is None:
            raise ValueError("Base encoder required when encoder_mode='encoder_only'")
        return base_encoder
    if encoder_mode == "variant_aware":
        if base_encoder is not None and variant == "base":
            return base_encoder
        if align_encoder is None:
            raise ValueError("Align encoder required when encoder_mode='variant_aware' and variant!='base'")
        return align_encoder
    raise ValueError(f"Unknown encoder_mode '{encoder_mode}'")


def encode_with_module(
    array: NDArray[float32], encoder: nn.Module, device: torch.device, *, pin_if_possible: bool
) -> NDArray[float32]:
    """Encode `array` with `encoder`, returning a rounded float32 numpy array."""
    tensor: torch.Tensor = torch.from_numpy(array)
    if pin_if_possible:
        with contextlib.suppress(Exception):
            tensor = tensor.pin_memory()
    encoded: torch.Tensor = encoder(tensor.to(device, non_blocking=pin_if_possible).float())
    rounded: NDArray[np.float32] = torch.round(encoded).detach().to("cpu").numpy()
    return np.ascontiguousarray(rounded, dtype=float32)


@torch.inference_mode()
def run_search(
    env: ABCEnvironment[ABCState], cfg: SearchStageConfig, *, algorithm: SearchMode = "qstar"
) -> QStarResults:
    """Run one of the supported batched search algorithms.

    Args:
        env: The environment.
        cfg: The validated search-stage configuration.
        algorithm: Selects Q*, GBFS, or UCS.

    Returns:
        Aggregated per-pair search results.
    """
    algorithm_label: str
    max_itrs_cfg: int
    use_heuristic: bool
    if algorithm == "qstar":
        algorithm_label = "Q*"
        max_itrs_cfg = cfg.search.max_search_itrs
        use_heuristic = True
    elif algorithm == "gbfs":
        algorithm_label = "GBFS"
        max_itrs_cfg = cfg.search.gbfs_search_itrs
        use_heuristic = True
    else:
        algorithm_label = "UCS"
        max_itrs_cfg = cfg.search.max_search_itrs
        use_heuristic = False

    logger.info(f"Starting {algorithm_label} search")

    # Select an available device in CUDA, XPU, MPS, CPU order.
    device: torch.device
    on_gpu: bool
    device, _, on_gpu = get_device()
    apply_cuda_tf32_settings(cfg.search.allow_tf32, cfg.search.float32_matmul_precision, logger)

    model_cfg: ModelConfig = cfg.model
    start_idx: int = cfg.search.start_idx
    logger.info(f"device: {device}")

    # Configure compiler caches and optional portable artifact prewarm
    # Setup results directory and output path
    pathlib.Path(cfg.search.results_dir).mkdir(exist_ok=True, parents=True)
    results_json_file: str = f"{cfg.search.results_dir}/results.json"

    # get data (support HDF5 pairs datasets, folder mode, and single-image pair)
    pairs_file: str | None = cfg.search.pairs_file
    state_dir: str | None = cfg.search.state_dir
    goal_state_dir: str | None = cfg.search.goal_state_dir
    images_glob: str = cfg.search.images_glob
    # Note: input sources may be image files/datasets or HDF5 pairs. Validation
    # will attempt environment-based checks and fall back to latent checks as
    # needed.

    # Build a streaming pair iterator (HDF5 pairs, folder mode, or single image)
    total_pairs: int
    # pair_iter: Iterator[PairItem]
    streamer: H5PairStreamer | DirPairStreamer | SinglePairStreamer
    is_pairs_input: bool = False
    sel_s: list[str] = []
    sel_g: list[str] = []
    if pairs_file:
        logger.info(f"Indexing pairs dataset for streaming: {pairs_file}")
        h5f: h5py.File = h5py.File(pairs_file, "r", rdcc_nbytes=256 * 1024**2, rdcc_nslots=100_003, rdcc_w0=0.25)
        try:
            total_pairs, start_vg, goal_vg, start_base_ds, goal_base_ds, start_avail, goal_avail = (
                inspect_h5_pair_inputs(h5f)
            )
        except Exception:
            h5f.close()
            raise

        inc_s: list[str] | None = cfg.search.pairs_start_include
        exc_s: list[str] | None = cfg.search.pairs_start_exclude
        legacy_s: str | None = cfg.search.pairs_start_variant
        if inc_s is None and legacy_s:
            inc_s = [legacy_s]
        inc_g: list[str] | None = cfg.search.pairs_goal_include
        exc_g: list[str] | None = cfg.search.pairs_goal_exclude
        legacy_g: str | None = cfg.search.pairs_goal_variant
        if inc_g is None and legacy_g:
            inc_g = [legacy_g]

        sel_s = order_with_base_first(resolve_variant_selection(sorted(start_avail), inc_s, exc_s, label="start"))
        sel_g = order_with_base_first(resolve_variant_selection(sorted(goal_avail), inc_g, exc_g, label="goal"))

        logger.info(f"Selected start variants={sel_s}, goal variants={sel_g} -> streaming {total_pairs} indices")
        streamer = H5PairStreamer(h5f, start_vg, goal_vg, start_base_ds, goal_base_ds, sel_s, sel_g, start_idx)
        is_pairs_input = True

    elif state_dir is not None:
        # Folder mode
        start_paths: list[str] = ImageHandler.list_images_in_dir(state_dir, images_glob)
        if len(start_paths) == 0:
            raise ValueError(f"No images found in state_dir='{state_dir}' with pattern '{images_glob}'")
        goal_paths: list[str] | None = None
        if goal_state_dir is not None:
            goal_paths = ImageHandler.list_images_in_dir(goal_state_dir, images_glob)
            if len(goal_paths) == 0:
                raise ValueError(f"No images found in goal_state_dir='{goal_state_dir}' with pattern '{images_glob}'")
            if len(goal_paths) != len(start_paths):
                raise ValueError(
                    f"Start/goal directory counts differ: {len(start_paths)} vs {len(goal_paths)}."
                    " Provide same number of files when using paired folder mode."
                )
        streamer = DirPairStreamer(start_paths, goal_paths, cfg.search.goal_state_path, start_idx)
        total_pairs = len(start_paths)

    else:
        # Single-image mode: load two images and convert to (N, C, H, W)
        assert cfg.search.state_path is not None, "cfg.search.state_path must be set for single-image mode"
        assert cfg.search.goal_state_path is not None, "cfg.search.goal_state_path must be set for single-image mode"
        state_img: NDArray[float32] = ImageHandler.load_image(cfg.search.state_path)
        goal_img: NDArray[float32] = ImageHandler.load_image(cfg.search.goal_state_path)
        single_state: NDArray[float32] = np.expand_dims(state_img.transpose(2, 0, 1).astype(float32), axis=0)
        single_goal: NDArray[float32] = np.expand_dims(goal_img.transpose(2, 0, 1).astype(float32), axis=0)
        streamer = SinglePairStreamer(single_state, single_goal, start_idx)
        total_pairs = 1
        # streamer yields (idx, state, goal, start_variant, goal_variant, base_start_state|None, base_goal_state|None)

    # UCS does not load or evaluate the learned action-value model.
    heuristic_fn: CallableModelType | None = None
    if use_heuristic:
        dqn: nn.Module = env.get_dqn(model_cfg)
        dqn = load_model_ptu(
            dqn,
            device=device,
            pretrained_path=cfg.search.heuristic_model_path,
            strip_compiled_prefixes=True,
            strip_ddp_prefixes=True,
            strip_dataparallel_prefixes=True,
            freeze=True,
        )
        heuristic_fn = HeuristicFnWrapper(
            nnet=dqn, device=device, clip_zero=True, batch_size=cfg.search.nnet_batch_size
        )

    # Environment transition model.
    model_fn: CallableModelType
    env_model: nn.Module = env.get_env_model_disc(model_cfg)
    env_model = load_model_ptu(
        env_model,
        device=device,
        pretrained_path=cfg.search.env_model_path,
        strip_compiled_prefixes=True,
        strip_ddp_prefixes=True,
        strip_dataparallel_prefixes=True,
        freeze=True,
    )
    model_fn = ModelFnWrapper(nnet=env_model, device=device, batch_size=cfg.search.nnet_batch_size)

    encoder_mode: str = cfg.search.encoder_mode
    align_encoder: nn.Module | None = None
    if encoder_mode in {"align_only", "variant_aware"}:
        # Load alignment model (A) used for variant inputs
        alignment_model: nn.Module = env.get_alignment_model(model_cfg)
        align_encoder = load_model_ptu(
            model=alignment_model,
            device=device,
            pretrained_path=cfg.search.alignment_model_path,
            strip_compiled_prefixes=True,
            strip_ddp_prefixes=True,
            strip_dataparallel_prefixes=True,
            freeze=True,
        )
        align_encoder.eval()

    # Optionally load the base encoder only in modes that can use it
    # - encoder_only: required and exclusively used
    # - variant_aware: used for 'base' variant when available
    # - align_only: never load/use base encoder
    base_encoder: nn.Module | None = None
    base_encoder_path: str | None = cfg.search.encoder_model_path
    if encoder_mode in {"encoder_only", "variant_aware"} and base_encoder_path:
        try:
            base_encoder_model: nn.Module = env.get_encoder_disc(model_cfg)
            # Use load_model_ptu which handles prefixes and device mapping consistently
            base_encoder = load_model_ptu(
                model=base_encoder_model,
                device=device,
                pretrained_path=base_encoder_path,
                strip_compiled_prefixes=True,
                strip_ddp_prefixes=True,
                strip_dataparallel_prefixes=True,
                freeze=True,
            )
            base_encoder.eval()
            logger.info(f"Loaded base encoder for clean/base variants: {base_encoder.__class__.__name__}")
        except Exception as e:
            logger.info(f"Failed to load base encoder at {base_encoder_path}: {e!s}")
            base_encoder = None
    if encoder_mode == "encoder_only" and base_encoder is None:
        raise ValueError(
            "encoder_mode=encoder_only requires search.encoder_model_path to point to a loadable checkpoint"
        )

    is_solved_fn: IsSolvedFn = IsSolvedTolerance(cfg.search.per_eq_tol)

    # TODO: Actions are assumed to be of a fixed size
    num_actions_max: int = env.num_actions_max

    # Visualization config and context
    vis_recon_mode: str = cfg.search.vis_recon_mode
    vis_env_mode: str = cfg.search.vis_env_mode
    vis_combined_mode: str = cfg.search.vis_combined_mode
    vis_on_unsolved: bool = cfg.search.vis_on_unsolved
    validate_on_unsolved: bool = cfg.search.validate_on_unsolved
    log_moves_on_unsolved: bool = cfg.search.log_moves_on_unsolved
    vis_fps: int = cfg.search.save_vis_fps

    image_ctx: QStarImageContext = ImageHandler.build_for_qstar(
        results_dir=cfg.search.results_dir,
        vis_recon_mode=_as_vis_mode(vis_recon_mode),
        vis_env_mode=_as_vis_mode(vis_env_mode),
        vis_combined_mode=_as_vis_mode(vis_combined_mode),
        vis_on_unsolved=vis_on_unsolved,
        include_start_goal_header=cfg.search.vis_combined_include_start_goal,
        header_start_title=cfg.search.vis_start_title,
        header_goal_title=cfg.search.vis_goal_title,
        env_row_title=cfg.search.vis_env_row_title,
        recon_row_title=cfg.search.vis_recon_row_title,
        fps=vis_fps,
        env=env,
        decoder=None,  # will be auto-loaded if needed when possible
        device=device,
        model_cfg=model_cfg,
        decoder_model_path=cfg.search.decoder_model_path,
        indiv_size_mode=_as_individual_size_mode(cfg.search.vis_individual_size_mode),
        indiv_target_h=cfg.search.vis_individual_height,
        indiv_target_w=cfg.search.vis_individual_width,
        combined_row_h=cfg.search.vis_combined_row_height,
        combined_row_w=cfg.search.vis_combined_row_width,
    )

    # Initialize per-run logs.
    logs: list[QStarLogEntry] = []
    summary_tracker: QStarSummaryTracker = QStarSummaryTracker()
    running_summary_enabled: bool = cfg.search.qstar_show_running_summary

    # The schema defines cfg.search.start_idx as int, so use it directly when present.
    start_idx = cfg.search.start_idx

    logger.info(f"Starting at idx {start_idx}")
    logger.info(f"Total number of test states: {total_pairs}")

    # Cache for clean ABCState objects per index (populated from HDF5 'states_pickle' when available)
    base_states_cache: dict[int, tuple[ABCState, ABCState]] = {}

    # Use a visual progress bar when iterating over pairs
    # When using HDF5 pairs the streamer may emit multiple variants per index
    # (start/goal variants). Compute the visible total as total_pairs * (#start variants * #goal variants)
    # and also set the initial completed count when start_idx > 0 so the bar doesn't jump past the total.
    variants_mult: int = 1
    if is_pairs_input:
        variants_mult = len(sel_s) * len(sel_g)

    progress_total: int = total_pairs * variants_mult

    start_completed: int = start_idx * variants_mult

    # index count for display (may be '?' when streaming a generator with unknown length)
    indices_field: int | str = total_pairs

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold yellow]{algorithm_label} search {{task.fields[phase]}}", justify="left"),
        BarColumn(bar_width=40),
        TextColumn(
            "{task.completed}/{task.total} [{task.fields[indices]} idx x {task.fields[variants_mult]} variants]"
        ),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Initialize progress from a configured start index.
        # Pass dynamic fields for the UI: indices and variants_mult
        task_id: TaskID = progress.add_task(
            "qstar",
            total=progress_total,
            completed=start_completed,
            phase="run",
            indices=indices_field,
            variants_mult=variants_mult,
        )

        pair_idx: int
        path_cost: float
        soln: list[int]
        num_nodes_gen_idx: int
        path: list[NDArray[float32]]
        start_var_name: str
        goal_var_name: str
        base_start_state: ABCState | None
        base_goal_state: ABCState | None
        for seq_idx, item in enumerate(streamer):
            # Unpack streaming tuple
            (
                pair_idx,
                state_curr,
                state_goal_curr,
                start_var_name,
                goal_var_name,
                base_start_state,
                base_goal_state,
            ) = item
            # Cache clean ABCState when provided by streamer (for env validation/visuals)
            if base_start_state is not None and base_goal_state is not None:
                base_states_cache[pair_idx] = (base_start_state, base_goal_state)
            if logger.isEnabledFor(DEBUG):
                logger.debug(
                    f"Idx {pair_idx} StartVar={start_var_name} GoalVar={goal_var_name} "
                    f"State Shape: {state_curr.shape}, Goal Shape: {state_goal_curr.shape}"
                )

            start_time: float = time.time()
            num_itrs: int = 0
            # res is used to check whether to continue search or not
            res: bool = True
            # Minimize CPU<->GPU copies: use pinned memory for CUDA and non_blocking transfer
            can_pin: bool = device.type == "cuda"
            enc_start: nn.Module = _select_encoder_for_variant(
                start_var_name, encoder_mode, align_encoder, base_encoder
            )
            state_enc_np: NDArray[float32] = encode_with_module(state_curr, enc_start, device, pin_if_possible=can_pin)
            if logger.isEnabledFor(DEBUG):
                logger.debug(
                    "Start encoder selected: "
                    f"{enc_start.__class__.__name__ if hasattr(enc_start, '__class__') else str(type(enc_start))} "
                    f"for variant '{start_var_name}'"
                )

            enc_goal: nn.Module = _select_encoder_for_variant(goal_var_name, encoder_mode, align_encoder, base_encoder)
            state_goal_enc_np: NDArray[float32] = encode_with_module(
                state_goal_curr, enc_goal, device, pin_if_possible=can_pin
            )
            if logger.isEnabledFor(DEBUG):
                logger.debug(
                    "Goal encoder selected: "
                    f"{enc_goal.__class__.__name__ if hasattr(enc_goal, '__class__') else str(type(enc_goal))} "
                    f"for variant '{goal_var_name}'"
                )

            # If Sokoban, treat goal as boxes/walls fixed and agent allowed on any blank cell
            is_sokoban: bool = env.get_env_name().lower() == "sokoban"
            have_base_states: bool = base_start_state is not None and base_goal_state is not None

            goal_node_found: Node
            goal_flags: list[bool]
            solved_by_search: bool

            if algorithm == "gbfs":
                if heuristic_fn is None:
                    raise RuntimeError("GBFS requires a heuristic model")

                if is_sokoban and have_base_states:
                    if not isinstance(base_goal_state, SokobanState):
                        raise TypeError("Expected SokobanState for goal-side base state in Sokoban search")
                    goal_states_all: list[ABCState] = _sokoban_goal_states(base_goal_state)
                    goals_all_np: NDArray[float32] = env.state_to_real(goal_states_all)
                    goals_enc_np: NDArray[float32] = encode_with_module(
                        goals_all_np, enc_goal, device, pin_if_possible=can_pin
                    )
                    starts_enc_np: NDArray[float32] = np.repeat(state_enc_np, goals_enc_np.shape[0], axis=0)
                    greedy_result: GBFSResult = run_gbfs(
                        starts_enc_np, goals_enc_np, heuristic_fn, model_fn, is_solved_fn, max_itrs_cfg
                    )
                else:
                    greedy_result = run_gbfs(
                        state_enc_np, state_goal_enc_np, heuristic_fn, model_fn, is_solved_fn, max_itrs_cfg
                    )

                goal_flags = [bool(flag) for flag in greedy_result.solved.tolist()]
                solved_by_search = any(goal_flags)
                selected_indices: list[int] = [idx for idx, flag in enumerate(goal_flags) if flag]
                if not selected_indices:
                    selected_indices = list(range(len(greedy_result.nodes)))
                selected_nodes: list[Node] = [greedy_result.nodes[idx] for idx in selected_indices]
                goal_node_found = min(selected_nodes, key=_node_path_cost)
                num_nodes_gen_idx = greedy_result.num_nodes_generated
                algorithm_timings = greedy_result.timings
                num_itrs = int(np.max(greedy_result.num_steps)) if greedy_result.num_steps.size > 0 else 0
            else:
                qstar: QStarImag
                qstar_heuristic: CallableModelType | None = heuristic_fn if algorithm == "qstar" else None
                if is_sokoban and have_base_states:
                    if not isinstance(base_goal_state, SokobanState):
                        raise TypeError("Expected SokobanState for goal-side base state in Sokoban search")
                    goal_states_all = _sokoban_goal_states(base_goal_state)
                    goals_all_np = env.state_to_real(goal_states_all)
                    goals_enc_np = encode_with_module(goals_all_np, enc_goal, device, pin_if_possible=can_pin)
                    starts_enc_np = np.repeat(state_enc_np, goals_enc_np.shape[0], axis=0)
                    qstar = QStarImag(
                        starts_enc_np,
                        goals_enc_np,
                        qstar_heuristic,
                        weights=[cfg.search.qstar_weight] * starts_enc_np.shape[0],
                        num_actions_max=num_actions_max,
                    )
                    while res and not any(qstar.has_found_goal()) and num_itrs < max_itrs_cfg:
                        res = qstar.step(
                            qstar_heuristic,
                            model_fn,
                            is_solved_fn,
                            cfg.search.qstar_batch_size,
                            verbose=cfg.search.verbose,
                        )
                        num_itrs += 1

                    goal_flags = qstar.has_found_goal()
                    solved_by_search = any(goal_flags)
                    if res and solved_by_search:
                        solved_idxs: NDArray[np.intp] = np.where(np.asarray(goal_flags, dtype=np.bool_))[0].astype(
                            np.intp
                        )
                        goal_node_found = qstar.get_goal_node_smallest_path_cost(int(solved_idxs[0]))
                        for idx_i in solved_idxs[1:]:
                            cand: Node = qstar.get_goal_node_smallest_path_cost(int(idx_i))
                            if cand.path_cost < goal_node_found.path_cost:
                                goal_node_found = cand
                    else:
                        goal_node_found = (
                            qstar.last_node if qstar.last_node is not None else qstar.instances[0].root_node
                        )
                else:
                    qstar = QStarImag(
                        state_enc_np,
                        state_goal_enc_np,
                        qstar_heuristic,
                        weights=[cfg.search.qstar_weight],
                        num_actions_max=num_actions_max,
                    )
                    while res and not min(qstar.has_found_goal()) and num_itrs < max_itrs_cfg:
                        res = qstar.step(
                            qstar_heuristic,
                            model_fn,
                            is_solved_fn,
                            cfg.search.qstar_batch_size,
                            verbose=cfg.search.verbose,
                        )
                        num_itrs += 1

                    goal_flags = qstar.has_found_goal()
                    solved_by_search = all(goal_flags)
                    if res and solved_by_search:
                        goal_node_found = qstar.get_goal_node_smallest_path_cost(0)
                    else:
                        goal_node_found = (
                            qstar.last_node if qstar.last_node is not None else qstar.instances[0].root_node
                        )
                num_nodes_gen_idx = sum(qstar.get_num_nodes_generated(i) for i in range(len(qstar.instances)))
                algorithm_timings = qstar.timings
                num_itrs = qstar.step_num

            path, soln, path_cost = get_path(goal_node_found)
            timing_str: str = ", ".join([
                f"{k}: {algorithm_timings.get(k, 0.0):.2f}" for k in ["pop", "closed", "heur", "add", "itr"]
            ])

            solve_time: float = time.time() - start_time

            # Determine if search found a goal (latent equality check)
            # solved_by_search already computed in both branches

            # Validation/visualization policy per dataset type:
            # HDF5 pairs validate against ABCState, subject to the unsolved-only setting.
            # Image files lack ABCState, so they use the latent result and reconstruction-only visuals.

            # Prefer per-item states and fall back to the cached states.
            maybe_states_direct: tuple[ABCState, ABCState] | None = (
                (base_start_state, base_goal_state)
                if (base_start_state is not None and base_goal_state is not None)
                else None
            )
            maybe_states_cache: tuple[ABCState, ABCState] | None = base_states_cache.get(pair_idx)
            maybe_states: tuple[ABCState, ABCState] | None = maybe_states_direct or maybe_states_cache

            solved: bool
            if maybe_states:
                start_state: ABCState
                goal_state: ABCState
                start_state, goal_state = maybe_states
                want_validate: bool = solved_by_search or validate_on_unsolved

                if want_validate:
                    try:
                        solved = is_valid_soln(env, start_state, goal_state, soln)
                    except Exception:
                        if solved_by_search:
                            logger.warning("Environment validation failed. Falling back to a latent-equality check")
                        # On validation failure, retain a solved latent result and otherwise return false.
                        solved = solved_by_search
                else:
                    solved = False
            else:
                # Without ABCState values, retain the latent result and render reconstructions only.
                solved = solved_by_search

            # Visualizations via unified handler
            # Policy: allow visuals if solved_by_search OR config requests unsolved visuals.
            solved_for_visuals: bool = solved_by_search or vis_on_unsolved
            ImageHandler.save_pair(
                image_ctx,
                pair_idx=pair_idx,
                solved=solved_for_visuals,
                path=path,
                moves=list(soln),
                start_variant=start_var_name,
                goal_variant=goal_var_name,
                state_curr=state_curr,
                state_goal_curr=state_goal_curr,
                base_states=maybe_states,
            )

            nodes_per_sec: float = (num_nodes_gen_idx / solve_time) if solve_time > 0 else float("inf")
            # print to screen
            # timing_str already constructed with filtered keys
            # Render a compact panel for this pair's result
            # Hide moves when unsolved-by-search and not requested to log
            soln_shown: list[int] = soln if (solved_by_search or log_moves_on_unsolved) else []
            # Determine category and env validation attempt/result for display and logs
            env_validation_attempted: bool = (maybe_states is not None) and (solved_by_search or validate_on_unsolved)
            solved_env: bool | None = None
            if env_validation_attempted:
                # Here, 'solved' reflects the env result as derived above when attempted
                solved_env = solved

            solve_category: SolveCategory
            if solved_by_search and (solved_env is True):
                solve_category = "both"
            elif solved_by_search:
                solve_category = "search_only"
            else:
                solve_category = "none"

            # Build three renderables: a single-column top row, a two-column middle table,
            # and a single-column bottom row. The bottom row (solution) is left as raw Text
            # so Rich will wrap it dynamically according to the Panel width.
            top: Align = Align.center(Text(f"Times - {timing_str}, num_itrs: {num_itrs}"))
            rule: Rule = Rule(style="dim")

            middle: Table = Table(show_header=False, box=None, padding=(0, 1))
            middle.add_column(justify="right", ratio=1)
            middle.add_column(justify="left", ratio=3)

            # Key/value middle rows (State is in panel title)
            middle.add_row("[bold]Search Solved[/bold]", "Yes" if solved_by_search else "No")
            middle.add_row(
                "[bold]Env Validated[/bold]", ("Yes" if solved_env else ("No" if solved_env is False else "N/A"))
            )
            middle.add_row("[bold]Category[/bold]", f"{solve_category}")
            middle.add_row("[bold]SolnCost[/bold]", f"{path_cost:.2f}")
            middle.add_row("[bold]#Moves[/bold]", f"{len(soln)}")
            middle.add_row("[bold]#Nodes Gen[/bold]", f"{num_nodes_gen_idx:,}")
            middle.add_row("[bold]Time[/bold]", f"{solve_time:.2f}s")
            middle.add_row("[bold]Nodes/Sec[/bold]", f"{nodes_per_sec:.2E}")

            # Bottom: solution as Text so Panel will wrap it to its width
            soln_text: Text = Text(", ".join(str(m) for m in soln_shown)) if len(soln_shown) > 0 else Text("N/A")

            bottom_title: Align = Align.left("[bold]Solution:[/bold] ")
            bottom: Align = Align.left(soln_text)

            content: Align = Align.center(Group(top, rule, middle, rule, bottom_title, bottom))
            # Place the state/variant info in the panel title for emphasis and compactness
            panel_title: str = (
                f"[bold]Variant Index[/bold]={seq_idx}, "
                f"[bold]State Index[/bold]={pair_idx} "
                f"([bold]Start State[/bold]: {start_var_name} → [bold]Goal State[/bold]: {goal_var_name})"
            )
            logger.info(
                Panel(
                    content, title=panel_title, title_align="left", border_style="dim green", padding=(1, 2), width=120
                )
            )

            logs.append({
                "index": pair_idx,
                "sequence_index": seq_idx,
                "input_type": ("hdf5" if is_pairs_input else "images"),
                "start_variant": start_var_name,
                "goal_variant": goal_var_name,
                "solved_by_search": solved_by_search,
                "env_validation_attempted": env_validation_attempted,
                "solved_by_env": solved_env,
                "solve_category": solve_category,
                "path_cost": path_cost,
                "num_moves": len(soln),
                "logged_moves": list(soln_shown),
                "num_nodes_generated": num_nodes_gen_idx,
                "num_iterations": num_itrs,
                "elapsed_sec": solve_time,
                "nodes_per_sec": nodes_per_sec,
                "timings_sec": dict(algorithm_timings.items()),
                "visuals_emitted": solved_for_visuals,
            })
            if logs:
                summary_tracker.update(logs[-1])
                if running_summary_enabled:
                    logger.info(
                        build_summary_panel(
                            summary_tracker.stats(), title=f"{algorithm_label} Running Summary", border_style="cyan"
                        )
                    )

            # Release large intermediates before processing the next pair.
            del state_enc_np, state_goal_enc_np, path
            if on_gpu:
                torch.cuda.empty_cache()
            # advance progress bar
            progress.update(task_id, advance=1)

    # Close underlying resources (no-op for non-HDF5 streamers)
    with contextlib.suppress(Exception):
        streamer.close()

    stats: QStarSummaryStats = summary_tracker.stats()
    logger.info(build_summary_panel(stats, title=f"{algorithm_label} Search Summary", border_style="blue"))

    # Convert integer keys in per_index to strings for JSON serialization
    per_index_str_keys = {str(k): v for k, v in stats.per_index.items()}

    results_out: QStarResults = {
        "logs": logs,
        "summary": {
            "overall": {
                "entries_total": stats.entries_total,
                "solved_by_both": stats.solved_by_both,
                "solved_by_search_only": stats.solved_by_search_only,
                "unsolved_by_both": stats.unsolved_by_both,
                "search_success_rate": stats.search_success_rate,
                "avg_moves_solved_any": stats.avg_moves_solved_any,
                "avg_iterations_solved_any": stats.avg_iterations_solved_any,
                "avg_nodes_generated": stats.avg_nodes_generated,
                "avg_time_sec": stats.avg_time_sec,
                "avg_nodes_per_sec": stats.avg_nodes_per_sec,
            },
            # JSON object keys must be strings.
            "by_index": per_index_str_keys,
            "by_start_variant": stats.by_start_variant,
            "by_goal_variant": stats.by_goal_variant,
        },
    }

    # Save once at the end using orjson (fast, compact)
    pathlib.Path(results_json_file).write_bytes(
        orjson.dumps(results_out, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY)
    )
    return results_out
