from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray


DT = TypeVar("DT")
NDT = TypeVar("NDT", bound=np.generic)


def flatten(data: list[list[DT]]) -> tuple[list[DT], list[int]]:
    """Flattens a list of lists into a single list and returns the flattened list along with the split indices.

    Args:
        data: The list of lists to be flattened.

    Returns:
        A tuple containing the flattened `DT` items and the split indices.
    """
    num_each: list[int] = [len(x) for x in data]
    split_idxs: list[int] = list(np.cumsum(num_each)[:-1])

    data_flat: list[DT] = [item for sublist in data for item in sublist]

    return data_flat, split_idxs


def unflatten(data: list[DT], split_idxs: list[int]) -> list[list[DT]]:
    """Unflattens a list into a list of lists using the provided split indices.

    Args:
        data: The flattened list.
        split_idxs (list[int]): The indices to split the flattened list.

    Returns:
        The unflattened list of `DT` items.
    """
    data_split: list[list[DT]] = []

    start_idx: int = 0
    end_idx: int
    for end_idx in split_idxs:
        data_split.append(data[start_idx:end_idx])
        start_idx = end_idx

    data_split.append(data[start_idx:])

    return data_split


def split_and_stack(x: NDArray[NDT]) -> NDArray[NDT]:
    """Splits x of shape (n, 3, 32, 64) into two halves along width, then stacks them along the channel dimension.

    Will shape (n, 6, 32, 32).
    """
    # left half  (n, 3, 32, 32)
    left: NDArray[NDT] = x[:, :, :, :32]
    # right half (n, 3, 32, 32)
    right: NDArray[NDT] = x[:, :, :, 32:]
    # concatenate along channels -> (n, 6, 32, 32)
    return np.concatenate([left, right], axis=1)


def split_evenly(num_total: int, num_splits: int) -> list[int]:
    """Splits a total number into nearly equal parts.

    Args:
        num_total (int): The total number to be split.
        num_splits (int): The number of parts to split into.

    Returns:
        list[int]: A list containing the sizes of each part.
    """
    num_per: list[int] = [math.floor(num_total / num_splits) for _ in range(num_splits)]
    left_over: int = num_total % num_splits
    for idx in range(left_over):
        num_per[idx] += 1

    return num_per


# Time profiling


def record_time(times: dict[str, float], time_name: str, start_time: float, on_gpu: bool) -> None:
    """Records the elapsed time for a given time name and updates the times dictionary.

    Increments time if time_name is already in times. Synchronizes if on_gpu is true.

    Args:
        times (dict[str, float]): The dictionary to store the times.
        time_name (str): The name of the time entry.
        start_time (float): The start time to calculate the elapsed time.
        on_gpu (bool): Whether to synchronize with GPU before recording time.
    """
    if on_gpu:
        torch.cuda.synchronize()

    time_elapsed: float = time.time() - start_time
    if time_name in times:
        times[time_name] += time_elapsed
    else:
        times[time_name] = time_elapsed


def add_times(times: dict[str, float], times_to_add: dict[str, float]) -> None:
    """Adds times from one dictionary to another.

    Args:
        times (dict[str, float]): The dictionary to update with added times.
        times_to_add (dict[str, float]): The dictionary containing times to add.
    """
    for key, value in times_to_add.items():
        times[key] += value


def get_time_str(times: dict[str, float]) -> str:
    """Converts a dictionary of times into a formatted string.

    Args:
        times (dict[str, float]): The dictionary containing time entries.

    Returns:
        str: A formatted string representation of the times.
    """
    time_str_l: list[str] = []
    time_str_i: str
    for key, val in times.items():
        time_str_i = f"{key}: {val:.2f}"
        time_str_l.append(time_str_i)
    time_str: str = ", ".join(time_str_l)

    return time_str
