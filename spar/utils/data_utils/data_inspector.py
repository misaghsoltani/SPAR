"""Inspect and visualize data files produced by SPAR generation stages."""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import operator
import pathlib
import sys
import textwrap
import time
import tkinter as tk
import traceback
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar, overload
import warnings

import cv2
import h5py
from matplotlib import animation, pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import orjson
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .hdf5_common import (
    decode_attr_text,
    open_hdf5_for_read,
    read_variant_names_from_attrs,
    read_variant_type_from_attrs,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from typing import Literal, NoReturn, SupportsIndex, TypeAlias

    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from matplotlib.text import Text
    from matplotlib.typing import RcKeyType
    from numpy.typing import NDArray
    from typing_extensions import LiteralString, NotRequired

    ImageArray: TypeAlias = NDArray[np.uint8 | np.float32 | np.float64]
    # A single state's frames can be either a single ImageArray or a list of ImageArray (sequence of frames).
    StateFrames: TypeAlias = ImageArray | list[ImageArray]
    # A variation entry is a tuple of (variation_name, image_data)
    VariationEntry: TypeAlias = tuple[str, ImageArray]
    # Per-state variations is a list of VariationEntry. The full variations structure is a list (one per frame)
    PerStateVariations: TypeAlias = list[VariationEntry]

    # Dictionary-style variations where each key maps to either a single image or
    # a list of images (per-state or per-variation)
    VariationDict: TypeAlias = dict[str, ImageArray | list[ImageArray]]

    # Union type for variations passed around in functions
    VariationType: TypeAlias = list[PerStateVariations] | VariationDict

    # When variations are present the trajectory item is a tuple of (clean_states, variations_per_state)
    StateWithVariations: TypeAlias = tuple[StateFrames, VariationType]
    # Top-level alias for one element of `state_img_trajs` used across this module.
    StateImgTraj: TypeAlias = StateFrames | StateWithVariations

    # Types for displaying nested structures in tree view
    DisplayScalar: TypeAlias = int | float | str | bool
    ActionType: TypeAlias = int | float | np.floating | np.integer
    ActionTraj: TypeAlias = list[ActionType]
    DisplayData: TypeAlias = (
        DisplayScalar | ImageArray | list["DisplayData"] | tuple["DisplayData", ...] | dict[str, "DisplayData"]
    )
    DisplayInput: TypeAlias = (
        DisplayScalar
        | ImageArray
        | np.generic
        | pathlib.Path
        | list["DisplayInput"]
        | tuple["DisplayInput", ...]
        | dict[str, "DisplayInput"]
        | None
    )
    AdaptInput: TypeAlias = (
        DisplayInput
        | StateImgTraj
        | ActionTraj
        | VariationType
        | VariationEntry
        | PerStateVariations
        | list[StateImgTraj]
        | list[ActionTraj]
    )
    FormatValue: TypeAlias = DisplayScalar | pathlib.Path | datetime | None

    # Types for variation metadata in specs
    VariationShapesType: TypeAlias = dict[str, tuple[int, ...] | str]
    VariationDtypesType: TypeAlias = dict[str, str]
    MetadataEntry: TypeAlias = str | int | float | bool | list[str]
    H5Types: TypeAlias = (
        type[h5py.SoftLink | h5py.ExternalLink | h5py.HardLink]
        | h5py.Group
        | h5py.Dataset
        | h5py.Datatype
        | h5py.SoftLink
        | h5py.ExternalLink
        | h5py.HardLink
    )
    TextKey: TypeAlias = str


T = TypeVar("T")

console: Console = Console()
_HDF5_INSPECTOR_RDCC_NBYTES: int = 64 * 1024**2


def _variation_dict_item_key(item: tuple[str, ImageArray | list[ImageArray]]) -> str:
    """Return the name used to order dictionary-style variations."""
    return item[0]


def _variation_entry_key(item: VariationEntry) -> str:
    """Return the name used to order one variation entry."""
    return item[0]


plt.rcParams["figure.constrained_layout.use"] = True

# Suppress specific matplotlib warnings that are not actionable
warnings.filterwarnings("ignore", message=".*tight_layout.*")
warnings.filterwarnings("ignore", message=".*constrained_layout.*")
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg.*")


class LazyList(list[T], Generic[T]):
    """List that loads each element on first access.

    Args:
        size: Logical list length.
        loader: Function that loads an element by index.
        placeholder: Initial value stored in unloaded slots.
    """

    __slots__: tuple[str, ...] = ("_loaded", "_loader", "_placeholder", "_size")

    def __init__(self, size: int, loader: Callable[[int], T], placeholder: T) -> None:
        super().__init__([placeholder] * size)
        self._size: int = size
        self._loader: Callable[[int], T] = loader
        self._loaded: list[bool] = [False] * size
        self._placeholder: T = placeholder

    def __len__(self) -> int:
        """Return number of items (matches logical size without forcing loads)."""
        return self._size

    @overload
    def __getitem__(self, index: SupportsIndex, /) -> T: ...

    @overload
    def __getitem__(self, index: slice, /) -> list[T]: ...

    def __getitem__(self, index: SupportsIndex | slice, /) -> T | list[T]:
        """Return an item or slice, loading requested items on first access.

        Args:
            index: Integer-like index or slice to load.

        Returns:
            One loaded element for an index, or a list of loaded elements for a
            slice.
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(self._size)
            # Materialize only the elements selected by the slice.
            result: list[T] = [self[i] for i in range(start, stop, step)]
            return result

        idx: int = operator.index(index)
        normalized: int = idx if idx >= 0 else self._size + idx
        if normalized < 0 or normalized >= self._size:
            raise IndexError("list index out of range")
        if not self._loaded[normalized]:
            loaded: T = self._loader(normalized)
            super().__setitem__(normalized, loaded)
            self._loaded[normalized] = True
            return loaded
        return super().__getitem__(normalized)

    def __iter__(self) -> Iterator[T]:
        """Iterate items, lazily materializing each element when reached.

        Yields:
            Materialized list elements in index order.
        """
        for i in range(self._size):
            yield self[i]


class VariationDetail(TypedDict, total=False):
    """Details about a single variation entry (name, type, shape, dtype)."""

    name: str
    type: str
    shape: tuple[int, ...] | str
    dtype: str


class StateSpecs(TypedDict, total=False):
    """State-related specifications inferred from the dataset."""

    has_variations: bool
    variation_format: Literal["list_per_state", "dictionary", "list", "unknown"]
    variation_types: list[str]
    variation_details: list[VariationDetail]
    variation_shapes: VariationShapesType
    variation_dtypes: VariationDtypesType
    first_state_only: bool
    shape: tuple[int, ...]
    frame_shape: tuple[int, ...]
    dtype: str
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    sequence_length: int


class ActionSpecs(TypedDict, total=False):
    """Action-related specifications inferred from the dataset."""

    total_actions: int
    unique_actions: list[ActionType]
    action_counts: dict[ActionType, int]
    avg_actions_per_episode: float
    min_actions_per_episode: int
    max_actions_per_episode: int


class Specs(TypedDict, total=False):
    """Top-level dataset specifications for a single file."""

    num_episodes: int
    file_path: NotRequired[str]
    action_specs: ActionSpecs
    state_specs: StateSpecs


class ComparisonData(TypedDict):
    """Structured comparison metadata for multiple datasets."""

    dataset_names: list[str]
    num_episodes: list[int]
    episode_lengths: list[list[int]]
    has_variations: list[bool]
    action_stats: NotRequired[list[ActionStatsSummary]]
    state_shape_info: NotRequired[list[str]]


class ActionStatsEntry(TypedDict):
    """Summary of actions for a dataset used in comparison tables."""

    unique_actions: list[float]
    action_distribution: dict[float, int]
    most_common_action: float | None
    total_actions: int


class ActionStatsSummary(TypedDict):
    """Summary of actions for plotting across datasets."""

    unique_actions: list[float]
    action_distribution: dict[float, int]
    most_common_action: float | None
    total_actions: int


def _h5_read_pair_item(
    file_path: str,
    start_images_path: str,
    goal_images_path: str,
    var_names_start: list[str],
    var_names_goal: list[str],
    index: int,
) -> StateImgTraj:
    """Read a single search-pair item by index from HDF5 without loading all data."""
    with open_hdf5_for_read(file_path, rdcc_nbytes=_HDF5_INSPECTOR_RDCC_NBYTES) as f:
        start_obj: H5Types | None = f.get(start_images_path)
        goal_obj: H5Types | None = f.get(goal_images_path)
        if not (isinstance(start_obj, h5py.Dataset) and isinstance(goal_obj, h5py.Dataset)):
            raise TypeError("Invalid HDF5: start/goal images are not datasets")
        start_img: ImageArray = np.asarray(start_obj[index])
        goal_img: ImageArray = np.asarray(goal_obj[index])
        base_pair: ImageArray = np.stack((start_img, goal_img), axis=0)

        if not (var_names_start or var_names_goal):
            return base_pair

        variations_per_state: list[PerStateVariations] = [[], []]
        for name in var_names_start:
            ds_path = f"pairs/start/variations/{name}/images"
            obj: H5Types | None = f.get(ds_path)
            if isinstance(obj, h5py.Dataset):
                variations_per_state[0].append((name, np.asarray(obj[index])))
        for name in var_names_goal:
            ds_path = f"pairs/goal/variations/{name}/images"
            obj = f.get(ds_path)
            if isinstance(obj, h5py.Dataset):
                variations_per_state[1].append((name, np.asarray(obj[index])))
        return (base_pair, variations_per_state)


def _h5_read_episode_state(
    file_path: str, episode_key: str, variant_names: list[str], variant_type: str
) -> StateImgTraj:
    """Read a single episode's states and variations without loading all episodes."""
    with open_hdf5_for_read(file_path, rdcc_nbytes=_HDF5_INSPECTOR_RDCC_NBYTES) as f:
        episodes_root: H5Types | None = f.get("episodes")
        if not isinstance(episodes_root, h5py.Group):
            raise TypeError("Invalid HDF5 structure: 'episodes' is not a group")
        ep_obj: H5Types | None = episodes_root.get(episode_key)
        if not isinstance(ep_obj, h5py.Group):
            raise TypeError(f"Invalid episode group for key: {episode_key}")
        base_grp: H5Types | None = ep_obj.get("base")
        if not isinstance(base_grp, h5py.Group):
            raise TypeError("Episode missing 'base' group")
        states_ds: H5Types | None = base_grp.get("states")
        if not isinstance(states_ds, h5py.Dataset):
            raise TypeError("Episode base 'states' is not a dataset")
        base_states: ImageArray = np.asarray(states_ds[:])
        if len(variant_names) <= 1:
            return base_states
        num_states: int = int(base_states.shape[0])
        per_state: list[PerStateVariations] = [[] for _ in range(num_states)]
        for var_name in variant_names:
            if var_name == "base":
                continue
            var_grp: H5Types | None = ep_obj.get(var_name)
            if isinstance(var_grp, h5py.Group):
                var_states_ds: H5Types | None = var_grp.get("states")
                if isinstance(var_states_ds, h5py.Dataset):
                    vstates: ImageArray = np.asarray(var_states_ds[:])
                    if variant_type == "first_state_only" and num_states > 0:
                        per_state[0].append((var_name, vstates[0]))
                    else:
                        max_idx = min(num_states, int(vstates.shape[0]))
                        for s in range(max_idx):
                            per_state[s].append((var_name, vstates[s]))
        return (base_states, per_state)


def _h5_read_episode_actions(file_path: str, episode_key: str) -> ActionTraj:
    with open_hdf5_for_read(file_path, rdcc_nbytes=_HDF5_INSPECTOR_RDCC_NBYTES) as f:
        episodes_root: H5Types | None = f.get("episodes")
        if not isinstance(episodes_root, h5py.Group):
            return []
        ep_obj: H5Types | None = episodes_root.get(episode_key)
        if not isinstance(ep_obj, h5py.Group):
            return []
        actions_ds: H5Types | None = ep_obj.get("actions")
        if isinstance(actions_ds, h5py.Dataset):
            return list(np.asarray(actions_ds[:]))
        return []


# Global variable for file metadata
_file_metadata: dict[str, MetadataEntry] = {}

# Constants for compare_datasets function
_MAX_ACTIONS_DISPLAY: int = 15
_PLOT_FIGURE_SIZE: tuple[int, int] = (12, 8)
_PLOT_DPI: int = 150
_PLOT_STYLE: dict[RcKeyType, int | float] = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": _PLOT_DPI,
}


@dataclass(frozen=True)
class InspectorTextConfig:
    """Holds customizable textual labels used across inspector output.

    A ``None`` value for a key disables that specific text. Setting
    ``disable_all`` makes every key disabled regardless of individual
    values. This design keeps runtime checks extremely cheap and avoids
    complex conditionals in hot visualization paths.
    """

    mapping: dict[TextKey, str | None] = field(default_factory=dict)
    disable_all: bool = False

    # __slots__ removed to avoid conflicts with dataclass-generated fields

    @staticmethod
    def default() -> InspectorTextConfig:
        """Return a default label mapping used when no customization provided."""
        return InspectorTextConfig(
            mapping={
                "report.title": "SPAR DATA REPORT",
                "table.basic.title": "Basic Information",
                "table.state.title": "State Structure",
                "table.action.title": "Action Statistics",
                "table.action.dist.title": "Action Distribution",
                "table.variations.title": "Variations Overview",
                "tabular.title": "TABULAR DATA OVERVIEW",
                "structure.title": "DATA STRUCTURE",
                "visual.sample.title": "Sample Trajectory",
                "visual.timeline.title": "Timeline",
                "compare.title": "Dataset Comparison",
            }
        )

    def resolve(self, key: TextKey, default: str | None = None) -> str | None:
        """Return the configured value, or the default when the key is absent.

        Args:
            key : TextKey
                Identifier for the label.
            default : str | None
                Fallback string when key is absent.
        """
        if self.disable_all:
            return None
        return self.mapping.get(key, default)

    def is_enabled(self, key: TextKey) -> bool:
        """Return True if key maps to a non-None value and not globally disabled."""
        return (not self.disable_all) and (self.mapping.get(key, None) is not None)

    def override(self, updates: Mapping[TextKey, str | None]) -> InspectorTextConfig:
        """Return new config with updates merged (copy-on-write)."""
        if not updates:
            return self
        return InspectorTextConfig(mapping={**self.mapping, **updates}, disable_all=self.disable_all)

    def disable_keys(self, keys: Iterable[TextKey]) -> InspectorTextConfig:
        """Return new config with each listed key disabled."""
        new_map: dict[TextKey, str | None] = {**self.mapping}
        for k in keys:
            new_map[k] = None
        return InspectorTextConfig(mapping=new_map, disable_all=self.disable_all)


@dataclass(frozen=True)
class MatplotlibLayoutConfig:
    """Matplotlib layout + style parameters.

    Restricted surface that focuses on frequently tuned values to keep
    configuration explicit and predictable.
    """

    dpi: int = 150
    constrained: bool = True
    tight_layout: bool = False
    grid: bool = False
    wspace: float = 0.15
    hspace: float = 0.20
    padding: float = 0.05
    theme: Literal["default", "dark", "light"] = "default"

    # __slots__ removed to avoid conflicts with dataclass-generated fields

    def apply(self, fig: Figure) -> None:
        """Apply resolution, theme, and spacing settings to a figure.

        Args:
            fig: Figure to configure.
        """
        plt.rcParams["figure.dpi"] = self.dpi
        if self.theme == "dark":
            plt.style.use("dark_background")
        elif self.theme == "light":
            plt.style.use("default")
        if self.constrained:
            set_layout_engine: Callable[[str], None] | None = getattr(fig, "set_layout_engine", None)
            if callable(set_layout_engine):
                with contextlib.suppress(Exception):
                    set_layout_engine("constrained")
                    return
        fig.subplots_adjust(
            left=self.padding,
            right=1.0 - self.padding,
            top=1.0 - self.padding,
            bottom=self.padding,
            wspace=self.wspace,
            hspace=self.hspace,
        )
        if self.tight_layout:
            with contextlib.suppress(Exception):
                fig.tight_layout()


@dataclass(frozen=True)
class InspectorOptions:
    """Aggregate options controlling text, layout, and performance knobs."""

    text: InspectorTextConfig = field(default_factory=InspectorTextConfig.default)
    layout: MatplotlibLayoutConfig = field(default_factory=MatplotlibLayoutConfig)
    disable_all_text: bool = False
    show_titles: bool = True
    show_colorbars: bool = True
    max_grid_cols: int = 6
    perf_mode: bool = True  # skip heavier computations when True

    def effective_text(self) -> InspectorTextConfig:
        """Return text configuration honoring master disable flag."""
        if self.disable_all_text:
            return InspectorTextConfig(mapping=self.text.mapping, disable_all=True)
        return self.text


# Module-level inspector option cache.
_inspector_options_container: list[InspectorOptions] = [InspectorOptions()]  # mutable container avoids globals


def set_inspector_options(options: InspectorOptions | None) -> None:
    """Set global inspector options (primarily for CLI configuration).

    Args:
        options : InspectorOptions | None
            The options instance to install globally. ``None`` resets to defaults.
    """
    _inspector_options_container[0] = resolve_inspector_options(options)


def get_inspector_options() -> InspectorOptions:
    """Return current global inspector options (creating defaults lazily)."""
    return _inspector_options_container[0]


def resolve_text_key(key: TextKey, default: str | None = None, **fmt: FormatValue) -> str | None:
    """Resolve and format a text key, retaining the template on a missing field."""
    opts: InspectorOptions = get_inspector_options()
    cfg: InspectorTextConfig = opts.effective_text()
    raw: str | None = cfg.resolve(key, default)
    if raw is None:
        return None
    if fmt:
        with contextlib.suppress(Exception):
            return raw.format(**fmt)
    return raw


def is_text_enabled(key: TextKey) -> bool:
    """Return True if the given text key is currently enabled."""
    return get_inspector_options().effective_text().is_enabled(key)


def set_axis_title(ax: Axes, key: TextKey, default: str) -> None:
    """Apply a title to an axis using text configuration.

    If the resolved text is None, the title is omitted (disabled).
    """
    title: str | None = resolve_text_key(key, default)
    if title:
        ax.set_title(title)


def resolve_inspector_options(options: InspectorOptions | None) -> InspectorOptions:
    """Return caller options or a new default instance.

    Args:
        options: Caller-supplied inspector options.

    Returns:
        The supplied options, or default options when the argument is None.
    """
    return options if options is not None else InspectorOptions()


ensure_options = resolve_inspector_options


def resolve_text(
    text_config: InspectorTextConfig, key: TextKey, default: str | None = None, **format_kwargs: FormatValue
) -> str | None:
    """Resolve a string key using the provided configuration and format it when requested."""
    resolved: str | None = text_config.resolve(key, default)
    if resolved is None:
        return None
    if format_kwargs:
        try:
            return resolved.format(**format_kwargs)
        except KeyError:
            return resolved
    return resolved


def adapt_display_data(obj: DisplayInput) -> DisplayData:
    """Normalize arbitrary nested data into the DisplayData structure."""
    if isinstance(obj, (int, float, str, bool, np.ndarray)):
        return obj
    if isinstance(obj, list):
        return [adapt_display_data(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(adapt_display_data(x) for x in obj)
    if isinstance(obj, dict):
        return {str(k): adapt_display_data(v) for k, v in obj.items()}
    return str(obj)


def _normalize_image_array(arr: NDArray[np.generic]) -> ImageArray:
    """Convert an array to a dtype accepted by :data:`ImageArray`.

    Args:
        arr: Array to normalize.

    Returns:
        A float64 or float32 view when already compatible, otherwise a uint8
        conversion with float32 as the exception fallback.
    """
    if arr.dtype == np.float64:
        return np.asarray(arr, dtype=np.float64)
    if arr.dtype == np.float32:
        return np.asarray(arr, dtype=np.float32)
    # Convert other dtypes to the dashboard's integer display representation.
    try:
        return arr.astype(np.uint8)
    except Exception:
        # Some array subclasses reject uint8 conversion but accept float32.
        return arr.astype(np.float32)


def to_display_image(img_data: ImageArray | list[ImageArray], step_idx: int | None = None) -> ImageArray:
    """Convert arrays to display-ready H/W[/C] with channel handling.

    - Accepts (T,C,H,W), (C,H,W), (H,W,C), (H,W), and lists thereof.
    - Handles 6-channel images by splitting into two RGB images side-by-side.
    - Keeps dtype and scales unchanged for speed.
    """
    # Select element if list-like, and keep a separate raw variable
    candidate_raw: ImageArray | list[ImageArray] = img_data
    candidate_elem: ImageArray
    if isinstance(candidate_raw, list):
        if not candidate_raw:
            return np.zeros((32, 32, 3), dtype=np.uint8)
        idx: int = 0 if step_idx is None else max(0, min(step_idx, len(candidate_raw) - 1))
        candidate_elem = candidate_raw[idx]
    else:
        candidate_elem = candidate_raw

    frame: ImageArray = candidate_elem

    # Unpack time dimension if present (T, C, H, W)
    if frame.ndim == 4:
        t: int = frame.shape[0]
        idx = 0 if step_idx is None else max(0, min(step_idx, t - 1))
        frame = frame[idx]

    # If already 2D, return normalized dtype
    if frame.ndim == 2:
        return _normalize_image_array(frame)

    # Determine channel layout for 3D frames (C,H,W) or (H,W,C)
    if frame.ndim == 3:
        # Channels-first if first dim looks like channels and is smaller than spatial dims
        h, w = frame.shape[-2], frame.shape[-1]
        if frame.shape[0] in {1, 3, 6} and frame.shape[0] < h and frame.shape[0] < w:
            c: int = frame.shape[0]
            if c == 3:
                out: ImageArray = frame.transpose(1, 2, 0)
                return _normalize_image_array(out)
            if c == 6:
                img1: ImageArray = frame[:3].transpose(1, 2, 0)
                img2: ImageArray = frame[3:].transpose(1, 2, 0)
                concat: ImageArray = np.concatenate((img1, img2), axis=1)
                return _normalize_image_array(concat)
            if c == 1:
                hwc: ImageArray = frame.transpose(1, 2, 0)
                return _normalize_image_array(hwc[:, :, 0])

        # Channels-last if last dim looks like channels
        if frame.shape[-1] in {1, 3, 6} and frame.shape[-1] < frame.shape[0] and frame.shape[-1] < frame.shape[1]:
            c = frame.shape[-1]
            if c == 3:
                return _normalize_image_array(frame)
            if c == 6:
                img1 = frame[..., :3]
                img2 = frame[..., 3:]
                concat = np.concatenate((img1, img2), axis=1)
                return _normalize_image_array(concat)
            if c == 1:
                return _normalize_image_array(frame[:, :, 0])

        # Fallback: if more than 3 channels, keep first 3
        if frame.shape[-1] > 3:
            return _normalize_image_array(frame[..., :3])

    # Final fallback: normalize whatever we have
    return _normalize_image_array(frame)


def _screen_size_from_tk() -> tuple[float, float]:
    """Determine screen size using Tkinter display metrics."""
    # Initialize Tkinter and hide root window
    root = tk.Tk()
    root.withdraw()
    # Refresh screen dimensions before applying the figure cap.
    root.update_idletasks()
    # Get pixel dimensions
    px_w: int = root.winfo_screenwidth()
    px_h: int = root.winfo_screenheight()
    # Get physical dimensions in millimeters
    mm_w: int = root.winfo_screenmmwidth()
    mm_h: int = root.winfo_screenmmheight()
    root.destroy()
    # Prefer physical measurement if available
    if mm_w > 0 and mm_h > 0:
        in_w: float = mm_w / 25.4
        in_h: float = mm_h / 25.4
    else:
        # Fallback to pixel-based conversion
        dpi = plt.rcParams.get("figure.dpi", 100)
        in_w = px_w / dpi
        in_h = px_h / dpi

    return in_w, in_h


def get_screen_size() -> tuple[float, float]:
    """Get screen size in inches using physical dimensions when possible, fallback to pixel/DPI conversion."""
    try:
        return _screen_size_from_tk()
    except Exception:
        # Final fallback: assume 1920x1080 and convert using default DPI
        dpi = plt.rcParams.get("figure.dpi", 100)
        return 1920 / dpi, 1080 / dpi


def constrain_figure_size(
    desired_width: float, desired_height: float, margin_factor: float = 0.8
) -> tuple[float, float]:
    """Constrain figure size to fit within screen bounds.

    Args:
        desired_width: Desired figure width in inches
        desired_height: Desired figure height in inches
        margin_factor: Factor to scale down from full screen (0.8 = 80% of screen)

    Returns:
        tuple[float, float]: Constrained width and height in inches
    """
    # Clamp margin_factor to [0.0, 1.0]
    margin: float = max(0.0, min(margin_factor, 1))
    screen_width, screen_height = get_screen_size()
    max_width: float = screen_width * margin
    max_height: float = screen_height * margin

    # Calculate scaling factor to fit within bounds while maintaining aspect ratio
    width_ratio: float = max_width / desired_width if desired_width > max_width else 1.0
    height_ratio: float = max_height / desired_height if desired_height > max_height else 1.0

    scale_factor: float = min(width_ratio, height_ratio)

    constrained_width: float = desired_width * scale_factor
    constrained_height: float = desired_height * scale_factor

    return constrained_width, constrained_height


def actions_to_floats(actions: Iterable[ActionType]) -> list[float]:
    """Convert an iterable of ActionType into a list of Python floats.

    This helper converts NumPy scalar types and other numeric values to
    built-in Python floats for sorting and plotting.

    Args:
        actions: Iterable of numeric actions.

    Returns:
        List[float]: Converted action values as floats. Empty iterable -> [].
    """
    result: list[float] = []
    for a in actions:
        # Built-in numeric types need no NumPy conversion.
        if isinstance(a, (int, float)):
            result.append(float(a))
            continue

        # Numpy scalar types: np.floating, np.integer
        try:
            # np.asarray accepts NumPy scalars and arrays. .item() extracts the Python scalar.
            scalar: float = np.asarray(a).item()
            result.append(scalar)
            continue
        except Exception:
            # Fallback: attempt direct float conversion
            with contextlib.suppress(Exception):
                result.append(float(a))

    return result


def show_or_save_plot(save_path: str | None = None) -> None:
    """Helper function to either show a plot or save it to a file.

    Args:
        save_path: Path to save the plot to. If None, shows the plot instead.
                  Supports formats: PNG, JPG, JPEG, PDF, SVG, EPS, PS
    """
    if save_path:
        plt.savefig(save_path, dpi=_PLOT_DPI, bbox_inches="tight")
        plt.close()
        console.print(f"📁 Plot saved to: [bold cyan]{save_path}[/bold cyan]")
    else:
        plt.show()


def _raise_value_error(message: str) -> NoReturn:
    """Raise a value error via helper to keep control flow explicit."""
    raise ValueError(message)


def _load_data_file_checked(file_path: str, file_ext: str) -> tuple[list[StateImgTraj], list[ActionTraj]]:
    """Load data from a supported file path after caller-side path validation."""
    if file_ext in {".h5", ".hdf5"}:
        # Handle HDF5 files from both the RL generator (episodes) and the search pair generator (pairs)
        with open_hdf5_for_read(file_path, rdcc_nbytes=_HDF5_INSPECTOR_RDCC_NBYTES) as f:
            # Branch 1: New search pair format (spar/data/search_data_generator.py)
            if "pairs" in f:
                pairs_grp_obj = f["pairs"]
                pairs_grp: h5py.Group
                if not isinstance(pairs_grp_obj, h5py.Group):
                    _raise_value_error("Invalid HDF5 structure: 'pairs' is not a group")
                pairs_grp = pairs_grp_obj

                # Root metadata
                _file_metadata["env_name"] = str(f.attrs.get("env_name") or "unknown")
                num_pairs = int(f.attrs.get("num_pairs", 0))
                _file_metadata["num_pairs"] = num_pairs
                _file_metadata["num_episodes"] = num_pairs  # for downstream display consistency
                _file_metadata["reverse_goal"] = bool(f.attrs.get("reverse_goal", False))
                # Store as int when possible, otherwise keep original value
                gns: int | str = f.attrs.get("goal_num_steps", -1)
                try:
                    _file_metadata["goal_num_steps"] = int(gns)
                except Exception:
                    _file_metadata["goal_num_steps"] = gns

                # Identify dataset kind (pairs) for downstream UI logic
                raw_dk: np.generic | bytes | bytearray | str = f.attrs.get("dataset_kind", "search_pairs_v1")
                _file_metadata["dataset_kind"] = decode_attr_text(raw_dk)

                # Base images
                start_grp_obj = pairs_grp.get("start")
                goal_grp_obj = pairs_grp.get("goal")
                if not (isinstance(start_grp_obj, h5py.Group) and isinstance(goal_grp_obj, h5py.Group)):
                    _raise_value_error("Missing 'pairs/start' or 'pairs/goal' groups")
                start_grp: h5py.Group = start_grp_obj
                goal_grp: h5py.Group = goal_grp_obj

                start_images_ds_obj = start_grp.get("images")
                goal_images_ds_obj = goal_grp.get("images")
                if not (isinstance(start_images_ds_obj, h5py.Dataset) and isinstance(goal_images_ds_obj, h5py.Dataset)):
                    _raise_value_error("Missing base images under 'pairs/start/images' or 'pairs/goal/images'")
                start_images_ds: h5py.Dataset = start_images_ds_obj
                goal_images_ds: h5py.Dataset = goal_images_ds_obj

                # Load the two base arrays (N, C, H, W)
                # Determine dataset paths for on-demand reads
                start_images_path: str = str(start_images_ds.name)
                goal_images_path: str = str(goal_images_ds.name)

                # Discover variation names for start/goal (if any)
                var_names_start: list[str] = []
                var_names_goal: list[str] = []
                # Safely get variations group and use .keys() to extract names
                start_vg = start_grp.get("variations")
                goal_vg = goal_grp.get("variations")
                if isinstance(start_vg, h5py.Group):
                    var_names_start = sorted(start_vg.keys())
                if isinstance(goal_vg, h5py.Group):
                    var_names_goal = sorted(goal_vg.keys())

                # Union for reporting
                all_var_names: list[str] = sorted(set(var_names_start) | set(var_names_goal))
                _file_metadata["variant_names"] = all_var_names
                _file_metadata["has_variations"] = len(all_var_names) > 0
                # Pair-specific metadata
                _file_metadata["variant_names_start"] = var_names_start
                _file_metadata["variant_names_goal"] = var_names_goal
                # Where variations were applied: none/start/goal/both
                if var_names_start and var_names_goal:
                    _file_metadata["variant_sides"] = "both"
                elif var_names_start:
                    _file_metadata["variant_sides"] = "start"
                elif var_names_goal:
                    _file_metadata["variant_sides"] = "goal"
                else:
                    _file_metadata["variant_sides"] = "none"
                # Back-compat: expose a coarse variant_type field for generic UIs
                _file_metadata["variant_type"] = (
                    "first_state_only"
                    if _file_metadata["variant_sides"] == "start"
                    else "all_states"
                    if _file_metadata["variant_sides"] in {"both", "goal"}
                    else "none"
                )

                # Build lazy containers based on dataset shapes
                n = int(start_images_ds.shape[0])
                # Lazy states
                lazy_states: LazyList[StateImgTraj] = LazyList(
                    size=n,
                    loader=lambda idx: _h5_read_pair_item(
                        file_path=file_path,
                        start_images_path=start_images_path,
                        goal_images_path=goal_images_path,
                        var_names_start=var_names_start,
                        var_names_goal=var_names_goal,
                        index=idx,
                    ),
                    placeholder=np.zeros((0, 0), dtype=np.uint8),
                )
                # Lazy actions (empty lists)
                lazy_actions: LazyList[ActionTraj] = LazyList(n, loader=lambda _i: [], placeholder=[])

                return lazy_states, lazy_actions

            # Branch 2: Original RL episode format (generator.py)
            if "episodes" in f:
                # Read global metadata (if present)
                _file_metadata["num_episodes"] = f.attrs.get("num_episodes", 0)

                variant_type_str: str = read_variant_type_from_attrs(f.attrs, default="unknown")
                _file_metadata["variant_type"] = variant_type_str

                variant_names: list[str] = read_variant_names_from_attrs(f.attrs)
                _file_metadata["variant_names"] = variant_names

                has_variations: bool = len(variant_names) > 1
                _file_metadata["has_variations"] = has_variations

                episodes_group: h5py.Group | h5py.Dataset | h5py.Datatype = f["episodes"]
                if not isinstance(episodes_group, h5py.Group):
                    _raise_value_error("Invalid HDF5 structure: 'episodes' is not a group")

                # Prepare lazy episode containers
                episode_keys: list[str] = sorted(episodes_group.keys())
                _file_metadata["episode_keys"] = episode_keys

                variant_type: str = str(_file_metadata.get("variant_type", "unknown"))

                lazy_states_ep: LazyList[StateImgTraj] = LazyList(
                    len(episode_keys),
                    loader=lambda idx: _h5_read_episode_state(
                        file_path, episode_keys[idx], variant_names, variant_type
                    ),
                    placeholder=np.zeros((0, 0), dtype=np.uint8),
                )
                lazy_actions_ep: LazyList[ActionTraj] = LazyList(
                    len(episode_keys),
                    loader=lambda idx: _h5_read_episode_actions(file_path, episode_keys[idx]),
                    placeholder=[],
                )

                return lazy_states_ep, lazy_actions_ep

            # Unknown HDF5 layout
            _raise_value_error("Unsupported HDF5 layout: expected 'pairs' or 'episodes' group at root")

    _raise_value_error(f"Unsupported file extension: {file_ext}. Only .h5 and .hdf5 files are supported.")


def load_data_file(file_path: str) -> tuple[list[StateImgTraj], list[ActionTraj]]:
    """Load a data file generated by the SPAR data generation pipeline.

    Args:
        file_path (str): Path to the data file.

    Returns:
        tuple[list[StateImgTraj], list[ActionTraj]]: State image trajectories and action trajectories.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not recognized or loading fails.
    """
    # Reset metadata dict
    _file_metadata.clear()

    if not pathlib.Path(file_path).exists():
        # Try adding common extensions if file doesn't exist
        if pathlib.Path(f"{file_path}.h5").exists():
            file_path += ".h5"
        elif pathlib.Path(f"{file_path}.hdf5").exists():
            file_path += ".hdf5"
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    console.print(f"🔍 Loading data file: [bold cyan]{file_path}[/bold cyan]")
    _file_metadata["source_file_path"] = file_path

    # Check file extension
    file_ext: str = pathlib.Path(file_path).suffix.lower()

    try:
        return _load_data_file_checked(file_path, file_ext)
    except Exception as err:
        console.print(f"[red]❌ Error loading data file:[/red] {err}")
        traceback.print_exc()
        raise ValueError(f"Could not load data file: {file_path}. Error: {err}") from err


def get_data_specs(state_img_trajs: list[StateImgTraj], action_trajs: list[ActionTraj]) -> Specs:
    """Analyze loaded data and return detailed specifications.

    Args:
        state_img_trajs: State image trajectories.
        action_trajs (list[ActionTraj]): Action trajectories.

    Returns:
        Specs: Detailed specifications.

    """
    state_specs: StateSpecs = {}
    action_specs: ActionSpecs = {}
    specs: Specs = {"num_episodes": len(state_img_trajs), "action_specs": action_specs, "state_specs": state_specs}

    if not state_img_trajs:
        return specs

    # Analyze action data
    all_actions: ActionTraj = [action for traj in action_trajs for action in traj]
    specs["action_specs"] = {
        "total_actions": len(all_actions),
        # Convert unique actions to built-in floats for display and plotting.
        "unique_actions": sorted(actions_to_floats(set(all_actions))),
        # Keep counts keyed by the original action values (may include numpy scalars)
        "action_counts": {action: all_actions.count(action) for action in set(all_actions)},
        "avg_actions_per_episode": len(all_actions) / len(action_trajs) if len(action_trajs) > 0 else 0,
        "min_actions_per_episode": min(len(traj) for traj in action_trajs) if len(action_trajs) > 0 else 0,
        "max_actions_per_episode": max(len(traj) for traj in action_trajs) if len(action_trajs) > 0 else 0,
    }

    # Analyze state image data
    first_item: StateImgTraj = state_img_trajs[0]
    clean_states: ImageArray | list[ImageArray]

    # Check if this is a clean state or has variations
    if isinstance(first_item, tuple) and len(first_item) == 2:
        # Data contains variations
        clean_states, variations_per_state = first_item
        state_specs["has_variations"] = True

        # variations_per_state is a list of lists of (var_name, var_image) tuples
        # Extract variation types from the data
        all_variation_names: set[str] = {"base"}  # Always include base

        if isinstance(variations_per_state, list) and variations_per_state:
            all_variation_names.update(var_name for state_vars in variations_per_state for var_name, _ in state_vars)

        variation_types: list[str] = list(all_variation_names)
        state_specs["variation_format"] = "list_per_state"
        state_specs["variation_types"] = variation_types

        # Analyze variation details
        variation_specs: list[VariationDetail] = []
        # clean_states may be a single ndarray (T,C,H,W) or a list/sequence of ndarrays (per-step frames).
        base_shape: tuple[int, ...] | str = "unknown"
        base_dtype: str = "unknown"
        base_type: str = "unknown"

        if isinstance(clean_states, np.ndarray):
            base_shape = tuple(int(x) for x in clean_states.shape)
            base_dtype = str(clean_states.dtype)
            base_type = "ndarray"
        elif clean_states:
            first_frame = clean_states[0]
            base_shape = tuple(int(x) for x in first_frame.shape)
            base_dtype = str(first_frame.dtype)
            base_type = "ndarray_sequence"
        base_spec: VariationDetail = {"name": "base", "shape": base_shape, "dtype": base_dtype, "type": base_type}
        variation_specs.append(base_spec)

        # Analyze first variation of each type if available
        for var_name in variation_types:
            if var_name == "base":
                continue

            sample_var: ImageArray | None = None
            if isinstance(variations_per_state, list):
                sample_var = next(
                    (vdata for state_vars in variations_per_state for vname, vdata in state_vars if vname == var_name),
                    None,
                )

            if isinstance(sample_var, np.ndarray):
                variation_specs.append({
                    "name": var_name,
                    "shape": tuple(int(x) for x in sample_var.shape),
                    "dtype": str(sample_var.dtype),
                    "type": "ndarray",
                })

        state_specs["variation_details"] = variation_specs

        # Check if this is first_state_only pattern
        if isinstance(variations_per_state, list):
            has_variations_beyond_first: bool = any(len(state_vars) > 0 for state_vars in variations_per_state[1:])
            state_specs["first_state_only"] = not has_variations_beyond_first
        else:
            state_specs["first_state_only"] = False
    else:
        # No variations
        clean_states = first_item
        state_specs["has_variations"] = False

    # Analyze clean states
    if isinstance(clean_states, np.ndarray):
        state_specs["shape"] = tuple(int(x) for x in clean_states.shape)
        state_specs["dtype"] = str(clean_states.dtype)
        state_specs["min_value"] = float(np.min(clean_states))
        state_specs["max_value"] = float(np.max(clean_states))
        state_specs["mean_value"] = float(clean_states.mean())
        state_specs["std_value"] = float(clean_states.std())

    elif clean_states:  # non-empty list of frames
        state_specs["sequence_length"] = len(clean_states)
        state_specs["frame_shape"] = tuple(int(x) for x in clean_states[0].shape)
        state_specs["dtype"] = str(clean_states[0].dtype)

    return specs


def print_tabular_overview(state_img_trajs: list[StateImgTraj], _action_trajs: list[ActionTraj], specs: Specs) -> None:
    """Generate and print a tabular overview of the data structure.

    Args:
        state_img_trajs: State image trajectories.
        _action_trajs (list[ActionTraj]): Action trajectories.
        specs: Data specifications.

    """
    console.print()
    console.print(Panel.fit("📊 TABULAR DATA OVERVIEW", style="bold blue"))

    # TABLE 1: Basic Information
    basic_table: Table = Table(title="Basic Information", box=box.ROUNDED)
    basic_table.add_column("Property", style="bold cyan")
    basic_table.add_column("Value", style="white")

    num_episodes: int = specs.get("num_episodes", 0)
    basic_table.add_row("Number of Episodes", str(num_episodes))

    file_path: str = specs.get("file_path") or ""
    file_format: str = "Compressed (bz2)" if ".bz2" in file_path else "Uncompressed"
    basic_table.add_row("File Format", file_format)

    state_specs: StateSpecs = specs.get("state_specs", {})
    hv_bool: bool = state_specs.get("has_variations", False)
    has_variations: str = "Yes" if hv_bool else "No"
    basic_table.add_row("Has Variations", has_variations)

    variation_format: Literal["list_per_state", "dictionary", "list", "unknown"] | None
    if hv_bool:
        variation_format = state_specs.get("variation_format")
        if variation_format == "list":
            num_variations = state_specs.get("num_variations", "N/A")
            basic_table.add_row("Number of Variations", str(num_variations))
        elif variation_format == "dictionary":
            variation_types: list[str] = state_specs.get("variation_types", [])
            basic_table.add_row("Number of Variation Types", str(len(variation_types)))

    console.print(basic_table)

    # TABLE 2: State Structure
    state_table: Table = Table(title="State Structure", box=box.ROUNDED)
    state_table.add_column("Property", style="bold green")
    state_table.add_column("Value", style="white")

    first_item: StateImgTraj | None = state_img_trajs[0] if state_img_trajs else None

    # Extract clean states
    clean_states: StateFrames | None = (
        first_item[0] if isinstance(first_item, tuple) and len(first_item) == 2 else first_item
    )
    dataset_kind: str | float | list[str] | None = _file_metadata.get("dataset_kind")

    if isinstance(clean_states, np.ndarray):
        is_pairs: bool = (
            isinstance(dataset_kind, str)
            and dataset_kind.startswith("search_pairs")
            and clean_states.ndim >= 3
            and clean_states.shape[0] == 2
        )
        if is_pairs:
            state_table.add_row("State Type", "Pair (Start, Goal)")
            # Show per-frame shape (C,H,W) if available
            frame_shape: tuple[int, int, int] = clean_states.shape[1:] if clean_states.ndim >= 4 else clean_states.shape
            state_table.add_row("Pair Length", "2")
            state_table.add_row("Frame Shape", str(frame_shape))
            state_table.add_row("Data Type", str(clean_states.dtype))
            # Memory estimate per frame
            per_frame_elems = int(np.prod(clean_states.shape[1:] if clean_states.ndim >= 4 else clean_states.shape))
            mem_mb_per_frame: float = (per_frame_elems * clean_states.itemsize) / (1024 * 1024)
            state_table.add_row("Memory per Frame", f"{mem_mb_per_frame:.2f} MB")
        else:
            state_table.add_row("State Type", "Single Frame")
            state_table.add_row("State Shape", str(clean_states.shape))
            state_table.add_row("Data Type", str(clean_states.dtype))

            # Add memory usage estimate
            mem_mb: float = (clean_states.size * clean_states.itemsize) / (1024 * 1024)
            state_table.add_row("Memory per State", f"{mem_mb:.2f} MB")

    elif isinstance(clean_states, list) and clean_states:
        state_table.add_row("State Type", "Sequence")
        state_table.add_row("Sequence Length", str(len(clean_states)))
        state_table.add_row("Frame Shape", str(clean_states[0].shape))
        state_table.add_row("Data Type", str(clean_states[0].dtype))

    console.print(state_table)

    # TABLE 3: Action Statistics
    action_table: Table = Table(title="Action Statistics", box=box.ROUNDED)
    action_table.add_column("Property", style="bold yellow")
    action_table.add_column("Value", style="white")

    action_specs: ActionSpecs = specs.get("action_specs", {})
    total_actions: int | str = action_specs.get("total_actions", "N/A")
    unique_actions: int = len(action_specs.get("unique_actions", []))
    avg_actions: float | int = action_specs.get("avg_actions_per_episode", 0)
    min_actions: int | str = action_specs.get("min_actions_per_episode", "N/A")
    max_actions: int | str = action_specs.get("max_actions_per_episode", "N/A")

    action_table.add_row("Total Actions", str(total_actions))
    action_table.add_row("Unique Actions", str(unique_actions))
    action_table.add_row("Actions per Episode (avg)", f"{avg_actions:.2f}")
    action_table.add_row("Actions per Episode (min)", str(min_actions))
    action_table.add_row("Actions per Episode (max)", str(max_actions))

    console.print(action_table)

    # TABLE 4: Action Distribution (if not too many unique actions)
    action_counts: dict[ActionType, int] = action_specs.get("action_counts", {})
    if action_counts and len(action_counts) <= 20:  # Only show if not too many
        dist_table = Table(title="Action Distribution", box=box.ROUNDED)
        dist_table.add_column("Action", style="bold magenta")
        dist_table.add_column("Count", justify="right", style="white")
        dist_table.add_column("Percentage", justify="right", style="green")

        total: int = sum(action_counts.values())
        for action, count in sorted(action_counts.items(), key=operator.itemgetter(1), reverse=True):
            percentage: float = (count / total) * 100 if total > 0 else 0
            dist_table.add_row(str(action), str(count), f"{percentage:.1f}%")

        console.print(dist_table)
        # TABLE 5: Variations Overview
        if hv_bool:
            var_table = Table(title="Variations Overview", box=box.ROUNDED)

            if state_specs.get("variation_format") == "dictionary":
                # Setup columns for dictionary format
                var_table.add_column("Variation", style="bold red")
                var_table.add_column("Shape", style="white")
                var_table.add_column("Data Type", style="cyan")

                # Add base row first
                var_table.add_row("base", str(state_specs.get("shape", "N/A")), state_specs.get("dtype", "N/A"))

                # Add other variations
                for var_name in state_specs.get("variation_types", []):
                    if var_name != "base":
                        variation_shapes: VariationShapesType = state_specs.get("variation_shapes", {})
                        variation_dtypes: VariationDtypesType = state_specs.get("variation_dtypes", {})
                        var_table.add_row(
                            var_name, str(variation_shapes.get(var_name, "N/A")), variation_dtypes.get(var_name, "N/A")
                        )

            elif state_specs.get("variation_format") == "list":
                # Setup columns for list format
                var_table.add_column("Name", style="bold red")
                var_table.add_column("Type", style="yellow")
                var_table.add_column("Shape", style="white")
                var_table.add_column("Data Type", style="cyan")

                # Add rows for each variation
                for var in state_specs.get("variation_details", []):
                    var_table.add_row(
                        var.get("name", "Unknown"),
                        var.get("type", "Unknown"),
                        str(var.get("shape", "N/A")),
                        var.get("dtype", "N/A"),
                    )

            console.print(var_table)
    console.print()  # Add spacing


def display_data_structure(
    data: DisplayData, max_depth: int = 5, current_depth: int = 0, prefix: str = "", is_last: bool = True
) -> None:
    """Display the hierarchical structure of data in a tree-like format.

    Args:
        data (DisplayData): The data to display.
        max_depth (int): Maximum display depth.
        current_depth (int): Current recursion depth.
        prefix (str): Line prefix for formatting.
        is_last (bool): True if this is the last item in its branch.

    """
    # Common formatting setup
    branch: Literal["└──", "├──"] = "└──" if is_last else "├──"
    current_prefix: str = f"{prefix}{branch}"

    # Stop recursion if we've reached max depth
    if current_depth >= max_depth:
        console.print(f"{current_prefix} ... (max depth reached)")
        return

    # Common extension for child items
    extension: Literal["    ", "│   "] = "    " if is_last else "│   "
    child_prefix: str = prefix + extension

    # Different handling based on data type
    if isinstance(data, (list, tuple)):
        container_type: Literal["List", "Tuple"] = "List" if isinstance(data, list) else "Tuple"
        length: int = len(data)

        # Handle empty containers
        if length == 0:
            console.print(f"{current_prefix} {container_type}[0] (empty)")
            return

        # Show container type and length
        console.print(f"{current_prefix} {container_type}[{length}]")

        # Sample items if the list is very large
        if length > 10:
            # Show first 3 items
            for i in range(3):
                display_data_structure(data[i], max_depth, current_depth + 1, child_prefix, False)

            # Indicate skipped items
            console.print(f"{child_prefix}├── ... ({length - 6} more items)")

            # Show last 3 items
            for i in range(length - 3, length):
                display_data_structure(data[i], max_depth, current_depth + 1, child_prefix, i == length - 1)
        else:
            # Show all items for smaller lists
            for i, item in enumerate(data):
                display_data_structure(item, max_depth, current_depth + 1, child_prefix, i == length - 1)

    elif isinstance(data, dict):
        length = len(data)
        # Handle empty dict
        if length == 0:
            console.print(f"{current_prefix} Dict[0] (empty)")
            return

        # Show dict info
        console.print(f"{current_prefix} Dict[{length}]")

        # Process all keys
        keys: list[str] = list(data.keys())
        last_key_idx: int = length - 1
        for i, key in enumerate(keys):
            key_str: str = f"{key[:17]}..." if len(key) > 20 else key

            # Show key and value
            is_last_key: bool = i == last_key_idx
            key_branch: Literal["└──", "├──"] = "└──" if is_last_key else "├──"
            console.print(f"{child_prefix}{key_branch} {key_str}:")

            # Calculate prefix for the value
            value_prefix: str = child_prefix + ("    " if is_last_key else "│   ")
            display_data_structure(data[key], max_depth, current_depth + 1, value_prefix, True)

    elif isinstance(data, np.ndarray):
        # Get min/max values safely
        try:
            min_val, max_val = np.min(data), np.max(data)
            val_range: str = f", range: [{min_val:.4g}, {max_val:.4g}]"
        except Exception:
            val_range = ""

        console.print(f"{current_prefix} ndarray{data.shape} ({data.dtype}{val_range})")

        # For image-like arrays, suggest visualization
        if len(data.shape) in {2, 3, 4} and current_depth < max_depth - 1:
            console.print(f"{child_prefix}└── (Use --visualize to view image data)")

    elif hasattr(data, "__dict__"):
        # Handle custom objects
        console.print(f"{current_prefix} {data.__class__.__name__} object")

        # Handle attributes if not too deep
        if current_depth < max_depth - 1:
            attributes = vars(data)
            attrs: list[str] = list(attributes.keys())

            if attrs:
                last_attr_idx: int = len(attrs) - 1
                for i, attr in enumerate(attrs):
                    is_last_attr: bool = i == last_attr_idx
                    attr_branch: Literal["└──", "├──"] = "└──" if is_last_attr else "├──"
                    console.print(f"{child_prefix}{attr_branch} {attr}:")

                    # Calculate prefix for the attribute value
                    attr_prefix: str = child_prefix + ("    " if is_last_attr else "│   ")
                    display_data_structure(attributes[attr], max_depth, current_depth + 2, attr_prefix, True)
            else:
                console.print(f"{child_prefix}└── (no attributes)")

    else:
        # Handle primitive types
        data_str: str = str(data)
        if len(data_str) > 50:
            data_str = f"{data_str[:47]}..."

        console.print(f"{current_prefix} {type(data).__name__}: {data_str}")


def visualize_data_structure(
    state_img_trajs: list[StateImgTraj], action_trajs: list[ActionTraj], max_depth: int = 5
) -> None:
    """Visualize the hierarchical structure of the dataset.

    Args:
        state_img_trajs: State image trajectories.
        action_trajs: Action trajectories.
        max_depth: Maximum depth for visualization.
    """
    console.print()
    console.print(Panel.fit("🏗️  DATA STRUCTURE HIERARCHY", style="bold blue"))

    console.print("\n[bold green]Overall Dataset Structure:[/bold green]")
    console.print("├── state_img_trajs (images/states)")
    console.print("└── action_trajs (actions)")

    console.print("\n[bold cyan]Detailed State Images Structure:[/bold cyan]")
    # Adapt lists to DisplayData via recursive transform (since we removed casts)

    def _adapt(obj: AdaptInput) -> DisplayData:
        if isinstance(obj, (int, float, str, bool, np.ndarray)):
            return obj
        if isinstance(obj, list):
            return [_adapt(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_adapt(x) for x in obj)
        if isinstance(obj, dict):
            return {str(k): _adapt(v) for k, v in obj.items()}
        return str(obj)

    display_data_structure(_adapt(state_img_trajs), max_depth=max_depth, prefix="")

    console.print("\n[bold yellow]Detailed Actions Structure:[/bold yellow]")
    display_data_structure(_adapt(action_trajs), max_depth=max_depth, prefix="")

    console.print()


def get_file_format_info(file_path: str) -> str:
    """Determine file format based on extension.

    Args:
        file_path (str): Path to the data file.

    Returns:
        str: File format description.
    """
    file_ext: str = pathlib.Path(file_path).suffix.lower()
    if file_ext == ".pkl":
        return "Pickle file"
    if file_ext == ".bz2":
        return "Compressed pickle (bz2)"
    if file_ext in {".h5", ".hdf5"}:
        return "HDF5 file"
    return f"Unknown ({file_ext})" if file_ext else "Unknown"


def create_file_info_table(file_path: str, file_size_mb: float, compression_status: str) -> Table:
    """Create file information table.

    Args:
        file_path (str): Path to the data file.
        file_size_mb (float): File size in MB.
        compression_status (str): File format description.

    Returns:
        Table: Formatted file information table.
    """
    file_table = Table(title="File Information", box=box.ROUNDED)
    file_table.add_column("Property", style="bold cyan")
    file_table.add_column("Value", style="white")

    file_size: int = pathlib.Path(file_path).stat().st_size
    file_table.add_row("Path", file_path)
    file_table.add_row("Size", f"{file_size_mb:.2f} MB ({file_size:,} bytes)")
    file_table.add_row("Format", compression_status)

    return file_table


def add_metadata_rows(table: Table, metadata: Mapping[str, MetadataEntry]) -> None:
    """Add metadata rows to file table.

    Args:
        table (Table): Table to add rows to.
        metadata: File metadata.
    """
    if "created_timestamp" in metadata:
        ts_val: MetadataEntry = metadata["created_timestamp"]
        if isinstance(ts_val, (int, float, str)):
            try:
                creation_time: datetime = datetime.fromtimestamp(float(ts_val), tz=timezone.utc)
                table.add_row("Created", creation_time.strftime("%Y-%m-%d %H:%M:%S"))
            except Exception:
                pass
        # if it's not convertible (e.g., list), skip

    if "env_name" in metadata:
        table.add_row("Environment", str(metadata["env_name"]))

    if "num_episodes" in metadata:
        table.add_row("Episodes (metadata)", str(metadata["num_episodes"]))
    if "num_pairs" in metadata and metadata.get("num_pairs") != metadata.get("num_episodes"):
        table.add_row("Pairs (metadata)", str(metadata["num_pairs"]))

    if "has_variations" in metadata:
        table.add_row("Has variations (metadata)", "Yes" if bool(metadata["has_variations"]) else "No")
    if "variant_type" in metadata:
        table.add_row("Variation scope", str(metadata["variant_type"]))
    if "variant_names" in metadata:
        names_val: MetadataEntry = metadata["variant_names"]
        names_str: str = ", ".join(names_val) if isinstance(names_val, list) else str(names_val)
        table.add_row("Variation types", names_str)

    # Pairs-specific: which sides have variations and per-side names
    if "variant_sides" in metadata:
        table.add_row("Variation sides", str(metadata["variant_sides"]))
    if "variant_names_start" in metadata:
        vns: MetadataEntry = metadata["variant_names_start"]
        vns_str: str = ", ".join(vns) if isinstance(vns, list) and vns else "None"
        table.add_row("Start variations", vns_str)
    if "variant_names_goal" in metadata:
        vng: MetadataEntry = metadata["variant_names_goal"]
        vng_str: str = ", ".join(vng) if isinstance(vng, list) and vng else "None"
        table.add_row("Goal variations", vng_str)

    if "reverse_goal" in metadata:
        table.add_row("Reverse goal", "Yes" if bool(metadata["reverse_goal"]) else "No")

    # Safely display goal_num_steps which may be stored as int or string
    if "goal_num_steps" in metadata:
        gns_val: int | None = None
        raw: MetadataEntry = metadata["goal_num_steps"]
        if isinstance(raw, (int, float, np.integer, np.floating)):
            gns_val = int(raw)
        elif isinstance(raw, str):
            with contextlib.suppress(Exception):
                gns_val = int(raw)
        if gns_val is not None and gns_val >= 0:
            table.add_row("Goal scramble steps", str(gns_val))


def create_dataset_overview_table(specs: Specs) -> Table:
    """Create dataset overview table.

    Args:
        specs: Data specifications.

    Returns:
        Table: Formatted dataset overview table.
    """
    overview_table = Table(title="Dataset Overview", box=box.ROUNDED)
    overview_table.add_column("Property", style="bold green")
    overview_table.add_column("Value", style="white")
    overview_table.add_row("Number of episodes", str(specs.get("num_episodes", "N/A")))
    return overview_table


def create_action_info_table(action_specs: ActionSpecs) -> Table:
    """Create action information table.

    Args:
        action_specs: Action specifications.

    Returns:
        Table: Formatted action information table.
    """
    action_table: Table = Table(title="Action Information", box=box.ROUNDED)
    action_table.add_column("Property", style="bold yellow")
    action_table.add_column("Value", style="white")

    action_table.add_row("Total actions", str(action_specs.get("total_actions", "N/A")))
    action_table.add_row("Unique actions", str(len(action_specs.get("unique_actions", []))))
    action_table.add_row("Action distribution", str(action_specs.get("action_counts", "N/A")))
    action_table.add_row("Average actions per episode", f"{action_specs.get('avg_actions_per_episode', 0):.2f}")
    action_table.add_row("Min actions per episode", str(action_specs.get("min_actions_per_episode", "N/A")))
    action_table.add_row("Max actions per episode", str(action_specs.get("max_actions_per_episode", "N/A")))

    return action_table


def create_state_info_table(state_specs: StateSpecs) -> Table:
    """Create state information table.

    Args:
        state_specs: State specifications.

    Returns:
        Table: Formatted state information table.
    """
    state_table: Table = Table(title="State Information", box=box.ROUNDED)
    state_table.add_column("Property", style="bold magenta")
    state_table.add_column("Value", style="white")
    dataset_kind: str | float | list[str] | None = _file_metadata.get("dataset_kind")

    if state_specs.get("has_variations", False):
        state_table.add_row("Has variations", "Yes")

        # Add information about variation scope
        # Special-case search pair data: show start/goal semantics if available
        variant_scope_label: str | None = None
        if isinstance(dataset_kind, str) and dataset_kind.startswith("search_pairs"):
            sides = str(_file_metadata.get("variant_sides", "none"))
            if sides == "both":
                variant_scope_label = "First state and goal state"
            elif sides == "start":
                variant_scope_label = "First state only"
            elif sides == "goal":
                variant_scope_label = "Goal state only"
            else:
                variant_scope_label = "None"
        else:
            first_state_only = state_specs.get("first_state_only", False)
            variant_scope_label = "First state only" if first_state_only else "All states"
        state_table.add_row("Variation scope", variant_scope_label)

        # Show variation mapping if available
        if "variation_categories" in _file_metadata:
            categories: MetadataEntry = _file_metadata["variation_categories"]
            state_table.add_row("Variation categories", str(categories))

        # Different outputs based on variation format
        variation_format: Literal["list_per_state", "dictionary", "list", "unknown"] = state_specs.get(
            "variation_format", "unknown"
        )
        state_table.add_row("Variation format", variation_format)

        # Check if "clean" is included in the variations
        if "variation_categories" in _file_metadata:
            categories_val: MetadataEntry = _file_metadata["variation_categories"]
            if isinstance(categories_val, list):
                categories_list: list[str] = list(categories_val)
                if "clean" in categories_list:
                    clean_idx: int = categories_list.index("clean")
                    state_table.add_row("Ground truth", f"Available as 'clean' variation (index {clean_idx})")

        if variation_format == "dictionary":
            state_table.add_row("Variation types", str(state_specs.get("variation_types", "N/A")))
        elif variation_format == "list":
            state_table.add_row("Number of variations", str(state_specs.get("num_variations", "N/A")))

        # If this is a search-pairs dataset, show per-side variation names
        if isinstance(dataset_kind, str) and dataset_kind.startswith("search_pairs"):
            vn_start: MetadataEntry | None = _file_metadata.get("variant_names_start")
            vn_goal: MetadataEntry | None = _file_metadata.get("variant_names_goal")
            if isinstance(vn_start, list):
                state_table.add_row("Start variations", ", ".join(vn_start) if vn_start else "None")
            if isinstance(vn_goal, list):
                state_table.add_row("Goal variations", ", ".join(vn_goal) if vn_goal else "None")

    else:
        state_table.add_row("Has variations", "No")

    # Print state shape and data type information
    if "shape" in state_specs:
        state_table.add_row("State shape", str(state_specs["shape"]))
    elif "frame_shape" in state_specs:
        state_table.add_row("Sequence length", str(state_specs.get("sequence_length", "N/A")))
        state_table.add_row("Frame shape", str(state_specs.get("frame_shape", "N/A")))

    if "dtype" in state_specs:
        state_table.add_row("Data type", state_specs["dtype"])

    return state_table


def create_variation_details_table(state_specs: StateSpecs) -> Table | None:
    """Create variation details table if applicable.

    Args:
        state_specs: State specifications.

    Returns:
        Table | None: Formatted variation details table or None if not applicable.
    """
    if not state_specs.get("has_variations", False):
        return None

    variation_format: Literal["list_per_state", "dictionary", "list", "unknown"] = state_specs.get(
        "variation_format", "unknown"
    )

    if variation_format == "dictionary":
        var_details_table = Table(title="Variation Details", box=box.ROUNDED)
        var_details_table.add_column("Variation", style="bold red")
        var_details_table.add_column("Shape", style="white")
        var_details_table.add_column("Data Type", style="cyan")
        variation_shapes: VariationShapesType = state_specs.get("variation_shapes", {})
        variation_dtypes: VariationDtypesType = state_specs.get("variation_dtypes", {})
        for var_name in state_specs.get("variation_types", []):
            if var_name == "base":
                continue
            shape: tuple[int, ...] | str = variation_shapes.get(var_name, "N/A")
            dtype: str = variation_dtypes.get(var_name, "N/A")
            var_details_table.add_row(var_name, str(shape), dtype)
        return var_details_table

    if variation_format == "list" and "variation_details" in state_specs:
        var_details_table = Table(title="Variation Details", box=box.ROUNDED)
        var_details_table.add_column("Name", style="bold red")
        var_details_table.add_column("Type", style="yellow")
        var_details_table.add_column("Shape", style="white")
        var_details_table.add_column("Data Type", style="cyan")

        for var_detail in state_specs["variation_details"]:
            name: str = var_detail.get("name", "Unknown")
            var_type: str = var_detail.get("type", "Unknown")
            shape = var_detail.get("shape", "N/A")
            dtype = var_detail.get("dtype", "N/A")
            var_details_table.add_row(name, var_type, str(shape), dtype)

        return var_details_table

    return None


def create_state_statistics_table(state_specs: StateSpecs) -> Table | None:
    """Create state statistics table if data is available.

    Args:
        state_specs: State specifications.

    Returns:
        Table | None: Formatted state statistics table or None if not available.
    """
    if not all(key in state_specs for key in ["min_value", "max_value", "mean_value", "std_value"]):
        return None

    stats_table: Table = Table(title="State Statistics", box=box.ROUNDED)
    stats_table.add_column("Statistic", style="bold blue")
    stats_table.add_column("Value", style="white")

    stats_table.add_row("Min value", f"{state_specs.get('min_value', 0.0):.6f}")
    stats_table.add_row("Max value", f"{state_specs.get('max_value', 0.0):.6f}")
    stats_table.add_row("Mean value", f"{state_specs.get('mean_value', 0.0):.6f}")
    stats_table.add_row("Standard deviation", f"{state_specs.get('std_value', 0.0):.6f}")

    return stats_table


def print_data_report(file_path: str, specs: Specs) -> None:
    """Print a detailed report about the data file.

    Args:
        file_path (str): Path to the data file.
        specs: Data specifications.
    """
    file_size: int = pathlib.Path(file_path).stat().st_size
    file_size_mb: float = file_size / (1024 * 1024)

    console.print()
    console.print(Panel.fit(f"📄 DATA FILE REPORT: {pathlib.Path(file_path).name}", style="bold blue"))

    # File Information Panel
    compression_status: str = get_file_format_info(file_path)
    file_table: Table = create_file_info_table(file_path, file_size_mb, compression_status)

    # Show metadata info if available
    if _file_metadata:
        add_metadata_rows(file_table, _file_metadata)

    console.print(file_table)

    # Dataset Overview
    overview_table: Table = create_dataset_overview_table(specs)
    console.print(overview_table)

    # Action Information
    action_table: Table = create_action_info_table(specs.get("action_specs", {}))
    console.print(action_table)

    # State Information
    state_table: Table = create_state_info_table(specs.get("state_specs", {}))
    console.print(state_table)

    # Detailed Variation Information (if applicable)
    variation_details_table: Table | None = create_variation_details_table(specs.get("state_specs", {}))
    if variation_details_table:
        console.print(variation_details_table)

    # State Statistics (if available)
    stats_table: Table | None = create_state_statistics_table(specs.get("state_specs", {}))
    if stats_table:
        console.print(stats_table)

    console.print()


def get_variation_info(var_data: ImageArray | list[ImageArray]) -> tuple[str, str, str, str]:
    """Extract variation information for table display.

    Args:
        var_data: Variation data

    Returns:
        Tuple of (count, shape, dtype, min_max) strings
    """
    # Handle array and list inputs separately.
    if isinstance(var_data, np.ndarray):
        count = "1"
        shape = str(var_data.shape)
        dtype = str(var_data.dtype)
        try:
            min_max = f"{np.min(var_data):.3f}/{np.max(var_data):.3f}"
        except Exception:
            min_max = "N/A"
        return count, shape, dtype, min_max

    # Treat as list of ImageArray (per function signature)
    count = str(len(var_data))
    if len(var_data) == 0:
        return count, "Empty", "N/A", "N/A"
    first_item = var_data[0]
    shape = str(first_item.shape)
    dtype = str(first_item.dtype)
    try:
        min_max = f"{np.min(first_item):.3f}/{np.max(first_item):.3f}"
    except Exception:
        min_max = "N/A"
    return count, shape, dtype, min_max


def display_variation_plot(var_data: ImageArray, title: str, save_path: str | None = None) -> None:
    """Display or save one variation image.

    Args:
        var_data: Image data to display.
        title: Plot title.
        save_path: Path to save the plot to. If None, shows the plot instead.
    """
    # Apply screen size constraints to prevent oversized windows
    constrained_width, constrained_height = constrain_figure_size(5, 5, margin_factor=0.5)
    _fig, ax = plt.subplots(figsize=(constrained_width, constrained_height), constrained_layout=True)
    display_img: ImageArray = to_display_image(var_data, step_idx=0)
    ax.imshow(display_img, cmap="viridis")

    # Apply text wrapping and styling
    style_subplot(ax, title, is_variation=True)
    show_or_save_plot(save_path)


def handle_dictionary_variations(variations: VariationDict, detailed: bool, save_path: str | None = None) -> None:
    """Handle dictionary-based variations display.

    Args:
        variations: Dictionary of variations
        detailed: Whether to show detailed plots
        save_path: Path to save plots to. If None, shows plots instead.
    """
    console.print("[bold green]Variation format:[/bold green] Dictionary")
    console.print(f"[bold green]Number of variation types:[/bold green] {len(variations)}")

    # Create table for variation details
    var_table = Table(title="Dictionary Variations", box=box.ROUNDED)
    var_table.add_column("Variation Name", style="bold cyan")
    var_table.add_column("Count", style="white")
    var_table.add_column("Shape", style="green")
    var_table.add_column("Data Type", style="yellow")
    var_table.add_column("Min/Max", style="magenta")

    # Sort variations alphabetically by name
    for var_name, var_data in sorted(variations.items(), key=_variation_dict_item_key):
        count, shape, dtype, min_max = get_variation_info(var_data)
        var_table.add_row(var_name, count, shape, dtype, min_max)
        # Display plot for first element if detailed and list-like with content
        if detailed and isinstance(var_data, list) and var_data:
            plot_save_path = None
            if save_path:
                name_safe: str = var_name.replace(" ", "_").replace("/", "_")
                plot_save_path = save_path.replace(".", f"_variation_{name_safe}.")
            display_variation_plot(var_data[0], f"Variation: {var_name}", plot_save_path)

    console.print(var_table)


def handle_list_variations(variations: list[PerStateVariations], detailed: bool, save_path: str | None = None) -> None:
    """Handle list-based variations display.

    Args:
        variations: List of variations
        detailed: Whether to show detailed plots
        save_path: Path to save plots to. If None, shows plots instead.
    """
    console.print("[bold green]Variation format:[/bold green] List (variations per state)")
    console.print(f"[bold green]Number of states with variations:[/bold green] {len(variations)}")

    # Find all unique variation names
    all_var_names: set[str] = {var_item[0] for state_vars in variations for var_item in state_vars}
    console.print(f"[bold green]Unique variation types found:[/bold green] {', '.join(sorted(all_var_names))}")

    # Create table for variations per state
    var_table: Table = Table(title="Variations Per State", box=box.ROUNDED)
    var_table.add_column("State Index", style="bold cyan")
    var_table.add_column("Variations Count", style="white")
    var_table.add_column("Variation Names", style="green")

    for state_idx, state_vars in enumerate(variations):
        var_count: int = len(state_vars)
        var_names: list[str] = [var_item[0] for var_item in state_vars]
        var_names_sorted: list[str] = sorted(var_names) if var_names else []
        var_names_str: str = ", ".join(var_names_sorted) if var_names_sorted else "None"
        var_table.add_row(str(state_idx), str(var_count), var_names_str)

    console.print(var_table)

    # Show detailed view if requested
    if detailed:
        # Detect search-pairs to show both start and goal when available
        dataset_kind: str | float | list[str] | None = _file_metadata.get("dataset_kind")
        is_pairs: bool = isinstance(dataset_kind, str) and dataset_kind.startswith("search_pairs")

        header: LiteralString = (
            "\n[bold blue]Detailed View - Start and Goal Variations:[/bold blue]"
            if is_pairs
            else "\n[bold blue]Detailed View - First State Variations:[/bold blue]"
        )
        console.print(header)

        # Gather which states to show (for pairs, prefer indices 0 and 1)
        states_to_show: list[int] = []
        if is_pairs and len(variations) >= 2:
            states_to_show = [0, 1]
        else:
            # Find first state that has variations
            for idx, state_vars in enumerate(variations):
                if state_vars:
                    states_to_show = [idx]
                    break

        for state_idx in states_to_show:
            if state_idx >= len(variations):
                continue
            state_vars = variations[state_idx]
            if not state_vars:
                continue

            console.print(f"\n[bold cyan]State {state_idx} variations:[/bold cyan]")

            # Sort variations alphabetically by name before displaying
            sorted_variations: list[tuple[str, ImageArray]] = sorted(
                [(v[0], v[1]) for v in state_vars], key=_variation_entry_key
            )

            for var_name, var_data in sorted_variations:
                console.print(f"  {var_name}: shape={var_data.shape}, dtype={var_data.dtype}")
                plot_save_path: str | None = None
                if save_path:
                    # Create unique filename for each variation
                    name_safe: str = var_name.replace(" ", "_").replace("/", "_")
                    plot_save_path = save_path.replace(".", f"_state{state_idx}_{name_safe}.")
                display_variation_plot(var_data, f"State {state_idx} - {var_name}", plot_save_path)


def examine_variations(
    state_img_trajs: list[StateImgTraj], sample_idx: int = 0, detailed: bool = False, save_path: str | None = None
) -> None:
    """Examine and display details about variations in the data.

    Args:
        state_img_trajs (list[StateImgTraj]): State image trajectories.
        sample_idx (int): Index of sample to examine.
        detailed (bool): True to display plots.
        save_path: Path to save plots to. If None, shows plots instead.

    """
    if sample_idx >= len(state_img_trajs):
        console.print(f"[red]Sample index {sample_idx} out of range (max: {len(state_img_trajs) - 1})[/red]")
        return

    sample = state_img_trajs[sample_idx]

    if not (isinstance(sample, tuple) and len(sample) == 2):
        console.print("[yellow]This sample does not contain variations[/yellow]")
        return

    # The caller supplies StateWithVariations here, so extract its variation mapping.
    _, variations = sample

    console.print()
    console.print(Panel.fit(f"🔍 VARIATION DETAILS (Sample {sample_idx})", style="bold blue"))

    if isinstance(variations, dict):
        handle_dictionary_variations(variations, detailed, save_path)
    else:  # variations is list
        handle_list_variations(variations, detailed, save_path)
    # No other variation types per VariationType

    console.print()


def setup_plotting_style() -> None:
    """Set up matplotlib plotting style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 12,
        "figure.dpi": 150,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })


def calculate_grid_layout(n_variants: int) -> tuple[int, int, float, float]:
    """Arrange variants in a near-square grid within fixed size limits.

    Args:
        n_variants: Number of variants to display.

    Returns:
        Row count, column count, figure width, and figure height.
    """
    # A square grid minimizes unused cells for the supported figure sizes.
    ncols: int = math.ceil(math.sqrt(n_variants))
    nrows: int = math.ceil(n_variants / ncols)

    # Allocate three inches in each dimension per image.
    base_width_per_image = 3
    base_height_per_image = 3

    fig_width: int = ncols * base_width_per_image
    fig_height: int = nrows * base_height_per_image

    # Bound the output to 18 by 12 inches.
    fig_width = min(fig_width, 18)
    fig_height = min(fig_height, 12)

    return nrows, ncols, fig_width, fig_height


def wrap_text_for_labels(text: str, max_width: int = 15) -> str:
    """Wrap a label at a fixed character width.

    Args:
        text: Text to wrap.
        max_width: Maximum characters per line.

    Returns:
        Text with newline separators inserted between wrapped lines.
    """
    # Break long tokens when a label contains no natural wrap point.
    wrapped: list[str] = textwrap.wrap(text, width=max_width, break_long_words=True)
    return "\n".join(wrapped)


def calculate_dynamic_font_size(text: str, base_font: int = 5, max_length: int = 20) -> int:
    """Reduce a label's font size when its text exceeds a length threshold.

    Args:
        text: Text to size.
        base_font: Base font size.
        max_length: Text length threshold for scaling.

    Returns:
        Font size clamped by the function's configured lower bound.
    """
    if not text:
        return base_font

    text_length: int = len(text)
    if text_length <= max_length:
        return base_font

    # Scale by the ratio between the threshold and the observed length.
    scale_factor: float = max_length / text_length
    return max(8, min(base_font, int(base_font * scale_factor)))


def adjust_figure_size_for_labels(base_width: float, labels: list[str]) -> float:
    """Increase figure width when the longest label exceeds 20 characters.

    Args:
        base_width: Base figure width.
        labels: Labels that will appear in the figure.

    Returns:
        Figure width after applying the label-length multiplier.
    """
    if not labels:
        return base_width

    max_label_len: int = max(len(label) for label in labels)
    if max_label_len > 20:
        # Add 0.05 inches for every character beyond the threshold.
        extra_width: float = (max_label_len - 20) * 0.05  # 0.05 inches per extra character
        return base_width + extra_width

    return base_width


def style_subplot(ax: Axes, title: str, is_variation: bool = True) -> None:
    """Hide axes and add a wrapped title to one image subplot.

    Args:
        ax: Matplotlib axes object.
        title: Title for the subplot.
        is_variation: Whether to expand a variation identifier into title case.
    """
    ax.axis("off")

    if is_variation:
        # Format variation name and apply text wrapping
        title = title.replace("_", " ").title()

    # Apply text wrapping for non-variation titles too
    wrapped_title: str = wrap_text_for_labels(title, max_width=40)
    font_size: int = calculate_dynamic_font_size(title, base_font=5, max_length=40)

    ax.set_title(
        wrapped_title,
        fontsize=font_size,
        pad=5,
        fontweight="medium",
        bbox={"facecolor": "aliceblue", "edgecolor": "steelblue", "alpha": 0.8, "boxstyle": "round,pad=0.3"},
    )

    # Add subtle border and frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("steelblue")
        spine.set_linewidth(1)

    ax.patch.set_edgecolor("white")
    ax.patch.set_linewidth(2)


def visualize_variations(
    sample: StateWithVariations, sample_idx: int, actions: ActionTraj, save_path: str | None = None
) -> None:
    """Handle visualization of samples with variations using matplotlib practices.

    Args:
        sample: Tuple of (clean_states, variations)
        sample_idx: Sample index
        actions: Action list for this sample
        save_path: Path to save the plot to. If None, shows the plot instead.
    """
    # sample is guaranteed to be a (clean_states, variations) tuple by type
    clean_states, variations_per_state = sample

    # Determine step to visualize
    num_steps: int
    if isinstance(clean_states, np.ndarray) and clean_states.ndim == 4:
        num_steps = clean_states.shape[0]
    elif isinstance(clean_states, list):
        num_steps = len(clean_states)
    else:
        num_steps = 1

    step_idx: int = min(num_steps // 2, num_steps - 1)
    action_str: str = f"{actions[step_idx]}" if step_idx < len(actions) else "N/A"

    # Extract variations for the selected step
    step_variations: PerStateVariations = []
    if isinstance(variations_per_state, list) and step_idx < len(variations_per_state):
        step_variations = variations_per_state[step_idx]

    n_variants: int = len(step_variations) + 1  # +1 for clean state

    # Calculate grid layout
    nrows, ncols, fig_width, fig_height = calculate_grid_layout(n_variants)

    # Sort variations alphabetically by variation name
    step_variations = sorted(step_variations, key=_variation_entry_key)

    # Collect all labels for dynamic sizing
    all_labels: list[str] = ["Original (Clean)"]
    all_labels.extend([var_name for var_name, _ in step_variations])

    # Extend the figure when a label is longer than 20 characters.
    adjusted_width: float = adjust_figure_size_for_labels(fig_width, all_labels)

    # Create figure without constrained_layout and manually adjust for spacing
    # Apply screen size constraints to prevent oversized windows
    constrained_width, constrained_height = constrain_figure_size(adjusted_width, fig_height)
    fig: Figure = plt.figure(
        figsize=(constrained_width, constrained_height), constrained_layout=False, tight_layout=True
    )
    # Manually leave space for title and caption
    fig.subplots_adjust(top=0.5, bottom=0.2)

    # Use GridSpec for layout with uniform padding
    # Constrain subplots between title (bottom at ~0.96) and caption (top at ~0.08)
    # More aggressive spacing to prevent any overlap
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.2, wspace=0.2, top=0.88, bottom=0.1)  # More conservative spacing
    grid: list[Axes] = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(nrows * ncols)]

    # Add figure titles
    title: str = f"Visualization of Sample {sample_idx}, Step {step_idx}"
    subtitle: str = f"Action: {action_str}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # Process and display clean state
    clean_img: ImageArray = to_display_image(clean_states, step_idx=step_idx)
    ax: Axes = grid[0]
    ax.imshow(clean_img)
    style_subplot(ax, "Original (Clean)", is_variation=False)

    # Process variations for the selected step
    for i, (var_name, var_arr) in enumerate(step_variations):
        if i + 1 < len(grid):
            img: ImageArray = to_display_image(var_arr, step_idx=0)  # Single variation image
            ax = grid[i + 1]
            ax.imshow(img)
            style_subplot(ax, var_name, is_variation=True)

    # Hide cells left over after the final variant.
    for ax in grid[len(step_variations) + 1 :]:
        ax.axis("off")

    # Add subtitle with action information (positioned below title, above subplots)
    fig.text(
        0.5,
        0.92,
        subtitle,
        ha="center",
        fontsize=11,
        bbox={"facecolor": "lavender", "alpha": 0.8, "edgecolor": "gray", "boxstyle": "round,pad=0.5"},
    )

    # Add caption
    fig.text(
        0.5,
        0.02,
        f"Sample {sample_idx} from dataset with {len(step_variations)} variations",
        ha="center",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "lightgray"},
    )

    show_or_save_plot(save_path)


def visualize_trajectory_array(
    arr: ImageArray, sample_idx: int, actions: ActionTraj, max_frames: int, save_path: str | None = None
) -> None:
    """Visualize numpy array trajectory following matplotlib practices.

    Args:
        arr: Numpy array to visualize
        sample_idx: Sample index
        actions: Action list
        max_frames: Maximum frames to display
        save_path: Path to save the plot to. If None, shows the plot instead.
    """
    if arr.ndim == 4:  # (T, C, H, W)
        num_frames: int = min(arr.shape[0], max_frames)

        # Fit the selected frames into a near-square grid.
        nrows, ncols, fig_width, fig_height = calculate_grid_layout(num_frames)

        # Create labels for dynamic sizing
        labels: list[str] = []
        for i in range(num_frames):
            action_str = f"Action: {actions[i]}" if i < len(actions) else "No Action"
            labels.append(f"Step {i} {action_str}")

        # Adjust figure size for labels
        adjusted_width: float = adjust_figure_size_for_labels(fig_width, labels)

        # Create figure with constrained layout
        # Apply screen size constraints to prevent oversized windows
        constrained_width, constrained_height = constrain_figure_size(adjusted_width, fig_height)
        fig: Figure = plt.figure(figsize=(constrained_width, constrained_height), constrained_layout=True)

        # Use GridSpec for layout
        # Constrain subplots between title and potential caption
        # More aggressive spacing to prevent overlap
        gs = GridSpec(
            nrows, ncols, figure=fig, hspace=0.3, wspace=0.2, top=0.85, bottom=0.15
        )  # More conservative spacing
        grid: list[Axes] = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(nrows * ncols)]

        fig.suptitle(f"Sample {sample_idx} Trajectory", fontsize=10, fontweight="bold")

        for i in range(num_frames):
            img: ImageArray = to_display_image(arr, step_idx=i)
            ax: Axes = grid[i]
            ax.imshow(img)

            title: str = f"Step {i}\nAction: {actions[i]}" if i < len(actions) else f"Step {i}"
            style_subplot(ax, title, is_variation=False)

        # Disable unused axes
        for ax in grid[num_frames:]:
            ax.axis("off")

        if arr.shape[0] > max_frames:
            fig.text(
                0.5,
                0.02,
                f"Showing {max_frames} of {arr.shape[0]} total frames",
                ha="center",
                fontsize=10,
                bbox={"facecolor": "lightyellow", "alpha": 0.5, "pad": 5},
            )
        show_or_save_plot(save_path)

    elif arr.ndim == 3:
        img = to_display_image(arr)
        # Apply screen size constraints to prevent oversized windows
        constrained_width, constrained_height = constrain_figure_size(6, 6)
        fig, ax = plt.subplots(figsize=(constrained_width, constrained_height), constrained_layout=True)
        ax.imshow(img)
        action_str = str(actions[0]) if len(actions) > 0 else "N/A"
        title = f"Sample {sample_idx}\nAction: {action_str}"
        style_subplot(ax, title, is_variation=False)
        show_or_save_plot(save_path)
    else:
        console.print(f"[red]❌ Unsupported image shape:[/red] {arr.shape}")


def visualize_trajectory_list(
    states_list: list[ImageArray], sample_idx: int, actions: ActionTraj, max_frames: int, save_path: str | None = None
) -> None:
    """Visualize list-based trajectory following matplotlib practices.

    Args:
        states_list: List of states
        sample_idx: Sample index
        actions: Action list
        max_frames: Maximum frames to display
        save_path: Path to save the plot to. If None, shows the plot instead.
    """
    num_frames: int = min(len(states_list), max_frames)

    # Fit the selected frames into a near-square grid.
    nrows, ncols, fig_width, fig_height = calculate_grid_layout(num_frames)

    # Create labels for dynamic sizing
    labels: list[str] = []
    for i in range(num_frames):
        action_str = f"Action: {actions[i]}" if i < len(actions) else "No Action"
        labels.append(f"Step {i} {action_str}")

    # Adjust figure size for labels
    adjusted_width: float = adjust_figure_size_for_labels(fig_width, labels)

    # Create figure with constrained layout
    # Apply screen size constraints to prevent oversized windows
    constrained_width, constrained_height = constrain_figure_size(adjusted_width, fig_height)
    fig: Figure = plt.figure(figsize=(constrained_width, constrained_height), constrained_layout=True)

    # Use GridSpec for layout
    # Constrain subplots between title and potential caption
    # More aggressive spacing to prevent overlap
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.4, wspace=0.3, top=0.85, bottom=0.15)  # More conservative spacing
    grid: list[Axes] = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(nrows * ncols)]

    fig.suptitle(f"Sample {sample_idx} Trajectory", fontsize=14, fontweight="bold")

    for i in range(num_frames):
        arr = states_list[i]
        img: ImageArray = to_display_image(arr, step_idx=0)

        ax: Axes = grid[i]
        ax.imshow(img)
        title: str = f"Step {i}\nAction: {actions[i]}" if i < len(actions) else f"Step {i}"
        style_subplot(ax, title, is_variation=False)

    # Disable unused axes
    for ax in grid[num_frames:]:
        ax.axis("off")

    if len(states_list) > max_frames:
        fig.text(
            0.5,
            0.02,
            f"Showing {max_frames} of {len(states_list)} total frames",
            ha="center",
            fontsize=10,
            bbox={"facecolor": "lightyellow", "alpha": 0.5, "pad": 5},
        )

    show_or_save_plot(save_path)


def visualize_sample(
    state_img_trajs: list[StateImgTraj],
    action_trajs: list[ActionTraj],
    sample_idx: int = 0,
    max_frames: int = 5,
    save_path: str | None = None,
) -> None:
    """Visualize a sample trajectory with styling.

    Args:
        state_img_trajs (list[StateImgTraj]): State image trajectories.
        action_trajs (list[ActionTraj]): Action trajectories.
        sample_idx (int): Index of sample to visualize.
        max_frames (int): Max frames to display.
        save_path (str | None): Path to save visualization instead of showing.

    """
    if sample_idx >= len(state_img_trajs):
        console.print(f"[red]Sample index {sample_idx} out of range (max: {len(state_img_trajs) - 1})[/red]")
        return

    setup_plotting_style()

    sample: StateImgTraj = state_img_trajs[sample_idx]
    actions: ActionTraj = action_trajs[sample_idx]

    # Check if sample has variations
    has_tuple_structure: bool = isinstance(sample, tuple) and len(sample) == 2
    has_variations_list: bool = has_tuple_structure and isinstance(sample[1], list) and len(sample[1]) > 0
    has_nested_variations: bool = False
    if has_variations_list and isinstance(sample, tuple):
        variations_candidate: VariationType = sample[1]
        if isinstance(variations_candidate, list):
            variations_list: list[PerStateVariations] = variations_candidate
            has_nested_variations = bool(variations_list and variations_list[0])

    if has_nested_variations and isinstance(sample, tuple):
        visualize_variations(sample, sample_idx, actions, save_path)
        return

    # Extract clean states
    clean_states: StateFrames = sample[0] if isinstance(sample, tuple) and len(sample) == 2 else sample

    # Visualize based on data type
    if isinstance(clean_states, np.ndarray):
        visualize_trajectory_array(clean_states, sample_idx, actions, max_frames, save_path)
    elif clean_states:  # list of frames
        visualize_trajectory_list(clean_states, sample_idx, actions, max_frames, save_path)


def process_state_image_for_timeline(state_img: StateFrames) -> ImageArray:
    """Convert state image to RGB format for timeline display using unified converter."""
    try:
        img: ImageArray = to_display_image(state_img, step_idx=0)
        if img.ndim == 2:
            img = _normalize_image_array(np.stack([img] * 3, axis=-1))
        elif img.ndim == 3 and img.shape[-1] > 3:
            img = img[..., :3]
        return img.astype(np.uint8)
    except Exception:
        h, w = 32, 32
        return np.zeros((h, w, 3), dtype=np.uint8)


def create_action_image(action: ActionType, dimensions: tuple[int, int]) -> NDArray[np.uint8]:
    """Create an action image with text overlay.

    Args:
        action: Action value to display.
        dimensions: Tuple of (height, width) for the image.

    Returns:
        Action image with text overlay.
    """
    h, w = dimensions
    action_img: NDArray[np.uint8] = np.zeros((h, w, 3), dtype=np.uint8)

    # Scale text size based on image dimensions
    font_scale: float = min(h, w) / 100 * 0.7  # Scale font based on image size
    text_x: int = int(w * 0.15)  # Position text horizontally at 15% of width
    text_y: int = int(h * 0.6)  # Position text vertically at 60% of height

    cv2.putText(action_img, str(action), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    return action_img


def process_episode_for_timeline(
    episode_states: StateImgTraj, episode_actions: ActionTraj, max_steps: int
) -> list[ImageArray]:
    """Process a single episode for timeline display.

    Args:
        episode_states: Episode state data.
        episode_actions: Episode action data.
        max_steps: Maximum number of steps to include.

    Returns:
        List of processed images for the episode.
    """
    # Handle episodes with variations
    if isinstance(episode_states, tuple) and len(episode_states) == 2:
        clean_states, _variations_per_state = episode_states
        # Timeline videos use the unmodified state sequence.
        episode_states = clean_states

    # Limit the number of steps to max_steps
    limited_states: StateFrames = episode_states[:max_steps]
    limited_actions: ActionTraj = episode_actions[:max_steps]

    # Create a grid of images for this episode
    episode_images: list[ImageArray] = []

    # Add state images
    for state_img in limited_states:
        processed_img: ImageArray = process_state_image_for_timeline(state_img)
        episode_images.append(processed_img)

    # Add action images (as text)
    for action in limited_actions:
        # Create an image with the action text using dimensions from state images
        # If episode_images is empty, use default dimensions
        if episode_images:
            h, w = episode_images[0].shape[0], episode_images[0].shape[1]
        else:
            h, w = 32, 32

        action_img: NDArray[np.uint8] = create_action_image(action, (h, w))
        episode_images.append(action_img)

    return episode_images


class TimelineGridAnimator:
    """Animator for the grid-style timeline view (no nested update function)."""

    __slots__: tuple[str, ...] = ("all_state_images", "ax")

    def __init__(self, all_state_images: list[list[ImageArray]], ax: Axes) -> None:
        self.all_state_images: list[list[ImageArray]] = all_state_images
        self.ax: Axes = ax

    def update(self, frame: int) -> list[AxesImage]:
        """Update the timeline grid for a frame.

        Parameters:
            frame (int): Current frame index.

        Returns:
            list[AxesImage]: Rendered image artists for this frame.
        """
        ax: Axes = self.ax
        ax.clear()
        ax.set_title("Timeline View of State and Action Changes")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Episode Index")
        artists: list[AxesImage] = []
        for episode_idx, episode_images in enumerate(self.all_state_images):
            if frame < len(episode_images):
                img = episode_images[frame]
                im: AxesImage = ax.imshow(img, aspect="auto", extent=(frame, frame + 1, episode_idx, episode_idx + 1))
                artists.append(im)
        return artists


def create_timeline_view(
    state_img_trajs: list[StateImgTraj],
    action_trajs: list[ActionTraj],
    output_file: str = "timeline_view.mp4",
    max_steps: int = 100,
    fps: int = 10,
) -> None:
    """Create a timeline video of state and action changes over time.

    Args:
        state_img_trajs: State image trajectories.
        action_trajs (list[ActionTraj]): Action trajectories.
        output_file (str): Output video file name.
        max_steps (int): Maximum number of steps to include in the video.
        fps (int): Frames per second for the video.

    """
    # Determine the number of episodes
    num_episodes: int = len(state_img_trajs)

    if num_episodes == 0:
        console.print("[red]No episodes found in the data.[/red]")
        return

    # Create a figure for the timeline view
    # Apply screen size constraints to prevent oversized windows
    constrained_width, constrained_height = constrain_figure_size(10, 6)
    fig, ax = plt.subplots(figsize=(constrained_width, constrained_height))
    plt.title("Timeline View of State and Action Changes")
    plt.xlabel("Time Steps")
    plt.ylabel("Episode Index")

    # Prepare data for plotting
    all_state_images: list[list[ImageArray]] = []
    for _episode_idx, (episode_states, episode_actions) in enumerate(zip(state_img_trajs, action_trajs, strict=False)):
        episode_images: list[ImageArray] = process_episode_for_timeline(episode_states, episode_actions, max_steps)
        all_state_images.append(episode_images)

    # Create an animation using animator class (no nested functions)
    grid_animator = TimelineGridAnimator(all_state_images, ax)
    ani = animation.FuncAnimation(fig, grid_animator.update, frames=max_steps, repeat=False)

    # Save the animation as a video file
    ani.save(output_file, writer="ffmpeg", fps=fps)

    plt.close(fig)
    console.print(f"✅ Timeline video saved as: [bold green]{output_file}[/bold green]")


def prepare_trajectory_data(
    sample: StateImgTraj, _actions: ActionTraj, variation_labels: list[str] | None
) -> tuple[StateFrames | list[ImageArray] | None, bool, VariationType | None]:
    """Prepare and validate trajectory data."""
    # Detect clean states vs. variations
    clean_states: StateFrames
    has_variations: bool
    variations: VariationType | None
    variations_per_state: VariationType
    if isinstance(sample, tuple) and len(sample) == 2:
        clean_states, variations_per_state = sample
        has_variations = True

        # Relabel variations if labels provided
        if variation_labels and isinstance(variations_per_state, list):
            new_vars_per_state: list[list[tuple[str, ImageArray]]] = []
            for state_vars in variations_per_state:
                new_state_vars: list[tuple[str, ImageArray]] = []
                for i, (name, data) in enumerate(state_vars):
                    label: str = variation_labels[i] if i < len(variation_labels) else name
                    new_state_vars.append((label, data))
                new_vars_per_state.append(new_state_vars)
            variations_per_state = new_vars_per_state

        variations = variations_per_state
    else:
        # No variations case
        clean_states = sample
        has_variations = False
        variations = None

    # Validate sequence data for timeline
    is_ndarray: bool = isinstance(clean_states, np.ndarray)
    is_list: bool = isinstance(clean_states, list)
    is_valid: bool = getattr(clean_states, "ndim", 0) > 2 if is_ndarray else is_list
    if not is_valid:
        error_msg: LiteralString = (
            "[red]❌ Timeline view requires image sequence data.[/red]"
            if is_ndarray
            else "[red]❌ Timeline view requires sequence data.[/red]"
        )
        console.print(error_msg)
        return None, False, None

    # If a single 3D array, wrap or split into steps
    if isinstance(clean_states, np.ndarray) and clean_states.ndim == 3:
        clean_states = [clean_states] if clean_states.shape[0] <= 3 else list(clean_states)

    return clean_states, has_variations, variations


def setup_figure_and_axes(
    actions: ActionTraj, has_variations: bool, variations: VariationType | None
) -> tuple[Figure, Axes, Axes, Axes | None, Axes, ImageArray | None]:
    """Set up the matplotlib figure and axes for animation."""
    # Set up plotting style
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 150,
    })

    # Constrained layout keeps the animation panels inside the screen-size cap.
    constrained_width, constrained_height = constrain_figure_size(12, 8)
    fig: Figure = plt.figure(figsize=(constrained_width, constrained_height), constrained_layout=True)
    # Note: wspace and hspace are ignored when using constrained_layout
    # Constrain subplots between title and caption with more aggressive spacing
    gs = GridSpec(3, 3, figure=fig, top=0.85, bottom=0.15)

    # First plot: Main image display
    ax_main: Axes = fig.add_subplot(gs[0:2, 0:2])
    ax_main.set_title("State Trajectory")
    ax_main.axis("off")

    # Second plot: Action history
    ax_action: Axes = fig.add_subplot(gs[2, 0:2])
    ax_action.set_title("Action History")
    ax_action.set_xlim(-0.5, min(20, len(actions)) - 0.5)
    if actions:
        action_vals: list[float] = actions_to_floats(actions)
        if action_vals:
            ax_action.set_ylim(min(action_vals) - 0.5, max(action_vals) + 0.5)

    else:
        ax_action.set_ylim(-0.5, 0.5)

    ax_action.set_xlabel("Time Step")
    ax_action.set_ylabel("Action")
    ax_action.set_facecolor("#f5f5f5")
    ax_action.grid(True, linestyle="--", alpha=0.7)

    # Third plot area: Variation display (if available)
    var_data: ImageArray | None = None
    ax_var: Axes | None = None
    if has_variations:
        ax_var = fig.add_subplot(gs[0:2, 2])
        ax_var.set_title("Variations")
        ax_var.axis("off")

    # Get first variation if available
    if isinstance(variations, list) and variations:
        first_state_variations = variations[0]
        if first_state_variations:
            var_name, var_data = first_state_variations[0]
            if ax_var is not None:
                ax_var.set_title(f"Variation: {var_name}")
    elif isinstance(variations, dict) and variations:
        first_key: str = next(iter(variations.keys()))
        first_list: StateFrames = variations[first_key]
        if first_list:
            var_data = first_list[0]
            if ax_var is not None:
                ax_var.set_title(f"Variation: {first_key}")

    else:
        ax_var = None

    # Statistics and info plot
    ax_info: Axes = fig.add_subplot(gs[2, 2])
    ax_info.set_title("Statistics")
    ax_info.axis("off")
    ax_info.set_facecolor("#f8f8f8")
    for spine in ax_info.spines.values():
        spine.set_visible(True)
        spine.set_color("#cccccc")
        spine.set_linewidth(0.5)

    return fig, ax_main, ax_action, ax_var, ax_info, var_data


def calculate_frame_statistics(state: ImageArray | list[ImageArray] | None, frame: int, total_frames: int) -> str:
    """Calculate and format statistics for the current frame."""
    if frame < total_frames:
        info_str: str = f"Frame: {frame}/{total_frames - 1}\n"
        if isinstance(state, np.ndarray):
            info_str += f"Shape: {state.shape}\n"
            info_str += f"Min: {np.min(state):.4f}\n"
            info_str += f"Max: {np.max(state):.4f}\n"
            info_str += f"Mean: {np.mean(state):.4f}\n"
            info_str += f"Std: {np.std(state):.4f}"

        else:
            info_str += f"Type: {type(state).__name__}"

    else:
        info_str = "End of sequence"

    return info_str


class TimelineAnimator:
    """Encapsulates state and update logic for the trajectory timeline animation (no nested functions)."""

    __slots__: tuple[str, ...] = (
        "actions",
        "ax_action",
        "ax_info",
        "ax_main",
        "ax_var",
        "clean_states",
        "fig",
        "has_variations",
        "main_img",
        "var_img",
        "variation_index",
        "variation_name",
        "variation_warning_shown",
        "variations",
    )

    def __init__(
        self,
        clean_states: list[ImageArray] | ImageArray,
        actions: ActionTraj,
        has_variations: bool,
        variations: VariationType | None,
        variation_name: str | None,
        variation_index: int,
    ) -> None:
        fig, ax_main, ax_action, ax_var, ax_info, _ = setup_figure_and_axes(actions, has_variations, variations)
        self.clean_states: StateFrames = clean_states
        self.actions = actions
        self.has_variations: bool = has_variations
        self.variations: VariationType | None = variations
        self.variation_name: str | None = variation_name
        self.variation_index: int = variation_index
        self.variation_warning_shown = False
        self.fig: Figure = fig
        self.ax_main: Axes = ax_main
        self.ax_action: Axes = ax_action
        self.ax_var: Axes | None = ax_var
        self.ax_info: Axes = ax_info
        self.main_img: AxesImage | None = None
        self.var_img: AxesImage | None = None

    def update(self, frame: int) -> list[AxesImage | Text]:
        """Update figure artists for a single frame.

        Args:
            frame: int
                The current frame index provided by FuncAnimation.

        Returns:
            list[matplotlib.image.AxesImage | matplotlib.text.Text]: The list of updated
            artists. The caller can use them when blitting is enabled, and they remain available for potential
            future optimization even though blit=False now).
        """
        clean_states: StateFrames = self.clean_states
        actions: ActionTraj = self.actions
        has_variations: bool = self.has_variations
        variations: VariationType | None = self.variations
        ax_main: Axes = self.ax_main
        ax_action: Axes = self.ax_action
        ax_var: Axes | None = self.ax_var
        ax_info: Axes = self.ax_info

        if frame < len(clean_states):
            state: ImageArray = clean_states[frame]
            img: ImageArray = to_display_image(state, step_idx=0)
            if self.main_img is None:
                if img.ndim == 2:
                    self.main_img = ax_main.imshow(img, cmap="viridis")
                    plt.colorbar(self.main_img, ax=ax_main, orientation="vertical", fraction=0.046, pad=0.04)
                else:
                    self.main_img = ax_main.imshow(img)
            else:
                self.main_img.set_array(img)
            action_txt: str = f", Action: {actions[frame]}" if frame < len(actions) else ""
            ax_main.set_title(f"State at t={frame}{action_txt}")

        if has_variations and ax_var is not None and isinstance(variations, list):
            current_frame_variations: PerStateVariations = variations[frame] if frame < len(variations) else []
            if current_frame_variations:
                selected_variation: tuple[str, ImageArray]
                variation_idx: int = -1
                if self.variation_name:
                    found: tuple[str, ImageArray] | None = None
                    for idx, (vn, vd) in enumerate(current_frame_variations):
                        if vn == self.variation_name:
                            found = (vn, vd)
                            variation_idx = idx
                            break
                    if found is None:
                        if not self.variation_warning_shown:
                            console.print(
                                "[yellow]Warning: Variation '"
                                f"{self.variation_name}"
                                "' not found, using first available[/yellow]"
                            )
                            self.variation_warning_shown = True
                        variation_idx = 0
                        selected_variation = current_frame_variations[0]
                    else:
                        selected_variation = found
                else:
                    variation_idx = min(self.variation_index, len(current_frame_variations) - 1)
                    selected_variation = current_frame_variations[variation_idx]
                var_name, current_var_data = selected_variation
                var_display: ImageArray = to_display_image(current_var_data, step_idx=0)
                variation_info: str = (
                    f"Variation: {var_name} (idx: {variation_idx}/{len(current_frame_variations) - 1})"
                )
                if self.var_img is None:
                    self.var_img = ax_var.imshow(var_display, cmap="plasma")
                    ax_var.set_title(variation_info)
                else:
                    self.var_img.set_array(var_display)
                    ax_var.set_title(variation_info)

        ax_action.clear()
        ax_action.set_title("Action History")
        ax_action.set_xlabel("Time Step")
        ax_action.set_ylabel("Action")
        ax_action.set_facecolor("#f5f5f5")
        ax_action.grid(True, linestyle="--", alpha=0.7)
        visible_window = 20
        start_idx: int = max(0, frame - visible_window + 1)
        displayed_actions: list[ActionType] = actions[start_idx : frame + 1] if start_idx < len(actions) else []
        x_vals: list[int] = list(range(start_idx, start_idx + len(displayed_actions)))
        if displayed_actions:
            y_vals: list[float] = actions_to_floats(displayed_actions)
            ax_action.plot(x_vals, y_vals, "b-o", linewidth=1.5)
            action_vals: list[float] = actions_to_floats(actions)
            if action_vals:
                ax_action.set_ylim(min(action_vals) - 0.5, max(action_vals) + 0.5)
        else:
            ax_action.set_ylim(-0.5, 0.5)
        ax_action.set_xlim(start_idx - 0.5, start_idx + visible_window - 0.5)
        ax_action.axvline(x=frame, color="r", linestyle="-", alpha=0.7)

        ax_info.clear()
        ax_info.set_title("Statistics")
        ax_info.axis("off")
        info_str: str = calculate_frame_statistics(
            clean_states[frame] if frame < len(clean_states) else None, frame, len(clean_states)
        )
        info_text: Text = ax_info.text(
            0.1,
            0.9,
            info_str,
            transform=ax_info.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "lightgray"},
        )
        artists: list[AxesImage | Text] = []
        if self.main_img is not None:
            artists.append(self.main_img)
        artists.append(info_text)
        if self.var_img is not None:
            artists.append(self.var_img)
        return artists

    def to_animation(self, max_frames: int = 100) -> animation.Animation:
        """Create a matplotlib.animation.FuncAnimation for the stored trajectory.

        Args:
            max_frames: int, default=100
                Maximum number of frames to render (caps long trajectories for faster previews).

        Returns:
            animation.Animation: The constructed animation instance.
        """
        frames: int = min(len(self.clean_states), max_frames)
        return animation.FuncAnimation(self.fig, self.update, frames=frames, interval=200, blit=False, repeat=True)


def _save_timeline_animation(anim: animation.Animation, fig: Figure, save_path: str) -> bool:
    """Save a timeline animation and report whether the caller should return immediately."""
    # Determine writer based on file extension
    ext: str = save_path.lower().split(".")[-1]
    if ext == "mp4":
        writer = "ffmpeg"
    elif ext == "gif":
        writer = "pillow"
    elif ext == "html":
        writer = "html"
    elif ext == "pdf":
        # For PDF, save first frame as static image since animations aren't supported
        first_frame_path = save_path.replace(".pdf", "_frame0.pdf")
        fig.savefig(first_frame_path, format="pdf", dpi=150, bbox_inches="tight")
        console.print(f"📁 First frame saved as PDF to: [bold cyan]{first_frame_path}[/bold cyan]")
        console.print("[yellow]Note: PDF format saves only the first frame as animations are not supported[/yellow]")
        return True
    else:
        writer = "ffmpeg"  # Default

    anim.save(save_path, writer=writer, fps=5, dpi=150, bitrate=1800)
    console.print(f"📁 Animation saved to: [bold cyan]{save_path}[/bold cyan]")
    return False


def create_timeline_animation(
    state_img_trajs: list[StateImgTraj],
    action_trajs: list[ActionTraj],
    sample_idx: int = 0,
    variation_labels: list[str] | None = None,
    variation_index: int = 0,
    variation_name: str | None = None,
    save_path: str | None = None,
) -> animation.Animation | None:
    """Create an animated timeline view of a sample trajectory.

    Args:
        state_img_trajs: State image trajectories.
        action_trajs (list[ActionTraj]): Action trajectories.
        sample_idx (int): Index of the sample to visualize.
        variation_labels (list[str] | None): Optional list of custom labels for variations.
        variation_index (int): Index of the variation to display (default: 0 for first variation).
        variation_name (str | None): Name of the variation to display (overrides variation_index if provided).
        save_path (str | None): Path to save the animation to. If None, shows the animation instead.

    """
    if sample_idx >= len(state_img_trajs):
        console.print(f"[red]Sample index {sample_idx} out of range (max: {len(state_img_trajs) - 1})[/red]")
        return None

    # Get the selected sample and normalize actions
    sample: StateImgTraj = state_img_trajs[sample_idx]
    actions: ActionTraj = action_trajs[sample_idx] if sample_idx < len(action_trajs) else []

    # actions originate from loader as ActionTraj

    # Prepare trajectory data
    clean_states, has_variations, variations = prepare_trajectory_data(sample, actions, variation_labels)
    if clean_states is None:
        return None

    # Display variation information if variations exist
    if has_variations and isinstance(variations, list):
        # Get all unique variation names from the first frame that has variations
        available_variations: list[str] = []
        for frame_variations in variations:
            if frame_variations:
                available_variations = [var_name for var_name, _ in frame_variations]
                break

        if available_variations:
            console.print(f"[cyan]Available variations:[/cyan] {', '.join(available_variations)}")
            if variation_name and variation_name not in available_variations:
                console.print(f"[yellow]Warning: Requested variation '{variation_name}' not found![/yellow]")
                console.print(f"[yellow]Available variations: {', '.join(available_variations)}[/yellow]")
            elif variation_name:
                console.print(f"[green]Using variation:[/green] {variation_name}")
            else:
                selected_idx: int = min(variation_index, len(available_variations) - 1)
                console.print(
                    f"[green]Using variation:[/green] {available_variations[selected_idx]} (index {selected_idx})"
                )

    # Create animator (handles figure + update logic)
    animator = TimelineAnimator(
        clean_states=clean_states,
        actions=actions,
        has_variations=has_variations,
        variations=variations,
        variation_name=variation_name,
        variation_index=variation_index,
    )

    fig: Figure = animator.fig
    fig.suptitle(f"Trajectory Animation - Sample {sample_idx}", fontsize=14, fontweight="bold")
    plt.figtext(
        0.5,
        0.01,
        (
            "Total frames: "
            f"{len(clean_states)} | Actions: {len(actions)} | Has variations: "
            f"{'Yes' if has_variations else 'No'}"
        ),
        ha="center",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "lightgray"},
    )

    anim: animation.Animation = animator.to_animation(max_frames=100)

    # Handle display/save based on parameters
    if save_path:
        # Save animation to file
        try:
            if _save_timeline_animation(anim, fig, save_path):
                return anim
        except Exception as e:
            console.print(f"[red]❌ Failed to save animation: {e}[/red]")
    else:
        # Show animation
        backend: str = plt.get_backend()
        console.print(f"[dim]Using matplotlib backend: {backend}[/dim]")

        # Check if backend supports interactive display
        if backend.lower() not in {"agg", "svg", "pdf", "ps"}:
            # Interactive backend - try to show
            try:
                plt.show()
                console.print("[green]✓[/green] Timeline animation displayed")
            except Exception as e:
                # Display failed (e.g., headless environment)
                console.print(f"[yellow]⚠️ Could not display animation: {e}[/yellow]")
                console.print("[dim]Tip: Run in environment with display or save animation to file.[/dim]")
        else:
            # Non-interactive backend
            console.print("[yellow]⚠️ Non-interactive backend detected. Animation created but not displayed.[/yellow]")
            console.print(f"[dim]Backend '{backend}' doesn't support interactive display[/dim]")

    # Return animation reference to prevent garbage collection
    return anim


def calculate_action_statistics(action_trajs: list[ActionTraj]) -> ActionStatsSummary:
    """Calculate action counts and episode-length statistics.

    Args:
        action_trajs: Per-episode action sequences.

    Returns:
        Counts, distribution, and episode-length summary values.
    """
    # Flatten all actions across episodes
    all_actions: list[ActionType] = [action for traj in action_trajs for action in traj]

    if not all_actions:
        return {"unique_actions": [], "action_distribution": {}, "most_common_action": None, "total_actions": 0}

    # Count all actions in one pass.
    action_counter: Counter[ActionType] = Counter(all_actions)
    # Normalize to floats for plotting and consistency
    unique_actions: list[float] = sorted(actions_to_floats(action_counter.keys()))
    action_distribution: dict[float, int] = {float(a): c for a, c in action_counter.items()}
    most_common_action_val = action_counter.most_common(1)[0][0]
    most_common_action: float | None = float(most_common_action_val) if action_counter else None

    return {
        "unique_actions": unique_actions,
        "action_distribution": action_distribution,
        "most_common_action": most_common_action,
        "total_actions": len(all_actions),
    }


def extract_state_shape_info(specs: list[Specs]) -> list[str]:
    """Extract state shape information from specifications."""
    result: list[str] = []
    for spec in specs:
        state_specs: StateSpecs = spec.get("state_specs", {})
        if "shape" in state_specs:
            result.append(str(state_specs["shape"]))

        elif "frame_shape" in state_specs:
            result.append(str(state_specs["frame_shape"]))

        else:
            result.append("N/A")

    return result


def create_comparison_table(comparison: ComparisonData) -> Table:
    """Create the main comparison table."""
    table = Table(title="Dataset Comparison", box=box.ROUNDED)
    table.add_column("Dataset", style="bold cyan")
    table.add_column("Episodes", justify="right", style="white")
    table.add_column("Avg. Episode Length", justify="right", style="white")
    table.add_column("Has Variations", justify="center", style="green")

    for i, name in enumerate(comparison["dataset_names"]):
        episode_lengths: list[int] = comparison["episode_lengths"][i]
        avg_length: float = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
        has_vars: str = "Yes" if comparison["has_variations"][i] else "No"
        table.add_row(name, str(comparison["num_episodes"][i]), f"{avg_length:.2f}", has_vars)

    return table


def create_action_stats_table(comparison: ComparisonData, action_stats: list[ActionStatsEntry]) -> Table:
    """Create the action statistics table."""
    table = Table(title="Action Statistics", box=box.ROUNDED)
    table.add_column("Dataset", style="bold cyan")
    table.add_column("Unique Actions", justify="right", style="white")
    table.add_column("Most Common", style="white")
    table.add_column("Total Actions", justify="right", style="white")

    for i, name in enumerate(comparison["dataset_names"]):
        stats: ActionStatsEntry = action_stats[i]
        most_common_action: float | None = stats["most_common_action"]
        most_common_count: int = stats["action_distribution"].get(most_common_action or -1, 0)
        most_common_display: str = (
            f"{most_common_action} ({most_common_count})" if most_common_action is not None else "None"
        )
        table.add_row(name, str(len(stats["unique_actions"])), most_common_display, str(stats["total_actions"]))

    return table


def create_state_shape_table(comparison: ComparisonData, state_shape_info: list[str]) -> Table:
    """Create the state shape information table."""
    table = Table(title="State Shape Information", box=box.ROUNDED)
    table.add_column("Dataset", style="bold cyan")
    table.add_column("State Shape", style="white")

    for i, name in enumerate(comparison["dataset_names"]):
        table.add_row(name, state_shape_info[i])

    return table


def truncate_dataset_names(dataset_names: list[str], max_length: int = 15) -> list[str]:
    """Truncate dataset names for display purposes."""
    return [f"{name[:max_length]}..." if len(name) > max_length else name for name in dataset_names]


def plot_episode_counts(ax: Axes, names: list[str], episode_counts: list[int]) -> None:
    """Plot episode count comparison."""
    bars: BarContainer = ax.bar(names, episode_counts, color="steelblue", alpha=0.8)
    ax.set_title("Number of Episodes")
    ax.set_ylabel("Count")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=45)

    # Add count labels above bars
    for bar in bars:
        height: float = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.1, f"{int(height)}", ha="center", va="bottom", fontsize=9
        )

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_episode_length_distribution(ax: Axes, names: list[str], episode_lengths: list[list[int]]) -> None:
    """Plot episode length distribution."""
    for i, lengths in enumerate(episode_lengths):
        if lengths:  # Only plot if we have data
            ax.hist(lengths, alpha=0.6, bins=15, label=names[i])

    ax.set_title("Episode Length Distribution")
    ax.set_xlabel("Steps per Episode")
    ax.set_ylabel("Frequency")
    ax.grid(linestyle="--", alpha=0.7)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_action_distribution(
    ax: Axes, names: list[str], action_stats: list[ActionStatsSummary], num_datasets: int
) -> None:
    """Plot action distribution comparison."""
    # Find common action space across datasets
    all_unique_actions: list[float] = sorted(set().union(*[stats["unique_actions"] for stats in action_stats]))

    # Limit to top actions if there are many
    if len(all_unique_actions) > _MAX_ACTIONS_DISPLAY:
        # Select subset based on frequency across all datasets
        action_counts: Counter[float] = Counter()
        for stats in action_stats:
            action_counts.update(stats["action_distribution"])
        all_unique_actions = [action for action, _ in action_counts.most_common(_MAX_ACTIONS_DISPLAY)]

    # Set up bar positions
    width: float = 0.8 / max(1, num_datasets)
    positions: NDArray[np.intp] = np.arange(len(all_unique_actions))

    # Plot bars for each dataset
    for i, stats in enumerate(action_stats):
        counts: list[int] = [stats["action_distribution"].get(action, 0) for action in all_unique_actions]
        # Normalize to percentage
        if sum(counts) > 0:
            percents = [count / stats["total_actions"] * 100 for count in counts]
            offset = width * (i - num_datasets / 2 + 0.5)
            ax.bar(positions + offset, percents, width, label=names[i], alpha=0.8)

    ax.set_title("Action Distribution Comparison")
    ax.set_xlabel("Action")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(a) for a in all_unique_actions])
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_comparison_plots(
    comparison: ComparisonData, action_stats: list[ActionStatsSummary], num_datasets: int, save_path: str | None = None
) -> None:
    """Generate all comparison plots following matplotlib practices."""
    # Set plotting style
    plt.rcParams.update(_PLOT_STYLE)

    # Use constrained_layout for appearance
    # Apply screen size constraints to prevent oversized windows
    constrained_width, constrained_height = constrain_figure_size(*_PLOT_FIGURE_SIZE)
    fig: Figure = plt.figure(figsize=(constrained_width, constrained_height), constrained_layout=True)
    fig.suptitle("Dataset Comparison", fontweight="bold", fontsize=14)

    # Truncate names for display
    names: list[str] = truncate_dataset_names(comparison["dataset_names"])

    # Plot 1: Episode count comparison
    ax1: Axes = plt.subplot(2, 2, 1)
    plot_episode_counts(ax1, names, comparison["num_episodes"])

    # Plot 2: Episode length distribution
    ax2: Axes = plt.subplot(2, 2, 2)
    plot_episode_length_distribution(ax2, names, comparison["episode_lengths"])

    # Plot 3: Action distribution comparison
    ax3: Axes = plt.subplot(2, 1, 2)
    plot_action_distribution(ax3, names, action_stats, num_datasets)

    show_or_save_plot(save_path)


def compare_datasets(
    datasets: list[tuple[list[StateImgTraj], list[ActionTraj]]],
    dataset_names: list[str],
    specs: list[Specs],
    show_plot: bool = True,
    save_path: str | None = None,
) -> ComparisonData:
    """Compare multiple datasets and show their differences.

    Args:
        datasets (list[tuple[list[StateImgTraj], list[ActionTraj]]]): List of datasets (state_img_trajs, action_trajs).
        dataset_names (list[str]): Names of the datasets.
        specs (list[Specs]): Specifications for each dataset.
        show_plot (bool): Whether to visualize comparison plots.
        save_path (str | None): Path to save the comparison plot to. If None, shows the plot instead.

    Returns:
        dict: Dictionary with comparison statistics.

    """
    if len(datasets) < 2:
        console.print("[red]❌ Need at least 2 datasets to compare.[/red]")
        return {"dataset_names": [], "num_episodes": [], "episode_lengths": [], "has_variations": []}

    console.print(f"\n{'=' * 80}")
    console.print("DATASET COMPARISON")
    console.print("=" * 80)

    # Extract basic comparison data
    comparison: ComparisonData = {
        "dataset_names": dataset_names,
        "num_episodes": [len(ds[0]) for ds in datasets],
        "episode_lengths": [[len(action_traj) for action_traj in ds[1]] for ds in datasets],
        "has_variations": [spec.get("state_specs", {}).get("has_variations", False) for spec in specs],
    }

    # Print comparison table
    comparison_table: Table = create_comparison_table(comparison)
    console.print(comparison_table)

    # Calculate action statistics and state shape info
    action_stats: list[ActionStatsSummary] = [calculate_action_statistics(action_trajs) for _, action_trajs in datasets]
    state_shape_info: list[str] = extract_state_shape_info(specs)

    # Add to comparison dict
    comparison["action_stats"] = action_stats
    comparison["state_shape_info"] = state_shape_info

    # Print additional tables
    console.print("\n")
    action_stats_table: Table = create_action_stats_table(comparison, action_stats)
    console.print(action_stats_table)

    console.print("\n")
    state_shape_table: Table = create_state_shape_table(comparison, state_shape_info)
    console.print(state_shape_table)

    # Generate comparison plots
    if show_plot:
        generate_comparison_plots(comparison, action_stats, len(datasets), save_path)

    return comparison


def export_summary(file_paths: list[str], specs: list[Specs], output_path: str, pretty: bool = True) -> None:
    """Export dataset summaries to a JSON file.

    Args:
        file_paths (list[str]): List of file paths for the datasets.
        specs: Specifications for each dataset.
        output_path (str): Path to save the JSON file.
        pretty (bool): Whether to format the JSON with indentation.

    """
    if not specs:
        console.print("[red]❌ No valid datasets to export.[/red]")
        return

    # Add file paths to specs (orjson handles numpy types natively)
    datasets: list[dict[str, int | str | ActionSpecs | StateSpecs]] = []
    for i, spec in enumerate(specs):
        # Create a copy of spec and add file path
        dataset_spec: dict[str, int | str | ActionSpecs | StateSpecs] = {}
        if "num_episodes" in spec:
            dataset_spec["num_episodes"] = spec["num_episodes"]
        if "file_path" in spec:
            dataset_spec["file_path"] = spec["file_path"]
        if "action_specs" in spec:
            dataset_spec["action_specs"] = spec["action_specs"]
        if "state_specs" in spec:
            dataset_spec["state_specs"] = spec["state_specs"]
        if i < len(file_paths):
            dataset_spec["file_path"] = file_paths[i]
        datasets.append(dataset_spec)

    # Prepare export data
    export_data: dict[
        str, str | int | list[dict[str, int | str | ActionSpecs | StateSpecs]] | dict[str, int | list[str]]
    ] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "num_datasets": len(specs), "datasets": datasets}

    # Add summary information for multiple datasets
    if len(specs) > 1:
        export_data["summary"] = {
            "total_episodes": sum(spec.get("num_episodes", 0) for spec in specs),
            "datasets": [pathlib.Path(path).name for path in file_paths],
        }

    try:
        option = orjson.OPT_INDENT_2 if pretty else 0
        serialized_data = orjson.dumps(export_data, option=option).decode("utf-8")
        pathlib.Path(output_path).write_text(serialized_data, encoding="utf-8")

        console.print(f"[green]✅ Summary exported to:[/green] {output_path}")

    except Exception as e:
        console.print(f"[red]❌ Error exporting summary:[/red] {e!s}")
        return


def _run_cli(args: argparse.Namespace) -> None:
    """Run the selected data-inspector CLI actions."""
    # Load the data
    state_img_trajs, action_trajs = load_data_file(args.file_path)

    # Get and print specifications
    specs: Specs = get_data_specs(state_img_trajs, action_trajs)
    print_data_report(args.file_path, specs)

    # Show tabular overview if requested
    if args.table:
        print_tabular_overview(state_img_trajs, action_trajs, specs)

    # Show hierarchical data structure if requested
    if args.structure:
        visualize_data_structure(state_img_trajs, action_trajs, args.depth)

    # Examine variations if requested
    if args.variations:
        examine_variations(state_img_trajs, args.sample, args.detailed)

    # Visualize if requested
    if args.visualize:
        visualize_sample(state_img_trajs, action_trajs, args.sample, args.frames)

    # Create timeline view if requested
    if args.timeline:
        create_timeline_view(state_img_trajs, action_trajs, args.output, max_steps=args.frames, fps=args.fps)

    # Export summary if requested
    if args.export:
        export_summary([args.file_path], [specs], args.export_path, args.pretty)


def main() -> int:
    """Entry point for the SPAR Data Inspector CLI.

    Returns:
        int: Exit code (0 for success).

    """
    parser = argparse.ArgumentParser(description="SPAR Data Inspector")
    parser.add_argument("file_path", help="Path to the data file")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize a sample trajectory")
    parser.add_argument("--variations", "-var", action="store_true", help="Display detailed info about variations")
    parser.add_argument("--structure", "-str", action="store_true", help="Display detailed hierarchical data structure")
    parser.add_argument("--table", "-t", action="store_true", help="Display tabular overview of data")
    parser.add_argument("--depth", "-dep", type=int, default=5, help="Maximum depth for structure visualization")
    parser.add_argument("--sample", "-s", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--frames", "-f", type=int, default=5, help="Maximum number of frames to display")
    parser.add_argument(
        "--detailed", "-d", action="store_true", help="Show detailed variation information including plots"
    )
    parser.add_argument(
        "--timeline", "-tl", action="store_true", help="Create a timeline video of state and action changes"
    )
    parser.add_argument("--output", "-o", type=str, default="timeline_view.mp4", help="Output file for timeline video")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for timeline video")
    parser.add_argument("--export", "-e", action="store_true", help="Export dataset summary to JSON")
    parser.add_argument(
        "--export_path", "-ep", type=str, default="summary.json", help="Output file for dataset summary JSON"
    )
    parser.add_argument("--pretty", action="store_true", help="Format JSON with indentation")

    args: argparse.Namespace = parser.parse_args()

    try:
        _run_cli(args)
    except Exception as e:
        console.print(f"[red]❌ Error:[/red] {e!s}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
