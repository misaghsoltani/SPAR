"""Visualize unsolved Q* search pairs.

Loads a results.json file (produced by the Q* runner), finds entries that were not solved by either search or
environment, and creates side-by-side visualizations of the start and goal images for each failure. When the search was
run against an HDF5 "pairs" file the script resolves the correct variant for start/goal ("base" or named variants) and
extracts the same images used during the run. Directory and single-image inputs are also supported.
"""

from __future__ import annotations

from collections import defaultdict
import contextlib
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
import re
from typing import TYPE_CHECKING, TypedDict

import h5py
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import orjson
from PIL import Image
import torch
from torch import nn

from spar.utils.pytorch_utils.nnet_utils import load_model

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from logging import Logger
    from typing import Literal, TypeAlias

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    from numpy.typing import NDArray
    from PIL.Image import Image as ImageFile
    from torch import Tensor

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.utils.config_utils.config_schema import ModelConfig, VisualizeUnsolvedQStarSPARConfig


logger: Logger = getLogger(__name__)

JSONType: TypeAlias = dict[str, "JSONType"] | list["JSONType"] | str | int | float | bool | None


class QStarLogRow(TypedDict, total=False):
    """TypedDict describing a single Q* result log row (partial).

    Only a subset of keys is declared. The loader accepts extra keys.
    """

    index: int
    idx: int
    pair_index: int
    start_variant: str
    goal_variant: str
    solved_by_search: bool
    solved_by_env: bool
    num_nodes_generated: int
    elapsed_sec: float
    solve_category: str
    moves: list[int]
    num_moves: int
    num_iterations: int


@dataclass(frozen=True, slots=True)
class _VisualizationContext:
    cfg: VisualizeUnsolvedQStarSPARConfig
    h5f: h5py.File | None
    has_single_pair_inputs: bool
    use_models: bool
    alignment_model: nn.Module | None
    decoder_model: nn.Module | None
    encoder_model: nn.Module | None
    device: torch.device | None
    figsize: tuple[float, float]
    outfmt: str


def load_results(path: str) -> dict[str, JSONType]:
    """Load results.json using orjson and return the parsed object.

    Returns a dictionary whose top-level ``logs`` key contains per-pair rows.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"results file not found: {path}")
    payload: JSONType = orjson.loads(p.read_bytes())
    if not isinstance(payload, dict):
        raise TypeError(f"results file did not contain a JSON object: {path}")
    return payload


def _parse_figsize(cfg_figsize: str | tuple[float, float] | list[float] | None) -> tuple[float, float]:
    """Parse figsize from config.

    Accepts strings like "10x6" / "10X6" / "10,6" (with optional spaces) or a 2-sequence of numbers.
    Falls back to (10.0, 6.0) on any error so downstream plotting never crashes.
    """
    default: tuple[float, float] = (10.0, 6.0)
    if cfg_figsize is None:
        return default

    # Tuple/list provided
    if isinstance(cfg_figsize, (tuple, list)) and len(cfg_figsize) == 2:
        try:
            return cfg_figsize[0], cfg_figsize[1]
        except Exception:
            return default

    if isinstance(cfg_figsize, str):
        # Normalize separators and strip spaces
        tokens: list[str] = re.split(r"[xX,]", cfg_figsize.replace(" ", ""))
        if len(tokens) == 2:
            try:
                return float(tokens[0]), float(tokens[1])
            except Exception:
                return default
    return default


def find_unsolved_logs(logs: Iterable[JSONType]) -> list[dict[str, JSONType]]:
    """Return log rows where neither search nor env solved the pair.

    Only dictionaries with the expected string keys are considered. Other JSON
    values and rows with missing fields are skipped.
    """
    out: list[dict[str, JSONType]] = []
    for entry in logs:
        if not isinstance(entry, dict):
            continue
        row: dict[str, JSONType] = entry
        if bool(row.get("solved_by_search")) or bool(row.get("solved_by_env")):
            continue

        qrow: dict[str, JSONType] = {}
        idx_v: JSONType = row.get("index") or row.get("idx") or row.get("pair_index")
        if isinstance(idx_v, int):
            qrow["index"] = idx_v
        elif isinstance(idx_v, str):
            with contextlib.suppress(ValueError):
                qrow["index"] = int(idx_v)

        sv: JSONType = row.get("start_variant")
        if isinstance(sv, str):
            qrow["start_variant"] = sv
        gv: JSONType = row.get("goal_variant")
        if isinstance(gv, str):
            qrow["goal_variant"] = gv

        sbs: JSONType = row.get("solved_by_search")
        if isinstance(sbs, bool):
            qrow["solved_by_search"] = sbs
        sbe: JSONType = row.get("solved_by_env")
        if isinstance(sbe, bool):
            qrow["solved_by_env"] = sbe

        out.append(qrow)
    return out


def _variant_bucket_key(row: Mapping[str, JSONType]) -> str:
    """Return a variant identifier for per-variant capping.

    Prefer the start variant, fall back to goal variant, and default to "base"
    if neither is present. This mirrors how variants are encoded in the results.
    """
    sv: JSONType = row.get("start_variant")
    if isinstance(sv, str) and sv:
        return sv
    gv: JSONType = row.get("goal_variant")
    if isinstance(gv, str) and gv:
        return gv
    return "base"


def _row_priority_score(row: Mapping[str, JSONType]) -> float:
    """Score an unsolved row for best/worst prioritization.

    The primary metric is ``num_nodes_generated``. Missing values fall back to ``elapsed_sec``,
    ``path_cost``, then ``num_iterations``. Index is used as a final tie-breaker
    to keep ordering deterministic.
    """
    for key in ("num_nodes_generated", "elapsed_sec", "path_cost", "num_iterations"):
        val: JSONType = row.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return float(_get_index_from_row(row))


def _order_unsolved(rows: list[dict[str, JSONType]], prioritize: str) -> list[dict[str, JSONType]]:
    """Order unsolved rows according to the requested priority."""
    mode: str = (prioritize or "none").lower()
    if mode not in {"best", "worst"}:
        return list(rows)

    reverse: bool = mode == "worst"

    def _row_priority_key(row: dict[str, JSONType]) -> tuple[float, int]:
        return _row_priority_score(row), _get_index_from_row(row)

    return sorted(rows, key=_row_priority_key, reverse=reverse)


def _select_unsolved_rows(
    rows: list[dict[str, JSONType]], max_total: int, max_per_var: int | None, prioritize: str
) -> list[dict[str, JSONType]]:
    """Select rows respecting overall cap, per-variant cap, and priority ordering."""
    if max_total <= 0:
        return []

    ordered: list[dict[str, JSONType]] = _order_unsolved(rows, prioritize)
    per_var_limit: int | None = max_per_var if max_per_var is not None and max_per_var > 0 else None

    if per_var_limit is None:
        return ordered[:max_total]

    counts: defaultdict[str, int] = defaultdict(int)
    selected: list[dict[str, JSONType]] = []
    for row in ordered:
        if len(selected) >= max_total:
            break
        key: str = _variant_bucket_key(row)
        if counts[key] >= per_var_limit:
            continue
        counts[key] += 1
        selected.append(row)

    return selected


def _count_by_variant(rows: Iterable[Mapping[str, JSONType]]) -> dict[str, int]:
    """Return a dict of variant -> count for logging/debugging selections."""
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counts[_variant_bucket_key(row)] += 1
    return dict(counts)


def _get_index_from_row(row: Mapping[str, JSONType]) -> int:
    """Extract a numeric index from a log row.

    Args:
        row: Log record that may contain an index field.

    Returns:
        The first integer index found under a recognized key, or zero when no
        recognized value is present.
    """
    for k in ("index", "idx", "pair_index", "i"):
        v: JSONType = row.get(k)
        if v is None:
            continue
        # Accept integers and strings that parse as integers. Convert other
        # numeric values only after checking their runtime type.
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                continue
        if isinstance(v, float):
            return int(v)
        # For other types, try to coerce via numpy to extract a python scalar.
        # This handles numpy scalar types and other numeric-like objects while
        # avoiding impossible isinstance() combinations on unions.
        scalar: int | float | None
        try:
            scalar = np.asarray(v).item()
        except Exception:
            scalar = None
        if isinstance(scalar, (int, np.integer)):
            return int(scalar)

        # Not convertible
        continue

    return 0


def _to_hwc_uint8(img: NDArray[np.uint8 | np.float32]) -> NDArray[np.uint8]:
    """Convert numpy image (CHW or HWC, float in [0,1] or uint8) to HWC uint8 for plotting.

    This helper is for visualization only. If a float array is provided it is
    assumed to be in [0,1] and will be scaled to 0..255. If a uint8 array is
    provided it will be returned (with shape adjustments) as-is.
    """
    if img.ndim == 3 and img.shape[0] in {1, 3}:
        # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    # If grayscale (H,W) expand to H,W,3
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    # If float, assume range [0,1] and scale for display
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    # If single channel expand
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return np.asarray(img, dtype=np.uint8)


def _float_to_vis_uint8(img_float: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Convert HWC float32 image in [0,1] to HWC uint8 for visualization.

    Use this to prepare decoder outputs for plotting without affecting float
    computations used for MSE.
    """
    if img_float.ndim == 3 and img_float.shape[0] in {1, 3}:
        img_float = np.transpose(img_float, (1, 2, 0))
    if img_float.ndim == 2:
        img_float = np.stack([img_float] * 3, axis=-1)
    return np.clip(img_float * 255.0, 0, 255).astype(np.uint8)


def load_image_from_h5(h5f: h5py.File, index: int, variant: str, side: str) -> NDArray[np.uint8]:
    """Load an image from an HDF5 pairs file for given index and variant.

    side must be 'start' or 'goal'. Variant 'base' uses the group's top-level
    ``images`` dataset. Other variants are read from ``variations/<variant>/images``.
    Returns an HWC uint8 numpy array.
    """
    # Validate the requested HDF5 group before reading datasets from it.
    grp_obj = h5f.get(f"pairs/{side}")
    if not isinstance(grp_obj, h5py.Group):
        raise KeyError(f"Missing or invalid group pairs/{side} in HDF5")
    grp: h5py.Group = grp_obj

    # base dataset
    if variant == "base":
        ds_obj = grp.get("images")
        if not isinstance(ds_obj, h5py.Dataset):
            raise KeyError(f"Missing images dataset under pairs/{side}")
        ds: h5py.Dataset = ds_obj
    else:
        vargrp_obj = grp.get("variations")
        if not isinstance(vargrp_obj, h5py.Group):
            raise KeyError(f"No variations group for pairs/{side} in HDF5")
        v_obj = vargrp_obj.get(variant)
        if not isinstance(v_obj, h5py.Group):
            raise KeyError(f"Variant '{variant}' not found under pairs/{side}/variations")
        ds_obj = v_obj.get("images")
        if not isinstance(ds_obj, h5py.Dataset):
            raise KeyError(f"Missing images dataset under pairs/{side}/variations/{variant}")
        ds = ds_obj

    # Read a single example. Datasets in SPAR are frequently stored as NCHW
    arr: NDArray[np.uint8 | np.float32] = np.array(ds[index : index + 1], copy=False)
    if arr.shape[0] == 1:
        arr = arr[0]

    return _to_hwc_uint8(arr)


def load_image_from_dirs(
    start_dir: str, goal_dir: str | None, index: int, variant: str
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Load start/goal from directories by index.

    Returns HWC uint8 numpy arrays.
    """
    # Use top-level helpers
    # Use the module-level helpers for listing and loading images
    # (implemented as _list_sorted_images / _load_pil_rgb)

    # start
    start_base: str = start_dir
    variant_dir: Path = Path(start_dir) / variant if variant != "base" else Path(start_dir)
    start_list: list[str] = _list_sorted_images(
        str(variant_dir) if (variant_dir.exists() and variant_dir.is_dir()) else start_base
    )

    if index >= len(start_list):
        raise IndexError(f"index {index} out of range for start images ({len(start_list)})")
    start_img: NDArray[np.uint8] = _load_pil_rgb(start_list[index])

    # goal
    goal_img: NDArray[np.uint8]
    if goal_dir is None:
        goal_img = start_img.copy()
    else:
        goal_base: str = goal_dir
        variant_dir_g: Path = Path(goal_dir) / variant if variant != "base" else Path(goal_dir)
        goal_list: list[str] = _list_sorted_images(
            str(variant_dir_g) if (variant_dir_g.exists() and variant_dir_g.is_dir()) else goal_base
        )

        if index >= len(goal_list):
            raise IndexError(f"index {index} out of range for goal images ({len(goal_list)})")

        goal_img = _load_pil_rgb(goal_list[index])

    return _to_hwc_uint8(start_img), _to_hwc_uint8(goal_img)


def _list_sorted_images(d: str) -> list[str]:
    p = Path(d)
    exts: set[str] = {".png", ".jpg", ".jpeg", ".bmp"}
    files: list[str] = [str(x) for x in sorted(p.iterdir()) if x.suffix.lower() in exts]
    return files


def _load_pil_rgb(path: str) -> NDArray[np.uint8]:
    return np.array(Image.open(path).convert("RGB"))


def _img_np_to_tensor(img: NDArray[np.uint8], device: torch.device) -> Tensor:
    """HWC uint8 -> CHW float32 [0,1] tensor on device."""
    t: Tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float() / 255.0
    return t.unsqueeze(0).to(device)


def _tensor_to_hwc_float01(t: Tensor) -> NDArray[np.float32]:
    """Convert a tensor to an HWC float32 array.

    Args:
        t: Tensor in CHW or NCHW layout.

    Returns:
        An HWC float32 array. Values above 1.5 are interpreted as the
        ``[0, 255]`` range and divided by 255. Values are not clipped.
    """
    a: NDArray[np.float32] = t.detach().cpu().numpy()
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    # CHW -> HWC
    if a.shape[0] in {1, 3}:
        a = np.transpose(a, (1, 2, 0))
    a = a.astype(np.float32)
    # Scale values in [0, 255] to [0, 1]. Leave values already in [0, 1] unchanged.
    if float(np.max(a)) > 1.5:
        a /= 255.0
    return a


def _encode_img_get_bits(
    encoder_model: nn.Module, img_np: NDArray[np.uint8 | np.float32], device: torch.device
) -> Tensor:
    t: Tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float() / 255.0
    t = t.unsqueeze(0).to(device)
    out: Tensor
    with torch.inference_mode():
        out = encoder_model(t)
    if out.ndim > 2:
        out = out.reshape(out.shape[0], -1)
    if out.min() < -0.1 or out.max() > 1.1:
        out = torch.sigmoid(out)

    return torch.round(out).clamp(0, 1).detach().cpu()


def create_visualization(
    start_img: NDArray[np.uint8],
    goal_img: NDArray[np.uint8],
    base_img: NDArray[np.uint8] | None,
    meta: QStarLogRow | None,
    outpath: str,
    *,
    dpi: int = 150,
    figsize: tuple[float, float] = (12.0, 6.0),
    # overlay_moves: bool = False,
    # moves: list[int] | None = None,
    recon_img: NDArray[np.uint8] | None = None,
    recon_mse: float | None = None,
    bitwise_eq_pct: float | None = None,
    font_scale: float = 1.25,
    image_border: float = 0.0,
    image_border_padding: float = 6.0,
) -> None:
    """Create a side-by-side visualization and write to ``outpath``.

    All descriptive text (titles, captions, optional move overlays, metrics) is kept in
    dedicated axes so it never overlaps the rendered images, even for very small figures
    or large ``font_scale`` values.
    """
    _ = meta

    def _scaled_font(key: Literal["axes.titlesize", "font.size"], scale: float, fallback: float = 12.0) -> float:
        try:
            raw = plt.rcParams[key]
            return float(raw) * scale
        except Exception:
            return fallback * scale

    # Build caption content up-front so we can size the GridSpec to give it its own row.
    # Only keep metric summaries on the image.
    caption_segments: list[str] = []
    if recon_mse is not None:
        caption_segments.append(f"MSE: {recon_mse:.3g}")
    if bitwise_eq_pct is not None:
        caption_segments.append(f"Bitwise Equality: {bitwise_eq_pct:.2f}%")

    caption_lines: list[str] = [" • ".join(caption_segments)] if caption_segments else []

    # Convert border/padding (points) to figure-relative margins.
    padding_in: float = max(image_border_padding, 0.0) / 72.0
    border_pts: float = max(image_border, 0.0)
    fig_w, fig_h = figsize
    left_margin: float = min(0.45, padding_in / max(fig_w, 1e-6))
    right_margin: float = left_margin
    bottom_margin: float = min(0.45, padding_in / max(fig_h, 1e-6))
    top_margin: float = bottom_margin

    has_caption: bool = len(caption_lines) > 0
    caption_height: float = 0.0
    if has_caption:
        # Reserve a compact caption row and keep it close to the images.
        base_height: float = 0.18 * max(font_scale, 0.8)
        extra_height: float = 0.06 * max(len(caption_lines) - 1, 0) * max(font_scale, 1.0)
        caption_height = base_height + extra_height

    fig: Figure = plt.figure(figsize=figsize, constrained_layout=True)
    if has_caption:
        gs: GridSpec = fig.add_gridspec(
            3,
            2,
            height_ratios=[1.0, 1.0, caption_height],
            hspace=0.06,
            wspace=0.05,
            left=left_margin,
            right=1.0 - right_margin,
            bottom=bottom_margin,
            top=1.0 - top_margin,
        )
        caption_ax: Axes | None = fig.add_subplot(gs[2, :])
        assert caption_ax is not None
        caption_ax.axis("off")
    else:
        gs = fig.add_gridspec(
            2,
            2,
            hspace=0.18,
            wspace=0.05,
            left=left_margin,
            right=1.0 - right_margin,
            bottom=bottom_margin,
            top=1.0 - top_margin,
        )
        caption_ax = None

    ax_start: Axes = fig.add_subplot(gs[0, 0])
    ax_goal: Axes = fig.add_subplot(gs[0, 1])
    ax_base: Axes = fig.add_subplot(gs[1, 0])
    ax_recon: Axes = fig.add_subplot(gs[1, 1])

    for ax, img in ((ax_start, start_img), (ax_goal, goal_img)):
        ax.imshow(img)
        ax.axis("off")

    if base_img is not None:
        ax_base.imshow(base_img)
    ax_base.axis("off")

    if recon_img is not None:
        ax_recon.imshow(recon_img)
    ax_recon.axis("off")

    title_fs: float = _scaled_font("axes.titlesize", font_scale, fallback=12.0)
    title_pad: float = 6.0 * font_scale
    ax_start.set_title("Start State (Input to Search)", fontsize=title_fs, pad=title_pad)
    ax_goal.set_title("Goal State", fontsize=title_fs, pad=title_pad)
    ax_base.set_title("Start State (Ground Truth)", fontsize=title_fs, pad=title_pad)
    ax_recon.set_title("Reconstruction", fontsize=title_fs, pad=title_pad)

    if caption_ax is not None:
        caption_fs: float = _scaled_font("font.size", font_scale * 0.95, fallback=12.0)
        # Space lines evenly within the caption band, biased toward the center
        n_lines: int = len(caption_lines)
        y_positions: NDArray[np.float32] = (
            np.linspace(0.8, 0.6, n_lines, dtype=np.float32) if n_lines > 1 else np.array([0.75])
        )
        for y, line in zip(y_positions, caption_lines, strict=False):
            caption_ax.text(
                0.5,
                float(y),
                line,
                ha="center",
                va="center",
                fontsize=caption_fs,
                bbox={"facecolor": "white", "alpha": 0.9, "pad": 4, "edgecolor": "none"},
            )

    if border_pts > 0.0:
        fig.patches.append(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=fig.transFigure,
                fill=False,
                linewidth=border_pts,
                edgecolor="black",
                zorder=1000,
                clip_on=False,
            )
        )

    # Save with tight bounding box to respect constrained layout and avoid extra margins
    # Matplotlib's format argument can override an extension in outpath.
    fmt: str | None = None
    # Use an existing extension. Otherwise leave fmt as None and
    # matplotlib will infer from `outpath` or use provided fmt in caller.
    suff: str = Path(outpath).suffix
    if suff:
        fmt = suff.lstrip(".")
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", pad_inches=0.0, format=fmt)
    plt.close(fig)


def _load_visualization_images(
    cfg: VisualizeUnsolvedQStarSPARConfig,
    h5f: h5py.File | None,
    idx: int,
    start_var: str,
    goal_var: str,
    has_single_pair_inputs: bool,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]] | None:
    if h5f is not None:
        start_img = load_image_from_h5(h5f, idx, start_var, "start")
        goal_img = load_image_from_h5(h5f, idx, goal_var, "goal")
        # load base/clean versions for encoder / MSE comparison
        start_img_base = load_image_from_h5(h5f, idx, "base", "start")
        # goal base not needed for current visualizations
        return start_img, goal_img, start_img_base
    if cfg.state_dir is not None:
        start_img, goal_img = load_image_from_dirs(cfg.state_dir, cfg.goal_dir, idx, start_var)
        start_img_base, _ = load_image_from_dirs(cfg.state_dir, cfg.goal_dir, idx, "base")
        # goal base not needed for current visualizations
        return start_img, goal_img, start_img_base
    if has_single_pair_inputs:
        assert cfg.single_state is not None
        assert cfg.single_goal is not None
        start_img = _to_hwc_uint8(_load_pil_rgb(cfg.single_state))
        goal_img = _to_hwc_uint8(_load_pil_rgb(cfg.single_goal))
        # single-image mode: base fallback to provided image
        start_img_base = start_img.copy()
        return start_img, goal_img, start_img_base
    return None


def _resize_recon_to_base_safe(
    recon_img: NDArray[np.uint8], base_np: NDArray[np.float32], recon_np: NDArray[np.float32]
) -> NDArray[np.float32]:
    try:
        return _resize_recon_to_base(recon_img, base_np)
    except Exception:
        return recon_np


def _resize_recon_to_base(recon_img: NDArray[np.uint8], base_np: NDArray[np.float32]) -> NDArray[np.float32]:
    recon_pil: ImageFile = Image.fromarray(recon_img)
    recon_pil = recon_pil.resize((base_np.shape[1], base_np.shape[0]))
    return np.array(recon_pil).astype(np.float32)


def _compute_reconstruction_metrics(
    start_img: NDArray[np.uint8],
    start_img_base: NDArray[np.uint8],
    alignment_model: nn.Module,
    decoder_model: nn.Module,
    device: torch.device,
) -> tuple[NDArray[np.uint8], float, Tensor]:
    # prepare input: variant image (start_img) -> alignment model -> round -> decoder -> recon
    inp: Tensor = _img_np_to_tensor(start_img, device)
    out_align: Tensor = alignment_model(inp)
    # Add a batch dimension to produce shape (1, N).
    out_align = out_align.detach()
    # if outputs look like images, try global pooling -> flatten
    if out_align.ndim > 2:
        out_align = out_align.reshape(out_align.shape[0], -1)
    # normalize with sigmoid if values outside [0,1]
    if out_align.min() < -0.1 or out_align.max() > 1.1:
        out_align = torch.sigmoid(out_align)

    out_align_rounded: Tensor = torch.round(out_align).clamp(0, 1)

    # decoder expects bits -> reconstruct
    recon_t: Tensor = decoder_model(out_align_rounded)
    recon_img: NDArray[np.uint8] = _float_to_vis_uint8(_tensor_to_hwc_float01(recon_t))

    # compute MSE against base (clean) start image in float [0,1]
    base_np: NDArray[np.float32] = start_img_base.astype(np.float32)
    recon_np: NDArray[np.float32] = recon_img.astype(np.float32)
    if base_np.shape != recon_np.shape:
        # attempt to resize recon to base using PIL
        recon_np = _resize_recon_to_base_safe(recon_img, base_np, recon_np)
    base_np /= 255.0
    recon_np /= 255.0
    recon_mse = float(np.mean((recon_np - base_np) ** 2))
    logger.info(f"Reconstruction MSE (vs base): {recon_mse:.4f}")
    return recon_img, recon_mse, out_align_rounded


def _compute_bitwise_eq_pct(
    encoder_model: nn.Module, start_img_base: NDArray[np.uint8], alignment_bits: Tensor | None, device: torch.device
) -> float | None:
    if alignment_bits is None:
        return None
    try:
        return _compute_bitwise_eq_pct_checked(encoder_model, start_img_base, alignment_bits, device)
    except Exception:
        return None


def _compute_bitwise_eq_pct_checked(
    encoder_model: nn.Module, start_img_base: NDArray[np.uint8], alignment_bits: Tensor, device: torch.device
) -> float | None:
    enc_bits: Tensor = _encode_img_get_bits(encoder_model, start_img_base, device)
    # Convert both arrays to integer binary values.
    align_bits_np: NDArray[np.int32] = alignment_bits.detach().cpu().numpy().ravel().astype(int)
    enc_bits_np: NDArray[np.int32] = enc_bits.numpy().ravel().astype(int)
    # pad to min length
    nbits: int = min(len(align_bits_np), len(enc_bits_np))
    if nbits <= 0:
        return None
    eq = int((align_bits_np[:nbits] == enc_bits_np[:nbits]).sum())
    bitwise_eq_pct = 100.0 * float(eq) / float(nbits)
    logger.info(f"Bitwise equality between alignment bits and encoder bits: {bitwise_eq_pct:.4f}%")
    return bitwise_eq_pct


def _build_visualization_meta(row: Mapping[str, JSONType], idx: int, start_var: str, goal_var: str) -> QStarLogRow:
    # Build a minimal QStarLogRow from available information
    meta: QStarLogRow = {}
    meta["index"] = idx
    meta["start_variant"] = start_var
    meta["goal_variant"] = goal_var
    # Copy a couple of optional numeric metadata fields if present
    n_nodes: JSONType = row.get("num_nodes_generated")
    if isinstance(n_nodes, int):
        meta["num_nodes_generated"] = n_nodes
    elapsed: JSONType = row.get("elapsed_sec")
    if isinstance(elapsed, (int, float)):
        meta["elapsed_sec"] = float(elapsed)
    num_moves: JSONType = row.get("num_moves")
    if isinstance(num_moves, int):
        meta["num_moves"] = num_moves
    return meta


def _coerce_moves(row: Mapping[str, JSONType]) -> list[int] | None:
    moves_list: list[int] | None = None
    moves_val: JSONType = row.get("moves")
    if isinstance(moves_val, list):
        moves_list = []
        for mv in moves_val:
            if isinstance(mv, (int, np.integer)):
                moves_list.append(int(mv))
            elif isinstance(mv, str):
                with contextlib.suppress(ValueError):
                    moves_list.append(int(mv))
    return moves_list


def _visualize_selected_row(
    row: Mapping[str, JSONType], context: _VisualizationContext, idx: int, start_var: str, goal_var: str
) -> bool:
    loaded_images = _load_visualization_images(
        context.cfg, context.h5f, idx, start_var, goal_var, context.has_single_pair_inputs
    )
    if loaded_images is None:
        return False

    start_img, goal_img, start_img_base = loaded_images
    recon_img: NDArray[np.uint8] | None = None
    recon_mse: float | None = None
    bitwise_eq_pct: float | None = None
    out_align_rounded: Tensor | None = None

    if (
        context.use_models
        and context.device is not None
        and isinstance(context.alignment_model, nn.Module)
        and isinstance(context.decoder_model, nn.Module)
    ):
        recon_img, recon_mse, out_align_rounded = _compute_reconstruction_metrics(
            start_img, start_img_base, context.alignment_model, context.decoder_model, context.device
        )

    if (
        context.use_models
        and context.device is not None
        and isinstance(context.encoder_model, nn.Module)
        and isinstance(context.alignment_model, nn.Module)
    ):
        bitwise_eq_pct = _compute_bitwise_eq_pct(
            context.encoder_model, start_img_base, out_align_rounded, context.device
        )

    outname: str = f"unsolved_idx{idx}_sv-{start_var}_gv-{goal_var}.{context.outfmt}"
    outpath: str = str(Path(context.cfg.outdir) / outname)
    meta = _build_visualization_meta(row, idx, start_var, goal_var)
    _coerce_moves(row)
    create_visualization(
        start_img,
        goal_img,
        start_img_base,
        meta,
        outpath,
        dpi=context.cfg.dpi,
        figsize=context.figsize,
        recon_img=recon_img,
        recon_mse=recon_mse,
        bitwise_eq_pct=bitwise_eq_pct,
        font_scale=context.cfg.font_scale,
        image_border=context.cfg.image_border,
        image_border_padding=context.cfg.image_border_padding,
    )
    logger.info(f"Wrote: {outpath}")
    return True


def visualize(env: ABCEnvironment[ABCState], cfg: VisualizeUnsolvedQStarSPARConfig) -> None:
    """Stage entrypoint for the visualizer used by the SPAR CLI.

    Args:
        env: environment instance (unused by this script but kept for signature compatibility)
        cfg: validated Pydantic config instance for this stage (VisualizeUnsolvedQStarSPARConfig)
    """
    if cfg.results is None:
        raise RuntimeError("'results' must be set in the stage config (configs/stage/visualize_unsolved_qstar.yaml)")

    if cfg.verbose:
        logger.setLevel("DEBUG")

    logger.info(
        f"Visualizing unsolved Q* pairs - results={cfg.results}, "
        f"outdir={cfg.outdir}, max={cfg.max}, max_per_var={cfg.max_per_var}, "
        f"prioritize={cfg.prioritize}, device={cfg.device}"
    )

    data: dict[str, JSONType] = load_results(cfg.results)
    logs_obj: JSONType = data.get("logs", [])

    logs_list: list[JSONType] = logs_obj if isinstance(logs_obj, list) else []

    # Consider only entries the environment did not solve. The helper will
    # perform runtime validation and return typed rows.
    unsolved: list[dict[str, JSONType]] = find_unsolved_logs(logs_list)
    logger.info(f"Loaded results: total_rows={len(logs_list)}, unsolved_not_solved_by_env={len(unsolved)}")
    if not unsolved:
        logger.info("No entries unsolved by environment found in results.")
        return

    priority_mode: str = (cfg.prioritize or "none").lower()
    selected_unsolved: list[dict[str, JSONType]] = _select_unsolved_rows(
        unsolved, max_total=cfg.max, max_per_var=cfg.max_per_var, prioritize=priority_mode
    )
    if not selected_unsolved:
        logger.info(f"No unsolved entries selected after applying max={cfg.max} and max_per_var={cfg.max_per_var}.")
        return

    if cfg.max_per_var:
        variant_counts: dict[str, int] = _count_by_variant(selected_unsolved)
        if variant_counts:
            counts_str: str = ", ".join(f"{k}:{v}" for k, v in sorted(variant_counts.items()))
            logger.info(f"Per-variant selection counts (max_per_var={cfg.max_per_var}): {counts_str}")

    logger.info(
        f"Selected {len(selected_unsolved)} unsolved rows (from {len(unsolved)} total) "
        f"with prioritize={priority_mode} and max_per_var={cfg.max_per_var}"
    )

    Path(cfg.outdir).mkdir(exist_ok=True, parents=True)

    h5f: h5py.File | None = None
    if cfg.pairs_file:
        h5f = h5py.File(cfg.pairs_file, "r", rdcc_nbytes=128 * 1024**2, rdcc_nslots=10_007, rdcc_w0=0.25)
        logger.info(f"Opened pairs HDF5 file: {cfg.pairs_file}")
    else:
        logger.info("No pairs HDF5 file provided. Using directory-based image loading.")

    # parse figsize
    figsize: tuple[float, float] = _parse_figsize(cfg.figsize)

    count: int = 0
    # optional model loading
    alignment_model: nn.Module | None = None
    decoder_model: nn.Module | None = None
    encoder_model: nn.Module | None = None
    device: torch.device | None = None
    use_models: bool = False
    # Optional model loading (use environment getters + load_model from nnet_utils)
    if cfg.alignment_model_path or cfg.decoder_model_path or cfg.encoder_model_path:
        use_models = True
        device = torch.device(cfg.device)

        is_continuous: bool = cfg.alignment_model_type == "continuous"
        get_encoder: Callable[[ModelConfig], nn.Module] = (
            env.get_encoder_cont if is_continuous else env.get_encoder_disc
        )
        get_decoder: Callable[[ModelConfig], nn.Module] = (
            env.get_decoder_cont if is_continuous else env.get_decoder_disc
        )

        # Alignment model
        if cfg.alignment_model_path:
            alignment_model = load_model(
                model=env.get_alignment_model(cfg.model),
                device=device,
                pretrained_path=cfg.alignment_model_path,
                freeze=True,
                compile_cfg=None,
            )
            logger.info(f"Alignment model instantiated and loaded from: {cfg.alignment_model_path}")
        else:
            logger.info("No alignment model provided.")

        # Decoder
        if cfg.decoder_model_path:
            decoder_model = load_model(
                model=get_decoder(cfg.model),
                device=device,
                pretrained_path=cfg.decoder_model_path,
                freeze=True,
                compile_cfg=None,
            )
            logger.info(f"Decoder instantiated and loaded from: {cfg.decoder_model_path}")
        else:
            logger.info("No decoder model provided.")

        # Encoder
        if cfg.encoder_model_path:
            encoder_model = load_model(
                model=get_encoder(cfg.model),
                device=device,
                pretrained_path=cfg.encoder_model_path,
                freeze=True,
                compile_cfg=None,
            )
            logger.info(f"Encoder instantiated and loaded from: {cfg.encoder_model_path}")
        else:
            logger.info("No encoder model provided.")
    # Validate output format option
    allowed_formats: set[str] = {"png", "pdf", "jpeg", "jpg"}
    outfmt: str = (cfg.output_format or "png").lower()
    if outfmt == "jpg":
        outfmt = "jpeg"
    if outfmt not in allowed_formats:
        logger.warning(f"Invalid output_format '{outfmt}' specified, falling back to 'png'.")
        outfmt = "png"

    has_single_pair_inputs: bool = cfg.single_state is not None and cfg.single_goal is not None
    if h5f is None and cfg.state_dir is None and not has_single_pair_inputs:
        raise RuntimeError(
            "No input source specified. Provide one of:"
            " --pairs-file, --state-dir/--goal-dir, or --single-state/--single-goal"
        )

    render_context = _VisualizationContext(
        cfg=cfg,
        h5f=h5f,
        has_single_pair_inputs=has_single_pair_inputs,
        use_models=use_models,
        alignment_model=alignment_model,
        decoder_model=decoder_model,
        encoder_model=encoder_model,
        device=device,
        figsize=figsize,
        outfmt=outfmt,
    )

    for row in selected_unsolved:
        if count >= cfg.max:
            break
        idx: int = _get_index_from_row(row)
        # Image loaders receive variant names as plain strings.
        sv_val: JSONType = row.get("start_variant")
        start_var: str = sv_val if isinstance(sv_val, str) else "base"
        gv_val: JSONType = row.get("goal_variant")
        goal_var: str = gv_val if isinstance(gv_val, str) else "base"

        logger.info(f"Processing idx={idx} start_var={start_var} goal_var={goal_var}")

        try:
            wrote = _visualize_selected_row(row, render_context, idx, start_var, goal_var)
        except Exception as e:  # keep going on errors
            logger.info(f"Skipping idx={idx} due to error: {e}")
            continue
        if wrote:
            count += 1

    if h5f is not None:
        h5f.close()
