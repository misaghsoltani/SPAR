"""Evaluate discrete and continuous SPAR models in one pass.

The evaluator runs both world models on the same inputs and produces grids
with three rows:

- Row 1: Discrete Reconstruction
- Row 2: Continuous Reconstruction
- Row 3: Ground Truth (Base)

Per-step grids and per-variation summaries use the existing output locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
import operator
import pathlib
from typing import TYPE_CHECKING, TypedDict

from matplotlib import pyplot as plt
import numpy as np
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import torch
import torch.nn.functional as F

from spar.testing.visualization import MetricsTracker, VisualizationData
from spar.testing.visualization.image_utils import ImageProcessor, extract_final_rendered_image, format_metric_value
from spar.utils.log_utils.console_logger import terminal_console as console
from spar.utils.viz_utils.highlighter import derive_highlight_geometry, highlight_differences_with_contrast_fill
from spar.utils.viz_utils.image_grid import GridStyle, create_image_grid

if TYPE_CHECKING:
    from logging import Logger

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from rich.progress import TaskID
    from torch import Tensor, nn

    from spar.data.testing_dataset import TestDataLoader, VariationInfo
    from spar.testing.visualization.types import SuptitleStyle, SuptitleStyleValue
    from spar.utils.config_utils.config_schema import BBoxConfig, SuptitleConfig


logger: Logger = getLogger(__name__)

_METRIC_PRIORITY_KEYS: list[str] = ["reconstruction_mse", "eq_bit", "cosine_similarity", "similarity"]


def _visualization_step_key(value: VisualizationData) -> int:
    """Return the step number used to order visualization records."""
    return value.step


def _normalize_trirow_labels(labels: list[str] | None) -> list[str]:
    """Return exactly three row labels in (discrete, continuous, ground-truth) order.

    The combined tester accepts optional user-provided labels that may have been
    intended for the original two-row visualizations. To avoid mislabeling rows in
    the 3-row grids, we always expand or truncate to three entries while keeping
    sensible defaults for any missing positions.
    """
    defaults: list[str] = ["Discrete Recon", "Continuous Recon", "Ground Truth"]
    if not labels:
        return defaults

    # When three or more labels are provided, respect the first three directly.
    if len(labels) >= 3:
        return list(labels[:3])

    # Two-label inputs usually come from the two-row (reconstruction / GT) setup.
    if len(labels) == 2:
        recon_label: str = labels[0] or defaults[0]
        gt_label: str = labels[1] or defaults[2]
        return [recon_label, defaults[1], gt_label]

    # Single-label inputs: treat it as the reconstruction label and keep defaults
    # for the continuous row and ground-truth row.
    return [labels[0] or defaults[0], defaults[1], defaults[2]]


def pick_and_format_metrics(md: dict[str, float | int | str] | None, priority_keys: list[str] | None = None) -> str:
    """Format up to two selected metrics.

    Args:
        md: Metric names and values.
        priority_keys: Preferred key order. The default metric order is used when
            this argument is omitted.

    Returns:
        A compact metric string, or an empty string when no metrics are given.
    """
    if not md:
        return ""
    keys_src: list[str] = priority_keys or _METRIC_PRIORITY_KEYS
    ordered: list[str] = [k for k in keys_src if k in md]
    if not ordered:
        ordered = list(md.keys())[:2]
    out: list[str] = []
    for k in ordered[:2]:
        raw: float | int | str | None = md.get(k, None)
        try:
            val: float | None = float(raw) if isinstance(raw, (int, float)) else None
        except Exception:
            val = None
        if val is None:
            continue
        out.append(f"{format_metric_value(val, k)}")
    return " | ".join(out)


def _format_dual_metrics_title(
    base_title: str,
    dis_metrics: dict[str, float | int | str] | None,
    cont_metrics: dict[str, float | int | str] | None,
    *,
    priority_keys: list[str] | None = None,
) -> str:
    """Compose a multi-line title that shows base, discrete and continuous metrics."""
    d_line: str = pick_and_format_metrics(dis_metrics, priority_keys)
    c_line: str = pick_and_format_metrics(cont_metrics, priority_keys)
    parts: list[str] = [base_title]
    if d_line:
        parts.append(f"D: {d_line}")
    if c_line:
        parts.append(f"C: {c_line}")
    return "\n".join(parts)


def _apply_suptitle_style(style: GridStyle, suptitle_style: SuptitleStyle | None) -> None:
    """Apply suptitle style dict to a GridStyle instance (in-place).

    Accepts the project's SuptitleStyle alias (a mapping of specific
    suptitle-related keys to well-typed values). Non-dict or missing
    values are ignored.
    """
    if not isinstance(suptitle_style, dict):
        return
    st: SuptitleStyle = suptitle_style
    if "font_family" in st and isinstance(st["font_family"], str):
        style.suptitle.font_family = st["font_family"]
    if "font_serif" in st and isinstance(st["font_serif"], (list, tuple)):
        style.suptitle.font_serif = list(st["font_serif"])
    if "font_size" in st:
        val = st["font_size"]
        if isinstance(val, (int, float)):
            style.suptitle.font_size = int(val)
        elif isinstance(val, str):
            try:
                style.suptitle.font_size = int(float(val))
            except Exception:
                logger.warning(f"Invalid suptitle font_size value: {val!r} (expected numeric)")
    if "font_weight" in st and isinstance(st["font_weight"], str):
        style.suptitle.font_weight = st["font_weight"]
    if "color" in st and isinstance(st["color"], str):
        style.suptitle.color = st["color"]
    if "bbox_style" in st and isinstance(st["bbox_style"], dict):
        # bbox_style is expected to be a dict[str, str|int|float|bool]
        style.suptitle.bbox_style = dict(st["bbox_style"])


def get_highlight_params_from_axes(image_axes: list[Axes]) -> tuple[int, int, int, int]:
    """Derive highlight parameters from the first GT axis in a 3-row grid.

    Returns: (min_area, kernel_size, morph_iterations, circle_thickness)
    with the same widened/tuned values as in the original implementation.
    """
    if not image_axes or len(image_axes) < 3:
        raise RuntimeError("No image axes found for highlighting")
    sample_gt: NDArray[np.float32] = extract_final_rendered_image(image_axes[2])
    hparams: dict[str, int | tuple[int, int, int]] = derive_highlight_geometry(
        height_px=int(sample_gt.shape[0]), width_px=int(sample_gt.shape[1]), dpi=300.0
    )
    min_area: int = hparams["min_area"] * 20 if isinstance(hparams["min_area"], int) else hparams["min_area"][0] * 20
    kernel_size: int = hparams["kernel_size"] if isinstance(hparams["kernel_size"], int) else hparams["kernel_size"][0]
    morph_iterations: int = 3 * (
        hparams["morph_iterations"] if isinstance(hparams["morph_iterations"], int) else hparams["morph_iterations"][0]
    )
    circle_thickness: int = 12
    return min_area, kernel_size, morph_iterations, circle_thickness


def highlight_trirow_figure(fig: Figure, col_titles: list[str]) -> None:
    """Apply difference highlighting to both reconstruction rows of a 3-row grid figure."""
    image_axes: list[Axes] = [ax for ax in fig.axes if len(ax.get_images()) > 0]
    n_rows: int = 3
    n_cols: int = len(col_titles)
    min_area, kernel_size, morph_iterations, circle_thickness = get_highlight_params_from_axes(image_axes)
    for c in range(n_cols):
        if col_titles[c].startswith("Step 0"):
            continue
        for r in (0, 1):
            recon_ax_idx: int = c * n_rows + r
            gt_ax_idx: int = c * n_rows + 2
            if recon_ax_idx >= len(image_axes) or gt_ax_idx >= len(image_axes):
                continue
            recon_ax: Axes = image_axes[recon_ax_idx]
            gt_ax2: Axes = image_axes[gt_ax_idx]
            recon_img: NDArray[np.float32] = extract_final_rendered_image(recon_ax)
            base_img: NDArray[np.float32] = extract_final_rendered_image(gt_ax2)
            if recon_img.shape == base_img.shape:
                highlighted, _ = highlight_differences_with_contrast_fill(
                    recon_img,
                    base_img,
                    min_area=min_area,
                    kernel_size=kernel_size,
                    highlight_mode="first",
                    morph_iterations=morph_iterations,
                    circle_thickness=circle_thickness,
                    use_contrast_fill=True,
                    fallback_fill_color=(0, 150, 255),
                    fallback_alpha=0.35,
                )
                for img in recon_ax.get_images():
                    img.set_array(highlighted)


def _mean_std(sum_val: float, sq_sum: float, count: int) -> tuple[float, float]:
    """Compute mean and std from sums over `count` observations."""
    denom: int = max(1, count)
    mean: float = sum_val / denom
    var: float = max(0.0, (sq_sum / denom) - mean * mean)
    return mean, np.sqrt(var)


class EvaluateResults(TypedDict):
    """Results from evaluating the models over the entire test dataset."""

    overall_metrics: dict[str, float]
    variation_metrics: dict[str, dict[str, float]]
    episode_metrics: list[dict[str, float | int | str]]
    total_episodes: int
    total_batches: int


@dataclass
class ModelBundle:
    """A bundle of models for one type (discrete or continuous)."""

    encoder: nn.Module
    transition: nn.Module
    decoder: nn.Module
    alignment: nn.Module | None = None


def step_visual(
    *,
    viz_data_discrete: VisualizationData,
    viz_data_continuous: VisualizationData,
    row_labels: list[str],
    apply_highlighting: bool,
    metric_priority_keys: list[str] | None = None,
) -> None:
    """Create a 3-row per-step grid from two VisualizationData instances.

    Top row uses discrete reconstructions, middle row uses continuous reconstructions,
    bottom row uses base ground-truth images.
    """
    # Basic sanity: start from discrete column titles (we'll enrich them with both models' metrics)
    col_titles: list[str] = list(viz_data_discrete.col_titles)

    # Prepare row labels (always 3 rows: discrete, continuous, ground truth)
    row_labels = _normalize_trirow_labels(row_labels)

    # Convert images
    proc: ImageProcessor = ImageProcessor()

    # Step 0 and Step N images
    disc_step0_top: NDArray[np.float32] = proc.convert_for_matplotlib(viz_data_discrete.step0_top)
    disc_stepn_top: NDArray[np.float32] = proc.convert_for_matplotlib(viz_data_discrete.step_n_top)
    cont_step0_top: NDArray[np.float32] = proc.convert_for_matplotlib(viz_data_continuous.step0_top)
    cont_stepn_top: NDArray[np.float32] = proc.convert_for_matplotlib(viz_data_continuous.step_n_top)
    gt_step0: NDArray[np.float32] = proc.convert_for_matplotlib(
        viz_data_discrete.base_step0_bottom
        if viz_data_discrete.base_step0_bottom is not None
        else viz_data_discrete.step0_bottom
    )
    gt_stepn: NDArray[np.float32] = proc.convert_for_matplotlib(viz_data_discrete.step_n_bottom)

    # Build grid images as [row][col]
    imgs: list[list[NDArray[np.float32]]] = [
        [disc_step0_top, disc_stepn_top],
        [cont_step0_top, cont_stepn_top],
        [gt_step0, gt_stepn],
    ]

    # Use style from model tester visuals
    style: GridStyle = GridStyle.model_tester_style()

    # Suptitle/gap/style
    suptitle: str | None = viz_data_discrete.suptitle
    suptitle_gap: float = viz_data_discrete.suptitle_gap if viz_data_discrete.suptitle_gap is not None else 0.15
    _apply_suptitle_style(style, viz_data_discrete.suptitle_style)

    # Left panel: show variant starting state if applicable
    left_panel_image: NDArray[np.float32] | None = None
    left_panel_title: str | None = None
    if viz_data_discrete.variation_name != "base" and viz_data_discrete.base_step0_bottom is not None:
        left_panel_image = proc.convert_for_matplotlib(viz_data_discrete.step0_bottom)
        left_panel_title = viz_data_discrete.variant_panel_title or "Noisy starting state"

    # Compute Step 0 MSEs for both reconstructions vs base GT and place in first column title
    try:
        mse0_d = float(np.mean((disc_step0_top.astype(np.float32) - gt_step0.astype(np.float32)) ** 2))
        mse0_c = float(np.mean((cont_step0_top.astype(np.float32) - gt_step0.astype(np.float32)) ** 2))
        if col_titles:
            col_titles[0] = _format_dual_metrics_title(
                base_title="Step 0",
                dis_metrics={"reconstruction_mse": mse0_d},
                cont_metrics={"reconstruction_mse": mse0_c},
                priority_keys=metric_priority_keys,
            )
    except Exception:
        pass

    # Enrich the last column title with both models' step-N metrics if available
    if len(col_titles) >= 2:
        col_titles[-1] = _format_dual_metrics_title(
            base_title=col_titles[-1].split("\n")[0] if col_titles[-1] else f"Step {viz_data_discrete.step}",
            dis_metrics=viz_data_discrete.metrics_dict,
            cont_metrics=viz_data_continuous.metrics_dict,
            priority_keys=metric_priority_keys,
        )

    # Build a temporary grid to extract final rendered images for highlighting
    temp_fig = create_image_grid(
        imgs=imgs,
        row_labels=row_labels,
        col_titles=col_titles,
        padding_size=0.06,
        col_gap=0.04,
        frame_width_in=4.0,
        row_label_gap=1.0,
        col_title_gap=0.04,
        suptitle=suptitle,
        suptitle_gap=suptitle_gap,
        style=style,
        rightmost_col_row_labels=viz_data_discrete.rightmost_col_row_labels,
        rightmost_col_row_labels_side=getattr(viz_data_discrete, "rightmost_col_row_labels_side", "left"),
        left_panel_image=left_panel_image,
        left_panel_title=left_panel_title,
        left_panel_main_gap=0.1,
    )

    # Optional highlighting on both reconstruction rows (skip Step 0 column)
    if apply_highlighting:
        try:
            highlight_trirow_figure(temp_fig, col_titles)
        except Exception as e:  # A highlighting failure does not invalidate model metrics.
            logger.warning(f"3-row highlighting failed: {e}")

    # Save and close
    temp_fig.savefig(viz_data_discrete.save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)

    plt.close(temp_fig)


def trirow_summary_visual(
    *,
    episode_viz_data_discrete: list[VisualizationData],
    episode_viz_data_continuous: list[VisualizationData],
    variation_name: str,
    metric_value: float,
    metric_name: str,
    summary_type: str,
    output_dir: str,
    visualization_format: str,
    episode_number: int,
    apply_highlighting: bool,
    row_labels: list[str] | None,
    rightmost_col_row_labels: list[str] | None,
    suptitle_text: str | None,
    suptitle_style: SuptitleStyle | None,
    suptitle_gap: float | None,
    metric_priority_keys: list[str] | None = None,
) -> None:
    """Create a summary visualization with 3 rows across all steps for an episode.

    Uses a two-pass approach (temporary grid without left panel for consistent axis extraction,
    then a final grid with the optional left panel), mirroring the separate tester behavior.
    """
    assert len(episode_viz_data_discrete) == len(episode_viz_data_continuous), (
        "Discrete and continuous visualization lists must have the same length"
    )
    row_labels = _normalize_trirow_labels(row_labels)

    processor = ImageProcessor()

    # Build columns and titles
    col_titles: list[str] = ["Step 0"]
    images: list[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]] = []

    # Starting column (Step 0)
    disc0: NDArray[np.float32] = processor.convert_for_matplotlib(episode_viz_data_discrete[0].step0_top)
    cont0: NDArray[np.float32] = processor.convert_for_matplotlib(episode_viz_data_continuous[0].step0_top)
    gt0: NDArray[np.float32] = processor.convert_for_matplotlib(
        episode_viz_data_discrete[0].base_step0_bottom
        if episode_viz_data_discrete[0].base_step0_bottom is not None
        else episode_viz_data_discrete[0].step0_bottom
    )
    images.append((disc0, cont0, gt0))
    # Enrich Step 0 title with both D and C reconstruction MSEs vs base GT
    try:
        mse0_d = float(np.mean((disc0.astype(np.float32) - gt0.astype(np.float32)) ** 2))
        mse0_c = float(np.mean((cont0.astype(np.float32) - gt0.astype(np.float32)) ** 2))
        col_titles[0] = _format_dual_metrics_title(
            "Step 0", {"reconstruction_mse": mse0_d}, {"reconstruction_mse": mse0_c}, priority_keys=metric_priority_keys
        )
    except Exception:
        pass

    # Step columns (Step 1..N)
    for viz_disc, viz_cont in zip(episode_viz_data_discrete, episode_viz_data_continuous, strict=False):
        disc_n: NDArray[np.float32] = processor.convert_for_matplotlib(viz_disc.step_n_top)
        cont_n: NDArray[np.float32] = processor.convert_for_matplotlib(viz_cont.step_n_top)
        gt_n: NDArray[np.float32] = processor.convert_for_matplotlib(viz_disc.step_n_bottom)
        images.append((disc_n, cont_n, gt_n))
        # Title with both D and C metrics
        title = _format_dual_metrics_title(
            f"Step {viz_disc.step}", viz_disc.metrics_dict, viz_cont.metrics_dict, priority_keys=metric_priority_keys
        )
        col_titles.append(title)

    # Convert list-of-tuples into grid rows
    disc_row: list[NDArray[np.float32]] = [it[0] for it in images]
    cont_row: list[NDArray[np.float32]] = [it[1] for it in images]
    gt_row: list[NDArray[np.float32]] = [it[2] for it in images]
    imgs: list[list[NDArray[np.float32]]] = [disc_row, cont_row, gt_row]

    # Build suptitle
    if suptitle_text is None:
        display_variation_name: str = variation_name.replace("_", " ").title()

        metric_display: str = metric_name.replace("_", " ").title()
        match_type: str = "Best" if summary_type == "best" else ("Worst" if summary_type == "worst" else "Selected")
        suptitle: str = (
            f"{display_variation_name} {match_type} {metric_display} "
            f"({format_metric_value(metric_value, metric_name)}) - Episode {episode_number}"
        )
    else:
        suptitle = suptitle_text

    # Style
    style: GridStyle = GridStyle.model_tester_style()
    _apply_suptitle_style(style, suptitle_style)
    sgap: float = 0.15 if suptitle_gap is None else suptitle_gap

    # Build a temporary grid (no left panel) to extract final rendered images
    temp_fig: Figure = create_image_grid(
        imgs=imgs,
        row_labels=row_labels,
        col_titles=col_titles,
        padding_size=0.06,
        col_gap=0.04,
        frame_width_in=4.0,
        row_label_gap=0.001,  # summary style tighter
        col_title_gap=0.04,
        suptitle=suptitle,
        suptitle_gap=sgap,
        style=style,
        rightmost_col_row_labels=rightmost_col_row_labels,
    )

    # Optional highlighting for both reconstruction rows using ground truth row
    if apply_highlighting:
        try:
            highlight_trirow_figure(temp_fig, col_titles)
        except Exception as e:
            logger.warning(f"3-row summary highlighting failed: {e}")

    # Extract final rendered images for building the final grid (with optional left panel)
    image_axes: list[Axes] = [ax for ax in temp_fig.axes if len(ax.get_images()) > 0]
    # Build rendered tri-row images (column-major indexing)
    n_cols: int = len(col_titles)
    rendered_disc: list[NDArray[np.float32]] = []
    rendered_cont: list[NDArray[np.float32]] = []
    rendered_gt: list[NDArray[np.float32]] = []
    for c in range(n_cols):
        # row 0: discrete recon, row 1: continuous recon, row 2: ground truth
        idx_d: int = c * 3 + 0
        idx_c: int = c * 3 + 1
        idx_g: int = c * 3 + 2
        if idx_g >= len(image_axes) or idx_c >= len(image_axes) or idx_d >= len(image_axes):
            continue
        rendered_disc.append(extract_final_rendered_image(image_axes[idx_d]))
        rendered_cont.append(extract_final_rendered_image(image_axes[idx_c]))
        rendered_gt.append(extract_final_rendered_image(image_axes[idx_g]))

    plt.close(temp_fig)

    # Optional left panel for non-base variations
    left_panel_img: NDArray[np.float32] | None = None
    left_panel_title: str | None = None
    if episode_viz_data_discrete and (
        episode_viz_data_discrete[0].variation_name != "base"
        and episode_viz_data_discrete[0].base_step0_bottom is not None
    ):
        left_panel_img = ImageProcessor().convert_for_matplotlib(episode_viz_data_discrete[0].step0_bottom)
        left_panel_title = episode_viz_data_discrete[0].variant_panel_title or "Noisy starting state"

    # Build the final grid from highlighted rendered images.
    final_imgs: list[list[NDArray[np.float32]]] = [rendered_disc, rendered_cont, rendered_gt]
    final_fig: Figure = create_image_grid(
        imgs=final_imgs,
        row_labels=row_labels,
        col_titles=col_titles,
        padding_size=0.06,
        col_gap=0.04,
        frame_width_in=4.0,
        row_label_gap=0.001,
        col_title_gap=0.04,
        suptitle=suptitle,
        suptitle_gap=sgap,
        style=style,
        rightmost_col_row_labels=rightmost_col_row_labels,
        left_panel_image=left_panel_img,
        left_panel_title=left_panel_title,
        left_panel_main_gap=0.035,
    )

    save_path = str(
        pathlib.Path(output_dir) / f"summary_{summary_type}_ep{episode_number}_{variation_name}.{visualization_format}"
    )
    final_fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(final_fig)


# Combined tester


class CombinedModelTester:
    """Run discrete and continuous models together and generate 3-row visuals."""

    def __init__(
        self,
        discrete: ModelBundle,
        continuous: ModelBundle,
        *,
        device: str | torch.device = "cpu",
        output_dir: str = "./outputs",
        variations_to_use: list[str] | None = None,
        variations_to_ignore: list[str] | None = None,
        use_variation_for_all_states: bool = False,
        save_interval: int = 10,
        visualization_format: str = "png",
        visualization_episode_index: int = 0,
        visualization_steps: list[int] | None = None,
        log_interval: int = 5,
        apply_diff_highlighting: bool = False,
        metrics_to_save_discrete: list[str] | None = None,
        metrics_to_save_continuous: list[str] | None = None,
        row_labels: list[str] | None = None,
        variant_panel_title: str | None = None,
        suptitle_cfg: SuptitleConfig | None = None,
        column_metric_priority: list[str] | None = None,
    ) -> None:
        self.device: torch.device = torch.device(device) if isinstance(device, str) else device
        self.output_dir: str = output_dir
        self.variations_to_use: list[str] | None = variations_to_use
        self.variations_to_ignore: list[str] | None = variations_to_ignore
        self.use_variation_for_all_states: bool = use_variation_for_all_states
        self.save_interval: int = save_interval
        self.visualization_format: str = visualization_format
        self.visualization_episode_index: int = visualization_episode_index
        self.visualization_steps: list[int] | None = visualization_steps
        self.log_interval: int = log_interval
        self.apply_diff_highlighting: bool = apply_diff_highlighting

        # 3-row labels
        self.row_labels: list[str] = _normalize_trirow_labels(row_labels)
        self.variant_panel_title: str = variant_panel_title or "Noisy starting state"
        self.suptitle_cfg: SuptitleConfig | None = suptitle_cfg
        self.column_metric_priority: list[str] | None = column_metric_priority

        # Setup model bundles
        self.disc: ModelBundle = discrete
        self.cont: ModelBundle = continuous
        for m in (self.disc.encoder, self.disc.transition, self.disc.decoder):
            m.to(self.device, non_blocking=True).eval()
        if self.disc.alignment is not None:
            self.disc.alignment.to(self.device, non_blocking=True).eval()
        for m in (self.cont.encoder, self.cont.transition, self.cont.decoder):
            m.to(self.device, non_blocking=True).eval()
        if self.cont.alignment is not None:
            self.cont.alignment.to(self.device, non_blocking=True).eval()

        pathlib.Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        # Metrics trackers (separate, so JSON is clearly separated per model type)
        self.metrics_tracker_disc = MetricsTracker(
            str(pathlib.Path(self.output_dir) / "discrete"),
            metrics_to_save_discrete or ["reconstruction_mse", "eq_bit"],
        )
        self.metrics_tracker_cont = MetricsTracker(
            str(pathlib.Path(self.output_dir) / "continuous"),
            metrics_to_save_continuous or ["reconstruction_mse", "cosine_similarity"],
        )

    def _get_suptitle_style_dict(self) -> dict[str, SuptitleStyleValue] | None:
        cfg: SuptitleConfig | None = self.suptitle_cfg
        if cfg is None:
            return None

        style: dict[str, SuptitleStyleValue] = {}

        raw_font_family: str | None = cfg.font_family
        if raw_font_family is not None:
            style["font_family"] = raw_font_family
        raw_font_serif: list[str] | None = cfg.font_serif
        if raw_font_serif is not None:
            style["font_serif"] = raw_font_serif
        style["font_size"] = cfg.font_size
        style["font_weight"] = cfg.font_weight
        style["color"] = cfg.color
        raw_bbox: BBoxConfig = cfg.bbox_style
        style["bbox_style"] = {
            "facecolor": raw_bbox.facecolor,
            "edgecolor": raw_bbox.edgecolor,
            "boxstyle": raw_bbox.boxstyle,
            "pad": raw_bbox.pad,
            "alpha": raw_bbox.alpha,
            "linewidth": raw_bbox.linewidth,
        }

        return style

    def _get_suptitle_gap(self) -> float | None:
        cfg: SuptitleConfig | None = self.suptitle_cfg
        if cfg is None:
            return None

        raw: float = cfg.gap
        return raw

    def _resolve_step_suptitle(self, variation_name: str, step: int, episode: int) -> str | None:
        cfg: SuptitleConfig | None = self.suptitle_cfg
        if cfg is None:
            return None

        template: str | None = cfg.step_template
        if template is None:
            return None

        if not template:
            return ""

        return template.format(
            episode=episode,
            step=step,
            variation=variation_name,
            variation_display=variation_name.replace("_", " ").title(),
        )

    def _resolve_summary_suptitle(
        self, variation_name: str, summary_type: str, metric_name: str, metric_value_formatted: str, episode: int
    ) -> str | None:
        cfg: SuptitleConfig | None = self.suptitle_cfg
        if cfg is None:
            return None

        template: str | None = cfg.summary_template
        if template is None:
            return None

        if not template:
            return ""

        # Allow user overrides for summary type labels
        custom_labels: dict[str, str] | None = None
        labels: dict[str, str] | None = cfg.summary_type_labels
        if isinstance(labels, dict):
            custom_labels = dict(labels.items())

        summary_type_label: str | None = (
            custom_labels.get(summary_type, custom_labels.get(summary_type.lower()))
            if custom_labels
            else ("Best" if summary_type == "best" else ("Worst" if summary_type == "worst" else "Selected"))
        )
        return template.format(
            variation=variation_name,
            variation_display=variation_name.replace("_", " ").title(),
            summary_type=summary_type,
            summary_type_label=summary_type_label,
            metric_name=metric_name,
            metric_name_display=metric_name.replace("_", " ").title(),
            metric_value_formatted=metric_value_formatted,
            episode=episode,
        )

    @torch.inference_mode()
    def _encode_disc(self, state: Tensor) -> Tensor:
        enc = self.disc.alignment(state) if self.disc.alignment is not None else self.disc.encoder(state)
        return torch.round(enc)

    @torch.inference_mode()
    def _encode_cont(self, state: Tensor) -> Tensor:
        return self.cont.alignment(state) if self.cont.alignment is not None else self.cont.encoder(state)

    # Evaluation

    def evaluate_dataloader(self, dataloader: TestDataLoader) -> EvaluateResults:
        """Evaluate both models on every variation and generate three-row figures.

        Args:
            dataloader: Test batches grouped by render variation.

        Returns:
            Aggregate and per-variation evaluation results.
        """
        logger.info("Starting combined model evaluation (discrete+continuous)...")

        variation_info: VariationInfo = dataloader.get_variation_info()
        logger.info(
            f"Evaluating on {len(variation_info['variations'])} variations: {variation_info['variations']}\n"
            f"Ignored variations: {self.variations_to_ignore or 'None'}\n"
        )

        overall_episode_metrics: list[dict[str, float | int | str]] = []
        variation_metrics: dict[str, dict[str, float]] = {}
        total_episodes = 0
        total_batches = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task: TaskID = progress.add_task("Evaluating...", total=len(dataloader))

            for variation_name in variation_info["variations"]:
                logger.info(f"\nEvaluating variation: {variation_name}")

                # Reset metrics trackers per variation
                self.metrics_tracker_disc.reset_variation_data()
                self.metrics_tracker_cont.reset_variation_data()

                # Reset dataloader for this variation
                dataloader.reset_tracking(variation_name)
                per_variation_episode_metrics: list[dict[str, float | int | str]] = []

                for batch_idx, batch in enumerate(
                    dataloader(variation_name, use_variation_for_all_states=self.use_variation_for_all_states)
                ):
                    # Move tensors once
                    states: Tensor = batch["states"].to(self.device, non_blocking=True)
                    actions: Tensor = batch["actions"].to(self.device, non_blocking=True)
                    target_states: Tensor = batch["target_states"].to(self.device, non_blocking=True)
                    episode_indices: Tensor = batch["episode_indices"]

                    batch_metrics: dict[str, float | int | str] = self._evaluate_episode_batch(
                        states=states,
                        actions=actions,
                        target_states=target_states,
                        variation_name=variation_name,
                        episode_indices=episode_indices,
                        batch_idx=batch_idx,
                    )

                    per_variation_episode_metrics.append(batch_metrics)
                    overall_episode_metrics.append(batch_metrics)

                    total_batches += 1
                    total_episodes += len(episode_indices)
                    if batch_idx % self.log_interval == 0:
                        logger.info(f"Processed batch {batch_idx} for variation {variation_name}")
                    progress.update(task, advance=1)

                # Aggregate metrics for this variation
                variation_metrics[variation_name] = self._aggregate_variation_metrics(per_variation_episode_metrics)

                # Save per-variation JSONs for both trackers
                json_name_disc: str = f"{variation_name}_evaluation_metrics.json"
                json_name_cont: str = f"{variation_name}_evaluation_metrics.json"
                saved_disc: str = self.metrics_tracker_disc.save_variation_metrics_to_json(
                    variation_name, json_name_disc
                )
                saved_cont: str = self.metrics_tracker_cont.save_variation_metrics_to_json(
                    variation_name, json_name_cont
                )
                logger.info(f"Saved discrete metrics JSON to: {saved_disc}")
                logger.info(f"Saved continuous metrics JSON to: {saved_cont}")

        overall_metrics: dict[str, float] = self._aggregate_variation_metrics(overall_episode_metrics)

        logger.info("Combined evaluation completed")
        logger.info(f"Total episodes: {total_episodes}")
        logger.info(f"Total batches: {total_batches}")

        return {
            "overall_metrics": overall_metrics,
            "variation_metrics": variation_metrics,
            "episode_metrics": overall_episode_metrics,
            "total_episodes": total_episodes,
            "total_batches": total_batches,
        }

    @staticmethod
    def _aggregate_variation_metrics(episode_metrics_list: list[dict[str, float | int | str]]) -> dict[str, float]:
        if not episode_metrics_list:
            return {}
        aggregated: dict[str, float] = {}
        keys: set[str] = set()
        for m in episode_metrics_list:
            keys.update(m.keys())
        for k in keys:
            vals: list[float] = []
            for m in episode_metrics_list:
                v: float | int | str | None = m.get(k)
                if isinstance(v, (int, float, np.number)):
                    vals.append(float(v))
            if vals:
                aggregated[k] = float(np.mean(vals))
        return aggregated

    @torch.inference_mode()
    def _evaluate_episode_batch(
        self,
        *,
        states: Tensor,
        actions: Tensor,
        target_states: Tensor,
        variation_name: str,
        episode_indices: Tensor,
        batch_idx: int,
    ) -> dict[str, float | int | str]:
        """Standard evaluation logic for combined models with on-the-fly 3-row visuals."""
        batch_size, episode_len = states.shape[:2]

        # Accumulators for per-episode-per-step metrics (summed across the whole batch)
        mses_disc_sum = 0.0
        mses_disc_sq = 0.0
        mses_cont_sum = 0.0
        mses_cont_sq = 0.0
        eqbit_sum = 0.0
        eqbit_sq = 0.0
        cosine_sum = 0.0
        cosine_sq = 0.0
        obs_count = 0  # number of per-episode observations (batch_size * num_steps)

        # Prepare per-step visualization capture for a single selected episode index
        selected_disc_viz: list[VisualizationData] = []
        selected_cont_viz: list[VisualizationData] = []
        selected_ep_number: int | None = None

        # Track per-episode visualization data and metrics for best/worst summaries
        episode_viz_map_disc: dict[int, list[VisualizationData]] = {}
        episode_viz_map_cont: dict[int, list[VisualizationData]] = {}
        episode_metrics_map_disc: dict[int, list[float]] = {}

        batch_needs_visualization = batch_idx % 50 == 0
        suptitle_style: dict[str, SuptitleStyleValue] | None = self._get_suptitle_style_dict()
        suptitle_gap: float | None = self._get_suptitle_gap()

        current_state: Tensor = states[:, 0]
        # Encodings at step 0
        enc0_disc: Tensor = self._encode_disc(current_state)
        enc0_cont: Tensor = self._encode_cont(current_state)
        # Reconstructions for Step 0 (top rows)
        recon0_disc: Tensor = self.disc.decoder(enc0_disc)
        recon0_cont: Tensor = self.cont.decoder(enc0_cont)

        # Iterate over steps 1..N
        cur_enc_disc: Tensor = enc0_disc
        cur_enc_cont: Tensor = enc0_cont
        for step in range(episode_len - 1):
            action: Tensor = actions[:, step]
            # Predict next encodings
            next_enc_disc: Tensor = torch.round(self.disc.transition(cur_enc_disc, action))
            next_enc_cont: Tensor = self.cont.transition(cur_enc_cont, action)

            # Decode predictions
            recon_disc: Tensor = self.disc.decoder(next_enc_disc)
            recon_cont: Tensor = self.cont.decoder(next_enc_cont)

            # Ground-truth next state
            gt_next: Tensor = target_states[:, step + 1]

            # Per-episode MSE (vs BASE target)
            mse_disc_ep: Tensor = F.mse_loss(recon_disc, gt_next, reduction="none").view(batch_size, -1).mean(dim=1)
            mse_cont_ep: Tensor = F.mse_loss(recon_cont, gt_next, reduction="none").view(batch_size, -1).mean(dim=1)

            # Aggregate per-episode observations for accurate mean/std across the whole batch
            mses_disc_sum += float(mse_disc_ep.sum().item())
            mses_disc_sq += float((mse_disc_ep**2).sum().item())
            mses_cont_sum += float(mse_cont_ep.sum().item())
            mses_cont_sq += float((mse_cont_ep**2).sum().item())
            obs_count += batch_size

            mse_disc = float(mse_disc_ep.mean().item())
            mse_cont = float(mse_cont_ep.mean().item())

            # Compute the model-specific metrics recorded during evaluation.
            # Discrete: eq_bit percentage against target encoding
            tgt_enc_disc: Tensor = self._encode_disc(gt_next)
            eq_bits: Tensor = next_enc_disc == tgt_enc_disc
            # Per-episode equality percentage, then aggregate
            eq_bit_ep: Tensor = 100.0 * eq_bits.float().reshape(batch_size, -1).mean(dim=1)
            eqbit_sum += float(eq_bit_ep.sum().item())
            eqbit_sq += float((eq_bit_ep**2).sum().item())
            eq_bit = float(eq_bit_ep.mean().item())
            # Continuous: cosine similarity
            tgt_enc_cont: Tensor = self._encode_cont(gt_next)
            cosine_ep: Tensor = F.cosine_similarity(next_enc_cont, tgt_enc_cont, dim=1)
            cosine_sum += float(cosine_ep.sum().item())
            cosine_sq += float((cosine_ep**2).sum().item())
            cosine_sim = float(cosine_ep.mean().item())

            # Track on-the-fly metrics
            self.metrics_tracker_disc.update_step_metrics(step=step, reconstruction_mse=mse_disc, eq_bit=eq_bit)
            self.metrics_tracker_cont.update_step_metrics(
                step=step, reconstruction_mse=mse_cont, cosine_similarity=cosine_sim
            )

            # Visualization handling (every save_interval or requested step)
            if batch_needs_visualization and (
                (s := step + 1) % self.save_interval == 0
                or ((vs := self.visualization_steps) is not None and s in vs)
                or s == episode_len - 1
            ):
                # Per-episode metric arrays for titles and best/worst episode selection
                # Discrete: reconstruction MSE per episode
                per_ep_mse_disc: list[float] = mse_disc_ep.detach().cpu().tolist()
                # Continuous: reconstruction MSE per episode
                per_ep_mse_cont: list[float] = mse_cont_ep.detach().cpu().tolist()
                # Continuous: cosine similarity per episode
                per_ep_cos: list[float] = cosine_ep.detach().cpu().tolist()

                # Prepare VisualizationData instances for both models with identical meta
                # Note: we store the same GT images (BASE) for both to keep consistent comparison
                # Step-0 reconstructions are from enc0_*
                viz_disc = VisualizationData(
                    step=step + 1,
                    episode_number=int(episode_indices[self.visualization_episode_index].item()),
                    step0_bottom=current_state[self.visualization_episode_index].cpu().numpy(),
                    step0_top=recon0_disc[self.visualization_episode_index].cpu().numpy(),
                    step_n_bottom=gt_next[self.visualization_episode_index].cpu().numpy(),
                    step_n_top=recon_disc[self.visualization_episode_index].cpu().numpy(),
                    batch_idx=batch_idx,
                    output_dir=self.output_dir,
                    variation_name=variation_name,
                    metric=mse_disc,
                    metrics_dict={"reconstruction_mse": mse_disc, "eq_bit": eq_bit},
                    visualization_format=self.visualization_format,
                    row_labels=["Reconstruction", "Ground Truth"],  # not used in tri-row renderer
                    rightmost_col_row_labels=None,
                    base_step0_bottom=target_states[:, 0][self.visualization_episode_index].cpu().numpy(),
                    variant_panel_title=self.variant_panel_title if variation_name != "base" else None,
                    suptitle=self._resolve_step_suptitle(
                        variation_name, step + 1, int(episode_indices[self.visualization_episode_index].item())
                    ),
                    suptitle_style=suptitle_style,
                    suptitle_gap=suptitle_gap,
                )
                viz_cont = VisualizationData(
                    step=step + 1,
                    episode_number=int(episode_indices[self.visualization_episode_index].item()),
                    step0_bottom=current_state[self.visualization_episode_index].cpu().numpy(),
                    step0_top=recon0_cont[self.visualization_episode_index].cpu().numpy(),
                    step_n_bottom=gt_next[self.visualization_episode_index].cpu().numpy(),
                    step_n_top=recon_cont[self.visualization_episode_index].cpu().numpy(),
                    batch_idx=batch_idx,
                    output_dir=self.output_dir,
                    variation_name=variation_name,
                    metric=mse_cont,
                    metrics_dict={"reconstruction_mse": mse_cont, "cosine_similarity": cosine_sim},
                    visualization_format=self.visualization_format,
                    row_labels=["Reconstruction", "Ground Truth"],
                    rightmost_col_row_labels=None,
                    base_step0_bottom=target_states[:, 0][self.visualization_episode_index].cpu().numpy(),
                    variant_panel_title=self.variant_panel_title if variation_name != "base" else None,
                    suptitle=self._resolve_step_suptitle(
                        variation_name, step + 1, int(episode_indices[self.visualization_episode_index].item())
                    ),
                    suptitle_style=suptitle_style,
                    suptitle_gap=suptitle_gap,
                )

                # Save a combined 3-row grid
                step_visual(
                    viz_data_discrete=viz_disc,
                    viz_data_continuous=viz_cont,
                    row_labels=self.row_labels,
                    apply_highlighting=self.apply_diff_highlighting,
                    metric_priority_keys=self.column_metric_priority,
                )

                # Collect for summary of the selected episode (one episode number per variation)
                if selected_ep_number is None:
                    selected_ep_number = int(episode_indices[self.visualization_episode_index].item())
                if int(episode_indices[self.visualization_episode_index].item()) == selected_ep_number:
                    selected_disc_viz.append(viz_disc)
                    selected_cont_viz.append(viz_cont)

                # Collect per-episode viz data for best/worst summaries
                for eps_idx_in_batch in range(batch_size):
                    ep_num = int(episode_indices[eps_idx_in_batch].item())
                    # Discrete viz data per episode
                    v_disc = VisualizationData(
                        step=step + 1,
                        episode_number=ep_num,
                        step0_bottom=current_state[eps_idx_in_batch].cpu().numpy(),
                        step0_top=recon0_disc[eps_idx_in_batch].cpu().numpy(),
                        step_n_bottom=gt_next[eps_idx_in_batch].cpu().numpy(),
                        step_n_top=recon_disc[eps_idx_in_batch].cpu().numpy(),
                        batch_idx=batch_idx,
                        output_dir=self.output_dir,
                        variation_name=variation_name,
                        metric=per_ep_mse_disc[eps_idx_in_batch],
                        metrics_dict={"reconstruction_mse": per_ep_mse_disc[eps_idx_in_batch]},
                        visualization_format=self.visualization_format,
                        row_labels=["Reconstruction", "Ground Truth"],
                        rightmost_col_row_labels=None,
                        base_step0_bottom=target_states[:, 0][eps_idx_in_batch].cpu().numpy(),
                        variant_panel_title=self.variant_panel_title if variation_name != "base" else None,
                        suptitle=self._resolve_step_suptitle(variation_name, step + 1, ep_num),
                        suptitle_style=suptitle_style,
                        suptitle_gap=suptitle_gap,
                    )
                    # Continuous viz data per episode
                    v_cont = VisualizationData(
                        step=step + 1,
                        episode_number=ep_num,
                        step0_bottom=current_state[eps_idx_in_batch].cpu().numpy(),
                        step0_top=recon0_cont[eps_idx_in_batch].cpu().numpy(),
                        step_n_bottom=gt_next[eps_idx_in_batch].cpu().numpy(),
                        step_n_top=recon_cont[eps_idx_in_batch].cpu().numpy(),
                        batch_idx=batch_idx,
                        output_dir=self.output_dir,
                        variation_name=variation_name,
                        metric=per_ep_mse_cont[eps_idx_in_batch],
                        metrics_dict={
                            "reconstruction_mse": per_ep_mse_cont[eps_idx_in_batch],
                            "cosine_similarity": per_ep_cos[eps_idx_in_batch],
                        },
                        visualization_format=self.visualization_format,
                        row_labels=["Reconstruction", "Ground Truth"],
                        rightmost_col_row_labels=None,
                        base_step0_bottom=target_states[:, 0][eps_idx_in_batch].cpu().numpy(),
                        variant_panel_title=self.variant_panel_title if variation_name != "base" else None,
                        suptitle=self._resolve_step_suptitle(variation_name, step + 1, ep_num),
                        suptitle_style=suptitle_style,
                        suptitle_gap=suptitle_gap,
                    )

                    episode_viz_map_disc.setdefault(ep_num, []).append(v_disc)
                    episode_viz_map_cont.setdefault(ep_num, []).append(v_cont)
                    episode_metrics_map_disc.setdefault(ep_num, []).append(per_ep_mse_disc[eps_idx_in_batch])

            # Advance encodings
            cur_enc_disc = next_enc_disc
            cur_enc_cont = next_enc_cont

        num_steps: int = episode_len - 1

        disc_mean, disc_std = _mean_std(mses_disc_sum, mses_disc_sq, obs_count)
        cont_mean, cont_std = _mean_std(mses_cont_sum, mses_cont_sq, obs_count)
        eq_bit_mean, eq_bit_std = _mean_std(eqbit_sum, eqbit_sq, obs_count)
        cosine_mean, cosine_std = _mean_std(cosine_sum, cosine_sq, obs_count)

        # Finalize trackers for on-the-fly plotting
        self.metrics_tracker_disc.finalize_episode_batch(batch_size, variation_name)
        self.metrics_tracker_cont.finalize_episode_batch(batch_size, variation_name)

        # Save summary visualization for the selected episode per variation
        if selected_disc_viz and selected_cont_viz:
            selected_disc_sorted: list[VisualizationData] = sorted(selected_disc_viz, key=_visualization_step_key)
            selected_cont_sorted: list[VisualizationData] = sorted(selected_cont_viz, key=_visualization_step_key)

            # Use mean of per-step MSEs from discrete as the displayed metric (consistent with original)
            metric_vals: list[float] = [v.metric for v in selected_disc_sorted]
            metric_value: float = float(np.mean(metric_vals)) if metric_vals else 0.0

            trirow_summary_visual(
                episode_viz_data_discrete=selected_disc_sorted,
                episode_viz_data_continuous=selected_cont_sorted,
                variation_name=variation_name,
                metric_value=metric_value,
                metric_name="reconstruction_mse",
                summary_type="selected",
                output_dir=self.output_dir,
                visualization_format=self.visualization_format,
                episode_number=selected_disc_sorted[0].episode_number,
                apply_highlighting=self.apply_diff_highlighting,
                row_labels=self.row_labels,
                rightmost_col_row_labels=None,
                suptitle_text=self._resolve_summary_suptitle(
                    variation_name,
                    "selected",
                    "reconstruction_mse",
                    format_metric_value(metric_value, "reconstruction_mse"),
                    selected_disc_sorted[0].episode_number,
                ),
                suptitle_style=suptitle_style,
                suptitle_gap=suptitle_gap,
                metric_priority_keys=self.column_metric_priority,
            )

        # Save best and worst episode summaries (based on discrete reconstruction MSE)
        if episode_metrics_map_disc and episode_viz_map_disc and episode_viz_map_cont:
            # Compute mean metric per episode number
            means: dict[int, float] = {
                ep: float(np.mean(vals)) for ep, vals in episode_metrics_map_disc.items() if vals
            }
            if means:
                # pick episode id with smallest/largest mean value
                best_ep: int = min(means.items(), key=operator.itemgetter(1))[0]
                worst_ep: int = max(means.items(), key=operator.itemgetter(1))[0]

                # Save best
                disc_list: list[VisualizationData] = sorted(
                    episode_viz_map_disc.get(best_ep, []), key=_visualization_step_key
                )
                cont_list: list[VisualizationData] = sorted(
                    episode_viz_map_cont.get(best_ep, []), key=_visualization_step_key
                )
                if disc_list and cont_list:
                    mval = float(np.mean([v.metric for v in disc_list]))
                    trirow_summary_visual(
                        episode_viz_data_discrete=disc_list,
                        episode_viz_data_continuous=cont_list,
                        variation_name=variation_name,
                        metric_value=mval,
                        metric_name="reconstruction_mse",
                        summary_type="best",
                        output_dir=self.output_dir,
                        visualization_format=self.visualization_format,
                        episode_number=best_ep,
                        apply_highlighting=self.apply_diff_highlighting,
                        row_labels=self.row_labels,
                        rightmost_col_row_labels=None,
                        suptitle_text=self._resolve_summary_suptitle(
                            variation_name,
                            "best",
                            "reconstruction_mse",
                            format_metric_value(mval, "reconstruction_mse"),
                            best_ep,
                        ),
                        suptitle_style=suptitle_style,
                        suptitle_gap=suptitle_gap,
                        metric_priority_keys=self.column_metric_priority,
                    )

                # Save worst
                disc_list = sorted(episode_viz_map_disc.get(worst_ep, []), key=_visualization_step_key)
                cont_list = sorted(episode_viz_map_cont.get(worst_ep, []), key=_visualization_step_key)
                if disc_list and cont_list:
                    mval = float(np.mean([v.metric for v in disc_list]))
                    trirow_summary_visual(
                        episode_viz_data_discrete=disc_list,
                        episode_viz_data_continuous=cont_list,
                        variation_name=variation_name,
                        metric_value=mval,
                        metric_name="reconstruction_mse",
                        summary_type="worst",
                        output_dir=self.output_dir,
                        visualization_format=self.visualization_format,
                        episode_number=worst_ep,
                        apply_highlighting=self.apply_diff_highlighting,
                        row_labels=self.row_labels,
                        rightmost_col_row_labels=None,
                        suptitle_text=self._resolve_summary_suptitle(
                            variation_name,
                            "worst",
                            "reconstruction_mse",
                            format_metric_value(mval, "reconstruction_mse"),
                            worst_ep,
                        ),
                        suptitle_style=suptitle_style,
                        suptitle_gap=suptitle_gap,
                        metric_priority_keys=self.column_metric_priority,
                    )

        # Return per-batch metrics (aggregated at variation and overall later)
        return {
            "variation_name": variation_name,
            "reconstruction_mse_mean_discrete": disc_mean,
            "reconstruction_mse_std_discrete": disc_std,
            "reconstruction_mse_mean_continuous": cont_mean,
            "reconstruction_mse_std_continuous": cont_std,
            "eq_bit_mean_discrete": eq_bit_mean,
            "eq_bit_std_discrete": eq_bit_std,
            "cosine_similarity_mean_continuous": cosine_mean,
            "cosine_similarity_std_continuous": cosine_std,
            "num_steps": num_steps,
        }
