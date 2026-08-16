"""Image processing and visualization utilities for model testing."""

from __future__ import annotations

import contextlib
import io
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from spar.utils.viz_utils.highlighter import derive_highlight_geometry, highlight_differences_with_contrast_fill
from spar.utils.viz_utils.image_grid import GridStyle, create_image_grid
from spar.utils.viz_utils.percentage_formatting import format_percentage

if TYPE_CHECKING:
    from logging import Logger
    from typing import TypedDict

    from cv2.typing import MatLike
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from torch import Tensor

    from spar.testing.visualization.types import SuptitleStyle, SuptitleStyleValue, VisualizationData

    class HighlightParams(TypedDict, total=False):
        """Keyword parameters accepted by :func:`highlight_differences_with_contrast_fill`.

        Attributes:
            min_area: Minimum contour area to highlight.
            kernel_size: Morphological kernel size.
            morph_iterations: Number of morphology passes.
            circle_thickness: Highlight border thickness.
            highlight_mode: Which output images receive highlights.
            target_contrast_ratio: Desired highlight/background contrast.
            min_alpha: Minimum overlay opacity.
            max_alpha: Maximum overlay opacity.
            use_contrast_fill: Whether to compute per-region highlight colors.
            fallback_fill_color: Fill color used when contrast selection is disabled.
            fallback_alpha: Fill opacity used when contrast selection is disabled.
            contour_order: Contour ordering strategy.
        """

        min_area: int
        kernel_size: int
        morph_iterations: int
        circle_thickness: int
        highlight_mode: str
        target_contrast_ratio: float
        min_alpha: float
        max_alpha: float
        use_contrast_fill: bool
        fallback_fill_color: tuple[int, int, int]
        fallback_alpha: float
        contour_order: str


logger: Logger = getLogger(__name__)


def _visualization_step_key(value: VisualizationData) -> int:
    """Return the step number used to order visualization records."""
    return value.step


def format_metric_value(metric_value: float, metric_name: str) -> str:
    """Format metric value based on metric type and characteristics.

    Args:
        metric_value: The numeric metric value
        metric_name: The name of the metric to determine formatting

    Returns:
        Formatted string representation of the metric value
    """
    metric_name_lower: str = metric_name.lower()

    # Handle percentage-based metrics (typically values >= 1.0 for percentages)
    if any(keyword in metric_name_lower for keyword in ["percent", "eq", "bit"]) or metric_value >= 1.0:
        return format_percentage(metric_value, use_special_rounding=True)

    # Handle MSE and error metrics with scientific notation
    if any(keyword in metric_name_lower for keyword in ["mse", "error"]):
        # Always use human-readable scientific notation for MSE/error
        return _format_scientific_notation(metric_value)

    # Handle similarity metrics (typically 0-1 range)
    if "similarity" in metric_name_lower or "cosine" in metric_name_lower:
        return f"{metric_value:.4f}"

    # Handle distance metrics
    if "distance" in metric_name_lower or "l1" in metric_name_lower:
        if metric_value < 1e-3:
            return _format_scientific_notation(metric_value)

        return f"{metric_value:.6f}"

    # Default fallback formatting
    if metric_value >= 1.0:
        return format_percentage(metric_value, use_special_rounding=True)

    return f"{metric_value:.4f}"


def _format_scientific_notation(value: float) -> str:
    """Format a number in human-readable scientific notation.

    Args:
        value: The numeric value to format

    Returns:
        Human-readable scientific notation string (e.g., "1.23x10^-3")
    """
    if value == 0:
        return "0"

    # Get scientific notation components
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa: float = value / (10**exponent)

    # Format mantissa to 2 decimal places
    mantissa_str: str = f"{mantissa:.2f}"

    # Format exponent with proper superscript characters
    exp_str: str
    if exponent == 0:
        return mantissa_str

    if exponent > 0:
        exp_str = _format_superscript(exponent)
        return f"{mantissa_str}x10{exp_str}"

    exp_str = _format_superscript(abs(exponent))
    return f"{mantissa_str}x10^-{exp_str}"


def format_metric_value_for_filename(metric_value: float, metric_name: str) -> str:
    """Return a filename-safe ASCII representation of a metric value.

    This avoids Unicode superscripts, multiplication signs, and percent symbols
    so that generated filenames remain portable across filesystems.
    """
    _ = metric_name
    try:
        # The ``g`` format switches to exponent notation outside its fixed-point range.
        s = f"{metric_value:.6g}"
        # Normalize common problematic characters
        s = s.replace("+", "")
        s = s.replace("/", "_")
        s = s.replace(" ", "_")
        return s.replace("%", "pct")
    except Exception:
        # Fall back to ASCII conversion.
        return str(metric_value)


def _format_superscript(number: int) -> str:
    """Convert a number to superscript Unicode characters.

    Args:
        number: The number to convert to superscript

    Returns:
        String with superscript Unicode characters
    """
    superscript_map: dict[str, str] = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
    }
    return "".join(superscript_map[digit] for digit in str(number))


def _int_from_union(value: int | tuple[int, ...]) -> int:
    """Normalize a value that may be an int or a tuple of ints into a single int."""
    if isinstance(value, tuple):
        # Prefer the first element when a tuple is returned
        return value[0]
    return value


def _int_from_suptitle_value(value: SuptitleStyleValue) -> int:
    """Safely convert a SuptitleStyleValue to int when possible.

    Accepts int, float, or strings containing integer values. Raises ValueError
    if conversion is not possible.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError("suptitle font_size value is not int/float/str")


class ImageProcessor:
    """Convert image layouts and ranges for visualization."""

    @staticmethod
    def convert_for_matplotlib(img: Tensor | NDArray[np.float32 | np.uint8] | None) -> NDArray[np.float32]:
        """Convert image tensor/array to matplotlib-compatible format."""
        # Convert to numpy array
        img_np: NDArray[np.float32] = (
            img.cpu().detach().numpy() if isinstance(img, torch.Tensor) else np.asarray(img, dtype=np.float32)
        )

        # Transpose channels if needed (C, H, W) -> (H, W, C)
        if img_np.ndim == 3 and img_np.shape[0] in {1, 3, 4}:
            img_np = np.moveaxis(img_np, 0, -1)

        return np.clip(
            img_np.astype(np.float32, copy=False) * np.float32(1.0 / 255.0)
            if img_np.dtype in {np.uint8, np.uint16, np.long, np.int32, np.int64}
            else img_np,
            0,
            1,
        )


def create_step_visualization(viz_data: VisualizationData, *, apply_highlighting: bool = False) -> None:
    """Create layout with images using the image_grid utility.

    Args:
        viz_data: Visualization data for the current step
        apply_highlighting: Whether to apply highlighting to differences
    """
    col_titles: list[str] = viz_data.col_titles

    # If a metric is provided, append a formatted value to the step column title
    if viz_data.metrics_dict and len(col_titles) >= 2:
        # Use the first metric as the primary for display
        metric_name = next(iter(viz_data.metrics_dict.keys()))
        metric_val_raw = viz_data.metrics_dict[metric_name]
        metric_val = float(metric_val_raw) if isinstance(metric_val_raw, (int, float)) else viz_data.metric
        formatted_metric = format_metric_value(metric_val, metric_name)
        # Update the last column title (Step N)
        col_titles[-1] = f"{col_titles[-1]}\n({formatted_metric})"

    n_cols: int = len(col_titles)
    n_rows: int = 2

    # Define images
    processor = ImageProcessor()

    # Convert images for matplotlib
    img_pairs: list[tuple[NDArray[np.float32 | np.uint8], NDArray[np.float32 | np.uint8]]] = [
        (viz_data.step0_top, viz_data.step0_bottom),
        (viz_data.step_n_top, viz_data.step_n_bottom),
    ]
    processed_images = [
        (processor.convert_for_matplotlib(top), processor.convert_for_matplotlib(bottom)) for top, bottom in img_pairs
    ]

    # Create image grid data
    reconstruction_row, ground_truth_row = map(list, zip(*processed_images, strict=False))
    img_grid = [reconstruction_row, ground_truth_row]

    # Use the model_tester styling
    style: GridStyle = GridStyle.model_tester_style()
    # Apply optional suptitle style overrides
    st_local: SuptitleStyle | None = None
    if getattr(viz_data, "suptitle_style", None):
        st_local = viz_data.suptitle_style
    if st_local:
        if "font_family" in st_local:
            style.suptitle.font_family = str(st_local["font_family"])
        if "font_serif" in st_local and isinstance(st_local["font_serif"], (list, tuple)):
            style.suptitle.font_serif = list(st_local["font_serif"])
            if "font_size" in st_local:
                with contextlib.suppress(Exception):
                    style.suptitle.font_size = _int_from_suptitle_value(st_local["font_size"])
        if "font_weight" in st_local:
            style.suptitle.font_weight = str(st_local["font_weight"])
        if "color" in st_local:
            style.suptitle.color = str(st_local["color"])
        if "bbox_style" in st_local and isinstance(st_local["bbox_style"], dict):
            style.suptitle.bbox_style = dict(st_local["bbox_style"])
    vgap = getattr(viz_data, "suptitle_gap", None)
    suptitle_gap_val: float = 0.15 if vgap is None else float(vgap)

    # If variation is not base and base_step0_bottom is provided, extract variant for left panel
    left_panel_img = None
    left_panel_title = None
    if viz_data.variation_name != "base" and viz_data.base_step0_bottom is not None:
        # The current step0 images carry the variant. Show the variant bottom in the left panel,
        # and replace the main grid's bottom-left with the base version so the main grid compares base GT everywhere.
        left_panel_img = processor.convert_for_matplotlib(viz_data.step0_bottom)
        left_panel_title = viz_data.variant_panel_title or "Noisy starting state"
        # Replace bottom-left with base
        img_grid[1][0] = processor.convert_for_matplotlib(viz_data.base_step0_bottom)

    # Compute and display Step 0 metric (MSE) like other steps
    try:
        step0_recon = img_grid[0][0]
        step0_gt = img_grid[1][0]
        # processed_images and img_grid come from NumPy arrays, so compute MSE directly.
        mse0 = float(np.mean((step0_recon.astype(np.float32) - step0_gt.astype(np.float32)) ** 2))
        col_titles[0] = f"Step 0\n({format_metric_value(mse0, 'reconstruction_mse')})"
    except Exception:
        # If computation fails for any reason, leave the default title
        pass

    # Create the final grid
    fig = create_image_grid(
        imgs=img_grid,
        row_labels=viz_data.row_labels,
        col_titles=viz_data.col_titles,
        padding_size=0.06,
        col_gap=0.04,
        frame_width_in=4.0,
        row_label_gap=1,
        col_title_gap=0.04,
        suptitle=viz_data.suptitle,
        suptitle_gap=suptitle_gap_val,
        style=style,
        rightmost_col_row_labels=viz_data.rightmost_col_row_labels,
        rightmost_col_row_labels_side=getattr(viz_data, "rightmost_col_row_labels_side", "left"),
        left_panel_image=left_panel_img,
        left_panel_title=left_panel_title,
        # Gap between the left panel and the first main column (per-step visualization)
        left_panel_main_gap=0.1,
    )

    if apply_highlighting:
        # Get highlight parameters
        sample_image: NDArray[np.float32] = processed_images[0][1]
        height_px, width_px = sample_image.shape[:2]

        highlight_params_dict: dict[str, int | tuple[int, int, int]] = derive_highlight_geometry(
            height_px=height_px, width_px=width_px, dpi=300.0
        )

        # Convert geometry values to integers before scaling.
        min_area: int = _int_from_union(highlight_params_dict["min_area"]) * 20
        kernel_size: int = _int_from_union(highlight_params_dict["kernel_size"])
        morph_iterations: int = 3 * _int_from_union(highlight_params_dict["morph_iterations"])
        circle_thickness: int = 12

        # Find image axes in the grid
        image_axes: list[Axes] = [ax for ax in fig.axes if len(ax.get_images()) > 0]

        # Apply highlighting to each column except "Step 0"
        for col_idx in range(n_cols):
            if col_titles[col_idx] == "Step 0":
                continue

            recon_ax_idx: int = col_idx * n_rows + 0
            if recon_ax_idx >= len(image_axes):
                continue

            recon_ax: Axes = image_axes[recon_ax_idx]
            final_recon: NDArray[np.float32] = extract_final_rendered_image(recon_ax)

            # Get comparison image
            # Always use ground truth from the rendered grid for comparison
            gt_ax_idx = col_idx * n_rows + 1
            if gt_ax_idx < len(image_axes):
                comparison_image: NDArray[np.float32] = extract_final_rendered_image(image_axes[gt_ax_idx])
            else:
                continue

            # Apply highlighting and replace in grid
            try:
                if hasattr(comparison_image, "shape"):
                    # Resize the comparison image to the reconstruction dimensions.
                    comparison_resized = comparison_image
                    # Apply highlighting
                    highlighted_recon, _ = highlight_differences_with_contrast_fill(
                        final_recon,
                        comparison_resized,
                        min_area=min_area,
                        kernel_size=kernel_size,
                        highlight_mode="first",
                        morph_iterations=morph_iterations,
                        circle_thickness=circle_thickness,
                        use_contrast_fill=True,
                        fallback_fill_color=(0, 150, 255),
                        fallback_alpha=0.35,
                    )
                    # Replace the image in the axis
                    for img in recon_ax.get_images():
                        img.set_array(highlighted_recon)

            except Exception as e:
                logger.warning(f"Highlighting failed for column {col_idx}: {e!s}")

    # Save the figure
    fig.savefig(viz_data.save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def _highlight_summary_images(
    *,
    final_recon: NDArray[np.float32],
    final_gt: NDArray[np.float32],
    col_idx: int,
    min_area: int,
    kernel_size: int,
    morph_iterations: int,
    circle_thickness: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return highlighted images for one summary visualization column."""
    # Extract the base image from its grid placement before comparison.
    # rather than creating a new figure. This path uses the ground-truth image.
    # Always compare to ground truth, resizing if needed
    comparison_final: NDArray[np.float32] = final_gt

    # Verify images have the same shape before highlighting
    highlighted_recon: NDArray[np.float32]
    if comparison_final.shape == final_recon.shape:
        # Apply highlighting to show differences - only highlight reconstruction
        highlighted_recon, _ = highlight_differences_with_contrast_fill(
            final_recon,
            comparison_final,
            min_area=min_area,
            kernel_size=kernel_size,
            highlight_mode="first",  # Only highlight the reconstruction
            morph_iterations=morph_iterations,
            circle_thickness=circle_thickness,
            use_contrast_fill=True,
            fallback_fill_color=(50, 180, 200),
            fallback_alpha=0.35,
        )
        return highlighted_recon, final_gt

    # Log the specific error and fallback to original images
    logger.warning(
        f"Highlighting failed for column {col_idx}: "
        f"Input images must have the same shape. "
        f"Reconstruction: {final_recon.shape}, Comparison: {comparison_final.shape}. "
        f"Using original images."
    )
    return final_recon, final_gt


def create_summary_visualization(
    episode_viz_data: list[VisualizationData],
    variation_name: str,
    metric_value: float,
    metric_name: str,
    summary_type: str,
    output_dir: str,
    visualization_format: str = "png",
    episode_number: int = 0,
    apply_highlighting: bool = False,
    *,
    row_labels: list[str] | None = None,
    rightmost_col_row_labels: list[str] | None = None,
    suptitle_text: str | None = None,
    suptitle_style: SuptitleStyle | None = None,
    suptitle_gap: float | None = None,
) -> None:
    """Create summary visualization showing all steps for an episode.

    Args:
        episode_viz_data: List of visualization data for all steps of an episode.
        variation_name: Name of the variation.
        metric_value: The metric value for this episode.
        metric_name: Name of the metric.
        summary_type: Type of summary ("best", "worst", or "selected").
        output_dir: Directory where the visualization image will be saved.
        visualization_format: Format for saving visualizations (default is "png").
        episode_number: Episode number for this visualization.
        apply_highlighting: Whether to apply highlighting to differences in reconstructions vs ground truth.
        row_labels: Optional labels for the two rows (left side). Defaults to ["Reconstruction", "Ground Truth"].
        rightmost_col_row_labels: Optional labels to display adjacent to the right-most column.
        suptitle_text: Optional text override for the suptitle. If "", the suptitle is
            hidden.
        suptitle_style: Optional dict of style overrides for suptitle (font_family,
            font_serif, font_size, font_weight, color, bbox_style).
        suptitle_gap: Optional gap above column titles for the suptitle. Defaults to 0.15.
            when not provided.
    """
    if not episode_viz_data:
        return

    # Sort frames by step number.
    episode_viz_data = sorted(episode_viz_data, key=_visualization_step_key)

    # Process image format (reuse the same function as the original)
    processor = ImageProcessor()

    # Create image grid with all steps
    # First column is the starting state, then all other steps
    images: list[tuple[NDArray[np.uint8 | np.float32], NDArray[np.uint8 | np.float32]]] = []
    col_titles: list[str] = []

    # Add starting state (step 0)
    first_viz: VisualizationData = episode_viz_data[0]
    images.append((first_viz.step0_top, first_viz.step0_bottom))
    col_titles.append("Step 0")

    # Add all other steps with metric information
    for viz_data in episode_viz_data:
        images.append((viz_data.step_n_top, viz_data.step_n_bottom))

        # Prefer the per-step metric and fall back to viz_data.metric.
        metric_val = viz_data.metric
        metric_name_for_fmt: str = "metric"
        if viz_data.metrics_dict and len(viz_data.metrics_dict) > 0:
            # Use the first key/value pair to determine formatting for this column
            first_key = next(iter(viz_data.metrics_dict.keys()))
            metric_name_for_fmt = first_key
            raw_val = viz_data.metrics_dict[first_key]
            if isinstance(raw_val, (int, float)):
                metric_val = float(raw_val)

        # Format metric based on type
        formatted_metric = format_metric_value(metric_val, metric_name_for_fmt)

        title: str = f"Step {viz_data.step}\n({formatted_metric})"
        col_titles.append(title)

    # Create image grid
    # Format: [row][col] where row 0 = reconstruction, row 1 = ground truth

    # Process all images and create initial grid
    processed_images: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = [
        (processor.convert_for_matplotlib(top_img), processor.convert_for_matplotlib(bottom_img))
        for top_img, bottom_img in images
    ]

    # Create initial image grid to get final rendered versions
    reconstruction_row: list[NDArray[np.float32]] = [reconstruction for reconstruction, _ in processed_images]
    ground_truth_row: list[NDArray[np.float32]] = [ground_truth for _, ground_truth in processed_images]
    img_grid: list[list[NDArray[np.float32]]] = [reconstruction_row, ground_truth_row]

    # Create labels
    if row_labels is None:
        row_labels = ["Reconstruction", "Ground Truth"]

    # Suptitle resolution with optional overrides
    if suptitle_text is None:
        # Create default suptitle with appropriate metric formatting
        display_variation_name: str = variation_name.replace("_", " ").title()
        formatted_metric_value: str = format_metric_value(metric_value, metric_name)
        if summary_type == "worst":
            match_type: str = "Worst"
        elif summary_type == "selected":
            match_type = "Selected"
        else:
            match_type = "Best"
        metric_display_name: str = metric_name.replace("_", " ").title()
        suptitle: str | None = (
            f"{display_variation_name} {match_type} {metric_display_name} "
            f"({formatted_metric_value}) - Episode {episode_number}"
        )
    else:
        suptitle = suptitle_text  # can be "" to disable

    style = GridStyle.model_tester_style()
    # Optional suptitle style overrides
    if suptitle_style:
        suptitle_overrides: SuptitleStyle = suptitle_style
        if "font_family" in suptitle_overrides:
            style.suptitle.font_family = str(suptitle_overrides["font_family"])
        if "font_serif" in suptitle_overrides and isinstance(suptitle_overrides["font_serif"], (list, tuple)):
            style.suptitle.font_serif = list(suptitle_overrides["font_serif"])
            if "font_size" in suptitle_overrides:
                with contextlib.suppress(Exception):
                    style.suptitle.font_size = _int_from_suptitle_value(suptitle_overrides["font_size"])
        if "font_weight" in suptitle_overrides:
            style.suptitle.font_weight = str(suptitle_overrides["font_weight"])
        if "color" in suptitle_overrides:
            style.suptitle.color = str(suptitle_overrides["color"])
        if "bbox_style" in suptitle_overrides and isinstance(suptitle_overrides["bbox_style"], dict):
            style.suptitle.bbox_style = dict(suptitle_overrides["bbox_style"])
    suptitle_gap_val2: float = 0.15 if suptitle_gap is None else suptitle_gap

    # Create temporary image grid to get final rendered images
    # If the first viz is from a non-base variation and carries base_step0_bottom, prepare left panel and
    # replace the bottom-left (starting state's GT) with base while the left panel shows the variant image.
    left_panel_img: NDArray[np.float32] | None = None
    left_panel_title: str | None = None
    if first_viz.variation_name != "base" and getattr(first_viz, "base_step0_bottom", None) is not None:
        processor2 = ImageProcessor()
        left_panel_img = processor2.convert_for_matplotlib(first_viz.step0_bottom)
        left_panel_title = first_viz.variant_panel_title or "Noisy starting state"
        # Replace bottom-left with base in the processed grid
        processed_images[0] = (processed_images[0][0], processor2.convert_for_matplotlib(first_viz.base_step0_bottom))
        # Rebuild the main grid rows to reflect the replacement in the actual grid
        reconstruction_row = [reconstruction for reconstruction, _ in processed_images]
        ground_truth_row = [ground_truth for _, ground_truth in processed_images]
        img_grid = [reconstruction_row, ground_truth_row]

    # Compute Step 0 metric (MSE) and place it in the first column title
    try:
        step0_recon = processed_images[0][0]
        step0_gt = processed_images[0][1]
        mse0 = float(np.mean((step0_recon.astype(np.float32) - step0_gt.astype(np.float32)) ** 2))
        if col_titles:
            col_titles[0] = f"Step 0\n({format_metric_value(mse0, 'reconstruction_mse')})"
    except Exception:
        pass

    # Omit the left panel so the temporary grid has one image axis per input.
    # Match n_cols * n_rows so extraction and highlighting use the same indices.
    temp_fig: Figure = create_image_grid(
        imgs=img_grid,
        row_labels=row_labels,
        col_titles=col_titles,
        padding_size=0.06,
        col_gap=0.04,
        frame_width_in=4.0,
        row_label_gap=1,
        col_title_gap=0.04,
        suptitle=suptitle,
        suptitle_gap=suptitle_gap_val2,
        style=style,
        rightmost_col_row_labels=rightmost_col_row_labels,
        # Do NOT include left panel in this temporary grid (prevents axis count mismatch)
    )

    # Extract final rendered images and prepare base images for highlighting
    highlighted_images: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = []
    n_cols = len(col_titles)
    n_rows = 2  # Always 2 rows: reconstruction and ground truth

    # Find image axes (those that contain actual images, not frame axes)
    image_axes: list[Axes] = [ax for ax in temp_fig.axes if len(ax.get_images()) > 0]

    # Validate that we have the expected number of axes
    expected_axes = n_cols * n_rows
    if len(image_axes) != expected_axes:
        logger.warning(
            f"Expected {expected_axes} image axes but found {len(image_axes)}. "
            f"This may cause incorrect axis indexing in summary visualizations."
        )

    # We always compare reconstruction to the ground truth image rendered from the grid.

    for col_idx in range(n_cols):
        # For column-major order: col_idx * n_rows + row_idx
        recon_ax_idx = col_idx * n_rows + 0  # First row (reconstruction)
        gt_ax_idx = col_idx * n_rows + 1  # Second row (ground truth)

        # Bounds checking to prevent index errors
        if recon_ax_idx >= len(image_axes) or gt_ax_idx >= len(image_axes):
            logger.error(
                f"Axis index out of bounds: trying to access axes {recon_ax_idx}, {gt_ax_idx} "
                f"but only {len(image_axes)} axes available. Skipping column {col_idx}."
            )
            continue

        recon_ax = image_axes[recon_ax_idx]
        gt_ax = image_axes[gt_ax_idx]

        # Extract final rendered images
        final_recon: NDArray[np.float32] = extract_final_rendered_image(recon_ax)
        final_gt: NDArray[np.float32] = extract_final_rendered_image(gt_ax)

        # Get image dimensions from numpy array shape
        height_px, width_px = final_gt.shape[:2]
        highlight_params: dict[str, int | tuple[int, int, int]] = derive_highlight_geometry(
            height_px=height_px, width_px=width_px, dpi=300.0
        )

        # Extract parameters
        min_area: int = _int_from_union(highlight_params["min_area"]) * 20
        kernel_size: int = _int_from_union(highlight_params["kernel_size"])
        morph_iterations: int = _int_from_union(highlight_params["morph_iterations"])
        circle_thickness: int = 12  # highlight_params["circle_thickness"]

        # Skip highlighting for the "Step 0" column (first column)
        # Also check for quality metrics for this specific step
        step_quality_metric: float | None = None
        if col_titles[col_idx] != "Step 0" and col_idx > 0:
            # For non-starting states, get quality metric from the corresponding viz_data
            step_viz_data: VisualizationData = episode_viz_data[
                col_idx - 1
            ]  # -1 because starting state is not in episode_viz_data
            # Try to get any available quality metric from metrics_dict
            if step_viz_data.metrics_dict:
                # Look for common quality metrics (could be eq_bit, cosine_similarity, etc.)
                for quality_key in ["step_bit_equality", "eq_bit", "cosine_similarity", "similarity"]:
                    if quality_key in step_viz_data.metrics_dict:
                        try:
                            step_quality_metric = float(step_viz_data.metrics_dict[quality_key])
                        except Exception:
                            step_quality_metric = None
                        break

        # Apply highlighting based on quality metric and the apply_highlighting flag
        should_apply_highlighting: bool = (
            apply_highlighting
            and col_titles[col_idx] != "Step 0"
            and (
                step_quality_metric is None
                or (step_quality_metric >= 1.0 and step_quality_metric < 100.0)  # Percentage-based metrics
                or (step_quality_metric < 1.0 and step_quality_metric < 0.99)  # Decimal-based metrics
            )
        )

        # Apply highlighting to final rendered images
        if should_apply_highlighting:
            try:
                highlighted_images.append(
                    _highlight_summary_images(
                        final_recon=final_recon,
                        final_gt=final_gt,
                        col_idx=col_idx,
                        min_area=min_area,
                        kernel_size=kernel_size,
                        morph_iterations=morph_iterations,
                        circle_thickness=circle_thickness,
                    )
                )
            except (ValueError, RuntimeError, TypeError) as e:
                # Log the specific error and fallback to original images
                error_msg = str(e)
                if "Input images must have the same shape" in error_msg:
                    logger.warning(
                        f"Highlighting failed for column {col_idx}: {error_msg}. "
                        f"Reconstruction: {final_recon.shape}, Ground truth: {final_gt.shape}. "
                        f"Using original images."
                    )
                else:
                    logger.warning(f"Highlighting failed for column {col_idx}: {error_msg}. Using original images.")

                highlighted_images.append((final_recon, final_gt))

        else:
            # Don't apply highlighting for "Step 0" column or when bit equality is 100%
            highlighted_images.append((final_recon, final_gt))

    plt.close(temp_fig)

    # Create final image grid with highlighted versions
    reconstruction_row = [pair[0] for pair in highlighted_images]
    ground_truth_row = [pair[1] for pair in highlighted_images]
    img_grid = [reconstruction_row, ground_truth_row]

    # Create titles (row_labels already set above)

    # Create suptitle for summary with dynamic formatting
    display_variation_name = variation_name.replace("_", " ").title()
    match_type = "Worst" if summary_type == "worst" else "Best"
    metric_display_name = metric_name.replace("_", " ").title()

    # Format the metric value using the same formatting function
    formatted_metric_value = format_metric_value(metric_value, metric_name)

    # Resolve final suptitle and style again for final grid
    if suptitle_text is None:
        suptitle = (
            f"{display_variation_name} {match_type} {metric_display_name} "
            f"({formatted_metric_value}) - Episode {episode_number}"
        )
    else:
        suptitle = suptitle_text
    style = GridStyle.model_tester_style()
    if suptitle_style:
        st2: SuptitleStyle = suptitle_style
        if "font_family" in st2:
            style.suptitle.font_family = str(st2["font_family"])
        if "font_serif" in st2 and isinstance(st2["font_serif"], (list, tuple)):
            style.suptitle.font_serif = list(st2["font_serif"])
        if "font_size" in st2:
            with contextlib.suppress(Exception):
                style.suptitle.font_size = _int_from_suptitle_value(st2["font_size"])
        if "font_weight" in st2:
            style.suptitle.font_weight = str(st2["font_weight"])
        if "color" in st2:
            style.suptitle.color = str(st2["color"])
        if "bbox_style" in st2 and isinstance(st2["bbox_style"], dict):
            style.suptitle.bbox_style = dict(st2["bbox_style"])
    suptitle_gap_val3: float = 0.15 if suptitle_gap is None else suptitle_gap

    # Create the final image grid
    fig: Figure = create_image_grid(
        imgs=img_grid,
        row_labels=row_labels,
        col_titles=col_titles,
        padding_size=0.06,
        col_gap=0.04,
        frame_width_in=4.0,
        row_label_gap=0.001,
        col_title_gap=0.04,
        suptitle=suptitle,
        suptitle_gap=suptitle_gap_val3,
        style=style,
        rightmost_col_row_labels=rightmost_col_row_labels,
        left_panel_image=left_panel_img,
        left_panel_title=left_panel_title,
        # Slightly increase gap between left panel and first column for better label spacing
        left_panel_main_gap=0.035,
    )

    # Save the figure
    save_path = Path(output_dir) / f"summary_{summary_type}_ep{episode_number}_{variation_name}.{visualization_format}"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def extract_final_rendered_image(ax: Axes) -> NDArray[np.float32]:
    """Extract the final rendered image from a matplotlib axis.

    Rendering the axis applies its Matplotlib transforms before extraction.

    Args:
        ax: Matplotlib axis containing the image

    Returns:
        Final rendered image as float32 array [0,1] with shape (H,W,3)
    """
    # Matplotlib Figure-like objects may return an array or a sequence from
    # get_size_inches(). Normalize either result to a pair of floats.
    figsize: tuple[float, float]
    get_size = getattr(ax.figure, "get_size_inches", None)
    if callable(get_size):
        size_val = get_size()
        # Normalize an array or sequence to two floats.
        figsize = (
            (float(size_val[0]), float(size_val[1]))
            if (isinstance(size_val, np.ndarray) and size_val.size >= 2)
            or (isinstance(size_val, (list, tuple)) and len(size_val) >= 2)
            else (8.0, 6.0)
        )
    else:
        figsize = (8.0, 6.0)

    temp_fig: Figure = plt.figure(figsize=figsize)
    temp_ax: Axes = temp_fig.add_subplot(111)

    # Copy the image data
    for image in ax.get_images():
        tmp_img_array: NDArray[np.float32] = np.asarray(image.get_array(), dtype=np.float32)
        temp_ax.imshow(tmp_img_array, extent=image.get_extent())

    temp_ax.axis("off")
    temp_ax.set_aspect("equal")

    # Render to buffer
    buf = io.BytesIO()
    temp_fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=300)
    buf.seek(0)

    # Load as numpy array
    img_array: NDArray[np.uint8] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    final_img_opt: MatLike | None = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if final_img_opt is None:
        raise ValueError("Could not decode rendered image buffer.")
    final_img: MatLike = final_img_opt
    final_img_rgb: MatLike = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    plt.close(temp_fig)
    buf.close()

    return final_img_rgb.astype(np.float32) / 255.0


def apply_highlighting_to_final_images(
    temp_fig: Figure, col_titles: list[str], highlight_params: HighlightParams | None
) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
    """Apply highlighting to final rendered images from a matplotlib figure.

    Always compares reconstructions to the final rendered ground truth (base) images.
    """
    # Find the image axes (skip frame axes)
    image_axes = [ax for ax in temp_fig.axes if len(ax.get_images()) > 0]
    n_cols = len(col_titles)
    n_rows = 2  # Always 2 rows: reconstruction and ground truth

    reconstruction_row: list[NDArray[np.float32]] = []
    ground_truth_row: list[NDArray[np.float32]] = []

    for col_idx in range(n_cols):
        # For column-major order: col_idx * n_rows + row_idx
        recon_ax: Axes = image_axes[col_idx * n_rows + 0]  # First row (reconstruction)
        gt_ax: Axes = image_axes[col_idx * n_rows + 1]  # Second row (ground truth)

        # Extract final rendered images
        final_recon: NDArray[np.float32] = extract_final_rendered_image(recon_ax)
        final_gt: NDArray[np.float32] = extract_final_rendered_image(gt_ax)

        reconstruction_row.append(final_recon)
        ground_truth_row.append(final_gt)

    # Skip highlighting if no parameters provided
    if highlight_params is None:
        return reconstruction_row, ground_truth_row

    highlighted_reconstruction: list[NDArray[np.float32]] = []
    for idx, recon_img in enumerate(reconstruction_row):
        # Skip first column (Step 0)
        if idx == 0 or col_titles[idx] == "Step 0":
            highlighted_reconstruction.append(recon_img)
            continue

        base_img: NDArray[np.float32] = ground_truth_row[idx]
        if recon_img.shape == base_img.shape:
            try:
                highlighted, _ = highlight_differences_with_contrast_fill(recon_img, base_img, **highlight_params)
                highlighted_reconstruction.append(highlighted)
                continue

            except Exception:
                pass

        # Fallback
        highlighted_reconstruction.append(recon_img)

    return highlighted_reconstruction, ground_truth_row
