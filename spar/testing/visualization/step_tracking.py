"""Track and render the highest- and lowest-scoring evaluation steps."""

from __future__ import annotations

import heapq
from logging import getLogger
import operator
import pathlib
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from spar.testing.visualization.image_utils import (
    ImageProcessor,
    _format_scientific_notation,
    extract_final_rendered_image,
    format_metric_value,
    format_metric_value_for_filename,
)
from spar.utils.viz_utils.highlighter import derive_highlight_geometry, highlight_differences_with_contrast_fill
from spar.utils.viz_utils.image_grid import GridStyle, create_image_grid

if TYPE_CHECKING:
    from logging import Logger

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from spar.testing.visualization.types import GridData, StepEntry


logger: Logger = getLogger(__name__)


class BestWorstStepTracker:
    """Tracks the top-k best and worst steps for end-to-end + use_variation_for_all_states evaluation."""

    def __init__(
        self,
        top_k: int = 5,
        output_dir: str = "./outputs",
        visualization_format: str = "png",
        apply_highlighting: bool = False,
        *,
        suptitle_style: dict[
            str, (str | int | float | list[str] | tuple[str, ...] | dict[str, str | int | float | bool])
        ]
        | None = None,
        suptitle_gap: float | None = None,
        step_tracker_template: str | None = None,
        rightmost_col_row_labels: list[str] | None = None,
        rightmost_col_row_labels_side: str = "right",
    ) -> None:
        """Initialize the step tracker.

        Args:
            top_k: Number of best and worst steps to track
            output_dir: Directory to save visualizations
            visualization_format: Format for saving visualizations
            apply_highlighting: Whether to apply highlighting to differences
            suptitle_style: Optional style overrides for suptitle
            suptitle_gap: Optional gap size for suptitle
            step_tracker_template: Optional template for suptitle formatting
            rightmost_col_row_labels: Optional row labels for the rightmost column when rendering grids
            rightmost_col_row_labels_side: Where to place the rightmost column's row labels ("left" or "right")
        """
        self.top_k: int = top_k
        self.output_dir: str = output_dir
        self.visualization_format: str = visualization_format
        self.apply_highlighting: bool = apply_highlighting
        self.suptitle_style = suptitle_style
        self.suptitle_gap = suptitle_gap
        self.step_tracker_template = step_tracker_template

        # Copy lists to avoid accidental external mutation
        self.rightmost_col_row_labels = list(rightmost_col_row_labels) if rightmost_col_row_labels is not None else None
        self.rightmost_col_row_labels_side = rightmost_col_row_labels_side

        # Create output directory for step visualizations
        self.step_viz_dir = str(pathlib.Path(self.output_dir) / "best_worst_steps")
        pathlib.Path(self.step_viz_dir).mkdir(exist_ok=True, parents=True)

        # Each top-k heap entry stores encoded_metric, counter, and StepEntry.
        self.best_steps: list[tuple[float, int, StepEntry]] = []
        self.worst_steps: list[tuple[float, int, StepEntry]] = []

        # Track current variation for file naming
        self.current_variation: str = ""
        self.current_batch_idx: int = 0

        # A monotonic counter makes heap entries unique without comparing dictionaries.
        self._counter: int = 0

        # Initialize image processor
        self.image_processor = ImageProcessor()

    def update_step(
        self,
        step: int,
        batch_idx: int,
        episode_number: int,
        metric: float,
        is_higher_better: bool,
        step_data: GridData,
        metric_name: str | None = None,
    ) -> None:
        """Update tracking with a new step, maintaining only the top-k entries.

        Args:
            step: Step number.
            batch_idx: Batch index.
            episode_number: Episode number.
            metric: Metric value for this step.
            is_higher_better: Indicates if higher metric values are better.
            step_data: Complete visualization data for this step.
            metric_name: Name of the metric used in titles. Defaults to ``metric``.
        """
        # Create a compact step entry
        step_entry: StepEntry = {
            "step": step,
            "batch_idx": batch_idx,
            "episode_number": episode_number,
            "metric": metric,
            "metric_name": metric_name or "metric",
            "is_higher_better": is_higher_better,
            "step_data": step_data,
        }

        # Use the same counter value to order entries in both heaps.
        self._counter += 1
        counter: int = self._counter

        # Determine sign encoding for best and worst depending on direction of optimality
        # For "higher is better": best_sign = 1.0, worst_sign = -1.0.
        # For "lower is better": best_sign = -1.0, worst_sign = 1.0.
        best_sign: float = 1.0 if is_higher_better else -1.0
        worst_sign: float = -best_sign

        # Encode the metric using the corresponding sign for best and worst
        best_value: float = best_sign * metric
        worst_value: float = worst_sign * metric

        # Update heaps using the common helper to avoid code duplication
        self._update_heap(self.best_steps, best_value, counter, step_entry)
        self._update_heap(self.worst_steps, worst_value, counter, step_entry)

    def _update_heap(
        self, heap: list[tuple[float, int, StepEntry]], value: float, counter: int, step_entry: StepEntry
    ) -> None:
        """Push or replace an entry in a heap based on its encoded metric value.

        The heap stores tuples of the form (encoded_metric, counter, step_entry) and maintains
        the smallest encoded value at the root. This helper retains only the top-k entries
        remain in the heap by replacing the root when appropriate.

        Args:
            heap: The heap to update.
            value: The encoded metric value (may be signed).
            counter: The unique counter associated with the entry.
            step_entry: The step entry payload.
        """
        if len(heap) < self.top_k:
            heapq.heappush(heap, (value, counter, step_entry))
        # Only replace the root if the new value is larger, which corresponds to a better
        # candidate for the particular heap's purpose (best or worst).
        elif value > heap[0][0]:
            heapq.heapreplace(heap, (value, counter, step_entry))

    def set_variation(self, variation_name: str) -> None:
        """Set the current variation name for file naming."""
        self.current_variation = variation_name.replace(" ", "_").replace("/", "_")

    def set_batch_idx(self, batch_idx: int) -> None:
        """Set the current batch index for file naming."""
        self.current_batch_idx = batch_idx

    def save_best_worst_visualizations(self) -> None:
        """Save visualizations for the top-k best and worst steps."""
        if not self.current_variation:
            logger.warning("No variation set for step tracker")
            return

        # Sort best by true metric (descending if higher is better, ascending otherwise)
        best_sorted_entries: list[StepEntry] = [t[2] for t in self.best_steps]
        if best_sorted_entries:
            is_higher_better_flag = best_sorted_entries[0]["is_higher_better"]
            best_sorted_entries.sort(key=operator.itemgetter("metric"), reverse=is_higher_better_flag)
        for i, step_entry in enumerate(best_sorted_entries):
            self._save_step_visualization(
                step_entry["step_data"],
                f"best_{i + 1:02d}",
                step_entry["step"],
                step_entry["episode_number"],
                step_entry["metric"],
                step_entry["metric_name"],
            )

        # Sort worst by true metric (ascending if higher is better, descending otherwise)
        worst_sorted_entries: list[StepEntry] = [t[2] for t in self.worst_steps]
        if worst_sorted_entries:
            is_higher_better_flag = worst_sorted_entries[0]["is_higher_better"]
            worst_sorted_entries.sort(key=operator.itemgetter("metric"), reverse=not is_higher_better_flag)
        for i, step_entry in enumerate(worst_sorted_entries):
            self._save_step_visualization(
                step_entry["step_data"],
                f"worst_{i + 1:02d}",
                step_entry["step"],
                step_entry["episode_number"],
                step_entry["metric"],
                step_entry["metric_name"],
            )

        logger.info(
            f"Saved {len(best_sorted_entries)} best and {len(worst_sorted_entries)} worst step "
            f"visualizations for {self.current_variation}"
        )

    def _save_step_visualization(
        self, viz_data: GridData, rank_prefix: str, step: int, episode_number: int, metric: float, metric_name: str
    ) -> None:
        """Save a single step visualization with all required components."""
        # Create filename
        # Use human-readable scientific for MSE-like metrics
        # Use filename-safe metric formatting to avoid unicode in filenames
        metric_str = format_metric_value_for_filename(metric, metric_name)
        filename: str = (
            f"{rank_prefix}_s{step:02d}_b{self.current_batch_idx:03d}_"
            f"ep{episode_number:04d}_{self.current_variation}_q{metric_str}.{self.visualization_format}"
        )
        filepath: str = str(pathlib.Path(self.step_viz_dir) / filename)
        # Create the variant-versus-base comparison grid.
        self._create_step_comparison_grid(viz_data, filepath, step, episode_number, metric, metric_name)

    def _save_highlighted_step_comparison_grid(
        self,
        *,
        variant_recon_final: NDArray[np.float32],
        variant_true_final: NDArray[np.float32],
        base_recon_final: NDArray[np.float32],
        base_true_final: NDArray[np.float32],
        min_area: int,
        kernel_size: int,
        morph_iterations: int,
        circle_thickness: int,
        temp_fig: Figure,
        filepath: str,
        row_labels: list[str],
        col_titles: list[str],
        title: str | None,
        suptitle_gap: float,
        style: GridStyle,
    ) -> None:
        """Save the step comparison grid with highlighted variant differences."""
        variant_recon_highlighted: NDArray[np.float32]
        if self.apply_highlighting:
            # Apply highlighting to the variant reconstruction (compare against variant true)
            variant_recon_highlighted, _ = highlight_differences_with_contrast_fill(
                variant_recon_final,
                variant_true_final,
                min_area=min_area,
                kernel_size=kernel_size,
                highlight_mode="first",
                morph_iterations=morph_iterations,
                circle_thickness=circle_thickness,
                use_contrast_fill=True,
                fallback_fill_color=(50, 180, 200),
                fallback_alpha=0.35,
            )
        else:
            # No highlighting - use original image
            variant_recon_highlighted = variant_recon_final

        # Create new grid with highlighted image in the same column order (Variant, Base)
        new_imgs: list[list[NDArray[np.float32]]] = [
            [variant_recon_highlighted, base_recon_final],
            [variant_true_final, base_true_final],
        ]

        # Close the temporary figure
        plt.close(temp_fig)

        # Create final figure with highlighted images
        fig: Figure = create_image_grid(
            imgs=new_imgs,
            row_labels=row_labels,
            col_titles=col_titles,
            padding_size=0.06,
            col_gap=0.04,
            frame_width_in=4.0,
            row_label_gap=1.5,
            col_title_gap=0.04,
            suptitle=title,
            suptitle_gap=suptitle_gap,
            style=style,
            rightmost_col_row_labels=list(self.rightmost_col_row_labels)
            if self.rightmost_col_row_labels is not None
            else None,
            rightmost_col_row_labels_side=self.rightmost_col_row_labels_side,
        )

        # Save the figure with highlighted images
        fig.savefig(filepath, dpi=150, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)

    def _create_step_comparison_grid(
        self, viz_data: GridData, filepath: str, step: int, episode_number: int, metric: float, metric_name: str
    ) -> None:
        """Create a variant-versus-base grid with the step MSE in its title."""
        # Map GridData fields to expected names:
        # step0_top = reconstruction of: initial step or variation version of initial step
        #             -> should be shown in the top row of the very first columns.
        # step0_bottom = ground truth of: initial step or variation version of initial step
        #                -> should be shown in the bottom row of the very first columns.
        # step_n_top = reconstruction of: current step (n) or base/target state of the variation in column 0
        #              -> should be shown in the top row.
        # step_n_bottom = ground truth of: current step (n) or base/target state of the variation in starting column 0
        #                 -> should be shown in the bottom row.

        # Use step_n images for both Variant and Base columns to have consistent step visuals
        state_variation: NDArray[np.float32] = self.image_processor.convert_for_matplotlib(viz_data.step0_bottom)
        state_variation_reconstruction: NDArray[np.float32] = self.image_processor.convert_for_matplotlib(
            viz_data.step0_top
        )
        state_base: NDArray[np.float32] = self.image_processor.convert_for_matplotlib(viz_data.step_n_bottom)
        state_base_reconstruction: NDArray[np.float32] = self.image_processor.convert_for_matplotlib(
            viz_data.step_n_top
        )

        # Calculate MSE between reconstructions and their corresponding ground truths (before highlighting)
        # Always compute MSE against the BASE target for the corresponding state/step
        variant_recon_mse: float = float(np.mean((state_variation_reconstruction - state_base) ** 2))
        base_recon_mse: float = float(np.mean((state_base_reconstruction - state_base) ** 2))

        # Format MSE values using scientific format always
        variant_mse_str: str = _format_scientific_notation(variant_recon_mse)
        base_mse_str: str = _format_scientific_notation(base_recon_mse)

        # Set up the grid columns for best/worst: Variant first, then Base
        col_titles: list[str] = [f"Variant\n(MSE: {variant_mse_str})", f"Base\n(MSE: {base_mse_str})"]
        # Arrange images: first column Variant, second column Base
        imgs: list[list[NDArray[np.float32]]] = [
            [state_variation_reconstruction, state_base_reconstruction],
            [state_variation, state_base],
        ]
        row_labels: list[str] = ["Reconstruction", "Ground Truth"]
        # Resolve suptitle using optional template from config
        metric_display_name: str = (metric_name or "Metric").replace("_", " ").title()
        metric_str = format_metric_value(metric, metric_name or "metric")
        if self.step_tracker_template is None:
            # Default template
            title: str | None = f"Step {step} | Episode {episode_number} | {metric_display_name}: {metric_str}"
        elif not self.step_tracker_template:
            title = ""
        else:
            try:
                title = self.step_tracker_template.format(
                    step=step,
                    episode=episode_number,
                    episode_number=episode_number,
                    metric_name=metric_name or "metric",
                    metric_name_display=metric_display_name,
                    metric_value=metric,
                    metric_value_formatted=metric_str,
                )
            except Exception:
                # On formatting error, fallback to default
                title = f"Step {step} | Episode {episode_number} | {metric_display_name}: {metric_str}"

        # Prepare style with optional suptitle overrides
        style = GridStyle.model_tester_style()
        st = self.suptitle_style
        if st:
            fam = st.get("font_family")
            if isinstance(fam, str):
                style.suptitle.font_family = fam
            serif = st.get("font_serif")
            if isinstance(serif, (list, tuple)):
                style.suptitle.font_serif = list(serif)
            fsize = st.get("font_size")
            if isinstance(fsize, (int, float)):
                style.suptitle.font_size = int(fsize)
            fweight = st.get("font_weight")
            if isinstance(fweight, str):
                style.suptitle.font_weight = fweight
            col = st.get("color")
            if isinstance(col, str):
                style.suptitle.color = col
            bbox = st.get("bbox_style")
            if isinstance(bbox, dict):
                style.suptitle.bbox_style = bbox
        suptitle_gap = 0.15 if self.suptitle_gap is None else self.suptitle_gap

        # Use a temporary grid to determine the image dimensions.
        temp_fig: Figure = create_image_grid(
            imgs=imgs,
            row_labels=row_labels,
            col_titles=col_titles,
            padding_size=0.06,
            col_gap=0.04,
            frame_width_in=4.0,
            row_label_gap=1.5,
            col_title_gap=0.04,
            suptitle=title,
            suptitle_gap=suptitle_gap,
            style=style,
            rightmost_col_row_labels=list(self.rightmost_col_row_labels)
            if self.rightmost_col_row_labels is not None
            else None,
            rightmost_col_row_labels_side=self.rightmost_col_row_labels_side,
        )

        # Find image axes in the grid
        image_axes: list[Axes] = [ax for ax in temp_fig.axes if len(ax.get_images()) > 0]

        # Extract final rendered images
        if len(image_axes) >= 4:  # A 2 by 2 grid contains four image axes.
            # Axes order given create_image_grid placement (per column: top then bottom):
            # 0: variant_recon (col0, top)
            # 1: variant_true  (col0, bottom)
            # 2: base_recon    (col1, top)
            # 3: base_true     (col1, bottom)
            variant_recon_ax: Axes = image_axes[0]
            variant_true_ax: Axes = image_axes[1]
            base_recon_ax: Axes = image_axes[2]
            base_true_ax: Axes = image_axes[3]

            # Extract final rendered images
            variant_recon_final: NDArray[np.float32] = extract_final_rendered_image(variant_recon_ax)
            base_recon_final: NDArray[np.float32] = extract_final_rendered_image(base_recon_ax)
            variant_true_final: NDArray[np.float32] = extract_final_rendered_image(variant_true_ax)
            base_true_final: NDArray[np.float32] = extract_final_rendered_image(base_true_ax)

            # Get highlight parameters
            height_px, width_px = variant_recon_final.shape[:2]
            h_params: dict[str, int | tuple[int, int, int]] = derive_highlight_geometry(
                height_px=height_px, width_px=width_px, dpi=300.0
            )

            # The highlight params may return either a scalar int/float or a small tuple where
            # the first element is the desired value. Handle both shapes explicitly.
            min_area_value: int | tuple[int, int, int] = h_params["min_area"]
            min_area = min_area_value * 20 if isinstance(min_area_value, (int, float)) else min_area_value[0] * 20

            kernel_size_value: int | tuple[int, int, int] = h_params["kernel_size"]
            kernel_size = kernel_size_value if isinstance(kernel_size_value, (int, float)) else kernel_size_value[0]

            morph_iterations_value: int | tuple[int, int, int] = h_params["morph_iterations"]
            morph_iterations = 3 * (
                morph_iterations_value
                if isinstance(morph_iterations_value, (int, float))
                else morph_iterations_value[0]
            )

            circle_thickness: int = 12

            try:
                self._save_highlighted_step_comparison_grid(
                    variant_recon_final=variant_recon_final,
                    variant_true_final=variant_true_final,
                    base_recon_final=base_recon_final,
                    base_true_final=base_true_final,
                    min_area=min_area,
                    kernel_size=kernel_size,
                    morph_iterations=morph_iterations,
                    circle_thickness=circle_thickness,
                    temp_fig=temp_fig,
                    filepath=filepath,
                    row_labels=row_labels,
                    col_titles=col_titles,
                    title=title,
                    suptitle_gap=suptitle_gap,
                    style=style,
                )
            except Exception as e:
                logger.warning(f"Highlighting failed: {e}")
                # Continue with original approach if highlighting fails
                plt.close(temp_fig)
            else:
                return
        else:
            logger.warning(f"Expected 4 image axes but found {len(image_axes)}. Using fallback approach.")
            plt.close(temp_fig)

        # Fallback: keep the original (Variant, Base) column order
        imgs = [[state_variation_reconstruction, state_base_reconstruction], [state_variation, state_base]]

        # Create and save figure with original images
        fig = create_image_grid(
            imgs=imgs,
            row_labels=row_labels,
            col_titles=col_titles,
            padding_size=0.06,
            col_gap=0.04,
            frame_width_in=4.0,
            row_label_gap=1.5,
            col_title_gap=0.04,
            suptitle=title,
            suptitle_gap=suptitle_gap,
            style=style,
            rightmost_col_row_labels=list(self.rightmost_col_row_labels)
            if self.rightmost_col_row_labels is not None
            else None,
            rightmost_col_row_labels_side=self.rightmost_col_row_labels_side,
        )

        fig.savefig(filepath, dpi=150, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)

    def reset_for_new_variation(self) -> None:
        """Reset tracking for a new variation."""
        self.best_steps.clear()
        self.worst_steps.clear()
        self._counter = 0  # Reset counter for new variation
