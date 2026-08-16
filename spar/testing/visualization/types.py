"""Type definitions for the visualization system."""

from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import TYPE_CHECKING, TypeAlias, TypedDict

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


SuptitleStyleValue: TypeAlias = str | int | float | list[str] | tuple[str, ...] | dict[str, str | int | float | bool]
SuptitleStyle: TypeAlias = dict[str, SuptitleStyleValue]
AccumulatorDictValue: TypeAlias = float | list[float]
AccumulatorDict: TypeAlias = dict[str, AccumulatorDictValue]
MetricsDict: TypeAlias = dict[str, int | list[int] | str] | AccumulatorDict
EpisodeMetricsValue: TypeAlias = AccumulatorDictValue | int | str | list[int]
EpisodeMetrics: TypeAlias = dict[str, EpisodeMetricsValue]


@dataclass(slots=True)
class GridData:
    """Data structure for grid visualization."""

    step: int
    episode_number: int
    step0_bottom: NDArray[np.uint8 | np.float32]
    step0_top: NDArray[np.uint8 | np.float32]
    step_n_bottom: NDArray[np.uint8 | np.float32]
    step_n_top: NDArray[np.uint8 | np.float32]
    batch_idx: int
    # Optional: base version of the starting state's bottom image (for replacing variant in main grid)
    base_step0_bottom: NDArray[np.uint8 | np.float32] | None
    # Optional: title for a separate left-side single-cell panel to display the variant/noisy env
    variant_panel_title: str | None
    # Note: Additional optional fields should be declared in VisualizationData to avoid
    # dataclass inheritance ordering issues.


@dataclass(slots=True)
class VisualizationData(GridData):
    """Data structure for steps visualization."""

    output_dir: str
    variation_name: str
    metric: float = 0.0
    row_labels: list[str] = field(default_factory=lambda: ["Reconstruction", "Ground Truth"])
    # Optional: Row labels specifically for the right-most column.
    # When provided, the renderer can draw a second set of row labels near the last column.
    rightmost_col_row_labels: list[str] | None = None
    # Controls where to place the rightmost-col row labels: 'left' (between columns) or 'right' (outside).
    rightmost_col_row_labels_side: str = "left"
    col_titles: list[str] = field(default_factory=list)
    metrics_dict: dict[str, float | int | str] | None = None
    # If provided, overrides auto title. If set to empty string "", suptitle is disabled.
    suptitle: str | None = None
    # Optional style overrides for suptitle (maps into GridStyle.suptitle fields)
    suptitle_style: SuptitleStyle | None = None
    suptitle_gap: float | None = None
    save_path: str = "None"
    visualization_format: str = "png"

    def __post_init__(self) -> None:
        """Initialize defaults for titles and save path."""
        if not self.col_titles:
            self.col_titles = ["Step 0", f"Step {self.step}"]

        # Auto-generate suptitle only when not explicitly provided.
        # When set to empty string "", keep it to disable suptitle rendering.
        if self.suptitle is None:
            display_variation_name: str = self.variation_name.replace("_", " ").title()
            self.suptitle = f"Episode {self.episode_number}, Step {self.step} - {display_variation_name}"

        if self.save_path == "None":
            filename: str = (
                f"ep{self.episode_number}_b{self.batch_idx}_s{self.step}_"
                f"{self.variation_name}.{self.visualization_format}"
            )
            self.save_path = str(pathlib.Path(self.output_dir) / filename)


class QualityMetrics(TypedDict, total=False):
    """Metrics for episode tracking."""

    metric_values: list[float]
    metric_name: str
    is_higher_better: bool
    viz_data: list[GridData]


class EpisodeTrackingData(TypedDict, total=False):
    """Data structure for tracking best/worst episodes."""

    episode_number: int
    metric_value: float
    metric_name: str
    is_higher_better: bool
    viz_data: list[GridData]


class StepEntry(TypedDict):
    """Data structure for step tracking entries."""

    step: int
    batch_idx: int
    episode_number: int
    metric: float
    metric_name: str
    is_higher_better: bool
    step_data: GridData
