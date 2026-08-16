"""Image grid visualization module with customizable layout and styling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing import TypeAlias

    from matplotlib.figure import Figure
    from matplotlib.typing import RcKeyType
    import numpy as np
    from numpy.typing import NDArray


RcParamsMap: TypeAlias = dict["RcKeyType", str | float | int | bool | Sequence[str]]
BboxStyle: TypeAlias = Mapping[str, str | float | int | bool]
HorizontalAlignment: TypeAlias = Literal["center"]
VerticalAlignment: TypeAlias = Literal["bottom", "center"]


@dataclass
class FrameStyle:
    """Styling configuration for frame elements."""

    facecolor: str = "lightgray"
    edgecolor: str = "gray"
    linewidth: float = 3.0
    alpha: float = 1.0
    rounding_size: float = 0.02
    inset: float = 0.01


@dataclass
class TextStyle:
    """Styling configuration for text elements."""

    # Preferred font family (matplotlib accepts strings like 'serif' or 'sans-serif')
    font_family: str = "sans-serif"
    # Optional ordered list of serif fonts
    font_serif: Sequence[str] | None = None
    font_size: int = 8
    font_weight: str = "normal"
    color: str = "black"
    # bbox styling passed directly into matplotlib.text->bbox. Values are
    # typically strings or numeric types (alpha, linewidth, pad, etc.).
    bbox_style: BboxStyle | None = None


@dataclass
class FigureStyle:
    """Styling configuration for figure elements."""

    facecolor: str = "white"
    edgecolor: str = "none"
    dpi: int = 300
    # Matplotlib rcParams values are heterogeneous (strings, numbers, booleans,
    # or sequences such as font.serif). We type them precisely here.
    rcparams: RcParamsMap | None = None


@dataclass
class GridStyle:
    """Complete styling configuration for the image grid."""

    frame: FrameStyle = field(default_factory=FrameStyle)
    column_title: TextStyle = field(default_factory=TextStyle)
    row_label: TextStyle = field(default_factory=TextStyle)
    suptitle: TextStyle = field(default_factory=lambda: TextStyle(font_size=8, font_weight="bold"))
    figure: FigureStyle = field(default_factory=FigureStyle)

    @classmethod
    def model_tester_style(cls) -> GridStyle:
        """Create styling that matches the model_tester.py visualization style."""
        return cls(
            frame=FrameStyle(
                facecolor="#E8E8E8",  # Light gray color
                edgecolor="#95A5A6",  # gray color
                linewidth=2.0,
                alpha=0.95,
                rounding_size=0.02,
                inset=0.01,
            ),
            column_title=TextStyle(
                font_family="serif",
                font_serif=["Times New Roman", "Computer Modern", "DejaVu Serif"],
                font_size=8,
                font_weight="bold",
                color="#2C3E50",  # Dark blue color
                bbox_style={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "#F8F9FA",
                    "edgecolor": "#3498DB",
                    "alpha": 0.9,
                    "linewidth": 1.0,
                },
            ),
            row_label=TextStyle(
                font_family="serif",
                font_serif=["Times New Roman", "Computer Modern", "DejaVu Serif"],
                font_size=8,
                font_weight="bold",
                color="#27AE60",  # Dark green color
                bbox_style={
                    "boxstyle": "round,pad=0.5",
                    "facecolor": "white",
                    "edgecolor": "#27AE60",
                    "linewidth": 1.0,
                    "alpha": 0.95,
                },
            ),
            suptitle=TextStyle(
                font_family="serif",
                font_serif=["Times New Roman", "Computer Modern", "DejaVu Serif"],
                font_size=10,
                font_weight="bold",
                color="#2C3E50",  # Dark blue color
                bbox_style={
                    "boxstyle": "round,pad=0.7",
                    "facecolor": "#E8F4FD",
                    "edgecolor": "#3498DB",
                    "alpha": 0.95,
                    "linewidth": 1.0,
                },
            ),
            figure=FigureStyle(
                facecolor="white",  # "#FAFBFC",
                edgecolor="none",
                dpi=300,
                rcparams={
                    "font.family": "serif",
                    "font.serif": ["Times New Roman", "Computer Modern", "DejaVu Serif"],
                    "font.size": 12,
                    "axes.linewidth": 1.0,
                    "figure.facecolor": "white",  # "#FAFBFC",
                    "axes.facecolor": "white",
                    "savefig.facecolor": "white",  # "#FAFBFC",
                    "savefig.dpi": 300,
                },
            ),
        )


def create_custom_style(
    *,
    frame_color: str | None = None,
    frame_edge_color: str | None = None,
    text_color: str | None = None,
    text_font_family: str | None = None,
    background_color: str | None = None,
    use_300_dpi: bool = False,
) -> GridStyle:
    """Create a custom style configuration with common styling options.

    Args:
        frame_color: Background color for frames (e.g., "lightblue", "#E8F4FD")
        frame_edge_color: Edge color for frames (e.g., "navy", "#3498DB")
        text_color: Color for all text elements (e.g., "darkblue", "#2C3E50")
        text_font_family: Font family for text ("serif", "sans-serif", "monospace")
        background_color: Background color for the entire figure
        use_300_dpi: Whether to set figure and save resolution to 300 DPI.

    Returns:
        GridStyle: Configured style object
    """
    style = GridStyle()

    # Apply frame styling
    if frame_color:
        style.frame.facecolor = frame_color
    if frame_edge_color:
        style.frame.edgecolor = frame_edge_color

    # Apply text styling
    if text_color:
        style.column_title.color = text_color
        style.row_label.color = text_color
        style.suptitle.color = text_color

    if text_font_family:
        style.column_title.font_family = text_font_family
        style.row_label.font_family = text_font_family
        style.suptitle.font_family = text_font_family

    # Apply figure styling
    if background_color:
        style.figure.facecolor = background_color

    # Apply the 300 DPI preset.
    if use_300_dpi:
        style.figure.dpi = 300
        style.figure.rcparams = {"font.size": 12, "axes.linewidth": 1.2, "savefig.dpi": 300}

    return style


def _add_figure_text(
    fig: Figure,
    x: float,
    y: float,
    text: str,
    *,
    style: TextStyle,
    ha: HorizontalAlignment,
    va: VerticalAlignment,
    rotation: Literal["vertical"] | None = None,
) -> None:
    if style.font_family and style.bbox_style:
        fig.text(
            x,
            y,
            text,
            rotation=rotation,
            ha=ha,
            va=va,
            fontsize=style.font_size,
            fontweight=style.font_weight,
            color=style.color,
            fontfamily=style.font_family,
            bbox=dict(style.bbox_style),
        )
        return
    if style.font_family:
        fig.text(
            x,
            y,
            text,
            rotation=rotation,
            ha=ha,
            va=va,
            fontsize=style.font_size,
            fontweight=style.font_weight,
            color=style.color,
            fontfamily=style.font_family,
        )
        return
    if style.bbox_style:
        fig.text(
            x,
            y,
            text,
            rotation=rotation,
            ha=ha,
            va=va,
            fontsize=style.font_size,
            fontweight=style.font_weight,
            color=style.color,
            bbox=dict(style.bbox_style),
        )
        return
    fig.text(
        x,
        y,
        text,
        rotation=rotation,
        ha=ha,
        va=va,
        fontsize=style.font_size,
        fontweight=style.font_weight,
        color=style.color,
    )


def create_image_grid(
    imgs: Sequence[Sequence[NDArray[np.float32]]],
    row_labels: Sequence[str],
    col_titles: Sequence[str],
    *,
    padding_size: float = 0.05,  # Identical gap on all 4 sides (0 < p < .25)
    col_gap: float = 0.15,  # Gap between frames (frame-width units)
    frame_width_in: float = 3.0,  # Width of each grey frame (inches)
    row_label_gap: float = 2.0,  # 0 is flush to the frame. 1 reaches the padding edge.
    col_title_gap: float = 2.0,  # Vertical gap above frame (figure-fraction units)
    suptitle: str | None = None,  # Global title, optional
    suptitle_gap: float = 0.08,  # Gap between column titles & suptitle (figure frac)
    style: GridStyle | None = None,  # Optional styling configuration
    rightmost_col_row_labels: Sequence[str] | None = None,  # Optional: draw row labels near last column
    # Controls where the optional rightmost-col row labels are drawn: to the 'left' of the last
    # column, between the final two columns by default, or to the right of the final column.
    # (outside the grid on the far right).
    rightmost_col_row_labels_side: str = "left",
    left_panel_image: NDArray[np.float32] | None = None,  # Optional: single image panel on the far left
    left_panel_title: str | None = None,  # Title for the left panel
    # Optional: overrides only the gap between the left panel and the first main column.
    # Units match col_gap (frame-width units)
    left_panel_main_gap: float | None = None,
) -> Figure:
    """Build a rectangular image grid with symmetrical padding.

    Args:
        imgs: 2D list of numpy arrays with shape [n_rows][n_cols]. Each array
            represents an image to be displayed in the grid.
        row_labels: Sequence of strings with length equal to n_rows. Labels
            displayed on the left side of each row.
        col_titles: Sequence of strings with length equal to n_cols. Titles
            displayed above each column frame.
        padding_size: Padding as fraction of frame width. Must be between 0 and 0.25.
        col_gap: Gap between frames measured in frame-width units.
        frame_width_in: Width of each grey frame in inches.
        row_label_gap: Distance from frame edge to row label. 0 means glued to frame,
            1 means center of pad area. Values greater than 1 are allowed.
        col_title_gap: Vertical gap between frame and column title measured in
            figure-fraction units.
        suptitle: Global heading displayed above all column titles. Optional.
        suptitle_gap: Extra gap above column titles measured in figure-fraction units.
        style: Optional styling configuration. If None, uses default styling.
        rightmost_col_row_labels: Optional sequence of strings of length n_rows. If provided,
            a second set of row labels will be drawn adjacent to the right-most column.
        rightmost_col_row_labels_side: Controls where the right-most column row labels are placed.
            Accepts "left" (default) to place labels between the last two columns, or "right" to
            place labels outside the grid, on the far right of the last column.
        left_panel_image: Optional single image to display in its own framed cell on the far left of the grid.
            When provided, the main grid shifts right by one column and this panel is vertically centered
            relative to the two-row image block.
        left_panel_title: Optional title string to display above the left panel. Defaults to None.
        left_panel_main_gap: If provided together with left_panel_image, only the gap between the left
            panel and the first main column will use this value (in frame-width units). All other inter-column
            gaps continue to use 'col_gap'.

    Returns:
        matplotlib.figure.Figure: The created figure containing the image grid.

    Raises:
        ValueError: If rows in 'imgs' have unequal length, or if the length of
            'row_labels' doesn't match n_rows, or if the length of 'col_titles'
            doesn't match n_cols, or if 'padding_size' is not in range (0, 0.25).
    """
    # Use default style if none provided
    if style is None:
        style = GridStyle()

    # Apply matplotlib rcParams if provided
    if style.figure.rcparams:
        plt.rcParams.update(style.figure.rcparams)
    n_rows: int = len(imgs)
    # Note: imgs is a sequence of equal-length sequences. Use first row to get n_cols.
    n_cols: int = len(imgs[0])

    if any(len(r) != n_cols for r in imgs):
        raise ValueError("All rows in 'imgs' must be equal length.")
    # Pad missing row labels with empty strings.
    # Only error when more labels than rows are provided.
    if len(row_labels) > n_rows:
        raise ValueError("'row_labels' length must be <= n_rows.")
    if len(row_labels) < n_rows:
        # convert to list and pad with empty labels to preserve indexing below
        row_labels = list(row_labels) + [""] * (n_rows - len(row_labels))
    if len(col_titles) != n_cols:
        raise ValueError("'col_titles' length must equal n_cols.")
    if not (0 < padding_size < 0.25):
        raise ValueError("'padding_size' must lie in (0, .25).")

    p: float = padding_size
    # GridSpec weight w so that one pad block = 'p' of total width
    # pad_fraction = w / (1 + 2 w) -> w = p / (1 - 2 p)
    w: float = p / (1 - 2 * p)
    width_ratios: list[float] = [w, 1, w]  # L-pad | cell | R-pad

    # Figure size
    panel_cols: int = 1 if left_panel_image is not None else 0
    if left_panel_image is not None and left_panel_main_gap is not None:
        # Nested layout: left panel | main grid. Use custom gap only between these two blocks.
        fig_w: float = (
            1 * frame_width_in
            + left_panel_main_gap * frame_width_in
            + n_cols * frame_width_in
            + (n_cols - 1) * col_gap * frame_width_in
        )
    else:
        # Single-layer layout using global col_gap for all inter-column gaps
        fig_w = (n_cols + panel_cols) * frame_width_in + (n_cols - 1 + panel_cols) * col_gap * frame_width_in
    unit_in: float = frame_width_in / (1 + 2 * w)  # inch per weight unit

    # Column-wise tallest stack
    col_heights: list[float] = []
    col_aspects: list[list[float]] = []
    for c_idx in range(n_cols):
        # compute per-image aspect ratios (height / width)
        aspects_c: list[float] = []
        for r in range(n_rows):
            img = imgs[r][c_idx]
            aspects_c.append(float(img.shape[0]) / float(img.shape[1]))
        col_aspects.append(aspects_c)
        col_heights.append(sum(aspects_c) + (n_rows + 1) * w)  # Images + (n_rows + 1) pads

    fig_h: float = unit_in * max(col_heights)
    fig: Figure = plt.figure(figsize=(fig_w, fig_h), facecolor=style.figure.facecolor)
    fig.patch.set_edgecolor(style.figure.edgecolor)

    # Build outer GridSpec
    use_nested_layout: bool = left_panel_image is not None and left_panel_main_gap is not None
    if use_nested_layout:
        # Two columns: [left panel] | [main grid container]
        # Width ratios approximate relative widths of panel vs. main grid area
        outer_gs: GridSpec = GridSpec(
            1,
            2,
            figure=fig,
            wspace=left_panel_main_gap,
            width_ratios=[1.0, float(n_cols) + float(n_cols - 1) * col_gap],
        )
    else:
        # Outer 1 * (n_cols + extra_left_cols) GridSpec with uniform gaps
        outer_total_cols: int = n_cols + panel_cols
        outer_gs = GridSpec(1, outer_total_cols, figure=fig, wspace=col_gap)

    # Vertical centers of each image row
    row_centers: dict[int, float] = {}

    # Build columns
    first_col_frame_pos = None
    last_col_frame_pos = None
    left_panel_frame_pos = None  # will be set if a left panel is added
    main_parent_spec: GridSpecFromSubplotSpec | None = None
    start_c: int = panel_cols
    # Parent spec for main grid columns (nested when using custom left-panel gap)
    if use_nested_layout:
        main_parent_spec = GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer_gs[0, 1], wspace=col_gap, hspace=0.0)
    else:
        # Determine starting column index in the outer gridspec for the main grid
        start_c = panel_cols
    for c in range(n_cols):
        aspects: list[float] = col_aspects[c]
        height_ratios: list[float] = [w]
        for r_aspect in aspects:
            height_ratios += [r_aspect, w]

        # Select parent slot for this column (nested or flat layout)
        if use_nested_layout:
            assert main_parent_spec is not None
            parent_spec_slot = main_parent_spec[0, c]
        else:
            parent_spec_slot = outer_gs[0, start_c + c]

        # Grey rounded frame
        ax_frame = fig.add_subplot(parent_spec_slot)
        ax_frame.axis("off")
        ax_frame.add_patch(
            FancyBboxPatch(
                (style.frame.inset, style.frame.inset),
                1.0 - 2 * style.frame.inset,
                1.0 - 2 * style.frame.inset,
                boxstyle=f"round,pad=0.005,rounding_size={style.frame.rounding_size}",
                edgecolor=style.frame.edgecolor,
                facecolor=style.frame.facecolor,
                linewidth=style.frame.linewidth,
                alpha=style.frame.alpha,
                transform=ax_frame.transAxes,
            )
        )

        # Capture frame positions for label anchoring
        if c == 0:
            first_col_frame_pos = ax_frame.get_position()
        # Track last column position on each iteration
        last_col_frame_pos = ax_frame.get_position()

        # Inner GridSpec (pads | image | pads vertically, with side pads)
        inner_gs: GridSpecFromSubplotSpec = GridSpecFromSubplotSpec(
            len(height_ratios),
            3,
            subplot_spec=parent_spec_slot,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            wspace=0.0,
            hspace=0.0,
        )

        # Place images
        for r in range(n_rows):
            ax = fig.add_subplot(inner_gs[2 * r + 1, 1])
            ax.imshow(imgs[r][c])
            ax.axis("off")
            ax.set_aspect("equal")

            if c == 0:  # Capture row center
                pos = ax.get_position()
                row_centers[r] = 0.5 * (pos.y0 + pos.y1)

        # Column title
        pos = ax_frame.get_position()
        _add_figure_text(
            fig,
            0.5 * (pos.x0 + pos.x1),
            pos.y1 + col_title_gap,
            col_titles[c],
            style=style.column_title,
            ha="center",
            va="bottom",
        )

    # Optional left single-cell panel (variant/noisy environment)
    if left_panel_image is not None:
        # Create a frame for the left panel
        ax_left_frame = fig.add_subplot(outer_gs[0, 0])
        ax_left_frame.axis("off")
        ax_left_frame.add_patch(
            FancyBboxPatch(
                (style.frame.inset, style.frame.inset),
                1.0 - 2 * style.frame.inset,
                1.0 - 2 * style.frame.inset,
                boxstyle=f"round,pad=0.005,rounding_size={style.frame.rounding_size}",
                edgecolor=style.frame.edgecolor,
                facecolor=style.frame.facecolor,
                linewidth=style.frame.linewidth,
                alpha=style.frame.alpha,
                transform=ax_left_frame.transAxes,
            )
        )
        # Build inner grid for left panel: pad | image | pad vertically
        # Height ratios: w (top pad), aspect (image), w (bottom pad) to match spacing
        left_aspect = (
            float(left_panel_image.shape[0]) / float(left_panel_image.shape[1]) if left_panel_image.size > 0 else 1.0
        )
        left_inner_gs: GridSpecFromSubplotSpec = GridSpecFromSubplotSpec(
            3,
            3,
            subplot_spec=outer_gs[0, 0],
            height_ratios=[w, left_aspect, w],
            width_ratios=width_ratios,
            wspace=0.0,
            hspace=0.0,
        )

        ax_left_img = fig.add_subplot(left_inner_gs[1, 1])
        ax_left_img.imshow(left_panel_image)
        ax_left_img.axis("off")
        ax_left_img.set_aspect("equal")

        # Capture left panel frame position for label placement logic
        left_panel_frame_pos = ax_left_frame.get_position()

        # Title for left panel, if provided
        if left_panel_title:
            pos = ax_left_frame.get_position()
            _add_figure_text(
                fig,
                0.5 * (pos.x0 + pos.x1),
                pos.y1 + col_title_gap,
                left_panel_title,
                style=style.column_title,
                ha="center",
                va="bottom",
            )

    # global col_gap controls spacing without a dedicated spacer column.

    # Row labels (left side): place between the left panel and first MAIN column when panel exists.
    # Use the captured frame position for the first main column. Fall back when unavailable.
    if first_col_frame_pos is None:
        # As a conservative fallback, try to derive from any axes that belongs to the main grid
        # fig.axes[0] may belong to the left panel, so derive the position from GridSpec.
        first_col_frame_pos = fig.get_axes()[-1].get_position()
    pad_pix: float = p * first_col_frame_pos.width
    first_left = first_col_frame_pos.x0
    # Compute margins from column/panel. For summary views (very small row_label_gap),
    # place labels with a small but readable space closer to the first column than the left panel.
    if row_label_gap <= 0.0015:
        # Small normal space from the first column ("Step 0")
        margin_from_col = max(0.012, 0.85 * pad_pix)
        # Leave a small buffer after the optional left panel.
        margin_from_panel = max(margin_from_col + 0.008, 0.025, 1.8 * pad_pix)
    else:
        # Original, more conservative spacing
        margin_from_col = max(0.012, 1.2 * pad_pix)
        margin_from_panel = max(margin_from_col + 0.005, 0.02, 2.2 * pad_pix)
    # Default: place labels just to the left of the first column with a small, readable margin
    x_lab: float = first_left - margin_from_col
    # If a left panel exists, compute bounds and clamp within margins
    if left_panel_image is not None and left_panel_frame_pos is not None:
        panel_right = left_panel_frame_pos.x1
        gap = max(0.0, first_left - panel_right)
        lower_bound = panel_right + margin_from_panel
        upper_bound = first_left - margin_from_col
        if row_label_gap <= 0.0015:
            # Summary view: place near the first column with a small readable buffer
            x_lab = upper_bound if upper_bound >= lower_bound else panel_right + 0.5 * gap
        else:
            # Non-summary: preserve the original balanced placement toward the column
            desired = panel_right + 0.6 * gap
            x_lab = min(max(lower_bound, desired), upper_bound)
    for r, label in enumerate(row_labels):
        _add_figure_text(
            fig, x_lab, row_centers[r], label, style=style.row_label, ha="center", va="center", rotation="vertical"
        )

    # Optional: Row labels for right-most block (placed adjacent to the last main column)
    if rightmost_col_row_labels is not None and n_cols >= 2:
        # Validate/pad rightmost_col_row_labels similar to left row_labels
        if len(rightmost_col_row_labels) > n_rows:
            raise ValueError("'rightmost_col_row_labels' length must be <= n_rows.")
        if len(rightmost_col_row_labels) < n_rows:
            rightmost_col_row_labels = list(rightmost_col_row_labels) + [""] * (n_rows - len(rightmost_col_row_labels))
        # Anchor to the actual last main column frame position
        anchor_pos = last_col_frame_pos
        if anchor_pos is None:
            # Fallback: approximate with the last axes position
            anchor_pos = fig.axes[-1].get_position()
        # Determine placement side for the rightmost labels
        place_on_right: bool = rightmost_col_row_labels_side.lower() == "right"
        if place_on_right:
            # Place labels just to the RIGHT of the last column with a readable margin
            # Use a margin proportional to pad as base, with a minimum to avoid touching the frame
            margin_from_col_right = max(0.012, 0.85 * pad_pix)
            x_lab_right: float = anchor_pos.x1 + margin_from_col_right
        else:
            # Default behavior: just to the LEFT of the last column (between the last two columns)
            x_lab_right = anchor_pos.x0 - 0.5 * row_label_gap * pad_pix
        for r, label in enumerate(rightmost_col_row_labels):
            _add_figure_text(
                fig,
                x_lab_right,
                row_centers[r],
                label,
                style=style.row_label,
                ha="center",
                va="center",
                rotation="vertical",
            )

    # Global suptitle
    if suptitle:
        # Top of column titles
        top_title_y: float = max(ax.get_position().y1 for ax in fig.axes[: int(n_cols)])

        assert suptitle is not None
        _add_figure_text(fig, 0.5, top_title_y + suptitle_gap, suptitle, style=style.suptitle, ha="center", va="bottom")

    # Reset matplotlib settings if they were modified
    if style.figure.rcparams:
        plt.rcdefaults()

    return fig
