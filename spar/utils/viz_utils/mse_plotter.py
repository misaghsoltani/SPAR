"""Plot MSE trajectories with configurable statistics and style presets.

The plotter accepts one or more arrays shaped as runs by time steps. It computes
means, spread statistics, quantiles, and smoothed series, with chunked handling
and decimation for long inputs.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field, replace
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogFormatterMathtext, MaxNLocator, ScalarFormatter
import numpy as np

from spar.utils.viz_utils.percentage_formatting import round_percentages

if TYPE_CHECKING:
    from collections.abc import Sequence
    from logging import Logger
    from typing import ClassVar, Literal

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.typing import LegendLocType, LineStyleType
    from numpy.typing import NDArray


# Warn after 50 Matplotlib figures remain open.
mpl.rcParams["figure.max_open_warning"] = 50

logger: Logger = getLogger(__name__)


class StyleName(str, Enum):
    """Named color, font, and line-style presets."""

    CLASSIC_MINIMALIST = "classic_minimalist"
    EDITORIAL_GREYSCALE = "editorial_greyscale"
    PARULA_CB_SAFE = "parula_cb_safe"
    OKLAB_SINGLE_HUE = "oklab_single_hue"
    FLAT_UI_CONTRAST = "flat_ui_contrast"
    NEON_ON_CHARCOAL = "neon_on_charcoal"
    EARTH_TONE_FIELD = "earth_tone_field"
    MONOCHROME_SERIF = "monochrome_serif"
    HIGH_CONTRAST_INK = "high_contrast_ink"
    DARK_MODE_GREYSCALE = "dark_mode_greyscale"
    NATURE_JOURNAL = "nature_journal"
    AAAI_CONFERENCE = "aaai_conference"
    SOLARIZED_LIGHT = "solarized_light"
    DARK_ACADEMIA = "dark_academia"
    BLUE_GREY_SANS = "blue_grey_sans"


class SmoothingMethod(str, Enum):
    """Available smoothing methods for time series."""

    EXPONENTIAL = "exponential"
    SAVGOL = "savgol"
    NONE = "none"


class UncertaintyMethod(str, Enum):
    """Available uncertainty visualization methods."""

    STD = "std"
    QUANTILES = "quantiles"
    MINMAX = "minmax"


class Statistics(NamedTuple):
    """Statistical summary of MSE time series."""

    mean: NDArray[np.float32]
    std: NDArray[np.float32]
    min: NDArray[np.float32]
    max: NDArray[np.float32]
    quantiles: dict[float, NDArray[np.float32]]
    smoothed_mean: NDArray[np.float32]


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for MSE plotting."""

    # Visual style
    style: StyleName = StyleName.NATURE_JOURNAL
    use_log_scale: bool = True
    # Linear-scale formatting controls (applied when use_log_scale is False)
    y_use_scientific_notation: bool = True
    y_decimal_places: int = 3
    y_symlog_threshold: float = 1.0

    # Statistical parameters
    smoothing_method: SmoothingMethod = SmoothingMethod.EXPONENTIAL
    smoothing_window_ratio: float = 0.01  # Fraction of time_steps
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.QUANTILES
    quantiles: tuple[float, float] = (0.1, 0.9)

    # Figure parameters
    figsize: tuple[float, float] = (3.45, 2.3)  # Single column width
    dpi: int = 300
    aspect_ratio: float = 1.5  # 3:2 ratio for Nature

    # Large input handling
    max_points_before_decimation: int = 10000
    streaming_threshold: int = 1000000  # Use streaming for T > 1M

    # Export settings
    export_formats: tuple[str, ...] = ("pdf", "png")
    png_transparent: bool = False

    # Legend fine-grained controls
    legend_show_raw_mean: bool | None = None
    legend_show_minmax: bool | None = None
    legend_show_band: bool | None = None

    # Legend label templates
    legend_smoothed_label_template: str = "{source} (smoothed mean)"
    legend_raw_label_template: str = "{source} (mean)"
    legend_minmax_label: str = "min / max"
    legend_band_label_quantiles_template: str = "{percent}% range"
    legend_band_label_std: str = "+/- std range"

    # Optional color overrides (honor config-provided palettes)
    variant_colors: tuple[str, ...] | None = None  # Ordered list applied to sources
    label_color_map: dict[str, str] | None = field(
        default=None, compare=False, hash=False
    )  # Explicit mapping from source label to color

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.smoothing_window_ratio < 1:
            raise ValueError(f"smoothing_window_ratio must be in (0, 1), got {self.smoothing_window_ratio}")

        if not 0 < self.quantiles[0] < self.quantiles[1] < 1:
            raise ValueError(f"quantiles must be in (0, 1) with first < second, got {self.quantiles}")


@dataclass
class StyleSpec:
    """Complete specification for a visual style."""

    # Colors (hex codes)
    background: str
    primary: str  # Main line (smoothed mean)
    secondary: str  # Raw mean
    band: str  # Uncertainty band
    minmax: str  # Min/max lines
    grid: str
    text: str

    # Transparency
    band_alpha: float = 0.20
    minmax_alpha: float = 0.30
    grid_alpha: float = 0.25

    # Line styles
    primary_linewidth: float = 2.0
    secondary_linewidth: float = 1.0
    minmax_linewidth: float = 0.5
    secondary_linestyle: LineStyleType = "--"
    minmax_linestyle: LineStyleType = "-"

    # Typography
    font_family: str = "sans-serif"
    font_size_ticks: int = 7
    font_size_labels: int = 8
    font_size_legend: int = 7
    font_weight_labels: str = "normal"

    # Grid and layout
    show_grid: bool = True
    grid_which: Literal["major", "minor", "both"] = "major"
    grid_linestyle: LineStyleType = "-"
    grid_linewidth: float = 0.4
    legend_framealpha: float = 0.0
    legend_location: LegendLocType = "best"

    # Special effects
    use_spines: bool = True
    spine_alpha: float = 1.0
    use_shadow: bool = False
    use_hatching: bool = False
    hatch_pattern: str = "////"

    def get_multi_source_palette(
        self, n_sources: int, override_colors: Sequence[str | None] | None = None
    ) -> dict[str, list[str]]:
        """Generate distinct color palette for multiple sources while maintaining style coherence.

        Args:
            n_sources: Number of data sources to generate colors for.
            override_colors: Optional sequence of color overrides per source.

        Returns:
            Dictionary mapping color roles to lists of colors for each source.
        """
        if n_sources == 1:
            palette: dict[str, list[str]] = {
                "primary": [self.primary],
                "secondary": [self.secondary],
                "band": [self.band],
                "minmax": [self.minmax],
            }
        else:
            # Derive source colors from the preset's hue and luminance values.
            base_colors: dict[str, str] = self._get_base_colors_for_style()

            palette = {
                "primary": StyleSpec._generate_color_variants(base_colors["primary"], n_sources),
                "secondary": StyleSpec._generate_color_variants(base_colors["secondary"], n_sources),
                "band": StyleSpec._generate_color_variants(base_colors["band"], n_sources),
                "minmax": StyleSpec._generate_color_variants(base_colors["minmax"], n_sources),
            }

        # Apply explicit color overrides (e.g., from config) while keeping fallbacks for any missing entries
        if override_colors:
            overrides: list[str | None] = list(override_colors)
            if len(overrides) < n_sources:
                overrides.extend([None] * (n_sources - len(overrides)))

            for idx, override in enumerate(overrides[:n_sources]):
                if override:
                    palette["primary"][idx] = override
                    palette["secondary"][idx] = override
                    palette["band"][idx] = override
                    palette["minmax"][idx] = override

        return palette

    def _get_base_colors_for_style(self) -> dict[str, str]:
        """Get the base colors used to derive per-source color variations."""
        return {"primary": self.primary, "secondary": self.secondary, "band": self.band, "minmax": self.minmax}

    @staticmethod
    def _generate_color_variants(base_color: str, n_variants: int) -> list[str]:
        """Generate perceptually distinct color variants from a base color."""
        if n_variants == 1:
            return [base_color]

        # Convert hex to RGB
        r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)

        variants: list[str] = []
        for i in range(n_variants):
            # Use hue rotation and lightness variation for distinctness while maintaining style
            factor: float = i / max(1, n_variants - 1)

            # Vary hue and luminance so adjacent sources remain distinct.
            if n_variants <= 3:
                # For 2-3 sources: use lightness variation
                lightness_factor = 0.7 + (factor * 0.6)  # Range: 0.7 to 1.3
                new_r: int = min(255, max(0, int(r * lightness_factor)))
                new_g: int = min(255, max(0, int(g * lightness_factor)))
                new_b: int = min(255, max(0, int(b * lightness_factor)))
            else:
                # For 4+ sources: use hue rotation with lightness
                h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
                # Rotate hue and vary lightness
                h = (h + factor * 0.8) % 1.0
                v = 0.3 + (0.7 * ((factor * 1.4) % 1.0))  # Vary brightness
                r_new, g_new, b_new = colorsys.hsv_to_rgb(h, s, v)
                new_r, new_g, new_b = int(r_new * 255), int(g_new * 255), int(b_new * 255)

            variants.append(f"#{new_r:02x}{new_g:02x}{new_b:02x}")

        return variants


def savgol_filter(x: NDArray[np.float32], window_length: int, polyorder: int) -> NDArray[np.float32]:
    """Apply a Savitzky-Golay filter to a one-dimensional array.

    Args:
        x: Input values.
        window_length: Number of samples in the filter window.
        polyorder: Polynomial order used to derive the convolution coefficients.

    Returns:
        The filtered values as a float32 array.
    """
    if window_length < 3:
        return x.copy()

    half: int = window_length // 2
    # Build design matrix
    a: NDArray[np.float64] = np.vander(np.arange(-half, half + 1), polyorder + 1).astype(np.float64, copy=False)
    # Compute pseudoinverse and filter coefficients for smoothing (zeroth derivative)
    coeffs = np.linalg.pinv(a)[polyorder]  # center coefficients

    # Pad signal at the boundaries
    pad_left = x[0] * np.ones(half, dtype=np.float32)
    pad_right = x[-1] * np.ones(half, dtype=np.float32)
    x_padded = np.concatenate([pad_left, x, pad_right])

    y = np.convolve(x_padded, coeffs[::-1], mode="valid")
    return np.asarray(y, dtype=np.float32)


def _norm_ppf(q: float) -> float:
    """Approximate inverse CDF (percent point function) for standard normal distribution.

    Uses the Peter J. Acklam rational approximation for the inverse normal CDF.
    """
    # Coefficients in rational approximations
    # Implementation follows the public-domain algorithm by Peter J. Acklam
    if q <= 0.0 or q >= 1.0:
        raise ValueError("q must be in (0, 1)")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]

    plow = 0.02425
    phigh = 1 - plow

    if q < plow:
        # Rational approximation for lower region
        ql = np.sqrt(-2.0 * np.log(q))
        return (((((c[0] * ql + c[1]) * ql + c[2]) * ql + c[3]) * ql + c[4]) * ql + c[5]) / (
            (((d[0] * ql + d[1]) * ql + d[2]) * ql + d[3]) * ql + 1.0
        )

    if q > phigh:
        # Rational approximation for upper region
        ql = np.sqrt(-2.0 * np.log(1.0 - q))
        return -(
            (((((c[0] * ql + c[1]) * ql + c[2]) * ql + c[3]) * ql + c[4]) * ql + c[5])
            / ((((d[0] * ql + d[1]) * ql + d[2]) * ql + d[3]) * ql + 1.0)
        )

    # Rational approximation for central region
    qm = q - 0.5
    r = qm * qm
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * qm
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


class MSEPlotter:
    """Compute MSE summaries and render them with a selected style preset."""

    def __init__(self, config: PlotConfig | None = None) -> None:
        """Initialize plotter with configuration.

        Args:
            config: Plot configuration. Uses default Nature style if None.
        """
        self.config: PlotConfig = config or PlotConfig()
        self._style_cache: dict[StyleName, StyleSpec] = {}
        self._setup_matplotlib_backends()

    @staticmethod
    def _setup_matplotlib_backends() -> None:
        """Select Matplotlib's noninteractive Agg backend."""
        # Agg renders plots on hosts without a display server.
        try:
            mpl.use("Agg")  # Non-interactive backend for server environments
        except ImportError:
            logger.warning("Could not set matplotlib backend")

    def _build_color_palette(self, style_spec: StyleSpec, source_names: Sequence[str]) -> dict[str, list[str]]:
        """Construct a color palette honoring explicit config overrides.

        Preference order:
        1) label_color_map (exact label match)
        2) variant_colors (by source index)
        3) style-derived palette (fallback)
        """
        names: list[str] = list(source_names) if source_names else [""]

        # Start with label-specific overrides when provided
        overrides: list[str | None] = []
        if self.config.label_color_map:
            overrides = [self.config.label_color_map.get(name) for name in names]

        # Fill remaining slots using ordered variant_colors
        if self.config.variant_colors:
            for idx in range(len(names)):
                if idx < len(self.config.variant_colors):
                    if len(overrides) <= idx:
                        overrides.append(self.config.variant_colors[idx])
                    elif overrides[idx] is None:
                        overrides[idx] = self.config.variant_colors[idx]

        if overrides:
            overrides = (overrides + [None] * len(names))[: len(names)]

        overrides_param: list[str | None] | None = (
            overrides if overrides and any(color is not None for color in overrides) else None
        )
        return style_spec.get_multi_source_palette(len(names), overrides_param)

    def _get_style_spec(self, style: StyleName) -> StyleSpec:
        """Return a cached style specification.

        Args:
            style: Style name.

        Returns:
            Complete style specification.
        """
        if style in self._style_cache:
            return self._style_cache[style]

        spec: StyleSpec = self._create_style_spec(style)
        self._style_cache[style] = spec
        return spec

    def get_style_spec(self, style: StyleName) -> StyleSpec:
        """Return style specification through the public API."""
        return self._get_style_spec(style)

    @staticmethod
    def _create_style_spec(style: StyleName) -> StyleSpec:
        """Create style specification for given style name.

        Args:
            style: Style name.

        Returns:
            Complete style specification.
        """
        # Preset values are derived from the repository's plot style reference.
        style_definitions: dict[StyleName, StyleSpec] = {
            StyleName.CLASSIC_MINIMALIST: StyleSpec(
                background="#ffffff",
                primary="#000000",
                secondary="#666666",
                band="#666666",
                minmax="#333333",
                grid="#cccccc",
                text="#000000",
                font_family="serif",
                band_alpha=0.20,
                grid_alpha=0.35,
                grid_linestyle=":",
                secondary_linestyle="--",
                minmax_linestyle=":",
                primary_linewidth=1.5,
                secondary_linewidth=0.8,
                minmax_linewidth=0.5,
            ),
            StyleName.EDITORIAL_GREYSCALE: StyleSpec(
                background="#ffffff",
                primary="#000000",  # 90% black for μ̃
                secondary="#999999",  # 60% grey for μ
                band="#cccccc",  # 30% grey for spread
                minmax="#666666",
                grid="#eeeeee",  # 15% grey for grid
                text="#000000",
                font_family="serif",
                font_weight_labels="bold",
                band_alpha=0.0,  # No fill, use hatching
                use_hatching=True,
                hatch_pattern="////",  # 45° hatching
                grid_which="major",  # Grid only on y major
            ),
            StyleName.PARULA_CB_SAFE: StyleSpec(
                background="#ffffff",
                primary="#440154",
                secondary="#3b528b",
                band="#21908c",
                minmax="#5dc863",
                grid="#e6e6e6",
                text="#000000",
                font_family="Inter",
                band_alpha=0.25,
                grid_alpha=0.15,
            ),
            StyleName.OKLAB_SINGLE_HUE: StyleSpec(
                background="#ffffff",
                primary="#1f4e79",
                secondary="#4472a8",
                band="#7d9ac7",
                minmax="#b6c8e6",
                grid="#e6e6e6",
                text="#000000",
                font_family="serif",
                font_size_ticks=7,
            ),
            StyleName.FLAT_UI_CONTRAST: StyleSpec(
                background="#ffffff",
                primary="#e74c3c",
                secondary="#2ecc71",
                band="#3498db",
                minmax="#9b59b6",
                grid="#ecf0f1",
                text="#2c3e50",
                font_family="Helvetica",
                primary_linewidth=1.8,
                use_shadow=True,
            ),
            StyleName.NEON_ON_CHARCOAL: StyleSpec(
                background="#1e1e1e",
                primary="#00ffff",
                secondary="#ff00ff",
                band="#ff00ff",
                minmax="#00ff00",
                grid="#404040",
                text="#ffffff",
                font_family="monospace",
                band_alpha=0.12,
                minmax_linewidth=0.8,
            ),
            StyleName.EARTH_TONE_FIELD: StyleSpec(
                background="#f8f6f0",
                primary="#4f4a3c",
                secondary="#6b5b3a",
                band="#a1b56c",
                minmax="#8b7355",
                grid="#e8e6e0",
                text="#3a3530",
                font_family="serif",
                use_hatching=True,
                spine_alpha=0.5,
            ),
            StyleName.MONOCHROME_SERIF: StyleSpec(
                background="#ffffff",
                primary="#000000",  # Pure black
                secondary="#000000",  # Pure black
                band="#000000",  # Pure black
                minmax="#000000",  # Pure black
                grid="#ffffff",  # No grid
                text="#000000",
                font_family="serif",
                secondary_linestyle="-.",  # Dot-dash for μ
                minmax_linestyle=":",  # Dotted for min/max
                primary_linewidth=2.0,  # Thick solid for μ̃
                secondary_linewidth=0.8,
                minmax_linewidth=0.4,
                show_grid=False,
                band_alpha=0.0,  # No band, use whiskers instead
            ),
            StyleName.HIGH_CONTRAST_INK: StyleSpec(
                background="#ffffff",
                primary="#000000",
                secondary="#000000",
                band="#666666",
                minmax="#000000",
                grid="#cccccc",
                text="#000000",
                font_family="serif",
                font_weight_labels="bold",
                primary_linewidth=2.0,
                secondary_linewidth=0.8,
                minmax_linewidth=0.4,
                band_alpha=0.15,
            ),
            StyleName.DARK_MODE_GREYSCALE: StyleSpec(
                background="#2b2b2b",
                primary="#ffffff",
                secondary="#aaaaaa",
                band="#cccccc",
                minmax="#888888",
                grid="#404040",
                text="#ffffff",
                font_family="monospace",
                primary_linewidth=2.0,
                use_spines=False,
            ),
            StyleName.NATURE_JOURNAL: StyleSpec(
                background="#ffffff",
                primary="#3c5488",  # Nature blue
                secondary="#3c5488",  # Same hue, lighter for raw mean
                band="#3c5488",  # Same hue at 20% opacity
                minmax="#666666",
                grid="#ffffff",  # Gridless
                text="#000000",
                font_family="Helvetica Neue",
                font_size_ticks=7,
                font_size_labels=8,
                primary_linewidth=1.8,  # Thick smoothed mean
                secondary_linewidth=0.8,  # Thin raw mean dashed
                band_alpha=0.25,
                show_grid=False,
            ),
            StyleName.AAAI_CONFERENCE: StyleSpec(
                background="#ffffff",
                primary="#0d3b66",
                secondary="#555555",
                band="#a2c6f2",
                minmax="#555555",
                grid="#dddddd",
                text="#000000",
                font_family="serif",
                font_size_ticks=8,
                font_size_labels=9,
                font_weight_labels="bold",
                primary_linewidth=1.2,
                secondary_linewidth=0.8,
                grid_linestyle=":",
                legend_framealpha=1.0,
                font_size_legend=7,
            ),
            StyleName.SOLARIZED_LIGHT: StyleSpec(
                background="#fdf6e3",
                primary="#b58900",
                secondary="#268bd2",
                band="#93a1a1",
                minmax="#6c71c4",
                grid="#eee8d5",
                text="#586e75",
                font_family="Fira Sans",
                primary_linewidth=1.5,
                band_alpha=0.20,
            ),
            StyleName.DARK_ACADEMIA: StyleSpec(
                background="#292524",
                primary="#ffe6a7",
                secondary="#b5838d",
                band="#6d6875",
                minmax="#a85751",
                grid="#292524",
                text="#ffe6a7",
                font_family="serif",
                band_alpha=0.15,
                show_grid=False,
            ),
            StyleName.BLUE_GREY_SANS: StyleSpec(
                background="#ffffff",
                primary="#0066ff",
                secondary="#666666",
                band="#d0e2ff",
                minmax="#999999",
                grid="#f0f0f0",
                text="#333333",
                font_family="SF Pro Text",
                font_size_labels=9,
                band_alpha=0.30,
                grid_linewidth=1.0,
            ),
        }

        return style_definitions[style]

    @staticmethod
    def _apply_style(style_spec: StyleSpec) -> None:
        """Apply style specification to matplotlib.

        Args:
            style_spec: Complete style specification.
        """
        # Reset to defaults first
        plt.rcdefaults()

        # Get available font with fallback
        available_font: str = FontManager.get_available_font(style_spec.font_family)

        # Apply style parameters
        plt.rcParams.update({
            "figure.facecolor": style_spec.background,
            "axes.facecolor": style_spec.background,
            "text.color": style_spec.text,
            "axes.labelcolor": style_spec.text,
            "xtick.color": style_spec.text,
            "ytick.color": style_spec.text,
            "font.family": available_font,
            "font.size": style_spec.font_size_labels,
            "xtick.labelsize": style_spec.font_size_ticks,
            "ytick.labelsize": style_spec.font_size_ticks,
            "legend.fontsize": style_spec.font_size_legend,
            "axes.labelweight": style_spec.font_weight_labels,
            "grid.alpha": style_spec.grid_alpha,
            "grid.linewidth": style_spec.grid_linewidth,
            "grid.linestyle": style_spec.grid_linestyle,
            "axes.grid": style_spec.show_grid,
            "axes.grid.which": style_spec.grid_which,
            "legend.framealpha": style_spec.legend_framealpha,
        })

        if not style_spec.use_spines:
            plt.rcParams.update({
                "axes.spines.left": False,
                "axes.spines.bottom": False,
                "axes.spines.top": False,
                "axes.spines.right": False,
            })

    def compute_statistics(
        self, mse: NDArray[np.float32], *, streaming: bool = False, chunk_size: int = 10000
    ) -> Statistics:
        """Compute statistical summary of MSE time series.

        Args:
            mse: MSE array of shape (N, T) where N is number of runs, T is time steps.
            streaming: Whether to process time steps in fixed-size chunks.
            chunk_size: Size of chunks for streaming computation.

        Returns:
            Statistical summary containing mean, std, quantiles, etc.
        """
        n_samples, n_steps = mse.shape

        if streaming or self.config.streaming_threshold < n_steps:
            return self._compute_statistics_streaming(mse, chunk_size)

        # Vectorized computation for sizes that fit in memory
        # Compute all basic statistics in single pass to minimize memory access
        # Use ddof=0 when only one sample is available to avoid RuntimeWarnings
        ddof = 0 if n_samples <= 1 else 1
        mean = np.asarray(np.nanmean(mse, axis=0), dtype=np.float32)
        std = np.asarray(np.nanstd(mse, axis=0, ddof=ddof), dtype=np.float32)
        min_vals = np.asarray(np.nanmin(mse, axis=0), dtype=np.float32)
        max_vals = np.asarray(np.nanmax(mse, axis=0), dtype=np.float32)

        # Compute only required quantiles to avoid redundant calculations
        required_quantiles = {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95}
        required_quantiles.update(self.config.quantiles)

        # One nanquantile call computes every required value
        quantile_list = sorted(required_quantiles)
        quantile_arrays = np.nanquantile(mse, quantile_list, axis=0)
        quantile_values = dict(
            zip(quantile_list, [np.asarray(q, dtype=np.float32) for q in quantile_arrays], strict=True)
        )

        # Compute smoothed mean
        smoothed_mean = np.asarray(self._smooth_time_series(mean, n_steps), dtype=np.float32)

        return Statistics(
            mean=mean, std=std, min=min_vals, max=max_vals, quantiles=quantile_values, smoothed_mean=smoothed_mean
        )

    def _compute_statistics_streaming(self, mse: NDArray[np.float32], chunk_size: int) -> Statistics:
        """Compute statistics over fixed-size time-step chunks.

        Args:
            mse: MSE array of shape (N, T).
            chunk_size: Size of temporal chunks.

        Returns:
            Statistical summary.
        """
        n_samples, n_steps = mse.shape
        n_chunks: int = (n_steps + chunk_size - 1) // chunk_size

        # Accumulate summary values one chunk at a time.
        chunk_means: list[NDArray[np.float32]] = []
        chunk_mins: list[NDArray[np.float32]] = []
        chunk_maxs: list[NDArray[np.float32]] = []

        # Process in chunks so large T does not exhaust memory
        for i in range(n_chunks):
            start_idx: int = i * chunk_size
            end_idx: int = min((i + 1) * chunk_size, n_steps)
            chunk: NDArray[np.float32] = mse[:, start_idx:end_idx]

            # Compute statistics for this chunk
            chunk_means.append(np.asarray(np.mean(chunk, axis=0), dtype=np.float32))
            chunk_mins.append(np.asarray(np.min(chunk, axis=0), dtype=np.float32))
            chunk_maxs.append(np.asarray(np.max(chunk, axis=0), dtype=np.float32))

        # Concatenate each chunk into float32 arrays.
        mean: NDArray[np.float32] = np.asarray(np.concatenate(chunk_means), dtype=np.float32)
        min_vals: NDArray[np.float32] = np.asarray(np.concatenate(chunk_mins), dtype=np.float32)
        max_vals: NDArray[np.float32] = np.asarray(np.concatenate(chunk_maxs), dtype=np.float32)

        # For large arrays, fall back to chunked variance computation
        ddof: int = 0 if n_samples <= 1 else 1
        if n_steps <= 100000:  # Exact computation for moderate sizes
            std: NDArray[np.float32] = np.asarray(np.nanstd(mse, axis=0, ddof=ddof), dtype=np.float32)
        else:
            # Approximate std using chunk-based computation
            logger.warning(f"Using approximate std computation for n_steps={n_steps}")
            chunk_vars: list[NDArray[np.float32]] = []
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_steps)
                chunk = mse[:, start_idx:end_idx]
                chunk_vars.append(np.asarray(np.nanvar(chunk, axis=0, ddof=ddof), dtype=np.float32))
            std = np.asarray(np.sqrt(np.concatenate(chunk_vars)), dtype=np.float32)

        # Compute quantiles (may need approximation for very large n_steps)
        quantile_values: dict[float, NDArray[np.float32]] = {}
        required_quantiles = {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95}
        required_quantiles.update(self.config.quantiles)

        if n_steps <= 100000:  # Exact quantiles for reasonable sizes
            quantile_list: list[float] = sorted(required_quantiles)
            quantile_arrays: NDArray[np.float32] = np.nanquantile(mse, quantile_list, axis=0)
            quantile_values = dict(
                zip(quantile_list, [np.asarray(q, dtype=np.float32) for q in quantile_arrays], strict=True)
            )
        else:
            # Approximation for very large n_steps using normal distribution assumption
            logger.warning(f"Using quantile approximation for n_steps={n_steps}")
            # Use local approximation for inverse normal CDF
            for q in required_quantiles:
                # Use normal approximation: quantile ≈ mean + z_score * std
                z_score = _norm_ppf(q)
                quantile_values[q] = np.asarray(mean + z_score * std, dtype=np.float32)

        # Smooth the mean (cast to float32)
        smoothed_mean: NDArray[np.float32] = np.asarray(self._smooth_time_series(mean, n_steps), dtype=np.float32)

        return Statistics(
            mean=mean, std=std, min=min_vals, max=max_vals, quantiles=quantile_values, smoothed_mean=smoothed_mean
        )

    def _smooth_time_series(self, data: NDArray[np.float32], time_steps: int) -> NDArray[np.float32]:
        """Apply the configured smoothing method to a time series.

        Args:
            data: 1D array of time series data.
            time_steps: Length of time series.

        Returns:
            Smoothed time series.
        """
        # Smoothing method dispatch
        if self.config.smoothing_method == SmoothingMethod.SAVGOL:
            # Apply the local Savitzky-Golay implementation.
            window_length: int = max(5, int(time_steps / 100))

            # Savitzky-Golay windows require an odd length.
            if window_length % 2 == 0:
                window_length += 1

            # Cap the window at the series length.
            window_length = min(window_length, len(data))
            if window_length < 5:
                logger.warning(f"Data too short for Savgol filter, using window_length={window_length:d}")
                return data.copy()

            polyorder: int = min(3, window_length - 1)
            filtered: NDArray[np.float32] = savgol_filter(np.asarray(data, dtype=np.float32), window_length, polyorder)
            return np.asarray(filtered, dtype=np.float32)

        if self.config.smoothing_method == SmoothingMethod.EXPONENTIAL:
            # Exponential moving average (standard smoothing used across the codebase)
            # Derive a sensible window from smoothing_window_ratio
            window: int = max(1, int(max(1, time_steps * self.config.smoothing_window_ratio)))
            # Common EMA alpha heuristic
            alpha: float = 2.0 / (window + 1.0)

            out: NDArray[np.float32] = np.empty_like(data, dtype=np.float32)
            s: NDArray[np.float32] = np.asarray(data[0], dtype=np.float32)
            out[0] = s
            for t in range(1, len(data)):
                s = (alpha * np.asarray(data[t], dtype=np.float32) + (1.0 - alpha) * s).astype(np.float32)
                out[t] = s
            return out

        if self.config.smoothing_method == SmoothingMethod.NONE:
            return data.copy()

        raise ValueError(f"Unknown smoothing method: {self.config.smoothing_method}")

    def _decimate_if_needed(
        self, data_arrays: dict[str, NDArray[np.float32] | NDArray[np.intp]], time_steps: int
    ) -> dict[str, NDArray[np.float32] | NDArray[np.intp]]:
        """Decimate data if it exceeds maximum points for performance.

        Args:
            data_arrays: Dictionary of named data arrays.
            time_steps: Original time series length.

        Returns:
            Potentially decimated data arrays.
        """
        if self.config.max_points_before_decimation >= time_steps:
            return data_arrays

        # Clamp the decimation factor to at least one.
        decim_factor: int = max(1, time_steps // self.config.max_points_before_decimation)
        indices = np.arange(0, time_steps, decim_factor)

        logger.info(f"Decimating data from {time_steps:d} to {len(indices):d} points for performance")
        # Decimate all arrays. Keep "steps" integer, others as float32.
        decimated: dict[str, NDArray[np.float32] | NDArray[np.intp]] = {}
        for name, array in data_arrays.items():
            arr: NDArray[np.float32 | np.intp] = np.asarray(array)
            if name == "steps":
                # steps must remain integer indices
                decimated[name] = arr[indices].astype(np.intp)
            else:
                decimated[name] = arr[indices].astype(np.float32)

        return decimated

    def plot_mse(
        self,
        mse: NDArray[np.float32] | dict[str, NDArray[np.float32]],
        *,
        title: str | None = None,
        xlabel: str = "Step",
        ylabel: str = "MSE",
        save_path: str | Path | None = None,
        show_minmax: bool = True,
        show_smoothed_line: bool = True,
        show_raw_mean: bool = True,
        show_legend: bool = True,
        primary_label: str | None = None,
        secondary_label: str | None = None,
        sample_size_text: str | None = None,
        annotation_lines: Sequence[tuple[int, str]] | None = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plot MSE values from one array or a mapping of named arrays.

        Args:
            mse: MSE data. Can be:
                - NDArray of shape (N, T) for single source
                - dict[str, NDArray[np.float32]] mapping source names to MSE arrays for multi-source
            title: Plot title. If None, no title is shown.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            save_path: Path to save figure. If None, figure is not saved.
            show_minmax: Whether to show min/max envelope.
            show_smoothed_line: Whether to show smoothed mean line.
            show_raw_mean: Whether to show raw (unsmoothed) mean.
            show_legend: Whether to display the legend.
            primary_label: Legend label for the smoothed line. Defaults to ``smoothed mean``.
            secondary_label: Legend label for the raw line. Defaults to ``mean``.
            sample_size_text: Override sample size annotation text (e.g., "N = 100"). If None, computed.
            annotation_lines: List of (x_position, label) for vertical annotation lines.
            return_fig: Whether to return the figure object.

        Returns:
            Figure object if return_fig=True, otherwise None.
        """
        # Handle both single and multi-source inputs
        if isinstance(mse, dict):
            return self._plot_mse_multi_source(
                mse,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                save_path=save_path,
                show_minmax=show_minmax,
                show_smoothed_line=show_smoothed_line,
                show_raw_mean=show_raw_mean,
                show_legend=show_legend,
                annotation_lines=annotation_lines,
                return_fig=return_fig,
                primary_label=primary_label,
                secondary_label=secondary_label,
                sample_size_text=sample_size_text,
            )
        return self._plot_mse_single_source(
            mse,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=save_path,
            show_minmax=show_minmax,
            show_smoothed_line=show_smoothed_line,
            show_raw_mean=show_raw_mean,
            show_legend=show_legend,
            primary_label=primary_label,
            secondary_label=secondary_label,
            sample_size_text=sample_size_text,
            annotation_lines=annotation_lines,
            return_fig=return_fig,
        )

    def _plot_mse_single_source(
        self,
        mse: NDArray[np.float32],
        *,
        title: str | None = None,
        xlabel: str = "Step",
        ylabel: str = "MSE",
        save_path: str | Path | None = None,
        show_minmax: bool = True,
        show_smoothed_line: bool = True,
        show_raw_mean: bool = True,
        show_legend: bool = True,
        primary_label: str | None = None,
        secondary_label: str | None = None,
        sample_size_text: str | None = None,
        yscale: Literal["linear", "log", "symlog"] | None = None,
        annotation_lines: Sequence[tuple[int, str]] | None = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plot MSE values from one array of runs and time steps.

        Args:
            mse: MSE array of shape (N, T) where N is number of runs, T is time steps.
            title: Plot title. If None, no title is shown.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            save_path: Path to save figure. If None, figure is not saved.
            show_minmax: Whether to show min/max envelope.
            show_smoothed_line: Whether to show smoothed mean line.
            show_raw_mean: Whether to show raw (unsmoothed) mean.
            show_legend: Whether to display the legend.
            primary_label: Legend label for the smoothed line. Defaults to ``smoothed mean``.
            secondary_label: Legend label for the raw line. Defaults to ``mean``.
            sample_size_text: Override sample size annotation text (e.g., "N = 100"). If None, computed.
            yscale: Override y-axis scale ("linear", "log", or "symlog"). If None, uses config.
            annotation_lines: List of (x_position, label) for vertical annotation lines.
            return_fig: Whether to return the figure. When false, this method
                closes the figure before returning.

        Returns:
            Figure when ``return_fig`` is true, or None after closing the figure.

        Raises:
            ValueError: If mse array has wrong shape or invalid parameters.
        """
        if mse.ndim != 2:
            raise ValueError(f"MSE array must be 2D (N, T), got shape {mse.shape}")

        n_samples: int
        n_steps: int
        n_samples, n_steps = mse.shape

        if n_samples < 1 or n_steps < 1:
            raise ValueError(f"MSE array must have positive dimensions, got shape {mse.shape}")

        # Compute statistics
        stats: Statistics = self.compute_statistics(mse)

        # Get style specification and apply it
        style_spec: StyleSpec = self._get_style_spec(self.config.style)
        self._apply_style(style_spec)

        # Resolve palette with config overrides (single-source uses first color)
        source_label: str = primary_label or secondary_label or "series"
        color_palette: dict[str, list[str]] = self._build_color_palette(style_spec, [source_label])
        primary_color: str = color_palette["primary"][0]
        secondary_color: str = color_palette["secondary"][0]
        band_color: str = color_palette["band"][0]
        minmax_color: str = color_palette["minmax"][0]

        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

        # Prepare data arrays for potential decimation
        steps: NDArray[np.intp] = np.arange(n_steps)
        data_arrays: dict[str, NDArray[np.float32] | NDArray[np.intp]] = {
            "steps": steps,
            "mean": stats.mean,
            "smoothed_mean": stats.smoothed_mean,
            "min": stats.min,
            "max": stats.max,
        }

        # Add quantile data
        q_low, q_high = self.config.quantiles
        if q_low in stats.quantiles and q_high in stats.quantiles:
            data_arrays["q_low"] = stats.quantiles[q_low]
            data_arrays["q_high"] = stats.quantiles[q_high]

        # Use intp steps and float32 values before decimation.
        data_arrays_precise: dict[str, NDArray[np.float32] | NDArray[np.intp]] = {}
        for k, v in data_arrays.items():
            if k == "steps":
                data_arrays_precise[k] = np.asarray(v, dtype=np.intp)
            else:
                data_arrays_precise[k] = np.asarray(v, dtype=np.float32)

        # Decimate if needed for performance.
        data_arrays = self._decimate_if_needed(data_arrays_precise, n_steps)

        # Keep decimated step indices as integers.
        steps = np.asarray(data_arrays["steps"], dtype=np.intp)
        mean = np.asarray(data_arrays["mean"], dtype=np.float32)
        smoothed_mean = np.asarray(data_arrays["smoothed_mean"], dtype=np.float32)
        min_vals = np.asarray(data_arrays["min"], dtype=np.float32)
        max_vals = np.asarray(data_arrays["max"], dtype=np.float32)

        # Plot uncertainty band (lowest z-order for background)
        if self.config.uncertainty_method == UncertaintyMethod.QUANTILES:
            if "q_low" in data_arrays and "q_high" in data_arrays:
                # Convert plot values to float32 rather than object arrays.
                q_low_data = np.asarray(data_arrays["q_low"], dtype=np.float32)
                q_high_data = np.asarray(data_arrays["q_high"], dtype=np.float32)

                # Add hatching if specified in style
                hatch: str | None = style_spec.hatch_pattern if style_spec.use_hatching else None

                # Type-safe call to fill_between with proper array handling
                band_label: str | None = None
                if (self.config.legend_show_band is None and show_legend) or (
                    self.config.legend_show_band is True and show_legend
                ):
                    q_pct_raw: float = (q_high - q_low) * 100
                    q_pct = int(round_percentages([q_pct_raw])[0])
                    band_label = self.config.legend_band_label_quantiles_template.format(percent=q_pct)

                ax.fill_between(
                    steps,  # x values
                    q_low_data,  # y1 values
                    q_high_data,  # y2 values
                    alpha=style_spec.band_alpha,
                    color=band_color,
                    hatch=hatch,
                    label=band_label,
                    zorder=1,  # Background layer
                )

        elif self.config.uncertainty_method == UncertaintyMethod.STD:
            std_data = data_arrays.get("std")
            if std_data is None:
                std_data = stats.std
            # Convert values to float32 before plotting.
            std_data = np.asarray(std_data, dtype=np.float32)
            # Type-safe call to fill_between
            band_label = None
            if (self.config.legend_show_band is None and show_legend) or (
                self.config.legend_show_band is True and show_legend
            ):
                band_label = self.config.legend_band_label_std
            ax.fill_between(
                steps,
                mean - std_data,
                mean + std_data,
                alpha=style_spec.band_alpha,
                color=band_color,
                label=band_label,
                zorder=1,  # Background layer
            )

        # Plot min/max envelope (mid-layer, behind main lines)
        if show_minmax:
            ax.plot(
                steps,
                min_vals,
                linewidth=style_spec.minmax_linewidth,
                linestyle=style_spec.minmax_linestyle,
                alpha=style_spec.minmax_alpha,
                color=minmax_color,
                label=None,
                zorder=2,  # Mid-background layer
            )
            ax.plot(
                steps,
                max_vals,
                linewidth=style_spec.minmax_linewidth,
                linestyle=style_spec.minmax_linestyle,
                alpha=style_spec.minmax_alpha,
                color=minmax_color,
                label=(
                    "min / max"
                    if ((self.config.legend_show_minmax is None and show_legend) or self.config.legend_show_minmax)
                    else None
                ),
                zorder=2,  # Mid-background layer
            )

        # Plot smoothed mean first (with reduced opacity to allow other lines to show through)
        if show_smoothed_line:
            ax.plot(
                steps,
                smoothed_mean,
                linewidth=style_spec.primary_linewidth,
                color=primary_color,
                alpha=0.85,  # Slight transparency to reveal lines behind
                label=(primary_label or "smoothed mean"),
                zorder=3,  # Main data layer
            )

        # When smoothing is disabled, draw the raw mean with the primary solid style.
        if show_raw_mean:
            raw_color = primary_color if not show_smoothed_line else secondary_color
            raw_linewidth = style_spec.primary_linewidth if not show_smoothed_line else style_spec.secondary_linewidth
            raw_linestyle = "-" if not show_smoothed_line else style_spec.secondary_linestyle
            ax.plot(
                steps,
                mean,
                linewidth=raw_linewidth,
                linestyle=raw_linestyle,
                alpha=0.9 if not show_smoothed_line else 0.8,
                color=raw_color,
                label=(
                    (secondary_label or "mean")
                    if ((self.config.legend_show_raw_mean is None and show_legend) or self.config.legend_show_raw_mean)
                    else None
                ),
                zorder=4,  # Top data layer
            )

        # Add annotation lines if specified (top layer)
        if annotation_lines:
            # Place the annotation inside the axes and away from the legend.
            y_min, y_max = ax.get_ylim()
            y_range: float = y_max - y_min

            # Position text in data coordinates, accounting for title space
            text_y_pos: float = y_max - y_range * 0.15 if title else y_max - y_range * 0.08

            for x_pos, label in annotation_lines:
                # Draw vertical line
                ax.axvline(x=x_pos, color=style_spec.text, linestyle="--", alpha=0.5, linewidth=0.8, zorder=5)

                # Draw the value on a translucent background.
                ax.text(
                    x_pos,
                    text_y_pos,
                    label,
                    rotation=90,
                    fontsize=style_spec.font_size_ticks,
                    alpha=0.8,  # Keep the label background mostly opaque.
                    zorder=6,
                    verticalalignment="top",  # Align text from top
                    horizontalalignment="center",  # Center on line
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": style_spec.background,
                        "edgecolor": "none",
                        "alpha": 0.7,
                    },  # Semi-transparent background for readability
                )

        # Configure axes
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title, fontweight="bold")

        selected_scale = yscale or ("log" if self.config.use_log_scale else "linear")
        if selected_scale == "log":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
        elif selected_scale == "symlog":
            ax.set_yscale("symlog", linthresh=self.config.y_symlog_threshold)
            sf = ScalarFormatter(useMathText=True)
            sf.set_powerlimits((-3, 3))
            sf.set_scientific(True)
            ax.yaxis.set_major_formatter(sf)
        elif self.config.y_use_scientific_notation:
            sf = ScalarFormatter(useMathText=True)
            sf.set_powerlimits((-3, 3))
            sf.set_scientific(True)
            ax.yaxis.set_major_formatter(sf)
        else:
            fmt = f"%.{max(0, self.config.y_decimal_places)}f"
            ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

        # Let MaxNLocator cap the number of visible ticks.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        # Configure legend
        if show_legend:
            ax.legend(
                framealpha=style_spec.legend_framealpha,
                loc=style_spec.legend_location,
                fontsize=style_spec.font_size_legend,
            )

        # Configure grid
        if style_spec.show_grid:
            ax.grid(
                True,
                which=style_spec.grid_which,
                alpha=style_spec.grid_alpha,
                linewidth=style_spec.grid_linewidth,
                linestyle=style_spec.grid_linestyle,
            )

        # Apply tight layout with annotation space consideration
        if annotation_lines:
            # Reserve extra space for annotations to prevent overlap
            fig.tight_layout(rect=(0, 0, 1, 0.95))  # Leave 5% at top for annotations
        else:
            fig.tight_layout()

        # Add sample size annotation only when provided and non-empty
        if sample_size_text:
            fig.text(0.02, 0.02, sample_size_text, fontsize=style_spec.font_size_ticks, alpha=0.7)

        # Save figure if path provided
        if save_path:
            self.save_figure(fig, save_path)

        # Close figure to free memory if not returning it
        if not return_fig:
            plt.close(fig)
            return None

        return fig

    def _plot_mse_multi_source(
        self,
        mse_dict: dict[str, NDArray[np.float32]],
        *,
        title: str | None = None,
        xlabel: str = "Step",
        ylabel: str = "MSE",
        save_path: str | Path | None = None,
        show_minmax: bool = True,
        show_smoothed_line: bool = True,
        show_raw_mean: bool = True,
        show_legend: bool = True,
        primary_label: str | None = None,
        secondary_label: str | None = None,
        sample_size_text: str | None = None,
        yscale: Literal["linear", "log", "symlog"] | None = None,
        annotation_lines: Sequence[tuple[int, str]] | None = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plot named MSE arrays on one set of axes.

        Args:
            mse_dict: Dictionary mapping source names to MSE arrays.
            title: Plot title. If None, no title is shown.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            save_path: Path to save figure. If None, figure is not saved.
            show_minmax: Whether to show min/max envelope.
            show_smoothed_line: Whether to show smoothed mean line.
            show_raw_mean: Whether to show raw (unsmoothed) mean.
            show_legend: Whether to display the legend.
            primary_label: Unused in multi-source mode because mapping keys label each source.
            secondary_label: Unused in multi-source mode because mapping keys label each source.
            sample_size_text: Override sample size annotation text (e.g., "N = 100"). If None, computed.
            yscale: Override y-axis scale ("linear", "log", or "symlog"). If None, uses config.
            annotation_lines: List of (x_position, label) for vertical annotation lines.
            return_fig: Whether to return the figure object.

        Returns:
            Figure object if return_fig=True, otherwise None.
        """
        _ = (primary_label, secondary_label)
        if not mse_dict:
            raise ValueError("mse_dict cannot be empty")

        # Validate all arrays have correct shape
        source_names: list[str] = list(mse_dict.keys())
        n_sources: int = len(source_names)
        for name, mse in mse_dict.items():
            if mse.ndim != 2:
                raise ValueError(f"MSE array for '{name}' must be 2D (N, T), got shape {mse.shape}")
            n_samples, n_steps = mse.shape
            if n_samples < 1 or n_steps < 1:
                raise ValueError(f"MSE array for '{name}' must have positive dimensions, got shape {mse.shape}")

        # Get style specification and apply it
        style_spec: StyleSpec = self._get_style_spec(self.config.style)
        self._apply_style(style_spec)

        # Get multi-source color palette (respects config-provided colors)
        color_palette: dict[str, list[str]] = self._build_color_palette(style_spec, source_names)

        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

        # Process each source
        all_statistics: dict[str, Statistics] = {}
        all_decimated_data: dict[str, dict[str, NDArray[np.float32] | NDArray[np.intp]]] = {}

        stats: Statistics
        for source_name, mse in mse_dict.items():
            # Compute statistics for this source
            stats = self.compute_statistics(mse)
            all_statistics[source_name] = stats

            n_samples, n_steps = mse.shape

            # Prepare data arrays for potential decimation
            steps: NDArray[np.intp] = np.arange(n_steps, dtype=np.intp)
            data_arrays: dict[str, NDArray[np.float32] | NDArray[np.intp]] = {
                "steps": steps,
                "mean": stats.mean,
                "smoothed_mean": stats.smoothed_mean,
                "min": stats.min,
                "max": stats.max,
            }

            # Add quantile data
            q_low, q_high = self.config.quantiles
            if q_low in stats.quantiles and q_high in stats.quantiles:
                data_arrays["q_low"] = stats.quantiles[q_low]
                data_arrays["q_high"] = stats.quantiles[q_high]

            data_arrays_precise: dict[str, NDArray[np.float32] | NDArray[np.intp]] = {}
            for k, v in data_arrays.items():
                if k == "steps":
                    data_arrays_precise[k] = np.asarray(v, dtype=np.intp)
                else:
                    data_arrays_precise[k] = np.asarray(v, dtype=np.float32)

            # Decimate if needed for performance (keep precise typing)
            decimated: dict[str, NDArray[np.float32] | NDArray[np.intp]] = self._decimate_if_needed(
                data_arrays_precise, n_steps
            )
            all_decimated_data[source_name] = decimated

        # Plot each source with distinct colors from the palette
        for i, source_name in enumerate(source_names):
            # Retrieve decimated arrays for this source
            decimated_arrays: dict[str, NDArray[np.float32] | NDArray[np.intp]] = all_decimated_data[source_name]
            stats = all_statistics[source_name]

            # Extract colors for this source
            primary_color: str = color_palette["primary"][i]
            secondary_color: str = color_palette["secondary"][i]
            band_color: str = color_palette["band"][i]
            minmax_color: str = color_palette["minmax"][i]

            # Extract decimated arrays
            steps = np.asarray(decimated_arrays["steps"], dtype=np.intp)
            mean: NDArray[np.float32] = np.asarray(decimated_arrays["mean"], dtype=np.float32)
            smoothed_mean: NDArray[np.float32] = np.asarray(decimated_arrays["smoothed_mean"], dtype=np.float32)
            min_vals: NDArray[np.float32] = np.asarray(decimated_arrays["min"], dtype=np.float32)
            max_vals: NDArray[np.float32] = np.asarray(decimated_arrays["max"], dtype=np.float32)

            # Plot uncertainty band (lowest z-order for background)
            if self.config.uncertainty_method == UncertaintyMethod.QUANTILES:
                if "q_low" in decimated_arrays and "q_high" in decimated_arrays:
                    q_low_data: NDArray[np.float32] = np.asarray(decimated_arrays["q_low"], dtype=np.float32)
                    q_high_data: NDArray[np.float32] = np.asarray(decimated_arrays["q_high"], dtype=np.float32)

                    # Add hatching if specified in style
                    hatch: str | None = style_spec.hatch_pattern if style_spec.use_hatching else None
                    q_pct_raw: float = (self.config.quantiles[1] - self.config.quantiles[0]) * 100
                    q_pct: int = int(round_percentages([q_pct_raw])[0])

                    ax.fill_between(
                        steps,
                        q_low_data,
                        q_high_data,
                        alpha=style_spec.band_alpha,
                        color=band_color,
                        hatch=hatch,
                        label=f"{source_name} {q_pct}%" if i == 0 else None,
                        zorder=1,
                    )

            elif self.config.uncertainty_method == UncertaintyMethod.STD:
                std_data: NDArray[np.float32] | NDArray[np.intp] | None = decimated_arrays.get("std")
                if std_data is None:
                    std_data = stats.std
                std_data = np.asarray(std_data, dtype=np.float32)
                ax.fill_between(
                    steps,
                    np.asarray(mean - std_data, dtype=np.float32),
                    np.asarray(mean + std_data, dtype=np.float32),
                    alpha=style_spec.band_alpha,
                    color=band_color,
                    zorder=1,
                )
            # Plot min/max envelope (mid-layer, behind main lines)
            if show_minmax and i == 0:  # Only show for first source to avoid clutter
                ax.plot(
                    steps,
                    min_vals,
                    linewidth=style_spec.minmax_linewidth,
                    linestyle=style_spec.minmax_linestyle,
                    alpha=style_spec.minmax_alpha,
                    color=minmax_color,
                    label=None,
                    zorder=2,
                )
                ax.plot(
                    steps,
                    max_vals,
                    linewidth=style_spec.minmax_linewidth,
                    linestyle=style_spec.minmax_linestyle,
                    alpha=style_spec.minmax_alpha,
                    color=minmax_color,
                    label=(
                        "min / max"
                        if ((self.config.legend_show_minmax is None and show_legend) or self.config.legend_show_minmax)
                        else None
                    ),
                    zorder=2,
                )

            # Plot smoothed mean (main line)
            if show_smoothed_line:
                ax.plot(
                    steps,
                    smoothed_mean,
                    linewidth=style_spec.primary_linewidth,
                    color=primary_color,
                    alpha=0.85,
                    label=f"{source_name}",
                    zorder=3,
                )

            # When smoothing is disabled, draw each raw mean with its primary solid style.
            if show_raw_mean:
                r_color: str = primary_color if not show_smoothed_line else secondary_color
                r_linewidth: float = (
                    style_spec.primary_linewidth if not show_smoothed_line else style_spec.secondary_linewidth
                )
                r_linestyle: LineStyleType = "-" if not show_smoothed_line else style_spec.secondary_linestyle
                r_label: str | None = (
                    source_name if not show_smoothed_line else (f"{source_name} (raw)" if n_sources <= 3 else None)
                )
                ax.plot(
                    steps,
                    mean,
                    linewidth=r_linewidth,
                    linestyle=r_linestyle,
                    alpha=0.9 if not show_smoothed_line else 0.6,
                    color=r_color,
                    label=r_label,
                    zorder=4,
                )

        # Add annotation lines if specified (top layer)
        if annotation_lines:
            y_min, y_max = ax.get_ylim()
            y_range: float = y_max - y_min
            text_y_positions: NDArray[np.float32] = np.linspace(
                y_max - 0.1 * y_range, y_max - 0.3 * y_range, len(annotation_lines), dtype=np.float32
            )

            for (x_pos, label), text_y in zip(annotation_lines, text_y_positions, strict=True):
                ax.axvline(x=x_pos, color=style_spec.text, alpha=0.5, linestyle="--", linewidth=0.8, zorder=5)
                ax.text(x_pos, text_y, label, fontsize=style_spec.font_size_ticks, alpha=0.8, ha="center")

        # Configure axes
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title, fontsize=style_spec.font_size_labels, fontweight=style_spec.font_weight_labels)

        # Unified y-axis scaling and formatting
        selected_scale = yscale or ("log" if self.config.use_log_scale else "linear")
        if selected_scale == "log":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
        elif selected_scale == "symlog":
            ax.set_yscale("symlog", linthresh=self.config.y_symlog_threshold)
            sf = ScalarFormatter(useMathText=True)
            sf.set_powerlimits((-3, 3))
            sf.set_scientific(True)
            ax.yaxis.set_major_formatter(sf)
        elif self.config.y_use_scientific_notation:
            sf = ScalarFormatter(useMathText=True)
            sf.set_powerlimits((-3, 3))
            sf.set_scientific(True)
            ax.yaxis.set_major_formatter(sf)
        else:
            fmt = f"%.{max(0, self.config.y_decimal_places)}f"
            ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

        # Let MaxNLocator cap the number of visible ticks.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        # Configure legend (more selective for multi-source to avoid clutter)
        if show_legend:
            legend_elements: list[Artist] = []
            if show_smoothed_line:
                for i, source_name in enumerate(source_names):
                    # Smoothed entries
                    line = Line2D(
                        [0],
                        [0],
                        color=color_palette["primary"][i],
                        linewidth=style_spec.primary_linewidth,
                        label=self.config.legend_smoothed_label_template.format(source=source_name),
                    )
                    legend_elements.append(line)
                if self.config.legend_show_raw_mean is True and show_raw_mean:
                    for i, source_name in enumerate(source_names):
                        raw_line = Line2D(
                            [0],
                            [0],
                            color=color_palette["secondary"][i],
                            linewidth=style_spec.secondary_linewidth,
                            linestyle=style_spec.secondary_linestyle,
                            alpha=0.6,
                            label=self.config.legend_raw_label_template.format(source=source_name),
                        )
                        legend_elements.append(raw_line)
            else:
                # Raw-only: single solid entry per source using primary palette
                for i, source_name in enumerate(source_names):
                    line = Line2D(
                        [0],
                        [0],
                        color=color_palette["primary"][i],
                        linewidth=style_spec.primary_linewidth,
                        linestyle="-",
                        label=source_name,
                    )
                    legend_elements.append(line)

            # Optionally include a MIN/MAX legend sample (single entry)
            if self.config.legend_show_minmax is True and show_minmax:
                minmax_line = Line2D(
                    [0],
                    [0],
                    color=color_palette["minmax"][0] if color_palette.get("minmax") else style_spec.minmax,
                    linewidth=style_spec.minmax_linewidth,
                    linestyle=style_spec.minmax_linestyle,
                    alpha=style_spec.minmax_alpha,
                    label=self.config.legend_minmax_label,
                )
                legend_elements.append(minmax_line)

            # Add optional uncertainty band legend (only once) when bands are plotted
            if self.config.legend_show_band is not False and (
                self.config.uncertainty_method in {UncertaintyMethod.QUANTILES, UncertaintyMethod.STD}
            ):
                if self.config.uncertainty_method == UncertaintyMethod.QUANTILES:
                    q_low, q_high = self.config.quantiles
                    q_pct_raw = (q_high - q_low) * 100
                    q_pct = int(round_percentages([q_pct_raw])[0])
                    band_label = self.config.legend_band_label_quantiles_template.format(percent=q_pct)
                else:
                    # The enclosing guard restricts the method to QUANTILES or STD.
                    band_label = self.config.legend_band_label_std

                legend_elements.append(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=(color_palette.get("band", [style_spec.band])[0]),
                        alpha=style_spec.band_alpha,
                        label=band_label,
                    )
                )

            ax.legend(
                handles=legend_elements,
                framealpha=style_spec.legend_framealpha,
                loc=style_spec.legend_location,
                fontsize=style_spec.font_size_legend,
            )

        # Configure grid
        if style_spec.show_grid:
            ax.grid(
                True,
                alpha=style_spec.grid_alpha,
                linewidth=style_spec.grid_linewidth,
                linestyle=style_spec.grid_linestyle,
                which=style_spec.grid_which,
            )

        # Apply tight layout with annotation space consideration
        if annotation_lines:
            fig.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            fig.tight_layout()

        # Add sample size annotation only when provided and non-empty
        if sample_size_text:
            fig.text(0.02, 0.02, sample_size_text, fontsize=style_spec.font_size_ticks, alpha=0.7)

        # Save figure if path provided
        if save_path:
            self.save_figure(fig, save_path)

        # Close figure to free memory if not returning it
        if not return_fig:
            plt.close(fig)
            return None

        return fig

    def save_figure(self, fig: Figure, save_path: str | Path, *, formats: Sequence[str] | None = None) -> None:
        """Save a figure in each requested format.

        Args:
            fig: Figure to save.
            save_path: Base path for saving (without extension).
            formats: File formats to save. Uses config default if None.
        """
        save_path = Path(save_path)
        formats = formats or self.config.export_formats

        # Create the parent directory before writing any format.
        save_path.parent.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            output_path = save_path.with_suffix(f".{fmt}")

            if fmt.lower() == "pdf":
                fig.savefig(str(output_path), format="pdf", bbox_inches="tight", pad_inches=0.1, backend="pdf")
            elif fmt.lower() == "png":
                fig.savefig(
                    str(output_path),
                    format="png",
                    dpi=self.config.dpi,
                    bbox_inches="tight",
                    pad_inches=0.1,
                    transparent=self.config.png_transparent,
                )
            elif fmt.lower() == "svg":
                fig.savefig(str(output_path), format="svg", bbox_inches="tight", pad_inches=0.1)
            else:
                logger.warning(f"Unknown format {fmt}, saving as-is")
                fig.savefig(str(output_path), bbox_inches="tight", pad_inches=0.1)

        logger.info(f"Saved figure to {save_path} in formats: {list(formats)}")

    def create_style_comparison(
        self, mse: NDArray[np.float32], styles: Sequence[StyleName] | None = None, save_path: str | Path | None = None
    ) -> Figure:
        """Create comparison plot showing multiple styles.

        Args:
            mse: MSE array of shape (N, T).
            styles: Styles to compare. Uses subset if None.
            save_path: Path to save comparison figure.

        Returns:
            Figure with style comparison.
        """
        if styles is None:
            # Select representative styles
            styles = [
                StyleName.NATURE_JOURNAL,
                StyleName.AAAI_CONFERENCE,
                StyleName.CLASSIC_MINIMALIST,
                StyleName.DARK_MODE_GREYSCALE,
            ]

        n_styles = len(styles)
        cols = 2
        rows = (n_styles + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols, figsize=(self.config.figsize[0] * cols, self.config.figsize[1] * rows), dpi=self.config.dpi
        )

        # Flatten axes for one-dimensional indexing.
        axes_array: list[Axes] = list(axes.flat) if isinstance(axes, np.ndarray) else [axes]

        for i, style in enumerate(styles):
            # Temporarily change style
            original_style = self.config.style
            self.config = replace(self.config, style=style)

            # Create subplot with this style
            ax = axes_array[i]
            stats = self.compute_statistics(mse)
            style_spec = self._get_style_spec(style)

            # Apply minimal styling for comparison with proper z-order
            steps = np.arange(len(stats.mean))

            ax.fill_between(
                steps,
                stats.quantiles[self.config.quantiles[0]],
                stats.quantiles[self.config.quantiles[1]],
                alpha=style_spec.band_alpha,
                color=style_spec.band,
                zorder=1,  # Background layer
            )

            ax.plot(
                steps,
                stats.smoothed_mean,
                linewidth=style_spec.primary_linewidth,
                color=style_spec.primary,
                alpha=0.85,  # Slight transparency for consistency
                zorder=3,  # Main data layer
            )

            ax.set_title(style.value.replace("_", " ").title())
            ax.set_facecolor(style_spec.background)

            # Restore original style
            self.config = replace(self.config, style=original_style)

        # Hide unused subplots
        for i in range(n_styles, len(axes_array)):
            ax_to_hide = axes_array[i]
            ax_to_hide.set_visible(False)

        fig.tight_layout()

        if save_path:
            self.save_figure(fig, save_path)

        return fig


# Module-level plotting functions


def plot_mse_with_defaults(
    mse: NDArray[np.float32] | dict[str, NDArray[np.float32]],
    *,
    style: StyleName = StyleName.NATURE_JOURNAL,
    title: str | None = None,
    xlabel: str = "Step",
    ylabel: str = "MSE",
    save_path: str | Path | None = None,
    show_minmax: bool = True,
    show_smoothed_line: bool = True,
    show_raw_mean: bool = True,
    show_legend: bool = True,
    primary_label: str | None = None,
    secondary_label: str | None = None,
    sample_size_text: str | None = None,
    annotation_lines: Sequence[tuple[int, str]] | None = None,
    return_fig: bool = False,
) -> Figure | None:
    """Plot one or more MSE sources with the default configuration.

    Args:
        mse: One MSE array or a mapping from source names to arrays.
        style: Plot style preset.
        title: Optional plot title.
        xlabel: Label for the horizontal axis.
        ylabel: Label for the vertical axis.
        save_path: Optional output path.
        show_minmax: Whether to draw the minimum and maximum envelope.
        show_smoothed_line: Whether to draw the smoothed mean.
        show_raw_mean: Whether to draw the unsmoothed mean.
        show_legend: Whether to display the legend.
        primary_label: Optional label for the smoothed line.
        secondary_label: Optional label for the raw line.
        sample_size_text: Optional sample-count annotation.
        annotation_lines: Optional vertical annotations as ``(step, label)``
            pairs.
        return_fig: Whether to return the figure instead of closing it.

    Returns:
        The figure when ``return_fig`` is true, otherwise ``None``.
    """
    config = PlotConfig(style=style)
    plotter = MSEPlotter(config)
    return plotter.plot_mse(
        mse,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=save_path,
        show_minmax=show_minmax,
        show_smoothed_line=show_smoothed_line,
        show_raw_mean=show_raw_mean,
        show_legend=show_legend,
        primary_label=primary_label,
        secondary_label=secondary_label,
        sample_size_text=sample_size_text,
        annotation_lines=annotation_lines,
        return_fig=return_fig,
    )


def compare_mse_series(
    mse_dict: dict[str, NDArray[np.float32]],
    *,
    style: StyleName = StyleName.NATURE_JOURNAL,
    save_path: str | Path | None = None,
    ncols: int = 2,
) -> Figure:
    """Compare multiple MSE series in a grid layout.

    Args:
        mse_dict: Dictionary mapping names to MSE arrays.
        style: Visual style to use.
        save_path: Optional save path.
        ncols: Number of columns in grid.

    Returns:
        Figure with comparison grid.
    """
    n_series = len(mse_dict)
    nrows = (n_series + ncols - 1) // ncols

    config = PlotConfig(style=style)
    plotter = MSEPlotter(config)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(config.figsize[0] * ncols, config.figsize[1] * nrows), dpi=config.dpi
    )

    # Flatten axes for one-dimensional indexing.
    axes_array: list[Axes] = list(axes.flat) if isinstance(axes, np.ndarray) else [axes]

    for i, (name, mse) in enumerate(mse_dict.items()):
        ax = axes_array[i]

        # Plot on specific axis - use current figure/axis instead of creating new one
        plt.sca(ax)

        # Temporarily create a plotter instance for this subplot
        temp_fig, _ = plt.subplots(figsize=(1, 1))  # Dummy figure
        plt.close(temp_fig)  # The comparison uses only the configured plotter.

        # Plot directly on our axis
        stats = plotter.compute_statistics(mse)
        style_spec = plotter.get_style_spec(plotter.config.style)

        # Apply minimal styling for comparison
        steps = np.arange(len(stats.mean))

        ax.fill_between(
            steps,
            stats.quantiles[plotter.config.quantiles[0]],
            stats.quantiles[plotter.config.quantiles[1]],
            alpha=style_spec.band_alpha,
            color=style_spec.band,
            zorder=1,
        )

        ax.plot(
            steps,
            stats.smoothed_mean,
            linewidth=style_spec.primary_linewidth,
            color=style_spec.primary,
            alpha=0.85,
            zorder=3,
        )

        ax.set_title(name)

    # Hide unused subplots
    for i in range(n_series, len(axes_array)):
        ax_to_hide = axes_array[i]
        ax_to_hide.set_visible(False)

    fig.tight_layout()

    if save_path:
        plotter.save_figure(fig, save_path)

    return fig


class FontManager:
    """font management with graceful fallbacks."""

    # Font fallback chains for each category
    _FONT_FALLBACKS: ClassVar[dict[str, list[str]]] = {
        "Helvetica Neue": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "Inter": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "Fira Sans": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "SF Pro Text": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "Helvetica": ["Arial", "DejaVu Sans", "sans-serif"],
        "serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "monospace": ["DejaVu Sans Mono", "Monaco", "Courier New", "monospace"],
        "sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    }

    _font_cache: ClassVar[dict[str, str]] = {}

    @classmethod
    def get_available_font(cls, requested_font: str) -> str:
        """Get available font with fallback.

        Args:
            requested_font: The desired font family name.

        Returns:
            Available font name (may be a fallback).
        """
        if requested_font in cls._font_cache:
            return cls._font_cache[requested_font]

        # Treat generic families as always available to avoid noisy warnings
        if requested_font in {"serif", "sans-serif", "monospace"}:
            cls._font_cache[requested_font] = requested_font
            return requested_font

        # Check if the requested font is available
        available_fonts: list[str] = [f.name for f in fm.fontManager.ttflist]

        # Try exact match first
        if requested_font in available_fonts:
            cls._font_cache[requested_font] = requested_font
            return requested_font

        # Try fallback chain
        fallbacks: list[str] = cls._FONT_FALLBACKS.get(requested_font, [requested_font])
        for fallback in fallbacks:
            if fallback in {"serif", "sans-serif", "monospace"}:
                cls._font_cache[requested_font] = fallback
                return fallback
            if fallback in available_fonts:
                cls._font_cache[requested_font] = fallback
                logger.info(f"Font '{requested_font}' not available, using fallback: '{fallback}'")
                return fallback

        # Use generic fallback based on font category
        if "mono" in requested_font.lower():
            fallback = "monospace"
        elif any(serif in requested_font.lower() for serif in ["serif", "times", "baskerville"]):
            fallback = "serif"
        else:
            fallback = "sans-serif"

        cls._font_cache[requested_font] = fallback
        if fallback not in {"serif", "sans-serif", "monospace"}:
            logger.warning(f"Font '{requested_font}' and fallbacks not available, using: '{fallback}'")
        return fallback

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the font cache."""
        cls._font_cache.clear()
