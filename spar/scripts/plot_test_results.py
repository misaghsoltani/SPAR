"""Plot reconstruction MSE from model-test metric files.

The stage can plot one variant, compare discrete and continuous models, or
compare several variants. Inputs may be explicit files or directory patterns.
"""

from __future__ import annotations

import fnmatch
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import orjson
from rich import box
from rich.panel import Panel
from rich.table import Table

from spar.utils.config_utils.config_schema import MSEPlotterSPARConfig
from spar.utils.viz_utils.mse_plotter import MSEPlotter, PlotConfig, SmoothingMethod, StyleName, UncertaintyMethod

if TYPE_CHECKING:
    from logging import Logger
    from typing import NoReturn, TypeAlias

    from numpy.typing import NDArray

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.utils.config_utils.config_schema import ColorSchemeConfig, MSEPlotterConfig, SPARConfig


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
EpisodeKey: TypeAlias = Literal["min", "max", "mean"]
FloatMetricKey: TypeAlias = Literal["min_val_overall", "max_val_overall", "mean_val_overall"]


logger: Logger = getLogger(__name__)


class EpisodeInfo(TypedDict, total=False):
    """TypedDict for individual episode entries in the metric JSON."""

    values_per_step: list[float]


class MetricInfo(TypedDict, total=False):
    """TypedDict for metric information stored per-variant per-metric."""

    min: EpisodeInfo
    max: EpisodeInfo
    mean: EpisodeInfo
    num_episodes: int | float | str
    min_val_overall: float
    max_val_overall: float
    mean_val_overall: float


def _title_from_identifier(name: str) -> str:
    """Convert a snake-case or kebab-case identifier to title case."""
    friendly = name.replace("_", " ").replace("-", " ")
    # Collapse multiple spaces
    friendly = " ".join(friendly.split())
    return friendly.title()


def _variant_display_name(name: str, plotter_cfg: MSEPlotterConfig) -> str:
    """Return the display label for a variant, honoring config overrides.

    Names may include a model-type suffix, which is removed before lookup.
    Falls back to humanized base name when no override is provided.
    """
    # Strip any model suffix that may be present
    base: str = name.replace("_discrete", "").replace("_continuous", "")
    try:
        label: str | None = _variant_label_override(base, plotter_cfg)
    except Exception:
        label = None
    if label is not None:
        return label
    return _title_from_identifier(base)


def _variant_label_override(base: str, plotter_cfg: MSEPlotterConfig) -> str | None:
    overrides = getattr(plotter_cfg, "variant_label_overrides", None)
    if not overrides:
        return None
    # Some configs may provide OmegaConf or mapping-like objects.
    try:
        label = overrides.get(base)
    except Exception:
        # Fallback for attribute-style access (unlikely).
        label = getattr(overrides, base, None)
    return label if isinstance(label, str) and label.strip() else None


def _discover_files_from_directory(
    input_directory: str | Path,
    file_pattern: str = "*.json",
    discrete_pattern: str = "*discrete*",
    continuous_pattern: str = "*continuous*",
) -> tuple[list[Path], list[Path]]:
    """Discover discrete and continuous model files from a directory.

    Args:
        input_directory: Directory to search for files.
        file_pattern: Pattern for JSON files.
        discrete_pattern: Pattern for discrete model files.
        continuous_pattern: Pattern for continuous model files.

    Returns:
        Tuple of (discrete_files, continuous_files).
    """
    input_dir = Path(input_directory)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all JSON files matching the pattern (non-recursive, then recursive fallback)
    all_files: list[Path] = list(input_dir.glob(file_pattern))
    if not all_files:
        all_files = list(input_dir.rglob(file_pattern))

    # Separate discrete and continuous files
    discrete_files: list[Path] = []
    continuous_files: list[Path] = []

    for file_path in all_files:
        file_str: str = str(file_path)
        name_str: str = file_path.name

        # Prefer explicit patterns from config
        matches_discrete: bool = fnmatch.fnmatch(name_str, discrete_pattern) or fnmatch.fnmatch(
            file_str, discrete_pattern
        )
        matches_continuous: bool = fnmatch.fnmatch(name_str, continuous_pattern) or fnmatch.fnmatch(
            file_str, continuous_pattern
        )

        if matches_discrete and not matches_continuous:
            discrete_files.append(file_path)
            continue
        if matches_continuous and not matches_discrete:
            continuous_files.append(file_path)
            continue

        # Fallback to substring inference from path parts
        lower_path: str = file_str.lower()
        if "discrete" in lower_path and "continuous" not in lower_path:
            discrete_files.append(file_path)
        elif "continuous" in lower_path and "discrete" not in lower_path:
            continuous_files.append(file_path)
        else:
            # Ambiguous - default to discrete but log it
            logger.debug(f"Ambiguous file type for {file_path}, defaulting to discrete")
            discrete_files.append(file_path)

    return discrete_files, continuous_files


# The sample-size helper is omitted because annotations are disabled.


def _load_single_variant_metrics(file_path: str | Path) -> tuple[str, dict[str, MetricInfo]]:
    """Load metrics data from a single variant JSON file.

    Args:
        file_path: Path to the single-variant metrics JSON file.

    Returns:
        Tuple of (variant_name, metrics_data).

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        ValueError: If the file format is invalid.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Metrics data file not found: {file_path}")

    try:
        return _parse_single_variant_metrics(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load or parse metrics data file {file_path}: {e}") from e


def _parse_single_variant_metrics(file_path: Path) -> tuple[str, dict[str, MetricInfo]]:
    data: dict[str, JSONValue] = orjson.loads(file_path.read_bytes())

    variant_name_raw: JSONValue | None = data.get("variant_name")
    metrics_raw: JSONValue | None = data.get("metrics")
    if not isinstance(variant_name_raw, str) or not isinstance(metrics_raw, dict):
        _raise_invalid_metrics_format()

    variant_name: str = variant_name_raw
    metrics_data: dict[str, MetricInfo] = {}
    metric_name: str
    metric_payload: JSONValue
    for metric_name, metric_payload in metrics_raw.items():
        if not isinstance(metric_payload, dict):
            continue
        metric_info: MetricInfo = {}
        for episode_key in ("min", "max", "mean"):
            episode_values = _extract_episode_values_from_payload(metric_payload, episode_key)
            if episode_values is not None:
                metric_info[episode_key] = {"values_per_step": episode_values}
        num_episodes_value: JSONValue | None = metric_payload.get("num_episodes")
        if isinstance(num_episodes_value, (int, float, str)):
            metric_info["num_episodes"] = num_episodes_value
        float_metric_keys: tuple[FloatMetricKey, ...] = ("min_val_overall", "max_val_overall", "mean_val_overall")
        for float_key in float_metric_keys:
            float_value: JSONValue | None = metric_payload.get(float_key)
            if isinstance(float_value, (int, float)):
                metric_info[float_key] = float(float_value)
        metrics_data[metric_name] = metric_info

    logger.debug(f"Loaded variant '{variant_name}' from {file_path.name}")
    return variant_name, metrics_data


def _raise_invalid_metrics_format() -> NoReturn:
    """Raise a stable parsing error for malformed metrics files."""
    raise ValueError("Invalid JSON format: missing 'variant_name' or 'metrics' fields")


def _extract_episode_values_from_payload(
    metric_payload: dict[str, JSONValue], episode_key: EpisodeKey
) -> list[float] | None:
    """Extract a float list for one episode type from raw metric payload."""
    episode_payload: JSONValue | None = metric_payload.get(episode_key)
    if not isinstance(episode_payload, dict):
        return None
    values_raw: JSONValue | None = episode_payload.get("values_per_step")
    if not isinstance(values_raw, list) or not values_raw:
        return None
    values: list[float] = [float(value) for value in values_raw if isinstance(value, (int, float))]
    return values or None


def _extract_episode_values(metric_info: MetricInfo, episode_key: EpisodeKey) -> list[float] | None:
    """Extract episode values from typed metric info."""
    episode_info = metric_info.get(episode_key)
    if not isinstance(episode_info, dict):
        return None
    values = episode_info.get("values_per_step")
    if not isinstance(values, list) or not values:
        return None
    return [float(x) for x in values]


def _convert_to_metric_arrays(
    data: dict[str, dict[str, MetricInfo]], metric_key: str = "reconstruction_mse"
) -> dict[str, NDArray[np.float32]]:
    """Convert new structured JSON data to numpy arrays suitable for plotting.

    Args:
        data: Structured metrics data from new JSON format.
        metric_key: The metric key to extract for plotting.

    Returns:
        Dictionary mapping variant names to metric arrays of shape (N_episodes, N_steps).
    """
    metric_arrays: dict[str, NDArray[np.float32]] = {}

    for variant_name, metrics_data in data.items():
        metric_info: MetricInfo | None = metrics_data.get(metric_key)
        if not isinstance(metric_info, dict):
            logger.warning(f"Metric '{metric_key}' not found or invalid for variant '{variant_name}'")
            continue

        # Collect available episode series
        episode_data_lists: list[list[float]] = []
        episode_keys: tuple[EpisodeKey, ...] = ("min", "max", "mean")
        for ep_name in episode_keys:
            values: list[float] | None = _extract_episode_values(metric_info, ep_name)
            if values is not None:
                episode_data_lists.append(values)

        if not episode_data_lists:
            logger.warning(f"No valid episode data found for metric '{metric_key}' in variant '{variant_name}'")
            continue

        # Pad sequences to equal length and convert to numpy array
        max_length: int = max(len(ep) for ep in episode_data_lists)
        padded_data: list[list[float]] = [
            list(ep) + [float(np.nan)] * (max_length - len(ep)) for ep in episode_data_lists
        ]
        metric_array: NDArray[np.float32] = np.array(padded_data, dtype=np.float32)
        metric_arrays[variant_name] = metric_array

        # Log conversion info
        num_episodes: int | float | str = metric_info.get("num_episodes", len(episode_data_lists))
        try:
            logger.info(
                f"Converted variant '{variant_name}': metric '{metric_key}', "
                f"{len(episode_data_lists)} episode samples, {max_length} steps, "
                f"shape {metric_array.shape}"
            )
            logger.info(
                f"  - Data range: {metric_info.get('min_val_overall', 'N/A')} "
                f"to {metric_info.get('max_val_overall', 'N/A')}"
            )
            logger.info(f"  - Overall mean: {metric_info.get('mean_val_overall', 'N/A')}")
            logger.info(f"  - Total episodes in dataset: {num_episodes}")
        except Exception:
            # A logging failure does not invalidate the converted metrics.
            pass

    return metric_arrays


def run_mse_plotter(_env: ABCEnvironment[ABCState], cfg: SPARConfig) -> None:
    """Run the MSE plotter stage using mse_plotter utility.

    The stage loads JSON files written by ``MetricsTracker`` and creates plots
    for individual variants or model comparisons.

    Args:
        _env: Environment instance (not used but required by stage interface).
        cfg: SPAR configuration containing plotter settings.
    """
    logger.info("Starting MSE plotter stage")

    if not isinstance(cfg, MSEPlotterSPARConfig):
        raise TypeError("run_mse_plotter expects MSEPlotterSPARConfig")
    mse_plotter_cfg: MSEPlotterSPARConfig = cfg
    plotter_cfg: MSEPlotterConfig = mse_plotter_cfg.plotter

    # Create output directory
    output_dir = Path(plotter_cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")

    # Choose plotting approach based on plot_mode
    if plotter_cfg.plot_mode == "directory":
        _handle_directory_based_plotting(plotter_cfg, output_dir)
    elif plotter_cfg.plot_mode == "files":
        _handle_file_based_plotting(plotter_cfg, output_dir)
    else:
        raise ValueError(f"Unknown plot_mode: {plotter_cfg.plot_mode}. Must be 'directory' or 'files'")


def _handle_directory_based_plotting(plotter_cfg: MSEPlotterConfig, output_dir: Path) -> None:
    """Handle directory-based plotting with automatic model type separation.

    Args:
        plotter_cfg: Plotter configuration.
        output_dir: Output directory for plots.
    """
    logger.info("Using directory-based plotting mode")

    # Discover discrete, continuous, and optional extra files
    discrete_files: list[Path] = []
    continuous_files: list[Path] = []
    extra_files: list[Path] = []

    if plotter_cfg.discrete_input_directory or plotter_cfg.continuous_input_directory:
        # Use separate directories when provided (they take precedence over input_directory)
        if plotter_cfg.discrete_input_directory:
            ddir = Path(plotter_cfg.discrete_input_directory)
            if not ddir.exists():
                raise FileNotFoundError(f"Discrete input directory not found: {ddir}")
            # Prefer the discrete-model pattern, then use the shared file pattern.
            dpattern: str = plotter_cfg.discrete_file_pattern or plotter_cfg.file_pattern
            discrete_files = list(ddir.glob(dpattern))
            if not discrete_files:
                discrete_files = list(ddir.rglob(dpattern))

        if plotter_cfg.continuous_input_directory:
            cdir = Path(plotter_cfg.continuous_input_directory)
            if not cdir.exists():
                raise FileNotFoundError(f"Continuous input directory not found: {cdir}")

            cpattern: str = plotter_cfg.continuous_file_pattern or plotter_cfg.file_pattern
            continuous_files = list(cdir.glob(cpattern))
            if not continuous_files:
                continuous_files = list(cdir.rglob(cpattern))
        extra_dir_val: str | None = getattr(plotter_cfg, "extra_input_directory", None)
        if extra_dir_val:
            edir = Path(extra_dir_val)
            if not edir.exists():
                raise FileNotFoundError(f"Extra input directory not found: {edir}")
            epattern: str = plotter_cfg.extra_file_pattern or plotter_cfg.file_pattern
            extra_files = list(edir.glob(epattern))
            if not extra_files:
                extra_files = list(edir.rglob(epattern))
    else:
        if plotter_cfg.input_directory is None:
            raise ValueError("Input directory must be specified for directory-based plotting")
        discrete_files, continuous_files = _discover_files_from_directory(
            plotter_cfg.input_directory,
            plotter_cfg.file_pattern,
            plotter_cfg.discrete_file_pattern,
            plotter_cfg.continuous_file_pattern,
        )
        # In single directory mode, try to also detect extra files via extra_file_pattern if provided
        extra_pat_val: str | None = getattr(plotter_cfg, "extra_file_pattern", None)
        if extra_pat_val:
            input_dir = Path(plotter_cfg.input_directory)
            epattern = extra_pat_val
            extra_files = list(input_dir.glob(epattern)) or list(input_dir.rglob(epattern))

    # Log discovery results
    discovery_table: Table = Table.grid(padding=(0, 1))
    discovery_table.add_column(justify="left", style="bold blue")
    discovery_table.add_column(style="bright_white")
    discovery_table.add_row("Directory", str(plotter_cfg.input_directory))
    if plotter_cfg.discrete_input_directory:
        discovery_table.add_row("Discrete Dir", plotter_cfg.discrete_input_directory)
    if plotter_cfg.continuous_input_directory:
        discovery_table.add_row("Continuous Dir", plotter_cfg.continuous_input_directory)
    discovery_table.add_row("Pattern", plotter_cfg.file_pattern)
    discovery_table.add_row("Discrete Pattern", plotter_cfg.discrete_file_pattern)
    discovery_table.add_row("Continuous Pattern", plotter_cfg.continuous_file_pattern)
    extra_pat: str | None = getattr(plotter_cfg, "extra_file_pattern", None)
    if extra_pat:
        discovery_table.add_row("Extra Pattern", extra_pat)
    discovery_table.add_row("Discrete Files", str(len(discrete_files)))
    discovery_table.add_row("Continuous Files", str(len(continuous_files)))
    if extra_files:
        discovery_table.add_row("Extra Files", str(len(extra_files)))

    logger.info(
        Panel(
            discovery_table,
            title="[bold blue]📁 Directory Discovery Results[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
            width=120,
        )
    )

    # If an extra source is provided, create a single multi-source mean-only plot across the three
    if extra_files:
        # Choose one representative file per source: when multiple files exist, prefer the first
        representative_files: list[Path] = []
        if discrete_files:
            representative_files.append(discrete_files[0])
        if continuous_files:
            representative_files.append(continuous_files[0])
        if extra_files:
            representative_files.append(extra_files[0])

        # Build labels from config
        src_names: list[str] = list(getattr(plotter_cfg, "source_names", []) or [])
        extra_name_val: str | None = getattr(plotter_cfg, "extra_source_name", None)
        if extra_name_val and len(src_names) == 2:
            src_names += [extra_name_val]

        _create_multi_source_mean_plot_from_files(representative_files, src_names, plotter_cfg, output_dir)
        # Continue to load all variants and create per-variant plots that include the extra source
        # rather than returning early so users get per-variant 3-source comparisons.

    # Load all variants for each model type (include extra variants when present)
    discrete_variants: dict[str, dict[str, MetricInfo]] = _load_variants_from_files(discrete_files, "discrete")
    continuous_variants: dict[str, dict[str, MetricInfo]] = _load_variants_from_files(continuous_files, "continuous")
    extra_variants: dict[str, dict[str, MetricInfo]] = (
        _load_variants_from_files(extra_files, "extra") if extra_files else {}
    )

    # Get all unique variant names (excluding model type suffix)
    all_variant_names: set[str] = set()
    for variant_name in discrete_variants:
        # Remove _discrete suffix if present
        base_name: str = variant_name.replace("_discrete", "")
        all_variant_names.add(base_name)
    for variant_name in continuous_variants:
        # Remove _continuous suffix if present
        base_name = variant_name.replace("_continuous", "")
        all_variant_names.add(base_name)
    for variant_name in extra_variants:
        base_name = variant_name.replace("_extra", "")
        all_variant_names.add(base_name)

    # Configure MSEPlotter
    plot_config: PlotConfig = _create_plot_config(plotter_cfg)
    plotter = MSEPlotter(plot_config)

    # Create plots for each variant
    for base_variant in sorted(all_variant_names):
        _create_variant_plots(
            base_variant, discrete_variants, continuous_variants, extra_variants, plotter, plotter_cfg, output_dir
        )

    # Create variant comparison plots if requested
    if getattr(plotter_cfg, "create_variant_comparison_plots", False):
        if len(discrete_variants) > 1:
            _create_variant_comparison_plots(discrete_variants, "discrete", plotter, plotter_cfg, output_dir)
        if len(continuous_variants) > 1:
            _create_variant_comparison_plots(continuous_variants, "continuous", plotter, plotter_cfg, output_dir)

    # Optional: aggregated mean across variants (per model type and D vs C)
    if getattr(plotter_cfg, "create_cross_variant_mean_plots", False):
        _create_cross_variant_mean_plots(discrete_variants, continuous_variants, plotter, plotter_cfg, output_dir)


def _handle_file_based_plotting(plotter_cfg: MSEPlotterConfig, output_dir: Path) -> None:
    """Handle file-based plotting using specific file list.

    Args:
        plotter_cfg: Plotter configuration.
        output_dir: Output directory for plots.
    """
    logger.info("Using file-based plotting mode")

    # Validate input files
    if not plotter_cfg.input_files:
        raise ValueError("Input files must be specified for file-based plotting")

    input_files: list[Path] = [Path(f) for f in plotter_cfg.input_files]

    # Check that all files exist
    missing_files: list[Path] = [f for f in input_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"Input files not found: {[str(f) for f in missing_files]}")

    # If 3+ files are provided, create a single multi-source comparison using only raw mean
    # This enables plotting Discrete (with latent state), Continuous (with latent state), and
    # Continuous (predicting next state) together.
    src_names: list[str]
    try:
        src_names = list(getattr(plotter_cfg, "source_names", []) or [])
    except Exception:
        src_names = []

    if len(input_files) >= 3:
        _create_multi_source_mean_plot_from_files(input_files, src_names, plotter_cfg, output_dir)
        return

    # Split files by model type using configured patterns, with safe fallbacks
    discrete_files: list[Path] = []
    continuous_files: list[Path] = []

    for f in input_files:
        name: str = f.name
        full: str = str(f)
        is_disc: bool = fnmatch.fnmatch(name, plotter_cfg.discrete_file_pattern) or fnmatch.fnmatch(
            full, plotter_cfg.discrete_file_pattern
        )
        is_cont: bool = fnmatch.fnmatch(name, plotter_cfg.continuous_file_pattern) or fnmatch.fnmatch(
            full, plotter_cfg.continuous_file_pattern
        )
        if is_disc and not is_cont:
            discrete_files.append(f)
        elif is_cont and not is_disc:
            continuous_files.append(f)
        else:
            lower: str = full.lower()
            if "discrete" in lower and "continuous" not in lower:
                discrete_files.append(f)
            elif "continuous" in lower and "discrete" not in lower:
                continuous_files.append(f)
            else:
                # Ambiguous files default to the discrete group and emit a log message.
                logger.debug(f"Ambiguous model type for {f}, defaulting to discrete in files-mode")
                discrete_files.append(f)

    # Load variants by model type
    discrete_variants: dict[str, dict[str, MetricInfo]] = (
        _load_variants_from_files(discrete_files, "discrete") if discrete_files else {}
    )
    continuous_variants: dict[str, dict[str, MetricInfo]] = (
        _load_variants_from_files(continuous_files, "continuous") if continuous_files else {}
    )
    # File mode does not accept an extra source. Retain an empty mapping for API compatibility.
    extra_variants: dict[str, dict[str, MetricInfo]] = {}

    if not discrete_variants and not continuous_variants:
        raise ValueError("No valid variant data could be loaded from input files")

    # Compute all base variant names (without suffixes)
    all_variant_names: set[str] = set()
    all_variant_names.update(name.replace("_discrete", "") for name in discrete_variants)
    all_variant_names.update(name.replace("_continuous", "") for name in continuous_variants)

    # Configure plotter
    plot_config: PlotConfig = _create_plot_config(plotter_cfg)
    plotter = MSEPlotter(plot_config)

    # Create plots per base variant
    for base_variant in sorted(all_variant_names):
        _create_variant_plots(
            base_variant, discrete_variants, continuous_variants, extra_variants, plotter, plotter_cfg, output_dir
        )

    # Optional: create within-model-type variant comparison plots
    if getattr(plotter_cfg, "create_variant_comparison_plots", False):
        if len(discrete_variants) > 1:
            _create_variant_comparison_plots(discrete_variants, "discrete", plotter, plotter_cfg, output_dir)
        if len(continuous_variants) > 1:
            _create_variant_comparison_plots(continuous_variants, "continuous", plotter, plotter_cfg, output_dir)

    if getattr(plotter_cfg, "create_cross_variant_mean_plots", False):
        _create_cross_variant_mean_plots(discrete_variants, continuous_variants, plotter, plotter_cfg, output_dir)


def _extract_mean_series_array(metric_info: MetricInfo) -> NDArray[np.float32] | None:
    """Extract the mean series as a (1, T) float32 array, padded if needed.

    Returns None if the series is missing or empty.
    """
    try:
        values: list[float] | None = _extract_episode_values(metric_info, "mean")
        if not isinstance(values, list) or len(values) == 0:
            return None
        arr: NDArray[np.float32] = np.asarray(values, dtype=np.float32)[None, :]
    except Exception:
        return None
    else:
        return arr


def _load_label_from_metrics_file(metrics_file: Path) -> str:
    """Load a humanized source label from a metrics file."""
    try:
        variant_name, _ = _load_single_variant_metrics(metrics_file)
    except Exception:
        return _title_from_identifier(metrics_file.stem)
    return _title_from_identifier(variant_name)


def _load_mean_series_for_source(metrics_file: Path, metric_key: str) -> NDArray[np.float32] | None:
    _variant_name, metrics = _load_single_variant_metrics(metrics_file)
    info: MetricInfo | None = metrics.get(metric_key)
    if not isinstance(info, dict):
        logger.warning(f"Metric '{metric_key}' missing in {metrics_file}")
        return None
    arr: NDArray[np.float32] | None = _extract_mean_series_array(info)
    if arr is None:
        logger.warning(f"Empty mean series for metric '{metric_key}' in {metrics_file}")
        return None
    return arr


def _create_multi_source_mean_plot_from_files(
    input_files: list[Path], source_names: list[str] | None, plotter_cfg: MSEPlotterConfig, output_dir: Path
) -> None:
    """Create a single comparison plot with 3+ sources, plotting only raw mean for each.

    Labels are taken from `source_names` when provided (and aligned with files),
    otherwise from the internal variant names or file stems.
    """
    # Configure plotter once
    plot_config: PlotConfig = _create_plot_config(plotter_cfg)
    plotter = MSEPlotter(plot_config)

    # Build labels for sources
    labels: list[str] = []
    if source_names and len(source_names) == len(input_files):
        labels = list(source_names)
    else:
        # Fallback: derive from variant_name if present, else file stem
        labels = [_load_label_from_metrics_file(f) for f in input_files]

    # Process each requested metric separately
    for metric_key in plotter_cfg.metric_keys_to_plot:
        series_map: dict[str, NDArray[np.float32]] = {}

        for idx, f in enumerate(input_files):
            try:
                arr = _load_mean_series_for_source(f, metric_key)
            except Exception:
                logger.exception(f"Failed loading {f}")
                continue
            if arr is None:
                continue
            series_map[labels[idx]] = arr

        if len(series_map) < 2:
            logger.warning(
                f"Insufficient sources with valid mean data for metric '{metric_key}' (have {len(series_map)})"
            )
            continue

        # Determine title
        comparison_title: str | None = plotter_cfg.comparison_title
        title: str = comparison_title or "Sources Comparison"

        # Plot only raw mean lines, no smoothing, no min/max, no sample size text
        try:
            plotter.plot_mse(
                series_map,
                title=title,
                xlabel=plotter_cfg.x_axis.label or "Step",
                ylabel=plotter_cfg.y_axis.label or "Reconstruction MSE",
                save_path=output_dir / f"multi_source_{metric_key}_raw_mean",
                show_minmax=False,
                show_smoothed_line=False,
                show_raw_mean=True,
                show_legend=getattr(plotter_cfg.legend, "show_legend", True),
                sample_size_text="",
                annotation_lines=None,
            )
            logger.info(f"Created multi-source raw-mean comparison for metric '{metric_key}'")
        except Exception:
            logger.exception(f"Failed to create multi-source plot for metric '{metric_key}'")


def _load_variants_from_files(files: list[Path], model_type: str) -> dict[str, dict[str, MetricInfo]]:
    """Load variant data from a list of files.

    Args:
        files: List of file paths to load.
        model_type: Model type ('discrete' or 'continuous').

    Returns:
        Dictionary mapping variant names to metrics data.
    """
    variants: dict[str, dict[str, MetricInfo]] = {}

    for file_path in files:
        loaded: tuple[str, dict[str, MetricInfo]] | None = _load_variant_for_model(file_path, model_type)
        if loaded is None:
            continue
        variant_name, metrics_data = loaded
        variant_with_type = f"{variant_name}_{model_type}"
        variants[variant_with_type] = metrics_data
        logger.debug(f"Loaded {model_type} variant: {variant_name}")

    return variants


def _load_variant_for_model(file_path: Path, model_type: str) -> tuple[str, dict[str, MetricInfo]] | None:
    """Load one variant metrics file for a specific model type."""
    try:
        return _load_single_variant_metrics(file_path)
    except (FileNotFoundError, ValueError):
        logger.exception(f"Failed to load {model_type} file {file_path}")
        return None


def _create_variant_plots(
    base_variant: str,
    discrete_variants: dict[str, dict[str, MetricInfo]],
    continuous_variants: dict[str, dict[str, MetricInfo]],
    extra_variants: dict[str, dict[str, MetricInfo]] | None,
    plotter: MSEPlotter,
    plotter_cfg: MSEPlotterConfig,
    output_dir: Path,
) -> None:
    """Create plots for a specific variant across model types.

    Args:
        base_variant: Base variant name (without model type suffix).
        discrete_variants: Discrete model variant data.
        continuous_variants: Continuous model variant data.
        extra_variants: Optional mapping of extra-source variants (e.g., next-state continuous) keyed by
            '<variant>_extra' to their metrics dict. May be an empty dict when not provided.
        plotter: Configured MSEPlotter instance.
        plotter_cfg: Plotter configuration.
        output_dir: Output directory.
    """
    # Create variant-specific output directory
    variant_dir: Path = output_dir / base_variant
    variant_dir.mkdir(exist_ok=True)

    # Look for discrete and continuous data for this variant
    discrete_key: str = f"{base_variant}_discrete"
    continuous_key: str = f"{base_variant}_continuous"
    extra_key: str = f"{base_variant}_extra"

    has_discrete: bool = discrete_key in discrete_variants
    has_continuous: bool = continuous_key in continuous_variants

    if not has_discrete and not has_continuous and not (extra_variants and extra_key in extra_variants):
        logger.warning(f"No data found for variant '{base_variant}'")
        return

    # Process each metric
    for metric_key in plotter_cfg.metric_keys_to_plot:
        metric_dir: Path = variant_dir / metric_key
        metric_dir.mkdir(exist_ok=True)

        # Create individual model type plots if requested
        if getattr(plotter_cfg, "create_individual_model_plots", True):
            if has_discrete:
                _create_model_type_plot(
                    base_variant,
                    discrete_variants[discrete_key],
                    "discrete",
                    metric_key,
                    plotter,
                    plotter_cfg,
                    metric_dir,
                )

            if has_continuous:
                _create_model_type_plot(
                    base_variant,
                    continuous_variants[continuous_key],
                    "continuous",
                    metric_key,
                    plotter,
                    plotter_cfg,
                    metric_dir,
                )

        # Create comparison plot(s): handle 2-source (D vs C) and optional 3-source when extra provided
        if getattr(plotter_cfg, "create_model_comparison_plots", True):
            # 3-source case: discrete + continuous + extra
            has_extra: dict[str, dict[str, MetricInfo]] | bool | None = extra_variants and extra_key in extra_variants
            if has_discrete and has_continuous and has_extra:
                extra_data: dict[str, MetricInfo] | None = (
                    extra_variants[extra_key] if extra_variants is not None else None
                )
                _create_model_comparison_plot(
                    base_variant,
                    discrete_variants[discrete_key],
                    continuous_variants[continuous_key],
                    metric_key,
                    plotter,
                    plotter_cfg,
                    metric_dir,
                    extra_data,
                )
            # Fallback to classic 2-source comparison
            elif has_discrete and has_continuous:
                _create_model_comparison_plot(
                    base_variant,
                    discrete_variants[discrete_key],
                    continuous_variants[continuous_key],
                    metric_key,
                    plotter,
                    plotter_cfg,
                    metric_dir,
                )


def _create_model_type_plot(
    variant_name: str,
    metrics_data: dict[str, MetricInfo],
    model_type: str,
    metric_key: str,
    plotter: MSEPlotter,
    plotter_cfg: MSEPlotterConfig,
    output_dir: Path,
) -> None:
    """Create plots for a single model type including individual episode analysis.

    Args:
        variant_name: Variant name.
        metrics_data: Metrics data for this variant.
        model_type: Model type ('discrete' or 'continuous').
        metric_key: Metric to plot.
        plotter: Configured MSEPlotter instance.
        plotter_cfg: Plotter configuration.
        output_dir: Output directory.
    """
    if metric_key not in metrics_data:
        logger.warning(f"Metric '{metric_key}' not found for {model_type} {variant_name}")
        return

    metric_info: MetricInfo = metrics_data[metric_key]

    # Create individual episode plots if requested
    if getattr(plotter_cfg, "create_individual_plots", True):
        _create_individual_episode_plots(
            variant_name, model_type, metric_key, metric_info, plotter, plotter_cfg, output_dir
        )

    # Create summary plot with mean trajectory
    _create_summary_plot(variant_name, model_type, metric_key, metric_info, plotter, plotter_cfg, output_dir)


def _create_individual_episode_plots(
    variant_name: str,
    model_type: str,
    metric_key: str,
    metric_info: MetricInfo,
    plotter: MSEPlotter,
    plotter_cfg: MSEPlotterConfig,
    output_dir: Path,
) -> None:
    """Create individual plots for mean, best, and worst episodes.

    Args:
        variant_name: Variant name.
        model_type: Model type ('discrete' or 'continuous').
        metric_key: Metric to plot.
        metric_info: Metric information from JSON.
        plotter: Configured MSEPlotter instance.
        plotter_cfg: Plotter configuration.
        output_dir: Output directory.
    """
    episodes: tuple[EpisodeKey, ...] = ("mean", "min", "max")
    # Use Min/Max naming per request
    episode_labels: dict[str, str] = {"mean": "Mean", "min": "Min", "max": "Max"}

    for episode_type in episodes:
        episode_values: list[float] | None = _extract_episode_values(metric_info, episode_type)
        if episode_values is None:
            logger.warning(f"Episode data '{episode_type}' not found for {variant_name} {model_type}")
            continue

        if not episode_values:
            logger.warning(f"Empty episode data for {episode_type} in {variant_name} {model_type}")
            continue

        # Convert to numpy array (single episode, so shape (1, T))
        episode_array: NDArray[np.float32] = np.array([episode_values], dtype=np.float32)

        # Create plot title (format template if provided)
        human_variant: str = _variant_display_name(variant_name, plotter_cfg)
        model_label: str = model_type.title()
        metric_label: str = _title_from_identifier(metric_key)
        episode_label: str = episode_labels[episode_type]
        if plotter_cfg.title_template:
            try:
                title: str = plotter_cfg.title_template.format(
                    variant=human_variant,
                    model=model_label,
                    model_type=model_label,
                    metric=metric_label,
                    episode=episode_label,
                )
            except Exception:
                title = f"{human_variant} - {model_label} - {episode_label}"
        else:
            title = f"{human_variant} - {model_label} - {episode_label}"

        # Create the plot
        try:
            plotter.plot_mse(
                episode_array,
                title=title,
                xlabel=plotter_cfg.x_axis.label or "Step",
                ylabel=plotter_cfg.y_axis.label or "Reconstruction MSE",
                save_path=output_dir / f"{variant_name}_{model_type}_{episode_type}_{metric_key}",
                show_minmax=False,  # Single episode: min/max envelope is not meaningful (N=1)
                show_smoothed_line=getattr(plotter_cfg, "show_smoothed", True),
                show_raw_mean=plotter_cfg.show_raw_mean,
                show_legend=getattr(plotter_cfg.legend, "show_legend", True),
                primary_label=f"smoothed {episode_label.lower()}",
                secondary_label=episode_label.lower(),
                sample_size_text="",  # Disable sample size display
                annotation_lines=_get_annotation_lines(plotter_cfg),
            )
            logger.debug(f"Created {episode_type} plot for {variant_name} {model_type}")
        except Exception:
            logger.exception(f"Failed to create {episode_type} plot for {variant_name} {model_type}")


def _create_summary_plot(
    variant_name: str,
    model_type: str,
    metric_key: str,
    metric_info: MetricInfo,
    plotter: MSEPlotter,
    plotter_cfg: MSEPlotterConfig,
    output_dir: Path,
) -> None:
    """Create a summary plot showing all episode types together.

    Args:
        variant_name: Variant name.
        model_type: Model type ('discrete' or 'continuous').
        metric_key: Metric to plot.
        metric_info: Metric information from JSON.
        plotter: Configured MSEPlotter instance.
        plotter_cfg: Plotter configuration.
        output_dir: Output directory.
    """
    # Build a single-source array stacking available episodes so bands (e.g., 80% quantile) can show
    stacked: list[NDArray[np.float32]] = []
    episode_keys: tuple[EpisodeKey, ...] = ("mean", "min", "max")
    for ep in episode_keys:
        vals: list[float] | None = _extract_episode_values(metric_info, ep)
        if vals:
            stacked.append(np.asarray(vals, dtype=np.float32))

    if len(stacked) == 0:
        logger.debug(f"No episode data for summary plot of {variant_name} {model_type}")
        return

    # Pad shorter series with NaN to match the longest series.
    max_len: int = max(arr.shape[0] for arr in stacked)
    padded: list[NDArray[np.float32]] = [
        np.pad(arr, (0, max_len - arr.shape[0]), constant_values=np.nan) for arr in stacked
    ]
    stacked_arr: NDArray[np.float32] = np.vstack(padded)  # Shape (N, T)

    # Title formatting
    human_variant: str = _variant_display_name(variant_name, plotter_cfg)
    model_label: str = model_type.title()
    metric_label: str = _title_from_identifier(metric_key)
    if plotter_cfg.title_template:
        try:
            title: str = plotter_cfg.title_template.format(
                variant=human_variant, model=model_label, model_type=model_label, metric=metric_label, episode="Summary"
            )
        except Exception:
            title = f"{human_variant} - {model_label} - Summary"
    else:
        title = f"{human_variant} - {model_label} - Summary"

    try:
        # Show min/max lines and quantile band (e.g., 80% if configured as 0.1-0.9)
        plotter.plot_mse(
            stacked_arr,
            title=title,
            xlabel=plotter_cfg.x_axis.label or "Step",
            ylabel=plotter_cfg.y_axis.label or "Reconstruction MSE",
            save_path=output_dir / f"{variant_name}_{model_type}_summary_{metric_key}",
            show_minmax=plotter_cfg.show_minmax,
            show_smoothed_line=getattr(plotter_cfg, "show_smoothed", True),
            show_raw_mean=plotter_cfg.show_raw_mean,
            show_legend=getattr(plotter_cfg.legend, "show_legend", True),
            primary_label="smoothed mean",
            secondary_label="mean",
            sample_size_text="",  # Disable sample size display
            annotation_lines=_get_annotation_lines(plotter_cfg),
        )
        logger.debug(f"Created summary plot for {variant_name} {model_type}")
    except Exception:
        logger.exception(f"Failed to create summary plot for {variant_name} {model_type}")


def _create_model_comparison_plot(
    variant_name: str,
    discrete_metrics: dict[str, MetricInfo],
    continuous_metrics: dict[str, MetricInfo],
    metric_key: str,
    plotter: MSEPlotter,
    plotter_cfg: MSEPlotterConfig,
    output_dir: Path,
    extra_metrics: dict[str, MetricInfo] | None = None,
) -> None:
    """Create a comparison plot between discrete and continuous models.

    Args:
        variant_name: Variant name.
        discrete_metrics: Discrete model metrics data.
        continuous_metrics: Continuous model metrics data.
        metric_key: Metric to plot.
        plotter: Configured MSEPlotter instance.
        plotter_cfg: Plotter configuration.
        output_dir: Output directory.
        extra_metrics: Optional metrics dict for a third source (e.g., continuous next-state) to include in a
            three-source comparison. Pass None when unavailable.
    """
    # Validate existence for required metric(s)
    if metric_key not in discrete_metrics or metric_key not in continuous_metrics:
        logger.warning(f"Metric '{metric_key}' not available for both model types in {variant_name}")
        return

    # Prepare data for two or three sources, using friendly labels from config when available
    source_names: list[str] = list(getattr(plotter_cfg, "source_names", []) or [])
    extra_name: str | None = getattr(plotter_cfg, "extra_source_name", None)

    combined_data: dict[str, dict[str, MetricInfo]] = {}
    # Label order: Discrete, Continuous, Extra
    d_label: str = source_names[0] if len(source_names) >= 1 else "Discrete"
    c_label: str = source_names[1] if len(source_names) >= 2 else "Continuous"
    combined_data[d_label] = {metric_key: discrete_metrics[metric_key]}
    combined_data[c_label] = {metric_key: continuous_metrics[metric_key]}

    if extra_metrics is not None and metric_key in extra_metrics:
        e_label: str = source_names[2] if len(source_names) >= 3 else (extra_name or "Extra")
        combined_data[e_label] = {metric_key: extra_metrics[metric_key]}

    metric_arrays: dict[str, NDArray[np.float32]] = _convert_to_metric_arrays(combined_data, metric_key)

    if len(metric_arrays) < 2:
        logger.warning(f"Insufficient data for comparison plot of {variant_name} - {metric_key}")
        return

    human_variant: str = _variant_display_name(variant_name, plotter_cfg)
    # Prefer the configured title and otherwise omit the metric name from the default.
    if plotter_cfg.comparison_title:
        try:
            title: str = plotter_cfg.comparison_title.format(variant=human_variant)
        except Exception:
            title = f"{human_variant} - Discrete vs Continuous"
    else:
        title = f"{human_variant} - Discrete vs Continuous"

    # Sample-size annotation disabled

    # Create comparison plot
    plotter.plot_mse(
        metric_arrays,
        title=title,
        xlabel=plotter_cfg.x_axis.label or "Step",
        ylabel=plotter_cfg.y_axis.label or "Reconstruction MSE",
        save_path=output_dir / f"{variant_name}_model_comparison_{metric_key}",
        show_minmax=plotter_cfg.show_minmax,
        show_smoothed_line=getattr(plotter_cfg, "show_smoothed", True),
        show_raw_mean=plotter_cfg.show_raw_mean,
        show_legend=getattr(plotter_cfg.legend, "show_legend", True),
        sample_size_text="",  # Disable sample size display
        annotation_lines=_get_annotation_lines(plotter_cfg),
    )


def _create_variant_comparison_plots(
    variants: dict[str, dict[str, MetricInfo]],
    model_type: str,
    plotter: MSEPlotter,
    plotter_cfg: MSEPlotterConfig,
    output_dir: Path,
) -> None:
    """Create comparison plots between multiple variants of the same model type.

    Args:
        variants: Dictionary of variant data for the same model type.
        model_type: Model type ('discrete' or 'continuous').
        plotter: Configured MSEPlotter instance.
        plotter_cfg: Plotter configuration.
        output_dir: Output directory.
    """
    if len(variants) < 2:
        logger.info(f"Skipping variant comparison for {model_type} - only {len(variants)} variant(s) available")
        return

    # Create model-type-specific output directory
    model_type_dir: Path = output_dir / f"{model_type}_variant_comparisons"
    model_type_dir.mkdir(exist_ok=True)

    # Process each metric
    for metric_key in plotter_cfg.metric_keys_to_plot:
        # Prepare data for all variants
        variant_metric_arrays: dict[str, NDArray[np.float32]] = {}

        for variant_name, metrics_data in variants.items():
            if metric_key in metrics_data:
                # Human-friendly variant label without model-type suffix (with overrides)
                base: str = variant_name.replace("_discrete", "").replace("_continuous", "")
                friendly: str = _variant_display_name(base, plotter_cfg)
                variant_data: dict[str, dict[str, MetricInfo]] = {friendly: {metric_key: metrics_data[metric_key]}}
                metric_arrays: dict[str, NDArray[np.float32]] = _convert_to_metric_arrays(variant_data, metric_key)
                if metric_arrays:
                    variant_metric_arrays.update(metric_arrays)

        if len(variant_metric_arrays) < 2:
            logger.warning(f"Insufficient data for {model_type} variant comparison - {metric_key}")
            continue

        # Title without metric name to avoid redundancy
        title: str = f"{model_type.title()} Variants Comparison"

        # Create comparison plot
        plotter.plot_mse(
            variant_metric_arrays,
            title=title,
            xlabel=plotter_cfg.x_axis.label or "Step",
            ylabel=plotter_cfg.y_axis.label or "Reconstruction MSE",
            save_path=model_type_dir / f"{model_type}_variants_{metric_key}_comparison",
            show_minmax=plotter_cfg.show_minmax,
            show_smoothed_line=getattr(plotter_cfg, "show_smoothed", True),
            show_raw_mean=plotter_cfg.show_raw_mean,
            show_legend=getattr(plotter_cfg.legend, "show_legend", True),
            sample_size_text="",  # Disable sample size display
            annotation_lines=_get_annotation_lines(plotter_cfg),
        )

        logger.info(f"Created {model_type} variant comparison plot for {metric_key}")


def _build_label_color_map(
    plotter_cfg: MSEPlotterConfig, variant_colors: tuple[str, ...] | None
) -> dict[str, str] | None:
    names_for_colors: list[str] = []
    src_names: list[str] | None = getattr(plotter_cfg, "source_names", None)
    if src_names:
        names_for_colors.extend(list(src_names))
    extra_name: str | None = getattr(plotter_cfg, "extra_source_name", None)
    if extra_name:
        names_for_colors.append(extra_name)
    if variant_colors and names_for_colors and len(variant_colors) >= len(names_for_colors):
        return {name: variant_colors[i] for i, name in enumerate(names_for_colors)}
    return None


def _create_plot_config(plotter_cfg: MSEPlotterConfig) -> PlotConfig:
    """Create a PlotConfig from MSEPlotterConfig.

    Args:
        plotter_cfg: MSE plotter configuration.

    Returns:
        PlotConfig instance.
    """
    # Validate and convert style
    try:
        style: StyleName = StyleName(plotter_cfg.style)
    except ValueError:
        logger.warning(f"Invalid style '{plotter_cfg.style}', using 'nature_journal'")
        style = StyleName.NATURE_JOURNAL

    # Validate and convert smoothing method
    try:
        # Honor show_smoothed=False by switching to NONE (raw-only)
        smoothing_method: SmoothingMethod = (
            SmoothingMethod.NONE
            if not getattr(plotter_cfg, "show_smoothed", True)
            else SmoothingMethod(plotter_cfg.smoothing_method)
        )
    except ValueError:
        logger.warning(f"Invalid smoothing method '{plotter_cfg.smoothing_method}', using 'exponential'")
        smoothing_method = SmoothingMethod.EXPONENTIAL

    # Validate and convert uncertainty method
    try:
        # Honor show_uncertainty=False by disabling shaded bands (use MINMAX placeholder)
        uncertainty_method: UncertaintyMethod = (
            UncertaintyMethod.MINMAX
            if not getattr(plotter_cfg, "show_uncertainty", True)
            else UncertaintyMethod(plotter_cfg.uncertainty_method)
        )
    except ValueError:
        logger.warning(f"Invalid uncertainty method '{plotter_cfg.uncertainty_method}', using 'quantiles'")
        uncertainty_method = UncertaintyMethod.QUANTILES

    # Color overrides from config (variant order and optional label mapping)
    variant_colors: tuple[str, ...] | None = None
    label_color_map: dict[str, str] | None = None
    try:
        colors_cfg: ColorSchemeConfig | None = getattr(plotter_cfg, "colors", None)
        if colors_cfg and getattr(colors_cfg, "variant_colors", None):
            variant_colors = tuple(colors_cfg.variant_colors)
    except Exception:
        variant_colors = None

    try:
        label_color_map = _build_label_color_map(plotter_cfg, variant_colors)
    except Exception:
        label_color_map = None

    return PlotConfig(
        style=style,
        use_log_scale=(plotter_cfg.y_axis.scale == "log"),
        y_use_scientific_notation=getattr(plotter_cfg.y_axis, "use_scientific_notation", True),
        y_decimal_places=getattr(plotter_cfg.y_axis, "decimal_places", 3),
        smoothing_method=smoothing_method,
        smoothing_window_ratio=plotter_cfg.smoothing_window_ratio,
        uncertainty_method=uncertainty_method,
        quantiles=(plotter_cfg.quantiles[0], plotter_cfg.quantiles[1]),
        figsize=(plotter_cfg.figsize[0], plotter_cfg.figsize[1]),
        dpi=plotter_cfg.dpi,
        max_points_before_decimation=plotter_cfg.max_points_before_decimation,
        streaming_threshold=plotter_cfg.streaming_threshold,
        export_formats=tuple(plotter_cfg.export_formats),
        png_transparent=plotter_cfg.png_transparent,
        legend_show_raw_mean=getattr(plotter_cfg, "legend_show_raw_mean", None),
        legend_show_minmax=getattr(plotter_cfg, "legend_show_minmax", None),
        legend_show_band=getattr(plotter_cfg, "legend_show_band", None),
        legend_smoothed_label_template=getattr(
            plotter_cfg, "legend_smoothed_label_template", "{source} (smoothed mean)"
        ),
        legend_raw_label_template=getattr(plotter_cfg, "legend_raw_label_template", "{source} (mean)"),
        legend_minmax_label=getattr(plotter_cfg, "legend_minmax_label", "min / max"),
        legend_band_label_quantiles_template=getattr(
            plotter_cfg, "legend_band_label_quantiles_template", "{percent}% range"
        ),
        legend_band_label_std=getattr(plotter_cfg, "legend_band_label_std", "+/- std range"),
        variant_colors=variant_colors,
        label_color_map=label_color_map,
    )


def _get_annotation_lines(plotter_cfg: MSEPlotterConfig) -> list[tuple[int, str]] | None:
    """Extract vertical annotation lines from config as (x, label) tuples.

    Only vertical lines are supported by the underlying plotter API.
    """
    try:
        lines = _collect_annotation_lines(plotter_cfg)
    except Exception:
        return None
    return lines or None


def _collect_annotation_lines(plotter_cfg: MSEPlotterConfig) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for item in plotter_cfg.annotations.vertical_lines:
        x_val = item.get("x")
        x: int | None = None
        if isinstance(x_val, (int, float)) or (isinstance(x_val, str) and x_val.isdigit()):
            x = int(x_val)
        label = str(item.get("label", f"x={x_val}"))
        if x is not None:
            lines.append((x, label))
    return lines


def _aggregate_mean(variants: dict[str, dict[str, MetricInfo]], metric_key: str) -> NDArray[np.float32] | None:
    series: list[NDArray[np.float32]] = []
    for metrics in variants.values():
        info: MetricInfo | None = metrics.get(metric_key)
        if not isinstance(info, dict):
            continue
        if not info:
            continue
        vals: list[float] = info.get("mean", {}).get("values_per_step", [])
        if vals:
            series.append(np.asarray(vals, dtype=np.float32))
    if not series:
        return None
    max_len: int = max(s.shape[0] for s in series)
    padded: list[NDArray[np.float32]] = [np.pad(s, (0, max_len - s.shape[0]), constant_values=np.nan) for s in series]
    stacked: NDArray[np.float32] = np.vstack(padded)
    avg: NDArray[np.float32] = np.nanmean(stacked, axis=0).astype(np.float32)
    return avg[None, :]


def _create_cross_variant_mean_plots(
    discrete_variants: dict[str, dict[str, MetricInfo]] | None,
    continuous_variants: dict[str, dict[str, MetricInfo]] | None,
    plotter: MSEPlotter,
    plotter_cfg: MSEPlotterConfig,
    output_dir: Path,
) -> None:
    """Create plots of the mean of means across all variants for each model type, and D vs C.

    For each metric, compute the per-step average of the mean series from each variant.
    """
    output_dir /= "cross_variant_means"
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_key in plotter_cfg.metric_keys_to_plot:
        d: NDArray[np.float32] | None = (
            _aggregate_mean(discrete_variants or {}, metric_key) if discrete_variants else None
        )
        c: NDArray[np.float32] | None = (
            _aggregate_mean(continuous_variants or {}, metric_key) if continuous_variants else None
        )

        # Sample-size summaries disabled

        if d is not None:
            plotter.plot_mse(
                d,
                title="Discrete - Mean Across Variants",
                xlabel=plotter_cfg.x_axis.label or "Step",
                ylabel=plotter_cfg.y_axis.label or "Reconstruction MSE",
                save_path=output_dir / f"discrete_{metric_key}_mean_across_variants",
                show_minmax=plotter_cfg.show_minmax,
                show_smoothed_line=getattr(plotter_cfg, "show_smoothed", True),
                show_raw_mean=plotter_cfg.show_raw_mean,
                show_legend=getattr(plotter_cfg.legend, "show_legend", True),
                # Disable sample size display
                sample_size_text="",
                annotation_lines=_get_annotation_lines(plotter_cfg),
            )
        if c is not None:
            plotter.plot_mse(
                c,
                title="Continuous - Mean Across Variants",
                xlabel=plotter_cfg.x_axis.label or "Step",
                ylabel=plotter_cfg.y_axis.label or "Reconstruction MSE",
                save_path=output_dir / f"continuous_{metric_key}_mean_across_variants",
                show_minmax=plotter_cfg.show_minmax,
                show_smoothed_line=getattr(plotter_cfg, "show_smoothed", True),
                show_raw_mean=plotter_cfg.show_raw_mean,
                show_legend=getattr(plotter_cfg.legend, "show_legend", True),
                sample_size_text="",  # Disable sample size display
                annotation_lines=_get_annotation_lines(plotter_cfg),
            )

        if d is not None and c is not None:
            plotter.plot_mse(
                {"Discrete": d, "Continuous": c},
                title="Discrete vs Continuous - Mean Across Variants",
                xlabel=plotter_cfg.x_axis.label or "Step",
                ylabel=plotter_cfg.y_axis.label or "Reconstruction MSE",
                save_path=output_dir / f"disc_vs_cont_{metric_key}_mean_across_variants",
                show_minmax=plotter_cfg.show_minmax,
                show_smoothed_line=getattr(plotter_cfg, "show_smoothed", True),
                show_raw_mean=plotter_cfg.show_raw_mean,
                show_legend=getattr(plotter_cfg.legend, "show_legend", True),
                sample_size_text="",  # Disable sample size display
                annotation_lines=_get_annotation_lines(plotter_cfg),
            )
