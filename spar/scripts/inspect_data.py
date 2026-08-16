"""Inspect, compare, and visualize SPAR data files.

Example usage:
    python inspect_data.py /path/to/your/data_file.h5

Or with visualization:
    python inspect_data.py /path/to/your/data_file.h5 --visualize

Save visualization to file:
    python inspect_data.py /path/to/your/data_file.h5 --visualize --save plot.pdf

Animation and timeline view:
    python inspect_data.py /path/to/your/data_file.h5 --timeline

Compare multiple datasets:
    python inspect_data.py /path/to/data1.h5 /path/to/data2.h5 --compare

Export summary as JSON:
    python inspect_data.py /path/to/data_file.h5 --export summary.json

The module can also be imported and called from another Python process.
"""

from __future__ import annotations

from argparse import SUPPRESS, ArgumentParser
from collections.abc import Mapping
import contextlib
import io
import os
import pathlib
import sys
import traceback
from typing import TYPE_CHECKING

from spar.utils.data_utils.data_inspector import (
    InspectorOptions,
    InspectorTextConfig,
    MatplotlibLayoutConfig,
    compare_datasets,
    create_timeline_animation,
    examine_variations,
    export_summary,
    get_data_specs,
    load_data_file,
    print_data_report,
    print_tabular_overview,
    set_inspector_options,
    visualize_data_structure,
    visualize_sample,
)

if TYPE_CHECKING:
    from argparse import Action, Namespace, _ArgumentGroup as ArgumentGroup
    from pathlib import Path

    from spar.utils.data_utils.data_inspector import ActionTraj, Specs, StateImgTraj


# Add the project root to the path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def _run_inspector_steps(
    file_path: str,
    visualize: bool,
    show_variations: bool,
    show_structure: bool,
    show_table: bool,
    *,
    sample_idx: int,
    max_frames: int,
    detailed: bool,
    depth: int,
    save_path: str | None,
) -> None:
    print(f"Analyzing SPAR data file: {file_path}")

    # Load data and get specs
    state_img_trajs, action_trajs = load_data_file(file_path)
    specs: Specs = get_data_specs(state_img_trajs, action_trajs)
    print_data_report(file_path, specs)

    # Run requested visualizations
    if show_structure:
        visualize_data_structure(state_img_trajs, action_trajs, depth)
    if show_table:
        print_tabular_overview(state_img_trajs, action_trajs, specs)
    if show_variations:
        examine_variations(state_img_trajs, sample_idx, detailed, save_path=save_path)
    if visualize:
        print("\nVisualizing sample from the dataset...")
        visualize_sample(state_img_trajs, action_trajs, sample_idx, max_frames, save_path=save_path)


def run_inspector(
    file_path: str,
    visualize: bool = False,
    show_variations: bool = False,
    show_structure: bool = False,
    show_table: bool = False,
    *,
    sample_idx: int = 0,
    max_frames: int = 5,
    detailed: bool = False,
    depth: int = 5,
    save_path: str | None = None,
) -> int:
    """Run the data inspection with various visualization options."""
    try:
        _run_inspector_steps(
            file_path,
            visualize,
            show_variations,
            show_structure,
            show_table,
            sample_idx=sample_idx,
            max_frames=max_frames,
            detailed=detailed,
            depth=depth,
            save_path=save_path,
        )

    except Exception as e:
        print(f"Error analyzing data file: {e}")
        traceback.print_exc()

        return 1
    else:
        return 0


def find_data_file(file_pattern: str) -> str:
    """Find a data file based on a pattern or partial path.

    Args:
        file_pattern: A file pattern that might include the "#file:" prefix.

    Returns:
        Full path to the data file if found, otherwise returns the original pattern.
    """
    if not file_pattern.startswith("#file:"):
        return file_pattern

    file_name: str = file_pattern[6:].strip()

    # Define common data directories
    cwd: Path = pathlib.Path.cwd()
    data_dirs: list[str] = [str(cwd), str(cwd / "data"), str(cwd / "spar" / "data" / "offline_data")]

    if spar_env := os.environ.get("SPAR_DATA_DIR"):
        data_dirs.append(spar_env)

    # Try each combination of directory and extension
    for d in filter(os.path.exists, data_dirs):
        for ext in ("", ".h5", ".hdf5"):
            path = pathlib.Path(d) / f"{file_name}{ext}"
            if path.exists():
                print(f"Found data file: {path}")
                return str(path)

    print(f"Warning: Could not find data file matching pattern: {file_name}")
    return file_pattern


def search_data_files(path_pattern: str) -> list[str]:
    """Search for data files matching a pattern.

    Args:
        path_pattern: File path pattern, can include wildcards

    Returns:
        List of matching file paths
    """
    # Handle the #file: syntax
    if path_pattern.startswith("#file:"):
        found_file: str = find_data_file(path_pattern)
        return [found_file] if found_file else []

    # Direct file check (fast path)
    if pathlib.Path(path_pattern).is_file():
        return [path_pattern]

    # Handle directory search for .h5/.hdf5 files
    if pathlib.Path(path_pattern).is_dir():
        return [str(p) for p in pathlib.Path(path_pattern).rglob("*.h5*")]

    # Try wildcard pattern matching
    pattern = pathlib.Path(path_pattern)
    if pattern.is_absolute():
        return [str(p) for p in pattern.parent.glob(pattern.name)]
    return [str(p) for p in pathlib.Path().glob(path_pattern)]


def build_parser() -> ArgumentParser:
    """Construct hierarchical CLI parser (no auto-help to allow contextual help)."""
    parser = ArgumentParser(prog="inspect_data", description="SPAR Data Inspector CLI", add_help=False)

    # Global/common options group
    parser.add_argument("file_paths", nargs="*", help="Path(s) to data file(s) or directories / glob patterns")
    parser.add_argument("--export", "-e", metavar="FILE", help="Export summary JSON to FILE")
    parser.add_argument("--save", metavar="PATH", help="Save figure/animation instead of showing GUI")

    # Text customization
    txt: ArgumentGroup = parser.add_argument_group("text", "Text customization & disabling")
    txt.add_argument("--text-override", nargs=2, action="append", metavar=("KEY", "VALUE"), help="Override a text key")
    txt.add_argument("--disable-text-key", nargs=1, action="append", metavar="KEY", help="Disable specific text key")
    txt.add_argument("--disable-all-text", action="store_true", help="Disable all textual labels/titles")

    # Layout config
    lay: ArgumentGroup = parser.add_argument_group("layout", "Matplotlib layout & style")
    lay.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    lay.add_argument("--theme", choices=["default", "dark", "light"], default="default", help="Color theme")
    lay.add_argument("--no-constrained", action="store_true", help="Disable constrained layout engine")
    lay.add_argument("--tight", action="store_true", help="Apply tight_layout after adjustments")
    lay.add_argument("--grid", action="store_true", help="Enable background grid on applicable axes")
    lay.add_argument("--wspace", type=float, default=0.15, help="Horizontal subplot spacing")
    lay.add_argument("--hspace", type=float, default=0.20, help="Vertical subplot spacing")
    lay.add_argument("--padding", type=float, default=0.05, help="Figure edge padding fraction")

    perf: ArgumentGroup = parser.add_argument_group("performance", "Performance & computation toggles")
    perf.add_argument("--no-perf-mode", action="store_true", help="Disable perf mode (compute full stats)")

    # Subcommands via mutually exclusive flags (kept backward compatible)
    modes: ArgumentGroup = parser.add_argument_group("modes", "Primary operations (choose any combination)")
    modes.add_argument("--visualize", "-v", action="store_true", help="Visualize a sample trajectory")
    modes.add_argument("--variations", "-var", action="store_true", help="Display variation images")
    modes.add_argument("--structure", "-str", action="store_true", help="Print hierarchical structure")
    modes.add_argument("--table", "-t", action="store_true", help="Print tabular overview")
    modes.add_argument("--timeline", "-tl", action="store_true", help="Create timeline animation")
    modes.add_argument("--compare", "-c", action="store_true", help="Compare multiple datasets")

    # Visualization customization
    viz: ArgumentGroup = parser.add_argument_group("visualization", "Parameters controlling visualization details")
    viz.add_argument("--sample", "-s", type=int, default=0, help="Sample index to visualize (default: 0)")
    viz.add_argument("--frames", "-f", type=int, default=5, help="Max frames to display in static visualization")
    viz.add_argument("--depth", "-dep", type=int, default=8, help="Max depth for structure visualization")
    viz.add_argument("--detailed", "-d", action="store_true", help="Show detailed variation information")
    viz.add_argument("--variation-labels", "-vl", nargs="+", help="Custom labels for variations")
    viz.add_argument("--variation-index", "-vi", type=int, default=0, help="Variation index for timeline")
    viz.add_argument("--variation-name", "-vn", help="Variation name (overrides index) for timeline")

    return parser


HELP_FLAGS: set[str] = {"-h", "--help", "--help-all"}


CONTEXTUAL_GROUP_HINTS: dict[str, tuple[str, ...]] = {
    "visualize": ("visualization", "layout", "text"),
    "variations": ("visualization", "text"),
    "structure": ("modes",),
    "table": ("modes",),
    "timeline": ("visualization", "layout", "text"),
    "compare": ("modes", "performance", "text"),
    "text_override": ("text",),
    "disable_text_key": ("text",),
    "disable_all_text": ("text",),
    "grid": ("layout",),
    "tight": ("layout",),
    "no_constrained": ("layout",),
    "wspace": ("layout",),
    "hspace": ("layout",),
    "padding": ("layout",),
    "dpi": ("layout",),
    "theme": ("layout",),
    "sample": ("visualization",),
    "frames": ("visualization",),
    "depth": ("visualization",),
    "detailed": ("visualization",),
    "variation_labels": ("visualization",),
    "variation_index": ("visualization",),
    "variation_name": ("visualization",),
    "no_perf_mode": ("performance",),
    "export": ("optional arguments",),
    "save": ("optional arguments",),
}


def _format_action_help(action: Action) -> str:
    """Return a compact, single-argument help line for an action."""
    opts: str = " ".join(action.option_strings) if action.option_strings else action.dest
    metavar: str = f" {action.metavar}" if getattr(action, "metavar", None) else ""
    help_text: str = action.help or ""
    default: str = (
        f" [default: {action.default}]"
        if action.default not in {None, False, SUPPRESS} and not isinstance(action.default, list)
        else ""
    )
    return f"  {opts}{metavar}\n      {help_text}{default}"


def _filter_help_tokens(argv: list[str]) -> list[str]:
    return [arg for arg in argv[1:] if arg not in HELP_FLAGS]


def _attempt_partial_parse(parser: ArgumentParser, args: list[str]) -> Namespace | None:
    if not args:
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            namespace, _ = parser.parse_known_args(args)
    except SystemExit:
        return None
    return namespace


def _gather_selected_actions(parser: ArgumentParser, argv: list[str], namespace: Namespace | None) -> set[Action]:
    selected: set[Action] = set()
    option_mapping = getattr(parser, "_option_string_actions", {})
    option_lookup: dict[str, Action] = dict(option_mapping) if isinstance(option_mapping, Mapping) else {}
    stripped_args: list[str] = _filter_help_tokens(argv)

    for token in stripped_args:
        if token in option_lookup:
            selected.add(option_lookup[token])

    if namespace is None:
        return selected

    for action in getattr(parser, "_actions", ()):
        # Skip help-like actions without importing private classes
        if action.dest == "help" or any(
            opt in {"-h", "--help", "--help-all"} for opt in getattr(action, "option_strings", [])
        ):
            continue
        if action.dest in {SUPPRESS, None}:
            continue
        if not hasattr(namespace, action.dest):
            continue
        value = getattr(namespace, action.dest)
        default = parser.get_default(action.dest)
        if default is SUPPRESS:
            default = None
        if action.option_strings:
            # Detect store_true/store_false via public attributes
            is_store_true: bool = getattr(action, "nargs", None) == 0 and getattr(action, "const", None) is True
            is_store_false: bool = getattr(action, "nargs", None) == 0 and getattr(action, "const", None) is False

            if is_store_true:
                if bool(value):
                    selected.add(action)
            elif is_store_false:
                if value is False and default is True:
                    selected.add(action)
            elif value is not None and value != default:
                selected.add(action)
        elif isinstance(value, list):
            if value:
                selected.add(action)
        elif value not in {None, False}:
            selected.add(action)

    return selected


def show_contextual_help(argv: list[str], parser: ArgumentParser) -> None:
    """Render contextual help based on the options present alongside -h/--help."""
    filtered_args: list[str] = _filter_help_tokens(argv)
    namespace: Namespace | None = _attempt_partial_parse(parser, filtered_args)
    selected_actions: set[Action] = _gather_selected_actions(parser, argv, namespace)

    if not selected_actions:
        parser.print_help()
        return

    action_to_group: dict[Action, ArgumentGroup] = {}
    for gp in getattr(parser, "_action_groups", ()):
        for action in getattr(gp, "_group_actions", ()):
            action_to_group[action] = gp

    selected_groups: set[str] = {"positional arguments", "optional arguments"}
    for action in selected_actions:
        group: ArgumentGroup | None = action_to_group.get(action)
        if group is not None and group.title:
            selected_groups.add(group.title.lower())
        if action.dest:
            selected_groups.update(hint.lower() for hint in CONTEXTUAL_GROUP_HINTS.get(action.dest, ()))

    prog: str = parser.prog or pathlib.Path(argv[0]).name
    print(f"Usage: {prog} [options] [file_paths]")
    print()

    for group in getattr(parser, "_action_groups", ()):
        if group is None:
            continue
        title: str = group.title or ""
        if not title:
            continue
        if title.lower() not in selected_groups:
            continue
        relevant_actions: list[Action] = [
            a for a in getattr(group, "_group_actions", ()) if getattr(a, "help", None) is not SUPPRESS
        ]
        if not relevant_actions:
            continue
        print(f"{title}:")
        for action in relevant_actions:
            line: str = _format_action_help(action)
            print(line)
        print()

    print("To see full help, run with --help-all or without specific mode flags.")


def parse_options(args: Namespace) -> None:
    """Translate CLI args into InspectorOptions and install globally."""
    # Text overrides and disabling
    mapping: dict[str, str | None] = {}
    if args.text_override:
        for pair in args.text_override:
            if len(pair) == 2:
                k, v = pair
                mapping[k] = v
    if args.disable_text_key:
        for k_list in args.disable_text_key:
            if k_list:
                mapping[k_list[0]] = None
    text_cfg: InspectorTextConfig = (
        InspectorTextConfig.default().override(mapping) if mapping else InspectorTextConfig.default()
    )
    if args.disable_all_text:
        text_cfg = InspectorTextConfig(mapping=text_cfg.mapping, disable_all=True)

    layout = MatplotlibLayoutConfig(
        dpi=args.dpi,
        constrained=not args.no_constrained,
        tight_layout=args.tight,
        grid=args.grid,
        wspace=args.wspace,
        hspace=args.hspace,
        padding=args.padding,
        theme=args.theme,
    )

    opts = InspectorOptions(
        text=text_cfg, layout=layout, disable_all_text=args.disable_all_text, perf_mode=not args.no_perf_mode
    )
    set_inspector_options(opts)


def _load_dataset_for_compare(path: str) -> tuple[list[StateImgTraj], list[ActionTraj], Specs] | None:
    """Load one dataset path for compare mode."""
    try:
        print(f"Loading dataset: {path}")
        state_img_trajs, action_trajs = load_data_file(path)
        spec: Specs = get_data_specs(state_img_trajs, action_trajs)
    except Exception as exc:
        print(f"Error loading {path}: {exc!s}")
        return None
    else:
        return state_img_trajs, action_trajs, spec


def main() -> int:
    """Entry point for hierarchical CLI inspector."""
    parser: ArgumentParser = build_parser()

    # Intercept help flags to provide contextual help and avoid argparse errors
    argv: list[str] = sys.argv
    if "--help-all" in argv:
        # Full help requested
        parser.print_help()
        return 0
    if "--help" in argv or "-h" in argv:
        # Contextual help based on flags present
        show_contextual_help(argv, parser)
        return 0

    args: Namespace = parser.parse_args()
    parse_options(args)

    # Get all file paths
    all_files: list[str] = []

    # Determine patterns to search - either from args or direct #file: reference
    patterns: list[str] = args.file_paths
    if not patterns and len(sys.argv) > 1 and sys.argv[1].startswith("#file:"):
        patterns = [sys.argv[1]]

    # Process all patterns
    for pattern in patterns:
        matches: list[str] = search_data_files(pattern)
        if matches:
            all_files.extend(matches)
        else:
            print(f"Warning: No files found matching pattern: {pattern}")

    # Filter out any falsy values and remove duplicates while preserving order
    seen: set[str] = set()
    unique_files: list[str] = []
    for f in all_files:
        if not f:
            continue
        if f in seen:
            continue
        seen.add(f)
        unique_files.append(f)

    all_files = unique_files

    # Print summary of found files
    if all_files:
        print(f"Found {len(all_files)} data file(s):")
        for f in all_files:
            print(f"  - {f}")

    if not all_files:
        print("Usage: python inspect_data.py <data_file_path> [options]")
        print("   or: python inspect_data.py #file:your_file [options]")
        print("   or: python inspect_data.py file1.h5 file2.h5 --compare")
        return 1

    # Check for compare mode (multiple files)
    if args.compare and len(all_files) > 1:
        # Load all datasets and compare
        datasets: list[tuple[list[StateImgTraj], list[ActionTraj]]] = []
        specs: list[Specs] = []
        names: list[str] = []

        for path in all_files:
            loaded = _load_dataset_for_compare(path)
            if loaded is None:
                continue
            state_img_trajs, action_trajs, spec = loaded
            datasets.append((state_img_trajs, action_trajs))
            specs.append(spec)
            names.append(pathlib.Path(path).name)

        if len(datasets) >= 2:
            compare_datasets(datasets, names, specs, show_plot=True, save_path=args.save)
        else:
            print("Need at least 2 valid datasets to compare.")
            return 1

        # Export if requested
        if args.export:
            export_summary(all_files, specs, args.export)

        return 0

    file_path: str = all_files[0]  # Use the first file

    # Export only
    if args.export and not (args.visualize or args.variations or args.structure or args.table or args.timeline):
        try:
            state_img_trajs, action_trajs = load_data_file(file_path)
            spec = get_data_specs(state_img_trajs, action_trajs)
            export_summary([file_path], [spec], args.export)
            print(f"Summary exported to {args.export}")

        except Exception as e:
            print(f"Error exporting data: {e!s}")
            return 1
        else:
            return 0

    # Timeline animation
    if args.timeline:
        try:
            print(f"Creating timeline animation for {file_path}")
            state_img_trajs, action_trajs = load_data_file(file_path)
            create_timeline_animation(
                state_img_trajs,
                action_trajs,
                sample_idx=args.sample,
                variation_labels=args.variation_labels,
                variation_index=args.variation_index,
                variation_name=args.variation_name,
                save_path=args.save,
            )

        except Exception as e:
            print(f"Error creating timeline animation: {e!s}")
            traceback.print_exc()
            return 1
        else:
            return 0

    # Standard inspection
    return run_inspector(
        file_path=file_path,
        visualize=args.visualize,
        show_variations=args.variations,
        show_structure=args.structure,
        show_table=args.table,
        sample_idx=args.sample,
        max_frames=args.frames,
        detailed=args.detailed,
        depth=args.depth,
        save_path=args.save,
    )


if __name__ == "__main__":
    sys.exit(main())
