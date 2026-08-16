"""Search pair data generation utilities for SPAR.

Generates start/goal state pairs for search experiments with optional visual
variations applied to the start images, goal images, or both.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from logging import Logger, getLogger
import pathlib
import pickle
from typing import TYPE_CHECKING, Literal, TypeAlias

import h5py
import numpy as np
from numpy.typing import NDArray
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.rule import Rule
from rich.table import Table

from spar.utils.config_utils.samplers import Sampler
from spar.utils.data_utils.hdf5_common import (
    CompressionType,
    create_array_dataset,
    create_utf8_string_dataset,
    ensure_hdf5_path,
    get_chunk_shape_4d,
    normalize_compression_value,
    open_hdf5_for_write,
    write_utf8_string_list_attr,
)
from spar.utils.env_utils import get_environment
from spar.utils.env_utils.effects_core import (
    EffectConfigMap,
    EffectConfigValue,
    EffectStage,
    StagePipelines,
    build_stage_pipelines,
    freeze as freeze_effects_cfg,
)
from spar.utils.log_utils.console_logger import terminal_console as console

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.progress import TaskID

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.utils.config_utils.config_schema import GenSearchDataSPARConfig, SearchPairsDataConfig

logger: Logger = getLogger(__name__)

ImageArray: TypeAlias = NDArray[np.float32]
JSONParamValue: TypeAlias = str | int | float | bool | list["JSONParamValue"] | dict[str, "JSONParamValue"] | None


@dataclass(slots=True)
class VariationResult:
    """Holds rendered images and parameter strings per-variation.

    Attributes:
        name: Variation name (effect key)
        images_start: Rendered start images (N, C, H, W) or None
        images_goal: Rendered goal images (N, C, H, W) or None
        params_start: Per-pair JSON-like strings of parameters (N,) or None
        params_goal: Per-pair JSON-like strings of parameters (N,) or None
    """

    name: str
    images_start: ImageArray | None
    images_goal: ImageArray | None
    params_start: list[str] | None
    params_goal: list[str] | None


def _extract_params(pipes: StagePipelines) -> dict[str, dict[str, dict[str, JSONParamValue]]]:
    """Extract concrete parameter values from a StagePipelines instance.

    Returns a nested mapping: {stage: {effect_name: params_dict}}
    """
    out: dict[str, dict[str, dict[str, JSONParamValue]]] = {"pre": {}, "obj": {}, "post": {}}
    for stage in (EffectStage.PRE_RENDER, EffectStage.OBJECT_RENDER, EffectStage.POST_RENDER):
        for eff, params in pipes.get_effects_by_stage(stage):
            out["pre" if stage == EffectStage.PRE_RENDER else "obj" if stage == EffectStage.OBJECT_RENDER else "post"][
                eff.__effect_metadata__.name
            ] = {k: _to_json_value(v) for k, v in params.items()}
    return out


def _to_json_value(value: EffectConfigValue | NDArray[np.generic]) -> JSONParamValue:
    """Convert NumPy and container values to JSON-compatible Python values."""
    if isinstance(value, Sampler):
        return repr(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        numeric = value.item()
        if isinstance(numeric, bool):
            return numeric
        if isinstance(numeric, int):
            return numeric
        return float(numeric)
    if isinstance(value, np.ndarray):
        return [_to_json_value(item) for item in value.tolist()]
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return repr(value)
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_value(item) for item in value]
    return str(value)


def save_pairs(
    out_path: str,
    env_name: str,
    start_imgs: ImageArray,
    goal_imgs: ImageArray,
    *,
    variations: list[VariationResult] | None = None,
    reverse_goal: bool = False,
    goal_num_steps: int | None = None,
    effects_config: EffectConfigMap | None = None,
    compression: CompressionType = "none",
    states_start: list[ABCState] | None = None,
    states_goal: list[ABCState] | None = None,
) -> None:
    """Save start/goal pairs with optional variations to an HDF5 file.

    Datasets:
    - /pairs/start/images: (N, C, H, W)
    - /pairs/goal/images:  (N, C, H, W)
    - /pairs/start/states_pickle: (N,) variable-length bytes of pickled ABCState objects (optional)
    - /pairs/goal/states_pickle:  (N,) variable-length bytes of pickled ABCState objects (optional)
    - /pairs/start/variations/<name>/images (optional)
    - /pairs/start/variations/<name>/params (optional, per-pair serialized dict)
    - /pairs/goal/variations/<name>/images  (optional)
    - /pairs/goal/variations/<name>/params  (optional)

    Root attributes (for tooling/inspectors):
    - env_name: str
    - num_pairs: int
    - reverse_goal: bool
    - goal_num_steps: int
    - dataset_kind: "search_pairs_v1"
    - variant_names: list[str] (union across start/goal plus "base")
    - variant_names_start: list[str]
    - variant_names_goal: list[str]
    - variant_sides: one of {"none","start","goal","both"}
    """
    path: str = ensure_hdf5_path(out_path)
    pathlib.Path(path).parent.mkdir(exist_ok=True, parents=True)

    # Progress-friendly write
    # Count total write steps: base start + base goal + each variation images (+params if present)
    total_steps: int = 2
    if states_start is not None and states_goal is not None:
        total_steps += 2
    if variations:
        for var in variations:
            if var.images_start is not None:
                total_steps += 1
                if var.params_start is not None:
                    total_steps += 1
            if var.images_goal is not None:
                total_steps += 1
                if var.params_goal is not None:
                    total_steps += 1

    with (
        open_hdf5_for_write(path) as f,
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Writing search pairs"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed:>3}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress,
    ):
        write_task: TaskID = progress.add_task("Writing HDF5", total=total_steps)

        # Root attributes
        f.attrs["env_name"] = env_name
        f.attrs["num_pairs"] = int(start_imgs.shape[0])
        f.attrs["reverse_goal"] = reverse_goal
        f.attrs["goal_num_steps"] = -1 if goal_num_steps is None else goal_num_steps
        if effects_config is not None:
            f.attrs["effects_config_frozen"] = str(freeze_effects_cfg(effects_config))
        # Mark file type for downstream tooling (inspector, etc.)
        f.attrs["dataset_kind"] = "search_pairs_v1"

        # Base images
        grp_pairs: h5py.Group = f.create_group("pairs")
        grp_s: h5py.Group = grp_pairs.create_group("start")
        grp_g: h5py.Group = grp_pairs.create_group("goal")

        image_chunks: tuple[int, int, int, int] = get_chunk_shape_4d(start_imgs)
        create_array_dataset(grp_s, "images", start_imgs, compression=compression, chunks=image_chunks)
        progress.advance(write_task, 1)
        create_array_dataset(grp_g, "images", goal_imgs, compression=compression, chunks=image_chunks)
        progress.advance(write_task, 1)

        # Optional: store pickled ABCState objects for exact environment validation
        if states_start is not None and states_goal is not None:
            # Validate pair counts before writing one variable-length byte array per state.
            if len(states_start) != len(states_goal):
                raise ValueError(
                    f"states_start and states_goal length mismatch: {len(states_start)} vs {len(states_goal)}"
                )
            n: int = len(states_start)
            dt: h5py.vlen_dtype = h5py.vlen_dtype(np.dtype("uint8"))
            # Create datasets with correct shape and dtype, then assign per-index
            if compression == "none":
                ds_s: h5py.Dataset = grp_s.create_dataset("states_pickle", shape=(n,), dtype=dt, track_times=False)
                ds_g: h5py.Dataset = grp_g.create_dataset("states_pickle", shape=(n,), dtype=dt, track_times=False)
            else:
                ds_s = grp_s.create_dataset(
                    "states_pickle", shape=(n,), dtype=dt, compression=compression, track_times=False
                )
                ds_g = grp_g.create_dataset(
                    "states_pickle", shape=(n,), dtype=dt, compression=compression, track_times=False
                )

            def _encode_state_pair(i: int) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
                try:
                    b_s: bytes = pickle.dumps(states_start[i], protocol=pickle.HIGHEST_PROTOCOL)
                    b_g: bytes = pickle.dumps(states_goal[i], protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as exc:
                    raise RuntimeError(f"Failed to pickle ABCState at index {i}: {exc}") from exc
                return np.frombuffer(b_s, dtype=np.uint8), np.frombuffer(b_g, dtype=np.uint8)

            for i in range(n):
                enc_s: NDArray[np.uint8]
                enc_g: NDArray[np.uint8]
                enc_s, enc_g = _encode_state_pair(i)
                ds_s[i] = enc_s
                ds_g[i] = enc_g
            progress.advance(write_task, 2)
            # progress.advance(write_task, 1)

        # Variations
        start_variant_names: set[str] = set()
        goal_variant_names: set[str] = set()
        start_variations_grp: h5py.Group | None = None
        goal_variations_grp: h5py.Group | None = None
        if variations:
            for var in variations:
                # Start
                if var.images_start is not None:
                    if start_variations_grp is None:
                        start_variations_grp = grp_s.create_group("variations")
                    vg: h5py.Group = start_variations_grp.create_group(var.name)
                    create_array_dataset(
                        vg,
                        "images",
                        var.images_start,
                        compression=compression,
                        chunks=get_chunk_shape_4d(var.images_start),
                    )
                    progress.advance(write_task, 1)
                    start_variant_names.add(var.name)
                    if var.params_start is not None:
                        create_utf8_string_dataset(vg, "params", var.params_start)
                        progress.advance(write_task, 1)

                # Goal
                if var.images_goal is not None:
                    if goal_variations_grp is None:
                        goal_variations_grp = grp_g.create_group("variations")
                    vg = goal_variations_grp.create_group(var.name)
                    create_array_dataset(
                        vg,
                        "images",
                        var.images_goal,
                        compression=compression,
                        chunks=get_chunk_shape_4d(var.images_goal),
                    )
                    progress.advance(write_task, 1)
                    goal_variant_names.add(var.name)
                    if var.params_goal is not None:
                        create_utf8_string_dataset(vg, "params", var.params_goal)
                        progress.advance(write_task, 1)

        # Write variation-related attributes for inspector compatibility
        # Include 'base' implicitly in the union for consistency with episode format
        union_variant_names: list[str] = sorted({"base", *start_variant_names, *goal_variant_names})
        write_utf8_string_list_attr(f.attrs, "variant_names", union_variant_names)

        # Per-side names can help UIs show goal-specific info
        write_utf8_string_list_attr(f.attrs, "variant_names_start", sorted(start_variant_names))
        write_utf8_string_list_attr(f.attrs, "variant_names_goal", sorted(goal_variant_names))

        # Indicate where variations were applied among {none,start,goal,both}
        if start_variant_names and goal_variant_names:
            f.attrs["variant_sides"] = "both"
        elif start_variant_names:
            f.attrs["variant_sides"] = "start"
        elif goal_variant_names:
            f.attrs["variant_sides"] = "goal"
        else:
            f.attrs["variant_sides"] = "none"

        # Advance the progress bar to its total before closing it.
        try:
            progress.update(write_task, completed=total_steps)
            progress.refresh()
        except Exception:
            # If progress is already finalized or any issue occurs, ignore safely
            pass

    size_mb: float = pathlib.Path(path).stat().st_size / (1024**2)
    logger.info(f"[green]✓[/green] Saved data: {path} ({size_mb:.1f} MB)")


def generate_search_pairs(
    env_name: str,
    num_pairs: int,
    out_path: str,
    *,
    reverse_goal: bool = False,
    goal_num_steps: int | None = None,
    goal_seeds: list[int] | NDArray[np.intp] | None = None,
    start_level_seed: int | None = None,
    num_start_levels: int | None = None,
    effects_config: EffectConfigMap | None = None,
    apply_variations_to: Literal["none", "start", "goal", "both"] = "none",
    compression: CompressionType = "none",
) -> str:
    """Generate N start/goal pairs (and optional variations) and save to HDF5.

    Args:
        env_name: Environment identifier (e.g., "cube3").
        num_pairs: Number of pairs to generate.
        out_path: Output file path (.h5 appended if missing).
        reverse_goal: If True, scramble goal states instead of using canonical goals.
        goal_num_steps: Scramble length for reverse goals (or None to sample per-pair).
        goal_seeds: Optional per-pair seeds used by env.generate_goal_states.
        start_level_seed: Optional seed to derive level_seeds for start states.
        num_start_levels: Optional number of unique start levels.
        effects_config: Variations configuration (same schema as generator.py).
        apply_variations_to: Where to apply variations: none/start/goal/both.
        compression: HDF5 compression for images.

    Returns:
        The absolute path to the saved file.
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")

    env: ABCEnvironment[ABCState] = get_environment(env_name)

    logger.info(f"Generating {num_pairs:,} start/goal pairs...")

    # Generate start states (respect level_seeds if supported by the environment)
    level_seeds: list[int] | None = None
    if "level_seeds" in env.generate_start_states.__code__.co_varnames and (
        (num_start_levels is not None and num_start_levels > 0)
        or (start_level_seed is not None and start_level_seed >= 0)
    ):
        if num_start_levels is None or num_start_levels < 1:
            num_start_levels = num_pairs
        if start_level_seed is None or start_level_seed < 0:
            start_level_seed = int(np.random.randint(0, 2**31 - 1))
        trajs_per_level: int = num_pairs // num_start_levels
        extra_trajs: int = num_pairs % num_start_levels
        levels: NDArray[np.int64] = np.arange(start_level_seed, start_level_seed + num_start_levels, dtype=np.int64)
        seeds_np: NDArray[np.int64] = np.concatenate((np.tile(levels, trajs_per_level), levels[:extra_trajs]))
        np.random.shuffle(seeds_np)
        level_seeds = [int(x) for x in seeds_np.tolist()]

    starts: list[ABCState] = (
        env.generate_start_states(num_pairs, level_seeds=level_seeds)
        if level_seeds is not None
        else env.generate_start_states(num_pairs)
    )
    logger.info("[green]✓[/green] Start states generated")

    # Generate goal states
    goal_state_fn: Callable[..., list[ABCState]] = env.generate_goal_states
    goals: list[ABCState] = goal_state_fn(starts, goal_num_steps, seeds=goal_seeds, reverse_goal=bool(reverse_goal))
    logger.info("[green]✓[/green] Goal states generated")

    # Render each base side in one batch.
    start_imgs: ImageArray = env.state_to_real(starts)
    goal_imgs: ImageArray = env.state_to_real(goals)
    logger.info("[green]✓[/green] Rendered base start/goal images")

    # Variations
    variations_out: list[VariationResult] = []
    if effects_config and apply_variations_to != "none":
        var_names: list[str] = list(effects_config.keys())
        n, c, h, w = start_imgs.shape
        apply_to_start: bool = apply_variations_to in {"start", "both"}
        apply_to_goal: bool = apply_variations_to in {"goal", "both"}

        # Progress for variations across all names and pairs
        total_units: int = len(var_names) * n * ((1 if apply_to_start else 0) + (1 if apply_to_goal else 0))
        total_units = max(total_units, 1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Applying variations"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed:>3}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            var_task = progress.add_task("Variations", total=total_units)

            for name in var_names:
                # Allocate outputs lazily for this variation
                imgs_s: ImageArray | None = np.zeros((n, c, h, w), dtype=np.float32) if apply_to_start else None
                imgs_g: ImageArray | None = np.zeros((n, c, h, w), dtype=np.float32) if apply_to_goal else None
                params_s: list[str] | None = [] if apply_to_start else None
                params_g: list[str] | None = [] if apply_to_goal else None

                cfg_single: dict[str, EffectConfigValue] = {name: effects_config[name]}

                # Sample fresh parameters per-pair by (re)building the pipeline each time
                for i in range(n):
                    pipelines: dict[str, StagePipelines] = build_stage_pipelines(cfg_single)
                    # Skip entries that are disabled or invalid.
                    if name not in pipelines:
                        # Advance for both potential units to keep bar honest
                        if apply_to_start:
                            progress.advance(var_task, 1)
                        if apply_to_goal:
                            progress.advance(var_task, 1)
                        continue
                    pipes: StagePipelines = pipelines[name]

                    # Extract and serialize params (store as string for HDF5 compatibility)
                    params_dict: dict[str, dict[str, dict[str, JSONParamValue]]] = _extract_params(pipes)
                    params_str = str(params_dict)

                    if apply_to_start and imgs_s is not None and params_s is not None:
                        imgs_s[i] = env.state_to_real([starts[i]], effects=pipes)[0]
                        params_s.append(params_str)
                        progress.advance(var_task, 1)
                    if apply_to_goal and imgs_g is not None and params_g is not None:
                        imgs_g[i] = env.state_to_real([goals[i]], effects=pipes)[0]
                        params_g.append(params_str)
                        progress.advance(var_task, 1)

                # Skip empty output when pipelines and start or goal images are disabled.
                if (apply_to_start and imgs_s is None) and (apply_to_goal and imgs_g is None):
                    continue
                variations_out.append(
                    VariationResult(
                        name=name, images_start=imgs_s, images_goal=imgs_g, params_start=params_s, params_goal=params_g
                    )
                )

    # Save
    save_pairs(
        out_path=out_path,
        env_name=env_name,
        start_imgs=start_imgs,
        goal_imgs=goal_imgs,
        variations=variations_out or None,
        reverse_goal=reverse_goal,
        goal_num_steps=goal_num_steps,
        effects_config=effects_config,
        compression=compression,
        states_start=starts,
        states_goal=goals,
    )

    return str(pathlib.Path(ensure_hdf5_path(out_path)).resolve())


def generate_search_data(env: ABCEnvironment[ABCState], cfg: GenSearchDataSPARConfig) -> None:
    """Stage entry: generate start/goal search pairs using config-driven datasets.

    Supports both cfg.data (preferred) and cfg.search_data (for configs packaged as
    `@package search_data` like `spar/configs/search_data/default.yaml`).
    Also supports a global `effects` block under the top-level search data config,
    which individual datasets can override via their own `effects` field.
    """
    # Prefer the explicit data field and fall back to search_data for default.yaml compatibility.
    data_cfg: SearchPairsDataConfig | None = getattr(cfg, "data", None)
    if data_cfg is None or not getattr(data_cfg, "datasets", None):
        # Try alternate field populated by `/search_data: default`
        alt: SearchPairsDataConfig | None = getattr(cfg, "search_data", None)
        if alt is not None:
            data_cfg = alt
    if data_cfg is None:
        logger.info(Panel("No search data configuration found (data/search_data)", border_style="yellow"))
        return
    save_dir: str = getattr(data_cfg, "save_dir", "data/search_pairs")

    if not data_cfg.datasets:
        logger.info(Panel("No datasets specified in search_data.datasets", border_style="yellow"))
        return

    # Summary panel similar to generator.py
    # High-level info panel
    msg_str: str = f"Save directory: {pathlib.Path(save_dir).resolve()}\nProcessing {len(data_cfg.datasets)} datasets."
    logger.info(
        Panel(
            msg_str,
            border_style="bright_blue",
            padding=(1, 2),
            title="[bright_blue]SPAR Search Pair Generation Pipeline[/bright_blue]",
            width=120,
        )
    )

    # Dataset summary table
    summary = Table(show_header=True, box=None, padding=(0, 1))
    summary.add_column("Dataset", style="bold cyan")
    summary.add_column("Pairs", style="bright_white", justify="right")
    summary.add_column("Reverse", style="bright_white", justify="center")
    total_pairs = 0
    for ds in data_cfg.datasets:
        name: str = getattr(ds, "name", "unnamed")
        num_pairs: int = getattr(ds, "num_pairs", 0)
        reverse_goal: bool = getattr(ds, "reverse_goal", False)
        summary.add_row(name, f"{num_pairs:,}", "yes" if reverse_goal else "no")
        total_pairs += num_pairs

    summary.add_row("", "", "", end_section=True)
    summary.add_row("[bold]TOTAL[/bold]", f"[bold]{total_pairs:,}[/bold]", "[dim]-[/dim]")
    logger.info(Panel(summary, title="[bold]Dataset Summary[/bold]", border_style="blue", padding=(1, 1), width=120))

    # Process datasets
    # Resolve optional global effects config once
    global_effects_config: EffectConfigMap | None = getattr(data_cfg, "effects", None)

    for idx, ds in enumerate(data_cfg.datasets, 1):
        if idx > 1:
            logger.info("")
            logger.info(Rule())
            logger.info("")
        name = getattr(ds, "name", f"dataset_{idx}")
        logger.info(f"[bold blue]Dataset {idx}/{len(data_cfg.datasets)}: {name.upper()}[/bold blue]")

        out_dir: str = getattr(ds, "save_dir", None) or save_dir
        pathlib.Path(out_dir).mkdir(exist_ok=True, parents=True)
        file_name: str = getattr(ds, "file_name", f"{cfg.env.name}_{name}_{getattr(ds, 'num_pairs', 0)}pairs")
        out_path: str = str(pathlib.Path(out_dir) / file_name)

        compression_raw: str | bool | None = getattr(ds, "compression", None)
        compression: CompressionType = normalize_compression_value(compression_raw)

        # Seeds for start states
        start_seed: int | None = getattr(ds, "start_seed", None)
        num_seeds: int | None = getattr(ds, "num_seeds", None)

        # Generate pairs
        # Use dataset-specific effects when present and otherwise use the shared effects config.
        dataset_effects: EffectConfigMap | None = getattr(ds, "effects", None) or global_effects_config

        # Small details panel per dataset
        details_lines: list[str] = []
        details_lines.append(
            f"Pairs: {getattr(ds, 'num_pairs', 0)} • "
            f"Reverse goal: {'yes' if getattr(ds, 'reverse_goal', False) else 'no'}"
        )
        apply_to: Literal["none", "start", "goal", "both"] = getattr(ds, "apply_variations_to", "none") or "none"
        if getattr(ds, "effects", None) or global_effects_config:
            details_lines.append(f"Variations: {apply_to} • Compression: {compression}")
        else:
            details_lines.append(f"Variations: disabled • Compression: {compression}")
        logger.info(
            Panel(
                "\n".join(details_lines),
                border_style="dim",
                padding=(0, 2),
                title=f"[dim]{name} details[/dim]",
                width=120,
            )
        )

        abs_path: str = generate_search_pairs(
            env_name=env.get_env_name(),
            num_pairs=getattr(ds, "num_pairs", 0),
            out_path=out_path,
            reverse_goal=getattr(ds, "reverse_goal", False),
            goal_num_steps=getattr(ds, "goal_num_steps", None),
            goal_seeds=getattr(ds, "goal_seeds", None),
            start_level_seed=start_seed,
            num_start_levels=num_seeds,
            effects_config=dataset_effects,
            apply_variations_to=apply_to,
            compression=compression,
        )

        logger.info(f"[green]✓[/green] Saved: [bold cyan]{abs_path}[/bold cyan]")

    logger.info("\n[bold green]✓ Search pair data generation complete.[/bold green]")
    logger.info(f"Files saved to: {pathlib.Path(save_dir).resolve()}")
