"""Compose Hydra configs and render command-line configuration help."""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import math
import os
from pathlib import Path
import random
import time
from typing import TYPE_CHECKING

from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import OmegaConf

from .help_render import (
    render_common_flags,
    render_context,
    render_examples,
    render_experiments,
    render_footer,
    render_group_tree,
    render_header,
    render_resolved_defaults,
    render_usage,
)
from .samplers import encode_choice_sampler, encode_integer_range_sampler, encode_range_sampler

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from omegaconf import DictConfig, ListConfig
    from omegaconf.nodes import (
        AnyNode,
        BooleanNode,
        BytesNode,
        EnumNode,
        FloatNode,
        IntegerNode,
        PathNode,
        StringNode,
        ValueNode,
    )

    from .config_schema import SPARConfig

    NodeType: TypeAlias = (
        AnyNode | BooleanNode | BytesNode | EnumNode | FloatNode | IntegerNode | PathNode | StringNode | ValueNode
    )

    Pruned: TypeAlias = NodeType | dict[str, "Pruned"] | list["Pruned"] | tuple["Pruned", ...]

    PrunedDict: TypeAlias = dict[str, Pruned]


ResolverScalar = str | int | float | bool


def _range_resolver(low: ResolverScalar, high: ResolverScalar) -> str:
    """Encode a continuous range resolver value."""
    return encode_range_sampler(float(low), float(high))


def _integer_range_resolver(low: ResolverScalar, high: ResolverScalar) -> str:
    """Encode an integer range resolver value."""
    return encode_integer_range_sampler(int(float(low)), int(float(high)))


def _multiply_resolver(x: ResolverScalar, y: ResolverScalar) -> int:
    """Multiply two resolver values and return an integer."""
    return int(float(x) * float(y))


def _add_resolver(x: ResolverScalar, y: ResolverScalar) -> int:
    """Add two resolver values and return an integer."""
    return int(float(x) + float(y))


def _subtract_resolver(x: ResolverScalar, y: ResolverScalar) -> int:
    """Subtract two resolver values and return an integer."""
    return int(float(x) - float(y))


def _divide_resolver(x: ResolverScalar, y: ResolverScalar) -> int:
    """Divide two resolver values and return an integer."""
    return int(float(x) / float(y))


def _square_root_resolver(x: ResolverScalar) -> float:
    """Compute the square root of a resolver value."""
    return math.sqrt(float(x))


def _conditional_resolver(
    cond: ResolverScalar, true_value: ResolverScalar, false_value: ResolverScalar
) -> ResolverScalar:
    """Select one resolver value based on a condition."""
    return true_value if OmegaConf.to_object(cond) else false_value


def _contains_resolver(string: ResolverScalar, substring: ResolverScalar) -> bool:
    """Check whether one resolver value occurs in another."""
    return str(substring) in str(string)


def register_omega_conf_resolvers() -> None:
    """Register custom OmegaConf resolvers for config interpolation."""
    has: Callable[[str], bool] = OmegaConf.has_resolver
    # Samplers
    if not has("range"):
        OmegaConf.register_new_resolver("range", _range_resolver)
    if not has("irange"):
        OmegaConf.register_new_resolver("irange", _integer_range_resolver)
    if not has("choice"):
        OmegaConf.register_new_resolver("choice", encode_choice_sampler, use_cache=True)
    # Math operations
    if not has("mul"):
        OmegaConf.register_new_resolver("mul", _multiply_resolver)
    if not has("add"):
        OmegaConf.register_new_resolver("add", _add_resolver)
    if not has("sub"):
        OmegaConf.register_new_resolver("sub", _subtract_resolver)
    if not has("div"):
        OmegaConf.register_new_resolver("div", _divide_resolver)
    if not has("sqrt"):
        OmegaConf.register_new_resolver("sqrt", _square_root_resolver)

    # Date and time
    if not has("timestamp"):
        OmegaConf.register_new_resolver("timestamp", lambda: int(time.time()))

    # Path operations
    if not has("join_path"):
        OmegaConf.register_new_resolver("join_path", os.path.join)
    if not has("basename"):
        OmegaConf.register_new_resolver("basename", os.path.basename)
    if not has("dirname"):
        OmegaConf.register_new_resolver("dirname", os.path.dirname)

    # Conditional resolver
    if not has("if_else"):
        OmegaConf.register_new_resolver("if_else", _conditional_resolver)

    # String operations
    if not has("contains"):
        OmegaConf.register_new_resolver("contains", _contains_resolver)

    # Help rendering (dynamic Rich-ish output)
    if not has("spar_help_usage"):
        OmegaConf.register_new_resolver(
            "spar_help_usage",
            lambda stage=None, env=None, stage_choice=None, env_choice=None: render_usage(
                stage, env, stage_choice, env_choice
            ),
            use_cache=False,
        )
    if not has("spar_help_context"):
        OmegaConf.register_new_resolver(
            "spar_help_context",
            lambda stage=None, env=None, stage_choice=None, env_choice=None: render_context(
                stage, env, stage_choice, env_choice
            ),
            use_cache=False,
        )
    if not has("spar_help_examples"):
        OmegaConf.register_new_resolver(
            "spar_help_examples",
            lambda stage=None, env=None, stage_choice=None, env_choice=None: render_examples(
                stage, env, stage_choice, env_choice
            ),
            use_cache=False,
        )
    if not has("spar_help_tree"):
        OmegaConf.register_new_resolver(
            "spar_help_tree",
            lambda stage=None, env=None, stage_choice=None, env_choice=None: render_group_tree(
                stage, env, stage_choice, env_choice
            ),
            use_cache=False,
        )
    if not has("spar_help_experiments"):
        OmegaConf.register_new_resolver(
            "spar_help_experiments",
            lambda env=None, env_choice=None: render_experiments(env, env_choice),
            use_cache=False,
        )

    # Lower section panels
    if not has("spar_help_resolved_defaults"):
        OmegaConf.register_new_resolver(
            "spar_help_resolved_defaults",
            lambda save_dir=None, debug=None, stage_choice=None, env_block=None: render_resolved_defaults(
                save_dir, debug, stage_choice, env_block
            ),
            use_cache=False,
        )
    if not has("spar_help_common_flags"):
        OmegaConf.register_new_resolver("spar_help_common_flags", render_common_flags, use_cache=False)
    if not has("spar_help_footer"):
        OmegaConf.register_new_resolver("spar_help_footer", render_footer, use_cache=False)
    if not has("spar_help_header"):
        OmegaConf.register_new_resolver("spar_help_header", render_header, use_cache=False)


def setup_seed(cfg: SPARConfig) -> None:
    """Seed all relevant RNGs and enforce deterministic kernels when requested."""
    if (seed := getattr(cfg, "seed", None)) is None:
        return

    torch = importlib.import_module("torch")

    # Seed Python, NumPy, and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # if not cfg.deterministic:
    #     torch.use_deterministic_algorithms(True, warn_only=True)
    #     torch.backends.cudnn.benchmark = False


def _keep(x: Pruned) -> bool:
    # empty containers (dict/list/tuple/set) are falsey
    if isinstance(x, (dict, list, tuple, set)):
        return bool(x)
    return True


def prune_none(
    params: Pruned | Mapping[str, Pruned] | list[Pruned] | tuple[Pruned, ...] | set[Pruned],
    *,
    return_dict: bool = False,
) -> Pruned:
    """Recursively remove all None values and any containers that become empty.

    Goes to the deepest level of nesting across dataclasses, dicts, lists,
    tuples, and sets, in any combination.

    Args:
        params: Supported container or atomic value.
        return_dict: If true, return built-in dict and list types. If false, rebuild the original types.

    Returns:
        A pruned version of 'params', matching its original type or built-in containers.
    """
    # Helper to test whether a cleaned value should be kept

    if return_dict:
        # Mapping -> dict
        if isinstance(params, Mapping):
            result_dict: dict[str, Pruned] = {}
            for k, v in params.items():
                cleaned: Pruned = prune_none(v, return_dict=True)
                if _keep(cleaned):
                    # keys from OmegaConf DictConfig can be non-str, coerce to str
                    result_dict[k] = cleaned
            return result_dict

        lst: list[Pruned]
        # Sequence (list/tuple) -> list
        if isinstance(params, (list, tuple)):
            lst = [prune_none(v, return_dict=True) for v in params]
            return [v for v in lst if _keep(v)]

        # Set -> list
        if isinstance(params, set):
            lst = [prune_none(v, return_dict=True) for v in params]
            return [v for v in lst if _keep(v)]

        # Atomic
        return params

    # Non-return_dict flow: reconstruct same container shapes when possible
    if isinstance(params, Mapping):
        result_map: dict[str, Pruned] = {}
        for k, v in params.items():
            cleaned = prune_none(v, return_dict=False)
            if _keep(cleaned):
                result_map[k] = cleaned
        return result_map

    if isinstance(params, (list, tuple)):
        lst = [prune_none(v, return_dict=False) for v in params]
        filtered: list[Pruned] = [v for v in lst if _keep(v)]
        return tuple(filtered) if isinstance(params, tuple) else filtered

    if isinstance(params, set):
        lst = [prune_none(v, return_dict=False) for v in params]
        return [v for v in lst if _keep(v)]

    return params


def save_resolved_config(cfg: DictConfig | ListConfig, *, filename: str = "resolved_config.yaml") -> None:
    """Save a fully-resolved YAML copy of the running Hydra config next to Hydra's raw config files.

    The function attempts to locate the Hydra run directory from the provided config (expects a
    `hydra.run.dir` value after interpolation). If unavailable it falls back to the current working
    directory. It writes the resolved YAML both at the run root and inside the run's `.hydra/` folder
    so it's colocated with Hydra's own saved config files.

    Args:
        cfg: The OmegaConf DictConfig (or any object accepted by OmegaConf.to_container).
        filename: The file name to write (defaults to `resolved_config.yaml`).
    """
    # Resolve interpolations and convert to YAML
    resolved_yaml: str = OmegaConf.to_yaml(cfg, resolve=True)

    run_dir_str: str | None = None
    if HydraConfig.initialized():
        run_dir_str = HydraConfig.get().runtime.output_dir

    # Prefer hydra.run.dir and fall back to the current directory.
    run_dir: Path = Path(run_dir_str).resolve() if run_dir_str else Path(Path.cwd())
    run_dir.mkdir(parents=True, exist_ok=True)

    # Also write inside .hydra alongside Hydra's saved raw config files
    hydra_dir: Path = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    hydra_target: Path = hydra_dir / filename
    hydra_target.write_text(resolved_yaml, encoding="utf-8")
