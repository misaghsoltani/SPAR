"""Search-space parsing and sampling helpers for SPAR Optuna studies."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from decimal import Decimal
from math import isfinite
import re
from typing import TYPE_CHECKING

from spar.utils.config_utils.config_schema import ParameterSpec

from .compat import convert_wandb_sweep_parameters
from .path_utils import get_path_value, sanitize_param_name, scoped_step_path

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.trial import Trial

    from spar.utils.config_utils.config_schema import OptunaConfig, WorkflowStep

    from .types import PathValue, SampledValue


_INLINE_SAMPLER_RE: re.Pattern[str] = re.compile(r"^\$\{(range|irange|choice):(.+)\}$")


def _parse_scalar(token: str) -> SampledValue:
    token = token.strip()
    if token in {"true", "True"}:
        return True
    if token in {"false", "False"}:
        return False
    if token in {"null", "None"}:
        return None
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _split_args(raw: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth: int = 0
    quote: str | None = None
    for char in raw:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in {"[", "(", "{"}:
            depth += 1
            current.append(char)
            continue
        if char in {"]", ")", "}"}:
            depth = max(0, depth - 1)
            current.append(char)
            continue
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part]


def _parse_choice_args(raw: str) -> list[SampledValue]:
    stripped: str = raw.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed: list[SampledValue] | None = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            values: list[SampledValue] = []
            for value in parsed:
                if not isinstance(value, bool | int | float | str) and value is not None:
                    raise TypeError(f"Unsupported inline choice value type: {type(value).__name__}")
                values.append(value)
            return values
    return [_parse_scalar(part) for part in _split_args(raw)]


def _parse_inline_sampler(path: str, raw_value: str, *, step_name: str) -> ParameterSpec | None:
    match: re.Match[str] | None = _INLINE_SAMPLER_RE.match(raw_value.strip())
    if match is None:
        return None
    kind_name: str = match.group(1)
    payload: str = match.group(2).strip()

    if kind_name == "range":
        low_raw, high_raw = _split_args(payload)
        return ParameterSpec(path=path, step=step_name, kind="float", low=float(low_raw), high=float(high_raw))
    if kind_name == "irange":
        low_raw, high_raw = _split_args(payload)
        return ParameterSpec(path=path, step=step_name, kind="int", low=int(float(low_raw)), high=int(float(high_raw)))
    if kind_name == "choice":
        choices: list[SampledValue] = _parse_choice_args(payload)
        if choices and all(isinstance(choice, bool) for choice in choices):
            return ParameterSpec(path=path, step=step_name, kind="bool", choices=choices)
        return ParameterSpec(path=path, step=step_name, kind="categorical", choices=choices)
    return None


def _walk_inline_specs(value: PathValue, *, prefix: str, step_name: str, specs: list[ParameterSpec]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_prefix: str = f"{prefix}.{key}" if prefix else str(key)
            _walk_inline_specs(item, prefix=child_prefix, step_name=step_name, specs=specs)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            _walk_inline_specs(item, prefix=child_prefix, step_name=step_name, specs=specs)
        return
    if isinstance(value, str):
        parsed_spec: ParameterSpec | None = _parse_inline_sampler(prefix, value, step_name=step_name)
        if parsed_spec is not None:
            specs.append(parsed_spec)


def scan_inline_parameter_specs(
    raw_step_cfg: Mapping[str, PathValue], *, step_name: str, scan_paths: list[str], ignored_paths: set[str]
) -> list[ParameterSpec]:
    """Scan configured subtrees for inline ``${range/...}`` sampler declarations."""
    specs: list[ParameterSpec] = []
    for scan_path in scan_paths:
        subtree: PathValue = get_path_value(raw_step_cfg, scan_path)
        if subtree is None:
            continue
        _walk_inline_specs(subtree, prefix=scan_path, step_name=step_name, specs=specs)
    return [spec for spec in specs if spec.path not in ignored_paths]


def _parameter_applies_to_step(spec: ParameterSpec, *, step_name: str, default_step_name: str) -> bool:
    target_step: str = spec.step or default_step_name
    return target_step == step_name


def _parameter_name(spec: ParameterSpec, *, step_name: str, default_step_name: str, multi_step: bool) -> str:
    base_name: str = spec.name or spec.path
    if multi_step and spec.step is None:
        base_name = f"{step_name}:{base_name}"
    elif spec.step is not None and (multi_step or spec.step != default_step_name):
        base_name = f"{spec.step}:{base_name}"
    return sanitize_param_name(base_name)


def _validate_grid_values(name: str, values: Sequence[SampledValue]) -> list[SampledValue]:
    """Validate scalar values used by an Optuna grid dimension.

    Args:
        name: Optuna parameter name.
        values: Candidate values for the parameter.

    Returns:
        A validated list that preserves the configured order.

    Raises:
        ValueError: If the values are empty, duplicated, or non-finite.
        TypeError: If a value cannot be persisted by Optuna.
    """
    if not values:
        raise ValueError(f"Grid parameter {name!r} requires at least one value")

    validated: list[SampledValue] = []
    seen: set[tuple[type[SampledValue], SampledValue]] = set()
    for value in values:
        if not isinstance(value, bool | int | float | str) and value is not None:
            raise TypeError(f"Grid parameter {name!r} has unsupported value type {type(value).__name__}")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"Grid parameter {name!r} contains a non-finite float")
        identity = (type(value), value)
        if identity in seen:
            raise ValueError(f"Grid parameter {name!r} contains duplicate value {value!r}")
        seen.add(identity)
        validated.append(value)
    return validated


def _finite_grid_values(name: str, spec: ParameterSpec) -> list[SampledValue] | None:
    """Expand one parameter specification into finite grid values.

    Args:
        name: Optuna parameter name.
        spec: Parameter specification to expand.

    Returns:
        Finite values for a sampled parameter, or ``None`` for a fixed value.

    Raises:
        ValueError: If the specification is not finite or is invalid.
        TypeError: If a categorical value cannot be persisted by Optuna.
    """
    if spec.kind == "fixed":
        return None

    if spec.when_all:
        raise ValueError(
            f"Grid sampler cannot exactly represent conditional parameter {name!r} because when_all is set"
        )

    if spec.kind == "categorical":
        if spec.choices is None:
            raise ValueError(f"Categorical grid parameter {name!r} requires choices")
        return _validate_grid_values(name, spec.choices)

    if spec.kind == "bool":
        choices: Sequence[SampledValue] = spec.choices if spec.choices is not None else [False, True]
        if any(not isinstance(choice, bool) for choice in choices):
            raise TypeError(f"Boolean grid parameter {name!r} requires boolean choices")
        return _validate_grid_values(name, choices)

    if spec.low is None or spec.high is None:
        raise ValueError(f"Grid parameter {name!r} requires low and high bounds")

    if spec.kind == "int":
        low = int(spec.low)
        high = int(spec.high)
        step = int(spec.step_size) if spec.step_size is not None else 1
        if low > high:
            raise ValueError(f"Integer grid parameter {name!r} requires low less than or equal to high")
        if step <= 0:
            raise ValueError(f"Integer grid parameter {name!r} requires a positive step_size")
        if spec.log and step != 1:
            raise ValueError(f"Log-scaled integer grid parameter {name!r} requires step_size equal to 1")
        return list(range(low, high + 1, step))

    if spec.kind == "float":
        if spec.log or spec.step_size is None:
            raise ValueError(
                f"Grid sampler requires a finite value list for float parameter {name!r}. "
                "Set a positive step_size without log scaling or use categorical choices"
            )
        low_decimal = Decimal(str(spec.low))
        high_decimal = Decimal(str(spec.high))
        step_decimal = Decimal(str(spec.step_size))
        if not all(isfinite(float(value)) for value in (low_decimal, high_decimal, step_decimal)):
            raise ValueError(f"Float grid parameter {name!r} requires finite bounds and step_size")
        if low_decimal > high_decimal:
            raise ValueError(f"Float grid parameter {name!r} requires low less than or equal to high")
        if step_decimal <= 0:
            raise ValueError(f"Float grid parameter {name!r} requires a positive step_size")
        num_steps = int((high_decimal - low_decimal) // step_decimal)
        values = [float(low_decimal + step_decimal * index) for index in range(num_steps + 1)]
        return _validate_grid_values(name, values)

    raise ValueError(f"Unsupported grid parameter kind: {spec.kind}")


def build_grid_search_space(
    parameter_specs_by_step: Mapping[str, Sequence[ParameterSpec]], *, default_step_name: str, multi_step: bool
) -> dict[str, list[SampledValue]]:
    """Build an exact finite Optuna grid from workflow parameter specs.

    Args:
        parameter_specs_by_step: Parameter specifications grouped by workflow step.
        default_step_name: Name used for unscoped single-step parameters.
        multi_step: Whether the workflow contains multiple steps.

    Returns:
        Parameter names mapped to their finite candidate values.

    Raises:
        ValueError: If the configured space cannot be represented exactly as a grid.
    """
    search_space: dict[str, list[SampledValue]] = {}
    for step_name, parameter_specs in parameter_specs_by_step.items():
        for spec in parameter_specs:
            name = _parameter_name(
                spec, step_name=step_name, default_step_name=default_step_name, multi_step=multi_step
            )
            values = _finite_grid_values(name, spec)
            if values is None:
                continue
            previous_values = search_space.get(name)
            if previous_values is not None and previous_values != values:
                raise ValueError(f"Grid parameter {name!r} is defined with conflicting value lists")
            search_space[name] = values

    if not search_space:
        raise ValueError("Grid sampler requires at least one finite non-fixed parameter")
    return search_space


def _condition_matches(
    conditions: Mapping[str, SampledValue],
    *,
    current_step: str,
    sampled_values_by_step: Mapping[str, Mapping[str, SampledValue]],
    base_cfgs_by_step: Mapping[str, Mapping[str, PathValue]],
) -> bool:
    for raw_ref, expected in conditions.items():
        scope_step, scope_path = scoped_step_path(raw_ref, current_step)
        if scope_step is None:
            return False
        actual: PathValue | SampledValue | None = sampled_values_by_step.get(scope_step, {}).get(scope_path)
        if actual is None and scope_path not in sampled_values_by_step.get(scope_step, {}):
            actual = get_path_value(base_cfgs_by_step.get(scope_step, {}), scope_path)
        if actual != expected:
            return False
    return True


def _suggest_value(trial: Trial, name: str, spec: ParameterSpec) -> SampledValue:
    step: float | None
    if spec.kind == "float":
        if spec.low is None or spec.high is None:
            raise ValueError(f"Float parameter {name} requires low/high")
        step = float(spec.step_size) if spec.step_size is not None else None
        return trial.suggest_float(name, float(spec.low), float(spec.high), log=spec.log, step=step)
    if spec.kind == "int":
        if spec.low is None or spec.high is None:
            raise ValueError(f"Int parameter {name} requires low/high")
        step = int(spec.step_size) if spec.step_size is not None else 1
        return trial.suggest_int(name, int(spec.low), int(spec.high), log=spec.log, step=step)
    if spec.kind == "categorical":
        if not spec.choices:
            raise ValueError(f"Categorical parameter {name} requires choices")
        values: list[SampledValue] = []
        for choice in spec.choices:
            if not isinstance(choice, bool | int | float | str) and choice is not None:
                raise TypeError(f"Categorical parameter {name} has unsupported choice type {type(choice).__name__}")
            values.append(choice)
        return trial.suggest_categorical(name, values)
    if spec.kind == "bool":
        choices: list[bool] = [False, True]
        if spec.choices is not None:
            choices = []
            for choice in spec.choices:
                if not isinstance(choice, bool):
                    raise TypeError(f"Boolean parameter {name} requires boolean choices")
                choices.append(choice)
        return trial.suggest_categorical(name, choices)
    if spec.kind == "fixed":
        if not isinstance(spec.value, bool | int | float | str) and spec.value is not None:
            raise TypeError(f"Fixed parameter {name} has unsupported value type {type(spec.value).__name__}")
        return spec.value
    raise ValueError(f"Unsupported parameter kind: {spec.kind}")


def collect_step_parameter_specs(
    optuna_cfg: OptunaConfig, *, step: WorkflowStep, raw_step_cfg: Mapping[str, PathValue], default_step_name: str
) -> tuple[list[ParameterSpec], list[str]]:
    """Merge explicit, converted, and scanned parameter specs for one workflow step."""
    warnings: list[str] = []
    explicit_specs: list[ParameterSpec] = [
        spec
        for spec in optuna_cfg.parameters
        if _parameter_applies_to_step(spec, step_name=step.name, default_step_name=default_step_name)
    ]
    explicit_paths: set[str] = {spec.path for spec in explicit_specs}

    imported_specs: list[ParameterSpec] = []
    if optuna_cfg.import_sweep_parameters and optuna_cfg.sweep_parameters:
        imported_specs, imported_warnings = convert_wandb_sweep_parameters(
            optuna_cfg.sweep_parameters, default_step=step.name, raw_step_cfg=raw_step_cfg
        )
        warnings.extend(imported_warnings)

    imported_specs = [spec for spec in imported_specs if spec.path not in explicit_paths]
    imported_paths: set[str] = explicit_paths | {spec.path for spec in imported_specs}

    scan_paths: list[str] = list(step.scan_paths) if step.scan_paths else list(optuna_cfg.scan_paths)
    scanned_specs: list[ParameterSpec] = scan_inline_parameter_specs(
        raw_step_cfg, step_name=step.name, scan_paths=scan_paths, ignored_paths=imported_paths
    )

    return explicit_specs + imported_specs + scanned_specs, warnings


def sample_step_parameters(
    trial: Trial,
    *,
    step: WorkflowStep,
    parameter_specs: list[ParameterSpec],
    default_step_name: str,
    sampled_values_by_step: Mapping[str, Mapping[str, SampledValue]],
    base_cfgs_by_step: Mapping[str, Mapping[str, PathValue]],
    multi_step: bool,
) -> tuple[dict[str, SampledValue], dict[str, SampledValue]]:
    """Sample and return path/value plus parameter-name/value mappings for one step."""
    sampled_by_path: dict[str, SampledValue] = {}
    named_values: dict[str, SampledValue] = {}

    for spec in parameter_specs:
        if spec.when_all and not _condition_matches(
            spec.when_all,
            current_step=step.name,
            sampled_values_by_step=sampled_values_by_step,
            base_cfgs_by_step=base_cfgs_by_step,
        ):
            continue

        param_name: str = _parameter_name(
            spec, step_name=step.name, default_step_name=default_step_name, multi_step=multi_step
        )
        value: SampledValue = _suggest_value(trial, param_name, spec)
        sampled_by_path[spec.path] = value
        named_values[param_name] = value

    return sampled_by_path, named_values
