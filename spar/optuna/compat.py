"""Compatibility helpers between existing SPAR sweep configs and Optuna."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from spar.utils.config_utils.config_schema import ParameterSpec

from .path_utils import get_path_value

if TYPE_CHECKING:
    from spar.utils.config_utils.config_schema import HydraValue

    from .types import PathValue, SampledValue


def _infer_categorical_kind(values: list[SampledValue]) -> str:
    if values and all(isinstance(value, bool) for value in values):
        return "bool"
    return "categorical"


def convert_wandb_sweep_parameters(
    parameters: Mapping[str, PathValue | HydraValue],
    *,
    default_step: str | None = None,
    raw_step_cfg: Mapping[str, PathValue] | None = None,
) -> tuple[list[ParameterSpec], list[str]]:
    """Best-effort conversion of W&B sweep parameter mappings into Optuna specs."""
    converted: list[ParameterSpec] = []
    warnings: list[str] = []

    for raw_path, raw_spec in parameters.items():
        if raw_step_cfg is not None and get_path_value(raw_step_cfg, raw_path) is None:
            warnings.append(f"Sweep parameter path does not exist in composed config: {raw_path}")
        if not isinstance(raw_spec, Mapping):
            warnings.append(f"Skipped unsupported sweep parameter spec for {raw_path}: {raw_spec!r}")
            continue

        if "values" in raw_spec:
            values_raw = raw_spec["values"]
            if not isinstance(values_raw, list):
                warnings.append(f"Skipped sweep values for {raw_path}: expected list, got {type(values_raw).__name__}")
                continue
            values: list[SampledValue] = []
            invalid_value = False
            for value in values_raw:
                if not isinstance(value, bool | int | float | str) and value is not None:
                    warnings.append(
                        f"Skipped sweep values for {raw_path}: unsupported categorical value type "
                        f"{type(value).__name__}"
                    )
                    invalid_value = True
                    break
                values.append(value)
            if invalid_value:
                continue
            converted.append(
                ParameterSpec(
                    path=raw_path, step=default_step, kind=_infer_categorical_kind(values), choices=list(values)
                )
            )
            continue

        distribution: HydraValue | PathValue | None = raw_spec.get("distribution")
        low: HydraValue | PathValue | None = raw_spec.get("min")
        high: HydraValue | PathValue | None = raw_spec.get("max")
        if (
            isinstance(distribution, str)
            and distribution in {"uniform", "log_uniform", "qlog_uniform"}
            and isinstance(low, int | float)
            and isinstance(high, int | float)
        ):
            converted.append(
                ParameterSpec(
                    path=raw_path,
                    step=default_step,
                    kind="float",
                    low=float(low),
                    high=float(high),
                    log=distribution != "uniform",
                )
            )
            continue

        unsupported_spec: dict[int | str, HydraValue | PathValue] = dict(raw_spec.items())
        warnings.append(f"Skipped unsupported sweep parameter mapping for {raw_path}: {unsupported_spec!r}")

    return converted, warnings
