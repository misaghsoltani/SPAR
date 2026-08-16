"""Shared type aliases for SPAR's Optuna integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import TypeAlias

    import numpy as np
    from numpy.typing import NDArray
    from torch import Tensor, nn


ScalarMetric: TypeAlias = float | int | bool | str | None
SampledValue: TypeAlias = ScalarMetric
TemplateValue: TypeAlias = (
    "ScalarMetric | list[TemplateValue] | tuple[TemplateValue, ...] | Mapping[str, TemplateValue]"
)
PathValue: TypeAlias = "ScalarMetric | list[PathValue] | tuple[PathValue, ...] | Mapping[str | int, PathValue]"
StageLeaf: TypeAlias = "ScalarMetric | Path | bytes | NDArray[np.generic] | Tensor | nn.Module"
StageValue: TypeAlias = "StageLeaf | list[StageValue] | tuple[StageValue, ...] | Mapping[str | int, StageValue]"
ReporterMetricMap: TypeAlias = "Mapping[str, ScalarMetric]"
ReporterValue: TypeAlias = ScalarMetric | ReporterMetricMap
ReporterPayload: TypeAlias = "Mapping[str, ReporterValue]"
