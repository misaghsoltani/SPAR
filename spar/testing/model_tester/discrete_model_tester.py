"""Evaluate discrete world models on rendered state variations."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, TypedDict

import torch

from spar.testing.model_tester.base_tester import ModelTesterBase
from spar.utils.viz_utils.percentage_formatting import round_percentages

if TYPE_CHECKING:
    from logging import Logger

    from torch import Tensor


# Type-specific accumulator structure for discrete models
class DiscreteAccumulatorDict(TypedDict):
    """Serialized accumulator values for discrete metrics."""

    percent_on_sum: float
    percent_on_squared_sum: float
    eq_sum: float
    eq_squared_sum: float
    eq_bit_sum: float
    eq_bit_squared_sum: float
    eq_bit_min_sum: float
    eq_bit_min_squared_sum: float
    eq_bit_min_values: list[float]


class DiscreteAccumulators:
    """Accumulator for discrete model metrics."""

    __slots__: tuple[str, ...] = (
        "eq_bit_min_squared_sum",
        "eq_bit_min_sum",
        "eq_bit_min_values",
        "eq_bit_squared_sum",
        "eq_bit_sum",
        "eq_squared_sum",
        "eq_sum",
        "percent_on_squared_sum",
        "percent_on_sum",
    )

    def __init__(self) -> None:
        # Initialize every float accumulator before the evaluation loop.
        self.percent_on_sum: float = 0.0
        self.percent_on_squared_sum: float = 0.0
        self.eq_sum: float = 0.0
        self.eq_squared_sum: float = 0.0
        self.eq_bit_sum: float = 0.0
        self.eq_bit_squared_sum: float = 0.0
        self.eq_bit_min_sum: float = 0.0
        self.eq_bit_min_squared_sum: float = 0.0
        # Only keep list for values that need individual tracking
        self.eq_bit_min_values: list[float] = []

    def to_dict(self) -> DiscreteAccumulatorDict:
        """Convert to dictionary format expected by base class."""
        return {
            "percent_on_sum": self.percent_on_sum,
            "percent_on_squared_sum": self.percent_on_squared_sum,
            "eq_sum": self.eq_sum,
            "eq_squared_sum": self.eq_squared_sum,
            "eq_bit_sum": self.eq_bit_sum,
            "eq_bit_squared_sum": self.eq_bit_squared_sum,
            "eq_bit_min_sum": self.eq_bit_min_sum,
            "eq_bit_min_squared_sum": self.eq_bit_min_squared_sum,
            "eq_bit_min_values": self.eq_bit_min_values,
        }

    @classmethod
    def from_dict(cls, data: DiscreteAccumulatorDict) -> DiscreteAccumulators:
        """Create from dictionary format."""
        instance: DiscreteAccumulators = cls()
        instance.percent_on_sum = data["percent_on_sum"]
        instance.percent_on_squared_sum = data["percent_on_squared_sum"]
        instance.eq_sum = data["eq_sum"]
        instance.eq_squared_sum = data["eq_squared_sum"]
        instance.eq_bit_sum = data["eq_bit_sum"]
        instance.eq_bit_squared_sum = data["eq_bit_squared_sum"]
        instance.eq_bit_min_sum = data["eq_bit_min_sum"]
        instance.eq_bit_min_squared_sum = data["eq_bit_min_squared_sum"]
        instance.eq_bit_min_values = list(data["eq_bit_min_values"])
        return instance


logger: Logger = getLogger(__name__)


class DiscreteModelTester(ModelTesterBase[DiscreteAccumulatorDict]):
    """Tester for discrete world models with all discrete-specific functionality."""

    @staticmethod
    def _preprocess_encoding(encoding: Tensor) -> Tensor:
        """Apply rounding for discrete encodings."""
        return torch.round(encoding)

    @staticmethod
    def _should_apply_highlighting(state_encoding_pred: Tensor, target_state_encoding: Tensor) -> bool:
        return not torch.equal(state_encoding_pred, target_state_encoding)

    @staticmethod
    def _compute_model_specific_step_metrics(pred_encoding: Tensor, target_encoding: Tensor) -> dict[str, float]:
        """Compute discrete-specific metrics for a single step."""
        # Use rounded predictions for discrete metrics
        pred_rounded: Tensor = torch.round(pred_encoding)
        target_rounded: Tensor = torch.round(target_encoding)

        # Calculate percent_on: percentage of predicted bits that are "on" (>= 0.5)
        percent_on_raw: float = 100 * torch.mean((pred_rounded >= 0.5).float()).item()
        percent_on: float = round_percentages([percent_on_raw])[0]

        # Calculate eq_bits: element-wise equality
        eq_bits: Tensor = pred_rounded == target_rounded

        # Calculate eq: percentage of samples where all bits match exactly
        eq_raw: float = 100 * torch.all(eq_bits, dim=1).float().mean().item()
        eq = round_percentages([eq_raw])[0]

        # Calculate eq_bit: percentage of individual bits that match
        eq_bit_raw: float = 100 * eq_bits.float().mean().item()
        eq_bit = round_percentages([eq_bit_raw])[0]

        # Calculate eq_bit_min: minimum bit-wise accuracy across samples
        eq_bit_min_raw: float = 100 * eq_bits.float().mean(dim=1).min().item()
        eq_bit_min = round_percentages([eq_bit_min_raw])[0]

        return {"percent_on": percent_on, "eq": eq, "eq_bit": eq_bit, "eq_bit_min": eq_bit_min}

    @staticmethod
    def _initialize_model_specific_accumulators() -> DiscreteAccumulatorDict:
        """Initialize discrete model-specific metric accumulators."""
        accumulator = DiscreteAccumulators()
        return accumulator.to_dict()

    @staticmethod
    def _accumulate_model_specific_metrics(
        step_metrics: dict[str, float], model_specific_accumulators: DiscreteAccumulatorDict
    ) -> None:
        """Accumulate discrete-specific metrics for a single step."""
        # Convert to type-safe accumulator
        accumulator = DiscreteAccumulators.from_dict(model_specific_accumulators)

        percent_on = step_metrics.get("percent_on", 0.0)
        eq = step_metrics.get("eq", 0.0)
        eq_bit = step_metrics.get("eq_bit", 0.0)
        eq_bit_min = step_metrics.get("eq_bit_min", 0.0)

        # Accumulate sums and squared sums with type safety
        accumulator.percent_on_sum += percent_on
        accumulator.percent_on_squared_sum += percent_on**2
        accumulator.eq_sum += eq
        accumulator.eq_squared_sum += eq**2
        accumulator.eq_bit_sum += eq_bit
        accumulator.eq_bit_squared_sum += eq_bit**2
        accumulator.eq_bit_min_sum += eq_bit_min
        accumulator.eq_bit_min_squared_sum += eq_bit_min**2

        # Append to eq_bit_min_values to track the minimum bit equality over all steps
        accumulator.eq_bit_min_values.append(eq_bit_min)

        # Update the original dictionary
        model_specific_accumulators.update(accumulator.to_dict())

    @staticmethod
    def _finalize_model_specific_metrics(
        model_specific_accumulators: DiscreteAccumulatorDict, num_steps: int
    ) -> dict[str, float]:
        """Finalize and calculate mean/std for discrete-specific metrics."""
        # Convert to type-safe accumulator
        accumulator: DiscreteAccumulators = DiscreteAccumulators.from_dict(model_specific_accumulators)

        def compute_mean_std(sum_val: float, squared_sum_val: float, num_steps: int) -> tuple[float, float]:
            mean = sum_val / num_steps
            variance = max(0.0, (squared_sum_val / num_steps) - mean**2)
            return mean, float(variance**0.5)

        # Calculate means and standard deviations with type safety
        percent_on_mean, percent_on_std = compute_mean_std(
            accumulator.percent_on_sum, accumulator.percent_on_squared_sum, num_steps
        )
        eq_mean, eq_std = compute_mean_std(accumulator.eq_sum, accumulator.eq_squared_sum, num_steps)
        eq_bit_mean, eq_bit_std = compute_mean_std(accumulator.eq_bit_sum, accumulator.eq_bit_squared_sum, num_steps)
        eq_bit_min_mean, _ = compute_mean_std(accumulator.eq_bit_min_sum, accumulator.eq_bit_min_squared_sum, num_steps)

        # Calculate minimum eq_bit_min across all steps
        eq_bit_min_min = min(accumulator.eq_bit_min_values) if accumulator.eq_bit_min_values else 0.0

        return {
            "percent_on_mean": percent_on_mean,
            "percent_on_std": percent_on_std,
            "eq_mean": eq_mean,
            "eq_std": eq_std,
            "eq_bit_mean": eq_bit_mean,
            "eq_bit_std": eq_bit_std,
            "eq_bit_min_mean": eq_bit_min_mean,
            "eq_bit_min_min": eq_bit_min_min,
        }

    @staticmethod
    def _calculate_episode_metrics(pred_encoding: Tensor, target_encoding: Tensor, batch_size: int) -> list[float]:
        """Calculate eq_bit_min metrics for each episode in the batch."""
        episode_metric_values: list[float] = []

        for episode_idx in range(batch_size):
            episode_encoding_pred: Tensor = pred_encoding[episode_idx]
            episode_encoding_target: Tensor = target_encoding[episode_idx]

            # For discrete models, use eq_bit_min (as percentage)
            eq_bits: Tensor = episode_encoding_pred == episode_encoding_target
            episode_metric: float = 100 * eq_bits.float().mean().item()
            episode_metric_values.append(episode_metric)

        return episode_metric_values
