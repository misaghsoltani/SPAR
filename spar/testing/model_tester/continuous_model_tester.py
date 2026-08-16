"""Evaluate continuous world models on rendered state variations."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, TypedDict

import torch
import torch.nn.functional as F

from spar.testing.model_tester.base_tester import ModelTesterBase

if TYPE_CHECKING:
    from logging import Logger

    from torch import Tensor


logger: Logger = getLogger(__name__)


# Type-specific accumulator structure for continuous models
class ContinuousAccumulatorDict(TypedDict):
    """Serialized accumulator values for continuous metrics."""

    cosine_similarity_sum: float
    cosine_similarity_squared_sum: float
    l1_distance_sum: float
    l1_distance_squared_sum: float
    relative_error_sum: float
    relative_error_squared_sum: float


class ContinuousAccumulators:
    """Accumulator for continuous model metrics."""

    __slots__: tuple[str, ...] = (
        "cosine_similarity_squared_sum",
        "cosine_similarity_sum",
        "l1_distance_squared_sum",
        "l1_distance_sum",
        "relative_error_squared_sum",
        "relative_error_sum",
    )

    def __init__(self) -> None:
        # Initialize every float accumulator before the evaluation loop.
        self.cosine_similarity_sum: float = 0.0
        self.cosine_similarity_squared_sum: float = 0.0
        self.l1_distance_sum: float = 0.0
        self.l1_distance_squared_sum: float = 0.0
        self.relative_error_sum: float = 0.0
        self.relative_error_squared_sum: float = 0.0

    def to_dict(self) -> ContinuousAccumulatorDict:
        """Convert to dictionary format expected by base class."""
        return {
            "cosine_similarity_sum": self.cosine_similarity_sum,
            "cosine_similarity_squared_sum": self.cosine_similarity_squared_sum,
            "l1_distance_sum": self.l1_distance_sum,
            "l1_distance_squared_sum": self.l1_distance_squared_sum,
            "relative_error_sum": self.relative_error_sum,
            "relative_error_squared_sum": self.relative_error_squared_sum,
        }

    @classmethod
    def from_dict(cls, data: ContinuousAccumulatorDict) -> ContinuousAccumulators:
        """Create from dictionary format."""
        instance = cls()
        instance.cosine_similarity_sum = data["cosine_similarity_sum"]
        instance.cosine_similarity_squared_sum = data["cosine_similarity_squared_sum"]
        instance.l1_distance_sum = data["l1_distance_sum"]
        instance.l1_distance_squared_sum = data["l1_distance_squared_sum"]
        instance.relative_error_sum = data["relative_error_sum"]
        instance.relative_error_squared_sum = data["relative_error_squared_sum"]
        return instance


class ContinuousModelTester(ModelTesterBase[ContinuousAccumulatorDict]):
    """Evaluate continuous encodings, transitions, and reconstructions."""

    @staticmethod
    def _preprocess_encoding(encoding: Tensor) -> Tensor:
        """Return a continuous encoding unchanged.

        Args:
            encoding: Continuous encoding tensor.

        Returns:
            The input tensor.
        """
        return encoding

    @staticmethod
    def _should_apply_highlighting(state_encoding_pred: Tensor, target_state_encoding: Tensor) -> bool:
        """Highlight differences in continuous models."""
        return bool(F.cosine_similarity(state_encoding_pred, target_state_encoding).mean() < 0.999)

    @staticmethod
    def _compute_model_specific_step_metrics(pred_encoding: Tensor, target_encoding: Tensor) -> dict[str, float]:
        """Compute continuous-specific metrics for a single step."""
        # Calculate cosine similarity
        cosine_similarity = F.cosine_similarity(pred_encoding, target_encoding, dim=1).mean().item()

        # Calculate L1 distance
        l1_distance = torch.mean(torch.abs(pred_encoding - target_encoding)).item()

        # Calculate relative error
        relative_error = torch.mean(
            torch.abs(pred_encoding - target_encoding) / (torch.abs(target_encoding) + 1e-8)
        ).item()

        return {"cosine_similarity": cosine_similarity, "l1_distance": l1_distance, "relative_error": relative_error}

    @staticmethod
    def _initialize_model_specific_accumulators() -> ContinuousAccumulatorDict:
        """Initialize continuous model-specific metric accumulators."""
        accumulator = ContinuousAccumulators()
        return accumulator.to_dict()

    @staticmethod
    def _accumulate_model_specific_metrics(
        step_metrics: dict[str, float], model_specific_accumulators: ContinuousAccumulatorDict
    ) -> None:
        """Accumulate continuous-specific metrics for a single step."""
        # Convert to type-safe accumulator
        accumulator = ContinuousAccumulators.from_dict(model_specific_accumulators)

        cosine_sim = step_metrics.get("cosine_similarity", 0.0)
        l1_distance = step_metrics.get("l1_distance", 0.0)
        relative_error = step_metrics.get("relative_error", 0.0)

        # Accumulate sums and squared sums with type safety
        accumulator.cosine_similarity_sum += cosine_sim
        accumulator.cosine_similarity_squared_sum += cosine_sim**2
        accumulator.l1_distance_sum += l1_distance
        accumulator.l1_distance_squared_sum += l1_distance**2
        accumulator.relative_error_sum += relative_error
        accumulator.relative_error_squared_sum += relative_error**2

        # Update the original dictionary
        model_specific_accumulators.update(accumulator.to_dict())

    @staticmethod
    def _finalize_model_specific_metrics(
        model_specific_accumulators: ContinuousAccumulatorDict, num_steps: int
    ) -> dict[str, float]:
        """Finalize and calculate mean/std for continuous-specific metrics."""
        # Convert to type-safe accumulator
        accumulator = ContinuousAccumulators.from_dict(model_specific_accumulators)

        def compute_mean_std(sum_val: float, squared_sum_val: float, num_steps: int) -> tuple[float, float]:
            mean = sum_val / num_steps
            variance = max(0.0, (squared_sum_val / num_steps) - mean**2)
            return mean, float(variance**0.5)

        # Calculate means and standard deviations with type safety
        cosine_sim_mean, cosine_sim_std = compute_mean_std(
            accumulator.cosine_similarity_sum, accumulator.cosine_similarity_squared_sum, num_steps
        )
        l1_dist_mean, l1_dist_std = compute_mean_std(
            accumulator.l1_distance_sum, accumulator.l1_distance_squared_sum, num_steps
        )
        rel_error_mean, rel_error_std = compute_mean_std(
            accumulator.relative_error_sum, accumulator.relative_error_squared_sum, num_steps
        )

        return {
            "cosine_similarity_mean": cosine_sim_mean,
            "cosine_similarity_std": cosine_sim_std,
            "l1_distance_mean": l1_dist_mean,
            "l1_distance_std": l1_dist_std,
            "relative_error_mean": rel_error_mean,
            "relative_error_std": rel_error_std,
        }

    @staticmethod
    def _calculate_episode_metrics(pred_encoding: Tensor, target_encoding: Tensor, batch_size: int) -> list[float]:
        """Calculate cosine similarity metrics for each episode in the batch."""
        episode_metric_values: list[float] = []

        for episode_idx in range(batch_size):
            episode_encoding_pred = pred_encoding[episode_idx]
            episode_encoding_target = target_encoding[episode_idx]

            # For continuous models, use cosine similarity
            episode_metric = F.cosine_similarity(
                episode_encoding_pred.unsqueeze(0), episode_encoding_target.unsqueeze(0), dim=-1
            ).item()
            episode_metric_values.append(episode_metric)

        return episode_metric_values
