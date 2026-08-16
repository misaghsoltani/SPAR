"""Track and plot reconstruction MSE metrics."""

from __future__ import annotations

from logging import getLogger
import pathlib
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import orjson

if TYPE_CHECKING:
    from logging import Logger


logger: Logger = getLogger(__name__)


class MetricSummary(TypedDict):
    """TypedDict describing summary statistics for a single metric."""

    mean: float
    std: float
    min: float
    max: float


class VariationSummary(TypedDict):
    """TypedDict describing aggregate summary for a variation."""

    episode_count: int
    max_steps: int
    metrics: dict[str, MetricSummary]


class _MetricEpisodePayload(TypedDict):
    episode_num_in_dataset: int | None
    episode_len: int
    values_per_step: list[float]


class _MetricJsonPayload(TypedDict):
    dtype: str
    num_episodes: int
    max_val_overall: float
    min_val_overall: float
    mean_val_overall: float
    min: _MetricEpisodePayload
    max: _MetricEpisodePayload
    mean: _MetricEpisodePayload


class MetricsTracker:
    """Metrics tracking system for model evaluation with structured JSON export."""

    def __init__(self, output_dir: str, metrics_to_save: list[str] | None = None) -> None:
        """Initialize the metrics tracker.

        Args:
            output_dir: Directory to save metrics files
            metrics_to_save: List of metric names to save in JSON format
        """
        self.output_dir = output_dir
        self.metrics_to_save = metrics_to_save or ["reconstruction_mse"]
        pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

        # Initialize data collection containers
        self.reset_variation_data()

        # Global metrics data structure for new JSON format
        self.variation_metrics: dict[str, dict[str, list[list[float]]]] = {}

    def reset_variation_data(self) -> None:
        """Reset data for a new variation - called at start of each variation."""
        self.current_batch_metrics: dict[int, dict[str, float]] = {}
        self.current_variation_name: str | None = None
        self.current_episode_count: int = 0
        self.max_steps_seen: int = 0

    def update_step_metrics(self, step: int, **metrics: float) -> None:
        """Update step metrics on-the-fly with support for multiple metrics.

        Args:
            step: Current step number
            **metrics: Metrics to track (e.g., reconstruction_mse=0.1, eq_bit=95.0)
        """
        self.max_steps_seen = max(self.max_steps_seen, step + 1)

        if step not in self.current_batch_metrics:
            self.current_batch_metrics[step] = {}

        # Store only the metrics we're configured to save
        for metric_name, metric_value in metrics.items():
            if metric_name in self.metrics_to_save:
                self.current_batch_metrics[step][metric_name] = metric_value

    def finalize_episode_batch(self, batch_size: int, variation_name: str) -> None:
        """Process current batch data and store in metrics structure.

        Args:
            batch_size: Number of episodes in current batch
            variation_name: Name of the current variation
        """
        if not self.current_batch_metrics:
            return

        # Initialize variation if not exists
        if variation_name not in self.variation_metrics:
            self.variation_metrics[variation_name] = {}
            # Initialize all metrics we're tracking
            for metric_name in self.metrics_to_save:
                self.variation_metrics[variation_name][metric_name] = []

        # Process each episode in the batch
        for _ in range(batch_size):
            # Build values per step for each metric
            for metric_name in self.metrics_to_save:
                episode_values: list[float] = []
                for step in range(self.max_steps_seen):
                    if step in self.current_batch_metrics and metric_name in self.current_batch_metrics[step]:
                        episode_values.append(self.current_batch_metrics[step][metric_name])
                    else:
                        # Fill missing steps with 0.0 or last known value
                        episode_values.append(0.0)

                # Store episode data for this metric
                self.variation_metrics[variation_name][metric_name].append(episode_values)

        self.current_episode_count += batch_size
        self.current_batch_metrics.clear()

    def save_variation_metrics_to_json(self, variation_name: str, filename: str) -> str:
        """Save metrics for a single variation in the new structured format.

        Args:
            variation_name: Name of the variation to save
            filename: Name of the output JSON file

        Returns:
            Path to the saved file
        """
        output_path = str(pathlib.Path(self.output_dir) / filename)

        if variation_name not in self.variation_metrics:
            logger.warning(f"No metrics data found for variation '{variation_name}'")
            return output_path

        # Build the structured JSON format
        metrics_out: dict[str, _MetricJsonPayload] = {}

        variation_data = self.variation_metrics[variation_name]

        for metric_name in self.metrics_to_save:
            if metric_name not in variation_data or not variation_data[metric_name]:
                continue

            episodes_data = variation_data[metric_name]

            # Calculate statistics across all episodes and steps
            all_values: list[float] = []
            for episode_values in episodes_data:
                all_values.extend(episode_values)

            if not all_values:
                continue

            all_values_array = np.array(all_values)
            overall_mean = float(np.mean(all_values_array))
            overall_min = float(np.min(all_values_array))
            overall_max = float(np.max(all_values_array))

            # Find episodes with min/max overall values
            episode_means = [np.mean(episode_values) for episode_values in episodes_data]
            min_episode_idx = int(np.argmin(episode_means))
            max_episode_idx = int(np.argmax(episode_means))

            # Determine data type (check whether values are integer-valued)
            dtype = "float" if any(isinstance(v, float) and not v.is_integer() for v in all_values[:10]) else "int"

            # Calculate mean values per step across all episodes
            max_steps = max((len(episode_values) for episode_values in episodes_data), default=0)
            mean_values_per_step: list[float] = []
            for step in range(max_steps):
                step_values = [
                    episodes_data[ep][step] for ep in range(len(episodes_data)) if step < len(episodes_data[ep])
                ]
                if step_values:
                    mean_values_per_step.append(float(np.mean(step_values)))
                else:
                    mean_values_per_step.append(0.0)

            # Store the computed summary for this metric.
            metrics_out[metric_name] = {
                "dtype": dtype,
                "num_episodes": len(episodes_data),
                "max_val_overall": overall_max,
                "min_val_overall": overall_min,
                "mean_val_overall": overall_mean,
                "min": {
                    "episode_num_in_dataset": min_episode_idx,
                    "episode_len": len(episodes_data[min_episode_idx]),
                    "values_per_step": episodes_data[min_episode_idx],
                },
                "max": {
                    "episode_num_in_dataset": max_episode_idx,
                    "episode_len": len(episodes_data[max_episode_idx]),
                    "values_per_step": episodes_data[max_episode_idx],
                },
                "mean": {
                    "episode_num_in_dataset": None,  # Mean is calculated, not from specific episode
                    "episode_len": max_steps,
                    "values_per_step": mean_values_per_step,
                },
            }

        structured_data = {"variant_name": variation_name, "metrics": metrics_out}

        json_bytes = orjson.dumps(
            structured_data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY
        )

        # Create the parent directory for a nested output filename.
        parent_dir = pathlib.Path(output_path).parent
        parent_dir.mkdir(exist_ok=True, parents=True)

        pathlib.Path(output_path).write_bytes(json_bytes)

        logger.info(f"Variation '{variation_name}' metrics data saved to {output_path}")
        return output_path

    def get_variation_summary(self, variation_name: str) -> VariationSummary:
        """Get summary statistics for a variation.

        Args:
            variation_name: Name of the variation

        Returns:
            Dictionary with summary statistics
        """
        # Always return a consistent summary structure (no empty/ambiguous types)
        if variation_name not in self.variation_metrics:
            return VariationSummary(episode_count=0, max_steps=0, metrics={})

        variation_data = self.variation_metrics[variation_name]
        summary: VariationSummary = VariationSummary(episode_count=0, max_steps=0, metrics={})

        for metric_name in self.metrics_to_save:
            if variation_data.get(metric_name):
                episodes_data = variation_data[metric_name]
                summary["episode_count"] = len(episodes_data)
                # Determine maximum steps seen across episodes for this metric
                max_steps_for_metric = max((len(ep) for ep in episodes_data), default=0)
                # summary["max_steps"] is an int
                summary["max_steps"] = max(summary["max_steps"], max_steps_for_metric)

                # Calculate summary statistics
                all_values: list[float] = []
                for episode_values in episodes_data:
                    all_values.extend(episode_values)

                if all_values:
                    # Create a mutable metrics mapping when the field is absent.
                    metrics_map = summary["metrics"]
                    metrics_map[metric_name] = {
                        "mean": float(np.mean(all_values)),
                        "std": float(np.std(all_values)),
                        "min": float(np.min(all_values)),
                        "max": float(np.max(all_values)),
                    }
        return summary
