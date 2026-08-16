"""Record per-step and aggregate metrics from model evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from logging import getLogger
import operator
import pathlib
from typing import TYPE_CHECKING, TypedDict
import uuid

import orjson
import torch

from spar.utils.log_utils.wandb_logger import get_active_tracking_session
from spar.utils.viz_utils.percentage_formatting import format_percentage

if TYPE_CHECKING:
    from logging import Logger
    from typing import TypeAlias

    from spar.utils.log_utils.wandb_logger import WandbTrackingSession


logger: Logger = getLogger(__name__)

MetricScalar: TypeAlias = str | int | float | bool | None
MetricValue: TypeAlias = MetricScalar | list["MetricValue"] | Mapping[str, "MetricValue"]
MetricDict: TypeAlias = dict[str, MetricValue]
_NESTED_LOG_METRICS = __import__("spar.utils.log_utils.wandb_logger", fromlist=["log_metrics"]).log_metrics


def _log_nested_metrics(session: WandbTrackingSession, metrics: Mapping[str, MetricValue], *, commit: bool) -> None:
    _NESTED_LOG_METRICS(session, dict(metrics), commit=commit)


class ExecutionTimeline(TypedDict):
    """Timeline fields for one evaluation run."""

    start_timestamp: str
    end_timestamp: str | None
    total_duration_seconds: float | None


class DataStatistics(TypedDict):
    """Dataset processing counters for one evaluation run."""

    total_variations_evaluated: int
    total_episodes_processed: int
    total_batches_processed: int
    total_prediction_steps: int


class PerformanceStatistics(TypedDict):
    """Throughput statistics for one evaluation run."""

    average_batch_processing_time_ms: float | None
    episodes_per_second: float | None
    steps_per_second: float | None


class EvaluationSummary(TypedDict):
    """Summary metadata for one evaluation run."""

    execution_timeline: ExecutionTimeline
    data_statistics: DataStatistics
    performance_statistics: PerformanceStatistics


class EvaluationMetadata(TypedDict):
    """Static metadata captured when an evaluation starts."""

    evaluation_id: str
    timestamp: str
    test_model_type: str
    evaluation_mode: str
    model_architecture: MetricDict
    test_configuration: MetricDict
    system_environment: MetricDict


class VariationMetadata(TypedDict):
    """Counters and timestamps for one variation."""

    variation_name: str
    start_timestamp: str
    end_timestamp: str | None
    duration_seconds: float | None
    total_episodes: int
    total_batches: int
    total_prediction_steps: int


class BatchData(TypedDict):
    """Metrics stored for one evaluated batch."""

    batch_idx: int
    timestamp: str
    episode_indices: list[int]
    batch_size: int
    sequence_length: int
    metrics: Mapping[str, MetricValue]
    step_level_data: Mapping[str, MetricValue]


class VariationData(TypedDict):
    """All logged data for one variation."""

    variation_metadata: VariationMetadata
    batch_data: list[BatchData]
    aggregated_variation_metrics: Mapping[str, MetricValue]


class MetricRankEntry(TypedDict):
    """Ranked metric value for one variation."""

    rank: int
    variation: str
    value: float


class MetricExtreme(TypedDict):
    """Best or worst variation value for one metric."""

    variation: str
    value: float


class MetricBestWorstAnalysis(TypedDict):
    """Best and worst values for one metric."""

    best: MetricExtreme
    worst: MetricExtreme


class CrossVariationAnalytics(TypedDict):
    """Cross-variation comparisons for key metrics."""

    performance_ranking: dict[str, list[MetricRankEntry]]
    best_worst_analysis: dict[str, MetricBestWorstAnalysis]


class CrossVariationAnalyticsMessage(TypedDict):
    """Message stored when cross-variation analytics cannot be computed."""

    message: str


CrossVariationPayload: TypeAlias = CrossVariationAnalytics | CrossVariationAnalyticsMessage


class MetricsData(TypedDict):
    """Complete evaluation metrics document written as JSON."""

    evaluation_metadata: EvaluationMetadata
    evaluation_summary: EvaluationSummary
    variation_data: dict[str, VariationData]
    cross_variation_analytics: CrossVariationPayload
    global_aggregated_metrics: Mapping[str, MetricValue]


def _format_console_metric(value: float, metric_name: str) -> str:
    """Format metric values for console display.

    Args:
        value: The metric value to format
        metric_name: The name of the metric

    Returns:
        Formatted string for console display
    """
    metric_name_lower: str = metric_name.lower()

    # Handle percentage-based metrics
    if any(keyword in metric_name_lower for keyword in ["percent", "eq", "bit"]):
        return format_percentage(value, use_special_rounding=True)

    # Handle MSE and error metrics with scientific notation
    if any(keyword in metric_name_lower for keyword in ["mse", "error"]):
        return f"{value:.2e}" if value < 1e-3 else f"{value:.6f}"

    # Handle similarity metrics
    if "similarity" in metric_name_lower or "cosine" in metric_name_lower:
        return f"{value:.4f}"

    # Handle distance metrics
    if "distance" in metric_name_lower or "l1" in metric_name_lower:
        return f"{value:.2e}" if value < 1e-3 else f"{value:.6f}"

    # Default formatting
    return f"{value:.6f}"


def _metric_float(value: MetricValue | None, default: float) -> float:
    """Return a metric value as a float.

    Args:
        value: Metric value read from a nested metrics payload.
        default: Value to return when `value` is not numeric.

    Returns:
        Numeric metric value as a float.
    """
    return float(value) if isinstance(value, (int, float)) else default


def _metric_int_list(value: MetricValue | None) -> list[int]:
    """Return a metric value as a list of integers.

    Args:
        value: Metric value read from a nested metrics payload.

    Returns:
        Integer list when `value` is a list of integers, otherwise an empty list.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int)]


class MetricsLogger:
    """Record prediction-step and evaluation metrics in a nested document."""

    def __init__(
        self, output_dir: str, model_info: MetricDict, test_config: MetricDict, test_model_type: str, end_to_end: bool
    ) -> None:
        """Initialize the metrics logger.

        Args:
            output_dir: Directory to save metrics file
            model_info: Information about the models being tested
            test_config: Test configuration parameters
            test_model_type: Type of model ("discrete" or "continuous")
            end_to_end: Whether using end-to-end evaluation
        """
        self.output_dir: str = output_dir
        self.test_model_type: str = test_model_type
        self.metrics_file: str = str(pathlib.Path(output_dir) / "evaluation_metrics.json")
        self.evaluation_id: str = str(uuid.uuid4())
        self.start_time: datetime = datetime.now()

        # Initialize structured metrics data
        self.metrics_data: MetricsData = {
            "evaluation_metadata": {
                "evaluation_id": self.evaluation_id,
                "timestamp": self.start_time.isoformat(),
                "test_model_type": test_model_type,
                "evaluation_mode": "end_to_end" if end_to_end else "standard",
                "model_architecture": model_info,
                "test_configuration": test_config,
                "system_environment": self._get_system_environment(),
            },
            "evaluation_summary": {
                "execution_timeline": {
                    "start_timestamp": self.start_time.isoformat(),
                    "end_timestamp": None,
                    "total_duration_seconds": None,
                },
                "data_statistics": {
                    "total_variations_evaluated": 0,
                    "total_episodes_processed": 0,
                    "total_batches_processed": 0,
                    "total_prediction_steps": 0,
                },
                "performance_statistics": {
                    "average_batch_processing_time_ms": None,
                    "episodes_per_second": None,
                    "steps_per_second": None,
                },
            },
            "variation_data": {},
            "cross_variation_analytics": {"performance_ranking": {}, "best_worst_analysis": {}},
            "global_aggregated_metrics": {},
        }

        # Performance tracking
        self._batch_count = 0
        self._step_count = 0

    @staticmethod
    def _get_system_environment() -> MetricDict:
        """Collect device and PyTorch runtime metadata.

        Returns:
            Device type, CUDA availability, PyTorch version, and CUDA memory
            counters when CUDA is available.
        """
        cuda_available: bool = torch.cuda.is_available()
        env: MetricDict = {
            "device_type": str(torch.cuda.get_device_name(0)) if cuda_available else "CPU",
            "cuda_available": cuda_available,
            "pytorch_version": torch.__version__,
        }
        if cuda_available:
            env["memory_info"] = {
                "cuda_memory_allocated": torch.cuda.memory_allocated(),
                "cuda_memory_reserved": torch.cuda.memory_reserved(),
            }
        return env

    def log_variation_start(self, variation_name: str) -> None:
        """Initialize logging for a new variation."""
        self.metrics_data["variation_data"][variation_name] = {
            "variation_metadata": {
                "variation_name": variation_name,
                "start_timestamp": datetime.now().isoformat(),
                "end_timestamp": None,
                "duration_seconds": None,
                "total_episodes": 0,
                "total_batches": 0,
                "total_prediction_steps": 0,
            },
            "batch_data": [],
            "aggregated_variation_metrics": {},
        }

    def log_episode_batch(
        self,
        variation_name: str,
        batch_idx: int,
        episode_indices: list[int],
        batch_results: Mapping[str, MetricValue],
        states_shape: tuple[int, ...],
        step_level_detailed_metrics: Mapping[str, MetricValue] | None = None,
    ) -> None:
        """Log metrics for a batch with essential information.

        Args:
            variation_name: Name of the current variation
            batch_idx: Index of the current batch
            episode_indices: List of episode indices in this batch
            batch_results: Results from evaluate_episode_batch
            states_shape: Shape of input states tensor
            step_level_detailed_metrics: Optional detailed step-level metrics
        """
        if variation_name not in self.metrics_data["variation_data"]:
            self.log_variation_start(variation_name)

        variation_data: VariationData = self.metrics_data["variation_data"][variation_name]
        raw_batch_metrics = batch_results["metrics"]
        batch_metrics: Mapping[str, MetricValue] = raw_batch_metrics if isinstance(raw_batch_metrics, dict) else {}
        batch_size: int = len(episode_indices)
        sequence_length: int = states_shape[1] if len(states_shape) > 1 else 1

        # Store essential batch data
        batch_data: BatchData = {
            "batch_idx": batch_idx,
            "timestamp": datetime.now().isoformat(),
            "episode_indices": episode_indices,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "metrics": batch_metrics,
            "step_level_data": step_level_detailed_metrics or {},
        }

        variation_data["batch_data"].append(batch_data)

        # Update counters
        variation_data["variation_metadata"]["total_episodes"] += batch_size
        variation_data["variation_metadata"]["total_batches"] += 1
        variation_data["variation_metadata"]["total_prediction_steps"] += sequence_length * batch_size

        self._batch_count += 1
        self._step_count += sequence_length * batch_size

    def log_variation_complete(self, variation_name: str, aggregated_metrics: Mapping[str, MetricValue]) -> None:
        """Log completion of variation evaluation."""
        if variation_name in self.metrics_data["variation_data"]:
            end_time: datetime = datetime.now()
            variation_data: VariationData = self.metrics_data["variation_data"][variation_name]
            metadata: VariationMetadata = variation_data["variation_metadata"]

            metadata["end_timestamp"] = end_time.isoformat()
            start_time: datetime = datetime.fromisoformat(metadata["start_timestamp"])
            metadata["duration_seconds"] = (end_time - start_time).total_seconds()

            variation_data["aggregated_variation_metrics"] = aggregated_metrics
            self.metrics_data["evaluation_summary"]["data_statistics"]["total_variations_evaluated"] += 1

    def finalize_and_save(
        self, overall_metrics: Mapping[str, MetricValue], total_episodes: int, total_batches: int
    ) -> str:
        """Finalize metrics collection and save to JSON file.

        Args:
            overall_metrics: Overall aggregated metrics across all variations
            total_episodes: Total number of episodes evaluated
            total_batches: Total number of batches processed

        Returns:
            Path to the saved metrics file
        """
        end_time: datetime = datetime.now()
        duration: float = (end_time - self.start_time).total_seconds()

        # Update evaluation summary
        self.metrics_data["evaluation_summary"]["execution_timeline"]["end_timestamp"] = end_time.isoformat()
        self.metrics_data["evaluation_summary"]["execution_timeline"]["total_duration_seconds"] = duration

        self.metrics_data["evaluation_summary"]["data_statistics"].update({
            "total_episodes_processed": total_episodes,
            "total_batches_processed": total_batches,
            "total_prediction_steps": self._step_count,
        })

        # Calculate performance statistics
        if duration > 0:
            self.metrics_data["evaluation_summary"]["performance_statistics"].update({
                "episodes_per_second": total_episodes / duration,
                "steps_per_second": self._step_count / duration,
            })

        self.metrics_data["global_aggregated_metrics"] = overall_metrics
        self._compute_cross_variation_analytics()

        # Save to file
        pathlib.Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(self.metrics_file).write_bytes(orjson.dumps(self.metrics_data, option=orjson.OPT_INDENT_2))

        return self.metrics_file

    def _compute_cross_variation_analytics(self) -> None:
        """Compute cross-variation analytics and comparisons."""
        variations: list[str] = list(self.metrics_data["variation_data"].keys())

        if len(variations) < 2:
            self.metrics_data["cross_variation_analytics"] = {
                "message": "Cross-variation analytics require at least 2 variations"
            }
            return

        cross_analytics: CrossVariationAnalytics = {"performance_ranking": {}, "best_worst_analysis": {}}

        # Compute performance rankings for key metrics
        for metric_name in ["encoder_mse_mean", "transition_model_mse_mean", "reconstruction_mse_mean"]:
            metric_values: dict[str, float] = {
                variation: _metric_float(
                    self.metrics_data["variation_data"][variation]["aggregated_variation_metrics"].get(metric_name),
                    float("inf"),
                )
                for variation in variations
            }

            # Create performance ranking (lower is better for MSE metrics)
            sorted_variations: list[tuple[str, float]] = sorted(metric_values.items(), key=operator.itemgetter(1))
            cross_analytics["performance_ranking"][metric_name] = [
                {"rank": i + 1, "variation": var, "value": val} for i, (var, val) in enumerate(sorted_variations)
            ]

            # Best and worst analysis
            if sorted_variations:
                cross_analytics["best_worst_analysis"][metric_name] = {
                    "best": {"variation": sorted_variations[0][0], "value": sorted_variations[0][1]},
                    "worst": {"variation": sorted_variations[-1][0], "value": sorted_variations[-1][1]},
                }

        self.metrics_data["cross_variation_analytics"] = cross_analytics

    # ---------------------------------------------------------------------
    # Helper methods for aggregation and printing
    # ---------------------------------------------------------------------
    @staticmethod
    def _aggregate_metrics_by_weight(
        episode_metrics_list: list[Mapping[str, MetricValue]], metric_keys: list[str]
    ) -> dict[str, float]:
        """Compute weighted average for a list of metric keys.

        Weights are determined by the number of episodes in each metrics dict.
        If the list is empty or the total number of episodes is zero, an empty
        dictionary is returned.
        """
        if not episode_metrics_list:
            return {}

        total_episodes: int = sum(len(_metric_int_list(m.get("episode_indices"))) for m in episode_metrics_list)
        if total_episodes == 0:
            return dict.fromkeys(metric_keys, 0.0)

        aggregated: dict[str, float] = dict.fromkeys(metric_keys, 0.0)
        for metrics in episode_metrics_list:
            weight: float = len(_metric_int_list(metrics.get("episode_indices"))) / total_episodes
            for key in metric_keys:
                aggregated[key] += _metric_float(metrics.get(key), 0.0) * weight

        return aggregated

    @staticmethod
    def _print_header(title: str) -> None:
        """Print a formatted section header."""
        logger.info(f"\n{'=' * 80}")
        logger.info(title)
        logger.info("=" * 80)

    @staticmethod
    def _print_metric_lines(
        metrics: Mapping[str, MetricValue], metric_keys: list[tuple[str, str]], indent: str = ""
    ) -> None:
        """Print metrics according to specified keys and display names."""
        for key, label in metric_keys:
            value: float = _metric_float(metrics.get(key), 0.0)
            formatted_value: str = _format_console_metric(value, key)
            logger.info(f"{indent}{label}: {formatted_value}")

    # Unified logging methods for different model types
    @staticmethod
    def log_batch_metrics_discrete(metrics: Mapping[str, MetricValue], batch_idx: int, variation_name: str) -> None:
        """Log discrete model batch metrics with formatting."""
        # Metrics already use a 0-to-100 scale, so formatting does not rescale them.
        eq_percent_str = _format_console_metric(_metric_float(metrics.get("eq_mean"), 0.0), "eq_mean")
        eq_bit_percent_str = _format_console_metric(_metric_float(metrics.get("eq_bit_mean"), 0.0), "eq_bit_mean")

        logger.info(f"  Batch {batch_idx} ({variation_name}):")
        MetricsLogger._print_metric_lines(
            metrics,
            [
                ("encoder_mse_mean", "Encoder MSE"),
                ("transition_model_mse_mean", "Transition MSE"),
                ("reconstruction_mse_mean", "Reconstruction MSE"),
            ],
            indent="    ",
        )
        logger.info(f"    Exact Match: {eq_percent_str}")
        logger.info(f"    Bit Accuracy: {eq_bit_percent_str}")

    @staticmethod
    def log_batch_metrics_continuous(metrics: Mapping[str, MetricValue], batch_idx: int, variation_name: str) -> None:
        """Log continuous model batch metrics with formatting."""
        logger.info(f"  Batch {batch_idx} ({variation_name}):")
        MetricsLogger._print_metric_lines(
            metrics,
            [
                ("encoder_mse_mean", "Encoder MSE"),
                ("transition_model_mse_mean", "Transition MSE"),
                ("reconstruction_mse_mean", "Reconstruction MSE"),
                ("cosine_similarity_mean", "Cosine Similarity"),
                ("l1_distance_mean", "L1 Distance"),
            ],
            indent="    ",
        )

    @staticmethod
    def aggregate_discrete_variation_metrics(episode_metrics_list: list[Mapping[str, MetricValue]]) -> dict[str, float]:
        """Aggregate metrics for discrete model variation."""
        return MetricsLogger._aggregate_metrics_by_weight(
            episode_metrics_list,
            [
                "encoder_mse_mean",
                "transition_model_mse_mean",
                "reconstruction_mse_mean",
                "eq_mean",
                "eq_bit_mean",
                "percent_on_mean",
            ],
        )

    @staticmethod
    def aggregate_continuous_variation_metrics(
        episode_metrics_list: list[Mapping[str, MetricValue]],
    ) -> dict[str, float]:
        """Aggregate metrics for continuous model variation."""
        return MetricsLogger._aggregate_metrics_by_weight(
            episode_metrics_list,
            [
                "encoder_mse_mean",
                "transition_model_mse_mean",
                "reconstruction_mse_mean",
                "cosine_similarity_mean",
                "l1_distance_mean",
                "relative_error_mean",
            ],
        )

    @staticmethod
    def log_final_results_discrete(
        overall_metrics: dict[str, float],
        variation_metrics: dict[str, dict[str, float]],
        use_wandb: bool = False,
        tracking_session: WandbTrackingSession | None = None,
    ) -> None:
        """Log final discrete model results with formatting."""
        MetricsLogger._print_header("DISCRETE MODEL EVALUATION RESULTS")

        # Overall results
        logger.info("\nOVERALL RESULTS:")
        MetricsLogger._print_metric_lines(
            overall_metrics,
            [
                ("encoder_mse_mean", "Encoder MSE"),
                ("transition_model_mse_mean", "Transition MSE"),
                ("reconstruction_mse_mean", "Reconstruction MSE"),
            ],
            indent="  ",
        )
        eq_percent_str: str = _format_console_metric(overall_metrics.get("eq_mean", 0.0), "eq_mean")
        eq_bit_percent_str: str = _format_console_metric(overall_metrics.get("eq_bit_mean", 0.0), "eq_bit_mean")

        logger.info(f"  Exact Match: {eq_percent_str}")
        logger.info(f"  Bit Accuracy: {eq_bit_percent_str}")

        # Per-variation results
        logger.info("\nPER-VARIATION RESULTS:")
        for variation_name, metrics in variation_metrics.items():
            logger.info(f"  {variation_name}:")
            eq_percent_str_var: str = _format_console_metric(metrics.get("eq_mean", 0.0), "eq_mean")
            eq_bit_percent_str_var: str = _format_console_metric(metrics.get("eq_bit_mean", 0.0), "eq_bit_mean")
            logger.info(f"    Exact Match: {eq_percent_str_var}")
            logger.info(f"    Bit Accuracy: {eq_bit_percent_str_var}")
            MetricsLogger._print_metric_lines(metrics, [("encoder_mse_mean", "Encoder MSE")], indent="    ")

        if use_wandb:
            session = tracking_session or get_active_tracking_session()
            if session is not None:
                _log_nested_metrics(session, {"final_results": overall_metrics}, commit=False)
                _log_nested_metrics(session, {"variation_results": variation_metrics}, commit=True)

    @staticmethod
    def log_final_results_continuous(
        overall_metrics: dict[str, float],
        variation_metrics: dict[str, dict[str, float]],
        use_wandb: bool = False,
        tracking_session: WandbTrackingSession | None = None,
    ) -> None:
        """Log final continuous model results with formatting."""
        MetricsLogger._print_header("CONTINUOUS MODEL EVALUATION RESULTS")

        # Overall results
        logger.info("\nOVERALL RESULTS:")
        MetricsLogger._print_metric_lines(
            overall_metrics,
            [
                ("encoder_mse_mean", "Encoder MSE"),
                ("transition_model_mse_mean", "Transition MSE"),
                ("reconstruction_mse_mean", "Reconstruction MSE"),
                ("cosine_similarity_mean", "Cosine Similarity"),
                ("l1_distance_mean", "L1 Distance"),
            ],
            indent="  ",
        )

        # Per-variation results
        logger.info("\nPER-VARIATION RESULTS:")
        for variation_name, metrics in variation_metrics.items():
            logger.info(f"  {variation_name}:")
            MetricsLogger._print_metric_lines(
                metrics,
                [
                    ("cosine_similarity_mean", "Cosine Similarity"),
                    ("l1_distance_mean", "L1 Distance"),
                    ("encoder_mse_mean", "Encoder MSE"),
                ],
                indent="    ",
            )

        if use_wandb:
            session = tracking_session or get_active_tracking_session()
            if session is not None:
                _log_nested_metrics(session, {"final_results": overall_metrics}, commit=False)
                _log_nested_metrics(session, {"variation_results": variation_metrics}, commit=True)


def extract_model_info(
    encoder: torch.nn.Module,
    transition_model: torch.nn.Module,
    decoder: torch.nn.Module,
    alignment_model: torch.nn.Module | None,
) -> MetricDict:
    """Extract essential model information for logging.

    Args:
        encoder: Encoder model
        transition_model: Transition model
        decoder: Decoder model
        alignment_model: Alignment model (optional)

    Returns:
        Dictionary containing model information
    """

    def _get_model_info(model: torch.nn.Module, _name: str) -> MetricDict:
        return {
            "class_name": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

    model_info: MetricDict = {
        "encoder": _get_model_info(encoder, "encoder"),
        "transition_model": _get_model_info(transition_model, "transition_model"),
        "decoder": _get_model_info(decoder, "decoder"),
    }

    if alignment_model is not None:
        model_info["alignment_model"] = _get_model_info(alignment_model, "alignment_model")

    return model_info
