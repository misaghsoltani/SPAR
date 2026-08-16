"""Shared evaluation loop for discrete and continuous world models."""

from __future__ import annotations

from abc import abstractmethod
from logging import getLogger
import pathlib
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar

import numpy as np
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import torch
import torch.nn.functional as F

from spar.testing.visualization import (
    BestWorstStepTracker,
    GridData,
    MetricsTracker,
    QualityMetrics,
    VisualizationData,
    create_step_visualization,
    create_summary_visualization,
)
from spar.testing.visualization.image_utils import format_metric_value
from spar.utils.log_utils.console_logger import terminal_console as console

if TYPE_CHECKING:
    from logging import Logger

    from rich.progress import TaskID
    from torch import Tensor, nn
    from torch.nn import Module

    from spar.data.testing_dataset import TestDataLoader, VariationInfo
    from spar.testing.visualization.types import EpisodeMetrics, EpisodeMetricsValue, SuptitleStyleValue
    from spar.utils.config_utils.config_schema import BBoxConfig, SuptitleConfig


logger: Logger = getLogger(__name__)
AccumulatorT = TypeVar("AccumulatorT")


def _visualization_step_key(value: VisualizationData) -> int:
    """Return the step number used to order visualization records."""
    return value.step


class EpisodeSummary(TypedDict, total=False):
    """Summary of a single episode's evaluation results."""

    episode_number: int
    metric_value: float
    metric_name: str
    is_higher_better: bool
    viz_data: list[GridData]
    metric_values: list[float]


class EvaluateResults(TypedDict):
    """Results from evaluating the model over the entire test dataset."""

    overall_metrics: dict[str, float]
    variation_metrics: dict[str, dict[str, float]]
    episode_metrics: list[EpisodeMetrics]
    total_episodes: int
    total_batches: int


class ModelTesterBase(Generic[AccumulatorT]):
    """Abstract base class for model evaluation with shared functionality."""

    def __init__(
        self,
        encoder: nn.Module,
        transition_model: nn.Module,
        decoder: nn.Module,
        alignment_model: nn.Module | None = None,
        *,
        device: str | torch.device = "cpu",
        test_model_type: str = "discrete",
        use_alignment_model: bool = False,
        output_dir: str = "./outputs",
        variations_to_use: list[str] | None = None,
        variations_to_ignore: list[str] | None = None,
        use_variation_for_all_states: bool = False,
        save_interval: int = 10,
        visualization_format: str = "png",
        visualization_episode_index: int = 0,
        visualization_steps: list[int] | None = None,
        log_interval: int = 5,
        use_wandb: bool = False,
        end_to_end: bool = False,
        top_k: int = 1,
        apply_diff_highlighting: bool = False,
        metrics_to_save: list[str] | None = None,
        row_labels: list[str] | None = None,
        rightmost_col_row_labels: list[str] | None = None,
        rightmost_col_row_labels_side: str = "right",
        variant_panel_title: str | None = None,
        suptitle_cfg: SuptitleConfig | None = None,
        column_metric_priority: list[str] | None = None,
    ) -> None:
        """Initialize the model evaluator.

        Args:
            encoder: Pre-trained encoder model.
            transition_model: Pre-trained transition model.
            decoder: Pre-trained decoder model.
            alignment_model: Pre-trained alignment model (optional).
            device: Device to run evaluation on.
            test_model_type: Type of model ("discrete" or "continuous").
            use_alignment_model: Whether to use alignment model for encoding.
            output_dir: Directory to save outputs.
            variations_to_use: List of variations to include (None for all).
            variations_to_ignore: List of variations to exclude.
            use_variation_for_all_states: Whether to apply variation to all states.
            save_interval: Interval for saving reconstruction images.
            visualization_format: Format for saving visualizations (default is "png").
            visualization_episode_index: Index of episode in batch to use for visualization (default is 0).
            visualization_steps: Additional specific steps to visualize (in addition to save_interval).
            log_interval: Interval for logging batch metrics.
            use_wandb: Whether to log metrics to Weights & Biases.
            end_to_end: Whether to use end-to-end evaluation mode with different logic.
            top_k: Number of top best/worst states to consider for evaluation.
            apply_diff_highlighting: Whether to apply highlighting for differences in visualizations.
            metrics_to_save: List of metrics to save in JSON format (e.g., ["reconstruction_mse", "eq_bit"]).
            row_labels: Optional labels for the two grid rows (left-side labels). Defaults to
                ["Reconstruction", "Ground Truth"].
            rightmost_col_row_labels: Optional labels to display adjacent to the right-most column
                (e.g., for labeling the predicted vs ground-truth rows differently for the right block).
            rightmost_col_row_labels_side: Placement for the right-most column row labels in per-step
                visualizations. ``right`` places them outside on the far right. ``left`` places them between
                the last two columns. Defaults to "right" for per-step while summaries remain default.
            variant_panel_title: Optional title for a separate left-side single-cell panel that shows the
                variant/noisy environment.
            suptitle_cfg: Optional config object or dict controlling suptitle text templates, style, and gap.
            column_metric_priority: Optional list of metric keys to prioritize for column titles
        """
        self.device: torch.device = torch.device(device) if isinstance(device, str) else device
        self.test_model_type: str = test_model_type
        self.use_alignment_model: bool = use_alignment_model
        self.output_dir: str = output_dir
        self.variations_to_use: list[str] | None = variations_to_use
        self.variations_to_ignore: list[str] | None = variations_to_ignore
        self.use_variation_for_all_states: bool = use_variation_for_all_states
        self.save_interval: int = save_interval
        self.visualization_format: str = visualization_format
        self.visualization_episode_index: int = visualization_episode_index
        self.visualization_steps: list[int] | None = visualization_steps
        self.log_interval: int = log_interval
        self.use_wandb: bool = use_wandb
        self.end_to_end: bool = end_to_end
        self.top_k: int = top_k
        self.apply_diff_highlighting: bool = apply_diff_highlighting
        self.metrics_to_save: list[str] = metrics_to_save or ["reconstruction_mse"]
        # Row labels for visualization
        self.row_labels: list[str] = row_labels or ["Reconstruction", "Ground Truth"]
        self.rightmost_col_row_labels: list[str] | None = rightmost_col_row_labels
        # Placement of the optional right-most column row labels for per-step visualizations
        self.rightmost_col_row_labels_side: str = rightmost_col_row_labels_side
        self.variant_panel_title: str = variant_panel_title or "Noisy starting state"
        # The suptitle config may be a dataclass or mapping.
        self.suptitle_cfg: SuptitleConfig | None = suptitle_cfg
        # Column metric priority for titles (first matching key wins)
        self.column_metric_priority: list[str] | None = column_metric_priority

        # Store models and move to device
        self.encoder: Module = encoder.to(self.device, non_blocking=True)
        self.transition_model: Module = transition_model.to(self.device, non_blocking=True)
        self.decoder: Module = decoder.to(self.device, non_blocking=True)
        self.alignment_model: Module | None = (
            alignment_model.to(self.device, non_blocking=True) if alignment_model is not None else None
        )

        # Set models to eval mode with torch.no_grad for performance
        self.encoder.eval()
        self.transition_model.eval()
        self.decoder.eval()

        if self.alignment_model is not None:
            self.alignment_model.eval()

        # Create output directory
        pathlib.Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        # Initialize metrics tracking system
        self.metrics_tracker = MetricsTracker(self.output_dir, self.metrics_to_save)

        cfg: SuptitleConfig | None = self.suptitle_cfg
        step_tracker_template: str | None = cfg.step_tracker_template if cfg is not None else None

        # Add step tracking
        self.step_tracker: BestWorstStepTracker | None = (
            BestWorstStepTracker(
                top_k,
                output_dir=self.output_dir,
                visualization_format=self.visualization_format,
                apply_highlighting=self.apply_diff_highlighting,
                suptitle_style=self._get_suptitle_style_dict(),
                suptitle_gap=self._get_suptitle_gap(),
                step_tracker_template=step_tracker_template,
                # Apply rightmost labels only to best/worst step visuals
                rightmost_col_row_labels=self.rightmost_col_row_labels,
                rightmost_col_row_labels_side=self.rightmost_col_row_labels_side,
            )
            if top_k > 0
            else None
        )

        self._is_discrete: bool = test_model_type == "discrete"
        # Initialize best/worst episode tracking variables
        self.best_episode_data: EpisodeSummary | None = None
        self.worst_episode_data: EpisodeSummary | None = None
        self.best_metric_value: float | None = None
        self.worst_metric_value: float | None = None
        self.metric_name: str | None = None
        self.is_higher_better: bool | None = None
        # Track per-step visualization data for the selected episode index
        self._selected_episode_viz_data: list[VisualizationData] = []
        self._selected_episode_number: int | None = None

    # Suptitle helpers

    def _get_suptitle_style_dict(self) -> dict[str, SuptitleStyleValue] | None:
        cfg: SuptitleConfig | None = self.suptitle_cfg
        if cfg is None:
            return None

        style: dict[str, SuptitleStyleValue] = {}

        raw_font_family: str | None = cfg.font_family
        if raw_font_family is not None:
            style["font_family"] = raw_font_family
        raw_font_serif: list[str] | None = cfg.font_serif
        if raw_font_serif is not None:
            style["font_serif"] = raw_font_serif
        style["font_size"] = cfg.font_size
        style["font_weight"] = cfg.font_weight
        style["color"] = cfg.color
        raw_bbox: BBoxConfig = cfg.bbox_style
        style["bbox_style"] = {
            "facecolor": raw_bbox.facecolor,
            "edgecolor": raw_bbox.edgecolor,
            "boxstyle": raw_bbox.boxstyle,
            "pad": raw_bbox.pad,
            "alpha": raw_bbox.alpha,
            "linewidth": raw_bbox.linewidth,
        }

        return style

    def _get_suptitle_gap(self) -> float | None:
        cfg: SuptitleConfig | None = self.suptitle_cfg
        if cfg is None:
            return None

        return cfg.gap

    def _resolve_step_suptitle(self, variation_name: str, step: int, episode: int) -> str | None:
        cfg: SuptitleConfig | None = self.suptitle_cfg
        if cfg is None:
            return None

        template: str | None = cfg.step_template
        if template is None:
            return None  # use default auto title in VisualizationData

        # If explicitly empty, disable suptitle
        if not template:
            return ""

        return template.format(
            episode=episode,
            step=step,
            variation=variation_name,
            variation_display=variation_name.replace("_", " ").title(),
        )

    def _resolve_summary_suptitle(
        self, variation_name: str, summary_type: str, metric_name: str, metric_value_formatted: str, episode: int
    ) -> str | None:
        cfg: SuptitleConfig | None = self.suptitle_cfg
        if cfg is None:
            return None

        template: str | None = cfg.summary_template
        if template is None:
            return None
        if not template:
            return ""
        # Optional override labels from config
        custom_labels: dict[str, str] | None = None
        labels: dict[str, str] | None = cfg.summary_type_labels
        if isinstance(labels, dict):
            # Cast values to str safely
            custom_labels = dict(labels.items())

        if custom_labels is not None:
            summary_type_label: str | None = custom_labels.get(summary_type, custom_labels.get(summary_type.lower()))
            if summary_type_label is None:
                summary_type_label = (
                    "Best" if summary_type == "best" else ("Worst" if summary_type == "worst" else "Selected")
                )
        else:
            summary_type_label = (
                "Best" if summary_type == "best" else ("Worst" if summary_type == "worst" else "Selected")
            )
        return template.format(
            variation=variation_name,
            variation_display=variation_name.replace("_", " ").title(),
            summary_type=summary_type,
            summary_type_label=summary_type_label,
            metric_name=metric_name,
            metric_name_display=metric_name.replace("_", " ").title(),
            metric_value_formatted=metric_value_formatted,
            episode=episode,
        )

    def _order_metrics_for_display(
        self, md: dict[str, float | int | str] | None
    ) -> dict[str, float | int | str] | None:
        """Reorder metrics dict so preferred key(s) appear first.

        Keeps original values, only changes insertion order for display logic that
        uses the first key (e.g., column titles).
        """
        if not md:
            return md
        prio: list[str] | None = self.column_metric_priority
        if not prio:
            return md
        # Build ordered dict with preferred keys first (if present), then the rest
        seen: set[str] = set()
        ordered: dict[str, float | int | str] = {}
        for key in prio:
            if key in md and key not in seen:
                ordered[key] = md[key]
                seen.add(key)
        ordered.update({key: val for key, val in md.items() if key not in seen})
        return ordered

    def _encode_state(self, state: Tensor) -> Tensor:
        """State encoding with device transfer and caching.

        Args:
            state: State tensor to encode.

        Returns:
            Encoded state tensor.
        """
        # state = state.to(self.device, non_blocking=True)

        with torch.inference_mode():
            encoded: Tensor = (
                self.alignment_model(state)
                if self.use_alignment_model and self.alignment_model is not None
                else self.encoder(state)
            )

        # Round outputs in discrete mode
        return self._preprocess_encoding(encoded)

    @staticmethod
    @abstractmethod
    def _should_apply_highlighting(state_encoding_pred: Tensor, target_state_encoding: Tensor) -> bool:
        """Determine if highlighting should be applied in standard evaluation mode."""
        raise NotImplementedError("Must be implemented by subclass")

    @staticmethod
    @abstractmethod
    def _preprocess_encoding(encoding: Tensor) -> Tensor:
        """Preprocess encoding based on model type (e.g., rounding for discrete).

        Args:
            encoding: Raw encoding tensor.

        Returns:
            Preprocessed encoding tensor.
        """
        raise NotImplementedError("Must be implemented by subclass")

    @staticmethod
    @abstractmethod
    def _compute_model_specific_step_metrics(pred_encoding: Tensor, target_encoding: Tensor) -> dict[str, float]:
        """Compute model-specific metrics for a single step.

        Args:
            pred_encoding: Predicted encoding tensor.
            target_encoding: Target encoding tensor.

        Returns:
            Dictionary of step-specific metrics.
        """
        raise NotImplementedError("Must be implemented by subclass")

    @staticmethod
    @abstractmethod
    def _initialize_model_specific_accumulators() -> AccumulatorT:
        """Initialize model-specific metric accumulators."""
        raise NotImplementedError("Must be implemented by subclass")

    @staticmethod
    @abstractmethod
    def _accumulate_model_specific_metrics(
        step_metrics: dict[str, float], model_specific_accumulators: AccumulatorT
    ) -> None:
        """Accumulate model-specific metrics for a single step.

        Args:
            step_metrics: Dictionary of metrics from _compute_model_specific_step_metrics.
            model_specific_accumulators: Dictionary of accumulators to update.
        """
        raise NotImplementedError("Must be implemented by subclass")

    @staticmethod
    @abstractmethod
    def _finalize_model_specific_metrics(model_specific_accumulators: AccumulatorT, num_steps: int) -> dict[str, float]:
        """Finalize and calculate mean/std for model-specific metrics.

        Args:
            model_specific_accumulators: Dictionary of accumulated metrics.
            num_steps: Number of steps over which metrics were accumulated.

        Returns:
            Dictionary of finalized metrics with mean/std values.
        """
        raise NotImplementedError("Must be implemented by subclass")

    @staticmethod
    @abstractmethod
    def _calculate_episode_metrics(pred_encoding: Tensor, target_encoding: Tensor, batch_size: int) -> list[float]:
        """Calculate metrics for each episode in the batch.

        Args:
            pred_encoding: Predicted encodings for the batch.
            target_encoding: Target encodings for the batch.
            batch_size: int | float of episodes in the batch.

        Returns:
            List of metric values for each episode.
        """
        raise NotImplementedError("Must be implemented by subclass")

    @staticmethod
    def _aggregate_variation_metrics(episode_metrics_list: list[EpisodeMetrics]) -> dict[str, float]:
        """Aggregate metrics for a single variation."""
        if not episode_metrics_list:
            return {}

        # Average each metric across episodes.
        aggregated: dict[str, float] = {}
        metric_keys: set[str] = set()
        for episode_metrics in episode_metrics_list:
            metric_keys.update(episode_metrics.keys())

        for key in metric_keys:
            values: list[float] = []
            for episode_metrics in episode_metrics_list:
                if key in episode_metrics:
                    value: EpisodeMetricsValue = episode_metrics[key]
                    # Only include numeric values in aggregation (exclude strings, tensors, arrays)
                    if isinstance(value, int | float | np.number):
                        values.append(float(value))

            if values:
                aggregated[key] = float(np.mean(values))

        return aggregated

    @staticmethod
    def _compute_mean_std(sum_val: float, squared_sum_val: float, num_steps: int) -> tuple[float, float]:
        """Compute the mean and standard deviation given a sum of values and the sum of squared values.

        Args:
            sum_val: Sum of values.
            squared_sum_val: Sum of squared values.
            num_steps: Number of steps over which the sums were accumulated.

        Returns:
            A tuple containing the mean and standard deviation.
        """
        mean: float = sum_val / num_steps
        variance: float = max(0.0, (squared_sum_val / num_steps) - mean**2)
        return mean, float(np.sqrt(variance))

    def evaluate_episode_batch(
        self,
        states: Tensor,
        actions: Tensor,
        target_states: Tensor,
        encoded_target_states: Tensor | None,
        variation_name: str,
        episode_indices: Tensor,
        batch_idx: int,
    ) -> dict[str, EpisodeMetrics]:
        """Test a batch of episodes with on-the-fly computation.

        Args:
            states: Input states with variation applied (kept on CPU until needed).
            actions: Action sequences (kept on CPU until needed).
            target_states: Ground truth target states (always base variant, kept on CPU until needed).
            encoded_target_states: Pre-encoded target states (if available, kept on CPU until needed).
            variation_name: Name of the variation being evaluated.
            episode_indices: Episode indices in the batch.
            batch_idx: Current batch index to determine if images should be saved.

        Returns:
            Dictionary containing evaluation metrics and data.
        """
        if self.end_to_end:
            return self._evaluate_episode_batch_end_to_end(
                states, actions, target_states, encoded_target_states, variation_name, episode_indices, batch_idx
            )

        return self._evaluate_episode_batch_standard(
            states, actions, target_states, encoded_target_states, variation_name, episode_indices, batch_idx
        )

    def _evaluate_episode_batch_end_to_end(
        self,
        states: Tensor,
        actions: Tensor,
        target_states: Tensor,
        encoded_target_states: Tensor | None,
        variation_name: str,
        episode_indices: Tensor,
        batch_idx: int,
    ) -> dict[str, EpisodeMetrics]:
        """End-to-end evaluation.

        Args:
            states: Input states with variation applied.
            actions: Action sequences.
            target_states: Ground truth target states.
            encoded_target_states: Pre-encoded target states (if available).
            variation_name: Name of the variation being evaluated.
            episode_indices: Episode indices in the batch.
            batch_idx: Current batch index.

        Returns:
            Dictionary containing evaluation metrics and data.
        """
        batch_size, episode_len = states.shape[:2]

        # Set batch index for step tracker if enabled
        if self.step_tracker is not None:
            self.step_tracker.set_batch_idx(batch_idx)

        # Metric accumulators
        # [enc_mse, enc_mse**2, trans_mse, trans_mse**2, recon_mse, recon_mse**2]
        metrics_sums: Tensor = torch.zeros(6, device=self.device)

        # Initialize model-specific metric accumulators
        model_specific_accumulators: AccumulatorT = self._initialize_model_specific_accumulators()

        # Extract and set up accumulator access
        model_specific_sums: Tensor
        if self._is_discrete:
            model_specific_sums = torch.zeros(8, device=self.device)  # Pre-allocate for all discrete metrics
        else:
            model_specific_sums = torch.zeros(6, device=self.device)  # Pre-allocate for continuous metrics

        # Track visualization data
        episode_visualization_data: list[GridData] = []
        episode_viz_metrics: dict[int, QualityMetrics] = {}
        batch_needs_visualization: bool = batch_idx % 20 == 0

        with torch.inference_mode():
            initial_state: Tensor = states[:, 0]
            # In end-to-end mode with use_variation_for_all_states=False, the first state shown
            # should be the actual varied initial state, not the base/target initial state.
            # The varied initial state makes the first column display the matching
            # "Reconstruction vs True Image" pairing for step 0.
            initial_state_true: Tensor = initial_state
            metrics_dicts_per_episode: list[dict[str, float | int | str]]
            if not self.use_variation_for_all_states:
                # Mode 1: Sequential reconstruction
                initial_state_encoding: Tensor = self._encode_state(initial_state)
                initial_state_reconstructed: Tensor = self.decoder(initial_state_encoding)
                current_state: Tensor = initial_state
                for step in range(episode_len - 1):
                    current_state_encoding: Tensor = self._encode_state(current_state)
                    action: Tensor = actions[:, step]

                    # Predict next encoding
                    next_state_encoding_pred: Tensor = self._preprocess_encoding(
                        self.transition_model(current_state_encoding, action)
                    )

                    # Decode prediction
                    reconstructed_next_state: Tensor = self.decoder(next_state_encoding_pred)
                    target_next_state: Tensor = target_states[:, step + 1]

                    # Calculate per-episode MSE (vs BASE target)
                    batch_mse_values: Tensor = F.mse_loss(reconstructed_next_state, target_next_state, reduction="none")
                    batch_mse_values = batch_mse_values.view(batch_size, -1).mean(dim=1)  # shape: [batch_size]

                    # Track MSE data for each episode
                    # Continue with aggregated MSE for existing metrics
                    recon_mse: int | float | bool = batch_mse_values.mean().item()
                    metrics_sums[4] += recon_mse
                    metrics_sums[5] += recon_mse * recon_mse

                    # Visualization handling
                    if batch_needs_visualization and (
                        (s := step + 1) % self.save_interval == 0
                        or ((vs := self.visualization_steps) is not None and s in vs)
                        or s == episode_len - 1
                    ):
                        # For end-to-end mode with use_variation_for_all_states=False,
                        # primary metric is reconstruction MSE
                        # Build per-episode metrics
                        per_episode_mse: list[float] = batch_mse_values.detach().cpu().tolist()
                        metrics_dicts_per_episode = [{"reconstruction_mse": v} for v in per_episode_mse]
                        primary_metric_values: list[float] = list(per_episode_mse)
                        primary_is_higher_better: bool = False

                        # Compute step-level encoding metrics for display (aggregated over batch)
                        try:
                            target_next_state_encoding: Tensor = self._preprocess_encoding(
                                self.alignment_model(target_next_state)
                                if self.use_alignment_model and self.alignment_model is not None
                                else self.encoder(target_next_state)
                            )
                            step_metrics_display: dict[str, float] | None = self._compute_model_specific_step_metrics(
                                next_state_encoding_pred, target_next_state_encoding
                            )
                        except Exception:
                            step_metrics_display = None

                        self._handle_step_visualization(
                            step=step,
                            batch_idx=batch_idx,
                            step_n_top_state=reconstructed_next_state,
                            step_n_bottom_state=target_next_state,
                            step0_top_state=initial_state_reconstructed,
                            step0_bottom_state=initial_state_true,
                            base_step0_bottom_state=target_states[:, 0],
                            batch_size=batch_size,
                            episode_indices=episode_indices,
                            variation_name=variation_name,
                            episode_visualization_data=episode_visualization_data,
                            episode_viz_metrics=episode_viz_metrics,
                            metrics_dicts_per_episode=metrics_dicts_per_episode,
                            step_metrics_for_display=step_metrics_display,
                            primary_metric_values_per_episode=primary_metric_values,
                            is_higher_better=primary_is_higher_better,
                            apply_highlighting=self.apply_diff_highlighting,
                        )

                    # Record step-level metrics for JSON export
                    self.metrics_tracker.update_step_metrics(step=step, reconstruction_mse=recon_mse)

                    # Update current state for next iteration
                    current_state = reconstructed_next_state.detach()

            else:
                # Mode 2: Direct encoding/decoding
                # initial_state_reconstructed = torch.empty_like(initial_state)
                for step in range(episode_len):
                    current_state = states[:, step]
                    current_state_encoding = self._encode_state(current_state)
                    reconstructed_current_state: Tensor = self.decoder(current_state_encoding)
                    target_current_state: Tensor = target_states[:, step]

                    # if step == 0:
                    #     initial_state_reconstructed = reconstructed_current_state.clone()

                    # Target encoding
                    target_current_state_encoding: Tensor = (
                        encoded_target_states[:, step]
                        if encoded_target_states is not None
                        else self._preprocess_encoding(self.encoder(target_current_state))
                    )

                    # Metric computation - calculate per episode MSE
                    enc_mse: int | float | bool = F.mse_loss(
                        current_state_encoding, target_current_state_encoding
                    ).item()

                    # Calculate per-episode reconstruction MSE (vs BASE target)
                    batch_recon_mse_values: Tensor = F.mse_loss(
                        reconstructed_current_state, target_current_state, reduction="none"
                    )
                    batch_recon_mse_values = batch_recon_mse_values.view(batch_size, -1).mean(dim=1)  # Per episode MSE

                    # Continue with aggregated MSE for existing metrics
                    recon_mse = batch_recon_mse_values.mean().item()

                    metrics_sums[0] += enc_mse
                    metrics_sums[1] += enc_mse * enc_mse
                    metrics_sums[4] += recon_mse
                    metrics_sums[5] += recon_mse * recon_mse
                    # No transition model in this mode

                    # Model-specific metrics
                    self._compute_model_metrics(
                        current_state_encoding, target_current_state_encoding, model_specific_sums
                    )

                    # Visualization
                    if batch_needs_visualization and (
                        (s := step + 1) % self.save_interval == 0
                        or ((vs := self.visualization_steps) is not None and s in vs)
                        or s == episode_len - 1
                    ):
                        # For end-to-end mode with use_variation_for_all_states=True,
                        # primary metric is encoding comparison (eq_bit for discrete, cosine_sim for continuous)
                        step_metrics: dict[str, float] = self._compute_model_specific_step_metrics(
                            current_state_encoding, target_current_state_encoding
                        )

                        # For visualization/tracking we still prefer reconstruction MSE per-episode
                        primary_is_higher_better = False

                        # Build per-episode metrics for visualization
                        per_episode_mse = batch_recon_mse_values.detach().cpu().tolist()
                        metrics_dicts_per_episode = [{"reconstruction_mse": float(v)} for v in per_episode_mse]
                        # Primary metric values are the per-episode reconstruction MSEs
                        primary_metric_values = [float(v) for v in per_episode_mse]

                        # Update metrics tracker with all computed metrics including model-specific ones
                        all_metrics: dict[str, float] = {"reconstruction_mse": recon_mse, **step_metrics}
                        self.metrics_tracker.update_step_metrics(step=step, **all_metrics)

                        self._handle_step_visualization(
                            step=step,
                            batch_idx=batch_idx,
                            step_n_top_state=self.decoder(target_current_state_encoding),
                            step_n_bottom_state=target_current_state,
                            step0_top_state=reconstructed_current_state,
                            step0_bottom_state=current_state,
                            base_step0_bottom_state=target_states[:, 0],
                            batch_size=batch_size,
                            episode_indices=episode_indices,
                            variation_name=variation_name,
                            episode_visualization_data=episode_visualization_data,
                            episode_viz_metrics=episode_viz_metrics,
                            metrics_dicts_per_episode=metrics_dicts_per_episode,
                            step_metrics_for_display=step_metrics,
                            primary_metric_values_per_episode=primary_metric_values,
                            is_higher_better=primary_is_higher_better,
                            apply_highlighting=self.apply_diff_highlighting
                            and self._should_apply_highlighting(current_state_encoding, target_current_state_encoding),
                        )

                    # Compute model-specific metrics for tracking purposes
                    step_metrics = self._compute_model_specific_step_metrics(
                        current_state_encoding, target_current_state_encoding
                    )

                    # Update metrics tracker with all computed metrics
                    all_metrics = {"reconstruction_mse": recon_mse, **step_metrics}
                    self.metrics_tracker.update_step_metrics(step=step, **all_metrics)

        num_steps: int = episode_len - 1 if not self.use_variation_for_all_states else episode_len
        metrics_means: Tensor = metrics_sums[::2] / num_steps  # Every other element (sums, not squares)
        metrics_vars: Tensor = torch.clamp(metrics_sums[1::2] / num_steps - metrics_means**2, min=0.0)
        metrics_stds: Tensor = torch.sqrt(metrics_vars)

        # Convert to NumPy
        enc_mse_mean, trans_mse_mean, recon_mse_mean = metrics_means.cpu().numpy()
        enc_mse_std, trans_mse_std, recon_mse_std = metrics_stds.cpu().numpy()

        # Construct results
        episode_indices_list: list[int] = episode_indices.cpu().numpy().tolist()
        batch_metrics: EpisodeMetrics
        batch_metrics = {
            "variation_name": variation_name,
            "episode_indices": episode_indices_list,
            "encoder_mse_mean": float(enc_mse_mean),
            "encoder_mse_std": float(enc_mse_std),
            "transition_model_mse_mean": float(trans_mse_mean),
            "transition_model_mse_std": float(trans_mse_std),
            "reconstruction_mse_mean": float(recon_mse_mean),
            "reconstruction_mse_std": float(recon_mse_std),
            "num_steps": num_steps,
        }

        # Add model-specific metrics
        model_specific_metrics: dict[str, float] = self._finalize_model_specific_metrics(
            model_specific_accumulators, num_steps
        )
        batch_metrics.update(model_specific_metrics)

        # Handle episode tracking
        if episode_visualization_data:
            self._update_best_worst_episodes(episode_viz_metrics)

        self.metrics_tracker.finalize_episode_batch(batch_size, variation_name)

        return {"metrics": batch_metrics}

    def _evaluate_episode_batch_standard(
        self,
        states: Tensor,
        actions: Tensor,
        target_states: Tensor,
        encoded_target_states: Tensor | None,
        variation_name: str,
        episode_indices: Tensor,
        batch_idx: int,
    ) -> dict[str, EpisodeMetrics]:
        """Standard evaluation logic (original implementation)."""
        batch_size, episode_len = states.shape[:2]

        # Set batch index for step tracker if enabled
        if self.step_tracker is not None:
            self.step_tracker.set_batch_idx(batch_idx)

        # Initialize metric accumulators for MSE metrics.  The order is [recon_sum, recon_sq_sum].
        metrics_sums: list[float] = [0.0, 0.0]

        # Initialize model-specific metric accumulators using the abstract method to set up any
        # subclass-specific tracking.
        model_specific_accumulators: AccumulatorT = self._initialize_model_specific_accumulators()

        # Track visualization data for this episode batch
        episode_visualization_data: list[GridData] = []

        # Track per-episode metrics for summary visualizations
        episode_viz_metrics: dict[int, QualityMetrics] = {}  # episode_idx -> QualityMetrics

        # Check if this batch might need visualization
        batch_needs_visualization: bool = batch_idx % 50 == 0

        # Track previous step bit equality for detecting drops from 100%
        previous_step_bit_equality: dict[int, float] = {}  # episode_idx -> bit_equality

        with torch.inference_mode():
            # Start with the first state
            current_state: Tensor = states[:, 0]
            current_state_encoding: Tensor = self._encode_state(current_state)
            initial_state_reconstructed: Tensor = self.decoder(current_state_encoding).clone()

            # Initialize bit equality tracking for discrete models
            if self.test_model_type == "discrete":
                # Initialize with 100% for all episodes (perfect match at start)
                for episode_idx_in_batch in range(batch_size):
                    previous_step_bit_equality[episode_idx_in_batch] = 100.0

            for step in range(episode_len - 1):
                # Get action for this step
                action: Tensor = actions[:, step]

                # Predict next encoding using transition model
                next_state_encoding_pred = self._preprocess_encoding(
                    self.transition_model(current_state_encoding, action)
                )

                # Get target encoding for next state
                target_next_state: Tensor = target_states[:, step + 1]

                target_next_state_encoding: Tensor = (
                    encoded_target_states[:, step + 1]
                    if encoded_target_states is not None
                    else self._preprocess_encoding(self.encoder(target_next_state))
                )

                # # 2. Transition Model MSE
                # transition_model_mse = F.mse_loss(next_state_encoding_pred, target_next_state_encoding).item()

                # 3. Reconstruction MSE - calculate per episode
                reconstructed_next_state: Tensor = self.decoder(next_state_encoding_pred).clone()
                batch_reconstruction_mse_values: Tensor = F.mse_loss(
                    reconstructed_next_state, target_next_state, reduction="none"
                )
                batch_reconstruction_mse_values = batch_reconstruction_mse_values.view(batch_size, -1).mean(dim=1)

                # Continue with aggregated MSE for existing metrics
                reconstruction_mse: int | float | bool = batch_reconstruction_mse_values.mean().item()

                # Accumulate MSE metrics into metrics_sums
                metrics_sums[0] += reconstruction_mse
                metrics_sums[1] += reconstruction_mse**2

                # Compute model-specific metrics and accumulate them.
                step_metrics: dict[str, float] = self._compute_model_specific_step_metrics(
                    next_state_encoding_pred, target_next_state_encoding
                )
                self._accumulate_model_specific_metrics(step_metrics, model_specific_accumulators)

                # Handle visualization
                if batch_needs_visualization and (
                    (s := step + 1) % self.save_interval == 0
                    or ((vs := self.visualization_steps) is not None and s in vs)
                    or s == episode_len - 1
                ):
                    # For standard mode, primary metric is always reconstruction MSE
                    # Build per-episode metrics for visualization
                    per_episode_mse: list[float] = batch_reconstruction_mse_values.detach().cpu().tolist()
                    metrics_dicts_per_episode: list[dict[str, float | int | str]] = [
                        {"reconstruction_mse": v} for v in per_episode_mse
                    ]
                    self._handle_step_visualization(
                        step=step,
                        batch_idx=batch_idx,
                        step_n_top_state=reconstructed_next_state,
                        # Always use BASE target for ground truth/MSE display
                        step_n_bottom_state=target_next_state,
                        step0_top_state=initial_state_reconstructed,
                        step0_bottom_state=current_state,
                        base_step0_bottom_state=target_states[:, 0],
                        batch_size=batch_size,
                        episode_indices=episode_indices,
                        variation_name=variation_name,
                        episode_visualization_data=episode_visualization_data,
                        episode_viz_metrics=episode_viz_metrics,
                        metrics_dicts_per_episode=metrics_dicts_per_episode,
                        step_metrics_for_display=step_metrics,
                        primary_metric_values_per_episode=list(per_episode_mse),
                        is_higher_better=False,
                        apply_highlighting=self.apply_diff_highlighting
                        and self._should_apply_highlighting(next_state_encoding_pred, target_next_state_encoding),
                    )

                # Update current encoding for next iteration
                current_state_encoding = next_state_encoding_pred

                # Update metrics tracker with reconstruction MSE and model-specific metrics
                self.metrics_tracker.update_step_metrics(
                    step=step, **{"reconstruction_mse": reconstruction_mse, **step_metrics}
                )

        # Compute final statistics and return
        num_steps: int = episode_len - 1

        # Compute mean and standard deviation for the three MSE metrics
        recon_mean, recon_std = self._compute_mean_std(metrics_sums[0], metrics_sums[1], num_steps)

        # Convert episode indices
        episode_indices_list: list[int] = episode_indices.cpu().numpy().tolist()

        # Base metrics dictionary
        episode_metrics: EpisodeMetrics = {
            "variation_name": variation_name,
            "episode_indices": episode_indices_list,
            "reconstruction_mse_mean": recon_mean,
            "reconstruction_mse_std": recon_std,
            "num_steps": num_steps,
        }

        # Add model-specific metrics
        model_specific_metrics: dict[str, float] = self._finalize_model_specific_metrics(
            model_specific_accumulators, num_steps
        )
        episode_metrics.update(model_specific_metrics)

        # Track episode data for summary visualizations
        if episode_visualization_data:
            self._update_best_worst_episodes(episode_viz_metrics)

        # Finalize batch processing for on-the-fly MSE plotting
        self.metrics_tracker.finalize_episode_batch(batch_size, variation_name)

        return {"metrics": episode_metrics}

    def evaluate_dataloader(self, dataloader: TestDataLoader) -> EvaluateResults:
        """Test model on all data in the dataloader.

        Args:
            dataloader: TestDataLoader containing evaluation data.

        Returns:
            Dictionary containing aggregated evaluation results.
        """
        logger.info("Starting model evaluation...")

        # Local accumulators
        all_episode_metrics: list[EpisodeMetrics] = []
        variation_metrics: dict[str, dict[str, float]] = {}

        # Get variation info
        variation_info: VariationInfo = dataloader.get_variation_info()
        logger.info(
            f"Evaluating on {len(variation_info['variations'])} variations: {variation_info['variations']}\n"
            f"Ignored variations: {self.variations_to_ignore or 'None'}\n"
        )

        total_episodes = 0
        total_batches = 0

        # Progress tracking
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            eval_task: TaskID = progress.add_task("Evaluating...", total=len(dataloader))

            for variation_name in variation_info["variations"]:
                logger.info(f"\nEvaluating variation: {variation_name}")

                # Reset tracking for this variation
                self._reset_variation_tracking()

                # Reset step tracker for new variation if enabled
                if self.step_tracker is not None:
                    self.step_tracker.reset_for_new_variation()
                    self.step_tracker.set_variation(variation_name)

                # Reset dataloader for this variation
                dataloader.reset_tracking(variation_name)
                variation_episode_metrics: list[EpisodeMetrics] = []

                # Iterate through batches for this variation
                for batch_idx, batch in enumerate(
                    dataloader(variation_name, use_variation_for_all_states=self.use_variation_for_all_states)
                ):
                    episode_indices: Tensor = batch["episode_indices"]
                    # Test this batch
                    encoded_target_states: Tensor | None = batch.get("encoded_target_states")
                    if encoded_target_states is not None:
                        encoded_target_states = encoded_target_states.to(self.device, non_blocking=True)
                    batch_results: dict[str, EpisodeMetrics] = self.evaluate_episode_batch(
                        states=batch["states"].to(self.device, non_blocking=True),
                        actions=batch["actions"].to(self.device, non_blocking=True),
                        target_states=batch["target_states"].to(self.device, non_blocking=True),
                        encoded_target_states=encoded_target_states,
                        variation_name=variation_name,
                        episode_indices=episode_indices,
                        batch_idx=batch_idx,
                    )

                    # Store metrics
                    batch_metrics: EpisodeMetrics = batch_results["metrics"]
                    variation_episode_metrics.append(batch_metrics)
                    all_episode_metrics.append(batch_metrics)

                    # Log progress every log_interval batches
                    if batch_idx % self.log_interval == 0:
                        logger.info(f"Processed batch {batch_idx} for variation {variation_name}")

                    total_batches += 1
                    total_episodes += len(episode_indices)

                    progress.update(eval_task, advance=1)

                # Aggregate metrics for this variation
                variation_metrics[variation_name] = self._aggregate_variation_metrics(variation_episode_metrics)

                logger.info(
                    f"Completed evaluation for variation {variation_name}: {len(variation_episode_metrics)} batches"
                )

                # Save summary visualizations for this variation immediately
                self._save_variation_summary_visualizations(variation_name)

        # Compute overall aggregated metrics
        overall_metrics: dict[str, float] = self._aggregate_variation_metrics(all_episode_metrics)

        # Log final results
        logger.info("Evaluation completed")
        logger.info(f"Total episodes: {total_episodes}")
        logger.info(f"Total batches: {total_batches}")

        return {
            "overall_metrics": overall_metrics,
            "variation_metrics": variation_metrics,
            "episode_metrics": all_episode_metrics,
            "total_episodes": total_episodes,
            "total_batches": total_batches,
        }

    def _reset_variation_tracking(self) -> None:
        """Reset tracking variables for a new variation."""
        self.best_episode_data = None
        self.worst_episode_data = None
        self.best_metric_value = None
        self.worst_metric_value = None
        self.metric_name = None
        self.is_higher_better = None
        # Reset metrics tracker data for new variation
        self.metrics_tracker.reset_variation_data()
        # Reset selected episode viz data for this variation
        self._selected_episode_viz_data = []
        self._selected_episode_number = None

    def _update_best_worst_episodes(self, episode_viz_metrics: dict[int, QualityMetrics]) -> None:
        """Update best and worst episode tracking for current variation.

        Args:
            episode_viz_metrics: Dictionary of episode visualization metrics
        """
        if not episode_viz_metrics:
            return

        # Process each episode's metrics
        for episode_number, episode_data in episode_viz_metrics.items():
            metric_values: list[float] = episode_data.get("metric_values", [])
            is_higher_better: bool = episode_data.get("is_higher_better", True)
            metric_name: str = episode_data.get("metric_name", "unknown")

            if not metric_values:
                continue

            # Calculate episode metric (minimum value across all steps)
            episode_metric: float = min(metric_values)

            # Initialize tracking variables on first episode
            if self.metric_name is None:
                self.metric_name = metric_name
                self.is_higher_better = is_higher_better

            # Check if this is the best episode so far
            is_new_best: bool = (
                self.best_metric_value is None
                or (is_higher_better and episode_metric > self.best_metric_value)
                or (not is_higher_better and episode_metric < self.best_metric_value)
            )

            if is_new_best:
                self.best_metric_value = episode_metric
                self.best_episode_data = {
                    "episode_number": episode_number,
                    "metric_value": episode_metric,
                    "metric_name": metric_name,
                    "is_higher_better": is_higher_better,
                    "viz_data": episode_data.get("viz_data", []),
                    # Store per-step values so summary columns show correct metrics per step
                    "metric_values": episode_data.get("metric_values", []),
                }

            # Check if this is the worst episode so far
            is_new_worst: bool = (
                self.worst_metric_value is None
                or (is_higher_better and episode_metric < self.worst_metric_value)
                or (not is_higher_better and episode_metric > self.worst_metric_value)
            )

            if is_new_worst:
                self.worst_metric_value = episode_metric
                self.worst_episode_data = {
                    "episode_number": episode_number,
                    "metric_value": episode_metric,
                    "metric_name": metric_name,
                    "is_higher_better": is_higher_better,
                    "viz_data": episode_data.get("viz_data", []),
                    # Store per-step values so summary columns show correct metrics per step
                    "metric_values": episode_data.get("metric_values", []),
                }

    def _compute_model_metrics(self, pred_encoding: Tensor, target_encoding: Tensor, metrics_tensor: Tensor) -> None:
        """Model-specific metrics computation using the same logic as subclasses.

        Args:
            pred_encoding: Predicted encoding tensor
            target_encoding: Target encoding tensor
            metrics_tensor: Pre-allocated tensor for accumulating metrics
        """
        # Use the subclass-implemented method for consistent logic
        step_metrics: dict[str, float] = self._compute_model_specific_step_metrics(pred_encoding, target_encoding)

        if self._is_discrete:
            # Disc metrics order:[percent_on, percent_on**2, eq, eq**2, eq_bit, eq_bit**2, eq_bit_min, eq_bit_min**2]
            percent_on: float = step_metrics["percent_on"]
            eq: float = step_metrics["eq"]
            eq_bit: float = step_metrics["eq_bit"]
            eq_bit_min: float = step_metrics["eq_bit_min"]

            metrics_tensor[0] += percent_on
            metrics_tensor[1] += percent_on**2
            metrics_tensor[2] += eq
            metrics_tensor[3] += eq**2
            metrics_tensor[4] += eq_bit
            metrics_tensor[5] += eq_bit**2
            metrics_tensor[6] += eq_bit_min
            metrics_tensor[7] += eq_bit_min**2
        else:
            # Continuous metrics order: [cosine_sim, cosine_sim**2, l1_dist, l1_dist**2, rel_error, rel_error**2]
            cosine_sim: float = step_metrics["cosine_similarity"]
            l1_dist: float = step_metrics["l1_distance"]
            rel_error: float = step_metrics["relative_error"]

            metrics_tensor[0] += cosine_sim
            metrics_tensor[1] += cosine_sim**2
            metrics_tensor[2] += l1_dist
            metrics_tensor[3] += l1_dist**2
            metrics_tensor[4] += rel_error
            metrics_tensor[5] += rel_error**2

    def _handle_step_visualization(
        self,
        *,
        step: int,
        batch_idx: int,
        step_n_top_state: Tensor,
        step_n_bottom_state: Tensor,
        step0_top_state: Tensor,
        step0_bottom_state: Tensor,
        base_step0_bottom_state: Tensor | None = None,
        batch_size: int,
        episode_indices: Tensor,
        variation_name: str,
        episode_visualization_data: list[GridData],
        episode_viz_metrics: dict[int, QualityMetrics],
        metrics_dicts_per_episode: list[dict[str, float | int | str]] | None,
        step_metrics_for_display: dict[str, float] | None = None,
        primary_metric_values_per_episode: list[float] | None,
        is_higher_better: bool = True,
        apply_highlighting: bool = False,
    ) -> None:
        """Visualization handling with per-episode metrics to avoid duplication."""
        # Determine primary metric name (use first key of the first dict if available)
        if metrics_dicts_per_episode and len(metrics_dicts_per_episode) > 0 and metrics_dicts_per_episode[0]:
            primary_metric_key: str = next(iter(metrics_dicts_per_episode[0].keys()))
        else:
            primary_metric_key = "unknown"

        for eps_idx_in_batch in range(batch_size):
            eps_number = int(episode_indices[eps_idx_in_batch].item())

            grid_data = GridData(
                step=step + 1,
                episode_number=eps_number,
                step0_bottom=step0_bottom_state[eps_idx_in_batch].cpu().numpy(),
                step0_top=step0_top_state[eps_idx_in_batch].cpu().numpy(),
                step_n_bottom=step_n_bottom_state[eps_idx_in_batch].cpu().numpy(),
                step_n_top=step_n_top_state[eps_idx_in_batch].cpu().numpy(),
                batch_idx=batch_idx,
                base_step0_bottom=(
                    base_step0_bottom_state[eps_idx_in_batch].cpu().numpy()
                    if (variation_name != "base" and base_step0_bottom_state is not None)
                    else None
                ),
                variant_panel_title=(self.variant_panel_title if variation_name != "base" else None),
            )

            episode_visualization_data.append(grid_data)

            # Populate episode_viz_metrics for tracking best/worst episodes
            if eps_number not in episode_viz_metrics:
                episode_viz_metrics[eps_number] = QualityMetrics(
                    metric_values=[], metric_name=primary_metric_key, is_higher_better=is_higher_better, viz_data=[]
                )

            # Add metric value and visualization data to this episode's tracking
            episode_data: QualityMetrics = episode_viz_metrics[eps_number]
            if "metric_values" not in episode_data:
                episode_data["metric_values"] = []
            if "viz_data" not in episode_data:
                episode_data["viz_data"] = []

            # Select per-episode metric value, fallback to 0.0
            primary_metric_value: float = (
                primary_metric_values_per_episode[eps_idx_in_batch]
                if primary_metric_values_per_episode is not None
                and eps_idx_in_batch < len(primary_metric_values_per_episode)
                else 0.0
            )
            episode_data["metric_values"].append(primary_metric_value)
            episode_data["viz_data"].append(grid_data)

            if self.step_tracker is not None:
                self.step_tracker.update_step(
                    step=step + 1,
                    batch_idx=batch_idx,
                    episode_number=eps_number,
                    metric=primary_metric_value,
                    is_higher_better=is_higher_better,
                    step_data=grid_data,
                    metric_name=primary_metric_key,
                )

            if eps_idx_in_batch == self.visualization_episode_index:
                # Choose per-episode metrics dict for display if provided
                metrics_dict: dict[str, float | int | str] | None = None
                if metrics_dicts_per_episode is not None and eps_idx_in_batch < len(metrics_dicts_per_episode):
                    metrics_dict = dict(metrics_dicts_per_episode[eps_idx_in_batch])
                # Merge in step-level metrics (aggregated) for display purposes if provided
                if step_metrics_for_display:
                    metrics_dict = {**(metrics_dict or {}), **step_metrics_for_display}
                # Reorder by priority
                metrics_dict = self._order_metrics_for_display(metrics_dict)
                viz_data = VisualizationData(
                    step=step + 1,
                    episode_number=eps_number,
                    step0_bottom=step0_bottom_state[eps_idx_in_batch].cpu().numpy(),
                    step0_top=step0_top_state[eps_idx_in_batch].cpu().numpy(),
                    step_n_bottom=step_n_bottom_state[eps_idx_in_batch].cpu().numpy(),
                    step_n_top=step_n_top_state[eps_idx_in_batch].cpu().numpy(),
                    batch_idx=batch_idx,
                    output_dir=self.output_dir,
                    variation_name=variation_name,
                    metric=primary_metric_value,
                    metrics_dict=metrics_dict,
                    visualization_format=self.visualization_format,
                    row_labels=self.row_labels,
                    # Do not draw rightmost labels in inline per-step grids
                    rightmost_col_row_labels=None,
                    # For per-step visualizations (inline within step grid), keep default placement
                    # so labels are not placed on the far right of the main step grid.
                    # Best/Worst step tracker visuals handle right-side placement separately.
                    base_step0_bottom=(
                        base_step0_bottom_state[eps_idx_in_batch].cpu().numpy()
                        if (variation_name != "base" and base_step0_bottom_state is not None)
                        else None
                    ),
                    variant_panel_title=(self.variant_panel_title if variation_name != "base" else None),
                    suptitle=self._resolve_step_suptitle(variation_name, step + 1, eps_number),
                    suptitle_style=self._get_suptitle_style_dict(),
                    suptitle_gap=self._get_suptitle_gap(),
                )
                create_step_visualization(viz_data=viz_data, apply_highlighting=apply_highlighting)
                # Collect viz data for creating a summary of the selected episode at variation end.
                # Only collect for a single episode number per variation to avoid mixing across batches.
                if self._selected_episode_number is None:
                    self._selected_episode_number = eps_number
                if eps_number == self._selected_episode_number:
                    self._selected_episode_viz_data.append(viz_data)

    def _save_variation_summary_visualizations(self, variation_name: str) -> None:
        """Save summary visualizations for a variation immediately after completion.

        Args:
            variation_name: Name of the variation to save visualizations for
        """
        # Save metrics data to JSON for this variation only
        json_path: str = self.metrics_tracker.save_variation_metrics_to_json(
            variation_name, f"{variation_name}_evaluation_metrics.json"
        )
        logger.info(f"Saved evaluation metrics to: {json_path}")

        if not self.best_episode_data and not self.worst_episode_data:
            # Still attempt to save the selected episode summary if available
            pass

        # Save best episode visualization
        if self.best_episode_data:
            best_viz_data: list[GridData] | None = self.best_episode_data.get("viz_data")
            if best_viz_data:
                viz_data_list: list[VisualizationData] = []
                metric_values: list[float] = self.best_episode_data.get("metric_values", [])
                metric_name_overall: str = self.best_episode_data.get("metric_name", "unknown")
                metric_value_overall: float = self.best_episode_data.get("metric_value", 0.0)
                episode_number_overall: int = self.best_episode_data.get("episode_number", 0)

                for i, grid_data in enumerate(best_viz_data):
                    # Get the metric value for this specific step
                    step_metric_value: float = metric_values[i] if i < len(metric_values) else metric_value_overall

                    viz_data = VisualizationData(
                        step=grid_data.step,
                        episode_number=grid_data.episode_number,
                        step0_bottom=grid_data.step0_bottom,
                        step0_top=grid_data.step0_top,
                        step_n_bottom=grid_data.step_n_bottom,
                        step_n_top=grid_data.step_n_top,
                        batch_idx=grid_data.batch_idx,
                        output_dir=self.output_dir,
                        variation_name=variation_name,
                        metric=step_metric_value,
                        metrics_dict={metric_name_overall: step_metric_value},
                        visualization_format=self.visualization_format,
                        row_labels=self.row_labels,
                        rightmost_col_row_labels=self.rightmost_col_row_labels,
                        base_step0_bottom=grid_data.base_step0_bottom,
                        variant_panel_title=grid_data.variant_panel_title,
                        suptitle=self._resolve_step_suptitle(variation_name, grid_data.step, grid_data.episode_number),
                        suptitle_style=self._get_suptitle_style_dict(),
                        suptitle_gap=self._get_suptitle_gap(),
                    )
                    viz_data_list.append(viz_data)

                create_summary_visualization(
                    episode_viz_data=viz_data_list,
                    variation_name=variation_name,
                    metric_value=metric_value_overall,
                    metric_name=metric_name_overall,
                    summary_type="best",
                    output_dir=self.output_dir,
                    visualization_format=self.visualization_format,
                    episode_number=episode_number_overall,
                    apply_highlighting=self.apply_diff_highlighting,
                    row_labels=self.row_labels,
                    rightmost_col_row_labels=self.rightmost_col_row_labels,
                    suptitle_text=self._resolve_summary_suptitle(
                        variation_name,
                        "best",
                        metric_name_overall,
                        format_metric_value(metric_value_overall, metric_name_overall),
                        episode_number_overall,
                    ),
                    suptitle_style=self._get_suptitle_style_dict(),
                    suptitle_gap=self._get_suptitle_gap(),
                )
        if self.worst_episode_data:
            worst_viz_data: list[GridData] | None = self.worst_episode_data.get("viz_data")
            if worst_viz_data:
                viz_data_list = []
                metric_values = self.worst_episode_data.get("metric_values", [])
                metric_name_overall = self.worst_episode_data.get("metric_name", "unknown")
                metric_value_overall = self.worst_episode_data.get("metric_value", 0.0)
                episode_number_overall = self.worst_episode_data.get("episode_number", 0)

                for i, grid_data in enumerate(worst_viz_data):
                    # Get the metric value for this specific step
                    step_metric_value = metric_values[i] if i < len(metric_values) else metric_value_overall

                    viz_data = VisualizationData(
                        step=grid_data.step,
                        episode_number=grid_data.episode_number,
                        step0_bottom=grid_data.step0_bottom,
                        step0_top=grid_data.step0_top,
                        step_n_bottom=grid_data.step_n_bottom,
                        step_n_top=grid_data.step_n_top,
                        batch_idx=grid_data.batch_idx,
                        output_dir=self.output_dir,
                        variation_name=variation_name,
                        metric=step_metric_value,
                        metrics_dict={metric_name_overall: step_metric_value},
                        visualization_format=self.visualization_format,
                        row_labels=self.row_labels,
                        rightmost_col_row_labels=self.rightmost_col_row_labels,
                        base_step0_bottom=grid_data.base_step0_bottom,
                        variant_panel_title=grid_data.variant_panel_title,
                        suptitle=self._resolve_step_suptitle(variation_name, grid_data.step, grid_data.episode_number),
                        suptitle_style=self._get_suptitle_style_dict(),
                        suptitle_gap=self._get_suptitle_gap(),
                    )
                    viz_data_list.append(viz_data)

                create_summary_visualization(
                    episode_viz_data=viz_data_list,
                    variation_name=variation_name,
                    metric_value=metric_value_overall,
                    metric_name=metric_name_overall,
                    summary_type="worst",
                    output_dir=self.output_dir,
                    visualization_format=self.visualization_format,
                    episode_number=episode_number_overall,
                    apply_highlighting=self.apply_diff_highlighting,
                    row_labels=self.row_labels,
                    rightmost_col_row_labels=self.rightmost_col_row_labels,
                    suptitle_text=self._resolve_summary_suptitle(
                        variation_name,
                        "worst",
                        metric_name_overall,
                        format_metric_value(metric_value_overall, metric_name_overall),
                        episode_number_overall,
                    ),
                    suptitle_style=self._get_suptitle_style_dict(),
                    suptitle_gap=self._get_suptitle_gap(),
                )

        # Save summary visualization for the selected episode index (from args)
        if self._selected_episode_viz_data:
            # Sort metrics by step.
            selected_viz_sorted: list[VisualizationData] = sorted(
                self._selected_episode_viz_data, key=_visualization_step_key
            )
            # Determine metric name and aggregate metric value for title
            metric_name: str = "reconstruction_mse"
            if selected_viz_sorted and selected_viz_sorted[0].metrics_dict:
                metric_name = next(iter(selected_viz_sorted[0].metrics_dict.keys()))
            # Use mean of per-step metrics for the selected episode
            metric_value: float = float(np.mean([v.metric for v in selected_viz_sorted]))

            create_summary_visualization(
                episode_viz_data=selected_viz_sorted,
                variation_name=variation_name,
                metric_value=metric_value,
                metric_name=metric_name,
                summary_type="selected",
                output_dir=self.output_dir,
                visualization_format=self.visualization_format,
                episode_number=self._selected_episode_number or 0,
                apply_highlighting=self.apply_diff_highlighting,
                row_labels=self.row_labels,
                rightmost_col_row_labels=self.rightmost_col_row_labels,
                suptitle_text=self._resolve_summary_suptitle(
                    variation_name,
                    "selected",
                    metric_name,
                    format_metric_value(metric_value, metric_name),
                    self._selected_episode_number or 0,
                ),
                suptitle_style=self._get_suptitle_style_dict(),
                suptitle_gap=self._get_suptitle_gap(),
            )

        # Save best/worst step visualizations if step tracker is enabled
        if self.step_tracker is not None:
            self.step_tracker.save_best_worst_visualizations()
