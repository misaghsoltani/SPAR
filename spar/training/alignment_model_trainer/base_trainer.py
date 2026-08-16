"""Alignment Model Trainer for SPAR."""

from __future__ import annotations

from logging import getLogger
import pathlib
import time
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.rule import Rule
import torch

from spar.training.deferred_scalar_buffer import DeferredScalarBuffer
from spar.utils.log_utils.console_logger import terminal_console as console
from spar.utils.log_utils.wandb_logger import log_metrics, log_metrics_batched

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from logging import Logger
    from pathlib import Path
    from typing import TypeAlias

    from rich.progress import TaskID
    from torch import Tensor
    from torch.nn import Module as nnModule
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

    from spar.utils.log_utils.wandb_logger import MetricPayload, WandbTrackingSession
    from spar.utils.log_utils.wandb_osh import WandbOshSyncTrigger
    from spar.utils.pytorch_utils.model_stripper import CheckpointValue


logger: Logger = getLogger(__name__)

LogOutputValue: TypeAlias = str | int | float | None
LogOutput: TypeAlias = dict[str, LogOutputValue]
MetricHistoryValue: TypeAlias = float | int | list[float]
MetricHistory: TypeAlias = dict[str, list[MetricHistoryValue]]


class RequiredMetricsDict(TypedDict):
    """Required metrics dictionary for alignment training: loss must be present."""

    loss: Tensor


class MetricsDict(RequiredMetricsDict, total=False):
    """Metrics dictionary structure for alignment training."""

    # Loss components
    loss_steps: list[float]

    # Alignment-specific metrics
    cosine_similarity: float
    l1_distance: float
    relative_error: float
    percent_aligned: float

    # Additional metrics
    percent_on: float
    eq: float
    eq_bit: float
    eq_bit_min: float

    # Additional data
    predicted_encodings: Tensor
    target_encodings: Tensor


class AlignmentModelTrainerBase:
    """Base trainer for alignment models."""

    def __init__(
        self,
        train_dataloader: DataLoader[dict[str, Tensor]],
        val_dataloader: DataLoader[dict[str, Tensor]],
        test_dataloader: DataLoader[dict[str, Tensor]] | None = None,
        *,
        pretrained_encoder: nnModule | None,
        alignment_model: nnModule,
        transition_model: nnModule | None = None,
        decoder: nnModule | None = None,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: str | torch.device = "cpu",
        log_interval: int = 100,
        checkpoint_interval: int = 100,
        use_wandb: bool = False,
        tracking_session: WandbTrackingSession | None = None,
        wandb_osh_trigger: WandbOshSyncTrigger | None = None,
        checkpoint_path: str = "",
        starting_iteration: int = 0,
        ending_iteration: int = 100_000,
        current_iteration: int = 0,
        training_metrics: dict[str, CheckpointValue] | None = None,
        end_to_end: bool = False,
        freeze_pretrained_models: bool = True,
        is_transition_frozen: bool = False,
        is_decoder_frozen: bool = False,
        **kwargs: CheckpointValue | WandbTrackingSession | WandbOshSyncTrigger,
    ) -> None:
        if isinstance(device, str):
            device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.device: torch.device = device

        self.pretrained_encoder: nnModule | None = pretrained_encoder
        self.alignment_model: nnModule = alignment_model
        self.transition_model: nnModule | None = transition_model
        self.decoder: nnModule | None = decoder

        self.optimizer: Optimizer = optimizer
        self.scheduler: LRScheduler = scheduler

        self.train_dataloader: DataLoader[dict[str, Tensor]] = train_dataloader
        self.val_dataloader: DataLoader[dict[str, Tensor]] = val_dataloader
        self.test_dataloader: DataLoader[dict[str, Tensor]] | None = test_dataloader

        # Combined storage for Train/Validation logs
        self.log_outputs: dict[str, LogOutput] = {}
        self.log_interval: int = log_interval
        self.checkpoint_interval: int = checkpoint_interval
        self.use_wandb: bool = use_wandb
        self.tracking_session: WandbTrackingSession | None = tracking_session
        self.wandb_osh_trigger: WandbOshSyncTrigger | None = wandb_osh_trigger

        # Internal state for checkpointing
        self.current_iteration: int = current_iteration
        self.starting_iteration: int = starting_iteration
        self.ending_iteration: int = ending_iteration
        self.checkpoint_path: str = checkpoint_path
        self.training_metrics: dict[str, CheckpointValue] = training_metrics.copy() if training_metrics else {}

        self.end_to_end: bool = end_to_end
        self.freeze_pretrained_models: bool = freeze_pretrained_models

        # Track individual model freeze states for proper mode management
        self.is_transition_frozen: bool = is_transition_frozen
        self.is_decoder_frozen: bool = is_decoder_frozen

        # Move models to device
        if self.pretrained_encoder is not None:
            self.pretrained_encoder.to(self.device)
            # Freeze pretrained encoder
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False
            self.pretrained_encoder.eval()

        self.alignment_model.to(self.device)

        if self.transition_model is not None:
            self.transition_model.to(self.device)
            # Respect upstream freeze decision for transition model
            if self.is_transition_frozen:
                for param in self.transition_model.parameters():
                    param.requires_grad = False
                self.transition_model.eval()
            else:
                for param in self.transition_model.parameters():
                    param.requires_grad = True
                self.transition_model.train()

        if self.decoder is not None:
            self.decoder.to(self.device)
            # Respect upstream freeze decision for decoder
            if self.is_decoder_frozen:
                for param in self.decoder.parameters():
                    param.requires_grad = False
                self.decoder.eval()
            else:
                for param in self.decoder.parameters():
                    param.requires_grad = True
                self.decoder.train()

        # Clearer init log: show component presence and status
        has_transition_path: bool = bool(kwargs.pop("has_transition_path", False))
        has_decoder_path: bool = bool(kwargs.pop("has_decoder_path", False))

        transition_status: str
        if self.transition_model is None:
            transition_status = "unused" if has_transition_path and not self.end_to_end else "absent"
        else:
            transition_status = "frozen" if self.is_transition_frozen else "trainable"

        decoder_status: str
        if self.decoder is None:
            decoder_status = "unused" if has_decoder_path and not self.end_to_end else "absent"
        else:
            decoder_status = "frozen" if self.is_decoder_frozen else "trainable"

        encoder_status: str = "absent" if self.pretrained_encoder is None else "frozen"

        logger.info(
            f"Alignment trainer init | end_to_end={self.end_to_end} | "
            f"encoder={encoder_status} | transition={transition_status} | decoder={decoder_status}"
        )

        # Save each arg in kwargs in a variable
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.METRIC_DISPLAY_CONFIG: dict[str, dict[str, str]] = {
            "mse_loss": {"format": ".2E", "label": "mse"},
            "percent_on": {"format": ".2f", "label": "%on"},
            "eq": {"format": ".2f", "label": "eq"},
            "eq_bit": {"format": ".2f", "label": "eq_bit"},
            "eq_bit_min": {"format": ".2f", "label": "eq_bit_min"},
            "cosine_similarity": {"format": ".3f", "label": "cos_sim"},
            "l1_distance": {"format": ".2E", "label": "l1_dist"},
            "relative_error": {"format": ".3f", "label": "rel_err"},
        }

    def _basic_lr_update(self, lr: float) -> None:
        """Update optimizer and scheduler fields shared by other schedulers.

        Args:
            lr: New base learning rate.
        """
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        if hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = [lr] * len(self.optimizer.param_groups)

        if hasattr(self.scheduler, "_last_lr"):
            last_lr_attr = "_last_lr"
            setattr(self.scheduler, last_lr_attr, [lr] * len(self.optimizer.param_groups))

    def update_lr(self, lr: float) -> None:
        """Reset the base learning rate without changing scheduler history.

        Args:
            lr: New base learning rate.
        """
        sched: LRScheduler = self.scheduler
        opt: Optimizer = self.optimizer
        name: str = sched.__class__.__name__

        # All schedulers expose base_lrs: keep them in-sync with the new base
        if hasattr(sched, "base_lrs"):
            sched.base_lrs = [lr] * len(opt.param_groups)

        epoch: int = getattr(sched, "last_epoch", 0)  # public, always present
        gamma: float = getattr(sched, "gamma", 1.0)  # present on the three we handle
        factor: float = 1.0  # multiplicative decay factor

        if name == "ExponentialLR":  # lr * gamma**step
            step: int = getattr(sched, "_step_count", epoch)
            factor = gamma**step

        elif name == "StepLR":  # lr * gamma**(epoch//step_size)
            step_size: int = getattr(sched, "step_size", 1)
            factor = gamma ** (epoch // step_size)

        elif name == "MultiStepLR":  # lr * gamma**(passed milestones)
            milestones: list[int] = list(getattr(sched, "milestones", []))
            factor = gamma ** sum(epoch >= m for m in milestones)

        else:  # anything else -> no decay yet
            return self._basic_lr_update(lr)

        new_lr: float = lr * factor  # final LR at current step

        # Update the optimizer and keep scheduler caches coherent
        for g in opt.param_groups:
            g["lr"] = new_lr

        if hasattr(sched, "_last_lr"):
            last_lr_attr = "_last_lr"
            setattr(sched, last_lr_attr, [new_lr] * len(opt.param_groups))
        return None

    def get_encoded_targets(self, base_states: Tensor) -> Tensor:
        """Get encoded targets by encoding base states online.

        Args:
            base_states: Base states to encode

        Returns:
            Encoded base states

        Raises:
            ValueError: If pretrained_encoder is None when online encoding is needed
        """
        if self.pretrained_encoder is None:
            raise ValueError(
                "Cannot perform online encoding: pretrained_encoder is None. "
                "This typically means targets should be precomputed or provided in the dataset."
            )

        # Online encoding using pretrained encoder
        self.pretrained_encoder.eval()
        with torch.inference_mode():
            encoded_targets = self.pretrained_encoder(base_states)
        # Clone outside inference-mode so downstream autograd loss paths can consume targets safely.
        return encoded_targets.clone()

    def get_last_lr(self) -> float:
        """Get the last learning rate used by the scheduler."""
        return float(self.scheduler.get_last_lr()[0])

    def update_iterations_range(self, starting_iteration: int, ending_iteration: int) -> None:
        """Update the range of iterations for training."""
        self.starting_iteration = starting_iteration
        self.ending_iteration = ending_iteration

    def update_checkpoint_path(self, checkpoint_path: str) -> None:
        """Update the checkpoint path for saving/loading model state."""
        pathlib.Path(pathlib.Path(checkpoint_path).parent).mkdir(exist_ok=True, parents=True)
        self.checkpoint_path = checkpoint_path

    def set_eval_mode(self) -> None:
        """Set the alignment model to evaluation mode."""
        self.alignment_model.eval()
        if self.transition_model is not None:
            self.transition_model.eval()
        if self.decoder is not None:
            self.decoder.eval()
        # Pretrained encoder is always in eval mode (if present)

    def set_train_mode(self) -> None:
        """Set the alignment model to training mode."""
        self.alignment_model.train()
        if self.transition_model is not None:
            # Only set to train mode if not frozen
            if not self.is_transition_frozen:
                self.transition_model.train()
            else:
                self.transition_model.eval()
        if self.decoder is not None:
            # Only set to train mode if not frozen
            if not self.is_decoder_frozen:
                self.decoder.train()
            else:
                self.decoder.eval()
        # Pretrained encoder stays in eval mode (if present)

    def log_step(
        self,
        prefix: str,
        itr: int,
        results: MetricsDict | None = None,
        lr: float | None = None,
        elapsed_time: float = 0.0,
        trigger_display: bool = False,
    ) -> None:
        """Store logs for display, optionally prepare W&B, and trigger console output."""
        prefix_l = prefix.lower()
        if results is None:
            self.log_outputs[prefix_l] = {"itr": itr, "elapsed_time": elapsed_time}
            if trigger_display:
                self._display_combined(self.log_outputs)

            return

        loss = results["loss"]

        # Human-readable metrics string
        loss_steps = results.get("loss_steps")
        metrics_str = self._format_metrics_str(loss, loss_steps, results, self.METRIC_DISPLAY_CONFIG)

        # Store for combined display
        self.log_outputs[prefix_l] = {"metrics_str": metrics_str, "itr": itr, "lr": lr, "elapsed_time": elapsed_time}

        # If both Train & Validation present, render panel
        if trigger_display:
            self._display_combined(self.log_outputs)

    @staticmethod
    def _format_metrics_str(
        loss: Tensor,
        _loss_steps: Sequence[float] | None,
        results: MetricsDict,
        metric_display_config: dict[str, dict[str, str]],
    ) -> str:
        parts: list[str] = [f"loss: {loss.item():.2E}"]

        excluded: set[str] = {"loss", "loss_steps", "predicted_encodings", "target_encodings"}

        for key, val in results.items():
            if key in excluded or val is None:
                continue

            cfg: dict[str, str] | None = metric_display_config.get(key)

            if not cfg:
                continue

            # Only handle scalar Tensors or numeric types
            scalar: float | int | None = None
            if isinstance(val, torch.Tensor) and val.numel() == 1:
                scalar = val.item()
            elif isinstance(val, (int, float)):
                scalar = val
            else:
                continue
            parts.append(f"{cfg['label']}: {scalar:{cfg['format']}}")

        return ", ".join(parts)

    @staticmethod
    def _build_wandb_dict(prefix: str, itr: int, lr: float, results: MetricsDict) -> MetricPayload:
        """Build a dict of metrics for W&B logging."""
        # Normalize prefix to lowercase
        prefix_l: str = prefix.lower()
        # RequiredMetricsDict records that every step result contains ``loss``.
        req_results: RequiredMetricsDict = results
        # Initialize W&B metrics dict
        wandb: dict[str, int | float | bool | str | Tensor | None] = {
            f"{prefix_l}/loss": req_results["loss"].item(),
            f"{prefix_l}/itr": itr,
            f"{prefix_l}/lr": lr,
            "global_step": itr,
        }

        # Add other scalar metrics
        for k, v in results.items():
            if k == "loss" or v is None:
                continue

            if isinstance(v, (int, float)):
                wandb[f"{prefix_l}/{k}"] = v

            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                wandb[f"{prefix_l}/{k}"] = v.item()

        return wandb

    @staticmethod
    def _display_combined(log_outputs: dict[str, LogOutput]) -> None:
        train: LogOutput | None = log_outputs.get("train")
        val: LogOutput | None = log_outputs.get("validation")
        misc: LogOutput = log_outputs.get("misc", {})

        if not train or not val:
            logger.info("No training or validation data to display.")
            return

        total_time_raw: LogOutputValue = misc.get("elapsed_time", 0.0)
        total_time: float = float(total_time_raw) if isinstance(total_time_raw, (int, float)) else 0.0
        content: Align = Align.center(
            Group(
                f"[bold green]Train[/bold green]\n{train['metrics_str']}",
                Rule(style="dim blue"),
                f"[bold blue]Validation[/bold blue]\n{val['metrics_str']}",
            )
        )
        title: str = f"Itr: {val['itr']}, lr: {val['lr']:.2E}"
        subtitle: str = (
            f"Times - Train/Val: {train['elapsed_time']:.2f}s/{val['elapsed_time']:.2f}s, All: {total_time:.2f}s"
        )

        logger.info(
            Panel(
                content,
                title=title,
                title_align="left",
                subtitle=subtitle,
                subtitle_align="right",
                border_style="dim blue",
                padding=(1, 2),
                width=120,
            )
        )
        logger.info("")

    def train(
        self,
        phase_name: str = "",
        starting_iteration: int = 0,
        ending_iteration: int = 100_000,
        initial_lr: float = 1e-3,
        **kwargs: Tensor,
    ) -> MetricHistory:
        """Train loop for alignment model."""
        # metrics values may include floats, ints, or lists of floats
        # Metrics store lists of floats and ints
        metrics: MetricHistory = {
            k: [] for k in ("train_loss", "train_steps", "val_loss", "val_steps", "train_loss_itr", "val_loss_itr")
        }
        train_loss_buffer = DeferredScalarBuffer(max(1, self.log_interval))

        self.update_iterations_range(starting_iteration, ending_iteration)
        self.update_lr(initial_lr)

        self.set_train_mode()
        use_non_blocking_transfer: bool = self.device.type == "cuda"
        # Initialize persistent dataloader iterators
        train_iter: Iterator[dict[str, Tensor]] = iter(self.train_dataloader)
        val_iter: Iterator[dict[str, Tensor]] = iter(self.val_dataloader)

        total_iters: int = ending_iteration - starting_iteration

        time_train: float = 0.0
        time_all: float = 0.0
        itr_start: float = 0.0

        # assert self.starting_iteration < self.ending_iteration, "Invalid training iteration range"
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]Training Phase {task.fields[phase]}\n"),
            BarColumn(bar_width=40),
            TextColumn("{task.completed}/{task.total} iterations"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            train_task: TaskID = progress.add_task("Training", total=total_iters, phase=phase_name)
            for itr in range(starting_iteration, ending_iteration):
                self.current_iteration = itr

                itr_start = time.time()
                batch_data: dict[str, Tensor]
                try:
                    batch_data = next(train_iter)
                except StopIteration:
                    # Re-initialise when we exhaust the dataloader
                    train_iter = iter(self.train_dataloader)
                    batch_data = next(train_iter)

                states: Tensor = batch_data["batch_states"].to(self.device, non_blocking=use_non_blocking_transfer)
                base_states: Tensor = batch_data["batch_base_states"].to(
                    self.device, non_blocking=use_non_blocking_transfer
                )

                # Get encoded targets and actions if available
                step_kwargs: dict[str, Tensor] = kwargs.copy()
                if "batch_encoded_targets" in batch_data:
                    step_kwargs["encoded_targets"] = batch_data["batch_encoded_targets"].to(
                        self.device, non_blocking=use_non_blocking_transfer
                    )
                if "batch_actions" in batch_data:
                    step_kwargs["actions"] = batch_data["batch_actions"].to(
                        self.device, non_blocking=use_non_blocking_transfer
                    )

                # Reset gradients
                self.optimizer.zero_grad(set_to_none=True)

                step_res: MetricsDict = self.step_model(
                    self.pretrained_encoder,
                    self.alignment_model,
                    self.transition_model,
                    self.decoder,
                    states,
                    base_states,
                    **step_kwargs,
                )
                # Access required loss via RequiredMetricsDict
                req_step_res: RequiredMetricsDict = step_res
                loss: Tensor = req_step_res["loss"]
                loss_steps: list[float] | None = step_res.get("loss_steps")
                train_loss_buffer.append(loss)
                if itr % self.log_interval == 0 and loss_steps is not None:
                    metrics["train_steps"].append(loss_steps)

                loss.backward()
                self.optimizer.step()

                lr_before_update: float = self.get_last_lr()

                # Update learning rate after optimizer step
                self.scheduler.step()
                progress.advance(train_task)

                # Build W&B dict if requested
                train_wandb_metrics: MetricPayload = {}
                if self.use_wandb:
                    train_wandb_metrics = self._build_wandb_dict("Train", itr, lr_before_update, step_res)

                # Stop train timer
                time_train += time.time() - itr_start

                will_validate: bool = (itr % self.log_interval == 0) or (itr == ending_iteration - 1)

                if will_validate:
                    self.log_step(
                        prefix="Train",
                        itr=itr,
                        results=step_res,
                        lr=lr_before_update,
                        elapsed_time=time_train,
                        trigger_display=False,
                    )

                # Log training-only wandb metrics for non-validation iterations
                if self.use_wandb and self.tracking_session is not None and train_wandb_metrics and not will_validate:
                    log_metrics(self.tracking_session, train_wandb_metrics, step=itr)

                if will_validate:
                    train_loss_buffer.flush_into(metrics["train_loss"])
                    # Start validation timer
                    val_start: float = time.time()
                    metrics["train_loss_itr"].append(itr)

                    self.set_eval_mode()
                    with torch.inference_mode():
                        val_batch_data: dict[str, Tensor]
                        try:
                            val_batch_data = next(val_iter)
                        except StopIteration:
                            val_iter = iter(self.val_dataloader)
                            val_batch_data = next(val_iter)
                        val_states: Tensor = val_batch_data["batch_states"].to(
                            self.device, non_blocking=use_non_blocking_transfer
                        )
                        val_base_states: Tensor = val_batch_data["batch_base_states"].to(
                            self.device, non_blocking=use_non_blocking_transfer
                        )

                        # Get encoded targets and actions if available
                        val_step_kwargs: dict[str, Tensor] = kwargs.copy()
                        if "batch_encoded_targets" in val_batch_data:
                            val_step_kwargs["encoded_targets"] = val_batch_data["batch_encoded_targets"].to(
                                self.device, non_blocking=use_non_blocking_transfer
                            )
                        if "batch_actions" in val_batch_data:
                            val_step_kwargs["actions"] = val_batch_data["batch_actions"].to(
                                self.device, non_blocking=use_non_blocking_transfer
                            )

                        val_res: MetricsDict = self.step_model(
                            pretrained_encoder=self.pretrained_encoder,
                            alignment_model=self.alignment_model,
                            transition_model=self.transition_model,
                            decoder=self.decoder,
                            states=val_states,
                            base_states=val_base_states,
                            **val_step_kwargs,
                        )
                    self.set_train_mode()

                    val_wandb_metrics: MetricPayload = {}
                    if self.use_wandb:
                        val_wandb_metrics = self._build_wandb_dict("Validation", itr, lr_before_update, val_res)

                    self.log_step(
                        prefix="Validation",
                        itr=itr,
                        results=val_res,
                        lr=lr_before_update,
                        elapsed_time=time.time() - val_start,
                        trigger_display=False,
                    )

                    # Access required loss via RequiredMetricsDict
                    req_val_res: RequiredMetricsDict = val_res
                    vloss: Tensor = req_val_res["loss"]
                    vloss_steps: list[float] | None = val_res.get("loss_steps")
                    metrics["val_loss"].append(vloss.item())
                    if vloss_steps is not None:
                        metrics["val_steps"].append(vloss_steps)
                    metrics["val_loss_itr"].append(itr)

                    # Log combined wandb metrics when validation is done
                    if (
                        self.use_wandb
                        and self.tracking_session is not None
                        and train_wandb_metrics
                        and val_wandb_metrics
                    ):
                        log_metrics_batched(self.tracking_session, [train_wandb_metrics, val_wandb_metrics], step=itr)

                    if self.wandb_osh_trigger is not None:
                        self.wandb_osh_trigger.maybe_trigger()

                    self.training_metrics.update(metrics)
                    if self.checkpoint_path:
                        self.save_checkpoint()

                    # First accumulation & display
                    time_before_log: float = time.time()
                    time_all += time_before_log - itr_start
                    self.log_step(prefix="Misc", itr=itr, elapsed_time=time_all, trigger_display=True)

                    # Reset all timers after display
                    time_train = 0.0
                    time_all = time.time() - time_before_log  # Keep overhead for next accumulation

                else:
                    # Non-validation single accumulation
                    time_all += time.time() - itr_start

        if self.wandb_osh_trigger is not None and self.wandb_osh_trigger.trigger_on_phase_end:
            self.wandb_osh_trigger.maybe_trigger(force=True)

        return metrics

    def load_checkpoint(self, strict: bool = True) -> dict[str, CheckpointValue]:
        """Load trainer state and separately stored model parameters.

        Args:
            strict: Whether to strictly enforce that the keys in state_dict match.

        Returns:
            Dictionary containing checkpoint metadata and any additional info.
        """
        if not self.checkpoint_path:
            raise ValueError("Checkpoint path not set during initialization.")

        if not pathlib.Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Validate required paths exist in checkpoint
        required_paths: set[str] = {"alignment_model_path"}
        missing_paths: set[str] = required_paths - set(checkpoint.keys())
        if missing_paths:
            raise ValueError(f"Checkpoint missing required model paths: {missing_paths}")

        # Load alignment model from separate .pt file
        alignment_model_path: str = checkpoint["alignment_model_path"]

        # Verify alignment model file exists
        if not pathlib.Path(alignment_model_path).exists():
            raise FileNotFoundError(f"Alignment model file not found: {alignment_model_path}")

        # Load alignment model state
        try:
            alignment_model_state: dict[str, Tensor] = torch.load(alignment_model_path, map_location=self.device)
            self.alignment_model.load_state_dict(alignment_model_state, strict=strict)
            logger.info(f"Alignment model state loaded from {alignment_model_path}")
        except Exception:
            logger.exception("Failed to load alignment model state")
            if strict:
                raise

        # Load pretrained encoder if it exists
        if "pretrained_encoder_path" in checkpoint:
            pretrained_encoder_path = checkpoint["pretrained_encoder_path"]
            if self.pretrained_encoder is not None:
                if pathlib.Path(pretrained_encoder_path).exists():
                    try:
                        encoder_state = torch.load(pretrained_encoder_path, map_location=self.device)
                        self.pretrained_encoder.load_state_dict(encoder_state, strict=strict)
                        logger.info(f"Pretrained encoder state loaded from {pretrained_encoder_path}")
                    except Exception:
                        logger.exception("Failed to load pretrained encoder state")
                        if strict:
                            raise
                else:
                    logger.warning(f"Pretrained encoder file not found: {pretrained_encoder_path}")
            else:
                logger.warning("Checkpoint contains pretrained encoder path but trainer has no pretrained encoder")
        elif self.pretrained_encoder is not None:
            logger.warning("Trainer has pretrained encoder but checkpoint does not contain its path")

        # Load transition model if it exists
        if "transition_model_path" in checkpoint:
            transition_model_path: str = checkpoint["transition_model_path"]
            if self.transition_model is not None:
                if pathlib.Path(transition_model_path).exists():
                    try:
                        transition_state: dict[str, Tensor] = torch.load(
                            transition_model_path, map_location=self.device
                        )
                        self.transition_model.load_state_dict(transition_state, strict=strict)
                        logger.info(f"Transition model state loaded from {transition_model_path}")
                    except Exception:
                        logger.exception("Failed to load transition model state")
                        if strict:
                            raise
                else:
                    logger.warning(f"Transition model file not found: {transition_model_path}")
            else:
                logger.warning("Checkpoint contains transition model path but trainer has no transition model")
        elif self.transition_model is not None:
            logger.warning("Trainer has transition model but checkpoint does not contain its path")

        # Load decoder if it exists
        if "decoder_path" in checkpoint:
            decoder_path: str = checkpoint["decoder_path"]
            if self.decoder is not None:
                if pathlib.Path(decoder_path).exists():
                    try:
                        decoder_state: dict[str, Tensor] = torch.load(decoder_path, map_location=self.device)
                        self.decoder.load_state_dict(decoder_state, strict=strict)
                        logger.info(f"Decoder state loaded from {decoder_path}")

                    except Exception:
                        logger.exception("Failed to load decoder state")
                        if strict:
                            raise
                else:
                    logger.warning(f"Decoder file not found: {decoder_path}")
            else:
                logger.warning("Checkpoint contains decoder path but trainer has no decoder")
        elif self.decoder is not None:
            logger.warning("Trainer has decoder but checkpoint does not contain its path")

        # Load optimizer and scheduler states
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Optimizer state loaded from checkpoint")
            except Exception:
                logger.exception("Failed to load optimizer state")
                if strict:
                    raise

        if "scheduler_state_dict" in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Scheduler state loaded from checkpoint")
            except Exception:
                logger.exception("Failed to load scheduler state")
                if strict:
                    raise

        # Restore random number generator states
        if "torch_rng_state" in checkpoint:
            try:
                torch.set_rng_state(checkpoint["torch_rng_state"])
                logger.info("PyTorch RNG state restored from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to restore PyTorch RNG state: {e}")

        if "numpy_rng_state" in checkpoint:
            try:
                np.random.set_state(checkpoint["numpy_rng_state"])
                logger.info("NumPy RNG state restored from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to restore NumPy RNG state: {e}")

        # Restore trainer configuration and state
        state_mappings: dict[str, str] = {
            "current_iteration": "current_iteration",
            "starting_iteration": "starting_iteration",
            "ending_iteration": "ending_iteration",
            "log_interval": "log_interval",
            "checkpoint_interval": "checkpoint_interval",
            "use_wandb": "use_wandb",
        }

        for checkpoint_key, attr_name in state_mappings.items():
            if checkpoint_key in checkpoint:
                setattr(self, attr_name, checkpoint[checkpoint_key])
                logger.debug(f"Restored {attr_name} from checkpoint")

        # Restore training metrics
        if "training_metrics" in checkpoint:
            self.training_metrics = checkpoint["training_metrics"].copy()
            logger.info("Training metrics restored from checkpoint")

        # Extract metadata for return
        metadata: dict[str, CheckpointValue] = {
            "iteration": checkpoint.get("current_iteration", 0),
            "starting_iteration": checkpoint.get("starting_iteration", 0),
            "ending_iteration": checkpoint.get("ending_iteration", 100_000),
            "device": checkpoint.get("device", str(self.device)),
            "training_metrics": checkpoint.get("training_metrics", {}),
            "save_timestamp": checkpoint.get("save_timestamp"),
            "pytorch_version": checkpoint.get("pytorch_version"),
            "checkpoint_path": checkpoint.get("checkpoint_path", self.checkpoint_path),
            "alignment_model_path": checkpoint.get("alignment_model_path"),
            "pretrained_encoder_path": checkpoint.get("pretrained_encoder_path"),
            "transition_model_path": checkpoint.get("transition_model_path"),
            "decoder_path": checkpoint.get("decoder_path"),
        }

        logger.info(f"Checkpoint loaded from {self.checkpoint_path} (iteration: {metadata['iteration']!r})")

        return metadata

    def save_checkpoint(self) -> None:
        """Save trainer state and model parameters to separate files."""
        if not self.checkpoint_path:
            raise ValueError("Checkpoint path not set during initialization.")

        # Create the checkpoint directory before writing files.
        checkpoint_dir: Path = pathlib.Path(self.checkpoint_path).parent
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Get base filename without extension for model files
        checkpoint_base: str = str(pathlib.Path(self.checkpoint_path).with_suffix(""))

        # Save individual model files
        alignment_model_path: str = f"{checkpoint_base}_alignment_model.pt"
        torch.save(self.alignment_model.state_dict(), alignment_model_path)

        # Prepare trainer state without model parameter dictionaries.
        checkpoint: dict[str, CheckpointValue] = {
            # Model file paths for reference
            "alignment_model_path": alignment_model_path,
            # Optimizer and scheduler states
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            # Random number generator states for reproducibility
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            # Training progress and configuration
            "current_iteration": self.current_iteration,
            "starting_iteration": self.starting_iteration,
            "ending_iteration": self.ending_iteration,
            # Training parameters
            "log_interval": self.log_interval,
            "checkpoint_interval": self.checkpoint_interval,
            # Device and system info
            "device": str(self.device),
            "use_wandb": self.use_wandb,
            # Training metrics and history
            "training_metrics": self.training_metrics.copy() if self.training_metrics else {},
            # Checkpoint metadata
            "checkpoint_path": self.checkpoint_path,
            "save_timestamp": time.time(),
            "pytorch_version": torch.__version__,
        }

        # Save pretrained encoder if it exists
        if self.pretrained_encoder is not None:
            pretrained_encoder_path: str = f"{checkpoint_base}_pretrained_encoder.pt"
            torch.save(self.pretrained_encoder.state_dict(), pretrained_encoder_path)
            checkpoint["pretrained_encoder_path"] = pretrained_encoder_path

        # Save transition model and decoder if they exist (for end-to-end training)
        if self.transition_model is not None:
            transition_model_path: str = f"{checkpoint_base}_transition_model.pt"
            torch.save(self.transition_model.state_dict(), transition_model_path)
            checkpoint["transition_model_path"] = transition_model_path

        if self.decoder is not None:
            decoder_path: str = f"{checkpoint_base}_decoder.pt"
            torch.save(self.decoder.state_dict(), decoder_path)
            checkpoint["decoder_path"] = decoder_path

        # Add CUDA RNG state if available
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        # Add any additional trainer-specific attributes that might be set by subclasses
        for attr_name in dir(self):
            if (
                not attr_name.startswith("_")
                and attr_name not in checkpoint
                and attr_name
                not in {
                    "pretrained_encoder",
                    "alignment_model",
                    "transition_model",
                    "decoder",
                    "optimizer",
                    "scheduler",
                    "train_dataloader",
                    "val_dataloader",
                    "test_dataloader",
                }
                and not callable(getattr(self, attr_name))
            ):
                try:
                    attr_value = getattr(self, attr_name)
                    # Only save serializable attributes
                    if isinstance(attr_value, (int, float, str, bool, list, dict, tuple, type(None))):
                        checkpoint[f"extra_{attr_name}"] = attr_value
                    elif hasattr(attr_value, "copy"):
                        checkpoint[f"extra_{attr_name}"] = attr_value.copy()
                except (AttributeError, TypeError):
                    # Skip attributes that can't be serialized
                    pass

        # Save checkpoint metadata
        torch.save(checkpoint, self.checkpoint_path)

    def step_model(
        self,
        pretrained_encoder: nnModule | None,
        alignment_model: nnModule,
        transition_model: nnModule | None,
        decoder: nnModule | None,
        states: Tensor,
        base_states: Tensor,
        **kwargs: Tensor,
    ) -> MetricsDict:
        """Step the alignment model for a single training step.

        Args:
            pretrained_encoder: Pretrained encoder module (frozen) or None.
            alignment_model: Alignment model to train.
            transition_model: Transition model module (for end-to-end training) or None.
            decoder: Decoder module (for end-to-end training) or None.
            states: Variation states tensor.
            base_states: Corresponding base states tensor.
            **kwargs: Additional keyword arguments for model-specific parameters.

        Returns:
            MetricsDict: Dictionary containing all computed metrics and data.
        """
        raise NotImplementedError
