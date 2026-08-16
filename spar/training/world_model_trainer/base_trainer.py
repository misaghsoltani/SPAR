"""World Model Trainer for SPAR."""

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
    from collections.abc import Callable, Mapping, Sequence
    from logging import Logger
    from typing import Literal

    from numpy.typing import NDArray
    from rich.progress import TaskID
    from torch import Tensor, nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

    from spar.utils.log_utils.wandb_logger import WandbTrackingSession
    from spar.utils.log_utils.wandb_osh import WandbOshSyncTrigger
    from spar.utils.pytorch_utils.model_stripper import CheckpointValue


logger: Logger = getLogger(__name__)


def _prepare_test_rollout_batches(
    state_episodes: list[NDArray[np.float32]],
    action_episodes: list[list[int]],
    start_idxs: NDArray[np.intp],
    num_steps: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Materialize rollout states/actions for model testing with minimal temporaries.

    Args:
        state_episodes: Sampled episodes of states.
        action_episodes: Sampled episodes of actions aligned with ``state_episodes``.
        start_idxs: Per-episode starting indices.
        num_steps: Number of rollout steps.

    Returns:
        Tuple of ``(all_states, all_actions)`` with shapes
        ``[num_steps+1, batch, ...]`` and ``[num_steps, batch]`` respectively.
    """
    batch_size: int = len(state_episodes)
    state_shape: tuple[int, ...] = state_episodes[0].shape[1:]

    all_states_np: NDArray[np.float32] = np.empty((num_steps + 1, batch_size, *state_shape), dtype=np.float32)
    all_actions_np: NDArray[np.int64] = np.empty((num_steps, batch_size), dtype=np.int64)

    batch_idx: int
    episode_start: int
    for batch_idx, (state_episode, action_episode, episode_start_np) in enumerate(
        zip(state_episodes, action_episodes, start_idxs, strict=True)
    ):
        episode_start = int(episode_start_np)
        episode_stop: int = episode_start + num_steps + 1
        all_states_np[:, batch_idx] = state_episode[episode_start:episode_stop]
        all_actions_np[:, batch_idx] = np.asarray(
            action_episode[episode_start : episode_start + num_steps], dtype=np.int64
        )

    return all_states_np, all_actions_np


class RequiredMetricsDict(TypedDict):
    """Required metrics dictionary for operations that need loss."""

    loss: Tensor


class MetricsDict(RequiredMetricsDict, total=False):
    """Metrics dictionary structure for logging."""

    # Loss components
    loss_recon: Tensor | float
    loss_env: Tensor | float
    loss_steps: list[float]

    # Model-specific metrics
    percent_on: float
    mean_act: float
    eq: float
    cos_sim: float
    eq_bit: float
    relative_acc: float
    eq_bit_min: float
    min_relative_acc: float

    # Additional data
    trajectories: list[NDArray[np.float32]] | None
    decoded_states: Tensor
    predicted_states: Tensor


class WorldModelTrainerBase:
    """Base trainer for world models."""

    def __init__(
        self,
        train_dataloader: DataLoader[dict[str, Tensor]],
        val_dataloader: DataLoader[dict[str, Tensor]],
        test_state_episodes: list[NDArray[np.float32]],
        test_action_episodes: list[list[int]],
        *,
        encoder: nn.Module,
        transition_model: nn.Module,
        decoder: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        env_coeff: float | None = None,
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
        training_metrics: dict[str, float | int | list[float]] | None = None,
        end_to_end: bool = False,
        **kwargs: str | int | float | bool,
    ) -> None:
        if isinstance(device, str):
            device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.device: torch.device = device

        self.encoder: nn.Module = encoder
        self.transition_model: nn.Module = transition_model
        self.decoder: nn.Module = decoder

        self.optimizer: Optimizer = optimizer
        self.scheduler: LRScheduler = scheduler

        self.train_dataloader: DataLoader[dict[str, Tensor]] = train_dataloader
        self.val_dataloader: DataLoader[dict[str, Tensor]] = val_dataloader
        self.test_state_episodes: list[NDArray[np.float32]] = test_state_episodes
        self.test_action_episodes: list[list[int]] = test_action_episodes

        # Combined storage for Train/Validation logs. Values can be str/int/float/None
        self.log_outputs: dict[str, dict[str, float | int | str | None]] = {}
        self.log_interval: int = log_interval
        self.checkpoint_interval: int = checkpoint_interval
        self.use_wandb: bool = use_wandb
        self.tracking_session: WandbTrackingSession | None = tracking_session
        self.wandb_osh_trigger: WandbOshSyncTrigger | None = wandb_osh_trigger

        self.env_coeff: float | None = env_coeff

        # Internal state for checkpointing
        self.current_iteration: int = current_iteration
        self.starting_iteration: int = starting_iteration
        self.ending_iteration: int = ending_iteration
        self.checkpoint_path: str = checkpoint_path
        self.training_metrics: dict[str, float | int | list[float]] = (
            training_metrics.copy() if training_metrics else {}
        )

        self.encoder.to(self.device)
        self.transition_model.to(self.device)
        self.decoder.to(self.device)

        self.end_to_end: bool = end_to_end

        # Read scalar keyword arguments into local variables.
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.METRIC_DISPLAY_CONFIG: dict[str, dict[str, str]] = {
            # Loss components
            "loss_recon": {"format": ".2E", "label": "l_recon"},
            "loss_env": {"format": ".2E", "label": "l_env"},
            # Discrete model metrics
            "percent_on": {"format": ".2f", "label": "%on"},
            "eq": {"format": ".2f", "label": "eq"},
            "eq_bit": {"format": ".2f", "label": "eq_bit"},
            "eq_bit_min": {"format": ".2f", "label": "eq_bit_min"},
            # Continuous model metrics
            "mean_act": {"format": ".2f", "label": "act"},
            "cos_sim": {"format": ".2f", "label": "cos_sim"},
            "relative_acc": {"format": ".2f", "label": "rel_acc"},
            "min_relative_acc": {"format": ".2f", "label": "rel_acc_min"},
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

        epoch: int = getattr(sched, "last_epoch", 0)
        gamma: float = getattr(sched, "gamma", 1.0)
        factor: float = 1.0  # multiplicative decay factor

        if name == "ExponentialLR":  # lr * gamma**step
            step: int = getattr(sched, "_step_count", epoch)
            factor = gamma**step

        elif name == "StepLR":  # lr * gamma**(epoch//step_size)
            step_size: int = getattr(sched, "step_size", 1)
            factor = gamma ** (epoch // step_size)

        elif name == "MultiStepLR":  # lr * gamma**(passed milestones)
            milestones: list[int] = list(getattr(sched, "milestones", ()))
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

    def update_env_coeff(self, env_coeff: float | None) -> None:
        """Update the environment coefficient used in loss calculation."""
        self.env_coeff = env_coeff

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
        """Set the model components to evaluation mode."""
        self.encoder.eval()
        self.transition_model.eval()
        self.decoder.eval()

    def set_train_mode(self) -> None:
        """Set the model components to training mode."""
        self.encoder.train()
        self.transition_model.train()
        self.decoder.train()

    def freeze_models(self, model_names: list[Literal["encoder", "transition_model", "decoder"]]) -> None:
        """Freeze specified model components to prevent training.

        Args:
            model_names (list[Literal["encoder", "transition_model", "decoder"]]): List of model names to freeze.
        """
        for model_name in model_names:
            if model_name == "encoder":
                for param in self.encoder.parameters():
                    param.requires_grad = False

            elif model_name == "transition_model":
                for param in self.transition_model.parameters():
                    param.requires_grad = False

            elif model_name == "decoder":
                for param in self.decoder.parameters():
                    param.requires_grad = False

    def unfreeze_models(self, model_names: list[Literal["encoder", "transition_model", "decoder"]]) -> None:
        """Unfreeze specified model components to allow training.

        Args:
            model_names (list[Literal["encoder", "transition_model", "decoder"]]): List of model names to unfreeze.
        """
        for model_name in model_names:
            if model_name == "encoder":
                for param in self.encoder.parameters():
                    param.requires_grad = True

            elif model_name == "transition_model":
                for param in self.transition_model.parameters():
                    param.requires_grad = True

            elif model_name == "decoder":
                for param in self.decoder.parameters():
                    param.requires_grad = True

    def log_step(
        self,
        prefix: str,
        itr: int,
        results: MetricsDict | None = None,
        lr: float | None = None,
        env_coeff: float | None = None,
        elapsed_time: float = 0.0,
        trigger_display: bool = False,
    ) -> None:
        """Store logs for display, optionally prepare W&B, and trigger console output."""
        prefix_l: str = prefix.lower()
        if results is None:
            self.log_outputs[prefix_l] = {"itr": itr, "elapsed_time": elapsed_time}
            if trigger_display:
                self._display_combined(self.log_outputs)

            return

        # Human-readable metrics string
        loss: Tensor = results["loss"]
        loss_steps: list[float] | None = results.get("loss_steps", None)
        metrics_str: str = self._format_metrics_str(loss, loss_steps, results, self.METRIC_DISPLAY_CONFIG)

        # Store for combined display
        self.log_outputs[prefix_l] = {
            "metrics_str": metrics_str,
            "itr": itr,
            "lr": lr,
            "env_coeff": env_coeff,
            "elapsed_time": elapsed_time,
        }

        # If both Train & Validation present, render panel
        if trigger_display:
            self._display_combined(self.log_outputs)

    @staticmethod
    def _format_metrics_str(
        loss: Tensor,
        loss_steps: Sequence[float] | None,
        results: MetricsDict,
        metric_display_config: dict[str, dict[str, str]],
    ) -> str:
        parts: list[str] = [f"loss: {loss.item():.2E}"]

        if loss_steps and len(loss_steps) >= 2:
            parts.extend((f"l_recon: {loss_steps[0]:.2E}", f"l_env: {loss_steps[1]:.2E}"))

        excluded = {"loss", "loss_steps", "trajectories", "decoded_states", "predicted_states"}
        if loss_steps and len(loss_steps) >= 2:
            excluded |= {"loss_recon", "loss_env"}

        for key, val in results.items():
            if key in excluded or val is None:
                continue

            cfg: dict[str, str] | None = metric_display_config.get(key)

            if not cfg:
                continue

            if isinstance(val, torch.Tensor):
                scalar: int | float | bool = val.item()
            elif isinstance(val, (int, float, bool)):
                scalar = val
            else:
                continue
            parts.append(f"{cfg['label']}: {scalar:{cfg['format']}}")

        return ", ".join(parts)

    @staticmethod
    def _build_wandb_dict(
        prefix: str, itr: int, lr: float, env_coeff: float | None, results: MetricsDict
    ) -> dict[str, float | int]:
        prefix_l: str = prefix.lower()
        wandb: dict[str, float | int] = {
            f"{prefix_l}/loss": results["loss"].item(),
            f"{prefix_l}/itr": itr,
            f"{prefix_l}/lr": lr,
            "global_step": itr,
        }

        for k, v in results.items():
            if k == "loss" or v is None:
                continue

            if isinstance(v, (int, float)):
                wandb[f"{prefix_l}/{k}"] = v

            # Only call .item() on single-element tensors
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                wandb[f"{prefix_l}/{k}"] = v.item()

        if env_coeff is not None:
            wandb[f"{prefix_l}/env_coeff"] = env_coeff

        return wandb

    @staticmethod
    def _display_combined(log_outputs: Mapping[str, Mapping[str, float | int | str | None]]) -> None:
        """Display combined Train/Validation logs in a rich panel."""
        train: Mapping[str, float | int | str | None] | None = log_outputs.get("train")
        val: Mapping[str, float | int | str | None] | None = log_outputs.get("validation")
        misc: Mapping[str, float | int | str | None] = log_outputs.get("misc", {})

        if not train or not val:
            logger.info("No training or validation data to display.")
            return

        total_time: float | int | str | None = misc.get("elapsed_time", 0.0)
        content: Align = Align.center(
            Group(
                f"[bold green]Train[/bold green]\n{train['metrics_str']}",
                Rule(style="dim blue"),
                f"[bold blue]Validation[/bold blue]\n{val['metrics_str']}",
            )
        )
        title: str = f"Itr: {val['itr']}, lr: {val['lr']:.2E}, env_coeff: {train['env_coeff']}"
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
        env_coeff: float | None = None,
        _path_len_incr_itr: int | None = None,
        **kwargs: dict[str, float | int | Tensor] | None,
    ) -> dict[str, list[float] | int]:
        """Train loop."""
        if env_coeff is not None:
            assert 0.0 <= env_coeff <= 1.0, "env_coeff must be between 0.0 and 1.0 inclusive."

        if env_coeff is not None and env_coeff <= 0.0:
            self.freeze_models(model_names=["transition_model"])
            self.unfreeze_models(model_names=["encoder", "decoder"])
        elif env_coeff is not None and env_coeff >= 1.0:
            self.freeze_models(model_names=["encoder", "decoder"])
            self.unfreeze_models(model_names=["transition_model"])
        else:
            self.unfreeze_models(model_names=["encoder", "transition_model", "decoder"])

        metrics: dict[str, list[float]] = {k: [] for k in ("train_loss", "val_loss", "train_loss_itr", "val_loss_itr")}
        # Special handling for step metrics which store lists
        metrics_steps: dict[str, list[list[float]]] = {k: [] for k in ("train_steps", "val_steps")}
        train_loss_buffer = DeferredScalarBuffer(max(1, self.log_interval))

        self.update_iterations_range(starting_iteration, ending_iteration)
        self.update_env_coeff(env_coeff)
        self.update_lr(initial_lr)

        self.set_train_mode()
        use_non_blocking_transfer: bool = self.device.type == "cuda"
        train_iter = iter(self.train_dataloader)
        val_iter = iter(self.val_dataloader)

        total_iters: int = ending_iteration - starting_iteration

        time_train: float = 0.0
        time_all: float = 0.0
        itr_start: float = 0.0

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
                    train_iter = iter(self.train_dataloader)
                    batch_data = next(train_iter)

                states: Tensor = batch_data["states"].to(self.device, non_blocking=use_non_blocking_transfer)
                actions: Tensor = batch_data["actions"].to(self.device, non_blocking=use_non_blocking_transfer)
                next_states: Tensor = batch_data["next_states"].to(self.device, non_blocking=use_non_blocking_transfer)
                target_states_source: Tensor = batch_data.get("target_states", states)
                target_states: Tensor = target_states_source.to(self.device, non_blocking=use_non_blocking_transfer)
                target_next_states_source: Tensor = batch_data.get("target_next_states", next_states)
                target_next_states: Tensor = target_next_states_source.to(
                    self.device, non_blocking=use_non_blocking_transfer
                )

                # Reset gradients
                self.optimizer.zero_grad(set_to_none=True)

                step_res: MetricsDict = self.step_model(
                    encoder=self.encoder,
                    transition_model=self.transition_model,
                    decoder=self.decoder,
                    states=states,
                    actions=actions,
                    next_states=next_states,
                    target_states=target_states,
                    target_next_states=target_next_states,
                    env_coeff=self.env_coeff,
                    **kwargs,
                )
                loss: Tensor = step_res["loss"]
                loss_steps: list[float] | None = step_res.get("loss_steps", None)

                train_loss_buffer.append(loss)

                if itr % self.log_interval == 0 and loss_steps is not None:
                    metrics_steps["train_steps"].append(loss_steps)

                loss.backward()
                self.optimizer.step()

                lr_before_update: float = self.get_last_lr()

                # Update learning rate after optimizer step
                self.scheduler.step()
                progress.advance(train_task)

                # Build W&B dict if requested
                train_wandb_metrics: dict[str, float | int] = {}
                if self.use_wandb:
                    train_wandb_metrics = self._build_wandb_dict("Train", itr, lr_before_update, env_coeff, step_res)

                # Stop train timer
                time_train += time.time() - itr_start

                will_validate: bool = (itr % self.log_interval == 0) or (itr == ending_iteration - 1)

                if will_validate:
                    self.log_step(
                        prefix="Train",
                        itr=itr,
                        results=step_res,
                        lr=lr_before_update,
                        env_coeff=self.env_coeff,
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
                        val_states: Tensor = val_batch_data["states"].to(
                            self.device, non_blocking=use_non_blocking_transfer
                        )
                        val_actions: Tensor = val_batch_data["actions"].to(
                            self.device, non_blocking=use_non_blocking_transfer
                        )
                        val_next_states: Tensor = val_batch_data["next_states"].to(
                            self.device, non_blocking=use_non_blocking_transfer
                        )
                        val_target_states_source: Tensor = val_batch_data.get("target_states", val_states)
                        val_target_states: Tensor = val_target_states_source.to(
                            self.device, non_blocking=use_non_blocking_transfer
                        )
                        val_target_next_states_source: Tensor = val_batch_data.get(
                            "target_next_states", val_next_states
                        )
                        val_target_next_states: Tensor = val_target_next_states_source.to(
                            self.device, non_blocking=use_non_blocking_transfer
                        )
                        val_res: MetricsDict = self.step_model(
                            encoder=self.encoder,
                            transition_model=self.transition_model,
                            decoder=self.decoder,
                            states=val_states,
                            actions=val_actions,
                            next_states=val_next_states,
                            target_states=val_target_states,
                            target_next_states=val_target_next_states,
                            env_coeff=env_coeff,
                            **kwargs,
                        )
                    self.set_train_mode()

                    val_wandb_metrics: dict[str, float | int] = {}
                    if self.use_wandb:
                        val_wandb_metrics = self._build_wandb_dict(
                            "Validation", itr, lr_before_update, env_coeff, val_res
                        )

                    self.log_step(
                        prefix="Validation",
                        itr=itr,
                        results=val_res,
                        lr=lr_before_update,
                        elapsed_time=time.time() - val_start,
                        env_coeff=self.env_coeff,
                        trigger_display=False,
                    )

                    vloss, vloss_steps = val_res["loss"], val_res.get("loss_steps", None)
                    metrics["val_loss"].append(vloss.item())
                    if vloss_steps is not None:
                        metrics_steps["val_steps"].append(vloss_steps)
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

        # Combine metrics for return
        combined_metrics: dict[str, list[float] | int] = {}
        combined_metrics.update(metrics)
        # Add step metrics as serialized lists (convert to lists of floats)
        for key, val in metrics_steps.items():
            if val:  # Only add non-empty step metrics
                combined_metrics[key] = [item for sublist in val for item in sublist]

        self.unfreeze_models(model_names=["encoder", "transition_model", "decoder"])
        if self.wandb_osh_trigger is not None and self.wandb_osh_trigger.trigger_on_phase_end:
            self.wandb_osh_trigger.maybe_trigger(force=True)

        return combined_metrics

    def save_checkpoint(self) -> None:
        """Save trainer state and model parameters to separate files."""
        if not self.checkpoint_path:
            raise ValueError("Checkpoint path not set during initialization.")

        # Create the checkpoint directory before writing files.
        checkpoint_dir: pathlib.Path = pathlib.Path(self.checkpoint_path).parent
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Get base filename without extension for model files
        checkpoint_base: str = str(pathlib.Path(self.checkpoint_path).with_suffix(""))

        # Save individual model files
        encoder_path: str = f"{checkpoint_base}_encoder.pt"
        transition_model_path: str = f"{checkpoint_base}_transition_model.pt"
        decoder_path: str = f"{checkpoint_base}_decoder.pt"

        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.transition_model.state_dict(), transition_model_path)
        torch.save(self.decoder.state_dict(), decoder_path)

        # Prepare trainer state without model parameter dictionaries.
        checkpoint: dict[str, CheckpointValue] = {
            # Model file paths for reference
            "encoder_path": encoder_path,
            "transition_model_path": transition_model_path,
            "decoder_path": decoder_path,
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
            "env_coeff": self.env_coeff,
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
                    "encoder",
                    "transition_model",
                    "decoder",
                    "optimizer",
                    "scheduler",
                    "train_dataloader",
                    "val_dataloader",
                    "test_state_episodes",
                    "test_action_episodes",
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
        # logger.info(f"Checkpoint saved to {self.checkpoint_path} at iteration {self.current_iteration}")

    def _load_model_component(
        self, path: str, module: nn.Module, info_label: str, error_label: str, *, strict: bool
    ) -> None:
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"{info_label} model file not found: {path}")
        try:
            state_dict: dict[str, CheckpointValue] = torch.load(path, map_location=self.device)
            module.load_state_dict(state_dict, strict=strict)
            logger.info(f"{info_label} state loaded from {path}")
        except Exception:
            logger.exception(f"Failed to load {error_label} state")
            if strict:
                raise

    @staticmethod
    def _run_with_logging(
        action: Callable[[], None],
        *,
        success_message: str | None,
        failure_message: str,
        raise_on_error: bool,
        warn_on_error: bool = False,
    ) -> None:
        try:
            action()
            if success_message:
                logger.info(success_message)
        except Exception as exc:
            log_func: Callable[[str], None] = logger.warning if warn_on_error else logger.error
            log_func(f"{failure_message}: {exc}")
            if raise_on_error:
                raise

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
        required_paths: set[str] = {"encoder_path", "transition_model_path", "decoder_path"}
        missing_paths: set[str] = required_paths - set(checkpoint.keys())
        if missing_paths:
            raise ValueError(f"Checkpoint missing required model paths: {missing_paths}")

        # Load models from separate .pt files
        encoder_path = checkpoint["encoder_path"]
        transition_model_path = checkpoint["transition_model_path"]
        decoder_path = checkpoint["decoder_path"]

        model_components: tuple[tuple[str, nn.Module, str, str], ...] = (
            (encoder_path, self.encoder, "Encoder", "encoder"),
            (transition_model_path, self.transition_model, "Transition model", "transition model"),
            (decoder_path, self.decoder, "Decoder", "decoder"),
        )

        for component_path, module, info_label, error_label in model_components:
            self._load_model_component(component_path, module, info_label, error_label, strict=strict)

        # Load optimizer and scheduler states
        if "optimizer_state_dict" in checkpoint:
            self._run_with_logging(
                lambda: self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]),
                success_message="Optimizer state loaded from checkpoint",
                failure_message="Failed to load optimizer state",
                raise_on_error=strict,
            )

        if "scheduler_state_dict" in checkpoint:
            self._run_with_logging(
                lambda: self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"]),
                success_message="Scheduler state loaded from checkpoint",
                failure_message="Failed to load scheduler state",
                raise_on_error=strict,
            )

        # Restore random number generator states for reproducibility
        if "torch_rng_state" in checkpoint:
            self._run_with_logging(
                lambda: torch.set_rng_state(checkpoint["torch_rng_state"]),
                success_message="PyTorch RNG state restored from checkpoint",
                failure_message="Failed to restore PyTorch RNG state",
                raise_on_error=False,
                warn_on_error=True,
            )

        if "numpy_rng_state" in checkpoint:
            self._run_with_logging(
                lambda: np.random.set_state(checkpoint["numpy_rng_state"]),
                success_message="NumPy RNG state restored from checkpoint",
                failure_message="Failed to restore NumPy RNG state",
                raise_on_error=False,
                warn_on_error=True,
            )

        if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
            self._run_with_logging(
                lambda: torch.cuda.set_rng_state(checkpoint["cuda_rng_state"]),
                success_message="CUDA RNG state restored from checkpoint",
                failure_message="Failed to restore CUDA RNG state",
                raise_on_error=False,
                warn_on_error=True,
            )

        if "cuda_rng_state_all" in checkpoint and torch.cuda.is_available():
            self._run_with_logging(
                lambda: torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"]),
                success_message="CUDA RNG state (all devices) restored from checkpoint",
                failure_message="Failed to restore CUDA RNG state (all devices)",
                raise_on_error=False,
                warn_on_error=True,
            )

        # Restore trainer configuration and state
        state_mappings: dict[str, str] = {
            "current_iteration": "current_iteration",
            "iteration": "current_iteration",  # Backward compatibility
            "starting_iteration": "starting_iteration",
            "ending_iteration": "ending_iteration",
            "env_coeff": "env_coeff",
            "log_interval": "log_interval",
            "checkpoint_interval": "checkpoint_interval",
            "use_wandb": "use_wandb",
        }

        for checkpoint_key, attr_name in state_mappings.items():
            if checkpoint_key in checkpoint:
                setattr(self, attr_name, checkpoint[checkpoint_key])
                logger.debug(f"Restored {attr_name} from checkpoint")

        # Restore training metrics and history
        if "training_metrics" in checkpoint:
            self.training_metrics = checkpoint["training_metrics"].copy()
            logger.info("Training metrics restored from checkpoint")

        # Restore additional trainer-specific attributes
        extra_attrs: dict[str, CheckpointValue] = {}
        for key, value in checkpoint.items():
            if key.startswith("extra_"):
                attr_name = key[6:]  # Remove "extra_" prefix
                try:
                    setattr(self, attr_name, value)
                    extra_attrs[attr_name] = value
                    logger.debug(f"Restored extra attribute {attr_name} from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to restore extra attribute {attr_name}: {e}")

        # Extract the iteration, metrics, and checkpoint paths for the caller.
        metadata: dict[str, CheckpointValue] = {
            "iteration": checkpoint.get("current_iteration", checkpoint.get("iteration", 0)),
            "starting_iteration": checkpoint.get("starting_iteration", 0),
            "ending_iteration": checkpoint.get("ending_iteration", 100_000),
            "device": checkpoint.get("device", str(self.device)),
            "env_coeff": checkpoint.get("env_coeff", 0.5),
            "training_metrics": checkpoint.get("training_metrics", {}),
            "save_timestamp": checkpoint.get("save_timestamp"),
            "pytorch_version": checkpoint.get("pytorch_version"),
            "checkpoint_path": checkpoint.get("checkpoint_path", self.checkpoint_path),
            "extra_attributes": extra_attrs,
            "encoder_path": checkpoint.get("encoder_path"),
            "transition_model_path": checkpoint.get("transition_model_path"),
            "decoder_path": checkpoint.get("decoder_path"),
        }

        # Add any remaining checkpoint data not explicitly handled
        excluded_keys: set[str] = {
            "encoder_path",
            "transition_model_path",
            "decoder_path",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "torch_rng_state",
            "numpy_rng_state",
            "cuda_rng_state",
            "cuda_rng_state_all",
            "current_iteration",
            "iteration",
            "starting_iteration",
            "ending_iteration",
            "device",
            "env_coeff",
            "log_interval",
            "checkpoint_interval",
            "use_wandb",
            "training_metrics",
            "save_timestamp",
            "pytorch_version",
            "checkpoint_path",
        }

        for key, value in checkpoint.items():
            if key not in excluded_keys and not key.startswith("extra_"):
                metadata[f"unhandled_{key}"] = value

        checkpoint_path_str: str = self.checkpoint_path
        logger.info(f"Checkpoint loaded from {checkpoint_path_str} (iteration: {metadata['iteration']!r})")

        # Log version compatibility warning if needed
        if metadata.get("pytorch_version") and metadata["pytorch_version"] != torch.__version__:
            saved_version = metadata["pytorch_version"]
            if isinstance(saved_version, (bytes, bytearray)):
                saved_version = saved_version.decode()

            logger.warning(
                f"PyTorch version mismatch: checkpoint saved with {saved_version}, "
                f"current version is {torch.__version__}"
            )

        return metadata

    def step_model(
        self,
        encoder: nn.Module,
        transition_model: nn.Module,
        decoder: nn.Module,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
        target_states: Tensor | None = None,
        target_next_states: Tensor | None = None,
        env_coeff: float | None = None,
        **kwargs: dict[str, float | int | Tensor] | None,
    ) -> MetricsDict:
        """Step the model for a single training step.

        Args:
            encoder: Encoder module.
            transition_model: Transition model module.
            decoder: Decoder module.
            states: Current states tensor.
            actions: Actions tensor.
            next_states: Ground truth next states tensor.
            target_states: Reconstruction target for current states (e.g., base/clean variant).
            target_next_states: Reconstruction target for next states.
            env_coeff: Environment coefficient (used by discrete models).
            **kwargs: Additional keyword arguments for model-specific parameters.

        Returns:
            MetricsDict: Dictionary containing all computed metrics and data.
        """
        raise NotImplementedError

    @staticmethod
    def _test_model(
        encoder: nn.Module,
        transition_model: nn.Module,
        all_states: Tensor,
        all_actions: Tensor,
        num_steps: int,
        batch_size: int,
        message: str = "Testing model",
        **kwargs: dict[str, float | int | Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Test the model by rolling out predictions for multiple steps.

        Args:
            encoder: Encoder module.
            transition_model: Transition model module.
            all_states: All states tensor.
            all_actions: All actions tensor.
            num_steps: Number of steps to simulate.
            batch_size: Batch size to use for testing.
            message: Message to print before testing.
            **kwargs: Additional keyword arguments.

        Returns:
            all_eq: Tensor of equality metrics for each step.
            all_eq_bit: Tensor of bitwise equality metrics for each step.
            all_eq_bit_min: Tensor of minimum bitwise equality metrics for each step.
            all_match_flags: Tensor of match flags for each step.
        """
        raise NotImplementedError

    def test_model(self, batch_size: int, num_steps: int | None = None, message: str = "Testing model") -> None:
        """Tests the model by rolling out predictions for multiple steps.

        Args:
            batch_size: Batch size to use for testing.
            num_steps: Number of steps to simulate. If None, uses the max length.
            message: Message to print before testing.
        """
        max_episode_length: int = max(len(actions) for actions in self.test_action_episodes)
        num_steps = max_episode_length if num_steps is None else min(num_steps, max_episode_length)

        episode_lens: NDArray[np.intp] = np.fromiter(
            (state_episode.shape[0] for state_episode in self.test_state_episodes),
            dtype=np.intp,
            count=len(self.test_state_episodes),
        )

        episode_idxs: NDArray[np.intp] = np.random.randint(len(self.test_state_episodes), size=batch_size)
        max_start_offsets: NDArray[np.intp] = episode_lens[episode_idxs] - num_steps - 1
        start_idxs: NDArray[np.intp] = np.rint(np.random.uniform(0.0, 1.0, size=batch_size) * max_start_offsets).astype(
            np.intp, copy=False
        )

        state_episodes: list[NDArray[np.float32]] = [self.test_state_episodes[int(idx)] for idx in episode_idxs]
        action_episodes: list[list[int]] = [self.test_action_episodes[int(idx)] for idx in episode_idxs]

        # Prepare all states/actions with direct contiguous slice copies.
        all_states_np: NDArray[np.float32]
        all_actions_np_int: NDArray[np.int64]
        all_states_np, all_actions_np_int = _prepare_test_rollout_batches(
            state_episodes, action_episodes, start_idxs, num_steps
        )

        all_states: Tensor = torch.tensor(all_states_np, device=self.device).float().contiguous()
        all_actions: Tensor = torch.tensor(all_actions_np_int, device=self.device).float()

        self.set_eval_mode()
        with console.status("Running tests...", spinner="line"), torch.no_grad():
            self._test_model(
                self.encoder, self.transition_model, all_states, all_actions, num_steps, batch_size, message
            )
        self.set_train_mode()
