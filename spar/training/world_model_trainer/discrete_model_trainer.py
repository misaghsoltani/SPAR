"""Discrete World Model Trainer for SPAR."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
import torch
import torch.nn.functional as F

from spar.training.world_model_trainer.base_trainer import WorldModelTrainerBase
from spar.utils.pytorch_utils.pytorch_models import STEThresh

if TYPE_CHECKING:
    from logging import Logger

    from numpy import float32
    from numpy.typing import NDArray
    from torch import Tensor
    from torch.nn import Module as nnModule
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

    from spar.training.world_model_trainer.base_trainer import MetricsDict
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession
    from spar.utils.log_utils.wandb_osh import WandbOshSyncTrigger


logger: Logger = logging.getLogger(__name__)


class DiscreteWorldModelTrainer(WorldModelTrainerBase):
    """Trainer for discrete world models."""

    def __init__(
        self,
        train_dataloader: DataLoader[dict[str, Tensor]],
        val_dataloader: DataLoader[dict[str, Tensor]],
        test_state_episodes: list[NDArray[float32]],
        test_action_episodes: list[list[int]],
        *,
        encoder: nnModule,
        transition_model: nnModule,
        decoder: nnModule,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        env_coeff: float | None = None,
        device: str | torch.device = "cpu",
        log_interval: int = 100,
        checkpoint_interval: int = 100,
        use_wandb: bool = False,
        checkpoint_path: str = "",
        starting_iteration: int = 0,
        ending_iteration: int = 100_000,
        current_iteration: int = 0,
        training_metrics: dict[str, float | int | list[float]] | None = None,
        end_to_end: bool = False,
        tracking_session: WandbTrackingSession | None = None,
        wandb_osh_trigger: WandbOshSyncTrigger | None = None,
        **kwargs: str | int | float | bool,
    ) -> None:
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_state_episodes=test_state_episodes,
            test_action_episodes=test_action_episodes,
            encoder=encoder,
            transition_model=transition_model,
            decoder=decoder,
            optimizer=optimizer,
            scheduler=scheduler,
            env_coeff=env_coeff,
            device=device,
            log_interval=log_interval,
            checkpoint_interval=checkpoint_interval,
            use_wandb=use_wandb,
            checkpoint_path=checkpoint_path,
            starting_iteration=starting_iteration,
            ending_iteration=ending_iteration,
            current_iteration=current_iteration,
            training_metrics=training_metrics,
            end_to_end=end_to_end,
            tracking_session=tracking_session,
            wandb_osh_trigger=wandb_osh_trigger,
            **kwargs,
        )
        self.ste_module: nnModule | None = STEThresh(threshold=0.5) if end_to_end else None

    def set_train_mode(self) -> None:
        """Set the trainer to training mode."""
        super().set_train_mode()
        if self.ste_module:
            self.ste_module.train()

    def set_eval_mode(self) -> None:
        """Set the trainer to evaluation mode."""
        super().set_eval_mode()
        if self.ste_module:
            self.ste_module.eval()

    @staticmethod
    def _step_model_end_to_end(
        encoder: nnModule,
        transition_model: nnModule,
        decoder: nnModule,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
        ste_module: nnModule,
        target_next_states: Tensor | None = None,
    ) -> MetricsDict:
        """Step the discrete world model for end-to-end training.

        Args:
            encoder: Encoder module.
            transition_model: Transition model module.
            decoder: Decoder module.
            states: Current states tensor.
            actions: Actions tensor.
            next_states: Ground truth next states tensor.
            ste_module: Straight-through estimator module for discrete model.
            target_next_states: Optional target next states tensor for reconstruction loss.

        Returns:
            MetricsDict containing only the loss.
        """
        # Encode current states
        enc_disc: Tensor = encoder(states)

        # Predict next encoding
        pred_enc_n_d: Tensor = ste_module(transition_model(enc_disc, actions))

        # Decode prediction
        pred_dec: Tensor = decoder(pred_enc_n_d)

        target_next_states = next_states if target_next_states is None else target_next_states

        # Reconstruction loss (MSE between predicted decoded state and ground truth next state)
        loss: Tensor = F.mse_loss(pred_dec, target_next_states)

        return {"loss": loss}

    def step_model(
        self,
        encoder: nnModule,
        transition_model: nnModule,
        decoder: nnModule,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
        target_states: Tensor | None = None,
        target_next_states: Tensor | None = None,
        env_coeff: float | None = None,
        **_kwargs: dict[str, float | int | Tensor] | None,
    ) -> MetricsDict:
        """Step the discrete world model for single-step prediction.

        Args:
            encoder: Encoder module.
            transition_model: Transition model module.
            decoder: Decoder module.
            states: Current states tensor.
            actions: Actions tensor.
            next_states: Ground truth next states tensor.
            target_states: Optional target states tensor for reconstruction loss.
            target_next_states: Optional target next states tensor for reconstruction loss.
            env_coeff: Environment coefficient.
            end_to_end: Whether to use end-to-end training (encoder + transition model + decoder).
            ste_module: Straight-through estimator module for discrete model
                (used with transition model in end-to-end training).

        Returns:
            MetricsDict containing all computed metrics and data.
        """
        target_states = states if target_states is None else target_states
        target_next_states = next_states if target_next_states is None else target_next_states

        if self.end_to_end:
            if self.ste_module is None:
                raise ValueError("STE module is required for end-to-end training")
            return DiscreteWorldModelTrainer._step_model_end_to_end(
                encoder, transition_model, decoder, states, actions, next_states, self.ste_module, target_next_states
            )

        # Autoencode current states
        enc_disc: Tensor = encoder(states)
        dec: Tensor = decoder(enc_disc)

        # Autoencode next states
        enc_n_disc: Tensor = encoder(next_states)
        dec_n: Tensor = decoder(enc_n_disc)

        # Reconstruction loss (variant inputs, clean targets)
        loss_recon: Tensor = 0.5 * (F.mse_loss(dec, target_states) + F.mse_loss(dec_n, target_next_states))

        # Active bits
        percent_on: Tensor = 100 * (torch.mean(enc_disc) + torch.mean(enc_n_disc)) * 0.5

        # Flatten for transition model
        # enc_disc = enc_disc.reshape((enc_disc.shape[0], -1))
        enc_n_disc = enc_n_disc.reshape((enc_n_disc.shape[0], -1))

        # Predict next encoding
        pred_enc_n: Tensor = transition_model(enc_disc, actions)

        # Transition model loss
        loss_env: Tensor = 0.5 * (
            F.mse_loss(enc_n_disc, torch.round(pred_enc_n.detach())) + F.mse_loss(enc_n_disc.detach(), pred_enc_n)
        )

        # Accuracy metrics
        eq_t: Tensor = torch.eq(torch.round(pred_enc_n), torch.round(enc_n_disc))
        eq: Tensor = 100 * torch.all(eq_t, dim=1).float().mean()
        eq_bit: Tensor = 100 * eq_t.float().mean()
        eq_bit_min: Tensor = 100 * eq_t.float().mean(dim=1).min()

        # Environment coefficient handling
        if env_coeff is None:
            env_coeff = 0.5  # Default value

        # Total loss
        loss: Tensor = (1 - env_coeff) * loss_recon + env_coeff * loss_env

        return {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_env": loss_env,
            "loss_steps": [loss_recon.item(), loss_env.item()],
            "percent_on": percent_on.item(),
            "eq_bit": eq_bit.item(),
            "eq_bit_min": eq_bit_min.item(),
            "eq": eq.item(),
            "decoded_states": dec,
        }

    @staticmethod
    def _test_model(
        encoder: nnModule,
        transition_model: nnModule,
        all_states: Tensor,
        all_actions: Tensor,
        num_steps: int,
        batch_size: int,
        message: str = "Testing model",
        **_kwargs: dict[str, float | int | Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Test the discrete world model by rolling out predictions for multiple steps."""
        # Encode states
        flat_states: Tensor = all_states.reshape(-1, *all_states.shape[2:])
        encs_all: Tensor = encoder(flat_states)
        encs_all = encs_all.reshape(num_steps + 1, batch_size, -1)
        encs: Tensor = encs_all[0]

        metrics_lines: list[str] = []
        match_count = 0
        eq_bit_mins: list[float] = []

        # For return values
        all_eq: Tensor = torch.zeros(num_steps)
        all_eq_bit: Tensor = torch.zeros(num_steps)
        all_eq_bit_min: Tensor = torch.zeros(num_steps)
        all_match_flags: Tensor = torch.zeros(num_steps, dtype=torch.bool)

        for step in range(num_steps):
            encs_pred: Tensor = torch.round(transition_model(encs, all_actions[step]))
            encs_gt: Tensor = torch.round(encs_all[step + 1])
            eq: Tensor = encs_pred == encs_gt

            eq_all: float = eq.all(dim=1).float().mean().item() * 100
            eq_bit: float = eq.float().mean().item() * 100
            eq_bit_min: float = eq.float().mean(dim=1).min().item() * 100
            match: bool = bool(eq.all(dim=1).all().item())

            # Store return values
            all_eq[step] = eq_all
            all_eq_bit[step] = eq_bit
            all_eq_bit_min[step] = eq_bit_min
            all_match_flags[step] = match

            metrics_lines.append(
                f"[cyan]step {step}[/cyan], "
                f"[green]eq_bit[/green]: {eq_bit:.2f}%, "
                f"[blue]eq_bit_min[/blue]: {eq_bit_min:.2f}%, "
                f"[yellow]eq[/yellow]: {eq_all:.2f}%, "
                f"[magenta]match[/magenta]: {int(match)}"
            )
            eq_bit_mins.append(eq_bit_min)
            match_count += int(match)
            encs = encs_pred

        metrics_lines.extend((
            f"[green]eq_bit_min (all):[/green] {min(eq_bit_mins):.2f}%",
            f"[green]{match_count} out of {num_steps} have all match[/green]",
        ))

        logger.info(
            Panel(
                Align.center(Group(*metrics_lines), width=120),
                title=f"[bold magenta]{message}[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
                width=120,
            )
        )

        return all_eq, all_eq_bit, all_eq_bit_min, all_match_flags
