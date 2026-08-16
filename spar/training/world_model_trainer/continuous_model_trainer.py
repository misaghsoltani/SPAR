"""Continuous World Model Trainer for SPAR."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
import torch
import torch.nn.functional as F

from spar.training.world_model_trainer.base_trainer import WorldModelTrainerBase

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


class ContinuousWorldModelTrainer(WorldModelTrainerBase):
    """Trainer for continuous world models."""

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

    @staticmethod
    def _step_model_end_to_end(
        encoder: nnModule,
        transition_model: nnModule,
        decoder: nnModule,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
        target_next_states: Tensor | None = None,
    ) -> MetricsDict:
        """Step the continuous world model for end-to-end training.

        Args:
            encoder: Encoder module.
            transition_model: Transition model module.
            decoder: Decoder module.
            states: Current states tensor.
            actions: Actions tensor.
            next_states: Ground truth next states tensor.
            target_next_states: Optional target next states tensor for reconstruction loss.

        Returns:
            Tuple containing (loss, loss_recon, loss_env, mean_act, cosine_sim,
                             rel_acc, rel_acc_min, dec).
        """
        # Autoencode current states
        enc_cont: Tensor = encoder(states)

        # # Flatten for transition model
        # enc_cont = enc_cont.reshape((enc_cont.shape[0], -1))
        pred_enc_n: Tensor = transition_model(enc_cont, actions)

        # Decode prediction
        pred_dec: Tensor = decoder(pred_enc_n)

        target_next_states = next_states if target_next_states is None else target_next_states

        # Reconstruction loss
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
        """Step the continuous world model for single-step prediction.

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

        Returns:
            MetricsDict containing all computed metrics and data.
        """
        target_states = states if target_states is None else target_states
        target_next_states = next_states if target_next_states is None else target_next_states

        if self.end_to_end:
            # End-to-end training: directly predict next states from current states and actions
            result: MetricsDict = ContinuousWorldModelTrainer._step_model_end_to_end(
                encoder, transition_model, decoder, states, actions, next_states, target_next_states
            )
            return result

        # Autoencode current states
        enc_cont: Tensor = encoder(states)
        dec: Tensor = decoder(enc_cont)

        # Autoencode next states
        enc_n_cont: Tensor = encoder(next_states)
        dec_n: Tensor = decoder(enc_n_cont)

        # Reconstruction loss (variant inputs, clean targets)
        loss_recon: Tensor = 0.5 * (F.mse_loss(dec, target_states) + F.mse_loss(dec_n, target_next_states))

        # Mean activation magnitude
        mean_act: Tensor = 100 * (torch.mean(torch.abs(enc_cont)) + torch.mean(torch.abs(enc_n_cont))) / 2.0

        # Transition metrics consume one flattened feature vector per sample.
        # Decoding uses enc_cont above. Metrics below use flattened views.
        enc_n_cont_flat: Tensor = enc_n_cont.view(enc_n_cont.size(0), -1)

        # Predict next encoding
        pred_enc_n: Tensor = transition_model(enc_cont, actions)
        pred_enc_n_flat: Tensor = pred_enc_n.view(pred_enc_n.size(0), -1)

        # Symmetric transition loss stops gradients on each target side once.
        loss_env: Tensor = 0.5 * (
            F.mse_loss(enc_n_cont_flat, pred_enc_n_flat.detach())
            + F.mse_loss(enc_n_cont_flat.detach(), pred_enc_n_flat)
        )

        # Similarity metrics on flattened representations
        cosine_sim: Tensor = F.cosine_similarity(pred_enc_n_flat, enc_n_cont_flat, dim=1, eps=1e-8)
        cos_sim: Tensor = 100 * cosine_sim.mean()
        denom: Tensor = torch.clamp(enc_n_cont_flat.abs(), min=1e-8)
        rel_err: Tensor = (pred_enc_n_flat - enc_n_cont_flat).abs() / denom
        relative_acc: Tensor = 100 * (1 - rel_err.mean()).clamp(0, 1)
        min_relative_acc: Tensor = 100 * (1 - rel_err.mean(dim=1).max()).clamp(0, 1)

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
            "relative_acc": relative_acc.item(),
            "min_relative_acc": min_relative_acc.item(),
            "mean_act": mean_act.item(),
            "cos_sim": cos_sim.item(),
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
        """Test the continuous world model by rolling out predictions for multiple steps."""
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
            encs_pred: Tensor = transition_model(encs, all_actions[step])
            encs_gt: Tensor = encs_all[step + 1]

            cosine_sim: Tensor = F.cosine_similarity(encs_pred, encs_gt, dim=1)
            eq_all: float = cosine_sim.mean().item() * 100

            relative_error = torch.abs(encs_pred - encs_gt) / (torch.abs(encs_gt) + 1e-8)
            eq_bit: float = (1 - relative_error.mean()).item() * 100
            eq_bit_min: float = (1 - relative_error.mean(dim=1).max()).item() * 100

            match: bool = eq_all > 99.0

            # Store return values
            all_eq[step] = eq_all
            all_eq_bit[step] = eq_bit
            all_eq_bit_min[step] = eq_bit_min
            all_match_flags[step] = match

            metrics_lines.append(
                f"[cyan]step {step}[/cyan], "
                f"[green]rel_acc[/green]: {eq_bit:.2f}%, "
                f"[blue]rel_acc_min[/blue]: {eq_bit_min:.2f}%, "
                f"[yellow]cos_sim[/yellow]: {eq_all:.2f}%, "
                f"[magenta]match[/magenta]: {int(match)}"
            )
            eq_bit_mins.append(eq_bit_min)
            match_count += int(match)
            encs = encs_pred

        metrics_lines.extend((
            f"[green]rel_acc_min (all):[/green] {min(eq_bit_mins):.2f}%",
            f"[green]{match_count} out of {num_steps} have high similarity[/green]",
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
