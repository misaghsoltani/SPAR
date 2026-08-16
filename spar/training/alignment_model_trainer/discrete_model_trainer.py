"""Alignment Model Trainer for SPAR."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from spar.utils.pytorch_utils.pytorch_models import STEThresh

from .base_trainer import AlignmentModelTrainerBase

if TYPE_CHECKING:
    from logging import Logger

    from torch import Tensor, nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

    from spar.training.alignment_model_trainer.base_trainer import MetricsDict
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession
    from spar.utils.log_utils.wandb_osh import WandbOshSyncTrigger
    from spar.utils.pytorch_utils.model_stripper import CheckpointValue


logger: Logger = getLogger(__name__)


class DiscreteAlignmentModelTrainer(AlignmentModelTrainerBase):
    """Trainer for discrete alignment models."""

    def __init__(
        self,
        train_dataloader: DataLoader[dict[str, Tensor]],
        val_dataloader: DataLoader[dict[str, Tensor]],
        test_dataloader: DataLoader[dict[str, Tensor]] | None = None,
        *,
        pretrained_encoder: nn.Module | None,
        alignment_model: nn.Module,
        transition_model: nn.Module | None = None,
        decoder: nn.Module | None = None,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: str | torch.device = "cpu",
        log_interval: int = 100,
        checkpoint_interval: int = 100,
        use_wandb: bool = False,
        checkpoint_path: str = "",
        starting_iteration: int = 0,
        ending_iteration: int = 100_000,
        current_iteration: int = 0,
        training_metrics: dict[str, CheckpointValue] | None = None,
        end_to_end: bool = False,
        freeze_pretrained_models: bool = True,
        is_transition_frozen: bool = False,
        is_decoder_frozen: bool = False,
        tracking_session: WandbTrackingSession | None = None,
        wandb_osh_trigger: WandbOshSyncTrigger | None = None,
        **kwargs: CheckpointValue,
    ) -> None:
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            pretrained_encoder=pretrained_encoder,
            alignment_model=alignment_model,
            transition_model=transition_model,
            decoder=decoder,
            optimizer=optimizer,
            scheduler=scheduler,
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
            freeze_pretrained_models=freeze_pretrained_models,
            is_transition_frozen=is_transition_frozen,
            is_decoder_frozen=is_decoder_frozen,
            tracking_session=tracking_session,
            wandb_osh_trigger=wandb_osh_trigger,
            **kwargs,
        )

        # Create STE module for discrete models in end-to-end training
        self.ste_module: nn.Module | None = STEThresh(threshold=0.5) if end_to_end else None

    def set_train_mode(self) -> None:
        """Set the alignment model to training mode."""
        super().set_train_mode()
        if self.ste_module:
            self.ste_module.train()

    def set_eval_mode(self) -> None:
        """Set the alignment model to evaluation mode."""
        super().set_eval_mode()
        if self.ste_module:
            self.ste_module.eval()

    @staticmethod
    def _step_model_end_to_end(
        alignment_model: nn.Module,
        transition_model: nn.Module,
        decoder: nn.Module,
        states: Tensor,
        actions: Tensor,
        target_base_states: Tensor,
        ste_module: nn.Module,
    ) -> MetricsDict:
        """Step the discrete alignment model for end-to-end training.

        Args:
            alignment_model: Alignment model to train (acts as encoder).
            transition_model: Transition model module.
            decoder: Decoder module.
            states: Current variation states tensor.
            actions: Actions tensor.
            target_base_states: Target base state images (raw next state images).
            ste_module: Straight-through estimator module for discrete model.

        Returns:
            MetricsDict containing only the loss for end-to-end training.
        """
        # Encode current variation states using alignment model
        enc_disc: Tensor = alignment_model(states)

        # Predict next states's encoding
        pred_enc_n: Tensor = ste_module(transition_model(enc_disc, actions))

        # Decode prediction to get predicted next base state
        pred_base_state: Tensor = decoder(pred_enc_n)

        # Calculate reconstruction loss (MSE between predicted base state and target base state)
        loss: Tensor = F.mse_loss(target_base_states, pred_base_state)

        return {"loss": loss}

    def step_model(
        self,
        pretrained_encoder: nn.Module | None,
        alignment_model: nn.Module,
        transition_model: nn.Module | None,
        decoder: nn.Module | None,
        states: Tensor,
        base_states: Tensor,
        **kwargs: Tensor,
    ) -> MetricsDict:
        """Step the discrete alignment model for single-step prediction.

        Args:
            pretrained_encoder: Pretrained encoder module (frozen) or None.
            alignment_model: Alignment model to train.
            transition_model: Transition model module (for end-to-end training) or None.
            decoder: Decoder module (for end-to-end training) or None.
            states: Variation states tensor.
            base_states: Corresponding base states tensor.
            **kwargs: Additional keyword arguments including actions for end-to-end training.

        Returns:
            MetricsDict containing all computed metrics and data.
        """
        _ = pretrained_encoder
        # Check if we're in end-to-end training mode
        if self.end_to_end and "actions" in kwargs:
            if transition_model is None or decoder is None or self.ste_module is None:
                raise ValueError("End-to-end training requires both transition_model and decoder")
            # In end-to-end mode, base_states are the raw next state images, not encoded
            actions: Tensor = kwargs["actions"]
            return self._step_model_end_to_end(
                alignment_model, transition_model, decoder, states, actions, base_states, self.ste_module
            )

        # Standard alignment training (current state alignment)
        # Get target encodings - use precomputed if available, otherwise encode online
        enc_target_disc: Tensor = (
            kwargs["encoded_targets"] if "encoded_targets" in kwargs else self.get_encoded_targets(base_states)
        )

        # Get predicted encodings from alignment model
        enc_pred: Tensor = alignment_model(states)
        enc_pred_disc: Tensor = torch.round(enc_pred)

        # Calculate alignment loss (MSE between predicted and target encodings)
        loss: Tensor = F.mse_loss(enc_pred, enc_target_disc)

        # Active bits (percent_on)
        percent_on: Tensor = 100 * (torch.mean(enc_pred_disc) + torch.mean(enc_target_disc)) * 0.5

        # Accuracy metrics (matching world model trainer)
        eq_bits: Tensor = torch.eq(enc_pred_disc, enc_target_disc)
        eq: Tensor = 100 * torch.all(eq_bits, dim=1).float().mean()
        eq_bit: Tensor = 100 * eq_bits.float().mean()
        eq_bit_min: Tensor = 100 * eq_bits.float().mean(dim=1).min()

        return {
            "loss": loss,
            "loss_steps": [loss.item()],
            "percent_on": percent_on.item(),
            "eq_bit": eq_bit.item(),
            "eq_bit_min": eq_bit_min.item(),
            "eq": eq.item(),
            "predicted_encodings": enc_pred,
            "target_encodings": enc_target_disc,
        }
