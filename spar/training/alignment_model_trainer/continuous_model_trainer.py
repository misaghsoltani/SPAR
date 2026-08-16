"""Alignment Model Trainer for SPAR."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

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


class ContinuousAlignmentModelTrainer(AlignmentModelTrainerBase):
    """Trainer for continuous alignment models."""

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

    @staticmethod
    def _step_model_end_to_end(
        alignment_model: nn.Module,
        transition_model: nn.Module,
        decoder: nn.Module,
        states: Tensor,
        actions: Tensor,
        target_base_states: Tensor,
    ) -> MetricsDict:
        """Step the continuous alignment model for end-to-end training.

        Args:
            alignment_model: Alignment model to train (acts as encoder).
            transition_model: Transition model module.
            decoder: Decoder module.
            states: Current variation states tensor.
            actions: Actions tensor.
            target_base_states: Target base state images (raw next state images).

        Returns:
            MetricsDict containing only the loss for end-to-end training.
        """
        # Encode current variation states using alignment model
        enc_cont: Tensor = alignment_model(states)

        # Predict next encoding using transition model
        pred_enc_n: Tensor = transition_model(enc_cont, actions)

        # Decode prediction to get predicted next base state
        pred_base_state: Tensor = decoder(pred_enc_n)

        # if current_iteration % 100 == 0:
        #     testing_next_pred_base_state_np = np.clip(pred_base_state[0].detach().cpu().numpy(), 0, 1)
        #     testing_next_target_base_states_np = np.clip(target_base_states[0].detach().cpu().numpy(), 0, 1)
        #     testing_current_state_np = np.clip(states[0].detach().cpu().numpy(), 0, 1)

        #     plt.figure(figsize=(12, 4))
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(testing_current_state_np.transpose(1, 2, 0))
        #     plt.title("Current State")
        #     plt.axis("off")

        #     plt.subplot(1, 3, 2)
        #     plt.imshow(testing_next_pred_base_state_np.transpose(1, 2, 0))
        #     plt.title("Predicted Next State")
        #     plt.axis("off")

        #     plt.subplot(1, 3, 3)
        #     plt.imshow(testing_next_target_base_states_np.transpose(1, 2, 0))
        #     plt.title("Target Next State")
        #     plt.axis("off")

        #     # save to file
        #     plt.savefig("alignment_visualization.png")
        #     plt.close()

        # Calculate reconstruction loss (MSE between predicted base state and target base state)
        loss: Tensor = F.mse_loss(pred_base_state, target_base_states)

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
        """Step the continuous alignment model for single-step prediction.

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
        if self.end_to_end:
            if transition_model is None or decoder is None:
                raise ValueError("End-to-end training requires both transition_model and decoder")
            if "actions" not in kwargs:
                raise ValueError("End-to-end training requires actions in the data")
            # In end-to-end mode, base_states are the raw next state images, not encoded
            return self._step_model_end_to_end(
                alignment_model, transition_model, decoder, states, kwargs["actions"], base_states
            )

        # Standard alignment training (current state alignment)
        # Get target encodings - use precomputed if available, otherwise encode online
        target_encodings: Tensor
        if "encoded_targets" in kwargs:
            target_encodings = kwargs["encoded_targets"]
        else:
            target_encodings = self.get_encoded_targets(base_states)

        # Get predicted encodings from alignment model
        predicted_encodings: Tensor = alignment_model(states)

        # Calculate alignment loss (MSE between predicted and target encodings)
        loss: Tensor = F.mse_loss(predicted_encodings, target_encodings)

        # Flatten features so metrics operate on one vector per sample.
        pred_flat: Tensor = predicted_encodings.view(predicted_encodings.size(0), -1)
        tgt_flat: Tensor = target_encodings.view(target_encodings.size(0), -1)

        # Cosine similarity (per-sample), then mean
        cos_sim_per_sample: Tensor = F.cosine_similarity(pred_flat, tgt_flat, dim=1, eps=1e-8)
        mean_cosine_similarity: Tensor = cos_sim_per_sample.mean()

        # L1 distance
        l1_distance: Tensor = torch.mean(torch.abs(pred_flat - tgt_flat))

        # Relative error (normalized by target magnitude)
        relative_error: Tensor = torch.mean(torch.abs(pred_flat - tgt_flat) / (torch.abs(tgt_flat) + 1e-8))

        # Percentage of "well-aligned" samples (high cosine similarity)
        # high_similarity_threshold: float = 0.95
        percent_aligned: Tensor = 100 * (cos_sim_per_sample > 0.95).float().mean()

        return {
            "loss": loss,
            "loss_steps": [loss.item()],
            "cosine_similarity": mean_cosine_similarity.item(),
            "l1_distance": l1_distance.item(),
            "relative_error": relative_error.item(),
            "percent_aligned": percent_aligned.item(),
            "predicted_encodings": predicted_encodings,
            "target_encodings": target_encodings,
        }
