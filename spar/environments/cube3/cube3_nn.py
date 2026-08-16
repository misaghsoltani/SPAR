from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from spar.environments.abstracts import (
    ABCDQN,
    ABCAlignmentModel,
    ABCDecoder,
    ABCEncoder,
    ABCTransitionModelCont,
    ABCTransitionModelDisc,
)
from spar.models.factory import ModelFactory

if TYPE_CHECKING:
    from torch import Tensor, nn

    from spar.utils.config_utils.config_schema import ModelArchitectureConfig


class EncoderDisc(ABCEncoder):
    """Encoder class for Cube3 environment."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        """Initialize the Encoder.

        Args:
            cfg (ModelArchitectureConfig): Configuration dictionary.
        """
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through the encoder.

        Args:
            state (Tensor): Input tensor. Shape: [batch_size, 3, 32, 64]

        Returns:
            Tensor: Real-valued encoded tensor. Shape: [batch_size, encoded_dim]
        """
        # Split the 32x64 input horizontally into two 32x32 images
        img1: Tensor = state[:, :, :, :32]  # First image: left half
        img2: Tensor = state[:, :, :, 32:]  # Second image: right half

        # Concatenate along channel dimension to form [batch_size, 6, 32, 32]
        state = torch.cat([img1, img2], dim=1)

        return self.nnet(state)


class EncoderCont(ABCEncoder):
    """Encoder class for Cube3 environment."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        """Initialize the Encoder.

        Args:
            cfg (ModelArchitectureConfig): Configuration dictionary.
        """
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through the encoder.

        Args:
            state (Tensor): Input tensor. Shape: [batch_size, 3, 32, 64]

        Returns:
            Tensor: Real-valued encoded tensor. Shape: [batch_size, encoded_dim]
        """
        # Split the 32x64 input horizontally into two 32x32 images
        img1: Tensor = state[:, :, :, :32]  # First image: left half
        img2: Tensor = state[:, :, :, 32:]  # Second image: right half

        # Concatenate along channel dimension to form [batch_size, 6, 32, 32]
        state = torch.cat([img1, img2], dim=1)

        return self.nnet(state)


class DecoderDisc(ABCDecoder):
    """Decoder class for Cube3 environment."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        """Initialize the Decoder.

        Args:
            cfg (ModelArchitectureConfig): Configuration dictionary.
        """
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, latent: Tensor) -> Tensor:
        """Forward pass of the decoder.

        Args:
            latent: Input latent tensor to decode from encoded representation. Shape: [batch_size, latent_dim]

        Returns:
            Reconstructed tensor in 6-channel format. Shape: [batch_size, 3, 32, 64]
        """
        # Network outputs a flattened tensor that needs to be reshaped to (batch_size, 6, 32, 32)
        network_output: Tensor = self.nnet(latent)

        # Reshape from flattened [B, 2 * 3 * 32 * 32] to [B, 6, 32, 32]
        reconstructed: Tensor = torch.reshape(network_output, (latent.shape[0], 6, 32, 32))
        left_recon: Tensor = reconstructed[:, :3, :, :]  # Left half
        right_recon: Tensor = reconstructed[:, 3:, :, :]  # Right half
        # Concatenate left and right halves to form [batch_size, 3, 32, 64]
        return torch.cat([left_recon, right_recon], dim=-1)


class DecoderCont(ABCDecoder):
    """Decoder class for Cube3 environment."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        """Initialize the Decoder.

        Args:
            cfg (ModelArchitectureConfig): Configuration dictionary.
        """
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, latent: Tensor) -> Tensor:
        """Forward pass of the decoder.

        Args:
            latent: Input latent tensor to decode from encoded representation. Shape: [batch_size, latent_dim]

        Returns:
            Reconstructed tensor in 6-channel format. Shape: [batch_size, 3, 32, 64]
        """
        # Network outputs a flattened tensor that needs to be reshaped to (batch_size, 6, 32, 32)
        network_output: Tensor = self.nnet(latent)

        # Reshape from flattened [B, 2 * 3 * 32 * 32] to [B, 6, 32, 32]
        reconstructed: Tensor = torch.reshape(network_output, (latent.shape[0], 6, 32, 32))
        left_recon: Tensor = reconstructed[:, :3, :, :]  # Left half
        right_recon: Tensor = reconstructed[:, 3:, :, :]  # Right half
        # Concatenate left and right halves to form [batch_size, 3, 32, 64]
        return torch.cat([left_recon, right_recon], dim=-1)


class TransitionModelDisc(ABCTransitionModelDisc):
    """Abstract base class for discrete transition models.

    All subclasses must override forward() to predict next encoded state.
    """

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()

        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state_encoded: Tensor, action: Tensor) -> Tensor:
        """Forward pass of the discrete transition model.

        Args:
            state_encoded: Current encoded state tensor representation. Shape: [batch_size, encoded_dim]
            action: Integer index of the discrete action or tensor representing action indices. Shape: [batch_size]

        Returns:
            Next encoded state tensor with same shape as input state_encoded. Shape: [batch_size, encoded_dim]
        """
        actions_oh: Tensor = F.one_hot(action.long(), 12)
        actions_oh = actions_oh.float()
        actions_oh = actions_oh.view(-1, 12)
        states_actions: Tensor = torch.cat((state_encoded.float(), actions_oh), dim=1)
        states_next: Tensor = self.nnet(states_actions)

        return states_next


class TransitionModelCont(ABCTransitionModelCont):
    """Abstract base class for continuous transition models.

    All subclasses must override forward() to predict next encoded state.
    """

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()

        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state_encoded: Tensor, action: Tensor) -> Tensor:
        """Forward pass of the continuous transition model.

        Args:
            state_encoded: Current encoded state tensor representation
            action: Continuous action tensor with real-valued actions or integer if discrete actions encoded as indices

        Returns:
            Next encoded state tensor with same shape as input state_encoded
        """
        actions_oh: Tensor = F.one_hot(action.long(), 12)
        actions_oh = actions_oh.float()
        actions_oh = actions_oh.view(-1, 12)
        states_actions: Tensor = torch.cat((state_encoded.float(), actions_oh), dim=1)
        states_next: Tensor = self.nnet(states_actions)

        return states_next


class DQN(ABCDQN):
    """DQN class for Cube3 environment."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        """Initialize the DQN.

        Args:
            cfg (ModelArchitectureConfig): Configuration dictionary.
        """
        super().__init__()
        self.dqn: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state_encoded: Tensor, goal_encoded: Tensor) -> Tensor:
        """Forward pass through the DQN.

        Args:
            state_encoded (Tensor): Input tensor representing the encoded state.
            goal_encoded (Tensor): Input tensor representing the encoded goal.

        Returns:
            Tensor: Q-values for each action.
        """
        q_values: Tensor = self.dqn(torch.cat((state_encoded, goal_encoded), dim=1))
        return q_values


class AlignmentModel(ABCAlignmentModel):
    """Alignment model."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        """Initialize the Alignment Model.

        Args:
            cfg (ModelArchitectureConfig): Configuration dictionary.
        """
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through the alignment model.

        Args:
            state (Tensor): Input tensor. Shape: [batch_size, 3, 32, 64]

        Returns:
            Tensor: Encoded tensors.
        """
        # Split the 32x64 input horizontally into two 32x32 images
        img1: Tensor = state[:, :, :, :32]  # First image: left half
        img2: Tensor = state[:, :, :, 32:]  # Second image: right half

        # Concatenate along channel dimension to form [batch_size, 6, 32, 32]
        state_cat: Tensor = torch.cat([img1, img2], dim=1)

        # state shape is now [batch_size, 6, 32, 32]
        return self.nnet(state_cat)
