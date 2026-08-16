from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import functional as F

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
            state (Tensor): Input tensor. Shape: [batch_size, 3, 32, 32]

        Returns:
            Tensor: Discrete encoded tensors.
        """
        batch_size, channels, height, width = state.shape
        # Stack two side-by-side images along the channel dimension -> [batch_size, 3, 32, 32]
        # Split width and stack along channels
        state = state.view(batch_size, channels, height, 2, width).permute(0, 3, 1, 2, 4)
        state = state.contiguous().view(batch_size, 3, height, width)

        # state shape is now [batch_size, 3, 32, 32]
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
            state (Tensor): Input tensor.

        Returns:
            Tensor: Real-valued encoded tensor.
        """
        batch_size, channels, height, width = state.shape
        # Stack two side-by-side images along the channel dimension -> [batch_size, 3, 32, 32]
        # Split width and stack along channels
        state = state.view(batch_size, channels, height, 2, width).permute(0, 3, 1, 2, 4)
        state = state.contiguous().view(batch_size, 3, height, width)

        # state shape is now [batch_size, 3, 32, 32]
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
            latent: Input latent tensor to decode from encoded representation

        Returns:
            Reconstructed tensor (matching original input dimensions).
        """
        batch_size = latent.shape[0]
        # Last layer of self.nnet is nn.Unflatten(1, (3, 32, 32))
        reconstructed = self.nnet(latent)
        # Reshape to match original input dimensions: [batch_size, 3, 32, 32]
        return reconstructed.view(batch_size, 3, 32, 32)


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
            latent: Input latent tensor to decode from encoded representation

        Returns:
            Reconstructed tensor (matching original input dimensions).
        """
        batch_size = latent.shape[0]
        # Last layer of self.nnet is nn.Unflatten(1, (3, 32, 32))
        reconstructed = self.nnet(latent)
        # Reshape to match original input dimensions: [batch_size, 3, 32, 32]
        return reconstructed.view(batch_size, 3, 32, 32)


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
            state_encoded: Current encoded state tensor representation
            action: Integer index of the discrete action or tensor representing action indices

        Returns:
            Next encoded state tensor with same shape as input state_encoded
        """
        # Pad state tensor with zeros on the right for action one-hot slots
        # Allocates a tensor of shape (B, state_dim + num_actions)
        input_tensor = F.pad(state_encoded, (0, 12))  # constant pad = 0
        # Scatter to set the one-hot in the padded region
        # Offset indices by state_dim to target the correct columns
        idx = action.long().unsqueeze(1) + 400
        input_tensor.scatter_(1, idx, 1.0)
        return self.nnet(input_tensor)


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
        # Pad state tensor with zeros on the right for action one-hot slots
        # Allocates a tensor of shape (B, state_dim + num_actions)
        input_tensor = F.pad(state_encoded, (0, 12))  # constant pad = 0
        # Scatter to set the one-hot in the padded region
        # Offset indices by state_dim to target the correct columns
        idx = action.long().unsqueeze(1) + 400
        input_tensor.scatter_(1, idx, 1.0)
        return self.nnet(input_tensor)


class DQN(ABCDQN):
    """DQN class for Cube3 environment."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        """Initialize the DQN.

        Args:
            cfg (ModelArchitectureConfig): Configuration dictionary.
        """
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state_encoded: Tensor, goal_encoded: Tensor) -> Tensor:
        """Forward pass through the DQN.

        Args:
            state_encoded (Tensor): Input tensor representing the encoded state.
            goal_encoded (Tensor): Input tensor representing the encoded goal.

        Returns:
            Tensor: Q-values for each action.
        """
        return self.nnet(state_encoded, goal_encoded)


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
            state (Tensor): Input tensor. Shape: [batch_size, 3, 32, 32]

        Returns:
            Tensor: Encoded tensors.
        """
        batch_size, channels, height, width = state.shape
        # Stack two side-by-side images along the channel dimension -> [batch_size, 3, 32, 32]
        # Split width and stack along channels
        state_new: Tensor = state.view(batch_size, channels, height, 2, width).permute(0, 3, 1, 2, 4)
        state_new = state_new.contiguous().view(batch_size, 3, height, width)

        # state shape is now [batch_size, 3, 32, 32]
        return self.nnet(state_new)
