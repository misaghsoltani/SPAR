"""Neural network wrappers for the Sokoban environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

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
    from torch import Tensor

    from spar.utils.config_utils.config_schema import ModelArchitectureConfig


class EncoderDisc(ABCEncoder):
    """Discrete encoder for Sokoban."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state: Tensor) -> Tensor:
        """Encode a batch of states.

        Args:
            state: Tensor of shape [B, C, H, W].

        Returns:
            Encoded representation tensor.
        """
        encoded: Tensor = self.nnet(state)
        return encoded


class EncoderCont(ABCEncoder):
    """Continuous encoder for Sokoban."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state: Tensor) -> Tensor:
        """Encode a batch of states.

        Args:
            state: Tensor of shape [B, C, H, W].

        Returns:
            Encoded representation tensor.
        """
        encoded: Tensor = self.nnet(state)
        return encoded


class DecoderDisc(ABCDecoder):
    """Discrete decoder for Sokoban."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, latent: Tensor) -> Tensor:
        """Decode a batch of latent codes to images.

        Args:
            latent: Encoded states tensor.

        Returns:
            Reconstructed images tensor.
        """
        decs: Tensor = torch.reshape(latent, (latent.shape[0], 16, 10, 10))
        return self.nnet(decs)


class DecoderCont(ABCDecoder):
    """Continuous decoder for Sokoban."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, latent: Tensor) -> Tensor:
        """Decode a batch of latent codes to images.

        Args:
            latent: Encoded states tensor.

        Returns:
            Reconstructed images tensor.
        """
        decs: Tensor = torch.reshape(latent, (latent.shape[0], 16, 10, 10))
        return self.nnet(decs)


class TransitionModelDisc(ABCTransitionModelDisc):
    """Discrete transition model for Sokoban."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state_encoded: Tensor, action: Tensor) -> Tensor:
        """Predict next encoded state given current encoded state and action.

        Args:
            state_encoded: Encoded state tensor.
            action: Integer action tensor.

        Returns:
            Next encoded state tensor.
        """
        states_conv: Tensor = state_encoded.view(-1, 16, 10, 10)
        action_oh: Tensor = nn.functional.one_hot(action.long(), 4).float()
        action_oh = action_oh.view(-1, 4, 1, 1)
        action_oh = action_oh.repeat(1, 1, states_conv.shape[2], states_conv.shape[3])
        states_action: Tensor = torch.cat((states_conv.float(), action_oh), dim=1)
        states_next: Tensor = self.nnet(states_action)

        return states_next


class TransitionModelCont(ABCTransitionModelCont):
    """Continuous transition model for Sokoban."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state_encoded: Tensor, action: Tensor) -> Tensor:
        """Predict next encoded state given state and action.

        Args:
            state_encoded: Encoded state tensor.
            action: Integer action tensor.

        Returns:
            Next encoded state tensor.
        """
        states_conv: Tensor = state_encoded.view(-1, 16, 10, 10)
        action_oh: Tensor = nn.functional.one_hot(action.long(), 4).float()
        action_oh = action_oh.view(-1, 4, 1, 1)
        action_oh = action_oh.repeat(1, 1, states_conv.shape[2], states_conv.shape[3])
        states_action: Tensor = torch.cat((states_conv.float(), action_oh), dim=1)
        states_next: Tensor = self.nnet(states_action)

        return states_next


class DQN(ABCDQN):
    """DQN model for Sokoban."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state_encoded: Tensor, goal_encoded: Tensor) -> Tensor:
        """Compute Q-values given encoded state and goal.

        Args:
            state_encoded: Encoded current state tensor.
            goal_encoded: Encoded goal state tensor.

        Returns:
            Q-values tensor.
        """
        q_values: Tensor = self.nnet(torch.cat((state_encoded, goal_encoded), dim=1))

        return q_values


class AlignmentModel(ABCAlignmentModel):
    """Alignment model for Sokoban."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state: Tensor) -> Tensor:
        """Align a batch of input images into the canonical domain.

        Args:
            state: Tensor of shape [B, C, H, W].

        Returns:
            Aligned images or latent tensor, depending on architecture.
        """
        encoded: Tensor = self.nnet(state)
        return encoded
