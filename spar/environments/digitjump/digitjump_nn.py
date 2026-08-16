"""Neural network wrappers for the DigitJump environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    """Discrete encoder wrapper for DigitJump."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass."""
        return self.nnet(state)


class EncoderCont(ABCEncoder):
    """Continuous encoder wrapper for DigitJump."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass."""
        return self.nnet(state)


class DecoderDisc(ABCDecoder):
    """Discrete decoder wrapper for DigitJump."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, latent: Tensor) -> Tensor:
        """Forward pass."""
        return self.nnet(latent)


class DecoderCont(ABCDecoder):
    """Continuous decoder wrapper for DigitJump."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, latent: Tensor) -> Tensor:
        """Forward pass."""
        return self.nnet(latent)


class TransitionModelDisc(ABCTransitionModelDisc):
    """Discrete transition model wrapper for DigitJump."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state_encoded: Tensor, action: Tensor) -> Tensor:
        """Forward pass."""
        return self.nnet(state_encoded, action)


class TransitionModelCont(ABCTransitionModelCont):
    """Continuous transition model wrapper for DigitJump."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state_encoded: Tensor, action: Tensor) -> Tensor:
        """Forward pass."""
        return self.nnet(state_encoded, action)


class DQN(ABCDQN):
    """DQN head wrapper for DigitJump."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state_encoded: Tensor, goal_encoded: Tensor) -> Tensor:
        """Forward pass."""
        return self.nnet(state_encoded, goal_encoded)


class AlignmentModel(ABCAlignmentModel):
    """Alignment model wrapper for DigitJump."""

    def __init__(self, cfg: ModelArchitectureConfig) -> None:
        super().__init__()
        self.nnet: nn.Sequential = ModelFactory.build_model(cfg)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass."""
        return self.nnet(state)
