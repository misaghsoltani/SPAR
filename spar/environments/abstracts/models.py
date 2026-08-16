from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class ABCEncoder(nn.Module, ABC):
    """Abstract base class for encoder models.

    All subclasses must override forward() to produce a real-valued encoding
    and a discrete encoding from an input state tensor.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, state: Tensor) -> Tensor:
        """Forward pass of the encoder.

        Args:
            state: Input state tensor to encode
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Encoded tensor.
        """


class ABCDecoder(nn.Module, ABC):
    """Abstract base class for decoder models.

    All subclasses must override forward() to produce a reconstructed tensor
    from a latent representation.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, latent: Tensor) -> Tensor:
        """Forward pass of the decoder.

        Args:
            latent: Input latent tensor to decode from encoded representation
            **kwargs: Additional keyword arguments.

        Returns:
            Reconstructed tensor (matching original input dimensions).
        """


class ABCDQN(nn.Module, ABC):
    """Abstract base class for Deep Q-Network (DQN) models.

    All subclasses must override forward() to produce Q-values from encoded states and goals.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, state_encoded: Tensor, goal_encoded: Tensor) -> Tensor:
        """Forward pass of the DQN.

        Args:
            state_encoded: Encoded state tensor representing the current environment state.
            goal_encoded: Goal tensor representing target state or objective.
            **kwargs: Additional keyword arguments.

        Returns:
            Q-values tensor of shape [..., num_actions], representing action values.
        """


class ABCTransitionModelCont(nn.Module, ABC):
    """Abstract base class for continuous transition models.

    All subclasses must override forward() to predict next encoded state.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, state_encoded: Tensor, action: Tensor) -> Tensor:
        """Forward pass of the continuous transition model.

        Args:
            state_encoded: Current encoded state tensor representation
            action: Continuous action tensor with real-valued actions or integer if discrete actions encoded as indices
            **kwargs: Additional keyword arguments

        Returns:
            Next encoded state tensor with same shape as input state_encoded
        """


class ABCTransitionModelDisc(nn.Module, ABC):
    """Abstract base class for discrete transition models.

    All subclasses must override forward() to predict next encoded state.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, state_encoded: Tensor, action: Tensor) -> Tensor:
        """Forward pass of the discrete transition model.

        Args:
            state_encoded: Current encoded state tensor representation
            action: Integer index of the discrete action or tensor representing action indices
            **kwargs: Additional keyword arguments

        Returns:
            Next encoded state tensor with same shape as input state_encoded
        """


class ABCAlignmentModel(nn.Module, ABC):
    """Abstract base class for alignment models.

    All subclasses must override forward() to produce an alignment tensor.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, state: Tensor) -> Tensor:
        """Forward pass of the alignment model.

        Args:
            state: Input state tensor to encode
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Encoded tensor.
        """
