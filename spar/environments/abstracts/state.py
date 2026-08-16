"""Abstract base class for states in the SPAR framework."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ABCState(ABC):
    """Abstract base class for states in the SPAR framework."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed: int | None = seed
        self._hash: int | None = None

    @abstractmethod
    def __hash__(self) -> int:
        """Compute the hash of the state.

        Returns:
            Hash value of the state.
        """
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check if two states are equal.

        Args:
            other (ABCState): The other state to compare with.

        Returns:
            bool: True if the states are equal, False otherwise.
        """
        raise NotImplementedError

    def get_opt_path_len(self) -> int | None:
        """Get the length of the optimal path.

        Returns:
            int | None: Length of the optimal path, or None if not available.
        """
        return None

    def get_solution(self) -> list[int] | None:
        """Get the list of actions to be taken to get to the goal.

        Returns:
            List[int]: The list of actions to be taken to get to the goal.
        """
        return None
