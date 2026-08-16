from __future__ import annotations

from abc import abstractmethod
import importlib
import random
from typing import TYPE_CHECKING

from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    from typing import TypeAlias

    from gymnasium import Space
    from gymnasium.core import RenderFrame
    from numpy.typing import NDArray
    from pygame import Surface
    from pygame.time import Clock


ObsType: TypeAlias = (
    "dict[str, Box] | Space[NDArray[np.uint8 | np.float32]] | dict[str, NDArray[np.uint8 | np.float32]]"
)
ActType: TypeAlias = "np.signedinteger | np.unsignedinteger | int"
ObsSpaceType: TypeAlias = "Space[ObsType] | Dict"
ActSpaceType: TypeAlias = "Space[ActType] | Discrete[np.int64]"
FrameArray: TypeAlias = "NDArray[np.uint8]"
InfoValue: TypeAlias = "tuple[int, int] | int | bool | None"
InfoType: TypeAlias = "dict[str, InfoValue]"


if TYPE_CHECKING:

    def _render_output(value: FrameArray | list[FrameArray] | None) -> RenderFrame | list[RenderFrame] | None: ...

else:

    def _render_output(value: FrameArray | list[FrameArray] | None) -> FrameArray | list[FrameArray] | None:
        """Preserve the concrete frame payload while exposing Gymnasium's render contract."""
        return value


def make_metadata(render_modes: list[str], render_fps: int) -> dict[str, list[str] | int | float]:
    """Build a metadata dictionary with explicit key/value types."""
    return {"render_modes": render_modes, "render_fps": render_fps}


class PuzzleEnv(Env[ObsType, ActType]):
    """Gymnasium base class for the IceSlider and DigitJump grid puzzles.

    Subclasses implement ``_reset``, ``_step``, and ``_get_rgb_array``. The
    base class owns Gymnasium lifecycle state and the ``human``, ``rgb_array``,
    and ``basic`` render modes.
    """

    # Class-level metadata for gymnasium v1.0.0 compatibility
    metadata: dict[str, list[str] | int | float] = make_metadata(["human", "rgb_array", "basic"], 4)

    __slots__: tuple[str, ...] = (
        "_action_to_direction",
        "_agent_location",
        "_already_solved",
        "_is_closed",
        "_needs_reset",
        "_rng",
        "_target_location",
        "_x",
        "_y",
        "action_space",
        "clock",
        "end",
        "grid",
        "max_tries",
        "min_sol_len",
        "observation_space",
        "pos",
        "pygame_module",
        "render_mode",
        "seed",
        "size",
        "solution",
        "start",
        "window",
        "window_size",
    )

    def __init__(
        self, *, size: int = 8, render_mode: str | None = None, min_sol_len: int = 8, max_tries: int = 1_000_000
    ) -> None:
        """Initialize core configuration and RNG for the puzzle environment.

        Args:
            size: Dimension of the square grid.
            render_mode: Rendering mode (e.g., "human", "rgb_array", "basic").
            min_sol_len: Minimum solution length when generating puzzles.
            max_tries: Maximum attempts to generate a solvable puzzle.

        Note:
            The seed parameter should be passed to reset() instead of __init__
            as per Gymnasium >= 1.0.0 guidelines.
        """
        # Initialize base Gymnasium Env for proper RNG and spec
        super().__init__()

        # Validate render mode
        render_modes = self.metadata["render_modes"]
        if not isinstance(render_modes, list):
            raise TypeError("metadata['render_modes'] must be a list of strings")
        if render_mode is not None and render_mode not in render_modes:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}, got: {render_mode}")

        # Direction offsets (N, E, W, S)
        self._x: tuple[int, int, int, int] = (0, 1, -1, 0)
        self._y: tuple[int, int, int, int] = (-1, 0, 0, 1)

        # Define spaces following gymnasium>=1.0.0 standards
        # Use concrete space instances and leave these attributes unannotated to
        # prevent static-type covariance conflicts with gymnasium.Env.
        self.action_space = Discrete(5)
        self.observation_space = Dict({
            "agent": Box(0, size - 1, shape=(2,), dtype=np.uint8),
            "target": Box(0, size - 1, shape=(2,), dtype=np.uint8),
        })

        self._agent_location: NDArray[np.uint8 | np.float32] = np.empty((2,), dtype=np.uint8)
        self._target_location: NDArray[np.uint8 | np.float32] = np.empty((2,), dtype=np.uint8)

        self._action_to_direction: dict[int, NDArray[np.uint8]] = {
            0: np.array([-1, 0]).astype(np.uint8),  # North (up)
            1: np.array([0, 1]).astype(np.uint8),  # East (right)
            2: np.array([0, -1]).astype(np.uint8),  # West (left)
            3: np.array([1, 0]).astype(np.uint8),  # South (down)
            4: np.array([0, 0]).astype(np.uint8),  # No-op
        }

        # Store render mode
        self.render_mode: str | None = render_mode

        # Initialize internal seed and random generator for backward compatibility
        # Note: Seed should be set via reset() as per Gymnasium >= 1.0.0
        self.seed: int | None = None
        self._rng: random.Random = random.Random()

        # Environment configuration
        self.size: int = size
        self.min_sol_len: int = min_sol_len
        self.max_tries: int = max_tries

        # State flags
        self._needs_reset: bool = True
        self._is_closed: bool = False

        # Game state - to be defined by subclasses
        self.grid: NDArray[np.uint8] | None = None
        self.pos: tuple[int, int] | None = None
        self.start: tuple[int, int] | None = None
        self.end: tuple[int, int] | None = None
        self.solution: list[ActType] | None = None
        self._already_solved: bool | None = None

        self.window_size: int = 512  # The size of the PyGame window
        # If human-rendering is used, 'self.window' will be a reference to the window that we draw to. 'self.clock' will
        # be the clock that limits human-mode rendering to the configured frame rate.
        # They will remain 'None' until human-mode is used for the first time.
        self.pygame_module: ModuleType | None = None
        self.window: Surface | None = None
        self.clock: Clock | None = None

    def _get_obs(self) -> dict[str, NDArray[np.uint8 | np.float32]]:
        """Get the current observation as a dictionary.

        Returns:
            Dictionary with keys 'agent' and 'target', each containing a 2D position as uint8 array.
        """
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self) -> InfoType:
        """Compute auxiliary information for debugging."""
        info: InfoType = {
            "agent_location": tuple(self._agent_location.tolist()) if self.pos is not None else None,
            "target_location": tuple(self._target_location.tolist()) if self.end is not None else None,
            "min_sol_len": self.min_sol_len,
            "max_tries": self.max_tries,
            "needs_reset": self._needs_reset,
            "is_closed": self._is_closed,
        }

        # Add optional game state information if available
        if self.pos is not None:
            info["current_position"] = self.pos
        if self.start is not None:
            info["start_position"] = self.start
        if self.end is not None:
            info["end_position"] = self.end
        if self.solution is not None:
            info["solution_length"] = len(self.solution)
            info["has_solution"] = True
        else:
            info["has_solution"] = False
        if self._already_solved is not None:
            info["already_solved"] = self._already_solved

        return info

    @property
    def x(self) -> tuple[int, ...]:
        """Direction offsets for x-coordinate (N, E, W, S)."""
        return self._x

    @property
    def y(self) -> tuple[int, ...]:
        """Direction offsets for y-coordinate (N, E, W, S)."""
        return self._y

    def reset(self, *, seed: int | None = None, options: InfoType | None = None) -> tuple[ObsType, InfoType]:
        """Reset the environment to a new episode following gymnasium>=1.0.0 API.

        Args:
            seed: Seed for random number generator
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info) where:
            - observation: Initial observation as RGB array
            - info: Dictionary with auxiliary information
        """
        _ = options
        # Call parent reset for proper seeding - this sets up self.np_random
        super().reset(seed=seed)

        self._needs_reset = False

        # Update internal RNG
        self._rng.seed(seed)

        self._already_solved = False

        observation, info = self._reset()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    @abstractmethod
    def _reset(self) -> tuple[ObsType, InfoType]:
        """Subclass-specific reset logic. Must return initial observation."""
        raise NotImplementedError

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, InfoType]:
        """Step the environment with the given action following gymnasium>=1.0.0 API.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info) where:
            - observation: New observation as RGB array
            - reward: Reward for this transition
            - terminated: Whether episode ended due to MDP termination
            - truncated: Whether episode ended due to time limit (always False)
            - info: Dictionary with auxiliary information
        """
        if self._needs_reset:
            raise RuntimeError("Environment must be reset before stepping.")

        if self._is_closed:
            raise RuntimeError("Cannot step a closed environment.")

        # Validate action is in action space
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not in action space {self.action_space}")

        observation, reward, terminated, truncated, info = self._step(action)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    @abstractmethod
    def _step(self, action: ActType) -> tuple[ObsType, float, bool, bool, InfoType]:
        """Subclass-specific step logic."""

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Render the environment based on the render_mode following Gymnasium >=1.0.0 API."""
        if self.render_mode is None:
            return None

        if self._needs_reset:
            raise RuntimeError("Environment must be reset before rendering.")

        if self._is_closed:
            raise RuntimeError("Cannot render a closed environment.")

        if self.render_mode == "human":
            self._render_frame()
            return None

        render_modes: list[str] | int | float = self.metadata["render_modes"]
        if isinstance(render_modes, list) and self.render_mode in render_modes:
            return _render_output(self.get_rgb_array())

        raise ValueError(f"Unsupported render mode: {self.render_mode}")

    @abstractmethod
    def get_rgb_array(self) -> FrameArray | list[FrameArray] | None:
        """Subclass must implement frame extraction (e.g., RGB image)."""
        raise NotImplementedError

    def _render_frame(self) -> None:
        """Render the current frame using pygame."""
        assert self.render_mode == "human", "This method should only be called in human render mode."

        pygame = importlib.import_module("pygame")

        self.pygame_module = pygame

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        window = self.window
        assert window is not None

        if self.clock is None:
            self.clock = pygame.time.Clock()
        clock = self.clock
        assert clock is not None

        canvas: Surface = pygame.Surface((self.window_size, self.window_size))

        # Get RGB array from the subclass implementation and convert to pygame surface
        rgb_val: FrameArray | list[FrameArray] | None = self.get_rgb_array()
        if rgb_val is None:
            return

        # Normalize either a single frame or a stacked frame sequence into one RGB image.
        frame: NDArray[np.uint8] = np.asarray(rgb_val, dtype=np.uint8)
        if frame.size == 0:
            return
        if frame.ndim == 4:
            frame = frame[0]

        # Scale the RGB array to fit the window size
        rgb_surface: Surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        scaled_surface: Surface = pygame.transform.scale(rgb_surface, (self.window_size, self.window_size))
        canvas.blit(scaled_surface, (0, 0))

        # Copy drawings from canvas to the visible window
        window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # Limit human-mode rendering to metadata["render_fps"].
        # The clock adds the delay needed to maintain the target frame rate.
        render_fps: list[str] | int | float = self.metadata["render_fps"]
        if not isinstance(render_fps, int | float):
            raise TypeError("metadata['render_fps'] must be numeric")
        clock.tick(float(render_fps))

    def close(self) -> None:
        """Close the environment and release any rendering resources."""
        self._is_closed = True

        if self.window and self.pygame_module is not None:
            self.pygame_module.display.quit()
            self.pygame_module.quit()

    def get_solution(self) -> list[ActType]:
        """Get the solution path for the current puzzle.

        Returns:
            List of actions representing the solution

        Raises:
            RuntimeError: If environment needs reset or is closed
        """
        if self._needs_reset:
            raise RuntimeError("Environment must be reset before retrieving solution.")

        if self._is_closed:
            raise RuntimeError("Cannot query a closed environment.")

        if self.solution is None:
            raise RuntimeError("No solution available.")

        return self.solution

    def labels(self) -> dict[str, ActType]:
        """Get current agent position labels.

        Returns:
            Dict with keys 'player_x' and 'player_y'.

        Raises:
            RuntimeError: If environment needs reset or is closed
        """
        if self._needs_reset:
            raise RuntimeError("Environment must be reset before querying labels.")

        if self._is_closed:
            raise RuntimeError("Cannot query a closed environment.")

        if self.pos is None:
            raise RuntimeError("Agent position not initialized.")

        return {"player_x": self.pos[0], "player_y": self.pos[1]}

    @classmethod
    def get_metadata(cls) -> dict[str, list[str] | int | float]:
        """Get environment metadata.

        Returns:
            Dictionary containing environment metadata including render_modes
        """
        return cls.metadata.copy()

    @property
    def render_modes(self) -> list[str]:
        """Supported render modes.

        Returns:
            List of supported render mode strings
        """
        render_modes: list[str] | int | float = self.metadata["render_modes"]
        if not isinstance(render_modes, list):
            raise TypeError("metadata['render_modes'] must be a list of strings")
        return render_modes
