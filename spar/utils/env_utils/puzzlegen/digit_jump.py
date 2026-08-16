from __future__ import annotations

from collections import deque
import pathlib
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy import float32, uint8

from .base import PuzzleEnv, make_metadata

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from random import Random

    from cv2.typing import MatLike
    from numpy.typing import NDArray

    from .base import ActType, InfoType, ObsType


def _solid_rgb_tile(red: float, green: float, blue: float, scale: float = 1.0) -> NDArray[float32]:
    """Create an 8x8 RGB tile with constant channels."""
    tile: NDArray[float32] = np.empty((8, 8, 3), dtype=float32)
    tile[..., 0] = np.float32(red * scale)
    tile[..., 1] = np.float32(green * scale)
    tile[..., 2] = np.float32(blue * scale)
    return tile


def _palette_tile(color: list[float]) -> NDArray[float32]:
    """Create an 8x8 RGB tile from a 3-value color vector."""
    color_vec: NDArray[float32] = np.asarray(color, dtype=float32)
    return np.broadcast_to(color_vec, (8, 8, 3)).copy()


class DigitJump(PuzzleEnv):
    """Digit Jump puzzle environment with numerical grid navigation.

    A puzzle environment where the player jumps across a grid with numbered cells.
    The number in each cell determines how far the player can jump in any direction.
    The goal is to reach the target position by strategically jumping across the grid.

    Attributes:
        render_mode: Visual mode for rendering ("mnist", "basic", "dice", "beta").
        grid: The game grid with numbers indicating jump distances.
        pos: Current player position.
        start: Starting position.
        end: Goal position.
        solution: Sequence of actions for optimal solution.
    """

    # Override metadata to include specific render modes and fps
    metadata = make_metadata(["human", "rgb_array", "basic", "dice", "beta", "mnist"], 4)

    __slots__: tuple[str, ...] = (
        "_already_solved",
        "_blocks_non_player",
        "_blocks_player",
        "_images_cache",
        "_non_player_rgb_cache",
        "_output_buffer",
        "_palette_cache",
        "_player_rgb_cache",
        "_targets_cache",
        "_temp_rgb",
        "_texture_cache",
        "render_mode",
    )

    def __init__(self, *, render_mode: str | None = None, **kwargs: int) -> None:
        """Initialize the Digit Jump environment.

        Args:
            render_mode: Rendering mode ("human", "rgb_array", "basic", "mnist", "dice", "beta").
            **kwargs: Additional arguments passed to PuzzleEnv.
        """
        super().__init__(render_mode=render_mode, **kwargs)
        self.render_mode: str | None = render_mode
        self.grid: NDArray[uint8] | None
        self.pos: tuple[int, int] | None
        self.start: tuple[int, int] | None
        self.end: tuple[int, int] | None
        self.solution: list[ActType] | None = None

        # Initialize cached texture and color attributes
        self._texture_cache: list[NDArray[float32]] | None = None
        self._player_rgb_cache: NDArray[float32] | None = None
        self._non_player_rgb_cache: NDArray[float32] | None = None
        self._palette_cache: list[NDArray[float32]] | None = None
        self._targets_cache: NDArray[float32] | None = None
        self._images_cache: NDArray[float32] | None = None

        # Pre-allocate output buffer for performance. Use RenderFrame alias
        # from gymnasium.core so the method return annotation lines up exactly
        # with PuzzleEnv.get_rgb_array.
        self._output_buffer: NDArray[uint8] | None = None
        self._temp_rgb: NDArray[float32] | None = None

        # Baked 8*8*3 tiles
        self._blocks_non_player: list[NDArray[uint8]] | None = None
        self._blocks_player: list[NDArray[uint8]] | None = None
        self._already_solved: bool | None = False

    def _can_go(self, r: int, c: int) -> bool:
        """Check if position is within grid bounds."""
        return (0 <= r < self.size) and (0 <= c < self.size)

    def _move(self, r: int, c: int, direction: int, dist: int) -> tuple[int, int]:
        """Move from position (r,c) in given direction by distance dist.

        Args:
            r: Current row position.
            c: Current column position.
            direction: Direction index (0=North, 1=East, 2=West, 3=South).
            dist: Distance to move in the given direction.

        Returns:
            New position after movement, or original position if move is invalid.
        """
        if direction < 0 or direction >= len(self.x):
            # Treat out-of-range directions (e.g., no-op) as staying in place.
            return (r, c)

        new_r: int = r + self.y[direction] * dist
        new_c: int = c + self.x[direction] * dist
        return (new_r, new_c) if self._can_go(new_r, new_c) else (r, c)

    def move(self, r: int, c: int, direction: int, dist: int) -> tuple[int, int]:
        """Public movement helper used by wrapper environments."""
        return self._move(r, c, direction, dist)

    def load_rendering_assets_if_needed(self) -> None:
        """Load baked rendering assets on first use."""
        if self._blocks_non_player is None or self._blocks_player is None:
            self._load_rendering_assets()

    def _reset(self) -> tuple[ObsType, InfoType]:
        """Reset the environment and return initial observation."""
        self._create_level()

        # Update agent and target locations for observation space
        assert self.pos is not None, "Position must be set after level creation"
        assert self.end is not None, "Goal position must be set after level creation"
        self._agent_location[0] = self.pos[0]
        self._agent_location[1] = self.pos[1]
        self._target_location[0] = self.end[0]
        self._target_location[1] = self.end[1]

        return self._get_obs(), self._get_info()

    def _create_level(self) -> None:
        """Create a random Digit Jump puzzle level.

        Generates a solvable puzzle by creating a grid with random jump distances
        and finding a valid path from start to end using BFS.

        Raises:
            RuntimeError: If unable to generate a solvable puzzle after max_tries attempts.
        """
        size: int = self.size
        start: tuple[int, int] = (0, 0)
        end: tuple[int, int] = (size - 1, size - 1)
        rng: Random = self._rng
        move: Callable[[int, int, int, int], tuple[int, int]] = self._move
        max_j: int = min(6, size - 1)

        total_cells: int = size * size
        start_idx: int = start[0] * size + start[1]
        end_idx: int = end[0] * size + end[1]

        parents: NDArray[np.int32] = np.empty((total_cells,), dtype=np.int32)
        parent_dir: NDArray[np.int8] = np.empty((total_cells,), dtype=np.int8)
        visited: NDArray[np.bool_] = np.empty((total_cells,), dtype=np.bool_)

        grid: NDArray[uint8] = np.empty((size, size), dtype=uint8)
        found: bool = False
        randint: Callable[[int, int], int] = rng.randint

        for _ in range(self.max_tries):
            for i in range(size):
                row: NDArray[uint8] = grid[i]
                for j in range(size):
                    row[j] = randint(1, max_j)

            parents.fill(-1)
            parent_dir.fill(-1)
            visited.fill(False)

            queue: deque[int] = deque([start_idx])
            visited[start_idx] = True
            parents[start_idx] = start_idx
            found = False

            while queue:
                idx: int = queue.popleft()
                if idx == end_idx:
                    found = True
                    break

                r: int = idx // size
                c: int = idx % size
                step: int = int(grid[r, c])
                for d in range(4):
                    nr, nc = move(r, c, d, step)
                    next_idx: int = nr * size + nc
                    if not visited[next_idx]:
                        visited[next_idx] = True
                        parents[next_idx] = idx
                        parent_dir[next_idx] = np.int8(d)
                        queue.append(next_idx)

            if found:
                break

        if not found:
            raise RuntimeError("Unable to generate a solvable Digit Jump puzzle within max_tries")

        path_dirs: list[int] = []
        cur_idx: int = end_idx
        while cur_idx != start_idx:
            prev_idx: int = parents[cur_idx]
            if prev_idx < 0:
                raise RuntimeError("Broken parent linkage while constructing Digit Jump solution")
            path_dirs.append(int(parent_dir[cur_idx]))
            cur_idx = prev_idx

        path_dirs.reverse()

        # Assign results just as before
        self.grid = grid
        self.pos = start
        self.start = start
        self.end = end
        self.solution = [uint8(d) for d in path_dirs]
        self._already_solved = False

        # Lazy-load render assets
        if self.render_mode in {"rgb_array", "human", "basic", "dice", "beta", "mnist"}:
            self._load_rendering_assets()

    def _load_rendering_asset_caches(self) -> None:
        """Load mode-specific rendering caches before block baking.

        Raises:
            ValueError: If an unknown rendering mode is specified.
        """
        # MNIST / human / rgb_array modes
        if self.render_mode in {"mnist", "human", "rgb_array"}:
            if self._texture_cache is None:
                data_path: Path = pathlib.Path(__file__).resolve().parent / "data/mnist.csv"
                data: NDArray[np.float32] = np.loadtxt(data_path, delimiter=",", dtype=np.float32)
                # Raw MNIST targets & images
                self._targets_cache = data[:, -1].astype(float32, copy=False)
                self._images_cache = 1.0 - data[:, :-1].reshape(-1, 8, 8, 1).astype(float32, copy=False) / 16.0
                # Pick one example per digit 0-6, then drop the zero
                self._texture_cache = [
                    self._images_cache[next(iter(np.where(self._targets_cache == i)[0]))] for i in range(7)
                ][1:]
                # Player / Non-player tint colors
                self._player_rgb_cache = _solid_rgb_tile(0.923, 0.386, 0.209, scale=255.0)
                self._non_player_rgb_cache = _solid_rgb_tile(0.56, 0.692, 0.195, scale=255.0)

        # Basic mode
        elif self.render_mode == "basic":
            if self._player_rgb_cache is None:
                # Use an 8 by 8 player mask.
                self._player_rgb_cache = (
                    np
                    .pad(np.ones((6, 6)), ((1, 1), (1, 1)), mode="constant", constant_values=0)
                    .reshape((8, 8, 1))
                    .astype(float32)
                )
                # Six pastel palette colors
                palette_colors: list[list[float]] = [
                    [132.0, 94.0, 194.0],
                    [214.0, 93.0, 177.0],
                    [255.0, 111.0, 145.0],
                    [255.0, 150.0, 113.0],
                    [255.0, 199.0, 95.0],
                    [249.0, 248.0, 113.0],
                ]
                self._palette_cache = [_palette_tile(p) for p in palette_colors]

        # Dice mode
        elif self.render_mode == "dice":
            if self._texture_cache is None:
                data_path = pathlib.Path(__file__).resolve().parent / "data/faces.csv"
                data = np.loadtxt(data_path, delimiter=",", dtype=np.float32) / 16.0
                images: NDArray[np.float32] = data.reshape((-1, 8, 8, 1)).astype(float32, copy=False)
                self._texture_cache = list(images)
                self._player_rgb_cache = np.full((8, 8, 3), 255.0, dtype=float32)
                palette_colors = [
                    [132.0, 94.0, 194.0],
                    [214.0, 93.0, 177.0],
                    [255.0, 111.0, 145.0],
                    [255.0, 150.0, 113.0],
                    [255.0, 199.0, 95.0],
                    [249.0, 248.0, 113.0],
                ]
                self._palette_cache = [_palette_tile(p) for p in palette_colors]

        # Beta mode
        elif self.render_mode == "beta":
            if self._palette_cache is None:
                # Seven grayscale ramp colors (drop zero)
                self._palette_cache = [
                    _solid_rgb_tile(1.0, 0.0, 0.0, scale=(float(i) * (255.0 / 6.0))) for i in range(1, 7)
                ]
                # Draw the player as a red square.
                center_mask: NDArray[float32] = np.pad(
                    np.ones((4, 4), dtype=float32), ((2, 2), (2, 2)), mode="constant", constant_values=0
                )
                player_rgb: NDArray[float32] = np.zeros((8, 8, 3), dtype=float32)
                player_rgb[..., 1] = center_mask * 255.0
                self._player_rgb_cache = player_rgb

        else:
            raise ValueError(f"Unknown rendering mode: {self.render_mode}")

    def _load_rendering_assets(self) -> None:
        """Load rendering assets based on render mode.

        Textures and colors are loaded once, cached, and reused by later frames.

        Raises:
            FileNotFoundError: If required data files cannot be found.
        """
        # Load & bake all rendering assets into 8*8*3 blocks, then free big caches."""
        try:
            self._load_rendering_asset_caches()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required data file not found for render mode '{self.render_mode}': {e}") from e

        # Bake all small 8*8*3 blocks and free big caches
        if self._blocks_non_player is None:
            mode = self.render_mode
            if mode is None:
                raise ValueError(f"Unknown rendering mode: {mode}")
            # Determine how many digit-tiles we have
            count: int = (
                len(self._texture_cache)
                if self._texture_cache is not None
                else len(self._palette_cache)
                if self._palette_cache is not None
                else 0
            )
            bn: list[NDArray[uint8]] = []
            bp: list[NDArray[uint8]] = []
            if mode in {"mnist", "human", "rgb_array"}:
                assert self._texture_cache is not None
                assert self._non_player_rgb_cache is not None
                assert self._player_rgb_cache is not None
                for i in range(count):
                    tex = self._texture_cache[i]  # 8*8*1
                    bn.append((self._non_player_rgb_cache * tex).astype(uint8))
                    bp.append((self._player_rgb_cache * tex).astype(uint8))
            elif mode == "basic":
                assert self._palette_cache is not None
                assert self._player_rgb_cache is not None
                for i in range(count):
                    pal = self._palette_cache[i]  # 8*8*3
                    bn.append(pal.astype(uint8))
                    bp.append((pal * self._player_rgb_cache).astype(uint8))  # Broadcast 1->3
            elif mode == "dice":
                assert self._texture_cache is not None
                assert self._palette_cache is not None
                assert self._player_rgb_cache is not None
                for i in range(count):
                    tex = self._texture_cache[i]
                    pal = self._palette_cache[i]
                    bn.append((pal * tex).astype(uint8))
                    bp.append((self._player_rgb_cache * tex).astype(uint8))
            else:  # Beta
                assert self._palette_cache is not None
                assert self._player_rgb_cache is not None
                for i in range(count):
                    pal = self._palette_cache[i]
                    bn.append(pal.astype(uint8))
                    bp.append((pal + self._player_rgb_cache).astype(uint8))

            self._blocks_non_player = bn
            self._blocks_player = bp

            # Free caches
            self._texture_cache = None
            self._palette_cache = None
            self._images_cache = None
            self._targets_cache = None
            self._player_rgb_cache = None
            self._non_player_rgb_cache = None

    def _step(self, action: ActType) -> tuple[ObsType, float, bool, bool, InfoType]:
        """Step the environment by applying the given action.

        Args:
            action: The action to take (0=North, 1=East, 2=West, 3=South, 4=No-op).

        Returns:
            Tuple containing (observation, reward, terminated, truncated, info).
        """
        if action in self.action_space and action < 4:
            assert self.pos is not None, "Position not initialized"
            assert self.grid is not None, "Grid not initialized"
            self.pos = self._move(*self.pos, int(action), int(self.grid[self.pos]))

            # Update agent location for observation space
            self._agent_location[0] = self.pos[0]
            self._agent_location[1] = self.pos[1]

        reward: float = 10.0 if (self.pos == self.end and not self._already_solved) else 0.0
        terminated: bool = (self.pos == self.end) and not self._already_solved
        self._already_solved = True if self.pos == self.end else self._already_solved

        return self._get_obs(), reward, terminated, False, self._get_info()

    def get_rgb_array(self) -> NDArray[uint8] | list[NDArray[uint8]] | None:
        """Get the current frame as an RGB array."""
        if self.grid is None or self.pos is None or self.end is None:
            raise RuntimeError("Environment state not initialized")

        size: int = self.size
        pi, pj = self.pos

        # Allocate once per mode
        if self._temp_rgb is None:
            # Full source at 8*8 per cell
            self._temp_rgb = np.empty((size * 8, size * 8, 3), dtype=float32)
        if self._output_buffer is None:
            self._output_buffer = np.empty((64, 64, 3), dtype=uint8)

        bn: list[NDArray[uint8]] | None
        bp: list[NDArray[uint8]] | None
        src = self._temp_rgb
        bn, bp = self._blocks_non_player, self._blocks_player
        assert bn is not None, "Non-player rendering blocks not initialized"
        assert bp is not None, "Player rendering blocks not initialized"

        for i in range(size):
            base_i: int = i * 8
            for j in range(size):
                el: int = int(self.grid[i, j]) - 1
                # Pick the correct 8*8*3 block
                block = bp[el] if (i == pi and j == pj) else bn[el]
                src[base_i : base_i + 8, j * 8 : j * 8 + 8] = block

        # Resize + Clip + Cast -> Buffer
        rescaled: MatLike = cv2.resize(src, (64, 64), interpolation=cv2.INTER_NEAREST)
        np.clip(rescaled, 0, 255, out=rescaled)
        # Copy & Cast
        np.copyto(self._output_buffer, rescaled, casting="unsafe")

        return self._output_buffer

    def close(self) -> None:
        """Clean up resources and close the environment.

        Clears texture caches and temporary buffers to free memory when the
        environment is no longer needed.
        """
        if self._is_closed:
            return

        super().close()

        # Clear cached assets
        self._texture_cache = None
        self._player_rgb_cache = None
        self._non_player_rgb_cache = None
        self._palette_cache = None
        self._targets_cache = None
        self._images_cache = None
        self._output_buffer = None
        self._temp_rgb = None
