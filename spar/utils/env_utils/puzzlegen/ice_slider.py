from __future__ import annotations

from collections import deque
import contextlib
import importlib
import pathlib
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .base import PuzzleEnv, make_metadata

if TYPE_CHECKING:
    from random import Random
    from types import ModuleType
    from typing import ClassVar

    from cv2.typing import MatLike
    from networkx import DiGraph, MultiDiGraph
    from numpy.typing import NDArray

    from .base import ActType, InfoType, ObsType


class IceSlider(PuzzleEnv):
    """Ice slider puzzle environment with sliding mechanics.

    A puzzle environment where the player slides on ice until hitting obstacles.
    The goal is to reach the target position by sliding across the grid.

    Attributes:
        easy: Whether to use easy mode (no dead-end checking).
        ice_density: Density parameter for ice generation.
        size: Grid size (inherited from PuzzleEnv).
        grid: The game grid where 0=rock, 1=ice.
        pos: Current player position.
        start: Starting position.
        end: Goal position.
        solution: Sequence of actions for optimal solution.
    """

    # Override metadata to include specific render modes and fps
    metadata: dict[str, list[str] | int | float] = make_metadata(["human", "rgb_array", "basic"], 30)
    pos: tuple[int, int] | None
    start: tuple[int, int] | None
    end: tuple[int, int] | None
    grid: NDArray[np.uint8] | None
    solution: list[ActType] | None

    __slots__: tuple[str, ...] = (
        "_cell_textures",
        "_temp_rgb",
        "_transitions",
        "easy",
        "end",
        "goal_rgb",
        "ice_density",
        "ice_rgb",
        "player_rgb",
        "pos",
        "rock_rgb",
        "start",
    )

    DIR_LOOKUP: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (0, -1), (1, 0))

    _BASIC_TEXTURES: ClassVar[dict[str, NDArray[np.uint8]]] = {
        "rock": np.array([[[255, 0, 0]]], dtype=np.uint8),
        "ice": np.array([[[255, 255, 255]]], dtype=np.uint8),
        "player": np.array([[[0, 255, 0]]], dtype=np.uint8),
        # Goal matches ICE color/texture by design
        "goal": np.array([[[255, 255, 255]]], dtype=np.uint8),
    }
    _RGB_TEXTURE_ASSETS: ClassVar[dict[str, str]] = {
        "rock": "data/rock.png",
        "ice": "data/ice.png",
        "player": "data/player.png",
        "goal": "data/ice.png",
    }
    _RGB_TEXTURES: ClassVar[dict[str, NDArray[np.uint8]]] = {}

    def __init__(
        self,
        ice_density: int = 4,
        easy: bool = False,
        size: int = 8,
        render_mode: str | None = None,
        min_sol_len: int = 8,
        max_tries: int = 1_000_000,
    ) -> None:
        super().__init__(size=size, render_mode=render_mode, min_sol_len=min_sol_len, max_tries=max_tries)
        self.easy: bool = easy
        self.ice_density: int = ice_density

        # Will lazily become a size*size*3 uint8 buffer
        self._temp_rgb: NDArray[np.uint8] | None = None

        # For slide table
        self._transitions: NDArray[np.uint8] | None = None

        # Textures & RGB attrs
        self._cell_textures: dict[str, NDArray[np.uint8]] = {}
        self.rock_rgb: NDArray[np.uint8] | None = None
        self.ice_rgb: NDArray[np.uint8] | None = None
        self.player_rgb: NDArray[np.uint8] | None = None
        self.goal_rgb: NDArray[np.uint8] | None = None
        self._tile_buffer: NDArray[np.uint8]
        self._output_buffer: NDArray[np.uint8]
        self._textures_stack: NDArray[np.uint8]
        self._cell_dims: tuple[int, int]
        self._output_buffer_tex: NDArray[np.uint8]

        # RGB rendering requires the texture assets.
        if self.render_mode in {"rgb_array", "human", "basic"}:
            self._load_textures()

        # Track if we've already given the goal reward
        self._already_solved: bool | None = False

    def _load_textures(self) -> None:
        ct: dict[str, NDArray[np.uint8]]
        if self.render_mode == "basic":
            ct = IceSlider._BASIC_TEXTURES

        elif self.render_mode in {"rgb_array", "human"}:
            if not IceSlider._RGB_TEXTURES:
                base = pathlib.Path(__file__).parent
                d: dict[str, NDArray[np.uint8]] = {}
                for name, path in IceSlider._RGB_TEXTURE_ASSETS.items():
                    img = self._imread(str(base / path))
                    d[name] = img.astype(np.uint8)

                IceSlider._RGB_TEXTURES.clear()
                IceSlider._RGB_TEXTURES.update(d)

            ct = IceSlider._RGB_TEXTURES

        else:
            raise ValueError(f"Unknown rendering style: {self.render_mode}")

        self._cell_textures = ct
        self.rock_rgb = ct["rock"]
        self.ice_rgb = ct["ice"]
        self.player_rgb = ct["player"]
        self.goal_rgb = ct["goal"]

    def clear_rendering_cache(self) -> None:
        """Clear cached arrays derived from textures or render mode."""
        for attr in (
            "_cached_render",
            "_cm",
            "_textures_stack",
            "_tile_buffer",
            "_output_buffer",
            "_output_buffer_tex",
            "_cell_dims",
        ):
            if hasattr(self, attr):
                with contextlib.suppress(Exception):
                    delattr(self, attr)

    def reload_textures(self) -> None:
        """Reload texture arrays for the current render mode."""
        self._load_textures()

    def set_basic_colormap(self, rock_color: NDArray[np.uint8], ice_color: NDArray[np.uint8]) -> None:
        """Set the two-color map used by basic rendering mode.

        Args:
            rock_color: RGB color used for rock cells.
            ice_color: RGB color used for ice cells.
        """
        self._cm = np.array([rock_color, ice_color], dtype=np.uint8)

    def _slide_until(self, r: int, c: int, d: ActType, grid: list[list[bool]] | NDArray[np.bool_]) -> tuple[int, int]:
        """Slide until hitting a rock or boundary.

        Args:
            r (int): Row index of the current position.
            c (int): Column index of the current position.
            d (int): Direction (0=up, 1=right, 2=left, 3=down).
            grid (list[list[bool]] | NDArray[np.bool_]): The grid to slide on.

        Returns:
            tuple[int, int]: The final position after sliding.
        """
        dy, dx = IceSlider.DIR_LOOKUP[d]
        nr, nc = r + dy, c + dx

        # Can go for one step?
        if not (0 <= nr < self.size and 0 <= nc < self.size and grid[nr][nc]):
            return r, c

        # Otherwise keep going until you hit rock or boundary
        while True:
            tr, tc = nr + dy, nc + dx
            if not (0 <= tr < self.size and 0 <= tc < self.size and grid[tr][tc]):
                return nr, nc

            nr, nc = tr, tc

    def _build_transition_table(self) -> None:
        """Precompute landing cell for each (r,c,d)."""
        transitions: NDArray[np.uint8] = np.empty((self.size, self.size, 4, 2), dtype=np.uint8)
        # Rock mask
        rock_mask: NDArray[np.bool_] = np.asarray(self.grid == 0, dtype=np.bool_)
        # Rock cells map to themselves
        rr, cc = np.nonzero(rock_mask)
        transitions[rr, cc, :, 0] = rr[:, None]
        transitions[rr, cc, :, 1] = cc[:, None]

        # Ice cells: slide until stop
        bool_grid: NDArray[np.bool_] = np.asarray(self.grid != 0, dtype=np.bool_)
        for r in range(self.size):
            for c in range(self.size):
                if not rock_mask[r, c]:
                    for d in range(4):
                        nr, nc = self._slide_until(r, c, d, bool_grid)
                        transitions[r, c, d, 0] = nr
                        transitions[r, c, d, 1] = nc

        self._transitions = transitions

    def build_transition_table(self) -> None:
        """Public transition-table builder for wrapper environments."""
        self._build_transition_table()

    def get_transitions_view(self) -> NDArray[np.uint8]:
        """Return a read-only view of the precomputed transition table."""
        if self._transitions is None:
            raise RuntimeError("Transition table not initialized")
        transitions: NDArray[np.uint8] = self._transitions.view()
        transitions.setflags(write=False)
        return transitions

    def set_transitions(self, transitions: NDArray[np.uint8]) -> None:
        """Set a precomputed transition table."""
        self._transitions = transitions

    def _step(self, action: ActType) -> tuple[ObsType, float, bool, bool, InfoType]:
        """Step the environment by applying the given action.

        Processes the specified action (0=up, 1=right, 2=left, 3=down, 4=no-op) to update the agent's position using the
        precomputed transition table. Calculates the reward and determines if the puzzle has been solved, marking the
        episode as terminated when appropriate.

        Args:
            action (int): Index of the movement direction to take (0: up, 1: right, 2: left, 3: down, 4: no-op).

        Returns:
            tuple[ObsType, float, bool, bool, InfoType]:
            A tuple containing:
              - observation (ObsType): Dictionary with 'agent' and 'target' locations.
              - reward (float): The reward earned by taking the action (10 if the puzzle is solved, otherwise 0).
              - terminated (bool): True if the puzzle was just solved on this step.
              - truncated (bool): Always False (no time limits).
              - info: Additional labels or metadata for the state.
        """
        if 0 <= action < 4:
            # Build the transition table on its first use.
            assert self._transitions is not None, "Transition table not initialized. Environment needs reset."
            assert self.pos is not None, "Current position not set. Environment needs reset."
            nr: int
            nc: int
            nr, nc = self._transitions[self.pos[0], self.pos[1], action]
            self.pos = (int(nr), int(nc))

            # Update agent location for observation space
            self._agent_location[0] = self.pos[0]
            self._agent_location[1] = self.pos[1]

        terminated: bool = (self.pos == self.end) and not self._already_solved
        reward: float = 10.0 if terminated else 0.0
        if terminated:
            self._already_solved = True

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _create_level(self) -> None:
        """Create a random Ice Slider puzzle level."""
        size: int = self.size

        rng: Random = self._rng

        # Placeholders
        start: tuple[int, int] = (0, 0)
        end: tuple[int, int] = (size - 1, 0)
        solution: list[ActType] = []
        transitions: NDArray[np.uint8] = np.empty((size, size, 4, 2), dtype=np.uint8)
        nr: int
        nc: int
        found: bool = False
        for _ in range(self.max_tries):
            # Random start/end
            start = (0, rng.randint(0, size - 1))
            end = (size - 1, rng.randint(0, size - 1))

            # Generate the grid
            raw: NDArray[np.uint8] = np.array([
                [self._rng.randint(0, self.ice_density) for _ in range(self.size)] for _ in range(self.size)
            ]).astype(np.uint8)
            raw[0, start[1]] = 1
            raw[-1, end[1]] = 1
            grid_bool: NDArray[np.bool_] = raw != 0

            # Precompute all slide transitions
            transitions = np.empty((size, size, 4, 2), dtype=np.uint8)
            rock_mask: NDArray[np.bool_] = np.logical_not(grid_bool)
            for r in range(size):
                for c in range(size):
                    if rock_mask[r, c]:
                        transitions[r, c, :, :] = [r, c]

                    else:
                        for d in range(4):
                            nr, nc = self._slide_until(r, c, d, grid_bool)
                            transitions[r, c, d] = [nr, nc]

            # BFS to find a solution path
            q: deque[tuple[int, int]] = deque([start])
            visited: set[tuple[int, int]] = {start}
            parents: dict[tuple[int, int], tuple[tuple[int, int], ActType]] = {}
            found = False
            prev_node: tuple[int, int]
            action_taken: ActType
            while q:
                curr: tuple[int, int] = q.popleft()
                if curr == end:
                    rev_actions: list[ActType] = []
                    node: tuple[int, int] = curr
                    while node != start:
                        prev_node, action_taken = parents[node]
                        rev_actions.append(action_taken)
                        node = prev_node
                    solution = list(reversed(rev_actions))
                    found = True
                    break

                r0, c0 = curr
                for d, (dy, dx) in enumerate(IceSlider.DIR_LOOKUP):
                    ar, ac = r0 + dy, c0 + dx
                    if 0 <= ar < size and 0 <= ac < size and grid_bool[ar, ac]:
                        nr, nc = transitions[r0, c0, d]
                        nxt: tuple[int, int] = (int(nr), int(nc))
                        if nxt not in visited:
                            visited.add(nxt)
                            q.append(nxt)
                            parents[nxt] = (curr, d)

            # Validate path length
            if not found or len(solution) <= self.min_sol_len:
                continue

            if self.easy:
                break

            # Hard mode: dead-end SCC check
            nx: ModuleType = importlib.import_module("networkx")
            node_id: dict[tuple[int, int], int] = {
                pos: idx for idx, pos in enumerate(zip(*np.nonzero(raw != 0), strict=False))
            }
            g: MultiDiGraph[int] = nx.MultiDiGraph()
            g.add_nodes_from(node_id.values())

            for (r, c), idx in node_id.items():
                for d in range(4):
                    nr, nc = transitions[r, c, d]
                    g.add_edge(idx, node_id[int(nr), int(nc)])

            cnd: DiGraph[int] = nx.condensation(g)
            s0: int = cnd.graph["mapping"][node_id[start]]
            t0: int = cnd.graph["mapping"][node_id[end]]

            # Is hard enough?
            if any(nx.has_path(cnd, s0, comp) and not nx.has_path(cnd, comp, t0) for comp in cnd):
                break

        if not found:
            raise RuntimeError("Unable to generate an Ice Slider puzzle within max_tries")

        # Finalize
        self.start = start
        self.end = end
        self.grid = np.asarray(raw != 0, dtype=np.uint8)
        self.pos = start
        self.solution = list(solution)
        self._already_solved = False
        self._transitions = transitions

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

    def get_rgb_array(self) -> NDArray[np.uint8] | list[NDArray[np.uint8]] | None:
        """Return a 64*64 RGB image of the current environment.

        Returns:
            NDArray[np.uint8]: A 64*64*3 array representing the rendered RGB image.

        Raises:
            RuntimeError: If the environment state has not been initialized.
        """
        if self.grid is None or self.pos is None or self.end is None:
            raise RuntimeError("Environment state not initialized")

        assert self.goal_rgb is not None, "Goal RGB must be set"
        assert self.player_rgb is not None, "Player RGB must be set"

        if self.render_mode == "basic":
            # Pre-allocate output array
            if not hasattr(self, "_output_buffer"):
                self._output_buffer = np.empty((64, 64, 3), dtype=np.uint8)

            # Lazy initialize color map
            if not hasattr(self, "_cm"):
                # Both rock_rgb and ice_rgb must exist before building the color map.
                # In the grid, 0 -> rock, 1 -> ice
                if self.rock_rgb is None or self.ice_rgb is None:
                    # Fallback colors
                    rock_color: NDArray[np.uint8] = np.array([139, 69, 19], dtype=np.uint8)  # Brown
                    ice_color: NDArray[np.uint8] = np.array([173, 216, 230], dtype=np.uint8)  # Light blue
                else:
                    # Extract a representative pixel from each texture
                    if hasattr(self.rock_rgb, "shape") and getattr(self.rock_rgb, "ndim", 1) > 1:
                        rock_color = self.rock_rgb.reshape(-1, self.rock_rgb.shape[-1])[0][:3].astype(np.uint8)
                    else:
                        rock_color = (
                            self.rock_rgb[:3].astype(np.uint8)
                            if len(self.rock_rgb) >= 3
                            else np.array([139, 69, 19], dtype=np.uint8)
                        )

                    if hasattr(self.ice_rgb, "shape") and getattr(self.ice_rgb, "ndim", 1) > 1:
                        ice_color = self.ice_rgb.reshape(-1, self.ice_rgb.shape[-1])[0][:3].astype(np.uint8)
                    else:
                        ice_color = (
                            self.ice_rgb[:3].astype(np.uint8)
                            if len(self.ice_rgb) >= 3
                            else np.array([173, 216, 230], dtype=np.uint8)
                        )

                # Map 0->rock, 1->ice
                self.set_basic_colormap(rock_color, ice_color)

            temp: NDArray[np.uint8] = self._cm[self.grid]

            # For basic mode, render the goal with the same color as ICE for consistency
            temp[self.end] = self._cm[1]

            if hasattr(self, "player_rgb"):
                player_pixel: NDArray[np.uint8] | None
                # Extract a representative pixel for comparison/assignment
                if self.player_rgb.ndim == 1 and len(self.player_rgb) == temp.shape[-1]:
                    player_pixel = self.player_rgb
                elif self.player_rgb.ndim >= 2:
                    # Flatten and take first pixel if it's a texture
                    player_pixel = self.player_rgb.reshape(-1, self.player_rgb.shape[-1])[0]
                else:
                    player_pixel = None

                # If player color matches ice, force a distinct, visible color
                if player_pixel is not None:
                    pp: NDArray[np.uint8] = player_pixel[: temp.shape[-1]].astype(np.uint8)
                    ice_col = self._cm[1]
                    if pp.shape == ice_col.shape and np.allclose(pp, ice_col, atol=2):
                        pp = np.array([255, 20, 147], dtype=np.uint8)  # Deep pink
                    temp[self.pos] = pp

            # Validate temp array before resize
            if temp.size == 0 or temp.shape[0] == 0 or temp.shape[1] == 0:
                # Create a fallback image if temp is empty
                temp = np.full((8, 8, 3), 128, dtype=np.uint8)  # Gray fallback
            # Perform resize and copy into output buffer
            try:
                resized: MatLike = cv2.resize(temp, (64, 64), interpolation=cv2.INTER_NEAREST)
                self._output_buffer[:] = resized
            except Exception:
                # Fallback to uniform gray if resize fails
                self._output_buffer[:] = np.full((64, 64, 3), 128, dtype=np.uint8)

            return self._output_buffer

        # Use detailed texture approach
        if not hasattr(self, "_tile_buffer"):
            self._tile_buffer = np.empty_like(self.grid, dtype=np.uint8)
            self._output_buffer_tex = np.empty((64, 64, 3), dtype=np.uint8)

        # Initialize stacked textures
        if not hasattr(self, "_textures_stack"):
            assert self.rock_rgb is not None, "Rock texture must be loaded"
            assert self.ice_rgb is not None, "Ice texture must be loaded"
            assert self.player_rgb is not None, "Player texture must be loaded"
            assert self.goal_rgb is not None, "Goal texture must be loaded"

            # Resize textures to compatible shapes before stacking.
            def normalize_texture(texture: NDArray[np.uint8], name: str) -> NDArray[np.uint8]:
                if texture.ndim == 1:
                    # Convert 1D to 2D RGB texture (assume it's RGB triplets)
                    if len(texture) >= 3:
                        # Take first 3 values as RGB and create a 1x1x3 texture
                        return texture[:3].reshape(1, 1, 3)
                    # Fallback gray texture
                    return np.array([128, 128, 128], dtype=np.uint8).reshape(1, 1, 3)
                if texture.ndim == 2:
                    # Add RGB channel dimension
                    return np.stack([texture] * 3, axis=-1)
                if texture.ndim == 3:
                    return texture
                raise ValueError(f"Invalid texture dimension for {name}: {texture.ndim}")

            rock_rgb: NDArray[np.uint8] = normalize_texture(self.rock_rgb, "rock")
            ice_rgb: NDArray[np.uint8] = normalize_texture(self.ice_rgb, "ice")
            player_rgb: NDArray[np.uint8] = normalize_texture(self.player_rgb, "player")
            goal_rgb: NDArray[np.uint8] = normalize_texture(self.goal_rgb, "goal")

            # Resize every texture to the shared spatial dimensions.
            target_h, target_w = ice_rgb.shape[:2]

            def resize_if_needed(texture: NDArray[np.uint8], target_shape: tuple[int, int]) -> NDArray[np.uint8]:
                if texture.shape[:2] != target_shape:
                    resized: MatLike = cv2.resize(
                        texture, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST
                    )
                    return resized.astype(np.uint8)
                return texture

            rock_rgb = resize_if_needed(rock_rgb, (target_h, target_w))
            player_rgb = resize_if_needed(player_rgb, (target_h, target_w))
            goal_rgb = resize_if_needed(goal_rgb, (target_h, target_w))

            self._textures_stack = np.stack(arrays=[rock_rgb, ice_rgb, player_rgb, goal_rgb], axis=0)

        np.copyto(self._tile_buffer, self.grid)

        # Render goal as ICE (index 1) so it matches ice texture by design
        self._tile_buffer[self.end] = 1
        self._tile_buffer[self.pos] = 2

        if not hasattr(self, "_cell_dims"):
            assert self.ice_rgb is not None, "Ice texture must be loaded for cell dimensions"
            # Reshape the texture to two spatial dimensions.
            if self.ice_rgb.ndim >= 2:
                self._cell_dims = self.ice_rgb.shape[:2]
            else:
                # Fallback for 1D arrays - assume square cell
                size: int = int(np.sqrt(len(self.ice_rgb) // 3)) if len(self.ice_rgb) >= 3 else 8
                self._cell_dims = (size, size)

        cell_h, cell_w = self._cell_dims
        board: NDArray[np.uint8] = self._textures_stack[self._tile_buffer]

        final_h, final_w = self.size * cell_h, self.size * cell_w
        board = board.transpose(0, 2, 1, 3, 4).reshape(final_h, final_w, 3)

        # Validate board before resize
        if board.size == 0 or board.shape[0] == 0 or board.shape[1] == 0:
            # Create a fallback image if board is empty
            board = np.full((64, 64, 3), 128, dtype=np.uint8)  # Gray fallback
            cv2.resize(board, (64, 64), dst=self._output_buffer_tex, interpolation=cv2.INTER_NEAREST)
        else:
            # Perform safe resize into output buffer
            try:
                resized_board: MatLike = cv2.resize(board, (64, 64), interpolation=cv2.INTER_NEAREST)
                self._output_buffer_tex[:] = resized_board
            except Exception:
                # Fallback to gray image if resize fails
                self._output_buffer_tex[:] = np.full((64, 64, 3), 128, dtype=np.uint8)

        return self._output_buffer_tex

    def close(self) -> None:
        """Clean up resources and close the environment.

        Clears texture caches, temporary RGB buffers, and transition tables to free memory when the environment is
        no longer needed.
        """
        if self._is_closed:
            return

        super().close()

        # Rebind instead of clearing in place: `_cell_textures` aliases the
        # shared class-level texture registry (`_BASIC_TEXTURES` or
        # `_RGB_TEXTURES`), and clearing it would corrupt every instance
        # constructed afterwards.
        self._cell_textures = {}
        self._temp_rgb = None
        self._transitions = None

        for attr in ("rock_rgb", "ice_rgb", "player_rgb", "goal_rgb"):
            if hasattr(self, attr):
                delattr(self, attr)

    @staticmethod
    def _imread(path: str) -> NDArray[np.float32]:
        """Read an image as an RGB float32 array.

        Args:
            path: Image path.

        Returns:
            The RGB image, or a one-pixel color selected from the filename when
            the file cannot be read.
        """
        bgr_image: MatLike | None = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if bgr_image is None:
            # Choose a one-pixel fallback color from the asset name.
            base: str = pathlib.Path(path).name.lower()
            color: NDArray[np.uint8]
            if "rock" in base:
                color = np.array([139, 69, 19], dtype=np.uint8)  # Brown
            elif "ice" in base:
                color = np.array([173, 216, 230], dtype=np.uint8)  # Light blue
            elif "player" in base:
                color = np.array([255, 20, 147], dtype=np.uint8)  # Deep pink
            elif "goal" in base:
                color = np.array([50, 205, 50], dtype=np.uint8)  # Lime green
            else:
                color = np.array([128, 128, 128], dtype=np.uint8)  # Gray
            fallback: NDArray[np.uint8] = color.reshape(1, 1, 3)
            return fallback.astype(np.float32)

        img: NDArray[np.uint8]
        try:
            img = np.asarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        except Exception:
            # If conversion fails, assume already RGB
            img = np.asarray(bgr_image, dtype=np.uint8)
        return np.asarray(img, dtype=np.float32)
