"""IceSliderEnv environment for the SPAR framework."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

from spar.environments.abstracts import ABCEnvironment, ABCState
from spar.utils.env_utils.effects_core import EffectStage
from spar.utils.env_utils.puzzlegen import IceSlider

from .iceslider_nn import (
    AlignmentModel,
    DecoderCont,
    DecoderDisc,
    EncoderCont,
    EncoderDisc,
    TransitionModelCont,
    TransitionModelDisc,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import ClassVar

    from numpy.typing import NDArray
    from torch import nn

    from spar.utils.config_utils.config_schema import (
        AlignmentModelConfig,
        DecoderConfig,
        EncoderConfig,
        EnvModelConfig,
        ModelConfig,
    )
    from spar.utils.env_utils.effects_core import EffectValue, ImageArray, Pipeline, StagePipelines


class IceSliderState(ABCState):
    """State representation for the Ice Slider environment.

    The state stores the immutable grid, player and goal positions, a solution
    path, and an optional transition table.

    Attributes:
        grid: Compressed grid representation where 0=rock, 1=ice.
        pos: Current player position as (row, col).
        goal: Goal position as (row, col).
        solution: Cached optimal solution path.
        hash: Cached hash value to avoid recomputing the grid digest.
        size: Grid dimensions.
        _transitions: Precomputed transition table for sliding mechanics.
    """

    __slots__: tuple[str, ...] = ("_hash", "_transitions", "goal", "grid", "pos", "size", "solution")

    def __init__(
        self,
        grid: NDArray[np.uint8],
        pos: tuple[int, int],
        goal: tuple[int, int],
        solution: Sequence[int] | NDArray[np.uint8] | None = None,
        size: int | None = None,
    ) -> None:
        """Initialize the Ice Slider state.

        Args:
            grid: Binary grid where 0=rock, 1=ice.
            pos: Current player position as (row, col).
            goal: Goal position as (row, col).
            solution: Optional precomputed solution path.
            size: Grid size (inferred from grid if not provided).
        """
        super().__init__()

        grid_view: NDArray[np.uint8] = np.asarray(grid, dtype=np.uint8).view()
        grid_view.setflags(write=False)
        self.grid: NDArray[np.uint8] = grid_view
        self.pos: tuple[int, int] = pos
        self.goal: tuple[int, int] = goal
        if solution is None:
            sol_view = np.empty(0, dtype=np.uint8)
        else:
            sol_view = np.asarray(solution, dtype=np.uint8).reshape(-1).view()
        sol_view.setflags(write=False)
        self.solution: NDArray[np.uint8] = sol_view
        self.size: int = size if size is not None else grid.shape[0]
        self._hash: int | None = None
        self._transitions: NDArray[np.uint8] | None = None

    @classmethod
    def from_transition(cls, state: IceSliderState, pos: tuple[int, int]) -> IceSliderState:
        """Create a transitioned state from validated immutable storage.

        Args:
            state: Source state whose grid, goal, solution, and size are preserved.
            pos: Position reached by the transition.

        Returns:
            A fresh state with the same array-view aliasing as the public constructor.
        """
        transitioned: IceSliderState = cls.__new__(cls)
        transitioned.seed = None
        transitioned.grid = state.grid.view()
        transitioned.pos = pos
        transitioned.goal = state.goal
        transitioned.solution = state.solution.view()
        transitioned.size = state.size
        transitioned._hash = None
        transitioned._transitions = None
        return transitioned

    def __hash__(self) -> int:
        """Compute the hash of the state using grid and positions.

        Returns:
            Hash value of the state.
        """
        if hasattr(self, "_hash") and self._hash is not None:
            return self._hash

        grid_hash: int = hash(self.grid.tobytes())
        pos_hash: int = hash(self.pos)
        goal_hash: int = hash(self.goal)
        self._hash = hash((grid_hash, pos_hash, goal_hash))

        return self._hash

    def __eq__(self, other: object) -> bool:
        """Check if two states are equal.

        Args:
            other: Another state to compare with.

        Returns:
            True if the states are equal, False otherwise.
        """
        if not isinstance(other, IceSliderState):
            return False

        return self.pos == other.pos and self.goal == other.goal and np.array_equal(self.grid, other.grid)

    def get_solution(self) -> list[int] | None:
        """Get the solution to the puzzle.

        Returns:
            The list of actions to solve the puzzle.
        """
        return self.solution.tolist()

    def get_opt_path_len(self) -> int | None:
        """Get the length of the optimal path.

        Returns:
            The length of the optimal path.
        """
        return int(self.solution.size)

    def _set_pos(self, x: int, y: int) -> bool:
        """Set the position of the player.

        Args:
            x: The x-coordinate (column).
            y: The y-coordinate (row).

        Returns:
            Whether the position was set. A rock in the target cell prevents the move.
        """
        if not (0 <= y < self.size and 0 <= x < self.size):
            return False

        if self.grid[y, x] == 0:  # Rock cell
            return False

        self.pos = (y, x)
        self._hash = None  # Invalidate hash cache
        return True

    def _get_grid(self) -> NDArray[np.uint8]:
        """Get the grid representation of the puzzle.

        Returns:
            The grid representation of the puzzle.
        """
        return self.grid.copy()


class IceSliderEnv(ABCEnvironment[IceSliderState]):
    """Ice Slider environment implementation.

    A move slides the player until an obstacle stops it. Transition tables are
    cached by grid contents.
    """

    moves: ClassVar[dict[int, str]] = {0: "UP", 1: "RIGHT", 2: "LEFT", 3: "DOWN", 4: "NO-OP"}

    # Direction lookup for sliding mechanics
    _DIR_LOOKUP: ClassVar[tuple[tuple[int, int], ...]] = (
        (-1, 0),  # UP
        (0, 1),  # RIGHT
        (0, -1),  # LEFT
        (1, 0),  # DOWN
    )

    def __init__(self, shape: int = 64) -> None:
        """Initialize the Ice Slider environment.

        Args:
            shape: Shape of the environment (default: 64). Used for observation size.
        """
        super().__init__()
        self._shape: int = shape
        self._ice_slider_generator: IceSlider | None = None
        self._ice_slider_cache: dict[tuple[int, str | None], IceSlider] = {}
        # LRU cache for transition tables keyed by grid bytes
        self._transitions_cache: OrderedDict[bytes, NDArray[np.uint8]] = OrderedDict()
        # Max number of cached grids (tunable)
        self._transitions_cache_max: int = 64

    def _get_transitions_for_grid(self, grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Return precomputed transitions for a given grid, using an LRU cache.

        If not cached, build a temporary IceSlider, compute its transition table and cache it.
        """
        key: bytes = grid.tobytes()
        transitions: NDArray[np.uint8]
        # Move key to end if present (mark as recently used)
        if key in self._transitions_cache:
            transitions = self._transitions_cache.pop(key)
            self._transitions_cache[key] = transitions
            return transitions

        # Not cached: build transitions
        size = int(grid.shape[0])
        ice_slider: IceSlider = self._get_ice_slider_instance(size=size, render_mode=None)
        ice_slider.grid = grid
        ice_slider.build_transition_table()
        transitions = ice_slider.get_transitions_view()

        # Insert into cache and evict oldest if over capacity
        self._transitions_cache[key] = transitions
        if len(self._transitions_cache) > self._transitions_cache_max:
            self._transitions_cache.popitem(last=False)

        return transitions

    @staticmethod
    def get_env_name() -> str:
        """Get the name of the environment.

        Returns:
            The name of the environment.
        """
        return "iceslider"

    @property
    def num_actions_max(self) -> int:
        """Maximum number of actions.

        Returns:
            The maximum number of actions.
        """
        return len(self.moves)

    def _get_ice_slider_instance(self, size: int = 8, *, render_mode: str | None = "rgb_array") -> IceSlider:
        """Return an IceSlider instance, caching only non-RGB variants."""
        if render_mode == "rgb_array":
            return IceSlider(size=size, render_mode=render_mode)

        key: tuple[int, str | None] = (size, render_mode)
        slider: IceSlider | None = self._ice_slider_cache.get(key)
        if slider is None:
            slider = IceSlider(size=size, render_mode=render_mode)
            self._ice_slider_cache[key] = slider
        return slider

    def next_state(
        self, states: list[IceSliderState], actions: list[int]
    ) -> tuple[list[IceSliderState], list[np.float32]]:
        """Get the next state and transition cost given the current state and action.

        Args:
            states: List of current states.
            actions: List of actions to take.

        Returns:
            The next states and transition costs.
        """
        if len(states) != len(actions):
            raise ValueError("Number of states must match number of actions")

        num_states: int = len(states)
        if num_states == 0:
            return [], []

        # Array construction costs more than transition lookup for small search frontiers.
        if num_states < 32:
            next_states_small: list[IceSliderState] = []
            for state_small, action in zip(states, actions, strict=True):
                if action == 4:
                    next_states_small.append(state_small)
                    continue
                if not 0 <= action <= 3:
                    raise ValueError(f"Invalid action: {action}")

                transitions_small: NDArray[np.uint8] = self._get_transitions_for_grid(state_small.grid)
                nr_small, nc_small = transitions_small[state_small.pos[0], state_small.pos[1], action]
                next_states_small.append(IceSliderState.from_transition(state_small, (int(nr_small), int(nc_small))))

            return next_states_small, [np.float32(1.0)] * num_states

        actions_np: NDArray[np.intp] = np.asarray(actions, dtype=np.intp)
        invalid_indices: NDArray[np.intp] = np.flatnonzero((actions_np < 0) | (actions_np > 4))
        if invalid_indices.size:
            invalid_action: int = actions[int(invalid_indices[0])]
            raise ValueError(f"Invalid action: {invalid_action}")

        moving_indices: NDArray[np.intp] = np.flatnonzero(actions_np != 4)
        next_states: list[IceSliderState] = list(states)

        if moving_indices.size:
            grid_groups: dict[tuple[int, tuple[int, ...], tuple[int, ...]], tuple[NDArray[np.uint8], list[int]]] = {}
            for state_idx_np in moving_indices:
                state_idx: int = int(state_idx_np)
                state_group: IceSliderState = states[state_idx]
                grid_key: tuple[int, tuple[int, ...], tuple[int, ...]] = (
                    state_group.grid.ctypes.data,
                    state_group.grid.shape,
                    state_group.grid.strides,
                )
                group: tuple[NDArray[np.uint8], list[int]] | None = grid_groups.get(grid_key)
                if group is None:
                    grid_groups[grid_key] = (state_group.grid, [state_idx])
                else:
                    group[1].append(state_idx)

            next_positions: NDArray[np.uint8] = np.empty((num_states, 2), dtype=np.uint8)
            for grid, group_indices_list in grid_groups.values():
                group_indices: NDArray[np.intp] = np.asarray(group_indices_list, dtype=np.intp)
                positions: NDArray[np.intp] = np.asarray(
                    [states[state_idx].pos for state_idx in group_indices_list], dtype=np.intp
                )
                group_transitions: NDArray[np.uint8] = self._get_transitions_for_grid(grid)
                next_positions[group_indices] = group_transitions[
                    positions[:, 0], positions[:, 1], actions_np[group_indices]
                ]

            for state_idx_np in moving_indices:
                state_idx = int(state_idx_np)
                state_moving: IceSliderState = states[state_idx]
                nr_batched: np.uint8 = next_positions[state_idx, 0]
                nc_batched: np.uint8 = next_positions[state_idx, 1]
                next_states[state_idx] = IceSliderState.from_transition(
                    state_moving, (int(nr_batched), int(nc_batched))
                )

        transition_costs: list[np.float32] = [np.float32(1.0)] * num_states

        return next_states, transition_costs

    def rand_action(self, states: list[IceSliderState]) -> list[int]:
        """Get random actions that could be taken in each state.

        Args:
            states: List of current states.

        Returns:
            List of random actions.
        """
        return [np.random.randint(0, self.num_actions_max) for _ in states]

    @staticmethod
    def is_solved(states: list[IceSliderState], states_goal: list[IceSliderState]) -> NDArray[np.bool_]:
        """Check if the states are solved.

        Args:
            states: List of current states.
            states_goal: List of goal states.

        Returns:
            Boolean array indicating whether each state is solved.
        """
        if len(states) != len(states_goal):
            raise ValueError("Number of states must match number of goal states")

        solved: NDArray[np.bool_] = np.zeros(len(states), dtype=np.bool_)

        for i, (state, goal_state) in enumerate(zip(states, states_goal, strict=True)):
            solved[i] = state.pos == goal_state.pos

        return solved

    def state_to_real(
        self,
        states: list[IceSliderState],
        *,
        effects: Pipeline[ImageArray] | StagePipelines | None = None,
        **_kwargs: EffectValue,
    ) -> NDArray[np.float32]:
        """Convert states to real-world observations with optional effects.

        Args:
            states: List of current states.
            effects: Optional effects pipeline to apply during rendering:
                - PRE_RENDER stage: modify IceSlider instances (render mode, assets, tinting)
                - POST_RENDER stage: process the rendered image arrays
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Real-world observations as float32 array with shape (batch, channels, height, width).
        """
        batch_size: int = len(states)
        rendered: NDArray[np.float32] = np.zeros((batch_size, 3, self._shape, self._shape), dtype=np.float32)

        ice_slider: IceSlider
        for i, state in enumerate(states):
            # Instantiate a renderer while retaining class-level texture caches.
            ice_slider = self._get_ice_slider_instance(size=state.size, render_mode="rgb_array")

            # Set the internal state to match our state
            ice_slider.grid = state.grid
            ice_slider.pos = state.pos
            ice_slider.end = state.goal
            # Reuse the cached transition table when available.
            ice_slider.set_transitions(self._get_transitions_for_grid(state.grid))

            if effects and hasattr(effects, "apply_by_stage"):
                ice_slider = effects.apply_by_stage(ice_slider, EffectStage.PRE_RENDER)

            rgb_uint8: NDArray[np.uint8] | list[NDArray[np.uint8]] | None = ice_slider.get_rgb_array()
            assert rgb_uint8 is not None, "Renderer returned None"

            # Normalize the rendered image to float values in [0, 1].
            rgb_f: NDArray[np.float32] = np.asarray(rgb_uint8, dtype=np.float32) / 255.0

            if effects and hasattr(effects, "apply_by_stage"):
                # Prefer fresh parameter sampling for POST_RENDER when available
                if hasattr(effects, "apply_by_stage_with_fresh_params"):
                    rgb_f = effects.apply_by_stage_with_fresh_params(rgb_f, EffectStage.POST_RENDER)
                else:
                    rgb_f = effects.apply_by_stage(rgb_f, EffectStage.POST_RENDER)

            # Normalize the frame to uint8 HWC RGB.
            rgb_f = np.asarray(rgb_f, dtype=np.float32)
            if float(np.max(rgb_f)) > 1.0:
                rgb_f /= 255.0
            rgb_f = np.clip(rgb_f, 0.0, 1.0)
            # If CHW, convert to HWC
            if rgb_f.ndim == 3 and rgb_f.shape[0] in {1, 3} and rgb_f.shape[2] not in {1, 3}:
                rgb_f = np.transpose(rgb_f, (1, 2, 0))
            # If grayscale, expand to 3 channels
            if rgb_f.ndim == 2:
                rgb_f = np.repeat(rgb_f[..., None], 3, axis=2)
            elif rgb_f.ndim == 3 and rgb_f.shape[2] == 1:
                rgb_f = np.repeat(rgb_f, 3, axis=2)

            # Keep as float32 in [0,1] per ABC interface
            rendered[i] = rgb_f.transpose(2, 0, 1)

        return rendered

    @staticmethod
    def get_dqn(cfg: ModelConfig) -> nn.Module:
        """Get the DQN model for this environment.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: The DQN model.
        """
        if cfg.dqn is None:
            raise ValueError("DQN configuration is not available")

        return TransitionModelDisc(cfg.dqn)

    @staticmethod
    def get_env_model_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous environment neural network model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Continuous environment model.
        """
        if cfg.continuous is None:
            raise ValueError("Continuous configuration is not available")

        env_model: EnvModelConfig | None = cfg.continuous.env_model
        if env_model is None:
            raise ValueError("Continuous environment model configuration is not available")

        return TransitionModelCont(env_model)

    @staticmethod
    def get_env_model_disc(cfg: ModelConfig) -> nn.Module:
        """Return the environment neural network model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Environment model.
        """
        if cfg.discrete is None:
            raise ValueError("Discrete configuration is not available")

        env_model: EnvModelConfig | None = cfg.discrete.env_model
        if env_model is None:
            raise ValueError("Discrete environment model configuration is not available")

        return TransitionModelDisc(env_model)

    @staticmethod
    def get_encoder_disc(cfg: ModelConfig) -> nn.Module:
        """Return the encoder model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Encoder model.
        """
        if cfg.discrete is None:
            raise ValueError("Discrete configuration is not available")

        encoder: EncoderConfig | None = cfg.discrete.encoder
        if encoder is None:
            raise ValueError("Discrete encoder configuration is not available")

        return EncoderDisc(encoder)

    @staticmethod
    def get_encoder_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous encoder model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Continuous encoder model.
        """
        if cfg.continuous is None:
            raise ValueError("Continuous configuration is not available")

        encoder: EncoderConfig | None = cfg.continuous.encoder
        if encoder is None:
            raise ValueError("Continuous encoder configuration is not available")

        return EncoderCont(encoder)

    @staticmethod
    def get_decoder_disc(cfg: ModelConfig) -> nn.Module:
        """Return the decoder model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Decoder model.
        """
        if cfg.discrete is None:
            raise ValueError("Discrete configuration is not available")

        decoder: DecoderConfig | None = cfg.discrete.decoder
        if decoder is None:
            raise ValueError("Discrete decoder configuration is not available")

        return DecoderDisc(decoder)

    @staticmethod
    def get_decoder_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous decoder model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Continuous decoder model.
        """
        if cfg.continuous is None:
            raise ValueError("Continuous configuration is not available")

        decoder: DecoderConfig | None = cfg.continuous.decoder
        if decoder is None:
            raise ValueError("Continuous decoder configuration is not available")

        return DecoderCont(decoder)

    @staticmethod
    def get_alignment_model(cfg: ModelConfig) -> nn.Module:
        """Return the alignment model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Alignment model.
        """
        if cfg.discrete is not None:
            discrete_alignment: AlignmentModelConfig | None = cfg.discrete.alignment_model
            if discrete_alignment is not None:
                return AlignmentModel(discrete_alignment)

        if cfg.continuous is not None:
            continuous_alignment: AlignmentModelConfig | None = cfg.continuous.alignment_model
            if continuous_alignment is not None:
                return AlignmentModel(continuous_alignment)

        raise ValueError("Alignment model configuration is not available")

    def generate_start_states(self, num_states: int, level_seeds: list[int] | None = None) -> list[IceSliderState]:
        """Generate start states for the environment.

        Args:
            num_states: Number of start states to generate.
            level_seeds: Optional list of seeds for level generation.

        Returns:
            List of generated start states.
        """
        if level_seeds is not None and len(level_seeds) != num_states:
            raise ValueError("Number of level seeds must match number of states")

        start_states: list[IceSliderState] = []

        ice_slider: IceSlider
        state: IceSliderState
        for i in range(num_states):
            # Set seed if provided
            if level_seeds is not None:
                np.random.seed(level_seeds[i])

            # Generate a new ice slider puzzle using a cached instance without RGB textures
            if self._ice_slider_generator is None:
                self._ice_slider_generator = IceSlider(render_mode=None)
            ice_slider = self._ice_slider_generator
            ice_slider.reset()

            # reset populates the grid, agent, target, and solution fields.
            assert ice_slider.grid is not None
            assert ice_slider.pos is not None
            assert ice_slider.end is not None
            # reset assigns a list to solution before this branch runs.
            # Convert engine actions to the state representation's uint8 dtype.
            solution_arr: NDArray[np.uint8] = np.asarray(ice_slider.solution or (), dtype=np.uint8)

            # Extract state information
            state = IceSliderState(
                grid=ice_slider.grid,
                pos=ice_slider.pos,
                goal=ice_slider.end,
                solution=solution_arr,
                size=ice_slider.size,
            )

            start_states.append(state)

        return start_states

    def generate_goal_states(
        self,
        states: list[IceSliderState],
        num_steps: int | None,
        seeds: list[int] | NDArray[np.intp] | None = None,
        **kwargs: EffectValue,
    ) -> list[IceSliderState]:
        """Return goal states aligned with the provided states.

        For IceSlider, the goal state places the player at ``goal`` for the
        same grid/level. Other parameters are not used here.

        Args:
            states: Source states to derive goals from.
            num_steps: Unused for this environment.
            seeds: Unused for this environment.
            **kwargs: Unused.

        Returns:
            Goal states corresponding to the input states.
        """
        _ = (self, num_steps, seeds, kwargs)
        return [IceSliderState(grid=st.grid, pos=st.goal, goal=st.goal, solution=[], size=st.size) for st in states]
