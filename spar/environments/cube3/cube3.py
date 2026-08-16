"""Rubik's Cube 3x3x3 environment for the SPAR framework."""

from __future__ import annotations

from collections.abc import Sequence
from random import randrange
from typing import TYPE_CHECKING, overload

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np

from spar.environments.abstracts import ABCEnvironment, ABCState
from spar.utils.env_utils.effects_core import EffectStage
from spar.utils.env_utils.viz_utils import InteractiveCube

from .cube3_nn import (
    DQN,
    AlignmentModel,
    DecoderCont,
    DecoderDisc,
    EncoderCont,
    EncoderDisc,
    TransitionModelCont,
    TransitionModelDisc,
)

if TYPE_CHECKING:
    from typing import ClassVar, Literal

    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from torch import nn

    from spar.utils.config_utils.config_schema import (
        AlignmentModelConfig,
        DecoderConfig,
        EncoderConfig,
        EnvModelConfig,
        ModelArchitectureConfig,
        ModelConfig,
    )
    from spar.utils.env_utils.effects_core import EffectValue, ImageArray, Pipeline, StagePipelines


mpl.use("Agg")
_ARRAY_TYPE: type[NDArray[np.float32]] = type(np.empty(0, dtype=np.float32))


def _float_pair(value: EffectValue | None, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, _ARRAY_TYPE):
        values: Sequence[EffectValue] = value.reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        raise TypeError("Expected a two-item numeric sequence")
    if len(values) < 2:
        raise TypeError("Expected a two-item numeric sequence")
    first, second = values[0], values[1]
    if not isinstance(first, (int, float, np.number)) or not isinstance(second, (int, float, np.number)):
        raise TypeError("Expected a two-item numeric sequence")
    return float(first), float(second)


def _int_pair(value: EffectValue | None, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, _ARRAY_TYPE):
        values: Sequence[EffectValue] = value.reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        raise TypeError("Expected a two-item integer sequence")
    if len(values) < 2:
        raise TypeError("Expected a two-item integer sequence")
    first, second = values[0], values[1]
    if not isinstance(first, (int, float, np.number)) or not isinstance(second, (int, float, np.number)):
        raise TypeError("Expected a two-item integer sequence")
    return int(float(first)), int(float(second))


def _real_value(value: EffectValue | None, default: float) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float, np.number)):
        raise TypeError("Expected a numeric value")
    return float(value)


class Cube3State(ABCState):
    """ABCState representation for Cube3 environment."""

    __slots__: list[str] = ["_hash", "colors"]

    def __init__(self, colors: NDArray[np.uint8]) -> None:
        """Initialize the Cube3State.

        Args:
            colors (NDArray[np.uint8]): Colors of the cube.
        """
        super().__init__()
        self.colors: NDArray[np.uint8] = colors
        self._hash: int | None = None

    def __eq__(self, other: object) -> bool:
        """Value-based equality for Cube3State.

        Two cube states are equal when their color arrays match exactly.
        """
        if not isinstance(other, Cube3State):
            return False

        return bool(np.array_equal(self.colors, other.colors))

    def __hash__(self) -> int:
        """Compute and cache a stable hash for the cube colors."""
        if hasattr(self, "_hash") and self._hash is not None:
            return self._hash

        self._hash = hash(self.colors.tobytes())
        return self._hash


class Cube3(ABCEnvironment[Cube3State]):
    """Cube3 environment class."""

    # Moves are defined as face letter with direction (-1 or +1)
    moves: ClassVar[list[str]] = [f"{f}{n}" for f in ["U", "D", "L", "R", "B", "F"] for n in [-1, 1]]
    moves_rev: ClassVar[list[str]] = [f"{f}{n}" for f in ["U", "D", "L", "R", "B", "F"] for n in [1, -1]]

    def __init__(self, cube_len: int = 3, do_action_triples: bool = False) -> None:
        """Initialize the Cube3 environment."""
        super().__init__()
        self.cube_len: int = cube_len

        self.do_action_triples: bool = do_action_triples
        self.num_moves = 12

        # Solved state: colors from 0 to (cube_len**2) * 6 - 1
        self.goal_colors: NDArray[np.uint8] = np.arange(0, (self.cube_len**2) * 6, 1, dtype=np.uint8)

        # get idxs changed for moves (populated later)
        self.rotate_idxs_new: dict[str, NDArray[np.intp]]
        self.rotate_idxs_old: dict[str, NDArray[np.intp]]

        # adjacency faces
        self.adj_faces: dict[int, NDArray[np.intp]]
        self._get_adj()

        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(self.cube_len, self.moves)

    @staticmethod
    def get_env_name() -> str:
        """Get the name of the environment.

        Returns:
            str: The environment name ("cube3").
        """
        return "cube3"

    @property
    def num_actions_max(self) -> int:
        """Maximum number of actions.

        Returns:
            int: Maximum number of actions.
        """
        return self.num_moves

    def rand_action(self, states: list[Cube3State]) -> list[int]:
        """Return random actions for the given states.

        Args:
            states (list[Cube3State]): List of states.

        Returns:
            list[int]: List of random actions.
        """
        return list(np.random.randint(0, self.num_moves, size=len(states), dtype=np.intp))

    def next_state(self, states: list[Cube3State], actions: list[int]) -> tuple[list[Cube3State], list[np.float32]]:
        """Return the next states and transition costs for given states and actions.

        Args:
            states (list[Cube3State]): List of current states.
            actions (list[int]): List of actions.

        Returns:
            Tuple[list[Cube3State], list[float]]: Next states and transition costs.
        """
        states_cast: list[Cube3State] = list(states)
        states_np: NDArray[np.uint8] = np.stack([x.colors for x in states_cast], axis=0, dtype=np.uint8)

        states_next_np: NDArray[np.uint8] = np.zeros(states_np.shape, dtype=np.uint8)
        tcs_np: NDArray[np.float32] = np.zeros(len(states), dtype=np.float32)
        actions_np: NDArray[np.intp] = np.array(actions, dtype=np.intp)

        states_next_np_act: NDArray[np.uint8]
        tcs_act: list[int]
        for action in np.unique(actions_np):
            action_idxs: NDArray[np.bool_] = actions_np == action
            states_np_act: NDArray[np.uint8] = states_np[actions_np == action]

            states_next_np_act, tcs_act = self._move_np(states_np_act, int(action))

            states_next_np[action_idxs] = states_next_np_act
            tcs_np[action_idxs] = np.array(tcs_act, dtype=np.float32)

        states_next: list[Cube3State] = [Cube3State(x) for x in list(states_next_np)]
        transition_costs: list[np.float32] = list(tcs_np)

        return states_next, transition_costs

    @staticmethod
    def is_solved(states: list[Cube3State], states_goal: list[Cube3State]) -> NDArray[np.bool_]:
        """Check if the given states are solved.

        Args:
            states (List[Cube3State]): List of current states.
            states_goal (List[Cube3State]): List of goal states.

        Returns:
            NDArray[np.bool_]: Boolean array indicating if each state is solved.
        """
        states_np: NDArray[np.uint8] = np.stack([state.colors for state in states], axis=0, dtype=np.uint8)
        states_goal_np: NDArray[np.uint8] = np.stack([state.colors for state in states_goal], axis=0, dtype=np.uint8)

        is_equal: NDArray[np.bool_] = np.equal(states_np, states_goal_np)

        return np.all(is_equal, axis=1)

    def state_to_real(
        self,
        states: list[Cube3State],
        *,
        effects: Pipeline[ImageArray] | StagePipelines | None = None,
        **kwargs: EffectValue,
    ) -> NDArray[np.float32]:
        """Convert cube states to real images with optional effects and save to files.

        Args:
            states: List of cube states to render.
            effects: Effects to apply during rendering. Can be None for no effects,
                a Pipeline instance, or a StagePipelines instance created with create_effects().
            **kwargs: Additional keyword arguments for figure size and DPI.

        Returns:
            Array of images with shape (N, 3, H, W*2), where H and W are determined by
            the figsize, dpi parameters. Each state is rendered as two side-by-side views,
            then transposed so channels come first.

        Example:
            # Apply effects and save the rendered images.
            effects = create_effects().add_lighting(1.2).add_wear("moderate").add_noise(0.03)
            images = env.state_to_real(states, effects=effects, save_to="./output")
        """
        # SET UP FIGURE & CANVAS
        figsize: tuple[float, float] = _float_pair(kwargs.get("figsize"), (0.32, 0.32))
        dpi: float = _real_value(kwargs.get("dpi"), 100.0)
        fig: Figure = plt.figure(figsize=figsize, dpi=dpi)
        viz: InteractiveCube = InteractiveCube(3, self.generate_start_states(1)[0].colors)
        # This batch loop draws explicitly before every buffer read, so state
        # updates must not trigger additional eager draws of their own.
        viz.defer_draws = True
        fig.add_axes(viz)
        canvas: FigureCanvasAgg = FigureCanvasAgg(fig)

        # Apply PRE_RENDER effects to figure if pipeline supports it
        if effects is not None:
            fig = effects.apply_by_stage_with_fresh_params(fig, EffectStage.PRE_RENDER)

        # Compute dimensions directly
        width = int(figsize[0] * dpi)
        height = int(figsize[1] * dpi)
        num_states: int = len(states)

        # Prepare output buffer: (N, H, W*2, 3) float32 for side-by-side views
        states_img: NDArray[np.float32] = np.empty((num_states, height, width * 2, 3), dtype=np.float32)

        # Reusable RGB buffer and normalization factor
        tmp_rgb: NDArray[np.float32] = np.empty((height, width, 3), dtype=np.float32)
        inv255: np.float32 = np.float32(1.0 / 255.0)

        for idx, state in enumerate(states):
            # Update cube state
            viz.new_state(state.colors)

            # Render left and right views
            for side in (0, 1):
                viz.set_rot(side)  # Reset to default left(0)/right(1)

                # Apply OBJECT_RENDER effects to cube AFTER rotation is set
                if effects is not None:
                    viz = effects.apply_by_stage_with_fresh_params(viz, EffectStage.OBJECT_RENDER)

                # Apply camera offsets if they exist (for camera effects)
                if hasattr(viz, "apply_camera_offsets"):
                    viz.apply_camera_offsets()

                # Draw & fetch RGBA buffer
                canvas.draw()
                buf: memoryview[int] = canvas.buffer_rgba()
                arr: NDArray[np.uint8] = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)

                # Normalize RGB into tmp buffer
                np.multiply(arr[:, :, :3].astype(np.float32), inv255, out=tmp_rgb)

                # Write directly into correct width slice (side-by-side)
                states_img[idx, :, side * width : (side + 1) * width, :] = tmp_rgb

            # Apply POST_RENDER effects to images if pipeline supports it
            if effects is not None:
                left: NDArray[np.float32] = states_img[idx, :, :width, :]
                right: NDArray[np.float32] = states_img[idx, :, width:, :]

                # Apply POST_RENDER effects with independent parameters for each frame
                left_result: NDArray[np.float32] = effects.apply_by_stage_with_fresh_params(
                    left, EffectStage.POST_RENDER
                )
                right_result: NDArray[np.float32] = effects.apply_by_stage_with_fresh_params(
                    right, EffectStage.POST_RENDER
                )

                # Validate result shape before writing into the output batch.
                if left_result.shape == (height, width, 3) and right_result.shape == (height, width, 3):
                    states_img[idx, :, :width, :] = left_result
                    states_img[idx, :, width:, :] = right_result

        plt.close(fig)

        # Final shape: (N, 3, H, W*2)
        result: NDArray[np.float32] = np.transpose(states_img, (0, 3, 1, 2))

        return result

    def render_post_variation_from_base(self, base_image: ImageArray, effects: StagePipelines) -> NDArray[np.float32]:
        """Apply post-render effects without drawing the unchanged cube again.

        The discarded start state preserves the random-number consumption of
        one legacy singleton call to :meth:`state_to_real`.

        Args:
            base_image: Existing channels-first image with two horizontal views.
            effects: Pipeline containing no pre-render or object-render effects.

        Returns:
            The varied channels-first image.

        Raises:
            ValueError: If the pipeline contains an earlier-stage effect or
                ``base_image`` does not contain two RGB views.
        """
        if effects.pre is not None or effects.obj is not None:
            raise ValueError("Base-image reuse only supports post-render effects")

        self.generate_start_states(1)

        if base_image.ndim != 3 or base_image.shape[0] != 3 or base_image.shape[2] % 2 != 0:
            raise ValueError(f"Expected a channels-first image with two RGB views, got {base_image.shape}")

        varied_hwc: NDArray[np.float32] = np.transpose(np.asarray(base_image, dtype=np.float32), (1, 2, 0)).copy()
        height: int = varied_hwc.shape[0]
        width: int = varied_hwc.shape[1] // 2
        left: NDArray[np.float32] = varied_hwc[:, :width, :]
        right: NDArray[np.float32] = varied_hwc[:, width:, :]

        left_result: NDArray[np.float32] = effects.apply_by_stage_with_fresh_params(left, EffectStage.POST_RENDER)
        right_result: NDArray[np.float32] = effects.apply_by_stage_with_fresh_params(right, EffectStage.POST_RENDER)
        expected_shape: tuple[int, int, int] = (height, width, 3)
        if left_result.shape == expected_shape and right_result.shape == expected_shape:
            varied_hwc[:, :width, :] = left_result
            varied_hwc[:, width:, :] = right_result

        return np.transpose(varied_hwc, (2, 0, 1))

    @staticmethod
    def get_dqn(cfg: ModelConfig) -> nn.Module:
        """Get the DQN model for this environment.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: The DQN model.
        """
        dqn_config: ModelArchitectureConfig | None = cfg.dqn
        if dqn_config is None:
            raise ValueError("DQN configuration is not available. None was provided.")

        return DQN(dqn_config)

    @staticmethod
    def get_env_model_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous environment neural network model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Continuous environment model.
        """
        # Set the number of actions
        # cfg.continuous.env_model_continuous.num_actions = self.num_actions_max

        # Create the model
        if cfg.continuous is not None:
            env_model: EnvModelConfig | None = cfg.continuous.env_model
            if env_model is not None:
                return TransitionModelCont(env_model)

        raise ValueError("Continuous environment model configuration is not available")

    @staticmethod
    def get_env_model_disc(cfg: ModelConfig) -> nn.Module:
        """Return the environment neural network model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Environment model.
        """
        # Set the number of actions
        # cfg.discrete.env_model.num_actions = self.num_actions_max

        # Create the model
        if cfg.discrete is not None:
            env_model: EnvModelConfig | None = cfg.discrete.env_model
            if env_model is not None:
                return TransitionModelDisc(env_model)

        raise ValueError("Discrete environment model configuration is not available")

    @staticmethod
    def get_encoder_disc(cfg: ModelConfig) -> nn.Module:
        """Return the encoder model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Encoder model.
        """
        if cfg.discrete is not None:
            encoder: EncoderConfig | None = cfg.discrete.encoder
            if encoder is not None:
                return EncoderDisc(encoder)

        raise ValueError("Discrete encoder configuration is not available")

    @staticmethod
    def get_encoder_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous encoder model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Continuous encoder model.
        """
        if cfg.continuous is not None:
            encoder = cfg.continuous.encoder
            if encoder is not None:
                return EncoderCont(encoder)

        raise ValueError("Continuous encoder configuration is not available")

    @staticmethod
    def get_decoder_disc(cfg: ModelConfig) -> nn.Module:
        """Return the decoder model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Decoder model.
        """
        # Create the model
        if cfg.discrete is not None:
            decoder = cfg.discrete.decoder
            if decoder is not None:
                return DecoderDisc(decoder)

        raise ValueError("Discrete decoder configuration is not available")

    @staticmethod
    def get_decoder_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous decoder model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Continuous decoder model.
        """
        # Create the model
        if cfg.continuous is not None:
            decoder: DecoderConfig | None = cfg.continuous.decoder
            if decoder is not None:
                return DecoderCont(decoder)

        raise ValueError("Continuous decoder configuration is not available")

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

    def generate_start_states(self, num_states: int, level_seeds: list[int] | None = None) -> list[Cube3State]:
        """Generate start states by scrambling the solved state.

        Args:
            num_states (int): Number of start states to generate.
            level_seeds (list[int] | None): Optional per-state seeds. When
                provided, state ``i`` is scrambled with a generator seeded by
                ``level_seeds[i]``, so the same seed always reproduces the same
                state regardless of batch composition.

        Returns:
            list[Cube3State]: List of start states.
        """
        assert num_states > 0
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        backwards_range: tuple[int, int] = (100, 200)

        # Get goal states
        states_np_result = self._generate_canonical_goal_states(num_states, np_format=True)
        states_np: NDArray[np.uint8] = states_np_result

        if level_seeds is not None:
            if len(level_seeds) != num_states:
                raise ValueError(f"level_seeds must have {num_states} entries, got {len(level_seeds)}")

            # Seeded path: scramble each state with its own deterministic
            # generator so identical seeds reproduce identical states.
            num_env_moves_seeded: int = len(self.moves)
            for state_idx, seed in enumerate(level_seeds):
                rng: np.random.Generator = np.random.default_rng(seed)
                num_scrambles: int = int(rng.integers(backwards_range[0], backwards_range[1] + 1))
                scramble_moves: NDArray[np.int64] = rng.integers(0, num_env_moves_seeded, size=num_scrambles)

                row: NDArray[np.uint8] = states_np[state_idx : state_idx + 1]
                for scramble_move in scramble_moves:
                    row, _ = self._move_np(row, int(scramble_move))
                states_np[state_idx : state_idx + 1] = row

            return [Cube3State(x) for x in list(states_np)]

        # Unseeded path: batched scrambling driven by the global RNG.
        scrambs: list[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = len(self.moves)

        # Scrambles
        scramble_nums: NDArray[np.intp] = np.random.choice(scrambs, num_states)
        num_back_moves: NDArray[np.intp] = np.zeros(num_states, dtype=np.intp)

        # Go backward from goal state
        moves_lt: NDArray[np.bool_] = num_back_moves < scramble_nums
        while np.any(moves_lt):
            idxs: NDArray[np.intp] = np.where(moves_lt)[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            states_np[idxs], _ = self._move_np(states_np[idxs], move)

            num_back_moves[idxs] += 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        states: list[Cube3State] = [Cube3State(x) for x in list(states_np)]

        return states

    @overload
    def _generate_canonical_goal_states(
        self, num_states: int, np_format: Literal[True] = True
    ) -> NDArray[np.uint8]: ...

    @overload
    def _generate_canonical_goal_states(
        self, num_states: int, np_format: Literal[False] = False
    ) -> list[Cube3State]: ...

    def _generate_canonical_goal_states(
        self, num_states: int, np_format: bool = False
    ) -> NDArray[np.uint8] | list[Cube3State]:
        """Generate copies of the canonical Cube3 goal state.

        Args:
            num_states: Number of goal states to generate.
            np_format: Whether to return a NumPy array.

        Returns:
            A uint8 array with shape ``(N, 6, cube_len, cube_len)`` when
            ``np_format`` is true, otherwise a list of ``Cube3State`` objects.
        """
        if np_format:
            goal_np: NDArray[np.uint8] = np.expand_dims(self.goal_colors.copy(), 0)
            solved_states_np: NDArray[np.uint8] = np.repeat(goal_np, num_states, axis=0)
            return solved_states_np

        solved_states: list[Cube3State] = [Cube3State(self.goal_colors.copy()) for _ in range(num_states)]
        return solved_states

    def generate_goal_states(
        self,
        states: list[Cube3State],
        num_steps: int | None,
        seeds: list[int] | NDArray[np.intp] | None = None,
        **kwargs: EffectValue,
    ) -> list[Cube3State]:
        """Generate goal states for the provided states.

        Behavior:
        - By default (reverse_goal=False), returns the canonical solved state for each input.
        - If reverse_goal=True, returns a scrambled state for each input state by
          applying a random walk of length ``num_steps`` (or a sampled length if None).

        Args:
            states: List of input states (used only when reverse_goal=True).
            num_steps: Number of random steps for scrambling when reverse_goal=True. If None,
                the number of steps is sampled uniformly from a default range per state.
            seeds: Optional list/array of per-state seeds used to deterministically generate
                the scramble action sequences when reverse_goal=True.
            **kwargs: Additional keyword arguments.
                - reverse_goal (bool, default False): If True, scrambled states will be returned as goals.
                - scramble_range (tuple[int, int], default (100, 200)): Range to sample number
                  of steps from when num_steps is None and reverse_goal=True.

        Returns:
            list[Cube3State]: The generated goal states.
        """
        reverse_goal: bool = bool(kwargs.get("reverse_goal"))

        # Fast-path: canonical solved state as goal
        if not reverse_goal:
            return self._generate_canonical_goal_states(len(states), np_format=False)

        # Reverse goal: scramble each input state forward by num_steps (or sampled length)
        if len(states) == 0:
            return []

        # Determine scramble lengths
        scramble_range: tuple[int, int] = _int_pair(kwargs.get("scramble_range"), (100, 200))
        if scramble_range[0] > scramble_range[1]:
            scramble_range = (scramble_range[1], scramble_range[0])

        num_states: int = len(states)

        # Prepare RNGs per-state for determinism if seeds are provided
        rngs: list[np.random.Generator]
        if seeds is not None:
            seeds_list: list[int] = [int(s) for s in list(seeds)[:num_states]]
            # If fewer seeds are provided than states, generate the rest randomly
            if len(seeds_list) < num_states:
                seeds_list.extend(list(np.random.randint(0, 2**31 - 1, size=num_states - len(seeds_list))))
            rngs = [np.random.default_rng(seed) for seed in seeds_list]
        else:
            rngs = [np.random.default_rng() for _ in range(num_states)]

        # Determine steps per state
        if num_steps is None:
            scramble_nums: NDArray[np.int64] = np.array(
                [rng.integers(scramble_range[0], scramble_range[1] + 1) for rng in rngs], dtype=np.int64
            )
        else:
            scramble_nums = np.full((num_states,), num_steps, dtype=np.int64)

        # Stack current colors
        states_np: NDArray[np.uint8] = np.stack([s.colors for s in states], axis=0).astype(np.uint8, copy=True)

        # Apply actions in a vectorized per-step manner grouped by action
        max_steps: int = int(np.max(scramble_nums, initial=0))
        if max_steps == 0:
            return [Cube3State(colors.copy()) for colors in list(states_np)]

        for step in range(max_steps):
            # Active states for this step
            active_mask: NDArray[np.bool_] = scramble_nums > step
            if not np.any(active_mask):
                break

            active_idxs: NDArray[np.intp] = np.where(active_mask)[0]
            # Sample an action for each active state using its own RNG
            actions_sampled: list[int] = [int(rngs[i].integers(0, self.num_moves)) for i in active_idxs]

            # Group by unique action to batch _move_np calls
            if len(active_idxs) > 0:
                actions_np: NDArray[np.intp] = np.array(actions_sampled, dtype=np.intp)
                for action in np.unique(actions_np):
                    group_mask: NDArray[np.bool_] = actions_np == action
                    group_idxs: NDArray[np.intp] = active_idxs[group_mask]
                    if group_idxs.size == 0:
                        continue
                    moved_np, _ = self._move_np(states_np[group_idxs], int(action))
                    states_np[group_idxs] = moved_np

        # Wrap into Cube3State objects
        return [Cube3State(x) for x in list(states_np)]

    def _move_np(self, states_np: NDArray[np.uint8], action: int) -> tuple[NDArray[np.uint8], list[int]]:
        """Apply the given action to states.

        Args:
            states_np (NDArray[np.uint8]): Current states.
            action (int): Action to apply.

        Returns:
            Tuple[NDArray[np.uint8], list[int]]: Next states and transition costs.
        """
        states_next_np: NDArray[np.uint8] = states_np.copy()

        actions: list[int] = []
        if self.do_action_triples:
            action_triples: dict[int, list[int]] | None = getattr(self, "action_triples", None)
            actions = list(action_triples[action]) if action_triples is not None else [action]
        else:
            actions = [action]

        for action_part in actions:
            action_str: str = self.moves[action_part]
            states_next_np[:, self.rotate_idxs_new[action_str]] = states_next_np[:, self.rotate_idxs_old[action_str]]

        transition_costs: list[int] = [1 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs

    def _get_adj(self) -> None:
        """Initialize Cube3 face adjacency.

        The faces are represented by integers:
        - WHITE: 0
        - YELLOW: 1
        - BLUE: 2
        - GREEN: 3
        - ORANGE: 4
        - RED: 5

        The adjacency relationships are stored in the 'adj_faces' attribute.
        """
        self.adj_faces = {
            0: np.array([2, 5, 3, 4], dtype=np.intp),
            1: np.array([2, 4, 3, 5], dtype=np.intp),
            2: np.array([0, 4, 1, 5], dtype=np.intp),
            3: np.array([0, 5, 1, 4], dtype=np.intp),
            4: np.array([0, 3, 1, 2], dtype=np.intp),
            5: np.array([0, 2, 1, 3], dtype=np.intp),
        }

    def _compute_rotation_idxs(
        self, cube_len: int, moves: list[str]
    ) -> tuple[dict[str, NDArray[np.intp]], dict[str, NDArray[np.intp]]]:
        """Compute the rotation indices for the cube faces based on the given moves.

        Args:
            cube_len (int): The length of one side of the cube.
            moves (list[str]): A list of moves to be applied to the cube. Each move is represented
                as a string, where the first character is the face ('U', 'D', 'L', 'R', 'B', 'F')
                and the second character is the direction (1 or -1).

        Returns:
            Tuple[Dict[str, NDArray], Dict[str, NDArray]]: Two dictionaries containing the
                new and old rotation indices for each move.
                - The keys are the move strings.
                - The values are numpy arrays of flattened indices representing the positions of
                    the colors on the cube faces.
        """
        rotate_idxs_new: dict[str, NDArray[np.intp]] = {}
        rotate_idxs_old: dict[str, NDArray[np.intp]] = {}

        for move in moves:
            f: str = move[0]
            sign: int = int(move[1:])

            rotate_idxs_new[move] = np.array([], dtype=np.intp)
            rotate_idxs_old[move] = np.array([], dtype=np.intp)

            colors = np.zeros((6, cube_len, cube_len), dtype=np.int64)
            colors_new = np.copy(colors)

            # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5

            adj_idxs = {
                0: {
                    2: [range(cube_len), cube_len - 1],
                    3: [range(cube_len), cube_len - 1],
                    4: [range(cube_len), cube_len - 1],
                    5: [range(cube_len), cube_len - 1],
                },
                1: {2: [range(cube_len), 0], 3: [range(cube_len), 0], 4: [range(cube_len), 0], 5: [range(cube_len), 0]},
                2: {
                    0: [0, range(cube_len)],
                    1: [0, range(cube_len)],
                    4: [cube_len - 1, range(cube_len - 1, -1, -1)],
                    5: [0, range(cube_len)],
                },
                3: {
                    0: [cube_len - 1, range(cube_len)],
                    1: [cube_len - 1, range(cube_len)],
                    4: [0, range(cube_len - 1, -1, -1)],
                    5: [cube_len - 1, range(cube_len)],
                },
                4: {
                    0: [range(cube_len), cube_len - 1],
                    1: [range(cube_len - 1, -1, -1), 0],
                    2: [0, range(cube_len)],
                    3: [cube_len - 1, range(cube_len - 1, -1, -1)],
                },
                5: {
                    0: [range(cube_len), 0],
                    1: [range(cube_len - 1, -1, -1), cube_len - 1],
                    2: [cube_len - 1, range(cube_len)],
                    3: [0, range(cube_len - 1, -1, -1)],
                },
            }

            face_dict = {"U": 0, "D": 1, "L": 2, "R": 3, "B": 4, "F": 5}
            face = face_dict[f]

            faces_to = self.adj_faces[face]
            if sign == 1:
                faces_from = faces_to[(np.arange(0, len(faces_to)) + 1) % len(faces_to)]
            else:
                faces_from = faces_to[(np.arange(len(faces_to) - 1, len(faces_to) - 1 + len(faces_to))) % len(faces_to)]

            cubes_idxs = [
                [0, range(cube_len)],
                [range(cube_len), cube_len - 1],
                [cube_len - 1, range(cube_len - 1, -1, -1)],
                [range(cube_len - 1, -1, -1), 0],
            ]
            cubes_to = np.array([0, 1, 2, 3])
            if sign == 1:
                cubes_from = cubes_to[(np.arange(len(cubes_to) - 1, len(cubes_to) - 1 + len(cubes_to))) % len(cubes_to)]
            else:
                cubes_from = cubes_to[(np.arange(0, len(cubes_to)) + 1) % len(cubes_to)]

            for i in range(4):
                idxs_new = [
                    [idx1, idx2]
                    for idx1 in np.array([cubes_idxs[cubes_to[i]][0]]).flatten()
                    for idx2 in np.array([cubes_idxs[cubes_to[i]][1]]).flatten()
                ]
                idxs_old = [
                    [idx1, idx2]
                    for idx1 in np.array([cubes_idxs[cubes_from[i]][0]]).flatten()
                    for idx2 in np.array([cubes_idxs[cubes_from[i]][1]]).flatten()
                ]

                for idx_new, idx_old in zip(idxs_new, idxs_old, strict=False):
                    flat_idx_new = np.ravel_multi_index((face, idx_new[0], idx_new[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face, idx_old[0], idx_old[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

            # Rotate adjacent faces
            face_idxs = adj_idxs[face]
            # pylint: disable=consider-using-enumerate
            for i in range(len(faces_to)):
                face_to = faces_to[i]
                face_from = faces_from[i]
                idxs_new = [
                    [idx1, idx2]
                    for idx1 in np.array([face_idxs[face_to][0]]).flatten()
                    for idx2 in np.array([face_idxs[face_to][1]]).flatten()
                ]
                idxs_old = [
                    [idx1, idx2]
                    for idx1 in np.array([face_idxs[face_from][0]]).flatten()
                    for idx2 in np.array([face_idxs[face_from][1]]).flatten()
                ]
                for idx_new, idx_old in zip(idxs_new, idxs_old, strict=False):
                    flat_idx_new = np.ravel_multi_index((face_to, idx_new[0], idx_new[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face_from, idx_old[0], idx_old[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

        return rotate_idxs_new, rotate_idxs_old


class Cube3Triples(Cube3):
    """Cube3Triples environment class."""

    moves: ClassVar[list[str]] = [f"{f}{n}" for f in ["U", "D", "L", "R", "B", "F"] for n in [-1, 1]]
    moves_rev: ClassVar[list[str]] = [f"{f}{n}" for f in ["U", "D", "L", "R", "B", "F"] for n in [1, -1]]

    def __init__(self, cube_len: int = 3) -> None:
        """Initialize the Cube3Triples environment."""
        super().__init__(cube_len=cube_len, do_action_triples=True)

        self.num_moves = 12**3
        self.action_triples: list[tuple[int, int, int]] = []
        for i in range(12):
            for j in range(12):
                for k in range(12):
                    self.action_triples.append((i, j, k))

        # solved state
        self.goal_colors: NDArray[np.uint8] = np.arange(0, (self.cube_len**2) * 6, 1, dtype=np.uint8)

        # get idxs changed for moves
        self.rotate_idxs_new: dict[str, NDArray[np.intp]]
        self.rotate_idxs_old: dict[str, NDArray[np.intp]]

        self.adj_faces: dict[int, NDArray[np.intp]]
        self._get_adj()

        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(self.cube_len, self.moves)

    @staticmethod
    def get_env_name() -> str:
        """Get the name of the environment.

        Returns:
            str: The name of the environment, "cube3_triples".
        """
        return "cube3_triples"
