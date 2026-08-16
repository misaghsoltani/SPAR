"""Digit Jump environment for the SPAR framework."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from spar.environments.abstracts import ABCEnvironment, ABCState
from spar.utils.env_utils.effects_core import EffectStage
from spar.utils.env_utils.puzzlegen.digit_jump import DigitJump

from .digitjump_nn import (
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
    from numpy.typing import NDArray
    from torch import nn

    from spar.utils.config_utils.config_schema import (
        AlignmentModelConfig,
        DecoderConfig,
        EncoderConfig,
        EnvModelConfig,
        ModelConfig,
    )
    from spar.utils.env_utils.effects_core import EffectValue, Pipeline, StagePipelines


# Actions: 0=North(up),1=East(right),2=West(left),3=South(down),4=No-op


class DigitJumpState(ABCState):
    """State representation for Digit Jump environment.

    Attributes:
        grid: Integer grid of jump distances.
        pos: Current (row, col) position.
        end: Goal (row, col) position.
        size: Grid size used by the backend.
    """

    __slots__: list[str] = ["_hash", "_pz", "end", "grid", "pos", "size"]

    def __init__(self, grid: NDArray[np.uint8], pos: tuple[int, int], end: tuple[int, int], size: int) -> None:
        super().__init__()
        grid_view: NDArray[np.uint8] = np.asarray(grid, dtype=np.uint8)
        grid_view.setflags(write=False)
        self.grid: NDArray[np.uint8] = grid_view
        self.pos: tuple[int, int] = (pos[0], pos[1])
        self.end: tuple[int, int] = (end[0], end[1])
        self.size: int = size
        self._hash: int | None = None
        self._pz: DigitJump | None = None

    def __eq__(self, other: object) -> bool:
        """Value-based equality for DigitJumpState.

        Two states are equal when size, position, end and the grid contents match.
        """
        if not isinstance(other, DigitJumpState):
            return False

        if self.size != other.size or self.pos != other.pos or self.end != other.end:
            return False

        return bool(np.array_equal(self.grid, other.grid))

    def __hash__(self) -> int:
        """Compute a stable hash for the state and cache it.

        Combines size, pos, end and the raw grid bytes into a single hash.
        """
        if hasattr(self, "_hash") and self._hash is not None:
            return self._hash

        self._hash = hash((self.size, self.pos, self.end, self.grid.tobytes()))
        return self._hash


class DigitJumpEnv(ABCEnvironment[DigitJumpState]):
    """Digit Jump environment class."""

    def __init__(self, size: int = 8) -> None:
        """Initialize the Digit Jump environment.

        Args:
            size (int): Grid size.
        """
        super().__init__()

        self.size: int = size
        self.num_moves: int = 5

        self._pz = DigitJump(size=self.size, render_mode="rgb_array")
        self._render_rgb_buffer: NDArray[np.float32] | None = None

    @staticmethod
    def get_env_name() -> str:
        """Get the name of the environment.

        Returns:
            str: Environment name.
        """
        return "digitjump"

    @property
    def num_actions_max(self) -> int:
        """Maximum number of actions.

        Returns:
            int: Maximum number of actions.
        """
        return self.num_moves

    def rand_action(self, states: list[DigitJumpState]) -> list[int]:
        """Return random actions for the given states.

        Args:
            states (list[DigitJumpState]): List of states.

        Returns:
            list[int]: List of random actions.
        """
        return list(np.random.randint(0, self.num_moves, size=len(states)))

    def next_state(
        self, states: list[DigitJumpState], actions: list[int]
    ) -> tuple[list[DigitJumpState], list[np.float32]]:
        """Return the next states and transition costs for the given states and actions.

        Args:
            states (list[DigitJumpState]): List of current states.
            actions (list[int]): List of actions.

        Returns:
            tuple[list[DigitJumpState], list[np.float32]]: Tuple of next states and transition costs.
        """
        next_states: list[DigitJumpState] = []
        costs: list[np.float32] = []

        if states and self._pz.size != states[0].size:
            self._pz = DigitJump(size=states[0].size, render_mode="rgb_array")

        r: int
        c: int
        step: int
        nr: int
        nc: int
        for st, a in zip(states, actions, strict=True):
            r, c = st.pos
            step = int(st.grid[r, c])
            nr, nc = self._pz.move(r, c, a, step)
            next_states.append(DigitJumpState(st.grid, (nr, nc), st.end, st.size))
            costs.append(np.float32(1.0))

        return next_states, costs

    @staticmethod
    def is_solved(states: list[DigitJumpState], states_goal: list[DigitJumpState]) -> NDArray[np.bool_]:
        """Check if the given states are solved.

        Args:
            states (list[DigitJumpState]): List of states.
            states_goal (list[DigitJumpState]): List of goal states.

        Returns:
            NDArray: Boolean array indicating if each state is solved.
        """
        result: NDArray[np.bool_] = np.zeros((len(states),), dtype=np.bool_)
        s: DigitJumpState
        g: DigitJumpState
        for i, (s, g) in enumerate(zip(states, states_goal, strict=True)):
            result[i] = (s.pos[0] == g.pos[0]) and (s.pos[1] == g.pos[1])
        return result

    def generate_goal_states(
        self,
        states: list[DigitJumpState],
        num_steps: int | None,
        seeds: list[int] | NDArray[np.intp] | None = None,
        **kwargs: EffectValue,
    ) -> list[DigitJumpState]:
        """Return goal states corresponding to the provided states.

        For DigitJump, the goal state for a given state is the same grid with
        the agent positioned at the environment's target ``end``.

        Args:
            states: Source states to derive goals from.
            num_steps: Unused for this environment.
            seeds: Unused for this environment.
            **kwargs: Unused.

        Returns:
            List of goal states aligned with the input states.
        """
        _ = (self, num_steps, seeds, kwargs)
        return [DigitJumpState(st.grid, st.end, st.end, st.size) for st in states]

    def generate_start_states(self, num_states: int, level_seeds: list[int] | None = None) -> list[DigitJumpState]:
        """Generate start states using puzzlegen levels."""
        starts: list[DigitJumpState] = []
        seeds_iter: list[int] = (
            level_seeds
            if level_seeds is not None
            else [int(np.random.randint(0, 2**31 - 1)) for _ in range(num_states)]
        )
        env = DigitJump(size=self.size, render_mode="rgb_array")
        for seed in seeds_iter[:num_states]:
            env.reset(seed=seed)
            assert env.grid is not None
            assert env.start is not None
            assert env.end is not None
            starts.append(DigitJumpState(env.grid, env.start, env.end, env.size))
        return starts

    def state_to_real(
        self,
        states: list[DigitJumpState],
        *,
        effects: Pipeline[NDArray[np.float32]] | StagePipelines | None = None,
        **_kwargs: EffectValue,
    ) -> NDArray[np.float32]:
        """Convert the given states to their real pixel representations.

        Args:
            states (list[DigitJumpState]): List of states to convert.
            effects: Optional effects to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            NDArray[np.float32]: Array of shape (N, C, H, W) with pixel values.
        """
        if states and self._pz.size != states[0].size:
            self._pz = DigitJump(size=states[0].size, render_mode="rgb_array")

        # The puzzlegen backend lazily loads rendering assets during reset(),
        # but here we bypass reset and set grid and positions manually. Set the
        # pre-baked blocks exist before requesting an RGB frame.
        self._pz.load_rendering_assets_if_needed()

        out: NDArray[np.float32] = np.empty((len(states), 3, 64, 64), dtype=np.float32)
        if self._render_rgb_buffer is None:
            self._render_rgb_buffer = np.empty((64, 64, 3), dtype=np.float32)
        float_workspace: NDArray[np.float32] = self._render_rgb_buffer
        st: DigitJumpState
        rgb_uint8: NDArray[np.uint8] | list[NDArray[np.uint8]] | None
        rgb_float: NDArray[np.float32]
        for i, st in enumerate(states):
            # Populate backend state
            self._pz.grid = st.grid
            self._pz.pos = st.pos
            self._pz.end = st.end
            # Render one frame
            rgb_uint8 = self._pz.get_rgb_array()
            if rgb_uint8 is None:
                raise RuntimeError("puzzlegen returned no RGB frame")
            arr_uint8: NDArray[np.uint8] = np.asarray(rgb_uint8, dtype=np.uint8)
            np.divide(arr_uint8, 255.0, out=float_workspace, dtype=np.float32)
            rgb_float = float_workspace
            if effects is not None:
                rgb_float = effects.apply_by_stage(rgb_float, EffectStage.POST_RENDER)
            np.clip(rgb_float, 0.0, 1.0, out=float_workspace)
            out[i] = float_workspace.transpose(2, 0, 1)
        return out

    @staticmethod
    def get_dqn(cfg: ModelConfig) -> nn.Module:
        """Get the DQN model for this environment."""
        if cfg.dqn is None:
            raise ValueError("DQN configuration is not available")
        return DQN(cfg.dqn)

    @staticmethod
    def get_env_model_disc(cfg: ModelConfig) -> nn.Module:
        """Return the discrete environment model."""
        if cfg.discrete is None:
            raise ValueError("Discrete configuration is not available")
        env_model: EnvModelConfig | None = cfg.discrete.env_model
        if env_model is None:
            raise ValueError("Discrete env_model configuration is not available")

        return TransitionModelDisc(env_model)

    @staticmethod
    def get_env_model_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous environment model."""
        if cfg.continuous is None:
            raise ValueError("Continuous configuration is not available")
        env_model: EnvModelConfig | None = cfg.continuous.env_model
        if env_model is None:
            raise ValueError("Continuous env_model configuration is not available")

        return TransitionModelCont(env_model)

    @staticmethod
    def get_encoder_disc(cfg: ModelConfig) -> nn.Module:
        """Return the discrete encoder model."""
        if cfg.discrete is None:
            raise ValueError("Discrete configuration is not available")
        encoder: EncoderConfig | None = cfg.discrete.encoder
        if encoder is None:
            raise ValueError("Discrete encoder configuration is not available")
        return EncoderDisc(encoder)

    @staticmethod
    def get_encoder_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous encoder model."""
        if cfg.continuous is None:
            raise ValueError("Continuous configuration is not available")
        encoder: EncoderConfig | None = cfg.continuous.encoder
        if encoder is None:
            raise ValueError("Continuous encoder configuration is not available")
        return EncoderCont(encoder)

    @staticmethod
    def get_decoder_disc(cfg: ModelConfig) -> nn.Module:
        """Return the discrete decoder model."""
        if cfg.discrete is None:
            raise ValueError("Discrete configuration is not available")
        decoder: DecoderConfig | None = cfg.discrete.decoder
        if decoder is None:
            raise ValueError("Discrete decoder configuration is not available")
        return DecoderDisc(decoder)

    @staticmethod
    def get_decoder_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous decoder model."""
        if cfg.continuous is None:
            raise ValueError("Continuous configuration is not available")
        decoder: DecoderConfig | None = cfg.continuous.decoder
        if decoder is None:
            raise ValueError("Continuous decoder configuration is not available")
        return DecoderCont(decoder)

    @staticmethod
    def get_alignment_model(cfg: ModelConfig) -> nn.Module:
        """Return the alignment model."""
        if cfg.discrete is not None:
            disc: AlignmentModelConfig | None = cfg.discrete.alignment_model
            if disc is not None:
                return AlignmentModel(disc)
        if cfg.continuous is not None:
            cont: AlignmentModelConfig | None = cfg.continuous.alignment_model
            if cont is not None:
                return AlignmentModel(cont)
        raise ValueError("Alignment model configuration is not available")
