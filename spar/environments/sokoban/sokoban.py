"""Sokoban environment implementation for SPAR."""

from __future__ import annotations

from logging import getLogger
import pathlib
import pickle
from typing import TYPE_CHECKING
import zipfile

import cv2
import numpy as np

from spar.environments import ABCEnvironment, ABCState
from spar.utils.env_utils import SOKOBAN_DATA_DIR
from spar.utils.env_utils.effects_core import EffectStage

from .sokoban_nn import (
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
    from collections.abc import Iterable
    from logging import Logger
    from pathlib import Path

    from cv2.typing import MatLike
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


logger: Logger = getLogger(__name__)


class SokobanState(ABCState):
    """State representation for the Sokoban environment."""

    __slots__: list[str] = ["_hash", "agent", "boxes", "seed", "walls"]

    def __init__(
        self, agent: NDArray[np.intp], boxes: NDArray[np.bool_], walls: NDArray[np.bool_], seed: int | None = None
    ) -> None:
        """Initialize the SokobanState.

        Args:
            agent: Agent's position.
            boxes: Boxes' positions.
            walls: Walls' positions.
            seed: Random seed. Defaults to None.
        """
        super().__init__()
        self.agent: NDArray[np.intp] = agent
        self.boxes: NDArray[np.bool_] = boxes
        self.walls: NDArray[np.bool_] = walls
        self.seed: int | None = seed
        self._hash: int | None = None

    def __hash__(self) -> int:
        """Compute the hash of the state.

        Returns:
            Hash value of the state.
        """
        if self._hash is not None:
            return self._hash

        boxes_flat: NDArray[np.intp] = self.boxes.flatten().astype(np.intp)
        walls_flat: NDArray[np.intp] = self.walls.flatten().astype(np.intp)
        state: NDArray[np.intp] = np.concatenate((self.agent.astype(np.intp), boxes_flat, walls_flat), axis=0)

        self._hash = hash(state.tobytes())

        return self._hash

    def __eq__(self, other: object) -> bool:
        """Check if two states are equal.

        Args:
            other: Another state to compare with.

        Returns:
            True if the states are equal, False otherwise.
        """
        if not isinstance(other, SokobanState):
            return NotImplemented

        agents_eq: bool = np.array_equal(self.agent, other.agent)
        boxes_eq: bool = np.array_equal(self.boxes, other.boxes)
        walls_eq: bool = np.array_equal(self.walls, other.walls)

        return agents_eq and boxes_eq and walls_eq


def load_states(file_name: str) -> list[SokobanState]:
    """Load Sokoban states from a file.

    Args:
        file_name: Path to the file containing the states.

    Returns:
        List of loaded Sokoban states.

    """
    with pathlib.Path(file_name).open("rb") as f:
        states_np: NDArray[np.intp] = pickle.load(f)

    states: list[SokobanState] = []

    agent_idxs: tuple[NDArray[np.intp], ...] = np.where(states_np == 1)
    box_masks: NDArray[np.bool_] = states_np == 2
    wall_masks: NDArray[np.bool_] = states_np == 4

    for idx in range(states_np.shape[0]):
        agent_idx = np.array([agent_idxs[1][idx], agent_idxs[2][idx]], dtype=int)
        states.append(SokobanState(agent_idx, box_masks[idx], wall_masks[idx]))

    return states


def _imread(path: str) -> NDArray[np.uint8]:
    """Read an image from a file and return it as a numpy array (RGB, uint8).

    Args:
        path: Path to the image file.

    Returns:
        Image as a numpy array in RGB format with dtype uint8.
    """
    bgr_opt: MatLike | None = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr_opt is None:
        raise ValueError(f"Could not read image file: {path}")
    bgr: MatLike = bgr_opt
    img: MatLike = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # The renderer contract uses uint8 output.
    return img.astype(np.uint8, copy=False)


def _get_surfaces() -> list[NDArray[np.uint8]]:
    """Load surface images for Sokoban.

    Returns:
        List of surface images.

    """
    img_dir: Path = (pathlib.Path(__file__).parent / "../../.." / SOKOBAN_DATA_DIR / "surface").resolve()

    # Load images, representing the corresponding situation
    box: NDArray[np.uint8] = _imread(str(img_dir / "box.png"))
    floor: NDArray[np.uint8] = _imread(str(img_dir / "floor.png"))
    player: NDArray[np.uint8] = _imread(str(img_dir / "player.png"))
    wall: NDArray[np.uint8] = _imread(str(img_dir / "wall.png"))

    surfaces: list[NDArray[np.uint8]] = [wall, floor, player, box]

    return surfaces


def _env_data_exists(dir_path: str, item_name: str) -> bool:
    """Check if the specified item (file or folder) exists and process a ZIP file if needed.

    Args:
        dir_path: The directory to search in.
        item_name: The name of the file or folder to check for.

    Returns:
        True if the item or its corresponding ZIP file exists and was processed,
        False otherwise.

    """
    dir_path_obj = pathlib.Path(dir_path)
    item_path: Path = dir_path_obj / item_name
    name: str = item_path.stem
    ext: str = item_path.suffix
    name_zip: str = f"{name}.zip"
    zip_path: Path = dir_path_obj / name_zip

    if ext != ".zip" and item_path.exists():
        return True

    if zip_path.exists():
        logger.info(f"ZIP file '{name_zip}' found in '{dir_path}'.\nExtracting contents...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            extracted_files = zip_ref.namelist()
            if not extracted_files:
                raise ValueError(f"The ZIP file '{name_zip}' is empty or does not contain any files.")

            zip_ref.extractall(dir_path_obj)

        logger.info(f"Successfully extracted all contents from '{name_zip}' to '{dir_path}'.")
        return True

    logger.info(f"Neither '{item_name}' nor '{name_zip}' found in '{dir_path}'. One of these is required.")
    return False


class SokobanEnv(ABCEnvironment[SokobanState]):
    """Sokoban environment implementation."""

    def __init__(self, dim: int = 10, num_boxes: int = 4) -> None:
        """Initialize the Sokoban environment.

        Args:
            dim: Dimension of the environment. Default is 10.
            num_boxes: Number of boxes in the environment. Default is 4.
        """
        super().__init__()

        self.dim: int = dim
        self.num_boxes: int = num_boxes
        self.num_moves: int = 4

        goal_states_dir = (pathlib.Path(__file__).parent / "../../.." / SOKOBAN_DATA_DIR).resolve()
        goal_states_filename: str = "goal_states.pkl"
        assert _env_data_exists(str(goal_states_dir), goal_states_filename), (
            f"Expected '{goal_states_filename}' or '{goal_states_filename[:-4]}.zip' to exist in '{goal_states_dir}'"
        )

        self.states_train: list[SokobanState] = load_states(str(goal_states_dir / goal_states_filename))

        self.img_dim: int = 40
        self.chan_enc: int = 16
        enc_h: int = 10
        enc_w: int = 10
        self.enc_dim: int = enc_h * enc_w * self.chan_enc
        self.enc_hw: tuple[int, int] = (enc_h, enc_w)

        self._surfaces: list[NDArray[np.uint8]] = _get_surfaces()
        self._surface_stack: NDArray[np.uint8] = np.stack(self._surfaces, axis=0)

    @classmethod
    def from_render_assets(cls, *, dim: int, image_dim: int, surface_stack: NDArray[np.uint8]) -> SokobanEnv:
        """Create a render-only environment without loading goal-state data.

        Args:
            dim: Side length of the Sokoban board.
            image_dim: Side length of the rendered RGB image.
            surface_stack: Bundled tile images indexed by board cell value.

        Returns:
            Environment initialized only with the state needed for rendering.
        """
        env: SokobanEnv = cls.__new__(cls)
        env.dim = dim
        env.img_dim = image_dim
        env._surface_stack = surface_stack
        return env

    @staticmethod
    def get_env_name() -> str:
        """Get the name of the environment.

        Returns:
            The name of the environment.
        """
        return "sokoban"

    @property
    def num_actions_max(self) -> int:
        """Maximum number of actions.

        Returns:
            Maximum number of actions.
        """
        return self.num_moves

    def rand_action(self, states: list[SokobanState]) -> list[int]:
        """Generate random actions for the given states.

        Args:
            states: List of states.

        Returns:
            List of random actions.
        """
        return list(np.random.randint(0, self.num_moves, size=len(states)))

    @staticmethod
    def _sample_mask(
        mask_seq: tuple[NDArray[np.bool_], ...], coords: NDArray[np.intp], indices: NDArray[np.intp] | None = None
    ) -> NDArray[np.bool_]:
        """Sample boolean masks at specified coordinates."""
        iterator: Iterable[tuple[NDArray[np.bool_], NDArray[np.intp]]]
        if indices is None:
            iterator = zip(mask_seq, coords, strict=False)
        else:
            iterator = ((mask_seq[int(idx)], coords_item) for idx, coords_item in zip(indices, coords, strict=False))
        return np.fromiter((mask[row, col] for mask, (row, col) in iterator), dtype=np.bool_, count=coords.shape[0])

    def next_state(self, states: list[SokobanState], actions: list[int]) -> tuple[list[SokobanState], list[np.float32]]:
        """Compute the next state and transition cost given the current state and action.

        Args:
            states: List of current states.
            actions: List of actions to take.

        Returns:
            Next states and transition costs.
        """
        states_list: list[SokobanState] = list(states)
        num_states: int = len(states_list)
        if num_states == 0:
            return [], []

        actions_np: NDArray[np.intp] = np.asarray(actions, dtype=np.intp)
        if actions_np.shape[0] != num_states:
            raise ValueError("Number of actions must match number of states.")

        agent_curr: NDArray[np.intp] = np.stack([state.agent for state in states_list], axis=0)
        agent_next_tmp: NDArray[np.intp] = np.empty_like(agent_curr)
        self._get_next_idx(agent_curr, actions_np, out=agent_next_tmp)

        walls_seq: tuple[NDArray[np.bool_], ...] = tuple(state.walls for state in states_list)
        boxes_seq: tuple[NDArray[np.bool_], ...] = tuple(state.boxes for state in states_list)

        agent_wall: NDArray[np.bool_] = self._sample_mask(walls_seq, agent_next_tmp)
        agent_box: NDArray[np.bool_] = self._sample_mask(boxes_seq, agent_next_tmp)

        agent_next: NDArray[np.intp] = agent_next_tmp.copy()
        if np.any(agent_wall):
            agent_next[agent_wall] = agent_curr[agent_wall]

        agent_box[agent_wall] = False
        push_indices: NDArray[np.intp] = np.flatnonzero(agent_box)

        boxes_next_seq: list[NDArray[np.bool_]] = list(boxes_seq)

        if push_indices.size:
            box_next_tmp: NDArray[np.intp] = np.empty((push_indices.size, 2), dtype=np.intp)
            self._get_next_idx(agent_next_tmp[push_indices], actions_np[push_indices], out=box_next_tmp)

            box_hits_wall: NDArray[np.bool_] = self._sample_mask(walls_seq, box_next_tmp, indices=push_indices)
            box_hits_box: NDArray[np.bool_] = self._sample_mask(boxes_seq, box_next_tmp, indices=push_indices)
            obstacles: NDArray[np.bool_] = np.logical_or(box_hits_wall, box_hits_box)

            if np.any(obstacles):
                blocked_idxs: NDArray[np.intp] = push_indices[obstacles]
                agent_next[blocked_idxs] = agent_curr[blocked_idxs]

            success_mask: NDArray[np.bool_] = np.logical_not(obstacles)
            if np.any(success_mask):
                success_push_idxs: NDArray[np.intp] = push_indices[success_mask]
                source_coords: NDArray[np.intp] = agent_next_tmp[success_push_idxs]
                dest_coords: NDArray[np.intp] = box_next_tmp[success_mask]

                for idx_local, state_idx in enumerate(success_push_idxs):
                    boxes_next: NDArray[np.bool_] = boxes_seq[state_idx].copy()
                    src_row: np.intp = source_coords[idx_local, 0]
                    src_col: np.intp = source_coords[idx_local, 1]
                    boxes_next[src_row, src_col] = False

                    dst_row: np.intp = dest_coords[idx_local, 0]
                    dst_col: np.intp = dest_coords[idx_local, 1]
                    boxes_next[dst_row, dst_col] = True

                    boxes_next_seq[state_idx] = boxes_next

        states_next: list[SokobanState] = [
            SokobanState(agent_next[idx], boxes_next_seq[idx], walls_seq[idx]) for idx in range(num_states)
        ]

        transition_costs: list[np.float32] = [np.float32(1.0)] * num_states

        return states_next, transition_costs

    # The former instance get_dqn method is replaced by static get_dqn(cfg) below.

    def state_to_real(
        self,
        states: list[SokobanState],
        *,
        effects: Pipeline[ImageArray] | StagePipelines | None = None,
        **_kwargs: EffectValue,
    ) -> NDArray[np.float32]:
        """Convert states to real-world observations.

        Args:
            states: List of states.
            effects: Optional effects pipeline to apply during rendering:
                OBJECT_RENDER effects can replace the Sokoban renderer, and POST_RENDER effects then process the image.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Real-world observations.
        """
        num_states: int = len(states)
        if num_states == 0:
            return np.empty((0, 3, self.img_dim, self.img_dim), dtype=np.float32)

        observations: NDArray[np.float32] = np.empty((num_states, 3, self.img_dim, self.img_dim), dtype=np.float32)
        clip_min: np.float32 = np.float32(0.0)
        clip_max: np.float32 = np.float32(1.0)

        for idx, state in enumerate(states):
            render_target: SokobanState | ImageArray = state
            if effects and hasattr(effects, "apply_by_stage"):
                render_target = effects.apply_by_stage(render_target, EffectStage.OBJECT_RENDER)

            rgb: NDArray[np.float32]
            if isinstance(render_target, np.ndarray):
                rgb = self._normalize_rendered_rgb(render_target)
            else:
                rgb = self.state_to_rgb(render_target)

            if effects and hasattr(effects, "apply_by_stage"):
                rgb = effects.apply_by_stage(rgb, EffectStage.POST_RENDER)

            np.clip(rgb, clip_min, clip_max, out=rgb)
            observations[idx] = np.transpose(rgb.astype(np.float32, copy=False), (2, 0, 1))

        return observations

    def _normalize_rendered_rgb(self, image: NDArray[np.generic]) -> NDArray[np.float32]:
        """Normalize an object-render effect image to HWC float32 at env image size."""
        rgb: NDArray[np.float32] = image.astype(np.float32, copy=False)
        if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, np.newaxis], 3, axis=2)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"Object-render effect returned invalid Sokoban image shape: {image.shape}")
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        if rgb.shape[0] != self.img_dim or rgb.shape[1] != self.img_dim:
            rgb = np.asarray(
                cv2.resize(rgb, (self.img_dim, self.img_dim), interpolation=cv2.INTER_AREA), dtype=np.float32
            )
        if np.nanmax(rgb) > np.float32(1.5):
            rgb *= np.float32(1.0 / 255.0)
        return rgb.astype(np.float32, copy=False)

    def generate_start_states(self, num_states: int, level_seeds: list[int] | None = None) -> list[SokobanState]:
        """Generate start states for the Sokoban environment.

        Args:
            num_states: Number of start states to generate.
            level_seeds: Optional list of seeds for level generation.

        Returns:
            List of generated start states.
        """
        del level_seeds
        state_idxs: NDArray[np.intp] = np.random.randint(0, len(self.states_train), size=num_states)
        states: list[SokobanState] = [self.states_train[idx] for idx in state_idxs]

        step_range: tuple[int, int] = (0, 100)

        # Initialize
        scrambs: list[int] = list(range(step_range[0], step_range[1] + 1))

        # Scrambles
        step_nums: NDArray[np.intp] = np.random.choice(scrambs, num_states)
        step_nums_curr: NDArray[np.intp] = np.zeros(num_states, dtype=np.intp)

        # Go backward from goal state
        steps_lt: NDArray[np.bool_] = step_nums_curr < step_nums
        idxs: NDArray[np.intp]
        idx: np.intp
        states_to_move: list[SokobanState]
        actions: list[int]
        states_moved: list[SokobanState]
        while np.any(steps_lt):
            idxs = np.where(steps_lt)[0]

            states_to_move = [states[idx] for idx in idxs]
            actions = list(np.random.randint(0, self.num_moves, size=len(states_to_move)))

            states_moved, _ = self.next_state(states_to_move, actions)

            for idx_moved, idx in enumerate(idxs):
                states[idx] = states_moved[idx_moved]

            step_nums_curr[idxs] += 1
            steps_lt[idxs] = step_nums_curr[idxs] < step_nums[idxs]

        return states

    def generate_goal_states(
        self,
        states: list[SokobanState],
        num_steps: int | None,
        seeds: list[int] | NDArray[np.intp] | None = None,
        **kwargs: EffectValue,
    ) -> list[SokobanState]:
        """Return goal states for the provided states.

        For Sokoban, goals are sampled from the dataset of goal configurations,
        independent of the provided start states. If ``seeds`` is provided, it
        controls per-sample determinism.

        Args:
            states: Reference list used for sizing. Its contents are not inspected.
            num_steps: Unused for this environment.
            seeds: Optional seeds, length should equal ``len(states)`` when given.
            **kwargs: Unused.

        Returns:
            List of goal states with length equal to ``len(states)``.
        """
        _ = (num_steps, kwargs)
        n: int = len(states)
        if n == 0:
            return []

        max_idx: int = len(self.states_train)
        if max_idx == 0:
            raise ValueError("Sokoban goal states dataset is empty")

        goal_states: list[SokobanState] = []

        if seeds is not None:
            # Deterministic sampling per item
            rs: np.random.RandomState
            idx: int
            for sd in list(seeds)[:n]:
                rs = np.random.RandomState(int(sd))
                idx = int(rs.randint(0, max_idx))
                goal_states.append(self.states_train[idx])
        else:
            idxs: NDArray[np.intp] = np.random.randint(0, max_idx, size=n)
            goal_states = [self.states_train[int(i)] for i in idxs]

        return goal_states

    def get_render_array(self, state: SokobanState) -> NDArray[np.intp]:
        """Generate a 2D array representation of the state for rendering.

        Args:
            state: The current state of the environment.

        Returns:
            2D array representation of the state.
        """
        state_rendered: NDArray[np.intp] = np.ones((self.dim, self.dim), dtype=np.intp)
        state_rendered -= state.walls
        state_rendered[state.agent[0], state.agent[1]] = 2
        state_rendered += state.boxes * 2

        return state_rendered

    def state_to_rgb(self, state: SokobanState) -> NDArray[np.float32]:
        """Convert the state to an RGB image (HWC float32 in [0,1]).

        Args:
            state: The current state of the environment.

        Returns:
            RGB image representation of the state.
        """
        room: NDArray[np.intp] = self.get_render_array(state)

        h_src: int
        w_src: int
        h_src, w_src = room.shape

        tiles: NDArray[np.uint8] = self._surface_stack[room]
        room_rgb: NDArray[np.uint8] = tiles.transpose(0, 2, 1, 3, 4).reshape(h_src * 16, w_src * 16, 3)

        room_rgb_f: NDArray[np.float32] = room_rgb.astype(np.float32)
        room_rgb_f *= np.float32(1.0 / 255.0)
        room_rgb_resized: MatLike = cv2.resize(room_rgb_f, (self.img_dim, self.img_dim))

        return room_rgb_resized.astype(np.float32, copy=False)

    def _get_next_idx(
        self, curr_idxs: NDArray[np.intp], actions: NDArray[np.intp] | list[int], *, out: NDArray[np.intp] | None = None
    ) -> NDArray[np.intp]:
        """Compute the next indices for the agent based on the current indices and actions.

        Args:
            curr_idxs: Current indices of the agent.
            actions: List of actions to be taken.
            out: Optional array to write the result into.

        Returns:
            Next indices of the agent.
        """
        actions_np: NDArray[np.intp] = np.asarray(actions, dtype=np.intp)
        next_idxs: NDArray[np.intp]
        if out is None:
            next_idxs = curr_idxs.copy()
        else:
            next_idxs = out
            np.copyto(next_idxs, curr_idxs)

        mask_up: NDArray[np.bool_] = actions_np == 0
        mask_down: NDArray[np.bool_] = actions_np == 1
        mask_left: NDArray[np.bool_] = actions_np == 2
        mask_right: NDArray[np.bool_] = actions_np == 3

        delta_row: NDArray[np.intp] = mask_down.astype(np.intp) - mask_up.astype(np.intp)
        delta_col: NDArray[np.intp] = mask_right.astype(np.intp) - mask_left.astype(np.intp)

        next_idxs[:, 0] += delta_row
        next_idxs[:, 1] += delta_col

        np.maximum(next_idxs, 0, out=next_idxs)
        np.minimum(next_idxs, self.dim - 1, out=next_idxs)

        return next_idxs

    @staticmethod
    def get_env_model_disc(cfg: ModelConfig) -> nn.Module:
        """Return the discrete environment model based on ModelConfig."""
        if cfg.discrete is None:
            raise ValueError("Discrete configuration is not available. None was provided.")
        env_model: EnvModelConfig | None = cfg.discrete.env_model
        if env_model is None:
            raise ValueError("Discrete env_model configuration is not available. None was provided.")
        return TransitionModelDisc(env_model)

    @staticmethod
    def get_env_model_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous environment model based on ModelConfig."""
        if cfg.continuous is None:
            raise ValueError("Continuous configuration is not available. None was provided.")
        env_model: EnvModelConfig | None = cfg.continuous.env_model
        if env_model is None:
            raise ValueError("Continuous env_model configuration is not available. None was provided.")
        return TransitionModelCont(env_model)

    @staticmethod
    def get_encoder_disc(cfg: ModelConfig) -> nn.Module:
        """Return the discrete encoder based on ModelConfig."""
        if cfg.discrete is None:
            raise ValueError("Discrete configuration is not available. None was provided.")
        encoder: EncoderConfig | None = cfg.discrete.encoder
        if encoder is None:
            raise ValueError("Discrete encoder configuration is not available. None was provided.")
        return EncoderDisc(encoder)

    @staticmethod
    def get_encoder_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous encoder based on ModelConfig."""
        if cfg.continuous is None:
            raise ValueError("Continuous configuration is not available. None was provided.")
        encoder: EncoderConfig | None = cfg.continuous.encoder
        if encoder is None:
            raise ValueError("Continuous encoder configuration is not available. None was provided.")
        return EncoderCont(encoder)

    @staticmethod
    def get_decoder_disc(cfg: ModelConfig) -> nn.Module:
        """Return the discrete decoder based on ModelConfig."""
        if cfg.discrete is None:
            raise ValueError("Discrete configuration is not available. None was provided.")
        decoder: DecoderConfig | None = cfg.discrete.decoder
        if decoder is None:
            raise ValueError("Discrete decoder configuration is not available. None was provided.")
        return DecoderDisc(decoder)

    @staticmethod
    def get_decoder_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous decoder based on ModelConfig."""
        if cfg.continuous is None:
            raise ValueError("Continuous configuration is not available. None was provided.")
        decoder: DecoderConfig | None = cfg.continuous.decoder
        if decoder is None:
            raise ValueError("Continuous decoder configuration is not available. None was provided.")
        return DecoderCont(decoder)

    @staticmethod
    def is_solved(states: list[SokobanState], states_goal: list[SokobanState]) -> NDArray[np.bool_]:
        """Check if the states are solved.

        Args:
            states: List of states.
            states_goal: List of goal states.

        Returns:
            Boolean array indicating whether each state is solved.
        """
        if len(states) != len(states_goal):
            raise ValueError("states and states_goal must have the same length")

        return np.fromiter(
            (
                np.array_equal(state.boxes, goal_state.boxes) and np.array_equal(state.walls, goal_state.walls)
                for state, goal_state in zip(states, states_goal, strict=True)
            ),
            dtype=np.bool_,
            count=len(states),
        )

    @staticmethod
    def get_dqn(cfg: ModelConfig) -> nn.Module:
        """Return the DQN model based on ModelConfig."""
        dqn_config: ModelArchitectureConfig | None = cfg.dqn
        if dqn_config is None:
            raise ValueError("DQN configuration is not available. None was provided.")

        return DQN(dqn_config)

    @staticmethod
    def get_alignment_model(cfg: ModelConfig) -> nn.Module:
        """Return the alignment model based on ModelConfig."""
        # Try top-level alignment_model first

        if cfg.discrete is not None:
            discrete_alignment: AlignmentModelConfig | None = cfg.discrete.alignment_model
            if discrete_alignment is not None:
                return AlignmentModel(discrete_alignment)

        if cfg.continuous is not None:
            continuous_alignment: AlignmentModelConfig | None = cfg.continuous.alignment_model
            if continuous_alignment is not None:
                return AlignmentModel(continuous_alignment)

        raise ValueError("Alignment model configuration is not available. None was provided.")
