"""Abstract base class for environments used in the SPAR framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import nn

    from spar.environments.abstracts.state import ABCState
    from spar.utils.config_utils.config_schema import ModelConfig
    from spar.utils.env_utils.effects_core import EffectValue, ImageArray, Pipeline, StagePipelines


State = TypeVar("State", bound="ABCState")


# @enforce_init_defaults
class ABCEnvironment(ABC, Generic[State]):
    """Abstract base class defining the interface for environments in the SPAR framework."""

    def __init__(self) -> None:
        self.dtype: type = np.float32
        self.fixed_actions: bool = True

    @staticmethod
    @abstractmethod
    def get_env_name() -> str:
        """Get the name of the environment.

        Returns:
            str: The name of the environment.
        """

    @property
    @abstractmethod
    def num_actions_max(self) -> int:
        """Maximum number of actions available in the environment.

        Returns:
            int: Maximum number of actions.
        """

    @abstractmethod
    def next_state(self, states: list[State], actions: list[int]) -> tuple[list[State], list[np.float32]]:
        """Get the next state and transition cost given the current state and action.

        Args:
            states (list[State]): List of states.
            actions (list[int]): Actions to take.

        Returns:
            Tuple[list[State], list[float]]: Next states, transition costs. Input states may be modified!
        """

    @abstractmethod
    def rand_action(self, states: list[State]) -> list[int]:
        """Get random actions that could be taken in each state.

        Args:
            states (list[State]): List of states.

        Returns:
            list[int]: List of random actions.
        """

    @staticmethod
    @abstractmethod
    def is_solved(states: list[State], states_goal: list[State]) -> NDArray[np.bool_]:
        """Return whether or not state is solved.

        Args:
            states (list[State]): List of states.
            states_goal (list[State]): List of goal states.

        Returns:
            NDArray[np.bool_]: Boolean numpy array where the element at index i corresponds to whether or
                not the state at index i is solved.
        """

    @abstractmethod
    def state_to_real(
        self,
        states: list[State],
        *,
        effects: Pipeline[ImageArray] | StagePipelines | None = None,
        **kwargs: EffectValue,
    ) -> NDArray[np.float32]:
        """State to real-world observation.

        Args:
            states (list[State]): List of states.
            effects: Optional effects pipeline to apply to the states.
            **kwargs: Additional keyword arguments.

        Returns:
            NDArray[np.float32]: A numpy array.
        """

    @abstractmethod
    def generate_start_states(self, num_states: int, level_seeds: list[int] | None = None) -> list[State]:
        """Generate initial states for the environment.

        Args:
            num_states (int): Number of states to generate.
            level_seeds (Optional[list[int]], optional): Seeds for random state generation,
                defaults to None.

        Returns:
            list[State]: List of generated initial states.
        """

    @abstractmethod
    def generate_goal_states(
        self,
        states: list[State],
        num_steps: int | None,
        seeds: list[int] | NDArray[np.intp] | None = None,
        **kwargs: EffectValue,
    ) -> list[State]:
        """Get the goal states for the input list of states.

        Args:
            states (list[State]): List of states.
            num_steps (int | None): Number of random steps to be taken to specify the resulting
                state as a goal state. This may or may not be used in different environments.
            seeds (list[int] | NDArray[np.intp], optional): Random seeds for goal state generation.
            **kwargs: Additional keyword arguments.

        Returns:
            list[State]: List of goal states.
        """

    def generate_episodes(
        self, num_steps_l: list[float], start_level_seed: int | None = -1, num_levels: int | None = -1
    ) -> tuple[list[State], list[State], list[list[State]], list[list[int]]]:
        """Generate episodes based on the given parameters.

        Args:
            num_steps_l (list[float]): List of number of steps for each trajectory.
            start_level_seed (Optional[int], optional): Starting seed for level generation,
                defaults to -1.
            num_levels (Optional[int], optional): Number of levels to generate, defaults to -1.

        Returns:
            Tuple[list[State], list[State], list[list[State]], list[list[int]]]: Tuple containing
                start states, goal states, trajectories, and action trajectories.
        """
        num_trajs: int = len(num_steps_l)

        # Check if the implemented method 'generate_start_states()' accepts 'level_seeds' as an argument
        has_arg: bool = "level_seeds" in signature(self.generate_start_states).parameters

        # Initialize
        states: list[State]
        if has_arg:
            # Calculating the seeds
            seeds_lst: list[int] | None = None
            if (num_levels is not None and num_levels > 0) or (start_level_seed is not None and start_level_seed > -1):
                # This branch requires both level count and starting seed.
                if num_levels is None or num_levels < 1:
                    num_levels = num_trajs

                if start_level_seed is None or start_level_seed < 0:
                    start_level_seed = np.random.randint(0, 1000000)

                trajs_per_level = num_trajs // num_levels
                extra_trajs = num_trajs % num_levels
                levels = np.arange(start_level_seed, start_level_seed + num_levels)
                seeds_np = np.concatenate((np.tile(levels, trajs_per_level), levels[:extra_trajs]))
                np.random.shuffle(seeds_np)
                seeds_lst = seeds_np.tolist()

            states = self.generate_start_states(num_trajs, level_seeds=seeds_lst)

        else:
            states = self.generate_start_states(num_trajs)

        states_walk: list[State] = states

        # Num steps
        num_steps: NDArray[np.int32] = np.array(num_steps_l, dtype=np.int32)
        num_moves_curr: NDArray[np.int32] = np.zeros(len(states), dtype=np.int32)

        # Random walk
        trajs: list[list[State]] = [[state] for state in states]
        action_trajs: list[list[int]] = [[] for _ in range(len(states))]

        moves_lt = num_moves_curr < num_steps
        while np.any(moves_lt):
            idxs: NDArray[np.intp] = np.where(moves_lt)[0]
            states_to_move = [states_walk[idx] for idx in idxs]

            actions: list[int] = self.rand_action(states_to_move)
            states_moved, _ = self.next_state(states_to_move, actions)

            for move_idx, idx in enumerate(idxs):
                trajs[idx].append(states_moved[move_idx])
                action_trajs[idx].append(actions[move_idx])
                states_walk[idx] = states_moved[move_idx]

            num_moves_curr[idxs] += 1

            moves_lt[idxs] = num_moves_curr[idxs] < num_steps[idxs]

        # get state goal pairs
        states_start: list[State] = [traj[0] for traj in trajs]
        states_goal: list[State] = [traj[-1] for traj in trajs]

        return states_start, states_goal, trajs, action_trajs

    @staticmethod
    @abstractmethod
    def get_dqn(cfg: ModelConfig) -> nn.Module:
        """Get the DQN model for this environment.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: The DQN model.
        """

    @staticmethod
    @abstractmethod
    def get_env_model_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous environment neural network model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Continuous environment model.
        """

    @staticmethod
    @abstractmethod
    def get_env_model_disc(cfg: ModelConfig) -> nn.Module:
        """Return the environment neural network model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Environment model.
        """

    @staticmethod
    @abstractmethod
    def get_encoder_disc(cfg: ModelConfig) -> nn.Module:
        """Return the encoder model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Encoder model.
        """

    @staticmethod
    @abstractmethod
    def get_encoder_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous encoder model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Continuous encoder model.
        """

    @staticmethod
    @abstractmethod
    def get_decoder_disc(cfg: ModelConfig) -> nn.Module:
        """Return the decoder model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Decoder model.
        """

    @staticmethod
    @abstractmethod
    def get_decoder_cont(cfg: ModelConfig) -> nn.Module:
        """Return the continuous decoder model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Continuous decoder model.
        """

    @staticmethod
    @abstractmethod
    def get_alignment_model(cfg: ModelConfig) -> nn.Module:
        """Return the alignment model.

        Args:
            cfg: Configuration object.

        Returns:
            nn.Module: Alignment model.
        """
