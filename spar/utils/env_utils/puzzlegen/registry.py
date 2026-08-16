"""Registry for puzzle environments compatible with gymnasium>=1.0.0."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
from gymnasium.envs import registration

if TYPE_CHECKING:
    from typing import TypedDict

    from numpy import uint8
    from numpy.typing import NDArray
    from typing_extensions import Unpack

    class _IceSliderKwargs(TypedDict, total=False):
        ice_density: int
        easy: bool
        render_mode: str | None
        min_sol_len: int
        size: int
        max_tries: int
        max_episode_steps: int | None
        disable_env_checker: bool | None

    class _DigitJumpKwargs(TypedDict, total=False):
        render_mode: str | None
        min_sol_len: int
        size: int
        max_tries: int
        max_episode_steps: int | None
        disable_env_checker: bool | None


def register_puzzle_environments() -> None:
    """Register the puzzle environments and their Gymnasium specifications."""
    # Register IceSlider environment (check if already registered to avoid warnings)
    if "IceSlider-v0" not in registration.registry:
        registration.register(
            id="IceSlider-v0",
            entry_point="spar.utils.env_utils.puzzlegen.ice_slider:IceSlider",
            max_episode_steps=1000,
            reward_threshold=10.0,
            kwargs={"ice_density": 4, "easy": True, "render_mode": "rgb_array", "min_sol_len": 8},
        )

    # Register DigitJump environment
    if "DigitJump-v0" not in registration.registry:
        registration.register(
            id="DigitJump-v0",
            entry_point="spar.utils.env_utils.puzzlegen.digit_jump:DigitJump",
            max_episode_steps=1000,
            reward_threshold=10.0,
            kwargs={"render_mode": "rgb_array", "min_sol_len": 8},
        )


def make_ice_slider(**kwargs: Unpack[_IceSliderKwargs]) -> gym.Env[NDArray[uint8], int]:
    """Create an IceSlider environment with proper spec.

    Args:
        **kwargs: Environment configuration

    Returns:
        Configured IceSlider environment with spec
    """
    return gym.make("IceSlider-v0", **kwargs)


def make_digit_jump(**kwargs: Unpack[_DigitJumpKwargs]) -> gym.Env[NDArray[uint8], int]:
    """Create a DigitJump environment with proper spec.

    Args:
        **kwargs: Environment configuration

    Returns:
        Configured DigitJump environment with spec
    """
    return gym.make("DigitJump-v0", **kwargs)
