from __future__ import annotations

from typing import TYPE_CHECKING

from spar.utils.search_utils.viz_utils import ImageHandler

if TYPE_CHECKING:
    from numpy import float32
    from numpy.typing import NDArray
    import torch
    from torch import nn

    from spar.environments.abstracts.environment import ABCEnvironment
    from spar.environments.abstracts.state import ABCState


def is_valid_soln(
    env: ABCEnvironment[ABCState],
    state: ABCState,
    state_goal: ABCState,
    soln: list[int],
    *,
    decoder: nn.Module | None = None,
    device: torch.device | None = None,
    state_idx: int | None = None,
    path: list[NDArray[float32]] | None = None,
    save_imgs_dir: str | None = None,
    save_imgs: bool = False,
    save_gif: bool = False,
    gif_fps: int = 5,
) -> bool:
    """Checks if the solution is valid.

    Args:
        env (ABCEnvironment[ABCState]): The environment instance.
        state (ABCState): The initial state.
        state_goal (ABCState): The goal state.
        soln (list[int]): The list of moves.
        decoder (Optional[nn.Module], optional): The decoder neural network. Defaults to None.
        device (Optional[torch.device], optional): The device to run computations on. Defaults to
            None.
        state_idx (Optional[int], optional): The index of the state. Defaults to None.
        path: Optional path of float32 state arrays. Defaults to None.
        save_imgs_dir (Optional[str], optional): The directory to save images. Defaults to None.
        save_imgs (bool, optional): Whether to save images (PNG). Defaults to False.
        save_gif (bool, optional): Whether to save an animated GIF. Defaults to False.
        gif_fps (int, optional): Frames per second for GIFs. Defaults to 5.

    Returns:
        bool: True if the solution is valid, False otherwise.
    """
    state_soln: ABCState = state
    move: int

    for move in soln:
        state_soln = env.next_state([state_soln], [move])[0][0]

    if save_imgs or save_gif:
        assert decoder is not None, "decoder must be provided when saving visuals"
        assert device is not None, "device must be provided when saving visuals"
        assert state_idx is not None, "state_idx must be provided when saving visuals"
        assert path is not None, "path must be provided when saving visuals"
        assert save_imgs_dir is not None, "save_imgs_dir must be provided when saving visuals"

        ImageHandler.save_solution_visuals(
            env=env,
            state=state,
            soln=soln,
            decoder=decoder,
            device=device,
            state_idx=state_idx,
            path=path,
            save_imgs_dir=save_imgs_dir,
            save_imgs=save_imgs,
            save_gif=save_gif,
            gif_fps=gif_fps,
        )

    return env.is_solved([state_soln], [state_goal])[0]
