from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from spar.environments.cube3.cube3 import Cube3

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from spar.environments.abstracts.environment import ABCEnvironment
    from spar.environments.cube3 import Cube3State


def main() -> None:
    """Generate the canonical goal image for Cube3 and save it to disk."""
    out: Path = pathlib.Path("outputs/image_processing_cube3/cube3_canonical_goal.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    env: ABCEnvironment[Cube3State] = Cube3()
    start: list[Cube3State] = env.generate_start_states(1)
    goal: Cube3State = env.generate_goal_states(start, num_steps=0)[0]
    goal_img: NDArray[np.float32] = env.state_to_real([goal])[0]  # (3, H, W)
    goal_hwc: NDArray[np.float32] = np.transpose(goal_img, (1, 2, 0))
    goal_uint8: NDArray[np.uint8] = np.clip(goal_hwc * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(goal_uint8).save(out)
    print(out.resolve())
    print(goal_uint8.shape[1], goal_uint8.shape[0])


if __name__ == "__main__":
    main()
