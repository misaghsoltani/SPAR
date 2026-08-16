"""Sokoban style effect backed by exported visual style assets."""

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from spar.environments.sokoban.sokoban import SokobanEnv, _get_surfaces
from spar.utils.env_utils.effects_core import EffectCategory, EffectStage, register_effect

if TYPE_CHECKING:
    from typing import Final, TypeAlias

    from cv2.typing import MatLike
    from numpy.typing import NDArray

    from spar.environments.sokoban.sokoban import SokobanState


__all__: list[str] = [
    "NON_DEFAULT_STYLE_IDS",
    "SOKOBAN_STYLE_EFFECT",
    "STYLE_IDS",
    "render_sokoban_style",
    "sokoban_style",
]

SOKOBAN_STYLE_EFFECT: Final[str] = "sokoban_style"
GRID_SIZE: Final[int] = 10
SPAR_DEFAULT_IMAGE_DIM: Final[int] = 40
STYLE_RESIZE_INTERPOLATION: Final[int] = cv2.INTER_LINEAR
FLOOR: Final[int] = 0
WALL: Final[int] = 1
PLAYER: Final[int] = 2
BOX: Final[int] = 3

NON_DEFAULT_STYLE_IDS: Final[tuple[str, ...]] = (
    "retro-pixel",
    "isometric-blueprint",
    "zen-garden",
    "neon-arcade",
    "storybook-sketch",
    "lunar-outpost",
    "circuit-foundry",
    "botanical-glasshouse",
    "volcanic-forge",
    "arctic-expedition",
)
STYLE_IDS: Final[tuple[str, ...]] = ("spar-default", *NON_DEFAULT_STYLE_IDS)
SokobanStyle: TypeAlias = Literal[
    "spar-default",
    "retro-pixel",
    "isometric-blueprint",
    "zen-garden",
    "neon-arcade",
    "storybook-sketch",
    "lunar-outpost",
    "circuit-foundry",
    "botanical-glasshouse",
    "volcanic-forge",
    "arctic-expedition",
]
_STYLE_ALIASES: Final[dict[str, str]] = {style_id: style_id for style_id in STYLE_IDS} | {
    "default": "spar-default",
    "spar_default": "spar-default",
    "retro_pixel": "retro-pixel",
    "isometric_blueprint": "isometric-blueprint",
    "zen_garden": "zen-garden",
    "neon_arcade": "neon-arcade",
    "storybook_sketch": "storybook-sketch",
    "lunar_outpost": "lunar-outpost",
    "circuit_foundry": "circuit-foundry",
    "botanical_glasshouse": "botanical-glasshouse",
    "volcanic_forge": "volcanic-forge",
    "arctic_expedition": "arctic-expedition",
}


@register_effect(
    SOKOBAN_STYLE_EFFECT,
    category=EffectCategory.MATERIAL,
    stage=EffectStage.OBJECT_RENDER,
    description="Render Sokoban states with a configurable visual style.",
    performance_level=1,
)
def sokoban_style(
    state: SokobanState, style: SokobanStyle = "spar-default", *, image_dim: int = 40
) -> NDArray[np.float32]:
    """Render a Sokoban state using one configurable visual style."""
    return render_sokoban_style(state, style, image_dim=image_dim)


def render_sokoban_style(state: SokobanState, style: str, *, image_dim: int = 40) -> NDArray[np.float32]:
    """Render a Sokoban state with SPAR default assets or exported style sheets."""
    style_id: str = _normalize_style_id(style)
    if style_id == "spar-default":
        return _render_default_style(state, image_dim=image_dim)
    return _render_exported_style(state, style_id, image_dim=image_dim)


def _render_default_style(state: SokobanState, *, image_dim: int) -> NDArray[np.float32]:
    del image_dim
    env = _default_renderer_for_state(state, image_dim=SPAR_DEFAULT_IMAGE_DIM)
    return SokobanEnv.state_to_rgb(env, state).astype(np.float32, copy=False)


def _render_exported_style(state: SokobanState, style_id: str, *, image_dim: int) -> NDArray[np.float32]:
    room: NDArray[np.intp] = _state_room(state)
    tile_grid, cell_size = _exported_tile_grid(style_id)
    height, width = room.shape
    if height > GRID_SIZE or width > GRID_SIZE:
        raise ValueError(
            f"Sokoban style assets cover grids up to {GRID_SIZE}x{GRID_SIZE}. "
            f"got {height}x{width} for style {style_id!r}."
        )

    row_indices = np.arange(height, dtype=np.intp)[:, np.newaxis]
    col_indices = np.arange(width, dtype=np.intp)[np.newaxis, :]
    tiles: NDArray[np.uint8] = tile_grid[room, row_indices, col_indices]
    rgb_u8: NDArray[np.uint8] = tiles.transpose(0, 2, 1, 3, 4).reshape(height * cell_size, width * cell_size, 3)
    rgb = rgb_u8.astype(np.float32)
    rgb *= np.float32(1.0 / 255.0)
    return _resize_if_needed(rgb, image_dim)


def _resize_if_needed(rgb: NDArray[np.float32], image_dim: int) -> NDArray[np.float32]:
    target_dim = _normalized_image_dim(image_dim)
    if rgb.shape[0] != target_dim or rgb.shape[1] != target_dim:
        return cv2.resize(rgb, (target_dim, target_dim), interpolation=STYLE_RESIZE_INTERPOLATION).astype(
            np.float32, copy=False
        )
    return rgb.astype(np.float32, copy=False)


def _normalized_image_dim(image_dim: int) -> int:
    try:
        dim = image_dim
    except (TypeError, ValueError):
        return SPAR_DEFAULT_IMAGE_DIM
    if dim <= 0:
        return SPAR_DEFAULT_IMAGE_DIM
    return dim


def _state_room(state: SokobanState) -> NDArray[np.intp]:
    env: SokobanEnv = _default_renderer_for_state(state, image_dim=SPAR_DEFAULT_IMAGE_DIM)
    rendered_room: NDArray[np.intp] = SokobanEnv.get_render_array(env, state)
    render_to_style: NDArray[np.intp] = np.asarray([WALL, FLOOR, PLAYER, BOX], dtype=np.intp)
    return render_to_style[rendered_room]


@cache
def _default_surface_stack() -> NDArray[np.uint8]:
    return np.stack(_get_surfaces(), axis=0).astype(np.uint8, copy=False)


def _default_renderer_for_state(state: SokobanState, *, image_dim: int) -> SokobanEnv:
    return SokobanEnv.from_render_assets(
        dim=int(state.walls.shape[0]), image_dim=image_dim, surface_stack=_default_surface_stack()
    )


@cache
def _exported_tile_grid(style_id: str) -> tuple[NDArray[np.uint8], int]:
    path = _style_sheet_path(style_id)
    rgb = _load_style_sheet_rgb(path, style_id)
    cell_size = _sheet_cell_size(rgb, style_id)
    tile_grid: NDArray[np.uint8] = rgb.reshape(4, GRID_SIZE, cell_size, GRID_SIZE, cell_size, 3).transpose(
        0, 1, 3, 2, 4, 5
    )
    return np.ascontiguousarray(tile_grid), cell_size


def _load_style_sheet_rgb(path: str, style_id: str) -> NDArray[np.uint8]:
    sheet: MatLike | None = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if sheet is None:
        raise FileNotFoundError(f"Could not load Sokoban style asset sheet: {path}")
    if sheet.ndim != 3 or sheet.shape[2] not in {3, 4}:
        raise ValueError(f"Invalid Sokoban style sheet shape for {style_id!r}: {sheet.shape}")

    if sheet.shape[2] == 3:
        return np.asarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    rgba: NDArray[np.uint8] = np.asarray(cv2.cvtColor(sheet, cv2.COLOR_BGRA2RGBA), dtype=np.uint8)
    rgb_f = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) * np.float32(1.0 / 255.0)
    if np.any(alpha < np.float32(1.0)):
        rgb_f = _composite_transparent_tiles(rgb_f, alpha, style_id)

    return np.clip(np.rint(rgb_f), 0, 255).astype(np.uint8)


def _composite_transparent_tiles(
    rgb: NDArray[np.float32], alpha: NDArray[np.float32], style_id: str
) -> NDArray[np.float32]:
    cell_size: int = _sheet_cell_size(np.zeros((*rgb.shape[:2], 3), dtype=np.uint8), style_id)
    rgb_grid = rgb.reshape(4, GRID_SIZE, cell_size, GRID_SIZE, cell_size, 3).transpose(0, 1, 3, 2, 4, 5)
    alpha_grid = alpha.reshape(4, GRID_SIZE, cell_size, GRID_SIZE, cell_size, 1).transpose(0, 1, 3, 2, 4, 5)
    floor_rgb = rgb_grid[FLOOR]
    floor_alpha = alpha_grid[FLOOR]
    opaque_floor = floor_rgb * floor_alpha + np.float32(255.0) * (np.float32(1.0) - floor_alpha)
    composited = rgb_grid * alpha_grid + opaque_floor[np.newaxis, ...] * (np.float32(1.0) - alpha_grid)
    return composited.transpose(0, 1, 3, 2, 4, 5).reshape(rgb.shape)


def _style_sheet_path(style_id: str) -> str:
    normalized: str = _normalize_style_id(style_id)
    if normalized not in NON_DEFAULT_STYLE_IDS:
        raise ValueError(f"Style {style_id!r} does not use exported style assets.")
    package_name: str | None = __package__
    if package_name is None:
        raise RuntimeError("Sokoban style assets require a package context")
    return str(resources.files(package_name) / "assets" / "sokoban_styles" / f"{normalized}.png")


def _sheet_cell_size(sheet: NDArray[np.uint8], style_id: str) -> int:
    height, width, channels = sheet.shape
    if channels != 3 or width % GRID_SIZE != 0:
        raise ValueError(f"Invalid Sokoban style sheet shape for {style_id!r}: {sheet.shape}")
    cell_size = width // GRID_SIZE
    if height != cell_size * GRID_SIZE * 4:
        raise ValueError(f"Invalid Sokoban style sheet shape for {style_id!r}: {sheet.shape}")
    return cell_size


def _normalize_style_id(style: str) -> str:
    key: str = style.strip().lower().replace(" ", "-")
    normalized: str | None = _STYLE_ALIASES.get(key, _STYLE_ALIASES.get(key.replace("-", "_")))
    if normalized is None:
        choices: str = ", ".join(STYLE_IDS)
        raise ValueError(f"Unknown Sokoban style {style!r}. Expected one of: {choices}")
    return normalized
