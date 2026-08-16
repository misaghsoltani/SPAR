"""Register sticker, material, and camera effects for Cube3 renders."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import numpy as np

from spar.utils.env_utils.effects_core import EffectCategory, EffectStage, register_effect

if TYPE_CHECKING:
    from logging import Logger

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon
    from matplotlib.transforms import Bbox
    from numpy.typing import NDArray

    from spar.utils.env_utils import InteractiveCube
    from spar.utils.env_utils.viz_utils import Quaternion


__all__: list[str] = [
    "background_texture",
    "camera_left_angle",
    "camera_preset",
    "camera_variation",
    "cube_color_calibration",
    "cube_color_shift",
    "cube_directional_lighting",
    "cube_in_love",
    "cube_material_finish",
    "cube_sticker_imperfections",
    "cube_sticker_wear",
    "cube_texture_scratches",
    "cube_zoom",
]

logger: Logger = getLogger(__name__)


def _to_rgb_float32(color: str) -> NDArray[np.float32] | None:
    """Convert a Matplotlib-compatible color string to float32 RGB."""
    try:
        return np.array(mcolors.to_rgb(color), dtype=np.float32)

    except (TypeError, ValueError):
        return None


@register_effect(
    category=EffectCategory.COLOR,
    stage=EffectStage.OBJECT_RENDER,
    description="Shift cube face colors by adjusting RGB channels",
)
def cube_color_shift(cube: InteractiveCube, r: float = 0.0, g: float = 0.0, b: float = 0.0) -> InteractiveCube:
    """Shift the cube's face colors by adjusting RGB channels.

    Simulates color inconsistencies or tinting in lighting conditions.
    Can be used to create warm/cool lighting effects or simulate color blindness effects.

    Args:
        cube: The InteractiveCube instance to modify.
        r: Red offset in range [-1, 1].
        g: Green offset in range [-1, 1].
        b: Blue offset in range [-1, 1].

    Returns:
        The modified cube for pipeline compatibility.
    """
    r = np.clip(r, -1.0, 1.0)
    g = np.clip(g, -1.0, 1.0)
    b = np.clip(b, -1.0, 1.0)

    shift: NDArray[np.float32] = np.array([r, g, b], dtype=np.float32)
    new_face_colors: list[str] = []

    for color in cube.face_colors:
        rgb: NDArray[np.float32] | None = _to_rgb_float32(color)
        if rgb is None:
            new_face_colors.append(color)
            continue

        rgb = np.clip(rgb + shift, 0, 1)
        new_face_colors.append(mcolors.to_hex(tuple(rgb)))

    cube.face_colors = new_face_colors
    return cube


@register_effect(
    category=EffectCategory.COLOR,
    stage=EffectStage.OBJECT_RENDER,
    description="Scale cube face colors by RGB channel multipliers",
)
def cube_color_calibration(
    cube: InteractiveCube, r_scale: float = 1.0, g_scale: float = 1.0, b_scale: float = 1.0
) -> InteractiveCube:
    """Calibrate the cube's face colors by scaling RGB channels.

    Simulates inconsistent color reproduction, color balance issues,
    or camera sensor differences.

    Args:
        cube: The InteractiveCube instance to modify.
        r_scale: Red scale factor in range [0, 2].
        g_scale: Green scale factor in range [0, 2].
        b_scale: Blue scale factor in range [0, 2].

    Returns:
        The modified cube for pipeline compatibility.
    """
    r_scale = np.clip(r_scale, 0.0, 2.0)
    g_scale = np.clip(g_scale, 0.0, 2.0)
    b_scale = np.clip(b_scale, 0.0, 2.0)

    scale: NDArray[np.float32] = np.array([r_scale, g_scale, b_scale], dtype=np.float32)
    new_face_colors: list[str] = []

    for color in cube.face_colors:
        rgb: NDArray[np.float32] | None = _to_rgb_float32(color)
        if rgb is None:
            new_face_colors.append(color)
            continue

        rgb = np.clip(rgb * scale, 0, 1)
        new_face_colors.append(mcolors.to_hex(tuple(rgb)))

    cube.face_colors = new_face_colors
    return cube


@register_effect(
    category=EffectCategory.MATERIAL,
    stage=EffectStage.OBJECT_RENDER,
    description="Simulate sticker wear by blending colors with grayscale",
)
def cube_sticker_wear(cube: InteractiveCube, fade_factor: float = 0.5) -> InteractiveCube:
    """Simulate sticker wear by blending sticker colors with their grayscale equivalent.

    Simulates the fading and degradation of sticker colors over time and use.

    Args:
        cube: The InteractiveCube instance to modify.
        fade_factor: Fade factor in range [0, 1].

    Returns:
        The modified cube for pipeline compatibility.
    """
    fade_factor = np.clip(fade_factor, 0.0, 1.0)
    new_face_colors: list[str] = []

    for color in cube.face_colors:
        rgb: NDArray[np.float32] | None = _to_rgb_float32(color)
        if rgb is None:
            new_face_colors.append(color)
            continue

        gray: np.float32 = np.mean(rgb).astype(np.float32)
        rgb = rgb * (1 - fade_factor) + gray * fade_factor
        new_face_colors.append(mcolors.to_hex(tuple(rgb)))

    cube.face_colors = new_face_colors
    return cube


@register_effect(
    category=EffectCategory.MATERIAL,
    stage=EffectStage.OBJECT_RENDER,
    description="Add random noise to cube face colors to simulate scratches",
    requires_rng=True,
)
def cube_texture_scratches(cube: InteractiveCube, noise_level: float = 0.05) -> InteractiveCube:
    """Add random noise to cube face colors to simulate micro-texture scratches.

    Simulates small scratches, scuffs, and imperfections that develop
    on cube stickers with use.

    Args:
        cube: The InteractiveCube instance to modify.
        noise_level: Noise level in range [0, 0.5].

    Returns:
        The modified cube for pipeline compatibility.
    """
    noise_level = np.clip(noise_level, 0.0, 0.5)
    new_face_colors: list[str] = []

    for color in cube.face_colors:
        rgb: NDArray[np.float32] | None = _to_rgb_float32(color)
        if rgb is None:
            new_face_colors.append(color)
            continue

        noise: NDArray[np.float32] = np.random.uniform(-noise_level, noise_level, size=3).astype(np.float32)
        rgb = np.clip(rgb + noise, 0, 1, dtype=np.float32)
        assert rgb is not None
        new_face_colors.append(mcolors.to_hex(tuple(rgb)))

    cube.face_colors = new_face_colors
    return cube


@register_effect(
    category=EffectCategory.MATERIAL,
    stage=EffectStage.OBJECT_RENDER,
    description="Simulate different material finishes by setting plastic color",
)
def cube_material_finish(cube: InteractiveCube, finish: str = "matte") -> InteractiveCube:
    """Simulate different material finishes by setting the cube's plastic color.

    Simulates different cube materials and finishes that affect how light
    interacts with the cube surface.

    Args:
        cube: The InteractiveCube instance to modify.
        finish: Either "matte" or "glossy".

    Returns:
        The modified cube for pipeline compatibility.
    """
    finish = finish.lower()
    if finish not in {"matte", "glossy"}:
        logger.info("Warning: 'finish' should be either 'matte' or 'glossy'. Defaulting to 'matte'.")
        finish = "matte"

    cube.plastic_color = "silver" if finish == "glossy" else "black"
    return cube


@register_effect(
    category=EffectCategory.GEOMETRY,
    stage=EffectStage.OBJECT_RENDER,
    description="Randomly jitter sticker polygon vertices to simulate imperfections",
    requires_rng=True,
)
def cube_sticker_imperfections(cube: InteractiveCube, jitter: float = 0.05) -> InteractiveCube:
    """Randomly jitter sticker polygon vertices to simulate imperfections.

    Simulates physical sticker imperfections such as peeling edges,
    misalignment, or application inaccuracies.

    Args:
        cube: The InteractiveCube instance to modify.
        jitter: Maximum jitter value in range [0, 0.2].

    Returns:
        The modified cube for pipeline compatibility.
    """
    jitter = np.clip(jitter, 0.0, 0.2)

    verts: NDArray[np.float32]
    new_verts: list[list[float]]
    for patch in cube.ensure_sticker_polygons():
        verts = patch.get_xy().astype(np.float32)
        new_verts = []

        for i, (x, y) in enumerate(verts):
            if i < len(verts) - 1:
                dx, dy = np.random.uniform(-jitter, jitter, 2)
                new_verts.append([x + dx, y + dy])
            else:
                new_verts.append(verts[0])

        patch.set_xy(new_verts)

    return cube


# ================================================================================================
# TEXTURE EFFECTS
# ================================================================================================


@register_effect(
    category=EffectCategory.BACKGROUND,
    stage=EffectStage.PRE_RENDER,
    description="Apply textured background patterns to the figure",
    requires_rng=True,
)
def background_texture(
    fig: Figure,
    texture_type: str = "grid",
    color: str = "gray",
    density: float = 1.0,
    alpha: float = 0.3,
    seed: int | None = None,
) -> Figure:
    """Apply a textured background to the figure.

    Args:
        fig: The matplotlib figure to modify.
        texture_type: Type of texture. Valid options are "grid", "dots", "stripes",
            "noise", "crosshatch", or "waves".
        color: Color for the texture.
        density: Density multiplier in range [0.5, 2.0].
        alpha: Opacity in range [0.1, 1.0].
        seed: Random seed for reproducible patterns.

    Returns:
        The modified figure for pipeline compatibility.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Validate parameters
    color = color if mcolors.is_color_like(color) else "gray"
    density = float(np.clip(density, 0.5, 2.0))
    alpha = float(np.clip(alpha, 0.1, 1.0))

    # Get figure dimensions in pixels
    bbox: Bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    wpx, hpx = int(bbox.width * fig.dpi), int(bbox.height * fig.dpi)

    ax: Axes = fig.add_axes((0, 0, 1, 1), zorder=-1)
    ax.set_axis_off()

    tt: str = texture_type.lower()
    gap: int
    if tt == "grid":
        gap = max(1, int(10 / density))
        for x in range(0, wpx, gap):
            ax.axvline(x / wpx, color=color, lw=0.5, alpha=alpha)
        for y in range(0, hpx, gap):
            ax.axhline(y / hpx, color=color, lw=0.5, alpha=alpha)

    elif tt == "dots":
        gap = max(1, int(15 / density))
        dot_xs = np.arange(0, wpx, gap) / wpx
        dot_ys = np.arange(0, hpx, gap) / hpx
        dot_x_mesh, dot_y_mesh = np.meshgrid(dot_xs, dot_ys)
        ax.scatter(dot_x_mesh.ravel(), dot_y_mesh.ravel(), s=3 * density, color=color, alpha=alpha)

    elif tt == "stripes":
        orient = np.random.choice(["vertical", "horizontal"])
        gap = max(1, int(20 / density))
        if orient == "vertical":
            for x in range(0, wpx, gap):
                ax.axvline(x / wpx, color=color, lw=gap / 2, alpha=alpha)
        else:
            for y in range(0, hpx, gap):
                ax.axhline(y / hpx, color=color, lw=gap / 2, alpha=alpha)

    elif tt == "noise":
        g: int = int(100 * density)
        grid: NDArray[np.float32] = np.random.rand(g, g).astype(np.float32)
        noise_xs, noise_ys = np.linspace(0, 1, g), np.linspace(0, 1, g)
        ax.pcolormesh(noise_xs, noise_ys, grid, cmap="gray", alpha=alpha, shading="auto")

    elif tt == "crosshatch":
        gap = max(1, int(20 / density))
        for k in range(-hpx, wpx, gap):
            ax.plot([k / wpx, (k + hpx) / wpx], [0, 1], color=color, lw=0.5, alpha=alpha)
            ax.plot([k / wpx, (k + hpx) / wpx], [1, 0], color=color, lw=0.5, alpha=alpha)

    elif tt == "waves":
        wave_xs: NDArray[np.float32] = np.linspace(0, 1, wpx).astype(np.float32)
        n = int(5 * density)
        for ph in np.linspace(0, 2 * np.pi, int(3 * density)):
            wave_ys: NDArray[np.float64] = 0.5 + 0.4 * np.sin(n * 2 * np.pi * wave_xs + ph)
            ax.plot(wave_xs, wave_ys, color=color, lw=0.8, alpha=alpha)

    else:  # fallback
        logger.warning(f"Warning: Invalid texture_type '{texture_type}'. Using default 'grid'.")
        return background_texture(fig, "grid", color, density, alpha, seed)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if seed is not None:
        np.random.seed(None)

    return fig


@register_effect(
    category=EffectCategory.GEOMETRY,
    stage=EffectStage.OBJECT_RENDER,
    description="Apply random camera angle variations to cube visualization",
    requires_rng=True,
)
def camera_variation(cube: InteractiveCube, yaw: float = -20, pitch: float = -10, roll: float = -8) -> InteractiveCube:
    """Apply random camera angle variations to the cube visualization.

    The renderer recalculates its viewport after the pose changes so the cube
    remains inside the frame.

    Args:
        cube: The InteractiveCube instance to modify.
        yaw: Yaw angle in degrees (rotation around z-axis).
        pitch: Pitch angle in degrees (rotation around y-axis).
        roll: Roll angle in degrees (rotation around x-axis).

    Returns:
        The modified cube with camera pose applied and viewport adjusted.
    """
    cube.set_camera_offsets(yaw=yaw, pitch=pitch, roll=roll)

    return cube


@register_effect(
    category=EffectCategory.GEOMETRY,
    stage=EffectStage.OBJECT_RENDER,
    description="Set specific camera angles for left view",
)
def camera_left_angle(cube: InteractiveCube, yaw: float = 0, pitch: float = 0, roll: float = 0) -> InteractiveCube:
    """Set camera angles for the left view.

    The renderer recalculates its viewport after the pose changes so the cube remains inside the frame.

    Args:
        cube: The InteractiveCube instance to modify.
        yaw: Yaw angle in degrees.
        pitch: Pitch angle in degrees.
        roll: Roll angle in degrees.

    Returns:
        The modified cube with specified camera pose and adjusted viewport.
    """
    cube.set_camera_pose(yaw=yaw, pitch=pitch, roll=roll, base_quat=cube.get_start_rotation())

    return cube


@register_effect(
    category=EffectCategory.GEOMETRY,
    stage=EffectStage.OBJECT_RENDER,
    description="Apply preset camera angles for common viewpoints",
)
def camera_preset(cube: InteractiveCube, preset: str = "default") -> InteractiveCube:
    """Apply one of the named camera poses used by Cube3 figures.

    The renderer recalculates its viewport after the pose changes so the cube
    remains inside the frame.

    Args:
        cube: The InteractiveCube instance to modify.
        preset: Camera preset name. Valid options are "default", "top", "side",
            "corner", "low", or "high".

    Returns:
        The modified cube with preset camera pose and adjusted viewport.
    """
    presets: dict[str, tuple[float, float, float]] = {
        "default": (0, 0, 0),
        "top": (0, -45, 0),
        "side": (90, 0, 0),
        "corner": (45, -30, 15),
        "low": (30, 20, 0),
        "high": (-30, -20, 0),
    }

    if preset not in presets:
        logger.warning(f"Warning: Unknown preset '{preset}'. Using 'default'.")
        preset = "default"

    yaw, pitch, roll = presets[preset]

    cube.set_camera_pose(yaw=yaw, pitch=pitch, roll=roll, base_quat=cube.get_start_rotation())

    return cube


@register_effect(
    category=EffectCategory.GEOMETRY,
    stage=EffectStage.OBJECT_RENDER,
    description="Scale and position the cube without changing the background",
)
def cube_zoom(
    cube: InteractiveCube, zoom_factor: float = 1.0, offset_x: float = 0.0, offset_y: float = 0.0
) -> InteractiveCube:
    """Zoom in or out on the cube while keeping the background untouched.

    Scale the cube's 3D geometry, adjust the viewport, and apply the requested position offsets.

    Args:
        cube: The InteractiveCube instance to modify.
        zoom_factor: Zoom factor where 1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out.
            Values are clamped to [0.1, 5.0].
        offset_x: Horizontal offset of the cube in viewport coordinates.
            Range: [-2.0, 2.0], where positive moves right.
        offset_y: Vertical offset of the cube in viewport coordinates.
            Range: [-2.0, 2.0], where positive moves up.

    Returns:
        The modified cube with zoom effect and positioning applied.
    """
    # Clamp parameters to reasonable bounds
    zoom_factor = np.clip(zoom_factor, 0.1, 5.0)
    offset_x = np.clip(offset_x, -2.0, 2.0)
    offset_y = np.clip(offset_y, -2.0, 2.0)

    # Only apply zoom if it's different from 1.0 (no zoom)
    if abs(zoom_factor - 1.0) < 1e-6 and abs(offset_x) < 1e-6 and abs(offset_y) < 1e-6:
        return cube

    cube.zoom_geometry(zoom_factor=float(zoom_factor), offset_x=float(offset_x), offset_y=float(offset_y))

    return cube


@register_effect(
    category=EffectCategory.LIGHTING,
    stage=EffectStage.OBJECT_RENDER,
    description="Directional shading.",
    is_destructive=True,
)
def cube_directional_lighting(
    cube: InteractiveCube,
    ambient_intensity: float = 0.2,  # [0,1]
    shadow_intensity: float = 0.6,  # [0,1]
    gradient_strength: float = 1.5,  # > 0 (used as a gamma adjustment)
    subsurface_scattering: float = 0.1,  # [0,0.5]
    azdeg: float = 315.0,
    altdeg: float = 45.0,
) -> InteractiveCube:
    """Apply directional lighting effects to the cube stickers.

    Modulate each sticker's face color using a Lambert-style calculation based on the light direction.

    Args:
        cube: The InteractiveCube instance to modify.
        ambient_intensity: Ambient light intensity in range [0, 1].
        shadow_intensity: Shadow intensity in range [0, 1].
        gradient_strength: Gradient strength factor (> 0, used as gamma correction).
        subsurface_scattering: Subsurface scattering amount in range [0, 0.5].
        azdeg: Light azimuth angle in degrees.
        altdeg: Light altitude angle in degrees.

    Returns:
        The modified cube with lighting effects applied.

    Raises:
        RuntimeError: If sticker polygons are not available or count is incorrect.
    """
    polys: list[Polygon] = cube.ensure_sticker_polygons()

    if len(polys) != 6 * cube.N**2:
        raise RuntimeError(f"Expected {6 * cube.N**2} stickers, got {len(polys)}")

    # 1) Build the light direction
    az: float = np.deg2rad(azdeg)
    al: float = np.deg2rad(altdeg)
    light_dir: NDArray[np.float32] = np.array([np.cos(al) * np.cos(az), np.cos(al) * np.sin(az), np.sin(al)]).astype(
        np.float32
    )
    light_dir /= np.linalg.norm(light_dir)

    # 2) Clamp parameters
    amb = float(np.clip(ambient_intensity, 0.0, 1.0))
    shad = float(np.clip(shadow_intensity, 0.0, 1.0))
    subs = float(np.clip(subsurface_scattering, 0.0, 0.5))
    gamma = max(1e-3, gradient_strength)

    shadow_floor: float = 0.0 if shad <= 0.0 else 1.0 - shad
    max_inten: float = 1.5 if (amb < 0.3 and shad > 0.5) else 1.2

    # 3) Get rotation that includes any pending camera offsets
    current_rot: Quaternion = cube.get_rotation_with_camera_offsets()

    # 4) Grab the 6 canonical face normals, rotated by the final orientation
    face_normals: NDArray[np.float32] = np.array(
        [[0, 1, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1], [-1, 0, 0], [1, 0, 0]], dtype=np.float32
    )

    rot_matrix: NDArray[np.float32] = current_rot.as_rotation_matrix()
    face_normals = (rot_matrix @ face_normals.T).T

    # 5) Determine each sticker's base color and physical position for normals
    patches_per_face: int = cube.N**2
    face_ids: NDArray[np.uint8] = (
        cube.colors // patches_per_face
    )  # length 6P array of 0...5 (which face color each sticker has)
    base_rgbs: NDArray[np.float32] = np.array([mcolors.to_rgb(cube.face_colors[fid]) for fid in face_ids]).astype(
        np.float32
    )  # shape (6P,3)

    # Use physical position for normals: sticker i is physically on face (i // P)
    physical_face_ids: NDArray[np.uint8] = (np.arange(6 * patches_per_face) // patches_per_face).astype(
        np.uint8
    )  # [0,0,...,1,1,...,2,2,...,5,5,...]
    normals: NDArray[np.float32] = face_normals[physical_face_ids]  # shape (6P,3)

    # 6) Lambert + ambient + back-face bleed
    lam: NDArray[np.float32] = normals @ light_dir  # dot product per sticker: (6P,3) @ (3,) -> (6P,)
    lam = np.where(lam <= 0.0, subs, lam)  # subsurface for back-faces
    intens: NDArray[np.float32] = amb + (1.0 - amb) * lam  # ambient mix
    intens = np.clip(intens, shadow_floor, max_inten)

    # 7) Apply gamma-style contrast for gradient_strength.
    intens = np.power(intens, 1.0 / gamma)

    # 8) Modulate each patch's facecolor in one vectorized swoop
    lit_rgbs: NDArray[np.float32] = (base_rgbs * intens[:, None]).astype(np.float32)  # shape (6P,3)

    existing_alpha: float
    rgba: tuple[float, float, float, float]
    for poly, rgb in zip(polys, lit_rgbs, strict=False):
        # Preserve any existing alpha and pass a tuple (acceptable color type)
        existing_alpha = float(poly.get_facecolor()[-1])
        rgba = (float(rgb[0]), float(rgb[1]), float(rgb[2]), existing_alpha)
        poly.set_facecolor(rgba)

    cube.figure.canvas.draw_idle()
    return cube


def create_heart_path(center_x: float, center_y: float, scale: float) -> MplPath:
    """Create a heart-shaped path centered at the given coordinates.

    The path uses the parametric equations ``x = 16sin³(t)`` and ``y = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)``.

    Args:
        center_x: X coordinate of heart center in [0, 1].
        center_y: Y coordinate of heart center in [0, 1].
        scale: Scaling factor for heart size.

    Returns:
        A matplotlib Path object representing the heart shape.
    """
    t: NDArray[np.float32] = np.linspace(0, 2 * np.pi, 100, dtype=np.float32)

    # Parametric heart equations (normalized)
    x: NDArray[np.float32] = 16 * np.sin(t) ** 3
    y: NDArray[np.float32] = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

    # Normalize to unit size and scale
    x = np.asarray((x / 16) * scale * 0.5, dtype=np.float32)
    y = np.asarray((y / 16) * scale * 0.5, dtype=np.float32)

    # Center the heart
    x += center_x
    y += center_y

    # Create path vertices
    vertices: list[tuple[float, float]] = list(zip(x, y, strict=True))
    codes: list[np.uint8] = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 1)

    return MplPath(vertices, codes)


@register_effect(
    category=EffectCategory.BACKGROUND,
    stage=EffectStage.PRE_RENDER,
    description="Add heart shapes to the background",
    requires_rng=True,
)
def cube_in_love(
    fig: Figure,
    heart_count: int = 15,
    color: str = "red",
    size_range: tuple[float, float] = (0.02, 0.08),
    alpha: float = 0.6,
    seed: int | None = None,
) -> Figure:
    """Add heart shapes to the background of the cube visualization.

    Args:
        fig: The matplotlib figure to modify.
        heart_count: Number of hearts to draw in range [5, 50].
        color: Color for the hearts (accepts color names, hex codes, etc.).
        size_range: Tuple of (min_size, max_size) for heart scaling in range [0.01, 0.15].
        alpha: Opacity of hearts in range [0.1, 1.0].
        seed: Random seed for reproducible heart placement.

    Returns:
        The modified figure with hearts in background.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Validate parameters
    heart_count = int(np.clip(heart_count, 5, 50))
    color = color if mcolors.is_color_like(color) else "red"
    min_size, max_size = size_range
    min_size = float(np.clip(min_size, 0.01, 0.15))
    max_size = float(np.clip(max_size, min_size, 0.15))
    alpha = float(np.clip(alpha, 0.1, 1.0))

    # Create background axis for hearts
    ax: Axes = fig.add_axes((0, 0, 1, 1), zorder=-1)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Generate random heart positions and sizes
    x: float
    y: float
    size: float
    heart_path: MplPath
    color_variation: float
    heart_color: str | tuple[float, float, float]
    heart_patch: PathPatch
    for _ in range(heart_count):
        # Random position (avoid edges to prevent clipping)
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)

        # Random size within specified range
        size = np.random.uniform(min_size, max_size)

        # Create heart path
        heart_path = create_heart_path(x, y, size)

        # Vary the red and pink tones.
        color_variation = np.random.uniform(0.8, 1.2)
        if color == "red":
            heart_color = (min(1.0, 0.8 * color_variation), 0.1, 0.1)
        elif color == "pink":
            heart_color = (1.0, min(1.0, 0.7 * color_variation), min(1.0, 0.8 * color_variation))
        else:
            heart_color = color

        # Create and add heart patch
        heart_patch = PathPatch(heart_path, facecolor=heart_color, edgecolor="none", alpha=alpha, zorder=-1)
        ax.add_patch(heart_patch)

    # Reset random seed if it was set
    if seed is not None:
        np.random.seed(None)

    return fig
