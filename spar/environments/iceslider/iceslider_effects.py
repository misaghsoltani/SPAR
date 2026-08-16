"""Register render effects that operate on IceSlider instances and assets."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from spar.utils.env_utils.effects_core import EffectCategory, EffectStage, register_effect

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from spar.utils.env_utils.puzzlegen.ice_slider import IceSlider


__all__: list[str] = [
    "change_render_mode",
    "change_texture_assets",
    "color_shift",
    "color_tint_assets",
    "neon_glow",
    "retro_pixel",
    "vintage_filter",
]


# Global backup storage for original textures
_TEXTURE_BACKUPS: dict[int, dict[str, NDArray[np.uint8]]] = {}


def _gray_to_rgb_float32(gray: NDArray[np.float32]) -> NDArray[np.float32]:
    """Expand a grayscale image to RGB with float32 dtype and no extra temporaries."""
    gray_f: NDArray[np.float32] = np.asarray(gray, dtype=np.float32)
    rgb: NDArray[np.float32] = np.empty((*gray_f.shape, 3), dtype=np.float32)
    rgb[..., 0] = gray_f
    rgb[..., 1] = gray_f
    rgb[..., 2] = gray_f
    return rgb


def _clear_rendering_cache(ice_slider: IceSlider) -> None:
    """Clear render arrays after an effect changes renderer state.

    Args:
        ice_slider: Environment whose render cache is stale.
    """
    ice_slider.clear_rendering_cache()


def _fix_basic_mode_colormap(ice_slider: IceSlider) -> None:
    """Fix basic mode color map format to prevent tensor shape errors."""
    if getattr(ice_slider, "_cm", None) is None:
        return

    # Basic mode indexes a two-row RGB color map.
    # Basic rendering indexes a two-row RGB color map.
    # - index 0 = rock color (brown)
    # - index 1 = ice color (light blue)

    # Default colors if textures are missing/malformed
    default_rock: NDArray[np.uint8] = np.array([139, 69, 19], dtype=np.uint8)  # Brown
    default_ice: NDArray[np.uint8] = np.array([173, 216, 230], dtype=np.uint8)  # Light blue

    # Try to extract colors from existing textures if available
    rock_color: NDArray[np.uint8] = default_rock
    ice_color: NDArray[np.uint8] = default_ice

    if hasattr(ice_slider, "rock_rgb") and ice_slider.rock_rgb is not None:
        rock_tex = ice_slider.rock_rgb
        if rock_tex.size >= 3:
            if rock_tex.ndim >= 2:
                rock_color = rock_tex.reshape(-1, rock_tex.shape[-1])[0][:3].astype(np.uint8)
            else:
                rock_color = rock_tex[:3].astype(np.uint8)

    if hasattr(ice_slider, "ice_rgb") and ice_slider.ice_rgb is not None:
        ice_tex = ice_slider.ice_rgb
        if ice_tex.size >= 3:
            if ice_tex.ndim >= 2:
                ice_color = ice_tex.reshape(-1, ice_tex.shape[-1])[0][:3].astype(np.uint8)
            else:
                ice_color = ice_tex[:3].astype(np.uint8)

    # The two by three color map supports direct binary-mask indexing.
    ice_slider.set_basic_colormap(rock_color, ice_color)


def _get_predefined_asset_set(asset_set: str) -> dict[str, str]:
    """Get predefined asset paths for different themes."""
    base_path: str = "spar/data/environments/iceslider/assets"

    asset_sets: dict[str, dict[str, str]] = {
        "winter": {
            "rock": f"{base_path}/winter/rock_snow.png",
            "ice": f"{base_path}/winter/ice_blue.png",
            "player": f"{base_path}/winter/player_winter.png",
            "goal": f"{base_path}/winter/goal_cabin.png",
        },
        "desert": {
            "rock": f"{base_path}/desert/rock_sand.png",
            "ice": f"{base_path}/desert/sand_dune.png",
            "player": f"{base_path}/desert/player_nomad.png",
            "goal": f"{base_path}/desert/goal_oasis.png",
        },
        "neon": {
            "rock": f"{base_path}/neon/rock_metal.png",
            "ice": f"{base_path}/neon/ice_neon.png",
            "player": f"{base_path}/neon/player_cyber.png",
            "goal": f"{base_path}/neon/goal_portal.png",
        },
    }
    return asset_sets.get(asset_set, {})


def _load_texture_from_path(texture_path: str) -> NDArray[np.uint8]:
    """Load texture from file path with fallback to solid color."""
    if pathlib.Path(texture_path).exists():
        try:
            img = Image.open(texture_path).convert("RGB")
            return np.array(img, dtype=np.uint8)
        except Exception:
            pass

    # Fallback to solid colors based on filename
    if "rock" in texture_path.lower():
        return np.array([139, 69, 19], dtype=np.uint8)  # Brown
    if "ice" in texture_path.lower():
        return np.array([173, 216, 230], dtype=np.uint8)  # Light blue
    if "player" in texture_path.lower():
        return np.array([255, 20, 147], dtype=np.uint8)  # Deep pink
    if "goal" in texture_path.lower():
        return np.array([50, 205, 50], dtype=np.uint8)  # Lime green
    return np.array([128, 128, 128], dtype=np.uint8)  # Gray


def _apply_color_tint(texture: NDArray[np.uint8], tint: tuple[float, float, float]) -> NDArray[np.uint8]:
    """Apply color tinting to a texture."""
    if texture.ndim == 3 and texture.shape[2] >= 3:
        # Multi-channel texture
        tinted = texture.astype(np.float32)
        tinted[:, :, :3] *= tint  # Apply tint to RGB channels
        return np.clip(tinted, 0, 255).astype(np.uint8)
    # Single channel or basic texture - average the tint
    avg_tint = sum(tint) / 3.0
    tinted = texture.astype(np.float32) * avg_tint
    return np.clip(tinted, 0, 255).astype(np.uint8)


# IMAGE TYPE/RANGE HELPERS
def _uint8_to_float01(image: NDArray[np.number] | None) -> tuple[NDArray[np.float32] | None, bool]:
    """Convert input image to float32 in [0,1].

    Returns (image_float01, was_chw).
    - Accepts uint8 (0-255) and float arrays (assumed either 0-1 or 0-255).
    - Detects CHW vs HWC by treating first dim<=4 as CHW.
    - If input is None or empty, returns (None, False).
    """
    if image is None or getattr(image, "size", 0) == 0:
        return None, False

    arr = image
    return_chw = False
    # Detect CHW (channels-first) if first dim is small
    if arr.ndim == 3 and arr.shape[0] <= 4:
        # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
        return_chw = True

    # Now arr is HWC or 2D
    # If uint8 or large float values, convert from 0-255 -> 0-1
    if np.issubdtype(arr.dtype, np.integer):
        arr_f = arr.astype(np.float32) / 255.0
    else:
        # float types: decide by max value
        arr_f = arr.astype(np.float32)
        maxv = float(np.nanmax(arr_f)) if arr_f.size else 0.0
        if maxv > 1.5:
            arr_f /= 255.0

    arr_f = np.clip(arr_f, 0.0, 1.0).astype(np.float32)
    return arr_f, return_chw


def _float01_to_uint8(image_f: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Convert float image in [0,1] to uint8 [0,255] for OpenCV operations."""
    return np.clip((image_f * 255.0).round(), 0, 255).astype(np.uint8)


@register_effect(
    category=EffectCategory.MATERIAL,
    stage=EffectStage.PRE_RENDER,
    description="Change the rendering mode of the IceSlider instance",
)
def change_render_mode(ice_slider: IceSlider, mode: str = "rgb_array") -> IceSlider:
    """Change the rendering mode of the IceSlider instance.

    The available modes are:

    - ``basic`` for solid-color tiles.
    - ``rgb_array`` for PNG tile textures.
    - ``human`` for PNG tile textures with the human render-mode tag.

    Args:
        ice_slider: The IceSlider instance to modify
        mode: Rendering mode ("basic", "rgb_array", "human")

    Returns:
        Modified IceSlider instance (for pipeline compatibility)
    """
    valid_modes = {"basic", "rgb_array", "human"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid render mode: {mode}. Must be one of {valid_modes}")

    # Only change if different to avoid unnecessary reloading
    if ice_slider.render_mode != mode:
        ice_slider.render_mode = mode
        # Clear cached rendering data that depends on render mode
        _clear_rendering_cache(ice_slider)
        ice_slider.reload_textures()
        # Fix basic mode color map if needed
        _fix_basic_mode_colormap(ice_slider)

    return ice_slider


@register_effect(
    category=EffectCategory.MATERIAL,
    stage=EffectStage.PRE_RENDER,
    description="Replace texture assets with custom ones",
)
def change_texture_assets(
    ice_slider: IceSlider, assets: dict[str, str | NDArray[np.uint8]] | None = None, asset_set: str = "default"
) -> IceSlider:
    """Replace rock, ice, player, and goal textures.

    Args:
        ice_slider: IceSlider instance to modify.
        assets: Mapping from ``"rock"``, ``"ice"``, ``"player"``, or ``"goal"`` to a file path or uint8 texture array.
        asset_set: Named asset set. Available values are ``"default"``, ``"winter"``, ``"desert"``, and ``"neon"``.

    Returns:
        The modified IceSlider instance
    """
    if assets is None:
        assets = dict(_get_predefined_asset_set(asset_set))

    # Save each instance's original textures once
    instance_id = id(ice_slider)
    if instance_id not in _TEXTURE_BACKUPS:
        _TEXTURE_BACKUPS[instance_id] = {}

    _clear_rendering_cache(ice_slider)

    # Apply new textures
    for element_name, texture_data in assets.items():
        if element_name in {"rock", "ice", "player", "goal"}:
            # Save the original texture before replacing it
            original_attr = f"{element_name}_rgb"
            if original_attr not in _TEXTURE_BACKUPS[instance_id]:
                original = getattr(ice_slider, original_attr, None)
                if original is not None:
                    _TEXTURE_BACKUPS[instance_id][original_attr] = original.copy()

            # Load new texture
            texture = _load_texture_from_path(texture_data) if isinstance(texture_data, str) else texture_data

            setattr(ice_slider, original_attr, texture)

    _clear_rendering_cache(ice_slider)
    return ice_slider


@register_effect(
    category=EffectCategory.COLOR,
    stage=EffectStage.PRE_RENDER,
    description="Apply color tinting to IceSlider texture assets",
)
def color_tint_assets(
    ice_slider: IceSlider,
    rock_tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ice_tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    player_tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    goal_tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> IceSlider:
    """Apply color tinting to texture assets for visual variation.

    Args:
        ice_slider: The IceSlider instance to modify
        rock_tint: RGB tint multipliers for rock texture (r, g, b)
        ice_tint: RGB tint multipliers for ice texture (r, g, b)
        player_tint: RGB tint multipliers for player texture (r, g, b)
        goal_tint: RGB tint multipliers for goal texture (r, g, b)

    Returns:
        Modified IceSlider instance (for pipeline compatibility)
    """
    # Skip effect for basic mode as it has texture shape limitations
    if ice_slider.render_mode == "basic":
        return ice_slider

    # Store original textures for backup
    instance_id = id(ice_slider)
    if instance_id not in _TEXTURE_BACKUPS:
        _TEXTURE_BACKUPS[instance_id] = {}

    _clear_rendering_cache(ice_slider)

    # Apply tinting to each texture
    tints = [rock_tint, ice_tint, player_tint, goal_tint]
    attrs = ["rock_rgb", "ice_rgb", "player_rgb", "goal_rgb"]

    for attr, tint in zip(attrs, tints, strict=True):
        if tint != (1.0, 1.0, 1.0):  # Only process if tint is not neutral
            # Backup original if not already backed up
            if attr not in _TEXTURE_BACKUPS[instance_id]:
                original = getattr(ice_slider, attr, None)
                if original is not None:
                    _TEXTURE_BACKUPS[instance_id][attr] = original.copy()

            # Apply tint
            original = _TEXTURE_BACKUPS[instance_id].get(attr, getattr(ice_slider, attr, None))
            if original is not None:
                tinted = _apply_color_tint(original, (tint[0], tint[1], tint[2]))
                setattr(ice_slider, attr, tinted)

    # Fix basic mode color map if needed (though we skip basic mode above)
    _fix_basic_mode_colormap(ice_slider)

    return ice_slider


# POST-RENDER VISUAL EFFECTS


@register_effect(
    category=EffectCategory.COLOR,
    stage=EffectStage.POST_RENDER,
    description="Multiply RGB channels by a configurable glow color and brightness factor",
)
def neon_glow(
    image: NDArray[np.number] | None,
    glow_intensity: float = 2.0,
    glow_color: tuple[float, float, float] = (0.0, 1.0, 1.0),
) -> NDArray[np.float32] | None:
    """Multiply image channels by a glow color and brightness factor.

    Args:
        image: Input image array.
        glow_intensity: Glow brightness multiplier.
        glow_color: RGB glow color in the range [0, 1].

    Returns:
        Brightness-scaled image, or None when the input is empty.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return None

    img_f, return_chw = _uint8_to_float01(image)
    # _uint8_to_float01 may return (None, False) for empty input
    if img_f is None:
        return None

    # Replicate grayscale input across three channels.
    if img_f.ndim == 2 or (img_f.ndim == 3 and img_f.shape[2] < 3):
        gray = img_f.squeeze() if img_f.ndim == 3 else img_f
        img_f = _gray_to_rgb_float32(gray)

    result = img_f.astype(np.float32)

    # Scale each channel by its glow-color component.
    for i in range(3):
        glow_scaled = result[:, :, i] * (1.0 + glow_intensity * glow_color[i])
        result[:, :, i] = np.clip(glow_scaled, 0.0, 1.0)

    # Add overall brightness boost
    result = np.clip(result * (1.0 + glow_intensity * 0.3), 0.0, 1.0).astype(np.float32)

    if return_chw:
        return np.transpose(result, (2, 0, 1))
    return result


@register_effect(
    category=EffectCategory.DISTORTION,
    stage=EffectStage.POST_RENDER,
    description="Transform image into retro pixel art style",
)
def retro_pixel(
    image: NDArray[np.number] | None, pixel_size: int = 6, color_levels: int = 4
) -> NDArray[np.float32] | None:
    """Transform image into retro pixel art with color quantization.

    Args:
        image: Input image array
        pixel_size: Size of pixelation blocks (higher = more pixelated)
        color_levels: Number of color levels per channel (2-16)

    Returns:
        Pixel art styled image
    """
    if image is None or getattr(image, "size", 0) == 0:
        return None

    img_f, return_chw = _uint8_to_float01(image)
    if img_f is None:
        return None
    image_hwc = img_f
    h, w = image_hwc.shape[:2]
    if h == 0 or w == 0:
        return img_f
    # Clamp parameters.
    pixel_size = max(1, min(max(h, w), pixel_size))
    color_levels = max(2, min(16, color_levels))

    # Determine downsample size
    block_w = max(1, w // pixel_size)
    block_h = max(1, h // pixel_size)

    # Convert to uint8 for cv2 operations
    img_uint = _float01_to_uint8(image_hwc)

    try:
        small = cv2.resize(img_uint, (block_w, block_h), interpolation=cv2.INTER_AREA)
    except Exception:
        small = img_uint

    # Color quantization performed in float01
    if color_levels > 1 and small.ndim == 3:
        small_f = small.astype(np.float32) / np.float32(255.0) * np.float32(color_levels - 1)
        # Quantize to [0,1] with float32 precision
        small_q = (np.round(small_f) / np.float32(max(1, (color_levels - 1)))).astype(np.float32)
        small_q = np.clip(small_q, 0.0, 1.0)
    else:
        small_q = (small.astype(np.float32) / np.float32(255.0)).astype(np.float32)

    # Convert the quantized image to uint8 for upsampling, then back to float32.
    small_uint_q = _float01_to_uint8(small_q.astype(np.float32))

    try:
        result_uint = cv2.resize(small_uint_q, (w, h), interpolation=cv2.INTER_NEAREST)
    except Exception:
        result_uint = img_uint

    result = result_uint.astype(np.float32) / 255.0

    if return_chw:
        return np.transpose(result, (2, 0, 1))
    return result


@register_effect(
    category=EffectCategory.COLOR,
    stage=EffectStage.POST_RENDER,
    description="Offset RGB channels and scale the combined result",
)
def color_shift(
    image: NDArray[np.number] | None,
    r_shift: float = 0.0,
    g_shift: float = 0.0,
    b_shift: float = 0.0,
    intensity: float = 1.0,
) -> NDArray[np.float32] | None:
    """Offset each RGB channel and scale the combined result.

    Args:
        image: Input image array
        r_shift: Red channel shift (-1.0 to 1.0)
        g_shift: Green channel shift (-1.0 to 1.0)
        b_shift: Blue channel shift (-1.0 to 1.0)
        intensity: Overall effect intensity (0.0 to 2.0)

    Returns:
        Color-shifted image
    """
    if image is None or getattr(image, "size", 0) == 0:
        return None

    img_f: NDArray[np.float32] | None
    return_chw: bool
    img_f, return_chw = _uint8_to_float01(image)
    if img_f is None:
        return None

    if img_f.ndim == 2 or (img_f.ndim == 3 and img_f.shape[2] < 3):
        gray: NDArray[np.float32] = img_f.squeeze() if img_f.ndim == 3 else img_f
        img_f = _gray_to_rgb_float32(gray)

    result: NDArray[np.float32] = img_f.astype(np.float32)

    # Map shifts from [-1, 1] to additive offsets in [-0.5, 0.5].
    shifts: list[float] = [r_shift, g_shift, b_shift]
    for i in range(3):
        shift: float = shifts[i] * intensity * 0.5
        if shift != 0:
            result[:, :, i] = np.clip(result[:, :, i] + shift, 0.0, 1.0)

    if return_chw:
        return np.transpose(result.astype(np.float32), (2, 0, 1))

    return result.astype(np.float32)


@register_effect(
    category=EffectCategory.COLOR,
    stage=EffectStage.POST_RENDER,
    description="Apply vintage film effect with sepia tones and vignetting",
)
def vintage_filter(
    image: NDArray[np.number] | None, sepia_strength: float = 0.8, vignette_strength: float = 0.3
) -> NDArray[np.float32] | None:
    """Apply vintage film effect with sepia and vignetting.

    Args:
        image: Input image array
        sepia_strength: Strength of sepia effect (0.0 to 1.0)
        vignette_strength: Strength of vignette darkening (0.0 to 1.0)

    Returns:
        Vintage-styled image
    """
    if image is None or getattr(image, "size", 0) == 0:
        return None

    img_f: NDArray[np.float32] | None
    return_chw: bool
    img_f, return_chw = _uint8_to_float01(image)
    if img_f is None:
        return None

    if img_f.ndim < 3 or img_f.shape[2] < 3:
        gray = img_f if img_f.ndim == 2 else img_f.squeeze() if img_f.ndim == 3 else img_f
        img_f = _gray_to_rgb_float32(gray)

    h: int
    w: int
    result: NDArray[np.float32] = img_f.astype(np.float32)
    h, w = result.shape[:2]

    # Apply sepia effect in float01
    if sepia_strength > 0:
        sepia_matrix: NDArray[np.float32] = np.array(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], dtype=np.float32
        )

        original_shape = result.shape
        result_flat: NDArray[np.float32] = result.reshape(-1, 3)
        sepia_result_flat = np.dot(result_flat, sepia_matrix.T)
        # sepia_result may exceed 1.0, so clip
        sepia_result = np.clip(sepia_result_flat.reshape(original_shape), 0.0, 1.0)

        result = result * (1.0 - sepia_strength) + sepia_result * sepia_strength

    # Apply vignette effect
    if vignette_strength > 0 and h > 1 and w > 1:
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        center_y, center_x = h / 2.0, w / 2.0
        distance = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        normalized_distance = distance / max_distance if max_distance > 0 else 0
        vignette_mask = 1.0 - (normalized_distance * vignette_strength)
        vignette_mask = np.clip(vignette_mask, 0.1, 1.0)
        result *= vignette_mask[..., np.newaxis]

    result = np.clip(result, 0.0, 1.0).astype(np.float32)

    if return_chw:
        return np.transpose(result, (2, 0, 1))
    return result
