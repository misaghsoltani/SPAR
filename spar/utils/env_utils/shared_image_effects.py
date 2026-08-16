from __future__ import annotations

from functools import cache, lru_cache
from io import BytesIO
from logging import getLogger
from math import cos, radians, sin
from typing import TYPE_CHECKING

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from skimage.transform import AffineTransform, ProjectiveTransform, resize, rotate, warp

from spar.data.cifar_utils import load_cifar10_images
from spar.utils.env_utils.effects_core import EffectCategory, EffectStage, register_effect

if TYPE_CHECKING:
    from logging import Logger

    from cv2.typing import MatLike
    from numpy.typing import ArrayLike, NDArray

    from spar.utils.env_utils.effects_core import FigureType, ImageArray

logger: Logger = getLogger(__name__)


# Utility functions for common operations
def _clip_image(image: ImageArray) -> ImageArray:
    """Clip image values to [0,1] range ensuring float32 dtype."""
    return np.asarray(np.clip(image, 0.0, 1.0), dtype=np.float32)


def _safe_rgb(color: str, default: str = "#808080") -> ImageArray:
    """Safely convert color name to RGB array with fallback."""
    try:
        rgb = mcolors.to_rgb(color)
    except ValueError:
        logger.warning(f"Warning: Invalid color '{color}'. Using default '{default}'.")
        rgb = mcolors.to_rgb(default)
    return np.asarray(rgb, dtype=np.float32)


def _validate_range(value: float, min_val: float, max_val: float, name: str) -> float:
    """Validate and clamp parameter to valid range."""
    _ = name
    if value < min_val or value > max_val:
        # logger.warning(f"Warning: {name} should be in [{min_val}, {max_val}]. Clamping to range.")
        return np.clip(value, min_val, max_val)
    return value


_CLUSTERED_DOT_8X8 = np.asarray(
    [
        [62, 57, 48, 36, 37, 49, 58, 63],
        [56, 47, 35, 21, 22, 38, 50, 59],
        [46, 34, 20, 10, 11, 23, 39, 51],
        [33, 19, 9, 3, 0, 4, 12, 24],
        [32, 18, 8, 2, 1, 5, 13, 25],
        [45, 31, 17, 7, 6, 14, 26, 40],
        [55, 44, 30, 16, 15, 27, 41, 52],
        [61, 54, 43, 29, 28, 42, 53, 60],
    ],
    dtype=np.float32,
) / np.float32(64.0)


def _srgb_to_linear(rgb: ImageArray) -> ImageArray:
    return np.where(rgb <= np.float32(0.04045), rgb / np.float32(12.92), ((rgb + 0.055) / 1.055) ** 2.4).astype(
        np.float32, copy=False
    )


def _linear_to_srgb(value: ImageArray) -> ImageArray:
    return np.where(
        value <= np.float32(0.0031308),
        value * np.float32(12.92),
        np.float32(1.055) * np.power(value, np.float32(1.0 / 2.4)) - np.float32(0.055),
    ).astype(np.float32, copy=False)


@cache
def _clustered_dot_thresholds(height: int, width: int) -> ImageArray:
    reps_y = int(np.ceil(height / _CLUSTERED_DOT_8X8.shape[0]))
    reps_x = int(np.ceil(width / _CLUSTERED_DOT_8X8.shape[1]))
    return np.tile(_CLUSTERED_DOT_8X8, (reps_y, reps_x))[:height, :width].astype(np.float32, copy=False)


@cache
def _paper_texture(height: int, width: int) -> ImageArray:
    y, x = np.indices((height, width), dtype=np.float32)
    fiber = np.sin(x * np.float32(0.19) + y * np.float32(0.07))
    tooth = np.sin((x + y) * np.float32(1.71)) * np.sin(x * np.float32(0.53) - y * np.float32(0.47))
    grain = np.mod(np.sin(x * np.float32(12.9898) + y * np.float32(78.233)) * np.float32(43758.5453), 1.0)
    return (np.float32(0.45) * fiber + np.float32(0.35) * tooth + np.float32(0.2) * (grain - 0.5)).astype(
        np.float32, copy=False
    )


def _resolve_range_sample(value: float | tuple[float, float] | list[float], name: str) -> float:
    """Resolve scalar or [min, max] parameter values without changing config semantics."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must be a scalar or length-2 range")
        lo, hi = value[0], value[1]
        if not (0.0 < lo <= hi) and name == "factor":
            raise ValueError("factor range must satisfy 0 < min <= max")
        if lo > hi:
            lo, hi = hi, lo
        return float(np.random.uniform(lo, hi))
    return value


@lru_cache(maxsize=32)
def _normalized_grid(height: int, width: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return cached normalized Y/X coordinate grids in [0, 1]."""
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(xs, ys)
    return y_grid, x_grid


@lru_cache(maxsize=1)
def _cached_background_images() -> tuple[NDArray[np.float32], ...]:
    """Load reusable image backgrounds once per process."""
    try:
        return tuple(np.asarray(img, dtype=np.float32) for img in load_cifar10_images())
    except Exception:
        return ()


def _prepare_effect_image(image: ArrayLike) -> tuple[NDArray[np.float32], bool]:
    """Convert an image to float32 HxWxC while tracking grayscale input."""
    img = np.asarray(image, dtype=np.float32)
    was_grayscale = img.ndim == 2
    if was_grayscale:
        img = img[:, :, None]
    if img.ndim != 3:
        raise ValueError(f"Expected image with shape (H, W) or (H, W, C), got {img.shape}")
    if np.max(img, initial=0.0) > 1.0:
        img = img.copy()
        img /= np.float32(255.0)
    return img, was_grayscale


def _match_channels(rgb: NDArray[np.float32], channels: int) -> NDArray[np.float32]:
    """Match an RGB background to the channel count of the target image."""
    if channels == 3:
        return rgb.astype(np.float32, copy=False)
    if channels == 1:
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return gray[..., None].astype(np.float32, copy=False)

    out = np.empty((*rgb.shape[:2], channels), dtype=np.float32)
    out[..., : min(3, channels)] = rgb[..., : min(3, channels)]
    if channels > 3:
        out[..., 3:] = 1.0
    return out


def _resize_rgb_background(background: NDArray[np.float32], height: int, width: int) -> NDArray[np.float32]:
    """Resize arbitrary RGB-like image data into an HxWx3 float32 background."""
    bg = np.asarray(background, dtype=np.float32)
    if bg.ndim == 2:
        bg = np.stack((bg, bg, bg), axis=-1)
    elif bg.ndim != 3:
        raise ValueError(f"Background image must have shape (H, W) or (H, W, C), got {bg.shape}")
    if bg.shape[2] == 1:
        bg = np.repeat(bg, 3, axis=2)
    elif bg.shape[2] > 3:
        bg = bg[..., :3]
    if np.max(bg, initial=0.0) > 1.0:
        bg = bg.copy()
        bg /= np.float32(255.0)
    if bg.shape[:2] != (height, width):
        bg = np.asarray(cv2.resize(bg, (width, height), interpolation=cv2.INTER_LINEAR), dtype=np.float32)
    return np.asarray(bg, dtype=np.float32)


def _matplotlib_resize_rgb_exact(
    image_rgb: NDArray[np.float32], target_height: int, target_width: int, *, interpolation: str = "bilinear"
) -> NDArray[np.float32]:
    """Resize RGB image data through Matplotlib's Agg canvas path."""
    target_height = max(1, target_height)
    target_width = max(1, target_width)
    dpi = 300
    fig = Figure(figsize=(target_width / dpi, target_height / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1), frameon=False)
    ax.axis("off")
    ax.imshow(np.clip(image_rgb, 0.0, 1.0), interpolation=interpolation)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    buffer = canvas.buffer_rgba()
    rendered = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    img = rendered.astype(np.float32) / np.float32(255.0)

    if img.shape[:2] == (target_height, target_width):
        return img.astype(np.float32, copy=False)

    out = np.zeros((target_height, target_width, 3), dtype=np.float32)
    h_copy = min(img.shape[0], target_height)
    w_copy = min(img.shape[1], target_width)
    out[:h_copy, :w_copy, :] = img[:h_copy, :w_copy, :]
    return out


def _pil_codec_roundtrip_rgb(
    image_rgb: NDArray[np.float32],
    *,
    save_format: str,
    jpeg_quality: int,
    jpeg_subsampling: int,
    png_compress_level: int,
) -> NDArray[np.float32]:
    """Round trip RGB image data through the same uint8/PIL save path as image processing."""
    image_uint8 = (np.clip(image_rgb, 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8)
    fmt = save_format.strip().lower()
    if fmt in {"jpg", "jpeg"}:
        pil_format = "JPEG"
    elif fmt == "png":
        pil_format = "PNG"
    else:
        raise ValueError("save_format must be 'png', 'jpeg', or 'jpg'")

    buffer = BytesIO()
    if pil_format == "JPEG":
        pil_image.save(
            buffer,
            format=pil_format,
            quality=int(np.clip(jpeg_quality, 1, 100)),
            subsampling=int(np.clip(jpeg_subsampling, 0, 2)),
        )
    else:
        pil_image.save(buffer, format=pil_format, compress_level=int(np.clip(png_compress_level, 0, 9)))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded_rgb = decoded.convert("RGB")
        return (np.asarray(decoded_rgb, dtype=np.float32) / np.float32(255.0)).astype(np.float32, copy=False)


def _make_zoom_background(
    height: int,
    width: int,
    channels: int,
    *,
    background_mode: str,
    background_color: str,
    background_alpha: float,
    background_blur: float,
    background_noise: float,
    image_probability: float,
    image: ImageArray | list[list[float]] | None,
) -> NDArray[np.float32]:
    """Create a color or image background for exposed zoom-out pixels."""
    mode = background_mode.lower()
    use_image = image is not None or mode == "image"
    if mode == "auto":
        use_image = bool(np.random.random() < image_probability)

    base_color = _safe_rgb(background_color, "#d8d4ca").reshape(1, 1, 3)
    if use_image:
        if image is not None:
            bg_rgb = _resize_rgb_background(np.asarray(image, dtype=np.float32), height, width)
        else:
            bank = _cached_background_images()
            if bank:
                bg_rgb = _resize_rgb_background(bank[int(np.random.randint(0, len(bank)))], height, width)
            else:
                bg_rgb = np.broadcast_to(base_color, (height, width, 3)).copy()
        if background_alpha < 1.0:
            bg_rgb = background_alpha * bg_rgb + (1.0 - background_alpha) * base_color
    else:
        bg_rgb = np.broadcast_to(base_color, (height, width, 3)).copy()

    if background_noise > 0.0:
        y_grid, x_grid = _normalized_grid(height, width)
        slope_x = np.float32(np.random.uniform(-0.08, 0.08))
        slope_y = np.float32(np.random.uniform(-0.08, 0.08))
        gradient = (x_grid - 0.5) * slope_x + (y_grid - 0.5) * slope_y
        noise = np.random.normal(0.0, background_noise, bg_rgb.shape).astype(np.float32)
        bg_rgb = bg_rgb + gradient[..., None].astype(np.float32) + noise

    if background_blur > 0.0:
        bg_rgb = np.asarray(
            cv2.GaussianBlur(bg_rgb, (0, 0), sigmaX=background_blur, sigmaY=background_blur), dtype=np.float32
        )

    return _match_channels(np.clip(bg_rgb, 0.0, 1.0).astype(np.float32, copy=False), channels)


def _project_phone_capture_corners(
    height: int, width: int, *, factor: float, angle: float, translate_x: float, translate_y: float
) -> NDArray[np.float32]:
    """Return destination corners for a mildly tilted handheld phone capture."""
    half_w = np.float32((width - 1) * 0.5 * factor)
    half_h = np.float32((height - 1) * 0.5 * factor)
    corners = np.array(
        [[-half_w, -half_h, 0.0], [half_w, -half_h, 0.0], [half_w, half_h, 0.0], [-half_w, half_h, 0.0]],
        dtype=np.float32,
    )

    zoom_out = max(0.0, 1.0 - factor)
    tilt_budget = min(9.0, 1.25 + 8.0 * zoom_out + 0.04 * abs(angle))
    yaw = np.float32(np.deg2rad(np.random.uniform(-tilt_budget, tilt_budget) + 5.0 * translate_x))
    pitch = np.float32(np.deg2rad(np.random.uniform(-tilt_budget, tilt_budget) - 5.0 * translate_y))
    roll = np.float32(np.deg2rad(angle))

    cy, sy = np.float32(np.cos(yaw)), np.float32(np.sin(yaw))
    cp, sp = np.float32(np.cos(pitch)), np.float32(np.sin(pitch))
    cr, sr = np.float32(np.cos(roll)), np.float32(np.sin(roll))
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rot_z = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rotated = corners @ (rot_z @ rot_y @ rot_x).T

    focal = np.float32(max(height, width) * 3.2)
    denom = np.maximum(focal - rotated[:, 2], np.float32(0.25 * focal))
    projected = np.empty((4, 2), dtype=np.float32)
    projected[:, 0] = focal * rotated[:, 0] / denom
    projected[:, 1] = focal * rotated[:, 1] / denom
    projected[:, 0] += np.float32((width - 1) * 0.5 + translate_x * width)
    projected[:, 1] += np.float32((height - 1) * 0.5 + translate_y * height)
    return projected


def _warp_foreground_over_background(
    img: NDArray[np.float32], background: NDArray[np.float32], dst_corners: NDArray[np.float32], *, interpolation: int
) -> NDArray[np.float32]:
    """Perspective-warp an image using an antialiased alpha matte."""
    height, width, _channels = img.shape
    src_corners = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]], dtype=np.float32
    )
    homography = cv2.getPerspectiveTransform(src_corners, dst_corners.astype(np.float32, copy=False))
    warped = cv2.warpPerspective(
        img, homography, (width, height), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    alpha = cv2.warpPerspective(
        np.ones((height, width), dtype=np.float32),
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    if min(height, width) >= 8:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.35, sigmaY=0.35)

    shadow = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(0.8, min(height, width) * 0.025))
    shadow = np.clip(shadow - alpha, 0.0, 1.0)[..., None]
    shaded_background = background * (1.0 - np.float32(0.22) * shadow)
    alpha_mask: NDArray[np.float32] = np.asarray(alpha, dtype=np.float32)[..., None]
    blended = warped * alpha_mask + shaded_background * (np.float32(1.0) - alpha_mask)
    return np.asarray(blended, dtype=np.float32)


def _luminance(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute Rec. 709 luma for RGB image data."""
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


def _smoothstep(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Cubic smoothstep for soft optical thresholds."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return (x * x * (np.float32(3.0) - np.float32(2.0) * x)).astype(np.float32, copy=False)


def _shift_channel(channel: NDArray[np.float32], shift_x: float, shift_y: float) -> NDArray[np.float32]:
    """Translate one color channel for subtle chromatic separation."""
    matrix = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    shifted: MatLike = cv2.warpAffine(
        channel,
        matrix,
        (channel.shape[1], channel.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.asarray(shifted, dtype=np.float32)


def _match_light_channels(rgb: NDArray[np.float32], channels: int) -> NDArray[np.float32]:
    """Match optical light to target channels without injecting alpha/extra-channel energy."""
    if channels <= 3:
        return _match_channels(rgb, channels)
    out = np.zeros((*rgb.shape[:2], channels), dtype=np.float32)
    out[..., :3] = rgb[..., :3]
    return out


def _near_white_optical_tint(chromatic: float, temperature: float = 0.5) -> NDArray[np.float32]:
    """Return a low-saturation optical tint that cannot repaint scene colors."""
    chroma = np.float32(np.clip(chromatic, 0.0, 1.0) * 0.18)
    temperature = float(np.clip(temperature, 0.0, 1.0))
    neutral = np.ones(3, dtype=np.float32)
    warm = np.array([1.0, 0.975, 0.94], dtype=np.float32)
    cool = np.array([0.94, 0.975, 1.0], dtype=np.float32)
    target = (np.float32(1.0 - temperature) * warm + np.float32(temperature) * cool).astype(np.float32)
    return np.clip((np.float32(1.0) - chroma) * neutral + chroma * target, 0.92, 1.0).astype(np.float32)


def _normalize_map(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize a scalar map to [0, 1] without amplifying empty maps."""
    max_value = float(np.max(values, initial=0.0))
    if max_value <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.asarray(values / np.float32(max_value), dtype=np.float32)


def _bright_source_saliency(luma: NDArray[np.float32], threshold: float) -> NDArray[np.float32]:
    """Find bright, structured light-source pixels while suppressing flat white backgrounds."""
    bright = _smoothstep(((luma - threshold) / max(1.0 - threshold, 1e-6)).astype(np.float32))
    if float(bright.sum(dtype=np.float64)) <= 1e-5 and float(np.max(luma, initial=0.0)) > 1e-5:
        adaptive_floor = np.float32(max(float(np.quantile(luma, 0.985)) * 0.82, 1e-4))
        bright = _smoothstep(
            ((luma - adaptive_floor) / max(float(np.max(luma)) - float(adaptive_floor), 1e-6)).astype(np.float32)
        )

    blur = cv2.GaussianBlur(luma, (0, 0), sigmaX=1.2, sigmaY=1.2)
    blur2 = cv2.GaussianBlur(luma * luma, (0, 0), sigmaX=1.2, sigmaY=1.2)
    local_std = np.sqrt(np.maximum(blur2 - blur * blur, 0.0)).astype(np.float32)
    grad_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    structure = _normalize_map(np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float32) + local_std)
    saliency = (bright * (np.float32(0.24) + np.float32(0.76) * structure)).astype(np.float32, copy=False)
    return _normalize_map(saliency)


def _detect_glare_sources(
    luma: NDArray[np.float32],
    saliency: NDArray[np.float32],
    *,
    max_sources: int,
    source_x: float = -1.0,
    source_y: float = -1.0,
) -> list[tuple[float, float, float]]:
    """Return normalized source coordinates and relative energies for glare/reflection synthesis."""
    height, width = saliency.shape
    y_grid, x_grid = _normalized_grid(height, width)
    sources: list[tuple[float, float, float]] = []

    explicit_source = source_x > -0.5 and source_y > -0.5
    if explicit_source:
        sx = float(np.clip(source_x, -0.25, 1.25))
        sy = float(np.clip(source_y, -0.25, 1.25))
        ix = int(np.clip(round(sx * (width - 1)), 0, width - 1))
        iy = int(np.clip(round(sy * (height - 1)), 0, height - 1))
        sources.append((sx, sy, max(saliency[iy, ix], luma[iy, ix], 0.35)))

    max_saliency = float(np.max(saliency, initial=0.0))
    if max_saliency > 1e-6:
        threshold = max(0.18, 0.55 * max_saliency)
        mask = (saliency >= threshold).astype(np.uint8)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components: list[tuple[float, float, float, int]] = []
        max_area = max(4, int(0.28 * height * width))
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area <= 0 or area > max_area:
                continue
            comp_mask = labels == idx
            weights = saliency * comp_mask
            energy = float(weights.sum(dtype=np.float64))
            if energy <= 1e-6:
                continue
            cx = float((weights * x_grid).sum(dtype=np.float64) / energy)
            cy = float((weights * y_grid).sum(dtype=np.float64) / energy)
            components.append((cx, cy, energy, area))

        def _component_energy_key(item: tuple[float, float, float, int]) -> float:
            return item[2] / max(item[3], 1)

        components.sort(key=_component_energy_key, reverse=True)
        for cx, cy, energy, _area in components:
            if len(sources) >= max_sources:
                break
            if all((cx - sx) ** 2 + (cy - sy) ** 2 > 0.0025 for sx, sy, _e in sources):
                sources.append((cx, cy, energy))

    if not sources:
        if np.random.random() < 0.72:
            sources.append((float(np.random.uniform(0.03, 0.97)), float(np.random.uniform(0.03, 0.97)), 0.45))
        else:
            edge = int(np.random.randint(0, 4))
            if edge == 0:
                sources.append((float(np.random.uniform(-0.12, 1.12)), -0.10, 0.45))
            elif edge == 1:
                sources.append((1.10, float(np.random.uniform(-0.12, 1.12)), 0.45))
            elif edge == 2:
                sources.append((float(np.random.uniform(-0.12, 1.12)), 1.10, 0.45))
            else:
                sources.append((-0.10, float(np.random.uniform(-0.12, 1.12)), 0.45))

    max_energy = max(energy for _x, _y, energy in sources)
    return [(x, y, float(np.clip(energy / max(max_energy, 1e-6), 0.15, 1.0))) for x, y, energy in sources[:max_sources]]


def _elliptical_gaussian(
    x_grid: NDArray[np.float32],
    y_grid: NDArray[np.float32],
    *,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    angle: float,
    ring: float = 0.0,
) -> NDArray[np.float32]:
    """Render a rotated elliptical disk or aperture ring in normalized image coordinates."""
    theta = np.float32(np.deg2rad(angle))
    c, s = np.float32(np.cos(theta)), np.float32(np.sin(theta))
    local_x = x_grid - np.float32(center_x)
    local_y = y_grid - np.float32(center_y)
    u = local_x * c + local_y * s
    v = -local_x * s + local_y * c
    ell = ((u / np.float32(max(sigma_x, 1e-4))) ** 2 + (v / np.float32(max(sigma_y, 1e-4))) ** 2).astype(np.float32)
    disk = np.exp(-0.5 * ell).astype(np.float32)
    if ring <= 0.0:
        return disk
    radius = np.sqrt(np.maximum(ell, 0.0)).astype(np.float32)
    ring_map = np.exp(-0.5 * ((radius - 1.0) / np.float32(max(ring, 1e-3))) ** 2).astype(np.float32)
    return np.clip((1.0 - ring) * disk + ring * ring_map, 0.0, 1.0).astype(np.float32)


def _screen_blend(base: NDArray[np.float32], light: NDArray[np.float32]) -> NDArray[np.float32]:
    """Photographic screen blend for additive optical light."""
    light_clipped: NDArray[np.float32] = np.clip(light, 0.0, 1.0).astype(np.float32, copy=False)
    blended = np.float32(1.0) - (np.float32(1.0) - base) * (np.float32(1.0) - light_clipped)
    return blended.astype(np.float32, copy=False)


# ================================================================================================
# SHARED IMAGE EFFECTS
# ================================================================================================


# Background Effects
@register_effect(
    category=EffectCategory.BACKGROUND,
    stage=EffectStage.PRE_RENDER,
    description="Change figure background color to simulate different lighting conditions",
)
def background_color_shift(fig: FigureType, color: str = "#cccccc") -> FigureType:
    """Change the figure's background color.

    Simulates different lighting conditions or background settings that can affect
    contrast with objects and visual perception.

    Args:
        fig: The matplotlib figure to modify
        color: Hex color string for background

    Returns:
        Modified figure (for pipeline compatibility)
    """
    fig.patch.set_facecolor(color if mcolors.is_color_like(color) else "#cccccc")
    return fig


@register_effect(
    category=EffectCategory.BACKGROUND,
    stage=EffectStage.PRE_RENDER,
    description="Apply random background color for unpredictable lighting simulation",
    requires_rng=True,
)
def background_randomization(fig: FigureType) -> FigureType:
    """Randomize the figure's background color using a random RGB value.

    Simulates unpredictable lighting conditions and backgrounds by applying
    a completely random color to the figure background.

    Args:
        fig: The matplotlib figure to modify

    Returns:
        Modified figure (for pipeline compatibility)
    """
    fig.patch.set_facecolor(mcolors.to_hex(tuple(np.random.rand(3))))
    return fig


# Color Effects
@register_effect(
    category=EffectCategory.COLOR,
    stage=EffectStage.POST_RENDER,
    description="Adjust hue, saturation, and brightness of the image",
)
def hsv_shift(image: ImageArray, hue_shift: float = 0.0, sat_scale: float = 1.0, val_scale: float = 1.0) -> ImageArray:
    """Adjust the hue, saturation, and brightness of the image.

    Simulates changes in lighting conditions, image sensor calibration issues,
    or color balance problems that affect color perception and recognition.

    Args:
        image: Input image array with pixel values in [0,1]
        hue_shift: Hue shift value in range [-1.0, 1.0]
        sat_scale: Saturation scale factor in range [0, 2]
        val_scale: Brightness scale factor in range [0, 2]

    Returns:
        Modified image with shifted HSV values
    """
    hue_shift = _validate_range(hue_shift, -1.0, 1.0, "hue_shift")
    sat_scale = _validate_range(sat_scale, 0.0, 2.0, "sat_scale")
    val_scale = _validate_range(val_scale, 0.0, 2.0, "val_scale")

    image_copy = np.asarray(image, dtype=np.float32)
    if float(np.max(image_copy)) > 1.0:
        image_copy /= 255.0

    hsv = mcolors.rgb_to_hsv(image_copy)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 1)

    result = mcolors.hsv_to_rgb(hsv)
    if float(np.max(image)) > 1.0:
        result *= 255.0

    return result.astype(np.float32, copy=False)


@register_effect(
    category=EffectCategory.COLOR, stage=EffectStage.POST_RENDER, description="Modify image contrast and brightness"
)
def contrast_brightness(image: ImageArray, contrast: float = 1.0, brightness: float = 0.0) -> ImageArray:
    """Modify the image's contrast and brightness.

    Simulates different lighting conditions, camera exposure settings,
    and image processing variations.

    Args:
        image: Input image array with pixel values in [0,1]
        contrast: Contrast multiplier in range [0.1, 3.0]
        brightness: Brightness offset in range [-0.5, 0.5]

    Returns:
        Modified image with adjusted contrast and brightness
    """
    contrast = _validate_range(contrast, 0.1, 3.0, "contrast")
    brightness = _validate_range(brightness, -0.5, 0.5, "brightness")
    return _clip_image(contrast * image + brightness)


@register_effect(
    category=EffectCategory.COLOR,
    stage=EffectStage.POST_RENDER,
    description="Adjust image exposure using gamma correction",
)
def exposure_variation(image: ImageArray, gamma: float = 1.0) -> ImageArray:
    """Adjust image exposure using gamma correction.

    Simulates different camera exposure settings or lighting conditions
    that can affect how colors appear in an image.

    Args:
        image: Input image array with pixel values in [0,1]
        gamma: Gamma correction value in range [0.1, 3.0]

    Returns:
        Modified image with adjusted exposure
    """
    gamma = _validate_range(gamma, 0.1, 3.0, "gamma")
    if gamma <= 0:
        gamma = 1.0
    return _clip_image(image ** (1.0 / gamma))


@register_effect(
    category=EffectCategory.COLOR,
    stage=EffectStage.POST_RENDER,
    description="Adjust red and blue channels to simulate color temperature changes",
)
def color_temperature(image: ImageArray, shift: float = 0.1) -> ImageArray:
    """Adjust the red and blue channels to simulate a change in color temperature.

    Simulates different lighting color temperatures, affecting how colors appear.
    Color temperature variations are common in different lighting environments.

    Args:
        image: Input image array with pixel values in [0,1]
        shift: Temperature shift value in range [-0.3, 0.3]

    Returns:
        Modified image with adjusted color temperature
    """
    shift = _validate_range(shift, -0.5, 0.5, "shift")
    out = image.copy()
    out[..., 0] = np.clip(out[..., 0] + shift, 0, 1)  # Red channel
    out[..., 2] = np.clip(out[..., 2] - shift, 0, 1)  # Blue channel
    return out


@register_effect(
    category=EffectCategory.COLOR,
    stage=EffectStage.POST_RENDER,
    description="Convert an image to a realistic black-and-white printer halftone rendering",
)
def black_and_white_print(
    image: ImageArray,
    contrast: float = 1.15,
    edge_strength: float = 0.12,
    levels: int = 12,
    dot_gain: float = 0.18,
    toner_density: float = 0.96,
    paper_gray: float = 0.94,
    yule_nielsen_n: float = 1.8,
    paper_texture: float = 0.018,
) -> ImageArray:
    """Render an image as a monochrome halftone print with paper, toner, and dot-gain response."""
    contrast = _validate_range(contrast, 0.2, 3.0, "contrast")
    edge_strength = _validate_range(edge_strength, 0.0, 0.6, "edge_strength")
    dot_gain = _validate_range(dot_gain, 0.0, 0.45, "dot_gain")
    toner_density = _validate_range(toner_density, 0.5, 0.995, "toner_density")
    paper_gray = _validate_range(paper_gray, 0.75, 1.0, "paper_gray")
    yule_nielsen_n = _validate_range(yule_nielsen_n, 1.0, 4.0, "yule_nielsen_n")
    paper_texture = _validate_range(paper_texture, 0.0, 0.08, "paper_texture")
    num_levels = int(np.clip(levels, 2, 32))
    oversample = 4

    src = np.asarray(image, dtype=np.float32)
    input_uses_255 = bool(np.nanmax(src) > np.float32(1.5))
    norm = np.nan_to_num(src * np.float32(1.0 / 255.0) if input_uses_255 else src, nan=1.0, posinf=1.0, neginf=0.0)
    norm = np.clip(norm, 0.0, 1.0)
    if norm.ndim == 2:
        rgb = np.repeat(norm[..., np.newaxis], 3, axis=-1)
    elif norm.ndim == 3 and norm.shape[-1] >= 3:
        rgb = norm[..., :3]
    elif norm.ndim == 3 and norm.shape[-1] == 1:
        rgb = np.repeat(norm, 3, axis=-1)
    else:
        raise ValueError(f"black_and_white_print expects an HWC or HW image, got shape {image.shape}")

    linear = _srgb_to_linear(rgb)
    gray: NDArray[np.float32] = np.asarray(
        linear[..., 0] * np.float32(0.2126) + linear[..., 1] * np.float32(0.7152) + linear[..., 2] * np.float32(0.0722),
        dtype=np.float32,
    )
    gray = _linear_to_srgb(gray)
    gray = np.clip((gray - np.float32(0.5)) * np.float32(contrast) + np.float32(0.5), 0.0, 1.0)

    if edge_strength > 0.0:
        edges: NDArray[np.float32] = np.asarray(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), dtype=np.float32)
        edge_max = float(np.max(edges))
        if edge_max > 0.0:
            edges *= np.float32(1.0 / edge_max)
            gray = np.asarray(np.clip(gray - np.float32(edge_strength) * edges, 0.0, 1.0), dtype=np.float32)

    coverage = np.clip(np.float32(1.0) - gray, 0.0, 1.0)
    coverage = np.round(coverage * np.float32(num_levels - 1)) * np.float32(1.0 / (num_levels - 1))

    height, width = coverage.shape
    hi_size = (width * oversample, height * oversample)
    coverage_hi: NDArray[np.float32] = np.asarray(
        cv2.resize(coverage, hi_size, interpolation=cv2.INTER_CUBIC), dtype=np.float32
    )
    coverage_hi = np.clip(coverage_hi, 0.0, 1.0)
    threshold = _clustered_dot_thresholds(coverage_hi.shape[0], coverage_hi.shape[1])
    toner_mask = (coverage_hi > threshold).astype(np.float32, copy=False)

    if dot_gain > 0.0:
        sigma = max(0.35, dot_gain * oversample * 1.6)
        spread: NDArray[np.float32] = np.asarray(
            cv2.GaussianBlur(toner_mask, (0, 0), sigmaX=sigma, sigmaY=sigma), dtype=np.float32
        )
        effective_coverage = np.clip(toner_mask + np.float32(dot_gain) * spread, 0.0, 1.0)
    else:
        effective_coverage = toner_mask

    paper = np.clip(
        np.float32(paper_gray) + np.float32(paper_texture) * _paper_texture(coverage_hi.shape[0], coverage_hi.shape[1]),
        0.0,
        1.0,
    )
    toner_reflectance = np.float32(1.0 - toner_density)
    inv_n = np.float32(1.0 / yule_nielsen_n)
    reflectance_hi = np.power(
        (np.float32(1.0) - effective_coverage) * np.power(paper, inv_n)
        + effective_coverage * np.power(toner_reflectance, inv_n),
        np.float32(yule_nielsen_n),
    )
    reflectance: NDArray[np.float32] = np.asarray(
        cv2.resize(reflectance_hi, (width, height), interpolation=cv2.INTER_AREA), dtype=np.float32
    )
    reflectance = np.clip(reflectance, 0.0, 1.0)

    out: ImageArray = np.repeat(reflectance[..., np.newaxis], 3, axis=-1).astype(np.float32, copy=False)
    if input_uses_255:
        out *= np.float32(255.0)
    return out.astype(image.dtype, copy=False)


# Lighting Effects
@register_effect(
    category=EffectCategory.LIGHTING,
    stage=EffectStage.POST_RENDER,
    description="Apply ambient light effect via semi-transparent color overlay",
)
def ambient_light(image: ImageArray, color: str = "#ffd1a4", alpha: float = 0.2) -> ImageArray:
    """Apply an ambient light effect via a semi-transparent color overlay.

    Simulates different ambient lighting conditions that can affect
    the color appearance and recognition of objects.

    Args:
        image: Input image array with pixel values in [0,1]
        color: Overlay color as hex string
        alpha: Overlay opacity in range [0, 1]

    Returns:
        Modified image with ambient lighting effect
    """
    alpha = _validate_range(alpha, 0.0, 1.0, "alpha")
    overlay = _safe_rgb(color, "#ffd1a4")
    return _clip_image(np.asarray((1.0 - alpha) * image + alpha * overlay, dtype=np.float32))


@register_effect(
    category=EffectCategory.LIGHTING,
    stage=EffectStage.POST_RENDER,
    description="Adjust image exposure with gamma correction, simulate a "
    "directional light at any 0-360° angle, and optionally apply vignetting.",
)
def directional_light(
    image: ImageArray,
    gamma: float = 1.0,
    angle: float = 0.0,
    intensity: float = 1.0,
    ambient: float = 0.5,
    apply_vignette: bool = False,
    vignette_strength: float = 0.5,
) -> ImageArray:
    """Simulate complex lighting changes on an image.

    Applies gamma correction to adjust overall exposure, then simulates a
    directional light source coming from the specified angle, combining
    an ambient term with a directional intensity. Optionally adds a
    radial vignetting fall-off.

    Args:
        image: Input image array with pixel values in [0, 1], shape (H, W, C).
        gamma: Gamma correction in [0.1, 3.0].
        angle: Light direction in degrees, 0° = right ->, 90° = up, etc.
        intensity: Strength of the directional light (> 0).
        ambient: Base ambient light term in [0, 1].
        apply_vignette: Whether to apply radial vignetting.
        vignette_strength: Vignette fall-off factor in [0, 1].

    Returns:
        ImageArray with simulated exposure, directional lighting, and
        optional vignetting, clipped to [0, 1].
    """
    # Validate parameters
    gamma = _validate_range(gamma, 0.1, 3.0, "gamma")
    angle %= 360.0
    intensity = max(0.0, intensity)
    ambient = _validate_range(ambient, 0.0, 1.0, "ambient")
    vignette_strength = _validate_range(vignette_strength, 0.0, 1.0, "vignette_strength")

    # Gamma correction (exposure)
    img = image ** (1.0 / gamma)

    # Build directional light mask
    h, w = img.shape[:2]
    # normalized coordinates in [-1, 1]
    ys = (np.linspace(0, h - 1, h) / (h - 1)) * 2 - 1
    xs = (np.linspace(0, w - 1, w) / (w - 1)) * 2 - 1
    xv, yv = np.meshgrid(xs, ys)
    # light direction unit vector
    theta = radians(angle)
    dx, dy = cos(theta), sin(theta)
    # dot product gives how "lit" each pixel is: map from [-1,1]->[0,1]
    mask = (xv * dx + yv * dy) * 0.5 + 0.5
    mask = np.clip(mask, 0.0, 1.0)[..., np.newaxis]  # shape (H, W, 1)

    # combine ambient + directional
    light_term = ambient + intensity * mask
    img *= light_term

    # Optional radial vignetting
    if apply_vignette:
        # radial distance from center
        rv = np.sqrt(xv**2 + yv**2)
        # smooth fall-off: 1 at center -> 1 - strength at edges
        vignette = 1.0 - vignette_strength * np.clip(rv, 0.0, 1.0)
        img *= vignette[..., np.newaxis]

    return _clip_image(np.asarray(img, dtype=np.float32))


@register_effect(
    category=EffectCategory.LIGHTING,
    stage=EffectStage.POST_RENDER,
    description=(
        "Apply physically inspired phone-camera glare: thresholded bloom, "
        "veiling flare, lens ghosts, and source-aligned streaks"
    ),
    performance_level=2,
    requires_rng=True,
)
def lens_glare(
    image: ArrayLike,
    *,
    intensity: float = 0.35,
    threshold: float = 0.78,
    bloom_strength: float = 0.45,
    bloom_radius: float = 0.055,
    halo_strength: float = 0.28,
    halo_radius: float = 0.32,
    ghost_strength: float = 0.16,
    num_ghosts: int = 3,
    streak_strength: float = 0.18,
    streak_length: float = 0.85,
    streak_width: float = 0.018,
    angle: float = 0.0,
    source_x: float = -1.0,
    source_y: float = -1.0,
    chromatic: float = 0.18,
    glare_size: float = 1.0,
) -> ImageArray:
    """Add bloom, halo, lens-ghost, and streak components around bright regions.

    Args:
        image: Input image in a layout accepted by :func:`_prepare_effect_image`.
        intensity: Multiplier applied to all glare components.
        threshold: Luminance threshold used to locate bright sources.
        bloom_strength: Multiplier for Gaussian bloom.
        bloom_radius: Bloom radius as a fraction of the shorter image side.
        halo_strength: Multiplier for the radial halo.
        halo_radius: Radius of the radial halo in normalized coordinates.
        ghost_strength: Multiplier for lens-coating ghosts.
        num_ghosts: Number of ghost elements to place.
        streak_strength: Multiplier for directional streaks.
        streak_length: Streak length in normalized coordinates.
        streak_width: Streak width in normalized coordinates.
        angle: Streak angle in degrees.
        source_x: Source x-coordinate, or a negative value for automatic selection.
        source_y: Source y-coordinate, or a negative value for automatic selection.
        chromatic: Color-separation strength for optical components.
        glare_size: Shared scale factor for bloom, halo, ghosts, and streaks.

    Returns:
        Image with the configured glare components applied.
    """
    img, was_grayscale = _prepare_effect_image(image)
    h, w, channels = img.shape
    if h == 0 or w == 0:
        return np.asarray(image, dtype=np.float32)

    intensity = _validate_range(intensity, 0.0, 2.0, "intensity")
    if intensity <= 0.0:
        return img[:, :, 0] if was_grayscale else img.copy()
    threshold = _validate_range(threshold, 0.05, 0.99, "threshold")
    bloom_strength = _validate_range(bloom_strength, 0.0, 2.0, "bloom_strength")
    bloom_radius = _validate_range(bloom_radius, 0.002, 0.3, "bloom_radius")
    halo_strength = _validate_range(halo_strength, 0.0, 2.0, "halo_strength")
    halo_radius = _validate_range(halo_radius, 0.02, 1.5, "halo_radius")
    ghost_strength = _validate_range(ghost_strength, 0.0, 1.0, "ghost_strength")
    num_ghosts = int(_validate_range(num_ghosts, 0, 8, "num_ghosts"))
    streak_strength = _validate_range(streak_strength, 0.0, 1.0, "streak_strength")
    streak_length = _validate_range(streak_length, 0.05, 2.0, "streak_length")
    streak_width = _validate_range(streak_width, 0.002, 0.08, "streak_width")
    chromatic = _validate_range(chromatic, 0.0, 1.0, "chromatic")
    glare_size = _validate_range(glare_size, 0.1, 5.0, "glare_size")

    linear_img = _srgb_to_linear(img)
    rgb_for_luma: NDArray[np.float32] = linear_img[..., :3] if channels >= 3 else np.repeat(linear_img, 3, axis=2)
    luma = _luminance(rgb_for_luma)
    saliency = _bright_source_saliency(luma, threshold)
    y_grid, x_grid = _normalized_grid(h, w)

    sources = _detect_glare_sources(
        luma, saliency, max_sources=max(1, min(4, 1 + num_ghosts // 2)), source_x=source_x, source_y=source_y
    )
    source_x, source_y, _source_energy = sources[0]

    tint = _near_white_optical_tint(chromatic, temperature=0.45)

    glare_rgb = np.zeros((h, w, 3), dtype=np.float32)
    veil = np.zeros((h, w), dtype=np.float32)

    saliency_sum = float(saliency.sum(dtype=np.float64))
    if bloom_strength > 0.0 and saliency_sum > 1e-5:
        highlight_energy = (np.clip(luma, 0.0, 1.0) * saliency).astype(np.float32)
        highlights = np.repeat(highlight_energy[..., None], 3, axis=2)
        sigma_base = max(0.45, float(min(h, w)) * bloom_radius * glare_size)
        for scale, weight in ((0.42, 0.58), (1.25, 0.30), (3.4, 0.12), (8.0, 0.035)):
            sigma = sigma_base * scale
            bloom_rgb = np.asarray(cv2.GaussianBlur(highlights, (0, 0), sigmaX=sigma, sigmaY=sigma), dtype=np.float32)
            glare_rgb += bloom_rgb * np.float32(intensity * bloom_strength * weight)
            if scale >= 3.0:
                veil += _normalize_map(_luminance(bloom_rgb)) * np.float32(weight)

    for source_idx, (sx, sy, energy) in enumerate(sources):
        dx = x_grid - np.float32(sx)
        dy = y_grid - np.float32(sy)
        radius2 = dx * dx + dy * dy
        source_weight = np.float32(energy / (1.0 + 0.35 * source_idx))

        if halo_strength > 0.0:
            scaled_halo_radius = halo_radius * glare_size
            halo_radius2 = np.float32(scaled_halo_radius * scaled_halo_radius)
            inverse_square = halo_radius2 / (radius2 + halo_radius2 + np.float32(1e-6))
            broad_haze = np.exp(-radius2 / np.float32(2.0 * (scaled_halo_radius * 1.8) ** 2)).astype(np.float32)
            annular = np.exp(
                -0.5
                * (
                    (np.sqrt(np.maximum(radius2, 0.0)) - np.float32(scaled_halo_radius * 0.78))
                    / np.float32(0.055 * glare_size)
                )
                ** 2
            ).astype(np.float32)
            halo = np.clip(0.66 * inverse_square + 0.22 * broad_haze + 0.12 * annular, 0.0, 1.0).astype(np.float32)
            glare_rgb += (
                halo[..., None] * tint.reshape(1, 1, 3) * np.float32(0.82 * intensity * halo_strength) * source_weight
            )
            veil += halo * np.float32(0.08 * intensity * halo_strength) * source_weight

        if streak_strength > 0.0:
            diffraction = np.zeros((h, w), dtype=np.float32)
            dust = cv2.GaussianBlur(
                np.random.normal(0.8, 0.22, (h, w)).astype(np.float32), (0, 0), sigmaX=max(0.75, min(h, w) * 0.018)
            )
            dust = np.clip(dust, 0.25, 1.35).astype(np.float32)
            for axis_angle, axis_weight, width_scale, length_scale in (
                (angle, 1.0, 0.85, 1.15),
                (angle + 90.0, 0.45, 1.40, 0.72),
                (angle + 45.0, 0.26, 2.10, 0.48),
                (angle - 45.0, 0.20, 2.50, 0.42),
                (angle + 22.5, 0.12, 3.10, 0.34),
                (angle - 22.5, 0.10, 3.30, 0.30),
            ):
                theta = np.float32(np.deg2rad(axis_angle))
                c, s = np.float32(np.cos(theta)), np.float32(np.sin(theta))
                parallel = dx * c + dy * s
                perp = -dx * s + dy * c
                width_term = np.exp(
                    -((np.abs(perp) / np.float32(streak_width * glare_size * width_scale)) ** np.float32(1.12))
                )
                length_term = np.exp(
                    -((np.abs(parallel) / np.float32(streak_length * glare_size * length_scale)) ** np.float32(0.68))
                )
                diffraction += np.float32(axis_weight) * width_term.astype(np.float32) * length_term.astype(np.float32)
            diffraction *= dust
            glare_rgb += (
                diffraction[..., None]
                * tint.reshape(1, 1, 3)
                * np.float32(1.12 * intensity * streak_strength)
                * source_weight
            )

    if ghost_strength > 0.0 and num_ghosts > 0:
        center_x = np.float32(0.5)
        center_y = np.float32(0.5)
        ghost_tints = (
            np.array([1.0, 0.985, 0.965], dtype=np.float32),
            np.array([0.965, 0.99, 1.0], dtype=np.float32),
            np.array([1.0, 0.995, 0.975], dtype=np.float32),
            np.array([0.985, 0.975, 1.0], dtype=np.float32),
        )
        for source_idx, (sx, sy, energy) in enumerate(sources[:2]):
            ghost_count = num_ghosts if source_idx == 0 else max(1, num_ghosts // 2)
            for idx, t in enumerate(np.linspace(0.16, 1.28, ghost_count, dtype=np.float32)):
                jitter = np.float32(np.random.uniform(-0.020, 0.020))
                gx = center_x + (center_x - np.float32(sx)) * (t + jitter)
                gy = center_y + (center_y - np.float32(sy)) * (t - jitter * np.float32(0.7))
                major = np.float32((0.026 + 0.017 * idx + 0.022 * abs(float(t - 0.5))) * glare_size)
                minor = np.float32(major * np.random.uniform(0.42, 0.84))
                ring_mix = float(np.clip(0.28 + 0.10 * idx + np.random.uniform(-0.08, 0.08), 0.12, 0.72))
                ghost = _elliptical_gaussian(
                    x_grid,
                    y_grid,
                    center_x=float(gx),
                    center_y=float(gy),
                    sigma_x=float(major),
                    sigma_y=float(minor),
                    angle=float(angle + 21.0 * idx + np.random.uniform(-14.0, 14.0)),
                    ring=ring_mix,
                )
                if -0.35 <= float(gx) <= 1.35 and -0.35 <= float(gy) <= 1.35:
                    decay = np.float32(energy / (1.0 + 0.58 * idx + 0.35 * source_idx))
                    ghost_tint = ghost_tints[(idx + source_idx) % len(ghost_tints)]
                    glare_rgb += (
                        ghost[..., None]
                        * ghost_tint.reshape(1, 1, 3)
                        * np.float32(1.95 * intensity * ghost_strength)
                        * decay
                    )

    glare_rgb = np.clip(glare_rgb, 0.0, 1.0)
    if chromatic > 0.0:
        shift_scale = np.float32(min(h, w) * 0.006 * chromatic)
        shift_x = float((np.float32(source_x) - 0.5) * shift_scale)
        shift_y = float((np.float32(source_y) - 0.5) * shift_scale)
        glare_rgb[..., 0] = _shift_channel(glare_rgb[..., 0], shift_x, shift_y)
        glare_rgb[..., 2] = _shift_channel(glare_rgb[..., 2], -shift_x, -shift_y)

    veil = np.clip(cv2.GaussianBlur(veil, (0, 0), sigmaX=max(0.8, min(h, w) * 0.035)), 0.0, 1.0)
    glare_rgb += veil[..., None] * tint.reshape(1, 1, 3) * np.float32(0.06)
    glare = _match_light_channels(glare_rgb, channels)
    glare = np.clip(glare, 0.0, 1.0)
    out_linear = _screen_blend(linear_img, glare)
    if channels <= 3:
        out = _linear_to_srgb(out_linear)
    else:
        out = img.copy()
        out[..., :3] = _linear_to_srgb(out_linear[..., :3])
    out = _clip_image(out)
    if was_grayscale:
        return out[:, :, 0]
    return out


@register_effect(
    category=EffectCategory.LIGHTING,
    stage=EffectStage.POST_RENDER,
    description=(
        "Add physically inspired multi-scale specular reflections with varied "
        "sizes, coating tints, roughness, and chromatic separation"
    ),
    performance_level=2,
    requires_rng=True,
)
def reflections(
    image: ArrayLike,
    *,
    intensity: float = 0.35,
    num_reflections: int = 6,
    size_min: float = 0.025,
    size_max: float = 0.18,
    sharpness: float = 0.65,
    elongation: float = 1.8,
    angle: float = 0.0,
    color_temperature: float = 0.55,
    surface_gloss: float = 0.7,
    chromatic: float = 0.12,
    source_x: float = -1.0,
    source_y: float = -1.0,
    source_threshold: float = 0.76,
    reflection_x: float = -1.0,
    reflection_y: float = -1.0,
    position_spread: float = 0.18,
) -> ImageArray:
    """Render glass or lens reflections around bright image regions.

    Args:
        image: Input image in a layout accepted by :func:`_prepare_effect_image`.
        intensity: Multiplier applied to all reflection lobes.
        num_reflections: Number of reflection lobes to render.
        size_min: Minimum lobe radius relative to the shorter image side.
        size_max: Maximum lobe radius relative to the shorter image side.
        sharpness: Edge sharpness of each reflection lobe.
        elongation: Ratio between the lobe's major and minor axes.
        angle: Base lobe angle in degrees.
        color_temperature: Warm-to-cool balance of the coating tint.
        surface_gloss: Balance between diffuse and specular components.
        chromatic: Color-separation strength at lobe boundaries.
        source_x: Source x-coordinate, or a negative value for automatic selection.
        source_y: Source y-coordinate, or a negative value for automatic selection.
        source_threshold: Luminance threshold used to find bright sources.
        reflection_x: Reflection-field x-coordinate, or a negative value to derive it.
        reflection_y: Reflection-field y-coordinate, or a negative value to derive it.
        position_spread: Maximum displacement of lobes around the reflection field.

    Returns:
        Image with the configured reflection lobes applied.
    """
    img, was_grayscale = _prepare_effect_image(image)
    h, w, channels = img.shape
    if h == 0 or w == 0:
        return np.asarray(image, dtype=np.float32)

    intensity = _validate_range(intensity, 0.0, 2.0, "intensity")
    if intensity <= 0.0:
        return img[:, :, 0] if was_grayscale else img.copy()
    num_reflections = int(_validate_range(num_reflections, 0, 16, "num_reflections"))
    if num_reflections <= 0:
        return img[:, :, 0] if was_grayscale else img.copy()
    size_min = _validate_range(size_min, 0.002, 0.8, "size_min")
    size_max = _validate_range(size_max, 0.002, 0.9, "size_max")
    if size_min > size_max:
        size_min, size_max = size_max, size_min
    sharpness = _validate_range(sharpness, 0.0, 1.0, "sharpness")
    elongation = _validate_range(elongation, 0.25, 8.0, "elongation")
    color_temperature = _validate_range(color_temperature, 0.0, 1.0, "color_temperature")
    surface_gloss = _validate_range(surface_gloss, 0.0, 1.0, "surface_gloss")
    chromatic = _validate_range(chromatic, 0.0, 1.0, "chromatic")
    source_threshold = _validate_range(source_threshold, 0.05, 0.99, "source_threshold")
    position_spread = _validate_range(position_spread, 0.0, 1.2, "position_spread")

    linear_img = _srgb_to_linear(img)
    rgb_for_luma = linear_img[..., :3] if channels >= 3 else np.repeat(linear_img, 3, axis=2)
    luma = _luminance(rgb_for_luma)
    saliency = _bright_source_saliency(luma, source_threshold)
    sources = _detect_glare_sources(
        luma, saliency, max_sources=max(1, min(4, num_reflections)), source_x=source_x, source_y=source_y
    )
    y_grid, x_grid = _normalized_grid(h, w)

    base_tint: NDArray[np.float32] = _near_white_optical_tint(chromatic, temperature=color_temperature)
    coating_tints: tuple[NDArray[np.float32], ...] = (
        np.array([1.0, 0.985, 0.965], dtype=np.float32),
        np.array([0.965, 0.99, 1.0], dtype=np.float32),
        np.array([1.0, 0.975, 0.995], dtype=np.float32),
        np.array([0.98, 1.0, 0.975], dtype=np.float32),
    )

    blur: MatLike = cv2.GaussianBlur(luma, (0, 0), sigmaX=1.2, sigmaY=1.2)
    blur2: MatLike = cv2.GaussianBlur(luma * luma, (0, 0), sigmaX=1.2, sigmaY=1.2)
    local_structure: NDArray[np.float32] = _normalize_map(
        np.sqrt(np.maximum(blur2 - blur * blur, 0.0)).astype(np.float32)
    )
    radial: NDArray[np.float32] = np.sqrt((x_grid - 0.5) ** 2 + (y_grid - 0.5) ** 2).astype(np.float32)
    fresnel: NDArray[np.float32] = np.clip(0.22 + 1.55 * radial * radial, 0.0, 1.0).astype(np.float32)
    surface_mask: NDArray[np.float32] = np.clip(
        (np.float32(0.22) + np.float32(0.78) * local_structure)
        * (np.float32(0.35) + np.float32(0.65) * surface_gloss)
        * (np.float32(0.45) + np.float32(0.55) * fresnel),
        0.0,
        1.0,
    ).astype(np.float32)

    reflection_rgb: NDArray[np.float32] = np.zeros((h, w, 3), dtype=np.float32)
    sizes: NDArray[np.float32]
    if num_reflections == 1:
        sizes = np.array([size_max], dtype=np.float32)
    else:
        sizes = np.geomspace(max(size_min, 1e-4), max(size_max, size_min + 1e-4), num_reflections).astype(np.float32)

    explicit_reflection = reflection_x > -0.5 and reflection_y > -0.5
    anchor_x = np.float32(np.clip(reflection_x, -0.25, 1.25))
    anchor_y = np.float32(np.clip(reflection_y, -0.25, 1.25))

    for idx in range(num_reflections):
        sx, sy, energy = sources[idx % len(sources)]
        if explicit_reflection:
            scatter_angle = np.float32(np.deg2rad(angle + idx * 137.5 + np.random.uniform(-28.0, 28.0)))
            scatter_radius = np.float32(position_spread * np.sqrt((idx + 0.35) / max(num_reflections, 1)))
            scatter_radius *= np.float32(np.random.uniform(0.25, 1.0))
            reflected_x = anchor_x + np.float32(np.cos(scatter_angle)) * scatter_radius
            reflected_y = anchor_y + np.float32(np.sin(scatter_angle)) * scatter_radius
        else:
            t = np.float32(0.18 + 1.28 * (idx + 0.35) / max(num_reflections, 1))
            reflected_x = np.float32(0.5) + (np.float32(0.5) - np.float32(sx)) * t
            reflected_y = np.float32(0.5) + (np.float32(0.5) - np.float32(sy)) * t
            tangent_angle = np.float32(np.deg2rad(angle + 90.0))
            reflected_x += np.float32(np.cos(tangent_angle) * np.random.uniform(-0.018, 0.018))
            reflected_y += np.float32(np.sin(tangent_angle) * np.random.uniform(-0.018, 0.018))

        size = float(sizes[idx])
        gloss_sharp = 0.55 + 0.65 * surface_gloss
        major = size * (1.0 + (elongation - 1.0) * np.random.uniform(0.35, 1.0))
        minor = max(size / max(elongation, 1e-3), size * 0.22) * np.random.uniform(0.72, 1.18)
        lobe_angle = float(angle + np.random.uniform(-16.0, 16.0) + 13.0 * idx)
        core = _elliptical_gaussian(
            x_grid,
            y_grid,
            center_x=float(reflected_x),
            center_y=float(reflected_y),
            sigma_x=max(major * (1.0 - 0.55 * sharpness), 0.002),
            sigma_y=max(minor * (1.0 - 0.48 * sharpness), 0.002),
            angle=lobe_angle,
            ring=0.0,
        )
        patch = _elliptical_gaussian(
            x_grid,
            y_grid,
            center_x=float(reflected_x),
            center_y=float(reflected_y),
            sigma_x=major * (1.25 + 0.75 * (1.0 - surface_gloss)),
            sigma_y=minor * (1.35 + 0.85 * (1.0 - surface_gloss)),
            angle=lobe_angle,
            ring=float(np.clip(0.16 + 0.06 * idx, 0.0, 0.62)),
        )
        reflection = np.clip(sharpness * core + (1.0 - sharpness) * patch, 0.0, 1.0)
        roughness_noise = cv2.GaussianBlur(
            np.random.normal(1.0, 0.20 + 0.35 * (1.0 - surface_gloss), (h, w)).astype(np.float32),
            (0, 0),
            sigmaX=max(0.5, min(h, w) * 0.012),
        )
        reflection = np.clip(reflection * roughness_noise * (0.35 + 0.65 * surface_mask), 0.0, 1.0)

        tint = np.clip(0.84 * base_tint + 0.16 * coating_tints[idx % len(coating_tints)], 0.0, 1.0)
        weight = np.float32(intensity * energy * gloss_sharp / (1.0 + 0.22 * idx))
        reflection_rgb += reflection[..., None] * tint.reshape(1, 1, 3) * weight

    reflection_rgb = np.clip(reflection_rgb, 0.0, 1.0)
    if chromatic > 0.0:
        shift = np.float32(min(h, w) * 0.005 * chromatic)
        reflection_rgb[..., 0] = _shift_channel(reflection_rgb[..., 0], float(shift), 0.0)
        reflection_rgb[..., 2] = _shift_channel(reflection_rgb[..., 2], float(-shift), 0.0)

    reflection = _match_light_channels(reflection_rgb, channels)
    out_linear = _screen_blend(linear_img, reflection)
    if channels <= 3:
        out = _linear_to_srgb(out_linear)
    else:
        out = img.copy()
        out[..., :3] = _linear_to_srgb(out_linear[..., :3])
    out = _clip_image(out)
    if was_grayscale:
        return out[:, :, 0]
    return out


# Geometry Effects
@register_effect(
    category=EffectCategory.GEOMETRY, stage=EffectStage.POST_RENDER, description="Rotate the image by specified angle"
)
def rotate_image(image: ImageArray, angle: float = 0.0) -> ImageArray:
    """Rotate the image by the given angle.

    Simulates camera rotation or orientation issues that can affect
    the position and recognition of objects.

    Args:
        image: Input image array with pixel values in [0,1]
        angle: Rotation angle in degrees, in range [-30, 30]

    Returns:
        Rotated image with same dimensions
    """
    angle = _validate_range(angle, -180, 180, "angle")
    return np.asarray(rotate(image, angle, resize=False, mode="edge"), dtype=np.float32)


@register_effect(category=EffectCategory.GEOMETRY, stage=EffectStage.POST_RENDER, description="Simulate camera zoom")
def zoom_effect(
    image: ArrayLike,
    factor: float | tuple[float, float] = 1.2,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
) -> NDArray[np.float32]:
    """Apply a center zoom (in or out) to an image, returning an output of the same shape.

    Args:
        image: ArrayLike H*W*C (or H*W) array, float32 with values in [0, 1].
        factor: float or (float, float), default=1.2
            - If > 1.0, zoom in by cropping the center and up-sampling.
            - If < 1.0, zoom out by down-sampling and padding the borders.
            - If a 2-tuple (min, max), sample uniformly in that range.
        interpolation: int, default=cv2.INTER_LINEAR (1) One of OpenCV's interpolation flags:
            - 0: INTER_NEAREST
            - 1: INTER_LINEAR
            - 2: INTER_CUBIC
            - 3: INTER_AREA
            - 4: INTER_LANCZOS4
            - 5: INTER_LINEAR_EXACT
            - 6: INTER_NEAREST_EXACT
            - 7: INTER_MAX
            - 8: WARP_FILL_OUTLIERS
            - 16: WARP_INVERSE_MAP
        border_mode: int, default=cv2.BORDER_CONSTANT (0) OpenCV border mode to use when padding on zoom-out:
            - 0: BORDER_CONSTANT
            - 1: BORDER_REPLICATE
            - 2: BORDER_REFLECT
            - 3: BORDER_WRAP
            - 4: BORDER_REFLECT_101
            - 5: BORDER_TRANSPARENT
            - 16: BORDER_ISOLATED

    Returns:
        out: NDArray[np.float32] Zoomed image, same shape and dtype as input.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        # Make it H*W*1 for uniform handling
        img = img[:, :, None]

    h, w = img.shape[:2]

    # Sample factor if given as a range
    if isinstance(factor, tuple):
        min_f, max_f = factor
        if not (0 < min_f <= max_f):
            raise ValueError("factor tuple must be (min, max) with 0 < min <= max")

        f = float(np.random.uniform(min_f, max_f))

    else:
        f = factor

    if np.isclose(f, 1.0):
        return img

    # Zoom in
    if f > 1.0:
        crop_h = round(h / f)
        crop_w = round(w / f)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        patch = img[top : top + crop_h, left : left + crop_w]
        out = cv2.resize(patch, (w, h), interpolation=interpolation)

    # Zoom out
    else:
        shrink_h = round(h * f)
        shrink_w = round(w * f)
        # Retain at least one pixel in each dimension.
        shrink_h = max(1, shrink_h)
        shrink_w = max(1, shrink_w)

        small = cv2.resize(img, (shrink_w, shrink_h), interpolation=interpolation)

        pad_top = (h - shrink_h) // 2
        pad_bottom = h - shrink_h - pad_top
        pad_left = (w - shrink_w) // 2
        pad_right = w - shrink_w - pad_left

        # For copyMakeBorder, value is only used if border_mode == BORDER_CONSTANT
        border_value = (1.0, 1.0, 1.0) if img.ndim == 3 and img.shape[2] == 3 else (1.0,)  # based on image channels

        out = cv2.copyMakeBorder(
            small, pad_top, pad_bottom, pad_left, pad_right, borderType=border_mode, value=border_value
        )

    # If original was gray, drop the extra channel
    if out.shape[2] == 1:
        out = out[:, :, 0]

    return np.asarray(out, dtype=np.float32)


@register_effect(
    category=EffectCategory.GEOMETRY,
    stage=EffectStage.POST_RENDER,
    description="Scale and rotate the frame over a configurable background",
    performance_level=2,
    requires_rng=True,
)
def zoom_bg_rotate(
    image: ArrayLike,
    *,
    factor: float | tuple[float, float] | list[float] = 0.92,
    angle: float | tuple[float, float] | list[float] = 0.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    background_mode: str = "auto",
    background_color: str = "#d8d4ca",
    background_alpha: float = 0.85,
    background_blur: float = 0.7,
    background_noise: float = 0.012,
    image_probability: float = 0.5,
    image_background: ImageArray | list[list[float]] | None = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> ImageArray:
    """Composite a scaled and rotated frame over a separate background.

    Args:
        image: Input image.
        factor: Scale factor or inclusive sampling range.
        angle: Rotation angle in degrees or inclusive sampling range.
        translate_x: Horizontal translation as a fraction of image width.
        translate_y: Vertical translation as a fraction of image height.
        background_mode: Background source selection mode.
        background_color: Matplotlib-compatible fallback background color.
        background_alpha: Background opacity.
        background_blur: Gaussian blur radius for the background.
        background_noise: Standard deviation of background noise.
        image_probability: Probability of selecting the supplied image background
            when ``background_mode`` is ``"auto"``.
        image_background: Optional image used to fill exposed margins.
        interpolation: OpenCV interpolation flag.

    Returns:
        The composited image as a float32 array.
    """
    img, was_grayscale = _prepare_effect_image(image)
    h, w, channels = img.shape
    if h == 0 or w == 0:
        return np.asarray(image, dtype=np.float32)

    factor_value = _resolve_range_sample(factor, "factor")
    factor_value = _validate_range(factor_value, 0.2, 2.5, "factor")
    angle_value = _resolve_range_sample(angle, "angle")
    angle_value = _validate_range(angle_value, -45.0, 45.0, "angle")
    translate_x = _validate_range(translate_x, -0.5, 0.5, "translate_x")
    translate_y = _validate_range(translate_y, -0.5, 0.5, "translate_y")
    background_alpha = _validate_range(background_alpha, 0.0, 1.0, "background_alpha")
    background_blur = _validate_range(background_blur, 0.0, 8.0, "background_blur")
    background_noise = _validate_range(background_noise, 0.0, 0.2, "background_noise")
    image_probability = _validate_range(image_probability, 0.0, 1.0, "image_probability")

    background = _make_zoom_background(
        h,
        w,
        channels,
        background_mode=background_mode,
        background_color=background_color,
        background_alpha=background_alpha,
        background_blur=background_blur,
        background_noise=background_noise,
        image_probability=image_probability,
        image=image_background,
    )

    dst_corners = _project_phone_capture_corners(
        h, w, factor=factor_value, angle=angle_value, translate_x=translate_x, translate_y=translate_y
    )
    out = _warp_foreground_over_background(img, background, dst_corners, interpolation=interpolation)

    out = _clip_image(out)
    if was_grayscale:
        return out[:, :, 0]
    return out


@register_effect(
    category=EffectCategory.GEOMETRY,
    stage=EffectStage.POST_RENDER,
    description="Crop one edge of the image and resize to original dimensions",
    requires_rng=True,
)
def edge_cropping(image: ImageArray, crop_fraction: float = 0.2) -> ImageArray:
    """Crop one edge of the image and then resize to the original dimensions.

    Args:
        image: Input image array with pixel values in [0, 1].
        crop_fraction: Fraction of the image to crop in range [0.01, 0.5].

    Returns:
        The cropped image resized to the original dimensions.
    """
    crop_fraction = _validate_range(crop_fraction, 0.01, 0.5, "crop_fraction")

    h, w, _ = image.shape
    side = np.random.choice(["top", "bottom", "left", "right"])

    if side == "top":
        sliced = image[int(crop_fraction * h) :, :]
    elif side == "bottom":
        sliced = image[: int((1 - crop_fraction) * h), :]
    elif side == "left":
        sliced = image[:, int(crop_fraction * w) :]
    else:  # right
        sliced = image[:, : int((1 - crop_fraction) * w)]

    return np.asarray(resize(sliced, (h, w, 3), anti_aliasing=True), dtype=np.float32)


# Noise Effects
@register_effect(
    category=EffectCategory.NOISE,
    stage=EffectStage.POST_RENDER,
    description="Add Gaussian noise to simulate sensor noise",
    requires_rng=True,
)
def gaussian_noise(image: ImageArray, noise_level: float = 0.05) -> ImageArray:
    """Add Gaussian noise to the image.

    Simulates random sensor noise or image compression artifacts
    that commonly occur in real-world camera systems.

    Args:
        image: Input image array with pixel values in [0,1]
        noise_level: Standard deviation for noise in range [0.01, 0.3]

    Returns:
        Noisy image with values clipped to [0,1] range
    """
    noise_level = _validate_range(noise_level, 0.01, 0.5, "noise_level")
    return _clip_image(image + np.random.normal(0.0, noise_level, image.shape).astype(np.float32))


@register_effect(
    category=EffectCategory.NOISE,
    stage=EffectStage.POST_RENDER,
    description="Apply salt and pepper noise to simulate impulse noise",
    requires_rng=True,
)
def salt_pepper_noise(image: ImageArray, amount: float = 0.01, salt_vs_pepper: float = 0.5) -> ImageArray:
    """Apply salt and pepper noise to the image.

    Simulates impulse noise that occurs in electronic sensors and
    digital transmission errors.

    Args:
        image: Input image array with pixel values in [0,1]
        amount: Proportion of pixels affected in range [0.001, 0.1]
        salt_vs_pepper: Ratio between salt and pepper noise in range [0, 1]

    Returns:
        Modified image with salt and pepper noise
    """
    amount = _validate_range(amount, 0.001, 0.1, "amount")
    salt_vs_pepper = _validate_range(salt_vs_pepper, 0.0, 1.0, "salt_vs_pepper")

    out = image.copy()
    h, w = out.shape[:2]

    # Salt noise
    num_salt = int(np.ceil(amount * image.size * salt_vs_pepper))
    coords = np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)
    out[coords] = 1

    # Pepper noise
    num_pepper = int(np.ceil(amount * image.size * (1.0 - salt_vs_pepper)))
    coords = np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper)
    out[coords] = 0

    return out


@register_effect(
    category=EffectCategory.NOISE,
    stage=EffectStage.POST_RENDER,
    description="Apply Poisson noise based on image intensities",
    requires_rng=True,
)
def poisson_noise(image: ImageArray) -> ImageArray:
    """Apply Poisson noise based on the image intensities.

    Simulates shot noise (photon counting noise) that occurs in
    low-light imaging and is intensity-dependent.

    Args:
        image: Input image array with pixel values in [0,1]

    Returns:
        Image with Poisson noise applied
    """
    vals = 2 ** np.ceil(np.log2(len(np.unique(image))))
    return _clip_image(np.random.poisson(image * vals) / vals)


@register_effect(
    category=EffectCategory.NOISE,
    stage=EffectStage.POST_RENDER,
    description="Add multiplicative speckle noise",
    requires_rng=True,
)
def speckle_noise(image: ImageArray) -> ImageArray:
    """Add multiplicative speckle noise to the image.

    Simulates granular noise patterns that occur in certain imaging
    systems like ultrasound, radar, and SAR imaging.

    Args:
        image: Input image array with pixel values in [0,1]

    Returns:
        Image with speckle noise applied
    """
    return _clip_image((image + image * np.random.randn(*image.shape).astype(np.float32)).astype(np.float32))


@register_effect(
    category=EffectCategory.COLOR,
    stage=EffectStage.POST_RENDER,
    description="Reduce continuous color gradients into discrete bands",
)
def color_banding(image: ImageArray, levels: int = 8) -> ImageArray:
    """Reduce continuous color gradients into discrete bands.

    Simulates reduced color depth or bit-depth limitations
    in display devices or image compression.

    Args:
        image: Input image array with pixel values in [0,1]
        levels: Number of color levels in range [2, 32]

    Returns:
        Image with reduced color depth
    """
    levels = int(_validate_range(levels, 2, 32, "levels"))
    return np.asarray(np.floor(image * levels) / levels, dtype=np.float32)


# Blur Effects
@register_effect(
    category=EffectCategory.BLUR,
    stage=EffectStage.POST_RENDER,
    description="Apply Gaussian blur to simulate defocus",
    performance_level=2,
)
def gaussian_blur(image: ImageArray, sigma: float = 1.0) -> ImageArray:
    """Apply Gaussian defocus blur to the image.

    Simulates out-of-focus camera effects or depth-of-field issues
    that can occur during image capture.

    Args:
        image: Input image array with pixel values in [0,1]
        sigma: Standard deviation for Gaussian kernel in range [0.5, 5]

    Returns:
        Blurred image with defocus effect applied
    """
    sigma = _validate_range(sigma, 0.0001, 5.0, "sigma")
    k = max(1, round(6 * sigma))
    if k % 2 == 0:
        k += 1
    return np.asarray(cv2.GaussianBlur(image, (k, k), sigmaX=sigma), dtype=np.float32)


@register_effect(
    category=EffectCategory.BLUR,
    stage=EffectStage.POST_RENDER,
    description="Apply horizontal motion blur",
    performance_level=2,
)
def motion_blur(image: ImageArray, kernel_size: int = 5) -> ImageArray:
    """Apply horizontal motion blur to the image.

    Simulates camera or object movement during image capture,
    causing objects to appear streaked horizontally.

    Args:
        image: Input image array with pixel values in [0,1]
        kernel_size: Kernel size in range [3, 21] (must be odd)

    Returns:
        Image with horizontal motion blur applied
    """
    kernel_size = int(_validate_range(kernel_size, 3, 21, "kernel_size"))
    if kernel_size % 2 == 0:
        kernel_size += 1  # Gaussian kernels require an odd size.

    img: NDArray[np.float32] = np.asarray(image, dtype=np.float32)
    if img.ndim < 2:
        raise ValueError("motion_blur expects at least 2 dimensions (HxW or HxWxC).")

    # Circular moving-average along width (axis=1) without np.roll temporaries.
    out: NDArray[np.float32] = np.zeros_like(img, dtype=np.float32)
    half: int = kernel_size // 2
    for dx in range(-half, half + 1):
        if dx > 0:
            out[:, :dx, ...] += img[:, -dx:, ...]
            out[:, dx:, ...] += img[:, :-dx, ...]
        elif dx < 0:
            s: int = -dx
            out[:, :-s, ...] += img[:, s:, ...]
            out[:, -s:, ...] += img[:, :s, ...]
        else:
            out += img

    out /= np.float32(kernel_size)
    return np.asarray(out, dtype=np.float32)


@register_effect(
    category=EffectCategory.BLUR,
    stage=EffectStage.POST_RENDER,
    description="Apply Gaussian defocus blur to simulate out-of-focus effects",
)
def defocus_blur(image: ImageArray, sigma: float = 1.0) -> ImageArray:
    """Apply Gaussian defocus blur to the image.

    Simulates out-of-focus camera effects or depth-of-field issues
    that can occur during image capture.

    Args:
        image: Input image array with pixel values in [0,1]
        sigma: Standard deviation for the Gaussian kernel in range [0.5, 5]

    Returns:
        Blurred image with defocus effect applied
    """
    sigma = _validate_range(sigma, 0.0001, 5.0, "sigma")
    k = max(1, round(6 * sigma))
    if k % 2 == 0:
        k += 1
    return np.asarray(cv2.GaussianBlur(image, (k, k), sigmaX=sigma), dtype=np.float32)


@register_effect(
    category=EffectCategory.BLUR,
    stage=EffectStage.POST_RENDER,
    description="Simulate radial blur by averaging multiple rotated versions",
    requires_rng=True,
)
def radial_blur(image: ImageArray, num_rotations: int = 5, max_angle: float = 3.0) -> ImageArray:
    """Simulate radial blur by averaging multiple rotated versions.

    Simulates rotational camera shake or motion blur that occurs
    in a circular pattern.

    Args:
        image: Input image array with pixel values in [0,1]
        num_rotations: Number of rotations in range [3, 15]
        max_angle: Maximum rotation angle in degrees in range [1, 10]

    Returns:
        Image with radial blur applied
    """
    num_rotations = int(_validate_range(num_rotations, 1, 20, "num_rotations"))
    max_angle = _validate_range(max_angle, 1.0, 10.0, "max_angle")

    angles = np.linspace(-max_angle, max_angle, num_rotations)
    acc = sum(rotate(image, a, resize=False, mode="edge") for a in angles)
    return np.asarray(acc / num_rotations, dtype=np.float32)


# Distortion Effects
@register_effect(
    category=EffectCategory.DISTORTION,
    stage=EffectStage.POST_RENDER,
    description="Apply barrel (radial) distortion to simulate lens effects",
)
def barrel_distortion(image: ImageArray, strength: float = 1e-5) -> ImageArray:
    """Apply barrel (radial) distortion to the image.

    Simulates lens distortion effects that occur in real cameras,
    particularly with wide-angle lenses.

    Args:
        image: Input image array with pixel values in [0,1]
        strength: Distortion strength in range [1e-6, 1e-3]

    Returns:
        Image with barrel distortion effect applied
    """
    strength = _validate_range(strength, 1e-6, 1e-3, "strength")
    # Simplified barrel distortion
    return np.stack(
        [
            np.interp(
                np.arange(image.shape[0] * image.shape[1]),
                np.arange(image.shape[0] * image.shape[1]),
                image[..., c].ravel(),
            ).reshape(image.shape[:2])
            for c in range(3)
        ],
        axis=2,
    )


# Perspective Effects
@register_effect(
    category=EffectCategory.GEOMETRY,
    stage=EffectStage.POST_RENDER,
    description="Apply perspective transform to simulate viewing angle changes",
)
def perspective_transform(image: ImageArray, delta: float = 0.1) -> ImageArray:
    """Apply a perspective transform to simulate viewing angle changes.

    Simulates changes in the camera viewing angle or orientation
    relative to objects, affecting how faces and edges appear.

    Args:
        image: Input image array with pixel values in [0,1]
        delta: Perspective distortion strength in range [0.05, 0.3]

    Returns:
        Transformed image with perspective effect applied
    """
    delta = _validate_range(delta, 0.00001, 0.3, "delta")
    h, w, _ = image.shape
    d = delta * min(h, w)

    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = np.array([[d, 0], [w - d, d], [w - d, h - d], [d, h - d]], dtype=np.float32)

    tform = ProjectiveTransform()
    tform.estimate(src, dst)
    return np.asarray(warp(image, tform, output_shape=(h, w)), dtype=np.float32)


@register_effect(
    category=EffectCategory.GEOMETRY,
    stage=EffectStage.POST_RENDER,
    description="Apply random affine transformations",
    requires_rng=True,
)
def spatial_perturbation(
    image: ImageArray, max_translation: float = 0.05, max_rotation: float = 10.0, max_scale: float = 0.1
) -> ImageArray:
    """Apply random affine transformations to the image.

    Combines multiple spatial transformations to simulate camera movement
    and positioning variations.

    Args:
        image: Input image array with pixel values in [0,1]
        max_translation: Maximum translation fraction in range [0, 0.2]
        max_rotation: Maximum rotation in degrees in range [0, 20]
        max_scale: Maximum scale change in range [0, 0.3]

    Returns:
        Transformed image with random combination of transformations
    """
    max_translation = _validate_range(max_translation, 0.0, 0.2, "max_translation")
    max_rotation = _validate_range(max_rotation, 0.0, 50.0, "max_rotation")
    max_scale = _validate_range(max_scale, 0.0, 0.3, "max_scale")

    h, w, _ = image.shape
    tmax = max_translation * min(h, w)

    tform = AffineTransform(
        translation=np.random.uniform(-tmax, tmax, 2),
        rotation=np.deg2rad(np.random.uniform(-max_rotation, max_rotation)),
        scale=1.0 + np.random.uniform(-max_scale, max_scale, 2),
    )
    return np.asarray(warp(image, tform, output_shape=(h, w)), dtype=np.float32)


# Adversarial Effects
@register_effect(
    category=EffectCategory.NOISE,
    stage=EffectStage.POST_RENDER,
    description="Add sinusoidal noise pattern to simulate adversarial examples",
    requires_rng=True,
)
def adversarial_noise(image: ImageArray, frequency: int = 10, amplitude: float = 0.05) -> ImageArray:
    """Add a sinusoidal noise pattern to the image.

    Introduces structured noise patterns that can challenge
    machine learning models, simulating adversarial examples.

    Args:
        image: Input image array with pixel values in [0,1]
        frequency: Frequency of sinusoidal noise in range [5, 50]
        amplitude: Amplitude of noise in range [0.01, 0.2]

    Returns:
        Modified image with adversarial noise pattern
    """
    frequency = int(_validate_range(frequency, 1, 50, "frequency"))
    amplitude = _validate_range(amplitude, 0.01, 0.2, "amplitude")

    h, w, _ = image.shape
    xv, yv = np.meshgrid(np.linspace(0, 2 * np.pi, w), np.linspace(0, 2 * np.pi, h))
    pattern = amplitude * 0.5 * (np.sin(frequency * xv) + np.cos(frequency * yv))
    return _clip_image((image + pattern[..., None]).astype(np.float32))


# Occlusion Effects
@register_effect(
    category=EffectCategory.OCCLUSION,
    stage=EffectStage.POST_RENDER,
    description="Add random occluding shapes over the image",
    requires_rng=True,
)
def random_occluders(
    image: ImageArray, num_shapes: int = 3, size_fraction: float | tuple[float, float] = (0.1, 0.25)
) -> ImageArray:
    """Add random occluding shapes (rectangles or circles) over the image.

    Simulates partial occlusion of objects by other objects, which is a
    common challenge in real-world scenes.

    Args:
        image: Input image array with pixel values in [0,1]
        num_shapes: Number of occluding shapes in range [1, 10]
        size_fraction: Fraction (or tuple of min,max) of image dimension for occluder size, in [0,1]

    Returns:
        Image with random occluding shapes applied
    """
    num_shapes = int(_validate_range(num_shapes, 1, 10, "num_shapes"))
    # Validate size fraction range
    if isinstance(size_fraction, tuple):
        min_frac, max_frac = size_fraction
    else:
        min_frac = max_frac = size_fraction
    min_frac = _validate_range(min_frac, 0.0, 1.0, "size_fraction_min")
    max_frac = _validate_range(max_frac, 0.0, 1.0, "size_fraction_max")
    if min_frac > max_frac:
        min_frac, max_frac = max_frac, min_frac

    h, w, _ = image.shape
    out = image.copy()

    # Accumulate every shape into one mask before applying it to the image.
    mask_total = np.ones((h, w), dtype=out.dtype)
    y_grid, x_grid = np.ogrid[:h, :w]

    for _ in range(num_shapes):
        shape_type = np.random.choice(["rect", "circle"])

        if shape_type == "rect":
            min_rw = max(1, int(min_frac * w))
            max_rw = max(min_rw + 1, int(max_frac * w))
            rect_w = np.random.randint(min_rw, max_rw)
            min_rh = max(1, int(min_frac * h))
            max_rh = max(min_rh + 1, int(max_frac * h))
            rect_h = np.random.randint(min_rh, max_rh)
            top = np.random.randint(0, h - rect_h)
            left = np.random.randint(0, w - rect_w)
            mask_total[top : top + rect_h, left : left + rect_w] = 0
        else:
            center_x = np.random.randint(0, w)
            center_y = np.random.randint(0, h)
            min_rad = max(1, int(min_frac * min(h, w)))
            max_rad = max(min_rad + 1, int(max_frac * min(h, w)))
            radius = np.random.randint(min_rad, max_rad)
            # Apply circular occluder mask
            dist = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
            mask_total[dist <= radius] = 0

    # Apply cumulative mask to image
    out *= mask_total[..., np.newaxis]

    return out


# Background Image Effect
@register_effect(
    category=EffectCategory.BACKGROUND,
    stage=EffectStage.PRE_RENDER,
    description="Apply an image as background using CIFAR-10 dataset",
)
def background_image(
    fig: FigureType, image: ImageArray | list[list[float]] | None = None, alpha: float = 0.8, resize_mode: str = "fill"
) -> FigureType:
    """Apply an image as the figure background.

    When ``image`` is ``None``, the effect loads a CIFAR-10 image. The loader
    downloads the dataset when it is absent and uses a generated image if the
    dataset cannot be loaded.

    Args:
        fig: Matplotlib figure to modify.
        image: Background image, or ``None`` to load a CIFAR-10 image.
        alpha: Image opacity in the range ``[0.1, 1.0]``.
        resize_mode: Placement mode. Accepted values are ``fill``, ``contain``,
            ``cover``, and ``center``.

    Returns:
        The modified figure.
    """
    if image is None:
        cifar_images: list[NDArray[np.float64]] = []
        try:
            cifar_images = load_cifar10_images()

        except Exception:
            cifar_images = []
        if cifar_images:
            # Select a CIFAR image and convert
            idx = np.random.randint(0, len(cifar_images))
            raw_img = cifar_images[idx]
            # load_cifar10_images() returns ndarray images (float64) - convert to float32
            img = raw_img.astype(np.float32, copy=False)
        else:
            # Procedural fallback
            h, w = 256, 256
            noise = np.random.random((h, w, 3)).astype(np.float32) * 0.1

            # Add gradient
            y_grad = np.linspace(0.3, 0.7, h, dtype=np.float32).reshape(-1, 1, 1)
            x_grad = np.linspace(0.2, 0.6, w, dtype=np.float32).reshape(1, -1, 1)
            # Combine noise and gradients
            img = np.clip(noise + y_grad + x_grad, 0.0, 1.0).astype(np.float32)
    elif not isinstance(image, np.ndarray):
        # Coerce provided image into a float32 ndarray
        img = np.asarray(image, dtype=np.float32)
    else:
        img = image.astype(np.float32, copy=False)

    if img.ndim == 2:
        # Replicate grayscale values across RGB channels as float32.
        img = np.stack([img, img, img], axis=-1).astype(np.float32)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Image must have shape (H, W, 3), got {img.shape}")

    alpha = _validate_range(alpha, 0.0, 1.0, "alpha")

    # Figure size (pixels)
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig_h, fig_w = int(bbox.height * fig.dpi), int(bbox.width * fig.dpi)
    fig_aspect = fig_w / fig_h

    # Image aspect & target size
    img_h, img_w = img.shape[:2]
    img_aspect = img_w / img_h

    # Resize according to mode. CIFAR backgrounds already match the default
    # Cube3 render size (32x32), so avoid an expensive no-op skimage resize.
    if img_h == fig_h and img_w == fig_w and resize_mode in {"fill", "contain", "cover", "center"}:
        resized = img
    elif resize_mode == "fill":
        resized = resize(img, (fig_h, fig_w, 3), anti_aliasing=True)
    elif resize_mode == "contain":
        if img_aspect > fig_aspect:
            new_w = fig_w
            new_h = int(fig_w / img_aspect)
        else:
            new_h = fig_h
            new_w = int(fig_h * img_aspect)
        resized = resize(img, (new_h, new_w, 3), anti_aliasing=True)
    elif resize_mode == "cover":
        if img_aspect < fig_aspect:
            new_w = fig_w
            new_h = int(fig_w / img_aspect)
        else:
            new_h = fig_h
            new_w = int(fig_h * img_aspect)
        resized = resize(img, (new_h, new_w, 3), anti_aliasing=True)
        # Crop to fit
        y_offset = (new_h - fig_h) // 2
        x_offset = (new_w - fig_w) // 2
        resized = resized[y_offset : y_offset + fig_h, x_offset : x_offset + fig_w]
    else:  # center
        resized = img

    # Create background axes
    ax = fig.add_axes((0, 0, 1, 1), zorder=-100)
    ax.set_axis_off()
    ax.imshow(np.asarray(resized, dtype=np.float32), aspect="auto", alpha=alpha)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(fig_h, 0)

    return fig


# Weather Effects
@register_effect(
    category=EffectCategory.WEATHER,
    stage=EffectStage.POST_RENDER,
    description="Apply a fog or haze effect based on a radial gradient",
)
def fog_haze(image: ImageArray, spread: float = 0.5, intensity: float = 0.3) -> ImageArray:
    """Apply a fog or haze effect based on a radial gradient.

    The radial mask lowers contrast away from the image center.

    Args:
        image: Input image array with pixel values in [0, 1].
        spread: Gradient spread in range [0.1, 2.0].
        intensity: Haze intensity in range [0.1, 0.7].

    Returns:
        The image with the fog or haze effect applied.
    """
    spread = _validate_range(spread, 0.0001, 2.0, "spread")
    intensity = _validate_range(intensity, 0.0001, 1.0, "intensity")

    h, w, _ = image.shape
    x, y = np.meshgrid(np.linspace(-1, 1, w, dtype=np.float32), np.linspace(-1, 1, h, dtype=np.float32))
    gradient = np.exp(-(x**2 + y**2) / (spread**2))
    out = (image + intensity * (1.0 - gradient)[..., None]).astype(np.float32)
    # Raise contrast and brightness while keeping values in [0, 1].
    out = np.clip(out * (1.0 + 0.05) + 0.02, 0.0, 1.0)
    return _clip_image(out)


@register_effect(
    category=EffectCategory.WEATHER,
    stage=EffectStage.POST_RENDER,
    description="Simulate rain streaks by drawing white diagonal streaks",
    requires_rng=True,
)
def rain_effect(image: ImageArray, num_streaks: int = 20, thickness: int = 1) -> ImageArray:
    """Draw white diagonal streaks over the image.

    Args:
        image: Input image array with pixel values in [0, 1].
        num_streaks: Number of streaks in range [5, 100].
        thickness: Thickness of each streak in pixels in range [1, 3].

    Returns:
        The image with diagonal streaks applied.
    """
    num_streaks = int(_validate_range(num_streaks, 5, 100, "num_streaks"))
    thickness = int(_validate_range(thickness, 1, 3, "thickness"))

    h, w, _ = image.shape
    out = image.copy()

    # Vectorized RNG: exactly num_streaks x 3 draws
    draws = np.random.randint(low=[0, 0, 5], high=[w, h, 15], size=(num_streaks, 3))
    x_starts, y_starts, lengths = draws[:, 0], draws[:, 1], draws[:, 2]

    # Build a grid of offsets up to the maximum streak length
    max_len = np.max(lengths)
    i = np.arange(max_len)

    # Mask: which (streak, offset) pairs are actually used
    mask = i[None, :] < lengths[:, None]  # shape (num_streaks, max_len)

    # Compute all pixel coords for every valid streak position
    x_vals = x_starts[:, None] + i[None, :]
    y_vals = y_starts[:, None] + (i[None, :] // 2)
    np.minimum(x_vals, w - 1, out=x_vals)
    np.minimum(y_vals, h - 1, out=y_vals)

    # Flatten to 1D lists of coordinates
    x_flat = x_vals[mask]
    y_flat = y_vals[mask]

    # Draw streaks with the specified thickness
    if thickness == 1:
        out[y_flat, x_flat] = 1.0
    else:
        for x, y in zip(x_flat, y_flat, strict=True):
            x_min, x_max = max(0, x - thickness // 2), min(w, x + thickness // 2 + 1)
            y_min, y_max = max(0, y - thickness // 2), min(h, y + thickness // 2 + 1)
            out[y_min:y_max, x_min:x_max] = 1.0

    return _clip_image(out.astype(np.float32))


# Sensor Effects
@register_effect(
    category=EffectCategory.SENSOR,
    stage=EffectStage.POST_RENDER,
    description=(
        "Mimic Matplotlib bilinear resize quality loss and PIL PNG/JPEG save artifacts "
        "without changing output dimensions"
    ),
    performance_level=3,
)
def resize_degradation(
    image: ArrayLike,
    *,
    strength: float = 1.0,
    resize_scale: float = 0.5,
    target_height: int = 0,
    target_width: int = 0,
    interpolation: str = "bilinear",
    save_format: str = "png",
    jpeg_quality: int = 85,
    jpeg_subsampling: int = 2,
    png_compress_level: int = 6,
    roundtrips: int = 1,
) -> ImageArray:
    """Mimic quality loss from Matplotlib resize plus PNG/JPEG persistence.

    The effect intentionally returns the original image dimensions. It performs a
    virtual resize through Matplotlib's Agg canvas to a smaller target size, resizes
    that result back to the original size with the same Matplotlib path, and then
    round-trips through PIL's save/load codec. This reproduces the perceptual blur,
    8-bit quantization, and optional JPEG block/chroma artifacts seen in the
    ``process_image`` stage without changing array shape.
    """
    img, was_grayscale = _prepare_effect_image(image)
    h, w, channels = img.shape
    if h == 0 or w == 0:
        return np.asarray(image, dtype=np.float32)

    strength = _validate_range(strength, 0.0, 1.0, "strength")
    if strength <= 0.0:
        return img[:, :, 0] if was_grayscale else img.copy()

    resize_scale = _validate_range(resize_scale, 0.02, 4.0, "resize_scale")
    target_h = target_height if target_height > 0 else round(h * resize_scale)
    target_w = target_width if target_width > 0 else round(w * resize_scale)
    target_h = int(np.clip(target_h, 1, max(1, h * 4)))
    target_w = int(np.clip(target_w, 1, max(1, w * 4)))
    roundtrips = int(_validate_range(roundtrips, 1, 4, "roundtrips"))
    jpeg_quality = int(_validate_range(jpeg_quality, 1, 100, "jpeg_quality"))
    jpeg_subsampling = int(_validate_range(jpeg_subsampling, 0, 2, "jpeg_subsampling"))
    png_compress_level = int(_validate_range(png_compress_level, 0, 9, "png_compress_level"))

    allowed_interpolations = {
        "none",
        "nearest",
        "bilinear",
        "bicubic",
        "spline16",
        "spline36",
        "hanning",
        "hamming",
        "hermite",
        "kaiser",
        "quadric",
        "catrom",
        "gaussian",
        "bessel",
        "mitchell",
        "sinc",
        "lanczos",
    }
    interpolation = interpolation.strip().lower()
    if interpolation not in allowed_interpolations:
        logger.warning("Unknown Matplotlib interpolation '%s'. Using bilinear", interpolation)
        interpolation = "bilinear"

    rgb = np.repeat(img, 3, axis=2) if channels == 1 else img[..., :3]
    degraded = np.asarray(rgb, dtype=np.float32)
    for _ in range(roundtrips):
        down = _matplotlib_resize_rgb_exact(degraded, target_h, target_w, interpolation=interpolation)
        degraded = _matplotlib_resize_rgb_exact(down, h, w, interpolation=interpolation)
        degraded = _pil_codec_roundtrip_rgb(
            degraded,
            save_format=save_format,
            jpeg_quality=jpeg_quality,
            jpeg_subsampling=jpeg_subsampling,
            png_compress_level=png_compress_level,
        )

    mixed_rgb: NDArray[np.float32] = np.clip(
        (np.float32(1.0) - np.float32(strength)) * rgb + np.float32(strength) * degraded, 0.0, 1.0
    ).astype(np.float32, copy=False)
    if channels == 1:
        out = _luminance(mixed_rgb)[:, :, None]
    else:
        out = img.copy()
        out[..., :3] = mixed_rgb
    out = _clip_image(out)
    return out[:, :, 0] if was_grayscale else out


@register_effect(
    category=EffectCategory.SENSOR,
    stage=EffectStage.POST_RENDER,
    description="Simulate sensor inaccuracy by combining Gaussian and salt-pepper noise",
    requires_rng=True,
)
def sensor_inaccuracy(
    image: ImageArray, noise_level: float = 0.03, sp_amount: float = 0.005, sp_ratio: float = 0.5
) -> ImageArray:
    """Combine Gaussian and salt-and-pepper noise.

    Args:
        image: Input image array with pixel values in [0, 1].
        noise_level: Gaussian noise level in range [0.01, 0.1].
        sp_amount: Salt-and-pepper noise amount in range [0.001, 0.02].
        sp_ratio: Ratio of salt to pepper noise in range [0, 1].

    Returns:
        The image with both noise distributions applied.
    """
    noise_level = _validate_range(noise_level, 0.01, 0.5, "noise_level")
    sp_amount = _validate_range(sp_amount, 0.001, 0.02, "sp_amount")
    sp_ratio = _validate_range(sp_ratio, 0.0, 1.0, "sp_ratio")

    # Apply Gaussian noise first
    out = _clip_image((image + np.random.normal(0.0, noise_level, image.shape)).astype(np.float32))

    # Add salt and pepper noise
    h, w, _ = out.shape
    n = int(np.ceil(sp_amount * out.size))
    ns = int(np.ceil(n * sp_ratio))

    # Salt (white pixels)
    coords = np.random.randint(0, h, ns), np.random.randint(0, w, ns)
    out[coords] = 1.0

    # Pepper (black pixels)
    coords = np.random.randint(0, h, n - ns), np.random.randint(0, w, n - ns)
    out[coords] = 0.0

    return out
