from __future__ import annotations

import io
import math
import operator
from typing import TYPE_CHECKING

import cv2
from matplotlib.axes import Axes
import numpy as np
from numpy import float32, uint8

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cv2.typing import MatLike, Moments
    from matplotlib.figure import Figure

    # from matplotlib.figure import SubFigure
    from numpy import number as np_number
    from numpy.typing import NDArray


def rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convert RGB to HSV color space.

    Args:
        r: Red component in [0, 1]
        g: Green component in [0, 1]
        b: Blue component in [0, 1]

    Returns:
        Tuple of (hue, saturation, value) in [0, 1]
    """
    max_val: float = max(r, g, b)
    min_val: float = min(r, g, b)
    diff: float = max_val - min_val

    # Value
    v: float = max_val

    # Saturation
    s: float = 0 if max_val == 0 else diff / max_val

    # Hue
    h: float
    if diff == 0:
        h = 0.0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:  # max_val == b
        h = (60 * ((r - g) / diff) + 240) % 360

    return h / 360.0, s, v


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV to RGB color space.

    Args:
        h: Hue component in [0, 1]
        s: Saturation component in [0, 1]
        v: Value (brightness) component in [0, 1]

    Returns:
        Tuple of (red, green, blue) in [0, 1]
    """
    h *= 360.0  # Convert to degrees
    c: float = v * s
    x: float = c * (1.0 - abs((h / 60.0) % 2 - 1.0))
    m: float = v - c

    r: float
    g: float
    b: float
    if 0 <= h < 60:
        r, g, b = c, x, 0.0
    elif 60 <= h < 120:
        r, g, b = x, c, 0.0
    elif 120 <= h < 180:
        r, g, b = 0.0, c, x
    elif 180 <= h < 240:
        r, g, b = 0.0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x

    return r + m, g + m, b + m


def compute_relative_luminance(r: float, g: float, b: float) -> float:
    """Compute relative luminance using WCAG sRGB formula.

    Args:
        r: Red component in [0, 1]
        g: Green component in [0, 1]
        b: Blue component in [0, 1]

    Returns:
        Relative luminance in [0, 1]
    """
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def compute_contrast_ratio(l1: float, l2: float) -> float:
    """Compute WCAG contrast ratio between two luminance values.

    Args:
        l1: First relative luminance value in [0, 1]
        l2: Second relative luminance value in [0, 1]

    Returns:
        Contrast ratio (>= 1.0)
    """
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def contrast_potential(color: tuple[float, float, float], bg_luminance: float) -> float:
    """Estimate contrast potential of a color against a background luminance.

    Args:
        color: The RGB color as a tuple (r, g, b) in [0, 1]
        bg_luminance: The background luminance in [0, 1]

    Returns:
        Contrast potential score (higher is better)
    """
    # r, g, b = color
    luminance: float = compute_relative_luminance(*color)
    return compute_contrast_ratio(luminance, bg_luminance)


def generate_contrast_color_candidates(
    r_bg: float, g_bg: float, b_bg: float, l_bg: float, num_candidates: int = 50
) -> list[tuple[float, float, float]]:
    """Generate RGB candidates from background hue and luminance.

    Candidates include fixed hue offsets, evenly spaced HSV samples, and dark
    or bright endpoints. They are sorted by contrast against the background.

    Args:
        r_bg: Background red value in [0, 1]
        g_bg: Background green value in [0, 1]
        b_bg: Background blue value in [0, 1]
        l_bg: Background luminance in [0, 1]
        num_candidates: Number of candidate colors to generate

    Returns:
        List of (r, g, b) tuples representing candidate colors
    """
    candidates: list[tuple[float, float, float]] = []
    h_bg: float
    s_bg: float
    h_bg, s_bg, _v_bg = rgb_to_hsv(r_bg, g_bg, b_bg)

    # 1. Luminance-based adjustment
    # Generate target luminance values that will maximize contrast
    # Dark background - generate bright colors, Light background - generate dark colors
    target_luminances: NDArray[float32] = (
        np.linspace(0.6, 1.0, 8, dtype=float32) if l_bg < 0.5 else np.linspace(0.0, 0.4, 8, dtype=float32)
    )

    # Add some mid-range luminances for variety
    target_luminances = np.concatenate([target_luminances, np.linspace(0.4, 0.6, 3, dtype=float32)])

    # 2. Systematic hue exploration
    # Generate hues based on color theory relationships
    hue_strategies: list[float] = [
        # Complementary (180° opposite)
        (h_bg + 0.5) % 1.0,
        # Triadic (120° intervals)
        (h_bg + 0.333) % 1.0,
        (h_bg + 0.667) % 1.0,
        # Tetradic (90° intervals)
        (h_bg + 0.25) % 1.0,
        (h_bg + 0.75) % 1.0,
        # Split-complementary (150° and 210°)
        (h_bg + 0.417) % 1.0,
        (h_bg + 0.583) % 1.0,
        # Analogous with high contrast (60° offset)
        (h_bg + 0.167) % 1.0,
        (h_bg + 0.833) % 1.0,
    ]

    # Add systematic hue sampling
    hue_strategies.extend((h_bg + i / 12) % 1.0 for i in range(12))

    # 3. Select saturation levels from the background saturation.
    # Base saturation on background properties
    saturation_levels: list[float]
    if s_bg < 0.2:
        # Desaturated background - use high saturation for pop
        saturation_levels = [0.7, 0.9, 1.0]
    elif s_bg < 0.5:
        # Moderately saturated background - balance
        saturation_levels = [0.5, 0.8, 1.0]
    else:
        # For background saturation above 0.5, include low and medium saturation.
        saturation_levels = [0.3, 0.6, 0.9]

    # Add some desaturated options for subtle highlights
    saturation_levels.extend([0.2, 0.4])

    # 4. Generate candidates by combining hue, saturation, and target luminance
    hue: float
    sat: float
    r: float
    g: float
    b: float
    for target_l in target_luminances:
        for hue in hue_strategies:
            for sat in saturation_levels:
                # Use iterative approach to find value that achieves target luminance
                best_value = 0.5
                best_diff = float("inf")

                # Search for the value that gets closest to target luminance
                for v_test in np.linspace(0.1, 1.0, 20):
                    r_test, g_test, b_test = hsv_to_rgb(hue, sat, v_test)
                    l_test = compute_relative_luminance(r_test, g_test, b_test)
                    diff = abs(l_test - target_l)

                    if diff < best_diff:
                        best_diff = diff
                        best_value = v_test

                # Generate the color
                r, g, b = hsv_to_rgb(hue, sat, best_value)
                candidates.append((r, g, b))

    # 5. Add luminance endpoints.
    # Maximum contrast colors based on luminance
    if l_bg < 0.5:
        # Dark background - add bright pure colors
        candidates.extend([
            (1.0, 1.0, 1.0),  # White
            (1.0, 1.0, 0.0),  # Yellow (high luminance)
            (0.0, 1.0, 1.0),  # Cyan (high luminance)
        ])
    else:
        # Light background - add dark pure colors
        candidates.extend([
            (0.0, 0.0, 0.0),  # Black
            (1.0, 0.0, 0.0),  # Red
            (0.0, 0.0, 1.0),  # Blue
        ])

    # 6. Perceptual uniformity: add colors distributed evenly in LAB-like space
    # Simplified perceptual sampling
    val: float
    for i in range(8):
        hue = i / 8.0
        for j in range(3):
            sat = 0.4 + j * 0.3  # 0.4, 0.7, 1.0
            for k in range(3):
                val = 0.3 + k * 0.35  # 0.3, 0.65, 1.0
                r, g, b = hsv_to_rgb(hue, sat, val)
                candidates.append((r, g, b))

    # Remove duplicates and limit to requested number
    unique_candidates: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()

    key: tuple[float, float, float]
    for r, g, b in candidates:
        # Round to avoid floating point precision issues
        key = (round(r, 3), round(g, 3), round(b, 3))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((r, g, b))

    # Sort by potential contrast and return top candidates
    def _contrast_key(color: tuple[float, float, float]) -> float:
        return contrast_potential(color, l_bg)

    unique_candidates.sort(key=_contrast_key, reverse=True)
    return unique_candidates[:num_candidates]


def find_min_alpha_for_contrast(
    r_ov: float, g_ov: float, b_ov: float, target_ratio: float, l_bg: float
) -> tuple[float | None, float]:
    """Binary search to find minimum alpha that achieves target contrast ratio.

    Args:
        r_ov: Overlay red value in [0, 1]
        g_ov: Overlay green value in [0, 1]
        b_ov: Overlay blue value in [0, 1]
        target_ratio: Desired contrast ratio (e.g. 4.5)
        l_bg: Background luminance in [0, 1]

    Returns:
        Tuple of (minimum alpha in [0, 1] or None if not achievable, achieved contrast ratio)

    """
    l_ov: float = compute_relative_luminance(r_ov, g_ov, b_ov)

    # Check whether this color can reach the target contrast.
    max_contrast: float = compute_contrast_ratio(l_ov, l_bg)
    if max_contrast < target_ratio:
        return None, max_contrast

    # Binary search for minimum alpha
    alpha_low: float = 0.0
    alpha_high: float = 1.0

    alpha_mid: float
    l_blend: float
    contrast_blend: float
    for _ in range(40):
        alpha_mid = (alpha_low + alpha_high) / 2.0

        # Compute blended luminance
        l_blend = alpha_mid * l_ov + (1 - alpha_mid) * l_bg
        contrast_blend = compute_contrast_ratio(l_blend, l_bg)

        if contrast_blend >= target_ratio:
            alpha_high = alpha_mid
        else:
            alpha_low = alpha_mid

    return alpha_high, compute_contrast_ratio(alpha_high * l_ov + (1 - alpha_high) * l_bg, l_bg)


def select_contrast_fill(
    patch_pixels: NDArray[np_number],
    target_contrast_ratio: float = 4.5,
    min_alpha: float = 0.3,
    max_alpha: float = 0.75,
    large_area_threshold: int = 10000,
) -> tuple[tuple[int, int, int], float]:
    """Select a BGR fill and opacity from the region's background colors.

    Candidate colors are evaluated against the mean background luminance. The
    selected candidate is the highest weighted combination of contrast, opacity,
    hue separation, and saturation, subject to the requested alpha bounds.

    Args:
        patch_pixels: numpy array of shape (H, W, 3) with RGB values in [0, 1]
        target_contrast_ratio: Minimum WCAG contrast ratio (4.5 for normal, 3.0 for large areas)
        min_alpha: Minimum allowed opacity for the selected fill.
        max_alpha: Maximum allowed opacity to avoid over-occlusion
        large_area_threshold: Pixel area above which to use relaxed 3:1 contrast (WCAG large-text exception)

    Returns:
        Tuple of (fill_color_bgr, opacity) where:
        - fill_color_bgr: BGR color tuple (0-255) for OpenCV
        - opacity: Alpha value in [0, 1]
    """
    if patch_pixels.size == 0:
        # Fallback for empty patches
        return (0, 255, 0), 0.5

    # Adjust target for large areas (WCAG large-text exception)
    area: int = patch_pixels.shape[0] * patch_pixels.shape[1]
    if area > large_area_threshold:
        target_contrast_ratio = max(3.0, target_contrast_ratio * 0.67)

    # 1. Background analysis
    r_bg, g_bg, b_bg = patch_pixels.mean(axis=(0, 1))
    l_bg: float = compute_relative_luminance(r_bg, g_bg, b_bg)

    # Also analyze variance for additional context
    r_var, g_var, b_var = patch_pixels.var(axis=(0, 1))
    color_variance: float = (r_var + g_var + b_var) / 3.0

    # 2. Generate contrast color candidates.
    candidates: list[tuple[float, float, float]] = generate_contrast_color_candidates(r_bg, g_bg, b_bg, l_bg)

    # 3. Score candidates by alpha, contrast, hue distance, and saturation.
    best_color: tuple[float, float, float] | None = None
    best_alpha: float = max_alpha + 1
    best_score: float = -1.0

    for r_ov, g_ov, b_ov in candidates:
        min_alpha_needed, contrast_achieved = find_min_alpha_for_contrast(r_ov, g_ov, b_ov, target_contrast_ratio, l_bg)

        if min_alpha_needed is not None and min_alpha_needed <= max_alpha:
            # Multi-criteria scoring
            h_ov, s_ov, _v_ov = rgb_to_hsv(r_ov, g_ov, b_ov)
            h_bg, _s_bg, _v_bg = rgb_to_hsv(r_bg, g_bg, b_bg)

            # Score components
            alpha_score = (max_alpha - min_alpha_needed) / max_alpha  # Prefer lower alpha
            contrast_score = min(contrast_achieved / target_contrast_ratio, 2.0) / 2.0  # Bonus for exceeding target

            # Color harmony score (prefer complementary relationships)
            hue_diff = abs(h_ov - h_bg)
            hue_diff = min(hue_diff, 1.0 - hue_diff)  # Circular distance
            harmony_score = hue_diff  # Prefer colors that are different in hue

            # Saturation appropriateness
            sat_score = s_ov if color_variance > 0.01 else 1.0 - abs(s_ov - 0.7)

            # Combined score
            score = alpha_score * 0.4 + contrast_score * 0.3 + harmony_score * 0.2 + sat_score * 0.1

            if score > best_score:
                best_score = score
                best_color = (r_ov, g_ov, b_ov)
                best_alpha = min_alpha_needed

    # 4. Fallback if no color meets requirements
    if best_color is None:
        # Fall back to luminance inversion.
        # White for dark backgrounds, black for light backgrounds
        best_color = (1.0, 1.0, 1.0) if l_bg < 0.5 else (0.0, 0.0, 0.0)
        best_alpha = max_alpha

    # 5. Apply bounds and area-based adjustments
    chosen_alpha: float = np.clip(best_alpha, min_alpha, max_alpha)

    # Area-based adjustments
    if area < 100:
        chosen_alpha = min(max_alpha, chosen_alpha * 1.2)  # Slightly more opaque for tiny areas
    elif area > large_area_threshold:
        chosen_alpha = max(min_alpha, chosen_alpha * 0.9)  # Slightly less opaque for large areas

    # Convert to BGR for OpenCV
    b_val: int = np.clip(best_color[2] * 255, 0, 255, dtype=int)
    g_val: int = np.clip(best_color[1] * 255, 0, 255, dtype=int)
    r_val: int = np.clip(best_color[0] * 255, 0, 255, dtype=int)
    fill_color_bgr: tuple[int, int, int] = (b_val, g_val, r_val)

    return fill_color_bgr, chosen_alpha


def extract_image_array(img: Figure | Axes | MatLike) -> NDArray[float32]:
    """Extract numpy array from various image sources.

    Supports:
    - numpy arrays (float [0,1] or uint8 [0,255])
    - matplotlib Figure objects
    - OpenCV images (BGR uint8)

    Returns:
        NDArray[np.float32]: Array with values in [0,1] and shape (H, W, 3)
    """
    # NumPy array input (covers plain NumPy and OpenCV image arrays).
    if isinstance(img, np.ndarray):
        arr: NDArray[np_number] = img

        # Normalize to float32 in [0, 1]
        arr_f32: NDArray[float32]
        if np.issubdtype(arr.dtype, np.floating):  # floats (possibly already [0,1])
            arr_f32 = arr.astype(float32, copy=False)
            # If the dynamic range looks like [0,255], scale to [0,1]
            if float(np.max(arr_f32)) > 1.0:
                arr_f32 = (arr_f32 / 255.0).astype(float32, copy=False)
        else:
            # Integers -> float32 in [0,1]
            arr_f32 = arr.astype(float32, copy=False)
            arr_f32 *= 1.0 / 255.0

        # Replicate grayscale input across three channels.
        if arr_f32.ndim == 2:
            # Grayscale to RGB
            arr_f32 = np.repeat(arr_f32[:, :, None], 3, axis=2)
        elif arr_f32.ndim == 3:
            channels: int = arr_f32.shape[2]
            if channels == 1:
                arr_f32 = np.repeat(arr_f32, 3, axis=2)
            elif channels == 4:
                # RGBA -> RGB (drop alpha)
                arr_f32 = arr_f32[:, :, :3]

        return arr_f32.astype(float32, copy=False)

    # # Matplotlib Figure or Axes -> render to PNG buffer, decode with OpenCV.
    # if isinstance(img, (Figure, Axes)):

    # `img` is either a Figure or an Axes at this point.
    fig: Figure | None = img.get_figure(root=True) if isinstance(img, Axes) else img

    if fig is None:
        raise ValueError("Cannot extract image: `img.get_figure(root=True)` returned `None`.")

    # Save figure into an in-memory PNG
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
        buf.seek(0)
        # Convert buffer -> uint8 array with zero-copy memoryview when possible.
        png_bytes: NDArray[uint8] = np.frombuffer(buf.getbuffer(), dtype=uint8)

    # Decode PNG (BGR order for color images per OpenCV) and convert to RGB.
    img_bgr_opt: MatLike | None = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)
    if img_bgr_opt is None:
        raise ValueError("Could not decode figure PNG buffer.")

    img_rgb: MatLike = cv2.cvtColor(img_bgr_opt, cv2.COLOR_BGR2RGB)

    return img_rgb.astype(float32, copy=False) / 255.0


def derive_highlight_geometry(
    height_px: int, width_px: int, dpi: float = 300.0, viewing_distance_in: float = 24.0
) -> dict[str, int | tuple[int, int, int]]:
    """Return a dict of parameter values tuned so that the function catches only human-visible changes.

    Args:
        height_px (int): Image height in pixels.
        width_px (int): Image width in pixels.
        dpi (float): default 96 Display dots-per-inch (use the same dpi you pass to Matplotlib).
        viewing_distance_in (float): default 24. Typical desktop eye-to-screen distance in inches.
        base_color(tuple(int,int,int)): default green. BGR colour for the annotation graphics.

    Returns:
        dict with keys: min_area, kernel_size, morph_iterations, circle_color, circle_thickness
    """
    # Perceptual geometry
    ppd: float = dpi * viewing_distance_in * math.pi / 180.0  # pixels/degree
    r_visible: int = max(1, int(ppd * 0.05))  # 0.05 Degrees ~ 3 arc-min

    # Parameter derivation
    min_area: int = int(math.pi * r_visible**2)  # pixels**2
    kernel_size: int = max(3, (r_visible // 2) * 2 + 1)  # Odd integer
    morph_iterations: int = 1 if min(height_px, width_px) < 1000 else 2

    diag: float = math.hypot(height_px, width_px)
    circle_thickness: int = max(1, int(0.006 * diag))  # ~0.6 % of diag

    return {
        "min_area": min_area,
        "kernel_size": kernel_size,
        "morph_iterations": morph_iterations,
        "circle_thickness": circle_thickness,
    }


def highlight_differences(
    img1: NDArray[float32] | NDArray[np.float64] | NDArray[uint8] | Figure | Axes,
    img2: NDArray[float32] | NDArray[np.float64] | NDArray[uint8] | Figure | Axes,
    *,
    min_area: int = 50,
    kernel_size: int = 5,
    morph_iterations: int = 2,
    circle_color: tuple[int, int, int] = (0, 255, 0),
    circle_thickness: int = 2,
    highlight_mode: str = "both",
    fill_color: tuple[int, int, int] = (0, 150, 255),
    alpha: float = 0.15,
    contour_order: str = "weighted",
) -> tuple[NDArray[float32], NDArray[float32]]:
    """Detects regions of substantial pixel-wise difference between two images.

    Supports numpy arrays, matplotlib Figure objects, and OpenCV images.
    Draws circles around each differing region.

    Args:
        img1: First input image. Can be numpy array, matplotlib Figure, or OpenCV image.
        img2: Second input image. Same type constraints as img1.
        min_area: Ignore any detected contour whose area (in pixels) is smaller than this.
            Larger values filter out small specks and noise. Smaller values detect finer
            details.
        kernel_size: Side length of the square structuring element used for morphological
            closing (must be odd). Larger kernels merge and smooth broader regions,
            smaller kernels preserve detail but may leave gaps.
        morph_iterations: Number of times to apply closing (dilate->erode). Higher values fill
            larger holes and further merge adjacent regions. Zero disables cleaning.
        circle_color: BGR color used to draw each enclosing circle. Change for different
            annotation hues (e.g. red=(0,0,255), blue=(255,0,0)).
        circle_thickness: Line thickness (in pixels) for the circle border. Increase for more
            visible outlines or decrease for subtle marks.
        highlight_mode: Which image(s) to highlight. Options: "both", "first", "second".
        fill_color: BGR color tuple for circle fill (used with alpha blending).
        alpha: Opacity for the fill color (0.0 = transparent, 1.0 = opaque).
        contour_order: Contour ordering strategy. ``weighted`` combines area,
            pixel difference, center distance, signal variation, compactness,
            and solidity. ``area`` sorts only by contour area.

    Returns:
        Tuple of annotated images as float32 arrays with values in [0,1] and shape (H,W,3).
    """
    # Convert inputs to standardized numpy arrays (float32 [0,1])
    img1_array: NDArray[float32] = np.clip(extract_image_array(img1), 0.0, 1.0)
    img2_array: NDArray[float32] = np.clip(extract_image_array(img2), 0.0, 1.0)

    # Validate inputs
    if img1_array.shape != img2_array.shape:
        raise ValueError("Input images must have the same shape.")

    if highlight_mode not in {"both", "first", "second"}:
        raise ValueError("highlight_mode must be 'both', 'first', or 'second'.")

    if contour_order not in {"weighted", "area"}:
        raise ValueError("contour_order must be 'weighted' or 'area'.")

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    if morph_iterations < 0:
        raise ValueError("morph_iterations must be non-negative.")

    # Convert to uint8 for OpenCV operations (single conversion)
    img1_uint8: NDArray[uint8] = (img1_array * 255).astype(uint8)
    img2_uint8: NDArray[uint8] = (img2_array * 255).astype(uint8)

    # Compute absolute difference
    diff: MatLike = cv2.absdiff(img1_uint8, img2_uint8)

    # Convert to grayscale for thresholding
    gray: MatLike = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Automatic Otsu thresholding
    mask: MatLike = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Apply morphological cleaning if requested
    if morph_iterations > 0:
        kernel: MatLike = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Build list of (contour, area), filter by min_area
    valid_contours: list[MatLike] = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    circles: list[tuple[MatLike, float]]  # List of (contour, area)
    if contour_order == "weighted":
        # Compute the weighted score for each contour.
        scored_circles: list[tuple[MatLike, float, float]] = []
        img_height, img_width = img1_uint8.shape[:2]

        area: float
        for contour in valid_contours:
            area = cv2.contourArea(contour)

            # Get bounding box for extracting patches
            x, y, w, h = cv2.boundingRect(contour)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_width, x + w), min(img_height, y + h)

            # Extract patches from both images and difference
            img1_patch: NDArray[uint8] = img1_uint8[y1:y2, x1:x2]
            img2_patch: NDArray[uint8] = img2_uint8[y1:y2, x1:x2]
            diff_patch: MatLike = diff[y1:y2, x1:x2]

            # 1. Combine area, intensity, and center distance.
            total_pixels: int = img_height * img_width
            area_factor: float = min(area / total_pixels, 0.1) * 10.0  # Normalize, cap at 10% of image

            intensity_factor: float32 | float = np.mean(diff_patch) / 255.0 if diff_patch.size > 0 else 0.0

            # Position factor (center bias)
            moments: Moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                center_x: int = img_width // 2
                center_y: int = img_height // 2
                distance_from_center: float = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
                max_distance: float = np.sqrt(center_x**2 + center_y**2)
                position_factor: float = 1.0 - (distance_from_center / max_distance)
            else:
                position_factor = 0.5

            location_difference_score: float32 | float = (
                0.4 * area_factor + 0.4 * intensity_factor + 0.2 * position_factor
            )

            # 2. Scale the mean-to-standard-deviation ratio by contour area.
            signal_variation_score: float32 | float
            if img1_patch.size > 0 and img2_patch.size > 0:
                patch_diff: NDArray[float32] = np.abs(img1_patch.astype(float32) - img2_patch.astype(float32))
                mean_diff: float32 = np.mean(patch_diff)
                std_diff: float32 = np.std(patch_diff)
                snr: float32 | float = mean_diff / (std_diff + 1e-8)
                signal_variation_score = snr * np.sqrt(area) / 1000.0  # Normalize
            else:
                signal_variation_score = 0.0

            # 3. Shape quality (compactness and solidity)
            perimeter: float = cv2.arcLength(contour, True)
            shape_score: float
            if perimeter > 0:
                compactness: float = 4 * np.pi * area / (perimeter**2)
                hull: MatLike = cv2.convexHull(contour)
                hull_area: float = cv2.contourArea(hull)
                solidity: float = area / (hull_area + 1e-8)
                shape_score = 0.6 * compactness + 0.4 * solidity
            else:
                shape_score = 0.0

            # Combine the three score groups.
            weighted_score: float | float32 = (
                0.5 * location_difference_score + 0.35 * signal_variation_score + 0.25 * shape_score
            )

            scored_circles.append((contour, area, float(weighted_score)))

        # Sort by weighted score in descending order.
        scored_circles.sort(key=operator.itemgetter(2), reverse=True)

        # Extract contours and areas for compatibility with existing code
        circles = [(cnt, area) for cnt, area, _ in scored_circles]

    elif contour_order == "area":
        # Sort contours by area alone.
        circles = [(cnt, cv2.contourArea(cnt)) for cnt in valid_contours]
        circles.sort(key=operator.itemgetter(1), reverse=True)

    else:
        raise ValueError(f"Unsupported contour ordering method: {contour_order}")

    # Initialize output images
    out1: NDArray[uint8] | MatLike = img1_uint8.copy()
    out2: NDArray[uint8] | MatLike = img2_uint8.copy()

    # Create overlays for alpha blending
    need_overlay1: bool = highlight_mode in {"both", "first"} and alpha > 0
    need_overlay2: bool = highlight_mode in {"both", "second"} and alpha > 0

    overlay1: MatLike | None = out1.copy() if need_overlay1 else None
    overlay2: MatLike | None = out2.copy() if need_overlay2 else None

    # Keep a list of circles we actually draw (center_x, center_y, radius)
    accepted_circles: list[tuple[int, int, int]] = []

    # Process each contour
    for contour, _ in circles:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        # Get minimum enclosing circle
        (x_circle, y_circle), radius = cv2.minEnclosingCircle(contour)
        cx, cy, r = int(x_circle), int(y_circle), int(radius)

        # Skip if overlaps any accepted circle
        should_skip = False
        for ax, ay, ar in accepted_circles:
            # Euclidean distance between centers
            dist: float = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5
            if dist < (r + ar):
                should_skip = True
                break

        if should_skip:
            continue

        # Nothing overlapped: accept this circle
        accepted_circles.append((cx, cy, r))

        # Draw on first image
        if highlight_mode in {"both", "first"}:
            if overlay1 is not None:
                cv2.circle(overlay1, (cx, cy), r, fill_color, -1)

            cv2.circle(out1, (cx, cy), r, circle_color, circle_thickness)

        # Draw on second image
        if highlight_mode in {"both", "second"}:
            if overlay2 is not None:
                cv2.circle(overlay2, (cx, cy), r, fill_color, -1)

            cv2.circle(out2, (cx, cy), r, circle_color, circle_thickness)

    # Apply alpha blending for fill effects
    if overlay1 is not None:
        out1 = cv2.addWeighted(out1, 1 - alpha, overlay1, alpha, 0)
    if overlay2 is not None:
        out2 = cv2.addWeighted(out2, 1 - alpha, overlay2, alpha, 0)

    # Convert back to float [0,1]
    return out1.astype(float32) / 255.0, out2.astype(float32) / 255.0


def highlight_differences_with_contrast_fill(
    img1: NDArray[float32] | NDArray[np.float64] | NDArray[uint8] | Figure | Axes,
    img2: NDArray[float32] | NDArray[np.float64] | NDArray[uint8] | Figure | Axes,
    *,
    min_area: int = 50,
    kernel_size: int = 5,
    morph_iterations: int = 2,
    circle_thickness: int = 2,
    highlight_mode: str = "both",
    target_contrast_ratio: float = 0.001,
    min_alpha: float = 0.2,
    max_alpha: float = 0.35,
    use_contrast_fill: bool = True,
    fallback_fill_color: tuple[int, int, int] = (0, 150, 255),
    fallback_alpha: float = 0.15,
    contour_order: str = "weighted",
) -> tuple[NDArray[float32], NDArray[float32]]:
    """Highlight pixel-difference contours with per-region contrast fills.

    When contrast-based fill selection is enabled, the background below each
    enclosing circle determines its fill color and opacity.

    Args:
        img1: First input image. Can be numpy array, matplotlib Figure, or OpenCV image.
        img2: Second input image. Same type constraints as img1.
        min_area: Ignore any detected contour whose area (in pixels) is smaller than this.
        kernel_size: Side length of the square structuring element used for morphological
            closing (must be odd).
        morph_iterations: Number of times to apply closing (dilate -> erode).
        circle_thickness: Line thickness (in pixels) for the circle border.
        highlight_mode: Which image(s) to highlight. Options: "both", "first", "second".
        target_contrast_ratio: Minimum WCAG contrast ratio for visibility.
            - Higher values (e.g., 4.5 or above) force the function to select fill colors
              that are more visually distinct from the background, resulting in stronger,
              more noticeable highlights.
            - Lower values (e.g., 2.0 or 3.0) allow the function to choose colors that may
              be closer in luminance to the background, producing more subtle highlights
              that blend in more.
            - If the background is very light or dark, a high contrast ratio will push the
              algorithm to select a fill color that is as different as possible in brightness,
              while a low ratio may result in a fill color that is less contrasting.
        min_alpha: Minimum allowed opacity for the selected fill.
        max_alpha: Maximum allowed opacity to avoid over-occlusion.
        use_contrast_fill: If True, select a fill from each region's background
            luminance. If False, use ``fallback_fill_color``.
        fallback_fill_color: BGR color used when contrast-based selection is disabled.
        fallback_alpha: Alpha value used when contrast-based selection is disabled.
        contour_order: Contour ordering strategy. ``weighted`` combines area,
            pixel difference, center distance, signal variation, compactness,
            and solidity. ``area`` sorts only by contour area.

    Returns:
        Tuple of annotated images as float32 arrays with values in [0,1] and shape (H,W,3).
    """
    # Convert inputs to standardized numpy arrays
    img1_array: NDArray[float32] = np.clip(extract_image_array(img1), 0.0, 1.0)
    img2_array: NDArray[float32] = np.clip(extract_image_array(img2), 0.0, 1.0)

    # Validate inputs
    if img1_array.shape != img2_array.shape:
        raise ValueError("Input images must have the same shape.")

    if highlight_mode not in {"both", "first", "second"}:
        raise ValueError("highlight_mode must be 'both', 'first', or 'second'.")

    if contour_order not in {"weighted", "area"}:
        raise ValueError("contour_order must be 'weighted' or 'area'.")

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    if morph_iterations < 0:
        raise ValueError("morph_iterations must be non-negative.")

    # Convert to uint8 for OpenCV operations
    img1_uint8: NDArray[uint8] = (img1_array * 255).astype(uint8)
    img2_uint8: NDArray[uint8] = (img2_array * 255).astype(uint8)

    # Compute absolute difference
    diff: MatLike = cv2.absdiff(img1_uint8, img2_uint8)

    # Convert to grayscale for thresholding
    gray: MatLike = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Automatic Otsu thresholding
    mask: MatLike = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Apply morphological cleaning if requested
    if morph_iterations > 0:
        kernel: MatLike = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Find contours
    contours: Sequence[MatLike] = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Build list of (contour, area), filter by min_area
    valid_contours: list[MatLike] = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    circles: list[tuple[MatLike, float]]  # List of (contour, area)
    if contour_order == "weighted":
        # Compute the weighted score for each contour.
        scored_circles: list[tuple[MatLike, float, float]] = []
        img_height, img_width = img1_uint8.shape[:2]

        for contour in valid_contours:
            area: float = cv2.contourArea(contour)

            # Get bounding box for extracting patches
            x, y, w, h = cv2.boundingRect(contour)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_width, x + w), min(img_height, y + h)

            # Extract patches from both images and difference
            img1_patch: NDArray[uint8] = img1_uint8[y1:y2, x1:x2]
            img2_patch: NDArray[uint8] = img2_uint8[y1:y2, x1:x2]
            diff_patch: MatLike = diff[y1:y2, x1:x2]

            # 1. Combine area, intensity, and center distance.
            total_pixels: float = img_height * img_width
            area_factor: float = min(area / total_pixels, 0.1) * 10  # Normalize, cap at 10% of image

            intensity_factor: float | float32 = np.mean(diff_patch) / 255.0 if diff_patch.size > 0 else 0.0

            # Position factor (center bias)
            position_factor: float
            moments: Moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                center_x, center_y = img_width // 2, img_height // 2
                distance_from_center: float = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
                max_distance: float = np.sqrt(center_x**2 + center_y**2)
                position_factor = 1.0 - (distance_from_center / max_distance)

            else:
                position_factor = 0.5

            location_difference_score: float | float32 = (
                0.4 * area_factor + 0.4 * intensity_factor + 0.2 * position_factor
            )

            # 2. Scale the mean-to-standard-deviation ratio by contour area.
            if img1_patch.size > 0 and img2_patch.size > 0:
                patch_diff: NDArray[float32] = np.abs(img1_patch.astype(float32) - img2_patch.astype(float32))
                mean_diff: float32 = np.mean(patch_diff)
                std_diff: float32 = np.std(patch_diff)
                snr: float | float32 = mean_diff / (std_diff + 1e-8)
                signal_variation_score: float = snr * np.sqrt(area) / 1000.0  # Normalize

            else:
                signal_variation_score = 0.0

            # 3. Shape quality (compactness and solidity)
            shape_score: float
            perimeter: float = cv2.arcLength(contour, True)
            if perimeter > 0:
                compactness: float = 4 * np.pi * area / (perimeter**2)
                hull: MatLike = cv2.convexHull(contour)
                hull_area: float = cv2.contourArea(hull)
                solidity: float = area / (hull_area + 1e-8)
                shape_score = 0.6 * compactness + 0.4 * solidity

            else:
                shape_score = 0.0

            # Combine the three score groups.
            weighted_score: float | float32 = (
                0.5 * location_difference_score + 0.3 * signal_variation_score + 0.2 * shape_score
            )

            scored_circles.append((contour, area, float(weighted_score)))

        # Sort by weighted score in descending order.
        scored_circles.sort(key=operator.itemgetter(2), reverse=True)

        # Extract contours and areas for compatibility with existing code
        circles = [(cnt, area) for cnt, area, _ in scored_circles]

    elif contour_order == "area":
        # Sort contours by area alone.
        circles = [(cnt, cv2.contourArea(cnt)) for cnt in valid_contours]
        circles.sort(key=operator.itemgetter(1), reverse=True)

    else:
        raise ValueError(f"Unsupported contour ordering method: {contour_order}")

    # Initialize output images
    out1: NDArray[uint8] | MatLike = img1_uint8.copy()
    out2: NDArray[uint8] | MatLike = img2_uint8.copy()

    # Track accepted circles to avoid overlaps
    accepted_circles: list[tuple[int, int, int]] = []

    # Select a fill from the background below each contour.
    for contour, area in circles:
        if area < min_area:
            continue

        # Get minimum enclosing circle
        (x_circle, y_circle), radius = cv2.minEnclosingCircle(contour)
        cx, cy, r = int(x_circle), int(y_circle), int(radius)

        # Skip if overlaps any accepted circle
        should_skip = False
        for ax, ay, ar in accepted_circles:
            dist: float = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5
            if dist < (r + ar):
                should_skip = True
                break

        if should_skip:
            continue

        # Accept this circle
        accepted_circles.append((cx, cy, r))

        # Determine fill color and alpha
        if use_contrast_fill:
            # Extract patch from the background image for color analysis
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(img1_array.shape[1], cx + r), min(img1_array.shape[0], cy + r)

            # Use the average of both images for background analysis
            patch1: NDArray[uint8] | MatLike = img1_array[y1:y2, x1:x2]
            patch2: NDArray[uint8] | MatLike = img2_array[y1:y2, x1:x2]

            if patch1.size > 0 and patch2.size > 0:
                avg_patch = (patch1 + patch2) / 2.0
                fill_color, alpha = select_contrast_fill(
                    avg_patch,
                    target_contrast_ratio=target_contrast_ratio,
                    min_alpha=min_alpha,
                    max_alpha=max_alpha,
                    large_area_threshold=min_area * 20,  # Scale threshold based on min_area
                )
            else:
                fill_color, alpha = fallback_fill_color, fallback_alpha
        else:
            fill_color, alpha = fallback_fill_color, fallback_alpha

        # Create border color (same as fill but full opacity for border visibility)
        border_color: tuple[int, int, int] = fill_color

        # Draw on appropriate images with double-alpha borders
        border_alpha: float = min(1.0, 3 * alpha)

        if highlight_mode in {"both", "first"}:
            # Fill overlay
            overlay1: NDArray[uint8] | MatLike = out1.copy()
            cv2.circle(overlay1, (cx, cy), r, fill_color, -1)
            out1 = cv2.addWeighted(out1, 1 - alpha, overlay1, alpha, 0)

            # Border overlay with increased opacity
            border_overlay1: NDArray[uint8] | MatLike = out1.copy()
            cv2.circle(border_overlay1, (cx, cy), r, border_color, circle_thickness)
            out1 = cv2.addWeighted(out1, 1 - border_alpha, border_overlay1, border_alpha, 0)

        if highlight_mode in {"both", "second"}:
            # Fill overlay
            overlay2: NDArray[uint8] | MatLike = out2.copy()
            cv2.circle(overlay2, (cx, cy), r, fill_color, -1)
            out2 = cv2.addWeighted(out2, 1 - alpha, overlay2, alpha, 0)

            # Border overlay with increased opacity
            border_overlay2: NDArray[uint8] | MatLike = out2.copy()
            cv2.circle(border_overlay2, (cx, cy), r, border_color, circle_thickness)
            out2 = cv2.addWeighted(out2, 1 - border_alpha, border_overlay2, border_alpha, 0)

    # Convert back to float [0,1]
    return out1.astype(float32) / 255.0, out2.astype(float32) / 255.0
