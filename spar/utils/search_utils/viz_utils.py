from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import imageio.v3 as iio
import numpy as np
from numpy import float32
import torch

from spar.utils.search_utils.nnet_utils import load_nnet

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from cv2.typing import MatLike
    from numpy.typing import NDArray
    from torch import nn

    from spar.environments.abstracts.environment import ABCEnvironment
    from spar.environments.abstracts.state import ABCState
    from spar.utils.config_utils.config_schema import ModelConfig


class ImageHandler:
    """Unified visualization utilities for search."""

    # Typography and spacing constants using 8pt grid system
    BASE_UNIT = 8  # Base spacing unit in pixels (8pt grid)
    GOLDEN_RATIO = 1.618  # Golden ratio for aesthetically pleasing proportions

    # Typography scale (based on modular scale with 1.25 ratio)
    FONT_SCALE_TINY = 0.4
    FONT_SCALE_SMALL = 0.5
    FONT_SCALE_BODY = 0.65
    FONT_SCALE_LARGE = 0.8
    FONT_SCALE_TITLE = 1.0
    FONT_SCALE_HEADING = 1.25

    # Spacing scale (multiples of BASE_UNIT)
    SPACING_TIGHT = BASE_UNIT * 1  # 8px
    SPACING_NORMAL = BASE_UNIT * 2  # 16px
    SPACING_RELAXED = BASE_UNIT * 3  # 24px
    SPACING_LOOSE = BASE_UNIT * 4  # 32px

    # Minimum sizes for legibility
    MIN_TEXT_HEIGHT = 14
    MIN_MARGIN = 8
    MIN_THUMBNAIL_SIZE = 64

    # palette and layout tokens
    COLOR_BACKGROUND: tuple[float, float, float] = (0.956, 0.964, 0.980)
    COLOR_SURFACE: tuple[float, float, float] = (0.980, 0.983, 0.992)
    COLOR_SURFACE_ALT: tuple[float, float, float] = (0.925, 0.945, 0.980)
    COLOR_ACCENT: tuple[float, float, float] = (0.278, 0.407, 0.780)
    COLOR_ACCENT_SOFT: tuple[float, float, float] = (0.643, 0.752, 0.925)
    COLOR_TEXT_PRIMARY: tuple[float, float, float] = (0.145, 0.165, 0.210)
    COLOR_TEXT_SECONDARY: tuple[float, float, float] = (0.350, 0.365, 0.410)
    COLOR_SHADOW: tuple[float, float, float] = (0.110, 0.130, 0.180)

    CARD_PADDING = BASE_UNIT * 3  # 24px internal padding
    CARD_SHADOW_OFFSET: tuple[int, int] = (BASE_UNIT, BASE_UNIT)  # 8px offset for drop shadow
    CARD_SHADOW_BLUR = BASE_UNIT * 5 + 1  # Gaussian blur requires an odd kernel.
    STORYBOARD_PADDING = BASE_UNIT * 4  # outer padding for full canvas
    SECTION_SPACING = BASE_UNIT * 3  # gap between stacked sections

    def __init__(
        self,
        env: ABCEnvironment[ABCState],
        state_start: ABCState,
        decoder: nn.Module,
        path: list[NDArray[float32]],
        len_soln: int,
        state_idx: int,
        device: torch.device,
    ) -> None:
        # Instance configured for a single solution path
        self.env: ABCEnvironment[ABCState] = env
        self.state_start: ABCState = state_start
        self.decoder: nn.Module = decoder
        self.path: list[NDArray[float32]] = path
        self.len_soln: int = len_soln
        self.state_idx: int = state_idx
        self.device: torch.device = device

        # Step tracking
        self.moves_taken: list[int] = []

        # Optional optimal path
        self.optimal_soln: list[int] | None = None
        if hasattr(self.state_start, "get_solution"):
            try:
                sol: list[int] | None = self.state_start.get_solution()
                if isinstance(sol, (list, tuple)):
                    self.optimal_soln = list(sol)
            except Exception:
                self.optimal_soln = None

    def step(self, move: int) -> None:
        """Record a single move taken by the agent.

        Args:
            move: Action index taken at the current step.
        """
        self.moves_taken.append(move)

    @staticmethod
    def _step_texts(moves: list[int]) -> list[str]:
        texts: list[str] = ["Step 0 - Move: None"]
        texts.extend([f"Step {i + 1} - Move: {m}" for i, m in enumerate(moves)])
        return texts

    @staticmethod
    def _compute_text_position(img_h: int, img_w: int, is_title: bool = False) -> tuple[int, int]:
        """Compute a text baseline from the image dimensions.

        Args:
            img_h: Image height in pixels.
            img_w: Image width in pixels.
            is_title: Whether to use the title margin.

        Returns:
            Pixel coordinates of the text baseline.
        """
        # Align the horizontal margin to BASE_UNIT.
        margin_x: int = max(
            ImageHandler.SPACING_NORMAL, ImageHandler.BASE_UNIT * round(0.02 * img_w / ImageHandler.BASE_UNIT)
        )

        # Titles use a smaller vertical inset than body text.
        margin_y_ratio: float = 0.08 if is_title else 0.12

        pos_y: int = max(ImageHandler.MIN_TEXT_HEIGHT + ImageHandler.SPACING_TIGHT, round(margin_y_ratio * img_h))

        return (margin_x, pos_y)

    @staticmethod
    def _compute_font_scale(img_h: int, text_type: str = "body") -> tuple[float, int]:
        """Compute font scale and stroke width from image height.

        Args:
            img_h: Image height in pixels.
            text_type: One of ``tiny``, ``small``, ``body``, ``large``,
                ``title``, or ``heading``.

        Returns:
            OpenCV font scale and stroke width.
        """
        # Base font scale on image height with typography scale
        base_scale: float = max(0.3, min(1.8, (img_h / 400.0)))

        # Apply typography scale multipliers
        scale_multipliers = {
            "tiny": ImageHandler.FONT_SCALE_TINY,
            "small": ImageHandler.FONT_SCALE_SMALL,
            "body": ImageHandler.FONT_SCALE_BODY,
            "large": ImageHandler.FONT_SCALE_LARGE,
            "title": ImageHandler.FONT_SCALE_TITLE,
            "heading": ImageHandler.FONT_SCALE_HEADING,
        }

        multiplier: float = scale_multipliers.get(text_type, ImageHandler.FONT_SCALE_BODY)
        font_scale: float = base_scale * multiplier

        # Thickness proportional to font scale with minimum for legibility
        thickness: int = max(1, round(font_scale * 1.5))

        return (font_scale, thickness)

    @staticmethod
    def _get_text_size(text: str, font_scale: float, thickness: int) -> tuple[int, int]:
        """Get the size of rendered text.

        Args:
            text: Text string to measure
            font_scale: OpenCV font scale
            thickness: Line thickness

        Returns:
            (width, height) of the text bounding box
        """
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        return (text_w, text_h + baseline)

    @staticmethod
    def _color_to_uint8(color: tuple[float, float, float]) -> tuple[int, int, int]:
        """Convert an RGB color expressed in [0,1] floats to uint8 tuple."""
        r = int(np.clip(color[0], 0.0, 1.0) * 255.0)
        g = int(np.clip(color[1], 0.0, 1.0) * 255.0)
        b = int(np.clip(color[2], 0.0, 1.0) * 255.0)
        return (r, g, b)

    @staticmethod
    def _ensure_multiple(value: int, minimum_units: int = 1) -> int:
        """Snap ``value`` to the closest multiple of ``BASE_UNIT`` with a floor."""
        return ImageHandler.BASE_UNIT * max(minimum_units, round(value / ImageHandler.BASE_UNIT))

    @staticmethod
    def _pad_to_width(img: NDArray[float32], width: int, color: tuple[float, float, float]) -> NDArray[float32]:
        """Pad an image symmetrically to the requested width using the provided color."""
        if img.shape[1] >= width:
            return img
        diff: int = width - img.shape[1]
        left: int = diff // 2
        canvas: NDArray[float32] = np.empty((img.shape[0], width, 3), dtype=float32)
        canvas[:, :, 0] = color[0]
        canvas[:, :, 1] = color[1]
        canvas[:, :, 2] = color[2]
        canvas[:, left : left + img.shape[1]] = img
        return canvas

    @staticmethod
    def _wrap_with_card(
        content: NDArray[float32],
        *,
        background: tuple[float, float, float] | None = None,
        padding: int | None = None,
        add_shadow: bool = True,
    ) -> NDArray[float32]:
        """Wrap content inside a soft card with padding and optional drop shadow."""
        if content.ndim != 3:
            return content

        padding = padding if padding is not None else ImageHandler.CARD_PADDING
        base_bg: tuple[float, float, float] = background if background is not None else ImageHandler.COLOR_SURFACE

        pad_h: int = max(0, padding)
        pad_w: int = max(0, padding)
        card_h: int = content.shape[0] + pad_h * 2
        card_w: int = content.shape[1] + pad_w * 2

        card: NDArray[float32] = np.empty((card_h, card_w, 3), dtype=float32)
        card[:, :, 0] = base_bg[0]
        card[:, :, 1] = base_bg[1]
        card[:, :, 2] = base_bg[2]
        card[pad_h : pad_h + content.shape[0], pad_w : pad_w + content.shape[1]] = content

        if not add_shadow:
            return card

        offset_x: int
        offset_y: int
        offset_x, offset_y = ImageHandler.CARD_SHADOW_OFFSET
        offset_x = max(0, offset_x)
        offset_y = max(0, offset_y)

        board_h: int = card_h + offset_y
        board_w: int = card_w + offset_x
        board: NDArray[float32] = np.empty((board_h, board_w, 3), dtype=float32)
        board[:, :, 0] = ImageHandler.COLOR_BACKGROUND[0]
        board[:, :, 1] = ImageHandler.COLOR_BACKGROUND[1]
        board[:, :, 2] = ImageHandler.COLOR_BACKGROUND[2]

        mask: NDArray[float32] = np.zeros((board_h, board_w), dtype=float32)
        mask[offset_y : offset_y + card_h, offset_x : offset_x + card_w] = 1.0

        blur_kernel: int = ImageHandler.CARD_SHADOW_BLUR
        blur_kernel = max(3, blur_kernel | 1)
        shadow_arr: MatLike = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), sigmaX=0, sigmaY=0)
        shadow: NDArray[float32] = np.asarray(shadow_arr, dtype=float32)
        shadow_max: float = float(np.max(shadow))
        if shadow_max > 0:
            shadow /= shadow_max
        shadow_strength: float = 0.18
        shadow_color: NDArray[float32] = np.array(ImageHandler.COLOR_SHADOW, dtype=float32)
        blend_factor: NDArray[float32] = shadow[:, :, None] * shadow_strength
        one: float32 = np.float32(1.0)
        board *= one - blend_factor
        board += shadow_color * blend_factor
        board = np.clip(board, 0.0, 1.0, out=board)

        board[0:card_h, 0:card_w] = card
        return board

    @staticmethod
    def _compose_storyboard(
        panels: list[NDArray[float32]], *, title: str | None = None, subtitle: str | None = None
    ) -> NDArray[float32]:
        """Stack cards vertically with generous spacing on a branded background."""
        if len(panels) == 0:
            return np.ones((ImageHandler.BASE_UNIT * 4, ImageHandler.BASE_UNIT * 4, 3), dtype=float32)

        # Measure panels before creating the optional title banner.
        initial_width: int = max(panel.shape[1] for panel in panels)

        # Create the banner before finalizing the shared width.
        banner_card: NDArray[float32] | None = None
        if title is not None:
            banner_inner_w: int = initial_width - ImageHandler.CARD_PADDING * 2
            banner_inner_w = max(ImageHandler.BASE_UNIT * 8, banner_inner_w)
            banner: NDArray[float32] = ImageHandler._create_text_banner(
                banner_inner_w,
                title,
                text_type="heading",
                reference_height=max(panel.shape[0] for panel in panels),
                subtitle=subtitle,
            )
            banner_card = ImageHandler._wrap_with_card(banner, background=ImageHandler.COLOR_SURFACE_ALT)

        # Accommodate both panels and the banner.
        final_width: int = initial_width
        if banner_card is not None:
            final_width = max(final_width, banner_card.shape[1])

        # Pad panels and construct spacer using the final width
        padded_panels: list[NDArray[float32]] = [
            ImageHandler._pad_to_width(panel, final_width, ImageHandler.COLOR_BACKGROUND) for panel in panels
        ]

        spacer_h: int = ImageHandler.SECTION_SPACING
        spacer: NDArray[float32] = np.empty((spacer_h, final_width, 3), dtype=float32)
        spacer[:, :, 0] = ImageHandler.COLOR_BACKGROUND[0]
        spacer[:, :, 1] = ImageHandler.COLOR_BACKGROUND[1]
        spacer[:, :, 2] = ImageHandler.COLOR_BACKGROUND[2]

        stack: list[NDArray[float32]] = []
        if banner_card is not None:
            # Pad the banner card to the final width.
            banner_card = ImageHandler._pad_to_width(banner_card, final_width, ImageHandler.COLOR_BACKGROUND)
            stack.extend((banner_card, spacer.copy()))
            if len(padded_panels) == 0:
                stack.pop()

        for idx, panel in enumerate(padded_panels):
            stack.append(panel)
            if idx < len(padded_panels) - 1:
                stack.append(spacer.copy())

        canvas: NDArray[float32] = np.concatenate(stack, axis=0)

        outer_pad: int = ImageHandler.STORYBOARD_PADDING
        board_h: int = canvas.shape[0] + outer_pad * 2
        # Use final width for board calculations
        board_w: int = final_width + outer_pad * 2
        board: NDArray[float32] = np.empty((board_h, board_w, 3), dtype=float32)
        board[:, :, 0] = ImageHandler.COLOR_BACKGROUND[0]
        board[:, :, 1] = ImageHandler.COLOR_BACKGROUND[1]
        board[:, :, 2] = ImageHandler.COLOR_BACKGROUND[2]
        board[outer_pad : outer_pad + canvas.shape[0], outer_pad : outer_pad + canvas.shape[1]] = canvas
        return board

    @staticmethod
    def _build_section_card(
        content: NDArray[float32],
        *,
        title: str,
        subtitle: str | None = None,
        target_h: int | None = None,
        target_w: int | None = None,
        banner_type: str = "heading",
    ) -> NDArray[float32]:
        """Create a stylised panel with a banner title and softly padded content."""
        if content.ndim != 3:
            return content

        ref_h: int = content.shape[0]
        ref_w: int = content.shape[1]
        if target_h is not None or target_w is not None:
            out_h: int = target_h if target_h is not None else ref_h
            out_w: int = target_w if target_w is not None else ref_w
            out_h = ImageHandler._ensure_multiple(max(out_h, ImageHandler.MIN_THUMBNAIL_SIZE), minimum_units=2)
            out_w = ImageHandler._ensure_multiple(max(out_w, ImageHandler.MIN_THUMBNAIL_SIZE), minimum_units=4)
            content = ImageHandler.aspect_fit_pad(content, out_h, out_w)
            ref_h, ref_w = content.shape[0], content.shape[1]

        body_padding: int = ImageHandler.SPACING_NORMAL
        inner_h: int = ref_h + body_padding * 2
        inner_w: int = ref_w + body_padding * 2
        body: NDArray[float32] = np.empty((inner_h, inner_w, 3), dtype=float32)
        body[:, :, 0] = ImageHandler.COLOR_SURFACE[0]
        body[:, :, 1] = ImageHandler.COLOR_SURFACE[1]
        body[:, :, 2] = ImageHandler.COLOR_SURFACE[2]
        body[body_padding : body_padding + ref_h, body_padding : body_padding + ref_w] = content

        # Create banner first - it may expand the requested width to satisfy text/padding
        banner: NDArray[float32] = ImageHandler._create_text_banner(
            inner_w, title, text_type=banner_type, reference_height=ref_h, subtitle=subtitle
        )

        # Match accent and body widths before concatenation.
        final_inner_w: int = max(inner_w, banner.shape[1])
        if body.shape[1] != final_inner_w:
            body = ImageHandler._pad_to_width(body, final_inner_w, ImageHandler.COLOR_SURFACE)

        accent_h: int = max(2, ImageHandler.BASE_UNIT // 2)
        accent: NDArray[float32] = np.empty((accent_h, final_inner_w, 3), dtype=float32)
        accent[:, :, 0] = ImageHandler.COLOR_ACCENT[0]
        accent[:, :, 1] = ImageHandler.COLOR_ACCENT[1]
        accent[:, :, 2] = ImageHandler.COLOR_ACCENT[2]

        # If banner ended up narrower (unlikely), pad it as well for symmetry
        if banner.shape[1] != final_inner_w:
            banner = ImageHandler._pad_to_width(banner, final_inner_w, ImageHandler.COLOR_SURFACE_ALT)

        panel_inner: NDArray[float32] = np.concatenate([banner, accent, body], axis=0)
        card: NDArray[float32] = ImageHandler._wrap_with_card(panel_inner)
        return card

    @staticmethod
    def _humanize_tag(tag: str) -> str:
        """Convert snake-case or underscored identifiers into a readable phrase."""
        cleaned: str = tag.replace("__", " → ").replace("_", " ").strip()
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return "Sequence"
        return cleaned[0].upper() + cleaned[1:]

    @staticmethod
    def _create_text_banner(
        width: int,
        text: str,
        *,
        text_type: str = "large",
        bg_color: tuple[float, float, float] | None = None,
        reference_height: int | None = None,
        subtitle: str | None = None,
    ) -> NDArray[float32]:
        """Create a text banner with background for labeling.

        Args:
            width: Width of the banner
            text: Text to display
            text_type: Typography style
            bg_color: Optional background color override
            reference_height: Optional height to scale typography consistently with content
            subtitle: Optional supporting line rendered below the title

        Returns:
            Banner image as float32 HxWx3
        """
        base_color = np.array(bg_color if bg_color is not None else ImageHandler.COLOR_SURFACE_ALT, dtype=float32)
        target_color = np.array(ImageHandler.COLOR_SURFACE, dtype=float32)

        ref_h: int = max(ImageHandler.MIN_TEXT_HEIGHT * 2, reference_height or ImageHandler.SPACING_LOOSE * 2)

        font_scale, thickness = ImageHandler._compute_font_scale(ref_h, text_type)
        text_w, text_h = ImageHandler._get_text_size(text, font_scale, thickness)

        subtitle_scale: float = 0.0
        subtitle_thickness: int = 0
        subtitle_w: int = 0
        subtitle_h: int = 0
        subtitle_gap: int = 0
        if subtitle:
            subtitle_scale, subtitle_thickness = ImageHandler._compute_font_scale(ref_h, "body")
            subtitle_w, subtitle_h = ImageHandler._get_text_size(subtitle, subtitle_scale, subtitle_thickness)
            subtitle_gap = ImageHandler.SPACING_TIGHT

        padding_v: int = ImageHandler.SPACING_NORMAL
        padding_h: int = max(ImageHandler.SPACING_NORMAL * 2, ImageHandler.BASE_UNIT * 3)

        required_width: int = max(text_w, subtitle_w) + padding_h * 2
        width = max(width, required_width)
        width = ImageHandler._ensure_multiple(width, minimum_units=6)

        content_h: int = text_h + subtitle_gap + subtitle_h
        banner_h: int = content_h + padding_v * 2
        banner_h = ImageHandler._ensure_multiple(banner_h, minimum_units=3)

        gradient = np.linspace(0.0, 1.0, banner_h, dtype=float32)
        banner: NDArray[float32] = np.empty((banner_h, width, 3), dtype=float32)
        for c in range(3):
            banner[:, :, c] = base_color[c] + (target_color[c] - base_color[c]) * gradient[:, None]

        accent_h: int = max(2, ImageHandler.BASE_UNIT // 2)
        accent_color: NDArray[float32] = np.array(ImageHandler.COLOR_ACCENT_SOFT, dtype=float32)
        banner[-accent_h:, :, 0] = accent_color[0]
        banner[-accent_h:, :, 1] = accent_color[1]
        banner[-accent_h:, :, 2] = accent_color[2]

        text_x: int = max(padding_h, (width - text_w) // 2) if width < (text_w + padding_h * 2) else padding_h
        text_baseline: int = padding_v + text_h
        banner = ImageHandler.overlay_text(
            img=banner,
            text=text,
            pos=(text_x, text_baseline),
            color=ImageHandler._color_to_uint8(ImageHandler.COLOR_TEXT_PRIMARY),
            font_scale=font_scale,
            thickness=thickness,
            outline=False,
            text_type=text_type,
        )

        if subtitle:
            subtitle_x: int = padding_h
            subtitle_baseline: int = text_baseline + subtitle_gap + subtitle_h
            banner = ImageHandler.overlay_text(
                img=banner,
                text=subtitle,
                pos=(subtitle_x, subtitle_baseline),
                color=ImageHandler._color_to_uint8(ImageHandler.COLOR_TEXT_SECONDARY),
                font_scale=subtitle_scale,
                thickness=subtitle_thickness,
                outline=False,
                text_type="body",
            )

        return banner

    @staticmethod
    def _add_separator_line(
        height: int, width: int, color: tuple[float, float, float] = (0.85, 0.85, 0.85)
    ) -> NDArray[float32]:
        """Create a thin separator line for visual separation between rows.

        Args:
            height: Line thickness
            width: Line width
            color: Line color (medium gray by default)

        Returns:
            Separator as float32 HxWx3
        """
        separator: NDArray[float32] = np.ones((height, width, 3), dtype=float32)
        separator[:, :, 0] = color[0]
        separator[:, :, 1] = color[1]
        separator[:, :, 2] = color[2]
        return separator

    @staticmethod
    def _overlay_texts_per_frame(
        frames: NDArray[float32], texts: list[str], pos: tuple[int, int] | None = None
    ) -> NDArray[float32]:
        if frames.ndim != 4:
            return frames
        out: list[NDArray[float32]] = []
        time_steps: int = frames.shape[0]
        for t in range(time_steps):
            txt: str = texts[t] if t < len(texts) else texts[-1]
            if pos is None:
                h, w = int(frames[t].shape[0]), int(frames[t].shape[1])
                pos_dyn: tuple[int, int] = ImageHandler._compute_text_position(h, w, is_title=False)
            else:
                pos_dyn = pos
            # Use body text style for frame annotations
            out.append(ImageHandler.overlay_text(frames[t], txt, pos=pos_dyn, text_type="body"))
        return np.stack(out, axis=0)

    def save_images(self, save_imgs_dir: str) -> None:
        """Save a concatenated PNG showing environment and reconstructions.

        Args:
            save_imgs_dir: Directory to write the image file into.
        """
        Path(save_imgs_dir).mkdir(exist_ok=True, parents=True)
        env_frames: NDArray[float32] = ImageHandler.render_env_frames(self.env, self.state_start, self.moves_taken)
        recon_frames: NDArray[float32] = ImageHandler.decode_latents_to_images(self.decoder, self.device, self.path)
        env_frames = self._overlay_texts_per_frame(env_frames, self._step_texts(self.moves_taken))
        recon_frames = self._overlay_texts_per_frame(recon_frames, self._step_texts(self.moves_taken))
        env_strip: NDArray[float32] = ImageHandler.hstack_strip(env_frames)
        recon_strip: NDArray[float32] = ImageHandler.hstack_strip(recon_frames)

        step_summary: str = f"{len(self.moves_taken)} step(s) executed" if self.moves_taken else "No actions taken"
        env_card: NDArray[float32] = ImageHandler._build_section_card(
            env_strip,
            title="Environment Execution",
            subtitle=f"Real environment roll-out | {step_summary}",
            banner_type="heading",
        )

        recon_card: NDArray[float32] = ImageHandler._build_section_card(
            recon_strip,
            title="Model Reconstruction",
            subtitle="Decoder rendering of latent trajectory",
            banner_type="heading",
        )

        panels: list[NDArray[float32]] = [env_card, recon_card]

        if self.optimal_soln is not None and len(self.optimal_soln) > 0:
            opt_frames: NDArray[float32] = ImageHandler.render_env_frames(self.env, self.state_start, self.optimal_soln)
            opt_frames = self._overlay_texts_per_frame(opt_frames, self._step_texts(self.optimal_soln))
            opt_strip: NDArray[float32] = ImageHandler.hstack_strip(opt_frames)
            optimal_summary: str = f"Reference solution | {len(self.optimal_soln)} step(s)"
            opt_card: NDArray[float32] = ImageHandler._build_section_card(
                opt_strip, title="Optimal Strategy Playback", subtitle=optimal_summary, banner_type="heading"
            )
            panels.append(opt_card)

        board_title: str = f"State {self.state_idx} - Search Trajectory Overview"
        board_subtitle: str = "Contrasting executed environment roll-outs with neural reconstructions"
        canvas: NDArray[float32] = ImageHandler._compose_storyboard(panels, title=board_title, subtitle=board_subtitle)

        out_path: str = str(Path(save_imgs_dir) / f"state_{self.state_idx}.png")
        iio.imwrite(out_path, np.clip(canvas * 255.0, 0, 255).astype(np.uint8))

    def save_gif(self, save_imgs_dir: str, fps: int = 5) -> None:
        """Save a combined GIF visualising environment and reconstructions.

        Args:
            save_imgs_dir: Directory to write the GIF file into.
            fps: Frames per second for the output GIF.
        """
        Path(save_imgs_dir).mkdir(exist_ok=True, parents=True)
        env_frames: NDArray[float32] = ImageHandler.render_env_frames(self.env, self.state_start, self.moves_taken)
        recon_frames: NDArray[float32] = ImageHandler.decode_latents_to_images(self.decoder, self.device, self.path)
        env_frames = self._overlay_texts_per_frame(env_frames, self._step_texts(self.moves_taken))
        recon_frames = self._overlay_texts_per_frame(recon_frames, self._step_texts(self.moves_taken))
        entries: list[tuple[str, NDArray[float32], str]] = [
            ("Environment Execution", env_frames, "Real environment perspective"),
            ("Model Reconstruction", recon_frames, "Decoder perspective"),
        ]
        if self.optimal_soln is not None and len(self.optimal_soln) > 0:
            opt_frames: NDArray[float32] = ImageHandler.render_env_frames(self.env, self.state_start, self.optimal_soln)
            opt_frames = self._overlay_texts_per_frame(opt_frames, self._step_texts(self.optimal_soln))
            entries.append(("Optimal Strategy Playback", opt_frames, "Reference policy replay"))

        if all(arr.shape[0] == 0 for _, arr, _ in entries):
            return

        max_len: int = max(int(arr.shape[0]) for _, arr, _ in entries)
        content_heights: list[int] = [int(arr.shape[1]) for _, arr, _ in entries if arr.shape[0] > 0]
        content_widths: list[int] = [int(arr.shape[2]) for _, arr, _ in entries if arr.shape[0] > 0]
        target_h: int = ImageHandler._ensure_multiple(max(content_heights, default=160), minimum_units=8)
        target_w: int = ImageHandler._ensure_multiple(max(content_widths, default=240), minimum_units=10)

        frames_u8: list[NDArray[np.uint8]] = []
        for t in range(max_len):
            cards: list[NDArray[float32]] = []
            for title, arr, descriptor in entries:
                if arr.shape[0] == 0:
                    continue
                idx: int = min(t, arr.shape[0] - 1)
                subtitle: str = f"{descriptor} | Step {t + 1}/{max_len}"
                card: NDArray[float32] = ImageHandler._build_section_card(
                    arr[idx], title=title, subtitle=subtitle, target_h=target_h, target_w=target_w, banner_type="large"
                )
                cards.append(card)

            storyboard_title: str = f"State {self.state_idx} - Step {t + 1}/{max_len}"
            storyboard_subtitle: str = "Environment vs model trajectories"
            frame_canvas: NDArray[float32] = ImageHandler._compose_storyboard(
                cards, title=storyboard_title, subtitle=storyboard_subtitle
            )
            frames_u8.append(np.clip(frame_canvas * 255.0, 0, 255).astype(np.uint8))

        delay_ms: int = round(1000 / max(1, fps))
        out_path: str = str(Path(save_imgs_dir) / f"state_{self.state_idx}.gif")
        iio.imwrite(out_path, frames_u8, duration=delay_ms, loop=0)

    @classmethod
    def build_for_qstar(
        cls,
        *,
        results_dir: str,
        vis_recon_mode: Literal["none", "image", "gif", "both"] = "none",
        vis_env_mode: Literal["none", "image", "gif", "both"] = "none",
        vis_combined_mode: Literal["none", "image", "gif", "both"] = "none",
        vis_on_unsolved: bool = False,
        include_start_goal_header: bool = True,
        header_start_title: str = "Start",
        header_goal_title: str = "Goal",
        env_row_title: str = "Environment Rendered",
        recon_row_title: str = "Reconstruction",
        fps: int = 5,
        env: ABCEnvironment[ABCState] | None = None,
        decoder: nn.Module | None = None,
        device: torch.device | None = None,
        model_cfg: ModelConfig | None = None,
        decoder_model_path: str | None = None,
        # sizing controls
        indiv_size_mode: Literal["original", "custom"] = "original",
        indiv_target_h: int | None = None,
        indiv_target_w: int | None = None,
        combined_row_h: int | None = None,
        combined_row_w: int | None = None,
    ) -> QStarImageContext:
        """Prepare a context for Q* visualization with directories and settings.

        The returned context can be reused for each state pair.
        """
        root = Path(results_dir) / "qstar_vis"
        vis_dirs: dict[str, str] = {}
        need_decoder = False
        if vis_recon_mode != "none":
            vis_dirs["recon_images"] = str(root / "recon" / "images")
            vis_dirs["recon_gifs"] = str(root / "recon" / "gifs")
            need_decoder = True
        if vis_env_mode != "none":
            vis_dirs["env_images"] = str(root / "env" / "images")
            vis_dirs["env_gifs"] = str(root / "env" / "gifs")
        if vis_combined_mode != "none":
            vis_dirs["combined_images"] = str(root / "combined" / "images")
            vis_dirs["combined_gifs"] = str(root / "combined" / "gifs")
            need_decoder = True
        for d in vis_dirs.values():
            Path(d).mkdir(exist_ok=True, parents=True)

        # Load decoder if required and not provided
        if (
            need_decoder
            and decoder is None
            and (
                env is not None
                and device is not None
                and model_cfg is not None
                and decoder_model_path is not None
                and load_nnet is not None
            )
        ):
            try:
                decoder_model: nn.Module = env.get_decoder_disc(model_cfg)
                decoder = load_nnet(decoder_model_path, decoder_model, device)
                decoder.eval()
            except Exception:
                decoder = None

        return QStarImageContext(
            env=env,
            decoder=decoder,
            device=device,
            dirs=vis_dirs,
            vis_recon_mode=vis_recon_mode,
            vis_env_mode=vis_env_mode,
            vis_combined_mode=vis_combined_mode,
            vis_on_unsolved=vis_on_unsolved,
            include_start_goal_header=include_start_goal_header,
            header_start_title=header_start_title,
            header_goal_title=header_goal_title,
            env_row_title=env_row_title,
            recon_row_title=recon_row_title,
            fps=fps,
            indiv_size_mode=indiv_size_mode,
            indiv_target_h=indiv_target_h,
            indiv_target_w=indiv_target_w,
            combined_row_h=combined_row_h,
            combined_row_w=combined_row_w,
        )

    @staticmethod
    def save_pair(
        ctx: QStarImageContext,
        *,
        pair_idx: int,
        solved: bool,
        path: list[NDArray[float32]],
        moves: list[int],
        start_variant: str,
        goal_variant: str,
        state_curr: NDArray[float32] | None,
        state_goal_curr: NDArray[float32] | None,
        base_states: tuple[ABCState, ABCState] | None,
    ) -> None:
        """Produce all requested outputs for one pair according to the context settings."""
        if ctx.is_noop:
            return
        # Strictly respect unsolved gating from config
        if (not solved) and (not ctx.vis_on_unsolved):
            return

        decoder: nn.Module | None = ctx.decoder
        device: torch.device | None = ctx.device
        env: ABCEnvironment[ABCState] | None = ctx.env

        # Create a variant-aware tag to avoid overwrites when multiple variants share the same index
        def _sanitize(s: str) -> str:
            return "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in s)

        var_tag: str = f"{_sanitize(start_variant)}__{_sanitize(goal_variant)}"

        # Recon frames when needed
        recon_frames: NDArray[float32] | None = None
        if (
            decoder is not None
            and (ctx.vis_recon_mode != "none" or ctx.vis_combined_mode != "none")
            and device is not None
        ):
            recon_frames = ImageHandler.decode_latents_to_images(decoder, device, path)

        # Env frames when available
        env_frames: NDArray[float32] | None = None
        start_state: ABCState | None = None
        if base_states is not None:
            start_state = base_states[0]
        if (
            env is not None
            and start_state is not None
            and (ctx.vis_env_mode != "none" or ctx.vis_combined_mode != "none")
        ):
            try:
                env_frames = ImageHandler.render_env_frames(env, start_state, moves)
            except Exception:
                env_frames = None

        # Recon-only outputs
        if recon_frames is not None and ctx.vis_recon_mode != "none":
            if ctx.vis_recon_mode in {"image", "both"}:
                ImageHandler.save_frames_strip_image(
                    recon_frames,
                    ctx.dirs.get("recon_images", ctx.root_dir),
                    pair_idx,
                    name_suffix=f"recon__{var_tag}",
                    target_h=(ctx.indiv_target_h if ctx.indiv_size_mode == "custom" else None),
                    target_w=(ctx.indiv_target_w if ctx.indiv_size_mode == "custom" else None),
                )
            if ctx.vis_recon_mode in {"gif", "both"}:
                ImageHandler.save_frames_gif(
                    recon_frames,
                    ctx.dirs.get("recon_gifs", ctx.root_dir),
                    pair_idx,
                    name_suffix=f"recon__{var_tag}",
                    fps=ctx.fps,
                    target_h=(ctx.indiv_target_h if ctx.indiv_size_mode == "custom" else None),
                    target_w=(ctx.indiv_target_w if ctx.indiv_size_mode == "custom" else None),
                )

        # Env-only outputs
        if env_frames is not None and ctx.vis_env_mode != "none":
            if ctx.vis_env_mode in {"image", "both"}:
                ImageHandler.save_frames_strip_image(
                    env_frames,
                    ctx.dirs.get("env_images", ctx.root_dir),
                    pair_idx,
                    name_suffix=f"env__{var_tag}",
                    target_h=(ctx.indiv_target_h if ctx.indiv_size_mode == "custom" else None),
                    target_w=(ctx.indiv_target_w if ctx.indiv_size_mode == "custom" else None),
                )
            if ctx.vis_env_mode in {"gif", "both"}:
                ImageHandler.save_frames_gif(
                    env_frames,
                    ctx.dirs.get("env_gifs", ctx.root_dir),
                    pair_idx,
                    name_suffix=f"env__{var_tag}",
                    fps=ctx.fps,
                    target_h=(ctx.indiv_target_h if ctx.indiv_size_mode == "custom" else None),
                    target_w=(ctx.indiv_target_w if ctx.indiv_size_mode == "custom" else None),
                )

        # Combined outputs with optional header
        if (recon_frames is not None or ctx.decoder is not None) and ctx.vis_combined_mode != "none":
            # Compose titles
            header_titles: tuple[str, str] = (
                f"{ctx.header_start_title} ({start_variant})",
                f"{ctx.header_goal_title} ({goal_variant})",
            )
            row_titles: tuple[str, str] = (ctx.env_row_title, ctx.recon_row_title)

            # header images
            start_img_nchw: NDArray[float32] | None = (
                ImageHandler.first_like(state_curr)
                if (ctx.include_start_goal_header and state_curr is not None)
                else None
            )
            goal_img_nchw: NDArray[float32] | None = (
                ImageHandler.first_like(state_goal_curr)
                if (ctx.include_start_goal_header and state_goal_curr is not None)
                else None
            )

            if ctx.vis_combined_mode in {"image", "both"}:
                ImageHandler.save_combined_images(
                    decoder=decoder,
                    device=device,
                    path=path,
                    env=env,
                    start_state=start_state,
                    moves=moves,
                    start_img_nchw=start_img_nchw,
                    goal_img_nchw=goal_img_nchw,
                    save_imgs_dir=ctx.dirs.get("combined_images", ctx.root_dir),
                    state_idx=pair_idx,
                    row_titles=row_titles,
                    header_titles=header_titles,
                    out_name_suffix=var_tag,
                    row_target_h=ctx.combined_row_h,
                    row_target_w=ctx.combined_row_w,
                )
            if ctx.vis_combined_mode in {"gif", "both"}:
                ImageHandler.save_combined_gif(
                    decoder=decoder,
                    device=device,
                    path=path,
                    env=env,
                    start_state=start_state,
                    moves=moves,
                    start_img_nchw=start_img_nchw,
                    goal_img_nchw=goal_img_nchw,
                    save_imgs_dir=ctx.dirs.get("combined_gifs", ctx.root_dir),
                    state_idx=pair_idx,
                    row_titles=row_titles,
                    header_titles=header_titles,
                    fps=ctx.fps,
                    out_name_suffix=var_tag,
                    row_target_h=ctx.combined_row_h,
                    row_target_w=ctx.combined_row_w,
                )

    @staticmethod
    @torch.inference_mode()
    def decode_latents_to_images(
        decoder: nn.Module, device: torch.device, path: list[NDArray[float32]]
    ) -> NDArray[float32]:
        """Decode a list of latent vectors into image frames.

        Args:
            decoder: PyTorch decoder module that maps latents to images.
            device: Device to run the decoder on.
            path: Sequence of latent vectors as numpy arrays.

        Returns:
            Numpy array of float32 images in HWC order with values in [0,1].
        """
        if len(path) == 0:
            return np.zeros((0, 0, 0, 3), dtype=float32)
        latents: NDArray[float32] = np.array(path, dtype=float32)
        latents_t: torch.Tensor = torch.tensor(latents, device=device).float()
        dec: torch.Tensor = decoder(latents_t)
        imgs: NDArray[float32] = dec.detach().to("cpu").numpy()
        imgs = np.clip(imgs, 0.0, 1.0)
        if imgs.ndim == 4:
            imgs = imgs.transpose(0, 2, 3, 1)
        return imgs.astype(float32, copy=False)

    @staticmethod
    def save_recon_images(
        decoder: nn.Module, device: torch.device, path: list[NDArray[float32]], save_imgs_dir: str, state_idx: int
    ) -> None:
        """Save reconstructions decoded from latents as a horizontal strip PNG.

        Args:
            decoder: Decoder module.
            device: Device for decoding.
            path: Latent vectors.
            save_imgs_dir: Directory to write the image.
            state_idx: Identifier used in the filename.
        """
        Path(save_imgs_dir).mkdir(exist_ok=True, parents=True)
        imgs: NDArray[float32] = ImageHandler.decode_latents_to_images(decoder, device, path)
        if imgs.size == 0:
            return
        ImageHandler.save_frames_strip_image(imgs, save_imgs_dir, state_idx, name_suffix="recon")

    @staticmethod
    def save_recon_gif(
        decoder: nn.Module,
        device: torch.device,
        path: list[NDArray[float32]],
        save_imgs_dir: str,
        state_idx: int,
        fps: int = 5,
    ) -> None:
        """Save reconstructions decoded from latents as a GIF.

        Args:
            decoder: Decoder module.
            device: Device for decoding.
            path: Latent vectors.
            save_imgs_dir: Directory to write the GIF.
            state_idx: Identifier used in the filename.
            fps: Frames per second for the GIF.
        """
        Path(save_imgs_dir).mkdir(exist_ok=True, parents=True)
        imgs: NDArray[float32] = ImageHandler.decode_latents_to_images(decoder, device, path)
        if imgs.size == 0:
            return
        ImageHandler.save_frames_gif(imgs, save_imgs_dir, state_idx, name_suffix="recon", fps=fps)

    @staticmethod
    def save_frames_strip_image(
        frames: NDArray[float32],
        save_imgs_dir: str,
        state_idx: int,
        *,
        name_suffix: str,
        target_h: int | None = None,
        target_w: int | None = None,
    ) -> None:
        """Write a horizontal strip PNG composed of a sequence of frames.

        Args:
            frames: Array of frames (T, H, W, C) with values in [0,1].
            save_imgs_dir: Output directory.
            state_idx: Identifier for filename.
            name_suffix: Suffix describing the strip (e.g. 'recon').
            target_h: If provided, resize the output to this height.
            target_w: If provided, resize the output to this width.
        """
        Path(save_imgs_dir).mkdir(exist_ok=True, parents=True)
        if frames.ndim != 4 or frames.shape[0] == 0:
            return
        mosaic: NDArray[float32] = np.concatenate([frames[t] for t in range(frames.shape[0])], axis=1)
        target_h_val: int | None = target_h if target_h is not None else None
        target_w_val: int | None = target_w if target_w is not None else None
        card: NDArray[float32] = ImageHandler._build_section_card(
            mosaic,
            title="Frame Timeline",
            subtitle=ImageHandler._humanize_tag(name_suffix),
            target_h=target_h_val,
            target_w=target_w_val,
            banner_type="heading",
        )
        canvas: NDArray[float32] = ImageHandler._compose_storyboard(
            panels=[card], title=f"State {state_idx} - Frame Timeline", subtitle="Sequence export"
        )
        out_path: str = str(Path(save_imgs_dir) / f"state_{state_idx}_{name_suffix}.png")
        iio.imwrite(out_path, np.clip(canvas * 255.0, 0, 255).astype(np.uint8))

    @staticmethod
    def save_frames_gif(
        frames: NDArray[float32],
        save_imgs_dir: str,
        state_idx: int,
        *,
        name_suffix: str,
        fps: int,
        target_h: int | None = None,
        target_w: int | None = None,
    ) -> None:
        """Write frames as an animated GIF.

        Args:
            frames: Array of frames (T, H, W, C) values in [0,1].
            save_imgs_dir: Output directory.
            state_idx: Identifier for filename.
            name_suffix: Suffix for the filename.
            fps: Frames per second.
            target_h: If provided, resize each frame to this height.
            target_w: If provided, resize each frame to this width.
        """
        Path(save_imgs_dir).mkdir(exist_ok=True, parents=True)
        if frames.ndim != 4 or frames.shape[0] == 0:
            return
        delay_ms: int = round(1000 / max(1, fps))
        out_path: str = str(Path(save_imgs_dir) / f"state_{state_idx}_{name_suffix}.gif")

        target_h_val: int | None = target_h if target_h is not None else None
        target_w_val: int | None = target_w if target_w is not None else None

        frames_u8: list[NDArray[np.uint8]] = []
        total_frames: int = int(frames.shape[0])
        for t in range(total_frames):
            content: NDArray[float32] = frames[t]
            card: NDArray[float32] = ImageHandler._build_section_card(
                content,
                title="Frame Sequence",
                subtitle=f"{ImageHandler._humanize_tag(name_suffix)} | Frame {t + 1}/{total_frames}",
                target_h=target_h_val,
                target_w=target_w_val,
                banner_type="large",
            )
            canvas: NDArray[float32] = ImageHandler._compose_storyboard(
                panels=[card], title=f"State {state_idx} - Animated Sequence", subtitle="export"
            )
            frames_u8.append(np.clip(canvas * 255.0, 0, 255).astype(np.uint8))

        iio.imwrite(out_path, frames_u8, duration=delay_ms, loop=0)

    @staticmethod
    def concat_if_six_channels(img_nchw: NDArray[float32]) -> NDArray[float32]:
        """Convert one or more channel-first views into one HWC RGB image.

        A single channel is repeated three times. Channel counts divisible by
        three are treated as adjacent RGB views. Other channel counts use their
        first three channels.

        Args:
            img_nchw: Image array either (1, C, H, W) or (C, H, W).

        Returns:
            HWC float32 RGB image with values clipped to [0, 1].
        """
        if img_nchw.ndim == 4:
            c, _h, _w = int(img_nchw.shape[1]), int(img_nchw.shape[2]), int(img_nchw.shape[3])
            chw: NDArray[float32] = img_nchw[0]
        else:
            c, _h, _w = int(img_nchw.shape[0]), int(img_nchw.shape[1]), int(img_nchw.shape[2])
            chw = img_nchw

        # Single grayscale channel -> replicate to RGB
        if c == 1:
            gray: NDArray[float32] = chw[0:1, :, :]
            rgb: NDArray[float32] = np.repeat(gray, 3, axis=0)
            return np.clip(rgb.transpose(1, 2, 0), 0.0, 1.0).astype(float32, copy=False)

        # Exactly 3 channels -> standard conversion
        if c == 3:
            return np.clip(chw.transpose(1, 2, 0), 0.0, 1.0).astype(float32, copy=False)

        # Multiple-of-3 channels -> treat as horizontal strip of 3-channel views
        if c % 3 == 0:
            v: int = c // 3
            views: list[NDArray[float32]] = []
            for i in range(v):
                seg: NDArray[float32] = chw[i * 3 : (i + 1) * 3, :, :]
                views.append(seg.transpose(1, 2, 0))
            hwc: NDArray[float32] = np.concatenate(views, axis=1)
            return np.clip(hwc, 0.0, 1.0).astype(float32, copy=False)

        # Fallback: use first 3 channels
        rgb_fallback: NDArray[float32] = chw[:3, :, :].transpose(1, 2, 0)
        return np.clip(rgb_fallback, 0.0, 1.0).astype(float32, copy=False)

    @staticmethod
    def render_env_frames(env: ABCEnvironment[ABCState], start_state: ABCState, moves: list[int]) -> NDArray[float32]:
        """Simulate environment state sequence and return rendered frames.

        Args:
            env: Environment implementing `next_state` and `state_to_real`.
            start_state: Initial state object.
            moves: Sequence of integer actions to apply.

        Returns:
            Array of frames (T, H, W, C) as float32 in [0,1].
        """
        states: list[ABCState] = [start_state]
        cur: ABCState = start_state
        for mv in moves:
            cur = env.next_state([cur], [mv])[0][0]
            states.append(cur)
        imgs_nchw: NDArray[float32] = env.state_to_real(states)
        frames: list[NDArray[float32]] = [
            ImageHandler.concat_if_six_channels(imgs_nchw[t]) for t in range(imgs_nchw.shape[0])
        ]
        return np.stack(frames, axis=0)

    @staticmethod
    def hstack_strip(frames: NDArray[float32]) -> NDArray[float32]:
        """Return a single image which is frames concatenated horizontally.

        Args:
            frames: Array of frames (T, H, W, C).

        Returns:
            Hx(W*T)xC image as float32. If input is empty returns a 1x1 white image.
        """
        if frames.ndim != 4 or frames.shape[0] == 0:
            return np.zeros((1, 1, 3), dtype=float32)
        return np.concatenate([frames[t] for t in range(frames.shape[0])], axis=1)

    @staticmethod
    def ensure_size(img: NDArray[float32], h: int, w: int) -> NDArray[float32]:
        """Resize an HxW image to the requested size, preserving value range.

        Args:
            img: Image array HxW or HxWxC.
            h: Desired height.
            w: Desired width.

        Returns:
            Resized image as float32 clipped to [0,1].
        """
        if img.shape[0] == h and img.shape[1] == w:
            return img
        resized: MatLike = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return np.clip(resized, 0.0, 1.0).astype(float32, copy=False)

    @staticmethod
    def overlay_text(
        img: NDArray[float32],
        text: str,
        *,
        pos: tuple[int, int] | None = None,
        color: tuple[int, int, int] = (255, 255, 255),
        font_scale: float | None = None,
        thickness: int | None = None,
        outline: bool = True,
        text_type: str = "body",
    ) -> NDArray[float32]:
        """Overlay text on an HxWx3 image and return the result.

        Args:
            img: HxWx3 float32 image with values in [0,1].
            text: Text string to render.
            pos: Pixel position ``(x, y)`` for the text baseline. The image
                dimensions determine the position when this is ``None``.
            color: RGB color tuple.
            font_scale: Font scale multiplier. If None, computed from image height and text_type.
            thickness: Line thickness for the text. If None, computed from font_scale.
            outline: If True, draw a dark outline behind the text for legibility.
            text_type: Text category used for sizing. Accepted values are
                ``tiny``, ``small``, ``body``, ``large``, ``title``, and
                ``heading``.

        Returns:
            Image with text rendered, as float32 in [0,1].
        """
        out: NDArray[float32] = img.copy()
        bgr: NDArray[np.uint8] = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        h, w = int(out.shape[0]), int(out.shape[1])

        # Compute position if not provided
        if pos is None:
            pos = ImageHandler._compute_text_position(h, w, is_title=(text_type in {"title", "heading"}))

        # Compute font scale and thickness if not provided
        if font_scale is None or thickness is None:
            computed_scale, computed_thickness = ImageHandler._compute_font_scale(h, text_type)
            font_scale = font_scale if font_scale is not None else computed_scale
            thickness = thickness if thickness is not None else computed_thickness

        # Outline for legibility on complex backgrounds with proper spacing
        if outline:
            outline_thickness: int = max(2, thickness + round(thickness * 0.6))
            cv2.putText(
                img=bgr,
                text=text,
                org=pos,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(0, 0, 0),
                thickness=outline_thickness,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img=bgr,
            text=text,
            org=pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(color[2], color[1], color[0]),
            thickness=max(1, thickness),
            lineType=cv2.LINE_AA,
        )
        return (bgr.astype(float32) / 255.0).astype(float32, copy=False)

    @staticmethod
    def aspect_fit_pad(
        img: NDArray[float32], target_h: int, target_w: int, pad_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> NDArray[float32]:
        """Resize preserving aspect ratio and pad to fit target size.

        Args:
            img: Input HxWxC image.
            target_h: Target height.
            target_w: Target width.
            pad_color: RGB tuple in [0,1] for padding color.

        Returns:
            HxWxC float32 image, aspect preserved with padding.
        """
        h, w = int(img.shape[0]), int(img.shape[1])
        if h <= 0 or w <= 0:
            return np.ones((target_h, target_w, 3), dtype=float32)

        # Compute scale to fit within target while preserving aspect ratio
        scale: float = min(target_w / max(1, w), target_h / max(1, h))
        new_w: int = max(1, round(w * scale))
        new_h: int = max(1, round(h * scale))
        resized: NDArray[float32] = ImageHandler.ensure_size(img, new_h, new_w)

        # Create output canvas with padding color
        out: NDArray[float32] = np.empty((target_h, target_w, 3), dtype=float32)
        out[:, :, 0] = pad_color[0]
        out[:, :, 1] = pad_color[1]
        out[:, :, 2] = pad_color[2]

        # Center the resized image using proper alignment
        # Align the text origin to the spacing grid.
        x_offset: int = (target_w - new_w) // 2
        y_offset: int = (target_h - new_h) // 2

        # Align to BASE_UNIT grid for pixel-perfect rendering
        x_offset = ImageHandler.BASE_UNIT * (x_offset // ImageHandler.BASE_UNIT)
        y_offset = ImageHandler.BASE_UNIT * (y_offset // ImageHandler.BASE_UNIT)

        # Stop before the next line would overflow the panel.
        x_offset = max(0, min(x_offset, target_w - new_w))
        y_offset = max(0, min(y_offset, target_h - new_h))

        out[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
        return out

    @staticmethod
    def save_env_images(
        env: ABCEnvironment[ABCState], start_state: ABCState, moves: list[int], save_imgs_dir: str, state_idx: int
    ) -> None:
        """Save a PNG showing environment frames for a solution as a strip.

        Args:
            env: Environment instance.
            start_state: Starting state.
            moves: Sequence of actions applied.
            save_imgs_dir: Output directory.
            state_idx: Identifier for the filename.
        """
        Path(save_imgs_dir).mkdir(exist_ok=True, parents=True)
        frames: NDArray[float32] = ImageHandler.render_env_frames(env, start_state, moves)
        strip: NDArray[float32] = ImageHandler.hstack_strip(frames)
        card: NDArray[float32] = ImageHandler._build_section_card(
            strip,
            title="Environment Execution",
            subtitle=f"{len(moves)} step(s) | Simulator playback",
            banner_type="heading",
        )
        canvas: NDArray[float32] = ImageHandler._compose_storyboard(
            panels=[card], title=f"State {state_idx} - Environment Timeline", subtitle="Rendered from environment"
        )
        out_path: str = str(Path(save_imgs_dir) / f"state_{state_idx}_env.png")
        iio.imwrite(out_path, np.clip(canvas * 255.0, 0, 255).astype(np.uint8))

    @staticmethod
    def save_env_gif(
        env: ABCEnvironment[ABCState],
        start_state: ABCState,
        moves: list[int],
        save_imgs_dir: str,
        state_idx: int,
        *,
        fps: int,
    ) -> None:
        """Save an animated GIF of environment-rendered frames.

        Args:
            env: Environment instance.
            start_state: Starting state.
            moves: Sequence of actions applied.
            save_imgs_dir: Output directory.
            state_idx: Identifier for the filename.
            fps: Frames per second.
        """
        Path(save_imgs_dir).mkdir(exist_ok=True, parents=True)
        frames: NDArray[float32] = ImageHandler.render_env_frames(env, start_state, moves)
        delay_ms: int = round(1000 / max(1, fps))
        out_path: str = str(Path(save_imgs_dir) / f"state_{state_idx}_env.gif")

        frames_u8: list[NDArray[np.uint8]] = []
        total_frames: int = int(frames.shape[0])
        for idx in range(total_frames):
            card: NDArray[float32] = ImageHandler._build_section_card(
                frames[idx],
                title="Environment Execution",
                subtitle=f"Frame {idx + 1}/{total_frames} | Simulator playback",
                banner_type="large",
            )
            canvas: NDArray[float32] = ImageHandler._compose_storyboard(
                [card], title=f"State {state_idx} - Environment Animation", subtitle="Rendered from environment"
            )
            frames_u8.append(np.clip(canvas * 255.0, 0, 255).astype(np.uint8))

        iio.imwrite(out_path, frames_u8, duration=delay_ms, loop=0)

    @staticmethod
    def make_header_strip(
        start_img_nchw: NDArray[float32],
        goal_img_nchw: NDArray[float32],
        total_width: int,
        row_height: int,
        *,
        start_title: str = "Start",
        goal_title: str = "Goal",
    ) -> NDArray[float32]:
        """Create a header image with start and goal thumbnails and titles.

        Args:
            start_img_nchw: First image in NCHW or CHW form.
            goal_img_nchw: Second image in NCHW or CHW form.
            total_width: Width of the header canvas.
            row_height: Height allocated for the thumbnails.
            start_title: Title placed above the start image.
            goal_title: Title placed above the goal image.

        Returns:
            RGB header image as float32 in [0,1].
        """
        outer_margin: int = ImageHandler.SPACING_NORMAL
        title_band: int = max(ImageHandler.MIN_TEXT_HEIGHT + ImageHandler.SPACING_NORMAL, round(row_height * 0.3))
        title_band = ImageHandler.BASE_UNIT * max(2, round(title_band / ImageHandler.BASE_UNIT))

        thumb_h: int = max(ImageHandler.MIN_THUMBNAIL_SIZE, row_height)
        thumb_h = ImageHandler.BASE_UNIT * max(2, round(thumb_h / ImageHandler.BASE_UNIT))

        column_gap: int = ImageHandler.SPACING_NORMAL * 2
        available_w: int = max(1, total_width - 2 * outer_margin)
        column_gap = min(column_gap, max(ImageHandler.SPACING_NORMAL, available_w // 12))
        column_w: int = max(ImageHandler.MIN_THUMBNAIL_SIZE, (available_w - column_gap) // 2)
        column_w = ImageHandler.BASE_UNIT * max(2, round(column_w / ImageHandler.BASE_UNIT))

        layout_w: int = column_w * 2 + column_gap
        if layout_w > available_w:
            column_w = max(ImageHandler.MIN_THUMBNAIL_SIZE, (available_w - column_gap) // 2)
            layout_w = column_w * 2 + column_gap

        start_x: int = outer_margin + max(0, (available_w - layout_w) // 2)
        left_x: int = start_x
        right_x: int = start_x + column_w + column_gap

        canvas_h: int = title_band + thumb_h + outer_margin * 2
        canvas: NDArray[float32] = np.ones((canvas_h, total_width, 3), dtype=float32)
        canvas[...] = 1.0
        title_bg: float = 0.96
        canvas[outer_margin : outer_margin + title_band, :, :] = title_bg

        # Prepare images in HxWx3 and pad to consistent tiles
        si: NDArray[float32] = ImageHandler.concat_if_six_channels(start_img_nchw)
        gi: NDArray[float32] = ImageHandler.concat_if_six_channels(goal_img_nchw)

        si_tile: NDArray[float32] = ImageHandler.aspect_fit_pad(si, thumb_h, column_w)
        gi_tile: NDArray[float32] = ImageHandler.aspect_fit_pad(gi, thumb_h, column_w)

        thumb_y: int = outer_margin + title_band
        canvas[thumb_y : thumb_y + thumb_h, left_x : left_x + column_w] = si_tile
        canvas[thumb_y : thumb_y + thumb_h, right_x : right_x + column_w] = gi_tile

        # Titles in the title band with measured centering
        font_scale, thickness = ImageHandler._compute_font_scale(title_band, "large")
        text_y: int = outer_margin + max(ImageHandler.MIN_TEXT_HEIGHT, round(title_band * 0.65))
        start_text_w, _ = ImageHandler._get_text_size(start_title, font_scale, thickness)
        goal_text_w, _ = ImageHandler._get_text_size(goal_title, font_scale, thickness)

        text_x1: int = left_x + max(0, (column_w - start_text_w) // 2)
        text_x2: int = right_x + max(0, (column_w - goal_text_w) // 2)

        canvas = ImageHandler.overlay_text(
            canvas,
            start_title,
            pos=(text_x1, text_y),
            color=ImageHandler._color_to_uint8(ImageHandler.COLOR_TEXT_PRIMARY),
            font_scale=font_scale,
            thickness=thickness,
            outline=False,
            text_type="large",
        )
        return ImageHandler.overlay_text(
            canvas,
            goal_title,
            pos=(text_x2, text_y),
            color=ImageHandler._color_to_uint8(ImageHandler.COLOR_TEXT_PRIMARY),
            font_scale=font_scale,
            thickness=thickness,
            outline=False,
            text_type="large",
        )

    @staticmethod
    def save_combined_images(
        *,
        decoder: nn.Module | None,
        device: torch.device | None,
        path: list[NDArray[float32]],
        env: ABCEnvironment[ABCState] | None,
        start_state: ABCState | None,
        moves: list[int] | None,
        start_img_nchw: NDArray[float32] | None,
        goal_img_nchw: NDArray[float32] | None,
        save_imgs_dir: str,
        state_idx: int,
        row_titles: tuple[str, str] = ("Environment Rendered", "Reconstruction"),
        header_titles: tuple[str, str] = ("Start", "Goal"),
        out_name_suffix: str | None = None,
        row_target_h: int | None = None,
        row_target_w: int | None = None,
    ) -> None:
        """Save a combined PNG with optional header, env row and recon row.

        Args:
            decoder: Optional decoder for reconstructions.
            device: Device to use with decoder.
            path: Latent path used for reconstructions.
            env: Optional environment to render real frames.
            start_state: Start state for environment simulation.
            moves: Actions applied for the path.
            start_img_nchw: Optional start image for header (CHW or NCHW).
            goal_img_nchw: Optional goal image for header (CHW or NCHW).
            save_imgs_dir: Output directory for the combined image.
            state_idx: Index used in the filename.
            row_titles: Titles for the two rows.
            header_titles: Titles for the header thumbnails.
            out_name_suffix: Optional suffix for the filename.
            row_target_h: If provided, resize each row to this height.
            row_target_w: If provided, resize each row to this width.
        """
        Path(save_imgs_dir).mkdir(exist_ok=True, parents=True)
        recon_frames: NDArray[float32] | None = (
            ImageHandler.decode_latents_to_images(decoder, device, path)
            if (decoder is not None and device is not None)
            else None
        )

        env_strip: NDArray[float32] | None = None
        if env is not None and start_state is not None and moves is not None:
            env_frames: NDArray[float32] = ImageHandler.render_env_frames(env, start_state, moves)
            env_strip = ImageHandler.hstack_strip(env_frames)

        if recon_frames is None:
            recon_frames = np.zeros((1, 1, 1, 3), dtype=float32)
        recon_strip: NDArray[float32] = ImageHandler.hstack_strip(recon_frames)

        target_h_val: int | None = row_target_h if row_target_h is not None else None
        target_w_val: int | None = row_target_w if row_target_w is not None else None

        panels: list[NDArray[float32]] = []
        if start_img_nchw is not None and goal_img_nchw is not None:
            header_strip: NDArray[float32] = ImageHandler.make_header_strip(
                start_img_nchw=start_img_nchw,
                goal_img_nchw=goal_img_nchw,
                total_width=(target_w_val if target_w_val is not None else max(recon_strip.shape[1], 320)),
                row_height=max(ImageHandler.MIN_THUMBNAIL_SIZE, (target_h_val or recon_strip.shape[0]) // 2),
                start_title=header_titles[0],
                goal_title=header_titles[1],
            )
            header_card: NDArray[float32] = ImageHandler._build_section_card(
                header_strip,
                title=f"{header_titles[0]} & {header_titles[1]}",
                subtitle="Starting and goal configurations",
                banner_type="large",
            )
            panels.append(header_card)

        if env_strip is not None:
            panels.append(
                ImageHandler._build_section_card(
                    env_strip,
                    title=row_titles[0],
                    subtitle="Environment playback",
                    target_h=target_h_val,
                    target_w=target_w_val,
                    banner_type="heading",
                )
            )

        panels.append(
            ImageHandler._build_section_card(
                recon_strip,
                title=row_titles[1],
                subtitle="Decoder reconstruction",
                target_h=target_h_val,
                target_w=target_w_val,
                banner_type="heading",
            )
        )

        composite_title: str = f"State {state_idx} - Combined Overview"
        composite_subtitle: str = "Environment roll-out and model reconstruction"
        final: NDArray[float32] = ImageHandler._compose_storyboard(
            panels, title=composite_title, subtitle=composite_subtitle
        )

        name_mid: str = f"_{out_name_suffix}" if out_name_suffix else ""
        out_path: str = str(Path(save_imgs_dir) / f"state_{state_idx}{name_mid}_combined.png")
        iio.imwrite(out_path, np.clip(final * 255.0, 0, 255).astype(np.uint8))

    @staticmethod
    def save_combined_gif(
        *,
        decoder: nn.Module | None,
        device: torch.device | None,
        path: list[NDArray[float32]],
        env: ABCEnvironment[ABCState] | None,
        start_state: ABCState | None,
        moves: list[int] | None,
        start_img_nchw: NDArray[float32] | None,
        goal_img_nchw: NDArray[float32] | None,
        save_imgs_dir: str,
        state_idx: int,
        row_titles: tuple[str, str] = ("Environment Rendered", "Reconstruction"),
        header_titles: tuple[str, str] = ("Start", "Goal"),
        fps: int = 5,
        out_name_suffix: str | None = None,
        row_target_h: int | None = None,
        row_target_w: int | None = None,
    ) -> None:
        """Save an animated GIF where each frame composes header, env and recon rows.

        Args:
            decoder: Optional decoder for reconstructions.
            device: Optional device for decoding.
            path: Latent path used for reconstructions.
            env: Optional environment to render real frames.
            start_state: Start state for environment simulation.
            moves: Actions applied for the path.
            start_img_nchw: Optional start image for header.
            goal_img_nchw: Optional goal image for header.
            save_imgs_dir: Output directory.
            state_idx: Index used in the filename.
            row_titles: Titles for the rows.
            header_titles: Titles for the header thumbnails.
            fps: Frames per second.
            out_name_suffix: Optional suffix for the filename.
            row_target_h: If provided, resize each content row to this height.
            row_target_w: If provided, resize content to this width.
        """
        Path(save_imgs_dir).mkdir(exist_ok=True, parents=True)

        # Decode reconstructions if available
        recon_frames: NDArray[float32] | None = (
            ImageHandler.decode_latents_to_images(decoder, device, path)
            if (decoder is not None and device is not None)
            else None
        )

        # Render environment frames if available
        env_frames: NDArray[float32] | None = None
        if env is not None and start_state is not None and moves is not None:
            env_frames = ImageHandler.render_env_frames(env, start_state, moves)

        time_steps: int = max(
            0,
            recon_frames.shape[0] if recon_frames is not None else 0,
            env_frames.shape[0] if env_frames is not None else 0,
        )
        if time_steps <= 0:
            return

        base_height: int = ImageHandler._ensure_multiple(
            max(
                env_frames.shape[1] if env_frames is not None else 0,
                recon_frames.shape[1] if recon_frames is not None else 0,
                160,
            ),
            minimum_units=8,
        )
        base_width: int = ImageHandler._ensure_multiple(
            max(
                env_frames.shape[2] if env_frames is not None else 0,
                recon_frames.shape[2] if recon_frames is not None else 0,
                320,
            ),
            minimum_units=10,
        )

        target_h_val: int = row_target_h if row_target_h is not None else base_height
        target_w_val: int = row_target_w if row_target_w is not None else base_width

        header_card: NDArray[float32] | None = None
        if start_img_nchw is not None and goal_img_nchw is not None:
            header_strip: NDArray[float32] = ImageHandler.make_header_strip(
                start_img_nchw=start_img_nchw,
                goal_img_nchw=goal_img_nchw,
                total_width=target_w_val,
                row_height=max(ImageHandler.MIN_THUMBNAIL_SIZE, target_h_val // 2),
                start_title=header_titles[0],
                goal_title=header_titles[1],
            )
            header_card = ImageHandler._build_section_card(
                header_strip,
                title=f"{header_titles[0]} & {header_titles[1]}",
                subtitle="Starting and goal configurations",
                target_w=target_w_val,
                banner_type="large",
            )

        frames_u8: list[NDArray[np.uint8]] = []
        for step in range(time_steps):
            cards: list[NDArray[float32]] = []
            if header_card is not None:
                cards.append(header_card)

            if env_frames is not None and env_frames.shape[0] > 0:
                env_idx: int = min(step, env_frames.shape[0] - 1)
                env_card: NDArray[float32] = ImageHandler._build_section_card(
                    env_frames[env_idx],
                    title=row_titles[0],
                    subtitle=f"Environment playback | Step {step + 1}/{time_steps}",
                    target_h=target_h_val,
                    target_w=target_w_val,
                    banner_type="heading",
                )
                cards.append(env_card)

            if recon_frames is not None and recon_frames.shape[0] > 0:
                recon_idx: int = min(step, recon_frames.shape[0] - 1)
                recon_card: NDArray[float32] = ImageHandler._build_section_card(
                    recon_frames[recon_idx],
                    title=row_titles[1],
                    subtitle=f"Decoder reconstruction | Step {step + 1}/{time_steps}",
                    target_h=target_h_val,
                    target_w=target_w_val,
                    banner_type="heading",
                )
                cards.append(recon_card)

            storyboard_title: str = f"State {state_idx} - Step {step + 1}/{time_steps}"
            storyboard_subtitle: str = "Environment roll-out and model reconstruction"
            frame_canvas: NDArray[float32] = ImageHandler._compose_storyboard(
                cards, title=storyboard_title, subtitle=storyboard_subtitle
            )
            frames_u8.append(np.clip(frame_canvas * 255.0, 0, 255).astype(np.uint8))

        delay_ms: int = round(1000 / max(1, fps))
        name_mid: str = f"_{out_name_suffix}" if out_name_suffix else ""
        out_path: str = str(Path(save_imgs_dir) / f"state_{state_idx}{name_mid}_combined.gif")
        iio.imwrite(out_path, frames_u8, duration=delay_ms, loop=0)

    @staticmethod
    def first_like(x: NDArray[float32] | Sequence[NDArray[float32]] | None) -> NDArray[float32] | None:
        """Return the first image from a sequence or the image itself.

        Args:
            x: Either a single image (CHW/NCHW) or a sequence of such images.

        Returns:
            The first image if `x` is a sequence, the image itself if single, or None.
        """
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        # Sequence (list/tuple) -> return first element if present
        try:
            return x[0]
        except Exception:
            return None

    @staticmethod
    def load_image(processed_img_path: str) -> NDArray[float32]:
        """Load an image from disk and return it as float32 RGB in [0,1].

        Args:
            processed_img_path: Path to the image file.

        Returns:
            HxWx3 float32 image in RGB order.
        """
        image_bgr_opt: MatLike | None = cv2.imread(processed_img_path)
        if image_bgr_opt is None:
            raise ValueError(f"Could not read image file: {processed_img_path}")
        image_bgr: MatLike = image_bgr_opt
        image_rgb: MatLike = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image: NDArray[float32] = image_rgb.astype(float32) / 255.0
        return image

    @staticmethod
    def list_images_in_dir(dir_path: str, pattern: str) -> list[str]:
        """List files matching a glob pattern in a directory.

        Args:
            dir_path: Directory to search in.
            pattern: Glob pattern e.g. '*.png'.

        Returns:
            Sorted list of matching file paths.
        """
        paths: list[str] = [str(p) for p in Path(dir_path).glob(pattern) if p.is_file()]
        paths.sort()
        return paths

    @staticmethod
    def load_images_batch(paths: list[str]) -> NDArray[float32]:
        """Load a list of images and return them as NCHW float32 array.

        Args:
            paths: Ordered list of file paths.

        Returns:
            Array shaped (T, C, H, W) with values in [0,1].
        """
        if len(paths) == 0:
            return np.zeros((0, 3, 0, 0), dtype=float32)
        imgs_hwcn: list[NDArray[float32]] = [ImageHandler.load_image(p) for p in paths]
        h, w, c = imgs_hwcn[0].shape
        for p, arr in zip(paths, imgs_hwcn, strict=False):
            if arr.shape != (h, w, c):
                raise ValueError(f"Image size mismatch: {p} has {arr.shape}, expected {(h, w, c)}")
        imgs_nchw: NDArray[float32] = np.stack([arr.transpose(2, 0, 1) for arr in imgs_hwcn], axis=0).astype(float32)
        return imgs_nchw

    @staticmethod
    def save_solution_visuals(
        *,
        env: ABCEnvironment[ABCState],
        state: ABCState,
        soln: list[int],
        decoder: nn.Module,
        device: torch.device,
        state_idx: int,
        path: list[NDArray[float32]],
        save_imgs_dir: str,
        save_imgs: bool,
        save_gif: bool,
        gif_fps: int,
    ) -> None:
        """Save images or a GIF for a solution path.

        Args:
            env: Environment instance.
            state: Starting state object.
            soln: Sequence of integer actions representing a solution.
            decoder: Decoder module for reconstructions.
            device: Device used with the decoder.
            state_idx: Index used in output filenames.
            path: Latent trajectory corresponding to the solution.
            save_imgs_dir: Directory to write artifacts.
            save_imgs: Whether to save PNG images.
            save_gif: Whether to save GIFs.
            gif_fps: Frames per second for GIF output.
        """
        handler: ImageHandler = ImageHandler(env, state, decoder, path, len(soln), state_idx, device)
        for mv in soln:
            handler.step(mv)
        if save_imgs:
            handler.save_images(save_imgs_dir)
        elif save_gif:
            handler.save_gif(save_imgs_dir, fps=gif_fps)


class QStarImageContext:
    """Reusable Q* visualization settings and directories."""

    def __init__(
        self,
        *,
        env: ABCEnvironment[ABCState] | None,
        decoder: nn.Module | None,
        device: torch.device | None,
        dirs: dict[str, str],
        vis_recon_mode: str,
        vis_env_mode: str,
        vis_combined_mode: str,
        vis_on_unsolved: bool,
        include_start_goal_header: bool,
        header_start_title: str,
        header_goal_title: str,
        env_row_title: str,
        recon_row_title: str,
        fps: int,
        indiv_size_mode: str,
        indiv_target_h: int | None,
        indiv_target_w: int | None,
        combined_row_h: int | None,
        combined_row_w: int | None,
    ) -> None:
        self.env: ABCEnvironment[ABCState] | None = env
        self.decoder: nn.Module | None = decoder
        self.device: torch.device | None = device
        self.dirs: dict[str, str] = dirs
        # root dir is parent of any existing dir or the cwd fallback
        self.root_dir: str = os.path.commonpath(list(dirs.values())) if dirs else str(Path.cwd())
        self.vis_recon_mode: str = vis_recon_mode
        self.vis_env_mode: str = vis_env_mode
        self.vis_combined_mode: str = vis_combined_mode
        self.vis_on_unsolved: bool = vis_on_unsolved
        self.include_start_goal_header: bool = include_start_goal_header
        self.header_start_title: str = header_start_title
        self.header_goal_title: str = header_goal_title
        self.env_row_title: str = env_row_title
        self.recon_row_title: str = recon_row_title
        self.fps: int = fps
        # sizing
        self.indiv_size_mode: str = indiv_size_mode
        self.indiv_target_h: int | None = indiv_target_h
        self.indiv_target_w: int | None = indiv_target_w
        self.combined_row_h: int | None = combined_row_h
        self.combined_row_w: int | None = combined_row_w

    @property
    def is_noop(self) -> bool:
        """Report whether every visualization mode is disabled.

        Returns:
            ``True`` when the context requests no output.
        """
        return self.vis_recon_mode == "none" and self.vis_env_mode == "none" and self.vis_combined_mode == "none"
