"""Resize, concatenate, encode, and reconstruct RGB float32 images.

The processors accept RGB arrays in the range [0, 1]. Alignment models encode
the resized images, and trained decoders reconstruct those encodings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from logging import DEBUG, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
from numpy import float32
from PIL import Image
from pillow_heif import register_heif_opener
from rich.panel import Panel
import torch

from spar.utils.pytorch_utils.nnet_utils import load_model

if TYPE_CHECKING:
    from logging import Logger

    from cv2.typing import MatLike
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy import uint8
    from numpy.typing import NDArray
    from PIL.Image import Image as PILImageType
    from torch import Tensor, nn

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.utils.config_utils.config_schema import ProcessImageSPARConfig


mpl.use("Agg")
logger: Logger = getLogger(__name__)


def insert_suffix_before_ext(path: str, suffix: str) -> str:
    """Insert a suffix before the file extension of path.

    Examples:
        insert_suffix_before_ext('a/b/foo.png', '_matplotlib') -> 'a/b/foo_matplotlib.png'
    """
    p: Path = Path(path)
    stem: str = p.stem
    suffix_ext: str = p.suffix
    parent: Path = p.parent
    new_name: str = f"{stem}{suffix}{suffix_ext}"
    return str(parent / new_name)


class ImageProcessor(Protocol):
    """Protocol for image processing strategies."""

    def process_images(
        self, image1: NDArray[float32], image2: NDArray[float32] | None, target_height: int, target_width: int
    ) -> NDArray[float32]:
        """Process images according to the specific strategy.

        Args:
            image1: First image (H, W, C) in RGB order, float32 [0, 1]
            image2: Optional second image (H, W, C) in RGB order, float32 [0, 1]
            target_height: Target height for output
            target_width: Target width for output (per image if concatenating)

        Returns:
            Processed image (H, W, C) in RGB order, float32 [0, 1]
        """
        ...


def _imread_color(path: str) -> MatLike | None:
    """Typed wrapper for cv2.imread in color mode, returning None on failure."""
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _read_image_bgr(path: str) -> MatLike:
    """Read an image into BGR format with optional HEIC support."""
    image_bgr: MatLike | None = _imread_color(path)
    if image_bgr is not None:
        return image_bgr

    try:
        register_heif_opener()
    except Exception as exc:
        logger.warning(f"Failed to register HEIF opener for '{path}': {exc}")

    with Image.open(path) as img:
        rgb_array: NDArray[uint8] = np.array(img.convert("RGB"))

    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


class BaseImageProcessor(ABC):
    """Abstract base class for image processing strategies."""

    @abstractmethod
    def process_images(
        self, image1: NDArray[float32], image2: NDArray[float32] | None, target_height: int, target_width: int
    ) -> NDArray[float32]:
        """Process images according to the specific strategy.

        Args:
            image1: First image (H, W, C) in RGB order, float32 [0, 1]
            image2: Optional second image (H, W, C) in RGB order, float32 [0, 1]
            target_height: Target height for output
            target_width: Target width for output (per image if concatenating)

        Returns:
            Processed image (H, W, C) in RGB order, float32 [0, 1]
        """

    @staticmethod
    def concatenate_images(image1: NDArray[float32], image2: NDArray[float32]) -> NDArray[float32]:
        """Concatenate two RGB float32 images horizontally.

        Args:
            image1: First image (H, W, C) in RGB order, float32 [0, 1]
            image2: Second image (H, W, C) in RGB order, float32 [0, 1]

        Returns:
            Concatenated image (H, W1+W2, C) in RGB order, float32 [0, 1]
        """
        if image1.ndim != 3 or image2.ndim != 3:
            raise ValueError(f"Both images must be 3D arrays. Got shapes: {image1.shape}, {image2.shape}")
        if image1.dtype != float32 or image2.dtype != float32:
            raise ValueError(f"Images must be float32. Got dtypes: {image1.dtype}, {image2.dtype}")

        # Resize larger image to match smaller image dimensions while preserving aspect ratio
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        # Determine which image is larger by total pixel count
        if h1 * w1 > h2 * w2:
            # Image1 is larger, resize it to match image2's dimensions
            # Convert to uint8 for cv2.resize, then back to float32
            image1_uint8 = (image1 * 255).astype(np.uint8)
            image1_resized_uint8 = cv2.resize(image1_uint8, (w2, h2), interpolation=cv2.INTER_NEAREST)
            image1 = (image1_resized_uint8 / 255.0).astype(np.float32)
        elif h2 * w2 > h1 * w1:
            # Image2 is larger, resize it to match image1's dimensions
            # Convert to uint8 for cv2.resize, then back to float32
            image2_uint8 = (image2 * 255).astype(np.uint8)
            image2_resized_uint8 = cv2.resize(image2_uint8, (w1, h1), interpolation=cv2.INTER_NEAREST)
            image2 = (image2_resized_uint8 / 255.0).astype(np.float32)
        # Equal-size images already share spatial dimensions.

        # Horizontal concatenation requires matching channel counts.
        if image1.shape[2] != image2.shape[2]:
            raise ValueError(f"Images must have same number of channels: {image1.shape[2]} vs {image2.shape[2]}")

        # Concatenate horizontally
        return np.concatenate([image1, image2], axis=1)


class CV2ImageProcessor(BaseImageProcessor):
    """Image processor using OpenCV for resizing, working with RGB float32 images."""

    def __init__(self, interpolation: str = "LANCZOS") -> None:
        """Initialize with interpolation method.

        Args:
            interpolation: Interpolation method ('NEAREST', 'LINEAR', 'CUBIC', 'LANCZOS')
        """
        self.interpolation: str = interpolation

        # Map interpolation string to CV2 constant
        self._interpolation_map: dict[str, int] = {
            "NEAREST": cv2.INTER_NEAREST,
            "LINEAR": cv2.INTER_LINEAR,
            "CUBIC": cv2.INTER_CUBIC,
            "LANCZOS": cv2.INTER_LANCZOS4,
        }

    def process_images(
        self, image1: NDArray[float32], image2: NDArray[float32] | None, target_height: int, target_width: int
    ) -> NDArray[float32]:
        """Process images with RGB float32 format.

        Args:
            image1: First image (H, W, C) in RGB order, float32 [0, 1]
            image2: Optional second image (H, W, C) in RGB order, float32 [0, 1]
            target_height: Target height for output
            target_width: Target width for output (per image if concatenating)

        Returns:
            Processed image (H, W, C) in RGB order, float32 [0, 1]
        """
        if image2 is not None:
            # Resize each image individually to target dimensions, then concatenate
            image1_resized: NDArray[float32] = self._resize_image(image1, target_height, target_width)
            image2_resized: NDArray[float32] = self._resize_image(image2, target_height, target_width)
            return BaseImageProcessor.concatenate_images(image1_resized, image2_resized)
        # Single image: resize to target dimensions
        return self._resize_image(image1, target_height, target_width)

    def _resize_image(self, image: NDArray[float32], target_height: int, target_width: int) -> NDArray[float32]:
        """Resize a single RGB float32 image.

        Args:
            image: Input image (H, W, C) in RGB order, float32 [0, 1]
            target_height: Target height
            target_width: Target width

        Returns:
            Resized image (H, W, C) in RGB order, float32 [0, 1]
        """
        interpolation: str
        if self.interpolation not in self._interpolation_map:
            logger.warning(f"Unknown interpolation method '{self.interpolation}', using LANCZOS")
            interpolation = "LANCZOS"
        else:
            interpolation = self.interpolation

        cv2_interpolation: int = self._interpolation_map[interpolation]

        # Validate input format
        if image.dtype != float32:
            raise ValueError(f"Expected float32 image, got {image.dtype}")
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3D image array (H, W, C), got shape {image.shape}")

        # Resize - CV2 expects values in [0, 1] for float32
        resized: MatLike = cv2.resize(image, (target_width, target_height), interpolation=cv2_interpolation)

        # Return float32 values clipped to [0, 1].
        return np.clip(resized, 0.0, 1.0).astype(np.float32)


class MatplotlibImageProcessor(BaseImageProcessor):
    """Image processor using matplotlib for layout and resizing, working with RGB float32 images."""

    def process_images(
        self, image1: NDArray[float32], image2: NDArray[float32] | None, target_height: int, target_width: int
    ) -> NDArray[float32]:
        """Process images with RGB float32 format using matplotlib.

        Args:
            image1: First image (H, W, C) in RGB order, float32 [0, 1]
            image2: Optional second image (H, W, C) in RGB order, float32 [0, 1]
            target_height: Target height for output
            target_width: Target width for output (per image if concatenating)

        Returns:
            Processed image (H, W, C) in RGB order, float32 [0, 1]
        """
        # Validate input format
        if image1.dtype != float32:
            raise ValueError(f"Expected float32 image1, got {image1.dtype}")
        if len(image1.shape) != 3:
            raise ValueError(f"Expected 3D image1 array (H, W, C), got shape {image1.shape}")

        if image2 is not None:
            if image2.dtype != float32:
                raise ValueError(f"Expected float32 image2, got {image2.dtype}")
            if len(image2.shape) != 3:
                raise ValueError(f"Expected 3D image2 array (H, W, C), got shape {image2.shape}")

            # Resize each image individually to target dimensions
            image1_resized: NDArray[float32] = MatplotlibImageProcessor._resize_single_image(
                image1, target_height, target_width
            )
            image2_resized: NDArray[float32] = MatplotlibImageProcessor._resize_single_image(
                image2, target_height, target_width
            )

            # Concatenate the resized images
            combined_image: NDArray[float32] = BaseImageProcessor.concatenate_images(image1_resized, image2_resized)
            return combined_image

        # Single image: resize to target dimensions
        return MatplotlibImageProcessor._resize_single_image(image1, target_height, target_width)

    @staticmethod
    def _resize_single_image(image_rgb: NDArray[float32], target_height: int, target_width: int) -> NDArray[float32]:
        """Resize a single RGB float32 image using matplotlib.

        Args:
            image_rgb: Input image (H, W, C) in RGB order, float32 [0, 1]
            target_height: Target height
            target_width: Target width

        Returns:
            Resized image (H, W, C) in RGB order, float32 [0, 1]
        """
        dpi = 300
        fig_width, fig_height = target_width / dpi, target_height / dpi
        fig: Figure = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        ax: Axes = fig.add_axes((0, 0, 1, 1), frameon=False)
        ax.axis("off")

        # Show the image, letting imshow do the interpolation for us
        # Matplotlib expects floating image values in [0, 1].
        image_clipped: NDArray[float32] = np.clip(image_rgb, 0.0, 1.0)
        ax.imshow(image_clipped, interpolation="bilinear")

        # Render to canvas and grab the raw RGB buffer
        canvas: FigureCanvasAgg = FigureCanvasAgg(fig)
        canvas.draw()
        buf: memoryview[int] = canvas.buffer_rgba()
        width, height = canvas.get_width_height()
        plt.close(fig)

        # Convert to numpy array (H, W, C) format, dropping alpha channel
        img_uint8: NDArray[uint8] = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

        # Convert back to float32 [0, 1] range
        img: NDArray[float32] = img_uint8.astype(np.float32) / 255.0

        # Crop a renderer size mismatch to the requested dimensions.
        if img.shape[:2] != (target_height, target_width):
            logger.warning(
                f"Matplotlib rendered size {img.shape[:2]} doesn't match target {(target_height, target_width)}. "
                "Cropping/padding to exact size."
            )

            # Calculate copy region (crop if too big, pad if too small)
            h_copy: int = min(img.shape[0], target_height)
            w_copy: int = min(img.shape[1], target_width)

            # Copy the available data
            img = img[:h_copy, :w_copy, :]

        logger.info(f"Shape of processed image: {img.shape}\nMin: {np.min(img)}, Max: {np.max(img)}")

        # Return in (H, W, C) float32 [0, 1] RGB format
        return img


class NeuralNetworkProcessor:
    """Handles neural network operations for encoding and reconstruction."""

    def __init__(self, device: torch.device, alignment_model_type: str) -> None:
        """Initialize with device.

        Args:
            device: PyTorch device to run inference on
            alignment_model_type: Type of alignment model (discrete or continuous)
        """
        self.device: torch.device = device
        self.decoder: nn.Module | None = None
        self.alignment_model: nn.Module | None = None
        self.alignment_model_type: str = alignment_model_type

    def load_models(
        self,
        decoder_path: str | None,
        alignment_model_path: str | None,
        env: ABCEnvironment[ABCState],
        cfg: ProcessImageSPARConfig,
    ) -> None:
        """Load pretrained decoder and alignment model.

        Args:
            decoder_path: Path to decoder checkpoint
            alignment_model_path: Path to alignment model checkpoint
            env: Environment instance to get model architectures
            cfg: Configuration for model creation

        Raises:
            ValueError: If any of the required paths are None
        """
        if decoder_path is None:
            raise ValueError("decoder_path cannot be None")
        if alignment_model_path is None:
            raise ValueError("alignment_model_path cannot be None")

        # Get model architectures from environment based on model type
        decoder: nn.Module = (
            env.get_decoder_disc(cfg.model)
            if cfg.alignment_model_type == "discrete"
            else env.get_decoder_cont(cfg.model)
        )

        alignment_model: nn.Module = env.get_alignment_model(cfg.model)

        self.decoder = load_model(model=decoder, device=self.device, pretrained_path=decoder_path, freeze=True)
        self.alignment_model = load_model(
            model=alignment_model, device=self.device, pretrained_path=alignment_model_path, freeze=True
        )
        self.decoder.eval()
        self.alignment_model.eval()

    @staticmethod
    def preprocess_image(image: NDArray[float32], device: torch.device) -> Tensor:
        """Preprocess RGB float32 image for neural network input.

        Args:
            image: Input image (H, W, C) in RGB order, float32 [0, 1]
            device: Device to move the tensor to

        Returns:
            Preprocessed tensor (N, C, H, W) for neural network
        """
        # Validate input format
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3D image array (H, W, C), got shape {image.shape}")
        if image.dtype != float32:
            raise ValueError(f"Expected float32 image, got {image.dtype}")

        logger.info(
            f"Preprocessing image with shape: {image.shape}, dtype: {image.dtype} "
            f"and min/max: {np.min(image)}/{np.max(image)}"
        )

        # Clamp reconstructed values to [0, 1].
        image_float: NDArray[float32] = np.clip(image, 0.0, 1.0)

        # Convert from HWC (NumPy/RGB format) to CHW (PyTorch channel-first format)
        image_chw: NDArray[float32] = np.transpose(image_float, (2, 0, 1))

        # Convert to tensor and add batch dimension (N, C, H, W)
        image_tensor: Tensor = torch.from_numpy(image_chw).unsqueeze(0).to(device)
        return image_tensor

    @staticmethod
    def postprocess_image(image_tensor: Tensor) -> NDArray[float32]:
        """Postprocess neural network output to RGB float32 image.

        Args:
            image_tensor: Output tensor (N, C, H, W) or (C, H, W) from neural network

        Returns:
            Postprocessed image (H, W, C) in RGB order, float32 [0, 1]
        """
        # Validate input tensor format
        if len(image_tensor.shape) not in {3, 4}:
            raise ValueError(f"Expected 3D or 4D tensor (C, H, W) or (N, C, H, W), got shape {image_tensor.shape}")

        # Remove batch dimension if present (N, C, H, W) -> (C, H, W)
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.squeeze(0)

        # Convert from CHW (PyTorch channel-first) to HWC (NumPy/RGB format)
        image_hwc: Tensor = image_tensor.permute(1, 2, 0)

        # Convert to a float32 NumPy image in [0, 1].
        image_np: NDArray[float32] = np.clip(image_hwc.detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)

        return image_np

    def encode_and_reconstruct(self, image_tensor: Tensor) -> tuple[Tensor, Tensor]:
        """Encode image to discrete representation and reconstruct.

        Args:
            image_tensor: Input tensor (N, C, H, W) from preprocess_image

        Returns:
            Tuple of (discrete_encoding, reconstructed_tensor)
        """
        if self.decoder is None or self.alignment_model is None:
            raise RuntimeError("Models must be loaded before encoding. Call load_models() first.")

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            if logger.isEnabledFor(DEBUG):
                logger.debug(
                    f"Encoding image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}\n"
                    f"Using alignment model: {self.alignment_model.__class__.__name__}\n"
                    f"Decoder model: {self.decoder.__class__.__name__}\n"
                    f"The max value of the image tensor is: {image_tensor.max().item()}\n"
                    f"The min value of the image tensor is: {image_tensor.min().item()}"
                )

            # Use alignment model to get encoding
            alignment_preds: Tensor = self.alignment_model(image_tensor)
            encoding: Tensor = (
                torch.round(alignment_preds) if self.alignment_model_type == "discrete" else alignment_preds
            )

            if self.alignment_model_type == "discrete" and logger.isEnabledFor(DEBUG):
                logger.debug(
                    f"Discrete encoding shape: {encoding.shape}, dtype: {encoding.dtype}\n"
                    f"Discrete encoding max value: {encoding.max().item()}\n"
                    f"Discrete encoding min value: {encoding.min().item()}\n"
                    f"Discrete encoding mean value: {encoding.mean().item()}\n"
                    f"Has any values other than 0 or 1: {torch.any((encoding != 0) & (encoding != 1)).item()}"
                )

            # Reconstruct image from discrete encoding
            reconstructed = self.decoder(encoding)

        return encoding, reconstructed


class ImageSaver:
    """Handles saving images and metadata to files."""

    @staticmethod
    def save_image(image: NDArray[float32], output_path: str) -> None:
        """Save RGB float32 image to file.

        Args:
            image: Image (H, W, C) in RGB order, float32 [0, 1]
            output_path: Path to save the image
        """
        # Validate input format
        if image.dtype != float32:
            raise ValueError(f"Expected float32 image, got {image.dtype}")
        if len(image.shape) not in {2, 3}:
            raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")

        # Clamp to [0, 1] and convert to uint8 [0, 255] for saving.
        image_clipped: NDArray[float32] = np.clip(image, 0.0, 1.0)
        image_uint8: NDArray[uint8] = (image_clipped * 255.0).astype(np.uint8)

        # Save using PIL (PIL expects RGB format)
        pil_image: PILImageType = Image.fromarray(image_uint8)
        pil_image.save(output_path)

    @staticmethod
    def save_encoding_info(discrete_encoding: Tensor, output_path: str) -> None:
        """Save discrete encoding information to file.

        Args:
            discrete_encoding: The discrete encoding tensor
            output_path: Path to save the encoding info
        """
        with Path(output_path).open("w", encoding="utf-8") as f:
            f.write(f"Discrete Encoding Shape: {discrete_encoding.shape}\n")
            f.write(f"Discrete Encoding Min: {discrete_encoding.min().item():.6f}\n")
            f.write(f"Discrete Encoding Max: {discrete_encoding.max().item():.6f}\n")
            f.write(f"Discrete Encoding Mean: {discrete_encoding.mean().item():.6f}\n")
            f.write(f"Discrete Encoding Std: {discrete_encoding.std().item():.6f}\n")
            # Count number of ones out of total bits and percentage
            num_ones = int((discrete_encoding == 1).sum().item())
            total_bits = discrete_encoding.numel()
            percent_ones = (num_ones / total_bits) * 100 if total_bits > 0 else 0.0
            f.write(f"Discrete Encoding Ones: {num_ones} out of {total_bits} ({percent_ones:.2f}%)\n")

    @staticmethod
    def create_final_viz(
        original_image_path: str,
        processed_img_path: str,
        reconstruction_image_path: str,
        output_path: str,
        title: str = "SPAR Alignment Model's Result on Real-World Images",
        original_title: str = "Original Image",
        processed_title: str = "Resized Image",
        reconstructed_title: str = "Reconstructed Image",
        subtitle_template: str = (
            "Original Dimensions: {orig_w}x{orig_h}x{orig_c} | "
            "Resized Dimensions: {proc_w}x{proc_h}x{proc_c} | "
            "Reconstruction Dimensions: {recon_w}x{recon_h}x{recon_c}"
        ),
    ) -> None:
        """Create a visualization containing the three primary images.

        Args:
            original_image_path: Path to the original input image.
            processed_img_path: Path to the processed image.
            reconstruction_image_path: Path to the reconstructed image.
            output_path: Path for the visualization.
            title: Title for the visualization.
            original_title: Title for the original image.
            processed_title: Title for the processed image.
            reconstructed_title: Title for the reconstructed image.
            subtitle_template: Subtitle template with dimension placeholders.
        """
        try:
            ImageSaver._create_final_viz_impl(
                original_image_path=original_image_path,
                processed_img_path=processed_img_path,
                reconstruction_image_path=reconstruction_image_path,
                output_path=output_path,
                title=title,
                original_title=original_title,
                processed_title=processed_title,
                reconstructed_title=reconstructed_title,
                subtitle_template=subtitle_template,
            )
        except Exception:
            logger.exception("Error creating detailed comparison visualization")
            raise

    @staticmethod
    def _create_final_viz_impl(
        *,
        original_image_path: str,
        processed_img_path: str,
        reconstruction_image_path: str,
        output_path: str,
        title: str,
        original_title: str,
        processed_title: str,
        reconstructed_title: str,
        subtitle_template: str,
    ) -> None:
        # Load images
        original_bgr: MatLike = _read_image_bgr(original_image_path)
        processed_bgr: MatLike = _read_image_bgr(processed_img_path)
        reconstruction_bgr: MatLike = _read_image_bgr(reconstruction_image_path)

        # Convert BGR to RGB
        original_rgb: MatLike = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        processed_rgb: MatLike = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        reconstruction_rgb: MatLike = cv2.cvtColor(reconstruction_bgr, cv2.COLOR_BGR2RGB)

        # Create figure with clean layout and more padding
        fig: Figure = plt.figure(figsize=(16, 5), dpi=300)
        if title:  # Only show title if not empty
            fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)

        # Create three subplots with more padding
        ax1: Axes = fig.add_subplot(1, 3, 1)
        ax2: Axes = fig.add_subplot(1, 3, 2)
        ax3: Axes = fig.add_subplot(1, 3, 3)

        # Display the three main images with custom titles
        ax1.imshow(original_rgb)
        if original_title:  # Only show title if not empty
            ax1.set_title(original_title, fontsize=14, fontweight="bold", pad=10)
        ax1.axis("off")

        ax2.imshow(processed_rgb)
        if processed_title:  # Only show title if not empty
            ax2.set_title(processed_title, fontsize=14, fontweight="bold", pad=10)
        ax2.axis("off")

        ax3.imshow(reconstruction_rgb)
        if reconstructed_title:  # Only show title if not empty
            ax3.set_title(reconstructed_title, fontsize=14, fontweight="bold", pad=10)
        ax3.axis("off")

        # Add custom subtitle with dimensions and channels if template is not empty
        if subtitle_template:
            info_text: str = subtitle_template.format(
                orig_w=original_rgb.shape[1],
                orig_h=original_rgb.shape[0],
                orig_c=original_rgb.shape[2],
                proc_w=processed_rgb.shape[1],
                proc_h=processed_rgb.shape[0],
                proc_c=processed_rgb.shape[2],
                recon_w=reconstruction_rgb.shape[1],
                recon_h=reconstruction_rgb.shape[0],
                recon_c=reconstruction_rgb.shape[2],
            )
            fig.text(0.5, 0.02, info_text, ha="center", fontsize=10, style="italic")

        # Adjust layout with more padding
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.95, wspace=0.15)

        # Save at the configured 300 DPI resolution.
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.25)
        plt.close(fig)

        logger.info(f"Saved detailed comparison visualization to: {output_path}")


class ImageProcessingPipeline:
    """Main image processing pipeline that orchestrates all components."""

    def __init__(self, env: ABCEnvironment[ABCState], cfg: ProcessImageSPARConfig) -> None:
        """Initialize the pipeline with environment and configuration.

        Args:
            env: Environment instance
            cfg: Configuration for image processing
        """
        self.env: ABCEnvironment[ABCState] = env
        self.cfg: ProcessImageSPARConfig = cfg
        self.device: torch.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.nn_processor = NeuralNetworkProcessor(self.device, cfg.alignment_model_type)
        self.image_saver = ImageSaver()

        # Initialize image processor based on strategy
        self.image_processor: ImageProcessor = self._create_image_processor()

    def _create_image_processor(self) -> ImageProcessor:
        """Create the appropriate image processor based on configuration."""
        processing_method: str = self.cfg.processing.processing_method.lower()

        if processing_method == "matplotlib":
            return MatplotlibImageProcessor()
        if processing_method == "opencv":
            return CV2ImageProcessor(self.cfg.processing.quality_interpolation)
        logger.warning(f"Unknown processing method '{processing_method}', using default matplotlib method")
        return MatplotlibImageProcessor()

    def _load_input_images(self) -> tuple[NDArray[float32], NDArray[float32] | None]:
        """Load input images from configured paths and convert to RGB float32 format.

        Returns:
            Tuple of (image1, image2) where image2 can be None
            Both images are in RGB order, float32 [0, 1] format
        """
        logger.info("Loading input images...")

        # Load first image
        image1_path: str = self.cfg.paths.input_image
        image1_bgr: MatLike = _read_image_bgr(image1_path)

        # Convert BGR to RGB and normalize to float32 [0, 1]
        image1_rgb: MatLike = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)
        image1: NDArray[float32] = image1_rgb.astype(np.float32) / 255.0

        # Load second image if needed
        image2: NDArray[float32] | None = None
        if self.cfg.processing.concatenate and self.cfg.paths.input_image2:
            image2_path: str = self.cfg.paths.input_image2
            image2_bgr: MatLike = _read_image_bgr(image2_path)

            # Convert BGR to RGB and normalize to float32 [0, 1]
            image2_rgb: MatLike = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2RGB)
            image2 = image2_rgb.astype(np.float32) / 255.0

        return image1, image2

    def _process_images(self, image1: NDArray[float32], image2: NDArray[float32] | None) -> NDArray[float32]:
        """Process images using the configured strategy.

        Args:
            image1: First input image
            image2: Optional second input image

        Returns:
            Processed image ready for neural network
        """
        logger.info("Processing images...")
        logger.info(f"Target dimensions: {self.cfg.processing.target_height}x{self.cfg.processing.target_width}")

        final_image: NDArray[float32] = self.image_processor.process_images(
            image1, image2, self.cfg.processing.target_height, self.cfg.processing.target_width
        )

        logger.info(f"Final processed image shape: {final_image.shape}")
        return final_image

    def _run_neural_network_pipeline(self, image: NDArray[float32]) -> tuple[Tensor, NDArray[float32]]:
        """Run the neural network encoding and reconstruction pipeline.

        Args:
            image: Processed input image

        Returns:
            Tuple of (discrete_encoding, reconstructed_image)
        """
        # Load pretrained models
        logger.info("Loading pretrained models...")
        self.nn_processor.load_models(
            self.cfg.pretrained_model_paths.decoder_path,
            self.cfg.pretrained_model_paths.alignment_model_path,
            self.env,
            self.cfg,
        )

        # Preprocess image for model
        logger.info("Preprocessing image for neural network...")
        logger.info(f"Input image shape: {image.shape}, dtype: {image.dtype}")
        image_tensor: Tensor = NeuralNetworkProcessor.preprocess_image(image, self.device)
        logger.info(
            f"Preprocessed tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype} "
            f"(PyTorch channel-first: N, C, H, W)"
        )

        # Encode and reconstruct
        logger.info("Encoding image to discrete representation...")
        discrete_encoding, reconstructed_tensor = self.nn_processor.encode_and_reconstruct(image_tensor)

        # Postprocess reconstruction
        logger.info("Postprocessing reconstruction...")
        logger.info(f"Reconstructed tensor shape: {reconstructed_tensor.shape}, dtype: {reconstructed_tensor.dtype}")
        reconstructed_image = NeuralNetworkProcessor.postprocess_image(reconstructed_tensor)
        logger.info(f"Final reconstructed image shape: {reconstructed_image.shape}, dtype: {reconstructed_image.dtype}")

        return discrete_encoding, reconstructed_image

    def _save_processed_image(self, processed_image: NDArray[float32], suffix: str = "") -> str:
        """Save processed image to file (Phase 1).

        Args:
            processed_image: The processed input image
            suffix: Optional suffix for output filename

        Returns:
            Path to the saved processed image file
        """
        # Create the output directory before saving.
        output_dir = Path(self.cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use configured processed_image path if provided, else default to output_dir/resized_input{suffix}.png
        configured: str | Path | None = getattr(self.cfg.paths, "processed_image", None)
        output_path: Path
        if configured:
            base_path = Path(str(configured))
            # If a suffix is provided, insert before extension
            output_path = Path(insert_suffix_before_ext(str(base_path), suffix)) if suffix else base_path
        else:
            output_path = output_dir / f"resized_input{suffix}.png"

        self.image_saver.save_image(processed_image, str(output_path))
        logger.info(f"Saved processed input to: {output_path}")

        return str(output_path)

    def _save_reconstruction_results(
        self, reconstructed_image: NDArray[float32], discrete_encoding: Tensor, suffix: str = ""
    ) -> None:
        """Save reconstruction results to the output directory (Phase 3).

        Args:
            reconstructed_image: The reconstructed image from neural network
            discrete_encoding: The discrete encoding tensor
            suffix: Optional suffix for output filenames
        """
        # Create the output directory before saving.
        output_dir = Path(self.cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving reconstruction results...")

        # Reconstruction image path (use configured if present)
        configured_recon: str | Path | None = getattr(self.cfg.paths, "reconstruction_image", None)
        recon_path: Path
        if configured_recon:
            recon_path = Path(
                insert_suffix_before_ext(str(configured_recon), suffix) if suffix else str(configured_recon)
            )
        else:
            recon_path = output_dir / f"reconstruction{suffix}.png"

        self.image_saver.save_image(reconstructed_image, str(recon_path))
        logger.info(f"Saved reconstruction to: {recon_path}")

        # Encoding info path
        configured_info: str | Path | None = getattr(self.cfg.paths, "encoding_info", None)
        info_path: Path
        if configured_info:
            info_path = Path(insert_suffix_before_ext(str(configured_info), suffix) if suffix else str(configured_info))
        else:
            info_path = output_dir / f"discrete_encoding_info{suffix}.txt"

        self.image_saver.save_encoding_info(discrete_encoding, str(info_path))
        logger.info(f"Saved encoding info to: {info_path}")

    def _create_final_viz(self, suffix: str = "") -> None:
        """Create the final visualization comparing all images.

        Args:
            suffix: Optional suffix for filenames
        """
        # Setup paths
        output_dir = Path(self.cfg.paths.output_dir)
        # Resolve processed and reconstruction paths (use configured if available)
        processed_cfg: str | Path | None = getattr(self.cfg.paths, "processed_image", None)
        processed_img_path: Path
        if processed_cfg:
            processed_img_path = Path(
                insert_suffix_before_ext(str(processed_cfg), suffix) if suffix else str(processed_cfg)
            )
        else:
            processed_img_path = output_dir / f"resized_input{suffix}.png"

        reconstruction_cfg: str | Path | None = getattr(self.cfg.paths, "reconstruction_image", None)
        reconstruction_image_path: Path
        if reconstruction_cfg:
            reconstruction_image_path = Path(
                insert_suffix_before_ext(str(reconstruction_cfg), suffix) if suffix else str(reconstruction_cfg)
            )
        else:
            reconstruction_image_path = output_dir / f"reconstruction{suffix}.png"

        # Determine the original image to use for visualization
        # If we have concatenation enabled, create the concatenated original
        if self.cfg.processing.concatenate and self.cfg.paths.input_image2:
            # Load and concatenate the original images
            image1_bgr: MatLike = _read_image_bgr(self.cfg.paths.input_image)
            image2_bgr: MatLike = _read_image_bgr(self.cfg.paths.input_image2)

            # Convert to RGB and float32
            image1_rgb: NDArray[float32] = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            image2_rgb: NDArray[float32] = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Concatenate the original images
            concatenated_original: NDArray[float32] = BaseImageProcessor.concatenate_images(image1_rgb, image2_rgb)

            # Save the concatenated original temporarily
            concatenated_original_path: Path = output_dir / f"temp_concatenated_original{suffix}.png"
            ImageSaver.save_image(concatenated_original, str(concatenated_original_path))
            original_image_path = str(concatenated_original_path)
        else:
            # Use single original image
            original_image_path = self.cfg.paths.input_image

        # Create visualization (use configured final_viz if provided)
        final_cfg: str | Path | None = getattr(self.cfg.paths, "final_viz", None)
        final_viz_output_path: Path
        if final_cfg:
            final_viz_output_path = Path(insert_suffix_before_ext(str(final_cfg), suffix) if suffix else str(final_cfg))
        else:
            final_viz_output_path = output_dir / f"real_world_test{suffix}.jpeg"
        self.image_saver.create_final_viz(
            original_image_path=original_image_path,
            processed_img_path=str(processed_img_path),
            reconstruction_image_path=str(reconstruction_image_path),
            output_path=str(final_viz_output_path),
            title=self.cfg.processing.main_title,
            original_title=self.cfg.processing.original_title,
            processed_title=self.cfg.processing.processed_title,
            reconstructed_title=self.cfg.processing.reconstructed_title,
            subtitle_template=self.cfg.processing.subtitle_template,
        )

        # Clean up temporary concatenated original if created
        if self.cfg.processing.concatenate and self.cfg.paths.input_image2:
            concatenated_original_path = Path(original_image_path)
            if concatenated_original_path.exists():
                concatenated_original_path.unlink()

        logger.info("Created the processing pipeline visualization.")

    @staticmethod
    def _load_processed_image(processed_img_path: str) -> NDArray[float32]:
        """Load processed image from file and convert to RGB float32 format (Phase 2).

        Args:
            processed_img_path: Path to the processed image file

        Returns:
            Loaded image as numpy array in RGB order, float32 [0, 1]
        """
        logger.info(f"Loading resized image from: {processed_img_path}")

        image_bgr: MatLike = _read_image_bgr(processed_img_path)

        # Convert BGR to RGB and normalize to float32 [0, 1]
        image_rgb: MatLike = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image: NDArray[float32] = image_rgb.astype(np.float32) / 255.0

        logger.info(f"Loaded resized image shape: {image.shape}, dtype: {image.dtype}")
        return image

    def process_phase1(self) -> str:
        """Run Phase 1: Load and process input images, save as resized_input_*.png.

        Returns:
            Path to the saved processed image file
        """
        try:
            return self._process_phase1_impl()
        except Exception:
            logger.exception("Error during Phase 1 image processing")
            raise

    def _process_phase1_impl(self) -> str:
        # Get processor based on configuration
        image_processor: ImageProcessor = self._create_image_processor()
        method_name: str = type(image_processor).__name__.replace("ImageProcessor", "")

        logger.info(
            Panel(
                f"[bold blue]Starting Phase 1: Image Processing with {method_name}[/bold blue]",
                border_style="blue",
                width=120,
            )
        )

        # Update the image processor
        self.image_processor = image_processor

        # Load input images
        image1: NDArray[float32]
        image2: NDArray[float32] | None
        image1, image2 = self._load_input_images()

        # Process images
        processed_img: NDArray[float32] = self._process_images(image1, image2)

        # Save processed image
        suffix: str = f"_{method_name.lower()}"
        processed_img_path: str = self._save_processed_image(processed_img, suffix)

        logger.info(
            Panel(
                f"[bold green]Phase 1: image processing with {method_name} complete.[/bold green]",
                border_style="green",
                width=120,
            )
        )

        return processed_img_path

    def process_phase2(self, processed_img_path: str | None = None) -> None:
        """Load processed image, run through neural networks, save reconstruction.

        Args:
            processed_img_path: Optional path to processed image. If None, will look for default path.
        """
        try:
            self._process_phase2_impl(processed_img_path)
        except Exception:
            logger.exception("Error during Phase 2 neural network processing")
            raise

    def _process_phase2_impl(self, processed_img_path: str | None = None) -> None:
        # Get processor based on configuration
        image_processor: ImageProcessor = self._create_image_processor()
        method_name: str = type(image_processor).__name__.replace("ImageProcessor", "")

        logger.info(
            Panel("[bold blue]Starting Phase 2: Neural Network Processing[/bold blue]", border_style="blue", width=120)
        )
        logger.info(f"Using device: {self.device}")

        # Determine processed image path. Prefer provided arg, then configured path, then default pattern.
        if processed_img_path is None:
            suffix: str = f"_{method_name.lower()}"
            processed_cfg: str | Path | None = getattr(self.cfg.paths, "processed_image", None)
            if processed_cfg:
                processed_img_path = insert_suffix_before_ext(str(processed_cfg), suffix)
            else:
                output_dir = Path(self.cfg.paths.output_dir)
                processed_img_path = str(output_dir / f"resized_input{suffix}.png")

        # Load processed image from file
        processed_img: NDArray[float32] = ImageProcessingPipeline._load_processed_image(processed_img_path)

        # Run neural network pipeline
        discrete_encoding: Tensor
        reconstructed_image: NDArray[float32]
        discrete_encoding, reconstructed_image = self._run_neural_network_pipeline(processed_img)

        # Save reconstruction results
        suffix = f"_{method_name.lower()}"
        self._save_reconstruction_results(reconstructed_image, discrete_encoding, suffix)

        logger.info("Creating the final visualization...")
        self._create_final_viz(suffix)

        logger.info(
            Panel("[bold green]Neural Network Processing Complete.[/bold green]", border_style="green", width=120)
        )

    def process(self) -> None:
        """Run the complete image processing pipeline (both phases) using configured processing method."""
        # Process and save input images
        processed_img_path: str = self.process_phase1()

        # Load processed images and run neural network pipeline
        self.process_phase2(processed_img_path)


def process_image(env: ABCEnvironment[ABCState], cfg: ProcessImageSPARConfig) -> None:
    """Main image processing function (backward compatibility).

    Args:
        env: Environment instance
        cfg: Configuration for image processing
    """
    pipeline = ImageProcessingPipeline(env, cfg)
    pipeline.process()
