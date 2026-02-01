"""
Inpainting Module

Provides watermark removal through inpainting:
1. OpenCV inpainting (fast, traditional methods)
2. LaMa inpainting (deep learning, higher quality)
"""

import cv2
import numpy as np
from typing import Optional, Literal
from abc import ABC, abstractmethod


class BaseInpainter(ABC):
    """Abstract base class for inpainting methods."""

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint the masked regions of an image.

        Args:
            image: BGR image (OpenCV format)
            mask: Binary mask where 255 = region to inpaint

        Returns:
            Inpainted image
        """
        pass


class OpenCVInpainter(BaseInpainter):
    """
    OpenCV-based inpainting using traditional algorithms.
    Fast and works well for smaller watermarks.
    """

    def __init__(
        self,
        method: Literal["telea", "ns"] = "telea",
        radius: int = 5
    ):
        """
        Initialize OpenCV inpainter.

        Args:
            method: Inpainting algorithm - "telea" (fast) or "ns" (Navier-Stokes, smoother)
            radius: Radius of circular neighborhood for inpainting
        """
        self.method = method
        self.radius = radius

        if method == "telea":
            self.flags = cv2.INPAINT_TELEA
        else:
            self.flags = cv2.INPAINT_NS

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using OpenCV's algorithms."""
        # Ensure mask is single channel and uint8
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)

        # Apply inpainting
        result = cv2.inpaint(image, mask, self.radius, self.flags)

        return result


class LamaInpainter(BaseInpainter):
    """
    LaMa (Large Mask Inpainting) - deep learning based inpainting.
    Higher quality results, especially for larger masked regions.
    Requires PyTorch.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize LaMa inpainter.

        Args:
            device: Device to run on ("cuda", "mps", "cpu", or None for auto-detect)
        """
        self.device = device or self._detect_device()
        self.model = None
        self._model_loaded = False

    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self):
        """Lazy load the LaMa model."""
        if self._model_loaded:
            return

        try:
            import torch
            from .lama_model import LamaModel

            self.model = LamaModel(self.device)
            self._model_loaded = True
            print(f"LaMa model loaded on {self.device}")
        except ImportError as e:
            print(f"Failed to load LaMa model: {e}")
            print("Falling back to OpenCV inpainting")
            self.model = None

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using LaMa model."""
        self._load_model()

        if self.model is None:
            # Fallback to OpenCV
            fallback = OpenCVInpainter()
            return fallback.inpaint(image, mask)

        return self.model.inpaint(image, mask)


class SimpleInpainter(BaseInpainter):
    """
    Simple inpainting that uses surrounding pixel information.
    Very fast but lower quality. Good for small watermarks.
    """

    def __init__(self, blur_size: int = 15):
        self.blur_size = blur_size

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Simple blur-based inpainting."""
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Expand mask slightly
        kernel = np.ones((5, 5), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=2)

        # Create a blurred version of the image
        blurred = cv2.GaussianBlur(image, (self.blur_size * 2 + 1, self.blur_size * 2 + 1), 0)

        # Blend based on mask
        mask_3ch = cv2.cvtColor(expanded_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0

        result = (image * (1 - mask_3ch) + blurred * mask_3ch).astype(np.uint8)

        return result


class HybridInpainter(BaseInpainter):
    """
    Hybrid approach: Uses OpenCV for initial inpainting,
    then applies smoothing for better visual results.
    """

    def __init__(self, method: str = "telea", radius: int = 5):
        self.opencv_inpainter = OpenCVInpainter(method=method, radius=radius)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Hybrid inpainting with post-processing."""
        # First pass: OpenCV inpainting
        result = self.opencv_inpainter.inpaint(image, mask)

        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Create feathered mask for blending
        feather_size = 5
        feathered_mask = cv2.GaussianBlur(mask.astype(float), (feather_size * 2 + 1, feather_size * 2 + 1), 0)
        feathered_mask = feathered_mask / 255.0

        # Slight blur on inpainted regions for smoothness
        blurred_result = cv2.GaussianBlur(result, (3, 3), 0)

        # Blend
        feathered_3ch = np.stack([feathered_mask] * 3, axis=-1)
        final = (result * (1 - feathered_3ch * 0.3) + blurred_result * feathered_3ch * 0.3).astype(np.uint8)

        return final


def get_inpainter(
    method: Literal["opencv", "opencv-ns", "lama", "simple", "hybrid"] = "opencv",
    **kwargs
) -> BaseInpainter:
    """
    Factory function to get an inpainter instance.

    Args:
        method: Inpainting method to use
        **kwargs: Additional arguments passed to the inpainter

    Returns:
        Inpainter instance
    """
    if method == "opencv":
        return OpenCVInpainter(method="telea", **kwargs)
    elif method == "opencv-ns":
        return OpenCVInpainter(method="ns", **kwargs)
    elif method == "lama":
        return LamaInpainter(**kwargs)
    elif method == "simple":
        return SimpleInpainter(**kwargs)
    elif method == "hybrid":
        return HybridInpainter(**kwargs)
    else:
        raise ValueError(f"Unknown inpainting method: {method}")
