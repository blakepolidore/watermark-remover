"""
Watermark Detection Module

Detects Sora watermarks in video frames using:
1. Template matching for the Sora logo
2. OCR-based detection for "Sora" text
3. Color-based detection for white semi-transparent overlays
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import os


class WatermarkDetector:
    """Detects Sora watermarks in video frames."""

    def __init__(self, template_path: Optional[str] = None, use_ocr: bool = True):
        """
        Initialize the watermark detector.

        Args:
            template_path: Path to the Sora logo template image (optional)
            use_ocr: Whether to use OCR for text-based detection
        """
        self.template = None
        self.template_scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # Multi-scale matching
        self.use_ocr = use_ocr
        self.ocr_reader = None

        if template_path and os.path.exists(template_path):
            self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        if self.use_ocr:
            self._init_ocr()

    def _init_ocr(self):
        """Initialize OCR reader (lazy loading)."""
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=self._has_gpu())
        except ImportError:
            print("Warning: easyocr not installed. OCR detection disabled.")
            self.use_ocr = False

    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            return False

    def detect(self, frame: np.ndarray, padding: int = 15) -> np.ndarray:
        """
        Detect watermark in a frame and return a binary mask.

        Args:
            frame: BGR image (OpenCV format)
            padding: Extra pixels to add around detected regions

        Returns:
            Binary mask where 255 = watermark region, 0 = clean area
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Method 1: Detect white text/logo regions (color-based)
        white_mask = self._detect_white_overlay(frame)

        # Method 2: OCR-based detection for "Sora" text
        if self.use_ocr and self.ocr_reader:
            ocr_mask = self._detect_via_ocr(frame)
            white_mask = cv2.bitwise_or(white_mask, ocr_mask)

        # Method 3: Template matching (if template provided)
        if self.template is not None:
            template_mask = self._detect_via_template(frame)
            white_mask = cv2.bitwise_or(white_mask, template_mask)

        # Apply padding to the mask
        if padding > 0:
            kernel = np.ones((padding * 2, padding * 2), np.uint8)
            white_mask = cv2.dilate(white_mask, kernel, iterations=1)

        return white_mask

    def _detect_white_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect white semi-transparent overlays typical of watermarks.
        Focuses on the corners where watermarks usually appear.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White color detection (low saturation, high value)
        # Sora watermark is white/light gray
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Also detect light gray
        lower_gray = np.array([0, 0, 180])
        upper_gray = np.array([180, 40, 255])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)

        combined = cv2.bitwise_or(white_mask, gray_mask)

        # Focus on regions where watermarks typically appear (corners)
        # Check all four corners
        corner_size_h = h // 4
        corner_size_w = w // 3

        regions = [
            (0, 0, corner_size_w, corner_size_h),  # Top-left
            (w - corner_size_w, 0, w, corner_size_h),  # Top-right
            (0, h - corner_size_h, corner_size_w, h),  # Bottom-left
            (w - corner_size_w, h - corner_size_h, w, h),  # Bottom-right
        ]

        for x1, y1, x2, y2 in regions:
            roi = combined[y1:y2, x1:x2]

            # Find contours in this region
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                # Filter by area - watermark components are medium-sized
                if 100 < area < (corner_size_h * corner_size_w * 0.5):
                    # Offset contour to full image coordinates
                    contour_offset = contour + np.array([x1, y1])
                    cv2.drawContours(mask, [contour_offset], -1, 255, -1)

        return mask

    def _detect_via_ocr(self, frame: np.ndarray) -> np.ndarray:
        """Detect watermark by finding 'Sora' text using OCR."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if not self.ocr_reader:
            return mask

        try:
            # Run OCR on the frame
            results = self.ocr_reader.readtext(frame)

            for (bbox, text, confidence) in results:
                text_lower = text.lower().strip()
                # Look for "sora" or username patterns "@..."
                if 'sora' in text_lower or text_lower.startswith('@') or confidence > 0.5:
                    # Check if it looks like a watermark (white text)
                    pts = np.array(bbox, dtype=np.int32)

                    # Get the region and check if it's white/light colored
                    x_min, y_min = pts.min(axis=0)
                    x_max, y_max = pts.max(axis=0)

                    # Clamp to image bounds
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)

                    if x_max > x_min and y_max > y_min:
                        roi = frame[y_min:y_max, x_min:x_max]
                        mean_val = roi.mean()

                        # If the region is bright (likely white watermark)
                        if mean_val > 150 or 'sora' in text_lower:
                            # Add generous padding for the whole watermark area
                            pad = 20
                            y1 = max(0, y_min - pad)
                            y2 = min(h, y_max + pad * 3)  # Extra padding below for username
                            x1 = max(0, x_min - pad)
                            x2 = min(w, x_max + pad)
                            mask[y1:y2, x1:x2] = 255
        except Exception as e:
            print(f"OCR detection error: {e}")

        return mask

    def _detect_via_template(self, frame: np.ndarray) -> np.ndarray:
        """Detect watermark using template matching with the Sora logo."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if self.template is None:
            return mask

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        best_match = None
        best_val = 0
        best_scale = 1.0

        # Multi-scale template matching
        for scale in self.template_scales:
            template_resized = cv2.resize(
                self.template,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR
            )

            th, tw = template_resized.shape[:2]
            if th > h or tw > w:
                continue

            result = cv2.matchTemplate(gray, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_match = max_loc
                best_scale = scale

        # If we found a good match (threshold)
        if best_match and best_val > 0.6:
            th, tw = self.template.shape[:2]
            th, tw = int(th * best_scale), int(tw * best_scale)
            x, y = best_match

            # Add padding and extend down for text below logo
            pad = 10
            y1 = max(0, y - pad)
            y2 = min(h, y + th + pad + 50)  # Extra space for text
            x1 = max(0, x - pad)
            x2 = min(w, x + tw + pad + 80)  # Extra space for "Sora" text

            mask[y1:y2, x1:x2] = 255

        return mask

    def detect_watermark_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect and return the bounding box of the watermark.

        Returns:
            Tuple of (x, y, width, height) or None if not found
        """
        mask = self.detect(frame, padding=0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get bounding box of all contours combined
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        return (x, y, w, h)


class SoraWatermarkDetector(WatermarkDetector):
    """
    Specialized detector for Sora watermarks.
    Uses knowledge of Sora watermark characteristics for better detection.
    """

    def __init__(self, template_path: Optional[str] = None):
        super().__init__(template_path, use_ocr=True)

        # Sora-specific detection parameters
        self.sora_logo_color = (255, 255, 255)  # White
        self.watermark_keywords = ['sora', '@']

    def detect(self, frame: np.ndarray, padding: int = 20) -> np.ndarray:
        """
        Detect Sora watermark with enhanced detection.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Try OCR first as it's most reliable for Sora watermarks
        if self.use_ocr and self.ocr_reader:
            ocr_mask = self._detect_sora_text(frame)
            if ocr_mask.any():
                mask = cv2.bitwise_or(mask, ocr_mask)

        # Fallback to color-based detection
        if not mask.any():
            color_mask = self._detect_sora_by_color(frame)
            mask = cv2.bitwise_or(mask, color_mask)

        # Apply padding
        if padding > 0 and mask.any():
            kernel = np.ones((padding, padding), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _detect_sora_text(self, frame: np.ndarray) -> np.ndarray:
        """Detect 'Sora' text and associated watermark elements."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if not self.ocr_reader:
            return mask

        try:
            results = self.ocr_reader.readtext(frame)
            sora_regions = []

            for (bbox, text, confidence) in results:
                text_lower = text.lower().strip()

                # Detect "Sora" text or username
                if 'sora' in text_lower or (text_lower.startswith('@') and len(text_lower) > 1):
                    pts = np.array(bbox, dtype=np.int32)
                    x_min, y_min = pts.min(axis=0)
                    x_max, y_max = pts.max(axis=0)
                    sora_regions.append((x_min, y_min, x_max, y_max))

            if sora_regions:
                # Combine all detected regions into one bounding box
                all_x_min = min(r[0] for r in sora_regions)
                all_y_min = min(r[1] for r in sora_regions)
                all_x_max = max(r[2] for r in sora_regions)
                all_y_max = max(r[3] for r in sora_regions)

                # Add padding for the logo above the text
                logo_padding_top = 50
                logo_padding_left = 60
                text_padding_bottom = 30

                y1 = max(0, all_y_min - logo_padding_top)
                y2 = min(h, all_y_max + text_padding_bottom)
                x1 = max(0, all_x_min - logo_padding_left)
                x2 = min(w, all_x_max + 20)

                mask[y1:y2, x1:x2] = 255

        except Exception as e:
            print(f"OCR error: {e}")

        return mask

    def _detect_sora_by_color(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect Sora watermark by looking for characteristic white elements
        in typical watermark positions.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold for white/bright pixels
        _, bright_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

        # Check corners for clusters of white pixels
        corner_regions = [
            (0, 0, w // 3, h // 4, "top-left"),
            (2 * w // 3, 0, w, h // 4, "top-right"),
            (0, 3 * h // 4, w // 3, h, "bottom-left"),
            (2 * w // 3, 3 * h // 4, w, h, "bottom-right"),
        ]

        for x1, y1, x2, y2, name in corner_regions:
            roi = bright_mask[y1:y2, x1:x2]

            # Find contours
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Look for watermark-like patterns
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 50000:  # Reasonable size for watermark elements
                    x, y, cw, ch = cv2.boundingRect(contour)
                    # Map back to full image coordinates
                    mask[y1+y:y1+y+ch, x1+x:x1+x+cw] = 255

        return mask
