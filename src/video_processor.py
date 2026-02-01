"""
Video Processing Module

Handles the complete pipeline:
1. Video loading and frame extraction
2. Watermark detection and mask generation
3. Inpainting each frame
4. Video reconstruction with audio
"""

import cv2
import numpy as np
import subprocess
import tempfile
import shutil
import os
from pathlib import Path
from typing import Optional, Generator, Tuple, Callable
from tqdm import tqdm

from .detector import SoraWatermarkDetector, WatermarkDetector
from .inpainter import get_inpainter, BaseInpainter


class VideoProcessor:
    """Processes videos to remove watermarks."""

    def __init__(
        self,
        detector: Optional[WatermarkDetector] = None,
        inpainter: Optional[BaseInpainter] = None,
        inpaint_method: str = "opencv",
        template_path: Optional[str] = None,
    ):
        """
        Initialize the video processor.

        Args:
            detector: Watermark detector instance (creates default if None)
            inpainter: Inpainter instance (creates default if None)
            inpaint_method: Inpainting method if inpainter not provided
            template_path: Path to watermark template image
        """
        self.detector = detector or SoraWatermarkDetector(template_path)
        self.inpainter = inpainter or get_inpainter(inpaint_method)
        self.temp_dir = None

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        mask_padding: int = 20,
        preview_mask: bool = False,
    ) -> bool:
        """
        Process a video to remove watermarks.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            progress_callback: Optional callback(current_frame, total_frames)
            mask_padding: Padding around detected watermark
            preview_mask: If True, show mask overlay instead of inpainting

        Returns:
            True if successful, False otherwise
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return False

        # Create temp directory for frames
        self.temp_dir = tempfile.mkdtemp(prefix="watermark_remover_")
        frames_dir = Path(self.temp_dir) / "frames"
        processed_dir = Path(self.temp_dir) / "processed"
        frames_dir.mkdir()
        processed_dir.mkdir()

        try:
            # Get video properties
            video_info = self._get_video_info(str(input_path))
            if not video_info:
                print("Error: Could not read video properties")
                return False

            fps, width, height, total_frames, has_audio = video_info
            print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
            print(f"Audio: {'Yes' if has_audio else 'No'}")

            # Process frames
            print("\nProcessing frames...")
            cap = cv2.VideoCapture(str(input_path))

            frame_idx = 0
            pbar = tqdm(total=total_frames, desc="Removing watermark")

            # For caching the last known watermark position
            last_mask = None
            frames_without_detection = 0
            max_frames_to_reuse_mask = 30  # Reuse mask for up to 1 second at 30fps

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect watermark
                mask = self.detector.detect(frame, padding=mask_padding)

                # If no watermark detected, use last known position (watermarks are persistent)
                if not mask.any() and last_mask is not None and frames_without_detection < max_frames_to_reuse_mask:
                    mask = last_mask
                    frames_without_detection += 1
                elif mask.any():
                    last_mask = mask.copy()
                    frames_without_detection = 0

                # Process frame
                if mask.any():
                    if preview_mask:
                        # Show mask as red overlay for debugging
                        processed = frame.copy()
                        processed[mask > 0] = [0, 0, 255]
                    else:
                        # Inpaint
                        processed = self.inpainter.inpaint(frame, mask)
                else:
                    processed = frame

                # Save processed frame
                frame_path = processed_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), processed)

                frame_idx += 1
                pbar.update(1)

                if progress_callback:
                    progress_callback(frame_idx, total_frames)

            pbar.close()
            cap.release()

            print(f"\nProcessed {frame_idx} frames")

            # Reconstruct video
            print("\nReconstructing video...")
            success = self._reconstruct_video(
                processed_dir,
                str(input_path),
                str(output_path),
                fps,
                width,
                height,
                has_audio
            )

            if success:
                print(f"\nOutput saved to: {output_path}")
                return True
            else:
                print("Error: Failed to reconstruct video")
                return False

        finally:
            # Cleanup temp files
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

    def _get_video_info(self, video_path: str) -> Optional[Tuple[float, int, int, int, bool]]:
        """Get video properties using OpenCV and ffprobe."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Check for audio using ffprobe
        has_audio = self._check_audio(video_path)

        return fps, width, height, total_frames, has_audio

    def _check_audio(self, video_path: str) -> bool:
        """Check if video has audio stream."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    video_path
                ],
                capture_output=True,
                text=True
            )
            return "audio" in result.stdout.lower()
        except FileNotFoundError:
            print("Warning: ffprobe not found. Assuming video has audio.")
            return True

    def _reconstruct_video(
        self,
        frames_dir: Path,
        original_video: str,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        has_audio: bool
    ) -> bool:
        """Reconstruct video from processed frames using FFmpeg."""
        try:
            frame_pattern = str(frames_dir / "frame_%06d.png")

            if has_audio:
                # Combine frames with original audio
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", frame_pattern,
                    "-i", original_video,
                    "-map", "0:v",  # Video from frames
                    "-map", "1:a?",  # Audio from original (if exists)
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",  # High quality
                    "-pix_fmt", "yuv420p",  # YouTube compatible
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-movflags", "+faststart",  # Web optimization
                    output_path
                ]
            else:
                # No audio
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", frame_pattern,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    output_path
                ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False

            return True

        except FileNotFoundError:
            print("Error: FFmpeg not found. Please install FFmpeg.")
            return False

    def process_frame(self, frame: np.ndarray, mask_padding: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single frame.

        Args:
            frame: Input frame (BGR)
            mask_padding: Padding around detected watermark

        Returns:
            Tuple of (processed_frame, mask)
        """
        mask = self.detector.detect(frame, padding=mask_padding)

        if mask.any():
            processed = self.inpainter.inpaint(frame, mask)
        else:
            processed = frame

        return processed, mask

    def preview_detection(self, video_path: str, frame_number: int = 0) -> Optional[np.ndarray]:
        """
        Preview watermark detection on a specific frame.

        Args:
            video_path: Path to video
            frame_number: Frame number to preview

        Returns:
            Frame with detection overlay, or None if failed
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        mask = self.detector.detect(frame, padding=20)

        # Create visualization
        vis = frame.copy()
        # Red overlay on detected regions
        vis[mask > 0] = vis[mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5

        # Draw bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

        return vis


def remove_watermark(
    input_path: str,
    output_path: Optional[str] = None,
    method: str = "opencv",
    mask_padding: int = 20,
    template_path: Optional[str] = None,
) -> bool:
    """
    Convenience function to remove watermark from a video.

    Args:
        input_path: Path to input video
        output_path: Path for output video (default: input_cleaned.mp4)
        method: Inpainting method ("opencv", "opencv-ns", "lama", "hybrid")
        mask_padding: Padding around detected watermark
        template_path: Optional path to watermark template

    Returns:
        True if successful
    """
    if output_path is None:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_cleaned.mp4")

    processor = VideoProcessor(
        inpaint_method=method,
        template_path=template_path,
    )

    return processor.process_video(
        input_path,
        output_path,
        mask_padding=mask_padding,
    )
