#!/usr/bin/env python3
"""
Sora Watermark Remover

A tool to detect and remove Sora watermarks from videos using inpainting.

Usage:
    python remove_watermark.py input.mp4
    python remove_watermark.py input.mp4 -o output.mp4
    python remove_watermark.py input.mp4 --method lama --padding 25

Requirements:
    - Python 3.8+
    - OpenCV
    - FFmpeg (for video encoding)
    - EasyOCR (for watermark detection)
    - PyTorch (optional, for LaMa inpainting)
"""

import argparse
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check that required dependencies are installed."""
    missing = []

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        from tqdm import tqdm
    except ImportError:
        missing.append("tqdm")

    # Check FFmpeg
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
    except FileNotFoundError:
        missing.append("ffmpeg (system package)")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall Python packages with:")
        print("  pip install -r requirements.txt")
        print("\nInstall FFmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        return False

    return True


def preview_frame(input_path: str, frame_num: int = 0, output_path: str = None):
    """Preview watermark detection on a single frame."""
    from src.video_processor import VideoProcessor

    processor = VideoProcessor()
    vis = processor.preview_detection(input_path, frame_num)

    if vis is None:
        print(f"Error: Could not read frame {frame_num} from {input_path}")
        return False

    if output_path:
        import cv2
        cv2.imwrite(output_path, vis)
        print(f"Preview saved to: {output_path}")
    else:
        # Display in window
        import cv2
        cv2.imshow("Watermark Detection Preview", vis)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Remove Sora watermarks from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python remove_watermark.py video.mp4

  Specify output file:
    python remove_watermark.py video.mp4 -o clean_video.mp4

  Use higher quality inpainting (slower):
    python remove_watermark.py video.mp4 --method opencv-ns

  Adjust mask padding (larger = more aggressive removal):
    python remove_watermark.py video.mp4 --padding 30

  Preview detection on first frame:
    python remove_watermark.py video.mp4 --preview

  Preview specific frame:
    python remove_watermark.py video.mp4 --preview --frame 100
        """
    )

    parser.add_argument(
        "input",
        help="Path to input video file (MP4)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: input_cleaned.mp4)"
    )

    parser.add_argument(
        "--method",
        choices=["opencv", "opencv-ns", "hybrid", "simple"],
        default="opencv",
        help="Inpainting method (default: opencv)"
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding around detected watermark in pixels (default: 20)"
    )

    parser.add_argument(
        "--template",
        help="Path to watermark template image for better detection"
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview watermark detection without processing"
    )

    parser.add_argument(
        "--preview-output",
        help="Save preview image to file instead of displaying"
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame number for preview (default: 0)"
    )

    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR-based detection (faster but less accurate)"
    )

    parser.add_argument(
        "--debug-mask",
        action="store_true",
        help="Output video with mask overlay for debugging"
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if not input_path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        print(f"Warning: Unexpected file extension: {input_path.suffix}")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Preview mode
    if args.preview:
        print(f"Previewing watermark detection on frame {args.frame}...")
        success = preview_frame(
            str(input_path),
            args.frame,
            args.preview_output
        )
        sys.exit(0 if success else 1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.parent / f"{input_path.stem}_cleaned.mp4")

    # Check output directory exists
    output_dir = Path(output_path).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    print("=" * 60)
    print("Sora Watermark Remover")
    print("=" * 60)
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path}")
    print(f"Method:   {args.method}")
    print(f"Padding:  {args.padding}px")
    print("=" * 60)

    # Process video
    from src.video_processor import VideoProcessor

    processor = VideoProcessor(
        inpaint_method=args.method,
        template_path=args.template,
    )

    # Disable OCR if requested
    if args.no_ocr:
        processor.detector.use_ocr = False
        processor.detector.ocr_reader = None

    success = processor.process_video(
        str(input_path),
        output_path,
        mask_padding=args.padding,
        preview_mask=args.debug_mask,
    )

    if success:
        print("\n" + "=" * 60)
        print("Processing complete!")
        print(f"Output saved to: {output_path}")
        print("=" * 60)

        # Print file size comparison
        input_size = input_path.stat().st_size / (1024 * 1024)
        output_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\nInput size:  {input_size:.2f} MB")
        print(f"Output size: {output_size:.2f} MB")
    else:
        print("\nProcessing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
