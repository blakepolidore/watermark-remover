#!/usr/bin/env python3
"""
Sora Watermark Remover

A tool to remove Sora watermarks from videos.

Two methods available:
1. FAST (URL mode): Fetch original video directly from Sora link (~5 seconds)
2. INPAINT (File mode): Process local video frame-by-frame (minutes)

Usage:
    # Fast method - from Sora link (recommended)
    python remove_watermark.py https://sora.chatgpt.com/p/s_xxxxx

    # Inpainting method - from local file
    python remove_watermark.py video.mp4
    python remove_watermark.py video.mp4 --method opencv-ns

Requirements:
    - Python 3.8+
    - For URL mode: No additional dependencies
    - For file mode: OpenCV, FFmpeg, EasyOCR
"""

import argparse
import sys
import os
import re
from pathlib import Path


def is_sora_url(input_str: str) -> bool:
    """Check if input is a Sora URL."""
    return bool(re.search(r"sora\.chatgpt\.com", input_str))


def check_dependencies_for_inpainting():
    """Check that required dependencies for inpainting are installed."""
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
        print("Missing dependencies for inpainting mode:")
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


def download_from_url(url: str, output_path: str = None, quality: str = "source") -> bool:
    """
    Download video directly from Sora URL (fast method).

    This fetches the original video without watermark from OpenAI's CDN.
    """
    from src.sora_fetcher import SoraFetcher

    print("=" * 60)
    print("Sora Watermark Remover - FAST MODE")
    print("=" * 60)
    print(f"URL: {url}")
    print("Method: Direct CDN fetch (no processing needed)")
    print("=" * 60)

    fetcher = SoraFetcher()

    # First, get video info
    print("\nFetching video information...")
    try:
        info = fetcher.fetch_video_info(url)
        print(f"Title: {info.title}")
        print(f"Resolution: {info.width}x{info.height}")
        print(f"Duration: {info.duration}s")
        print(f"Frames: {info.frame_count}")
    except ValueError as e:
        print(f"\nError: {e}")
        return False
    except ConnectionError as e:
        print(f"\nConnection error: {e}")
        print("\nThis might mean:")
        print("  - The video is private or has been deleted")
        print("  - OpenAI has changed their API")
        print("  - Network connectivity issues")
        print("\nTry downloading the video manually and using inpainting mode:")
        print(f"  python remove_watermark.py video.mp4")
        return False

    # Generate output filename if not provided
    if output_path is None:
        safe_title = re.sub(r'[^\w\s-]', '', info.title)[:50].strip()
        if not safe_title:
            safe_title = info.video_id
        output_path = f"{safe_title}_no_watermark.mp4"

    # Download
    print(f"\nDownloading {quality} quality...")

    def progress(downloaded, total):
        pct = (downloaded / total) * 100 if total else 0
        mb_down = downloaded / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        print(f"\r  Progress: {mb_down:.1f}/{mb_total:.1f} MB ({pct:.1f}%)", end="", flush=True)

    try:
        path, _ = fetcher.download_video(url, output_path, quality, progress_callback=progress)
        print()  # New line after progress
    except ConnectionError as e:
        print(f"\n\nDownload failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"Output: {path}")

    # Show file size
    file_size = Path(path).stat().st_size / (1024 * 1024)
    print(f"Size: {file_size:.2f} MB")
    print(f"Quality: Original HD (no watermark, no compression)")

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


def process_local_file(args) -> bool:
    """Process a local video file using inpainting."""
    input_path = Path(args.input)

    # Check dependencies for inpainting
    if not check_dependencies_for_inpainting():
        sys.exit(1)

    # Preview mode
    if args.preview:
        print(f"Previewing watermark detection on frame {args.frame}...")
        success = preview_frame(
            str(input_path),
            args.frame,
            args.preview_output
        )
        return success

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
    print("Sora Watermark Remover - INPAINTING MODE")
    print("=" * 60)
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path}")
    print(f"Method:   {args.method}")
    print(f"Padding:  {args.padding}px")
    print("=" * 60)
    print("\nNote: This processes each frame individually.")
    print("For faster results, use a Sora share link instead.")
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

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Remove Sora watermarks from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two modes available:

  FAST MODE (Sora URL):
    Fetches original video directly from OpenAI's CDN - no watermark, no processing.

    python remove_watermark.py https://sora.chatgpt.com/p/s_xxxxx
    python remove_watermark.py https://sora.chatgpt.com/p/s_xxxxx -o output.mp4

  INPAINTING MODE (Local file):
    Processes each frame to detect and remove watermark using inpainting.

    python remove_watermark.py video.mp4
    python remove_watermark.py video.mp4 --method opencv-ns
    python remove_watermark.py video.mp4 --preview

Examples:
  # Download from Sora link (fast, ~5 seconds)
  python remove_watermark.py "https://sora.chatgpt.com/p/s_697f9e9b209c81918c9c1d6f076d2b79"

  # Process local file (slower, uses inpainting)
  python remove_watermark.py video.mp4 -o clean_video.mp4

  # Preview detection before processing
  python remove_watermark.py video.mp4 --preview

  # Higher quality inpainting
  python remove_watermark.py video.mp4 --method opencv-ns --padding 30
        """
    )

    parser.add_argument(
        "input",
        help="Sora URL (https://sora.chatgpt.com/p/...) or path to local video file"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: auto-generated)"
    )

    parser.add_argument(
        "--quality",
        choices=["source", "md", "ld"],
        default="source",
        help="Video quality for URL mode: source (HD), md (medium), ld (low). Default: source"
    )

    # Inpainting options (only for local file mode)
    inpaint_group = parser.add_argument_group("Inpainting options (local file mode only)")

    inpaint_group.add_argument(
        "--method",
        choices=["opencv", "opencv-ns", "hybrid", "simple"],
        default="opencv",
        help="Inpainting method (default: opencv)"
    )

    inpaint_group.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding around detected watermark in pixels (default: 20)"
    )

    inpaint_group.add_argument(
        "--template",
        help="Path to watermark template image for better detection"
    )

    inpaint_group.add_argument(
        "--preview",
        action="store_true",
        help="Preview watermark detection without processing"
    )

    inpaint_group.add_argument(
        "--preview-output",
        help="Save preview image to file instead of displaying"
    )

    inpaint_group.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame number for preview (default: 0)"
    )

    inpaint_group.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR-based detection (faster but less accurate)"
    )

    inpaint_group.add_argument(
        "--debug-mask",
        action="store_true",
        help="Output video with mask overlay for debugging"
    )

    args = parser.parse_args()

    # Determine mode based on input
    if is_sora_url(args.input):
        # URL mode - fast download
        success = download_from_url(args.input, args.output, args.quality)
    else:
        # File mode - inpainting
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            print("\nIf you meant to use a Sora URL, make sure it starts with https://")
            sys.exit(1)

        if not input_path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            print(f"Warning: Unexpected file extension: {input_path.suffix}")

        success = process_local_file(args)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
