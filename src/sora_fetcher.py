"""
Sora Video Fetcher Module

Fetches original videos directly from Sora share links without watermarks.
This is the fast method - no processing needed, just downloads the source file.
"""

import re
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class SoraVideoInfo:
    """Information about a Sora video."""
    video_id: str
    title: str
    width: int
    height: int
    duration: float
    frame_count: int
    source_url: str
    medium_url: Optional[str] = None
    low_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    created_at: Optional[str] = None


class SoraFetcher:
    """Fetches original Sora videos from share links."""

    # API endpoint for public generations
    API_BASE = "https://sora.chatgpt.com/backend/public/generations"

    # Patterns for extracting Sora IDs from different URL formats
    URL_PATTERNS = [
        # Share link format: https://sora.chatgpt.com/p/s_xxx
        r"sora\.chatgpt\.com/p/([a-zA-Z0-9_]+)",
        # Generation format: https://sora.chatgpt.com/g/gen_xxx
        r"sora\.chatgpt\.com/g/(gen_[a-zA-Z0-9_]+)",
        # Direct ID format: https://sora.chatgpt.com/generations/xxx
        r"sora\.chatgpt\.com/generations/([a-zA-Z0-9_]+)",
    ]

    def __init__(self, user_agent: Optional[str] = None):
        """
        Initialize the Sora fetcher.

        Args:
            user_agent: Custom user agent string (optional)
        """
        self.user_agent = user_agent or "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract the Sora video ID from a URL.

        Args:
            url: Sora share link or generation URL

        Returns:
            Video ID or None if not found
        """
        for pattern in self.URL_PATTERNS:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # Fallback: try to get the last path segment
        parts = url.rstrip("/").split("/")
        if parts:
            last_part = parts[-1]
            # Validate it looks like an ID
            if re.match(r"^[a-zA-Z0-9_]+$", last_part) and len(last_part) > 10:
                return last_part

        return None

    def fetch_video_info(self, url_or_id: str) -> SoraVideoInfo:
        """
        Fetch video information from Sora.

        Args:
            url_or_id: Either a Sora URL or a video ID

        Returns:
            SoraVideoInfo object with video details

        Raises:
            ValueError: If the URL/ID is invalid
            ConnectionError: If the API request fails
        """
        # Extract ID if URL provided
        if url_or_id.startswith("http"):
            video_id = self.extract_video_id(url_or_id)
            if not video_id:
                raise ValueError(f"Could not extract video ID from URL: {url_or_id}")
        else:
            video_id = url_or_id

        # Build API URL
        api_url = f"{self.API_BASE}/{video_id}"

        # Make request
        request = urllib.request.Request(
            api_url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            }
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(f"Video not found: {video_id}")
            elif e.code == 403:
                raise ConnectionError(f"Access denied. The video may be private or the API may have changed.")
            else:
                raise ConnectionError(f"API request failed with status {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Network error: {e.reason}")

        # Parse response
        return self._parse_video_info(video_id, data)

    def _parse_video_info(self, video_id: str, data: Dict[str, Any]) -> SoraVideoInfo:
        """Parse API response into SoraVideoInfo object."""
        encodings = data.get("encodings", {})

        # Get source (highest quality, no watermark)
        source = encodings.get("source", {})
        source_url = source.get("path") or source.get("url")

        if not source_url:
            # Try alternative structure
            if "video_url" in data:
                source_url = data["video_url"]
            elif "url" in data:
                source_url = data["url"]
            else:
                raise ValueError("Could not find video URL in API response")

        # Get other quality levels
        medium = encodings.get("md", {})
        low = encodings.get("ld", {})
        thumbnail = encodings.get("thumbnail", {})

        # Get dimensions
        dimensions = data.get("dimensions", {})
        width = dimensions.get("width", source.get("width", 0))
        height = dimensions.get("height", source.get("height", 0))

        # Get duration and frame count
        duration = data.get("duration", source.get("duration", 0))
        frame_count = data.get("frame_count", source.get("frame_count", 0))

        return SoraVideoInfo(
            video_id=video_id,
            title=data.get("title", "Untitled"),
            width=width,
            height=height,
            duration=duration,
            frame_count=frame_count,
            source_url=source_url,
            medium_url=medium.get("path") or medium.get("url"),
            low_url=low.get("path") or low.get("url"),
            thumbnail_url=thumbnail.get("path") or thumbnail.get("url"),
            created_at=data.get("created_at"),
        )

    def download_video(
        self,
        url_or_id: str,
        output_path: Optional[str] = None,
        quality: str = "source",
        progress_callback=None,
    ) -> Tuple[str, SoraVideoInfo]:
        """
        Download a Sora video.

        Args:
            url_or_id: Sora URL or video ID
            output_path: Output file path (auto-generated if None)
            quality: Quality level - "source", "md", or "ld"
            progress_callback: Optional callback(downloaded_bytes, total_bytes)

        Returns:
            Tuple of (output_path, video_info)
        """
        # Get video info
        info = self.fetch_video_info(url_or_id)

        # Select quality
        if quality == "source":
            video_url = info.source_url
        elif quality == "md":
            video_url = info.medium_url or info.source_url
        elif quality == "ld":
            video_url = info.low_url or info.medium_url or info.source_url
        else:
            raise ValueError(f"Invalid quality: {quality}. Use 'source', 'md', or 'ld'")

        # Generate output path if not provided
        if output_path is None:
            safe_title = re.sub(r'[^\w\s-]', '', info.title)[:50].strip()
            if not safe_title:
                safe_title = info.video_id
            output_path = f"{safe_title}_no_watermark.mp4"

        output_path = Path(output_path)

        # Download the video
        print(f"Downloading from: {video_url[:80]}...")

        request = urllib.request.Request(
            video_url,
            headers={"User-Agent": self.user_agent}
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 8192

                with open(output_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size:
                            progress_callback(downloaded, total_size)

        except urllib.error.HTTPError as e:
            raise ConnectionError(f"Download failed with status {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Download failed: {e.reason}")

        return str(output_path), info


def fetch_sora_video(
    url: str,
    output_path: Optional[str] = None,
    quality: str = "source",
) -> Tuple[str, SoraVideoInfo]:
    """
    Convenience function to fetch a Sora video.

    Args:
        url: Sora share link
        output_path: Output file path (optional)
        quality: Quality level - "source" (HD, no watermark), "md", or "ld"

    Returns:
        Tuple of (output_path, video_info)

    Example:
        path, info = fetch_sora_video("https://sora.chatgpt.com/p/s_xxx")
        print(f"Downloaded: {path} ({info.width}x{info.height})")
    """
    fetcher = SoraFetcher()
    return fetcher.download_video(url, output_path, quality)


if __name__ == "__main__":
    # Quick test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sora_fetcher.py <sora_url>")
        sys.exit(1)

    url = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        path, info = fetch_sora_video(url, output)
        print(f"\nSuccess!")
        print(f"Title: {info.title}")
        print(f"Resolution: {info.width}x{info.height}")
        print(f"Duration: {info.duration}s")
        print(f"Saved to: {path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
