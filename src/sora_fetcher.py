"""
Sora Video Fetcher Module

Fetches original videos directly from Sora share links without watermarks.
This is the fast method - no processing needed, just downloads the source file.

Note: OpenAI now requires authentication for the API. You can provide:
1. A cookies file exported from your browser (Netscape format)
2. Or manually provide cookies as a string
"""

import re
import json
import urllib.request
import urllib.error
import http.cookiejar
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

# Global verbose flag
VERBOSE = False

def set_verbose(enabled: bool):
    """Enable or disable verbose logging."""
    global VERBOSE
    VERBOSE = enabled

def log(message: str):
    """Print a log message if verbose mode is enabled."""
    if VERBOSE:
        print(f"[DEBUG] {message}")


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

    def __init__(
        self,
        user_agent: Optional[str] = None,
        cookies_file: Optional[str] = None,
        cookies_string: Optional[str] = None,
    ):
        """
        Initialize the Sora fetcher.

        Args:
            user_agent: Custom user agent string (optional)
            cookies_file: Path to cookies file in Netscape format (from browser export)
            cookies_string: Raw cookie string (e.g., "__Secure-next-auth.session-token=xxx")
        """
        self.user_agent = user_agent or "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        self.cookies_string = cookies_string
        self.cookie_jar = None

        # Load cookies from file if provided
        if cookies_file and Path(cookies_file).exists():
            self._load_cookies_from_file(cookies_file)

    def _load_cookies_from_file(self, cookies_file: str):
        """Load cookies from a Netscape format cookies file."""
        try:
            self.cookie_jar = http.cookiejar.MozillaCookieJar(cookies_file)
            self.cookie_jar.load(ignore_discard=True, ignore_expires=True)
            log(f"Loaded {len(self.cookie_jar)} cookies from {cookies_file}")
        except Exception as e:
            log(f"Failed to load cookies file: {e}")
            self.cookie_jar = None

    def _get_cookie_header(self) -> Optional[str]:
        """Get the Cookie header value for requests."""
        if self.cookies_string:
            return self.cookies_string

        if self.cookie_jar:
            # Extract cookies for sora.chatgpt.com
            cookies = []
            for cookie in self.cookie_jar:
                if "chatgpt.com" in cookie.domain or "openai.com" in cookie.domain:
                    cookies.append(f"{cookie.name}={cookie.value}")
            if cookies:
                return "; ".join(cookies)

        return None

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
            log(f"Input URL: {url_or_id}")
            video_id = self.extract_video_id(url_or_id)
            if not video_id:
                raise ValueError(f"Could not extract video ID from URL: {url_or_id}")
            log(f"Extracted video ID: {video_id}")
        else:
            video_id = url_or_id
            log(f"Using direct video ID: {video_id}")

        # Build API URL
        api_url = f"{self.API_BASE}/{video_id}"
        log(f"API URL: {api_url}")

        # Build request headers
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://sora.chatgpt.com/",
            "Origin": "https://sora.chatgpt.com",
        }

        # Add cookies if available
        cookie_header = self._get_cookie_header()
        if cookie_header:
            headers["Cookie"] = cookie_header
            log(f"Using cookies (length: {len(cookie_header)} chars)")
        else:
            log("No cookies provided - request may fail with 401")

        log(f"Request headers: {json.dumps({k: v[:50] + '...' if len(str(v)) > 50 else v for k, v in headers.items()}, indent=2)}")

        # Make request
        request = urllib.request.Request(api_url, headers=headers)

        try:
            log("Sending request...")
            with urllib.request.urlopen(request, timeout=30) as response:
                status = response.status
                resp_headers = dict(response.headers)
                body = response.read().decode("utf-8")

                log(f"Response status: {status}")
                log(f"Response headers: {json.dumps(resp_headers, indent=2)}")
                log(f"Response body (first 500 chars): {body[:500]}")

                data = json.loads(body)

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except:
                pass

            log(f"HTTP Error: {e.code} {e.reason}")
            log(f"Error headers: {dict(e.headers) if e.headers else 'None'}")
            log(f"Error body: {error_body[:500] if error_body else 'Empty'}")

            if e.code == 404:
                raise ValueError(f"Video not found: {video_id}")
            elif e.code == 401:
                raise ConnectionError(
                    f"Authentication required (HTTP 401).\n"
                    f"  OpenAI now requires login to access Sora videos.\n\n"
                    f"  To authenticate, provide cookies from your browser:\n"
                    f"  1. Log into sora.chatgpt.com in Chrome/Firefox\n"
                    f"  2. Open Developer Tools (F12) > Application > Cookies\n"
                    f"  3. Copy the cookie values and use --cookies flag\n\n"
                    f"  Example:\n"
                    f"    python remove_watermark.py URL --cookies \"__Secure-next-auth.session-token=YOUR_TOKEN\"\n\n"
                    f"  Or export cookies to a file using a browser extension and use --cookies-file"
                )
            elif e.code == 403:
                raise ConnectionError(
                    f"Access denied (HTTP 403). The video may be private.\n"
                    f"  API URL: {api_url}\n"
                    f"  Response: {error_body[:200] if error_body else 'No response body'}"
                )
            else:
                raise ConnectionError(
                    f"API request failed with status {e.code}: {e.reason}\n"
                    f"  API URL: {api_url}\n"
                    f"  Response: {error_body[:200] if error_body else 'No response body'}"
                )
        except urllib.error.URLError as e:
            log(f"URL Error: {e.reason}")
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
