# Sora Watermark Remover

Remove Sora watermarks from videos - two methods available:

| Method | Speed | How it works |
|--------|-------|--------------|
| **URL Mode** | ~5 seconds | Fetches original video directly from OpenAI's CDN |
| **File Mode** | 5-30 minutes | Processes local video frame-by-frame with inpainting |

## Quick Start

```bash
# Fast method - from Sora share link (recommended)
python remove_watermark.py "https://sora.chatgpt.com/p/s_xxxxx"

# Slow method - from local file (when URL doesn't work)
python remove_watermark.py video.mp4
```

## Installation

### For URL Mode (Fast)

No additional dependencies needed - just Python 3.8+.

### For File Mode (Inpainting)

```bash
# Install system dependencies
brew install ffmpeg tesseract  # macOS
# or
sudo apt install ffmpeg tesseract-ocr  # Ubuntu

# Install Python packages
pip install -r requirements.txt
```

## Usage

### Method 1: URL Mode (Fast) - Recommended

Fetches the original video directly from OpenAI's CDN without the watermark overlay.

```bash
# Basic usage
python remove_watermark.py "https://sora.chatgpt.com/p/s_697f9e9b209c81918c9c1d6f076d2b79"

# Specify output file
python remove_watermark.py "https://sora.chatgpt.com/p/s_xxxxx" -o my_video.mp4

# Download medium quality (smaller file)
python remove_watermark.py "https://sora.chatgpt.com/p/s_xxxxx" --quality md
```

**Quality options:**
- `source` - Original HD quality (default)
- `md` - Medium quality (smaller file)
- `ld` - Low quality (smallest file)

### Method 2: File Mode (Inpainting)

For when you have a local video file (e.g., downloaded with watermark, screen recorded).

```bash
# Basic usage
python remove_watermark.py video.mp4

# Specify output
python remove_watermark.py video.mp4 -o clean_video.mp4

# Preview detection before processing
python remove_watermark.py video.mp4 --preview

# Higher quality inpainting (slower)
python remove_watermark.py video.mp4 --method opencv-ns

# Increase removal area if watermark not fully removed
python remove_watermark.py video.mp4 --padding 30
```

**Inpainting methods:**
| Method | Speed | Quality |
|--------|-------|---------|
| `opencv` | Fast | Good (default) |
| `opencv-ns` | Medium | Better |
| `hybrid` | Medium | Better |
| `simple` | Very Fast | Basic |

## How It Works

### URL Mode

1. Extracts video ID from Sora share link
2. Calls OpenAI's public API: `https://sora.chatgpt.com/backend/public/generations/{id}`
3. Gets direct CDN URL for the original `source` quality video
4. Downloads the file - no watermark, no processing, full quality

This works because OpenAI stores the original video separately from the watermarked version displayed publicly.

### File Mode

1. Loads local video and extracts frames
2. Detects watermark using OCR ("Sora" text) and color detection (white overlay)
3. Creates mask around detected watermark region
4. Applies inpainting algorithm to fill in the masked area
5. Reconstructs video with original audio

## Troubleshooting

### URL Mode Issues

**"Connection error" or "Access denied"**
- The video may be private or deleted
- OpenAI may have changed their API
- Try downloading the video manually and use file mode

**"Could not extract video ID"**
- Make sure you're using a Sora share link
- Format should be: `https://sora.chatgpt.com/p/s_xxxxx` or `https://sora.chatgpt.com/g/gen_xxxxx`

### File Mode Issues

**"Watermark not fully removed"**
- Increase padding: `--padding 30` or `--padding 40`
- Try different method: `--method opencv-ns`

**"Too much being removed"**
- Decrease padding: `--padding 10`
- Use `--preview` to check detection

**"OCR not detecting watermark"**
- Make sure EasyOCR is installed: `pip install easyocr`
- The watermark may be too small or obscured

## Output Format

Both modes output:
- **Container**: MP4
- **Video Codec**: H.264 (libx264) for file mode, original for URL mode
- **Audio**: Preserved from original
- **Compatibility**: YouTube, Instagram, TikTok, Twitter

## API Reference

The tool can also be used as a Python library:

```python
# URL mode
from src.sora_fetcher import fetch_sora_video

path, info = fetch_sora_video("https://sora.chatgpt.com/p/s_xxxxx")
print(f"Downloaded: {path}")
print(f"Resolution: {info.width}x{info.height}")

# File mode
from src.video_processor import remove_watermark

remove_watermark("input.mp4", "output.mp4", method="opencv")
```

## Project Structure

```
watermark-remover/
├── remove_watermark.py      # Main CLI tool
├── requirements.txt         # Python dependencies
├── setup.sh                # Setup script
├── src/
│   ├── sora_fetcher.py     # URL mode - fetches from Sora CDN
│   ├── detector.py         # Watermark detection (OCR + color)
│   ├── inpainter.py        # Inpainting algorithms
│   └── video_processor.py  # Video processing pipeline
├── templates/              # Watermark templates (optional)
└── output/                 # Default output directory
```

## Credits

- URL extraction method inspired by [SoraChatGPTDownloader](https://github.com/AzozzALFiras/SoraChatGPTDownloader)
- Inpainting uses OpenCV's implementation of Telea and Navier-Stokes algorithms

## License

MIT License - Use freely for personal projects.

## Disclaimer

This tool is for personal use on your own content. The URL mode accesses publicly available API endpoints. Respect OpenAI's terms of service and content ownership rights.
