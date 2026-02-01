# Sora Watermark Remover

A command-line tool to detect and remove Sora watermarks from videos using computer vision and inpainting techniques.

## Features

- **Automatic Detection**: Detects Sora watermarks using OCR and color-based detection
- **Multiple Inpainting Methods**: Choose between fast (OpenCV) or high-quality options
- **Audio Preservation**: Maintains original audio track
- **YouTube Ready**: Outputs MP4 with H.264 encoding, compatible with all major platforms
- **Free**: Uses only open-source tools, no API costs

## Requirements

### System Dependencies

- **Python 3.8+**
- **FFmpeg** (for video encoding)
- **Tesseract OCR** (optional, improves detection)

Install system dependencies:

```bash
# macOS
brew install ffmpeg tesseract

# Ubuntu/Debian
sudo apt install ffmpeg tesseract-ocr

# Windows
# Download FFmpeg from https://ffmpeg.org/download.html
# Download Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

Or run the setup script:

```bash
./setup.sh
```

## Usage

### Basic Usage

```bash
python remove_watermark.py video.mp4
```

This will create `video_cleaned.mp4` in the same directory.

### Specify Output File

```bash
python remove_watermark.py video.mp4 -o clean_video.mp4
```

### Preview Detection

Before processing, preview what the tool detects as watermark:

```bash
# Display preview window
python remove_watermark.py video.mp4 --preview

# Save preview to file
python remove_watermark.py video.mp4 --preview --preview-output detection.png

# Preview a specific frame
python remove_watermark.py video.mp4 --preview --frame 100
```

### Inpainting Methods

| Method | Speed | Quality | Description |
|--------|-------|---------|-------------|
| `opencv` | Fast | Good | Default. Telea algorithm |
| `opencv-ns` | Medium | Better | Navier-Stokes algorithm |
| `hybrid` | Medium | Better | OpenCV with post-processing |
| `simple` | Very Fast | Basic | Blur-based (for small watermarks) |

```bash
# Use Navier-Stokes (smoother results)
python remove_watermark.py video.mp4 --method opencv-ns

# Use hybrid approach
python remove_watermark.py video.mp4 --method hybrid
```

### Adjust Detection Sensitivity

If the watermark isn't fully removed, increase the padding:

```bash
python remove_watermark.py video.mp4 --padding 30
```

If too much of the image is being removed, decrease it:

```bash
python remove_watermark.py video.mp4 --padding 10
```

### Debug Mode

To see exactly what's being detected:

```bash
python remove_watermark.py video.mp4 --debug-mask -o debug_output.mp4
```

This outputs a video with red overlay showing detected watermark regions.

### Faster Processing (Skip OCR)

If detection is working well, skip OCR for faster processing:

```bash
python remove_watermark.py video.mp4 --no-ocr
```

## How It Works

1. **Detection**: Each frame is analyzed to find the Sora watermark:
   - OCR looks for "Sora" text and @username patterns
   - Color detection finds white/light overlays in corners
   - Optional template matching for the Sora logo

2. **Mask Generation**: A binary mask is created covering the watermark area with configurable padding

3. **Inpainting**: The masked region is filled in using surrounding pixel information:
   - OpenCV algorithms analyze nearby textures and colors
   - The watermark area is seamlessly replaced

4. **Reconstruction**: Processed frames are combined with original audio into a new MP4 file

## Processing Time

On a MacBook Pro:
- **OpenCV method**: ~5-15 minutes for a 20-second video
- **Higher quality methods**: ~15-30 minutes for same video

Factors affecting speed:
- Video resolution (4K takes longer than 1080p)
- Video length
- Inpainting method chosen
- Whether OCR is enabled

## Troubleshooting

### "Watermark not detected"

1. Try increasing padding: `--padding 30`
2. Make sure OCR is enabled (don't use `--no-ocr`)
3. Use `--preview` to see what's being detected

### "Too much is being removed"

1. Decrease padding: `--padding 10`
2. Use `--preview` to check detection accuracy

### "FFmpeg not found"

Install FFmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`

### "easyocr import error"

```bash
pip install easyocr
```

### Slow processing

1. Use `--no-ocr` if detection is working
2. Use `--method simple` for fastest (lower quality)
3. Consider processing on a machine with GPU

## Output Format

The tool outputs:
- **Container**: MP4
- **Video Codec**: H.264 (libx264)
- **Audio Codec**: AAC (192kbps)
- **Quality**: CRF 18 (high quality)
- **Compatibility**: YouTube, Instagram, TikTok, Twitter

## License

MIT License - Use freely for personal projects.

## Disclaimer

This tool is for personal use on your own content. Respect copyright and terms of service of content platforms.
