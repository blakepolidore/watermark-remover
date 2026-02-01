#!/bin/bash
# Setup script for Sora Watermark Remover

echo "=================================="
echo "Sora Watermark Remover - Setup"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found. Please install pip."
    exit 1
fi

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "Warning: FFmpeg not found!"
    echo "Please install FFmpeg:"
    echo "  macOS:   brew install ffmpeg"
    echo "  Ubuntu:  sudo apt install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
    echo ""
fi

# Check if Tesseract is installed (needed for some OCR features)
if ! command -v tesseract &> /dev/null; then
    echo ""
    echo "Note: Tesseract OCR not found (optional, but recommended)"
    echo "Install with:"
    echo "  macOS:   brew install tesseract"
    echo "  Ubuntu:  sudo apt install tesseract-ocr"
    echo ""
fi

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment created and activated"
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Usage:"
echo "  python remove_watermark.py input.mp4"
echo ""
echo "For more options:"
echo "  python remove_watermark.py --help"
