#!/bin/bash
# ChatterBox Pipeline Setup Script

set -e

echo "=============================================="
echo "ChatterBox TTS Pipeline - Setup"
echo "=============================================="

# Check Python version
echo ""
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# Install dependencies
echo ""
echo "2. Installing dependencies..."
pip install -r requirements.txt

# Install ChatterBox from local source if available
if [ -d "../project2-chatterbox-tts" ]; then
    echo ""
    echo "3. Found local ChatterBox source, installing..."
    pip install -e ../project2-chatterbox-tts
else
    echo ""
    echo "3. Installing ChatterBox from pip..."
    pip install chatterbox-tts
fi

# Create directories
echo ""
echo "4. Creating directories..."
mkdir -p input output tests/output config

# Download models (will happen on first run)
echo ""
echo "5. Models will be downloaded automatically on first run"

# Run quick test
echo ""
echo "6. Running quick test..."
python tests/quick_test.py

echo ""
echo "=============================================="
echo "✅ Setup complete!"
echo "=============================================="
echo ""
echo "Quick start:"
echo "  python chatterbox_pipeline.py --text 'Hello world!'"
echo ""
echo "Run tests:"
echo "  python tests/test_multilingual.py"
echo ""
echo "See README.md for full documentation"
echo "=============================================="
