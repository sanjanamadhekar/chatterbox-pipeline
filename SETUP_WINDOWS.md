# Windows Setup Guide - ChatterBox TTS Pipeline

Complete setup instructions for Windows 10/11 with NVIDIA GPU.

---

## Prerequisites

- **Python 3.10+** installed
- **NVIDIA GPU** (for GPU acceleration)
- **NVIDIA Drivers** installed
- **Git** (for cloning repository)

---

## Step 1: Verify GPU and CUDA

```bash
# Check if NVIDIA driver is installed
nvidia-smi
```

You should see your GPU listed (e.g., RTX 6000 Ada). Note the **CUDA Version** shown.

---

## Step 2: Clone Repository

```bash
# Navigate to desired location
cd C:\Users\%USERNAME%

# Clone the repository
git clone https://github.com/sanjanamadhekar/chatterbox-pipeline.git

# Enter directory
cd chatterbox-pipeline
```

---

## Step 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

You should see `(venv)` in your command prompt.

---

## Step 4: Install PyTorch with CUDA

**IMPORTANT:** ChatterBox requires PyTorch 2.6.0 exactly.

**For CUDA 12.4+ (RTX 6000, RTX 4090, etc.):**
```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**For CUDA 12.1:**
```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only (no GPU):**
```bash
pip install torch==2.6.0 torchaudio==2.6.0
```

---

## Step 5: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install langdetect pydub librosa transformers diffusers safetensors

# Install ChatterBox TTS
pip install chatterbox-tts
```

---

## Step 6: Verify CUDA Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**Expected output:**
```
CUDA available: True
GPU: NVIDIA RTX 6000 Ada Generation
```

If you see `CUDA available: False`, PyTorch was installed without CUDA support. Go back to Step 4 and use the correct CUDA index URL.

---

## Step 7: Download Models

Models will download automatically on first run (~2-4GB). Or manually trigger:

```bash
python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cuda')"
python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; ChatterboxMultilingualTTS.from_pretrained(device='cuda')"
```

This will take a few minutes.

---

## Step 8: Run Tests

```bash
# Quick test (1 minute)
python tests\quick_test.py

# Full multilingual test (10-15 minutes)
python tests\test_multilingual.py

# Test specific languages
python tests\test_multilingual.py --languages en es fr
```

---

## Step 9: Use the Pipeline

```bash
# Basic usage
python chatterbox_pipeline.py --text "Hello from Windows!" --format wav

# Specify language
python chatterbox_pipeline.py --text "Hola mundo" --language es --format mp3

# From file
echo This is a test > input.txt
python chatterbox_pipeline.py --input input.txt --output my_output --format wav

# With custom settings
python chatterbox_pipeline.py --text "Advanced test" --device cuda --exaggeration 0.7 --format wav
```

---

## Troubleshooting

### CUDA Not Available

**Problem:** `torch.cuda.is_available()` returns `False`

**Solutions:**
1. Verify GPU drivers: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchaudio -y
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### MP3 Conversion Fails

**Problem:** Error when using `--format mp3`

**Solution:** Install ffmpeg
- Download from: https://ffmpeg.org/download.html
- Or use chocolatey: `choco install ffmpeg`
- Or use winget: `winget install ffmpeg`

### Out of Memory

**Problem:** CUDA out of memory error

**Solution:** Use CPU instead:
```bash
python chatterbox_pipeline.py --text "Test" --device cpu
```

### Models Don't Download

**Problem:** Network/firewall blocking downloads

**Solution:** Check firewall settings or download manually from Hugging Face

---

## Complete Setup Script (Copy-Paste)

```bash
cd C:\Users\%USERNAME%
git clone https://github.com/sanjanamadhekar/chatterbox-pipeline.git
cd chatterbox-pipeline
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install langdetect pydub librosa transformers diffusers safetensors
pip install chatterbox-tts
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python tests\quick_test.py
```

---

## Performance Tips

- **Use GPU:** Always use `--device cuda` for 5-20x speedup
- **Batch processing:** Process multiple files in a loop
- **Output format:** WAV is fastest, MP3 requires conversion

---

## Next Steps

- See [README.md](README.md) for full usage documentation
- See [VOICES.md](VOICES.md) for voice options
- Run comprehensive tests: `python tests\test_multilingual.py`

---

**Ready to use!** Your Windows cluster is now set up for ChatterBox TTS! 🚀
