# ChatterBox TTS Pipeline

**Barebones multilingual text-to-speech pipeline with automatic language detection**

Supports 23 languages with automatic language detection, GPU acceleration, and multiple output formats (WAV, MP3, PCM/RAW).

---

## Features

✅ **Automatic Language Detection** - Detects language from input text
✅ **23 Languages Supported** - Full multilingual support via ChatterBox
✅ **Multiple Output Formats** - WAV, MP3, PCM/RAW
✅ **GPU Accelerated** - Runs on CUDA-enabled GPUs
✅ **Voice Cloning** - Custom voice reference support
✅ **Comprehensive Tests** - Stress tests for all 23 languages

---

## Supported Languages

Arabic • Danish • German • Greek • English • Spanish • Finnish • French • Hebrew • Hindi • Italian • Japanese • Korean • Malay • Dutch • Norwegian • Polish • Portuguese • Russian • Swedish • Swahili • Turkish • Chinese

---

## Quick Start

### Windows One-Click Setup (Recommended for Windows Users)

**Automated setup and testing - just double-click and run:**

```bash
# Simply run the batch file
run_tests.bat
```

The batch file will automatically:
- ✅ Check Python installation
- ✅ Create virtual environment
- ✅ Install all dependencies (PyTorch with CUDA, ChatterBox TTS, etc.)
- ✅ Run comprehensive tests
- ✅ Generate sample audio files
- ✅ Display results and open output folder

**Requirements:** Python 3.10+ installed and added to PATH

---

### 1. Installation (Manual Setup)

**Linux:**
```bash
cd /home/user/Documents/chatterbox-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch 2.6.0 with CUDA support (for GPU)
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# OR for CPU only
# pip install torch==2.6.0 torchaudio==2.6.0

# Install other dependencies
pip install langdetect pydub librosa transformers diffusers safetensors

# Install ChatterBox TTS
pip install chatterbox-tts
```

**Windows:**
```bash
cd C:\Users\YourUsername\chatterbox-pipeline

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch 2.6.0 with CUDA support (for GPU)
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# OR for CPU only
# pip install torch==2.6.0 torchaudio==2.6.0

# Install other dependencies
pip install langdetect pydub librosa transformers diffusers safetensors

# Install ChatterBox TTS
pip install chatterbox-tts
```

### 2. Download ChatterBox Models

```bash
# Models will be downloaded automatically on first run
# OR manually download to cache:
python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')"
python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; ChatterboxMultilingualTTS.from_pretrained(device='cpu')"
```

### 3. Run Quick Test

```bash
# Verify everything works
python tests/quick_test.py
```

---

## Usage

### 1. Command Line Tool

```bash
# Basic usage (auto-detect language)
python chatterbox_pipeline.py --text "Hello, this is a test!"

# Specify language explicitly
python chatterbox_pipeline.py --text "Bonjour le monde!" --language fr

# From input file
python chatterbox_pipeline.py --input input/mytext.txt

# Custom output path and format
python chatterbox_pipeline.py \
  --text "Hello world" \
  --output output/hello \
  --format mp3

# Use custom voice reference
python chatterbox_pipeline.py \
  --text "Clone this voice" \
  --voice path/to/reference_audio.wav

# Full options
python chatterbox_pipeline.py \
  --text "Advanced usage example" \
  --language en \
  --output output/advanced \
  --format wav \
  --device cuda \
  --exaggeration 0.7 \
  --cfg-weight 0.5 \
  --temperature 0.8
```

### 2. Use as Python Library (Import in Your Code)

```python
from chatterbox_pipeline import ChatterBoxPipeline

# Initialize pipeline
pipeline = ChatterBoxPipeline(device="cuda")

# Generate from text (auto-detect language)
audio, sample_rate = pipeline.generate("Hello world!")

# Generate with specific language
audio, sample_rate = pipeline.generate(
    "Bonjour le monde!",
    language="fr"
)

# Save audio in different formats
pipeline.save_audio(audio, sample_rate, "output/test", format="wav")
pipeline.save_audio(audio, sample_rate, "output/test", format="mp3")
pipeline.save_audio(audio, sample_rate, "output/test", format="raw")

# Process text file
output_file = pipeline.process_file(
    "input/mytext.txt",
    output_path="output/result",
    output_format="wav"
)

# Process text directly
output_file = pipeline.process_text(
    "Direct text input",
    output_path="output/direct",
    output_format="mp3"
)
```

---

## Output Formats

### WAV (Default)
```bash
python chatterbox_pipeline.py --text "Test" --format wav
```
- Standard WAV audio file
- Uncompressed, high quality
- Compatible with all audio players

### MP3
```bash
python chatterbox_pipeline.py --text "Test" --format mp3
```
- Compressed audio (192kbps)
- Smaller file size
- Requires `ffmpeg` or `libav` installed

### PCM/RAW
```bash
python chatterbox_pipeline.py --text "Test" --format raw
```
- Raw 16-bit signed PCM data
- No header, just audio samples
- For custom audio processing pipelines

---

## Testing

### Automated Test Suite (Windows)
```bash
# Run everything automatically - setup, install, and test
run_tests.bat
```

### Quick Test (Manual - 5 tests)
```bash
python tests/quick_test.py
```

Tests:
- English auto-detection
- Spanish explicit language
- MP3 output
- PCM/RAW output

### Comprehensive Multilingual Test Suite

Test **all 23 languages** with challenging sentences:

```bash
# Test all languages (generates ~100+ audio files)
python tests/test_multilingual.py

# Test specific languages only
python tests/test_multilingual.py --languages en es fr de zh ja

# Test with different output format
python tests/test_multilingual.py --format mp3

# Custom output directory
python tests/test_multilingual.py --output my_test_results
```

**What's tested for each language:**
- Numbers and dates (1999, 15:45, $99.99, 15%)
- Punctuation and special characters
- Abbreviations (Dr., WWW, PM, LOL)
- Homographs and pronunciation challenges
- Language-specific tongue twisters
- URL and email-like patterns

**Test Coverage:**
- English: 6 test sentences
- Spanish: 5 test sentences
- French: 5 test sentences
- German: 5 test sentences
- Chinese: 5 test sentences
- Japanese: 5 test sentences
- Korean: 5 test sentences
- And more for all 23 languages...

---

## CLI Options

```
Options:
  --text TEXT              Text to synthesize
  --input FILE            Input text file
  --output PATH           Output file path (default: output/output)
  --format FORMAT         Output format: wav, mp3, raw (default: wav)
  --language CODE         Language code (auto-detected if not specified)
  --device DEVICE         Device: cuda, cpu, mps (default: cuda)
  --exaggeration FLOAT    Exaggeration level 0.25-2.0 (default: 0.5)
  --cfg-weight FLOAT      CFG/Pace weight 0.0-1.0 (default: 0.5)
  --temperature FLOAT     Temperature 0.05-5.0 (default: 0.8)
  --voice FILE            Path to custom voice reference audio
```

---

## Language Codes

| Code | Language   | Code | Language    | Code | Language  |
|------|------------|------|-------------|------|-----------|
| ar   | Arabic     | it   | Italian     | pt   | Portuguese|
| da   | Danish     | ja   | Japanese    | ru   | Russian   |
| de   | German     | ko   | Korean      | sv   | Swedish   |
| el   | Greek      | ms   | Malay       | sw   | Swahili   |
| en   | English    | nl   | Dutch       | tr   | Turkish   |
| es   | Spanish    | no   | Norwegian   | zh   | Chinese   |
| fi   | Finnish    | pl   | Polish      |      |           |
| fr   | French     | pt   | Portuguese  |      |           |
| he   | Hebrew     | ru   | Russian     |      |           |
| hi   | Hindi      | sv   | Swedish     |      |           |

---

## Project Structure

```
chatterbox-pipeline/
├── chatterbox_pipeline.py      # Main pipeline script
├── config/
│   └── language_config.py      # Language mappings and config
├── tests/
│   ├── quick_test.py          # Quick sanity test
│   └── test_multilingual.py   # Comprehensive 23-language tests
├── input/                      # Place input text files here
├── output/                     # Generated audio outputs
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- ChatterBox TTS models (downloaded automatically)
- ~4GB GPU VRAM for model inference

---

## Examples

### Example 1: Simple English
```bash
python chatterbox_pipeline.py \
  --text "Welcome to ChatterBox TTS!" \
  --format wav
```

### Example 2: Spanish with Custom Settings
```bash
python chatterbox_pipeline.py \
  --text "Hola, ¿cómo estás?" \
  --language es \
  --exaggeration 0.7 \
  --format mp3
```

### Example 3: Process Multiple Languages
```bash
# Create input files
echo "Hello world" > input/english.txt
echo "Bonjour le monde" > input/french.txt
echo "Hola mundo" > input/spanish.txt

# Process them
python chatterbox_pipeline.py --input input/english.txt --format wav
python chatterbox_pipeline.py --input input/french.txt --format wav
python chatterbox_pipeline.py --input input/spanish.txt --format wav
```

### Example 4: Batch Test All Languages
```bash
# Run comprehensive tests for quality assurance
python tests/test_multilingual.py --format wav
```

---

## Troubleshooting

### Models not downloading
```bash
# Manually trigger download
python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; ChatterboxMultilingualTTS.from_pretrained(device='cpu')"
```

### MP3 conversion fails
```bash
# Install ffmpeg
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

### CUDA out of memory
```bash
# Use CPU instead
python chatterbox_pipeline.py --text "Test" --device cpu
```

### Language detection fails
```bash
# Specify language explicitly
python chatterbox_pipeline.py --text "..." --language en
```

---

## Performance

**Typical generation times (on RTX GPU):**
- Short sentence (10-20 words): ~1-2 seconds
- Medium paragraph (50-100 words): ~3-5 seconds
- Long text (200+ words): ~8-12 seconds

**Real-Time Factor:** Usually 5-20x faster than audio duration

---

## License

This pipeline is a wrapper around ChatterBox TTS by Resemble AI.

- ChatterBox TTS: MIT License
- This pipeline code: MIT License

---

## Credits

- **ChatterBox TTS** by [Resemble AI](https://resemble.ai)
- Pipeline implementation for production use
