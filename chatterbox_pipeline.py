#!/usr/bin/env python3
"""
ChatterBox TTS Pipeline - Production Text-to-Speech System

A production-ready multilingual text-to-speech pipeline built on ChatterBox TTS by Resemble AI.
Provides automatic language detection, voice cloning, and multiple output formats for 23 languages.

Features:
    - Automatic language detection using langdetect
    - Support for 23 languages (see SUPPORTED_LANGUAGES in config/language_config.py)
    - Multiple output formats: WAV, MP3, PCM/RAW
    - GPU acceleration (CUDA, MPS) with CPU fallback
    - Zero-shot voice cloning from reference audio
    - Voice file caching for Windows compatibility

Architecture:
    - Uses ChatterboxTTS for English (higher quality)
    - Uses ChatterboxMultilingualTTS for all other languages
    - Downloads and caches remote voice files locally for cross-platform compatibility

Usage:
    Command line:
        python chatterbox_pipeline.py --text "Hello world" --format wav

    Python library:
        from chatterbox_pipeline import ChatterBoxPipeline
        pipeline = ChatterBoxPipeline()
        audio, sr = pipeline.generate("Hello world")

Author: Sanjana Madhekar
License: MIT
"""

import os
import sys
import warnings

# Suppress deprecation warnings from third-party libraries
warnings.filterwarnings('ignore', category=FutureWarning, module='diffusers')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

import torch
import torchaudio as ta
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from langdetect import detect
from pydub import AudioSegment
import urllib.request
import hashlib

# Set transformers to use eager attention to avoid SDPA warnings
os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'

# Add config directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))
from language_config import (
    LANGDETECT_TO_CHATTERBOX,
    get_language_name,
    get_default_voice,
    is_supported
)

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


# =============================================================================
# Voice File Management
# =============================================================================

def download_voice_file(url: str, cache_dir: str = "voices") -> Optional[str]:
    """
    Download voice file from URL to local cache directory.

    This function handles downloading remote voice files and caching them locally
    to solve Windows compatibility issues where librosa cannot load files directly
    from URLs. Files are named using MD5 hash of the URL to avoid duplicates.

    Configuration:
        cache_dir: Directory where voice files are cached (default: "voices/")
                  Can be changed to use a different cache location.

    Args:
        url: Full URL of the voice file to download (must be HTTP/HTTPS)
        cache_dir: Local directory to store cached voice files

    Returns:
        str: Absolute path to the cached voice file on success
        None: If download fails or encounters an error

    Example:
        >>> voice_path = download_voice_file("https://example.com/voice.flac")
        >>> print(voice_path)
        voices/a3f2e1b9c8d7f6e5.flac

    Note:
        - Files are only downloaded once and reused on subsequent calls
        - Uses MD5 hash of URL as filename to handle special characters
        - Creates cache_dir automatically if it doesn't exist
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Generate unique filename from URL using MD5 hash
    # This avoids issues with special characters in URLs and prevents duplicates
    url_hash = hashlib.md5(url.encode()).hexdigest()
    filename = f"{url_hash}.flac"
    cache_path = os.path.join(cache_dir, filename)

    # Download only if not already cached
    if not os.path.exists(cache_path):
        print(f"Downloading voice file from URL...")
        try:
            urllib.request.urlretrieve(url, cache_path)
            print(f"Cached voice file: {cache_path}")
        except Exception as e:
            print(f"WARNING: Failed to download voice file: {e}")
            return None

    return cache_path


def get_voice_path(audio_prompt_path: Optional[str]) -> Optional[str]:
    """
    Convert voice file path/URL to local file path, downloading if necessary.

    Handles both local file paths and remote URLs transparently. For URLs,
    downloads and caches the file locally. For local paths, returns them as-is.
    This ensures cross-platform compatibility, especially on Windows.

    Args:
        audio_prompt_path: Either a local file path or HTTP/HTTPS URL to voice file

    Returns:
        str: Local file path to the voice file (either original or cached)
        None: If input is None or download fails

    Example:
        >>> # URL input - downloads and caches
        >>> path = get_voice_path("https://example.com/voice.flac")
        >>> # Local path input - returns as-is
        >>> path = get_voice_path("/local/voice.wav")

    Note:
        - Automatically detects URLs by checking for "http://" or "https://" prefix
        - Falls back to download_voice_file() for remote URLs
        - Returns None if download fails (caller should handle this)
    """
    if audio_prompt_path is None:
        return None

    # Check if input is a URL (starts with http:// or https://)
    if audio_prompt_path.startswith("http://") or audio_prompt_path.startswith("https://"):
        # Download and cache the file locally
        return download_voice_file(audio_prompt_path)

    # Otherwise it's already a local path - return as-is
    return audio_prompt_path


# =============================================================================
# Main Pipeline Class
# =============================================================================

class ChatterBoxPipeline:
    """
    Main text-to-speech pipeline with automatic language detection.

    This class provides a high-level interface for converting text to speech across
    23 languages. It automatically detects the input language, selects the appropriate
    model (English-only or multilingual), and generates natural-sounding speech with
    optional voice cloning.

    Architecture:
        - Lazy-loads two models: ChatterboxTTS (English) and ChatterboxMultilingualTTS
        - English text uses the specialized English model for higher quality
        - All other languages use the multilingual model
        - Supports GPU (CUDA, MPS) with automatic CPU fallback

    Configurable Parameters:
        - device: Compute device ('cuda', 'cpu', 'mps')
        - exaggeration: Speech expressiveness (0.25-2.0, default 0.5)
        - cfg_weight: Classifier-free guidance weight (0.0-1.0, default 0.5)
        - temperature: Sampling temperature (0.05-5.0, default 0.8)

    Attributes:
        device (str): Actual device being used after initialization
        english_model (ChatterboxTTS): Model for English TTS
        multilingual_model (ChatterboxMultilingualTTS): Model for other languages

    Example:
        >>> pipeline = ChatterBoxPipeline(device="cuda")
        >>> audio, sr = pipeline.generate("Hello world!", language="en")
        >>> pipeline.save_audio(audio, sr, "output/hello", format="wav")
    """

    def __init__(self, device: str = "cuda", checkpoint_dir: Optional[str] = None):
        """
        Initialize the ChatterBox TTS pipeline with model loading.

        Loads both the English-only and multilingual models into memory. Models are
        downloaded automatically from HuggingFace on first run (~2-4GB total).

        Configuration:
            device: Preferred device for inference
                   - 'cuda': NVIDIA GPU (fastest, requires CUDA)
                   - 'cpu': CPU inference (slower but works everywhere)
                   - 'mps': Apple Silicon GPU (M1/M2/M3 Macs)

            checkpoint_dir: Optional custom path to model checkpoints
                           If None, uses default HuggingFace cache

        Args:
            device: Device to run inference on (cuda/cpu/mps)
            checkpoint_dir: Custom model checkpoint directory (optional)

        Raises:
            RuntimeError: If CUDA is requested but not available

        Note:
            - Automatically falls back to CPU if CUDA is not available
            - Models are cached in ~/.cache/huggingface by default
            - First run will download ~2-4GB of model files
        """
        # Auto-detect device availability and fallback to CPU if needed
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Initializing ChatterBox Pipeline on {self.device}")

        # Load both TTS models into memory
        print("Loading ChatterBox models...")
        self.english_model = ChatterboxTTS.from_pretrained(device=self.device)
        self.multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        print("Models loaded successfully")

    def detect_language(self, text: str) -> Tuple[str, str]:
        """
        Automatically detect language from input text.

        Uses langdetect library to identify the language, then maps it to a ChatterBox-
        compatible language code. Falls back to English if detection fails or if the
        detected language is not supported.

        Supported Languages (23 total):
            ar (Arabic), da (Danish), de (German), el (Greek), en (English),
            es (Spanish), fi (Finnish), fr (French), he (Hebrew), hi (Hindi),
            it (Italian), ja (Japanese), ko (Korean), ms (Malay), nl (Dutch),
            no (Norwegian), pl (Polish), pt (Portuguese), ru (Russian),
            sv (Swedish), sw (Swahili), tr (Turkish), zh (Chinese)

        Args:
            text: Input text to analyze (should be at least a few words)

        Returns:
            tuple: (language_code, language_name)
                  - language_code: Two-letter ISO code (e.g., 'en', 'es')
                  - language_name: Full English name (e.g., 'English', 'Spanish')

        Example:
            >>> pipeline.detect_language("Hello world")
            ('en', 'English')
            >>> pipeline.detect_language("Bonjour le monde")
            ('fr', 'French')

        Note:
            - Requires at least 2-3 words for accurate detection
            - Short texts may be misdetected
            - Always returns ('en', 'English') as fallback
        """
        try:
            # Use langdetect to identify the language
            detected = detect(text)

            # Map langdetect code to ChatterBox code (some differ)
            chatterbox_code = LANGDETECT_TO_CHATTERBOX.get(detected, detected)

            # Check if the detected language is supported by ChatterBox
            if not is_supported(chatterbox_code):
                print(f"WARNING: Detected language '{detected}' not supported, defaulting to English")
                return "en", "English"

            # Get full language name for logging
            lang_name = get_language_name(chatterbox_code)
            return chatterbox_code, lang_name

        except Exception as e:
            # If detection fails for any reason, default to English
            print(f"WARNING: Language detection failed: {e}, defaulting to English")
            return "en", "English"

    def generate(
        self,
        text: str,
        language: Optional[str] = None,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech audio from input text.

        This is the main TTS generation method. It handles language detection,
        model selection, voice cloning, and audio synthesis. Returns raw audio
        waveform as a numpy array.

        Configurable Parameters:
            language: Force a specific language (bypasses auto-detection)
                     Use two-letter codes: 'en', 'es', 'fr', etc.

            audio_prompt_path: Path or URL to voice reference audio for cloning
                              - Local path: "/path/to/voice.wav"
                              - Remote URL: "https://example.com/voice.flac"
                              - If None, uses default voice for the language

            exaggeration: Controls speech expressiveness and emotion
                         - Range: 0.25 to 2.0
                         - Default: 0.5 (neutral)
                         - Lower: More monotone, robotic
                         - Higher: More expressive, exaggerated

            cfg_weight: Classifier-free guidance weight (controls pacing)
                       - Range: 0.0 to 1.0
                       - Default: 0.5 (balanced)
                       - Lower: Faster, less controlled
                       - Higher: Slower, more controlled

            temperature: Sampling temperature (controls randomness)
                        - Range: 0.05 to 5.0
                        - Default: 0.8 (natural variation)
                        - Lower: More consistent, less varied
                        - Higher: More varied, less predictable

        Args:
            text: Text to convert to speech (any length)
            language: Language code (None for auto-detection)
            audio_prompt_path: Voice reference for cloning (None for default)
            exaggeration: Speech expressiveness level (0.25-2.0)
            cfg_weight: CFG/Pace control weight (0.0-1.0)
            temperature: Sampling randomness (0.05-5.0)

        Returns:
            tuple: (audio_waveform, sample_rate)
                  - audio_waveform: 1D numpy array of float32 audio samples
                  - sample_rate: Sample rate in Hz (typically 24000)

        Example:
            >>> # Auto-detect language with default voice
            >>> audio, sr = pipeline.generate("Hello world")

            >>> # Specify language and use custom voice
            >>> audio, sr = pipeline.generate(
            ...     "Hola mundo",
            ...     language="es",
            ...     audio_prompt_path="my_voice.wav",
            ...     exaggeration=0.7
            ... )

        Note:
            - English uses a specialized model for better quality
            - Voice cloning works best with 3-10 seconds of clear speech
            - Default voices are used if audio_prompt_path is None
            - URLs are automatically downloaded and cached
        """
        # Auto-detect language if not explicitly provided
        if language is None:
            language, lang_name = self.detect_language(text)
            print(f"Detected language: {lang_name} ({language})")
        else:
            lang_name = get_language_name(language)
            print(f"Using specified language: {lang_name} ({language})")

        # Use default voice if no custom voice provided
        # English has no default voice (model doesn't require one)
        if audio_prompt_path is None and language != "en":
            audio_prompt_path = get_default_voice(language)
            if audio_prompt_path:
                print(f"Using default voice for {lang_name}")

        # Convert URL to local path if needed (for Windows compatibility)
        audio_prompt_path = get_voice_path(audio_prompt_path)

        # Generate audio using appropriate model
        print(f"Generating speech...")

        if language == "en":
            # Use specialized English model for higher quality
            wav = self.english_model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
            sample_rate = self.english_model.sr
        else:
            # Use multilingual model for all other languages
            wav = self.multilingual_model.generate(
                text,
                language_id=language,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
            sample_rate = self.multilingual_model.sr

        # Convert from tensor to numpy array and remove batch dimension
        return wav.squeeze(0).numpy(), sample_rate

    def save_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str,
        format: str = "wav"
    ) -> str:
        """
        Save audio waveform to file in specified format.

        Supports three output formats: WAV (uncompressed), MP3 (compressed),
        and PCM/RAW (headerless). File extension is added automatically.

        Configuration:
            format: Output file format
                   - 'wav': Standard WAV format, uncompressed, high quality
                   - 'mp3': MP3 format, compressed at 192kbps (requires ffmpeg)
                   - 'raw'/'pcm': Raw 16-bit signed PCM data, no header

            output_path: Base path without extension (extension added automatically)

        Args:
            audio: Audio waveform as 1D numpy array
            sample_rate: Sample rate in Hz
            output_path: Output file path without extension
            format: Output format ('wav', 'mp3', 'raw', 'pcm')

        Returns:
            str: Full path to the saved audio file including extension

        Raises:
            ValueError: If format is not one of: wav, mp3, raw, pcm

        Example:
            >>> audio, sr = pipeline.generate("Hello")
            >>> # Saves as output/test.wav
            >>> path = pipeline.save_audio(audio, sr, "output/test", format="wav")
            >>> # Saves as output/test.mp3
            >>> path = pipeline.save_audio(audio, sr, "output/test", format="mp3")

        Note:
            - MP3 export requires ffmpeg or libav installed on the system
            - PCM format is 16-bit signed integer, mono channel
            - WAV format is the most compatible across all platforms
        """
        format = format.lower()

        # Remove any existing extension from output path
        output_path = str(Path(output_path).with_suffix(''))

        if format == "wav":
            # Save as uncompressed WAV file
            output_file = f"{output_path}.wav"
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            ta.save(output_file, audio_tensor, sample_rate)
            print(f"Saved WAV: {output_file}")

        elif format == "mp3":
            # Convert to MP3 via temporary WAV file
            # (pydub requires WAV intermediate for conversion)
            temp_wav = f"{output_path}_temp.wav"
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            ta.save(temp_wav, audio_tensor, sample_rate)

            # Convert WAV to MP3 at 192kbps
            output_file = f"{output_path}.mp3"
            sound = AudioSegment.from_wav(temp_wav)
            sound.export(output_file, format="mp3", bitrate="192k")

            # Clean up temporary WAV file
            os.remove(temp_wav)
            print(f"Saved MP3: {output_file}")

        elif format == "raw" or format == "pcm":
            # Save as raw PCM data (16-bit signed integer)
            output_file = f"{output_path}.raw"

            # Convert float32 audio to 16-bit signed integer
            # Assumes audio is normalized to [-1.0, 1.0]
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_int16.tofile(output_file)

            print(f"Saved PCM/RAW: {output_file}")
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Format: 16-bit signed PCM, mono")

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'wav', 'mp3', or 'raw'")

        return output_file

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        output_format: str = "wav",
        language: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """
        Process a text file and generate audio output.

        Convenience method that reads text from a file, generates speech,
        and saves the output. Handles file I/O automatically.

        Configuration:
            output_path: Where to save the audio file
                        - If None, auto-generated from input filename
                        - Example: "input.txt" -> "output/input_output.wav"

            output_format: Audio format (wav/mp3/raw)

            language: Force specific language (None for auto-detection)

            **generation_kwargs: Additional parameters passed to generate()
                                (exaggeration, cfg_weight, temperature, audio_prompt_path)

        Args:
            input_path: Path to input text file (UTF-8 encoded)
            output_path: Path for output audio file (None for auto)
            output_format: Output format ('wav', 'mp3', 'raw')
            language: Language code (None for auto-detection)
            **generation_kwargs: Additional generation parameters

        Returns:
            str: Path to the generated audio file

        Example:
            >>> # Auto-detect language, auto-generate output path
            >>> path = pipeline.process_file("input.txt")

            >>> # Specify everything
            >>> path = pipeline.process_file(
            ...     "input.txt",
            ...     output_path="custom/path",
            ...     output_format="mp3",
            ...     language="es",
            ...     exaggeration=0.7
            ... )

        Note:
            - Input file must be UTF-8 encoded
            - Auto-generated paths use format: output/{stem}_output.{ext}
            - Creates output directory automatically if needed
        """
        # Read input text file
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        print(f"Processing file: {input_path}")
        print(f"   Text length: {len(text)} characters")

        # Auto-generate output path if not provided
        if output_path is None:
            output_path = Path(input_path).stem
            output_path = f"output/{output_path}_output"

        # Generate speech audio
        audio, sample_rate = self.generate(text, language=language, **generation_kwargs)

        # Save to file
        output_file = self.save_audio(audio, sample_rate, output_path, output_format)

        return output_file

    def process_text(
        self,
        text: str,
        output_path: str = "output/output",
        output_format: str = "wav",
        language: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """
        Process text string directly and generate audio output.

        Convenience method that takes a text string, generates speech,
        and saves the output in one call.

        Configuration:
            output_path: Where to save the audio file (default: "output/output")

            output_format: Audio format (wav/mp3/raw)

            language: Force specific language (None for auto-detection)

            **generation_kwargs: Additional parameters passed to generate()
                                (exaggeration, cfg_weight, temperature, audio_prompt_path)

        Args:
            text: Text string to convert to speech
            output_path: Path for output audio file
            output_format: Output format ('wav', 'mp3', 'raw')
            language: Language code (None for auto-detection)
            **generation_kwargs: Additional generation parameters

        Returns:
            str: Path to the generated audio file

        Example:
            >>> # Basic usage with defaults
            >>> path = pipeline.process_text("Hello world")

            >>> # Full customization
            >>> path = pipeline.process_text(
            ...     "Hola mundo",
            ...     output_path="output/spanish",
            ...     output_format="mp3",
            ...     language="es",
            ...     exaggeration=0.8,
            ...     audio_prompt_path="voice.wav"
            ... )

        Note:
            - Creates output directory automatically if needed
            - More efficient than process_file() for programmatic use
            - Use this method when text is already in memory
        """
        print(f"Processing text: {text[:100]}...")

        # Generate speech audio
        audio, sample_rate = self.generate(text, language=language, **generation_kwargs)

        # Save to file
        output_file = self.save_audio(audio, sample_rate, output_path, output_format)

        return output_file


# =============================================================================
# Command-Line Interface
# =============================================================================

def main():
    """
    Command-line interface for ChatterBox TTS pipeline.

    Provides a CLI for converting text to speech without writing code. Supports
    all pipeline features through command-line arguments.

    Command-Line Arguments:
        Input (required, one of):
            --text TEXT           : Text string to synthesize
            --input FILE          : Path to text file to process

        Output (optional):
            --output PATH         : Output file path (default: output/output)
            --format FORMAT       : Output format: wav, mp3, raw (default: wav)

        Language (optional):
            --language CODE       : Language code (auto-detected if not specified)
                                   Use 2-letter codes: en, es, fr, de, etc.

        Voice (optional):
            --voice FILE          : Path or URL to voice reference audio
                                   For zero-shot voice cloning

        Generation Parameters (optional):
            --exaggeration FLOAT  : Expressiveness level (0.25-2.0, default: 0.5)
            --cfg-weight FLOAT    : CFG/Pace weight (0.0-1.0, default: 0.5)
            --temperature FLOAT   : Sampling temperature (0.05-5.0, default: 0.8)

        System (optional):
            --device DEVICE       : Compute device: cuda, cpu, mps (default: cuda)

    Examples:
        # Basic text to speech
        python chatterbox_pipeline.py --text "Hello world"

        # Specify language and format
        python chatterbox_pipeline.py --text "Hola" --language es --format mp3

        # Process file with custom voice
        python chatterbox_pipeline.py --input input.txt --voice voice.wav

        # Full customization
        python chatterbox_pipeline.py \\
            --text "Advanced test" \\
            --output custom/path \\
            --format wav \\
            --language en \\
            --device cuda \\
            --exaggeration 0.7 \\
            --cfg-weight 0.6 \\
            --temperature 0.9 \\
            --voice custom_voice.wav

    Exit Codes:
        0: Success
        1: Error (invalid arguments or processing failure)
    """
    import argparse

    # Set up argument parser with detailed help
    parser = argparse.ArgumentParser(
        description="ChatterBox TTS Pipeline - Multilingual Text-to-Speech",
        epilog="For more information, see README.md"
    )

    # Input arguments (mutually exclusive: text OR file)
    parser.add_argument("--text", type=str,
                       help="Text to synthesize")
    parser.add_argument("--input", type=str,
                       help="Input text file path")

    # Output arguments
    parser.add_argument("--output", type=str, default="output/output",
                       help="Output file path (default: output/output)")
    parser.add_argument("--format", type=str, default="wav",
                       choices=["wav", "mp3", "raw", "pcm"],
                       help="Output format (default: wav)")

    # Language argument
    parser.add_argument("--language", type=str,
                       help="Language code (auto-detected if not specified)")

    # System argument
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"],
                       help="Device to run on (default: cuda)")

    # Generation parameter arguments
    parser.add_argument("--exaggeration", type=float, default=0.5,
                       help="Exaggeration level 0.25-2.0 (default: 0.5)")
    parser.add_argument("--cfg-weight", type=float, default=0.5,
                       help="CFG/Pace weight 0.0-1.0 (default: 0.5)")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Temperature 0.05-5.0 (default: 0.8)")

    # Voice cloning argument
    parser.add_argument("--voice", type=str,
                       help="Path or URL to custom voice reference audio")

    args = parser.parse_args()

    # Validate that either --text or --input is provided
    if not args.text and not args.input:
        parser.error("Either --text or --input must be provided")

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Initialize pipeline with specified device
    pipeline = ChatterBoxPipeline(device=args.device)

    # Prepare generation parameters
    generation_kwargs = {
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
        "temperature": args.temperature,
        "audio_prompt_path": args.voice
    }

    # Process input (either file or text)
    if args.input:
        # Process text file
        output_file = pipeline.process_file(
            args.input,
            output_path=args.output,
            output_format=args.format,
            language=args.language,
            **generation_kwargs
        )
    else:
        # Process text string
        output_file = pipeline.process_text(
            args.text,
            output_path=args.output,
            output_format=args.format,
            language=args.language,
            **generation_kwargs
        )

    print(f"\nComplete! Audio saved to: {output_file}")


if __name__ == "__main__":
    main()
