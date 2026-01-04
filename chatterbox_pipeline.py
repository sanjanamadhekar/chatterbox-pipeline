#!/usr/bin/env python3
"""
ChatterBox TTS Pipeline
Automatic language detection and multilingual text-to-speech conversion
"""

import os
import sys
import torch
import torchaudio as ta
import numpy as np
from pathlib import Path
from typing import Optional
from langdetect import detect
from pydub import AudioSegment
import urllib.request
import hashlib

# Add config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))
from language_config import (
    LANGDETECT_TO_CHATTERBOX,
    get_language_name,
    get_default_voice,
    is_supported
)

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def download_voice_file(url: str, cache_dir: str = "voices") -> str:
    """
    Download voice file from URL to local cache

    Args:
        url: URL of the voice file
        cache_dir: Directory to cache voice files

    Returns:
        Path to cached voice file
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Create filename from URL hash to avoid duplicates
    url_hash = hashlib.md5(url.encode()).hexdigest()
    filename = f"{url_hash}.flac"
    cache_path = os.path.join(cache_dir, filename)

    # Download if not already cached
    if not os.path.exists(cache_path):
        print(f"📥 Downloading voice file from URL...")
        try:
            urllib.request.urlretrieve(url, cache_path)
            print(f"✅ Cached voice file: {cache_path}")
        except Exception as e:
            print(f"⚠️  Failed to download voice file: {e}")
            return None

    return cache_path


def get_voice_path(audio_prompt_path: Optional[str]) -> Optional[str]:
    """
    Get local voice path, downloading from URL if needed

    Args:
        audio_prompt_path: Path or URL to voice file

    Returns:
        Local path to voice file, or None if unavailable
    """
    if audio_prompt_path is None:
        return None

    # If it's a URL, download and cache it
    if audio_prompt_path.startswith("http://") or audio_prompt_path.startswith("https://"):
        return download_voice_file(audio_prompt_path)

    # Otherwise it's already a local path
    return audio_prompt_path


class ChatterBoxPipeline:
    """Main pipeline for ChatterBox TTS with automatic language detection"""

    def __init__(self, device: str = "cuda", checkpoint_dir: Optional[str] = None):
        """
        Initialize ChatterBox pipeline

        Args:
            device: Device to run on ('cuda', 'cpu', 'mps')
            checkpoint_dir: Path to model checkpoints (optional)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing ChatterBox Pipeline on {self.device}")

        # Load models
        print("📥 Loading ChatterBox models...")
        self.english_model = ChatterboxTTS.from_pretrained(device=self.device)
        self.multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        print("✅ Models loaded successfully")

    def detect_language(self, text: str) -> tuple[str, str]:
        """
        Detect language from text

        Args:
            text: Input text

        Returns:
            Tuple of (detected_lang_code, language_name)
        """
        try:
            detected = detect(text)
            chatterbox_code = LANGDETECT_TO_CHATTERBOX.get(detected, detected)

            if not is_supported(chatterbox_code):
                print(f"⚠️  Detected language '{detected}' not supported, defaulting to English")
                return "en", "English"

            lang_name = get_language_name(chatterbox_code)
            return chatterbox_code, lang_name
        except Exception as e:
            print(f"⚠️  Language detection failed: {e}, defaulting to English")
            return "en", "English"

    def generate(
        self,
        text: str,
        language: Optional[str] = None,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8
    ) -> tuple[np.ndarray, int]:
        """
        Generate speech from text

        Args:
            text: Input text to synthesize
            language: Language code (auto-detected if None)
            audio_prompt_path: Path to voice reference audio (optional)
            exaggeration: Speech exaggeration level (0.25-2.0)
            cfg_weight: CFG/Pace weight (0.0-1.0)
            temperature: Generation temperature (0.05-5.0)

        Returns:
            Tuple of (audio_waveform, sample_rate)
        """
        # Detect language if not provided
        if language is None:
            language, lang_name = self.detect_language(text)
            print(f"🔍 Detected language: {lang_name} ({language})")
        else:
            lang_name = get_language_name(language)
            print(f"🌍 Using specified language: {lang_name} ({language})")

        # Use default voice if no custom voice provided
        if audio_prompt_path is None and language != "en":
            audio_prompt_path = get_default_voice(language)
            if audio_prompt_path:
                print(f"🎤 Using default voice for {lang_name}")

        # Convert URL to local path if needed
        audio_prompt_path = get_voice_path(audio_prompt_path)

        # Generate audio
        print(f"🎵 Generating speech...")

        if language == "en":
            # Use English-only model for better quality
            wav = self.english_model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
            sample_rate = self.english_model.sr
        else:
            # Use multilingual model
            wav = self.multilingual_model.generate(
                text,
                language_id=language,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
            sample_rate = self.multilingual_model.sr

        return wav.squeeze(0).numpy(), sample_rate

    def save_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str,
        format: str = "wav"
    ) -> str:
        """
        Save audio to file in specified format

        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            output_path: Output file path (extension will be added)
            format: Output format ('wav', 'mp3', 'raw')

        Returns:
            Path to saved file
        """
        format = format.lower()

        # Remove extension if present
        output_path = str(Path(output_path).with_suffix(''))

        if format == "wav":
            # Save as WAV
            output_file = f"{output_path}.wav"
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            ta.save(output_file, audio_tensor, sample_rate)
            print(f"✅ Saved WAV: {output_file}")

        elif format == "mp3":
            # Save as WAV first, then convert to MP3
            temp_wav = f"{output_path}_temp.wav"
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            ta.save(temp_wav, audio_tensor, sample_rate)

            # Convert to MP3
            output_file = f"{output_path}.mp3"
            sound = AudioSegment.from_wav(temp_wav)
            sound.export(output_file, format="mp3", bitrate="192k")

            # Remove temp file
            os.remove(temp_wav)
            print(f"✅ Saved MP3: {output_file}")

        elif format == "raw" or format == "pcm":
            # Save as raw PCM
            output_file = f"{output_path}.raw"

            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_int16.tofile(output_file)

            print(f"✅ Saved PCM/RAW: {output_file}")
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
        Process text file and generate audio

        Args:
            input_path: Path to input text file
            output_path: Path for output file (auto-generated if None)
            output_format: Output format ('wav', 'mp3', 'raw')
            language: Language code (auto-detected if None)
            **generation_kwargs: Additional arguments for generate()

        Returns:
            Path to generated audio file
        """
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        print(f"📄 Processing file: {input_path}")
        print(f"   Text length: {len(text)} characters")

        # Generate output path if not provided
        if output_path is None:
            output_path = Path(input_path).stem
            output_path = f"output/{output_path}_output"

        # Generate audio
        audio, sample_rate = self.generate(text, language=language, **generation_kwargs)

        # Save audio
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
        Process text string and generate audio

        Args:
            text: Input text to synthesize
            output_path: Path for output file
            output_format: Output format ('wav', 'mp3', 'raw')
            language: Language code (auto-detected if None)
            **generation_kwargs: Additional arguments for generate()

        Returns:
            Path to generated audio file
        """
        print(f"💬 Processing text: {text[:100]}...")

        # Generate audio
        audio, sample_rate = self.generate(text, language=language, **generation_kwargs)

        # Save audio
        output_file = self.save_audio(audio, sample_rate, output_path, output_format)

        return output_file


def main():
    """CLI interface for ChatterBox pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="ChatterBox TTS Pipeline")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--input", type=str, help="Input text file")
    parser.add_argument("--output", type=str, default="output/output", help="Output file path")
    parser.add_argument("--format", type=str, default="wav", choices=["wav", "mp3", "raw", "pcm"],
                        help="Output format (default: wav)")
    parser.add_argument("--language", type=str, help="Language code (auto-detected if not specified)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Device to run on")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Exaggeration level (0.25-2.0)")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG/Pace weight (0.0-1.0)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature (0.05-5.0)")
    parser.add_argument("--voice", type=str, help="Path to custom voice reference audio")

    args = parser.parse_args()

    # Validate input
    if not args.text and not args.input:
        parser.error("Either --text or --input must be provided")

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Initialize pipeline
    pipeline = ChatterBoxPipeline(device=args.device)

    # Process input
    generation_kwargs = {
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
        "temperature": args.temperature,
        "audio_prompt_path": args.voice
    }

    if args.input:
        # Process file
        output_file = pipeline.process_file(
            args.input,
            output_path=args.output,
            output_format=args.format,
            language=args.language,
            **generation_kwargs
        )
    else:
        # Process text
        output_file = pipeline.process_text(
            args.text,
            output_path=args.output,
            output_format=args.format,
            language=args.language,
            **generation_kwargs
        )

    print(f"\n🎉 Complete! Audio saved to: {output_file}")


if __name__ == "__main__":
    main()
