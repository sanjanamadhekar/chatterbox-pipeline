#!/usr/bin/env python3
"""
Quick test to verify ChatterBox pipeline is working
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatterbox_pipeline import ChatterBoxPipeline

def main():
    print("="*60)
    print("CHATTERBOX PIPELINE - QUICK TEST")
    print("="*60)

    # Create output directory
    os.makedirs("tests/output", exist_ok=True)

    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = ChatterBoxPipeline()

    # Test 1: English with auto-detection
    print("\n2. Testing English (auto-detect)...")
    text = "Hello! This is a quick test of the ChatterBox pipeline."
    audio, sr = pipeline.generate(text)
    output = pipeline.save_audio(audio, sr, "tests/output/quick_test_en", "wav")
    print(f"   ✅ Generated: {output}")

    # Test 2: Spanish with explicit language
    print("\n3. Testing Spanish (explicit language)...")
    text = "Hola! Esta es una prueba rápida del pipeline de ChatterBox."
    audio, sr = pipeline.generate(text, language="es")
    output = pipeline.save_audio(audio, sr, "tests/output/quick_test_es", "wav")
    print(f"   ✅ Generated: {output}")

    # Test 3: MP3 output
    print("\n4. Testing MP3 output...")
    text = "Testing MP3 format output."
    audio, sr = pipeline.generate(text)
    output = pipeline.save_audio(audio, sr, "tests/output/quick_test_mp3", "mp3")
    print(f"   ✅ Generated: {output}")

    # Test 4: PCM/RAW output
    print("\n5. Testing PCM/RAW output...")
    text = "Testing raw PCM format output."
    audio, sr = pipeline.generate(text)
    output = pipeline.save_audio(audio, sr, "tests/output/quick_test_pcm", "raw")
    print(f"   ✅ Generated: {output}")

    print("\n" + "="*60)
    print("✅ ALL QUICK TESTS PASSED!")
    print("="*60)
    print(f"Test outputs saved to: {os.path.abspath('tests/output')}")
    print("="*60)

if __name__ == "__main__":
    main()
