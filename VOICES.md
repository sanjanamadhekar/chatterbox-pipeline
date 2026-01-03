# ChatterBox Available Voices

## Overview

ChatterBox does **NOT have traditional preset voices**. Instead, it uses **zero-shot voice cloning** - you provide a reference audio sample, and ChatterBox extracts the voice characteristics to generate speech in that voice.

---

## Default Reference Voices (23 Languages)

Resemble AI provides **one default reference voice per language** hosted on Google Cloud Storage. These are short audio samples (3-12 seconds) used for voice cloning.

### Voice List

| Language    | Code | Gender | Voice ID      | Duration | URL |
|-------------|------|--------|---------------|----------|-----|
| Arabic      | ar   | Female | ar_prompts2   | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac) |
| Danish      | da   | Male   | da_m1         | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac) |
| German      | de   | Female | de_f1         | ~8s      | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac) |
| Greek       | el   | Male   | el_m          | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac) |
| **English** | en   | **Female** | **en_f1** | **~3s**  | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac) |
| Spanish     | es   | Female | es_f1         | ~12s     | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac) |
| Finnish     | fi   | Male   | fi_m          | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac) |
| French      | fr   | Female | fr_f1         | ~6.6s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac) |
| Hebrew      | he   | Male   | he_m1         | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac) |
| Hindi       | hi   | Female | hi_f1         | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac) |
| Italian     | it   | Male   | it_m1         | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac) |
| Japanese    | ja   | Mixed  | ja_prompts1   | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac) |
| Korean      | ko   | Female | ko_f          | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac) |
| Malay       | ms   | Female | ms_f          | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac) |
| Dutch       | nl   | Male   | nl_m          | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac) |
| Norwegian   | no   | Female | no_f1         | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac) |
| Polish      | pl   | Male   | pl_m          | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac) |
| Portuguese  | pt   | Male   | pt_m1         | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac) |
| Russian     | ru   | Male   | ru_m          | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac) |
| Swedish     | sv   | Female | sv_f          | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac) |
| Swahili     | sw   | Male   | sw_m          | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac) |
| Turkish     | tr   | Male   | tr_m          | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac) |
| Chinese     | zh   | Female | zh_f2         | ~3-5s    | [Link](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac) |

**Summary:**
- **Total:** 23 voices (one per language)
- **Female:** 11 voices
- **Male:** 11 voices
- **Mixed/Unknown:** 1 voice (Japanese)

---

## How Voices Work

### 1. Voice Cloning Process

ChatterBox uses a **speaker encoder** to extract voice characteristics:

```python
# What happens internally:
1. Load reference audio (3-10 seconds recommended)
2. Resample to 16kHz
3. Extract 256-dimensional speaker embedding
4. Use embedding to condition TTS generation
```

### 2. Voice Requirements

**Reference Audio Requirements:**
- **Duration:** 3-10 seconds (optimal)
- **Quality:** Clear speech, minimal background noise
- **Format:** Any audio format (WAV, MP3, FLAC, etc.)
- **Content:** Natural speech in the target language
- **Sample Rate:** Any (automatically resampled to 16kHz)

**What Makes a Good Reference:**
- Single speaker
- Clear pronunciation
- Natural intonation
- Minimal background noise
- Matches target language (for best accent)

---

## Usage Examples

### Using Default Remote Voices

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Uses default English female voice (downloaded automatically)
wav = model.generate(
    "Hello world!",
    language_id="en",
    audio_prompt_path="https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac"
)
```

### Using Your Own Voice

```python
# Clone any voice from a local audio file
wav = model.generate(
    "Hello in my voice!",
    language_id="en",
    audio_prompt_path="/path/to/your_voice_sample.wav"
)
```

### Using Pipeline (Auto-selects Default Voice)

```python
from chatterbox_pipeline import ChatterBoxPipeline

pipeline = ChatterBoxPipeline()

# Automatically uses default voice for detected language
wav, sr = pipeline.generate("Bonjour le monde!")  # Uses fr_f1 voice
```

---

## Voice Options Summary

| Option | Type | Availability | Quality | Usage |
|--------|------|--------------|---------|-------|
| **Default Remote Voices** | Reference samples | 23 languages | Good baseline | Auto-downloaded URLs |
| **Your Own Audio** | Custom samples | Unlimited | Depends on sample | Local/remote files |
| **No Voice** | ❌ Not supported | N/A | N/A | Will error |

---

## Limitations

1. **No Preset Voice Library**
   - ChatterBox doesn't have voice presets like "David", "Sarah", etc.
   - Every voice requires a reference audio sample

2. **One Default Voice Per Language**
   - Only one male OR female default voice available
   - To get variety, you must provide your own samples

3. **Voice Quality Depends on Reference**
   - Better reference = better output
   - Poor quality reference = poor quality output

4. **Language Matching Recommended**
   - Best results when reference language matches target language
   - Cross-language cloning possible but may have accent transfer

---

## Adding More Voices

### Option 1: Download Default Voices Locally

```bash
# Download all default voices to local storage
mkdir -p voices
cd voices

# Download English female
wget https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac

# Download Spanish female
wget https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac

# etc...
```

### Option 2: Record Your Own Voices

```python
# Create your own voice library
my_voices = {
    "narrator_male": "voices/narrator_male.wav",
    "narrator_female": "voices/narrator_female.wav",
    "character_1": "voices/character_1.wav",
}

# Use in generation
wav = model.generate(
    "Text here",
    audio_prompt_path=my_voices["narrator_male"]
)
```

### Option 3: Use Public Audio Samples

Find clean speech samples from:
- Audiobooks
- Podcasts
- YouTube videos (extract audio)
- LibriVox (public domain)
- Voice acting samples

**Requirements:** 3-10 seconds of clean, clear speech

---

## Technical Details

### Voice Encoder Architecture
- **Model:** CAMPPlus speaker encoder
- **Output:** 256-dimensional speaker embedding
- **Input:** 16kHz mono audio
- **Processing:** Mel-spectrogram → encoder → L2-normalized embedding

### What's Extracted from Reference Audio
- Voice timbre (vocal quality)
- Pitch characteristics
- Speaking rate tendencies
- Accent patterns
- Prosody style

### What's NOT Extracted
- Exact emotions (controlled via `exaggeration` parameter)
- Background noise
- Audio quality issues
- Multiple speakers (uses first/dominant speaker)

---

## FAQ

**Q: Can I use any audio file as reference?**
A: Yes, but 3-10 seconds of clean speech works best.

**Q: Do I need different voices for different languages?**
A: No, but it's recommended for better accent matching.

**Q: Can I create multiple voices?**
A: Yes, just provide different reference audio files.

**Q: Are the default voices free to use?**
A: Yes, they're provided by Resemble AI for demo purposes.

**Q: Can I use celebrity voices?**
A: Technically possible, but may violate terms of service and laws.

**Q: How do I get a male voice for English?**
A: Either use a male voice sample or wait for additional defaults.

**Q: Can I adjust the voice pitch?**
A: The model will match the reference. For variations, adjust the reference audio.

---

## See Also

- [ChatterBox Documentation](https://github.com/resemble-ai/chatterbox)
- [Voice Cloning Guide](https://resemble.ai)
- [Speaker Encoder Details](https://github.com/resemble-ai/chatterbox)
