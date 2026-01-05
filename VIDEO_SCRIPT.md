# ChatterBox Pipeline - 15-Minute Code Walkthrough & Demo Video Script

**Total Duration: 15 minutes**
**Target Audience: Future teammates taking over the project**
**Format: Code walkthrough with live demonstrations**

---

## INTRO (1 minute)

**[Screen: VS Code with chatterbox-pipeline folder open]**

**Script:**

> "Hey team! This is a code walkthrough and demo of the ChatterBox Pipeline. I'm going to walk you through the actual code, show you how everything works under the hood, and then demonstrate it live.
>
> This is a production-ready text-to-speech system that supports 23 languages. We'll cover:
> - The main pipeline class and its methods
> - How language detection works
> - The dual-model architecture
> - Voice cloning implementation
> - Then we'll run live demos to see it all in action
>
> Let's start with the code."

---

## PART 1: PROJECT STRUCTURE OVERVIEW (1 minute)

**[Screen: File explorer showing project structure]**

**Script:**

> "Here's the project layout. Two main code files:
>
> **Core Code:**
> - `chatterbox_pipeline.py` - 790 lines, the main pipeline class
> - `config/language_config.py` - 99 lines, language configuration
>
> **Tests:**
> - `tests/quick_test.py` - 5 quick validation tests
> - `tests/test_multilingual.py` - 100+ comprehensive tests
>
> **Documentation:**
> - README.md, VOICES.md, PROJECT_SUMMARY.md, SETUP_WINDOWS.md
>
> We'll focus on the code. Let's open the main file."

---

## PART 2: MAIN PIPELINE CLASS - INITIALIZATION (2 minutes)

**[Screen: Open chatterbox_pipeline.py, scroll to class definition and __init__ method]**

**Script:**

> "Let's start with the ChatterBoxPipeline class. Opening `chatterbox_pipeline.py`.
>
> [Scroll to line 1-50, show imports and class definition]
>
> The imports tell us what we're working with:
> - `torch` and `torchaudio` for PyTorch and audio handling
> - `langdetect` for automatic language detection
> - `ChatterboxTTS` and `ChatterboxMultilingualTTS` - the two models
> - `pydub` for MP3 conversion
> - Standard libraries for file handling
>
> [Scroll to __init__ method around line 60-120]
>
> The initialization does three critical things:
>
> **1. Device Selection** (lines ~65-75)
> ```python
> if device is None:
>     if torch.cuda.is_available():
>         device = 'cuda'
>     elif torch.backends.mps.is_available():
>         device = 'mps'
>     else:
>         device = 'cpu'
> ```
> It automatically detects GPU (CUDA or MPS for Mac) and falls back to CPU if needed.
>
> **2. Dual Model Loading** (lines ~80-100)
> ```python
> self.english_model = ChatterboxTTS(device=self.device)
> self.multilingual_model = ChatterboxMultilingualTTS(device=self.device)
> ```
> This is key - we load BOTH models. The English model gives better quality for English, the multilingual handles the other 22 languages.
>
> **3. Voice File Caching** (lines ~105-115)
> ```python
> self.voice_cache_dir = Path.home() / '.chatterbox_voices'
> self.voice_cache_dir.mkdir(exist_ok=True)
> ```
> We cache downloaded voice files to avoid re-downloading. This is important for Windows compatibility.
>
> So when you create a pipeline object, both models are loaded into memory and ready to go."

---

## PART 3: LANGUAGE DETECTION - THE DETECT_LANGUAGE METHOD (1.5 minutes)

**[Screen: Scroll to detect_language method in chatterbox_pipeline.py]**

**Script:**

> "Let's look at how automatic language detection works. Scrolling to the `detect_language` method.
>
> [Show the method around lines 120-150]
>
> ```python
> def detect_language(self, text: str) -> str:
>     try:
>         detected = detect(text)
>         lang_name = LANGDETECT_TO_LANGUAGE.get(detected, 'English')
>         return lang_name
>     except:
>         return 'English'
> ```
>
> Pretty simple but powerful:
> 1. Uses the `langdetect` library to analyze the text
> 2. Maps the detected code to our language names using the config
> 3. Falls back to English if detection fails
>
> The mapping happens in `config/language_config.py`. Let me show you that quickly.
>
> [Open config/language_config.py]
>
> Look at this dictionary around line 20:
> ```python
> LANGDETECT_TO_LANGUAGE = {
>     'en': 'English',
>     'es': 'Spanish',
>     'fr': 'French',
>     'zh-cn': 'Chinese',
>     'ja': 'Japanese',
>     # ... 18 more languages
> }
> ```
>
> And we also have default voice URLs for each language:
> ```python
> DEFAULT_VOICES = {
>     'English': 'https://storage.googleapis.com/...',
>     'Spanish': 'https://storage.googleapis.com/...',
>     # ... all 23 languages
> }
> ```
>
> These are hosted on Google Cloud Storage and download automatically when needed. If you need to change the default voices or add language variants, this is where you do it.
>
> Back to the main file."

---

## PART 4: THE GENERATE METHOD - CORE TTS LOGIC (2.5 minutes)

**[Screen: Scroll to generate method in chatterbox_pipeline.py]**

**Script:**

> "Now the heart of the system - the `generate` method. This is where the actual text-to-speech happens.
>
> [Scroll to generate method around lines 150-250]
>
> Let me walk through the flow:
>
> **Step 1: Language Detection** (lines ~160-165)
> ```python
> if language is None:
>     language = self.detect_language(text)
>     print(f"Detected language: {language}")
> ```
> If no language specified, auto-detect it.
>
> **Step 2: Get Voice Reference** (lines ~170-180)
> ```python
> if voice_path is None:
>     voice_path = get_default_voice(language)
>
> voice_path = self.get_voice_path(voice_path)
> ```
> Either use provided voice, or get the default for that language. The `get_voice_path` method handles downloading if it's a URL.
>
> **Step 3: Model Selection** (lines ~190-210)
> ```python
> if language == 'English':
>     audio = self.english_model.generate(
>         text=text,
>         voice=voice_path,
>         exaggeration=exaggeration,
>         cfg_weight=cfg_weight,
>         temperature=temperature
>     )
> else:
>     audio = self.multilingual_model.generate(
>         text=text,
>         language=language,
>         voice=voice_path,
>         exaggeration=exaggeration,
>         cfg_weight=cfg_weight,
>         temperature=temperature
>     )
> ```
>
> This is the dual-model architecture in action. English gets special treatment with the English-only model. Everything else uses the multilingual model with an explicit language parameter.
>
> **Step 4: Return Audio** (lines ~220-230)
> ```python
> sample_rate = 24000  # ChatterBox uses 24kHz
> return audio, sample_rate
> ```
>
> The models return a PyTorch tensor of audio samples at 24kHz. We return both the audio and sample rate.
>
> **Key Parameters You Can Tune:**
> - `exaggeration` (0.25-2.0): Controls expressiveness
> - `cfg_weight` (0.0-1.0): Controls pacing
> - `temperature` (0.05-5.0): Controls variation
>
> These get passed directly to the underlying ChatterBox models."

---

## PART 5: VOICE CLONING - DOWNLOAD & CACHING (1.5 minutes)

**[Screen: Scroll to get_voice_path and download_voice_file methods]**

**Script:**

> "Voice cloning is a key feature. Let me show you how the voice file handling works.
>
> [Scroll to get_voice_path method around lines 250-280]
>
> ```python
> def get_voice_path(self, voice_path: str) -> str:
>     if voice_path.startswith('http://') or voice_path.startswith('https://'):
>         return self.download_voice_file(voice_path)
>     else:
>         return voice_path
> ```
>
> Simple: if it's a URL, download it. If it's a local path, use it directly.
>
> [Scroll to download_voice_file method around lines 285-330]
>
> The download logic is interesting:
> ```python
> def download_voice_file(self, url: str) -> str:
>     # Create MD5 hash of URL for unique filename
>     url_hash = hashlib.md5(url.encode()).hexdigest()
>     cache_filename = f"voice_{url_hash}.wav"
>     cache_path = self.voice_cache_dir / cache_filename
>
>     # Check if already cached
>     if cache_path.exists():
>         return str(cache_path)
>
>     # Download and save
>     response = requests.get(url)
>     with open(cache_path, 'wb') as f:
>         f.write(response.content)
>
>     return str(cache_path)
> ```
>
> **Why this matters:**
> 1. Uses MD5 hash of URL as filename - same URL always maps to same cached file
> 2. Checks cache first - avoids re-downloading
> 3. Saves to home directory (~/.chatterbox_voices) - works cross-platform
> 4. Critical for Windows - temporary URLs cause issues, so we cache locally
>
> When you use a default voice or provide a URL, this method handles everything automatically. You can also provide your own local .wav file for custom voices - just pass the file path."

---

## PART 6: SAVE_AUDIO & OUTPUT FORMATS (1.5 minutes)

**[Screen: Scroll to save_audio method]**

**Script:**

> "Let's look at how we save the generated audio in different formats.
>
> [Scroll to save_audio method around lines 340-420]
>
> The method signature:
> ```python
> def save_audio(self, audio: torch.Tensor, sample_rate: int,
>                output_path: str, format: str = 'wav') -> str:
> ```
>
> We support three formats: WAV, MP3, and raw PCM.
>
> **WAV Format** (default - lines ~360-375)
> ```python
> if format.lower() == 'wav':
>     output_file = f"{output_path}.wav"
>     torchaudio.save(output_file, audio.unsqueeze(0), sample_rate)
> ```
> Simple: use torchaudio to save directly. Uncompressed, high quality.
>
> **MP3 Format** (lines ~380-405)
> ```python
> elif format.lower() == 'mp3':
>     # First save as WAV
>     temp_wav = f"{output_path}_temp.wav"
>     torchaudio.save(temp_wav, audio.unsqueeze(0), sample_rate)
>
>     # Convert WAV to MP3 using pydub
>     audio_segment = AudioSegment.from_wav(temp_wav)
>     output_file = f"{output_path}.mp3"
>     audio_segment.export(output_file, format='mp3', bitrate='192k')
>
>     # Clean up temp file
>     os.remove(temp_wav)
> ```
> Two-step process: save as WAV, then convert to MP3 using pydub (which uses ffmpeg under the hood). We use 192kbps bitrate for good quality.
>
> **PCM/RAW Format** (lines ~410-420)
> ```python
> elif format.lower() in ['pcm', 'raw']:
>     output_file = f"{output_path}.pcm"
>     # Convert to 16-bit signed integers
>     audio_int16 = (audio * 32767).short()
>     # Write raw binary data
>     with open(output_file, 'wb') as f:
>         f.write(audio_int16.cpu().numpy().tobytes())
> ```
> Raw audio data with no header. Useful if you need to process the audio further in a custom pipeline. It's 16-bit signed integer PCM data.
>
> The method returns the output file path, so you can chain operations if needed."

---

## PART 7: CLI & PYTHON API (1 minute)

**[Screen: Scroll to the main block and CLI argument parsing]**

**Script:**

> "The code supports both CLI and Python library usage. Let me quickly show the CLI setup.
>
> [Scroll to the main block around lines 650-790]
>
> The argument parser is comprehensive:
> ```python
> parser = argparse.ArgumentParser(description='ChatterBox TTS Pipeline')
> parser.add_argument('--text', help='Text to convert to speech')
> parser.add_argument('--input', help='Input text file')
> parser.add_argument('--output', help='Output file path')
> parser.add_argument('--format', choices=['wav', 'mp3', 'pcm', 'raw'])
> parser.add_argument('--language', help='Language code')
> parser.add_argument('--voice', help='Voice reference file or URL')
> parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'])
> parser.add_argument('--exaggeration', type=float, default=0.5)
> parser.add_argument('--cfg-weight', type=float, default=0.5)
> parser.add_argument('--temperature', type=float, default=0.8)
> ```
>
> Then in the main execution:
> ```python
> pipeline = ChatterBoxPipeline(device=args.device)
> audio, sr = pipeline.generate(
>     text=args.text,
>     language=args.language,
>     voice_path=args.voice,
>     exaggeration=args.exaggeration,
>     cfg_weight=args.cfg_weight,
>     temperature=args.temperature
> )
> pipeline.save_audio(audio, sr, args.output, format=args.format)
> ```
>
> **As a Library:**
> You can also import it:
> ```python
> from chatterbox_pipeline import ChatterBoxPipeline
>
> pipeline = ChatterBoxPipeline()
> audio, sr = pipeline.generate("Hello!")
> pipeline.save_audio(audio, sr, "output/test", format="mp3")
> ```
>
> Both interfaces give you full control."

---

## PART 8: LIVE DEMO TIME (4 minutes)

**[Screen: Terminal ready to execute commands]**

**Script:**

> "Alright, enough code. Let's see this in action. I'll run through several demos to show you what it can do.
>
> **Demo 1: Basic English Generation**
> ```bash
> python chatterbox_pipeline.py --text "Hello! This is a demonstration of the ChatterBox pipeline."
> ```
>
> [Run the command, wait for completion]
>
> Watch the output - it detects the language as English, uses the English model, and saves to output/ directory.
>
> [Play the generated audio file]
>
> Clear, natural speech.
>
> **Demo 2: Multilingual - Spanish with MP3 Output**
> ```bash
> python chatterbox_pipeline.py --text "Hola! ¿Cómo estás? Bienvenido al pipeline de ChatterBox." --format mp3
> ```
>
> [Run command]
>
> Notice it auto-detected Spanish and switched to the multilingual model. Generated an MP3 file this time.
>
> [Play the Spanish MP3]
>
> **Demo 3: Processing a Text File**
> Let me show you the Spanish example file first:
> ```bash
> cat input/example_spanish.txt
> ```
>
> [Show the file contents]
>
> Now convert it:
> ```bash
> python chatterbox_pipeline.py --input input/example_spanish.txt --output output/spanish_demo --format wav
> ```
>
> [Run command]
>
> [Play the output]
>
> **Demo 4: Custom Parameters for Expressiveness**
> ```bash
> python chatterbox_pipeline.py --text "This is more dramatic!" --exaggeration 1.2 --temperature 1.0 --format mp3
> ```
>
> [Run command and play]
>
> Hear the difference? Higher exaggeration makes it more expressive.
>
> **Demo 5: Run the Quick Test Suite**
> ```bash
> python tests/quick_test.py
> ```
>
> [Run it, let it complete]
>
> This runs 5 tests: English, Spanish, MP3 format, PCM format, and basic functionality. Takes about 30-60 seconds. All tests should pass.
>
> [Show the generated output files]
>
> You can see all the generated audio files in the output/ directory. Let me play a couple more to show you the quality across different languages.
>
> [Play 1-2 more test outputs if time permits]
>
> That's the system working end-to-end - from text input to audio output, across multiple languages and formats."

---

## PART 9: KEY TAKEAWAYS & HANDOVER INFO (1 minute)

**[Screen: Split screen - code on left, documentation tree on right]**

**Script:**

> "Let me wrap up with the key code components and where to find everything.
>
> **Core Architecture - Remember These:**
>
> 1. **ChatterBoxPipeline class** - The main class in `chatterbox_pipeline.py`
>    - `__init__()` - Loads both models, sets up caching
>    - `detect_language()` - Auto language detection
>    - `generate()` - Core TTS logic with model selection
>    - `get_voice_path()` / `download_voice_file()` - Voice caching
>    - `save_audio()` - Multi-format output (WAV/MP3/PCM)
>
> 2. **Language Config** - `config/language_config.py`
>    - `LANGDETECT_TO_LANGUAGE` - Language mappings
>    - `DEFAULT_VOICES` - Voice URLs for 23 languages
>
> 3. **Dual-Model Strategy**
>    - English → `ChatterboxTTS` (higher quality)
>    - Other 22 languages → `ChatterboxMultilingualTTS`
>
> **Critical Code Patterns:**
> - Device auto-detection with GPU fallback
> - MD5-based voice file caching for cross-platform support
> - Two-step MP3 conversion via WAV intermediate
> - Language detection with English fallback
>
> **Where to Find Resources:**
> - **Installation** → README.md, SETUP_WINDOWS.md
> - **Voice Cloning** → VOICES.md
> - **Technical Details** → PROJECT_SUMMARY.md
> - **Language Config** → config/language_config.py
> - **All Code** → chatterbox_pipeline.py (790 lines, well-commented)
> - **Tests** → tests/quick_test.py (5 tests), tests/test_multilingual.py (100+ tests)
>
> **Your First Steps:**
> 1. Setup: Follow SETUP_WINDOWS.md or run setup.sh
> 2. Validate: Run `python tests/quick_test.py`
> 3. Experiment: Try the demo commands
> 4. Read the code: Start with the ChatterBoxPipeline class
> 5. Customize: Modify config/language_config.py if needed"

---

## CLOSING (30 seconds)

**[Screen: Show the project folder with all files visible]**

**Script:**

> "That's it! You've seen:
> - The complete code walkthrough - from initialization to audio output
> - How the dual-model architecture works
> - Language detection and voice cloning implementation
> - Live demos across multiple languages and formats
> - Where to find everything you need
>
> This is production-ready code. It's tested, documented, and ready to build on.
>
> **Key Files to Remember:**
> - `chatterbox_pipeline.py` - All the code
> - `config/language_config.py` - Language settings
> - `README.md` - Your primary reference
> - `tests/quick_test.py` - Validation
>
> Everything is here. The code is clean, the docs are comprehensive, and it works across 23 languages.
>
> Good luck with the project!"

---

## DEMO COMMAND CHEAT SHEET

**For Live Demo Copy-Paste:**

```bash
# Demo 1: Basic English
python chatterbox_pipeline.py --text "Hello! This is a demonstration of the ChatterBox pipeline."

# Demo 2: Spanish with MP3
python chatterbox_pipeline.py --text "Hola! ¿Cómo estás? Bienvenido al pipeline de ChatterBox." --format mp3

# Demo 3: Spanish from file
python chatterbox_pipeline.py --input input/example_spanish.txt --output output/spanish_demo --format wav

# Demo 4: With custom parameters (more expressive)
python chatterbox_pipeline.py --text "This is more dramatic!" --exaggeration 1.2 --temperature 1.0 --format mp3

# Demo 5: Quick test suite
python tests/quick_test.py

# Bonus: Show input file
cat input/example_spanish.txt
```

---

## PREPARATION CHECKLIST

**Before Recording:**

- [ ] Navigate to chatterbox-pipeline directory
- [ ] Activate virtual environment
- [ ] Clear output/ directory for clean demo: `rm -rf output/* || mkdir -p output`
- [ ] Test all demo commands work beforehand
- [ ] Have VS Code open with these files ready in tabs:
  - chatterbox_pipeline.py (main code)
  - config/language_config.py
  - tests/quick_test.py
  - README.md
- [ ] Have audio player ready (vlc, mpv, or system default)
- [ ] Check microphone and screen recording setup
- [ ] Terminal font size large enough to read (at least 14pt)
- [ ] Zoom level in VS Code readable (120-150%)

**Screen Setup:**
- **Primary window**: VS Code with chatterbox_pipeline.py open
- **Secondary window**: Terminal ready to execute commands
- **Have ready**: File browser for output/ directory
- **Audio player**: Ready to play .wav and .mp3 files

**Important Code Sections to Show (line numbers approximate):**
- Lines 1-50: Imports and class definition
- Lines 60-120: `__init__()` method
- Lines 120-150: `detect_language()` method
- Lines 150-250: `generate()` method (core TTS)
- Lines 250-330: `get_voice_path()` and `download_voice_file()` methods
- Lines 340-420: `save_audio()` method with format handling
- Lines 650-790: CLI argument parsing and main block

**Timing Breakdown:**
- Intro: 1 min
- Structure Overview: 1 min
- Initialization Code: 2 min
- Language Detection: 1.5 min
- Generate Method: 2.5 min
- Voice Caching: 1.5 min
- Save Audio Formats: 1.5 min
- CLI & API: 1 min
- Live Demos: 4 min
- Key Takeaways: 1 min
- Closing: 0.5 min
- **Total: ~15 minutes**

---

## PRESENTER NOTES

**Tone:** Technical but clear - this is a code walkthrough for developers

**Pacing:**
- Speak clearly and at a moderate pace
- When demos run, use that time to explain what's happening
- Don't rush through code sections - give viewers time to read
- Pause briefly when switching between files

**Key Messages:**
1. **Code is well-structured** - clean separation of concerns
2. **Dual-model architecture** - English gets special treatment for quality
3. **Smart caching** - Voice files handled intelligently for cross-platform support
4. **Production-ready** - Error handling, fallbacks, comprehensive testing
5. **Easy to extend** - Clear entry points in config/language_config.py

**What to Emphasize:**
- The `generate()` method - this is the heart of the system
- Dual-model selection logic (if English vs. else multilingual)
- Voice file caching strategy (MD5 hashing, local storage)
- Multi-format output handling (WAV direct, MP3 via conversion, PCM raw)
- How language detection works with the config mapping
- Code is documented and easy to follow

**What to Show Clearly:**
- Actual code snippets - scroll slowly
- Terminal output from demos
- Generated audio files in output/ directory
- File structure in VS Code explorer
- Play at least 2-3 audio samples

**Code Walkthrough Tips:**
- Use VS Code's minimap for navigation
- Use Cmd/Ctrl+F to jump to methods quickly
- Highlight important lines as you discuss them
- Show the full method, then explain key parts
- Reference line numbers when discussing code

**Demo Tips:**
- Let commands complete fully before moving on
- Show the terminal output - it's informative
- Play the audio to prove it works
- Show both English and at least one other language
- Demonstrate different output formats (WAV and MP3 minimum)

**If You Run Out of Time:**
- Skip or shorten the live demos
- Focus on code walkthrough - that's the priority
- Can reference test outputs instead of running live

**If You Have Extra Time:**
- Show config/language_config.py in detail
- Walk through a test file (quick_test.py)
- Demonstrate voice cloning with custom file
- Show the comprehensive test suite structure

Good luck with the recording! Remember: code walkthrough first, demos second.
