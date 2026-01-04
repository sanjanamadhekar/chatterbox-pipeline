@echo off
REM ============================================================================
REM ChatterBox TTS Pipeline - Automated Test Runner for Windows
REM ============================================================================
REM
REM This batch file handles the complete setup and testing process:
REM   1. Checks for Python installation
REM   2. Creates a virtual environment (if needed)
REM   3. Installs all required dependencies
REM   4. Runs comprehensive TTS tests
REM   5. Displays test results and generated files
REM
REM Usage:
REM   Simply double-click this file or run from command prompt:
REM   > run_tests.bat
REM
REM Requirements:
REM   - Python 3.10 or higher installed
REM   - Internet connection for downloading dependencies and models
REM   - ~10GB disk space for models and dependencies
REM   - (Optional) NVIDIA GPU with CUDA for faster processing
REM
REM Author: Sanjana Madhekar
REM License: MIT
REM ============================================================================

setlocal enabledelayedexpansion

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ============================================================================
echo ChatterBox TTS Pipeline - Windows Test Runner
echo ============================================================================
echo.
echo Current Directory: %CD%
echo Script Location: %SCRIPT_DIR%
echo.

REM ============================================================================
REM Step 1: Check Python Installation
REM ============================================================================
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.10 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo    ✓ Python %PYTHON_VERSION% found
echo.

REM ============================================================================
REM Step 2: Create Virtual Environment
REM ============================================================================
echo [2/6] Setting up virtual environment...
if not exist "venv\" (
    echo    Creating new virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Failed to create virtual environment!
        echo.
        pause
        exit /b 1
    )
    echo    ✓ Virtual environment created
) else (
    echo    ✓ Virtual environment already exists
)
echo.

REM ============================================================================
REM Step 3: Activate Virtual Environment
REM ============================================================================
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to activate virtual environment!
    echo.
    pause
    exit /b 1
)
echo    ✓ Virtual environment activated
echo.

REM ============================================================================
REM Step 4: Install Dependencies
REM ============================================================================
echo [4/6] Installing dependencies...
echo.
echo    This may take 5-10 minutes on first run...
echo    Dependencies will be cached for future runs.
echo.

REM Upgrade pip first
echo    [4a] Upgrading pip...
python -m pip install --upgrade pip --quiet
if %errorlevel% neq 0 (
    echo    WARNING: Failed to upgrade pip, continuing anyway...
)

REM Check if PyTorch is already installed
python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo    [4b] Installing PyTorch 2.6.0 with CUDA support...
    echo         (This is the largest dependency, ~2-3GB)
    pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --quiet
    if %errorlevel% neq 0 (
        echo.
        echo    WARNING: CUDA PyTorch installation failed, trying CPU version...
        pip install torch==2.6.0 torchaudio==2.6.0 --quiet
        if %errorlevel% neq 0 (
            echo.
            echo    ERROR: Failed to install PyTorch!
            echo.
            pause
            exit /b 1
        )
    )
    echo    ✓ PyTorch installed
) else (
    echo    ✓ PyTorch already installed
)

echo.
echo    [4c] Installing other dependencies...
pip install langdetect pydub librosa transformers diffusers safetensors --quiet
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo.
    pause
    exit /b 1
)

echo    [4d] Installing ChatterBox TTS...
pip install chatterbox-tts --quiet
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install ChatterBox TTS!
    echo.
    pause
    exit /b 1
)

echo    ✓ All dependencies installed
echo.

REM ============================================================================
REM Step 5: Run Tests
REM ============================================================================
echo [5/6] Running ChatterBox TTS tests...
echo.
echo ============================================================================
echo.

REM Run the quick test suite
python tests\quick_test.py
if %errorlevel% neq 0 (
    echo.
    echo ============================================================================
    echo ERROR: Tests failed!
    echo ============================================================================
    echo.
    echo Please check the error messages above for details.
    echo.
    echo Common issues:
    echo   - Out of memory: Try using CPU mode (edit the script)
    echo   - Network error: Check internet connection for model downloads
    echo   - Missing ffmpeg: MP3 conversion requires ffmpeg installed
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo.

REM ============================================================================
REM Step 6: Display Results
REM ============================================================================
echo [6/6] Test Results Summary
echo ============================================================================
echo.
echo ✓ All tests completed successfully!
echo.
echo Generated audio files:
echo.

REM List all generated test files
if exist "tests\output\" (
    for %%f in (tests\output\*.*) do (
        set "filename=%%~nxf"
        set "filesize=%%~zf"
        REM Convert bytes to KB
        set /a filesizeKB=!filesize! / 1024
        echo    - %%~nxf  (!filesizeKB! KB)
    )
    echo.
    echo Output directory: %CD%\tests\output\
) else (
    echo    No output files found (this shouldn't happen)
)

echo.
echo ============================================================================
echo Next Steps
echo ============================================================================
echo.
echo 1. Listen to the generated audio files in: tests\output\
echo.
echo 2. Run your own tests:
echo    python chatterbox_pipeline.py --text "Your text here"
echo.
echo 3. Try different languages:
echo    python chatterbox_pipeline.py --text "Bonjour le monde" --language fr
echo.
echo 4. See README.md for more usage examples and options
echo.
echo 5. To run comprehensive tests (all 23 languages):
echo    python tests\test_multilingual.py
echo.
echo ============================================================================
echo.
echo Press any key to open the output folder...
pause >nul

REM Open the output folder in Windows Explorer
if exist "tests\output\" (
    start "" "%CD%\tests\output\"
)

echo.
echo Done! You can close this window.
echo.
pause
