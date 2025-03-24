@echo off
ECHO ====================================================
ECHO Audio Transcription and Analysis Tool - Setup Script
ECHO ====================================================
ECHO.
ECHO This script will set up the environment for the Audio Transcription
ECHO and Analysis tool on your Windows system with NVIDIA GPU support.
ECHO.
ECHO Requirements:
ECHO - Python 3.8 or newer
ECHO - NVIDIA GPU with CUDA support
ECHO - Internet connection for downloading packages
ECHO.
ECHO The setup will:
ECHO 1. Create a Python virtual environment
ECHO 2. Install required Python packages
ECHO 3. Download models and resources
ECHO 4. Create necessary directories
ECHO.
ECHO Press Ctrl+C at any time to cancel.
ECHO.
PAUSE

:: Check if Python is installed
python --version > nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Python not found. Please install Python 3.8 or newer.
    GOTO :END
)

:: Create directories
ECHO.
ECHO Creating directories...
IF NOT EXIST audio_files mkdir audio_files
IF NOT EXIST output mkdir output

:: Create virtual environment
ECHO.
ECHO Creating Python virtual environment...
python -m venv venv
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Failed to create virtual environment.
    GOTO :END
)

:: Activate virtual environment
ECHO.
ECHO Activating virtual environment...
CALL venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Failed to activate virtual environment.
    GOTO :END
)

:: Install PyTorch with CUDA support
ECHO.
ECHO Installing PyTorch with CUDA support (this may take a while)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
IF %ERRORLEVEL% NEQ 0 (
    ECHO [WARNING] Failed to install PyTorch with CUDA support.
    ECHO Attempting to install CPU version as fallback...
    pip install torch torchvision torchaudio
)

:: Install other required packages
ECHO.
ECHO Installing required packages (this may take a while)...
pip install openai-whisper transformers nltk spacy sentence-transformers keybert matplotlib pandas seaborn wordcloud tqdm
pip install bert-extractive-summarizer markdown
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Failed to install required packages.
    GOTO :END
)

:: Download spaCy model
ECHO.
ECHO Downloading spaCy language model...
python -m spacy download en_core_web_md
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Failed to download spaCy model.
    GOTO :END
)

:: Download NLTK resources
ECHO.
ECHO Downloading NLTK resources...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Failed to download NLTK resources.
    GOTO :END
)

:: Check for FFmpeg
ECHO.
ECHO Checking for FFmpeg...
ffmpeg -version > nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [WARNING] FFmpeg not found in PATH.
    ECHO.
    ECHO To process audio files, you need to install FFmpeg:
    ECHO 1. Download from https://www.gyan.dev/ffmpeg/builds/ (ffmpeg-release-full version)
    ECHO 2. Extract the archive and copy the bin folder to a permanent location
    ECHO 3. Add the bin folder to your PATH environment variable
    ECHO.
    ECHO The analysis tool will still install, but audio conversion may not work.
)

:: Success message
ECHO.
ECHO ====================================================
ECHO Setup completed successfully!
ECHO ====================================================
ECHO.
ECHO To use the Audio Transcription and Analysis tool:
ECHO 1. Activate the virtual environment: venv\Scripts\activate
ECHO 2. Run the script: python audio_analysis.py
ECHO.
ECHO Enjoy analyzing your audio files!
ECHO.

:END
PAUSE