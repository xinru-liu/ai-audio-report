# Audio Transcription and Analysis Tool

A comprehensive tool for converting audio recordings (meetings, podcasts, lectures, etc.) into structured conference notes with AI-generated insights.

## Features

- **Audio Transcription**: Convert speech to text using OpenAI's Whisper model
- **Topic Extraction**: Identify the main themes discussed  
- **Action Item Detection**: Find potential tasks and commitments
- **Entity Recognition**: Identify people, organizations, and other entities mentioned
- **Key Takeaways**: Generate concise summary points of the most important information
- **Summarization**: Create condensed versions of each segment
- **Visualization**: Generate word clouds and topic distribution charts
- **GPU Acceleration**: Optimized for NVIDIA GPUs (specifically tested with RTX 3070)

## Windows Setup Instructions

### Prerequisites

- Windows 10 or 11
- Python 3.8 or newer
- NVIDIA GPU with at least 4GB VRAM (optimized for RTX 3070)
- NVIDIA drivers installed

### Automatic Setup

1. Download all files from this repository
2. Run `setup.bat` as administrator
3. Follow the on-screen instructions

### Manual Setup

1. Create a Python virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install PyTorch with CUDA support:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Install other required packages:
   ```
   pip install openai-whisper transformers nltk spacy sentence-transformers keybert matplotlib pandas seaborn wordcloud tqdm
   pip install bert-extractive-summarizer markdown
   ```

4. Download models and resources:
   ```
   python -m spacy download en_core_web_md
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
   ```

5. Install FFmpeg:
   - Download from https://www.gyan.dev/ffmpeg/builds/ (ffmpeg-release-full version)
   - Extract the archive and copy the bin folder to a permanent location
   - Add the bin folder to your PATH environment variable

6. Create directories for audio files and output:
   ```
   mkdir audio_files
   mkdir output
   ```

## Usage

1. Activate the virtual environment (if not already activated):
   ```
   venv\Scripts\activate
   ```

2. Run the script:
   ```
   python audio_analysis.py
   ```

3. Follow the on-screen instructions:
   - Select an audio file when prompted
   - Choose a model size based on your needs and available GPU memory
   - Wait for processing to complete
   - View the generated conference notes and visualizations

## Model Size Selection Guide

| Model Size | VRAM Required | Processing Speed | Accuracy | Recommended For |
|------------|---------------|------------------|----------|-----------------|
| tiny       | ~1GB          | Fastest          | Lowest   | Quick tests, short files |
| base       | ~1GB          | Fast             | Good     | General purpose, most users |
| small      | ~2GB          | Medium           | Better   | Better handling of accents |
| medium     | ~5GB          | Slow             | High     | When accuracy is important |
| large      | ~10GB         | Very Slow        | Highest  | Maximum accuracy (may not work on RTX 3070) |

## Output Files

The tool generates several files in the `output` directory:

- `conference_notes.md`: Markdown document with complete analysis
- `conference_notes.html`: HTML version of the notes for easy viewing
- `wordcloud.png`: Visual representation of frequently mentioned terms
- `topic_distribution.png`: Bar chart showing the relevance of key topics
- `[filename]_transcription.txt`: Raw transcript of the audio

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:
- Try a smaller model size (base or tiny)
- Close other applications using the GPU
- For very long recordings, try splitting the audio file

### Transcription Accuracy Issues

If transcription quality is poor:
- Try a larger model size if your GPU can handle it
- Ensure the audio quality is reasonable; consider cleaning it with audio software
- For non-English content, specify the language in the code

### FFmpeg Not Found

If you see "FFmpeg not found in PATH" errors:
- Make sure FFmpeg is properly installed
- Verify the bin directory is in your PATH environment variable
- Try restarting your computer after adding to PATH

## License

This project is provided for educational and personal use.

## Acknowledgments

This tool uses several open-source libraries and models:
- OpenAI's Whisper for speech recognition
- Hugging Face Transformers for NLP tasks
- BERT, BART and T5 models for various text analysis tasks
- SpaCy for named entity recognition
- NLTK for text processing