# Audio Transcription and Analysis for Conference Notes
# Windows Local Setup with NVIDIA GPU Support
# Optimized for NVIDIA 3070 with Robust Error Handling

# ==========================================
# SETUP INSTRUCTIONS (Run these in Command Prompt before running this script)
# ==========================================

# 1. Create a Python virtual environment (recommended)
#    > python -m venv venv
#    > venv\Scripts\activate
#
# 2. Install required packages:
#    > pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#    > pip install openai-whisper transformers nltk spacy sentence-transformers keybert matplotlib pandas seaborn wordcloud tqdm
#    > pip install bert-extractive-summarizer markdown
#    > python -m spacy download en_core_web_md
#    > python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
#
# 3. Install FFmpeg for Windows:
#    - Download from https://www.gyan.dev/ffmpeg/builds/ (ffmpeg-release-full version)
#    - Extract and add the bin folder to your PATH environment variable
#
# 4. Run this script:
#    > python audio_analysis.py [optional: path_to_audio_file]

# ==========================================
# LIBRARY IMPORTS
# ==========================================

import os
import sys
import re
import time
import json
import torch
import whisper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import subprocess
import threading
import traceback  # For detailed error reporting

# NLP Libraries
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from summarizer import Summarizer
import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from wordcloud import WordCloud
from rapidfuzz import fuzz  # Install with: pip install rapidfuzz

from transformers import BitsAndBytesConfig

# ==========================================
# CONFIGURATION
# ==========================================



# Define directory paths - these are relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio_files")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Create directories if they don't exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download required NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

# GPU Configuration for NVIDIA 3070
def setup_gpu():
    """
    Configure GPU settings for optimal performance on NVIDIA 3070.
    
    Returns:
        bool: True if GPU is available and configured, False otherwise
    """
    print("Checking GPU availability...")
    try:
        if torch.cuda.is_available():
            # Get GPU information
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
            
            print(f"GPU detected: {gpu_name}")
            print(f"GPU memory: {gpu_memory:.2f} GB")

            torch.cuda.empty_cache()
            
            # Setting memory settings to avoid out of memory errors
            # These can be adjusted based on your specific setup
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 90% of GPU memory
            
            # Enable autotuning for the best performance
            torch.backends.cudnn.benchmark = True
            
            return True
        else:
            print("No GPU detected, using CPU only.")
            print("This will be significantly slower than using a GPU.")
            print("Please check your CUDA installation if you expected GPU support.")
            return False
    except Exception as e:
        print(f"Error during GPU setup: {e}")
        print("Continuing with CPU only...")
        return False

# Check for FFmpeg installation
def check_ffmpeg():
    """
    Check if FFmpeg is installed and available in PATH.
    
    Returns:
        bool: True if FFmpeg is available, False otherwise
    """
    print("Checking for FFmpeg installation...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW  # Prevents console window from appearing
        )
        if result.returncode == 0:
            print("FFmpeg installation detected.")
            return True
        else:
            print(f"FFmpeg check returned error code: {result.returncode}")
            print(f"stderr: {result.stderr}")
            return False
    except FileNotFoundError:
        print("ERROR: FFmpeg not found in PATH.")
        print("Please install FFmpeg and add it to your PATH environment variable.")
        print("Download from: https://www.gyan.dev/ffmpeg/builds/")
        return False
    except Exception as e:
        print(f"Unexpected error checking FFmpeg: {e}")
        return False

# Try to load spaCy model with fallbacks for different versions
def load_spacy_model():
    """
    Try to load the spaCy model with fallbacks for different versions.
    
    Returns:
        spacy.Language: Loaded spaCy model or None on failure
    """
    print("Loading spaCy language model...")
    try:
        # First try to load the medium model
        return spacy.load("en_core_web_md")
    except (OSError, IOError) as e:
        print(f"Medium spaCy model error: {e}")
        try:
            # Fall back to small model
            print("Medium spaCy model not found, trying small model...")
            return spacy.load("en_core_web_sm")
        except (OSError, IOError) as e:
            print(f"Small spaCy model error: {e}")
            try:
                # If no model is installed, try to download the small model
                print("No spaCy model found. Downloading small English model...")
                spacy.cli.download("en_core_web_sm")
                return spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"Failed to download spaCy model: {e}")
                print("Some NLP features will be limited.")
                return None

# ==========================================
# AUDIO PROCESSING FUNCTIONS
# ==========================================

def load_audio(file_path):
    """
    Load and preprocess audio file for transcription.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        str: Processed file path
    """
    print(f"Loading audio file: {file_path}")
    
    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Check if file needs conversion
    filename, extension = os.path.splitext(file_path)
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac']
    
    if extension.lower() not in valid_extensions:
        print(f"Converting {extension} file to .wav format...")
        output_path = f"{filename}.wav"
        
        # Windows-friendly ffmpeg command
        cmd = [
            "ffmpeg", 
            "-v", "error", 
            "-i", file_path, 
            "-ar", "16000", 
            "-ac", "1", 
            output_path
        ]
        
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW  # Windows-specific flag
        )
        
        if result.returncode != 0:
            print(f"Error converting audio: {result.stderr}")
            return file_path
            
        print(f"Converted to: {output_path}")
        return output_path
    
    return file_path

def segment_audio_file(input_file, segment_length=600):  # 600 seconds = 10 minutes
    """Split audio file into smaller segments for more reliable processing"""
    
    output_dir = os.path.join(os.path.dirname(input_file), "segments")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get duration using ffprobe
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        input_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    
    # Create segments
    segment_files = []
    for i in range(0, int(duration), segment_length):
        output_file = os.path.join(output_dir, f"segment_{i//segment_length}.wav")
        
        # Use ffmpeg to extract segment
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", input_file,
            "-ss", str(i),
            "-t", str(min(segment_length, duration-i)),
            "-c:a", "pcm_s16le",
            "-ar", "16000",
            output_file
        ]
        
        subprocess.run(cmd)
        segment_files.append(output_file)
    
    return segment_files

def transcribe_audio(file_path, model_size="base", use_gpu=True, language=None):
    """
    Transcribe audio file using OpenAI's Whisper model with GPU acceleration.
    
    Args:
        file_path (str): Path to the audio file
        model_size (str): Whisper model size (tiny, base, small, medium, large)
        use_gpu (bool): Whether to use GPU for transcription
        language (str): Optional language code to force language detection
        
    Returns:
        dict: Transcription results
    """
    print(f"Transcribing audio using Whisper {model_size} model...")
    
    # Set the device (CUDA for GPU, CPU otherwise)
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Using GPU acceleration for transcription")
    else:
        print("Using CPU for transcription (slower)")
    
    start_time = time.time()
    
    # Load the Whisper model with the specified device
    model = whisper.load_model(model_size, device=device)
    
    # Transcribe the audio with options to improve quality
    # More options: https://github.com/openai/whisper/blob/main/whisper/transcribe.py
    result = model.transcribe(
        file_path, 
        verbose=False,
        fp16=device == "cuda",  # Use half-precision for GPU to save memory
        language=language,      # Set language if specified
        initial_prompt="This is a professional discussion in clear English.",  # Help model with context
        condition_on_previous_text=True,  # Maintain consistency
        temperature=0.0,  # Use greedy decoding for more predictable output
        compression_ratio_threshold=2.4,  # Helps avoid repetition loops
        no_speech_threshold=0.6,  # Higher threshold to ignore background noise
        patience=1.0,  # More deterministic beam search
        beam_size=5
    )
    
    elapsed_time = time.time() - start_time
    print(f"Transcription completed in {elapsed_time:.2f} seconds")
    
    # Save transcription to file
    base_name = os.path.basename(file_path).split('.')[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_transcription.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result["text"])
    
    # Also save detailed JSON with segments and timing
    json_path = os.path.join(OUTPUT_DIR, f"{base_name}_transcription.json")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"Transcription saved to: {output_path}")
    print(f"Detailed JSON with timing saved to: {json_path}")
    
    return result

def remove_repetitions(transcript_path):
    """
    Clean up a transcript by removing excessive repetitions
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Find repeated phrases (4+ words repeated)
    words = text.split()
    cleaned_words = []
    i = 0
    
    while i < len(words):
        # Check for repetitions of 4-word phrases
        if i + 8 < len(words):
            phrase1 = ' '.join(words[i:i+4])
            phrase2 = ' '.join(words[i+4:i+8])
            
            if phrase1.lower() == phrase2.lower():
                # Skip the repetition
                cleaned_words.extend(words[i:i+4])
                i += 8
                continue
        
        cleaned_words.append(words[i])
        i += 1
    
    cleaned_text = ' '.join(cleaned_words)
    
    # Save cleaned transcript
    clean_path = transcript_path.replace('.txt', '_cleaned.txt')
    with open(clean_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    return clean_path

# ==========================================
# TEXT ANALYSIS FUNCTIONS
# ==========================================

def segment_speakers(transcript, min_segment_length=100, max_segment_length=1000,segment_markers=None):
    """
    Segment transcript into probable speaker turns based on pauses and content.
    
    Args:
        transcript (str): Full transcript text
        min_segment_length (int): Minimum characters per segment
        max_segment_length (int): Maximum characters per segment
        
    Returns:
        list: List of segments
    """
    print("Segmenting transcript into probable speaker turns...")
    
    # If we have explicit segment markers from multi-segment processing
    if segment_markers:
        # Use these as additional split points
        for marker in segment_markers:
            transcript = transcript.replace(marker, "\n\n" + marker + "\n\n")
    
    # First split by clear paragraph or long pauses
    initial_segments = re.split(r'\n\n+|\.\s+(?=[A-Z])', transcript)
    
    # Refine segments to appropriate lengths
    refined_segments = []
    current_segment = ""
    
    for segment in initial_segments:
        segment = segment.strip()
        if not segment:
            continue
            
        if len(current_segment) + len(segment) < max_segment_length:
            current_segment += " " + segment if current_segment else segment
        else:
            if current_segment:
                refined_segments.append(current_segment.strip())
            current_segment = segment
    
    # Add the last segment if it exists
    if current_segment:
        refined_segments.append(current_segment.strip())
    
    # Further split any remaining long segments
    final_segments = []
    for segment in refined_segments:
        if len(segment) > max_segment_length:
            sentences = sent_tokenize(segment)
            temp_segment = ""
            
            for sentence in sentences:
                if len(temp_segment) + len(sentence) < max_segment_length:
                    temp_segment += " " + sentence if temp_segment else sentence
                else:
                    if temp_segment:
                        final_segments.append(temp_segment.strip())
                    temp_segment = sentence
                    
            if temp_segment:
                final_segments.append(temp_segment.strip())
        else:
            final_segments.append(segment)
    
    print(f"Identified {len(final_segments)} potential speaker segments")
    return final_segments

def extract_topics(transcript, num_topics=5, use_gpu=True):

    """
    Extract key topics from transcript using KeyBERT with GPU acceleration.
    
    Args:
        transcript (str): Transcript text
        num_topics (int): Number of topics to extract
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        list: List of extracted topics with scores
    """
    print("Extracting key topics from transcript...")
    
    try:
        # Set the device for sentence transformers
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Initialize KeyBERT with GPU support if available
        kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        
        # Force the model to use the specified device
        if hasattr(kw_model.model, 'to'):
            kw_model.model = kw_model.model.to(device)
        
        # Extract keywords with KeyBERT
        keywords = kw_model.extract_keywords(
            transcript, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            use_mmr=True, 
            diversity=0.7, 
            top_n=num_topics
        )
        
        print(f"Extracted {len(keywords)} topics")
        return keywords
    
    except Exception as e:
        print(f"Error extracting topics: {e}")
        print("Falling back to simple keyword extraction...")
        
        # Simple fallback for topic extraction using frequency
        try:
            # Tokenize and filter words
            words = word_tokenize(transcript.lower())
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
            
            # Count word frequencies
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top words by frequency
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:num_topics]
            
            # Normalize scores to 0-1 range
            max_score = top_words[0][1] if top_words else 1
            normalized_topics = [(word, score/max_score) for word, score in top_words]
            
            print(f"Extracted {len(normalized_topics)} topics using fallback method")
            return normalized_topics
        
        except Exception as inner_e:
            print(f"Even fallback topic extraction failed: {inner_e}")
            # Return default topics if all else fails
            return [("topic1", 1.0), ("topic2", 0.8), ("topic3", 0.6), 
                    ("topic4", 0.4), ("topic5", 0.2)]
                    
def generate_key_takeaways(transcript, summaries, topics, num_takeaways=5, use_gpu=True):
    """
    Generate insightful key takeaways from the transcript using DeepSeek 7B model with
    quantization to fit on consumer GPUs like RTX 3070.
    
    This function uses:
    1. Content prioritization to focus on important sections
    2. Topic-guided deep analysis
    3. Multi-stage reasoning process
    4. Format-aware generation
    
    Args:
        transcript (str): Full transcript text
        summaries (list): Segment summaries
        topics (list): Extracted topics with scores
        num_takeaways (int): Number of takeaways to generate
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        list: Insightful key takeaways
    """
    print("Generating key takeaways with DeepSeek 7B model...")
    print(f"Generating key takeaways from full transcript ({len(transcript)} chars)...")
    
    try:
        # Add necessary imports for DeepSeek and quantization
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        
        # Set device for model
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using {device} for takeaway generation")
        
        # Configure 4-bit quantization to fit 7B model in 8GB VRAM
        if device == "cuda":
            print("Setting up 4-bit quantization for DeepSeek model...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
            
        # Step 1: Content Prioritization - identify most important segments
        segment_importances = prioritize_content(summaries, topics)
        
        # Step 2: Multi-stage analysis
        all_takeaways = []
        
        # Load DeepSeek model with quantization
        print("Loading DeepSeek model (this may take a minute)...")
        model_name = "deepseek-ai/deepseek-llm-1.3b-chat"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model with quantization if on GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto"  # Automatically determine device mapping
        )
        
        # First stage: Analyze main topics individually
        topic_takeaways = analyze_topics_with_deepseek(
            transcript, summaries, topics, segment_importances, 
            tokenizer, model, device
        )
        all_takeaways.extend(topic_takeaways)
        
        # Second stage: Identify cross-topic relationships and implications
        if len(topics) > 1:
            relationship_takeaways = analyze_relationships_with_deepseek(
                transcript, summaries, topics, segment_importances,
                tokenizer, model, device
            )
            all_takeaways.extend(relationship_takeaways)
        
        # Third stage: Identify high-level insights and implications
        implication_takeaways = analyze_implications_with_deepseek(
            transcript, summaries, topics, segment_importances,
            tokenizer, model, device
        )
        all_takeaways.extend(implication_takeaways)
        
        # Step 3: Deduplication and quality filtering
        final_takeaways = filter_and_deduplicate(all_takeaways, num_takeaways)
        
        # Step 4: Format takeaways consistently
        formatted_takeaways = format_takeaways(final_takeaways)
        
        # Clean up GPU memory
        if device == "cuda":
            del model
            torch.cuda.empty_cache()
        
        print(f"Generated {len(formatted_takeaways)} insightful takeaways")
        return formatted_takeaways
        
    except Exception as e:
        print(f"Error in advanced takeaway generation with DeepSeek: {e}")
        # Fall back to a more basic approach
        return fallback_takeaway_generation(transcript, summaries, topics, num_takeaways, use_gpu)


def analyze_topics_with_deepseek(transcript, summaries, topics, importances, tokenizer, model, device):
    """
    Analyze individual topics in depth using DeepSeek model.
    
    Args:
        transcript (str): Full transcript text
        summaries (list): Segment summaries
        topics (list): Extracted topics [(topic, score), ...]
        importances (list): Importance scores for each segment
        tokenizer: Model tokenizer
        model: Pre-loaded DeepSeek model
        device: Computing device (cuda/cpu)
        
    Returns:
        list: Topic-specific takeaways
    """
    print("Analyzing main topics...")
    topic_takeaways = []
    
    # Focus on the top topics
    top_topics = topics[:min(5, len(topics))]
    
    for topic_name, topic_score in top_topics:
        # Create context for this topic by finding relevant segments
        relevant_context = extract_topic_context(topic_name, summaries, importances)
        
        # Create a specific prompt for deeper analysis of this topic using DeepSeek format
        prompt = f"""<User>
I have a meeting transcript about various topics, including "{topic_name}" (relevance score: {topic_score:.2f}).

Here is the relevant meeting content about this topic:
{relevant_context}

Based on this content, provide 1-2 insightful takeaways specifically about "{topic_name}".
Each takeaway should:
- Go beyond summarizing to provide real insight
- Include specific details from the discussion
- Explain why this is significant

Format each takeaway as a complete paragraph starting with "Takeaway:"
</User>"""
        
        # Generate analysis for this topic
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=300,  # Control length of generation
                temperature=0.3,     # Lower temperature for more focused output
                top_p=0.9,           # Nucleus sampling
                do_sample=True,      # Enable sampling
                pad_token_id=tokenizer.eos_token_id  # Prevent pad token errors
            )
        
        # Extract the generated response (excluding the input prompt)
        result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract takeaways from the result
        takeaway_matches = re.findall(r'Takeaway:\s*(.*?)(?=Takeaway:|$)', result, re.DOTALL)
        
        for match in takeaway_matches:
            takeaway = match.strip()
            if takeaway:
                topic_takeaways.append(f"[Topic: {topic_name}] {takeaway}")
    
    return topic_takeaways


def analyze_relationships_with_deepseek(transcript, summaries, topics, importances, tokenizer, model, device):
    """
    Analyze relationships between topics using DeepSeek model.
    
    Args:
        transcript (str): Full transcript text
        summaries (list): Segment summaries
        topics (list): Extracted topics [(topic, score), ...]
        importances (list): Importance scores for each segment
        tokenizer: Model tokenizer
        model: Pre-loaded DeepSeek model
        device: Computing device (cuda/cpu)
        
    Returns:
        list: Relationship takeaways
    """
    print("Analyzing relationships between topics...")
    
    # Only proceed if we have multiple topics
    if len(topics) < 2:
        return []
    
    # Get top topics for relationship analysis
    top_topics = topics[:min(4, len(topics))]
    topic_names = [t[0] for t in top_topics]
    
    # Combine top summaries for context
    top_indices = [i for i, score in sorted(enumerate(importances), key=lambda x: x[1], reverse=True)][:5]
    top_summaries = [summaries[i] for i in top_indices]
    context = "\n".join(top_summaries)
    
    # Create a prompt for relationship analysis using DeepSeek format
    prompt = f"""<User>
In a meeting, the following topics were discussed:
{", ".join(topic_names)}

Here is the key context from the meeting:
{context}

Analyze how these topics interconnect in the discussion. Provide 1-2 insightful takeaways about the relationships, 
connections, or contrasts between these topics. Focus on how they influence each other.

Format each takeaway as a complete paragraph starting with "Relationship:"
</User>"""
    
    # Generate analysis for topic relationships
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=300,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract the generated response
    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Extract relationship takeaways
    takeaway_matches = re.findall(r'Relationship:\s*(.*?)(?=Relationship:|$)', result, re.DOTALL)
    
    relationship_takeaways = []
    for match in takeaway_matches:
        takeaway = match.strip()
        if takeaway:
            relationship_takeaways.append(f"[Relationship] {takeaway}")
    
    return relationship_takeaways


def analyze_implications_with_deepseek(transcript, summaries, topics, importances, tokenizer, model, device):
    """
    Analyze broader implications using DeepSeek model.
    
    Args:
        transcript (str): Full transcript text
        summaries (list): Segment summaries
        topics (list): Extracted topics [(topic, score), ...]
        importances (list): Importance scores for each segment
        tokenizer: Model tokenizer
        model: Pre-loaded DeepSeek model
        device: Computing device (cuda/cpu)
        
    Returns:
        list: Implication takeaways
    """
    print("Analyzing broader implications...")
    
    # Get main topic themes
    topic_themes = ", ".join([t[0] for t in topics[:min(3, len(topics))]])
    
    # Combine important summaries to create context
    weighted_summaries = [(summary, importance) for summary, importance in zip(summaries, importances)]
    weighted_summaries.sort(key=lambda x: x[1], reverse=True)
    
    important_context = "\n".join([s[0] for s in weighted_summaries[:5]])
    
    # Create a prompt for implications analysis using DeepSeek format
    prompt = f"""<User>
In a meeting about {topic_themes}, the following key points were discussed:

{important_context}

Based on this discussion, provide 2-3 high-level takeaways focusing on:
1. The broader implications of what was discussed
2. Potential future impacts or directions
3. The most significant insights that weren't explicitly stated

Go beyond summarizing to provide real analytical value.
Format each takeaway as a complete paragraph starting with "Implication:"
</User>"""
    
    # Generate analysis for implications
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=350,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract the generated response
    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Extract implication takeaways
    takeaway_matches = re.findall(r'Implication:\s*(.*?)(?=Implication:|$)', result, re.DOTALL)
    
    implication_takeaways = []
    for match in takeaway_matches:
        takeaway = match.strip()
        if takeaway:
            implication_takeaways.append(f"[Implication] {takeaway}")
    
    return implication_takeaways


def prioritize_content(summaries, topics):
    """
    Prioritize content by assigning importance scores to each segment.
    
    Args:
        summaries (list): List of segment summaries
        topics (list): Extracted topics [(topic, score), ...]
        
    Returns:
        list: Importance scores for each segment
    """
    importances = []
    
    # Extract topic terms for checking
    top_topics = [topic[0].lower() for topic in topics[:min(5, len(topics))]]
    
    for summary in summaries:
        summary_lower = summary.lower()
        
        # Base importance score
        importance = 1.0
        
        # Topic relevance - segments mentioning top topics are more important
        topic_matches = sum(1 for topic in top_topics if topic in summary_lower)
        importance += topic_matches * 0.5
        
        # Content density - more information-rich segments are important
        words = summary.split()
        if len(words) > 0:
            # Segments with specific details (numbers, percentages, dates) are more important
            if re.search(r'\d+%|\d+\.\d+|\b\d{4}\b', summary):
                importance += 0.5
            
            # Segments with action-oriented language are important
            action_verbs = ['should', 'must', 'need', 'recommend', 'require', 'propose', 'plan']
            if any(verb in summary_lower for verb in action_verbs):
                importance += 0.3
        
        importances.append(importance)
    
    # Normalize scores to a 0-1 range
    if importances:
        max_importance = max(importances)
        if max_importance > 0:
            importances = [score / max_importance for score in importances]
    
    return importances


def extract_topic_context(topic, summaries, importances, max_chars=2000):
    """
    Extract relevant context for a specific topic.
    
    Args:
        topic (str): Topic to find context for
        summaries (list): Segment summaries
        importances (list): Importance scores for each segment
        max_chars (int): Maximum characters to include
        
    Returns:
        str: Relevant context for the topic
    """
    # Calculate relevance scores for this topic
    relevance_scores = []
    topic_lower = topic.lower()
    
    for i, summary in enumerate(summaries):
        score = importances[i]  # Base on general importance
        
        # Add bonus for segments mentioning the topic
        if topic_lower in summary.lower():
            score += 1.0
            
            # Extra bonus for segments where the topic appears multiple times
            matches = re.findall(re.escape(topic_lower), summary.lower())
            if len(matches) > 1:
                score += 0.2 * (len(matches) - 1)
        
        relevance_scores.append((i, score))
    
    # Sort segments by relevance to this topic
    relevance_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Build context with the most relevant segments first
    context = []
    current_length = 0
    
    for idx, _ in relevance_scores:
        summary = summaries[idx]
        if current_length + len(summary) <= max_chars:
            context.append(f"Segment {idx+1}: {summary}")
            current_length += len(summary)
        else:
            break
    
    return "\n\n".join(context)


def filter_and_deduplicate(takeaways, num_takeaways):
    """
    Filter, deduplicate and select the best takeaways.
    
    Args:
        takeaways (list): All generated takeaways
        num_takeaways (int): Number of takeaways to keep
        
    Returns:
        list: Filtered takeaways
    """
    if not takeaways:
        return []
    
    # Remove any very short takeaways
    valid_takeaways = [t for t in takeaways if len(t) > 50]
    
    # Remove duplicates and near-duplicates
    unique_takeaways = []
    seen_content = set()
    
    for takeaway in valid_takeaways:
        # Create simplified content for comparison
        # Remove the category prefix [Topic/Relationship/Implication]
        cleaned = re.sub(r'^\[(Topic|Relationship|Implication)[^\]]*\]\s*', '', takeaway)
        # Simplify to core content by removing extra spaces and lowercasing
        simple_content = re.sub(r'\s+', ' ', cleaned.lower()).strip()
        
        # Check if this is similar to anything we've seen
        is_duplicate = False
        for existing in seen_content:
            # Simple similarity check - could be improved with proper fuzzy matching
            # if two takeaways share 80% of their words in common, consider them similar
            words1 = set(simple_content.split())
            words2 = set(existing.split())
            
            if len(words1) == 0 or len(words2) == 0:
                continue
                
            overlap = len(words1.intersection(words2))
            similarity = overlap / min(len(words1), len(words2))
            
            if similarity > 0.7:  # Threshold for similarity
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_content.add(simple_content)
            unique_takeaways.append(takeaway)
    
    # Order by type (topic-specific first, then relationships, then implications)
    ordered_takeaways = []
    
    # First add topic takeaways
    for takeaway in unique_takeaways:
        if takeaway.startswith("[Topic:"):
            ordered_takeaways.append(takeaway)
    
    # Then relationship takeaways
    for takeaway in unique_takeaways:
        if takeaway.startswith("[Relationship]"):
            ordered_takeaways.append(takeaway)
    
    # Finally implication takeaways
    for takeaway in unique_takeaways:
        if takeaway.startswith("[Implication]"):
            ordered_takeaways.append(takeaway)
    
    # Add any that didn't match the patterns
    for takeaway in unique_takeaways:
        if not any(takeaway.startswith(prefix) for prefix in ["[Topic:", "[Relationship]", "[Implication]"]):
            ordered_takeaways.append(takeaway)
    
    # Limit to requested number
    return ordered_takeaways[:num_takeaways]


def format_takeaways(takeaways):
    """
    Format takeaways for final presentation.
    
    Args:
        takeaways (list): List of takeaways to format
        
    Returns:
        list: Formatted takeaways
    """
    formatted = []
    
    for i, takeaway in enumerate(takeaways, 1):
        # Remove the category prefix for cleaner presentation
        clean_takeaway = re.sub(r'^\[(Topic|Relationship|Implication)[^\]]*\]\s*', '', takeaway)
        
        # Ensure proper capitalization and punctuation
        if clean_takeaway:
            # Capitalize first letter
            clean_takeaway = clean_takeaway[0].upper() + clean_takeaway[1:]
            
            # Add period if missing end punctuation
            if not clean_takeaway[-1] in ['.', '!', '?']:
                clean_takeaway += '.'
            
            # Add to formatted list
            formatted.append(f"{i}. {clean_takeaway}")
    
    return formatted


def fallback_takeaway_generation(transcript, summaries, topics, num_takeaways=5, use_gpu=True):
    """
    Fallback method for takeaway generation when the advanced approach fails.
    
    This uses simpler methods like extractive summarization.
    
    Args:
        transcript (str): Full transcript text
        summaries (list): Segment summaries
        topics (list): Extracted topics
        num_takeaways (int): Number of takeaways to generate
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        list: Key takeaways
    """
    print("Using fallback takeaway generation method...")
    
    try:
        # Get all sentences from summaries
        all_sentences = []
        for summary in summaries:
            sentences = sent_tokenize(summary)
            for sentence in sentences:
                if len(sentence.split()) >= 10:  # Only consider substantial sentences
                    all_sentences.append(sentence)
        
        # Calculate importance based on topics
        sentence_scores = []
        for sentence in all_sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Higher score for sentences mentioning key topics
            for topic, topic_score in topics:
                if topic.lower() in sentence_lower:
                    score += topic_score
            
            # Higher score for sentences with numbers (often more specific)
            if re.search(r'\d+', sentence):
                score += 0.5
            
            sentence_scores.append((sentence, score))
        
        # Sort by score and add the top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Format final takeaways
        final_takeaways = []
        seen = set()
        
        for sentence, _ in sentence_scores:
            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue
                
            # Skip duplicates
            sentence_lower = sentence.lower()
            if sentence_lower in seen:
                continue
                
            seen.add(sentence_lower)
            
            # Ensure proper capitalization and punctuation
            formatted = sentence[0].upper() + sentence[1:]
            if not formatted[-1] in ['.', '!', '?']:
                formatted += '.'
                
            final_takeaways.append(formatted)
            
            if len(final_takeaways) >= num_takeaways:
                break
        
        # Limit to requested number and add numbering
        final_takeaways = final_takeaways[:num_takeaways]
        formatted_takeaways = [f"{i+1}. {t}" for i, t in enumerate(final_takeaways)]
        
        print(f"Generated {len(formatted_takeaways)} takeaways using fallback method")
        return formatted_takeaways
        
    except Exception as e:
        print(f"Error in fallback takeaway generation: {e}")
        # Ultimate fallback - return generic placeholders
        return [f"{i+1}. Key point from the transcript." for i in range(num_takeaways)]
        
def create_word_cloud(transcript, max_words=100, width=800, height=400):
    """
    Create a word cloud visualization of the transcript with improved filtering.
    
    Args:
        transcript (str): Transcript text
        max_words (int): Maximum number of words to include
        width (int): Width of the output image
        height (int): Height of the output image
        
    Returns:
        tuple: (Path to saved image, base64 encoded image for embedding)
    """
    print("Generating word cloud visualization...")
    
    try:
        # Get standard stopwords
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        
        # Add custom colloquial words and fillers to filter out
        custom_stopwords = {
            # Common filler words
            'yeah', 'uh', 'um', 'like', 'sort', 'kind', 'you know', 'i mean',
            # Affirmations
            'right', 'okay', 'ok', 'mhm', 'hmm', 'yep', 'yup', 'sure',
            # Hedges
            'actually', 'basically', 'literally', 'probably', 'maybe', 
            # Other low-information words
            'going', 'get', 'got', 'getting', 'go', 'goes', 'went',
            'thing', 'things', 'something', 'anything', 'everything', 'nothing',
            'just', 'really', 'very', 'quite', 'bit',
            # Time references
            'today', 'yesterday', 'tomorrow', 'now', 'then', 
            # Common weak verbs
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
            'has', 'have', 'had', 'having', 'do', 'does', 'did', 'doing',
            # Personal pronouns (usually not topically relevant)
            'i', 'me', 'my', 'mine', 'myself', 
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'we', 'us', 'our', 'ours', 'ourselves', 
            'they', 'them', 'their', 'theirs', 'themselves',
            # Single letters that might appear
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        }
        
        # Combine standard and custom stopwords
        all_stopwords = stop_words.union(custom_stopwords)
        
        # Create word cloud with enhanced settings
        wordcloud = WordCloud(
            width=width, 
            height=height,
            background_color='white',
            stopwords=all_stopwords,
            max_words=max_words,
            collocations=True,  # Allow bigrams for more meaningful phrases
            collocation_threshold=10,  # Higher threshold for more significant bigrams
            min_word_length=3,  # Ignore very short words
            random_state=42,  # Consistent layout between runs
            colormap='viridis'  # Professional color scheme
        ).generate(transcript)
        
        # Save the wordcloud image
        output_path = os.path.join(OUTPUT_DIR, 'wordcloud.png')
        plt.figure(figsize=(width/100, height/100))  # Convert pixels to inches
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Create base64 encoding for embedding in HTML/MD
        import base64
        from io import BytesIO
        
        img_buffer = BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        print(f"Word cloud saved to: {output_path}")
        return output_path, img_data
    
    except Exception as e:
        print(f"Error creating word cloud: {e}")
        return None, None

def summarize_transcript(segments, model_name="facebook/bart-large-cnn", use_gpu=True, batch_size=4):
    """
    Summarize transcript segments using a pretrained model with GPU acceleration.
    
    Args:
        segments (list): List of transcript segments
        model_name (str): Pretrained summarization model name
        use_gpu (bool): Whether to use GPU acceleration
        batch_size (int): Batch size for processing multiple segments at once
        
    Returns:
        list: Summarized segments
    """
    print(f"Summarizing transcript segments using {model_name}...")
    
    try:
        # Set device for transformers
        device = -1  # CPU
        if use_gpu and torch.cuda.is_available():
            device = 0  # First GPU
        
        # Initialize the summarization pipeline
        summarizer = pipeline(
            "summarization", 
            model=model_name, 
            device=device,
            framework="pt"  # Use PyTorch backend
        )
        
        summaries = []
        
        # Process segments in batches to improve GPU utilization
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            valid_batch = []
            indices = []
            
            # Filter out segments that are too short
            for j, segment in enumerate(batch):
                if len(segment.split()) >= 30:
                    valid_batch.append(segment)
                    indices.append(j)
                else:
                    # Add short segments directly
                    summaries.append(segment)
            
            if valid_batch:
                try:
                    # Generate summaries for the batch
                    batch_summaries = summarizer(
                        valid_batch,
                        max_length=100,
                        min_length=30,
                        do_sample=False,
                        batch_size=len(valid_batch)  # Process all at once
                    )
                    
                    # Insert the summaries back in the right order
                    for k, summary in enumerate(batch_summaries):
                        # Find where to insert this summary
                        while len(summaries) <= indices[k] + i:
                            summaries.append(None)
                        summaries[indices[k] + i] = summary['summary_text']
                    
                except Exception as e:
                    print(f"Error summarizing batch {i//batch_size}: {e}")
                    # Add the original segments on error
                    for j, segment in enumerate(valid_batch):
                        while len(summaries) <= indices[j] + i:
                            summaries.append(None)
                        summaries[indices[j] + i] = segment
        
        # Ensure we have no None values
        summaries = [s if s is not None else "No summary available." for s in summaries]
        
        print(f"Generated {len(summaries)} segment summaries")
        return summaries
    
    except Exception as e:
        print(f"Error during summarization: {e}")
        print("Falling back to extractive summarization...")
        
        try:
            # Fallback to simple extractive summarization
            bert_model = Summarizer()
            
            fallback_summaries = []
            for segment in segments:
                try:
                    if len(segment.split()) >= 30:
                        # Use BERT extractive summarization
                        summary = bert_model(segment, num_sentences=3)
                        fallback_summaries.append(summary)
                    else:
                        # Short segments don't need summarization
                        fallback_summaries.append(segment)
                except:
                    # If even this fails, just use the original
                    fallback_summaries.append(segment)
            
            print(f"Generated {len(fallback_summaries)} fallback summaries")
            return fallback_summaries
        
        except Exception as inner_e:
            print(f"Fallback summarization also failed: {inner_e}")
            # If all summarization fails, return original segments
            print("Using original segments as summaries")
            return segments

def extract_action_items(transcript, use_gpu=True):
    """
    Extract potential action items from the transcript with GPU acceleration.
    
    Args:
        transcript (str): Transcript text
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        list: Extracted action items
    """
    print("Extracting potential action items...")
    
    try:
        # Load spaCy model
        nlp = load_spacy_model()
        
        if nlp is None:
            # If spaCy failed to load, use NLTK for basic sentence splitting
            print("Using NLTK for basic sentence tokenization (limited functionality)")
            sentences = sent_tokenize(transcript)
        else:
            # Process the transcript
            # For large transcripts, process in chunks to avoid memory issues
            max_chars = 100000  # Maximum characters to process at once
            
            if len(transcript) > max_chars:
                # Process in chunks
                chunks = [transcript[i:i+max_chars] for i in range(0, len(transcript), max_chars)]
                sentences = []
                for chunk in chunks:
                    doc = nlp(chunk)
                    sentences.extend([sent.text.strip() for sent in doc.sents])
            else:
                # Process whole transcript
                doc = nlp(transcript)
                sentences = [sent.text.strip() for sent in doc.sents]
        
        # Define action item indicators
        action_verbs = ["need to", "should", "must", "will", "going to", 
                        "have to", "plan to", "decide", "implement", 
                        "create", "develop", "establish", "organize"]
        
        # Identify potential action item sentences
        action_sentences = []
        
        for sent in sentences:
            # Skip short sentences
            if len(sent.split()) < 5:
                continue
                
            # Check for action indicators
            for verb in action_verbs:
                if verb.lower() in sent.lower():
                    action_sentences.append(sent)
                    break
        
        # Further refine to remove duplicates and similar items
        filtered_actions = []
        
        # Try to use sentence embeddings if available
        try:
            if use_gpu and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                
            sentence_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
            sentence_embeddings = sentence_embeddings.to(device)
            
            if action_sentences:
                embeddings = sentence_embeddings.encode(action_sentences)
                
                # Simple clustering to remove similar items
                threshold = 0.8
                indices_to_include = [0]  # Always include the first sentence
                
                for i in range(1, len(embeddings)):
                    # Check similarity with already included sentences
                    max_similarity = max([np.dot(embeddings[i], embeddings[j]) / 
                                         (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])) 
                                         for j in indices_to_include])
                    
                    if max_similarity < threshold:
                        indices_to_include.append(i)
                
                filtered_actions = [action_sentences[i] for i in indices_to_include]
            
        except Exception as e:
            print(f"Error in action item deduplication: {e}")
            # If deduplication fails, use original action sentences
            filtered_actions = action_sentences[:10]  # Limit to top 10
        
        print(f"Extracted {len(filtered_actions)} potential action items")
        return filtered_actions
    
    except Exception as e:
        print(f"Error extracting action items: {e}")
        return []  # Return empty list on failure

def identify_entities(transcript):
    """
    Identify key named entities in the transcript using spaCy.
    
    Args:
        transcript (str): Transcript text
        
    Returns:
        dict: Dictionary of entity types and lists of entities
    """
    print("Identifying key entities in transcript...")
    
    # Initialize empty entity dictionary
    entities = {
        "PERSON": [],
        "ORG": [],
        "PRODUCT": [],
        "GPE": [],  # Geopolitical entities (countries, cities)
        "EVENT": [],
        "OTHER": []
    }
    
    try:
        # Load spaCy model
        nlp = load_spacy_model()
        
        if nlp is None:
            print("Skipping entity identification due to missing spaCy model")
            return entities
        
        # For large transcripts, process in chunks
        max_chars = 100000  # Maximum characters to process at once
        
        if len(transcript) > max_chars:
            # Process in chunks
            chunks = [transcript[i:i+max_chars] for i in range(0, len(transcript), max_chars)]
            
            for chunk in chunks:
                doc = nlp(chunk)
                
                # Extract entities
                for ent in doc.ents:
                    if ent.label_ in entities:
                        entities[ent.label_].append(ent.text)
                    else:
                        entities["OTHER"].append((ent.text, ent.label_))
        else:
            # Process whole transcript
            doc = nlp(transcript)
            
            # Extract entities
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
                else:
                    entities["OTHER"].append((ent.text, ent.label_))
        
        # Count frequencies and get unique entities
        for category in entities:
            if category != "OTHER":
                counter = {}
                for item in entities[category]:
                    counter[item] = counter.get(item, 0) + 1
                
                # Sort by frequency
                sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
                entities[category] = sorted_items
        
        print(f"Identified entities: {sum(len(entities[k]) for k in entities)} total")
        return entities
    
    except Exception as e:
        print(f"Error identifying entities: {e}")
        return entities  # Return empty collections on failure



# ==========================================
# TRANSCRIPT ANALYSIS FUNCTION
# ==========================================

def analyze_transcript(transcript_path, use_gpu=True):
    """
    Analyze an existing transcript file without performing audio transcription.
    
    Args:
        transcript_path (str): Path to transcript text file
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        dict: Paths to generated outputs
    """
    print(f"\n{'='*50}")
    print(f"ANALYZING EXISTING TRANSCRIPT")
    print(f"{'='*50}\n")
    
    outputs = {
        "notes_md": None,
        "notes_html": None,
        "wordcloud": None,
        "topic_viz": None
    }
    
    try:
        # Read the transcript file
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
            
        print(f"Read transcript from: {transcript_path}")
        print(f"Transcript length: {len(transcript)} characters")
        
        # Process the transcription
        # Step 1: Segment transcript
        segments = segment_speakers(transcript)
        
        # Step 2: Extract topics
        topics = extract_topics(transcript, use_gpu=use_gpu)
        
        # Step 3: Create word cloud
        wordcloud_path = create_word_cloud(transcript)
        outputs["wordcloud"] = wordcloud_path
        
        # Step 4: Summarize transcript
        summaries = summarize_transcript(segments, use_gpu=use_gpu)
        
        # Step 5: Extract action items
        action_items = extract_action_items(transcript, use_gpu=use_gpu)
        
        # Step 6: Identify entities
        entities = identify_entities(transcript)
        
        # Step 7: Generate key takeaways
        takeaways = generate_key_takeaways(transcript, summaries, topics, use_gpu=use_gpu)
        
        # Step 8: Visualize topics
        topic_viz_path = visualize_topics(topics)
        outputs["topic_viz"] = topic_viz_path
        
        # Step 9: Generate conference notes
        notes_path = generate_conference_notes(
            transcript,
            segments,
            summaries,
            topics,
            action_items,
            entities,
            takeaways
        )
        
        outputs["notes_md"] = notes_path
        outputs["notes_html"] = os.path.join(OUTPUT_DIR, "conference_notes.html")
        
        # Print completion message
        print(f"\n{'='*50}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*50}\n")
        
        # Summarize the files that were generated
        print("Files generated:")
        print(f"1. Conference Notes (MD): {notes_path}")
        print(f"2. Conference Notes (HTML): {outputs['notes_html']}")
        print(f"3. Word Cloud: {wordcloud_path}" if wordcloud_path else "3. Word Cloud: Failed to generate")
        print(f"4. Topic Distribution: {topic_viz_path}" if topic_viz_path else "4. Topic Distribution: Failed to generate")
        
        return outputs
    
    except Exception as e:
        print(f"\nERROR: An error occurred during analysis: {e}")
        traceback.print_exc()
        return outputs  # Return whatever outputs were successfully generated

def visualize_topics(topics):
    """
    Create a visual representation of key topics.
    
    Args:
        topics (list): List of (topic, score) tuples
        
    Returns:
        tuple: (Path to saved image, base64 encoded image for embedding)
    """
    print("Creating topic visualization...")
    
    try:
        # Extract topics and scores
        topic_names = [t[0] for t in topics]
        topic_scores = [t[1] for t in topics]
        
        # Create horizontal bar chart with enhanced styling
        plt.figure(figsize=(10, 6))
        
        # Use a horizontal bar chart
        bars = plt.barh(
            topic_names, 
            topic_scores, 
            color=sns.color_palette("viridis", len(topics)),
            height=0.6,  # Slightly thinner bars
            edgecolor='white',  # White edges for better separation
            linewidth=0.5
        )
        
        # Add value labels to the end of each bar
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.01,  # Slight offset
                bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                va='center',
                fontsize=9
            )
        
        # Enhance styling
        plt.xlabel('Relevance Score', fontsize=11)
        plt.title('Key Topics Discussed', fontsize=14, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(OUTPUT_DIR, 'topic_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Create base64 encoding for embedding in HTML/MD
        import base64
        from io import BytesIO
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        plt.close()
        
        print(f"Topic visualization saved to: {output_path}")
        return output_path, img_data
    
    except Exception as e:
        print(f"Error creating topic visualization: {e}")
        return None, None

# Add this new function to your script for creating condensed bullet points

def create_condensed_summary(summaries, topics, num_points=15, use_gpu=True):
    """
    Create a condensed, bulleted summary from segment summaries.
    
    Args:
        summaries (list): List of segment summaries
        topics (list): List of key topics with scores
        num_points (int): Maximum number of bullet points to generate
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        list: Condensed bullet points
    """
    print(f"Creating condensed summary with {num_points} bullet points...")
    
    try:
        # First approach: Use topic-guided extraction
        # This is more reliable than generation for ensuring complete bullet points
        
        # Get all sentences from summaries
        all_sentences = []
        for summary in summaries:
            # Split into sentences and filter out very short ones
            sentences = [s.strip() for s in sent_tokenize(summary) if len(s.strip()) > 20]
            all_sentences.extend(sentences)
        
        # Remove duplicate or very similar sentences
        unique_sentences = []
        seen_content = set()
        
        for sentence in all_sentences:
            # Create a simplified representation for comparison (lowercase, no punctuation)
            simple_content = re.sub(r'[^\w\s]', '', sentence.lower())
            simple_content = ' '.join(simple_content.split())  # Normalize whitespace
            
            # Skip if we've seen very similar content before
            if any(fuzz.ratio(simple_content, existing) > 80 for existing in seen_content):
                continue
                
            seen_content.add(simple_content)
            unique_sentences.append(sentence)
        
        # Score sentences based on key topics and informational content
        sentence_scores = []
        topic_terms = [topic[0].lower() for topic in topics]
        
        for sentence in unique_sentences:
            sentence_lower = sentence.lower()
            
            # Base score
            score = 0
            
            # Topic relevance score - reward sentences containing key topics
            for topic in topic_terms:
                if topic in sentence_lower:
                    score += 10
            
            # Content density score - reward information-rich sentences
            words = word_tokenize(sentence_lower)
            content_words = [w for w in words if w not in stopwords.words('english') and w.isalpha()]
            content_ratio = len(content_words) / max(1, len(words))
            score += content_ratio * 5
            
            # Penalize very long sentences
            if len(words) > 30:
                score -= (len(words) - 30) * 0.1
                
            # Bonus for sentences with numbers (often more specific information)
            if any(w.isdigit() for w in words):
                score += 3
                
            # Store score with sentence
            sentence_scores.append((sentence, score))
        
        # Sort by score and select top sentences
        top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_points*2]
        
        # Now re-sort to maintain original order (more coherent)
        top_sentences.sort(key=lambda x: all_sentences.index(x[0]) if x[0] in all_sentences else 9999)
        
        # Take the top N, ensuring they're from different parts of the transcript
        final_sentences = []
        sections = max(1, len(top_sentences) // num_points)
        
        for i in range(0, len(top_sentences), sections):
            section = top_sentences[i:i+sections]
            if section:
                # Take the highest scored from each section
                best = max(section, key=lambda x: x[1])
                final_sentences.append(best[0])
                if len(final_sentences) >= num_points:
                    break
        
        # Format as bullet points
        bullet_points = [f"• {sentence}" for sentence in final_sentences[:num_points]]
        
        # If we don't have enough bullet points, try a different approach
        if len(bullet_points) < min(5, num_points):
            # Fallback to generation approach
            return create_summary_with_generation(summaries, topics, num_points, use_gpu)
        
        print(f"Generated {len(bullet_points)} condensed bullet points using extraction")
        return bullet_points
        
    except Exception as e:
        print(f"Error in extraction-based summary: {e}")
        # Try generation approach as fallback
        return create_summary_with_generation(summaries, topics, num_points, use_gpu)


def create_summary_with_generation(summaries, topics, num_points=15, use_gpu=True):
    """
    Create a condensed summary using text generation approach as fallback.
    
    This is used when the extraction-based approach doesn't yield good results.
    """
    print("Attempting generation-based summary approach...")
    
    try:
        # Set device
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Use a more reliable approach with smaller chunks
        # Instead of trying to generate all bullet points at once, we'll do it in batches
        
        # Prepare topic keywords for context
        topic_keywords = ", ".join([topic[0] for topic in topics[:5]])  # Top 5 topics
        
        # 1. First split the summaries into manageable chunks
        chunk_size = 5
        summary_chunks = [summaries[i:i+chunk_size] for i in range(0, len(summaries), chunk_size)]
        
        all_bullets = []
        
        # Process each chunk separately with T5 model
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        # 2. Generate bullet points for each chunk
        for i, chunk in enumerate(summary_chunks):
            # Combine the summaries in this chunk
            chunk_text = " ".join(chunk)
            
            # Generate a prompt for each chunk
            prompt = f"Extract 3-5 key points from this text about {topic_keywords}. Format each point as a complete sentence starting with a bullet point: {chunk_text[:3000]}"
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
            
            # Generate with careful parameters to ensure complete sentences
            outputs = model.generate(
                inputs.input_ids,
                max_length=200,
                min_length=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,  # Added this line
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract bullet points with a reliable pattern
            # We're looking for lines that start with •, -, *, or a number followed by a period
            bullet_matches = re.findall(r'(?:^|\n)[•\-\*]\s*(.*?)(?=\n[•\-\*]|\n\d+\.|\Z)', decoded_output, re.DOTALL)
            numbered_matches = re.findall(r'(?:^|\n)\d+\.\s*(.*?)(?=\n[•\-\*]|\n\d+\.|\Z)', decoded_output, re.DOTALL)
            
            chunk_bullets = bullet_matches + numbered_matches
            
            # If regex didn't find structured bullets, just split by newlines
            if not chunk_bullets:
                chunk_bullets = [line.strip() for line in decoded_output.split('\n') 
                               if line.strip() and len(line.strip()) > 15]
            
            # Clean up and add to our collection
            for bullet in chunk_bullets:
                cleaned = bullet.strip()
                if cleaned and len(cleaned) > 15:  # Ensure meaningful content
                    all_bullets.append(cleaned)
        
        # 3. Remove duplicates and near-duplicates
        unique_bullets = []
        seen_content = set()
        
        for bullet in all_bullets:
            # Create a simplified representation for comparison
            simple_content = re.sub(r'[^\w\s]', '', bullet.lower())
            simple_content = ' '.join(simple_content.split())
            
            # Skip if we've seen very similar content
            if any(fuzz.ratio(simple_content, existing) > 75 for existing in seen_content):
                continue
                
            seen_content.add(simple_content)
            unique_bullets.append(bullet)
        
        # 4. Format and limit
        formatted_bullets = []
        for bullet in unique_bullets[:num_points]:
            # Ensure each starts with a bullet point and ends with proper punctuation
            clean_bullet = re.sub(r'^[•\-\*\d.]+\s*', '', bullet).strip()
            
            # Add period if missing
            if not clean_bullet.endswith(('.', '!', '?')):
                clean_bullet += '.'
                
            # Add bullet point
            formatted_bullets.append(f"• {clean_bullet}")
        
        print(f"Generated {len(formatted_bullets)} condensed bullet points using generation")
        return formatted_bullets
        
    except Exception as e:
        print(f"Error in generation-based summary: {e}")
        
        # Ultimate fallback to simple sentence extraction
        try:
            print("Using simple sentence extraction as final fallback...")
            
            # Get all sentences
            all_sentences = []
            for summary in summaries:
                all_sentences.extend([s.strip() for s in sent_tokenize(summary) if len(s.strip()) > 15])
            
            # Remove duplicates
            unique_sentences = []
            seen = set()
            for s in all_sentences:
                s_lower = s.lower()
                if s_lower not in seen:
                    seen.add(s_lower)
                    unique_sentences.append(s)
            
            # Select sentences evenly spaced through the document
            selected = []
            if unique_sentences:
                step = max(1, len(unique_sentences) // num_points)
                for i in range(0, len(unique_sentences), step):
                    if len(selected) < num_points:
                        selected.append(unique_sentences[i])
            
            # Format as bullets
            bullets = [f"• {s}" for s in selected[:num_points]]
            
            print(f"Generated {len(bullets)} bullet points using simple extraction")
            return bullets
            
        except Exception as inner_e:
            print(f"All summary methods failed: {inner_e}")
            return [f"• Key point {i+1} from the transcript." for i in range(min(num_points, 10))]

# Now modify the generate_conference_notes function to use your new condensed summary
# and remove the original text excerpts

def generate_conference_notes(transcript,segments, summaries, topics, action_items, entities, takeaways,
                             wordcloud_data=None, topic_viz_data=None):
    """
    Generate comprehensive conference notes with all analysis components and embedded visualizations.
    
    Args:
        transcript (str): Full transcript text
        segments (list): Segmented transcript
        summaries (list): Segment summaries
        topics (list): Extracted topics
        action_items (list): Extracted action items
        entities (dict): Identified entities
        takeaways (list): Key takeaways
        wordcloud_data (tuple): Tuple of (file_path, base64_data) for wordcloud
        topic_viz_data (tuple): Tuple of (file_path, base64_data) for topic visualization
        
    Returns:
        str: Path to the generated notes file
    """
    print("Generating comprehensive conference notes...")
    
    try:
        # Generate condensed bullet point summary
        condensed_bullets = create_condensed_summary(summaries, topics, num_points=15)
        
        # Create a markdown document
        notes = []
        notes_html = []  # We'll use this for the HTML version
        
        # Add header
        notes.append("# Conference Notes")
        notes.append(f"*Generated on {time.strftime('%Y-%m-%d at %H:%M')}*")
        notes.append("")
        
        notes_html.append("# Conference Notes")
        notes_html.append(f"*Generated on {time.strftime('%Y-%m-%d at %H:%M')}*")
        notes_html.append("")
        
        # Add key takeaways section
        notes.append("## Key Takeaways")
        notes_html.append("## Key Takeaways")
        
        for takeaway in takeaways:
            notes.append(takeaway)
            notes_html.append(takeaway)
        
        notes.append("")
        notes_html.append("")
        
        # Add condensed summary section
        notes.append("## Condensed Summary")
        notes_html.append("## Condensed Summary")
        
        for bullet in condensed_bullets:
            notes.append(bullet)
            notes_html.append(bullet)
        
        notes.append("")
        notes_html.append("")
        
        # Add key topics section with visualizations
        notes.append("## Key Topics and Visualizations")
        notes_html.append("## Key Topics and Visualizations")
        
        # Add topic list
        for topic, score in topics:
            notes.append(f"- **{topic}** (relevance score: {score:.2f})")
            notes_html.append(f"- **{topic}** (relevance score: {score:.2f})")
        
        notes.append("")
        notes_html.append("")
        
        # Add word cloud visualization
        notes.append("### Word Cloud Visualization")
        notes_html.append("### Word Cloud Visualization")
        
        if wordcloud_data and wordcloud_data[1]:  # Check if we have the base64 data
            # For Markdown - just add a link to the image file
            notes.append(f"![Word Cloud]({os.path.basename(wordcloud_data[0])})")
            
            # For HTML - embed the image directly using base64
            notes_html.append(f'<img src="data:image/png;base64,{wordcloud_data[1]}" alt="Word Cloud" style="max-width:100%;">')
        else:
            notes.append("*Word cloud visualization not available*")
            notes_html.append("*Word cloud visualization not available*")
        
        notes.append("")
        notes_html.append("")
        
        # Add topic distribution visualization
        notes.append("### Topic Distribution")
        notes_html.append("### Topic Distribution")
        
        if topic_viz_data and topic_viz_data[1]:  # Check if we have the base64 data
            # For Markdown - just add a link to the image file
            notes.append(f"![Topic Distribution]({os.path.basename(topic_viz_data[0])})")
            
            # For HTML - embed the image directly using base64
            notes_html.append(f'<img src="data:image/png;base64,{topic_viz_data[1]}" alt="Topic Distribution" style="max-width:100%;">')
        else:
            notes.append("*Topic distribution visualization not available*")
            notes_html.append("*Topic distribution visualization not available*")
        
        notes.append("")
        notes_html.append("")
        
        # Add action items section
        notes.append("## Action Items")
        notes_html.append("## Action Items")
        
        if action_items:
            for i, action in enumerate(action_items, 1):
                notes.append(f"{i}. {action}")
                notes_html.append(f"{i}. {action}")
        else:
            notes.append("No specific action items identified.")
            notes_html.append("No specific action items identified.")
        
        notes.append("")
        notes_html.append("")
        
        # Add key participants/entities section
        notes.append("## Key Participants and Organizations")
        notes_html.append("## Key Participants and Organizations")
        
        if entities["PERSON"] and len(entities["PERSON"]) > 0:
            notes.append("### People")
            notes_html.append("### People")
            
            for person, count in entities["PERSON"][:10]:  # Top 10 people
                notes.append(f"- {person} (mentioned {count} times)")
                notes_html.append(f"- {person} (mentioned {count} times)")
        
        if entities["ORG"] and len(entities["ORG"]) > 0:
            notes.append("\n### Organizations")
            notes_html.append("\n### Organizations")
            
            for org, count in entities["ORG"][:10]:  # Top 10 organizations
                notes.append(f"- {org} (mentioned {count} times)")
                notes_html.append(f"- {org} (mentioned {count} times)")
        
        notes.append("")
        notes_html.append("")
        
        # Add detailed summary section
        # notes.append("## Detailed Summary")
        # notes_html.append("## Detailed Summary")
        
        for i, summary in enumerate(summaries, 1):
            notes.append(f"### Segment {i}")
            notes.append(f"{summary}")
            notes.append("")
            
            notes_html.append(f"### Segment {i}")
            notes_html.append(f"{summary}")
            notes_html.append("")
        
        # Add full transcript section
        notes.append("## Full Transcript")
        notes.append("<details>")
        notes.append("<summary>Click to expand full transcript</summary>")
        notes.append("")
        notes.append("```")
        notes.append(transcript)
        notes.append("```")
        notes.append("</details>")
        
        notes_html.append("## Full Transcript")
        notes_html.append("<details>")
        notes_html.append("<summary>Click to expand full transcript</summary>")
        notes_html.append("<pre>")
        notes_html.append(transcript.replace("<", "&lt;").replace(">", "&gt;"))  # Escape HTML characters
        notes_html.append("</pre>")
        notes_html.append("</details>")
        
        # Join all sections
        notes_text = "\n".join(notes)
        notes_html_text = "\n".join(notes_html)
        
        # Save to markdown file
        output_path = os.path.join(OUTPUT_DIR, "conference_notes.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(notes_text)
        
        # Save HTML version
        html_output_path = os.path.join(OUTPUT_DIR, "conference_notes.html")
        try:
            # Create custom HTML directly (not using markdown conversion)
            # This allows us to embed the base64 images properly
            with open(html_output_path, 'w', encoding='utf-8') as f:
                f.write("<!DOCTYPE html>\n<html><head>")
                f.write("<title>Conference Notes</title>")
                f.write("<style>")
                f.write("body { font-family: system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.5; }")
                f.write("h1 { color: #333; border-bottom: 1px solid #ddd; padding-bottom: 10px; }")
                f.write("h2 { color: #444; margin-top: 30px; border-bottom: 1px solid #eee; padding-bottom: 5px; }")
                f.write("h3 { color: #555; }")
                f.write("pre { background: #f5f5f5; padding: 10px; overflow: auto; border-radius: 3px; }")
                f.write("img { max-width: 100%; margin: 15px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 3px; }")
                f.write("details { background: #f9f9f9; padding: 10px; border-radius: 3px; margin: 15px 0; }")
                f.write("summary { cursor: pointer; font-weight: bold; }")
                f.write("</style>")
                f.write("</head><body>")
                
                # Convert the markdown-like format to HTML
                import re
                html_content = notes_html_text
                
                # Handle headers
                html_content = re.sub(r'# (.*?)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
                html_content = re.sub(r'## (.*?)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
                html_content = re.sub(r'### (.*?)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
                
                # Handle bullet points
                html_content = re.sub(r'- (.*?)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
                html_content = re.sub(r'(\n<li>.*?</li>\n)+', r'<ul>\n\g<0>\n</ul>', html_content, flags=re.DOTALL)
                
                # Handle numbered lists
                html_content = re.sub(r'\d+\. (.*?)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
                html_content = re.sub(r'(\n<li>.*?</li>\n)+', r'<ol>\n\g<0>\n</ol>', html_content, flags=re.DOTALL)
                
                # Handle emphasis
                html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)
                html_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_content)
                
                # Handle paragraphs (consecutive lines)
                html_content = re.sub(r'(?<!\n)\n(?!\n)(?!<[uo]l>|<li>|<h|<p|<div|<img|<pre)', r'<br>', html_content)
                
                # Fix potential issues with list structures
                html_content = html_content.replace('<ul>\n<br>', '<ul>\n')
                html_content = html_content.replace('<br>\n</ul>', '\n</ul>')
                html_content = html_content.replace('<ol>\n<br>', '<ol>\n')
                html_content = html_content.replace('<br>\n</ol>', '\n</ol>')
                
                f.write(html_content)
                f.write("</body></html>")
            
            print(f"HTML notes saved to: {html_output_path}")
        except Exception as html_e:
            print(f"Error generating HTML: {html_e}")
            
            # Fallback to using markdown
            try:
                import markdown
                with open(html_output_path, 'w', encoding='utf-8') as f:
                    f.write("<html><head><title>Conference Notes</title>")
                    f.write("<style>body{font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto;max-width:800px;margin:0 auto;padding:20px;} h1{color:#333} h2{color:#555} pre{background:#f5f5f5;padding:10px;overflow:auto;}</style>")
                    f.write("</head><body>")
                    f.write(markdown.markdown(notes_text))
                    f.write("</body></html>")
                print(f"HTML notes (fallback method) saved to: {html_output_path}")
            except ImportError:
                print("Could not generate HTML output (markdown package not installed)")
        
        print(f"Conference notes saved to: {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Error generating conference notes: {e}")
        # Create a simple version on error
        simple_path = os.path.join(OUTPUT_DIR, "simple_notes.md")
        with open(simple_path, 'w', encoding='utf-8') as f:
            f.write("# Simple Conference Notes\n\n")
            f.write("*Error occurred during full notes generation*\n\n")
            f.write("## Transcript\n\n")
            f.write(transcript)
        
        print(f"Simple notes saved to: {simple_path}")
        return simple_path

# Update the process_audio_to_notes function to use these enhanced visualizations
# (This is just the relevant part that needs to be updated)

# ==========================================
# MAIN EXECUTION
# ==========================================
def process_audio_to_notes(audio_path, model_size="base", use_gpu=True, language=None, segment_length=0, clean_repetitions=False):
    """
    End-to-end process from audio file to conference notes.
    
    Args:
        audio_path (str): Path to the audio file
        model_size (str): Whisper model size
        use_gpu (bool): Whether to use GPU acceleration
        language (str): Optional language code to force language detection
        
    Returns:
        dict: Paths to generated outputs
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING AUDIO TO CONFERENCE NOTES")
    print(f"{'='*50}\n")
    
    # Dictionary to store output paths
    outputs = {
        "transcript_txt": None,
        "transcript_json": None,
        "notes_md": None,
        "notes_html": None,
        "wordcloud": None,
        "topic_viz": None
    }
    
    try:
        # Step 1: Load and process audio
        processed_audio = load_audio(audio_path)
        
         # Step 2: Check if we should segment the audio
        if segment_length > 0:
            print(f"Segmenting audio into {segment_length}-second chunks...")
            segment_files = segment_audio_file(processed_audio, segment_length)
            print(f"Created {len(segment_files)} segments")
            
            # Transcribe each segment
            full_transcript = ""
            all_segments = []
            
            for i, segment_file in enumerate(segment_files):
                print(f"Transcribing segment {i+1}/{len(segment_files)}...")
                segment_result = transcribe_audio(segment_file, model_size, use_gpu, language)
                full_transcript += segment_result["text"] + "\n\n"
                all_segments.extend(segment_result.get("segments", []))
                
                # Clean up memory after each segment
                if use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Create a combined result dictionary
            transcription_result = {
                "text": full_transcript,
                "segments": all_segments
            }
            # Use full_transcript for all subsequent processing
            transcript = full_transcript
        else:
            # No segmentation, transcribe the whole file
            transcription_result = transcribe_audio(processed_audio, model_size, use_gpu, language)
            transcript = transcription_result["text"]

        transcript = transcription_result["text"]
        
        # Save paths to outputs
        base_name = os.path.basename(audio_path).split('.')[0]
        transcript_path = os.path.join(OUTPUT_DIR, f"{base_name}_transcription.txt")
        outputs["transcript_txt"] = transcript_path
        
        # Save the full transcript
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)


        # Apply repetition removal if requested
        if clean_repetitions:  # Add this parameter to the function
            print("Cleaning up repetitions in transcript...")
            cleaned_path = remove_repetitions(transcript_path)
            outputs["transcript_txt_cleaned"] = cleaned_path
            
            # Use the cleaned transcript for further analysis
            with open(cleaned_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
                    
        # Save JSON with timing if available
        json_path = os.path.join(OUTPUT_DIR, f"{base_name}_transcription.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_result, f, indent=2)
        outputs["transcript_json"] = json_path
        
        # Let the user know transcription is done and analysis is beginning
        print("\nTranscription complete! Beginning analysis phase...")
        
        # Process the transcription
        # Step 3: Segment transcript
        segments = segment_speakers(transcript)
        
        # Step 4: Extract topics
        topics = extract_topics(transcript, use_gpu=use_gpu)
        
        # Step 5: Create word cloud
        wordcloud_data = create_word_cloud(transcript)  # Now returns (path, base64)
        outputs["wordcloud"] = wordcloud_data[0] if wordcloud_data else None
        
        # Step 6: Summarize transcript
        summaries = summarize_transcript(segments, use_gpu=use_gpu)
        
        # Step 7: Extract action items
        action_items = extract_action_items(summaries, use_gpu=use_gpu)
        
        # Step 8: Identify entities
        entities = identify_entities(transcript)
        
        # Step 9: Generate key takeaways
        takeaways = generate_key_takeaways(transcript, summaries, topics, use_gpu=use_gpu)
        
        # Step 10: Visualize topics
        topic_viz_data = visualize_topics(topics)  # Now returns (path, base64)
        outputs["topic_viz"] = topic_viz_data[0] if topic_viz_data else None
        
        # Step 11: Generate conference notes
        notes_path = generate_conference_notes(
            transcript,
            segments,
            summaries,
            topics,
            action_items,
            entities,
            takeaways,
            wordcloud_data,  # Pass both the file path and base64 data
            topic_viz_data
        )
        
        outputs["notes_md"] = notes_path
        outputs["notes_html"] = os.path.join(OUTPUT_DIR, "conference_notes.html")
        
        # Print completion message
        print(f"\n{'='*50}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*50}\n")
        
        # Summarize the files that were generated
        print("Files generated:")
        print(f"1. Conference Notes (MD): {notes_path}")
        print(f"2. Conference Notes (HTML): {outputs['notes_html']}")
        print(f"3. Word Cloud: {outputs['wordcloud']}" if outputs['wordcloud'] else "3. Word Cloud: Failed to generate")
        print(f"4. Topic Distribution: {outputs['topic_viz']}" if outputs['topic_viz'] else "4. Topic Distribution: Failed to generate") 
        print(f"5. Transcript: {outputs['transcript_txt']}")
        print(f"6. Detailed Transcript with Timing: {outputs['transcript_json']}")
        
        return outputs
    
    except Exception as e:
        print(f"\nERROR: An error occurred during processing: {e}")
        traceback.print_exc()
        return outputs  # Return whatever outputs were successfully generated

if __name__ == "__main__":
    try:
        print("\n" + "-"*60)
        print(" Audio Transcription and Analysis Tool")
        print(" Windows Local Setup with NVIDIA GPU Support")
        print("-"*60 + "\n")
        
        # Check system requirements
        have_ffmpeg = check_ffmpeg()
        if not have_ffmpeg:
            print("WARNING: FFmpeg not found. Audio conversion may not work.")
            print("Please install FFmpeg and add it to your PATH.")
            
        # Check if GPU is available and configured
        use_gpu = setup_gpu()
        print(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled (using CPU)'}")
        
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Audio Transcription and Analysis Tool")
        parser.add_argument("file_path", nargs="?", help="Path to the audio or transcript file")
        parser.add_argument("--transcribe-only", action="store_true", help="Only transcribe, don't analyze")
        parser.add_argument("--analyze-transcript", action="store_true", help="Analyze an existing transcript file")
        parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], default="base", 
                          help="Whisper model size (default: base)")
        parser.add_argument("--language", help="Force a specific language (e.g., 'en' for English)")
        parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")

        parser.add_argument("--segment", type=int, default=0, 
                  help="Segment audio into chunks of specified seconds (e.g., 600 for 10 minutes, 0 for no segmentation")
        parser.add_argument("--clean-repetitions", action="store_true", help="Clean up repetitions in the transcript")


        
        
        args = parser.parse_args()
        
        # Override GPU setting if requested
        if args.no_gpu:
            use_gpu = False
            print("GPU acceleration manually disabled")
        
        # Determine the mode of operation
        if args.transcribe_only and args.analyze_transcript:
            print("ERROR: Cannot specify both --transcribe-only and --analyze-transcript")
            exit(1)
        
        # Get the file path (from arguments or prompt)
        file_path = args.file_path
        if not file_path:
            if args.analyze_transcript:
                file_path = input("Enter the path to your transcript file: ")
            else:
                file_path = input("Enter the path to your audio file: ")
        
        if not file_path or not os.path.exists(file_path):
            print(f"ERROR: File not found at {file_path}")
            exit(1)
        
        print(f"Using file: {file_path}")
        
        # Process based on the selected mode
        if args.analyze_transcript:
            # Analyze existing transcript
            print(f"Analyzing transcript: {file_path}")
            output_paths = analyze_transcript(file_path, use_gpu=(not args.no_gpu))
        elif args.transcribe_only:
            # Transcribe only
            print(f"Transcribing only: {file_path}")
            output_paths = transcribe_only(file_path, args.model, use_gpu=(not args.no_gpu), language=args.language)
        else:
            # Full process
            print(f"Running full transcription and analysis: {file_path}")
            output_paths = process_audio_to_notes(file_path, args.model, use_gpu=(not args.no_gpu), language=args.language, segment_length=args.segment, clean_repetitions=args.clean_repetitions)
        
        # Try to open the results in browser if available
        if not args.transcribe_only and output_paths.get("notes_html"):
            try:
                import webbrowser
                html_path = output_paths["notes_html"]
                if os.path.exists(html_path):
                    print("\nOpening conference notes in your web browser...")
                    webbrowser.open(f"file://{os.path.abspath(html_path)}")
                else:
                    print("\nHTML notes file was not generated successfully.")
            except Exception as e:
                print(f"\nCould not open notes in browser: {e}")
                print(f"Please open {output_paths['notes_html']} in your browser manually.")
        
        # Final message
        print("\nThank you for using Xinru's Transcription and Analysis Tool!")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred:")
        print(f"{e}")
        print("\nDetailed error information:")
        traceback.print_exc()
    
    input("\nPress Enter to exit...")


# ==========================================
# DIRECT TRANSCRIPTION FUNCTION
# ==========================================

def transcribe_only(audio_path, model_size="base", use_gpu=True, language=None):
    """
    Transcribe audio file only, without further analysis.
    
    Args:
        audio_path (str): Path to the audio file
        model_size (str): Whisper model size
        use_gpu (bool): Whether to use GPU for transcription
        language (str): Optional language code to force language detection
        
    Returns:
        dict: Paths to generated transcripts
    """
    print(f"\n{'='*50}")
    print(f"AUDIO TRANSCRIPTION ONLY")
    print(f"{'='*50}\n")
    
    outputs = {
        "transcript_txt": None,
        "transcript_json": None
    }
    
    try:
        # Step 1: Load and process audio
        processed_audio = load_audio(audio_path)
        
        # Step 2: Transcribe audio
        transcription_result = transcribe_audio(processed_audio, model_size, use_gpu, language)
        
        # Save paths to outputs
        base_name = os.path.basename(audio_path).split('.')[0]
        outputs["transcript_txt"] = os.path.join(OUTPUT_DIR, f"{base_name}_transcription.txt")
        outputs["transcript_json"] = os.path.join(OUTPUT_DIR, f"{base_name}_transcription.json")
        
        print(f"\n{'='*50}")
        print(f"TRANSCRIPTION COMPLETE")
        print(f"{'='*50}\n")
        
        print("Files generated:")
        print(f"1. Transcript Text: {outputs['transcript_txt']}")
        print(f"2. Transcript JSON with Timing: {outputs['transcript_json']}")
        
        return outputs
    
    except Exception as e:
        print(f"\nERROR: An error occurred during transcription: {e}")
        traceback.print_exc()
        return outputs
