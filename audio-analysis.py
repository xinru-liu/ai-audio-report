# Audio Transcription and Analysis for Conference Notes
# Windows Local Setup with NVIDIA GPU Support
# Optimized for NVIDIA 3070

# ==========================================
# SETUP INSTRUCTIONS (Run these in Command Prompt before running this script)
# ==========================================

# 1. Create a Python virtual environment (recommended)
#    > python -m venv audio_env
#    > audio_env\Scripts\activate
#
# 2. Install required packages:
#    > pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#    > pip install openai-whisper transformers nltk spacy sentence-transformers keybert matplotlib pandas seaborn wordcloud tqdm
#    > pip install streamlit  # For the optional GUI
#    > pip install bert-extractive-summarizer
#    > python -m spacy download en_core_web_md
#    > python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
#
# 3. Install FFmpeg for Windows:
#    - Download from https://www.gyan.dev/ffmpeg/builds/ (ffmpeg-release-full version)
#    - Extract and add the bin folder to your PATH environment variable
#
# 4. Create directories for audio files and output:
#    > mkdir audio_files
#    > mkdir output
#
# 5. Run this script:
#    > python audio_analysis.py
#    Or for the GUI version:
#    > streamlit run audio_analysis_gui.py

# ==========================================
# LIBRARY IMPORTS
# ==========================================
import sys
import traceback
import os
import re
import time
import json
print("1. Basic imports loaded")
print("2. About to import torch and GPU-related libraries")
import torch
import whisper
print("4. whisper imported successfully")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import subprocess
import threading



# NLP Libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
print("5. transformers imported successfully")
from summarizer import Summarizer
import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from wordcloud import WordCloud
print("6. All imports completed")

print("Script starting...")
print("Python version:", sys.version)
print("Current directory:", os.getcwd())

# ==========================================
# CONFIGURATION
# ==========================================

# Define directory paths - these are relative to the script location
# Replace these with absolute paths if preferred
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio_files")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Create directories if they don't exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GPU Configuration for NVIDIA 3070
def setup_gpu():
    """Configure GPU settings for optimal performance on NVIDIA 3070."""
    print("Checking GPU availability...")
    
    try:
        if torch.cuda.is_available():
            # Get GPU information
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
            
            print(f"GPU detected: {gpu_name}")
            print(f"GPU memory: {gpu_memory:.2f} GB")
            
            # Setting memory settings to avoid out of memory errors
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
            
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
    """Check if FFmpeg is installed and available in PATH."""
    print("Checking for FFmpeg installation...")
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW  # Windows-specific flag
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

def transcribe_audio(file_path, model_size="base", use_gpu=True):
    """
    Transcribe audio file using OpenAI's Whisper model with GPU acceleration.
    
    Args:
        file_path (str): Path to the audio file
        model_size (str): Whisper model size (tiny, base, small, medium, large)
        use_gpu (bool): Whether to use GPU for transcription
        
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
    
    # Transcribe the audio
    # More options: https://github.com/openai/whisper/blob/main/whisper/transcribe.py
    result = model.transcribe(
        file_path, 
        verbose=False,
        fp16=device == "cuda",  # Use half-precision for GPU to save memory
        language="en",  # Force English language
        initial_prompt="This is a professional discussion in English.",
        condition_on_previous_text=True,
        temperature=0.0  # Use greedy decoding for more predictable output
    )
    
    elapsed_time = time.time() - start_time
    print(f"Transcription completed in {elapsed_time:.2f} seconds")
    
    # Save transcription to file
    base_name = os.path.basename(file_path).split('.')[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_transcription.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result["text"])
    
    print(f"Transcription saved to: {output_path}")
    
    return result

# ==========================================
# TEXT ANALYSIS FUNCTIONS
# ==========================================

def segment_speakers(transcript, min_segment_length=100, max_segment_length=1000):
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

def create_word_cloud(transcript):
    """
    Create a word cloud visualization of the transcript.
    
    Args:
        transcript (str): Transcript text
        
    Returns:
        WordCloud: Generated word cloud object
    """
    print("Generating word cloud visualization...")
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        stopwords=stop_words,
        max_words=100,
        collocations=False
    ).generate(transcript)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    # Save the word cloud image
    plt.savefig(os.path.join(OUTPUT_DIR, 'wordcloud.png'), dpi=300)
    
    # Don't show the plot here - we'll save it and display it in the final results
    plt.close()
    
    return wordcloud

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
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_md")
    
    # Process the transcript
    # For large transcripts, process in chunks to avoid memory issues
    max_chars = 100000  # Maximum characters to process at once
    
    if len(transcript) > max_chars:
        # Process in chunks
        chunks = [transcript[i:i+max_chars] for i in range(0, len(transcript), max_chars)]
        docs = list(nlp.pipe(chunks))
        # Combined sentences from all chunks
        sentences = []
        for doc in docs:
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
    
    # Use sentence embeddings with GPU if available
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
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
    
    print(f"Extracted {len(filtered_actions)} potential action items")
    return filtered_actions

def identify_entities(transcript):
    """
    Identify key named entities in the transcript using spaCy.
    
    Args:
        transcript (str): Transcript text
        
    Returns:
        dict: Dictionary of entity types and lists of entities
    """
    print("Identifying key entities in transcript...")
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_md")
    
    # For large transcripts, process in chunks
    max_chars = 100000  # Maximum characters to process at once
    entities = {
        "PERSON": [],
        "ORG": [],
        "PRODUCT": [],
        "GPE": [],  # Geopolitical entities (countries, cities)
        "EVENT": [],
        "OTHER": []
    }
    
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

def generate_key_takeaways(transcript, summaries, topics, num_takeaways=5, use_gpu=True):
    """
    Generate key takeaways from the transcript using Flan-T5 with GPU acceleration.
    
    Args:
        transcript (str): Full transcript text
        summaries (list): Segment summaries
        topics (list): Extracted topics
        num_takeaways (int): Number of takeaways to generate
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        list: Key takeaways
    """
    print("Generating key takeaways...")
    
    # Set device
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    # Load T5 model for takeaway generation
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    # Combine summaries into a condensed representation
    combined_summary = " ".join(summaries)
    
    # Prepare topic keywords for the prompt
    topic_keywords = ", ".join([topic[0] for topic in topics])
    
    # Create a prompt for generating takeaways
    prompt = f"Based on this conference transcript summary about {topic_keywords}, what are the {num_takeaways} most important takeaways? Summary: {combined_summary[:3000]}"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    # Generate takeaways
    outputs = model.generate(
        inputs.input_ids,
        max_length=200,
        min_length=50,
        num_return_sequences=1,
        temperature=0.7,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Split into individual takeaways
    takeaways = [t.strip() for t in re.split(r'\d+\.|\n-|\n•', decoded_output) if t.strip()]
    
    # Ensure we have the desired number of takeaways
    if len(takeaways) < num_takeaways:
        # If model didn't generate enough, use additional summarization
        bert_model = Summarizer()
        additional_summary = bert_model(transcript, num_sentences=num_takeaways - len(takeaways))
        additional_points = sent_tokenize(additional_summary)
        takeaways.extend(additional_points)
    
    # Limit to requested number and format nicely
    takeaways = takeaways[:num_takeaways]
    formatted_takeaways = [f"{i+1}. {takeaway}" for i, takeaway in enumerate(takeaways)]
    
    print(f"Generated {len(formatted_takeaways)} key takeaways")
    return formatted_takeaways

# ==========================================
# OUTPUT GENERATION FUNCTIONS
# ==========================================

def visualize_topics(topics):
    """
    Create a visual representation of key topics.
    
    Args:
        topics (list): List of (topic, score) tuples
        
    Returns:
        str: Path to the saved visualization
    """
    print("Creating topic visualization...")
    
    # Extract topics and scores
    topic_names = [t[0] for t in topics]
    topic_scores = [t[1] for t in topics]
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(topic_names, topic_scores, color=sns.color_palette("viridis", len(topics)))
    
    # Add labels and title
    plt.xlabel('Relevance Score')
    plt.title('Key Topics Discussed')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'topic_distribution.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def generate_conference_notes(transcript, segments, summaries, topics, action_items, entities, takeaways):
    """
    Generate comprehensive conference notes with all analysis components.
    
    Args:
        transcript (str): Full transcript text
        segments (list): Segmented transcript
        summaries (list): Segment summaries
        topics (list): Extracted topics
        action_items (list): Extracted action items
        entities (dict): Identified entities
        takeaways (list): Key takeaways
        
    Returns:
        str: Path to the generated notes file
    """
    print("Generating comprehensive conference notes...")
    
    # Create a markdown document
    notes = []
    
    # Add header
    notes.append("# Conference Notes")
    notes.append(f"*Generated on {time.strftime('%Y-%m-%d at %H:%M')}*")
    notes.append("")
    
    # Add key takeaways section
    notes.append("## Key Takeaways")
    for takeaway in takeaways:
        notes.append(takeaway)
    notes.append("")
    
    # Add key topics section
    notes.append("## Key Topics Discussed")
    for topic, score in topics:
        notes.append(f"- **{topic}** (relevance score: {score:.2f})")
    notes.append("")
    
    # Add action items section
    notes.append("## Action Items")
    for i, action in enumerate(action_items, 1):
        notes.append(f"{i}. {action}")
    notes.append("")
    
    # Add key participants/entities section
    notes.append("## Key Participants and Organizations")
    
    if entities["PERSON"]:
        notes.append("### People")
        for person, count in entities["PERSON"][:10]:  # Top 10 people
            notes.append(f"- {person} (mentioned {count} times)")
    
    if entities["ORG"]:
        notes.append("\n### Organizations")
        for org, count in entities["ORG"][:10]:  # Top 10 organizations
            notes.append(f"- {org} (mentioned {count} times)")
    
    notes.append("")
    
    # Add summary section
    notes.append("## Meeting Summary")
    
    # Combine segments and summaries
    for i, (segment, summary) in enumerate(zip(segments, summaries), 1):
        notes.append(f"### Segment {i}")
        notes.append(f"**Original:** {segment[:150]}...")
        notes.append(f"**Summary:** {summary}")
        notes.append("")
    
    # Add full transcript section
    notes.append("## Full Transcript")
    notes.append("<details>")
    notes.append("<summary>Click to expand full transcript</summary>")
    notes.append("")
    notes.append("```")
    notes.append(transcript)
    notes.append("```")
    notes.append("</details>")
    
    # Join all sections
    notes_text = "\n".join(notes)
    
    # Save to file
    output_path = os.path.join(OUTPUT_DIR, "conference_notes.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(notes_text)
    
    # Also save an HTML version for easy viewing
    html_output_path = os.path.join(OUTPUT_DIR, "conference_notes.html")
    try:
        import markdown
        with open(html_output_path, 'w', encoding='utf-8') as f:
            f.write("<html><head><title>Conference Notes</title>")
            f.write("<style>body{font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto;max-width:800px;margin:0 auto;padding:20px;} h1{color:#333} h2{color:#555} pre{background:#f5f5f5;padding:10px;overflow:auto;}</style>")
            f.write("</head><body>")
            f.write(markdown.markdown(notes_text))
            f.write("</body></html>")
        print(f"HTML notes saved to: {html_output_path}")
    except ImportError:
        print("Could not generate HTML output (markdown package not installed)")
    
    print(f"Conference notes saved to: {output_path}")
    
    return output_path

# ==========================================
# MAIN PROCESSING FUNCTION
# ==========================================

def process_audio_to_notes(audio_path, model_size="base", use_gpu=True):
    """
    End-to-end process from audio file to conference notes.
    
    Args:
        audio_path (str): Path to the audio file
        model_size (str): Whisper model size
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        str: Path to the generated notes
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING AUDIO TO CONFERENCE NOTES")
    print(f"{'='*50}\n")
    
    # Step 1: Load and process audio
    processed_audio = load_audio(audio_path)
    
    # Step 2: Transcribe audio
    transcription_result = transcribe_audio(processed_audio, model_size, use_gpu)
    transcript = transcription_result["text"]
    
    # Step 3: Segment transcript
    segments = segment_speakers(transcript)
    
    # Step 4: Extract topics
    topics = extract_topics(transcript, use_gpu=use_gpu)
    
    # Step 5: Create word cloud
    word_cloud = create_word_cloud(transcript)
    
    # Step 6: Summarize transcript
    summaries = summarize_transcript(segments, use_gpu=use_gpu)
    
    # Step 7: Extract action items
    action_items = extract_action_items(transcript, use_gpu=use_gpu)
    
    # Step 8: Identify entities
    entities = identify_entities(transcript)
    
    # Step 9: Generate key takeaways
    takeaways = generate_key_takeaways(transcript, summaries, topics, use_gpu=use_gpu)
    
    # Step 10: Visualize topics
    topic_viz_path = visualize_topics(topics)
    
    # Step 11: Generate conference notes
    notes_path = generate_conference_notes(
        transcript,
        segments,
        summaries,
        topics,
        action_items,
        entities,
        takeaways
    )
    
    # Print completion message
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}\n")
    
    # Summarize the files that were generated
    print("Files generated:")
    print(f"1. Conference Notes (MD): {notes_path}")
    print(f"2. Conference Notes (HTML): {os.path.join(OUTPUT_DIR, 'conference_notes.html')}")
    print(f"3. Word Cloud: {os.path.join(OUTPUT_DIR, 'wordcloud.png')}")
    print(f"4. Topic Distribution: {topic_viz_path}")
    print(f"5. Transcript: {os.path.join(OUTPUT_DIR, 'transcription.txt')}")
    
    # Try to open the HTML file automatically
    try:
        import webbrowser
        print("\nOpening conference notes in your web browser...")
        webbrowser.open(f"file://{os.path.abspath(os.path.join(OUTPUT_DIR, 'conference_notes.html'))}")
    except Exception as e:
        print(f"Couldn't open notes automatically: {e}")
        print(f"Please open {os.path.join(OUTPUT_DIR, 'conference_notes.html')} in your browser manually.")
    
    return notes_path

# 3. Last: Main execution block
# ==========================================
# MAIN EXECUTION
# ==========================================

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    try:
        print(f"\n{'-'*60}")
        print(" Audio Transcription and Analysis for Conference Notes")
        print(f" Windows Local Setup with NVIDIA GPU Support")
        print(f"{'-'*60}\n")
        
        # Import needed for command line arguments
        import sys
        
        # Check system requirements
        have_ffmpeg = check_ffmpeg()
        if not have_ffmpeg:
            print("WARNING: FFmpeg not found. Audio conversion may not work.")
            print("Please install FFmpeg and add it to your PATH.")
            
        # Check if GPU is available and configured
        use_gpu = setup_gpu()
        print(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled (using CPU)'}")
        
        # Step 1: Get the audio file path
        if len(sys.argv) > 1:
            # If path is provided as command line argument, use it
            audio_path = sys.argv[1]
            print(f"\nUsing audio file from command line: {audio_path}")
        else:
            # Otherwise prompt for path
            audio_path = input("\nEnter the path to your audio file: ")
        
        if not audio_path or not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            print("Please provide a valid file path.")
            exit(1)
        
        # Step 2: Let the user select a model size
        print("\nStep 2: Select a Whisper model size")
        print("NOTE: Larger models are more accurate but require more memory and processing time.")
        print("  1. tiny   - Fastest, least accurate (1GB VRAM needed)")
        print("  2. base   - Good balance for general use (1GB VRAM needed)")
        print("  3. small  - Better accuracy, still reasonable speed (2GB VRAM needed)")
        print("  4. medium - High accuracy, slower (5GB VRAM needed)")
        print("  5. large  - Highest accuracy, slowest (10GB VRAM needed)")
        
        model_choice = input("\nEnter your choice (1-5) [default: 2]: ") or "2"
        
        model_sizes = {
            "1": "tiny",
            "2": "base",
            "3": "small",
            "4": "medium",
            "5": "large"
        }
        
        model_size = model_sizes.get(model_choice, "base")
        print(f"Using {model_size} model for transcription")
        
        # Step 3: Process the audio file
        print("\nStep 3: Processing audio and generating conference notes")
        print("This may take several minutes depending on the file length and model size.")
        print("Please wait...\n")
        
        # Call the processing function
        result_path = process_audio_to_notes(audio_path, model_size, use_gpu)
        
        # Step 4: Display results and paths
        if result_path and os.path.exists(result_path):
            print("\nStep 4: Results")
            
            print("\nFiles generated:")
            print(f"1. Conference Notes (MD): {result_path}")
            print(f"2. Conference Notes (HTML): {os.path.join(OUTPUT_DIR, 'conference_notes.html')}")
            print(f"3. Word Cloud: {os.path.join(OUTPUT_DIR, 'wordcloud.png')}")
            print(f"4. Topic Distribution: {os.path.join(OUTPUT_DIR, 'topic_distribution.png')}")
            
            print("\nAll files have been saved to:")
            print(f"  {os.path.abspath(OUTPUT_DIR)}")
            
            # Try to open the HTML file in browser
            try:
                import webbrowser
                html_path = os.path.join(OUTPUT_DIR, 'conference_notes.html')
                if os.path.exists(html_path):
                    print("\nOpening HTML results in browser...")
                    webbrowser.open(f"file://{os.path.abspath(html_path)}")
            except Exception as e:
                print(f"Note: Could not open results in browser automatically: {e}")
            
            print("\nThank you for using the Audio Transcription and Analysis tool!")
        else:
            print("\nERROR: Processing failed or did not complete properly.")
            print("Please check the above error messages for details.")
    
    except Exception as e:
        import traceback
        print(f"\nERROR: An unexpected error occurred during processing:")
        print(f"{e}")
        print("\nDetailed error information:")
        traceback.print_exc()
        print("\nPlease check your installation and try again.")
    
    input("\nPress Enter to exit...")