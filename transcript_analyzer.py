# transcript_analyzer.py
# A standalone script for analyzing transcription text and generating conference notes

import os
import re
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import sys

# NLP Libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from summarizer import Summarizer
import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from wordcloud import WordCloud

# Define directory for storing output
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")
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

# Try to load spaCy model - handle alternate versions gracefully
def load_spacy_model():
    """Try to load the spaCy model with fallbacks for different versions."""
    try:
        # First try to load the medium model
        return spacy.load("en_core_web_md")
    except (OSError, IOError):
        try:
            # Fall back to small model
            print("Medium spaCy model not found, trying small model...")
            return spacy.load("en_core_web_sm")
        except (OSError, IOError):
            # If no model is installed, try to download the small model
            print("No spaCy model found. Downloading small English model...")
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

# Setup GPU
def setup_gpu():
    """Configure GPU settings for optimal performance."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU detected: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.2f} GB")
        
        # Setting memory settings to avoid out of memory errors
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cudnn.benchmark = True
        
        return True
    else:
        print("No GPU detected, using CPU only.")
        return False

def segment_speakers(transcript, min_segment_length=100, max_segment_length=1000):
    """Segment transcript into probable speaker turns based on pauses and content."""
    print("Segmenting transcript into logical chunks...")
    
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
    
    print(f"Identified {len(final_segments)} segments")
    return final_segments

def extract_topics(transcript, num_topics=5, use_gpu=True):
    """Extract key topics from transcript using KeyBERT."""
    print("Extracting key topics from transcript...")
    
    # Set the device for sentence transformers
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    # Initialize KeyBERT
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
    """Create a word cloud visualization of the transcript."""
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
    output_path = os.path.join(OUTPUT_DIR, 'wordcloud.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Word cloud saved to: {output_path}")
    return output_path

def summarize_transcript(segments, model_name="facebook/bart-large-cnn", use_gpu=True, batch_size=4):
    """Summarize transcript segments using a pretrained model."""
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
        framework="pt"
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
                    batch_size=len(valid_batch)
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
    """Extract potential action items from the transcript."""
    print("Extracting potential action items...")
    
    try:
        # Load spaCy model
        nlp = load_spacy_model()
        
        # Process the transcript in chunks if needed
        max_chars = 100000
        sentences = []
        
        if len(transcript) > max_chars:
            # Process in chunks
            chunks = [transcript[i:i+max_chars] for i in range(0, len(transcript), max_chars)]
            for chunk in chunks:
                doc = nlp(chunk)
                sentences.extend([sent.text.strip() for sent in doc.sents])
        else:
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
        
        # Further refine to remove duplicates
        # Use sentence embeddings with GPU if available
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        sentence_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
        sentence_embeddings = sentence_embeddings.to(device)
        
        filtered_actions = []
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
    
    except Exception as e:
        print(f"Error in action item extraction: {e}")
        print("Continuing with empty action items list")
        return []

def identify_entities(transcript):
    """Identify key named entities in the transcript."""
    print("Identifying key entities in transcript...")
    
    try:
        # Load spaCy model
        nlp = load_spacy_model()
        
        # Initialize entity categories
        entities = {
            "PERSON": [],
            "ORG": [],
            "PRODUCT": [],
            "GPE": [],  # Geopolitical entities (countries, cities)
            "EVENT": [],
            "OTHER": []
        }
        
        # Process in chunks for large transcripts
        max_chars = 100000
        
        if len(transcript) > max_chars:
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
        print(f"Error in entity identification: {e}")
        print("Continuing with empty entities dictionary")
        return {"PERSON": [], "ORG": [], "PRODUCT": [], "GPE": [], "EVENT": [], "OTHER": []}

def generate_key_takeaways(transcript, summaries, topics, num_takeaways=5, use_gpu=True):
    """Generate key takeaways from the transcript."""
    print("Generating key takeaways...")
    
    try:
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
        prompt = f"Based on this transcript summary about {topic_keywords}, what are the {num_takeaways} most important takeaways? Summary: {combined_summary[:3000]}"
        
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
    
    except Exception as e:
        print(f"Error generating takeaways: {e}")
        print("Falling back to extractive summarization")
        
        # Fallback to simple extractive summarization
        try:
            from nltk.tokenize import sent_tokenize
            
            # Get all sentences
            sentences = sent_tokenize(transcript)
            
            # Calculate sentence importance (simple word frequency approach)
            word_frequencies = {}
            for sentence in sentences:
                for word in word_tokenize(sentence.lower()):
                    if word not in stopwords.words('english'):
                        word_frequencies[word] = word_frequencies.get(word, 0) + 1
            
            # Calculate sentence scores
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                score = 0
                for word in word_tokenize(sentence.lower()):
                    if word in word_frequencies:
                        score += word_frequencies[word]
                sentence_scores[i] = score / max(1, len(word_tokenize(sentence)))
            
            # Get top sentences
            top_indices = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_takeaways]
            top_indices = sorted([i for i, _ in top_indices])
            
            # Format takeaways
            formatted_takeaways = [f"{i+1}. {sentences[idx]}" for i, idx in enumerate(top_indices)]
            
            print(f"Generated {len(formatted_takeaways)} fallback takeaways")
            return formatted_takeaways
        
        except Exception as inner_e:
            print(f"Even fallback summarization failed: {inner_e}")
            return [f"{i+1}. Key point from the transcript." for i in range(num_takeaways)]

def visualize_topics(topics):
    """Create a visual representation of key topics."""
    print("Creating topic visualization...")
    
    try:
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
        
        print(f"Topic visualization saved to: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None

def generate_conference_notes(transcript, segments, summaries, topics, action_items, entities, takeaways):
    """Generate comprehensive conference notes with all analysis components."""
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
    if action_items:
        for i, action in enumerate(action_items, 1):
            notes.append(f"{i}. {action}")
    else:
        notes.append("No specific action items identified.")
    notes.append("")
    
    # Add key participants/entities section
    notes.append("## Key Participants and Organizations")
    
    if entities["PERSON"] and len(entities["PERSON"]) > 0:
        notes.append("### People")
        for person, count in entities["PERSON"][:10]:  # Top 10 people
            notes.append(f"- {person} (mentioned {count} times)")
    
    if entities["ORG"] and len(entities["ORG"]) > 0:
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
    
    # Save to markdown file
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

def analyze_transcript(transcript_path, use_gpu=True):
    """
    Analyze a transcript file and generate conference notes.
    
    Args:
        transcript_path (str): Path to the transcript text file
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        str: Path to the generated notes
    """
    print(f"\n{'='*50}")
    print(f"ANALYZING TRANSCRIPT")
    print(f"{'='*50}\n")
    
    # Read the transcript file
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
            
        print(f"Read transcript file: {transcript_path}")
        print(f"Transcript length: {len(transcript)} characters")
        
        # Check if transcript is empty or too short
        if len(transcript) < 100:
            print("Warning: Transcript seems very short. Results may be limited.")
    except Exception as e:
        print(f"Error reading transcript file: {e}")
        return None
    
    # Step 1: Segment transcript
    segments = segment_speakers(transcript)
    
    # Step 2: Extract topics
    topics = extract_topics(transcript, use_gpu=use_gpu)
    
    # Step 3: Create word cloud
    word_cloud_path = create_word_cloud(transcript)
    
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
    
    # Print completion message
    print(f"\n{'='*50}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*50}\n")
    
    # Summarize the files that were generated
    print("Files generated:")
    print(f"1. Conference Notes (MD): {notes_path}")
    print(f"2. Conference Notes (HTML): {os.path.join(OUTPUT_DIR, 'conference_notes.html')}")
    print(f"3. Word Cloud: {word_cloud_path}")
    print(f"4. Topic Distribution: {topic_viz_path}")
    
    # Try to open the HTML file automatically
    try:
        import webbrowser
        print("\nOpening conference notes in your web browser...")
        webbrowser.open(f"file://{os.path.abspath(os.path.join(OUTPUT_DIR, 'conference_notes.html'))}")
    except Exception as e:
        print(f"Couldn't open notes automatically: {e}")
        print(f"Please open {os.path.join(OUTPUT_DIR, 'conference_notes.html')} in your browser manually.")
    
    return notes_path

if __name__ == "__main__":
    try:
        print("\n" + "-"*60)
        print(" Transcript Analysis Tool")
        print("-"*60 + "\n")
        
        # Check if GPU is available and configured
        use_gpu = setup_gpu()
        print(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled (using CPU)'}")
        
        # Get transcript file path
        if len(sys.argv) > 1:
            transcript_path = sys.argv[1]
            print(f"Using transcript file from command line: {transcript_path}")
        else:
            transcript_path = input("Enter the path to your transcript text file: ")
        
        if not transcript_path or not os.path.exists(transcript_path):
            print(f"Error: Transcript file not found at {transcript_path}")
            exit(1)
        
        # Process the transcript
        print("\nAnalyzing transcript and generating conference notes...")
        print("This may take several minutes depending on the transcript length.")
        print("Please wait...\n")
        
        result_path = analyze_transcript(transcript_path, use_gpu)
        
        if result_path:
            print("\nAnalysis completed successfully!")
            print(f"All output files have been saved to: {os.path.abspath(OUTPUT_DIR)}")
        else:
            print("\nERROR: Analysis failed or did not complete properly.")
        
    except Exception as e:
        import traceback
        print(f"\nERROR: An unexpected error occurred during processing:")
        print(f"{e}")
        print("\nDetailed error information:")
        traceback.print_exc()
    
    input("\nPress Enter to exit...")