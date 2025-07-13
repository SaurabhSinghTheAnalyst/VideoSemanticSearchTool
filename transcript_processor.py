from llama_index.core import Document
import json
from pathlib import Path

def process_transcript_to_documents(transcript_file):
    """Convert Deepgram transcript to LlamaIndex Documents"""
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        video_name = Path(transcript_file).stem.replace('_transcript', '')
        
        # Extract sentences from paragraphs
        results = data.get("results", {})
        channels = results.get("channels", [])
        
        if not channels:
            print(f"No channels found in {transcript_file}")
            return documents
            
        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            print(f"No alternatives found in {transcript_file}")
            return documents
            
        paragraphs_data = alternatives[0].get("paragraphs", {})
        paragraphs = paragraphs_data.get("paragraphs", [])
        
        if not paragraphs:
            print(f"No paragraphs found in {transcript_file}")
            return documents
        
        # Process each paragraph and sentence
        for para_idx, paragraph in enumerate(paragraphs):
            sentences = paragraph.get("sentences", [])
            
            for sent_idx, sentence in enumerate(sentences):
                # Create Document with metadata
                doc = Document(
                    text=sentence.get("text", ""),
                    metadata={
                        "video_file": video_name,
                        "start_time": sentence.get("start", 0),
                        "end_time": sentence.get("end", 0),
                        "sentence_id": f"{video_name}_p{para_idx}_s{sent_idx}",
                        "paragraph_index": para_idx,
                        "transcript_file": str(transcript_file)
                    }
                )
                documents.append(doc)
        
        print(f"Processed {len(documents)} sentences from {video_name}")
        return documents
        
    except Exception as e:
        print(f"Error processing {transcript_file}: {e}")
        return []

def get_all_transcript_files(transcripts_folder="transcripts"):
    """Get all transcript JSON files from the transcripts folder"""
    transcript_path = Path(transcripts_folder)
    if not transcript_path.exists():
        print(f"Transcripts folder '{transcripts_folder}' does not exist")
        return []
    
    transcript_files = list(transcript_path.glob("*_transcript.json"))
    print(f"Found {len(transcript_files)} transcript files")
    return transcript_files 