# Video Transcription & Search App

A powerful Streamlit app for uploading videos, extracting audio, generating transcriptions, and performing semantic search using Deepgram API and LlamaIndex embeddings.

## Features

### Video Processing
- 🎥 Upload single or multiple video files (MP4, AVI, MOV, MKV, WMV, FLV)
- 📁 Automatic saving to Data folder
- 🎵 Audio extraction from videos
- 📝 AI-powered transcription using Deepgram
- 📊 Real-time progress tracking across all videos
- 🔄 Batch processing with unified progress bar

### Semantic Search
- 🔍 Semantic search across all transcribed videos
- 🎯 Find relevant content by meaning, not just keywords
- ⏰ Get exact timestamps for each match
- 📺 Filter search by specific videos
- 📊 Similarity scoring for result relevance
- 🚀 Fast vector-based search using ChromaDB and HuggingFace embeddings

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Deepgram API key:**
   ```bash
   export DEEPGRAM_API_KEY="your_deepgram_api_key_here"
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to the URL shown in terminal (usually `http://localhost:8501`)

## Usage

### Upload & Process Tab
1. Upload one or multiple video files using the file uploader
2. Click "Process Video" (single file) or "Process X Videos" (multiple files) button
3. Watch the progress bar as the app:
   - Extracts audio from each video
   - Transcribes each audio file using AI
   - Shows progress: "Processing video X of Y: [current step]"
   - Automatically updates the search index

### Search Transcripts Tab
1. Enter your search query (e.g., "machine learning", "data analysis")
2. Optionally filter by specific video or adjust number of results
3. Browse results with exact timestamps and similarity scores
4. Use the "Index Management" section to check status or rebuild the search index

## Output Files

- **Videos**: Saved in `Data/` folder
- **Audio**: Extracted audio files in `audio/` folder (MP3 format)
- **Transcripts**: JSON transcription files in `transcripts/` folder

## Requirements

- Python 3.7+
- Deepgram API key (sign up at [deepgram.com](https://deepgram.com))
- Internet connection for transcription service 