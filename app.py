import streamlit as st
import os
from pathlib import Path
import time
from video_processor import DeepgramTranscriber
from embedding_manager import EmbeddingManager
from search_engine import SemanticSearchEngine
from batch_processor import process_all_transcripts
import dotenv
dotenv.load_dotenv()

def save_uploaded_file(uploaded_file, data_folder="Data"):
    """Save the uploaded file to the Data folder"""
    try:
        data_path = Path(data_folder)
        data_path.mkdir(exist_ok=True)
        
        file_path = data_path / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def process_video_with_progress(video_path, progress_bar, status_text, video_index=1, total_videos=1):
    """Process a single video with progress updates"""
    try:
        # Initialize the transcriber
        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        transcriber = DeepgramTranscriber(deepgram_api_key=deepgram_api_key)
        
        # Create necessary folders
        audio_folder = Path("audio")
        output_folder = Path("transcripts")
        audio_folder.mkdir(exist_ok=True)
        output_folder.mkdir(exist_ok=True)
        
        video_file = Path(video_path)
        audio_file = audio_folder / f"{video_file.stem}.mp3"
        transcript_file = output_folder / f"{video_file.stem}_transcript.json"
        
        # Calculate base progress for this video
        base_progress = ((video_index - 1) / total_videos) * 100
        step_size = 100 / total_videos / 4  # 4 steps per video
        
        # Step 1: Extract audio
        status_text.text(f"Processing video {video_index} of {total_videos}: Extracting audio from {video_file.name}...")
        progress_bar.progress(int(base_progress + step_size))
        
        try:
            transcriber.extract_audio(str(video_file), str(audio_file))
            progress_bar.progress(int(base_progress + step_size * 2))
            status_text.text(f"Processing video {video_index} of {total_videos}: Audio extraction completed")
            time.sleep(0.3)  # Brief pause for user to see the status
        except Exception as e:
            st.error(f"Error extracting audio from {video_file.name}: {e}")
            return False
        
        # Step 2: Transcribe audio
        status_text.text(f"Processing video {video_index} of {total_videos}: Transcribing audio...")
        progress_bar.progress(int(base_progress + step_size * 3))
        
        try:
            result = transcriber.transcribe_audio(
                str(audio_file), 
                save_json=False,  # We'll handle saving ourselves
                max_retries=3
            )
            
            if result:
                # Save transcript
                success = transcriber._write_file_with_retry(
                    transcript_file, 
                    result, 
                    is_json=True
                )
                
                if success:
                    progress_bar.progress(int(base_progress + step_size * 4))
                    status_text.text(f"Processing video {video_index} of {total_videos}: Transcription completed!")
                    return True
                else:
                    st.error(f"Failed to save transcript for {video_file.name}")
                    return False
            else:
                st.error(f"Transcription failed for {video_file.name}")
                return False
                
        except Exception as e:
            st.error(f"Error during transcription of {video_file.name}: {e}")
            return False
            
    except Exception as e:
        st.error(f"Error during processing of {video_file.name}: {e}")
        return False

def process_multiple_videos_with_progress(video_paths, progress_bar, status_text):
    """Process multiple videos with unified progress tracking"""
    total_videos = len(video_paths)
    successful_videos = []
    failed_videos = []
    
    for i, video_path in enumerate(video_paths, 1):
        success = process_video_with_progress(
            video_path, 
            progress_bar, 
            status_text, 
            video_index=i, 
            total_videos=total_videos
        )
        
        if success:
            successful_videos.append(video_path)
        else:
            failed_videos.append(video_path)
        
        # Small delay between videos
        if i < total_videos:
            time.sleep(0.2)
    
    # Final status update
    if len(successful_videos) == total_videos:
        status_text.text(f"All {total_videos} videos processed successfully! 🎉")
        progress_bar.progress(100)
    else:
        status_text.text(f"Completed: {len(successful_videos)}/{total_videos} videos processed successfully")
        progress_bar.progress(100)
    
    return successful_videos, failed_videos

@st.cache_resource
def load_search_engine():
    """Load the search engine (cached for performance)"""
    try:
        embedding_manager = EmbeddingManager()
        index = embedding_manager.load_existing_index()
        if index:
            return index, embedding_manager
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading search engine: {e}")
        return None, None

def rebuild_index_ui():
    """UI for rebuilding the search index"""
    st.subheader("🔧 Index Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Check Index Status"):
            try:
                embedding_manager = EmbeddingManager()
                stats = embedding_manager.get_collection_stats()
                if stats:
                    st.success(f"Index contains {stats['document_count']} documents")
                else:
                    st.warning("No index found")
            except Exception as e:
                st.error(f"Error checking index: {e}")
    
    with col2:
        if st.button("🔄 Rebuild Search Index"):
            with st.spinner("Rebuilding search index..."):
                try:
                    search_engine, embedding_manager = process_all_transcripts(rebuild_index=True)
                    if search_engine:
                        st.success("Search index rebuilt successfully!")
                        st.cache_resource.clear()  # Clear cache to reload
                    else:
                        st.error("Failed to rebuild search index")
                except Exception as e:
                    st.error(f"Error rebuilding index: {e}")

def search_interface():
    """Semantic search interface"""
    st.subheader("🔍 Semantic Search")
    
    # Load search engine
    index, embedding_manager = load_search_engine()
    
    if not index:
        st.warning("No search index found. Please process some videos first or rebuild the index.")
        rebuild_index_ui()
        return
    
    # Search input
    query = st.text_input(
        "Search transcripts:", 
        placeholder="What are you looking for? (e.g., 'machine learning', 'data analysis')",
        help="Enter your search query to find relevant segments across all transcribed videos"
    )
    
    # Search options
    col1, col2 = st.columns(2)
    with col1:
        num_results = st.slider("Number of results", 1, 10, 5)
    with col2:
        use_reranker = st.checkbox("Use AI Reranker", value=False, help="Use BGE reranker for better accuracy (slower)")
    
    # Perform search
    if query:
        # Create search engine with reranker option
        search_engine = SemanticSearchEngine(index, use_reranker=use_reranker)
        
        search_text = "Searching with AI reranker..." if use_reranker else "Searching..."
        with st.spinner(search_text):
            try:
                results = search_engine.search(query, video_filter=None, top_k=num_results)
                
                if results:
                    st.success(f"Found {len(results)} matches:")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(
                            f"📺 {result['video_file']} - {result['timestamp']} (Score: {result['similarity_score']:.3f})"
                        ):
                            # Create two columns: video preview on left, content on right
                            col_video, col_content = st.columns([1, 2])
                            
                            with col_video:
                                # Video preview
                                video_path = f"Data/{result['video_file']}"
                                
                                # Check if video file exists
                                if os.path.exists(video_path):
                                    st.video(video_path, start_time=int(result['start_time']))
                                else:
                                    # Try common video extensions
                                    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
                                    video_found = False
                                    
                                    for ext in video_extensions:
                                        test_path = f"Data/{result['video_file']}{ext}"
                                        if os.path.exists(test_path):
                                            st.video(test_path, start_time=int(result['start_time']))
                                            video_found = True
                                            break
                                    
                                    if not video_found:
                                        st.warning("🎥 Video file not found")
                                        st.caption(f"Expected: {video_path}")
                            
                            with col_content:
                                # Search result content
                                st.write("**Match:**")
                                st.write(result['text'])
                                
                                # Additional metadata
                                st.write("**Details:**")
                                meta_col1, meta_col2 = st.columns(2)
                                with meta_col1:
                                    st.caption(f"⏰ Time: {result['timestamp']}")
                                    st.caption(f"🎥 Video: {result['video_file']}")
                                with meta_col2:
                                    st.caption(f"📊 Similarity: {result['similarity_score']:.3f}")
                                    if 'rerank_score' in result:
                                        st.caption(f"🎯 Rerank: {result['rerank_score']:.3f}")
                                    st.caption(f"🔗 Start: {result['start_time']:.1f}s")
                else:
                    st.info("No matches found. Try different keywords or check if transcripts are available.")
                    
            except Exception as e:
                st.error(f"Search error: {e}")

def sidebar_index_management():
    """Sidebar section for index management (rebuild, status)"""
    st.sidebar.subheader("🔧 Index Management")
    if st.sidebar.button("🔄 Rebuild Search Index"):
        with st.spinner("Rebuilding search index..."):
            try:
                search_engine, embedding_manager = process_all_transcripts(rebuild_index=True)
                if search_engine:
                    st.sidebar.success("Search index rebuilt successfully!")
                    st.cache_resource.clear()  # Clear cache to reload
                else:
                    st.sidebar.error("Failed to rebuild search index")
            except Exception as e:
                st.sidebar.error(f"Error rebuilding index: {e}")
    if st.sidebar.button("📊 Check Index Status"):
        try:
            embedding_manager = EmbeddingManager()
            stats = embedding_manager.get_collection_stats()
            if stats:
                st.sidebar.success(f"Index contains {stats['document_count']} documents")
            else:
                st.sidebar.warning("No index found")
        except Exception as e:
            st.sidebar.error(f"Error checking index: {e}")


def main():
    st.set_page_config(
        page_title="Video Transcription & Search App",
        page_icon="🎥",
        layout="wide"
    )
    # Add sidebar index management
    sidebar_index_management()
    st.title("🎥 Video Transcription & Search App")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📤 Upload & Process", "🔍 Search Transcripts"])
    
    with tab1:
        st.write("Upload one or multiple video files to extract audio and generate transcriptions")
        
        # File upload (now supports multiple files)
        uploaded_files = st.file_uploader(
            "Choose video files",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'],
            help="Supported formats: MP4, AVI, MOV, MKV, WMV, FLV",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Process button
            button_text = "Process Video" if len(uploaded_files) == 1 else f"Process {len(uploaded_files)} Videos"
            if st.button(button_text, type="primary"):
                # Save all uploaded files
                with st.spinner("Saving video files..."):
                    video_paths = []
                    for uploaded_file in uploaded_files:
                        video_path = save_uploaded_file(uploaded_file)
                        if video_path:
                            video_paths.append(video_path)
                
                if video_paths:
                    # Create progress elements
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process the videos
                    if len(video_paths) == 1:
                        # Single video processing
                        success = process_video_with_progress(video_paths[0], progress_bar, status_text)
                        successful_videos = [video_paths[0]] if success else []
                        failed_videos = [] if success else [video_paths[0]]
                    else:
                        # Multiple video processing
                        successful_videos, failed_videos = process_multiple_videos_with_progress(
                            video_paths, progress_bar, status_text
                        )
                    
                    # Show results
                    if successful_videos:
                        if len(successful_videos) == len(video_paths):
                            st.success("🎉 All videos processed successfully!")
                            st.balloons()
                        else:
                            st.success(f"🎉 {len(successful_videos)} out of {len(video_paths)} videos processed successfully!")
                        
                        # Auto-rebuild search index after processing
                        with st.spinner("Updating search index..."):
                            try:
                                process_all_transcripts()
                                st.cache_resource.clear()  # Clear cache to reload search engine
                                st.info("Search index updated! You can now search the new transcripts.")
                            except Exception as e:
                                st.warning(f"Video processing completed, but search index update failed: {e}")
                    
                    if failed_videos:
                        st.error(f"❌ {len(failed_videos)} videos failed to process")
                else:
                    st.error("❌ Failed to save video files. Please try again.")
        
        # Info section
        st.divider()
        st.subheader("📋 How it works:")
        st.write("""
        1. **Upload**: Select and upload one or multiple video files
        2. **Save**: Videos are saved to the `Data` folder
        3. **Extract**: Audio is extracted from each video (MP3 format)
        4. **Transcribe**: Audio files are transcribed using Deepgram API
        5. **Index**: Transcripts are automatically indexed for semantic search
        """)
        
        # Requirements section
        st.subheader("⚙️ Requirements:")
        st.write("""
        - Set `DEEPGRAM_API_KEY` environment variable
        - Supported video formats: MP4, AVI, MOV, MKV, WMV, FLV
        - Internet connection for transcription service
        """)
    
    with tab2:
        search_interface()

if __name__ == "__main__":
    main()
