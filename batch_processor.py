from transcript_processor import process_transcript_to_documents, get_all_transcript_files
from embedding_manager import EmbeddingManager
from search_engine import SemanticSearchEngine
from pathlib import Path

def process_all_transcripts(transcripts_folder="transcripts", rebuild_index=False):
    """Process all transcript files and build unified search index"""
    
    print("🚀 Starting transcript processing...")
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Check if we should rebuild or load existing index
    if rebuild_index:
        print("🔄 Rebuilding index from scratch...")
        embedding_manager.clear_collection()
    
    # Get existing stats
    stats = embedding_manager.get_collection_stats()
    if stats:
        print(f"📊 Current collection stats: {stats['document_count']} documents")
    
    # Get all transcript files
    transcript_files = get_all_transcript_files(transcripts_folder)
    
    if not transcript_files:
        print(f"❌ No transcript files found in '{transcripts_folder}' folder")
        return None, None
    
    # Process all transcripts
    all_documents = []
    processed_videos = []
    
    for transcript_file in transcript_files:
        print(f"📝 Processing: {transcript_file.name}")
        documents = process_transcript_to_documents(transcript_file)
        
        if documents:
            all_documents.extend(documents)
            video_name = transcript_file.stem.replace('_transcript', '')
            processed_videos.append(video_name)
        else:
            print(f"⚠️  No documents extracted from {transcript_file.name}")
    
    if not all_documents:
        print("❌ No documents were processed successfully")
        return None, None
    
    print(f"📚 Total documents processed: {len(all_documents)}")
    print(f"🎥 Videos processed: {len(processed_videos)}")
    
    # Create or load index
    try:
        if stats and stats['document_count'] > 0 and not rebuild_index:
            print("📖 Loading existing index...")
            index = embedding_manager.load_existing_index()
            
            if index:
                # Add new documents to existing index (if any)
                index = embedding_manager.add_documents_to_index(index, all_documents)
            else:
                print("⚠️  Could not load existing index, creating new one...")
                index = embedding_manager.create_index(all_documents)
        else:
            print("🏗️  Creating new index...")
            index = embedding_manager.create_index(all_documents)
        
        if index:
            print("✅ Index created/updated successfully!")
            
            # Create search engine
            search_engine = SemanticSearchEngine(index)
            
            # Final stats
            final_stats = embedding_manager.get_collection_stats()
            if final_stats:
                print(f"📊 Final collection stats: {final_stats['document_count']} documents")
            
            return search_engine, embedding_manager
        else:
            print("❌ Failed to create index")
            return None, None
            
    except Exception as e:
        print(f"❌ Error during index creation: {e}")
        return None, None

def rebuild_search_index(transcripts_folder="transcripts"):
    """Rebuild the entire search index from scratch"""
    print("🔄 Rebuilding search index...")
    return process_all_transcripts(transcripts_folder, rebuild_index=True)

def get_index_stats():
    """Get current index statistics"""
    try:
        embedding_manager = EmbeddingManager()
        stats = embedding_manager.get_collection_stats()
        
        if stats:
            print(f"📊 Index Statistics:")
            print(f"   Documents: {stats['document_count']}")
            print(f"   Collection: {stats['collection_name']}")
            print(f"   Storage: {stats['persist_dir']}")
        else:
            print("❌ No index found or error retrieving stats")
            
        return stats
    except Exception as e:
        print(f"❌ Error getting index stats: {e}")
        return None

def test_search_functionality(query="machine learning", top_k=3):
    """Test the search functionality with a sample query"""
    print(f"🔍 Testing search with query: '{query}'")
    
    try:
        # Load existing index
        embedding_manager = EmbeddingManager()
        index = embedding_manager.load_existing_index()
        
        if not index:
            print("❌ No index found. Please run process_all_transcripts() first.")
            return
        
        # Create search engine and test
        search_engine = SemanticSearchEngine(index)
        results = search_engine.search(query, top_k=top_k)
        
        if results:
            print(f"✅ Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Video: {result['video_file']}")
                print(f"   Time: {result['timestamp']}")
                print(f"   Text: {result['text'][:100]}...")
                print(f"   Score: {result['similarity_score']:.3f}")
        else:
            print("❌ No results found")
            
    except Exception as e:
        print(f"❌ Error during search test: {e}")

if __name__ == "__main__":
    # Example usage
    print("🎬 Video Transcript Search Index Builder")
    print("=" * 50)
    
    # Build/load index
    search_engine, embedding_manager = process_all_transcripts()
    
    if search_engine:
        print("\n" + "=" * 50)
        print("🎯 Testing search functionality...")
        test_search_functionality()
    else:
        print("❌ Failed to create search system") 