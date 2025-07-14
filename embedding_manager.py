import os
import json
import time
from pathlib import Path
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import dotenv
dotenv.load_dotenv()
import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler, LlamaDebugHandler

class EmbeddingManager:
    """
    Comprehensive embedding manager that handles transcript processing, 
    vector index creation, and batch operations.
    
    This class consolidates functionality from:
    - transcript_processor.py: Processing Deepgram transcripts
    - batch_processor.py: Batch processing operations
    - Original EmbeddingManager: Vector indexing and search
    """
    
    def __init__(self, persist_dir="./chroma_db", collection_name="video_transcripts"):
        """Initialize embedding manager with ChromaDB persistence"""
        
        # Set OpenAI API key from environment variable
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key.")
        
        # Initialize embedding model with OpenAI
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large"
        )
        Settings.llm = None
        
        # Setup ChromaDB for persistence
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Create persist directory if it doesn't exist
        Path(persist_dir).mkdir(exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(collection_name)
            print(f"Created new collection: {collection_name}")
        
        # Setup vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Setup token counting and debug handlers
        self.token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model("text-embedding-3-large").encode,
            verbose=False  # Set to False to reduce noise
        )
        self.debug_handler = LlamaDebugHandler(print_trace_on_end=False)  # Set to False to reduce noise
        self.callback_manager = CallbackManager([self.token_counter, self.debug_handler])
    
    # === TRANSCRIPT PROCESSING METHODS (from transcript_processor.py) ===
    
    def process_transcript_to_documents(self, transcript_file):
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

    def get_all_transcript_files(self, transcripts_folder="transcripts"):
        """Get all transcript JSON files from the transcripts folder"""
        transcript_path = Path(transcripts_folder)
        if not transcript_path.exists():
            print(f"Transcripts folder '{transcripts_folder}' does not exist")
            return []
        
        transcript_files = list(transcript_path.glob("*_transcript.json"))
        print(f"Found {len(transcript_files)} transcript files")
        return transcript_files

    # === BATCH PROCESSING METHODS (from batch_processor.py) ===

    def process_all_transcripts(self, transcripts_folder="transcripts", rebuild_index=False):
        """
        Process all transcript files and build unified search index.
        
        Args:
            transcripts_folder (str): Path to folder containing transcript files
            rebuild_index (bool): Whether to rebuild the index from scratch
            
        Returns:
            VectorStoreIndex: The created/updated index, or None if failed
        """
        
        print("🚀 Starting transcript processing...")
        
        # Check if we should rebuild or load existing index
        if rebuild_index:
            print("🔄 Rebuilding index from scratch...")
            self.clear_collection()
        
        # Get existing stats
        stats = self.get_collection_stats()
        if stats:
            print(f"📊 Current collection stats: {stats['document_count']} documents")
        
        # Get all transcript files
        transcript_files = self.get_all_transcript_files(transcripts_folder)
        
        if not transcript_files:
            print(f"❌ No transcript files found in '{transcripts_folder}' folder")
            return None
        
        # Process all transcripts
        all_documents = []
        processed_videos = []
        
        for transcript_file in transcript_files:
            print(f"📝 Processing: {transcript_file.name}")
            documents = self.process_transcript_to_documents(transcript_file)
            
            if documents:
                all_documents.extend(documents)
                video_name = transcript_file.stem.replace('_transcript', '')
                processed_videos.append(video_name)
            else:
                print(f"⚠️  No documents extracted from {transcript_file.name}")
        
        if not all_documents:
            print("❌ No documents were processed successfully")
            return None
        
        print(f"📚 Total documents processed: {len(all_documents)}")
        print(f"🎥 Videos processed: {len(processed_videos)}")
        
        # Create or load index
        try:
            if stats and stats['document_count'] > 0 and not rebuild_index:
                print("📖 Loading existing index...")
                index = self.load_existing_index()
                
                if index:
                    # Add new documents to existing index (if any)
                    index = self.add_documents_to_index(index, all_documents)
                else:
                    print("⚠️  Could not load existing index, creating new one...")
                    index = self.create_index(all_documents)
            else:
                print("🏗️  Creating new index...")
                index = self.create_index(all_documents)
            
            if index:
                print("✅ Index created/updated successfully!")
                
                # Final stats
                final_stats = self.get_collection_stats()
                if final_stats:
                    print(f"📊 Final collection stats: {final_stats['document_count']} documents")
                
                return index
            else:
                print("❌ Failed to create index")
                return None
                
        except Exception as e:
            print(f"❌ Error during index creation: {e}")
            return None

    def rebuild_search_index(self, transcripts_folder="transcripts"):
        """Rebuild the entire search index from scratch"""
        print("🔄 Rebuilding search index...")
        return self.process_all_transcripts(transcripts_folder, rebuild_index=True)

    def get_index_stats(self):
        """Get current index statistics (convenience method for batch operations)"""
        try:
            stats = self.get_collection_stats()
            
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

    def test_search_functionality(self, query="machine learning", top_k=3):
        """Test the search functionality with a sample query"""
        print(f"🔍 Testing search with query: '{query}'")
        
        try:
            # Load existing index
            index = self.load_existing_index()
            
            if not index:
                print("❌ No index found. Please run process_all_transcripts() first.")
                return None
            
            # Create query engine and test
            query_engine = index.as_query_engine(
                similarity_top_k=top_k,
                callback_manager=self.callback_manager
            )
            
            start_time = time.time()
            response = query_engine.query(query)
            elapsed = time.time() - start_time
            
            print(f"✅ Search completed in {elapsed:.2f} seconds")
            print(f"Embedding Tokens Used: {self.token_counter.total_embedding_token_count}")
            
            # Parse results from response
            results = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    result = {
                        'video_file': node.metadata.get('video_file', 'Unknown'),
                        'timestamp': f"{node.metadata.get('start_time', 0):.1f}s",
                        'text': node.text,
                        'similarity_score': node.score if hasattr(node, 'score') else 0.0
                    }
                    results.append(result)
            
            if results:
                print(f"✅ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Video: {result['video_file']}")
                    print(f"   Time: {result['timestamp']}")
                    print(f"   Text: {result['text'][:100]}...")
                    print(f"   Score: {result['similarity_score']:.3f}")
            else:
                print("❌ No results found")
            
            return results
                
        except Exception as e:
            print(f"❌ Error during search test: {e}")
            return None

    # === CORE INDEX MANAGEMENT METHODS ===

    def create_index(self, documents):
        """Create vector index from documents and log token usage and timing"""
        if not documents:
            print("No documents provided for indexing")
            return None
            
        print(f"Creating index with {len(documents)} documents...")
        try:
            start_time = time.time()
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                show_progress=True,
                callback_manager=self.callback_manager
            )
            elapsed = time.time() - start_time
            print("Index created successfully!")
            print(f"Index creation time: {elapsed:.2f} seconds")
            print(f"Embedding Tokens Used: {self.token_counter.total_embedding_token_count}")
            return index
        except Exception as e:
            print(f"Error creating index: {e}")
            return None
    
    def load_existing_index(self):
        """Load existing index from storage"""
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )
            print("Loaded existing index successfully!")
            return index
        except Exception as e:
            print(f"Error loading existing index: {e}")
            return None
    
    def add_documents_to_index(self, index, new_documents):
        """Add new documents to existing index and log token usage and timing"""
        if not new_documents:
            print("No new documents to add")
            return index
            
        try:
            start_time = time.time()
            for doc in new_documents:
                index.insert(doc, callback_manager=self.callback_manager)
            elapsed = time.time() - start_time
            print(f"Added {len(new_documents)} new documents to index")
            print(f"Addition time: {elapsed:.2f} seconds")
            print(f"Embedding Tokens Used: {self.token_counter.total_embedding_token_count}")
            return index
        except Exception as e:
            print(f"Error adding documents to index: {e}")
            return index
    
    def get_collection_stats(self):
        """Get statistics about the current collection"""
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "persist_dir": self.persist_dir
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return None
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(self.collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            print(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False


# === CONVENIENCE FUNCTIONS (for backward compatibility) ===

def process_all_transcripts(transcripts_folder="transcripts", rebuild_index=False):
    """
    Convenience function for batch processing (maintains compatibility with batch_processor.py).
    Returns both search engine and embedding manager for backward compatibility.
    """
    from search_engine import SemanticSearchEngine
    
    print("🚀 Starting transcript processing...")
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Process transcripts
    index = embedding_manager.process_all_transcripts(transcripts_folder, rebuild_index)
    
    if index:
        # Create search engine
        search_engine = SemanticSearchEngine(index)
        return search_engine, embedding_manager
    else:
        return None, None

def rebuild_search_index(transcripts_folder="transcripts"):
    """Convenience function to rebuild the entire search index from scratch"""
    print("🔄 Rebuilding search index...")
    return process_all_transcripts(transcripts_folder, rebuild_index=True)

def get_index_stats():
    """Convenience function to get current index statistics"""
    try:
        embedding_manager = EmbeddingManager()
        return embedding_manager.get_index_stats()
    except Exception as e:
        print(f"❌ Error getting index stats: {e}")
        return None

def test_search_functionality(query="machine learning", top_k=3):
    """Convenience function to test the search functionality with a sample query"""
    print(f"🔍 Testing search with query: '{query}'")
    
    try:
        # Load existing index
        embedding_manager = EmbeddingManager()
        index = embedding_manager.load_existing_index()
        
        if not index:
            print("❌ No index found. Please run process_all_transcripts() first.")
            return
        
        # Create search engine and test
        from search_engine import SemanticSearchEngine
        search_engine = SemanticSearchEngine(index)
        results = search_engine.search(query, top_k=top_k)
        
        if results:
            print(f"✅ Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Video: {result['video_file']}")
                timestamp = result.get('timestamp', f"{result.get('start_time', 0):.1f}s")
                print(f"   Time: {timestamp}")
                print(f"   Text: {result['text'][:100]}...")
                print(f"   Score: {result['similarity_score']:.3f}")
        else:
            print("❌ No results found")
            
    except Exception as e:
        print(f"❌ Error during search test: {e}")


# Example usage when run as main
if __name__ == "__main__":
    print("🎬 Video Transcript Search Index Builder")
    print("=" * 50)
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Build/load index
    index = embedding_manager.process_all_transcripts()
    
    if index:
        print("\n" + "=" * 50)
        print("🎯 Testing search functionality...")
        embedding_manager.test_search_functionality()
    else:
        print("❌ Failed to create search system") 