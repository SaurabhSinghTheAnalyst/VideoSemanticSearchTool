import os
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from pathlib import Path
import dotenv
dotenv.load_dotenv()
import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler, LlamaDebugHandler

class EmbeddingManager:
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
            verbose=True
        )
        self.debug_handler = LlamaDebugHandler(print_trace_on_end=True)
        self.callback_manager = CallbackManager([self.token_counter, self.debug_handler])
    
    def create_index(self, documents):
        """Create vector index from documents and log token usage and timing"""
        import time
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
        import time
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