import os
import dotenv
from llama_index.core import QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
import cohere

# Load environment variables
dotenv.load_dotenv()

class SemanticSearchEngine:
    def __init__(self, index, use_reranker=False):
        """Initialize search engine with vector index"""
        self.index = index
        if index is None:
            raise ValueError("Index cannot be None")
        
        # Initialize reranker if requested
        self.use_reranker = use_reranker
        self.cohere_client = None
        if use_reranker:
            try:
                print("Loading Cohere reranker...")
                cohere_api_key = os.environ.get("COHERE_API_KEY")
                if not cohere_api_key:
                    print("⚠️  COHERE_API_KEY not found in environment variables")
                    self.use_reranker = False
                else:
                    self.cohere_client = cohere.ClientV2(api_key=cohere_api_key)
                    print("✅ Cohere reranker loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load Cohere reranker: {e}")
                self.use_reranker = False
    
    def search(self, query, video_filter=None, top_k=5):
        """Search for semantically similar sentences with optional reranking"""
        try:
            # Get more initial results if using reranker
            initial_top_k = top_k * 4 if self.use_reranker else top_k * 2
            
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=initial_top_k
            )
            
            # Retrieve nodes
            query_bundle = QueryBundle(query_str=query)
            nodes = retriever.retrieve(query_bundle)
            
            # Convert nodes to results
            candidate_results = []
            for node in nodes:
                metadata = node.metadata
                
                # Apply video filter if specified
                if video_filter and metadata.get("video_file") != video_filter:
                    continue
                
                result = {
                    "text": node.text,
                    "video_file": metadata.get("video_file", "unknown"),
                    "start_time": metadata.get("start_time", 0),
                    "end_time": metadata.get("end_time", 0),
                    "timestamp": format_timestamp(metadata.get("start_time", 0), metadata.get("end_time", 0)),
                    "similarity_score": getattr(node, 'score', 0.0),
                    "sentence_id": metadata.get("sentence_id", "unknown"),
                    "paragraph_index": metadata.get("paragraph_index", 0)
                }
                candidate_results.append(result)
            
            # Apply reranking if enabled
            if self.use_reranker and self.cohere_client and len(candidate_results) > 1:
                reranked_results = self._rerank_results(query, candidate_results, top_k)
                return reranked_results
            else:
                return candidate_results[:top_k+1]
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def _rerank_results(self, query, results, top_k):
        """Rerank results using Cohere reranker API"""
        try:
            # Prepare documents for Cohere API
            documents = [result["text"] for result in results]
            
            # Call Cohere rerank API (v2 doesn't use return_documents parameter)
            response = self.cohere_client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents))  # Don't exceed available documents
            )
            
            # Map reranked results back to original format
            reranked_results = []
            for reranked_item in response.results:
                # Get the original result using the index
                original_result = results[reranked_item.index].copy()
                # Add Cohere rerank score
                original_result["rerank_score"] = float(reranked_item.relevance_score)
                reranked_results.append(original_result)
            
            return reranked_results
            
        except Exception as e:
            print(f"Error during Cohere reranking: {e}")
            # Fallback to original similarity-based ranking
            return results[:top_k]
    
    def search_by_video(self, query, video_name, top_k=5):
        """Search within a specific video"""
        return self.search(query, video_filter=video_name, top_k=top_k)
    
    def get_context_around_result(self, result, context_sentences=2):
        """Get additional context sentences around a search result"""
        try:
            # This would require implementing context retrieval
            # For now, return the result as-is
            return result
        except Exception as e:
            print(f"Error getting context: {e}")
            return result

def format_timestamp(start_time, end_time):
    """Convert seconds to MM:SS format"""
    try:
        start_minutes = int(start_time // 60)
        start_seconds = int(start_time % 60)
        end_minutes = int(end_time // 60)
        end_seconds = int(end_time % 60)
        
        return f"{start_minutes:02d}:{start_seconds:02d} - {end_minutes:02d}:{end_seconds:02d}"
    except:
        return "00:00 - 00:00"

def format_duration(start_time, end_time):
    """Get duration of a segment"""
    try:
        duration = end_time - start_time
        return f"{duration:.1f}s"
    except:
        return "0.0s" 