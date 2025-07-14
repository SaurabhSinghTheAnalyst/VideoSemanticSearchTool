from llama_index.core import QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from FlagEmbedding import FlagReranker

class SemanticSearchEngine:
    def __init__(self, index, use_reranker=False):
        """Initialize search engine with vector index"""
        self.index = index
        if index is None:
            raise ValueError("Index cannot be None")
        
        # Initialize reranker if requested
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker:
            try:
                print("Loading BGE reranker...")
                self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
                print("✅ BGE reranker loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load reranker: {e}")
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
            if self.use_reranker and self.reranker and len(candidate_results) > 1:
                reranked_results = self._rerank_results(query, candidate_results, top_k)
                return reranked_results
            else:
                return candidate_results[:top_k+1]
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def _rerank_results(self, query, results, top_k):
        """Rerank results using BGE reranker"""
        try:
            # Prepare query-document pairs
            pairs = [[query, result["text"]] for result in results]
            
            # Get reranking scores
            scores = self.reranker.compute_score(pairs)
            
            # Ensure scores is a list
            if not isinstance(scores, list):
                scores = [scores]
            
            # Add rerank scores to results
            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)
            
            # Sort by rerank score (higher is better)
            reranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            print(f"Error during reranking: {e}")
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