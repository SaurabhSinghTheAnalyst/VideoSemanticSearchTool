import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_engine import SemanticSearchEngine
from embedding_manager import EmbeddingManager
import json
import time

# Diverse test cases for robust evaluation
# Each prompt is crafted to test different retrieval capabilities
# (paraphrase, summary, comparison, list, inference, negative, analogy, etc.)
test_cases = [
    # Paraphrased/Indirect
    {
        "prompt": "Describe the art of using gaps between APIs and customer needs.",
        "expected_docs": [{
            "video_name": "api",
            "text": "Now, the ability to see gaps and a sort of arbitrage between what an API or other people's code enables you to and what customers need, this is the art of API arbitrage and it's the closest thing we have to magic in 2025, right?",
            "start": 46.3,
            "end": 62.3
        }]
    },
    # Open-ended/Summary
    {
        "prompt": "Summarize the main advice for aspiring businesspeople.",
        "expected_docs": [{
            "video_name": "Nikhil Kamath_ My Honest Business Advice",
            "text": "When I actively wanted to become a businessman.",
            "start": 1.4,
            "end": 8.7
        }]
    },
    # Comparison/Contrast
    {
        "prompt": "How does contextual chunking differ from traditional chunking?",
        "expected_docs": [{
            "video_name": "Contextual Retrieval with Any LLM_ A Step-by-Step Guide",
            "text": "The contextual retrieval system adds a pre processing step in which we take each chunk, feed that into a specific prompt, which also looks at the whole document, and then that prompt identify or locate that chunk in the whole document and add contextual information related to that specific chunk.",
            "start": 76.0,
            "end": 95.0
        }]
    },
    # List/Steps/Process
    {
        "prompt": "List the steps to create a knowledge graph.",
        "expected_docs": [{
            "video_name": "AI",
            "text": "But how do you create that knowledge graph?",
            "start": 650.3,
            "end": 653.0
        }]
    },
    # Why/How/When/Where
    {
        "prompt": "Why is instruction tuning valuable for language models?",
        "expected_docs": [{
            "video_name": "Colbert",
            "text": "Instruction tuning is a technique that allows pretrained language models to learn from input and response pairs.",
            "start": 747.255,
            "end": 752.935
        }]
    },
    # Opinion/Cause/Consequence
    {
        "prompt": "What does the speaker say about profitable SaaS strategies?",
        "expected_docs": [{
            "video_name": "api",
            "text": "So the most profitable founders I know, they treat APIs and accessing other people's code like attacks on progress, right?",
            "start": 433.6,
            "end": 445.0
        }]
    },
    # Inference/Context
    {
        "prompt": "Which video would help a beginner learn about YouTube growth?",
        "expected_docs": [{
            "video_name": "How I Get Over 100,000,000 Views Per Video",
            "text": "In this video, I'm gonna be showing you how you can get access to all my YouTube secrets. Everything I use to get over a 100,000,000 views a video.",
            "start": 0.0,
            "end": 4.2
        }]
    },
    # Negative/Exclusion
    {
        "prompt": "Find a segment that does not mention Google search.",
        "expected_docs": [{
            "video_name": "Colbert",
            "text": "Welcome to another video on how to do better rag.",
            "start": 0.6,
            "end": 3.5
        }]
    },
    # Example/Analogy/Metaphor
    {
        "prompt": "What analogy is used to explain APIs?",
        "expected_docs": [{
            "video_name": "api",
            "text": "So in API, imagine you're at a party where everyone brings half built LEGO sets.",
            "start": 74.7,
            "end": 82.0
        }]
    },
    # Speaker/Timeframe
    {
        "prompt": "Who discusses building a brand and when?",
        "expected_docs": [{
            "video_name": "Nikhil Kamath_ My Honest Business Advice",
            "text": "It was not the other way around that, Let me create a brand and now let me find out what brand to create.",
            "start": 131.1,
            "end": 140.0
        }]
    },
    # Paraphrase/Indirect
    {
        "prompt": "Explain the main pattern for data applications in simple terms.",
        "expected_docs": [{
            "video_name": "AI",
            "text": "So the core pattern is actually really really simple, but really really powerful.",
            "start": 446.1,
            "end": 451.0
        }]
    },
    # List/Process
    {
        "prompt": "What are the requirements for using the video search app?",
        "expected_docs": [{
            "video_name": "AI",
            "text": "I basically dedicated my professional life towards getting developers to be able to build better applications and build applications better by leveraging not just individual data points, kind of retrieved at once, like one at a time, or summed up or group calculated averages, but individual data points connected by relationships.",
            "start": 13.28,
            "end": 37.010002
        }]
    },
    # Why/How/When/Where
    {
        "prompt": "How does thumbnail search help with video performance?",
        "expected_docs": [{
            "video_name": "How I Get Over 100,000,000 Views Per Video",
            "text": "It's called thumbnail search, and you can type in whatever you want, and it will pull up thumbnails for you to get inspired off of.",
            "start": 247.0,
            "end": 255.0
        }]
    },
    # Inference/Context
    {
        "prompt": "If you want to improve your video ranking, what should you do?",
        "expected_docs": [{
            "video_name": "How I Get Over 100,000,000 Views Per Video",
            "text": "You can hit ranking and see what it ranked out of 10 when I uploaded it.",
            "start": 156.6,
            "end": 162.0
        }]
    },
    # Example/Analogy
    {
        "prompt": "Give an example of ultradian cycles as discussed.",
        "expected_docs": [{
            "video_name": "Exploring VideoDB for Efficient Video Chunking and Embedding-2",
            "text": "So if I type something like what is the best ultradian cycle time.",
            "start": 60.9,
            "end": 65.0
        }]
    },
]

def evaluate_retrieval(query, retrieved_docs, expected_docs, k=5):
    """
    Evaluate the retrieval performance of a model.
    Returns precision@k and whether expected content was found in top-k results.
    """
    if not retrieved_docs:
        return 0.0, False
    
    # Check if any expected video appears in top-k results
    expected_video_names = {doc["video_name"] for doc in expected_docs}
    retrieved_video_names = {doc["video_file"] for doc in retrieved_docs[:k]}
    
    found_expected = bool(expected_video_names.intersection(retrieved_video_names))
    
    # Simple precision calculation based on relevance
    relevant_docs = [doc for doc in retrieved_docs[:k] if doc.get("relevant", False)]
    precision_at_k = len(relevant_docs) / k if k > 0 else 0.0
    
    return precision_at_k, found_expected

def run_evaluation():
    """Run comprehensive evaluation with timing and token usage logging."""
    print("🔧 Initializing unified embedding manager...")
    
    try:
        # Initialize unified embedding manager
        embedding_manager = EmbeddingManager()
        
        # Check if index exists
        stats = embedding_manager.get_collection_stats()
        if not stats or stats['document_count'] == 0:
            print("❌ No index found or empty index. Please run transcript processing first.")
            print("   Run: python -c 'from embedding_manager import EmbeddingManager; EmbeddingManager().process_all_transcripts()'")
            return
        
        print(f"📊 Index loaded with {stats['document_count']} documents")
        
        # Load index using unified embedding manager
        index = embedding_manager.load_existing_index()
        if not index:
            print("❌ Failed to load index")
            return
        
        # Initialize search engines
        search_engine_basic = SemanticSearchEngine(index, use_reranker=False)
        print("✅ Basic search engine ready!")
        
        search_engine_rerank = SemanticSearchEngine(index, use_reranker=True)
        print("✅ Reranker search engine ready!\n")
        
        # Evaluation metrics
        basic_scores = []
        rerank_scores = []
        basic_found_count = 0
        rerank_found_count = 0
        total_basic_time = 0
        total_rerank_time = 0
        
        # Reset token counter for evaluation
        initial_tokens = embedding_manager.token_counter.total_embedding_token_count
        
        print(f"🧪 STARTING EVALUATION OF {len(test_cases)} TEST CASES")
        print("=" * 100)
        
        for i, test_case in enumerate(test_cases, 1):
            prompt = test_case["prompt"]
            expected_docs = test_case["expected_docs"]
            
            print(f"\n🧪 TEST CASE {i}/{len(test_cases)}")
            print("=" * 80)
            print(f"📝 PROMPT: {prompt}")
            print("\n📋 EXPECTED DOCUMENT:")
            print(json.dumps(expected_docs, indent=2, ensure_ascii=False))
            
            # Basic search with timing
            print(f"\n🔍 BASIC SEARCH RESULTS:")
            print("-" * 60)
            
            start_time = time.time()
            basic_results = search_engine_basic.search(prompt, top_k=3)
            basic_time = time.time() - start_time
            total_basic_time += basic_time
            
            if basic_results:
                for j, doc in enumerate(basic_results, 1):
                    print(f"\n{j}. Video: {doc['video_file']}")
                    print(f"   Time: {doc['timestamp']}")
                    print(f"   Text: {doc['text'][:100]}...")
                    print(f"   Score: {doc['similarity_score']:.3f}")
                print(f"\n⏱️  Basic search time: {basic_time:.2f}s")
            else:
                print("No results found.")
            
            # Evaluate basic search
            basic_precision, basic_found = evaluate_retrieval(prompt, basic_results, expected_docs, k=3)
            basic_scores.append(basic_precision)
            if basic_found:
                basic_found_count += 1
            
            # Reranked search with timing
            print(f"\n🎯 RERANKED SEARCH RESULTS:")
            print("-" * 60)
            
            start_time = time.time()
            rerank_results = search_engine_rerank.search(prompt, top_k=3)
            rerank_time = time.time() - start_time
            total_rerank_time += rerank_time
            
            if rerank_results:
                for j, doc in enumerate(rerank_results, 1):
                    print(f"\n{j}. Video: {doc['video_file']}")
                    print(f"   Time: {doc['timestamp']}")
                    print(f"   Text: {doc['text'][:100]}...")
                    print(f"   Similarity Score: {doc['similarity_score']:.3f}")
                    if 'rerank_score' in doc:
                        print(f"   Rerank Score: {doc['rerank_score']:.3f}")
                print(f"\n⏱️  Rerank search time: {rerank_time:.2f}s")
            else:
                print("No results found.")
            
            # Evaluate reranked search
            rerank_precision, rerank_found = evaluate_retrieval(prompt, rerank_results, expected_docs, k=3)
            rerank_scores.append(rerank_precision)
            if rerank_found:
                rerank_found_count += 1
            
            print(f"\n📊 EVALUATION METRICS:")
            print(f"   Basic - Found Expected: {'✅' if basic_found else '❌'}")
            print(f"   Rerank - Found Expected: {'✅' if rerank_found else '❌'}")
            print(f"{'='*80}")
        
        # Final evaluation summary
        print(f"\n🎯 FINAL EVALUATION SUMMARY")
        print("=" * 100)
        
        avg_basic_precision = sum(basic_scores) / len(basic_scores) if basic_scores else 0
        avg_rerank_precision = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0
        
        basic_hit_rate = basic_found_count / len(test_cases) * 100
        rerank_hit_rate = rerank_found_count / len(test_cases) * 100
        
        print(f"📈 PERFORMANCE METRICS:")
        print(f"   Basic Search Hit Rate: {basic_hit_rate:.1f}% ({basic_found_count}/{len(test_cases)})")
        print(f"   Rerank Search Hit Rate: {rerank_hit_rate:.1f}% ({rerank_found_count}/{len(test_cases)})")
        print(f"   Average Basic Precision@3: {avg_basic_precision:.3f}")
        print(f"   Average Rerank Precision@3: {avg_rerank_precision:.3f}")
        
        print(f"\n⏱️  TIMING ANALYSIS:")
        print(f"   Total Basic Search Time: {total_basic_time:.2f}s")
        print(f"   Total Rerank Search Time: {total_rerank_time:.2f}s")
        print(f"   Average Basic Search Time: {total_basic_time/len(test_cases):.2f}s")
        print(f"   Average Rerank Search Time: {total_rerank_time/len(test_cases):.2f}s")
        print(f"   Rerank Overhead: {((total_rerank_time/total_basic_time)-1)*100:.1f}% slower")
        
        print(f"\n💰 TOKEN USAGE:")
        total_tokens_used = embedding_manager.token_counter.total_embedding_token_count - initial_tokens
        print(f"   Total Embedding Tokens Used: {total_tokens_used}")
        print(f"   Average Tokens per Query: {total_tokens_used/(len(test_cases)*2):.1f}")
        
        print(f"\n🏆 WINNER:")
        if rerank_hit_rate > basic_hit_rate:
            print(f"   🥇 Reranker wins with {rerank_hit_rate:.1f}% hit rate (+{rerank_hit_rate-basic_hit_rate:.1f}%)")
        elif basic_hit_rate > rerank_hit_rate:
            print(f"   🥇 Basic search wins with {basic_hit_rate:.1f}% hit rate (+{basic_hit_rate-rerank_hit_rate:.1f}%)")
        else:
            print(f"   🤝 Tie at {basic_hit_rate:.1f}% hit rate")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_evaluation() 