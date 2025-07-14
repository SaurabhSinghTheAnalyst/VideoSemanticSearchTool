import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_engine import SemanticSearchEngine
from embedding_manager import EmbeddingManager
import json
import time

# Define comprehensive test cases with prompts and expected documents based on actual transcript content
test_cases = [
    # Original test cases
    {
        "prompt": "How to earn your status?",
        "expected_docs": [{
            "video_name": "It's Never Been Easier To Make Money - Naval Ravikant",
            "text": "But even back then, you had to earn your status by taking care of the tribe.",
            "start": 29.17,
            "end": 32.77
        }]
    },
    {
        "prompt": "How to build applications better",
        "expected_docs": [{
            "video_name": "AI",
            "text": "I basically dedicated my professional life towards getting developers to be able to build better applications and build applications better by leveraging not just individual data points, kind of retrieved at once, like one at a time, or summed up or group calculated averages, but individual data points connected by relationships.",
            "start": 13.28,
            "end": 37.010002
        }]
    },
    {
        "prompt": "What is instruction tuning?",
        "expected_docs": [{
            "video_name": "Colbert",
            "text": "Instruction tuning is a technique that allows pretrained language models to learn from input and response pairs.",
            "start": 747.255,
            "end": 752.935
        }]
    },
    {
        "prompt": "What is contextual retrieval?",
        "expected_docs": [{
            "video_name": "Contextual Retrieval with Any LLM_ A Step-by-Step Guide",
            "text": "In a previous video, we looked at contextual retrieval, which is a new chunking strategy from Anthropic, which has shown to improve the failure rates by thirty five percent in contextual embeddings, and when combined with PM25, the failure rate can be reduced by forty nine percent.",
            "start": 0.0,
            "end": 18.265
        }]
    },
    
    # New test cases based on transcript analysis
    {
        "prompt": "What is API arbitrage?",
        "expected_docs": [{
            "video_name": "api",
            "text": "Now, the ability to see gaps and a sort of arbitrage between what an API or other people's code enables you to and what customers need, this is the art of API arbitrage and it's the closest thing we have to magic in 2025, right?",
            "start": 46.3,
            "end": 62.3
        }]
    },
    {
        "prompt": "How to do better RAG?",
        "expected_docs": [{
            "video_name": "Colbert",
            "text": "Welcome to another video on how to do better rag.",
            "start": 0.6,
            "end": 3.5
        }]
    },
    {
        "prompt": "What is the LEGO philosophy for APIs?",
        "expected_docs": [{
            "video_name": "api",
            "text": "So in API, imagine you're at a party where everyone brings half built LEGO sets.",
            "start": 74.7,
            "end": 82.0
        }]
    },
    {
        "prompt": "How to get YouTube views?",
        "expected_docs": [{
            "video_name": "How I Get Over 100,000,000 Views Per Video",
            "text": "In this video, I'm gonna be showing you how you can get access to all my YouTube secrets. Everything I use to get over a 100,000,000 views a video.",
            "start": 0.0,
            "end": 4.2
        }]
    },
    {
        "prompt": "What is thumbnail search?",
        "expected_docs": [{
            "video_name": "How I Get Over 100,000,000 Views Per Video",
            "text": "It's called thumbnail search, and you can type in whatever you want, and it will pull up thumbnails for you to get inspired off of.",
            "start": 247.0,
            "end": 255.0
        }]
    },
    {
        "prompt": "How to create knowledge graphs?",
        "expected_docs": [{
            "video_name": "AI",
            "text": "But how do you create that knowledge graph?",
            "start": 650.3,
            "end": 653.0
        }]
    },
    {
        "prompt": "What happened before Google search?",
        "expected_docs": [{
            "video_name": "AI",
            "text": "Everyone here in this room knows that the vast majority of web searches today are handled with Google. But some of you know that it didn't start that way, it started this way.",
            "start": 49.3,
            "end": 58.0
        }]
    },
    {
        "prompt": "How to become a businessman?",
        "expected_docs": [{
            "video_name": "Nikhil Kamath_ My Honest Business Advice",
            "text": "When I actively wanted to become a businessman.",
            "start": 1.4,
            "end": 8.7
        }]
    },
    {
        "prompt": "What are ultradian cycles?",
        "expected_docs": [{
            "video_name": "Exploring VideoDB for Efficient Video Chunking and Embedding-2",
            "text": "So if I type something like what is the best ultradian cycle time.",
            "start": 60.9,
            "end": 65.0
        }]
    },
    {
        "prompt": "How to use OpenAI for chatbots?",
        "expected_docs": [{
            "video_name": "api",
            "text": "So like, you know, you would feed ChatGPT, and there's hundreds of different tutorials, guys, on how to make like a website chatbot with OpenAI.",
            "start": 737.7,
            "end": 750.0
        }]
    },
    {
        "prompt": "What are profitable SaaS strategies?",
        "expected_docs": [{
            "video_name": "api",
            "text": "So the most profitable founders I know, they treat APIs and accessing other people's code like attacks on progress, right?",
            "start": 433.6,
            "end": 445.0
        }]
    },
    {
        "prompt": "What is video ranking?",
        "expected_docs": [{
            "video_name": "How I Get Over 100,000,000 Views Per Video",
            "text": "You can hit ranking and see what it ranked out of 10 when I uploaded it.",
            "start": 156.6,
            "end": 162.0
        }]
    },
    {
        "prompt": "How does contextual chunking work?",
        "expected_docs": [{
            "video_name": "Contextual Retrieval with Any LLM_ A Step-by-Step Guide",
            "text": "The contextual retrieval system adds a pre processing step in which we take each chunk, feed that into a specific prompt, which also looks at the whole document, and then that prompt identify or locate that chunk in the whole document and add contextual information related to that specific chunk.",
            "start": 76.0,
            "end": 95.0
        }]
    },
    {
        "prompt": "What embeddings are used in contextual retrieval?",
        "expected_docs": [{
            "video_name": "Contextual Retrieval with Any LLM_ A Step-by-Step Guide",
            "text": "For embeddings, we're using the OpenAI embedding.",
            "start": 319.6,
            "end": 323.0
        }]
    },
    {
        "prompt": "What is the core pattern for data applications?",
        "expected_docs": [{
            "video_name": "AI",
            "text": "So the core pattern is actually really really simple, but really really powerful.",
            "start": 446.1,
            "end": 451.0
        }]
    },
    {
        "prompt": "How to create a brand?",
        "expected_docs": [{
            "video_name": "Nikhil Kamath_ My Honest Business Advice",
            "text": "It was not the other way around that, Let me create a brand and now let me find out what brand to create.",
            "start": 131.1,
            "end": 140.0
        }]
    }
]

def format_time(seconds):
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def evaluate_retrieval(query, retrieved_docs, expected_docs, k=5):
    """
    Evaluate the retrieval performance of a model.
    Returns precision@k, recall@k, and whether expected content was found.
    """
    if not retrieved_docs:
        return 0.0, 0.0, False
    
    # Check if any expected video appears in top-k results
    expected_video_names = {doc["video_name"] for doc in expected_docs}
    retrieved_video_names = {doc["video_file"] for doc in retrieved_docs[:k]}
    
    found_expected = bool(expected_video_names.intersection(retrieved_video_names))
    
    # Calculate precision and recall
    relevant_docs = [doc for doc in retrieved_docs[:k] 
                     if doc["video_file"] in expected_video_names]
    
    precision_at_k = len(relevant_docs) / k if k > 0 else 0.0
    recall_at_k = len(relevant_docs) / len(expected_docs) if len(expected_docs) > 0 else 0.0
    
    return precision_at_k, recall_at_k, found_expected

def calculate_mrr(retrieved_docs, expected_docs):
    """Calculate Mean Reciprocal Rank."""
    expected_video_names = {doc["video_name"] for doc in expected_docs}
    
    for i, doc in enumerate(retrieved_docs, 1):
        if doc["video_file"] in expected_video_names:
            return 1.0 / i
    return 0.0

def run_comprehensive_evaluation():
    """Run comprehensive evaluation with detailed metrics and analysis."""
    print("🔧 Initializing search engines...")
    
    try:
        # Initialize unified embedding manager
        embedding_manager = EmbeddingManager()
        
        # Check if index exists
        stats = embedding_manager.get_collection_stats()
        if not stats or stats['document_count'] == 0:
            print("❌ No index found or empty index. Please run transcript processing first.")
            print("   Run: python embedding_manager.py")
            return
        
        print(f"Loaded existing index successfully!")
        
        # Load index
        index = embedding_manager.load_existing_index()
        if not index:
            print("❌ Failed to load index")
            return
        
        # Initialize search engines
        search_engine_basic = SemanticSearchEngine(index, use_reranker=False)
        print("✅ Basic search engine ready!")
        
        search_engine_rerank = SemanticSearchEngine(index, use_reranker=True)
        print("✅ Reranker search engine ready!")
        
        # Evaluation metrics storage
        basic_metrics = {
            'precisions': [], 'recalls': [], 'mrrs': [],
            'found_count': 0, 'times': []
        }
        rerank_metrics = {
            'precisions': [], 'recalls': [], 'mrrs': [],
            'found_count': 0, 'times': []
        }
        
        # Track token usage
        initial_tokens = embedding_manager.token_counter.total_embedding_token_count
        
        for i, test_case in enumerate(test_cases, 1):
            prompt = test_case["prompt"]
            expected_docs = test_case["expected_docs"]
            
            print(f"\n🧪 TEST CASE {i}/{len(test_cases)}")
            print("=" * 80)
            print(f"📝 PROMPT: {prompt}")
            print()
            
            # Display expected document
            print("📋 EXPECTED DOCUMENT:")
            print("[")
            for doc in expected_docs:
                print("  {")
                print(f'    "video_name": "{doc["video_name"]}",')
                print(f'    "text": "{doc["text"]}",')
                print(f'    "start": {doc["start"]},')
                print(f'    "end": {doc["end"]}')
                print("  }")
            print("]")
            print()
            
            # Basic search evaluation
            print("🔍 BASIC SEARCH RESULTS:")
            print("-" * 60)
            print()
            
            start_time = time.time()
            basic_results = search_engine_basic.search(prompt, top_k=4)
            basic_time = time.time() - start_time
            basic_metrics['times'].append(basic_time)
            
            if basic_results:
                for j, doc in enumerate(basic_results, 1):
                    print(f"{j}. Video: {doc['video_file']}")
                    print(f"   Time: {format_time(doc['start_time'])} - {format_time(doc['end_time'])}")
                    print(f"   Text: {doc['text']}")
                    print(f"   Score: {doc['similarity_score']:.3f}")
                    print("-" * 40)
                    print()
            else:
                print("No results found.")
                print()
            
            # Evaluate basic search
            basic_precision, basic_recall, basic_found = evaluate_retrieval(
                prompt, basic_results, expected_docs, k=5)
            basic_mrr = calculate_mrr(basic_results, expected_docs)
            
            basic_metrics['precisions'].append(basic_precision)
            basic_metrics['recalls'].append(basic_recall)
            basic_metrics['mrrs'].append(basic_mrr)
            if basic_found:
                basic_metrics['found_count'] += 1
            
            # Reranked search evaluation
            print("🎯 RERANKED SEARCH RESULTS:")
            print("-" * 60)
            print()
            
            start_time = time.time()
            rerank_results = search_engine_rerank.search(prompt, top_k=4)
            rerank_time = time.time() - start_time
            rerank_metrics['times'].append(rerank_time)
            
            if rerank_results:
                for j, doc in enumerate(rerank_results, 1):
                    print(f"{j}. Video: {doc['video_file']}")
                    print(f"   Time: {format_time(doc['start_time'])} - {format_time(doc['end_time'])}")
                    print(f"   Text: {doc['text']}")
                    print(f"   Similarity Score: {doc['similarity_score']:.3f}")
                    if 'rerank_score' in doc:
                        print(f"   Rerank Score: {doc['rerank_score']:.3f}")
                    print("-" * 40)
                    print()
            else:
                print("No results found.")
                print()
            
            # Evaluate reranked search
            rerank_precision, rerank_recall, rerank_found = evaluate_retrieval(
                prompt, rerank_results, expected_docs, k=5)
            rerank_mrr = calculate_mrr(rerank_results, expected_docs)
            
            rerank_metrics['precisions'].append(rerank_precision)
            rerank_metrics['recalls'].append(rerank_recall)
            rerank_metrics['mrrs'].append(rerank_mrr)
            if rerank_found:
                rerank_metrics['found_count'] += 1
            
            print("=" * 80)
            print()
        
        # Calculate final statistics - same as before
        print(f"\n🎯 COMPREHENSIVE EVALUATION RESULTS")
        print("=" * 100)
        
        # Calculate averages
        def calc_avg(values):
            return sum(values) / len(values) if values else 0.0
        
        basic_avg_precision = calc_avg(basic_metrics['precisions'])
        basic_avg_recall = calc_avg(basic_metrics['recalls'])
        basic_avg_mrr = calc_avg(basic_metrics['mrrs'])
        basic_hit_rate = (basic_metrics['found_count'] / len(test_cases)) * 100
        basic_avg_time = calc_avg(basic_metrics['times'])
        
        rerank_avg_precision = calc_avg(rerank_metrics['precisions'])
        rerank_avg_recall = calc_avg(rerank_metrics['recalls'])
        rerank_avg_mrr = calc_avg(rerank_metrics['mrrs'])
        rerank_hit_rate = (rerank_metrics['found_count'] / len(test_cases)) * 100
        rerank_avg_time = calc_avg(rerank_metrics['times'])
        
        print(f"📈 RETRIEVAL PERFORMANCE:")
        print(f"                    │   Basic    │  Reranker  │ Improvement")
        print(f"   ─────────────────┼────────────┼────────────┼─────────────")
        print(f"   Hit Rate         │   {basic_hit_rate:6.1f}%   │   {rerank_hit_rate:6.1f}%   │   {rerank_hit_rate-basic_hit_rate:+6.1f}%")
        print(f"   Precision@5      │   {basic_avg_precision:8.3f}   │   {rerank_avg_precision:8.3f}   │   {rerank_avg_precision-basic_avg_precision:+8.3f}")
        print(f"   Recall@5         │   {basic_avg_recall:8.3f}   │   {rerank_avg_recall:8.3f}   │   {rerank_avg_recall-basic_avg_recall:+8.3f}")
        print(f"   Mean Reciprocal  │   {basic_avg_mrr:8.3f}   │   {rerank_avg_mrr:8.3f}   │   {rerank_avg_mrr-basic_avg_mrr:+8.3f}")
        print(f"   Rank (MRR)       │            │            │")
        
        print(f"\n⏱️  PERFORMANCE TIMING:")
        print(f"   Average Query Time:")
        print(f"     Basic Search:    {basic_avg_time:.3f}s")
        print(f"     Reranked Search: {rerank_avg_time:.3f}s")
        print(f"     Rerank Overhead: {((rerank_avg_time/basic_avg_time)-1)*100:+.1f}%")
        
        print(f"\n💰 TOKEN USAGE ANALYSIS:")
        total_tokens_used = embedding_manager.token_counter.total_embedding_token_count - initial_tokens
        tokens_per_query = total_tokens_used / (len(test_cases) * 2) if len(test_cases) > 0 else 0
        
        print(f"   Total Embedding Tokens: {total_tokens_used:,}")
        print(f"   Tokens per Query: {tokens_per_query:.1f}")
        print(f"   Total Queries: {len(test_cases) * 2}")
        
        # Estimated cost (approximate OpenAI pricing)
        cost_per_1k_tokens = 0.00013  # text-embedding-3-large pricing
        estimated_cost = (total_tokens_used / 1000) * cost_per_1k_tokens
        print(f"   Estimated Cost: ${estimated_cost:.4f}")
        
        print(f"\n🏆 FINAL VERDICT:")
        
        improvements = {
            'hit_rate': rerank_hit_rate - basic_hit_rate,
            'precision': rerank_avg_precision - basic_avg_precision,
            'mrr': rerank_avg_mrr - basic_avg_mrr
        }
        
        if improvements['hit_rate'] > 5:  # Significant improvement
            print(f"   🥇 RERANKER WINS decisively!")
            print(f"      ↗️  Hit rate improved by {improvements['hit_rate']:.1f}%")
            print(f"      ↗️  MRR improved by {improvements['mrr']:.3f}")
            print(f"      💸 Cost: {((rerank_avg_time/basic_avg_time)-1)*100:.0f}% slower")
        elif improvements['hit_rate'] > 0:
            print(f"   🥈 RERANKER WINS marginally")
            print(f"      ↗️  Small improvements across metrics")
        elif improvements['hit_rate'] < -5:
            print(f"   🥇 BASIC SEARCH WINS!")
            print(f"      ↗️  Better performance, much faster")
        else:
            print(f"   🤝 TIE - Similar performance")
            print(f"      ⚖️  Choose based on speed vs accuracy needs")
        
        print(f"\n📝 RECOMMENDATION:")
        if rerank_hit_rate > basic_hit_rate + 5:
            print(f"   Use reranker for high-accuracy applications")
        elif basic_avg_time < rerank_avg_time * 0.5:
            print(f"   Use basic search for speed-critical applications")
        else:
            print(f"   Both options are viable - test with your specific use case")
        
        print("\n")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_evaluation() 