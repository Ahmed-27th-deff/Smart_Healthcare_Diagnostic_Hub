import time
from core.rag_engine import get_context
from core import hybrid_search as hs
from core import reranker as rr
from core.vector_db import collection

def print_div(char="=", length=80):
    print(char * length)

def test_basic_queries():
    # Added custom queries to specifically test BM25's exact keyword matching strength
    queries = [
        "What are diabetes complications?", # Tends to favor semantic search
        "High BMI and risk factors",        # Excellent for hybrid search
        "Symptoms of pneumonia",            # Good for hybrid search
        "HbA1c levels 6.5%",                # Strict test for BM25 (exact numbers and medical terms)
        "Metformin dosage for type 2",      # Test for BM25 (drug names)
    ]

    print_div("-")
    print("⚙️ Initializing BM25 Index...")
    
    # Properly load the BM25 index from the saved file instead of fetching from the vector database
    try:
        if len(hs.tokenized) == 0:
            if hs.load_index():
                print(f"✅ Successfully loaded {len(hs.all_docs)} documents from saved BM25 index.")
            else:
                print("⚠️ Could not load BM25 index from disk. Please run ingest.py first.")
    except Exception as e:
        print("❌ ERROR loading BM25:", e)
    print_div("-")

    for i, q in enumerate(queries):
        print("\n" + "=" * 80)
        print(f"🔍 Test {i+1}: '{q}'")
        print("=" * 80)

        start_time = time.time()
        
        # 1. Retrieve detailed hybrid search information
        info = hs.hybrid_search_debug(q, k=10)
        
        sem_docs = info.get("semantic_docs", [])
        sem_dists = info.get("semantic_distances", [])
        bm25_info = info.get("bm25", [])
        combined = info.get("combined", [])

        print("\n🧠 -- Semantic Search (Vector) Results --")
        if not sem_docs:
            print("   No semantic results found.")
        for idx, (doc, dist) in enumerate(zip(sem_docs[:3], sem_dists[:3])):
            short = doc[:150].replace("\n", " ").strip() + "..."
            print(f"   [{idx+1}] Dist: {dist:.4f} | {short}")

        print("\n🔤 -- BM25 (Keyword) Results --")
        if not bm25_info:
            print("   No BM25 keyword matches found (Score = 0).")
        for idx, score, doc in bm25_info[:3]:
            short = doc[:150].replace("\n", " ").strip() + "..."
            print(f"   [ID:{idx}] Score: {score:.4f} | {short}")

        # Analyze the intersection/overlap between semantic and keyword search
        bm25_docs_only = [doc for _, _, doc in bm25_info]
        overlap = set(sem_docs).intersection(set(bm25_docs_only))
        
        print(f"\n🔗 -- Hybrid Fusion (RRF) Stats --")
        print(f"   Total Semantic Docs: {len(sem_docs)}")
        print(f"   Total BM25 Docs: {len(bm25_docs_only)}")
        print(f"   Overlap (Found in both): {len(overlap)}")
        print(f"   Combined Unique Docs passed to Reranker: {len(combined)}")

        print("\n⚖️ -- Cross-Encoder Reranking (Top 3) --")
        ranked = rr.rerank_with_scores(q, combined)
        for j, (doc, score) in enumerate(ranked[:3]):
            short = doc[:150].replace("\n", " ").strip() + "..."
            
            # Identify the source of the winning document (Semantic, BM25, or both)
            source = []
            if doc in sem_docs: source.append("Semantic")
            if doc in bm25_docs_only: source.append("BM25")
            source_str = " + ".join(source)
            
            print(f"   [{j+1}] Score: {score:.4f} [Source: {source_str}] -> {short}")

        # 2. Fetch the final context used by the RAG engine
        final_context = get_context(q)
        end_time = time.time()
        
        print("\n📄 --- Final RAG Context (First 300 chars) ---")
        # Fix: Store cleaned context in a variable before printing to avoid Python < 3.12 f-string errors
        cleaned_context = final_context[:300].replace('\n', ' ')
        print(f"   {cleaned_context}...")
        print(f"\n⏱️ Query Latency: {(end_time - start_time):.4f} seconds")


def test_edge_cases():
    queries = ["", "asdasdasdasd", "123456789"]
    for q in queries:
        print_div()
        print(f"🛡️ Edge Case Test: '{q}'")
        try:
            start = time.time()
            _ = get_context(q)
            print(f"   ✅ Handled gracefully in {(time.time() - start):.4f} seconds (No crash)")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    print_div()

if __name__ == "__main__":
    print("\n🚀 Starting Retrieval & Hybrid Search Validation...\n")
    test_basic_queries()
    test_edge_cases()
    print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY\n")