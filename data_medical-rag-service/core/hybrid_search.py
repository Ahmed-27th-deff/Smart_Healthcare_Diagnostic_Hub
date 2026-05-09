import os
import re
import pickle
from pathlib import Path
from core.retriever import semantic_search, semantic_search_with_scores
from rank_bm25 import BM25Okapi

# 1. Use absolute paths to prevent random/duplicate directory creation
ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "vectordb" / "bm25_index.pkl"

all_docs = []
tokenized = []

# 2. Performance fix: Cache the BM25 instance in memory
_bm25_model = None

def _build_bm25():
    """Initialize the BM25 model once to avoid recalculating IDF across the corpus on every query."""
    global _bm25_model
    if tokenized:
        _bm25_model = BM25Okapi(tokenized)
    else:
        _bm25_model = None

def add_docs(docs):
    global all_docs, tokenized
    added = False
    
    for d in docs:
        if not isinstance(d, str):
            continue
        s = " ".join(d.split()).strip()
        if len(s) < 50:
            continue
        if not any(ch.isalpha() for ch in s):
            continue

        # Prevent exact duplicates
        norm = s.lower()
        if norm in (x.lower() for x in all_docs):
            continue

        toks = [t for t in re.findall(r"\w+", norm) if len(t) >= 2]
        if not toks:
            continue

        all_docs.append(s)
        tokenized.append(toks)
        added = True

    # Rebuild the in-memory model only if new documents were added
    if added:
        _build_bm25()

def _tokenize_text(text):
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) >= 2]

def save_index(path: str = None):
    p = path or str(INDEX_PATH)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"all_docs": all_docs, "tokenized": tokenized}, f)
        return True
    except Exception:
        return False

def load_index(path: str = None):
    p = path or str(INDEX_PATH)
    if not os.path.exists(p):
        return False

    try:
        with open(p, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            docs = data.get("all_docs", [])
            toks = data.get("tokenized", [])
            if docs and toks:
                global all_docs, tokenized
                all_docs = docs
                tokenized = toks
                
                # Initialize the model after successful load so it's ready for searching
                _build_bm25()
                return True
    except Exception:
        return False

    return False

def get_rrf_ranking(semantic_docs, bm25_docs, k_param=60):
    """
    3. Mathematical fusion using Reciprocal Rank Fusion (RRF).
    This function standardizes scores between semantic and keyword search,
    and elegantly removes duplicate documents from the final results.
    """
    rrf_scores = {}

    # Evaluate semantic search results
    for rank, doc in enumerate(semantic_docs):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0.0
        rrf_scores[doc] += 1.0 / (k_param + rank + 1)

    # Evaluate keyword search results (BM25)
    for rank, doc in enumerate(bm25_docs):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0.0
        rrf_scores[doc] += 1.0 / (k_param + rank + 1)

    # Sort final results based on the highest RRF Score
    ranked_docs = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
    return ranked_docs

def hybrid_search(query, k=5):
    # 4. Respect the 'k' parameter dynamically instead of using hardcoded values
    sem_docs = semantic_search(query, k=k)

    if not _bm25_model:
        return sem_docs

    q_toks = _tokenize_text(query)
    scores = _bm25_model.get_scores(q_toks)

    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    # Exclude results that have a score of exactly 0
    bm25_docs = [all_docs[i] for i in top_idx if scores[i] > 0]

    # Merge results and eliminate duplicates using the RRF algorithm
    combined = get_rrf_ranking(sem_docs, bm25_docs)

    return combined[:k]

def hybrid_search_debug(query, k=5):
    sem_docs, sem_dists = semantic_search_with_scores(query, k=k)

    if not _bm25_model:
        return {
            "semantic_docs": sem_docs,
            "semantic_distances": sem_dists,
            "bm25": [],
            "combined": sem_docs[:k],
            "all_docs_count": len(all_docs),
            "tokenized_count": len(tokenized),
        }

    q_toks = _tokenize_text(query)
    scores = _bm25_model.get_scores(q_toks)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    bm25_list = [(i, scores[i], all_docs[i]) for i in top_idx if i < len(all_docs) and scores[i] > 0]
    bm25_docs = [doc for _, _, doc in bm25_list]

    combined = get_rrf_ranking(sem_docs, bm25_docs)

    return {
        "semantic_docs": sem_docs,
        "semantic_distances": sem_dists,
        "bm25": bm25_list,
        "combined": combined[:k],
        "all_docs_count": len(all_docs),
        "tokenized_count": len(tokenized),
    }