from core.hybrid_search import hybrid_search, hybrid_search_debug, add_docs, tokenized
from core.reranker import rerank, rerank_with_scores
from core.vector_db import collection


def _ensure_bm25_loaded():
    # Try loading persisted index first
    try:
        from core import hybrid_search as hs
        if hs.load_index():
            return
    except Exception:
        pass

    if len(tokenized) == 0:
        try:
            res = collection.get(include=["documents"]) or {}
            docs = res.get("documents", [[]])[0]
            if docs:
                add_docs(docs)
        except Exception:
            # silently ignore; hybrid will still return semantic-only results
            pass


def get_context(query, top_k=4):
    """Return final context string (top_k) for given query."""
    _ensure_bm25_loaded()

    docs = hybrid_search(query, k=10)
    ranked = rerank(query, docs)

    final_docs = ranked[:top_k]
    return "\n".join(final_docs)


def get_context_debug(query, top_k=4):
    """Return detailed retrieval+rerank info for diagnostics.

    Returns dict with: semantic_docs, semantic_distances, bm25 (list), combined,
    reranked (list of (doc, score)), final_context (joined top_k).
    """
    _ensure_bm25_loaded()

    info = hybrid_search_debug(query, k=10)
    combined = info.get("combined", [])

    reranked = rerank_with_scores(query, combined)
    final_docs = [d for d, _ in reranked][:top_k]

    return {
        "semantic_docs": info.get("semantic_docs", []),
        "semantic_distances": info.get("semantic_distances", []),
        "bm25": info.get("bm25", []),
        "combined": combined,
        "reranked": reranked,
        "final_context": "\n".join(final_docs),
    }