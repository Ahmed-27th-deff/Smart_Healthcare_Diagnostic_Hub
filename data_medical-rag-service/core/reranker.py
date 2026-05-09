from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs):
    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)

    return [d for _, d in sorted(zip(scores, docs), reverse=True)]


def rerank_with_scores(query, docs):
    """Return list of (doc, score) sorted by score descending."""
    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)

    doc_score = list(zip(docs, scores))
    doc_score.sort(key=lambda x: x[1], reverse=True)
    return doc_score