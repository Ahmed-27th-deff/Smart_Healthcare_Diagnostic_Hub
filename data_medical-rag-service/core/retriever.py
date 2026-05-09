from core.embedding import embedding_model
from core.vector_db import collection


def semantic_search(query, k=5):
    q_emb = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    return results["documents"][0]


def semantic_search_with_scores(query, k=5):
    """Return (documents, distances) for query."""
    q_emb = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return docs, distances