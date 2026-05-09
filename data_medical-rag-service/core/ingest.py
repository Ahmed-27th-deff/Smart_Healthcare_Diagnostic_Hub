import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if __package__ is None:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.loader import load_pdf
from core.chunker import chunk_text
from core.embedding import embedding_model
from core.vector_db import collection
import uuid


def ingest_folder(folder, category):
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            text = load_pdf(os.path.join(folder, file))
            chunks = chunk_text(text)

            for c in chunks:
                emb = embedding_model.encode(c).tolist()

                collection.add(
                    documents=[c],
                    embeddings=[emb],
                    ids=[str(uuid.uuid4())],
                    metadatas=[{"category": category}]
                )


if __name__ == "__main__":
    # Ingest all PDFs dropped into the project `data/` folder (no subfolders required).
    data_dir = PROJECT_ROOT / "data"

    if not data_dir.exists():
        os.makedirs(data_dir, exist_ok=True)

    # track seen chunks during this ingestion to avoid duplicates
    seen_chunks = set()

    for f in os.listdir(str(data_dir)):
        if f.lower().endswith(".pdf"):
            path = os.path.join(str(data_dir), f)
            text = load_pdf(path)
            chunks = chunk_text(text)

            for c in chunks:
                # normalize and skip near-duplicates / tiny pieces
                snippet = " ".join(c.split())
                if len(snippet) < 50:
                    continue
                if not any(ch.isalpha() for ch in snippet):
                    continue

                key = snippet[:200].lower()
                if key in seen_chunks:
                    continue
                seen_chunks.add(key)

                emb = embedding_model.encode(snippet).tolist()
                collection.add(
                    documents=[snippet],
                    embeddings=[emb],
                    ids=[str(uuid.uuid4())],
                    metadatas=[{"source": f}]
                )

    print(" INGESTION DONE")
    # Populate BM25 in-memory index and persist it so retrieval works for others
    try:
        from core import hybrid_search

        # try loading existing index first
        loaded = hybrid_search.load_index()
        if not loaded:
            res = collection.get(include=["documents"]) or {}
            docs_outer = res.get("documents")

            # normalize into a flat list of documents
            docs = []
            if isinstance(docs_outer, list) and len(docs_outer) > 0:
                # chroma returns a list-of-lists, take first inner list if present
                first = docs_outer[0]
                if isinstance(first, list):
                    docs = first
                else:
                    docs = docs_outer

            if docs:
                hybrid_search.add_docs(docs)
                saved = hybrid_search.save_index()
                print(f" BM25 index built with {len(docs)} docs (saved={saved})")
            else:
                print(" No documents found in vectordb to build BM25 index")
        else:
            print(" BM25 index loaded from disk")
    except Exception as e:
        print(" Could not populate/persist BM25 index:", e)