def chunk_text(text, chunk_size=500, overlap=100):
    # Basic sliding-window chunker with trimming.
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end - overlap

    # remove very short chunks
    filtered = [c for c in chunks if len(c) >= 50 and any(ch.isalpha() for ch in c)]
    return filtered