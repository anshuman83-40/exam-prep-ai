"""Level 2: Smart search with embeddings + a FAISS vector index.

An embedding model turns text into a list of 384 numbers (a vector) that captures
its MEANING. Texts with similar meaning get vectors that point in a similar direction,
so "what does an agent sense with?" can find a chunk about "sensors" and "percepts"
even if the exact words are different. That's semantic search, not keyword search.

FAISS stores all chunk vectors and finds the closest ones to a question in milliseconds.

Run:  python levels/level2_search.py "your notes.pdf"
"""

import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from level1_read_pdf import PROJECT_DIR, read_pages, split_into_chunks

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # small, free, runs locally, strong at retrieval
# bge models work best when questions (not chunks) get this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 4


def build_index(chunks: list[tuple[int, str]], model: SentenceTransformer) -> faiss.Index:
    """Embed every chunk and store the vectors in a FAISS index."""
    texts = [text for _, text in chunks]
    # normalize -> every vector has length 1, so inner product == cosine similarity
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.asarray(vectors, dtype=np.float32))
    return index


def search(question: str, index: faiss.Index, model: SentenceTransformer, k: int = TOP_K):
    """Return the indexes and similarity scores of the k most similar chunks."""
    q = model.encode([QUERY_PREFIX + question], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype=np.float32), k)
    return list(zip(ids[0], scores[0]))


def main():
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(PROJECT_DIR.glob("*.pdf"))[0]

    pages = read_pages(pdf_path)
    chunks = [(n, piece) for n, text in enumerate(pages, start=1) for piece in split_into_chunks(text)]

    print(f"Loading embedding model {EMBED_MODEL} (first time downloads ~130 MB)...")
    model = SentenceTransformer(EMBED_MODEL)
    index = build_index(chunks, model)
    print(f"Indexed {index.ntotal} chunks, each a vector of {index.d} numbers.\n")

    while True:
        question = input("Search your notes (Enter to quit): ").strip()
        if not question:
            break
        for rank, (i, score) in enumerate(search(question, index, model), start=1):
            page, text = chunks[i]
            print(f"\n  #{rank}  page {page}  similarity {score:.2f}\n  {text[:250]}...")
        print()


if __name__ == "__main__":
    main()
