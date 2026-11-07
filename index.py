"""Milestone 4: index every PDF in data/ into Chroma.

Run:  python index.py

For each PDF: load -> chunk -> delete its old chunks -> embed + store.
Deleting first means re-running never creates duplicates, and never leaves
stale chunks behind (e.g. after changing CHUNK_SIZE).
"""

import sys
import time

import config
from rag import store
from rag.chunker import chunk_pages
from rag.loader import load_pdf


def main():
    pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {config.DATA_DIR}")
        return 1

    collection = store.get_collection()
    print(f"Collection '{config.COLLECTION_NAME}' in {config.CHROMA_DIR}: "
          f"{collection.count()} chunks before indexing\n")

    total_start = time.perf_counter()
    for path in pdfs:
        start = time.perf_counter()
        pages = load_pdf(path)
        chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        store.delete_file(collection, path.name)
        store.add_chunks(collection, chunks)
        print(f"  {path.name}: {len(pages)} pages -> {len(chunks)} chunks "
              f"({time.perf_counter() - start:.1f}s)")

    print(f"\nDone in {time.perf_counter() - total_start:.1f}s. "
          f"Collection now holds {collection.count()} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
