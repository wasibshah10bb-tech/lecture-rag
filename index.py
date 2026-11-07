"""Milestones 4 + 8: index the PDFs in data/ into Chroma, incrementally.

Run:  python index.py

For each PDF a fingerprint (hash of contents + indexing settings) is compared
with the one stored in Chroma:
  unchanged       -> skipped, nothing is embedded
  new / changed   -> its old chunks are deleted, then load -> chunk -> embed -> store
Files that were indexed but are no longer in data/ have their chunks removed.
"""

import hashlib
import sys
import time

import config
from rag import store
from rag.chunker import chunk_pages
from rag.loader import load_pdf


def fingerprint(path):
    """SHA-256 of the file's bytes plus every setting that changes the stored chunks.

    Including the settings means that changing CHUNK_SIZE, CHUNK_OVERLAP or
    EMBED_MODEL re-indexes every file, even though no PDF changed.
    (Changes to the loader/chunker *code* are not detected: delete chroma_db/
    or run with changed settings to force a full re-index.)
    """
    h = hashlib.sha256(path.read_bytes())
    h.update(f"|{config.CHUNK_SIZE}|{config.CHUNK_OVERLAP}|{config.EMBED_MODEL}".encode())
    return h.hexdigest()


def main():
    pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
    collection = store.get_collection()
    indexed = store.indexed_files(collection)
    print(f"Collection '{config.COLLECTION_NAME}': {collection.count()} chunks "
          f"from {len(indexed)} files before indexing\n")

    total_start = time.perf_counter()
    counts = {"unchanged": 0, "new": 0, "changed": 0, "removed": 0}

    # 1. Files that disappeared from data/: remove their chunks.
    for source in sorted(indexed.keys() - {p.name for p in pdfs}):
        store.delete_file(collection, source)
        counts["removed"] += 1
        print(f"  removed    {source}")

    # 2. New or changed files: re-embed. Unchanged files: skip.
    for path in pdfs:
        fp = fingerprint(path)
        if indexed.get(path.name) == fp:
            counts["unchanged"] += 1
            print(f"  unchanged  {path.name}")
            continue

        status = "new" if path.name not in indexed else "changed"
        start = time.perf_counter()
        chunks = chunk_pages(load_pdf(path), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        store.delete_file(collection, path.name)
        store.add_chunks(collection, chunks, fp)
        counts[status] += 1
        print(f"  {status:9}  {path.name}: {len(chunks)} chunks embedded "
              f"({time.perf_counter() - start:.1f}s)")

    summary = ", ".join(f"{n} {k}" for k, n in counts.items())
    print(f"\nDone in {time.perf_counter() - total_start:.1f}s ({summary}). "
          f"Collection now holds {collection.count()} chunks.")
    if not pdfs:
        print(f"Note: no PDFs found in {config.DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
