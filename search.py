"""Milestone 5: retrieval-only CLI. Type a question, see the top-k chunks.

Run:  python index.py     (once, if chroma_db/ is empty)
      python search.py
"""

import sys

import config
from rag import store
from rag.retriever import retrieve

PREVIEW_CHARS = 300


def print_hits(hits):
    for rank, h in enumerate(hits, start=1):
        preview = " ".join(h["text"].split())  # collapse newlines for display
        if len(preview) > PREVIEW_CHARS:
            preview = preview[:PREVIEW_CHARS] + "..."
        print(f"\n#{rank}  score {h['score']:.3f}  [{h['source']}, p. {h['page']}]")
        print(f"    {preview}")


def main():
    collection = store.get_collection()
    if collection.count() == 0:
        print("The index is empty. Run: python index.py")
        return 1
    print(f"{collection.count()} chunks indexed. Showing top {config.TOP_K}. "
          "Empty line or 'quit' to exit.")

    while True:
        try:
            question = input("\nQuestion> ").strip()
        except EOFError:  # e.g. Ctrl+Z on Windows, or end of piped input
            break
        if question.lower() in ("", "quit", "exit"):
            break
        print_hits(retrieve(question, collection=collection))
    return 0


if __name__ == "__main__":
    sys.exit(main())
