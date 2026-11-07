"""Milestone 3: show how a real PDF is chunked: sizes, overlaps, statistics.

Run:  python inspect_chunks.py data/tcs-01-finite-automata.pdf
      python inspect_chunks.py data/tcs-01-finite-automata.pdf 300 80   (try other size/overlap)
      (no PDF argument = first PDF in data/; no numbers = values from config.py)
"""

import sys
from pathlib import Path

import config
from rag.chunker import chunk_pages
from rag.loader import load_pdf


def shared_text(prev, curr):
    """Longest end of `prev` that is also the start of `curr`: the actual overlap."""
    for n in range(min(len(prev), len(curr)), 0, -1):
        if prev.endswith(curr[:n]):
            return curr[:n]
    return ""


def one_line(text, width=70):
    """Shorten text to one line for display."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def main():
    args = sys.argv[1:]
    path = Path(args[0]) if args else sorted(config.DATA_DIR.glob("*.pdf"))[0]
    size = int(args[1]) if len(args) > 1 else config.CHUNK_SIZE
    overlap = int(args[2]) if len(args) > 2 else config.CHUNK_OVERLAP

    pages = load_pdf(path)
    chunks = chunk_pages(pages, size, overlap)
    print(f"\n{path.name}: chunk_size={size}, overlap={overlap}\n")

    prev = None
    for c in chunks:
        print(f"[{c['source']} p.{c['page']} #{c['chunk_index']}] {len(c['text'])} chars")
        print(f"    starts: {one_line(c['text'])}")
        # Overlap only exists between chunks of the same page.
        if prev and prev["page"] == c["page"]:
            shared = shared_text(prev["text"], c["text"])
            print(f"    overlap with previous ({len(shared)} chars): \"{one_line(shared)}\"")
        prev = c

    lengths = [len(c["text"]) for c in chunks]
    split_pages = len({c["page"] for c in chunks if c["chunk_index"] > 0})
    print(f"\n{len(pages)} pages -> {len(chunks)} chunks "
          f"({split_pages} pages were split into more than one chunk)")
    print(f"chunk length: min {min(lengths)}, avg {sum(lengths) // len(lengths)}, max {max(lengths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
