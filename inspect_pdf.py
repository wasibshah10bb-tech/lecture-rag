"""Milestone 2: print one PDF page by page, with its metadata.

Run:  python inspect_pdf.py data/tcs-01-finite-automata.pdf
      (no argument = first PDF in data/)
"""

import sys
from pathlib import Path

import config
from rag.loader import load_pdf


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {config.DATA_DIR}")
            return 1
        path = pdfs[0]

    pages = load_pdf(path)
    for p in pages:
        print(f"===== source: {p['source']} | page: {p['page']} | {len(p['text'])} chars =====")
        print(p["text"])
        print()

    total = sum(len(p["text"]) for p in pages)
    print(f"Loaded {len(pages)} pages with text from {path.name}, {total} characters in total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
