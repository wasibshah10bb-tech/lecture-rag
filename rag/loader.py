"""Step 1 of the pipeline: Load.

Turn a PDF into a list of pages, each carrying the metadata we need for
citations later: which file it came from and which page it is.
"""

import logging
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

# pypdf logs a warning for every small structural flaw in a PDF file
# ("Ignoring wrong pointing object ..."). They don't affect the text, so hide them.
logging.getLogger("pypdf").setLevel(logging.ERROR)

# A line counts as a header/footer if it appears on at least this share of a PDF's pages.
REPEATED_LINE_SHARE = 0.5


def _line_key(line):
    """Normalize a line so that 'Slides 3' and 'Slides 12' count as the same line."""
    return re.sub(r"\d+", "#", line.strip())


def remove_repeated_lines(texts):
    """Remove header/footer lines that repeat on most pages of one document.

    Slide decks usually repeat something like 'Course – Topic  <page>  Lecturer' on
    every slide. It carries no meaning but would end up in every chunk and make
    unrelated chunks look similar to the embedding model.
    Returns (cleaned_texts, removed_keys).
    """
    if len(texts) < 3:  # too few pages to tell what "repeats" means
        return texts, []

    # Count on how many pages each normalized line occurs (once per page).
    counts = Counter()
    for text in texts:
        counts.update({_line_key(line) for line in text.splitlines() if line.strip()})

    min_pages = REPEATED_LINE_SHARE * len(texts)
    repeated = {key for key, n in counts.items() if n >= min_pages}

    cleaned = [
        "\n".join(line for line in text.splitlines() if _line_key(line) not in repeated).strip()
        for text in texts
    ]
    return cleaned, sorted(repeated)


def load_pdf(path):
    """Return [{'text': ..., 'source': 'file.pdf', 'page': 1}, ...] for one PDF.

    - 'page' is the physical page in the PDF, counted from 1 (what a PDF viewer shows),
      not a slide number printed on the page.
    - Repeated header/footer lines are removed (see remove_repeated_lines).
    - Pages with no extractable text (e.g. scanned images, blank slides) are skipped
      with a warning, since there is nothing to search in them. We do no OCR.
    """
    path = Path(path)
    reader = PdfReader(path)
    raw = [(pdf_page.extract_text() or "") for pdf_page in reader.pages]

    texts, removed = remove_repeated_lines(raw)
    for key in removed:
        print(f"info: {path.name}: removed repeated line '{key}'")

    pages = []
    for i, text in enumerate(texts, start=1):
        if not text:
            print(f"warning: {path.name} p. {i} has no extractable text, skipped")
            continue
        pages.append({"text": text, "source": path.name, "page": i})
    return pages
