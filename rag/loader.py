"""Step 1 of the pipeline: Load.

Turn a PDF into a list of pages, each carrying the metadata we need for
citations later: which file it came from and which page it is.
"""

import logging
from pathlib import Path

from pypdf import PdfReader

# pypdf logs a warning for every small structural flaw in a PDF file
# ("Ignoring wrong pointing object ..."). They don't affect the text, so hide them.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def load_pdf(path):
    """Return [{'text': ..., 'source': 'file.pdf', 'page': 1}, ...] for one PDF.

    - 'page' is the physical page in the PDF, counted from 1 (what a PDF viewer shows),
      not a slide number printed on the page.
    - Pages with no extractable text (e.g. scanned images, blank slides) are skipped
      with a warning, since there is nothing to search in them. We do no OCR.
    """
    path = Path(path)
    reader = PdfReader(path)
    pages = []
    for i, pdf_page in enumerate(reader.pages, start=1):
        text = (pdf_page.extract_text() or "").strip()
        if not text:
            print(f"warning: {path.name} p. {i} has no extractable text, skipped")
            continue
        pages.append({"text": text, "source": path.name, "page": i})
    return pages
