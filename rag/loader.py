"""Step 1 of the pipeline: Load.

Turn a file into a list of text units, each carrying the metadata we need for
citations later:
- PDF:       one unit per page             -> {'text', 'source', 'page'}
- Markdown:  one unit per heading section  -> {'text', 'source', 'line'}
- TXT:       one unit for the whole file   -> {'text', 'source', 'line'}
'line' is the 1-based line number where the unit's text starts.
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


# A Markdown heading: 1-6 '#' followed by a space, e.g. "## Pumping Lemma".
HEADING_RE = re.compile(r"^#{1,6}\s")


def _read_text(path):
    """Read a text file as UTF-8 (a BOM is ignored), falling back to Windows-1252."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        print(f"warning: {path.name} is not valid UTF-8, reading it as Windows-1252")
        return path.read_text(encoding="cp1252")


def load_text(path):
    """Return [{'text', 'source', 'line'}, ...] for a .md or .txt file.

    Markdown is split at headings, so a section and its heading stay together
    and no unit mixes two topics. '#' lines inside ``` code fences are not
    headings (e.g. shell comments). TXT has no structure: it is one unit.
    """
    path = Path(path)
    lines = _read_text(path).splitlines()

    # Indexes of the lines where a new unit starts.
    starts = [0]
    if path.suffix.lower() == ".md":
        in_fence = False
        for i, line in enumerate(lines):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            elif not in_fence and i > 0 and HEADING_RE.match(line):
                starts.append(i)

    units = []
    for start, end in zip(starts, starts[1:] + [len(lines)]):
        block = lines[start:end]
        # Skip leading blank lines so 'line' points at the first real text.
        while block and not block[0].strip():
            block = block[1:]
            start += 1
        text = "\n".join(block).strip()
        if text:
            units.append({"text": text, "source": path.name, "line": start + 1})

    if not units:
        print(f"warning: {path.name} contains no text, skipped")
    return units


LOADERS = {".pdf": load_pdf, ".md": load_text, ".txt": load_text}
SUPPORTED_SUFFIXES = set(LOADERS)


def load_file(path):
    """Load any supported file by picking the loader from its extension."""
    path = Path(path)
    return LOADERS[path.suffix.lower()](path)
