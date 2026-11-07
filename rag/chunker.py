"""Step 2 of the pipeline: Chunk.

Split page text into overlapping pieces small enough to embed and to paste
into the LLM prompt. Each chunk stays inside one page, so its citation
(file + page) is always exact.
"""


def _last_whitespace(text, lo, hi):
    """Index of the last space/newline in text[lo:hi], or -1."""
    return max(text.rfind(" ", lo, hi), text.rfind("\n", lo, hi))


def _first_whitespace(text, lo, hi):
    """Index of the first space/newline in text[lo:hi], or -1."""
    hits = [i for i in (text.find(" ", lo, hi), text.find("\n", lo, hi)) if i != -1]
    return min(hits) if hits else -1


def split_text(text, chunk_size, overlap):
    """Split text into chunks of at most chunk_size characters.

    Consecutive chunks share about `overlap` characters, so a sentence cut at a
    chunk border still appears whole in one of the two chunks.
    Cuts are moved to whitespace so no word is split in half; this makes chunks
    slightly shorter and overlaps slightly smaller than the configured values.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while True:
        end = start + chunk_size
        if end >= len(text):  # the rest fits into one chunk
            chunks.append(text[start:].strip())
            return chunks

        # Move the end back to the last whitespace, but not so far back that the
        # next chunk would start before this one (that would loop forever).
        # If there is no whitespace in that window (overlap almost as big as
        # chunk_size, or one huge "word"), we fall back to a hard cut at chunk_size.
        cut = _last_whitespace(text, start + overlap + 1, end)
        if cut != -1:
            end = cut
        chunks.append(text[start:end].strip())

        # The next chunk starts `overlap` characters before this one ended.
        # If that lands mid-word, move forward to the next word start. The search
        # includes position `end` itself, so in the worst case we start right
        # after this chunk (no overlap) rather than in the middle of a word.
        start = end - overlap
        if not text[start - 1].isspace():
            space = _first_whitespace(text, start, end + 1)
            if space != -1:
                start = space + 1


def chunk_pages(units, chunk_size, overlap):
    """Turn loader units (PDF pages or text sections) into chunks that keep their metadata.

    Returns [{'text', 'source', 'page' or 'line', 'chunk_index'}, ...] where
    chunk_index counts from 0 within each unit.
    - PDF chunks inherit the page number.
    - Text chunks get the exact line they start on: every chunk is a substring
      of the unit's text, so we find where it starts and count the newlines before it.
    """
    chunks = []
    for unit in units:
        texts = [t for t in split_text(unit["text"], chunk_size, overlap) if t]  # drop empty pieces
        search_from = 0
        for i, text in enumerate(texts):
            chunk = {"text": text, "source": unit["source"], "chunk_index": i}
            if "page" in unit:
                chunk["page"] = unit["page"]
            else:
                pos = unit["text"].find(text, search_from)
                chunk["line"] = unit["line"] + unit["text"].count("\n", 0, pos)
                search_from = pos + 1  # the next chunk starts after this one's start
            chunks.append(chunk)
    return chunks
