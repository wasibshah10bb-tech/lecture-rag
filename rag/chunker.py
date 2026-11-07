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


def chunk_pages(pages, chunk_size, overlap):
    """Turn loader pages into chunks that keep the page's metadata.

    Returns [{'text', 'source', 'page', 'chunk_index'}, ...] where chunk_index
    counts from 0 within each page.
    """
    chunks = []
    for page in pages:
        texts = [t for t in split_text(page["text"], chunk_size, overlap) if t]  # drop empty pieces
        for i, text in enumerate(texts):
            chunks.append({
                "text": text,
                "source": page["source"],
                "page": page["page"],
                "chunk_index": i,
            })
    return chunks
