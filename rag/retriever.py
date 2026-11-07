"""Step 5 of the pipeline: Retrieve.

Embed the question with the same model as the chunks, then let Chroma find
the chunks whose vectors point in the most similar direction.
"""

import config
from rag import ollama_client, store

# Counterpart of store.DOCUMENT_PREFIX: nomic-embed-text expects questions
# to be marked as search queries.
QUERY_PREFIX = "search_query: "


def retrieve(question, k=None, collection=None):
    """Return the k most similar chunks, best first:
    [{'text', 'source', 'page', 'line', 'chunk_index', 'score'}, ...]
    ('page' is set for PDFs, 'line' for Markdown/TXT; the other one is None.)

    score = cosine similarity = 1 - cosine distance. 1.0 would mean "same
    direction"; unrelated text scores clearly lower. Absolute values depend on
    the embedding model, so look at real scores before choosing a threshold.
    """
    collection = collection or store.get_collection()
    query_vector = ollama_client.embed([QUERY_PREFIX + question])[0]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k or config.TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma returns one list per query vector; we sent one, so take index [0].
    hits = []
    for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({
            "text": text,
            "source": meta["source"],
            "page": meta.get("page"),
            "line": meta.get("line"),
            "chunk_index": meta["chunk_index"],
            "score": 1 - distance,
        })
    return hits
