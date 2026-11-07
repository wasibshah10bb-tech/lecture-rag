"""Steps 3 + 4 of the pipeline: Embed and Store.

Chroma is the vector database: it keeps each chunk's text, its embedding
(a list of 768 floats from nomic-embed-text) and its metadata (source, page)
in a folder on disk, and can later find the chunks closest to a query vector.
We never let Chroma embed anything itself: all vectors come from Ollama.
"""

import chromadb

import config
from rag import ollama_client

# nomic-embed-text was trained with task prefixes: documents get "search_document: ",
# questions get "search_query: ". Leaving them out noticeably hurts retrieval.
DOCUMENT_PREFIX = "search_document: "


def get_collection():
    """Open (or create) the persistent collection in config.CHROMA_DIR."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        # Cosine distance = 1 - cosine similarity, so similarity = 1 - distance (0..1 in practice).
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,  # we always pass our own embeddings
    )


def chunk_id(chunk):
    """A stable, readable ID like 'tcs-01-finite-automata.pdf::p3::c0'."""
    return f"{chunk['source']}::p{chunk['page']}::c{chunk['chunk_index']}"


def embed_documents(texts):
    """Embed chunk texts in batches (one HTTP request per batch, not per chunk)."""
    vectors = []
    for i in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = [DOCUMENT_PREFIX + t for t in texts[i : i + config.EMBED_BATCH_SIZE]]
        vectors.extend(ollama_client.embed(batch))
    return vectors


def delete_file(collection, source):
    """Remove every chunk that came from this file name."""
    collection.delete(where={"source": source})


def indexed_files(collection):
    """Return {source: fingerprint} for every file that has chunks in the collection.

    All chunks of a file carry the same fingerprint, so reading any of them is enough.
    Chunks indexed before fingerprints existed get None, so they count as changed.
    """
    result = collection.get(include=["metadatas"])
    return {m["source"]: m.get("fingerprint") for m in result["metadatas"]}


def add_chunks(collection, chunks, fingerprint):
    """Embed chunks and store them with their text and metadata.

    The fingerprint identifies the file version (and settings) these chunks came
    from; index.py compares it on the next run to decide what to re-embed.
    """
    texts = [c["text"] for c in chunks]
    collection.upsert(
        ids=[chunk_id(c) for c in chunks],
        embeddings=embed_documents(texts),
        documents=texts,
        metadatas=[
            {"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"],
             "fingerprint": fingerprint}
            for c in chunks
        ],
    )
