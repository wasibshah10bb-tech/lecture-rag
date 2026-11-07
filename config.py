"""All settings for the app live here, so there is exactly one place to change them."""

from pathlib import Path

# --- Ollama ---
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"          # model that writes the answers
EMBED_MODEL = "nomic-embed-text"  # model that turns text into vectors
REQUEST_TIMEOUT = 300             # seconds; generous because a 7B model on CPU is slow

# --- Chunking (in characters) ---
# Most slides are 300-550 characters, so 800 keeps a typical slide in one chunk
# and only splits the densest ones.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150   # characters shared by consecutive chunks of the same page

# --- Embedding & storage ---
EMBED_BATCH_SIZE = 16           # chunks sent to Ollama per /api/embed request
COLLECTION_NAME = "lecture_notes"

# --- Retrieval ---
TOP_K = 5                       # how many chunks to retrieve per question

# --- Paths ---
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"         # put lecture PDFs here
CHROMA_DIR = PROJECT_DIR / "chroma_db"  # Chroma stores its files here (created in Milestone 4)
