"""All settings for the app live here, so there is exactly one place to change them."""

from pathlib import Path

# --- Ollama ---
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"          # model that writes the answers
EMBED_MODEL = "nomic-embed-text"  # model that turns text into vectors
REQUEST_TIMEOUT = 300             # seconds; generous because a 7B model on CPU is slow

# --- Paths ---
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"         # put lecture PDFs here
CHROMA_DIR = PROJECT_DIR / "chroma_db"  # Chroma stores its files here (created in Milestone 4)
