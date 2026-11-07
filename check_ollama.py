"""Milestone 1: check that Ollama is running and both configured models respond.

Run:  python check_ollama.py
Exit code 0 = all good, 1 = something failed.
"""

import sys
import time
import urllib.error

import config
from rag import ollama_client


def ok(msg):
    print(f"✓ {msg}")


def fail(msg, hint=None):
    print(f"✗ {msg}")
    if hint:
        print(f"  → {hint}")


def main():
    # 1. Is the server up?
    try:
        installed = ollama_client.list_models()
    except (urllib.error.URLError, ConnectionError) as e:
        fail(f"Cannot reach Ollama at {config.OLLAMA_URL} ({e})",
             "Start Ollama (open the Ollama app or run `ollama serve`).")
        return 1
    ok(f"Ollama server reachable at {config.OLLAMA_URL}")

    # 2. Are both models installed? Ollama names may carry a ':latest' tag,
    #    so 'nomic-embed-text' should match 'nomic-embed-text:latest'.
    all_present = True
    for model in (config.LLM_MODEL, config.EMBED_MODEL):
        if model in installed or f"{model}:latest" in installed:
            ok(f"Model '{model}' is installed")
        else:
            fail(f"Model '{model}' is not installed", f"Run: ollama pull {model}")
            all_present = False
    if not all_present:
        return 1

    # 3. Does the embedding model return a vector?
    try:
        start = time.perf_counter()
        vectors = ollama_client.embed(["hello world"])
        ok(f"Embedding OK: vector length {len(vectors[0])} ({time.perf_counter() - start:.1f}s)")
    except Exception as e:
        fail(f"Embedding failed: {e}")
        return 1

    # 4. Does the LLM answer? (First call is slow: the model is loaded into memory.)
    try:
        start = time.perf_counter()
        reply = ollama_client.chat([{"role": "user", "content": "Reply with exactly one word: OK"}])
        ok(f'LLM OK: replied "{reply.strip()}" ({time.perf_counter() - start:.1f}s)')
    except Exception as e:
        fail(f"LLM call failed: {e}")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
