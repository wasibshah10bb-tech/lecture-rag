"""A thin wrapper around Ollama's REST API, using only the standard library.

Ollama is just an HTTP server: every call below is one JSON request and one
JSON response. Keeping it this bare makes it obvious what goes over the wire.
API reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import urllib.request

import config


def _request(path, payload=None):
    """Send a GET (no payload) or POST (with JSON payload) and return the parsed JSON."""
    url = config.OLLAMA_URL + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)  # POST if data is set
    with urllib.request.urlopen(req, timeout=config.REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_models():
    """Return the names of all locally installed models, e.g. ['qwen2.5:7b', ...]."""
    return [m["name"] for m in _request("/api/tags")["models"]]


def _options(extra=None):
    """Model options sent with every request (see config.NUM_GPU_LAYERS)."""
    options = {}
    if config.NUM_GPU_LAYERS is not None:
        options["num_gpu"] = config.NUM_GPU_LAYERS
    options.update(extra or {})
    return options


def embed(texts, model=None):
    """Turn a list of strings into a list of vectors (one list of floats per string)."""
    payload = {"model": model or config.EMBED_MODEL, "input": texts, "options": _options()}
    return _request("/api/embed", payload)["embeddings"]


def chat(messages, model=None, options=None):
    """Send chat messages ([{'role': 'user', 'content': '...'}, ...]) and return the reply text.

    stream=False makes Ollama return the whole answer in one response
    instead of a token-by-token stream.
    options are model settings such as {'temperature': 0}.
    """
    payload = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": _options(options),
    }
    return _request("/api/chat", payload)["message"]["content"]
