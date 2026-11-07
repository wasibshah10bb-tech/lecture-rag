"""Step 6 of the pipeline: Generate.

The LLM never sees your files. It only sees the few chunks we retrieved,
pasted into the prompt. Everything it is allowed to know is in that prompt.
"""

import config
from rag import ollama_client
from rag.retriever import retrieve

NOT_FOUND = "I can't find this in your notes."

# The rules go into the system message, which models treat as higher priority
# than the user message.
SYSTEM_PROMPT = f"""You answer questions about a student's lecture notes.

Rules:
- Use ONLY the information in the context excerpts below. Do not use outside knowledge.
- After every statement, cite the excerpt it comes from, copying its label exactly, e.g. [lecture.pdf, p. 3].
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Be concise. Answer in the language of the question."""


def citation(hit):
    """The label a chunk gets in the prompt, which is also the citation format."""
    return f"[{hit['source']}, p. {hit['page']}]"


def build_messages(question, hits):
    """Put the retrieved chunks and the question into chat messages."""
    context = "\n\n".join(f"{citation(h)}\n{h['text']}" for h in hits)
    user = f"Context excerpts:\n\n{context}\n\n---\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def answer(question, collection=None):
    """Retrieve, prompt the LLM, and return (answer_text, hits)."""
    hits = retrieve(question, collection=collection)
    messages = build_messages(question, hits)
    reply = ollama_client.chat(messages, options={"temperature": config.LLM_TEMPERATURE})
    return reply.strip(), hits
