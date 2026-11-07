"""Step 6 of the pipeline: Generate.

The LLM never sees your files. It only sees the few chunks we retrieved,
pasted into the prompt. Everything it is allowed to know is in that prompt.
"""

import config
from rag import citations, ollama_client
from rag.citations import citation  # the label format; re-exported for ask.py, app.py, ...
from rag.retriever import retrieve

NOT_FOUND = "I can't find this in your notes."

# The rules go into the system message, which models treat as higher priority
# than the user message.
# Lessons from eval_questions.py, kept in the wording below:
# - A concrete example label (e.g. "[lecture.pdf, p. 3]") got copied literally as a
#   citation, so the format is described without a fake file name.
# - Slides are terse bullets ("Tibor Radó 1962: Busy Beaver"); without the note-form
#   hint the model refused questions whose answer was in the excerpt.
# - For a .txt excerpt the model once cited just "[notes.txt]", so the rule now
#   says explicitly that the page/line number is part of the label.
SYSTEM_PROMPT = f"""You answer questions about a student's lecture notes.

Rules:
- Use ONLY the information in the context excerpts below. Do not use outside knowledge.
- The excerpts are lecture slides written as short bullet points. If they contain the answer, even in note form, answer the question.
- After every statement, cite the excerpt it comes from by copying the label in square brackets that appears above that excerpt, character for character, including the page or line number.
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Be concise. Answer in the language of the question."""


def is_refusal(reply):
    """True if the reply is (or contains) the "not in your notes" sentence."""
    return NOT_FOUND.rstrip(".").lower() in reply.lower()


def build_messages(question, hits):
    """Put the retrieved chunks and the question into chat messages."""
    context = "\n\n".join(f"{citation(h)}\n{h['text']}" for h in hits)
    user = f"Context excerpts:\n\n{context}\n\n---\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def answer(question, collection=None):
    """Retrieve, prompt the LLM, check citations; return (answer_text, hits, citation_report).

    Two layers of "I don't know":
    1. Threshold gate (here): if even the best chunk is a weak match, refuse
       without calling the LLM at all: fast, and nothing to hallucinate from.
    2. The prompt rule: for questions that pass the gate but still aren't
       answered by the excerpts, the LLM is told to reply NOT_FOUND.
    Then every citation in a real answer is checked against the excerpts the
    model was given (see rag/citations.py). For refusals the report is None.
    """
    hits = retrieve(question, collection=collection)
    if hits[0]["score"] < config.SIMILARITY_THRESHOLD:
        return NOT_FOUND, hits, None
    messages = build_messages(question, hits)
    reply = ollama_client.chat(messages, options={"temperature": config.LLM_TEMPERATURE}).strip()
    if is_refusal(reply):
        return reply, hits, None
    checked, report = citations.check(reply, hits)
    return checked, hits, report
