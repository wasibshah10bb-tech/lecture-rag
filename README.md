# Lecture Notes Q&A: a fully local RAG app, built from scratch

Ask questions about your own lecture slides and notes and get answers **only from that material**, with citations like `[tcs-03-formal-languages.pdf, p. 43]`. If the notes don't cover the question, the app says *"I can't find this in your notes."* instead of guessing.

Everything runs **locally and offline**: no cloud APIs, no API keys. The LLM and the embedding model run through [Ollama](https://ollama.com).

```
Question> What is the pumping lemma used for?

The pumping lemma can be used to show that a language is not regular or context-free
[tcs-03-formal-languages.pdf, p. 43]. However, it cannot be used to show that a language
is regular or context-free [tcs-03-formal-languages.pdf, p. 43].

Question> How do I bake sourdough bread?

I can't find this in your notes.
```

## Why I built it

I'm a computer science student and wanted to understand how Retrieval-Augmented Generation (RAG) actually works, not just call a framework that hides it. So the rule was **no LangChain, no LlamaIndex**. Every pipeline step is plain Python that I can read, run on its own, and inspect. I built it in small milestones, each one runnable and committed before the next. I used my *Theoretical Computer Science* lecture slides (5 PDFs, 200 pages) as the test material. They are not included in this repo.

## How it works

```
 PDF / MD / TXT ──► load ──► chunk ──► embed ──► Chroma (on disk)
                                                     │
 question ──► embed ──► top-k similar chunks ◄───────┘
                              │
                 best score < threshold? ──► "I can't find this in your notes."
                              │ no
                              ▼
              prompt (rules + chunks + question) ──► LLM ──► answer with [file, p. X]
                                                              │
                                                  citation check (valid / repaired / flagged)
```

| Step | File | What it does |
|---|---|---|
| Load | `rag/loader.py` | Extracts text per PDF page (keeps file name + page number). Removes repeated slide footers. Splits Markdown at headings. Reads TXT as one unit. |
| Chunk | `rag/chunker.py` | Splits text into overlapping ~800-character chunks, cutting only at whitespace. Chunks never cross pages, so citations are always exact. |
| Embed + store | `rag/store.py` | Embeds chunks with `nomic-embed-text` via Ollama and stores text, vector and metadata in a persistent ChromaDB collection (cosine distance). |
| Index | `index.py` | Incremental: a SHA-256 fingerprint (file contents + chunk settings) per file, so only new or changed files get re-embedded. Deleted files are removed. |
| Retrieve | `rag/retriever.py` | Embeds the question and returns the top-k chunks with a similarity score (1 − cosine distance). |
| Generate | `rag/generator.py` | Builds the prompt, calls `qwen2.5:7b`, and refuses early if retrieval is too weak. |
| Check citations | `rag/citations.py` | Verifies every citation points to an excerpt the model was actually given. Repairs file-name typos and flags everything else. |

## What I learned (the interesting part)

- **Look at your extracted text before anything else.** Every slide had the same footer line (`TCS – … J. Schmidt …`), which would have been in every chunk and made unrelated chunks look similar. A generic "line repeats on ≥ 50 % of pages" filter removed exactly that line from all 5 PDFs and nothing else.
- **A GPU backend can silently corrupt output.** Answers came back with doubled tokens (`with with`, `p. 233`, `.pdf.pdf`). Testing the raw Ollama API showed that this wasn't my code or the model. It was Ollama's Vulkan backend on a 2 GB laptop GPU. Forcing CPU-only (`num_gpu: 0`) fixed it and was *faster*. I also had to re-index, because the embeddings had been affected too.
- **Similarity scores don't cleanly separate "answerable" from "not in my notes".** On my test set, answerable questions scored 0.578–0.835 and off-topic ones 0.505–0.688. They overlap: *"How does Dijkstra's algorithm work?"* scored higher than two questions my slides do answer. So the threshold is only a coarse first filter (set just below the lowest answerable score), and the LLM's own refusal rule catches the near-misses.
- **LLMs copy prompt examples literally.** With an example label `[lecture.pdf, p. 3]` in the prompt, the model cited a non-existent `lecture.pdf`. I fixed it by describing the format without a fake file name.
- **Too strict a prompt causes false refusals.** Slides are terse (`Tibor Radó 1962: Busy Beaver`), so the model refused to answer *"Who introduced the busy beaver function?"* until the prompt told it that note-form answers count.
- **Temperature 0 is not fully deterministic on CPU**, so my tests check *behaviour* (answered vs. refused, right page cited), not exact wording.
- **A citation is not a proof.** In one run the model attributed an NFA property to DFAs and cited the correct slide anyway. Citation validation catches citations that point nowhere, but not claims the cited source doesn't support.

## Evaluation

`test_questions.json` holds 18 hand-written questions:
- 12 that the slides answer, each with the page(s) that must be cited. One of them is in German.
- 6 that must be refused: 2 clearly off-topic, plus 4 *near-misses* (Dijkstra, quicksort, TCP, B-trees) that sound like CS lecture topics but aren't in these slides.

```
python eval_questions.py --scores-only   # retrieval scores + threshold sweep (seconds)
python eval_questions.py                 # full run with the LLM (~15–25 min on CPU)
```

Result: **18/18 passed**. 1 refusal came from the threshold, 5 from the LLM, and every answer cited an expected page. The test set is small, and I wrote the questions and tuned the threshold on the same set, so this is an optimistic number, not a benchmark.

## Setup (Windows, Python 3.13)

1. Install [Ollama](https://ollama.com) and pull the models:
   ```powershell
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
2. Create a virtual environment and install the dependencies:
   ```powershell
   py -V:3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Check that Ollama and both models respond:
   ```powershell
   python check_ollama.py
   ```
4. Put your PDFs, `.md` or `.txt` files into a `data/` folder and build the index:
   ```powershell
   python index.py
   ```

All settings (models, chunk size, top-k, threshold, GPU use) live in `config.py`.

## Usage

```powershell
python ask.py              # Q&A in the terminal (--show-prompt prints the exact prompt)
streamlit run app.py       # web UI on http://localhost:8501 (bound to localhost only)
python search.py           # retrieval only: top-k chunks with scores, no LLM
python inspect_pdf.py data\lecture.pdf      # see what text extraction produces
python inspect_chunks.py data\lecture.pdf   # see chunk sizes and overlaps
```

On a laptop CPU, `qwen2.5:7b` takes roughly 15–60 s per answer. `qwen3:4b-instruct` is faster if you change `LLM_MODEL` in `config.py`.

## Tech stack

Python 3.13 · Ollama (`qwen2.5:7b`, `nomic-embed-text`) · ChromaDB · pypdf · Streamlit. Ollama is called over its REST API with the standard library (`urllib`), so there's no client library in between.

## Limitations

- No OCR: scanned PDFs without a text layer are skipped.
- Diagrams don't survive text extraction, so questions about them retrieve poorly.
- Each question is answered on its own, with no chat memory, so follow-ups like "and why?" don't work.
- Changes to the loader or chunker *code* aren't detected by incremental indexing. Delete `chroma_db/` to rebuild.
