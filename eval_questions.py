"""Milestone 7: run the hand-written test questions and check the app's behaviour.

Run:  python eval_questions.py --scores-only   (retrieval only, fast: use it to choose the threshold)
      python eval_questions.py                 (full run with the LLM, several minutes on CPU)

We check behaviour, not exact wording (the LLM's wording varies between runs):
- "answer" questions must NOT be refused and must cite at least one expected page.
- "refuse" questions must be refused (by the threshold gate or by the LLM).
"""

import json
import re
import sys

import config
from rag import store
from rag.generator import NOT_FOUND, answer, citation
from rag.retriever import retrieve

# Matches citations like [tcs-01-finite-automata.pdf, p. 17]
CITATION_RE = re.compile(r"\[([^\[\]]+?\.pdf),\s*p\.\s*(\d+)\]")


def load_questions():
    with open(config.PROJECT_DIR / "test_questions.json", encoding="utf-8") as f:
        return json.load(f)


def scores_only(questions, collection):
    """Print each question's best retrieval score, then how well each threshold separates them."""
    rows = []
    for q in questions:
        best = retrieve(q["question"], k=1, collection=collection)[0]
        rows.append((q["expect"], best["score"], q["question"], citation(best)))

    for expect, score, question, cite in sorted(rows, key=lambda r: -r[1]):
        print(f"{score:.3f}  {expect:6}  {question[:55]:55}  {cite}")

    # For each candidate threshold: how many questions would the gate classify correctly?
    # (Only the gate: the LLM can still refuse questions above the threshold.)
    print("\nthreshold  correct  (answer kept / refuse blocked)")
    n_answer = sum(1 for r in rows if r[0] == "answer")
    n_refuse = len(rows) - n_answer
    for t in [x / 100 for x in range(50, 76, 2)]:
        kept = sum(1 for e, s, *_ in rows if e == "answer" and s >= t)
        blocked = sum(1 for e, s, *_ in rows if e == "refuse" and s < t)
        print(f"  {t:.2f}     {kept + blocked:2}/{len(rows)}   ({kept}/{n_answer} / {blocked}/{n_refuse})")


def full_run(questions, collection):
    passed = 0
    for i, q in enumerate(questions, start=1):
        reply, hits = answer(q["question"], collection=collection)
        refused = NOT_FOUND.rstrip(".").lower() in reply.lower()
        gated = hits[0]["score"] < config.SIMILARITY_THRESHOLD
        cited = {f"{src}, p. {page}" for src, page in CITATION_RE.findall(reply)}
        retrieved = {citation(h)[1:-1] for h in hits}  # labels the model actually saw

        if q["expect"] == "refuse":
            ok = refused
            detail = "refused by threshold" if gated else ("refused by LLM" if refused else "ANSWERED")
        else:
            hit = cited & set(q["sources"])
            ok = not refused and bool(hit)
            if refused:
                detail = "REFUSED by threshold" if gated else "REFUSED by LLM"
            else:
                detail = f"cited {sorted(hit)}" if hit else f"WRONG SOURCES, cited {sorted(cited) or 'nothing'}"

        invented = cited - retrieved  # citations of chunks the model was never shown
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  [{i:2}] best {hits[0]['score']:.3f}  {q['question']}")
        print(f"            {detail}")
        if invented:
            print(f"            warning: cited sources it was not given: {sorted(invented)}")
        if not ok:
            print(f"            reply: {' '.join(reply.split())[:300]}")

    print(f"\n{passed}/{len(questions)} passed (threshold {config.SIMILARITY_THRESHOLD})")
    return 0 if passed == len(questions) else 1


def main():
    collection = store.get_collection()
    questions = load_questions()
    if "--scores-only" in sys.argv:
        scores_only(questions, collection)
        return 0
    return full_run(questions, collection)


if __name__ == "__main__":
    sys.exit(main())
