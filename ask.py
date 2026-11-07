"""Milestone 6: ask a question, get an answer from your notes with citations.

Run:  python ask.py
      python ask.py --show-prompt     (also print the exact prompt sent to the LLM)
"""

import sys
import time

from rag import store
from rag.generator import answer, build_messages, citation


def main():
    show_prompt = "--show-prompt" in sys.argv
    collection = store.get_collection()
    if collection.count() == 0:
        print("The index is empty. Run: python index.py")
        return 1
    print("Ask about your lecture notes. Empty line or 'quit' to exit.")

    while True:
        try:
            question = input("\nQuestion> ").strip()
        except EOFError:
            break
        if question.lower() in ("", "quit", "exit"):
            break

        print("Thinking... (a 7B model on CPU can take a minute)")
        start = time.perf_counter()
        reply, hits = answer(question, collection=collection)

        if show_prompt:
            for m in build_messages(question, hits):
                print(f"\n----- {m['role']} -----\n{m['content']}")
            print("----- end of prompt -----")

        print(f"\n{reply}\n")
        print(f"Sources given to the model ({time.perf_counter() - start:.0f}s):")
        for h in hits:
            print(f"  {h['score']:.3f}  {citation(h)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
