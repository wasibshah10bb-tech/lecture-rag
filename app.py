"""Milestone 9: Streamlit UI for the lecture-notes Q&A.

Run:  streamlit run app.py
      (opens http://localhost:8501 in your browser)

Streamlit re-runs this whole script from top to bottom on every interaction
(e.g. submitting the form). That's why the Chroma connection is cached.
"""

import streamlit as st

import config
from rag import store
from rag.generator import NOT_FOUND, answer, citation


@st.cache_resource
def get_collection():
    """Open Chroma once per server process instead of on every re-run."""
    return store.get_collection()


st.set_page_config(page_title="Lecture Notes Q&A", page_icon="📚")
st.title("📚 Lecture Notes Q&A")

collection = get_collection()
if collection.count() == 0:
    st.error("The index is empty. Run `python index.py` first.")
    st.stop()

st.caption(
    f"{collection.count()} chunks indexed · LLM `{config.LLM_MODEL}` · "
    f"embeddings `{config.EMBED_MODEL}` · top {config.TOP_K} · "
    f"threshold {config.SIMILARITY_THRESHOLD}"
)

# A form so the question is only sent when you press the button (or Enter),
# not on every keystroke.
with st.form("ask"):
    question = st.text_input("Your question", placeholder="e.g. What is a trap state?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Thinking... (a 7B model on CPU can take a minute)"):
        reply, hits = answer(question.strip(), collection=collection)

    # Answer, and which layer refused if it was a refusal (see generator.answer).
    st.subheader("Answer")
    if NOT_FOUND.rstrip(".").lower() in reply.lower():
        st.warning(reply)
        if hits[0]["score"] < config.SIMILARITY_THRESHOLD:
            st.caption(f"Refused by the similarity threshold: the best chunk scored "
                       f"{hits[0]['score']:.3f} < {config.SIMILARITY_THRESHOLD}, "
                       "so the LLM was not called.")
        else:
            st.caption("Refused by the LLM: retrieval found chunks above the threshold, "
                       "but the model judged that they don't answer the question.")
    else:
        st.markdown(reply)

    # Every chunk the model was given, so citations can be checked.
    st.subheader("Sources given to the model")
    for rank, h in enumerate(hits, start=1):
        with st.expander(f"#{rank} · score {h['score']:.3f} · {citation(h)}"):
            st.text(h["text"])
