import hashlib

import streamlit as st

from llm import answer_question, generate_quiz_cards, summarize_notes
from rag import build_index, chunk_pages, extract_pdf_pages, search

MIN_CHARS = 50
RESULT_KEYS = ("ready", "summary", "chunks", "index", "messages", "quiz")

st.set_page_config(page_title="Smart Study Buddy", page_icon="🎓", layout="wide")
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold !important;
        width: 100% !important;
        height: 3em !important;
        border-radius: 10px !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa_pipe = pipeline("text2text-generation", model="google/flan-t5-large")
    return embedder, qa_pipe


def fingerprint(pages, filename=""):
    digest = hashlib.sha256()
    digest.update((filename or "").encode())
    for page in pages:
        digest.update(str(page.get("page")).encode())
        digest.update(page["text"].encode())
    return digest.hexdigest()


def clear_results():
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)


def source_label(hit):
    return f"Page {hit['page']}" if hit.get("page") else "Notes"


def render_sources(hits):
    with st.expander("Sources"):
        for hit in hits:
            st.caption(f"{source_label(hit)} · similarity {hit['score']:.2f}")
            st.write(hit["text"])


with st.sidebar:
    st.header("About")
    st.info("Built by J Bahulika")
    st.header("Connect")
    st.link_button("LinkedIn", "https://www.linkedin.com/in/j-bahulika-8b8237207/")
    st.link_button("GitHub", "https://github.com/JBahulika")

st.title("🎓 Smart Study Buddy")
st.caption("Summarize notes, ask questions with sources, and quiz yourself.")

input_method = st.radio("Input selection", ["Paste text", "Upload PDF"], horizontal=True)
pages = []
filename = ""

if input_method == "Paste text":
    pasted = st.text_area("Paste your study material here:", height=250)
    if pasted.strip():
        pages = [{"page": None, "text": pasted.strip()}]
else:
    uploaded = st.file_uploader("Upload notes (PDF)", type="pdf")
    if uploaded:
        filename = uploaded.name
        pages = extract_pdf_pages(uploaded)
        if pages:
            st.caption(f"{uploaded.name} · {len(pages)} page(s) with extractable text")
        else:
            st.warning("Could not extract text. This PDF may be scanned or image-only.")
    else:
        st.info("Upload a PDF to get started.")

char_count = sum(len(page["text"]) for page in pages)
source_key = fingerprint(pages, filename) if pages else ""
if st.session_state.get("source_key") != source_key:
    clear_results()
    st.session_state["source_key"] = source_key

can_generate = char_count >= MIN_CHARS
if pages and not can_generate:
    st.caption(f"Add at least {MIN_CHARS} characters before generating.")

if st.button("Generate study pack", disabled=not can_generate):
    embedder, qa_pipe = load_models()
    chunks = chunk_pages(pages)
    if not chunks:
        st.error("Not enough extractable text to study.")
    else:
        progress = st.progress(0.0, text="Indexing notes...")
        index = build_index(embedder, chunks)

        def on_progress(done, total, label):
            progress.progress(min(done / total, 1.0), text=label)

        summary = summarize_notes(qa_pipe, chunks, on_progress=on_progress)
        progress.empty()
        st.session_state.ready = True
        st.session_state.summary = summary
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.messages = []
        st.session_state.quiz = None
        st.rerun()

st.divider()

if st.session_state.get("ready"):
    chunk_count = len(st.session_state["chunks"])
    st.caption(f"Indexed {chunk_count} chunk(s) from your notes.")
    summary_tab, ask_tab, quiz_tab = st.tabs(["Summary", "Ask questions", "Quiz"])

    with summary_tab:
        st.subheader("Study summary")
        with st.container(border=True):
            st.write(st.session_state["summary"])
        st.download_button(
            "Download summary",
            data=st.session_state["summary"],
            file_name="study-summary.txt",
            mime="text/plain",
        )

    with ask_tab:
        st.subheader("Ask about your notes")
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and message.get("sources"):
                    render_sources(message["sources"])

        query = st.chat_input("Ask a specific question about your notes")
        if query:
            embedder, qa_pipe = load_models()
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("Searching your notes..."):
                    hits = search(
                        embedder,
                        st.session_state["index"],
                        st.session_state["chunks"],
                        query,
                    )
                    answer = answer_question(qa_pipe, query, hits)
                st.write(answer)
                if hits:
                    render_sources(hits)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": hits}
            )

    with quiz_tab:
        st.subheader("Quiz yourself")
        if st.button("Generate quiz"):
            _, qa_pipe = load_models()
            progress = st.progress(0.0, text="Writing quiz questions...")

            def on_quiz_progress(done, total, label):
                progress.progress(min(done / total, 1.0), text=label)

            st.session_state.quiz = generate_quiz_cards(
                qa_pipe,
                st.session_state["chunks"],
                on_progress=on_quiz_progress,
            )
            progress.empty()
            st.rerun()

        quiz = st.session_state.get("quiz")
        if quiz:
            for i, card in enumerate(quiz, start=1):
                title = f"Q{i}. {card['question']}"
                with st.expander(title):
                    st.write(card["answer"])
                    if card.get("page"):
                        st.caption(f"Source: page {card['page']}")
        else:
            st.info("Generate a short quiz from your notes when you are ready to review.")
else:
    st.write("### How to use")
    st.write("1. Paste notes or upload a PDF.")
    st.write("2. Click **Generate study pack** for a summary of the whole document.")
    st.write("3. Ask questions (with sources) or generate a quiz to review.")
