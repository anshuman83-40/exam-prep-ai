"""AI Exam Prep Assistant — chat with your notes. Run with: streamlit run app.py"""

import os
import re

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag import (EMBED_MODEL, PROJECT_DIR, RERANK_MODEL, KnowledgeBase, answer, extract_questions,
                 find_topics, make_chunks, make_quiz, read_pages)

load_dotenv(PROJECT_DIR / ".env")
st.set_page_config(page_title="AI Exam Prep Assistant", page_icon="📚", layout="centered")

st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; }
  .hero h1 { font-size: 2rem; margin: 0; padding: 0; }
  .hero p  { opacity: .75; margin: .25rem 0 0 0; }
  div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is missing. Copy .env.example to .env and add your key.")
    st.stop()

PAGES = ["💬 Ask", "📝 Quiz", "🎯 Important topics"]
SUGGESTIONS = ["Explain PEAS with an example", "What are the types of agents?",
               "Difference between BFS and DFS", "What is knowledge representation?"]


@st.cache_resource(show_spinner="Loading AI models (first start takes ~1 minute)...")
def load_models() -> tuple[SentenceTransformer, CrossEncoder]:
    return SentenceTransformer(EMBED_MODEL), CrossEncoder(RERANK_MODEL)


if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase(*load_models())
    st.session_state.files = {}  # file name -> {"pages": n, "chunks": n}
    st.session_state.docs = {}  # file name -> (pdf bytes, page texts), for topic analysis
    st.session_state.messages = []
    st.session_state.page = PAGES[0]
kb: KnowledgeBase = st.session_state.kb


def add_pdf(name: str, data: bytes):
    """Read a PDF (OCR if needed), chunk it and add it to the knowledge base."""
    if name in st.session_state.files:
        return
    bar = st.progress(0.0, text=f"Reading {name}...")

    def progress(done, total):
        bar.progress(done / total, text=f"Reading handwritten page {done}/{total} of {name}...")

    pages = read_pages(data, progress)
    chunks = make_chunks(pages, name)
    kb.add(chunks)
    st.session_state.files[name] = {"pages": len(pages), "chunks": len(chunks)}
    st.session_state.docs[name] = (data, pages)
    st.session_state.pop("topics", None)  # new file -> topic analysis is out of date
    bar.empty()


# ---------- helpers ----------

def plain(text: str) -> str:
    """Escape markdown symbols (#, *, $ ...) so note text isn't shown as headings or maths."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>$~<])", r"\\\1", text)


def with_page_badges(reply: str, hits) -> str:
    """Turn the LLM's [1], [2] citations into page badges like  p. 5 ."""
    def badge(m):
        n = int(m.group(1))
        return f" :violet-badge[p. {hits[n - 1][0].page}]" if 1 <= n <= len(hits) else ""
    return re.sub(r"\s?\[(\d+)\]", badge, reply)


def maths(text: str) -> str:
    """Show LaTeX the AI writes (e.g. \\sqrt{x^2}) as a formula instead of raw code."""
    if "$" not in text and re.search(r"\\[a-zA-Z]+|\^|_\{", text):
        return f"${text}$"
    return text


def render_sources(hits):
    pages = sorted({c.page for c, _ in hits})
    with st.expander(f"📖 Sources — page {', '.join(map(str, pages))}"):
        for c, score in hits:
            st.markdown(f":violet-badge[p. {c.page}] **{c.source}** · match {score:.0%}")
            st.caption(plain(c.text[:350]) + ("..." if len(c.text) > 350 else ""))


def go(page: str, **state):
    """Button callback: switch section and set some state (e.g. a question to ask)."""
    st.session_state.page = page
    st.session_state.update(state)


def ask(question: str):
    st.session_state.pending_question = question


# ---------- sidebar ----------

with st.sidebar:
    st.markdown("### 📚 Exam Prep AI")
    st.caption("Your notes → answers, quizzes and exam priorities.")

    # PDFs already in the project folder are loaded automatically
    for pdf in sorted(PROJECT_DIR.glob("*.pdf")):
        add_pdf(pdf.name, pdf.read_bytes())
    for f in st.file_uploader("Add notes or past papers (PDF)", type="pdf",
                              accept_multiple_files=True) or []:
        add_pdf(f.name, f.getvalue())

    st.markdown("**Loaded**")
    for name, info in st.session_state.files.items():
        with st.container(border=True):
            st.markdown(f"📄 **{name}**")
            st.caption(f"{info['pages']} pages · {info['chunks']} searchable chunks")

    st.divider()
    show_sources = st.toggle("Show sources under answers", value=True)
    if st.button("🧹 Clear chat", width="stretch", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

    with st.expander("⚙️ How it works"):
        st.markdown(
            "1. **OCR** — Gemini vision reads handwritten pages\n"
            "2. **Chunking** — text split into overlapping pieces\n"
            "3. **Hybrid search** — embeddings (meaning) + BM25 (keywords)\n"
            "4. **Re-ranking** — cross-encoder picks the best 5\n"
            "5. **Grounded answer** — Gemini answers only from those, with page citations\n\n"
            "Measured on 34 test questions: right page ranked **#1 for 53%** and in the "
            "**top 5 for 85%** (basic RAG: 41% / 71%); **100%** of off-topic questions refused."
        )


# ---------- header + navigation ----------

total_pages = sum(f["pages"] for f in st.session_state.files.values())
st.markdown(
    f"<div class='hero'><h1>📚 AI Exam Prep Assistant</h1>"
    f"<p>{len(st.session_state.files)} file(s) · {total_pages} pages · "
    f"{len(kb.chunks)} chunks — answers come only from your notes, with page numbers.</p></div>",
    unsafe_allow_html=True,
)
page = st.segmented_control("Section", PAGES, key="page", label_visibility="collapsed")
page = page or PAGES[0]  # clicking the selected button again deselects it
st.write("")


# ---------- 💬 Ask: chat with your notes ----------

if page == PAGES[0]:
    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("#### 👋 Ask anything from your notes")
            st.caption("Answers are written only from your PDFs and every claim shows its page. "
                       "If it's not in your notes, the assistant will say so.")
            st.pills("Try one:", SUGGESTIONS, key="suggestion",
                     on_change=lambda: (ask(st.session_state.suggestion),
                                        st.session_state.update(suggestion=None)))

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "📚"):
            st.markdown(msg["content"])
            if msg.get("hits") and show_sources:
                render_sources(msg["hits"])

    typed = st.chat_input("Ask a question about your notes...")
    question = typed or st.session_state.pop("pending_question", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="📚"):
            with st.spinner("Searching your notes and writing an answer..."):
                try:
                    reply, hits = answer(question, kb)
                    reply = with_page_badges(reply, hits)
                except Exception as e:
                    reply, hits = f"⚠️ Something went wrong: {e}", []
            st.markdown(reply)
            if hits and show_sources:
                render_sources(hits)
        st.session_state.messages.append({"role": "assistant", "content": reply, "hits": hits})


# ---------- 📝 Quiz: MCQs generated from your notes ----------

def new_quiz(topic: str, n: int, level: str):
    with st.spinner(f"Writing and fact-checking {n} questions on '{topic}'..."):
        try:
            st.session_state.quiz = make_quiz(topic, kb, n, level)
        except Exception as e:
            st.error(f"⚠️ Couldn't make a quiz: {e}")
            return
    st.session_state.quiz_topic = topic
    st.session_state.quiz_settings = (topic, n, level)
    st.session_state.quiz_checked = False
    st.session_state.quiz_picks = []
    st.session_state.quiz_round = st.session_state.get("quiz_round", 0) + 1  # fresh widgets


if page == PAGES[1]:
    with st.container(border=True):
        st.markdown("#### 📝 Practice quiz")
        st.caption("MCQs written from your notes, then double-checked against them by a second AI pass.")
        with st.form("quiz_settings_form", border=False):
            topic = st.text_input("Topic", key="quiz_topic_input",
                                  placeholder="e.g. types of agents, BFS, PEAS, A* search")
            c1, c2 = st.columns(2)
            n = c1.slider("Questions", 3, 10, 5)
            level = c2.segmented_control("Difficulty", ["easy", "medium", "hard"],
                                         default="medium") or "medium"
            make = st.form_submit_button("✨ Generate quiz", type="primary", width="stretch")

    auto_topic = st.session_state.pop("auto_quiz_topic", None)  # from "Quiz me" on Topics
    if make and not topic.strip():
        st.warning("Type a topic first.")
    elif make or auto_topic:
        new_quiz(topic.strip() if make else auto_topic, n if make else 5, level)
        if st.session_state.get("quiz") == []:
            st.warning("Couldn't make questions on that topic from your notes. Try another one.")

    quiz = st.session_state.get("quiz")
    if quiz:
        checked = st.session_state.quiz_checked
        answers = st.session_state.get("quiz_picks", [])
        rnd = st.session_state.quiz_round

        if checked:
            score = sum(a == q["answer_index"] for a, q in zip(answers, quiz))
            pct = score / len(quiz)
            with st.container(border=True):
                c1, c2 = st.columns([1, 2], vertical_alignment="center")
                c1.metric("Your score", f"{score} / {len(quiz)}")
                c2.progress(pct, text=("🏆 Perfect!" if pct == 1 else "👍 Good job!"
                                       if pct >= 0.6 else "📖 Keep revising!") + f"  {pct:.0%}")
                wrong_pages = sorted({q["page"] for a, q in zip(answers, quiz)
                                      if a != q["answer_index"]})
                if wrong_pages:
                    c2.caption(f"Revise page(s) {', '.join(map(str, wrong_pages))} of your notes.")
            if pct == 1:
                st.balloons()

        st.markdown(f"##### Quiz: {st.session_state.quiz_topic}")
        with st.form("quiz_answers", border=False):
            picks = []
            for i, q in enumerate(quiz):
                with st.container(border=True):
                    picks.append(st.radio(
                        f"**Q{i + 1}.** {q['question']}", range(4),
                        index=answers[i] if checked else None,
                        format_func=lambda j, q=q: maths(q["options"][j]),
                        key=f"q{i}-{rnd}-{'done' if checked else 'open'}", disabled=checked))
                    if checked:
                        right = maths(q["options"][q["answer_index"]])
                        if answers[i] == q["answer_index"]:
                            st.success(f"Correct! {q['explanation']}", icon="✅")
                        elif answers[i] is None:
                            st.warning(f"Skipped. Answer: **{right}**. {q['explanation']}", icon="⏭️")
                        else:
                            st.error(f"Answer: **{right}**. {q['explanation']}", icon="❌")
                        st.caption(f"📖 {q['source']}, page {q['page']}")
            submitted = st.form_submit_button("Check answers", type="primary",
                                              disabled=checked, width="stretch")

        if submitted:
            st.session_state.quiz_picks = picks
            st.session_state.quiz_checked = True
            st.rerun()

        if checked:
            c1, c2 = st.columns(2)
            if c1.button("🔁 Retry same questions", width="stretch"):
                st.session_state.quiz_checked = False
                st.session_state.quiz_picks = []
                st.session_state.quiz_round += 1
                st.rerun()
            if c2.button("✨ New quiz on this topic", width="stretch"):
                new_quiz(*st.session_state.quiz_settings)
                st.rerun()


# ---------- 🎯 Important topics: most-asked exam questions ----------

if page == PAGES[2]:
    with st.container(border=True):
        st.markdown("#### 🎯 What should I study first?")
        st.caption("Finds every exam question in your notes and past papers, groups similar ones "
                   "with clustering, and ranks topics by how often they're asked + their marks. "
                   "Add past papers in the sidebar for better results.")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        grouping = c1.select_slider(
            "Grouping", options=[0.2, 0.25, 0.3, 0.35, 0.4], value=0.3,
            format_func=lambda v: {0.2: "Very fine", 0.25: "Fine", 0.3: "Balanced",
                                   0.35: "Broad", 0.4: "Very broad"}[v],
            help="How similar questions must be to count as the same topic.")
        analyze = c2.button("🔍 Analyze", type="primary", width="stretch")

    if analyze:
        try:
            with st.spinner("Finding exam questions in your notes..."):
                questions = []
                for name, (data, pages) in st.session_state.docs.items():
                    questions += extract_questions(data, pages, name)
            with st.spinner(f"Grouping {len(questions)} questions into topics..."):
                st.session_state.topics = find_topics(questions, kb.embedder, grouping)
                st.session_state.topic_count = len(questions)
        except Exception as e:
            st.error(f"⚠️ Analysis failed: {e}")

    topics = st.session_state.get("topics")
    if topics == []:
        st.warning("No exam questions found. Add past papers or notes with questions like "
                   "'Q. ... (4 marks)'.")
    elif topics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Questions found", st.session_state.topic_count)
        m2.metric("Topics", len(topics))
        m3.metric("Marks covered", sum(t["total_marks"] for t in topics))
        st.success(f"Start with **{topics[0]['topic']}** — asked {topics[0]['times_asked']}× "
                   f"for {topics[0]['total_marks']} marks.", icon="🥇")

        df = pd.DataFrame([{"Topic": t["topic"], "Times asked": t["times_asked"],
                            "Total marks": t["total_marks"], "Importance": t["score"]}
                           for t in topics[:10]])
        chart = alt.Chart(df).mark_bar(cornerRadiusEnd=6, color="#7c6cf2").encode(
            x=alt.X("Importance:Q", title="Importance (times asked + marks ÷ 2)"),
            y=alt.Y("Topic:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=["Topic", "Times asked", "Total marks"],
        ).properties(height=34 * len(df))
        st.altair_chart(chart, width="stretch")

        st.markdown("##### Study in this order")
        for rank, t in enumerate(topics, start=1):
            with st.container(border=True):
                c1, c2 = st.columns([4, 2], vertical_alignment="center")
                badges = f":blue-badge[asked {t['times_asked']}×]" + (
                    f" :orange-badge[{t['total_marks']} marks]" if t["total_marks"] else "")
                c1.markdown(f"**{rank}. {t['topic']}**  \n{badges}")
                b1, b2 = c2.columns(2)
                b1.button("📝 Quiz", key=f"tq{rank}", width="stretch", help="Quiz me on this",
                          on_click=go, args=(PAGES[1],), kwargs={"auto_quiz_topic": t["topic"]})
                b2.button("💬 Explain", key=f"te{rank}", width="stretch",
                          help="Explain this from my notes", on_click=go, args=(PAGES[0],),
                          kwargs={"pending_question": f"Explain {t['topic']} for my exam"})
                with st.expander(f"{len(t['questions'])} question(s)", expanded=rank == 1):
                    for q in t["questions"]:
                        marks = f" :orange-badge[{q['marks']} marks]" if q["marks"] else ""
                        st.markdown(f"- {plain(q['question'])}{marks} "
                                    f":gray-badge[{q['source']} · p. {q['page']}]")
