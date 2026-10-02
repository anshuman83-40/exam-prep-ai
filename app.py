"""AI Exam Prep Assistant — chat with your notes. Run with: streamlit run app.py"""

import html
import os
import re
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

import progress as prog
from rag import (EMBED_MODEL, PROJECT_DIR, RERANK_MODEL, KnowledgeBase, answer, extract_questions,
                 extract_syllabus, find_topics, make_chunks, make_quiz, read_pages, study_plan)

load_dotenv(PROJECT_DIR / ".env")
st.set_page_config(page_title="AI Exam Prep Assistant", page_icon="📚", layout="wide")

# Dark purple "glass" theme. Streamlit's own colours are set in .streamlit/config.toml;
# this adds the gradient background, glass cards, pill buttons and the dashboard widgets.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], button, input, textarea { font-family: 'Exo 2', sans-serif; }
/* keep Streamlit's icon font for icons (otherwise they show as words like "arrow_right") */
span[data-testid="stIconMaterial"], [class*="material-symbols"], .material-icons {
  font-family: 'Material Symbols Rounded' !important; }
.stApp {
  background:
    radial-gradient(1200px 600px at 85% -10%, rgba(124,58,237,.35), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(91,33,182,.35), transparent 60%),
    #0b0715;
}
.block-container { max-width: 1240px; padding-top: 1.4rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }

/* glass cards = every bordered container */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]),
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(160deg, rgba(255,255,255,.065), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.09) !important;
  border-radius: 22px !important;
  box-shadow: 0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);
  backdrop-filter: blur(14px);
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #120b22, #0d0819);
  border-right: 1px solid rgba(255,255,255,.06);
}

/* buttons: purple primary, white "Continue"-style pills for the rest */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {
  border-radius: 999px !important; font-weight: 600;
}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primaryFormSubmit"] {
  background: linear-gradient(135deg, #8b5cf6, #6d28d9); border: none;
  box-shadow: 0 6px 18px rgba(124,58,237,.45);
}
.stButton button[kind="secondary"] {
  background: rgba(255,255,255,.92); color: #1b1030; border: none;
}
.stButton button[kind="secondary"]:hover { background: #fff; color: #5b21b6; }
.stButton button[kind="tertiary"] { color: #c4b5fd; }

/* section switcher looks like the app's top tabs */
div[data-testid="stButtonGroup"] button { border-radius: 999px !important; }

/* text inputs / chat input */
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-testid="stChatInput"] > div {
  border-radius: 999px !important; background: rgba(255,255,255,.06) !important;
}
div[data-testid="stChatInput"] textarea { min-height: 0 !important; }

/* dashboard pieces */
.topbar-title { font-size: 2rem !important; font-weight: 700; line-height: 1.2; letter-spacing: .3px; }
.chip { display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .75rem;
        border-radius:999px; background:rgba(255,255,255,.06);
        border:1px solid rgba(255,255,255,.08); font-size:.85rem; }
.avatar { width:30px; height:30px; border-radius:50%; display:inline-grid; place-items:center;
          background:linear-gradient(135deg,#f9a8d4,#a78bfa); font-weight:700; color:#1b1030; }
.stat-big { font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
.stat-sub { font-size: .78rem; opacity: .65; }
.up { color: #4ade80; font-size: .72rem; } .down { color: #f87171; font-size: .72rem; }
.ring { --p: 0; width: 112px; height: 112px; border-radius: 50%; margin: auto;
        background: conic-gradient(#a78bfa calc(var(--p) * 1%), rgba(255,255,255,.08) 0);
        display: grid; place-items: center; box-shadow: 0 0 30px rgba(139,92,246,.35); }
.ring::before { content: ""; width: 84px; height: 84px; border-radius: 50%; background: #150d26;
                grid-area: 1 / 1; }
.ring span { grid-area: 1 / 1; font-size: 1.35rem; font-weight: 700; z-index: 1; }
.orb { width: 120px; height: 120px; border-radius: 50%; margin: .6rem auto 1rem;
       background: radial-gradient(circle at 32% 28%, #f0e7ff 0%, #b18cff 18%, #6d28d9 45%,
                   #2e0f6b 72%, #12052c 100%);
       box-shadow: 0 0 45px rgba(139,92,246,.75), inset -12px -16px 30px rgba(0,0,0,.45);
       animation: float 5s ease-in-out infinite; }
@keyframes float { 0%,100% { transform: translateY(0) } 50% { transform: translateY(-8px) } }
.hello { font-size: 1.5rem; font-weight: 700; margin: 0; }
.hello-emoji { font-size: 4.2rem; text-align: right; line-height: 1; }
.lesson-title { font-weight: 600; margin-bottom: .25rem; }
.bar { height: 6px; border-radius: 999px; background: rgba(255,255,255,.12); overflow: hidden; }
.bar > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#c4b5fd,#fff); }
.slot { border-left: 2px solid rgba(167,139,250,.6); padding: .35rem .7rem; margin: .45rem 0;
        border-radius: 0 12px 12px 0; background: rgba(139,92,246,.10); }
.slot small { opacity: .7; }

/* phones: tighter spacing, smaller headings (Streamlit stacks columns below ~640px) */
@media (max-width: 640px) {
  .block-container { padding: .8rem .7rem 3rem; }
  .topbar-title { font-size: 1.5rem !important; }
  .hello { font-size: 1.25rem; } .hello-emoji { font-size: 3rem; }
  .stat-big { font-size: 1.4rem; }
  div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px !important; }
  /* rows INSIDE a card (Study|Quiz buttons, stat pairs) stay side by side on phones */
  div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important; gap: .5rem !important; }
  div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    min-width: 0 !important; flex: 1 1 0 !important; width: auto !important; }
}
</style>
""", unsafe_allow_html=True)

if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is missing. Copy .env.example to .env and add your key.")
    st.stop()

HOME, ASK, QUIZ, TOPICS, PLAN = ("🏠 Overview", "💬 Ask", "📝 Quiz", "🎯 Topics", "🗺️ Study plan")
PAGES = [HOME, ASK, QUIZ, TOPICS, PLAN]
PRIORITY_ICON = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}
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
    st.session_state.page = HOME
kb: KnowledgeBase = st.session_state.kb
progress = prog.load()  # quiz history, units done, exam date... (progress.json)


def read_with_progress(name: str, data: bytes) -> list[str]:
    """read_pages() with a progress bar while scanned pages are OCR'd."""
    bar = st.progress(0.0, text=f"Reading {name}...")

    def progress(done, total):
        bar.progress(done / total, text=f"Reading scanned page {done}/{total} of {name}...")

    pages = read_pages(data, progress)
    bar.empty()
    return pages


def add_pdf(name: str, data: bytes):
    """Read a PDF (OCR if needed), chunk it and add it to the knowledge base."""
    if name in st.session_state.files:
        return
    pages = read_with_progress(name, data)
    chunks = make_chunks(pages, name)
    kb.add(chunks)
    st.session_state.files[name] = {"pages": len(pages), "chunks": len(chunks)}
    st.session_state.docs[name] = (data, pages)
    st.session_state.pop("topics", None)  # new file -> topic analysis is out of date


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

    with st.expander("👤 Profile"):
        name = st.text_input("Your name", progress["name"])
        if name.strip() and name.strip() != progress["name"]:
            progress["name"] = name.strip()
            prog.save(progress)

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

def search_notes():
    """Top-bar search box: ask the typed text in the Ask section."""
    text = st.session_state.top_search.strip()
    st.session_state.top_search = ""
    if text:
        go(ASK, pending_question=text)


page = st.session_state.get("page") or HOME  # clicking the selected button again deselects it
total_pages = sum(f["pages"] for f in st.session_state.files.values())

with st.container(border=True):
    c1, c2, c3 = st.columns([1.3, 2, 1.2], vertical_alignment="center")
    c1.markdown(f"<div class='topbar-title'>{page.split(' ', 1)[1]}</div>", unsafe_allow_html=True)
    c2.text_input("Search", key="top_search", placeholder="🔍  Search your notes...",
                  label_visibility="collapsed", on_change=search_notes)
    initial = html.escape(progress["name"][:1].upper() or "S")
    c3.markdown(
        f"<div style='text-align:right'><span class='chip'>📄 {total_pages} pages</span> "
        f"<span class='chip'><span class='avatar'>{initial}</span>"
        f"<span><b>{html.escape(progress['name'])}</b><br><small>AI course</small></span></span></div>",
        unsafe_allow_html=True)
    st.segmented_control("Section", PAGES, key="page", label_visibility="collapsed")
st.write("")


# ---------- 🏠 Overview: dashboard ----------

def ask_from_dashboard():
    text = st.session_state.dash_ask.strip()
    st.session_state.dash_ask = ""
    if text:
        go(ASK, pending_question=text)


def set_exam_date():
    d = st.session_state.exam_date_input
    progress["exam_date"] = d.isoformat() if d else None
    prog.save(progress)


if page == HOME:
    plan_rows = progress["plan"]
    left, mid, right = st.columns([1.05, 1, 1.05], gap="medium")

    # ----- left: study schedule (the "calendar") -----
    with left, st.container(border=True):
        st.markdown("#### 📅 Study schedule")
        today = date.today()
        exam = date.fromisoformat(progress["exam_date"]) if progress["exam_date"] else None
        st.date_input("Exam date", value=exam, min_value=today + timedelta(days=1),
                      key="exam_date_input", on_change=set_exam_date, format="DD/MM/YYYY")
        if not plan_rows:
            st.caption("Build a study plan from your syllabus + PYQs and your days will be "
                       "planned here, most important units first.")
            st.button("🗺️ Build study plan", width="stretch", type="primary",
                      on_click=go, args=(PLAN,))
        elif not exam or exam <= today:
            st.caption("Set your exam date to spread your units over the days left.")
        else:
            sched = prog.schedule(plan_rows, exam, today)
            st.markdown(f"<span class='chip'>⏳ {(exam - today).days} days to exam</span>",
                        unsafe_allow_html=True)
            days = sorted(sched)[:6]
            labels = {d: d.strftime("%a %d") for d in days}
            picked = st.segmented_control("Day", days, format_func=labels.get, default=days[0],
                                          key="sched_day", label_visibility="collapsed")
            for unit in sched.get(picked or days[0], []):
                if unit.get("revision"):
                    st.markdown("<div class='slot'>🔁 <b>Revision + PYQ practice</b><br>"
                                "<small>Go through every High-priority unit again</small></div>",
                                unsafe_allow_html=True)
                    continue
                done = unit["title"] in progress["units_done"]
                st.markdown(
                    f"<div class='slot'>{'✅' if done else PRIORITY_ICON[unit['priority']][:1]} "
                    f"<b>Unit {unit['unit']}: {html.escape(unit['title'])}</b><br>"
                    f"<small>{unit['weightage']:.0%} weightage · {unit['priority']} priority</small>"
                    f"</div>", unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                b1.button("Study", key=f"s_{picked}_{unit['unit']}", width="stretch", on_click=go,
                          args=(ASK,),
                          kwargs={"pending_question": f"Summarise {unit['title']} for my exam"})
                b2.button("Quiz", key=f"q_{picked}_{unit['unit']}", width="stretch",
                          on_click=go, args=(QUIZ,), kwargs={"auto_quiz_topic": unit["title"]})

    # ----- middle: stats, progress ring, recent quizzes -----
    with mid:
        acc = prog.accuracy(progress)
        with st.container(border=True):
            a, b = st.columns(2)
            a.markdown(f"<div style='text-align:center'><div class='stat-big'>{prog.points(progress)}"
                       f"</div><div class='stat-sub'>Quiz points</div>"
                       f"<div class='up'>{len(progress['quizzes'])} quizzes taken</div></div>",
                       unsafe_allow_html=True)
            b.markdown(f"<div style='text-align:center'><div class='stat-big'>"
                       f"{f'{acc:.0%}' if acc is not None else '—'}</div>"
                       f"<div class='stat-sub'>Accuracy</div>"
                       f"<div class='up'>{progress['asked']} questions asked</div></div>",
                       unsafe_allow_html=True)

        with st.container(border=True):
            total_units = len(plan_rows)
            done_units = sum(u["title"] in progress["units_done"] for u in plan_rows)
            pct = round(100 * done_units / total_units) if total_units else 0
            a, b = st.columns([1.1, 1], vertical_alignment="center")
            a.markdown(f"<div class='stat-big'>{done_units}/{total_units or '—'}</div>"
                       f"<div class='stat-sub'>Units completed</div>", unsafe_allow_html=True)
            b.markdown(f"<div class='ring' style='--p:{pct}'><span>{pct}%</span></div>",
                       unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Last quizzes")
            recent = progress["quizzes"][::-1][:3]
            if not recent:
                st.caption("No quizzes yet — test yourself on any topic.")
                st.button("📝 Start a quiz", width="stretch", on_click=go, args=(QUIZ,))
            for i, q in enumerate(recent):
                score = q["score"] / q["total"] if q["total"] else 0
                with st.container(border=True):
                    st.markdown(
                        f"<div class='lesson-title'>{html.escape(q['topic'])}</div>"
                        f"<div class='stat-sub'>{'🏆 Perfect!' if score == 1 else '👍 Good going' if score >= .6 else '📖 You can do better!'}"
                        f" · {q['score']}/{q['total']}</div>"
                        f"<div class='bar'><div style='width:{score:.0%}'></div></div>",
                        unsafe_allow_html=True)
                    a, b = st.columns([1, 1], vertical_alignment="center")
                    a.caption(prog.ago(q["at"]))
                    b.button("Continue", key=f"cont{i}", width="stretch", on_click=go,
                             args=(QUIZ,), kwargs={"auto_quiz_topic": q["topic"]})

    # ----- right: greeting + AI assistant -----
    with right:
        with st.container(border=True):
            a, b = st.columns([2, 1], vertical_alignment="center")
            next_up = next((u for u in plan_rows if u["title"] not in progress["units_done"]), None)
            a.markdown(f"<div class='hello'>Hi, {html.escape(progress['name'])}!</div>"
                       f"<div class='stat-sub' style='font-size:.95rem'>Ready to make progress today?"
                       + (f"<br>Next up: <b>{html.escape(next_up['title'])}</b>" if next_up else "")
                       + "</div>", unsafe_allow_html=True)
            b.markdown("<div class='hello-emoji'>🧑‍🎓</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### AI Assistant")
            st.markdown("<div class='orb'></div>", unsafe_allow_html=True)
            st.pills("Tips", ["Explain PEAS", "Types of agents", "BFS vs DFS"], key="dash_tip",
                     label_visibility="collapsed",
                     on_change=lambda: (go(ASK, pending_question=st.session_state.dash_tip),
                                        st.session_state.update(dash_tip=None)))
            st.text_input("Ask", key="dash_ask", placeholder="🎙️  Ask me about your notes...",
                          label_visibility="collapsed", on_change=ask_from_dashboard)
            st.caption(f"Answers only from your {len(st.session_state.files)} file(s) · "
                       f"{len(kb.chunks)} chunks, with page numbers.")


# ---------- 💬 Ask: chat with your notes ----------

if page == ASK:
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
                    progress["asked"] += 1
                    prog.save(progress)
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


if page == QUIZ:
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
            prog.record_quiz(progress, st.session_state.quiz_topic,
                             sum(a == q["answer_index"] for a, q in zip(picks, quiz)), len(quiz))
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

if page == TOPICS:
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
                          on_click=go, args=(QUIZ,), kwargs={"auto_quiz_topic": t["topic"]})
                b2.button("💬 Explain", key=f"te{rank}", width="stretch",
                          help="Explain this from my notes", on_click=go, args=(ASK,),
                          kwargs={"pending_question": f"Explain {t['topic']} for my exam"})
                with st.expander(f"{len(t['questions'])} question(s)", expanded=rank == 1):
                    for q in t["questions"]:
                        marks = f" :orange-badge[{q['marks']} marks]" if q["marks"] else ""
                        st.markdown(f"- {plain(q['question'])}{marks} "
                                    f":gray-badge[{q['source']} · p. {q['page']}]")


# ---------- 🗺️ Study plan: syllabus + PYQs -> which chapter first, with weightage ----------

if page == PLAN:
    st.session_state.setdefault("plan_syllabus", None)  # (file name, pdf bytes)
    st.session_state.setdefault("plan_pyqs", {})  # file name -> pdf bytes
    # PDFs in the syllabus/ and pyqs/ folders are picked up automatically
    for pdf in sorted((PROJECT_DIR / "syllabus").glob("*.pdf"))[:1]:
        if st.session_state.plan_syllabus is None:
            st.session_state.plan_syllabus = (pdf.name, pdf.read_bytes())
    for pdf in sorted((PROJECT_DIR / "pyqs").glob("*.pdf")):
        st.session_state.plan_pyqs.setdefault(pdf.name, pdf.read_bytes())

    with st.container(border=True):
        st.markdown("#### 🗺️ Which chapter should I start with?")
        st.caption("Upload your course handout/syllabus and previous year papers (PYQs). "
                   "Each PYQ question is matched to a syllabus unit to work out its weightage. "
                   "Scanned papers work too.")
        c1, c2 = st.columns(2)
        syl = c1.file_uploader("1️⃣ Course handout / syllabus", type="pdf", key="syl_up")
        pyqs = c2.file_uploader("2️⃣ Previous year papers (PYQs)", type="pdf",
                                accept_multiple_files=True, key="pyq_up")
        if syl:
            st.session_state.plan_syllabus = (syl.name, syl.getvalue())
        for f in pyqs or []:
            st.session_state.plan_pyqs[f.name] = f.getvalue()

        if st.session_state.plan_syllabus or st.session_state.plan_pyqs:
            chips = []
            if st.session_state.plan_syllabus:
                chips.append(f":violet-badge[📘 {st.session_state.plan_syllabus[0]}]")
            chips += [f":blue-badge[📄 {n}]" for n in st.session_state.plan_pyqs]
            st.markdown(" ".join(chips))
            if st.button("Clear files", key="plan_clear"):
                st.session_state.plan_syllabus, st.session_state.plan_pyqs = None, {}
                st.session_state.pop("plan", None)
                st.rerun()

        build = st.button("📊 Build my study plan", type="primary", width="stretch",
                          disabled=st.session_state.plan_syllabus is None)
        if st.session_state.plan_syllabus is None:
            st.caption("⬆️ Add a syllabus first. PYQs are optional but give real weightage.")

    if build:
        try:
            name, data = st.session_state.plan_syllabus
            with st.spinner("Reading the syllabus units..."):
                syllabus = extract_syllabus(data, read_with_progress(name, data))
            papers = {}
            n_papers = len(st.session_state.plan_pyqs)
            for i, (pname, pdata) in enumerate(st.session_state.plan_pyqs.items(), start=1):
                with st.spinner(f"Finding questions in paper {i}/{n_papers}..."):
                    papers[pname] = extract_questions(pdata, read_with_progress(pname, pdata), pname)
            with st.spinner("Matching every question to a syllabus unit..."):
                st.session_state.plan = study_plan(syllabus, papers)
            # remember the ranking for the Overview schedule and progress ring
            progress["plan"] = [{k: r[k] for k in ("unit", "title", "weightage", "priority", "order")}
                                for r in st.session_state.plan["rows"]]
            prog.save(progress)
        except Exception as e:
            st.error(f"⚠️ Couldn't build the plan: {e}")

    plan = st.session_state.get("plan")
    if plan and not plan["rows"]:
        st.warning("No units found in that file. Is it the course handout/syllabus?")
    elif plan:
        rows = plan["rows"]
        if plan["course"]:
            st.markdown(f"##### 📘 {plan['course']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Units", len(rows))
        m2.metric("PYQ papers", plan["papers"])
        m3.metric("Questions", plan["n_questions"])
        m4.metric("Marks analysed", plan["total_marks"])

        top = rows[0]
        if plan["basis"] == "pyq":
            st.success(f"Start with **Unit {top['unit']}: {top['title']}** — "
                       f"{top['weightage']:.0%} of the PYQ "
                       f"{'marks' if plan['use_marks'] else 'questions'}.", icon="🥇")
        elif plan["basis"] == "hours":
            st.info("No PYQs added, so units are ranked by **lecture hours**. "
                    "Add previous year papers for real exam weightage.", icon="ℹ️")
        else:
            st.info("No PYQs or lecture hours found, so units are in **syllabus order**. "
                    "Add previous year papers for real exam weightage.", icon="ℹ️")

        weight_label = ("PYQ weightage" if plan["basis"] == "pyq" else
                        "Lecture-hour share" if plan["basis"] == "hours" else "Weightage")
        table = pd.DataFrame([{
            "#": r["order"],
            "Unit": f"{r['unit']}. {r['title']}",
            "Priority": PRIORITY_ICON[r["priority"]],
            weight_label: r["weightage"] * 100,
            "PYQ marks": r["marks"],
            "Asked in": f"{r['asked_in']}/{plan['papers']}" if plan["papers"] else "—",
            "Hours": r["hours"] or None,
        } for r in rows])
        if plan["basis"] == "pyq":
            table = table.drop(columns="Hours")
        else:
            table = table.drop(columns=["PYQ marks", "Asked in"])
        st.dataframe(
            table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3,
            column_config={
                "#": st.column_config.NumberColumn(width=40),
                "Unit": st.column_config.TextColumn(width="medium"),
                "Priority": st.column_config.TextColumn(width=95),
                weight_label: st.column_config.ProgressColumn(
                    weight_label, format="%.0f%%", min_value=0, max_value=100, width=130),
                "PYQ marks": st.column_config.NumberColumn("Marks", width=60),
                "Asked in": st.column_config.TextColumn("Papers", width=60,
                                                        help="Asked in how many of the papers"),
                "Hours": st.column_config.NumberColumn(format="%d h", width=60),
            },
        )
        csv = table.assign(Why=[r["why"] for r in rows], Topics=[", ".join(r["topics"]) for r in rows])
        st.download_button("⬇️ Download table (CSV)", csv.to_csv(index=False).encode("utf-8-sig"),
                           "study_plan.csv", "text/csv")

        if plan["basis"] == "pyq":
            chart = alt.Chart(table).mark_arc(innerRadius=60).encode(
                theta=alt.Theta(f"{weight_label}:Q"),
                color=alt.Color("Unit:N", legend=alt.Legend(orient="right", labelLimit=260)),
                tooltip=["Unit", alt.Tooltip(f"{weight_label}:Q", format=".0f"), "PYQ marks"],
            ).properties(height=260, title="Share of PYQ marks by unit")
            st.altair_chart(chart, width="stretch")

        st.markdown("##### Unit by unit")
        for r in rows:
            with st.container(border=True):
                c1, c2 = st.columns([4, 2], vertical_alignment="center")
                c1.markdown(f"**#{r['order']} · Unit {r['unit']}: {r['title']}**  \n"
                            f"{PRIORITY_ICON[r['priority']]} · {r['why']}")
                b1, b2 = c2.columns(2)
                b1.button("📝 Quiz", key=f"pq{r['unit']}", width="stretch",
                          on_click=go, args=(QUIZ,), kwargs={"auto_quiz_topic": r["title"]})
                b2.button("💬 Explain", key=f"pe{r['unit']}", width="stretch", on_click=go,
                          args=(ASK,),
                          kwargs={"pending_question": f"Summarise {r['title']} for my exam"})
                done = st.checkbox("Mark as done", value=r["title"] in progress["units_done"],
                                   key=f"done{r['unit']}")
                if done != (r["title"] in progress["units_done"]):
                    if done:
                        progress["units_done"].append(r["title"])
                    else:
                        progress["units_done"].remove(r["title"])
                    prog.save(progress)
                with st.expander(f"Topics & {r['n_questions']} PYQ question(s)"):
                    st.markdown("**Topics:** " + ", ".join(plain(t) for t in r["topics"]))
                    for q in r["questions"]:
                        marks = f" :orange-badge[{q['marks']} marks]" if q["marks"] else ""
                        st.markdown(f"- {plain(q['question'])}{marks} :gray-badge[{q['paper']}]")

        if plan["unmapped"]:
            with st.expander(f"⚠️ {len(plan['unmapped'])} question(s) didn't match any unit"):
                for q in plan["unmapped"]:
                    st.markdown(f"- {plain(q['question'])} :gray-badge[{q['paper']}]")
