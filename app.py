"""AI Exam Prep Assistant — chat with your notes. Run with: streamlit run app.py"""

import html
import importlib
import os
import re
import sys
from datetime import date, datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

# Streamlit re-runs this file on every update but keeps imported modules in memory (the file
# watcher is off for speed), so after a deploy the old rag.py/progress.py would keep running.
# Reload them whenever their file on disk is newer than the copy in memory.
for _name in ("rag", "progress"):
    _module = sys.modules.get(_name)
    if _module is not None and getattr(_module, "LOADED_MTIME", None) != os.path.getmtime(_module.__file__):
        importlib.reload(_module)

import progress as prog  # noqa: E402  (must come after the reload above)
from rag import (  # noqa: E402
    EMBED_MODEL, PROJECT_DIR, RERANK_MODEL, KnowledgeBase, answer, extract_questions,
                 extract_syllabus, file_kind, find_topics, grade_exam, make_chunks, make_flashcards,
                 make_mock_exam, make_quiz, read_pages, study_plan, transcribe_answer)

load_dotenv(PROJECT_DIR / ".env")
try:  # online (Streamlit Cloud): keys come from the app's Secrets box
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:  # no secrets file locally -> .env is used
    pass
st.set_page_config(page_title="AI Exam Prep Assistant", page_icon=":material/school:", layout="wide")

# Dark purple "glass" theme. Streamlit's own colours are set in .streamlit/config.toml;
# this adds the gradient background, glass cards, pill buttons and the dashboard widgets.
st.markdown("""
<style>
/* the Exo 2 font is set in .streamlit/config.toml so icons keep their own icon font */
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
/* sidebar buttons: subtle dark pills instead of bright white ones */
section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
  background: rgba(255,255,255,.06); color: #ede9fe; border: 1px solid rgba(255,255,255,.12);
}
section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
  background: rgba(139,92,246,.25); color: #fff; }

/* section switcher looks like the app's top tabs */
div[data-testid="stButtonGroup"] button { border-radius: 999px !important; }

/* text inputs / chat input */
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-testid="stChatInput"] > div {
  border-radius: 999px !important; background: rgba(255,255,255,.06) !important;
}
div[data-testid="stChatInput"] textarea { min-height: 0 !important; }
/* hide Streamlit's "Press Enter to submit form · 0/60" hint: it overlaps the placeholder text */
div[data-testid="InputInstructions"] { display: none !important; }

/* dashboard pieces */
.topbar-title { font-size: 2rem !important; font-weight: 700; line-height: 1.2; letter-spacing: .3px; }
.chip { display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .75rem;
        border-radius:999px; background:rgba(255,255,255,.06);
        border:1px solid rgba(255,255,255,.08); font-size:.85rem; }
.avatar { width:30px; height:30px; border-radius:50%; display:inline-grid; place-items:center;
          background:linear-gradient(135deg,#f9a8d4,#a78bfa); font-weight:700; color:#1b1030; }
.stat { text-align: center; margin-bottom: .5rem; }
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
.hello { font-size: clamp(1.15rem, 1.6vw, 1.5rem); font-weight: 700; margin: 0;
         word-break: keep-all; overflow-wrap: normal; }
.hello-emoji { font-size: clamp(2.6rem, 4vw, 4.2rem); text-align: right; line-height: 1; }
.lesson-title { font-weight: 600; margin-bottom: .25rem; }
.bar { height: 6px; border-radius: 999px; background: rgba(255,255,255,.12); overflow: hidden; }
.bar > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#c4b5fd,#fff); }
.slot { border-left: 2px solid rgba(167,139,250,.6); padding: .35rem .7rem; margin: .45rem 0;
        border-radius: 0 12px 12px 0; background: rgba(139,92,246,.10); }
.slot small { opacity: .7; }
.mi { font-family: 'Material Symbols Rounded' !important; font-weight: normal; font-style: normal;
      font-size: 1.15em; line-height: 1; vertical-align: -0.22em; letter-spacing: normal;
      text-transform: none; white-space: nowrap; direction: ltr; -webkit-font-smoothing: antialiased;
      font-feature-settings: 'liga'; color: #c4b5fd; }
.mi.done { color: #4ade80; }
.dot { display:inline-block; width:.6rem; height:.6rem; border-radius:50%; margin-right:.2rem;
       box-shadow: 0 0 8px currentColor; vertical-align: .05em; }
.flash { min-height: 120px; display: grid; place-items: center; text-align: center;
         padding: 1.4rem 1rem; margin: .4rem 0 .8rem; border-radius: 18px; font-size: 1.25rem;
         font-weight: 600; background: linear-gradient(145deg, rgba(139,92,246,.28), rgba(76,29,149,.18));
         border: 1px solid rgba(167,139,250,.35); white-space: pre-wrap; }
.flash.back { font-size: 1.05rem; font-weight: 500; background: rgba(255,255,255,.06);
              border-color: rgba(255,255,255,.12); }
.hello-art { text-align: right; }
.hello-art.center { text-align: center; margin: .4rem 0 .2rem; }
.welcome-title { font-size: 1.7rem; font-weight: 700; text-align: center; }
.welcome-sub { text-align: center; opacity: .72; margin: .3rem 0 1rem; font-size: .95rem; }
.hello-art svg { filter: drop-shadow(0 0 18px rgba(139,92,246,.6)); }

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

HOME, ASK, QUIZ, TOPICS, PLAN = (":material/dashboard: Overview", ":material/forum: Ask", ":material/quiz: Quiz",
                                   ":material/insights: Topics", ":material/map: Study plan")
EXAM, CARDS = ":material/assignment: Mock exam", ":material/style: Flashcards"
PAGES = [HOME, ASK, QUIZ, EXAM, CARDS, TOPICS, PLAN]
EXAM_PRESETS = {  # name -> ([(questions, marks each), ...], minutes)
    "Quick test · 20 marks": ([(5, 2), (2, 5)], 30),
    "Mid-sem · 30 marks": ([(5, 2), (2, 10)], 60),
    "End-sem · 50 marks": ([(5, 2), (4, 10)], 120),
}
PRIORITY_BADGE = {"High": ":red-badge[High]", "Medium": ":orange-badge[Medium]",
                  "Low": ":green-badge[Low]"}
PRIORITY_COLOR = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#4ade80"}
USER_AVATAR, BOT_AVATAR = ":material/person:", ":material/auto_awesome:"


def mi(name: str, cls: str = "") -> str:
    """A Material icon for use inside custom HTML (same icon set Streamlit uses)."""
    return f"<span class='mi {cls}'>{name}</span>"


def dot(priority: str) -> str:
    return f"<span class='dot' style='background:{PRIORITY_COLOR[priority]}'></span>"


# Greeting-card graphic: glowing graduation cap badge (replaces the emoji)
GRAD_CAP_SVG = """
<div class='hello-art'><svg viewBox="0 0 96 96" width="88" height="88" aria-hidden="true">
  <defs>
    <radialGradient id="g1" cx="35%" cy="30%" r="75%">
      <stop offset="0" stop-color="#e9d5ff"/><stop offset=".45" stop-color="#8b5cf6"/>
      <stop offset="1" stop-color="#3b0f86"/></radialGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="4" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <circle cx="48" cy="48" r="38" fill="url(#g1)" filter="url(#glow)"/>
  <path d="M48 28 L76 40 L48 52 L20 40 Z" fill="#fff"/>
  <path d="M32 46 V58 C32 64 64 64 64 58 V46 L48 53 Z" fill="#f5f3ff" opacity=".92"/>
  <path d="M74 41 V56" stroke="#fff" stroke-width="2.5" stroke-linecap="round"/>
  <circle cx="74" cy="58" r="3" fill="#fde68a"/>
</svg></div>"""
GRAD_CAP_SVG = " ".join(line.strip() for line in GRAD_CAP_SVG.splitlines())  # one line for markdown
SUGGESTIONS = ["Summarise my notes in 10 points", "What are the most important topics?",
               "Explain the hardest concept simply", "What questions might come in the exam?"]
TIPS = ["Summarise my notes", "Important topics", "Likely exam questions"]


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


# ---------- Welcome: every visitor enters their name and gets their own profile ----------

def start_profile(name: str, course: str = "", base: dict | None = None):
    new = prog.create(name, course, base)
    st.query_params["u"] = new["id"]  # the profile lives in the link, so a bookmark brings you back


progress = prog.load(st.query_params.get("u"))  # name, quiz history, plan... (profiles/<id>.json)
if progress is None:
    _, mid_col, _ = st.columns([1, 1.4, 1])
    with mid_col, st.container(border=True):
        st.markdown(GRAD_CAP_SVG.replace("hello-art", "hello-art center"), unsafe_allow_html=True)
        st.markdown("<div class='welcome-title'>Welcome to Exam Prep AI</div>"
                    "<div class='welcome-sub'>Your notes turned into answers, quizzes, mock exams "
                    "and a study plan. Let's set up your space.</div>", unsafe_allow_html=True)
        with st.form("welcome", border=False):
            name = st.text_input("What's your name?", max_chars=40, placeholder="e.g. Priya")
            course = st.text_input("Course or subject (optional)", max_chars=60,
                                   placeholder="e.g. B.Tech CSE")
            go_in = st.form_submit_button("Get started", icon=":material/arrow_forward:",
                                          type="primary", width="stretch")
        if go_in:
            if name.strip():
                start_profile(name, course)
                st.rerun()
            else:
                st.warning("Please enter your name.")
        if st.query_params.get("u"):
            st.caption("That profile link wasn't found, so you can start a new profile here.")
        old = prog.legacy()
        if old and old.get("name"):
            st.divider()
            st.caption("Progress from before profiles were added was found on this computer.")
            if st.button(f"Continue as {old['name']}", icon=":material/history:", width="stretch"):
                start_profile(old["name"], base=old)
                prog.retire_legacy()
                st.rerun()
        st.caption("Tip: bookmark the page after you start. Your progress is saved to that link.")
    st.stop()


def read_with_progress(name: str, data: bytes) -> list[str]:
    """read_pages() with a progress bar while scanned pages are OCR'd."""
    bar = st.progress(0.0, text=f"Opening {name}...")

    def progress(done, total):
        bar.progress(done / total, text=f"Reading {name}: page {done} of {total} "
                                        "(handwritten pages are read by AI — keep this page open)")

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


def uploaded_bytes(f) -> bytes | None:
    """The uploaded file's content if it's a PDF or a photo, else a clear message.

    Upload boxes accept any file (type=None): a ".pdf" filter makes some phone file pickers
    grey out PDFs shared from WhatsApp/Drive (they arrive with a generic file type), so
    files are checked here by content instead of by name."""
    data = f.getvalue()
    if file_kind(data) is None:
        st.warning(f"**{f.name}** isn't a PDF or a photo, so it can't be read. Upload a PDF, "
                   "or a photo (JPG, PNG, HEIC) of the page.", icon=":material/warning:")
        return None
    return data


def queue_pdf(name: str, data: bytes):
    """Remember an uploaded notes PDF until it's read. Kept in the session (not in the upload
    widget), so switching sections or a phone interrupting the page doesn't lose it."""
    if name not in st.session_state.files:
        st.session_state.setdefault("pending", {}).setdefault(name, data)


def process_pending():
    """Read every queued notes PDF, showing progress in the main area (the sidebar is hidden
    on phones). If interrupted, the next run continues — OCR resumes from saved pages."""
    for name, data in list(st.session_state.get("pending", {}).items()):
        try:
            add_pdf(name, data)
            n = st.session_state.files[name]["pages"]
            st.toast(f"Added {name}: {n} page{'s' if n != 1 else ''}", icon=":material/check_circle:")
        except Exception as e:
            st.error(f"Couldn't read {name}: {e}", icon=":material/error:")
        st.session_state.pending.pop(name, None)


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
    with st.expander(icon=":material/menu_book:", label=f"Sources — page {', '.join(map(str, pages))}"):
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
    st.markdown("### :material/school: Exam Prep AI")
    st.caption("Your notes → answers, quizzes and exam priorities.")

    # PDFs already in the project folder are loaded automatically
    for pdf in sorted(PROJECT_DIR.glob("*.pdf")):
        add_pdf(pdf.name, pdf.read_bytes())
    for f in st.file_uploader("Add notes (PDF or photos)", type=None,
                              accept_multiple_files=True) or []:
        if (data := uploaded_bytes(f)) is not None:
            queue_pdf(f.name, data)

    st.markdown("**Loaded**")
    for name, info in st.session_state.files.items():
        with st.container(border=True):
            st.markdown(f":material/description: **{name}**")
            st.caption(f"{info['pages']} pages · {info['chunks']} searchable chunks")

    st.divider()
    show_sources = st.toggle("Show sources under answers", value=True)
    if st.button("Clear chat", icon=":material/delete_sweep:", width="stretch", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Profile", icon=":material/person:"):
        name = st.text_input("Your name", progress["name"], max_chars=40)
        course = st.text_input("Course", progress["course"], max_chars=60)
        if (name.strip() and name.strip() != progress["name"]) or course.strip() != progress["course"]:
            progress["name"], progress["course"] = name.strip() or progress["name"], course.strip()
            prog.save(progress)
        st.caption("Bookmark this page to come back to your progress. Your profile code:")
        st.code(progress["id"], language=None)
        if st.button("Switch user", icon=":material/logout:", width="stretch"):
            st.query_params.clear()
            st.session_state.clear()
            st.rerun()

    with st.expander("How it works", icon=":material/settings:"):
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
if page != PLAN:  # (the Study plan reads its own files first, then any pending notes)
    process_pending()  # read uploaded notes first: progress shows at the top, counts stay right
total_pages = sum(f["pages"] for f in st.session_state.files.values())

with st.container(border=True):
    c0, c1, c2, c3 = st.columns([.42, 1.3, 2, 1.2], vertical_alignment="center")
    c0.button("", icon=":material/home:", key="home_btn", help="Home", type="primary",
              on_click=go, args=(HOME,))
    c1.markdown(f"<div class='topbar-title'>{page.split(' ', 1)[1]}</div>", unsafe_allow_html=True)
    c2.text_input("Search", key="top_search", placeholder="Search your notes...",
                  label_visibility="collapsed", on_change=search_notes)
    initial = html.escape(progress["name"][:1].upper() or "S")
    c3.markdown(
        f"<div style='text-align:right'><span class='chip'>{mi('description')} {total_pages} page{'s' if total_pages != 1 else ''}</span> "
        f"<span class='chip'><span class='avatar'>{initial}</span>"
        f"<span><b>{html.escape(progress['name'])}</b><br>"
        f"<small>{html.escape(progress.get('course') or 'Student')}</small></span></span></div>",
        unsafe_allow_html=True)
    st.segmented_control("Section", PAGES, key="page", label_visibility="collapsed")
st.write("")

# New visitors have no notes yet: make uploading them the obvious first step
# (the sidebar uploader is hidden behind a button on phones).
if not st.session_state.files and page != PLAN:
    with st.container(border=True):
        st.markdown("#### :material/upload_file: Add your notes to get started")
        st.caption("Upload your class notes as PDF — typed or photographed handwritten pages both "
                   "work. Everything (answers, quizzes, mock exams, flashcards) comes from them.")
        a, b = st.columns([3, 1.3], vertical_alignment="center")
        a.caption("Have a course handout/syllabus or previous year papers? They go in Study plan.")
        b.button("Open Study plan", icon=":material/map:", key="to_plan", width="stretch",
                 on_click=go, args=(PLAN,))
        new_files = st.file_uploader("Notes (PDF or photos)", type=None, accept_multiple_files=True,
                                     key="main_upload", label_visibility="collapsed") or []
        queued = False
        for f in new_files:
            if (data := uploaded_bytes(f)) is not None and f.name not in st.session_state.files:
                queue_pdf(f.name, data)
                queued = True
        if queued:
            st.rerun()  # read them at the top of the page (process_pending), with progress
    if page != HOME:
        st.stop()


# ---------- Overview: dashboard ----------

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
        st.markdown("#### :material/calendar_month: Study schedule")
        today = date.today()
        exam = date.fromisoformat(progress["exam_date"]) if progress["exam_date"] else None
        st.date_input("Exam date", value=exam, min_value=today + timedelta(days=1),
                      key="exam_date_input", on_change=set_exam_date, format="DD/MM/YYYY")
        if not plan_rows:
            st.caption("Build a study plan from your syllabus + PYQs and your days will be "
                       "planned here, most important units first.")
            st.button("Build study plan", icon=":material/map:", width="stretch", type="primary",
                      on_click=go, args=(PLAN,))
        elif not exam or exam <= today:
            st.caption("Set your exam date to spread your units over the days left.")
        else:
            sched = prog.schedule(plan_rows, exam, today)
            st.markdown(f"<span class='chip'>{mi('hourglass_top')} {(exam - today).days} days to exam</span>",
                        unsafe_allow_html=True)
            days = sorted(sched)[:6]
            labels = {d: d.strftime("%a %d") for d in days}
            picked = st.segmented_control("Day", days, format_func=labels.get, default=days[0],
                                          key="sched_day", label_visibility="collapsed")
            for unit in sched.get(picked or days[0], []):
                if unit.get("revision"):
                    st.markdown(f"<div class='slot'>{mi('replay')} <b>Revision + PYQ practice</b><br>"
                                "<small>Go through every High-priority unit again</small></div>",
                                unsafe_allow_html=True)
                    continue
                done = unit["title"] in progress["units_done"]
                st.markdown(
                    f"<div class='slot'>{mi('check_circle', 'done') if done else dot(unit['priority'])} "
                    f"<b>Unit {unit['unit']}: {html.escape(unit['title'])}</b><br>"
                    f"<small>{unit['weightage']:.0%} weightage · {unit['priority']} priority</small>"
                    f"</div>", unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                b1.button("Study", key=f"s_{picked}_{unit['unit']}", width="stretch", on_click=go,
                          args=(ASK,),
                          kwargs={"pending_question": f"Summarise {unit['title']} for my exam"})
                b2.button("Quiz", key=f"q_{picked}_{unit['unit']}", width="stretch",
                          on_click=go, args=(QUIZ,), kwargs={"auto_quiz_topic": unit["title"]})

    # ----- left, below the schedule: flashcards due + last mock exam -----
    with left, st.container(border=True):
        n_due = len(prog.due_cards(progress))
        last_mock = progress["mocks"][-1] if progress["mocks"] else None
        mock_text = f"{last_mock['scored']:g}/{last_mock['total']}" if last_mock else "—"
        a, b = st.columns(2)
        a.markdown(f"<div class='stat'><div class='stat-big'>{n_due}</div>"
                   f"<div class='stat-sub'>Flashcards due</div></div>", unsafe_allow_html=True)
        b.markdown(f"<div class='stat'><div class='stat-big'>{mock_text}</div>"
                   f"<div class='stat-sub'>Last mock exam</div></div>", unsafe_allow_html=True)
        a.button("Review", icon=":material/style:", width="stretch", on_click=go, args=(CARDS,),
                 key="dash_cards")
        b.button("Take a mock", icon=":material/assignment:", width="stretch", on_click=go,
                 args=(EXAM,), key="dash_mock")

    # ----- middle: stats, progress ring, recent quizzes -----
    with mid:
        acc = prog.accuracy(progress)
        with st.container(border=True):
            a, b = st.columns(2)
            acc_text = f"{acc:.0%}" if acc is not None else "—"
            a.markdown(f"<div class='stat'><div class='stat-big'>{prog.points(progress)}</div>"
                       f"<div class='stat-sub'>Quiz points</div>"
                       f"<div class='up'>{len(progress['quizzes'])} quizzes taken</div></div>",
                       unsafe_allow_html=True)
            b.markdown(f"<div class='stat'><div class='stat-big'>{acc_text}</div>"
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
                st.button("Start a quiz", icon=":material/quiz:", width="stretch", on_click=go, args=(QUIZ,))
            for i, q in enumerate(recent):
                score = q["score"] / q["total"] if q["total"] else 0
                with st.container(border=True):
                    st.markdown(
                        f"<div class='lesson-title'>{html.escape(q['topic'])}</div>"
                        f"<div class='stat-sub'>{'Perfect!' if score == 1 else 'Good going' if score >= .6 else 'You can do better!'}"
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
            a, b = st.columns([3, 1], vertical_alignment="center")
            next_up = next((u for u in plan_rows if u["title"] not in progress["units_done"]), None)
            a.markdown(f"<div class='hello'>Hi, {html.escape(progress['name'])}!</div>"
                       f"<div class='stat-sub' style='font-size:.95rem'>Ready to make progress today?"
                       + (f"<br>Next up: <b>{html.escape(next_up['title'])}</b>" if next_up else "")
                       + "</div>", unsafe_allow_html=True)
            b.markdown(GRAD_CAP_SVG, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### AI Assistant")
            st.markdown("<div class='orb'></div>", unsafe_allow_html=True)
            st.pills("Tips", TIPS, key="dash_tip",
                     label_visibility="collapsed",
                     on_change=lambda: (go(ASK, pending_question=st.session_state.dash_tip),
                                        st.session_state.update(dash_tip=None)))
            st.text_input("Ask", key="dash_ask", placeholder="Ask me about your notes...",
                          label_visibility="collapsed", on_change=ask_from_dashboard)
            n_files = len(st.session_state.files)
            st.caption(f"Answers only from your {n_files} file{'s' if n_files != 1 else ''} "
                       f"({total_pages} page{'s' if total_pages != 1 else ''}), with page numbers.")


# ---------- Ask: chat with your notes ----------

if page == ASK:
    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("#### :material/forum: Ask anything from your notes")
            st.caption("Answers are written only from your PDFs and every claim shows its page. "
                       "If it's not in your notes, the assistant will say so.")
            st.pills("Try one:", SUGGESTIONS, key="suggestion",
                     on_change=lambda: (ask(st.session_state.suggestion),
                                        st.session_state.update(suggestion=None)))

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
            st.markdown(msg["content"])
            if msg.get("hits") and show_sources:
                render_sources(msg["hits"])

    typed = st.chat_input("Ask a question about your notes...")
    question = typed or st.session_state.pop("pending_question", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner("Searching your notes and writing an answer..."):
                try:
                    reply, hits = answer(question, kb)
                    reply = with_page_badges(reply, hits)
                    progress["asked"] += 1
                    prog.save(progress)
                except Exception as e:
                    reply, hits = f"Something went wrong: {e}", []
            st.markdown(reply)
            if hits and show_sources:
                render_sources(hits)
        st.session_state.messages.append({"role": "assistant", "content": reply, "hits": hits})


# ---------- Quiz: MCQs generated from your notes ----------

def new_quiz(topic: str, n: int, level: str):
    with st.spinner(f"Writing and fact-checking {n} questions on '{topic}'..."):
        try:
            st.session_state.quiz = make_quiz(topic, kb, n, level)
        except Exception as e:
            st.error(f"Couldn't make a quiz: {e}", icon=":material/error:")
            return
    st.session_state.quiz_topic = topic
    st.session_state.quiz_settings = (topic, n, level)
    st.session_state.quiz_checked = False
    st.session_state.quiz_picks = []
    st.session_state.quiz_round = st.session_state.get("quiz_round", 0) + 1  # fresh widgets


if page == QUIZ:
    with st.container(border=True):
        st.markdown("#### :material/quiz: Practice quiz")
        st.caption("MCQs written from your notes, then double-checked against them by a second AI pass.")
        with st.form("quiz_settings_form", border=False):
            topic = st.text_input("Topic", key="quiz_topic_input",
                                  placeholder="e.g. types of agents, BFS, PEAS, A* search")
            c1, c2 = st.columns(2)
            n = c1.slider("Questions", 3, 10, 5)
            level = c2.segmented_control("Difficulty", ["easy", "medium", "hard"],
                                         default="medium") or "medium"
            make = st.form_submit_button("Generate quiz", icon=":material/auto_awesome:", type="primary", width="stretch")

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
                c2.progress(pct, text=("Perfect!" if pct == 1 else "Good job!"
                                       if pct >= 0.6 else "Keep revising!") + f"  {pct:.0%}")
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
                            st.success(f"Correct! {q['explanation']}", icon=":material/check_circle:")
                        elif answers[i] is None:
                            st.warning(f"Skipped. Answer: **{right}**. {q['explanation']}", icon=":material/skip_next:")
                        else:
                            st.error(f"Answer: **{right}**. {q['explanation']}", icon=":material/cancel:")
                        st.caption(f":material/menu_book: {q['source']}, page {q['page']}")
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
            if c1.button("Retry same questions", icon=":material/replay:", width="stretch"):
                st.session_state.quiz_checked = False
                st.session_state.quiz_picks = []
                st.session_state.quiz_round += 1
                st.rerun()
            if c2.button("New quiz on this topic", icon=":material/auto_awesome:", width="stretch"):
                new_quiz(*st.session_state.quiz_settings)
                st.rerun()


# ---------- Mock exam: a full paper in exam pattern, written answers graded by AI ----------

@st.fragment(run_every="1s")
def exam_timer(ends_at: datetime):
    left = int((ends_at - datetime.now()).total_seconds())
    if left <= 0:
        st.error("Time's up! Submit your answers now.", icon=":material/timer_off:")
        return
    h, rest = divmod(left, 3600)
    st.markdown(f"<span class='chip'>{mi('timer')} {h}:{rest // 60:02d}:{rest % 60:02d} left</span>",
                unsafe_allow_html=True)


def grade_band(pct: float) -> str:
    return ("Outstanding" if pct >= .9 else "Very good" if pct >= .75 else "Good" if pct >= .6
            else "Pass" if pct >= .4 else "Needs work")


if page == EXAM:
    exam = st.session_state.get("exam")

    if exam is None:  # ----- set up a paper -----
        with st.container(border=True):
            st.markdown("#### :material/assignment: Mock exam")
            st.caption("A full paper in university pattern, written from your notes. Type your "
                       "answers or upload photos of handwritten ones; the AI marks each answer "
                       "against the key points like an examiner.")
            preset = st.segmented_control("Paper", [*EXAM_PRESETS, "Custom"],
                                          default="Quick test · 20 marks") or "Quick test · 20 marks"
            if preset == "Custom":
                c1, c2, c3, c4, c5 = st.columns(5)
                n_short = c1.number_input("Short Qs", 0, 10, 5)
                m_short = c2.number_input("Marks each", 1, 5, 2)
                n_long = c3.number_input("Long Qs", 0, 8, 2)
                m_long = c4.number_input("Marks each ", 4, 20, 10)
                minutes = c5.number_input("Minutes", 10, 180, 45, step=5)
                pattern = [(n, m) for n, m in [(n_short, m_short), (n_long, m_long)] if n]
            else:
                pattern, minutes = EXAM_PRESETS[preset]
            total = sum(n * m for n, m in pattern)
            st.caption(" + ".join(f"{n} × {m} marks" for n, m in pattern)
                       + f" = **{total} marks** · {minutes} minutes")
            use_plan = bool(progress["plan"]) and st.toggle(
                "Weight questions by my study plan",
                help="Only if your study plan is for the same subject as these notes.")
            if st.button("Generate paper", icon=":material/auto_awesome:", type="primary",
                         width="stretch", disabled=not pattern):
                with st.spinner("Setting your paper from your notes..."):
                    try:
                        paper = make_mock_exam(kb, pattern, progress["plan"] if use_plan else None)
                    except Exception as e:
                        st.error(f"Couldn't make the paper: {e}", icon=":material/error:")
                        paper = None
                if paper and paper["questions"]:
                    st.session_state.exam = {**paper, "id": datetime.now().strftime("%H%M%S"),
                                             "ends_at": datetime.now() + timedelta(minutes=minutes),
                                             "results": None, "answers": []}
                    st.rerun()
                elif paper is not None:
                    st.warning("Not enough notes to set a paper. Add notes in the sidebar.")

    elif exam["results"] is None:  # ----- writing the paper -----
        qs = exam["questions"]
        total = sum(q["marks"] for q in qs)
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 1.2, 1], vertical_alignment="center")
            c1.markdown(f"#### Mock paper · {total} marks")
            with c2:
                exam_timer(exam["ends_at"])
            if c3.button("Discard paper", icon=":material/close:", width="stretch"):
                st.session_state.exam = None
                st.rerun()
            st.caption("Answer in your own words. For long answers, write points like you would "
                       "in the exam. Blank answers score 0.")

        with st.form("mock_answers", border=False):
            for i, q in enumerate(qs, start=1):
                with st.container(border=True):
                    st.markdown(f":violet-badge[Q{i}] :orange-badge[{q['marks']} marks]  \n"
                                f"**{maths(q['question'])}**")
                    st.text_area("Your answer", key=f"ans{i}-{exam['id']}",
                                 height=90 if q["marks"] <= 3 else 220,
                                 label_visibility="collapsed", placeholder="Write your answer...")
                    st.file_uploader("…or upload a photo of your handwritten answer",
                                     type=None, key=f"img{i}-{exam['id']}")
            submit = st.form_submit_button("Submit for grading", icon=":material/grading:",
                                           type="primary", width="stretch")
        if submit:
            answers = []
            with st.spinner("Reading handwritten answers and marking your paper..."):
                try:
                    for i in range(1, len(qs) + 1):
                        text = st.session_state.get(f"ans{i}-{exam['id']}") or ""
                        img = st.session_state.get(f"img{i}-{exam['id']}")
                        kind = file_kind(img.getvalue()) if img is not None else None
                        if kind and kind != "pdf":
                            text = (text + "\n" + transcribe_answer(img.getvalue(), kind)).strip()
                        answers.append(text)
                    exam["answers"] = answers
                    exam["results"] = grade_exam(exam, answers)
                    prog.record_mock(progress, sum(r["marks"] for r in exam["results"]), total, len(qs))
                    st.rerun()
                except Exception as e:
                    st.error(f"Couldn't grade the paper: {e}", icon=":material/error:")

    else:  # ----- results -----
        qs, results = exam["questions"], exam["results"]
        total = sum(q["marks"] for q in qs)
        scored = sum(r["marks"] for r in results)
        pct = scored / total if total else 0
        with st.container(border=True):
            c1, c2 = st.columns([1, 2], vertical_alignment="center")
            c1.markdown(f"<div class='stat-big'>{scored:g} / {total}</div>"
                        f"<div class='stat-sub'>Mock exam score</div>", unsafe_allow_html=True)
            c2.progress(pct, text=f"{grade_band(pct)} · {pct:.0%}")
            weak = sorted({q["page"] for q, r in zip(qs, results) if r["marks"] < q["marks"] * .6})
            if weak:
                c2.caption(f"Revise page(s) {', '.join(map(str, weak))} of your notes.")
            b1, b2 = st.columns(2)
            if b1.button("New paper", icon=":material/refresh:", width="stretch", type="primary"):
                st.session_state.exam = None
                st.rerun()
            lost = [(q, r) for q, r in zip(qs, results) if r["marks"] < q["marks"]]
            if b2.button(f"Make flashcards from {len(lost)} weak answer(s)", icon=":material/style:",
                         width="stretch", disabled=not lost):
                added = prog.add_cards(progress, [
                    {"front": q["question"], "back": q["model_answer"], "topic": "Mock exam",
                     "source": q["source"], "page": q["page"]} for q, _ in lost])
                st.toast(f"Added {added} flashcard(s) to your deck.", icon=":material/style:")

        for i, (q, r, a) in enumerate(zip(qs, results, exam["answers"]), start=1):
            with st.container(border=True):
                colour = "green" if r["marks"] == q["marks"] else "orange" if r["marks"] else "red"
                st.markdown(f":violet-badge[Q{i}] :{colour}-badge[{r['marks']:g} / {q['marks']} marks]"
                            f"  \n**{maths(q['question'])}**")
                if a:
                    with st.expander("Your answer"):
                        st.markdown(plain(a))
                for point in r["covered"]:
                    st.markdown(f":green[:material/check_circle:] {plain(point)}")
                for point in r["missed"]:
                    st.markdown(f":red[:material/cancel:] {plain(point)}")
                st.caption(r["feedback"])
                with st.expander("Model answer", icon=":material/menu_book:"):
                    st.markdown(maths(q["model_answer"]))
                    st.caption(f"From {q['source']}, page {q['page']}")


# ---------- Flashcards: spaced repetition ----------

def rate_card(card_id: str, quality: int):
    for card in progress["cards"]:
        if card["id"] == card_id:
            prog.review(card, quality)
    prog.save(progress)
    st.session_state.card_revealed = False


if page == CARDS:
    cards = progress["cards"]
    due = sorted(prog.due_cards(progress), key=lambda c: c["due"])
    learned = sum(c["reps"] >= 2 for c in cards)

    with st.container(border=True):
        a, b, c = st.columns(3)
        for col, value, label in [(a, len(due), "Due today"), (b, len(cards), "Cards"),
                                  (c, learned, "Learned")]:
            col.markdown(f"<div style='text-align:center'><div class='stat-big'>{value}</div>"
                         f"<div class='stat-sub'>{label}</div></div>", unsafe_allow_html=True)

    left, right = st.columns([1.6, 1], gap="medium")
    with left, st.container(border=True):
        st.markdown("#### :material/style: Review")
        if not cards:
            st.caption("No cards yet. Make some from any topic on the right.")
        elif not due:
            next_due = min(c["due"] for c in cards)
            st.success(f"All caught up! Next cards are due on {next_due}.",
                       icon=":material/task_alt:")
        else:
            card = due[0]
            st.caption(f"{len(due)} left today · {card['topic']} · {card['source']}, p. {card['page']}")
            st.markdown(f"<div class='flash'>{html.escape(card['front'])}</div>",
                        unsafe_allow_html=True)
            if not st.session_state.get("card_revealed"):
                st.button("Show answer", icon=":material/visibility:", type="primary",
                          width="stretch", on_click=st.session_state.update,
                          kwargs={"card_revealed": True})
            else:
                st.markdown(f"<div class='flash back'>{html.escape(card['back'])}</div>",
                            unsafe_allow_html=True)
                st.caption("How well did you remember it?")
                cols = st.columns(4)
                for col, (label, q) in zip(cols, [("Again", 1), ("Hard", 3), ("Good", 4), ("Easy", 5)]):
                    days = prog.next_interval(card, q)
                    col.button(f"{label} · {f'{days}d' if days else 'now'}", key=f"rate{q}",
                               width="stretch",
                               type="primary" if q == 4 else "secondary",
                               on_click=rate_card, args=(card["id"], q))

    with right, st.container(border=True):
        st.markdown("#### :material/auto_awesome: Make cards")
        with st.form("make_cards", border=False):
            topic = st.text_input("Topic", placeholder="e.g. types of agents, A* search",
                                  key="card_topic")
            n = st.slider("How many", 5, 20, 10)
            make = st.form_submit_button("Make flashcards", type="primary", width="stretch")
        if make and topic.strip():
            with st.spinner(f"Writing {n} cards on '{topic}' from your notes..."):
                try:
                    added = prog.add_cards(progress, make_flashcards(topic.strip(), kb, n))
                    st.toast(f"Added {added} new card(s).", icon=":material/style:")
                    st.rerun()
                except Exception as e:
                    st.error(f"Couldn't make cards: {e}", icon=":material/error:")
        elif make:
            st.warning("Type a topic first.")

        if cards:
            with st.expander(f"Your deck ({len(cards)})", icon=":material/folder:"):
                by_topic: dict[str, list] = {}
                for c in cards:
                    by_topic.setdefault(c["topic"], []).append(c)
                for t, group in by_topic.items():
                    x, y = st.columns([3, 1], vertical_alignment="center")
                    x.markdown(f"**{plain(t)}** · {len(group)} cards")
                    if y.button("Delete", key=f"del-{t}", icon=":material/delete:",
                                type="tertiary"):
                        progress["cards"] = [c for c in cards if c["topic"] != t]
                        prog.save(progress)
                        st.rerun()


# ---------- Important topics: most-asked exam questions ----------

if page == TOPICS:
    with st.container(border=True):
        st.markdown("#### :material/insights: What should I study first?")
        st.caption("Finds every exam question in your notes and past papers, groups similar ones "
                   "with clustering, and ranks topics by how often they're asked + their marks. "
                   "Add past papers in the sidebar for better results.")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        grouping = c1.select_slider(
            "Grouping", options=[0.2, 0.25, 0.3, 0.35, 0.4], value=0.3,
            format_func=lambda v: {0.2: "Very fine", 0.25: "Fine", 0.3: "Balanced",
                                   0.35: "Broad", 0.4: "Very broad"}[v],
            help="How similar questions must be to count as the same topic.")
        analyze = c2.button("Analyze", icon=":material/search:", type="primary", width="stretch")

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
            st.error(f"Analysis failed: {e}", icon=":material/error:")

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
                   f"for {topics[0]['total_marks']} marks.", icon=":material/emoji_events:")

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
                b1.button("Quiz", icon=":material/quiz:", key=f"tq{rank}", width="stretch", help="Quiz me on this",
                          on_click=go, args=(QUIZ,), kwargs={"auto_quiz_topic": t["topic"]})
                b2.button("Explain", icon=":material/forum:", key=f"te{rank}", width="stretch",
                          help="Explain this from my notes", on_click=go, args=(ASK,),
                          kwargs={"pending_question": f"Explain {t['topic']} for my exam"})
                with st.expander(f"{len(t['questions'])} question(s)", expanded=rank == 1):
                    for q in t["questions"]:
                        marks = f" :orange-badge[{q['marks']} marks]" if q["marks"] else ""
                        st.markdown(f"- {plain(q['question'])}{marks} "
                                    f":gray-badge[{q['source']} · p. {q['page']}]")


# ---------- Study plan: syllabus + PYQs -> which chapter first, with weightage ----------

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
        st.markdown("#### :material/map: Which chapter should I start with?")
        st.caption("Upload your course handout/syllabus and previous year papers (PYQs). "
                   "Each PYQ question is matched to a syllabus unit to work out its weightage. "
                   "Scanned papers work too.")
        c1, c2 = st.columns(2)
        syl = c1.file_uploader("Course handout / syllabus (PDF or photo)", type=None, key="syl_up")
        pyqs = c2.file_uploader("Previous year papers (PDF or photos)", type=None,
                                accept_multiple_files=True, key="pyq_up")
        # Copy uploads into the session straight away (they survive switching sections) and
        # confirm each new file instantly, so it's clear on a phone that the upload arrived.
        if syl and (data := uploaded_bytes(syl)) is not None \
                and st.session_state.plan_syllabus != (syl.name, data):
            st.session_state.plan_syllabus = (syl.name, data)
            st.toast(f"Got {syl.name} — building your plan…", icon=":material/check_circle:")
        for f in pyqs or []:
            if (data := uploaded_bytes(f)) is not None and st.session_state.plan_pyqs.get(f.name) != data:
                st.session_state.plan_pyqs[f.name] = data
                st.toast(f"Got {f.name}", icon=":material/check_circle:")

        if st.session_state.plan_syllabus or st.session_state.plan_pyqs:
            chips = []
            if st.session_state.plan_syllabus:
                chips.append(f":violet-badge[:material/menu_book: {st.session_state.plan_syllabus[0]}]")
            chips += [f":blue-badge[:material/description: {n}]" for n in st.session_state.plan_pyqs]
            st.markdown(" ".join(chips))
            if st.button("Clear files", key="plan_clear"):
                st.session_state.plan_syllabus, st.session_state.plan_pyqs = None, {}
                st.session_state.pop("plan", None)
                st.session_state.pop("plan_files", None)
                st.rerun()

        # The plan builds by itself whenever the uploaded files change (students kept
        # uploading and waiting, not noticing a separate button); the button rebuilds on demand.
        files_now = None
        if st.session_state.plan_syllabus:
            files_now = (st.session_state.plan_syllabus[0], *sorted(st.session_state.plan_pyqs))
        rebuild = st.button("Rebuild study plan" if st.session_state.get("plan") else "Build my study plan",
                            icon=":material/analytics:", type="primary", width="stretch",
                            disabled=files_now is None)
        if files_now is None:
            st.caption("Add your syllabus first — the plan builds automatically. "
                       "PYQs are optional but give real exam weightage.")
    build = rebuild or (files_now is not None and files_now != st.session_state.get("plan_files"))

    if build:
        # plan_files is only recorded once the build finishes (or fails with an error), so a
        # build interrupted on a phone simply runs again next time, resuming any saved OCR pages
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
            progress["course"] = progress["course"] or st.session_state.plan["course"]
            prog.save(progress)
        except Exception as e:
            st.error(f"Couldn't build the plan: {e}", icon=":material/error:")
        st.session_state.plan_files = files_now

    plan = st.session_state.get("plan")
    if plan and not plan["rows"]:
        st.warning("No units found in that file. Is it the course handout/syllabus?")
    elif plan:
        rows = plan["rows"]
        if plan["course"]:
            st.markdown(f"##### :material/menu_book: {plan['course']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Units", len(rows))
        m2.metric("PYQ papers", plan["papers"])
        m3.metric("Questions", plan["n_questions"])
        m4.metric("Marks analysed", plan["total_marks"])

        top = rows[0]
        if plan["basis"] == "pyq":
            st.success(f"Start with **Unit {top['unit']}: {top['title']}** — "
                       f"{top['weightage']:.0%} of the PYQ "
                       f"{'marks' if plan['use_marks'] else 'questions'}.", icon=":material/emoji_events:")
        elif plan["papers"] and not plan["n_questions"]:
            st.warning("No questions could be read from your PYQ file(s), so units are ranked "
                       f"by **{'lecture hours' if plan['basis'] == 'hours' else 'syllabus order'}** "
                       "for now. If it's a photo or scan, make sure the page is upright, in focus "
                       "and well lit, then upload it again.", icon=":material/warning:")
        elif plan["basis"] == "hours":
            st.info("No PYQs added, so units are ranked by **lecture hours**. "
                    "Add previous year papers for real exam weightage.", icon=":material/info:")
        else:
            st.info("No PYQs or lecture hours found, so units are in **syllabus order**. "
                    "Add previous year papers for real exam weightage.", icon=":material/info:")

        weight_label = ("PYQ weightage" if plan["basis"] == "pyq" else
                        "Lecture-hour share" if plan["basis"] == "hours" else "Weightage")
        table = pd.DataFrame([{
            "#": r["order"],
            "Unit": f"{r['unit']}. {r['title']}",
            "Priority": r["priority"],
            weight_label: r["weightage"] * 100,
            "PYQ marks": r["marks"],
            "Asked in": f"{r['asked_in']}/{plan['papers']}" if plan["papers"] else "—",
            "Hours": r["hours"] or None,
        } for r in rows])
        if plan["basis"] == "pyq":
            table = table.drop(columns="Hours")
        else:
            table = table.drop(columns=["PYQ marks", "Asked in"])
        # colour the Priority cell instead of using emoji dots
        styled = table.style.map(lambda p: f"color: {PRIORITY_COLOR[p]}; font-weight: 600;",
                                 subset=["Priority"])
        st.dataframe(
            styled, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3,
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
        st.download_button("Download table (CSV)", csv.to_csv(index=False).encode("utf-8-sig"),
                           "study_plan.csv", "text/csv", icon=":material/download:")

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
                            f"{PRIORITY_BADGE[r['priority']]} · {r['why']}")
                b1, b2 = c2.columns(2)
                b1.button("Quiz", icon=":material/quiz:", key=f"pq{r['unit']}", width="stretch",
                          on_click=go, args=(QUIZ,), kwargs={"auto_quiz_topic": r["title"]})
                b2.button("Explain", icon=":material/forum:", key=f"pe{r['unit']}", width="stretch", on_click=go,
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
            with st.expander(f"{len(plan['unmapped'])} question(s) didn't match any unit", icon=":material/warning:"):
                for q in plan["unmapped"]:
                    st.markdown(f"- {plain(q['question'])} :gray-badge[{q['paper']}]")

    process_pending()  # any notes uploaded elsewhere are read after the plan, not before it
