# 📚 AI Exam Prep Assistant

Turn your course notes — **even handwritten, photographed ones** — into a personal exam tutor.

- 🏠 **Overview dashboard** (dark glass UI, works on phone and laptop): study schedule up to
  your exam date, quiz points & accuracy, units-completed ring, recent quizzes with *Continue*,
  and an AI assistant box.
- 👋 **Personal profiles**: every visitor enters their name on a welcome screen and gets their
  own dashboard ("Hi, Priya!"). Progress is saved per student (in Supabase online, or
  `profiles/<id>.json` locally); the id is kept in the page link, so bookmarking it brings you
  back. *Switch user* in the sidebar.

- 💬 **Ask** questions and get answers written *only* from your notes, with page numbers
- 📝 **Quiz** yourself with MCQs generated from your notes and fact-checked by a second AI pass
- 🧾 **Mock exam**: a full paper in university pattern (e.g. 5×2 + 2×10), with a live timer.
  Type answers or upload photos of handwritten ones; the AI marks each answer against key points
  like an examiner, lists what you covered/missed, shows a model answer, and can turn weak
  answers into flashcards
- 🃏 **Flashcards with spaced repetition** (SM-2, the algorithm behind Anki): cards made from
  your notes come back just before you'd forget them — *Again / Hard / Good / Easy*
- 🎯 **Important topics**: finds every exam question in your notes/past papers, groups similar
  ones and tells you what to study first
- 🗺️ **Study plan**: upload your course handout/syllabus and previous year papers (PYQs) and get
  a **priority table** — which unit to start with, its PYQ **weightage %**, marks, how many papers
  asked it, and High/Medium/Low priority (downloadable as CSV)

If something isn't in your notes, it says so instead of making it up.

## Results (measured, not guessed)

Evaluated with [`levels/level6_evaluate.py`](levels/level6_evaluate.py) on 34 test questions
over 43 pages of handwritten AI-course notes:

| Retrieval setup | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| Vector search only (basic RAG) | 41% | 71% | 0.52 |
| + BM25 hybrid search | 41% | 76% | 0.55 |
| + small re-ranker (MiniLM) | 44% | 85% | 0.60 |
| **+ bge-reranker-base (this app)** | **53%** | **85%** | **0.67** |

- **Hit@1** = the right page is the #1 result, **Hit@5** = it's in the top 5 the LLM reads,
  **MRR** = mean reciprocal rank (1.0 = perfect).
- **Hallucination test:** 5/5 off-topic questions correctly refused (100%).
- Bigger embedding model (bge-base) was also tested: no gain at Hit@1, so the small one is kept.

## How it works

```
PDF ─► text layer? ──no──► Gemini vision OCR (keeps line structure) ─► cache
              │yes                                                   │
              └──────────────────► page texts ◄──────────────────────┘
                                       │
                         overlapping chunks (800 chars, 150 overlap)
                       ┌───────────────┴───────────────┐
            bge-small embeddings + FAISS          BM25 keyword index
                       └──── Reciprocal Rank Fusion ───┘
 Question ────────────────────────►│
                        top 15 ─► cross-encoder re-ranker ─► top 5
                                                              │
               Gemini answers ONLY from these 5, citing pages ◄┘

Quiz:   top chunks ─► Gemini writes MCQs (JSON schema) ─► Gemini checks each answer key
Topics: Gemini extracts exam questions ─► embeddings ─► Agglomerative Clustering ─► ranking
Plan:   syllabus ─► units + topics (JSON)   PYQs ─► questions + marks (JSON)
        each question ─► Gemini classifies into a unit ─► weightage = unit marks / all marks
```

**Study plan ranking:** units are ordered by PYQ weightage (share of marks; question count if the
papers show no marks), then by how many papers asked them. Priority is **High** if a unit's share is
≥ 1.25× the average unit share, **Low** if ≤ 0.6×, otherwise **Medium**. Without PYQs it falls back
to lecture hours from the handout, then to syllabus order. Put files in `syllabus/` and `pyqs/` to
load them automatically, or upload them in the app.

## Tech stack

| Part | Tool |
|---|---|
| UI | Streamlit, Altair |
| LLM, OCR, structured output | Google Gemini (free tier) |
| Embeddings | `BAAI/bge-small-en-v1.5` (local) |
| Vector index | FAISS |
| Keyword search | BM25 (`rank-bm25`) |
| Re-ranker | `BAAI/bge-reranker-base` (local) |
| Clustering | scikit-learn Agglomerative Clustering |
| PDF | pypdf, pypdfium2 |

## Run locally

```bash
python -m venv .venv          # tip: keep it outside OneDrive/Dropbox folders to save cloud space
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env        # then put your free Gemini API key in .env
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com/apikey. Put your PDFs in the project
folder (loaded automatically) or upload them in the sidebar. First start takes ~1 minute while
the local models load.

## Deploy (free)

**1. Streamlit Community Cloud** — [share.streamlit.io](https://share.streamlit.io) → sign in
with GitHub → *Create app* → repo `exam-prep-ai`, branch `main`, file `app.py` →
*Advanced settings*: Python **3.13**, and paste [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)
into *Secrets* with your values → *Deploy*. The first build takes ~5-10 minutes.

**2. Keep profiles (optional, recommended)** — the free server's disk is wiped on every restart.
Create a free project at [supabase.com](https://supabase.com), open *SQL Editor*, run:

```sql
create table profiles (
  id text primary key,
  data jsonb not null,
  updated_at timestamptz default now()
);
alter table profiles enable row level security;  -- only the server key can read/write
```

Then add `SUPABASE_URL` (Project Settings → API → Project URL) and `SUPABASE_KEY`
(the **service_role** key — it stays on the server, never in the browser) to the app's Secrets.

**Notes:** each visitor uploads their own notes; nothing in `*.pdf`, `notes_text/`, `profiles/`
or `.env` is ever committed. The app uses your Gemini key for everyone, so keep it on the free
tier (no billing) to avoid any cost.

**Hugging Face Spaces** (more RAM) works too: create a Docker/Streamlit Space, upload these
files and add the same secrets.

## Learning path (`levels/`)

| Level | Where | Concept |
|---|---|---|
| 0 | `levels/level0_hello_ai.py` | Calling an LLM API safely (keys in `.env`) |
| 1 | `levels/level1_read_pdf.py` | PDF text extraction, vision OCR, chunking |
| 2 | `levels/level2_search.py` | Embeddings, cosine similarity, FAISS |
| 3 | `rag.py` → `answer()` | RAG: hybrid search, re-ranking, grounded answers with citations |
| 4 | `rag.py` → `make_quiz()` | Structured output (Pydantic schema), LLM-as-a-judge verification |
| 5 | `rag.py` → `find_topics()` | Information extraction + Agglomerative Clustering |
| 5+ | `rag.py` → `study_plan()` | Syllabus parsing, LLM classification, weightage analysis |
| 5+ | `rag.py` → `make_mock_exam()`, `grade_exam()` | Rubric-based LLM grading, handwriting OCR of answers |
| 5+ | `progress.py` → `review()` | SM-2 spaced repetition scheduling |
| 6 | `levels/level6_evaluate.py` | Evaluation: synthetic test set, Hit@k, MRR, hallucination test |
| 7 | this README | Deployment, documentation |

## Challenges solved

- **Handwritten notes had no text layer** → Gemini vision OCR, cached so each PDF is read once.
- **Acronyms like "PEAS" were missed** by embedding search → hybrid BM25 + vector search and a
  re-ranker. Measured: Hit@5 71% → 85%.
- **Quiz answer keys were wrong** ("ChatGPT" attached to the wrong agent type) → root cause was
  flattening OCR text into one line, which lost which bullet belonged to which heading. Fixed by
  keeping line structure; a verification pass was added as a second safety net.
- **Choosing models by data, not guesswork** → compared 2 embedding × 2 re-ranker models; the
  larger re-ranker raised Hit@1 from 44% to 53%, the larger embedder added nothing.
- **Free-tier limits** (429/503 errors, a retired model name) → automatic retry with backoff
  and a `-latest` model alias.

## Limitations & future work

- Questions spanning many pages (e.g. "how are heuristic values assigned") are harder to pin
  to one page; page-level grouping or larger chunks could help.
- Diagrams are described in text by OCR but not searchable as images.
- Answer quality is checked by refusal tests; an LLM-judged faithfulness score is a next step.
