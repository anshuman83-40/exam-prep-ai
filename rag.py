"""Core RAG pipeline used by the app.

PDF -> text (OCR for handwritten pages) -> chunks -> embeddings -> FAISS
Question -> embedding -> top-k chunks -> Gemini answers ONLY from those chunks, with citations.
(Each step is explained one at a time in the levels/ folder.)
"""

import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pypdfium2 as pdfium
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")  # reads GEMINI_API_KEY
CACHE_DIR = PROJECT_DIR / "notes_text"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Chosen by evaluation (levels/level6_evaluate.py): bge-reranker-base ranks the right page #1
# for 53% of test questions vs 44% with the smaller MiniLM re-ranker (MRR 0.67 vs 0.60).
# On a small server, set RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 (5x faster).
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
QUERY_PREFIX ="Represent this sentence for searching relevant passages: "
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

OCR_PROMPT = """This is a page of a student's notes (may be handwritten).
Transcribe ALL the text exactly as written, top to bottom.
- Keep headings, bullet points and question numbers.
- Write arrows as "->".
- Fix obvious spelling mistakes only when you are sure of the word.
- For a drawing or diagram, write one line: [Diagram: short description].
Output only the transcribed text, nothing else."""

ANSWER_PROMPT = """You are a friendly study assistant helping a student prepare for exams.
Answer the question using ONLY the numbered notes below.
- Cite the notes you used like [1] or [2][3] right after the sentence.
- Write clearly with short paragraphs or bullet points, like a good exam answer.
- If the notes don't contain the answer, say "I couldn't find this in your notes."
  and do not make anything up.
- Start directly with the answer: no greeting, no "here is", no sign-off.

Notes:
{context}

Question: {question}

Answer:"""


@dataclass
class Chunk:
    text: str
    source: str  # file name
    page: int  # 1-based page number


# ---------- Gemini helpers ----------

def get_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def get_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")


def generate(client: genai.Client, contents, config=None, retries: int = 5) -> str:
    """Call Gemini, waiting and retrying if the free tier is rate-limited (429) or busy (503)."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=get_model(), contents=contents, config=config
            )
            return response.text or ""
        except Exception as e:
            if attempt == retries - 1 or not any(c in str(e) for c in ("429", "503")):
                raise
            time.sleep(10 * (attempt + 1))
    return ""


# ---------- Level 1: reading PDFs ----------

def clean(text: str) -> str:
    """Tidy spaces but KEEP line breaks: in notes, which line an item sits under carries
    meaning (e.g. "-> Chatgpt" belongs to the heading above it)."""
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def read_pages(data: bytes, on_progress=None) -> list[str]:
    """Return the text of each page. Pages without a text layer are OCR'd with Gemini.
    Results are cached in notes_text/ by file content, so each PDF is only OCR'd once."""
    cache_file = CACHE_DIR / f"{hashlib.sha1(data).hexdigest()[:16]}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    pages = [clean(p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages]
    empty = [i for i, t in enumerate(pages) if not t]
    if empty:
        client = get_client()
        doc = pdfium.PdfDocument(data)
        for n, i in enumerate(empty, start=1):
            if on_progress:
                on_progress(n, len(empty))
            buf = io.BytesIO()
            doc[i].render(scale=2).to_pil().convert("RGB").save(buf, format="JPEG", quality=85)
            image = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
            pages[i] = clean(generate(client, [image, OCR_PROMPT]))

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    return pages


def make_chunks(pages: list[str], source: str) -> list[Chunk]:
    """Split each page into overlapping chunks, remembering where each came from."""
    chunks = []
    for page_num, text in enumerate(pages, start=1):
        for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            piece = text[start : start + CHUNK_SIZE]
            if piece.strip():
                chunks.append(Chunk(piece, source, page_num))
    return chunks


# ---------- Level 2: embeddings + vector search ----------

# Filler words (and exam-question verbs) that would distract keyword search
STOPWORDS = set("""a an the is are was were be been of to in on at for with by from and or not
it its this that these those as into about what which who whom how why when where do does did
can could should would will shall may might must i you me my your we our they their he she
explain describe define write give short note notes discuss list state tell please""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class KnowledgeBase:
    """Two-stage retrieval, the same design used in production RAG systems.

    Stage 1 (fast, recall): hybrid search = FAISS (meaning) + BM25 (exact keywords), merged
      with Reciprocal Rank Fusion. Embeddings understand meaning ("sense surroundings" ~
      "sensors") but are weak on acronyms like "PEAS"; BM25 is the opposite.
    Stage 2 (accurate, precision): a cross-encoder re-ranker reads the question and each
      candidate TOGETHER and scores how well the chunk answers it.
    """

    RRF_K = 60  # standard constant for Reciprocal Rank Fusion
    CANDIDATES = 15  # how many stage-1 results the re-ranker looks at

    def __init__(self, embedder: SentenceTransformer, reranker: CrossEncoder | None = None,
                 hybrid: bool = True):
        self.embedder = embedder
        self.reranker = reranker
        self.hybrid = hybrid
        self.index = None
        self.bm25 = None
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]):
        if not chunks:
            return
        vectors = self.embedder.encode([c.text for c in chunks], normalize_embeddings=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def vector_search(self, query: str, n: int) -> list[tuple[int, float]]:
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype=np.float32), n)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0])]

    def keyword_search(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(scores)[::-1][:n] if scores[i] > 0]

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks with their cosine similarity to the question."""
        if self.index is None:
            return []
        vector_hits = self.vector_search(query, len(self.chunks))  # every chunk, best first
        similarity = dict(vector_hits)

        if self.hybrid:
            # Reciprocal Rank Fusion: ranked high in EITHER list -> high fused score
            fused: dict[int, float] = {}
            for ranking in ([i for i, _ in vector_hits[:20]], self.keyword_search(query, 20)):
                for rank, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0) + 1 / (self.RRF_K + rank + 1)
            candidates = sorted(fused, key=fused.get, reverse=True)
        else:
            candidates = [i for i, _ in vector_hits]

        if self.reranker:
            candidates = candidates[: self.CANDIDATES]
            scores = self.reranker.predict([(query, self.chunks[i].text) for i in candidates])
            candidates = [candidates[j] for j in np.argsort(scores)[::-1]]

        return [(self.chunks[i], similarity[i]) for i in candidates[:k]]


# ---------- Level 3: retrieval-augmented generation ----------

def answer(question: str, kb: KnowledgeBase, k: int = 5) -> tuple[str, list[tuple[Chunk, float]]]:
    """Retrieve the most relevant chunks and ask Gemini to answer only from them."""
    hits = kb.search(question, k)
    if not hits:
        return "Please add some notes first.", []

    context = "\n\n".join(
        f"[{i}] ({c.source}, page {c.page})\n{c.text}" for i, (c, _) in enumerate(hits, start=1)
    )
    reply = generate(get_client(), ANSWER_PROMPT.format(context=context, question=question))
    return reply, hits


# ---------- Level 4: quiz generator (structured output) ----------

class MCQ(BaseModel):
    question: str
    options: list[str] = Field(description="exactly 4 answer options, without A/B/C/D labels")
    answer_index: int = Field(description="index 0-3 of the correct option")
    explanation: str = Field(description="1-2 sentences on why the answer is correct")
    note_number: int = Field(description="the [n] number of the note the question is based on")


class Quiz(BaseModel):
    questions: list[MCQ]


QUIZ_PROMPT = """You are an exam setter. Using ONLY the numbered notes below, write {n} {level}
multiple-choice questions about "{topic}" for a student revising for an exam.
- Every question must be answerable from the notes. Do not use outside facts.
- Exactly 4 options per question, only one correct; wrong options should be believable.
- Spread questions across different facts; no two questions should test the same fact.
- Vary the position of the correct answer.

Notes:
{context}"""


class Verdict(BaseModel):
    question_number: int
    verdict: str = Field(description='"correct", "wrong_key" or "ambiguous"')
    correct_index: int = Field(description="index 0-3 of the option the notes actually support")


class Review(BaseModel):
    verdicts: list[Verdict]


VERIFY_PROMPT = """You are a strict exam checker. For each multiple-choice question below,
check the marked answer against the notes ONLY.
- "correct": the notes clearly support the marked answer and no other option.
- "wrong_key": the notes clearly support a DIFFERENT option; give its index.
- "ambiguous": the notes are unclear, could support several options, or don't cover it.
Handwritten notes were transcribed, so line breaks may be lost: be careful which label
an example (like a product name) belongs to.

Notes:
{context}

Questions (answer_index is 0-based):
{questions}"""


def make_quiz(topic: str, kb: KnowledgeBase, n: int = 5, level: str = "medium") -> list[dict]:
    """Generate MCQs from the notes most relevant to `topic`, then have a second AI pass
    verify each answer against the notes (LLM-as-a-judge). Wrong keys are fixed and
    ambiguous questions dropped. Returns dicts with question, options, answer_index,
    explanation, source, page."""
    hits = kb.search(topic, k=6)
    if not hits:
        return []
    context = "\n\n".join(
        f"[{i}] ({c.source}, page {c.page})\n{c.text}" for i, (c, _) in enumerate(hits, start=1)
    )
    client = get_client()

    # Step 1: generate. Structured output = Gemini must return JSON matching the Quiz schema.
    # Ask for a few extra in case the checker drops some.
    raw = generate(
        client,
        QUIZ_PROMPT.format(n=n + 2, level=level, topic=topic, context=context),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=Quiz
        ),
    )
    # Drop malformed questions instead of showing something broken
    candidates = [q for q in Quiz.model_validate_json(raw).questions
                  if len(q.options) == 4 and 0 <= q.answer_index < 4]

    # Step 2: verify every answer key against the notes
    listing = json.dumps([{"question_number": i, "question": q.question, "options": q.options,
                           "answer_index": q.answer_index} for i, q in enumerate(candidates)])
    review = Review.model_validate_json(generate(
        client,
        VERIFY_PROMPT.format(context=context, questions=listing),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=Review
        ),
    ))
    verdicts = {v.question_number: v for v in review.verdicts}

    questions = []
    for i, q in enumerate(candidates):
        v = verdicts.get(i)
        if v is None or v.verdict == "ambiguous":
            continue
        if v.verdict == "wrong_key":
            if not 0 <= v.correct_index < 4:
                continue
            q.answer_index = v.correct_index
            q.explanation = f"The notes support \"{q.options[v.correct_index]}\"."
        chunk = hits[q.note_number - 1][0] if 1 <= q.note_number <= len(hits) else hits[0][0]
        # "Note [2] says..." means nothing to a student -> "Your notes (page 6) say..."
        q.explanation = re.sub(
            r"\b[Nn]otes? \[(\d+)\]",
            lambda m: (f"your notes (page {hits[int(m.group(1)) - 1][0].page})"
                       if 1 <= int(m.group(1)) <= len(hits) else "your notes"),
            q.explanation)
        q.explanation = re.sub(r"\s?\[\d+\]", "", q.explanation)
        q.explanation = q.explanation[:1].upper() + q.explanation[1:]
        questions.append({**q.model_dump(), "source": chunk.source, "page": chunk.page,
                          "fixed": v.verdict == "wrong_key"})
    return questions[:n]


# ---------- Level 5: important-topics finder (extraction + clustering) ----------

class ExamQuestion(BaseModel):
    question: str = Field(description="the question text, cleaned up, without the 'Q.' prefix")
    marks: int = Field(description="marks if written, e.g. '(4 marks)' -> 4; 0 if not given")
    page: int = Field(description="the page number the question is on")


class QuestionList(BaseModel):
    questions: list[ExamQuestion]


EXTRACT_PROMPT = """Below are pages of a student's notes and/or past exam papers.
List EVERY exam-style question in them: lines starting with "Q.", "Q1", "Ques", numbered
past-paper questions, or lines the teacher marked as exam questions (e.g. with "(4 marks)",
"(exam)", "PYQ", "IMP"). Do NOT invent questions from ordinary notes.
Ignore worked numerical answers; keep only the question itself.

{pages}"""


class TopicName(BaseModel):
    group_number: int
    name: str = Field(description="short topic name, 2-5 words, e.g. 'PEAS of agents'")


class TopicNames(BaseModel):
    topics: list[TopicName]


NAME_PROMPT = """Each numbered group below contains similar exam questions.
Give each group a short, specific syllabus topic name (2-5 words).

{groups}"""


def extract_questions(data: bytes, pages: list[str], source: str) -> list[dict]:
    """Ask Gemini to list every exam question (with marks + page) in a document. Cached."""
    cache_file = CACHE_DIR / f"{hashlib.sha1(data).hexdigest()[:16]}.questions.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    text = "\n\n".join(f"=== Page {i} ===\n{t}" for i, t in enumerate(pages, start=1))
    raw = generate(
        get_client(),
        EXTRACT_PROMPT.format(pages=text),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=QuestionList
        ),
    )
    found = [{**q.model_dump(), "source": source}
             for q in QuestionList.model_validate_json(raw).questions if q.question.strip()]
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")
    return found


def find_topics(questions: list[dict], embedder: SentenceTransformer,
                threshold: float = 0.3) -> list[dict]:
    """Group similar questions with Agglomerative Clustering on their embeddings, name each
    group with Gemini, and rank groups by importance = times asked + total marks / 2."""
    if not questions:
        return []
    vectors = embedder.encode([q["question"] for q in questions], normalize_embeddings=True)

    if len(questions) == 1:
        labels = [0]
    else:
        # Merge questions whose cosine distance (1 - similarity) is below the threshold
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average"
        ).fit_predict(vectors)

    groups: dict[int, list[dict]] = {}
    for q, label in zip(questions, labels):
        groups.setdefault(int(label), []).append(q)
    groups_list = list(groups.values())

    listing = "\n\n".join(
        f"Group {g}:\n" + "\n".join(f"- {q['question']}" for q in qs)
        for g, qs in enumerate(groups_list)
    )
    raw = generate(
        get_client(),
        NAME_PROMPT.format(groups=listing),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=TopicNames
        ),
    )
    names = {t.group_number: t.name for t in TopicNames.model_validate_json(raw).topics}

    topics = []
    for g, qs in enumerate(groups_list):
        marks = sum(q["marks"] for q in qs)
        topics.append({
            "topic": names.get(g, qs[0]["question"][:40]),
            "times_asked": len(qs),
            "total_marks": marks,
            "score": len(qs) + marks / 2,
            "questions": sorted(qs, key=lambda q: -q["marks"]),
        })
    return sorted(topics, key=lambda t: -t["score"])
