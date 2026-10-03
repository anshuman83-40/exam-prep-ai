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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pypdfium2 as pdfium
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps
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
- For a drawing or diagram, write [Diagram: description] AND write out every number,
  label, grid row, node name and formula inside it.
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


OCR_BATCH = 3  # pages per Gemini request (fewer requests -> fewer free-tier rate limits)
OCR_WORKERS = 3  # requests running at the same time

OCR_BATCH_PROMPT = """These {n} images are consecutive pages of a student's notes (may be
handwritten). Transcribe EACH page separately, top to bottom, following these rules:
- Keep headings, bullet points, question numbers and line breaks.
- Write arrows as "->".
- Fix obvious spelling mistakes only when you are sure of the word.
- Give every page the same full detail you would give it alone; never shorten a page.
- For a drawing or diagram, write [Diagram: description] AND write out every number,
  label, grid row, node name and formula inside it (e.g. "[Diagram: 3x3 grid 2 8 3 / 1 _ 4 /
  7 6 5, arrows R L U D]").
Return exactly {n} strings in "pages": the text of image 1, image 2, ... in order."""


class PageTexts(BaseModel):
    pages: list[str]


MAX_SIDE = 2000  # px on the long side: plenty to read handwriting, small enough for any server


def page_image(doc, i: int) -> types.Part:
    """Render one PDF page as a JPEG for OCR. The zoom is capped by pixel size, because
    phone-scanner PDFs use huge pages (e.g. 3024x4032): rendering those at a fixed 2x zoom
    needs ~200 MB per page and can crash a small server."""
    width, height = doc[i].get_size()
    scale = min(2.0, MAX_SIDE / max(width, height, 1))
    buf = io.BytesIO()
    doc[i].render(scale=scale).to_pil().convert("RGB").save(buf, format="JPEG", quality=85)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def file_kind(data: bytes) -> str | None:
    """'pdf', an image MIME type (photos of notes/papers), or None for anything else.
    Checked from the file's content, not its name: phones often give odd names/types."""
    if b"%PDF" in data[:1024]:
        return "pdf"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp" and data[8:12] in (b"heic", b"heix", b"heif", b"mif1", b"msf1", b"hevc"):
        return "image/heic"  # iPhone photos
    return None


def photo_part(data: bytes, kind: str) -> types.Part:
    """A photo ready for OCR: turned upright (phones store rotation separately) and shrunk."""
    if kind == "image/heic":  # Gemini reads HEIC directly; Pillow can't open it
        return types.Part.from_bytes(data=data, mime_type=kind)
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def ocr_job(client, idxs: list[int], images: list, save) -> None:
    """OCR a few pages in one request (falls back to one page per request if the reply
    doesn't have one text per page), then hand the results to `save`. Runs in a worker thread."""
    results: dict[int, str] = {}
    if len(idxs) > 1:
        try:
            raw = generate(client, [*images, OCR_BATCH_PROMPT.format(n=len(idxs))],
                           config=types.GenerateContentConfig(
                               response_mime_type="application/json", response_schema=PageTexts))
            texts = PageTexts.model_validate_json(raw).pages
            if len(texts) == len(idxs):
                results = {i: clean(t) for i, t in zip(idxs, texts)}
        except Exception:
            results = {}
    for i, image in zip(idxs, images):
        if i not in results:
            results[i] = clean(generate(client, [image, OCR_PROMPT]))
    save(results)


def read_pages(data: bytes, on_progress=None) -> list[str]:
    """Return the text of each page. Pages without a text layer are OCR'd with Gemini,
    several pages per request and several requests in parallel.

    Every finished page is saved straight away (notes_text/<hash>.partial.json), so if the
    run is interrupted (phone screen locks, user taps elsewhere) it resumes where it stopped.
    The finished result is cached by file content, so each PDF is only OCR'd once.
    A photo (JPG/PNG/WEBP/HEIC) is read as a single page."""
    digest = hashlib.sha1(data).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{digest}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    kind = file_kind(data)
    if kind is None:
        raise ValueError("this file isn't a PDF or a photo (JPG, PNG, HEIC)")
    CACHE_DIR.mkdir(exist_ok=True)
    if kind != "pdf":
        if on_progress:
            on_progress(0, 1)
        pages = [clean(generate(get_client(), [photo_part(data, kind), OCR_PROMPT]))]
        cache_file.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
        return pages

    pages = [clean(p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages]
    empty = [i for i, t in enumerate(pages) if not t]
    if empty:
        partial_file = CACHE_DIR / f"{digest}.partial.json"
        try:
            done = {int(k): v for k, v in json.loads(partial_file.read_text(encoding="utf-8")).items()}
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            done = {}
        lock = threading.Lock()

        def save(results: dict[int, str]):  # called from worker threads
            with lock:
                done.update(results)
                partial_file.write_text(json.dumps({str(k): v for k, v in done.items()},
                                                   ensure_ascii=False), encoding="utf-8")

        todo = [i for i in empty if i not in done]
        if on_progress:
            on_progress(len(empty) - len(todo), len(empty))
        if todo:
            client = get_client()
            doc = pdfium.PdfDocument(data)  # rendering stays on this thread (pdfium isn't thread-safe)
            pool = ThreadPoolExecutor(max_workers=OCR_WORKERS)
            try:
                futures = []
                for b in range(0, len(todo), OCR_BATCH):
                    idxs = todo[b:b + OCR_BATCH]
                    futures.append(pool.submit(ocr_job, client, idxs,
                                               [page_image(doc, i) for i in idxs], save))
                for future in as_completed(futures):
                    future.result()  # re-raise a real failure (e.g. invalid API key)
                    if on_progress:
                        with lock:
                            finished = sum(i in done for i in empty)
                        on_progress(finished, len(empty))
            finally:
                # if interrupted: drop queued batches; running ones still finish and save()
                pool.shutdown(wait=False, cancel_futures=True)
        for i in empty:
            pages[i] = done[i]
        partial_file.unlink(missing_ok=True)

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
Marks: use the marks written next to a question ("(5)", "[10M]", "5 marks"); if a section says
"each question carries N marks" or "N x M = total", give each question in it N marks.
For "a)/b)" sub-parts with their own marks, list each sub-part separately.
For "Q3 OR Q4" choices, list both.

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


# ---------- Study plan: syllabus + PYQs -> chapter priority & weightage ----------

class Unit(BaseModel):
    number: int = Field(description="unit/chapter number as in the syllabus, 1-based")
    title: str
    topics: list[str] = Field(description="the topics listed under this unit")
    hours: int = Field(description="lecture hours/lectures for the unit if listed, else 0")


class Syllabus(BaseModel):
    course: str = Field(description="course name/code, or '' if not found")
    units: list[Unit]


SYLLABUS_PROMPT = """Below is a course handout / syllabus. List the course's units (or chapters /
modules / lessons) in order, with the topics under each and the lecture hours if given.
Only include teaching units, not evaluation schemes, reference books or lab lists.

{pages}"""


class Mapping(BaseModel):
    question_number: int
    unit_number: int = Field(description="the unit the question belongs to; 0 if it fits none")


class Mappings(BaseModel):
    mappings: list[Mapping]


MAP_PROMPT = """Match each exam question to the syllabus unit it tests. Use the unit's topics,
not just its title. If a question clearly fits no unit, use 0.

Syllabus units:
{units}

Questions:
{questions}"""


def extract_syllabus(data: bytes, pages: list[str]) -> dict:
    """Ask Gemini for the units/chapters (with topics + hours) in a course handout. Cached."""
    cache_file = CACHE_DIR / f"{hashlib.sha1(data).hexdigest()[:16]}.syllabus.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    text = "\n\n".join(f"=== Page {i} ===\n{t}" for i, t in enumerate(pages, start=1))
    raw = generate(
        get_client(), SYLLABUS_PROMPT.format(pages=text),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=Syllabus),
    )
    syllabus = Syllabus.model_validate_json(raw).model_dump()
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(syllabus, ensure_ascii=False, indent=1), encoding="utf-8")
    return syllabus


def map_questions(questions: list[dict], units: list[dict], batch: int = 60) -> list[int]:
    """Classify each question into a syllabus unit number (0 = not in syllabus)."""
    unit_text = "\n".join(f"Unit {u['number']}: {u['title']} — {', '.join(u['topics'])}"
                          for u in units)
    valid = {u["number"] for u in units}
    result = [0] * len(questions)
    client = get_client()
    for start in range(0, len(questions), batch):
        part = questions[start:start + batch]
        listing = "\n".join(f"{i}. {q['question']}" for i, q in enumerate(part, start=1))
        raw = generate(
            client, MAP_PROMPT.format(units=unit_text, questions=listing),
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=Mappings),
        )
        for m in Mappings.model_validate_json(raw).mappings:
            if 1 <= m.question_number <= len(part):
                result[start + m.question_number - 1] = m.unit_number if m.unit_number in valid else 0
    return result


def study_plan(syllabus: dict, papers: dict[str, list[dict]]) -> dict:
    """Combine the syllabus with PYQ papers ({paper name: questions}) into a priority table.

    Weightage = the unit's share of all PYQ marks (question count if papers show no marks).
    Ranking: weightage, then how many papers asked it; without PYQs, lecture hours; without
    hours, syllabus order. Priority: High if weightage >= 1.25x the average unit share,
    Low if <= 0.6x, else Medium."""
    units = syllabus["units"]
    questions = [{**q, "paper": name} for name, qs in papers.items() for q in qs]
    unit_of = map_questions(questions, units) if questions and units else []
    marked = [q["marks"] for q in questions if q["marks"] > 0]
    use_marks = bool(marked)
    avg_marks = sum(marked) / len(marked) if marked else 1

    def weight(q):  # a question without marks counts as an average question
        if not use_marks:
            return 1
        return q["marks"] or avg_marks

    total = sum(weight(q) for q, u in zip(questions, unit_of) if u) or 0
    total_hours = sum(u["hours"] for u in units)
    rows = []
    for u in units:
        qs = [q for q, n in zip(questions, unit_of) if n == u["number"]]
        share = (sum(weight(q) for q in qs) / total if total
                 else u["hours"] / total_hours if total_hours else 0)
        asked_in = len({q["paper"] for q in qs})
        rows.append({
            "unit": u["number"], "title": u["title"], "topics": u["topics"], "hours": u["hours"],
            "questions": sorted(qs, key=lambda q: -q["marks"]), "n_questions": len(qs),
            "marks": sum(q["marks"] for q in qs), "weightage": share, "asked_in": asked_in,
        })

    basis = "pyq" if total else "hours" if total_hours else "order"
    rows.sort(key=lambda r: (-r["weightage"], -r["asked_in"], r["unit"]))
    avg = 1 / len(rows) if rows else 0
    for order, r in enumerate(rows, start=1):
        r["order"] = order
        if basis == "order":
            r["priority"] = "Medium"
        elif r["weightage"] >= 1.25 * avg:
            r["priority"] = "High"
        elif r["weightage"] <= 0.6 * avg:
            r["priority"] = "Low"
        else:
            r["priority"] = "Medium"
        if basis == "pyq":
            r["why"] = (f"{r['weightage']:.0%} of PYQ {'marks' if use_marks else 'questions'}, "
                        f"asked in {r['asked_in']}/{len(papers)} paper(s)" if r["n_questions"]
                        else "not asked in these papers")
        elif basis == "hours":
            r["why"] = f"{r['hours']} lecture hours ({r['weightage']:.0%} of course)"
        else:
            r["why"] = "syllabus order (add PYQs for weightage)"

    unmapped = [q for q, n in zip(questions, unit_of) if n == 0]
    return {"course": syllabus.get("course", ""), "rows": rows, "basis": basis,
            "papers": len(papers), "n_questions": len(questions), "total_marks":
            sum(q["marks"] for q in questions), "unmapped": unmapped, "use_marks": use_marks}


# ---------- shared: pick note excerpts for a whole-syllabus task ----------

def coverage_context(kb: KnowledgeBase, units: list[dict] | None = None,
                     max_chunks: int = 18) -> list[Chunk]:
    """Excerpts spread over the whole course: the best chunks for each study-plan unit
    (more for high-weightage units) or, without a plan, chunks evenly spaced through the notes."""
    if not kb.chunks:
        return []
    if units:
        picked: list[Chunk] = []
        for u in units:
            per_unit = max(1, round(u.get("weightage", 0) * max_chunks)) if u.get("weightage") else 3
            for c, _ in kb.search(u["title"], k=per_unit):
                if c not in picked:
                    picked.append(c)
        return picked[: max_chunks + len(units)]
    step = max(1, len(kb.chunks) // max_chunks)
    return kb.chunks[::step][:max_chunks]


def numbered(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.source}, page {c.page})\n{c.text}"
                       for i, c in enumerate(chunks, start=1))


# ---------- Mock exam: generate a paper in the exam pattern, then grade written answers ----------

class ExamQ(BaseModel):
    question: str
    marks: int
    key_points: list[str] = Field(description="the points a full-marks answer must contain")
    model_answer: str = Field(description="a concise full-marks answer written from the notes")
    note_number: int = Field(description="the [n] note the question is mainly based on")


class ExamPaper(BaseModel):
    questions: list[ExamQ]


MOCK_PROMPT = """You are setting a university exam paper. Using ONLY the numbered notes below,
write questions in exactly this pattern:
{pattern}
- Short questions (1-3 marks): definitions, differences, one-line facts.
- Long questions (5+ marks): explain / apply / compare / solve, like real end-semester questions.
- Spread questions across as many different topics in the notes as possible; no repeats.
- For each question give the key points a full-marks answer needs (about one point per
  1-2 marks) and a concise model answer, both strictly from the notes.

Notes:
{context}"""


class Grade(BaseModel):
    question_number: int
    marks_awarded: float
    points_covered: list[str]
    points_missed: list[str]
    feedback: str = Field(description="one or two sentences of specific advice")


class Grades(BaseModel):
    grades: list[Grade]


GRADE_PROMPT = """You are a fair university examiner. Mark each student answer against its
key points and the notes. Give marks in steps of 0.5, never above the question's marks.
Award marks for correct ideas in the student's own words; ignore spelling and grammar.
A blank or irrelevant answer gets 0.

Notes:
{context}

Questions, key points and student answers:
{answers}"""


def make_mock_exam(kb: KnowledgeBase, pattern: list[tuple[int, int]],
                   units: list[dict] | None = None) -> dict:
    """pattern = [(number of questions, marks each), ...] e.g. [(5, 2), (3, 10)]."""
    chunks = coverage_context(kb, units)
    if not chunks:
        return {"questions": [], "chunks": []}
    pattern_text = "\n".join(f"- {n} question(s) of {m} marks each" for n, m in pattern)
    raw = generate(
        get_client(), MOCK_PROMPT.format(pattern=pattern_text, context=numbered(chunks)),
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           response_schema=ExamPaper),
    )
    # keep the requested pattern exactly, even if the model wrote extra questions
    pool = ExamPaper.model_validate_json(raw).questions
    questions = []
    for n, m in pattern:
        same = [q for q in pool if q.marks == m][:n]
        for q in same:
            pool.remove(q)
        questions += same
    out = []
    for q in questions:
        c = chunks[q.note_number - 1] if 1 <= q.note_number <= len(chunks) else chunks[0]
        out.append({**q.model_dump(), "source": c.source, "page": c.page})
    return {"questions": out, "chunks": [(c.source, c.page, c.text) for c in chunks]}


def transcribe_answer(image: bytes, kind: str) -> str:
    """Read a photo of a handwritten answer (exam answers are handwritten)."""
    part = photo_part(image, kind)  # upright and shrunk, like note photos
    prompt = ("Transcribe this handwritten exam answer exactly, keeping its structure. "
              "Describe any diagram in one line as [Diagram: ...]. Output only the text.")
    return generate(get_client(), [part, prompt]).strip()


def grade_exam(exam: dict, answers: list[str]) -> list[dict]:
    """Mark every written answer in one call. Returns per-question marks + feedback."""
    context = "\n\n".join(f"[{i}] ({s}, page {p})\n{t}"
                          for i, (s, p, t) in enumerate(exam["chunks"], start=1))
    listing = json.dumps([
        {"question_number": i, "question": q["question"], "marks": q["marks"],
         "key_points": q["key_points"], "student_answer": a.strip() or "(blank)"}
        for i, (q, a) in enumerate(zip(exam["questions"], answers), start=1)], ensure_ascii=False)
    raw = generate(
        get_client(), GRADE_PROMPT.format(context=context, answers=listing),
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           response_schema=Grades),
    )
    by_number = {g.question_number: g for g in Grades.model_validate_json(raw).grades}
    results = []
    for i, (q, a) in enumerate(zip(exam["questions"], answers), start=1):
        g = by_number.get(i)
        if not a.strip():  # never trust a model to give marks for nothing
            results.append({"marks": 0.0, "covered": [], "missed": q["key_points"],
                            "feedback": "Not attempted."})
        elif g is None:
            results.append({"marks": 0.0, "covered": [], "missed": [],
                            "feedback": "Could not be graded, try again."})
        else:
            marks = min(max(round(g.marks_awarded * 2) / 2, 0), q["marks"])
            results.append({"marks": marks, "covered": g.points_covered,
                            "missed": g.points_missed, "feedback": g.feedback})
    return results


# ---------- Flashcards ----------

class Card(BaseModel):
    front: str = Field(description="a short question or term, max ~15 words")
    back: str = Field(description="the answer, 1-3 short lines")
    note_number: int


class Cards(BaseModel):
    cards: list[Card]


CARDS_PROMPT = """Make {n} flashcards for revising "{topic}" from ONLY the numbered notes below.
- One fact per card: definitions, differences, steps, formulas, examples.
- Front: a short question or term. Back: a short, exact answer from the notes.
- No duplicates, no trivial cards, nothing that isn't in the notes.

Notes:
{context}"""


def make_flashcards(topic: str, kb: KnowledgeBase, n: int = 10) -> list[dict]:
    hits = kb.search(topic, k=6)
    if not hits:
        return []
    chunks = [c for c, _ in hits]
    raw = generate(
        get_client(), CARDS_PROMPT.format(n=n, topic=topic, context=numbered(chunks)),
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           response_schema=Cards),
    )
    out = []
    for card in Cards.model_validate_json(raw).cards[:n]:
        c = chunks[card.note_number - 1] if 1 <= card.note_number <= len(chunks) else chunks[0]
        out.append({"front": card.front.strip(), "back": card.back.strip(), "topic": topic,
                    "source": c.source, "page": c.page})
    return out
