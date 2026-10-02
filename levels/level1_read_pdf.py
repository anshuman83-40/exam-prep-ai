"""Level 1: Read a PDF (typed OR handwritten) and split the text into overlapping chunks.

Step 1  Try to read the text layer of each page (works for typed PDFs).
Step 2  If a page has no text (a scanned/photographed page), render it as an image
        and ask Gemini's vision model to transcribe it (OCR). Works on handwriting.
Step 3  Save the page texts to notes_text/ so we never OCR the same file twice.
Step 4  Split the text into overlapping chunks, ready for searching in Level 2.

Run:  python levels/level1_read_pdf.py "your notes.pdf"
"""

import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import pypdfium2 as pdfium
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / "notes_text"
CHUNK_SIZE = 800  # characters per chunk (~150 words)
CHUNK_OVERLAP = 150  # characters shared between neighbouring chunks

OCR_PROMPT = """This is a page of a student's notes (may be handwritten).
Transcribe ALL the text exactly as written, top to bottom.
- Keep headings, bullet points and question numbers.
- Write arrows as "->".
- Fix obvious spelling mistakes only when you are sure of the word.
- For a drawing or diagram, write one line: [Diagram: short description].
Output only the transcribed text, nothing else."""


def ocr_page(client, model: str, page) -> str:
    """Render one PDF page to a JPEG and ask Gemini to read it. Retries if rate-limited."""
    image = page.render(scale=2).to_pil().convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    parts = [types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), OCR_PROMPT]

    for attempt in range(5):
        try:
            return client.models.generate_content(model=model, contents=parts).text or ""
        except Exception as e:  # 429 = too many requests, 503 = server busy
            wait = 10 * (attempt + 1)
            print(f"    retry in {wait}s ({str(e)[:60]}...)")
            time.sleep(wait)
    return ""


def clean(text: str) -> str:
    """Tidy spaces but KEEP line breaks: in notes, which line an item sits under carries
    meaning (e.g. "-> Chatgpt" belongs to the heading above it)."""
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def read_pages(pdf_path: Path) -> list[str]:
    """Return the text of each page (index 0 = page 1), using OCR where needed."""
    # Name the saved file by a fingerprint (hash) of the PDF's contents, so a changed
    # PDF is read again but the same PDF is never OCR'd twice (shared with the app).
    cache_file = CACHE_DIR / f"{hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:16]}.json"
    if cache_file.exists():
        print(f"(using saved text from {cache_file.name})")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    # Step 1: text layer
    pages = [clean(p.extract_text() or "") for p in PdfReader(pdf_path).pages]

    # Step 2: OCR the pages that came back empty
    empty = [i for i, t in enumerate(pages) if not t]
    if empty:
        print(f"{len(empty)} page(s) are images -> reading them with Gemini vision...")
        load_dotenv(PROJECT_DIR / ".env")
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        doc = pdfium.PdfDocument(pdf_path)
        for n, i in enumerate(empty, start=1):
            print(f"  page {i + 1} ({n}/{len(empty)})")
            pages[i] = clean(ocr_page(client, model, doc[i]))

    # Step 3: save so next time is instant (and doesn't use API quota)
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    return pages


def split_into_chunks(text: str) -> list[str]:
    """Cut text into CHUNK_SIZE pieces that overlap by CHUNK_OVERLAP characters."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def main():
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:  # no file given: use the first PDF in the project folder
        pdfs = sorted(PROJECT_DIR.glob("*.pdf"))
        if not pdfs:
            raise SystemExit("No PDF found. Put a PDF in the project folder.")
        pdf_path = pdfs[0]

    pages = read_pages(pdf_path)
    chunks = []  # (page number, chunk text)
    for page_num, text in enumerate(pages, start=1):
        for piece in split_into_chunks(text):
            chunks.append((page_num, piece))

    empty_pages = [i for i, t in enumerate(pages, start=1) if not t]
    print(f"\nFile:        {pdf_path.name}")
    print(f"Pages:       {len(pages)}")
    print(f"Characters:  {sum(len(t) for t in pages):,}")
    print(f"Chunks:      {len(chunks)}  (size {CHUNK_SIZE}, overlap {CHUNK_OVERLAP})")
    if empty_pages:
        print(f"Warning:     no text found on page(s) {empty_pages}")

    if chunks:
        page_num, sample = chunks[min(2, len(chunks) - 1)]
        print(f"\n--- Sample chunk (from page {page_num}) ---\n{sample}")


if __name__ == "__main__":
    main()
