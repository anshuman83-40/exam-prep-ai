"""Level 6: Measure how good the assistant actually is (evaluation).

1. Test set: Gemini writes student-style questions for sampled chunks of the notes, using
   DIFFERENT words than the notes (so search can't just copy-match). The page each question
   came from is the "correct answer". Saved to eval/eval_set.json so every run is comparable.
   You can add your own questions to that file too.
2. Retrieval metrics for 4 search setups:
     Hit@1  = % of questions where the right page is the #1 result
     Hit@5  = % where the right page is anywhere in the top 5 (what the LLM gets to read)
     MRR    = Mean Reciprocal Rank: 1/rank of the first right result, averaged (1.0 = perfect)
3. Hallucination test: off-topic questions must be refused, not answered from thin air.

Run:  python levels/level6_evaluate.py            (add --refresh to rebuild the test set)
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import rag.py from project root

from google.genai import types
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag import (EMBED_MODEL, PROJECT_DIR, RERANK_MODEL, KnowledgeBase, answer, generate,
                 get_client, make_chunks, read_pages)

EVAL_DIR = PROJECT_DIR / "eval"
EVAL_SET = EVAL_DIR / "eval_set.json"
N_SYNTHETIC = 30

# Hand-written checks (pages verified against the handwritten notes)
MANUAL = [
    {"question": "Explain PEAS with an example", "pages": [5]},
    {"question": "What are the different types of agents?", "pages": [4]},
    {"question": "difference between structured and unstructured data", "pages": [2]},
    {"question": "which agent type is chatgpt an example of", "pages": [4]},
]

OFF_TOPIC = [
    "What is the capital of France?",
    "Who won the 2011 cricket world cup?",
    "Explain photosynthesis in plants",
    "What is the boiling point of water in Fahrenheit?",
    "Write a poem about the moon",
]

QGEN_PROMPT = """For each numbered excerpt from a student's notes, write ONE question a student
might ask that this excerpt answers. Rules:
- Use different wording than the excerpt (paraphrase, don't copy phrases).
- Ask about the main idea, not a tiny detail. Natural student style, e.g. "how does X work".
- Skip an excerpt (leave it out) if it has no clear fact to ask about.

{excerpts}"""


class GenQ(BaseModel):
    excerpt_number: int
    question: str


class GenQs(BaseModel):
    questions: list[GenQ]


def build_eval_set(chunks) -> list[dict]:
    random.seed(42)  # same sample every time
    pool = [c for c in chunks if len(c.text) > 250]
    sample = random.sample(pool, min(N_SYNTHETIC, len(pool)))
    excerpts = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(sample, start=1))
    raw = generate(get_client(), QGEN_PROMPT.format(excerpts=excerpts),
                   config=types.GenerateContentConfig(response_mime_type="application/json",
                                                      response_schema=GenQs))
    items = [{"question": q.question, "pages": [sample[q.excerpt_number - 1].page],
              "type": "synthetic"}
             for q in GenQs.model_validate_json(raw).questions
             if 1 <= q.excerpt_number <= len(sample)]
    items += [{**m, "type": "manual"} for m in MANUAL]
    EVAL_DIR.mkdir(exist_ok=True)
    EVAL_SET.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    return items


def retrieval_metrics(kb: KnowledgeBase, items: list[dict], k: int = 5) -> dict:
    hit1 = hit5 = rr = 0.0
    misses = []
    for item in items:
        pages = [c.page for c, _ in kb.search(item["question"], k)]
        rank = next((r for r, p in enumerate(pages, start=1) if p in item["pages"]), None)
        hit1 += rank == 1
        hit5 += rank is not None
        rr += 1 / rank if rank else 0
        if rank is None:
            misses.append(item["question"])
    n = len(items)
    return {"Hit@1": hit1 / n, "Hit@5": hit5 / n, "MRR": rr / n, "misses": misses}


def main():
    pdf = sorted(PROJECT_DIR.glob("*.pdf"))[0]
    chunks = make_chunks(read_pages(pdf.read_bytes()), pdf.name)
    print(f"Notes: {pdf.name}, {len(chunks)} chunks")

    if EVAL_SET.exists() and "--refresh" not in sys.argv:
        items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    else:
        print("Building test set with Gemini...")
        items = build_eval_set(chunks)
    print(f"Test set: {len(items)} questions ({EVAL_SET.relative_to(PROJECT_DIR)})\n")

    embedder, reranker = SentenceTransformer(EMBED_MODEL), CrossEncoder(RERANK_MODEL)
    small_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    setups = {
        "1. Vector search only": dict(reranker=None, hybrid=False),
        "2. Hybrid (vector + BM25)": dict(reranker=None, hybrid=True),
        "3. Hybrid + small re-ranker": dict(reranker=small_reranker, hybrid=True),
        "4. Hybrid + re-ranker (app)": dict(reranker=reranker, hybrid=True),
    }
    results = {}
    print(f"{'Setup':<30}{'Hit@1':>8}{'Hit@5':>8}{'MRR':>8}{'ms/query':>10}")
    for name, opts in setups.items():
        kb = KnowledgeBase(embedder, **opts)
        kb.add(chunks)
        start = time.perf_counter()
        m = retrieval_metrics(kb, items)
        m["ms_per_query"] = (time.perf_counter() - start) * 1000 / len(items)
        results[name] = m
        print(f"{name:<30}{m['Hit@1']:>8.0%}{m['Hit@5']:>8.0%}{m['MRR']:>8.2f}"
              f"{m['ms_per_query']:>10.0f}")

    print("\nStill missed by the app's setup:")
    for q in results["4. Hybrid + re-ranker (app)"]["misses"] or ["(none)"]:
        print("  -", q)

    print("\nHallucination test (off-topic questions should be refused):")
    kb = KnowledgeBase(embedder, reranker)
    kb.add(chunks)
    refused = 0
    for q in OFF_TOPIC:
        reply, _ = answer(q, kb)
        ok = "couldn't find" in reply.lower()
        refused += ok
        print(f"  {'✅ refused ' if ok else '❌ answered'}  {q}")
    results["refusal_rate"] = refused / len(OFF_TOPIC)
    print(f"Refusal rate: {results['refusal_rate']:.0%}")

    (EVAL_DIR / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nSaved to {(EVAL_DIR / 'results.json').relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
