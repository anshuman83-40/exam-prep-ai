"""Saves the student's progress (quiz scores, units done, exam date, plan) between sessions.

Stored as progress.json in the project folder (kept out of git).
"""

import json
import math
from datetime import date, datetime, timedelta

from rag import PROJECT_DIR

PROGRESS_FILE = PROJECT_DIR / "progress.json"
DEFAULTS = {"name": "Anshuman", "exam_date": None, "quizzes": [], "asked": 0,
            "units_done": [], "plan": []}


def load() -> dict:
    try:
        saved = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        saved = {}
    return {**DEFAULTS, **saved}


def save(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=1, ensure_ascii=False), encoding="utf-8")


def record_quiz(progress: dict, topic: str, score: int, total: int):
    progress["quizzes"].append({"topic": topic, "score": score, "total": total,
                                "at": datetime.now().isoformat(timespec="minutes")})
    save(progress)


def points(progress: dict) -> int:
    """10 points per correct quiz answer."""
    return 10 * sum(q["score"] for q in progress["quizzes"])


def accuracy(progress: dict) -> float | None:
    total = sum(q["total"] for q in progress["quizzes"])
    return sum(q["score"] for q in progress["quizzes"]) / total if total else None


def ago(iso: str) -> str:
    """'5 min ago', '2 h ago', '3 days ago'."""
    minutes = (datetime.now() - datetime.fromisoformat(iso)).total_seconds() / 60
    if minutes < 60:
        return f"{max(1, int(minutes))} min ago"
    if minutes < 60 * 24:
        return f"{int(minutes // 60)} h ago"
    return f"{int(minutes // (60 * 24))} days ago"


def schedule(plan: list[dict], exam: date, today: date | None = None) -> dict[date, list[dict]]:
    """Spread plan units over the days before the exam, in priority order, giving each unit a
    number of days proportional to its weightage. The day before the exam is for revision."""
    today = today or date.today()
    study_days = max(1, (exam - today).days - 1)  # keep the last day for revision
    days = [today + timedelta(days=i) for i in range(study_days)]
    out: dict[date, list[dict]] = {d: [] for d in days}
    weights = [max(u["weightage"], 0.02) for u in plan]  # every unit gets some time
    total = sum(weights) or 1
    before = 0.0
    for unit, w in zip(plan, weights):
        start = min(int(before / total * study_days), study_days - 1)
        before += w
        end = max(start, min(math.ceil(before / total * study_days) - 1, study_days - 1))
        for d in days[start:end + 1]:
            out[d].append(unit)
    if (exam - today).days >= 1:
        out[exam - timedelta(days=1)] = [{"title": "Revision + PYQ practice", "revision": True}]
    return out
