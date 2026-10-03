"""Saves each student's progress (name, quiz scores, units done, exam date, plan, flashcards).

Every student has their own profile, identified by a random code kept in the page link
(?u=<id>), so bookmarking the link brings you back. Profiles are stored:
- in a Supabase database table when SUPABASE_URL and SUPABASE_KEY are set (online deploys,
  where the server's disk is wiped on every restart), otherwise
- as files in profiles/<id>.json (local use; kept out of git).
"""

import json
import math
import os
import re
import uuid
from datetime import date, datetime, timedelta

import requests

from rag import PROJECT_DIR

PROFILE_DIR = PROJECT_DIR / "profiles"
LEGACY_FILE = PROJECT_DIR / "progress.json"  # single-user file from before profiles existed
DEFAULTS = {"name": "", "course": "", "exam_date": None, "quizzes": [], "asked": 0,
            "units_done": [], "plan": [], "cards": [], "mocks": []}
ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")  # ids come from the URL: never trust them as paths


def valid_id(uid: str | None) -> bool:
    return bool(uid) and bool(ID_PATTERN.match(uid))


def _path(uid: str):
    return PROFILE_DIR / f"{uid}.json"


def _supabase() -> tuple[str, dict] | None:
    """(table URL, headers) for the Supabase REST API, or None to use local files."""
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not (url and key):
        return None
    return (f"{url.rstrip('/')}/rest/v1/profiles",
            {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"})


def load(uid: str | None) -> dict | None:
    """The profile for this id, or None if the id is invalid or unknown."""
    if not valid_id(uid):
        return None
    db = _supabase()
    if db:
        table, headers = db
        r = requests.get(table, headers=headers, timeout=10,
                         params={"id": f"eq.{uid}", "select": "data"})
        r.raise_for_status()
        rows = r.json()
        saved = rows[0]["data"] if rows else None
    else:
        try:
            saved = json.loads(_path(uid).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            saved = None
    return {**DEFAULTS, **saved, "id": uid} if saved is not None else None


def save(progress: dict):
    db = _supabase()
    if db:
        table, headers = db
        r = requests.post(table, timeout=10,
                          headers={**headers, "Prefer": "resolution=merge-duplicates"},
                          json={"id": progress["id"], "data": progress,
                                "updated_at": datetime.now().astimezone().isoformat()})
        r.raise_for_status()
        return
    PROFILE_DIR.mkdir(exist_ok=True)
    _path(progress["id"]).write_text(json.dumps(progress, indent=1, ensure_ascii=False),
                                     encoding="utf-8")


def create(name: str, course: str = "", base: dict | None = None) -> dict:
    """Start a new profile (optionally from existing progress, e.g. the legacy file)."""
    progress = {**DEFAULTS, **(base or {}), "name": name.strip()[:40],
                "course": (course.strip() or (base or {}).get("course", ""))[:60],
                "id": uuid.uuid4().hex[:12]}
    save(progress)
    return progress


def retire_legacy():
    """After the old single-user file is moved into a profile, keep it only as a backup."""
    if LEGACY_FILE.exists():
        LEGACY_FILE.replace(LEGACY_FILE.with_name("progress.migrated.json"))


def legacy() -> dict | None:
    """Progress saved before profiles existed (one file for the whole app), if any."""
    try:
        return {**DEFAULTS, **json.loads(LEGACY_FILE.read_text(encoding="utf-8"))}
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def record_quiz(progress: dict, topic: str, score: int, total: int):
    progress["quizzes"].append({"topic": topic, "score": score, "total": total,
                                "at": datetime.now().isoformat(timespec="minutes")})
    save(progress)


def record_mock(progress: dict, scored: float, total: int, n_questions: int):
    progress["mocks"].append({"scored": scored, "total": total, "questions": n_questions,
                              "at": datetime.now().isoformat(timespec="minutes")})
    save(progress)


# ---------- flashcards: SM-2 spaced repetition (the algorithm behind Anki/SuperMemo) ----------

def add_cards(progress: dict, cards: list[dict]) -> int:
    """Add new cards (due today), skipping ones whose front already exists. Returns # added."""
    seen = {c["front"].lower() for c in progress["cards"]}
    added = 0
    for card in cards:
        if card["front"].lower() in seen:
            continue
        progress["cards"].append({**card, "id": uuid.uuid4().hex[:10], "ef": 2.5, "reps": 0,
                                  "interval": 0, "due": date.today().isoformat()})
        seen.add(card["front"].lower())
        added += 1
    save(progress)
    return added


def due_cards(progress: dict, today: date | None = None) -> list[dict]:
    """Cards due today, oldest first; cards just failed ("Again") go to the back of the queue."""
    today = (today or date.today()).isoformat()
    due = [c for c in progress["cards"] if c["due"] <= today]
    return sorted(due, key=lambda c: (c["due"], c.get("last", "")))


FIRST_STEP = {3: 1, 4: 2, 5: 4}  # days after the first successful review: Hard / Good / Easy


def review(card: dict, quality: int, today: date | None = None):
    """Update a card after a review. quality: 0-5 (Again=1, Hard=3, Good=4, Easy=5).

    SM-2 (with Anki-style first steps): a failed card is shown again in this session;
    otherwise the gap grows first step (1/2/4 days) -> 6 days -> previous gap x ease factor.
    The ease factor drops when a card feels hard and rises when it feels easy, so hard
    cards come back more often."""
    today = today or date.today()
    if quality < 3:
        card["reps"], card["interval"] = 0, 0  # due today again, after the other cards
    else:
        card["reps"] += 1
        card["interval"] = (FIRST_STEP[min(quality, 5)] if card["reps"] == 1
                            else 6 if card["reps"] == 2
                            else round(card["interval"] * card["ef"]))
    card["ef"] = max(1.3, card["ef"] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    card["due"] = (today + timedelta(days=card["interval"])).isoformat()
    card["last"] = datetime.now().isoformat(timespec="seconds")


def next_interval(card: dict, quality: int) -> int:
    """Days until the card returns if answered with this quality (shown on the buttons)."""
    trial = dict(card)
    review(trial, quality)
    return trial["interval"]


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


# when this file was loaded: app.py reloads the module if the file on disk changes (new deploy)
LOADED_MTIME = os.path.getmtime(__file__)
