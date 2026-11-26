"""Optional local persistence for user supplied review analyses."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "ctrl_trip.sqlite3"


def save_reviews(frame, path: Path = DB_PATH) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [(str(row.get("destination", "")), str(row.get("review", "")),
                str(row.get("sentiment", "")), datetime.now(timezone.utc).isoformat())
               for _, row in frame.iterrows() if str(row.get("review", "")).strip()]
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, destination TEXT, review TEXT NOT NULL, sentiment TEXT, saved_at TEXT NOT NULL)")
        conn.executemany("INSERT INTO reviews(destination, review, sentiment, saved_at) VALUES (?, ?, ?, ?)", records)
    return len(records)


def review_count(path: Path = DB_PATH) -> int:
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()
    return int(row[0])
