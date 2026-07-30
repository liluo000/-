from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class HistoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    scene_type TEXT NOT NULL,
                    accepted_count INTEGER NOT NULL,
                    review_count INTEGER NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )

    def save(self, filename: str, scene_type: str, rows: list[dict[str, Any]], result: dict[str, Any]) -> int:
        accepted = sum(1 for row in rows if row.get("状态") == "accepted")
        review = sum(1 for row in rows if row.get("状态") == "review")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO analyses(created_at, filename, scene_type, accepted_count, review_count, result_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    filename,
                    scene_type,
                    accepted,
                    review,
                    json.dumps({"rows": rows, "analysis": result}, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, created_at, filename, scene_type, accepted_count, review_count FROM analyses ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, record_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM analyses WHERE id = ?", (record_id,)).fetchone()
        if not row:
            return None
        value = dict(row)
        value["result"] = json.loads(value.pop("result_json"))
        return value
