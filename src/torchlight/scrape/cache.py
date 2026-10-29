import json
import sqlite3
import time
import zlib
from pathlib import Path


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(path TEXT PRIMARY KEY, status INTEGER NOT NULL, body BLOB NOT NULL, fetched_at REAL)"
        )
        self._db.commit()

    def get(self, path: str) -> tuple[int, str] | None:
        row = self._db.execute(
            "SELECT status, body FROM responses WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            return None
        return row[0], zlib.decompress(row[1]).decode()

    def get_json(self, path: str) -> dict | None:
        hit = self.get(path)
        if hit is None or hit[0] != 200:
            return None
        return json.loads(hit[1])

    def put(self, path: str, status: int, body: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
            (path, status, zlib.compress(body.encode()), time.time()),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
