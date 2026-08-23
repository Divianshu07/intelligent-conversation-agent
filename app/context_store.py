from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from app.models import ContextScope


SCOPES: tuple[ContextScope, ...] = ("category", "merchant", "customer", "trigger")


@dataclass(frozen=True)
class StoredContext:
    scope: ContextScope
    context_id: str
    version: int
    payload: dict[str, Any]
    stored_at: datetime


@dataclass(frozen=True)
class WriteResult:
    accepted: bool
    stored_at: datetime | None = None
    current_version: int | None = None


class ContextStore:
    """Small SQLite-backed, versioned context store for one challenge run."""

    def __init__(self, database_path: str) -> None:
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS contexts (
                scope TEXT NOT NULL,
                context_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                stored_at TEXT NOT NULL,
                PRIMARY KEY (scope, context_id)
            )
            """
        )
        self._connection.commit()

    def put(self, scope: ContextScope, context_id: str, version: int, payload: dict[str, Any]) -> WriteResult:
        """Atomically insert or replace context only when its version is newer."""
        now = datetime.now(UTC)
        with self._lock, self._connection:
            current = self._connection.execute(
                "SELECT version FROM contexts WHERE scope = ? AND context_id = ?",
                (scope, context_id),
            ).fetchone()
            if current is not None and current["version"] >= version:
                return WriteResult(accepted=False, current_version=int(current["version"]))

            self._connection.execute(
                """
                INSERT INTO contexts (scope, context_id, version, payload_json, stored_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scope, context_id) DO UPDATE SET
                    version = excluded.version,
                    payload_json = excluded.payload_json,
                    stored_at = excluded.stored_at
                """,
                (scope, context_id, version, json.dumps(payload, ensure_ascii=False), now.isoformat()),
            )
        return WriteResult(accepted=True, stored_at=now)

    def get(self, scope: ContextScope, context_id: str) -> StoredContext | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT scope, context_id, version, payload_json, stored_at FROM contexts WHERE scope = ? AND context_id = ?",
                (scope, context_id),
            ).fetchone()
        if row is None:
            return None
        return StoredContext(
            scope=row["scope"],
            context_id=row["context_id"],
            version=int(row["version"]),
            payload=json.loads(row["payload_json"]),
            stored_at=datetime.fromisoformat(row["stored_at"]),
        )

    def counts(self) -> dict[ContextScope, int]:
        with self._lock:
            rows = self._connection.execute("SELECT scope, COUNT(*) AS count FROM contexts GROUP BY scope").fetchall()
        result: dict[ContextScope, int] = {scope: 0 for scope in SCOPES}
        for row in rows:
            # Scopes outside the four challenge context scopes (e.g. the
            # internal "conversation" scope used by ReplyService) are not
            # part of the reported context counts.
            if row["scope"] in result:
                result[row["scope"]] = int(row["count"])
        return result

    def clear_all(self) -> None:
        """Remove every stored row, across all scopes (including internal
        scopes such as "conversation"), so a new test run starts clean.
        Safe to call on an already-empty store."""
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM contexts")

    def close(self) -> None:
        with self._lock:
            self._connection.close()
