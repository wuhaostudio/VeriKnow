from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from veriknow.config import Config, ensure_data_dirs
from veriknow.schemas import PublicationJob, RunRecord, TaskSpec, UserPreference, now_iso


class MemoryStore:
    def __init__(self, config: Config):
        self.config = config
        ensure_data_dirs(config)
        self.path = config.database_path
        self.initialize()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    raw_request TEXT NOT NULL,
                    task_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    task_id TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS publication_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_path TEXT NOT NULL,
                    target TEXT NOT NULL,
                    status TEXT NOT NULL,
                    job_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )

    def create_run(self, raw_request: str, task: TaskSpec) -> RunRecord:
        created_at = now_iso()
        run_id = self._new_run_id()
        run_dir = self.run_dir(run_id)
        (run_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)

        task_path = run_dir / "task.json"
        task_path.write_text(
            json.dumps(task.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts = {"task": str(task_path)}
        record = RunRecord(
            run_id=run_id,
            raw_request=raw_request,
            task=task,
            status="created",
            artifacts=artifacts,
            created_at=created_at,
            updated_at=created_at,
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, raw_request, task_json, status, artifacts_json,
                    created_at, updated_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.raw_request,
                    json.dumps(task.to_dict(), ensure_ascii=False),
                    record.status,
                    json.dumps(record.artifacts, ensure_ascii=False),
                    record.created_at,
                    record.updated_at,
                    record.completed_at,
                ),
            )
        return record

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> RunRecord:
        record = self.get_run(run_id)
        if record is None:
            raise KeyError(f"run not found: {run_id}")
        if status is not None:
            record.status = status
        if artifacts:
            record.artifacts.update(artifacts)
        record.updated_at = now_iso()
        if record.status in {"completed", "failed"} and record.completed_at is None:
            record.completed_at = record.updated_at

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, artifacts_json = ?, updated_at = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (
                    record.status,
                    json.dumps(record.artifacts, ensure_ascii=False),
                    record.updated_at,
                    record.completed_at,
                    record.run_id,
                ),
            )
        return record

    def complete_run(self, run_id: str, artifacts: dict[str, str] | None = None) -> RunRecord:
        return self.update_run(run_id, status="completed", artifacts=artifacts)

    def list_runs(self, limit: int = 20) -> list[RunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, raw_request, task_json, status, artifacts_json,
                       created_at, updated_at, completed_at
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, raw_request, task_json, status, artifacts_json,
                       created_at, updated_at, completed_at
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def append_preference(self, preference: UserPreference) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO preferences (key, value, source, task_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    preference.key,
                    preference.value,
                    preference.source,
                    preference.task_id,
                    preference.created_at,
                ),
            )

    def list_preferences(self, limit: int = 50) -> list[UserPreference]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT key, value, source, task_id, created_at
                FROM preferences
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            UserPreference(
                key=row["key"],
                value=row["value"],
                source=row["source"],
                task_id=row["task_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def append_publication_job(self, job: PublicationJob) -> PublicationJob:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO publication_jobs (
                    document_path, target, status, job_json, created_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job.document_path,
                    job.target,
                    job.status,
                    json.dumps(job.to_dict(), ensure_ascii=False),
                    job.created_at,
                    job.completed_at,
                ),
            )
        return job

    def list_publication_jobs(self, limit: int = 20) -> list[PublicationJob]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_json
                FROM publication_jobs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [PublicationJob.from_dict(json.loads(row["job_json"])) for row in rows]

    def is_approved_knowledge_document(self, document_path: str | Path) -> bool:
        candidate = Path(document_path).resolve()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT artifacts_json
                FROM runs
                WHERE status = 'completed'
                """
            ).fetchall()

        for row in rows:
            artifacts = json.loads(row["artifacts_json"])
            knowledge_document = artifacts.get("knowledge_document")
            if not knowledge_document:
                continue
            if Path(knowledge_document).resolve() == candidate:
                return True
        return False

    def run_dir(self, run_id: str) -> Path:
        return self.config.runs_dir / run_id

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            raw_request=row["raw_request"],
            task=TaskSpec.from_dict(json.loads(row["task_json"])),
            status=row["status"],
            artifacts=json.loads(row["artifacts_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    def _new_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"run-{timestamp}-{uuid.uuid4().hex[:8]}"
