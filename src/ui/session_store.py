"""Session persistence layer — JSON-file-backed CRUD for conversation histories."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


STORAGE_DIR = Path("data/sessions")
INDEX_FILE = STORAGE_DIR / "index.json"
MAX_SESSIONS = 50


@dataclass
class SessionRecord:
    session_id: str
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    task: str = ""
    dataset_path: str = ""
    messages: list[dict] = field(default_factory=list)
    metrics: dict | None = None
    code_history: list[str] = field(default_factory=list)
    final_report: str = ""
    completed: bool = False


class SessionStore:
    """JSON-file-backed session persistence with auto-indexing and max-limit pruning."""

    def __init__(self, storage_dir: Path | None = None, max_sessions: int = MAX_SESSIONS):
        self.storage_dir = storage_dir or STORAGE_DIR
        self.max_sessions = max_sessions
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> list[str]:
        if not INDEX_FILE.exists():
            return []
        try:
            data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_index(self, session_ids: list[str]) -> None:
        INDEX_FILE.write_text(
            json.dumps(session_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[SessionRecord]:
        """Return sessions sorted by updated_at descending."""
        ids = self._load_index()
        records: list[SessionRecord] = []
        for sid in ids:
            rec = self._load(sid)
            if rec is not None:
                records.append(rec)
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records

    def get_session(self, session_id: str) -> SessionRecord | None:
        return self._load(session_id)

    def create_session(self, task: str = "", dataset_path: str = "") -> SessionRecord:
        now = datetime.now(timezone.utc).isoformat()
        title = task[:40] if task else "新对话"
        return SessionRecord(
            session_id=uuid.uuid4().hex,
            title=title,
            created_at=now,
            updated_at=now,
            task=task,
            dataset_path=dataset_path,
        )

    def save_session(self, record: SessionRecord) -> None:
        record.updated_at = datetime.now(timezone.utc).isoformat()
        if not record.created_at:
            record.created_at = record.updated_at
        self._save(record)
        self._touch_index(record.session_id)
        self._prune()

    def delete_session(self, session_id: str) -> None:
        filepath = self.storage_dir / f"{session_id}.json"
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            pass
        ids = self._load_index()
        if session_id in ids:
            ids.remove(session_id)
            self._save_index(ids)

    def update_messages(self, session_id: str, messages: list[dict]) -> None:
        rec = self._load(session_id)
        if rec is None:
            return
        rec.messages = messages
        rec.updated_at = datetime.now(timezone.utc).isoformat()
        self._save(rec)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _filepath(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def _load(self, session_id: str) -> SessionRecord | None:
        fp = self._filepath(session_id)
        if not fp.exists():
            return None
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            return SessionRecord(**data)
        except (json.JSONDecodeError, TypeError, OSError):
            return None

    def _save(self, record: SessionRecord) -> None:
        fp = self._filepath(record.session_id)
        payload = {
            "session_id": record.session_id,
            "title": record.title,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "task": record.task,
            "dataset_path": record.dataset_path,
            "messages": record.messages,
            "metrics": record.metrics,
            "code_history": record.code_history,
            "final_report": record.final_report,
            "completed": record.completed,
        }
        tmp = fp.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(fp)

    def _touch_index(self, session_id: str) -> None:
        ids = self._load_index()
        if session_id in ids:
            ids.remove(session_id)
        ids.insert(0, session_id)
        self._save_index(ids)

    def _prune(self) -> None:
        ids = self._load_index()
        if len(ids) <= self.max_sessions:
            return
        to_remove = ids[self.max_sessions:]
        ids = ids[: self.max_sessions]
        self._save_index(ids)
        for sid in to_remove:
            fp = self._filepath(sid)
            try:
                fp.unlink(missing_ok=True)
            except OSError:
                pass
