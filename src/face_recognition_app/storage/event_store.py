"""
event_store.py
Journal persistant des événements de surveillance — backend SQLite.

Remplace l'ancien backend JSON pour des requêtes rapides même sur un grand historique.
L'API publique reste identique afin de ne pas casser le code existant.

Schéma :
  events(id, timestamp, camera_uid, camera_name, faces_json, snapshot_b64)
"""

from __future__ import annotations

import base64
import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "events.db"


# ── Modèle ────────────────────────────────────────────────────────────────────

class StoredEvent:
    """Représentation d'un événement de détection chargé depuis SQLite."""

    def __init__(
        self,
        id: int,
        timestamp: float,
        camera_uid: str,
        camera_name: str,
        faces: List[dict],
        snapshot_b64: Optional[str] = None,
    ) -> None:
        self.id = id
        self.timestamp = timestamp
        self.camera_uid = camera_uid
        self.camera_name = camera_name
        self.faces = faces
        self.snapshot_b64 = snapshot_b64

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)

    @property
    def known_names(self) -> List[str]:
        return [f["name"] for f in self.faces if f.get("is_known")]

    @property
    def has_unknown(self) -> bool:
        return any(not f.get("is_known", False) for f in self.faces)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "datetime": self.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_uid": self.camera_uid,
            "camera_name": self.camera_name,
            "faces": self.faces,
            "snapshot_b64": self.snapshot_b64,
        }


# ── Store ─────────────────────────────────────────────────────────────────────

class EventStore:
    """
    Gestionnaire des événements de surveillance (SQLite).

    Thread-safe : chaque thread obtient sa propre connexion via threading.local().
    """

    SNAPSHOT_WIDTH = 320
    SNAPSHOT_QUALITY = 60
    DEFAULT_RECENT_COUNT = 100

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_db()

    # ── Connexion ─────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        with self._write_lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    REAL    NOT NULL,
                    camera_uid   TEXT    NOT NULL,
                    camera_name  TEXT    NOT NULL,
                    faces_json   TEXT    NOT NULL,
                    snapshot_b64 TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts      ON events(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cam     ON events(camera_uid)")
            conn.commit()
            conn.close()

    # ── Écriture ──────────────────────────────────────────────────────────────

    def record(
        self,
        timestamp: float,
        camera_uid: str,
        camera_name: str,
        faces: List[dict],
        frame: Optional[np.ndarray] = None,
        save_snapshot: bool = True,
    ) -> StoredEvent:
        snapshot_b64 = None
        if frame is not None and save_snapshot:
            snapshot_b64 = self._encode_snapshot(frame)

        with self._write_lock:
            conn = self._conn()
            cur = conn.execute(
                "INSERT INTO events(timestamp, camera_uid, camera_name, faces_json, snapshot_b64) "
                "VALUES (?, ?, ?, ?, ?)",
                (timestamp, camera_uid, camera_name, json.dumps(faces), snapshot_b64),
            )
            conn.commit()
            event_id = cur.lastrowid

        return StoredEvent(
            id=event_id,
            timestamp=timestamp,
            camera_uid=camera_uid,
            camera_name=camera_name,
            faces=faces,
            snapshot_b64=snapshot_b64,
        )

    # ── Lecture ───────────────────────────────────────────────────────────────

    def get_recent(self, count: int = DEFAULT_RECENT_COUNT) -> List[StoredEvent]:
        rows = self._conn().execute(
            "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (count,)
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_by_id(self, event_id: int) -> Optional[StoredEvent]:
        row = self._conn().execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        return self._row_to_event(row) if row else None

    def get_for_date(self, date: datetime) -> List[StoredEvent]:
        start = datetime(date.year, date.month, date.day).timestamp()
        end = start + 86400
        rows = self._conn().execute(
            "SELECT * FROM events WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp DESC",
            (start, end),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_by_camera(self, camera_uid: str, count: int = DEFAULT_RECENT_COUNT) -> List[StoredEvent]:
        rows = self._conn().execute(
            "SELECT * FROM events WHERE camera_uid = ? ORDER BY timestamp DESC LIMIT ?",
            (camera_uid, count),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_by_person(self, name: str, count: int = DEFAULT_RECENT_COUNT) -> List[StoredEvent]:
        # SQLite LIKE sur le JSON (suffisant pour des noms simples)
        pattern = f'%"name": "{name}"%'
        rows = self._conn().execute(
            "SELECT * FROM events WHERE faces_json LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (pattern, count),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def stats(self) -> dict:
        """Statistiques globales : total, par caméra, par personne."""
        conn = self._conn()
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        by_camera = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT camera_name, COUNT(*) FROM events GROUP BY camera_name"
            ).fetchall()
        }
        return {"total": total, "by_camera": by_camera}

    def delete_before(self, before_timestamp: float) -> int:
        """Supprime les événements antérieurs à une date (nettoyage)."""
        with self._write_lock:
            conn = self._conn()
            cur = conn.execute("DELETE FROM events WHERE timestamp < ?", (before_timestamp,))
            conn.commit()
            return cur.rowcount

    # ── Utilitaires ───────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> StoredEvent:
        return StoredEvent(
            id=row["id"],
            timestamp=row["timestamp"],
            camera_uid=row["camera_uid"],
            camera_name=row["camera_name"],
            faces=json.loads(row["faces_json"]),
            snapshot_b64=row["snapshot_b64"],
        )

    def _encode_snapshot(self, frame: np.ndarray) -> Optional[str]:
        try:
            h, w = frame.shape[:2]
            if w > self.SNAPSHOT_WIDTH:
                scale = self.SNAPSHOT_WIDTH / w
                frame = cv2.resize(frame, (self.SNAPSHOT_WIDTH, int(h * scale)))
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.SNAPSHOT_QUALITY]
            )
            if ok:
                return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception as exc:
            logger.error("Erreur encodage snapshot : %s", exc)
        return None
