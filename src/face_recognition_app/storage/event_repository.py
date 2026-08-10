"""
event_repository.py
Journal persistant des événements de surveillance — SQLite.

Deux améliorations par rapport à la version précédente :

  - **Écriture asynchrone.** `record()` dépose l'événement dans une file et rend
    la main immédiatement ; un thread dédié encode le snapshot et écrit en base.
    Auparavant l'encodage JPEG et l'INSERT se faisaient dans le thread d'analyse,
    sous verrou global, ce qui sérialisait les caméras.
  - **Purge automatique.** `purge_expired()` applique `settings.event_retention_days`,
    empêchant la base de croître sans fin.

Schéma inchangé, `events.db` existant est lu sans migration :
    events(id, timestamp, camera_uid, camera_name, faces_json, snapshot_b64)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from ..domain.detection import DetectedFace
from ..settings import AppSettings

logger = logging.getLogger(__name__)

_SENTINELLE: Any = object()


@dataclass
class StoredEvent:
    """Un événement lu depuis la base."""

    id: int
    timestamp: float
    camera_uid: str
    camera_name: str
    faces: list[DetectedFace] = field(default_factory=list)
    snapshot_b64: str | None = None

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)

    @property
    def known_names(self) -> list[str]:
        return [f.name for f in self.faces if f.is_known]

    @property
    def unknown_count(self) -> int:
        return sum(1 for f in self.faces if not f.is_known)

    @property
    def has_unknown(self) -> bool:
        return self.unknown_count > 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "datetime": self.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_uid": self.camera_uid,
            "camera_name": self.camera_name,
            "faces": [f.to_dict() for f in self.faces],
            "snapshot_b64": self.snapshot_b64,
        }


def _escape_like(value: str) -> str:
    """Neutralise les jokers LIKE dans une valeur fournie par l'utilisateur."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class EventRepository:
    """Accès aux événements de surveillance persistés."""

    SNAPSHOT_WIDTH = 320
    SNAPSHOT_QUALITY = 60
    DEFAULT_COUNT = 100
    QUEUE_MAXSIZE = 256

    def __init__(self, settings: AppSettings) -> None:
        self._db_path = settings.events_db
        self._retention_days = settings.event_retention_days
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()
        self._write_lock = threading.Lock()

        self._queue: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self._writer: threading.Thread | None = None
        self._running = False

        self._init_db()

    # ── Connexions ────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            with self._conn_lock:
                self._connections.append(conn)
        return conn

    def _init_db(self) -> None:
        # Réutilise la connexion thread-local plutôt que d'en ouvrir une jetable :
        # sur un disque lent, chaque connexion WAL coûte un fsync.
        nouveau = not self._db_path.exists()
        conn = self._conn()
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts  ON events(timestamp DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cam ON events(camera_uid)")
        conn.commit()
        if nouveau:
            os.chmod(self._db_path, 0o600)

    def close_connections(self) -> None:
        """Ferme toutes les connexions ouvertes. Appelé à l'arrêt de l'application."""
        with self._conn_lock:
            for conn in self._connections:
                try:
                    conn.close()
                except sqlite3.Error as exc:
                    logger.warning("Fermeture de connexion SQLite échouée : %s", exc)
            self._connections.clear()
        self._local = threading.local()

    # ── Thread d'écriture ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Démarre le thread d'écriture. Idempotent."""
        if self._running:
            return
        self._running = True
        self._writer = threading.Thread(target=self._write_loop, name="event-writer", daemon=True)
        self._writer.start()
        logger.info("EventRepository démarré (rétention %d jours)", self._retention_days)

    def stop(self) -> None:
        """Draine la file, arrête le thread et ferme les connexions. Idempotent."""
        if self._running:
            self._running = False
            self._queue.put(_SENTINELLE)
            if self._writer is not None:
                self._writer.join(timeout=5.0)
            self._writer = None
        self.close_connections()

    def _write_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINELLE:
                break
            try:
                self.record_sync(**item)
            except Exception as exc:
                logger.error("Écriture d'événement échouée : %s", exc, exc_info=True)

    # ── Écriture ──────────────────────────────────────────────────────────────

    def record(
        self,
        timestamp: float,
        camera_uid: str,
        camera_name: str,
        faces: list[DetectedFace],
        frame: np.ndarray | None = None,
        save_snapshot: bool = True,
    ) -> None:
        """Dépose l'événement dans la file d'écriture. Ne bloque jamais l'appelant."""
        item = {
            "timestamp": timestamp,
            "camera_uid": camera_uid,
            "camera_name": camera_name,
            "faces": faces,
            "frame": frame,
            "save_snapshot": save_snapshot,
        }
        if not self._running:
            self.record_sync(**item)
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning("File d'écriture saturée, événement %s abandonné", camera_name)

    def record_sync(
        self,
        timestamp: float,
        camera_uid: str,
        camera_name: str,
        faces: list[DetectedFace],
        frame: np.ndarray | None = None,
        save_snapshot: bool = True,
    ) -> StoredEvent:
        """Écrit immédiatement l'événement et le retourne."""
        snapshot = self._encode_snapshot(frame) if (frame is not None and save_snapshot) else None
        faces_json = json.dumps([f.to_dict() for f in faces], ensure_ascii=False)

        with self._write_lock:
            conn = self._conn()
            cur = conn.execute(
                "INSERT INTO events(timestamp, camera_uid, camera_name, faces_json, snapshot_b64) "
                "VALUES (?, ?, ?, ?, ?)",
                (timestamp, camera_uid, camera_name, faces_json, snapshot),
            )
            conn.commit()
            event_id = int(cur.lastrowid or 0)

        return StoredEvent(
            id=event_id,
            timestamp=timestamp,
            camera_uid=camera_uid,
            camera_name=camera_name,
            faces=list(faces),
            snapshot_b64=snapshot,
        )

    def _encode_snapshot(self, frame: np.ndarray) -> str | None:
        try:
            h, w = frame.shape[:2]
            if w > self.SNAPSHOT_WIDTH:
                scale = self.SNAPSHOT_WIDTH / w
                frame = cv2.resize(frame, (self.SNAPSHOT_WIDTH, int(h * scale)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.SNAPSHOT_QUALITY])
            if ok:
                return base64.b64encode(buf.tobytes()).decode("ascii")
            logger.warning("Encodage JPEG du snapshot échoué")
        except Exception as exc:
            logger.error("Erreur d'encodage du snapshot : %s", exc)
        return None

    # ── Lecture ───────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> StoredEvent:
        return StoredEvent(
            id=row["id"],
            timestamp=row["timestamp"],
            camera_uid=row["camera_uid"],
            camera_name=row["camera_name"],
            faces=[DetectedFace.from_dict(d) for d in json.loads(row["faces_json"])],
            snapshot_b64=row["snapshot_b64"],
        )

    def _query(self, sql: str, params: tuple[Any, ...]) -> list[StoredEvent]:
        rows = self._conn().execute(sql, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_recent(self, count: int = DEFAULT_COUNT) -> list[StoredEvent]:
        return self._query("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (count,))

    def get_by_id(self, event_id: int) -> StoredEvent | None:
        row = self._conn().execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return self._row_to_event(row) if row else None

    def get_by_camera(self, camera_uid: str, count: int = DEFAULT_COUNT) -> list[StoredEvent]:
        return self._query(
            "SELECT * FROM events WHERE camera_uid = ? ORDER BY timestamp DESC LIMIT ?",
            (camera_uid, count),
        )

    def get_by_person(self, name: str, count: int = DEFAULT_COUNT) -> list[StoredEvent]:
        """Recherche par nom dans le JSON des visages, jokers LIKE neutralisés."""
        pattern = f'%"name": "{_escape_like(name)}"%'
        return self._query(
            "SELECT * FROM events WHERE faces_json LIKE ? ESCAPE '\\' "
            "ORDER BY timestamp DESC LIMIT ?",
            (pattern, count),
        )

    def count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def stats(self) -> dict:
        conn = self._conn()
        total = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        by_camera = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT camera_name, COUNT(*) FROM events GROUP BY camera_name"
            ).fetchall()
        }
        return {"total": total, "by_camera": by_camera}

    # ── Nettoyage ─────────────────────────────────────────────────────────────

    def delete_before(self, before_timestamp: float) -> int:
        with self._write_lock:
            conn = self._conn()
            cur = conn.execute("DELETE FROM events WHERE timestamp < ?", (before_timestamp,))
            conn.commit()
            return cur.rowcount

    def purge_expired(self) -> int:
        """Supprime les événements plus vieux que la rétention configurée."""
        cutoff = time.time() - self._retention_days * 86400
        supprimes = self.delete_before(cutoff)
        if supprimes:
            logger.info(
                "Purge : %d événement(s) au-delà de %d jours supprimé(s)",
                supprimes,
                self._retention_days,
            )
        return supprimes

    def vacuum(self) -> None:
        """Compacte la base après une purge importante."""
        with self._write_lock:
            self._conn().execute("VACUUM")
