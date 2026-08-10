"""
event_store.py
ADAPTATEUR TEMPORAIRE — conserve l'API utilisée par ui/ et services/.

`EventStore` délègue à `EventRepository`. Les visages sont acceptés sous forme
de dictionnaires (ancienne API) comme d'objets `DetectedFace`.

Sera supprimé en Phase 3.
"""

from __future__ import annotations

from ..domain.detection import DetectedFace
from ..settings import AppSettings
from .event_repository import EventRepository, StoredEvent

__all__ = ["EventStore", "StoredEvent"]


class EventStore:
    """Façade de compatibilité au-dessus d'EventRepository."""

    def __init__(self, db_path=None) -> None:
        settings = AppSettings.create()
        settings.ensure_directories()
        self._repo = EventRepository(settings)

    @property
    def repository(self) -> EventRepository:
        return self._repo

    @staticmethod
    def _to_faces(faces) -> list[DetectedFace]:
        return [f if isinstance(f, DetectedFace) else DetectedFace.from_dict(f) for f in faces]

    def record(self, timestamp, camera_uid, camera_name, faces, frame=None, save_snapshot=True):
        return self._repo.record_sync(
            timestamp, camera_uid, camera_name, self._to_faces(faces), frame, save_snapshot
        )

    def get_recent(self, count=100):
        return self._repo.get_recent(count)

    def get_by_id(self, event_id):
        return self._repo.get_by_id(event_id)

    def get_by_camera(self, camera_uid, count=100):
        return self._repo.get_by_camera(camera_uid, count)

    def get_by_person(self, name, count=100):
        return self._repo.get_by_person(name, count)

    def count(self):
        return self._repo.count()

    def stats(self):
        return self._repo.stats()

    def delete_before(self, before_timestamp):
        return self._repo.delete_before(before_timestamp)
