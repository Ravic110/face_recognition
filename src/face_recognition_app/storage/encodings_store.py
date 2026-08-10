"""
encodings_store.py
ADAPTATEUR TEMPORAIRE — conserve l'API par fonctions utilisée par services/ et ui/.

Délègue à EncodingsRepository. Sera supprimé en Phase 3, quand tous les appelants
auront reçu le repository par injection.
"""

from __future__ import annotations

import numpy as np

from ..settings import AppSettings
from .encodings_repository import EncodingsRepository

_repo: EncodingsRepository | None = None


def _repository() -> EncodingsRepository:
    global _repo
    if _repo is None:
        settings = AppSettings.create()
        settings.ensure_directories()
        _repo = EncodingsRepository(settings)
    return _repo


def set_repository(repo: EncodingsRepository) -> None:
    """Injecté par __main__ pour que l'adaptateur partage le repository de l'application."""
    global _repo
    _repo = repo


def load_existing_encodings() -> list[dict]:
    return [{"name": e.name, "encoding": e.encoding} for e in _repository().load_all()]


def load_encodings_map() -> dict[str, np.ndarray]:
    return _repository().load_map()


def load_metadata() -> dict:
    return _repository().load_metadata()


def save_face_encoding(name, encoding, image=None) -> None:
    _repository().save(name, np.asarray(encoding, dtype=float), image)


def load_image_for_name(name: str):
    return _repository().load_image(name)


def delete_encoding(name: str) -> list[str]:
    return _repository().delete(name)
