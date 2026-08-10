"""
camera_repository.py
Persistance des configurations de caméras dans `cameras.json`.

Une entrée invalide est ignorée et journalisée, sans empêcher le chargement des
autres : une caméra mal configurée ne doit pas priver l'utilisateur de tout son
système de surveillance.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable

from ..domain.camera import CameraConfig
from ..settings import AppSettings

logger = logging.getLogger(__name__)


class CameraRepository:
    """Lit et écrit les configurations de caméras."""

    def __init__(self, settings: AppSettings) -> None:
        self._file = settings.cameras_file

    def load_all(self) -> list[CameraConfig]:
        if not self._file.exists():
            return []
        try:
            items = json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("%s illisible, aucune caméra chargée : %s", self._file.name, exc)
            return []

        configs: list[CameraConfig] = []
        for item in items:
            try:
                configs.append(CameraConfig.from_dict(item))
            except (KeyError, TypeError, ValueError) as exc:
                logger.error("Caméra invalide ignorée (%s) : %s", item.get("name", "?"), exc)
        logger.info("%d caméra(s) chargée(s)", len(configs))
        return configs

    def save_all(self, configs: Iterable[CameraConfig]) -> None:
        payload = [c.to_dict() for c in configs]
        try:
            self._file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            logger.error("Écriture de %s échouée : %s", self._file.name, exc)
