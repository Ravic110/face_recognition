"""
camera_manager.py
Registre de toutes les caméras configurées.

Responsabilités :
  - CRUD sur les configs (persistées dans cameras.json)
  - Cycle de vie des CameraSource (start / stop)
  - Point d'entrée unique pour obtenir une frame depuis n'importe quelle caméra
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .camera_source import CameraConfig, CameraSource, create_camera_source

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Gestionnaire centralisé de toutes les sources vidéo.

    Usage typique :
        mgr = CameraManager(cameras_file)
        mgr.add_camera(CameraConfig(name="Salon", source_type="webcam", source=0))
        mgr.start_all()
        frame = mgr.get_frame("uid123")
        mgr.stop_all()
    """

    def __init__(self, cameras_file: Path) -> None:
        self._file = cameras_file
        self._configs: Dict[str, CameraConfig] = {}
        self._sources: Dict[str, CameraSource] = {}
        # Callbacks invoqués quand une caméra est ajoutée / supprimée
        self._on_change_callbacks: List[Callable[[], None]] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            items = json.loads(self._file.read_text(encoding="utf-8"))
            for item in items:
                cfg = CameraConfig.from_dict(item)
                self._configs[cfg.uid] = cfg
            logger.info("CameraManager : %d caméra(s) chargée(s)", len(self._configs))
        except Exception as exc:
            logger.error("Erreur lecture %s : %s", self._file, exc)

    def _save(self) -> None:
        try:
            self._file.write_text(
                json.dumps([c.to_dict() for c in self._configs.values()], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Erreur écriture %s : %s", self._file, exc)

    # ── CRUD configs ─────────────────────────────────────────────────────────

    def add_camera(self, config: CameraConfig) -> None:
        """Ajoute une nouvelle caméra et persiste la configuration."""
        self._configs[config.uid] = config
        self._save()
        self._notify()

    def remove_camera(self, uid: str) -> None:
        """Arrête et supprime une caméra."""
        self.stop_camera(uid)
        self._configs.pop(uid, None)
        self._save()
        self._notify()

    def update_camera(self, config: CameraConfig) -> None:
        """Met à jour une config existante (redémarre si la source était active)."""
        uid = config.uid
        was_running = uid in self._sources and self._sources[uid].is_running
        if was_running:
            self.stop_camera(uid)
        self._configs[uid] = config
        self._save()
        if was_running and config.enabled:
            self.start_camera(uid)
        self._notify()

    def list_configs(self) -> List[CameraConfig]:
        return list(self._configs.values())

    def get_config(self, uid: str) -> Optional[CameraConfig]:
        return self._configs.get(uid)

    # ── Cycle de vie ─────────────────────────────────────────────────────────

    def start_camera(self, uid: str) -> bool:
        """Démarre une caméra par son UID. Retourne True si OK."""
        config = self._configs.get(uid)
        if not config:
            logger.warning("start_camera : UID inconnu %s", uid)
            return False
        if not config.enabled:
            return False
        if uid in self._sources and self._sources[uid].is_running:
            return True   # déjà active

        source = create_camera_source(config)
        ok = source.start()
        if ok:
            self._sources[uid] = source
        return ok

    def stop_camera(self, uid: str) -> None:
        """Arrête une caméra."""
        source = self._sources.pop(uid, None)
        if source:
            source.stop()

    def start_all(self) -> None:
        """Démarre toutes les caméras actives."""
        for uid, cfg in self._configs.items():
            if cfg.enabled:
                self.start_camera(uid)

    def stop_all(self) -> None:
        """Arrête toutes les caméras en cours."""
        for uid in list(self._sources):
            self.stop_camera(uid)

    # ── Accès aux frames ─────────────────────────────────────────────────────

    def get_frame(self, uid: str):
        """Retourne la dernière frame de la caméra uid (None si indisponible)."""
        source = self._sources.get(uid)
        return source.get_frame() if source else None

    def get_source(self, uid: str) -> Optional[CameraSource]:
        return self._sources.get(uid)

    def get_all_sources(self) -> Dict[str, CameraSource]:
        return dict(self._sources)

    def is_running(self, uid: str) -> bool:
        source = self._sources.get(uid)
        return source is not None and source.is_running

    # ── Callbacks ────────────────────────────────────────────────────────────

    def on_change(self, callback: Callable[[], None]) -> None:
        """Enregistre un callback invoqué à chaque modification du registre."""
        self._on_change_callbacks.append(callback)

    def _notify(self) -> None:
        for cb in self._on_change_callbacks:
            try:
                cb()
            except Exception as exc:
                logger.error("Erreur callback on_change : %s", exc)
