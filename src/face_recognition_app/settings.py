"""
settings.py
Configuration de l'application, construite une fois et injectée.

Aucun module ne doit lire la configuration à l'import : `AppSettings` est
construit dans `__main__` puis passé explicitement aux composants qui en ont
besoin. C'est ce qui rend le stockage testable sans monkey-patching de
constantes de module.

Surcharges par variables d'environnement, toutes préfixées `FR_` :
  FR_API_HOST, FR_API_PORT, FR_API_ALLOW_CONTROL,
  FR_EVENT_RETENTION_DAYS, FR_CAPTURE_FPS, FR_MAX_CLIPS
Les secrets ne transitent jamais par un fichier : FR_API_KEY, FR_SMTP_PASSWORD.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "%s='%s' n'est pas un entier, valeur par défaut %d conservée", name, raw, default
        )
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "%s='%s' n'est pas un nombre, valeur par défaut %s conservée", name, raw, default
        )
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "oui"}


@dataclass(frozen=True)
class AppSettings:
    """Configuration complète de l'application. Immuable."""

    project_root: Path
    config_dir: Path
    encodings_dir: Path
    events_db: Path
    clips_dir: Path
    cameras_file: Path
    profiles_file: Path
    alerts_file: Path
    api_key_file: Path

    # Capture et affichage
    capture_fps: float = 15.0
    ui_refresh_ms: int = 200

    # Rétention. 0 = conserver indéfiniment, valeur par défaut : un système de
    # sécurité n'efface pas son historique de lui-même. La purge est une action
    # manuelle, ou s'active en fixant FR_EVENT_RETENTION_DAYS à une valeur > 0.
    event_retention_days: int = 0
    max_clips: int = 100
    max_clips_mb: int = 2048

    # API — fermée par défaut
    api_host: str = "127.0.0.1"
    api_port: int = 5000
    api_allow_control: bool = False

    # Seuil d'enrôlement. Le seuil de reconnaissance live vit dans le profil.
    duplicate_tolerance: float = 0.6

    @classmethod
    def create(cls, project_root: Path | None = None) -> AppSettings:
        """Construit la configuration à partir de la racine projet et de l'environnement."""
        root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        config_dir = root / ".config"
        return cls(
            project_root=root,
            config_dir=config_dir,
            encodings_dir=root / "encodings",
            events_db=root / "events.db",
            clips_dir=root / "clips",
            cameras_file=root / "cameras.json",
            profiles_file=root / "profiles.json",
            alerts_file=root / "alerts_config.json",
            api_key_file=config_dir / "api_key",
            capture_fps=_env_float("FR_CAPTURE_FPS", 15.0),
            ui_refresh_ms=_env_int("FR_UI_REFRESH_MS", 200),
            event_retention_days=_env_int("FR_EVENT_RETENTION_DAYS", 0),
            max_clips=_env_int("FR_MAX_CLIPS", 100),
            max_clips_mb=_env_int("FR_MAX_CLIPS_MB", 2048),
            api_host=os.environ.get("FR_API_HOST", "127.0.0.1"),
            api_port=_env_int("FR_API_PORT", 5000),
            api_allow_control=_env_bool("FR_API_ALLOW_CONTROL", False),
            duplicate_tolerance=_env_float("FR_DUPLICATE_TOLERANCE", 0.6),
        )

    def ensure_directories(self) -> None:
        """Crée les répertoires de données. Le répertoire de config est en 0700."""
        self.encodings_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.chmod(0o700)
