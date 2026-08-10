"""
config.py
ADAPTATEUR TEMPORAIRE — conserve les constantes utilisées par services/ et ui/.

Les valeurs proviennent désormais d'AppSettings. Ce module sera supprimé en
Phase 3, quand tous les appelants auront reçu la configuration par injection.

Les trois seuils historiques sont ramenés à deux :
    FACE_RECOGNITION_THRESHOLD → profil.recognition_threshold
    DUPLICATE_TOLERANCE, VIDEO_FACE_TOLERANCE → settings.duplicate_tolerance
"""

from __future__ import annotations

from ..domain.profile import DEFAULT_PROFILES
from ..settings import AppSettings

_settings = AppSettings.create()
_settings.ensure_directories()

PROJECT_ROOT = _settings.project_root
ENCODED_DIR = str(_settings.encodings_dir)
META_FILE = str(_settings.encodings_dir / "metadata.json")
CAMERAS_FILE = _settings.cameras_file

DUPLICATE_TOLERANCE = _settings.duplicate_tolerance
VIDEO_FACE_TOLERANCE = _settings.duplicate_tolerance
FACE_RECOGNITION_THRESHOLD = DEFAULT_PROFILES["present"].recognition_threshold
