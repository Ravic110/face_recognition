"""
profile_store.py
Profils de surveillance avec règles différenciées.

Profils prédéfinis :
  absent  → analyse toutes les caméras, alertes actives, enregistrement vidéo ON
  nuit    → sensibilité maximale, alertes bureau + email, enregistrement ON
  present → détection légère, pas d'alerte sur les personnes connues, enregistrement OFF

L'utilisateur peut créer ses propres profils.
Persisté dans profiles.json.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PROFILES_FILE = PROJECT_ROOT / "profiles.json"


@dataclass
class SurveillanceProfile:
    """Règles d'un profil de surveillance."""

    name: str
    label: str                              # Nom affiché dans l'UI
    detection_model: str = "hog"            # "hog" (CPU) | "cnn" (GPU)
    analysis_interval: float = 0.5          # Secondes entre deux analyses
    motion_required: bool = True            # N'analyser qu'en cas de mouvement
    motion_sensitivity: int = 500           # Seuil de sensibilité mouvement
    recognition_threshold: float = 0.5     # Distance de reconnaissance faciale
    record_video: bool = False              # Enregistrer des clips vidéo
    pre_record_seconds: float = 5.0
    post_record_seconds: float = 10.0
    alert_on_unknown: bool = True
    alert_on_known: bool = False
    target_persons: List[str] = field(default_factory=list)
    enabled_camera_uids: List[str] = field(default_factory=list)  # [] = toutes

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SurveillanceProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Profils par défaut ────────────────────────────────────────────────────────

DEFAULT_PROFILES: Dict[str, SurveillanceProfile] = {
    "present": SurveillanceProfile(
        name="present",
        label="Présent (domicile occupé)",
        detection_model="hog",
        analysis_interval=1.0,
        motion_required=True,
        motion_sensitivity=800,
        recognition_threshold=0.5,
        record_video=False,
        alert_on_unknown=False,
        alert_on_known=False,
    ),
    "absent": SurveillanceProfile(
        name="absent",
        label="Absent (surveillance active)",
        detection_model="hog",
        analysis_interval=0.5,
        motion_required=True,
        motion_sensitivity=400,
        recognition_threshold=0.5,
        record_video=True,
        pre_record_seconds=5.0,
        post_record_seconds=15.0,
        alert_on_unknown=True,
        alert_on_known=True,
    ),
    "nuit": SurveillanceProfile(
        name="nuit",
        label="Nuit (sensibilité maximale)",
        detection_model="hog",
        analysis_interval=0.3,
        motion_required=False,          # Analyser en continu la nuit
        motion_sensitivity=200,
        recognition_threshold=0.45,
        record_video=True,
        pre_record_seconds=10.0,
        post_record_seconds=20.0,
        alert_on_unknown=True,
        alert_on_known=False,
    ),
}


# ── Store ─────────────────────────────────────────────────────────────────────

class ProfileStore:
    """
    Gère les profils de surveillance.

    Usage :
        store = ProfileStore()
        profile = store.get_active()
        store.set_active("absent")
    """

    def __init__(self, profiles_file: Path = PROFILES_FILE) -> None:
        self._file = profiles_file
        self._profiles: Dict[str, SurveillanceProfile] = dict(DEFAULT_PROFILES)
        self._active_name: str = "present"
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            self._active_name = data.get("active", "present")
            for p_data in data.get("profiles", []):
                p = SurveillanceProfile.from_dict(p_data)
                self._profiles[p.name] = p
        except Exception as exc:
            logger.error("Erreur lecture profiles.json : %s", exc)

    def save(self) -> None:
        data = {
            "active": self._active_name,
            "profiles": [p.to_dict() for p in self._profiles.values()],
        }
        self._file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── API ────────────────────────────────────────────────────────────────────

    def list_profiles(self) -> List[SurveillanceProfile]:
        return list(self._profiles.values())

    def get(self, name: str) -> Optional[SurveillanceProfile]:
        return self._profiles.get(name)

    def get_active(self) -> SurveillanceProfile:
        return self._profiles.get(self._active_name, DEFAULT_PROFILES["present"])

    @property
    def active_name(self) -> str:
        return self._active_name

    def set_active(self, name: str) -> bool:
        if name not in self._profiles:
            return False
        self._active_name = name
        self.save()
        return True

    def save_profile(self, profile: SurveillanceProfile) -> None:
        self._profiles[profile.name] = profile
        self.save()

    def delete_profile(self, name: str) -> bool:
        if name in DEFAULT_PROFILES:
            return False   # Ne pas supprimer les profils par défaut
        if name == self._active_name:
            self._active_name = "present"
        removed = self._profiles.pop(name, None)
        if removed:
            self.save()
        return removed is not None
