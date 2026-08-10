"""
profile.py
Profil de surveillance — source unique de vérité des règles d'analyse et d'alerte.

Les champs d'alerte vivaient jusqu'ici en double, ici et dans `AlertConfig`,
et seule la version `AlertConfig` avait un effet. Ils sont désormais définis
uniquement dans ce module ; `AlertConfig` ne conserve que le transport
(serveur SMTP, URL de webhook, activation des canaux).

Profils prédéfinis :
  present → détection légère, aucune alerte, pas d'enregistrement
  absent  → surveillance active, alertes, enregistrement
  nuit    → sensibilité maximale, analyse continue, enregistrement
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from .detection import DetectedFace

DETECTION_MODELS = frozenset({"hog", "cnn"})


@dataclass
class SurveillanceProfile:
    """Règles appliquées par le moteur de surveillance."""

    name: str
    label: str

    # Analyse
    detection_model: str = "hog"
    analysis_interval: float = 0.5
    motion_required: bool = True
    motion_sensitivity: int = 500
    recognition_threshold: float = 0.5

    # Enregistrement
    record_video: bool = False
    pre_record_seconds: float = 5.0
    post_record_seconds: float = 10.0

    # Alertes — source unique de vérité
    alert_on_unknown: bool = True
    alert_on_known: bool = False
    target_persons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.detection_model not in DETECTION_MODELS:
            raise ValueError(
                f"detection_model invalide : {self.detection_model!r} "
                f"(attendu : {sorted(DETECTION_MODELS)})"
            )
        if not 0.0 < self.recognition_threshold <= 1.0:
            raise ValueError(
                f"recognition_threshold doit être dans ]0, 1], reçu {self.recognition_threshold}"
            )
        if self.analysis_interval < 0:
            raise ValueError(f"analysis_interval doit être positif, reçu {self.analysis_interval}")

    # ── Décisions ─────────────────────────────────────────────────────────────

    def should_alert(self, faces: Sequence[DetectedFace]) -> bool:
        """Faut-il émettre une alerte pour ces visages ?"""
        if not faces:
            return False
        known = [f for f in faces if f.is_known]
        if any(f.name in self.target_persons for f in known):
            return True
        if self.alert_on_unknown and any(not f.is_known for f in faces):
            return True
        return bool(self.alert_on_known and known)

    def should_record(self) -> bool:
        """Faut-il enregistrer un clip vidéo sur détection ?"""
        return self.record_video

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SurveillanceProfile:
        """Ignore les clés inconnues, dont `enabled_camera_uids` désormais supprimé."""
        champs = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in champs})


DEFAULT_PROFILES: dict[str, SurveillanceProfile] = {
    "present": SurveillanceProfile(
        name="present",
        label="Présent (domicile occupé)",
        analysis_interval=1.0,
        motion_sensitivity=800,
        record_video=False,
        alert_on_unknown=False,
        alert_on_known=False,
    ),
    "absent": SurveillanceProfile(
        name="absent",
        label="Absent (surveillance active)",
        analysis_interval=0.5,
        motion_sensitivity=400,
        record_video=True,
        post_record_seconds=15.0,
        alert_on_unknown=True,
        alert_on_known=True,
    ),
    "nuit": SurveillanceProfile(
        name="nuit",
        label="Nuit (sensibilité maximale)",
        analysis_interval=0.3,
        motion_required=False,
        motion_sensitivity=200,
        recognition_threshold=0.45,
        record_video=True,
        pre_record_seconds=10.0,
        post_record_seconds=20.0,
        alert_on_unknown=True,
        alert_on_known=False,
    ),
}
