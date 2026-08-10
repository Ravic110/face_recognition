"""
detection.py
Résultat d'une analyse : visages détectés et événement de surveillance.

Ces modèles traversent toutes les couches — moteur, bus, stockage, API, UI —
et ne dépendent donc d'aucune bibliothèque externe. Le champ `frame` est typé
`Any` pour transporter un tableau numpy sans imposer l'import à ce module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

UNKNOWN_NAME = "Inconnu"

# (haut, droite, bas, gauche) — convention de face_recognition
Location = tuple[int, int, int, int]


@dataclass
class DetectedFace:
    """Un visage détecté dans une frame."""

    location: Location
    name: str
    confidence: float
    is_known: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "is_known": self.is_known,
            "location": list(self.location),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DetectedFace:
        loc = data.get("location") or (0, 0, 0, 0)
        return cls(
            location=tuple(int(v) for v in loc),  # type: ignore[arg-type]
            name=data.get("name", UNKNOWN_NAME),
            confidence=float(data.get("confidence", 0.0)),
            is_known=bool(data.get("is_known", False)),
        )


@dataclass
class SurveillanceEvent:
    """Une détection sur une caméra, à un instant donné."""

    camera_uid: str
    camera_name: str
    timestamp: float
    faces: list[DetectedFace] = field(default_factory=list)
    motion_score: float = 0.0
    frame: Any = None

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
