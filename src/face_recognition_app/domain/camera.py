"""
camera.py
Configuration persistable d'une caméra.

Le format sérialisé est identique à celui de `cameras.json` existant :
la refactorisation ne demande aucune migration de ce fichier.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

SOURCE_TYPES = frozenset({"webcam", "ip"})
DETECTION_MODELS = frozenset({"hog", "cnn"})


@dataclass
class CameraConfig:
    """Paramètres d'une caméra. `source` est un index entier (webcam) ou une URL (IP)."""

    name: str
    source_type: str
    source: str | int
    enabled: bool = True
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    width: int = 640
    height: int = 480

    # Zone d'intérêt (x, y, largeur, hauteur) en pixels ; None = toute l'image
    roi: tuple[int, int, int, int] | None = None

    detection_model: str = "hog"

    # Retournement horizontal. `None` = décider selon le type de source : une
    # webcam sert de miroir, on s'attend à s'y voir comme dans une glace ; une
    # caméra IP de surveillance ne doit pas l'être, cela rendrait les textes de
    # la scène illisibles. Résolu en booléen dès la construction.
    mirror: bool | None = None

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(
                f"source_type invalide : {self.source_type!r} (attendu : {sorted(SOURCE_TYPES)})"
            )
        if self.detection_model not in DETECTION_MODELS:
            raise ValueError(
                f"detection_model invalide : {self.detection_model!r} "
                f"(attendu : {sorted(DETECTION_MODELS)})"
            )
        if self.roi is not None:
            self.roi = tuple(int(v) for v in self.roi)  # type: ignore[assignment]
        if self.mirror is None:
            self.mirror = self.source_type == "webcam"

    @property
    def is_ip(self) -> bool:
        return self.source_type == "ip"

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "name": self.name,
            "source_type": self.source_type,
            "source": self.source,
            "enabled": self.enabled,
            "width": self.width,
            "height": self.height,
            "roi": list(self.roi) if self.roi else None,
            "detection_model": self.detection_model,
            "mirror": self.mirror,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraConfig:
        roi_raw = data.get("roi")
        return cls(
            uid=data.get("uid", uuid.uuid4().hex[:8]),
            name=data["name"],
            source_type=data["source_type"],
            source=data["source"],
            enabled=data.get("enabled", True),
            width=data.get("width", 640),
            height=data.get("height", 480),
            roi=tuple(roi_raw) if roi_raw else None,
            detection_model=data.get("detection_model", "hog"),
            # Absent des cameras.json antérieurs : laisser __post_init__ décider.
            mirror=data.get("mirror"),
        )
