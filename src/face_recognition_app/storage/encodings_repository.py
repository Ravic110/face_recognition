"""
encodings_repository.py
Stockage des encodages faciaux — un fichier JSON par visage.

Format d'un fichier (inchangé par rapport à la version précédente) :
    {"name": str, "encoding": [128 flottants], "timestamp": iso, "image_base64": str?}

`metadata.json` indexe les uid vers nom et date de création.

Les fichiers sont écrits en 0600 : ce sont des données biométriques.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..domain.matching import ENCODING_LENGTH, find_duplicate
from ..settings import AppSettings

logger = logging.getLogger(__name__)


@dataclass
class StoredEncoding:
    """Un encodage lu depuis le disque."""

    uid: str
    name: str
    encoding: np.ndarray
    timestamp: str


class EncodingsRepository:
    """Accès aux encodages faciaux persistés."""

    def __init__(self, settings: AppSettings) -> None:
        self._dir = settings.encodings_dir
        self._meta_file = settings.encodings_dir / "metadata.json"
        self._default_tolerance = settings.duplicate_tolerance

    # ── Lecture ───────────────────────────────────────────────────────────────

    def _iter_files(self) -> Iterator[Path]:
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob("*.json")):
            if path.name == self._meta_file.name or "temp" in path.name:
                continue
            yield path

    def _read_json(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Encodage illisible ignoré (%s) : %s", path.name, exc)
            return None

    @staticmethod
    def _is_valid(data: dict) -> bool:
        if not all(k in data for k in ("name", "encoding", "timestamp")):
            return False
        return isinstance(data["encoding"], list) and len(data["encoding"]) == ENCODING_LENGTH

    def load_all(self) -> list[StoredEncoding]:
        """Tous les encodages valides, triés par nom de fichier."""
        resultats: list[StoredEncoding] = []
        for path in self._iter_files():
            data = self._read_json(path)
            if data is None:
                continue
            if not self._is_valid(data):
                logger.warning("Encodage invalide ignoré : %s", path.name)
                continue
            resultats.append(
                StoredEncoding(
                    uid=path.stem,
                    name=data["name"],
                    encoding=np.array(data["encoding"], dtype=float),
                    timestamp=data["timestamp"],
                )
            )
        return resultats

    def load_map(self) -> dict[str, np.ndarray]:
        """Un encodage par nom — le premier rencontré en cas de doublon."""
        mapping: dict[str, np.ndarray] = {}
        for entry in self.load_all():
            mapping.setdefault(entry.name, entry.encoding)
        return mapping

    def load_metadata(self) -> dict:
        data = self._read_json(self._meta_file)
        return data if isinstance(data, dict) else {}

    def load_image(self, name: str) -> np.ndarray | None:
        """Image du visage enregistré pour ce nom, ou None."""
        for path in self._iter_files():
            data = self._read_json(path)
            if data is None or data.get("name") != name:
                continue
            encoded = data.get("image_base64") or data.get("image")
            if isinstance(encoded, str):
                return self._decode_image(encoded)
        return None

    @staticmethod
    def _decode_image(encoded: str) -> np.ndarray | None:
        try:
            raw = base64.b64decode(encoded)
            return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.warning("Image d'encodage indécodable : %s", exc)
            return None

    # ── Écriture ──────────────────────────────────────────────────────────────

    def _write_json(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        os.chmod(path, 0o600)

    def save(
        self,
        name: str,
        encoding: np.ndarray,
        image: np.ndarray | None = None,
    ) -> str:
        """Enregistre un encodage et retourne son uid."""
        encoding = np.asarray(encoding, dtype=float)
        if encoding.shape != (ENCODING_LENGTH,):
            raise ValueError(
                f"Un encodage doit être un vecteur de {ENCODING_LENGTH} valeurs, "
                f"reçu {encoding.shape}"
            )

        timestamp = datetime.now().isoformat()
        uid = uuid.uuid4().hex[:12]
        data: dict = {
            "name": name,
            "encoding": encoding.tolist(),
            "timestamp": timestamp,
        }

        if image is not None:
            ok, buffer = cv2.imencode(".jpg", image)
            if ok:
                data["image_base64"] = base64.b64encode(buffer).decode("ascii")
            else:
                logger.warning(
                    "Encodage JPEG échoué pour '%s' ; visage enregistré sans image", name
                )

        self._dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self._dir / f"{uid}.json", data)
        self._update_metadata(uid, name, timestamp)
        logger.info("Visage '%s' enregistré (uid=%s)", name, uid)
        return uid

    def _update_metadata(self, uid: str, name: str, timestamp: str) -> None:
        metadata = self.load_metadata()
        metadata[uid] = {"name": name, "date_creation": timestamp}
        self._write_json(self._meta_file, metadata)

    def delete(self, name: str) -> list[str]:
        """Supprime tous les encodages portant ce nom. Retourne les uid supprimés."""
        supprimes: list[str] = []
        for path in self._iter_files():
            data = self._read_json(path)
            if data is None or data.get("name") != name:
                continue
            try:
                path.unlink()
                supprimes.append(path.stem)
            except OSError as exc:
                logger.error("Suppression impossible (%s) : %s", path.name, exc)

        if supprimes:
            metadata = self.load_metadata()
            for uid in supprimes:
                metadata.pop(uid, None)
            self._write_json(self._meta_file, metadata)
            logger.info("Visage '%s' supprimé (%d fichier(s))", name, len(supprimes))
        return supprimes

    # ── Doublons ──────────────────────────────────────────────────────────────

    def find_duplicate_name(
        self,
        encoding: np.ndarray,
        tolerance: float | None = None,
    ) -> str | None:
        """Nom de la personne déjà enregistrée correspondant à cet encodage, ou None."""
        existing = [(e.name, e.encoding) for e in self.load_all()]
        return find_duplicate(
            encoding,
            existing,
            self._default_tolerance if tolerance is None else tolerance,
        )
