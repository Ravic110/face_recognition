"""
camera_source.py
Abstraction de source vidéo.

Supporte :
  - WebcamSource   : caméra USB / intégrée (index OpenCV)
  - IPCameraSource : flux RTSP ou HTTP-MJPEG (caméras IP, smartphones)
                     → Android : application "IP Webcam"  → http://<ip>:8080/video
                     → iOS     : application "EpoCam"     → rtsp://<ip>/live
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    """Paramètres persistables d'une caméra."""

    name: str
    source_type: str          # 'webcam' | 'ip'
    source: str | int         # index (webcam) ou URL (IP)
    enabled: bool = True
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # Résolution souhaitée
    width: int = 640
    height: int = 480

    # Zone d'intérêt ROI (x, y, w, h) en pixels, None = toute l'image
    roi: Optional[Tuple[int, int, int, int]] = None

    # Modèle de détection : "hog" (CPU) | "cnn" (GPU/plus précis)
    detection_model: str = "hog"

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
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraConfig":
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
        )


# ── Classe de base ────────────────────────────────────────────────────────────

class CameraSource(ABC):
    """
    Source vidéo générique avec boucle de lecture en arrière-plan.

    Usage :
        src = WebcamSource(config)
        src.start()
        frame = src.get_frame()   # None si pas encore de frame
        src.stop()
    """

    # Paramètres du backoff exponentiel de reconnexion
    RECONNECT_DELAY_MIN = 2.0
    RECONNECT_DELAY_MAX = 60.0
    RECONNECT_BACKOFF_FACTOR = 2.0

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._reconnect_delay = self.RECONNECT_DELAY_MIN
        self._reconnect_count = 0

    # ── Propriétés ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def uid(self) -> str:
        return self.config.uid

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Ouvre la capture et démarre la boucle de lecture. Retourne True si OK."""
        if self._running:
            return True

        ok = self._open_capture()
        if not ok:
            logger.error("[%s] Impossible d'ouvrir la source : %s", self.name, self.config.source)
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name=f"cam-{self.uid}",
        )
        self._thread.start()
        logger.info("[%s] Démarré (source=%s)", self.name, self.config.source)
        return True

    def stop(self) -> None:
        """Arrête la capture proprement."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._release()
        logger.info("[%s] Arrêté", self.name)

    # ── Lecture de frame ──────────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """Retourne la dernière frame disponible (thread-safe), ou None."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    # ── Implémentation interne ────────────────────────────────────────────────

    def _open_capture(self) -> bool:
        cap = cv2.VideoCapture(self.config.source)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap = cap
        self._connected = True
        return True

    def _release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self._connected = False

    def _read_loop(self) -> None:
        while self._running:
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    with self._lock:
                        self._latest_frame = frame
                    self._connected = True
                    # Réinitialiser le backoff après une lecture réussie
                    self._reconnect_delay = self.RECONNECT_DELAY_MIN
                    self._reconnect_count = 0
                else:
                    self._connected = False
                    self._release()
                    self._reconnect_count += 1
                    logger.warning(
                        "[%s] Perte de flux (tentative %d), nouvelle connexion dans %.0fs…",
                        self.name, self._reconnect_count, self._reconnect_delay,
                    )
                    time.sleep(self._reconnect_delay)
                    # Backoff exponentiel plafonné
                    self._reconnect_delay = min(
                        self._reconnect_delay * self.RECONNECT_BACKOFF_FACTOR,
                        self.RECONNECT_DELAY_MAX,
                    )
                    self._open_capture()
            else:
                time.sleep(0.1)


# ── Implémentations concrètes ────────────────────────────────────────────────

class WebcamSource(CameraSource):
    """
    Caméra locale (USB ou intégrée).

    config.source = index entier (0, 1, 2…)
    """

    def _open_capture(self) -> bool:
        # Forcer l'index entier
        source = int(self.config.source) if not isinstance(self.config.source, int) else self.config.source
        self.config.source = source
        return super()._open_capture()


class IPCameraSource(CameraSource):
    """
    Caméra distante via RTSP ou HTTP-MJPEG.

    config.source = URL, exemples :
      rtsp://192.168.1.50:554/live          (caméra IP standard)
      http://192.168.1.50:8080/video        (Android IP Webcam)
      http://192.168.1.50:8080/mjpeg        (variante MJPEG)
    """
    pass


# ── Fabrique ─────────────────────────────────────────────────────────────────

def create_camera_source(config: CameraConfig) -> CameraSource:
    """Instancie la bonne sous-classe selon config.source_type."""
    if config.source_type == "webcam":
        return WebcamSource(config)
    if config.source_type == "ip":
        return IPCameraSource(config)
    raise ValueError(f"Type de source inconnu : {config.source_type!r}")
