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

import cv2
import numpy as np

from ..domain.camera import CameraConfig
from ..domain.connection import ConnectionState, ReconnectPolicy

# CameraConfig est réexporté : l'UI l'importe depuis ce module.
__all__ = [
    "CameraConfig",
    "CameraSource",
    "ConnectionState",
    "IPCameraSource",
    "ReconnectPolicy",
    "WebcamSource",
    "create_camera_source",
]

logger = logging.getLogger(__name__)


# ── Classe de base ────────────────────────────────────────────────────────────


class CameraSource:
    """
    Source vidéo générique avec boucle de lecture en arrière-plan.

    Machine à états :

        DISCONNECTED ──tentative──► CONNECTING ──succès──► CONNECTED
              ▲                          │                     │
              └──────échec, backoff──────┘◄───perte de flux────┘

    `DISCONNECTED` retente toujours. L'implémentation précédente restait
    définitivement inerte lorsqu'une réouverture échouait.

    Les attentes utilisent `Event.wait()` et non `time.sleep()`, afin que
    `stop()` rende la main immédiatement même au milieu d'un backoff de 60 s.

    Usage :
        src = WebcamSource(config)
        src.start()
        frame = src.get_frame()   # None si pas encore de frame
        src.stop()
    """

    # Pause de la boucle quand la source est saine, pour ne pas saturer le CPU
    IDLE_POLL = 0.005

    def __init__(self, config: CameraConfig, policy: ReconnectPolicy | None = None) -> None:
        self.config = config
        self._policy = policy or ReconnectPolicy()
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state = ConnectionState.DISCONNECTED
        self._attempts = 0
        self._next_retry_in = 0.0

    # ── Propriétés ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def uid(self) -> str:
        return self.config.uid

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_connected(self) -> bool:
        return self._state is ConnectionState.CONNECTED

    @property
    def reconnect_attempts(self) -> int:
        """Nombre d'échecs consécutifs. Remis à zéro dès qu'une ouverture réussit."""
        return self._attempts

    @property
    def next_retry_in(self) -> float:
        """Délai avant la prochaine tentative, 0 si la source est connectée."""
        return self._next_retry_in

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Ouvre la capture et démarre la boucle de lecture. Idempotent."""
        if self.is_running:
            return True

        self._stop_event.clear()
        self._state = ConnectionState.CONNECTING
        if not self._try_open():
            self._state = ConnectionState.DISCONNECTED
            logger.error("[%s] Impossible d'ouvrir la source : %s", self.name, self.config.source)
            return False

        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name=f"cam-{self.uid}",
        )
        self._thread.start()
        logger.info("[%s] Démarré (source=%s)", self.name, self.config.source)
        return True

    def stop(self) -> None:
        """Arrête la capture. Idempotent, et interrompt un backoff en cours."""
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("[%s] Le thread de lecture ne s'est pas arrêté", self.name)
        self._release()
        self._state = ConnectionState.DISCONNECTED
        self._next_retry_in = 0.0
        logger.info("[%s] Arrêté", self.name)

    # ── Lecture de frame ──────────────────────────────────────────────────────

    def get_frame(self) -> np.ndarray | None:
        """Dernière frame disponible (copie, thread-safe), ou None."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    # ── Points de surcharge ───────────────────────────────────────────────────

    def _open_capture(self) -> bool:
        """Ouvre la capture. Retourne True si elle est utilisable."""
        cap = cv2.VideoCapture(self.config.source)
        if not cap.isOpened():
            # Libérer même une capture inutilisable : sinon le descripteur fuit.
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap = cap
        return True

    def _read_raw(self) -> tuple[bool, np.ndarray | None]:
        """
        Lit une frame. Retourne (succès, frame).

        Le retournement miroir est appliqué ici, à la source, et non à
        l'affichage : reconnaissance, annotations, clips et instantanés portent
        ainsi tous sur la même image que celle montrée à l'écran. Le corollaire
        est que la ROI se définit dans le repère de l'image retournée — c'est
        bien celui que l'utilisateur voit quand il la trace.
        """
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        if self.config.mirror:
            frame = cv2.flip(frame, 1)
        return True, frame

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Machine à états ───────────────────────────────────────────────────────

    def _try_open(self) -> bool:
        """Tente une ouverture et met l'état à jour."""
        if self._open_capture():
            self._state = ConnectionState.CONNECTED
            self._attempts = 0
            self._next_retry_in = 0.0
            return True
        self._release()
        return False

    def _handle_failure(self) -> None:
        """Perte de flux ou ouverture ratée : libère, attend, puis retente."""
        self._state = ConnectionState.DISCONNECTED
        self._release()
        self._attempts += 1
        delai = self._policy.next_delay(self._attempts)
        self._next_retry_in = delai
        logger.warning(
            "[%s] Flux indisponible (tentative %d), nouvelle connexion dans %.0f s…",
            self.name,
            self._attempts,
            delai,
        )
        # Event.wait plutôt que time.sleep : stop() interrompt l'attente.
        if self._stop_event.wait(delai):
            return
        self._state = ConnectionState.CONNECTING
        self._try_open()

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._state is not ConnectionState.CONNECTED:
                # Toujours retenter — c'est ce qui manquait auparavant.
                self._handle_failure()
                continue

            ok, frame = self._read_raw()
            if not ok:
                self._handle_failure()
                continue

            with self._lock:
                self._latest_frame = frame
            self._stop_event.wait(self.IDLE_POLL)


# ── Implémentations concrètes ────────────────────────────────────────────────


class WebcamSource(CameraSource):
    """
    Caméra locale (USB ou intégrée).

    config.source = index entier (0, 1, 2…)
    """

    def _open_capture(self) -> bool:
        # Forcer l'index entier
        source = (
            int(self.config.source)
            if not isinstance(self.config.source, int)
            else self.config.source
        )
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
