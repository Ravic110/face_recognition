"""
surveillance_engine.py
Moteur de reconnaissance faciale multi-caméras — version améliorée.

Améliorations v2 :
  - Détection de mouvement (MOG2) avant la reconnaissance → économie CPU
  - File d'attente (Queue) pour découpler lecture caméra et analyse
  - Suivi FPS d'analyse par caméra
  - ROI (zone d'intérêt) configurable par caméra
  - Modèle de détection configurable (hog / cnn) via SurveillanceProfile
  - Intégration VideoRecorder (buffer + clip sur événement)
  - Intégration AlertManager
  - Application du SurveillanceProfile actif
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

from ..storage.config import FACE_RECOGNITION_THRESHOLD
from ..storage.encodings_store import load_encodings_map
from .camera_manager import CameraManager
from .motion_detector import MotionDetector

logger = logging.getLogger(__name__)


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class DetectedFace:
    location: Tuple[int, int, int, int]
    name: str
    confidence: float
    is_known: bool


@dataclass
class SurveillanceEvent:
    camera_uid: str
    camera_name: str
    timestamp: float
    faces: List[DetectedFace]
    motion_score: float = 0.0
    frame: Optional[np.ndarray] = None

    @property
    def known_names(self) -> List[str]:
        return [f.name for f in self.faces if f.is_known]

    @property
    def has_unknown(self) -> bool:
        return any(not f.is_known for f in self.faces)


EventCallback = Callable[[SurveillanceEvent], None]


# ── Stats par caméra ──────────────────────────────────────────────────────────

@dataclass
class CameraStats:
    fps: float = 0.0
    frames_analysed: int = 0
    detections: int = 0
    motion_triggers: int = 0
    last_detection_ts: float = 0.0

    def _fps_tick(self) -> None:
        pass   # calculé dans la boucle


# ── Moteur principal ──────────────────────────────────────────────────────────

class SurveillanceEngine:
    """
    Lance un thread d'analyse par caméra avec file d'attente, motion detection,
    ROI, modèle configurable, enregistrement vidéo et alertes.
    """

    QUEUE_MAXSIZE = 4           # frames en attente d'analyse par caméra
    CACHE_REFRESH_INTERVAL = 30.0

    def __init__(
        self,
        camera_manager: CameraManager,
        threshold: float = FACE_RECOGNITION_THRESHOLD,
    ) -> None:
        self._mgr = camera_manager
        self._threshold = threshold
        self._listeners: List[EventCallback] = []

        self._threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._queues: Dict[str, queue.Queue] = {}
        self._running = False

        # Stats publiques
        self.stats: Dict[str, CameraStats] = {}
        self._stats_lock = threading.Lock()

        # Cache encodages
        self._encodings_cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._last_cache_refresh = 0.0

        # Motion detectors (un par caméra)
        self._motion_detectors: Dict[str, MotionDetector] = {}

        # Paramètres du profil actif (mis à jour via apply_profile)
        self._motion_required = True
        self._detection_model = "hog"
        self._analysis_interval = 0.5

        # Composants optionnels (injectés après construction)
        self._recorder = None    # VideoRecorder
        self._alert_mgr = None   # AlertManager

    # ── Injection de dépendances optionnelles ─────────────────────────────────

    def set_recorder(self, recorder) -> None:
        self._recorder = recorder

    def set_alert_manager(self, alert_mgr) -> None:
        self._alert_mgr = alert_mgr

    # ── Profil de surveillance ────────────────────────────────────────────────

    def apply_profile(self, profile) -> None:
        """Applique un SurveillanceProfile. Peut être appelé même pendant l'analyse."""
        self._motion_required = profile.motion_required
        self._detection_model = profile.detection_model
        self._analysis_interval = profile.analysis_interval
        self._threshold = profile.recognition_threshold
        # Mettre à jour les détecteurs de mouvement existants
        for det in self._motion_detectors.values():
            det._sensitivity = profile.motion_sensitivity
        logger.info("Profil appliqué : %s", profile.label)

    # ── Listeners ─────────────────────────────────────────────────────────────

    def add_event_listener(self, callback: EventCallback) -> None:
        self._listeners.append(callback)

    def remove_event_listener(self, callback: EventCallback) -> None:
        self._listeners = [cb for cb in self._listeners if cb is not callback]

    def _emit(self, event: SurveillanceEvent) -> None:
        for cb in self._listeners:
            try:
                cb(event)
            except Exception as exc:
                logger.error("Erreur listener surveillance : %s", exc)

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._refresh_encodings()
        for uid in self._mgr.get_all_sources():
            self._start_camera_thread(uid)
        self._mgr.on_change(self._sync_threads)
        logger.info("SurveillanceEngine v2 démarré (%d caméra(s))", len(self._threads))

    def stop(self) -> None:
        self._running = False
        for evt in self._stop_events.values():
            evt.set()
        for t in self._threads.values():
            t.join(timeout=3.0)
        self._threads.clear()
        self._stop_events.clear()
        self._queues.clear()
        logger.info("SurveillanceEngine arrêté")

    def _sync_threads(self) -> None:
        active = set(self._mgr.get_all_sources().keys())
        running = set(self._threads.keys())
        for uid in active - running:
            self._start_camera_thread(uid)
        for uid in running - active:
            self._stop_camera_thread(uid)

    def _start_camera_thread(self, uid: str) -> None:
        if uid in self._threads:
            return
        stop_evt = threading.Event()
        q: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self._stop_events[uid] = stop_evt
        self._queues[uid] = q
        self._motion_detectors[uid] = MotionDetector()
        with self._stats_lock:
            self.stats[uid] = CameraStats()

        # Thread producteur : lit les frames et les met en queue
        producer = threading.Thread(
            target=self._produce_loop, args=(uid, q, stop_evt),
            daemon=True, name=f"prod-{uid}",
        )
        # Thread consommateur : analyse les frames de la queue
        consumer = threading.Thread(
            target=self._analyse_loop, args=(uid, q, stop_evt),
            daemon=True, name=f"surv-{uid}",
        )
        producer.start()
        consumer.start()
        self._threads[uid] = consumer   # on suit le consommateur

    def _stop_camera_thread(self, uid: str) -> None:
        stop_evt = self._stop_events.pop(uid, None)
        if stop_evt:
            stop_evt.set()
        t = self._threads.pop(uid, None)
        if t:
            t.join(timeout=3.0)
        self._queues.pop(uid, None)
        self._motion_detectors.pop(uid, None)

    # ── Thread producteur : pushes frames dans la Queue ───────────────────────

    def _produce_loop(self, uid: str, q: queue.Queue, stop_evt: threading.Event) -> None:
        while not stop_evt.is_set():
            frame = self._mgr.get_frame(uid)
            if frame is not None:
                # Alimenter le recorder si actif
                if self._recorder is not None:
                    try:
                        self._recorder.push_frame(uid, frame)
                    except Exception:
                        pass
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    pass   # Dropper les frames si l'analyse est trop lente
            stop_evt.wait(0.033)    # ~30 FPS producteur

    # ── Thread consommateur : analyse ────────────────────────────────────────

    def _analyse_loop(self, uid: str, q: queue.Queue, stop_evt: threading.Event) -> None:
        config = self._mgr.get_config(uid)
        cam_name = config.name if config else uid
        motion_det = self._motion_detectors.get(uid)

        fps_frames = 0
        fps_start = time.monotonic()

        while not stop_evt.is_set():
            t0 = time.monotonic()

            try:
                frame = q.get(timeout=0.5)
            except queue.Empty:
                continue

            # ── Détection de mouvement ────────────────────────────────────────
            motion_detected = True
            motion_score = 1.0
            if self._motion_required and motion_det is not None:
                motion_detected, motion_score = motion_det.update(frame)
                if not motion_detected:
                    fps_frames += 1
                    continue   # Pas de mouvement → on saute la reconnaissance

            with self._stats_lock:
                if uid in self.stats:
                    self.stats[uid].motion_triggers += 1

            # ── Mise à jour cache encodages ────────────────────────────────────
            self._maybe_refresh_encodings()

            # ── Appliquer le ROI sur la frame si configuré ─────────────────────
            analysis_frame = self._apply_roi(frame, config)

            # ── Reconnaissance ─────────────────────────────────────────────────
            faces = self._process_frame(analysis_frame)
            fps_frames += 1

            # ── FPS ─────────────────────────────────────────────────────────────
            elapsed_fps = time.monotonic() - fps_start
            if elapsed_fps >= 2.0:
                fps = fps_frames / elapsed_fps
                with self._stats_lock:
                    if uid in self.stats:
                        self.stats[uid].fps = round(fps, 1)
                fps_frames = 0
                fps_start = time.monotonic()

            if faces:
                annotated = self._annotate_frame(frame.copy(), faces, config)
                event = SurveillanceEvent(
                    camera_uid=uid,
                    camera_name=cam_name,
                    timestamp=time.time(),
                    faces=faces,
                    motion_score=motion_score,
                    frame=annotated,
                )
                with self._stats_lock:
                    if uid in self.stats:
                        self.stats[uid].detections += 1
                        self.stats[uid].frames_analysed += 1
                        self.stats[uid].last_detection_ts = event.timestamp

                # Déclencher l'enregistrement vidéo
                if self._recorder is not None:
                    try:
                        self._recorder.trigger_recording(uid, cam_name)
                    except Exception:
                        pass

                # Déclencher les alertes
                if self._alert_mgr is not None:
                    try:
                        faces_data = [
                            {"name": f.name, "confidence": f.confidence, "is_known": f.is_known}
                            for f in faces
                        ]
                        self._alert_mgr.notify(
                            camera_name=cam_name,
                            faces=faces_data,
                            snapshot_b64=None,
                        )
                    except Exception:
                        pass

                self._emit(event)

            # Respecter l'intervalle d'analyse
            elapsed = time.monotonic() - t0
            wait = max(0.0, self._analysis_interval - elapsed)
            if wait > 0:
                stop_evt.wait(wait)

    # ── Traitement d'une frame ────────────────────────────────────────────────

    def _apply_roi(self, frame: np.ndarray, config) -> np.ndarray:
        if config is None or config.roi is None:
            return frame
        x, y, w, h = config.roi
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2]
        return frame

    def _process_frame(self, frame: np.ndarray) -> List[DetectedFace]:
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb, model=self._detection_model)
        if not locations:
            return []

        encodings = face_recognition.face_encodings(rgb, locations)

        with self._cache_lock:
            known_names = list(self._encodings_cache.keys())
            known_encs = list(self._encodings_cache.values())

        results: List[DetectedFace] = []
        for loc, enc in zip(locations, encodings):
            top, right, bottom, left = loc
            loc_full = (top * 2, right * 2, bottom * 2, left * 2)

            name, confidence, is_known = "Inconnu", 0.0, False
            if known_encs:
                distances = face_recognition.face_distance(known_encs, enc)
                best_idx = int(np.argmin(distances))
                best_dist = float(distances[best_idx])
                if best_dist < self._threshold:
                    name = known_names[best_idx]
                    confidence = round(1.0 - best_dist, 3)
                    is_known = True
                else:
                    confidence = round(1.0 - best_dist, 3)

            results.append(DetectedFace(
                location=loc_full, name=name, confidence=confidence, is_known=is_known,
            ))
        return results

    def _annotate_frame(self, frame: np.ndarray, faces: List[DetectedFace], config) -> np.ndarray:
        # Décaler si ROI active
        off_x = off_y = 0
        if config and config.roi:
            off_x, off_y = config.roi[0], config.roi[1]

        for face in faces:
            top, right, bottom, left = face.location
            top += off_y; bottom += off_y; left += off_x; right += off_x
            color = (0, 200, 0) if face.is_known else (0, 0, 220)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{face.name} ({face.confidence:.0%})"
            cv2.rectangle(frame, (left, bottom - 22), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 4, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    # ── Cache encodages ───────────────────────────────────────────────────────

    def _maybe_refresh_encodings(self) -> None:
        if time.monotonic() - self._last_cache_refresh > self.CACHE_REFRESH_INTERVAL:
            self._refresh_encodings()

    def _refresh_encodings(self) -> None:
        try:
            new_map = load_encodings_map()
            with self._cache_lock:
                self._encodings_cache = {
                    name: np.array(enc) for name, enc in new_map.items()
                }
            self._last_cache_refresh = time.monotonic()
            logger.debug("Cache encodages rechargé (%d visage(s))", len(self._encodings_cache))
        except Exception as exc:
            logger.error("Erreur rechargement encodages : %s", exc)

    def force_refresh_encodings(self) -> None:
        self._last_cache_refresh = 0.0

    # ── Accès aux stats ───────────────────────────────────────────────────────

    def get_stats(self, uid: str) -> Optional[CameraStats]:
        with self._stats_lock:
            return self.stats.get(uid)
