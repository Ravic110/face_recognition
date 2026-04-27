"""
video_recorder.py
Enregistrement vidéo déclenché par événement.

Fonctionnement :
  - Buffer circulaire en mémoire : conserve les N dernières secondes de chaque caméra
  - Quand un événement de détection survient, le buffer est vidé dans un fichier .mp4
    (incluant les frames AVANT la détection + les frames APRÈS)
  - Les clips sont nommés avec la date, l'heure et le nom de la caméra

Structure des clips :
  clips/
    CAM_NOM-2026-04-16_14-30-00.mp4
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from ..storage.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

CLIPS_DIR = PROJECT_ROOT / "clips"


class CameraBuffer:
    """Buffer circulaire de frames pour une caméra."""

    def __init__(self, pre_seconds: float, fps: float = 15.0) -> None:
        self._fps = fps
        capacity = int(pre_seconds * fps)
        self._frames: Deque[np.ndarray] = deque(maxlen=max(1, capacity))
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frames.append(frame.copy())

    def drain(self) -> list:
        """Vide et retourne toutes les frames du buffer."""
        with self._lock:
            frames = list(self._frames)
            self._frames.clear()
            return frames

    @property
    def fps(self) -> float:
        return self._fps


class VideoRecorder:
    """
    Gestionnaire d'enregistrement de clips vidéo multi-caméras.

    Usage :
        rec = VideoRecorder(clips_dir, pre_seconds=5, post_seconds=10)
        rec.push_frame("cam1", frame)           # appelé à chaque frame
        rec.trigger_recording("cam1", "Salon")  # déclenche un clip
    """

    FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

    def __init__(
        self,
        clips_dir: Path = CLIPS_DIR,
        pre_seconds: float = 5.0,
        post_seconds: float = 10.0,
        fps: float = 15.0,
        max_clips: int = 100,
    ) -> None:
        self._dir = clips_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._pre = pre_seconds
        self._post = post_seconds
        self._fps = fps
        self._max_clips = max_clips

        # Buffer par caméra
        self._buffers: Dict[str, CameraBuffer] = {}
        self._buf_lock = threading.Lock()

        # Threads d'enregistrement en cours
        self._rec_threads: Dict[str, threading.Thread] = {}
        self._rec_lock = threading.Lock()

    # ── API principale ────────────────────────────────────────────────────────

    def push_frame(self, camera_uid: str, frame: np.ndarray) -> None:
        """Ajoute une frame au buffer de la caméra (appelé depuis le thread caméra)."""
        with self._buf_lock:
            if camera_uid not in self._buffers:
                self._buffers[camera_uid] = CameraBuffer(self._pre, self._fps)
            buf = self._buffers[camera_uid]
        buf.push(frame)

    def trigger_recording(self, camera_uid: str, camera_name: str) -> Optional[Path]:
        """
        Déclenche l'enregistrement d'un clip pour la caméra donnée.
        Lance un thread de post-capture non bloquant.
        Retourne le chemin du clip (prévu) ou None si déjà en cours.
        """
        with self._rec_lock:
            if camera_uid in self._rec_threads and self._rec_threads[camera_uid].is_alive():
                return None   # déjà en cours d'enregistrement

        with self._buf_lock:
            buf = self._buffers.get(camera_uid)
        if buf is None:
            return None

        pre_frames = buf.drain()
        clip_path = self._clip_path(camera_name)
        t = threading.Thread(
            target=self._write_clip,
            args=(camera_uid, pre_frames, clip_path),
            daemon=True,
        )
        with self._rec_lock:
            self._rec_threads[camera_uid] = t
        t.start()

        self._enforce_max_clips()
        return clip_path

    def is_recording(self, camera_uid: str) -> bool:
        with self._rec_lock:
            t = self._rec_threads.get(camera_uid)
        return t is not None and t.is_alive()

    def remove_camera(self, camera_uid: str) -> None:
        with self._buf_lock:
            self._buffers.pop(camera_uid, None)

    # ── Écriture du clip ──────────────────────────────────────────────────────

    def _write_clip(
        self,
        camera_uid: str,
        pre_frames: list,
        clip_path: Path,
    ) -> None:
        """Thread : écrit les frames pré-détection puis capture les frames post."""
        writer: Optional[cv2.VideoWriter] = None
        frame_size: Optional[Tuple[int, int]] = None
        post_count = int(self._post * self._fps)
        written = 0

        try:
            # Écrire les frames pré-détection
            for frame in pre_frames:
                if writer is None:
                    h, w = frame.shape[:2]
                    frame_size = (w, h)
                    writer = cv2.VideoWriter(
                        str(clip_path), self.FOURCC, self._fps, frame_size
                    )
                writer.write(frame)
                written += 1

            # Capturer les frames post-détection depuis le buffer
            deadline = time.monotonic() + self._post
            while time.monotonic() < deadline:
                with self._buf_lock:
                    buf = self._buffers.get(camera_uid)
                if buf is None:
                    break
                new_frames = buf.drain()
                for frame in new_frames:
                    if writer is None:
                        h, w = frame.shape[:2]
                        frame_size = (w, h)
                        writer = cv2.VideoWriter(
                            str(clip_path), self.FOURCC, self._fps, frame_size
                        )
                    writer.write(frame)
                    written += 1
                time.sleep(1.0 / self._fps)

            logger.info("Clip enregistré : %s (%d frames)", clip_path.name, written)
        except Exception as exc:
            logger.error("Erreur enregistrement clip : %s", exc)
        finally:
            if writer:
                writer.release()

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def _clip_path(self, camera_name: str) -> Path:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in camera_name)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return self._dir / f"{safe_name}-{ts}.mp4"

    def _enforce_max_clips(self) -> None:
        """Supprime les clips les plus anciens si la limite est atteinte."""
        clips = sorted(self._dir.glob("*.mp4"), key=os.path.getmtime)
        while len(clips) > self._max_clips:
            try:
                clips.pop(0).unlink()
            except OSError:
                break

    def list_clips(self) -> list:
        """Retourne la liste des clips triée du plus récent au plus ancien."""
        return sorted(self._dir.glob("*.mp4"), key=os.path.getmtime, reverse=True)
