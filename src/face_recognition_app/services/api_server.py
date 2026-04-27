"""
api_server.py
API REST locale (Flask) pour consulter le système depuis un navigateur ou smartphone.

Endpoints :
  GET  /api/status          → état général (caméras, surveillance, version)
  GET  /api/cameras         → liste des caméras configurées
  GET  /api/events          → événements récents (query: limit, camera, person)
  GET  /api/events/<id>     → détail d'un événement + snapshot base64
  GET  /api/faces           → personnes enregistrées dans la base
  GET  /api/clips           → liste des clips vidéo disponibles
  POST /api/surveillance/start  → démarrer la surveillance
  POST /api/surveillance/stop   → arrêter la surveillance
  GET  /api/snapshot/<uid>  → dernière frame d'une caméra (JPEG)

Le serveur tourne dans un thread daemon et s'arrête avec l'application.
"""

from __future__ import annotations

import base64
import logging
import threading
from typing import TYPE_CHECKING, Callable, Optional

import cv2

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .camera_manager import CameraManager
    from .surveillance_engine import SurveillanceEngine
    from ..storage.event_store import EventStore
    from .video_recorder import VideoRecorder


class ApiServer:
    """
    Serveur Flask embarqué.

    Usage :
        server = ApiServer(camera_manager, engine, event_store, recorder)
        server.start(host="0.0.0.0", port=5000)
        ...
        server.stop()
    """

    VERSION = "1.0"

    def __init__(
        self,
        camera_manager: "CameraManager",
        engine: "SurveillanceEngine",
        event_store: "EventStore",
        recorder: "VideoRecorder",
    ) -> None:
        self._mgr = camera_manager
        self._engine = engine
        self._store = event_store
        self._recorder = recorder
        self._thread: Optional[threading.Thread] = None
        self._app = self._build_app()

    # ── Construction Flask ────────────────────────────────────────────────────

    def _build_app(self):
        from flask import Flask, jsonify, request, Response

        app = Flask(__name__)
        app.config["JSON_SORT_KEYS"] = False

        # Désactiver les logs Flask en production
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        # ── Statut ────────────────────────────────────────────────────────────

        @app.route("/api/status")
        def status():
            return jsonify({
                "version": self.VERSION,
                "surveillance_active": self._engine._running,
                "cameras_total": len(self._mgr.list_configs()),
                "cameras_running": len(self._mgr.get_all_sources()),
            })

        # ── Caméras ───────────────────────────────────────────────────────────

        @app.route("/api/cameras")
        def cameras():
            return jsonify([
                {**c.to_dict(), "running": self._mgr.is_running(c.uid)}
                for c in self._mgr.list_configs()
            ])

        @app.route("/api/snapshot/<uid>")
        def snapshot(uid: str):
            frame = self._mgr.get_frame(uid)
            if frame is None:
                return jsonify({"error": "Caméra indisponible"}), 404
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                return jsonify({"error": "Encodage échoué"}), 500
            return Response(buf.tobytes(), mimetype="image/jpeg")

        # ── Événements ────────────────────────────────────────────────────────

        @app.route("/api/events")
        def events():
            limit = min(int(request.args.get("limit", 50)), 200)
            camera = request.args.get("camera")
            person = request.args.get("person")

            if person:
                evts = self._store.get_by_person(person, limit)
            elif camera:
                evts = self._store.get_by_camera(camera, limit)
            else:
                evts = self._store.get_recent(limit)

            return jsonify([
                {
                    "id": e.id,
                    "datetime": e.dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "camera_uid": e.camera_uid,
                    "camera_name": e.camera_name,
                    "faces": e.faces,
                    "has_snapshot": e.snapshot_b64 is not None,
                }
                for e in evts
            ])

        @app.route("/api/events/<int:event_id>")
        def event_detail(event_id: int):
            evt = self._store.get_by_id(event_id)
            if evt is None:
                return jsonify({"error": "Événement introuvable"}), 404
            d = {
                "id": evt.id,
                "datetime": evt.dt.strftime("%Y-%m-%d %H:%M:%S"),
                "camera_uid": evt.camera_uid,
                "camera_name": evt.camera_name,
                "faces": evt.faces,
                "snapshot_b64": evt.snapshot_b64,
            }
            return jsonify(d)

        # ── Personnes enregistrées ────────────────────────────────────────────

        @app.route("/api/faces")
        def faces():
            from ..storage.encodings_store import load_metadata
            meta = load_metadata()
            return jsonify([
                {"uid": uid, **info} for uid, info in meta.items()
            ])

        # ── Clips ─────────────────────────────────────────────────────────────

        @app.route("/api/clips")
        def clips():
            clip_list = [
                {"name": p.name, "size_mb": round(p.stat().st_size / 1_048_576, 2)}
                for p in self._recorder.list_clips()
            ]
            return jsonify(clip_list)

        # ── Contrôle surveillance ─────────────────────────────────────────────

        @app.route("/api/surveillance/start", methods=["POST"])
        def surv_start():
            if not self._engine._running:
                self._mgr.start_all()
                self._engine.start()
            return jsonify({"ok": True, "active": True})

        @app.route("/api/surveillance/stop", methods=["POST"])
        def surv_stop():
            if self._engine._running:
                self._engine.stop()
                self._mgr.stop_all()
            return jsonify({"ok": True, "active": False})

        return app

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def start(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        """Lance le serveur Flask dans un thread daemon."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._app.run,
            kwargs={"host": host, "port": port, "debug": False, "use_reloader": False},
            daemon=True,
        )
        self._thread.start()
        logger.info("API REST démarrée sur http://%s:%d/api/status", host, port)

    def stop(self) -> None:
        """Le thread est daemon, il s'arrête avec l'application."""
        logger.info("API REST arrêtée")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
