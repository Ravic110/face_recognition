"""
server.py
API REST locale — fermée par défaut, ouverte explicitement.

Trois changements par rapport à `services/api_server.py`, qui est supprimé :

  - **Authentification.** Toute route exige `X-API-Key`. L'API exposait
    auparavant les flux caméra, l'historique nominatif et l'arrêt de la
    surveillance sans aucun contrôle.
  - **Écoute locale.** L'hôte vient de `settings.api_host`, à `127.0.0.1` par
    défaut, au lieu d'un `0.0.0.0` codé en dur.
  - **Arrêt réel.** `werkzeug.serving.make_server` expose `shutdown()`. La
    version précédente se contentait de journaliser : le thread continuait
    d'écouter, `is_running` restait vrai, et l'API ne pouvait plus être
    réactivée.

Les routes d'écriture sont gouvernées par `settings.api_allow_control`, à
`false` par défaut : consulter ses caméras à distance n'oblige pas à laisser
la télécommande ouverte.

Endpoints :
  GET  /api/status              état général
  GET  /api/cameras             caméras configurées
  GET  /api/snapshot/<uid>      dernière frame (JPEG)
  GET  /api/events              événements récents (limit, camera, person)
  GET  /api/events/<id>         détail + snapshot base64
  GET  /api/faces               personnes enregistrées
  GET  /api/clips               clips disponibles
  POST /api/surveillance/start  démarrer   (si api_allow_control)
  POST /api/surveillance/stop   arrêter    (si api_allow_control)
"""

from __future__ import annotations

import logging
import threading
from functools import wraps
from typing import TYPE_CHECKING, Any

import cv2

from ..settings import AppSettings
from .auth import ApiKeyGuard, resolve_api_key

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..services.camera_manager import CameraManager
    from ..services.video_recorder import VideoRecorder
    from ..storage.event_repository import EventRepository

MAX_EVENTS = 200
SNAPSHOT_QUALITY = 70


class ApiServer:
    """Serveur Flask embarqué, authentifié et réellement arrêtable."""

    VERSION = "2.0"

    def __init__(
        self,
        settings: AppSettings,
        camera_manager: CameraManager,
        engine: Any,
        event_repository: EventRepository,
        recorder: VideoRecorder,
    ) -> None:
        self._settings = settings
        self._mgr = camera_manager
        self._engine = engine
        self._events = event_repository
        self._recorder = recorder

        self._key = resolve_api_key(settings)
        self._guard = ApiKeyGuard(self._key)

        self._server: Any = None
        self._thread: threading.Thread | None = None

        self.app = self._build_app()

    # ── Propriétés ────────────────────────────────────────────────────────────

    @property
    def api_key(self) -> str:
        return self._key

    @property
    def port(self) -> int:
        """Port réellement lié. Vaut le port configuré tant que rien n'écoute."""
        if self._server is not None:
            return int(self._server.server_port)
        return self._settings.api_port

    @property
    def url(self) -> str:
        return f"http://{self._settings.api_host}:{self.port}"

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build_app(self):
        from flask import Flask, Response, jsonify, request

        app = Flask(__name__)
        app.config["JSON_SORT_KEYS"] = False

        def authentifie(vue):
            """Exige une clé valide et applique la limitation par adresse."""

            @wraps(vue)
            def enveloppe(*args, **kwargs):
                ip = request.remote_addr or "inconnue"
                if self._guard.is_blocked(ip):
                    return jsonify({"error": "Trop de tentatives"}), 429
                if not self._guard.check(request.headers.get("X-API-Key"), ip):
                    return jsonify({"error": "Clé d'API invalide ou absente"}), 401
                return vue(*args, **kwargs)

            return enveloppe

        def controle_autorise(vue):
            """Refuse les routes d'écriture tant que api_allow_control est faux."""

            @wraps(vue)
            def enveloppe(*args, **kwargs):
                if not self._settings.api_allow_control:
                    return jsonify(
                        {"error": "Contrôle à distance désactivé (FR_API_ALLOW_CONTROL)"}
                    ), 403
                return vue(*args, **kwargs)

            return enveloppe

        # ── Statut ────────────────────────────────────────────────────────────

        @app.route("/api/status")
        @authentifie
        def status():
            return jsonify(
                {
                    "version": self.VERSION,
                    "surveillance_active": bool(getattr(self._engine, "is_running", False)),
                    "cameras_total": len(self._mgr.list_configs()),
                    "cameras_running": len(self._mgr.get_all_sources()),
                    "control_enabled": self._settings.api_allow_control,
                }
            )

        # ── Caméras ───────────────────────────────────────────────────────────

        @app.route("/api/cameras")
        @authentifie
        def cameras():
            return jsonify(
                [
                    {**c.to_dict(), "running": self._mgr.is_running(c.uid)}
                    for c in self._mgr.list_configs()
                ]
            )

        @app.route("/api/snapshot/<uid>")
        @authentifie
        def snapshot(uid: str):
            frame = self._mgr.get_frame(uid)
            if frame is None:
                return jsonify({"error": "Caméra indisponible"}), 404
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY])
            if not ok:
                logger.error("Encodage JPEG du snapshot de %s échoué", uid)
                return jsonify({"error": "Encodage échoué"}), 500
            return Response(buf.tobytes(), mimetype="image/jpeg")

        # ── Événements ────────────────────────────────────────────────────────

        @app.route("/api/events")
        @authentifie
        def events():
            try:
                limit = min(int(request.args.get("limit", 50)), MAX_EVENTS)
            except ValueError:
                return jsonify({"error": "Le paramètre 'limit' doit être un entier"}), 400

            personne = request.args.get("person")
            camera = request.args.get("camera")
            if personne:
                trouves = self._events.get_by_person(personne, limit)
            elif camera:
                trouves = self._events.get_by_camera(camera, limit)
            else:
                trouves = self._events.get_recent(limit)

            return jsonify(
                [
                    {
                        "id": e.id,
                        "datetime": e.dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "camera_uid": e.camera_uid,
                        "camera_name": e.camera_name,
                        "faces": [f.to_dict() for f in e.faces],
                        "has_snapshot": e.snapshot_b64 is not None,
                    }
                    for e in trouves
                ]
            )

        @app.route("/api/events/<int:event_id>")
        @authentifie
        def event_detail(event_id: int):
            evt = self._events.get_by_id(event_id)
            if evt is None:
                return jsonify({"error": "Événement introuvable"}), 404
            return jsonify(evt.to_dict())

        # ── Personnes et clips ────────────────────────────────────────────────

        @app.route("/api/faces")
        @authentifie
        def faces():
            from ..storage.encodings_repository import EncodingsRepository

            metadata = EncodingsRepository(self._settings).load_metadata()
            return jsonify([{"uid": uid, **info} for uid, info in metadata.items()])

        @app.route("/api/clips")
        @authentifie
        def clips():
            return jsonify(
                [
                    {"name": p.name, "size_mb": round(p.stat().st_size / 1_048_576, 2)}
                    for p in self._recorder.list_clips()
                ]
            )

        # ── Contrôle de la surveillance ───────────────────────────────────────

        @app.route("/api/surveillance/start", methods=["POST"])
        @authentifie
        @controle_autorise
        def surv_start():
            if not getattr(self._engine, "is_running", False):
                self._mgr.start_all()
                self._engine.start()
            return jsonify({"ok": True, "active": True})

        @app.route("/api/surveillance/stop", methods=["POST"])
        @authentifie
        @controle_autorise
        def surv_stop():
            if getattr(self._engine, "is_running", False):
                self._engine.stop()
                self._mgr.stop_all()
            return jsonify({"ok": True, "active": False})

        return app

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Lance le serveur. Idempotent. Retourne True si le serveur écoute."""
        if self.is_running:
            return True

        from werkzeug.serving import make_server

        hote, port = self._settings.api_host, self._settings.api_port
        try:
            self._server = make_server(hote, port, self.app, threaded=True)
        except OSError as exc:
            logger.error("Impossible d'écouter sur %s:%d : %s", hote, port, exc)
            self._server = None
            return False

        # poll_interval court : shutdown() attend au plus ce délai avant de rendre
        # la main. Le défaut de werkzeug (0,5 s) rend chaque arrêt perceptible.
        self._thread = threading.Thread(
            target=lambda: self._server.serve_forever(poll_interval=0.1),
            name="api-server",
            daemon=True,
        )
        self._thread.start()

        if hote not in ("127.0.0.1", "localhost"):
            logger.warning(
                "API exposée sur %s — accessible depuis le réseau. "
                "La clé d'API est le seul rempart.",
                hote,
            )
        logger.info("API REST démarrée sur %s/api/status", self.url)
        return True

    def stop(self) -> None:
        """Ferme réellement l'écoute. Idempotent."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("API REST arrêtée")
