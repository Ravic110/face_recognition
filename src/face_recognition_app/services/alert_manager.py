"""
alert_manager.py
Système d'alertes multi-canal.

Canaux supportés :
  - Notification bureau  (plyer)
  - Email SMTP           (smtplib standard)
  - Webhook HTTP POST    (requests)

Configuration persistée dans alerts_config.json.
Les alertes peuvent être filtrées : uniquement inconnus, uniquement personnes ciblées, etc.
"""

from __future__ import annotations

import json
import logging
import smtplib
import threading
from dataclasses import dataclass, field, asdict
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Optional

from ..storage.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

ALERTS_CONFIG_FILE = PROJECT_ROOT / "alerts_config.json"


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class AlertConfig:
    """Paramètres de configuration des alertes."""

    # Notification bureau
    desktop_enabled: bool = False

    # Email
    email_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_recipients: List[str] = field(default_factory=list)

    # Webhook
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_secret: str = ""

    # Filtres d'alerte
    alert_on_unknown: bool = True          # alerte si visage inconnu
    alert_on_known: bool = False           # alerte si visage connu
    target_persons: List[str] = field(default_factory=list)  # personnes ciblées spécifiquement

    # Anti-spam : délai minimal entre deux alertes (secondes)
    cooldown_seconds: float = 60.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AlertConfig":
        return cls(
            desktop_enabled=data.get("desktop_enabled", False),
            email_enabled=data.get("email_enabled", False),
            smtp_host=data.get("smtp_host", "smtp.gmail.com"),
            smtp_port=data.get("smtp_port", 587),
            smtp_user=data.get("smtp_user", ""),
            smtp_password=data.get("smtp_password", ""),
            email_recipients=data.get("email_recipients", []),
            webhook_enabled=data.get("webhook_enabled", False),
            webhook_url=data.get("webhook_url", ""),
            webhook_secret=data.get("webhook_secret", ""),
            alert_on_unknown=data.get("alert_on_unknown", True),
            alert_on_known=data.get("alert_on_known", False),
            target_persons=data.get("target_persons", []),
            cooldown_seconds=data.get("cooldown_seconds", 60.0),
        )


# ── Gestionnaire d'alertes ────────────────────────────────────────────────────

class AlertManager:
    """
    Émet des alertes selon la configuration quand un événement de surveillance est reçu.

    Usage :
        mgr = AlertManager()
        mgr.notify(camera_name="Salon", faces=[{"name": "Inconnu", "is_known": False}])
    """

    def __init__(self, config_file: Path = ALERTS_CONFIG_FILE) -> None:
        self._config_file = config_file
        self._config = AlertConfig()
        self._last_alert_time: float = 0.0
        self._lock = threading.Lock()
        self._load()

    # ── Config ────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._config_file.exists():
            try:
                data = json.loads(self._config_file.read_text(encoding="utf-8"))
                self._config = AlertConfig.from_dict(data)
            except Exception as exc:
                logger.error("Erreur lecture alerts_config.json : %s", exc)

    def save(self) -> None:
        self._config_file.write_text(
            json.dumps(self._config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def config(self) -> AlertConfig:
        return self._config

    @config.setter
    def config(self, value: AlertConfig) -> None:
        self._config = value
        self.save()

    # ── API principale ────────────────────────────────────────────────────────

    def notify(
        self,
        camera_name: str,
        faces: list,                        # [{"name": str, "is_known": bool, ...}]
        snapshot_b64: Optional[str] = None,
    ) -> None:
        """
        Analyse les visages détectés et envoie les alertes si les filtres le permettent.
        Non bloquant : les envois se font dans un thread séparé.
        """
        cfg = self._config
        if not (cfg.desktop_enabled or cfg.email_enabled or cfg.webhook_enabled):
            return

        # Vérifier les filtres
        unknown_faces = [f for f in faces if not f.get("is_known", False)]
        known_faces = [f for f in faces if f.get("is_known", False)]
        target_faces = [f for f in known_faces if f.get("name") in cfg.target_persons]

        should_alert = (
            (cfg.alert_on_unknown and unknown_faces)
            or (cfg.alert_on_known and known_faces)
            or target_faces
        )
        if not should_alert:
            return

        # Anti-spam (cooldown global)
        import time
        with self._lock:
            now = time.monotonic()
            if now - self._last_alert_time < cfg.cooldown_seconds:
                return
            self._last_alert_time = now

        # Construire le message
        parts = []
        if unknown_faces:
            parts.append(f"{len(unknown_faces)} visage(s) inconnu(s)")
        if target_faces:
            parts.append(", ".join(f["name"] for f in target_faces))
        elif known_faces and cfg.alert_on_known:
            parts.append(", ".join(f["name"] for f in known_faces))
        summary = " — ".join(parts) if parts else "Détection"
        title = f"Alerte Surveillance : {camera_name}"
        body = f"Caméra : {camera_name}\n{summary}"

        threading.Thread(
            target=self._dispatch,
            args=(title, body, summary, camera_name, snapshot_b64),
            daemon=True,
        ).start()

    # ── Canaux d'envoi ────────────────────────────────────────────────────────

    def _dispatch(
        self,
        title: str,
        body: str,
        summary: str,
        camera_name: str,
        snapshot_b64: Optional[str],
    ) -> None:
        cfg = self._config
        if cfg.desktop_enabled:
            self._send_desktop(title, summary)
        if cfg.email_enabled:
            self._send_email(title, body)
        if cfg.webhook_enabled:
            self._send_webhook(camera_name, summary, snapshot_b64)

    def _send_desktop(self, title: str, message: str) -> None:
        try:
            from plyer import notification
            notification.notify(
                title=title,
                message=message,
                app_name="Surveillance",
                timeout=8,
            )
        except Exception as exc:
            logger.error("Notification bureau échouée : %s", exc)

    def _send_email(self, subject: str, body: str) -> None:
        cfg = self._config
        if not cfg.smtp_user or not cfg.email_recipients:
            return
        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = cfg.smtp_user
            msg["To"] = ", ".join(cfg.email_recipients)
            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(cfg.smtp_user, cfg.smtp_password)
                server.send_message(msg)
            logger.info("Email alerte envoyé à %s", cfg.email_recipients)
        except Exception as exc:
            logger.error("Envoi email échoué : %s", exc)

    def _send_webhook(
        self,
        camera_name: str,
        summary: str,
        snapshot_b64: Optional[str],
    ) -> None:
        cfg = self._config
        if not cfg.webhook_url:
            return
        try:
            import time
            import requests
            payload: dict = {
                "event": "detection",
                "camera": camera_name,
                "summary": summary,
                "timestamp": time.time(),
            }
            if snapshot_b64:
                payload["snapshot_b64"] = snapshot_b64
            headers = {"Content-Type": "application/json"}
            if cfg.webhook_secret:
                headers["X-Surveillance-Secret"] = cfg.webhook_secret
            resp = requests.post(cfg.webhook_url, json=payload, headers=headers, timeout=5)
            logger.info("Webhook envoyé → %s (%d)", cfg.webhook_url, resp.status_code)
        except Exception as exc:
            logger.error("Webhook échoué : %s", exc)

    # ── Test des canaux ───────────────────────────────────────────────────────

    def test_desktop(self) -> bool:
        try:
            self._send_desktop("Test Surveillance", "Notification de test.")
            return True
        except Exception:
            return False

    def test_email(self) -> str:
        """Retourne '' si OK, sinon le message d'erreur."""
        try:
            self._send_email("Test Surveillance", "Email de test depuis le système de surveillance.")
            return ""
        except Exception as exc:
            return str(exc)

    def test_webhook(self) -> str:
        try:
            self._send_webhook("Test", "Test webhook", None)
            return ""
        except Exception as exc:
            return str(exc)
