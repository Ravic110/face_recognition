"""
auth.py
Authentification de l'API locale.

L'API exposait auparavant les flux caméra, l'historique nominatif et l'arrêt de
la surveillance sur `0.0.0.0:5000`, sans aucun contrôle. Toute requête exige
désormais un en-tête `X-API-Key`.

Origine de la clé, par ordre de priorité :
  1. la variable d'environnement `FR_API_KEY` ;
  2. le fichier `<config_dir>/api_key`, créé en 0600 au premier lancement.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import threading
import time

from ..settings import AppSettings

logger = logging.getLogger(__name__)

KEY_LENGTH = 32


def resolve_api_key(settings: AppSettings) -> str:
    """Clé de l'API : environnement, puis fichier, puis génération."""
    depuis_env = os.environ.get("FR_API_KEY", "").strip()
    if depuis_env:
        return depuis_env

    fichier = settings.api_key_file
    if fichier.exists():
        existante = fichier.read_text(encoding="utf-8").strip()
        if existante:
            return existante

    nouvelle = secrets.token_urlsafe(KEY_LENGTH)
    fichier.parent.mkdir(parents=True, exist_ok=True)
    fichier.write_text(nouvelle, encoding="utf-8")
    os.chmod(fichier, 0o600)
    logger.info("Clé d'API générée dans %s", fichier)
    return nouvelle


class ApiKeyGuard:
    """
    Vérifie la clé et limite les tentatives.

    La comparaison utilise `hmac.compare_digest` : une comparaison naïve fuite
    la longueur du préfixe correct par son temps d'exécution.
    """

    def __init__(
        self,
        api_key: str,
        max_attempts: int = 10,
        window_seconds: float = 300.0,
    ) -> None:
        self._key = api_key
        self._max_attempts = max_attempts
        self._window = window_seconds
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent_failures(self, client_ip: str, now: float) -> list[float]:
        """Échecs de cette adresse encore dans la fenêtre. Purge les plus anciens."""
        recents = [t for t in self._failures.get(client_ip, []) if now - t < self._window]
        if recents:
            self._failures[client_ip] = recents
        else:
            self._failures.pop(client_ip, None)
        return recents

    def is_blocked(self, client_ip: str) -> bool:
        with self._lock:
            return len(self._recent_failures(client_ip, time.monotonic())) >= self._max_attempts

    def reset(self, client_ip: str) -> None:
        with self._lock:
            self._failures.pop(client_ip, None)

    def check(self, provided: str | None, client_ip: str) -> bool:
        """Valide la clé fournie. Une adresse bloquée est refusée sans comparaison."""
        now = time.monotonic()
        with self._lock:
            if len(self._recent_failures(client_ip, now)) >= self._max_attempts:
                logger.warning("Requête refusée : %s a dépassé le quota de tentatives", client_ip)
                return False

            valide = bool(provided) and hmac.compare_digest(provided or "", self._key)
            if valide:
                self._failures.pop(client_ip, None)
                return True

            self._failures.setdefault(client_ip, []).append(now)
            logger.warning("Clé d'API invalide depuis %s", client_ip)
            return False
