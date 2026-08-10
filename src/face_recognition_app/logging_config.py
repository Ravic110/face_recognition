"""
logging_config.py
Configuration unique du logging, appliquée depuis le point d'entrée.

Aucun module ne doit appeler `logging.basicConfig` : deux modules le faisaient à
l'import, avec des destinations contradictoires, et le premier importé gagnait
silencieusement.

Sortie : console (INFO) et fichier tournant `app.log` à la racine du projet.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .settings import AppSettings

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s — %(message)s"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUPS = 3

_configured = False


def configure_logging(settings: AppSettings, verbose: bool = False) -> None:
    """Installe les handlers racine. Idempotent."""
    global _configured
    if _configured:
        return

    niveau = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(niveau)

    fichier = RotatingFileHandler(
        settings.project_root / "app.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUPS,
        encoding="utf-8",
    )
    fichier.setFormatter(formatter)
    fichier.setLevel(logging.DEBUG)

    racine = logging.getLogger()
    racine.setLevel(niveau)
    racine.handlers.clear()
    racine.addHandler(console)
    racine.addHandler(fichier)

    # Flask est bavard sur chaque requête ; on ne garde que ses erreurs.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    _configured = True
