"""
__main__.py
Point d'entrée de l'application — composition root.

Construit la configuration, installe le logging, instancie les repositories,
applique les migrations de démarrage, puis lance le tableau de bord.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import ttkbootstrap as ttk

from .logging_config import configure_logging
from .settings import AppSettings
from .storage.camera_repository import CameraRepository
from .storage.encodings_repository import EncodingsRepository
from .storage.event_repository import EventRepository
from .storage.profile_repository import ProfileRepository
from .theme import apply_theme

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Composants partagés, construits une fois et passés explicitement."""

    settings: AppSettings
    encodings: EncodingsRepository
    events: EventRepository
    profiles: ProfileRepository
    cameras: CameraRepository


def _verifier_opencv() -> None:
    """
    Signale l'installation simultanée d'opencv-python et opencv-python-headless.

    La variante headless l'emporte à l'import et casse silencieusement toute
    fonction d'affichage OpenCV.
    """
    from importlib.metadata import distributions

    installes = {d.metadata["Name"] for d in distributions() if d.metadata["Name"]}
    if {"opencv-python", "opencv-python-headless"} <= installes:
        logger.warning(
            "opencv-python et opencv-python-headless sont installés ensemble : "
            "l'affichage OpenCV sera cassé. Exécutez « pip uninstall opencv-python-headless »."
        )


def build_context(settings: AppSettings) -> AppContext:
    """Instancie les repositories et applique les migrations de démarrage."""
    settings.ensure_directories()

    profiles = ProfileRepository(settings)
    if profiles.migrate_alert_fields():
        logger.info("Migration des champs d'alerte appliquée")

    events = EventRepository(settings)
    events.start()
    # Aucune purge au démarrage : l'historique de détections est la raison d'être
    # du système. Le nettoyage est une action manuelle, depuis la fenêtre
    # d'historique, précédée d'une confirmation et d'une sauvegarde.

    context = AppContext(
        settings=settings,
        encodings=EncodingsRepository(settings),
        events=events,
        profiles=profiles,
        cameras=CameraRepository(settings),
    )

    # L'adaptateur d'encodages partage le repository de l'application.
    from .storage import encodings_store

    encodings_store.set_repository(context.encodings)

    return context


def main() -> None:
    settings = AppSettings.create()
    configure_logging(settings)
    logger.info("Démarrage — racine projet : %s", settings.project_root)

    _verifier_opencv()
    context = build_context(settings)

    # Un thème sombre standard d'abord : `Style.register_theme` est une méthode
    # d'instance, il faut donc une fenêtre pour lui soumettre la palette maison.
    root = ttk.Window(themename="darkly")
    apply_theme(root)
    root.withdraw()

    from .ui.surveillance_dashboard import SurveillanceDashboard

    SurveillanceDashboard(root)

    try:
        root.mainloop()
    finally:
        context.events.stop()
        logger.info("Arrêt terminé")


if __name__ == "__main__":
    main()
