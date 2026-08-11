"""
base.py
Socle des vues du tableau de bord.

L'application tient dans une seule fenêtre : la barre supérieure et le bandeau
de statut restent fixes, seule la zone centrale change. Chaque onglet correspond
à une `View`, construite à la première ouverture puis conservée — revenir sur un
onglet retrouve son état, ses filtres et sa position de défilement.

`on_show` et `on_hide` permettent à une vue de ne travailler que lorsqu'elle est
visible : l'historique se recharge à l'affichage, le tableau de bord suspend son
rafraîchissement quand on le quitte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ... import theme
from ..widgets import Frame

if TYPE_CHECKING:
    from ...api.server import ApiServer
    from ...services.alert_manager import AlertManager
    from ...services.camera_manager import CameraManager
    from ...services.video_recorder import VideoRecorder
    from ...settings import AppSettings
    from ...storage.event_repository import EventRepository
    from ...storage.profile_repository import ProfileRepository


@dataclass
class AppServices:
    """
    Composants partagés, passés à chaque vue.

    Les vues ne construisent aucun service : elles reçoivent ceux de
    l'application, ce qui évite deux repositories concurrents sur le même
    fichier et rend chaque vue testable avec des doublures.
    """

    settings: AppSettings
    cameras: CameraManager
    engine: Any
    events: EventRepository
    recorder: VideoRecorder
    alerts: AlertManager
    profiles: ProfileRepository
    api: ApiServer


class View(Frame):
    """Vue affichée dans la zone centrale de la fenêtre."""

    #: Libellé de l'onglet, en capitales.
    TITRE = ""

    def __init__(self, parent, services: AppServices) -> None:
        super().__init__(parent, bg=theme.BG_BASE)
        self.services = services
        self._construite = False

    def construire(self) -> None:
        """Construit l'interface. Appelée une seule fois, à la première ouverture."""
        raise NotImplementedError

    def assurer_construction(self) -> None:
        if not self._construite:
            self.construire()
            self._construite = True

    def on_show(self) -> None:
        """Appelée à chaque fois que la vue devient visible."""

    def on_hide(self) -> None:
        """Appelée quand on quitte la vue."""

    def fermer(self) -> None:
        """Libère les ressources propres à la vue, à la fermeture de l'application."""


# ── Fabriques partagées ───────────────────────────────────────────────────────


def entete_panneau(parent, titre: str) -> Frame:
    """
    En-tête de panneau : fond plus clair, titre en capitales, filet en bas.

    Retourne le conteneur du titre, pour y ajouter des éléments à droite.
    """
    from ..widgets import Label

    bloc = Frame(parent, bg=theme.BG_SURFACE)
    bloc.pack(fill="x")
    contenu = Frame(bloc, bg=theme.BG_SURFACE, padx=theme.PAD_M, pady=theme.PAD_S)
    contenu.pack(fill="x")
    Label(
        contenu,
        text=titre,
        bg=theme.BG_SURFACE,
        fg=theme.TEXT_PRIMARY,
        font=theme.FONT_HEADING(),
    ).pack(side="left")
    Frame(bloc, bg=theme.BORDER, height=theme.BORDER_W).pack(fill="x")
    return contenu


def filet(parent, horizontal: bool = True):
    """Filet de séparation de 1 px — la profondeur passe par des traits."""
    if horizontal:
        trait = Frame(parent, bg=theme.BORDER, height=theme.BORDER_W)
        trait.pack(fill="x")
    else:
        trait = Frame(parent, bg=theme.BORDER, width=theme.BORDER_W)
        trait.pack(fill="y", side="left")
    return trait
