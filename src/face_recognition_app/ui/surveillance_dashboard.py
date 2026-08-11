"""
surveillance_dashboard.py
Coquille de l'application : barre supérieure, bandeau de statut, vues.

L'application tient dans une seule fenêtre. La barre et le bandeau restent
fixes ; les onglets échangent le contenu de la zone centrale au lieu d'ouvrir
des fenêtres. Chaque vue est construite à sa première ouverture puis conservée,
si bien qu'y revenir retrouve ses filtres et sa position de défilement.

Le fichier ne contient plus que le châssis — les vues vivent dans `ui/views/`.
Il pesait 1 300 lignes avant ce découpage.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import messagebox, ttk

from .. import theme
from ..api.server import ApiServer
from ..services.alert_manager import AlertManager
from ..services.camera_manager import CameraManager
from ..services.surveillance_engine import SurveillanceEngine, SurveillanceEvent
from ..services.video_recorder import VideoRecorder
from ..settings import AppSettings
from ..storage.config import PROJECT_ROOT
from ..storage.event_store import EventStore
from ..storage.profile_store import ProfileStore
from .views.alerts_view import AlertsView
from .views.base import AppServices, View
from .views.config_view import ConfigView
from .views.dashboard_view import DashboardView
from .views.faces_view import FacesView
from .views.history_view import HistoryView
from .widgets import Frame, Label

logger = logging.getLogger(__name__)

CAMERAS_FILE = PROJECT_ROOT / "cameras.json"


class SurveillanceDashboard(tk.Toplevel):
    """
    Fenêtre unique du système de surveillance.

    Doit être créée comme Toplevel d'un root ttkbootstrap vivant. Le root est
    retiré (withdraw) puis détruit quand cette fenêtre se ferme.
    """

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.title("Surveillance Intelligente")
        self.geometry("1280x740")
        self.minsize(1024, 640)
        self.configure(bg=theme.BG_BASE)

        settings = AppSettings.create()

        # ── Services, construits une fois et partagés par toutes les vues
        cam_mgr = CameraManager(CAMERAS_FILE)
        event_store = EventStore()
        recorder = VideoRecorder()
        alert_mgr = AlertManager()
        profile_store = ProfileStore()

        engine = SurveillanceEngine(cam_mgr)
        engine.set_recorder(recorder)
        engine.set_alert_manager(alert_mgr)
        engine.apply_profile(profile_store.get_active())

        api = ApiServer(settings, cam_mgr, engine, event_store.repository, recorder)

        self.services = AppServices(
            settings=settings,
            cameras=cam_mgr,
            engine=engine,
            events=event_store.repository,
            recorder=recorder,
            alerts=alert_mgr,
            profiles=profile_store.repository,
            api=api,
        )
        self._event_store = event_store

        self._vues: dict[str, View] = {}
        self._onglets: dict[str, tuple[Label, Frame]] = {}
        self._vue_active: str | None = None

        self._build_shell()
        self._afficher(DashboardView.TITRE)

        engine.add_event_listener(self._on_surveillance_event)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Châssis ───────────────────────────────────────────────────────────────

    def _build_shell(self) -> None:
        entete = Frame(self, bg=theme.BG_SURFACE)
        entete.pack(fill=tk.X)

        # ── Bande 1 : identité et commande principale
        bande_haut = Frame(entete, bg=theme.BG_SURFACE, padx=theme.PAD_L, pady=theme.PAD_M)
        bande_haut.pack(fill=tk.X)

        Label(
            bande_haut,
            text="◉",
            bg=theme.BG_SURFACE,
            fg=theme.BRAND_ACCENT,
            font=(theme.font_sans(), 20),
        ).pack(side=tk.LEFT)
        Label(
            bande_haut,
            text="SURVEILLANCE",
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_PRIMARY,
            font=theme.FONT_TITLE(),
        ).pack(side=tk.LEFT, padx=(theme.PAD_S, theme.PAD_XL))

        self._start_btn = ttk.Button(
            bande_haut,
            text="START",
            command=self._start_surveillance,
            bootstyle="success-outline",
            width=10,
        )
        self._start_btn.pack(side=tk.LEFT)

        self._stop_btn = ttk.Button(
            bande_haut,
            text="STOP",
            command=self._stop_surveillance,
            state=tk.DISABLED,
            bootstyle="secondary-outline",
            width=10,
        )
        self._stop_btn.pack(side=tk.LEFT, padx=theme.PAD_S)

        self._profile_var = tk.StringVar(value=self.services.profiles.active_name)
        profil_cb = ttk.Combobox(
            bande_haut,
            textvariable=self._profile_var,
            values=[p.name for p in self.services.profiles.list_all()],
            width=10,
            state="readonly",
        )
        profil_cb.pack(side=tk.RIGHT)
        profil_cb.bind("<<ComboboxSelected>>", self._on_profile_change)
        Label(
            bande_haut,
            text="PROFIL",
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        ).pack(side=tk.RIGHT, padx=theme.PAD_S)

        Frame(entete, bg=theme.BORDER, height=theme.BORDER_W).pack(fill=tk.X)

        # ── Bande 2 : onglets
        bande_nav = Frame(entete, bg=theme.BG_SURFACE, padx=theme.PAD_L)
        bande_nav.pack(fill=tk.X)

        for titre in (
            DashboardView.TITRE,
            HistoryView.TITRE,
            FacesView.TITRE,
            AlertsView.TITRE,
            ConfigView.TITRE,
        ):
            self._creer_onglet(bande_nav, titre)

        # Import d'images et de vidéos restent des fenêtres : ce sont des tâches
        # ponctuelles, menées en parallèle de la surveillance.
        for libelle, commande in (
            ("VIDÉOS", self._open_video_importer),
            ("IMAGES", self._open_image_importer),
        ):
            lien = Label(
                bande_nav,
                text=libelle,
                bg=theme.BG_SURFACE,
                fg=theme.TEXT_SECONDARY,
                font=theme.FONT_BADGE(),
                cursor="hand2",
                padx=theme.PAD_S,
            )
            lien.pack(side=tk.RIGHT, pady=theme.PAD_S)
            lien.bind("<Button-1>", lambda _e, c=commande: c())

        Frame(entete, bg=theme.BORDER, height=theme.BORDER_W).pack(fill=tk.X)

        # ── Bandeau de statut
        bandeau = Frame(self, bg=theme.BG_BASE, padx=theme.PAD_L, pady=theme.PAD_S)
        bandeau.pack(fill=tk.X)
        Label(
            bandeau,
            text="SYSTEM STATUS",
            bg=theme.BG_BASE,
            fg=theme.BRAND_ACCENT,
            font=theme.FONT_BADGE(),
        ).pack(side=tk.LEFT, padx=(0, theme.PAD_S))
        self._status_var = tk.StringVar(
            value="Prêt. Démarrez la surveillance ou ajoutez une caméra."
        )
        Label(
            bandeau,
            textvariable=self._status_var,
            bg=theme.BG_BASE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_SMALL(),
            anchor=tk.W,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        Frame(self, bg=theme.BORDER, height=theme.BORDER_W).pack(fill=tk.X)

        # ── Zone centrale : une seule vue visible à la fois
        self._conteneur = Frame(self, bg=theme.BG_BASE)
        self._conteneur.pack(fill=tk.BOTH, expand=True)

    def _creer_onglet(self, parent, titre: str) -> None:
        bloc = Frame(parent, bg=theme.BG_SURFACE)
        bloc.pack(side=tk.LEFT, padx=(0, theme.PAD_L))
        etiquette = Label(
            bloc,
            text=titre,
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_NAV(),
            pady=theme.PAD_S,
            cursor="hand2",
        )
        etiquette.pack()
        soulignement = Frame(bloc, bg=theme.BG_SURFACE, height=2)
        soulignement.pack(fill=tk.X)
        etiquette.bind("<Button-1>", lambda _e, t=titre: self._afficher(t))
        self._onglets[titre] = (etiquette, soulignement)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _creer_vue(self, titre: str) -> View:
        if titre == DashboardView.TITRE:
            return DashboardView(self._conteneur, self.services, self._status_var.set)
        if titre == HistoryView.TITRE:
            return HistoryView(self._conteneur, self.services)
        if titre == FacesView.TITRE:
            return FacesView(self._conteneur, self.services)
        if titre == AlertsView.TITRE:
            return AlertsView(self._conteneur, self.services)
        if titre == ConfigView.TITRE:
            return ConfigView(self._conteneur, self.services)
        raise ValueError(f"Vue inconnue : {titre!r}")

    def _afficher(self, titre: str) -> None:
        """Bascule sur une vue, en la construisant si c'est la première fois."""
        if titre == self._vue_active:
            return

        if self._vue_active is not None:
            precedente = self._vues[self._vue_active]
            precedente.on_hide()
            precedente.pack_forget()

        vue = self._vues.get(titre)
        if vue is None:
            vue = self._creer_vue(titre)
            self._vues[titre] = vue
        vue.assurer_construction()
        vue.pack(fill=tk.BOTH, expand=True)
        vue.on_show()

        self._vue_active = titre
        for nom, (etiquette, soulignement) in self._onglets.items():
            actif = nom == titre
            etiquette.configure(fg=theme.BRAND_ACCENT if actif else theme.TEXT_SECONDARY)
            soulignement.configure(bg=theme.BRAND_ACCENT if actif else theme.BG_SURFACE)

    @property
    def _dashboard(self) -> DashboardView | None:
        vue = self._vues.get(DashboardView.TITRE)
        return vue if isinstance(vue, DashboardView) else None

    # ── Surveillance ──────────────────────────────────────────────────────────

    def _start_surveillance(self) -> None:
        if not self.services.cameras.list_configs():
            messagebox.showinfo("Aucune caméra", "Ajoutez au moins une caméra.", parent=self)
            return
        self.services.cameras.start_all()
        self.services.engine.start()
        self._start_btn.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL, bootstyle="danger-outline")
        self._status_var.set("Surveillance active…")
        if self._dashboard:
            self._dashboard.marquer_surveillance(True)

    def _stop_surveillance(self) -> None:
        self.services.engine.stop()
        self.services.cameras.stop_all()
        self._start_btn.configure(state=tk.NORMAL)
        self._stop_btn.configure(state=tk.DISABLED, bootstyle="secondary-outline")
        self._status_var.set("Surveillance arrêtée.")
        if self._dashboard:
            self._dashboard.marquer_surveillance(False)

    def _on_surveillance_event(self, event: SurveillanceEvent) -> None:
        """Reçu depuis un thread d'analyse : persister puis planifier l'affichage."""
        self.services.events.record(
            timestamp=event.timestamp,
            camera_uid=event.camera_uid,
            camera_name=event.camera_name,
            faces=event.faces,
            frame=event.frame,
            save_snapshot=True,
        )
        vue = self._dashboard
        if vue is None:
            return
        vue.encaisser_evenement(event)
        self.after(0, lambda e=event: vue.ajouter_au_journal(e))

    # ── Profils ───────────────────────────────────────────────────────────────

    def _on_profile_change(self, _event=None) -> None:
        nom = self._profile_var.get()
        if self.services.profiles.set_active(nom):
            profil = self.services.profiles.get_active()
            self.services.engine.apply_profile(profil)
            self._status_var.set(f"Profil activé : {profil.label}")

    # ── Fenêtres ponctuelles ──────────────────────────────────────────────────

    def _open_image_importer(self) -> None:
        from .image_importer import ImageImporterApp

        ImageImporterApp(self).focus()

    def _open_video_importer(self) -> None:
        from .video_importer import VideoImporterApp

        fenetre = tk.Toplevel(self)
        VideoImporterApp(fenetre)
        fenetre.focus()

    # ── Fermeture ─────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        for vue in self._vues.values():
            vue.fermer()
        self.services.engine.stop()
        self.services.cameras.stop_all()
        self.services.api.stop()
        self.master.destroy()
