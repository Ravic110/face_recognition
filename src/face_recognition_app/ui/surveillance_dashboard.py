"""
surveillance_dashboard.py
Tableau de bord principal de surveillance multi-caméras.

Fonctionnalités :
  - Grille de vignettes en temps réel (jusqu'à 6 caméras)
  - Journal des détections (panneau droit)
  - Ajout / modification / suppression de caméras
  - Démarrage / arrêt du moteur de surveillance
  - Accès aux modules d'import (images, vidéos)
  - Enregistrement automatique des événements dans EventStore
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

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
from .camera_config_dialog import CameraConfigDialog
from .widgets import Canvas, Frame, Label, Text

logger = logging.getLogger(__name__)

CAMERAS_FILE = PROJECT_ROOT / "cameras.json"

# Grille : max colonnes et lignes
GRID_COLS = 3
GRID_ROWS = 2
MAX_CAMERAS_VISIBLE = GRID_COLS * GRID_ROWS

# Taille de chaque vignette caméra (px)
THUMB_W = 320
THUMB_H = 240

# Fréquence de rafraîchissement de l'interface (ms)
REFRESH_INTERVAL_MS = 100


class TerminalFeed(tk.Frame):
    """
    Flux des dernières lignes du journal applicatif, en monospace.

    Se branche sur le logger racine via un `logging.Handler` : ce que le
    terminal affiche est exactement ce que le système journalise, sans source
    parallèle à maintenir.

    Les lignes arrivent depuis n'importe quel thread ; elles sont replanifiées
    sur la boucle Tk par `after`, Tkinter n'étant pas thread-safe.
    """

    MAX_LIGNES = 200

    NIVEAUX = {
        "WARNING": theme.STATE_WARN,
        "ERROR": theme.STATE_DANGER,
        "CRITICAL": theme.STATE_DANGER,
    }

    def __init__(self, parent: tk.Widget, hauteur: int = 7) -> None:
        super().__init__(parent, bg=theme.BG_BASE, autostyle=False)

        Frame(self, bg=theme.BORDER, height=theme.BORDER_W).pack(fill=tk.X)

        self._texte = Text(
            self,
            height=hauteur,
            bg=theme.BG_BASE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_MONO(),
            relief=tk.FLAT,
            bd=0,
            padx=theme.PAD_S,
            pady=theme.PAD_XS,
            state=tk.DISABLED,
            wrap=tk.NONE,
            insertbackground=theme.BRAND_ACCENT,
        )
        self._texte.pack(fill=tk.BOTH, expand=True)
        for niveau, couleur in self.NIVEAUX.items():
            self._texte.tag_configure(niveau, foreground=couleur)
        self._texte.tag_configure("INFO", foreground=theme.TEXT_SECONDARY)
        self._texte.tag_configure("prompt", foreground=theme.BRAND_ACCENT)

        self._handler = _TerminalHandler(self)
        logging.getLogger().addHandler(self._handler)

    def ajouter(self, ligne: str, niveau: str = "INFO") -> None:
        """Ajoute une ligne. Sûr depuis n'importe quel thread."""
        try:
            if self.winfo_exists():
                self.after(0, lambda: self._ecrire(ligne, niveau))
        except tk.TclError:
            pass  # fenêtre en cours de destruction

    def _ecrire(self, ligne: str, niveau: str) -> None:
        try:
            self._texte.configure(state=tk.NORMAL)
            self._texte.insert(tk.END, "> ", "prompt")
            self._texte.insert(tk.END, ligne + "\n", niveau)
            surplus = int(self._texte.index("end-1c").split(".")[0]) - self.MAX_LIGNES
            if surplus > 0:
                self._texte.delete("1.0", f"{surplus}.0")
            self._texte.see(tk.END)
            self._texte.configure(state=tk.DISABLED)
        except tk.TclError:
            pass  # widget détruit entre la planification et l'exécution

    def detacher(self) -> None:
        """Retire le handler du logger racine. À appeler à la fermeture."""
        logging.getLogger().removeHandler(self._handler)


class _TerminalHandler(logging.Handler):
    """Redirige les enregistrements du logger racine vers un `TerminalFeed`."""

    def __init__(self, feed: TerminalFeed) -> None:
        super().__init__(level=logging.INFO)
        self._feed = feed

    def emit(self, record: logging.LogRecord) -> None:
        try:
            module = record.name.rsplit(".", 1)[-1]
            self._feed.ajouter(f"{module}: {record.getMessage()}", record.levelname)
        except Exception:  # noqa: BLE001 — un handler ne doit jamais lever
            self.handleError(record)


_JAMAIS_PEINT = object()


class CameraTile(tk.Frame):
    """
    Vignette affichant le flux d'une caméra, au format des maquettes.

    Structure :
      - en-tête : « CAM 01: NOM » à gauche, cadence et pastille d'état à droite.
        L'en-tête se teinte de rouge quand la source est hors ligne — l'anomalie
        se repère alors sans lire le texte.
      - corps   : dernière frame annotée, ou un état vide explicite.
      - superposition : nombre de visages détectés, en bas à gauche du flux.

    Le badge exploite la machine à états de `CameraSource` et affiche le délai
    avant la prochaine tentative, au lieu du simple point rouge d'avant.
    """

    def __init__(
        self,
        parent: tk.Widget,
        uid: str,
        name: str,
        index: int = 1,
        on_fullscreen=None,
    ) -> None:
        super().__init__(
            parent, bg=theme.BG_CARD, highlightthickness=theme.BORDER_W, autostyle=False
        )
        self.configure(highlightbackground=theme.BORDER, highlightcolor=theme.BORDER)
        self.uid = uid
        self._on_fullscreen = on_fullscreen
        self._latest_frame = None
        # Sentinelle : `None` est un état légitime (source absente), il ne peut
        # donc pas servir de valeur initiale au garde de `set_status`.
        self._etat: object = _JAMAIS_PEINT

        # ── En-tête
        self._entete = Frame(self, bg=theme.BG_SURFACE)
        self._entete.pack(fill=tk.X)
        interieur = Frame(self._entete, bg=theme.BG_SURFACE, padx=theme.PAD_S, pady=theme.PAD_XS)
        interieur.pack(fill=tk.X)
        self._entete_interieur = interieur

        self._titre = Label(
            interieur,
            text=f"CAM {index:02d}: {name.upper()}",
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_PRIMARY,
            font=theme.FONT_BADGE(),
        )
        self._titre.pack(side=tk.LEFT)

        self._badge_dot = Label(
            interieur,
            text="●",
            bg=theme.BG_SURFACE,
            fg=theme.STATE_DANGER,
            font=theme.FONT_SMALL(),
        )
        self._badge_dot.pack(side=tk.RIGHT)
        self._badge_label = Label(
            interieur,
            text="HORS LIGNE",
            bg=theme.BG_SURFACE,
            fg=theme.STATE_DANGER,
            font=theme.FONT_BADGE(),
        )
        self._badge_label.pack(side=tk.RIGHT, padx=theme.PAD_XS)

        Frame(self, bg=theme.BORDER, height=theme.BORDER_W).pack(fill=tk.X)

        # ── Corps : le flux, coins francs pour ne perdre aucun pixel
        self._canvas = Canvas(
            self, width=THUMB_W, height=THUMB_H, bg=theme.BG_BASE, highlightthickness=0
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Double-Button-1>", self._handle_dblclick)

        self._photo_ref: ImageTk.PhotoImage | None = None
        self._last_annotated: object | None = None
        self._detections = ""

        self._draw_placeholder()

    def _handle_dblclick(self, _event) -> None:
        if self._on_fullscreen:
            self._on_fullscreen(self.uid)

    # ── Mise à jour ───────────────────────────────────────────────────────────

    def set_status(self, state, retry_in: float = 0.0) -> None:
        """Met à jour le badge et la teinte de l'en-tête depuis l'état de la source."""
        if state is self._etat:
            return
        self._etat = state
        libelle, couleur = theme.connection_badge(state, retry_in)
        hors_ligne = couleur == theme.STATE_DANGER
        fond = theme.HEADER_ALERT if hors_ligne else theme.BG_SURFACE

        self._badge_dot.configure(fg=couleur, bg=fond)
        self._badge_label.configure(text=libelle, fg=couleur, bg=fond)
        self._titre.configure(bg=fond)
        self._entete.configure(bg=fond)
        self._entete_interieur.configure(bg=fond)

    def update_frame(self, frame_bgr, detections: str = "", fps: str = "") -> None:
        """Appelé périodiquement depuis le thread Tkinter."""
        self._detections = detections

        if frame_bgr is None:
            return
        self._latest_frame = frame_bgr

        h, w = frame_bgr.shape[:2]
        cw = self._canvas.winfo_width() or THUMB_W
        ch = self._canvas.winfo_height() or THUMB_H
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame_bgr, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._photo_ref = photo

        self._canvas.delete("all")
        self._canvas.create_image((cw - nw) // 2, (ch - nh) // 2, anchor=tk.NW, image=photo)

        # Superpositions dans le style « HUD » des maquettes
        if fps:
            self._canvas.create_text(
                cw - theme.PAD_S,
                theme.PAD_S,
                text=fps,
                anchor=tk.NE,
                fill=theme.BRAND_ACCENT,
                font=theme.FONT_BADGE(),
            )
        if detections:
            self._canvas.create_text(
                theme.PAD_S,
                ch - theme.PAD_S,
                text=detections,
                anchor=tk.SW,
                fill=theme.TEXT_PRIMARY,
                font=theme.FONT_BADGE(),
            )

    def set_annotated(self, frame_bgr) -> None:
        """Reçoit la frame annotée depuis le moteur de surveillance."""
        self._last_annotated = frame_bgr

    def pop_annotated(self):
        """Consomme et retourne la frame annotée (ou None)."""
        f = self._last_annotated
        self._last_annotated = None
        return f

    def _draw_placeholder(self) -> None:
        self._canvas.delete("all")
        cw = self._canvas.winfo_width() or THUMB_W
        ch = self._canvas.winfo_height() or THUMB_H
        self._canvas.create_text(
            cw // 2,
            ch // 2,
            text="PAS DE SIGNAL",
            fill=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        )


# ── Tableau de bord ────────────────────────────────────────────────────────────


class SurveillanceDashboard(tk.Toplevel):
    """
    Fenêtre principale du système de surveillance.

    Doit être créé comme Toplevel d'un root ttkbootstrap vivant.
    Le root est retiré (withdraw) puis détruit quand cette fenêtre se ferme.
    """

    MAX_EVENEMENTS = 100

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.title("Surveillance Intelligente")
        self.geometry("1280x740")
        self.minsize(900, 600)
        self.configure(bg=theme.BG_BASE)

        # Services
        self._cam_mgr = CameraManager(CAMERAS_FILE)
        self._event_store = EventStore()
        self._recorder = VideoRecorder()
        self._alert_mgr = AlertManager()
        self._profile_store = ProfileStore()

        self._engine = SurveillanceEngine(self._cam_mgr)
        self._engine.set_recorder(self._recorder)
        self._engine.set_alert_manager(self._alert_mgr)
        self._engine.apply_profile(self._profile_store.get_active())

        self._api_server = ApiServer(
            AppSettings.create(),
            self._cam_mgr,
            self._engine,
            self._event_store.repository,
            self._recorder,
        )

        # Tiles par uid caméra
        self._tiles: dict[str, CameraTile] = {}
        self._annotated_lock = threading.Lock()
        self._annotated: dict[str, object] = {}  # uid → np.ndarray

        # Journal : les événements sont conservés pour pouvoir refiltrer
        self._evenements: list[SurveillanceEvent] = []
        self._event_photo_refs: list[ImageTk.PhotoImage] = []

        self._build_ui()
        self._rebuild_tiles()
        self._cam_mgr.on_change(self._on_cameras_changed)
        self._engine.add_event_listener(self._on_surveillance_event)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_refresh()

    # ── Interface ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Barre supérieure ───────────────────────────────────────────────────
        # Calquée sur les maquettes : deux bandes séparées par un filet de 1 px.
        # Le vert est l'accent de marque — démarrage, onglet actif, pastille API.
        # La profondeur passe par des traits, jamais par des ombres.
        entete = Frame(self, bg=theme.BG_SURFACE)
        entete.pack(fill=tk.X)

        # ── Bande 1 : identité et commande principale ──────────────────────────
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

        self._profile_var = tk.StringVar(value=self._profile_store.active_name)
        self._profile_cb = ttk.Combobox(
            bande_haut,
            textvariable=self._profile_var,
            values=[p.name for p in self._profile_store.list_profiles()],
            width=10,
            state="readonly",
        )
        self._profile_cb.pack(side=tk.RIGHT)
        self._profile_cb.bind("<<ComboboxSelected>>", self._on_profile_change)
        Label(
            bande_haut,
            text="PROFIL",
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        ).pack(side=tk.RIGHT, padx=theme.PAD_S)

        Frame(entete, bg=theme.BORDER, height=theme.BORDER_W).pack(fill=tk.X)

        # ── Bande 2 : navigation en onglets ────────────────────────────────────
        bande_nav = Frame(entete, bg=theme.BG_SURFACE, padx=theme.PAD_L)
        bande_nav.pack(fill=tk.X)

        # « Tableau de bord » est la vue courante ; les autres ouvrent leur
        # fenêtre. De vrais onglets dans une seule fenêtre demanderaient la
        # restructuration prévue en Phase 3.
        self._onglets: dict[str, tk.Frame] = {}
        for libelle, commande in (
            ("TABLEAU DE BORD", None),
            ("HISTORIQUE", self._open_event_browser),
            ("BASE DE VISAGES", self._open_encodings_manager),
            ("ALERTES", self._open_alerts_config),
            ("CONFIGURATION", self._add_camera),
        ):
            self._onglets[libelle] = self._creer_onglet(bande_nav, libelle, commande)

        # Indicateur d'API, à droite de la barre de navigation
        cadre_api = Frame(bande_nav, bg=theme.BG_SURFACE)
        cadre_api.pack(side=tk.RIGHT, pady=theme.PAD_S)
        self._api_dot = Label(
            cadre_api,
            text="●",
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_SMALL(),
        )
        self._api_dot.pack(side=tk.LEFT, padx=(0, theme.PAD_XS))
        self._api_var = tk.StringVar(value="API HORS LIGNE")
        self._api_btn = Label(
            cadre_api,
            textvariable=self._api_var,
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
            cursor="hand2",
        )
        self._api_btn.pack(side=tk.LEFT)
        self._api_btn.bind("<Button-1>", lambda _e: self._toggle_api())

        Frame(entete, bg=theme.BORDER, height=theme.BORDER_W).pack(fill=tk.X)

        # ── Bandeau de statut ──────────────────────────────────────────────────
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

        # ── Corps ─────────────────────────────────────────────────────────────
        body = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            bg=theme.BORDER,
            sashwidth=theme.BORDER_W,
            sashrelief=tk.FLAT,
            bd=0,
        )
        body.pack(fill=tk.BOTH, expand=True)

        # ── Panneau gauche : flux vidéo ────────────────────────────────────────
        left_pane = Frame(body, bg=theme.BG_BASE)
        body.add(left_pane, minsize=640)

        entete_flux = self._entete_panneau(left_pane, "FLUX VIDÉO EN DIRECT")
        self._grille_var = tk.StringVar(value="GRILLE 3×2")
        Label(
            entete_flux,
            textvariable=self._grille_var,
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        ).pack(side=tk.RIGHT)

        # ── Liste des caméras configurées, ancrée en bas
        cam_list_lf = Frame(left_pane, bg=theme.BG_BASE)
        cam_list_lf.pack(side=tk.BOTTOM, fill=tk.X)
        Frame(cam_list_lf, bg=theme.BORDER, height=theme.BORDER_W).pack(fill=tk.X)

        corps_liste = Frame(cam_list_lf, bg=theme.BG_BASE, padx=theme.PAD_S, pady=theme.PAD_S)
        corps_liste.pack(fill=tk.X)

        self._cam_list = ttk.Treeview(
            corps_liste, columns=("type", "source", "etat"), show="headings", height=4
        )
        self._cam_list.heading("type", text="TYPE")
        self._cam_list.heading("source", text="SOURCE")
        self._cam_list.heading("etat", text="ÉTAT")
        self._cam_list.column("type", width=80, anchor=tk.CENTER)
        self._cam_list.column("source", width=220)
        self._cam_list.column("etat", width=110, anchor=tk.CENTER)
        self._cam_list.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._cam_list.tag_configure("actif", foreground=theme.BRAND_ACCENT)
        self._cam_list.tag_configure("attente", foreground=theme.STATE_WARN)
        self._cam_list.tag_configure("inactif", foreground=theme.TEXT_SECONDARY)

        cam_btns = Frame(corps_liste, bg=theme.BG_BASE)
        cam_btns.pack(side=tk.RIGHT, fill=tk.Y, padx=theme.PAD_S)
        for libelle, commande in (
            ("＋ CAMÉRA", self._add_camera),
            ("MODIFIER", self._edit_camera),
            ("SUPPRIMER", self._remove_camera),
            ("TESTER", self._test_camera),
        ):
            ttk.Button(
                cam_btns, text=libelle, command=commande, bootstyle="secondary-outline", width=12
            ).pack(side=tk.LEFT, padx=theme.PAD_XS)

        # Grille des vignettes, dans l'espace restant
        self._grid_frame = Frame(left_pane, bg=theme.BG_BASE, padx=theme.PAD_XS)
        self._grid_frame.pack(fill=tk.BOTH, expand=True)

        # ── Panneau droit : journal des événements ─────────────────────────────
        right_pane = Frame(body, bg=theme.BG_BASE)
        body.add(right_pane, minsize=340)

        entete_journal = self._entete_panneau(right_pane, "JOURNAL DES ÉVÉNEMENTS")
        self._live_badge = Label(
            entete_journal,
            text=" LIVE ",
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        )
        self._live_badge.pack(side=tk.RIGHT)

        # Chips de filtre
        barre_filtres = Frame(right_pane, bg=theme.BG_BASE, padx=theme.PAD_S, pady=theme.PAD_S)
        barre_filtres.pack(fill=tk.X)
        self._filtre = tk.StringVar(value="TOUT")
        self._chips: dict[str, tk.Label] = {}
        for libelle in ("TOUT", "ALERTES", "RECONNUS"):
            chip = Label(
                barre_filtres,
                text=f" {libelle} ",
                bg=theme.BG_CARD,
                fg=theme.TEXT_SECONDARY,
                font=theme.FONT_BADGE(),
                padx=theme.PAD_S,
                pady=theme.PAD_XS,
                cursor="hand2",
            )
            chip.pack(side=tk.LEFT, padx=(0, theme.PAD_XS))
            chip.bind("<Button-1>", lambda _e, nom=libelle: self._changer_filtre(nom))
            self._chips[libelle] = chip
        self._event_count_var = tk.StringVar(value="")
        Label(
            barre_filtres,
            textvariable=self._event_count_var,
            bg=theme.BG_BASE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        ).pack(side=tk.RIGHT)
        self._changer_filtre("TOUT")

        # Journal défilant
        cadre_journal = Frame(right_pane, bg=theme.BG_BASE)
        cadre_journal.pack(fill=tk.BOTH, expand=True)
        self._event_canvas = Canvas(cadre_journal, bg=theme.BG_BASE, highlightthickness=0)
        event_scroll = ttk.Scrollbar(
            cadre_journal, orient=tk.VERTICAL, command=self._event_canvas.yview
        )
        self._event_canvas.configure(yscrollcommand=event_scroll.set)
        event_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._event_canvas.pack(fill=tk.BOTH, expand=True)

        self._event_inner = Frame(self._event_canvas, bg=theme.BG_BASE)
        self._event_canvas.create_window((0, 0), window=self._event_inner, anchor=tk.NW)
        self._event_inner.bind(
            "<Configure>",
            lambda e: self._event_canvas.configure(scrollregion=self._event_canvas.bbox("all")),
        )

        # ── Terminal : les dernières lignes du journal applicatif
        self._terminal = TerminalFeed(right_pane, hauteur=7)
        self._terminal.pack(side=tk.BOTTOM, fill=tk.X)

        self._event_count = 0

    # ── Fabriques de composants ───────────────────────────────────────────────

    def _entete_panneau(self, parent: tk.Widget, titre: str) -> tk.Frame:
        """En-tête de panneau : fond plus clair, titre en capitales, filet en bas."""
        bloc = Frame(parent, bg=theme.BG_SURFACE)
        bloc.pack(fill=tk.X)
        contenu = Frame(bloc, bg=theme.BG_SURFACE, padx=theme.PAD_M, pady=theme.PAD_S)
        contenu.pack(fill=tk.X)
        Label(
            contenu,
            text=titre,
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_PRIMARY,
            font=theme.FONT_HEADING(),
        ).pack(side=tk.LEFT)
        Frame(bloc, bg=theme.BORDER, height=theme.BORDER_W).pack(fill=tk.X)
        return contenu

    def _creer_onglet(self, parent: tk.Widget, libelle: str, commande) -> tk.Frame:
        """Onglet de navigation : libellé en capitales, soulignement si actif."""
        bloc = Frame(parent, bg=theme.BG_SURFACE)
        bloc.pack(side=tk.LEFT, padx=(0, theme.PAD_L))
        etiquette = Label(
            bloc,
            text=libelle,
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_NAV(),
            pady=theme.PAD_S,
            cursor="hand2" if commande else "",
        )
        etiquette.pack()
        soulignement = Frame(bloc, bg=theme.BG_SURFACE, height=2)
        soulignement.pack(fill=tk.X)
        if commande is not None:
            etiquette.bind("<Button-1>", lambda _e: commande())
        else:
            # Vue courante
            etiquette.configure(fg=theme.BRAND_ACCENT)
            soulignement.configure(bg=theme.BRAND_ACCENT)
        return bloc

    def _changer_filtre(self, nom: str) -> None:
        """Active un chip de filtre du journal."""
        self._filtre.set(nom)
        for libelle, chip in self._chips.items():
            actif = libelle == nom
            chip.configure(
                bg=theme.BRAND_ACCENT if actif else theme.BG_CARD,
                fg=theme.BG_BASE if actif else theme.TEXT_SECONDARY,
            )
        if hasattr(self, "_event_inner"):
            self._redessiner_journal()

    # ── Gestion des tuiles (grille) ────────────────────────────────────────────

    def _rebuild_tiles(self) -> None:
        """Reconstruit la grille de vignettes à partir des caméras configurées."""
        # Nettoyer
        for w in self._grid_frame.winfo_children():
            w.destroy()
        self._tiles.clear()

        configs = self._cam_mgr.list_configs()
        if not configs:
            Label(
                self._grid_frame,
                text="AUCUNE CAMÉRA CONFIGURÉE\n\nAjoutez-en une avec « ＋ CAMÉRA »",
                bg=theme.BG_BASE,
                fg=theme.TEXT_SECONDARY,
                font=theme.FONT_BADGE(),
                justify=tk.CENTER,
            ).pack(expand=True)
            return

        visible = configs[:MAX_CAMERAS_VISIBLE]
        for i, cfg in enumerate(visible):
            row, col = divmod(i, GRID_COLS)
            tile = CameraTile(
                self._grid_frame,
                cfg.uid,
                cfg.name,
                index=i + 1,
                on_fullscreen=self._open_fullscreen,
            )
            tile.grid(row=row, column=col, padx=theme.PAD_XS, pady=theme.PAD_XS, sticky=tk.NSEW)
            self._tiles[cfg.uid] = tile

        # Poids de grille égaux
        for c in range(GRID_COLS):
            self._grid_frame.columnconfigure(c, weight=1)
        for r in range(GRID_ROWS):
            self._grid_frame.rowconfigure(r, weight=1)

        self._grille_var.set(f"{len(visible)} / {MAX_CAMERAS_VISIBLE} FLUX")
        self._refresh_cam_list()

    def _refresh_cam_list(self) -> None:
        for row in self._cam_list.get_children():
            self._cam_list.delete(row)
        for cfg in self._cam_mgr.list_configs():
            type_label = "WEBCAM" if cfg.source_type == "webcam" else "IP"
            if self._cam_mgr.is_running(cfg.uid):
                etat, tag = "ACTIF", "actif"
            elif not cfg.enabled:
                etat, tag = "DÉSACTIVÉ", "inactif"
            else:
                etat, tag = "ARRÊTÉ", "attente"
            self._cam_list.insert(
                "", tk.END, iid=cfg.uid, values=(type_label, str(cfg.source), etat), tags=(tag,)
            )

    # ── Callbacks caméras ─────────────────────────────────────────────────────

    def _on_cameras_changed(self) -> None:
        self.after(0, self._rebuild_tiles)

    def _add_camera(self) -> None:
        dlg = CameraConfigDialog(self, title="Ajouter une caméra")
        self.wait_window(dlg)
        if dlg.result:
            self._cam_mgr.add_camera(dlg.result)
            if self._engine.is_running:
                self._cam_mgr.start_camera(dlg.result.uid)

    def _edit_camera(self) -> None:
        sel = self._cam_list.selection()
        if not sel:
            messagebox.showinfo("Sélection", "Sélectionnez une caméra à modifier.", parent=self)
            return
        uid = sel[0]
        config = self._cam_mgr.get_config(uid)
        if not config:
            return
        dlg = CameraConfigDialog(self, title="Modifier la caméra", config=config)
        self.wait_window(dlg)
        if dlg.result:
            self._cam_mgr.update_camera(dlg.result)

    def _remove_camera(self) -> None:
        sel = self._cam_list.selection()
        if not sel:
            messagebox.showinfo("Sélection", "Sélectionnez une caméra à supprimer.", parent=self)
            return
        uid = sel[0]
        config = self._cam_mgr.get_config(uid)
        name = config.name if config else uid
        if messagebox.askyesno("Confirmation", f"Supprimer la caméra « {name} » ?", parent=self):
            self._cam_mgr.remove_camera(uid)

    def _test_camera(self) -> None:
        """Tente d'ouvrir brièvement la caméra sélectionnée et affiche le résultat."""
        sel = self._cam_list.selection()
        if not sel:
            messagebox.showinfo("Sélection", "Sélectionnez une caméra à tester.", parent=self)
            return
        uid = sel[0]
        config = self._cam_mgr.get_config(uid)
        if not config:
            return

        import cv2 as _cv2

        cap = _cv2.VideoCapture(config.source)
        ok = cap.isOpened()
        cap.release()
        if ok:
            messagebox.showinfo("Test", f"✓ Connexion réussie à « {config.name} ».", parent=self)
        else:
            messagebox.showerror(
                "Test",
                f"✗ Impossible de se connecter à « {config.name} ».\nSource : {config.source}",
                parent=self,
            )

    # ── Surveillance ──────────────────────────────────────────────────────────

    def _start_surveillance(self) -> None:
        if not self._cam_mgr.list_configs():
            messagebox.showinfo("Aucune caméra", "Ajoutez au moins une caméra.", parent=self)
            return
        self._cam_mgr.start_all()
        self._engine.start()
        self._start_btn.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL)
        self._live_badge.configure(bg=theme.STATE_DANGER, fg=theme.TEXT_PRIMARY)
        self._status_var.set("Surveillance active…")
        self._refresh_cam_list()

    def _stop_surveillance(self) -> None:
        self._engine.stop()
        self._cam_mgr.stop_all()
        self._start_btn.configure(state=tk.NORMAL)
        self._stop_btn.configure(state=tk.DISABLED)
        self._live_badge.configure(bg=theme.BG_SURFACE, fg=theme.TEXT_SECONDARY)
        self._status_var.set("Surveillance arrêtée.")
        self._refresh_cam_list()

    # ── Callback moteur de surveillance ──────────────────────────────────────

    def _on_surveillance_event(self, event: SurveillanceEvent) -> None:
        """Reçu depuis un thread d'analyse — on stocke et planifie la mise à jour UI."""
        # Enregistrer l'événement
        self._event_store.record(
            timestamp=event.timestamp,
            camera_uid=event.camera_uid,
            camera_name=event.camera_name,
            faces=event.faces,
            frame=event.frame,
            save_snapshot=True,
        )

        # Stocker la frame annotée pour la tuile
        if event.frame is not None:
            with self._annotated_lock:
                self._annotated[event.camera_uid] = event.frame

        # Planifier l'ajout dans le journal UI (thread-safe via after)
        self.after(0, lambda e=event: self._add_event_to_log(e))

    def _add_event_to_log(self, event: SurveillanceEvent) -> None:
        """
        Insère une carte dans le journal.

        Deux types, comme dans les maquettes : « ALERTE INTRUSION » dès qu'un
        visage inconnu apparaît, « VISAGE RECONNU » sinon. Le type est porté par
        un bandeau coloré en tête de carte, lisible avant même le texte.
        """
        self._event_count += 1
        self._evenements.append(event)
        if len(self._evenements) > self.MAX_EVENEMENTS:
            self._evenements.pop(0)
        self._event_count_var.set(f"{self._event_count} DÉTECTIONS")

        ts = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        self._status_var.set(
            f"Dernière détection : {ts} — {event.camera_name} — "
            f"{', '.join(event.known_names) or 'inconnu'}"
        )
        self._redessiner_journal()

    def _correspond_au_filtre(self, event: SurveillanceEvent) -> bool:
        filtre = self._filtre.get()
        if filtre == "ALERTES":
            return event.has_unknown
        if filtre == "RECONNUS":
            return bool(event.known_names) and not event.has_unknown
        return True

    def _redessiner_journal(self) -> None:
        """Reconstruit la liste selon le filtre actif."""
        for w in self._event_inner.winfo_children():
            w.destroy()
        self._event_photo_refs.clear()

        visibles = [e for e in self._evenements if self._correspond_au_filtre(e)]
        if not visibles:
            Label(
                self._event_inner,
                text="AUCUN ÉVÉNEMENT",
                bg=theme.BG_BASE,
                fg=theme.TEXT_SECONDARY,
                font=theme.FONT_BADGE(),
                pady=theme.PAD_XL,
            ).pack(fill=tk.X)
            return

        for event in reversed(visibles):
            self._creer_carte_evenement(event)

        self._event_canvas.update_idletasks()
        self._event_canvas.yview_moveto(0.0)

    def _creer_carte_evenement(self, event: SurveillanceEvent) -> None:
        alerte = event.has_unknown
        accent = theme.STATE_DANGER if alerte else theme.BRAND_ACCENT
        titre = "ALERTE INTRUSION" if alerte else "VISAGE RECONNU"
        icone = "⚠" if alerte else "◉"
        ts = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")

        carte = Frame(self._event_inner, bg=theme.BG_CARD, highlightthickness=theme.BORDER_W)
        carte.configure(highlightbackground=accent, highlightcolor=accent)
        carte.pack(fill=tk.X, padx=theme.PAD_S, pady=theme.PAD_XS)

        # Bandeau de type
        bandeau = Frame(carte, bg=theme.BG_SURFACE, padx=theme.PAD_S, pady=theme.PAD_XS)
        bandeau.pack(fill=tk.X)
        Label(
            bandeau,
            text=f"{icone} {titre}",
            bg=theme.BG_SURFACE,
            fg=accent,
            font=theme.FONT_BADGE(),
        ).pack(side=tk.LEFT)
        Label(
            bandeau,
            text=ts,
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        ).pack(side=tk.RIGHT)

        corps = Frame(carte, bg=theme.BG_CARD, padx=theme.PAD_S, pady=theme.PAD_S)
        corps.pack(fill=tk.X)

        if event.frame is not None:
            try:
                thumb = cv2.resize(event.frame, (64, 48))
                rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self._event_photo_refs.append(photo)
                Label(corps, image=photo, bg=theme.BG_CARD).pack(
                    side=tk.LEFT, padx=(0, theme.PAD_S)
                )
            except Exception as exc:
                logger.warning("Miniature du journal non générée : %s", exc)

        info = Frame(corps, bg=theme.BG_CARD)
        info.pack(side=tk.LEFT, fill=tk.X, expand=True)

        sujet = ", ".join(event.known_names) if event.known_names else "Inconnu"
        Label(
            info,
            text=sujet,
            bg=theme.BG_CARD,
            fg=theme.TEXT_PRIMARY,
            font=theme.FONT_BODY(),
            anchor=tk.W,
        ).pack(anchor=tk.W)

        detail = event.camera_name.upper()
        if event.unknown_count:
            detail += f" · {event.unknown_count} INCONNU(S)"
        Label(
            info,
            text=detail,
            bg=theme.BG_CARD,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
            anchor=tk.W,
        ).pack(anchor=tk.W)

    # ── Rafraîchissement périodique de la grille ──────────────────────────────

    def _schedule_refresh(self) -> None:
        self._refresh_tiles()
        self.after(REFRESH_INTERVAL_MS, self._schedule_refresh)

    def _refresh_tiles(self) -> None:
        """Met à jour chaque tuile avec la dernière frame disponible."""
        with self._annotated_lock:
            annotated_copy = dict(self._annotated)
            self._annotated.clear()

        for uid, tile in self._tiles.items():
            # État réel de la source : le badge affiche le délai de reconnexion
            # plutôt qu'un simple point rouge.
            source = self._cam_mgr.get_source(uid)
            if source is None:
                tile.set_status(None)
            else:
                tile.set_status(source.state, source.next_retry_in)

            fps_str = ""
            stats = self._engine.get_stats(uid)
            if stats and stats.fps > 0:
                fps_str = f"{stats.fps:.1f} fps"

            annotated = annotated_copy.get(uid)
            frame = annotated if annotated is not None else self._cam_mgr.get_frame(uid)
            tile.update_frame(frame, fps=fps_str)

        if self._engine.is_running:
            self._refresh_cam_list()

    # ── Plein écran ───────────────────────────────────────────────────────────

    def _open_fullscreen(self, uid: str) -> None:
        """Ouvre une caméra en plein écran (double-clic sur la vignette)."""
        config = self._cam_mgr.get_config(uid)
        name = config.name if config else uid
        win = tk.Toplevel(self)
        win.title(f"Plein écran — {name}")
        win.configure(bg=theme.BG_BASE)
        win.state("zoomed")

        canvas = Canvas(win, bg=theme.BG_BASE, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.bind("<Escape>", lambda _: win.destroy())
        canvas.bind("<Double-Button-1>", lambda _: win.destroy())

        photo_ref = [None]
        running = [True]

        def _update():
            if not running[0]:
                return
            tile = self._tiles.get(uid)
            frame = None
            if tile and tile._latest_frame is not None:
                frame = tile._latest_frame
            else:
                frame = self._cam_mgr.get_frame(uid)
            if frame is not None:
                cw, ch = canvas.winfo_width() or 800, canvas.winfo_height() or 600
                h, w = frame.shape[:2]
                scale = min(cw / w, ch / h)
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (nw, nh))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                photo = ImageTk.PhotoImage(pil)
                photo_ref[0] = photo
                canvas.delete("all")
                canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)
            if running[0]:
                win.after(66, _update)

        def _on_close():
            running[0] = False
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)
        _update()

        Label(
            win,
            text="Appuyez sur Échap ou double-cliquez pour fermer",
            bg=theme.BG_BASE,
            fg=theme.TEXT_SECONDARY,
            font=("Helvetica", 9),
        ).pack(side=tk.BOTTOM, pady=4)

    # ── Import ────────────────────────────────────────────────────────────────

    def _open_image_importer(self) -> None:
        from .image_importer import ImageImporterApp

        win = ImageImporterApp(self)
        win.focus()

    def _open_video_importer(self) -> None:
        from .video_importer import VideoImporterApp

        win = VideoImporterApp(self)
        win.focus()

    def _open_event_browser(self) -> None:
        from .event_browser import EventBrowserApp

        win = EventBrowserApp(self, self._event_store)
        win.focus()

    def _open_encodings_manager(self) -> None:
        """Ouvre le gestionnaire d'encodages."""
        from .interface import FaceRecognitionApp

        win = tk.Toplevel(self)
        app = FaceRecognitionApp(win)
        app.open_manage_window()

    # ── Profils ───────────────────────────────────────────────────────────────

    def _on_profile_change(self, _event=None) -> None:
        name = self._profile_var.get()
        if self._profile_store.set_active(name):
            profile = self._profile_store.get_active()
            self._engine.apply_profile(profile)
            self._status_var.set(f"Profil activé : {profile.label}")

    # ── API REST ──────────────────────────────────────────────────────────────

    def _toggle_api(self) -> None:
        if self._api_server.is_running:
            self._api_server.stop()
            self._api_var.set("API HORS LIGNE")
            self._api_dot.configure(fg=theme.TEXT_SECONDARY)
            return

        if not self._api_server.start():
            messagebox.showerror(
                "API",
                f"Impossible d'écouter sur {self._api_server.url}.\n"
                "Le port est peut-être déjà utilisé.",
                parent=self,
            )
            return

        self._api_var.set(f"API :{self._api_server.port}")
        self._api_dot.configure(fg=theme.BRAND_ACCENT)
        self._afficher_cle_api()

    def _afficher_cle_api(self) -> None:
        """Montre l'URL et la clé, avec un bouton de copie."""
        fenetre = tk.Toplevel(self)
        fenetre.title("Accès à l'API")
        fenetre.resizable(False, False)
        fenetre.transient(self)

        Label(
            fenetre,
            text="Envoyez cette clé dans l'en-tête X-API-Key :",
            font=("Helvetica", 10),
        ).pack(padx=16, pady=(14, 6))

        cle = self._api_server.api_key
        champ = ttk.Entry(fenetre, width=52, justify=tk.CENTER)
        champ.insert(0, cle)
        champ.configure(state="readonly")
        champ.pack(padx=16)

        Label(
            fenetre,
            text=f"{self._api_server.url}/api/status",
            fg=theme.TEXT_SECONDARY,
            font=("Helvetica", 9),
        ).pack(pady=(6, 0))

        def copier() -> None:
            self.clipboard_clear()
            self.clipboard_append(cle)

        barre = Frame(fenetre)
        barre.pack(pady=12)
        ttk.Button(barre, text="Copier la clé", command=copier).pack(side=tk.LEFT, padx=4)
        ttk.Button(barre, text="Fermer", command=fenetre.destroy).pack(side=tk.LEFT, padx=4)

    # ── Alertes ───────────────────────────────────────────────────────────────

    def _open_alerts_config(self) -> None:
        _AlertConfigDialog(self, self._alert_mgr)

    # ── Fermeture ─────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._terminal.detacher()
        self._engine.stop()
        self._cam_mgr.stop_all()
        self._api_server.stop()
        # Détruire le root (ttkbootstrap Window) pour terminer l'application
        self.master.destroy()


# ── Dialogue configuration des alertes ───────────────────────────────────────


class _AlertConfigDialog(tk.Toplevel):
    """Fenêtre de configuration des canaux d'alerte."""

    def __init__(self, parent: tk.Widget, alert_mgr: AlertManager) -> None:
        super().__init__(parent)
        self.title("Configuration des alertes")
        self.resizable(False, False)
        self.grab_set()
        self._mgr = alert_mgr
        cfg = alert_mgr.config

        pad = {"padx": 10, "pady": 5}
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ── Onglet Bureau ──────────────────────────────────────────────────────
        tab_desk = Frame(nb)
        nb.add(tab_desk, text="Bureau")
        self._desk_en = tk.BooleanVar(value=cfg.desktop_enabled)
        ttk.Checkbutton(
            tab_desk, text="Activer les notifications bureau", variable=self._desk_en
        ).pack(anchor=tk.W, **pad)
        ttk.Button(tab_desk, text="Tester", command=lambda: alert_mgr.test_desktop()).pack(
            anchor=tk.W, padx=10
        )

        # ── Onglet Email ───────────────────────────────────────────────────────
        tab_mail = Frame(nb)
        nb.add(tab_mail, text="Email")
        self._mail_en = tk.BooleanVar(value=cfg.email_enabled)
        ttk.Checkbutton(tab_mail, text="Activer les alertes email", variable=self._mail_en).pack(
            anchor=tk.W, **pad
        )
        fields = [
            ("Serveur SMTP", "smtp_host"),
            ("Port", "smtp_port"),
            ("Utilisateur", "smtp_user"),
            ("Mot de passe", "smtp_password"),
            ("Destinataires (virgule)", "email_recipients"),
        ]
        self._mail_vars: dict = {}
        for label, key in fields:
            row = Frame(tab_mail)
            row.pack(fill=tk.X, padx=10, pady=2)
            Label(row, text=label, width=22, anchor=tk.W).pack(side=tk.LEFT)
            val = getattr(cfg, key)
            if isinstance(val, list):
                val = ", ".join(val)
            var = tk.StringVar(value=str(val))
            self._mail_vars[key] = var
            ttk.Entry(row, textvariable=var, width=28, show="*" if "password" in key else "").pack(
                side=tk.LEFT
            )

        # ── Onglet Webhook ────────────────────────────────────────────────────
        tab_wh = Frame(nb)
        nb.add(tab_wh, text="Webhook")
        self._wh_en = tk.BooleanVar(value=cfg.webhook_enabled)
        ttk.Checkbutton(tab_wh, text="Activer le webhook", variable=self._wh_en).pack(
            anchor=tk.W, **pad
        )
        for label, key in [("URL", "webhook_url"), ("Secret", "webhook_secret")]:
            row = Frame(tab_wh)
            row.pack(fill=tk.X, padx=10, pady=2)
            Label(row, text=label, width=10, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value=getattr(cfg, key))
            setattr(self, f"_wh_{key}", var)
            ttk.Entry(row, textvariable=var, width=36).pack(side=tk.LEFT)

        # ── Onglet Filtres ────────────────────────────────────────────────────
        tab_f = Frame(nb)
        nb.add(tab_f, text="Filtres")
        self._unk_var = tk.BooleanVar(value=cfg.alert_on_unknown)
        self._kn_var = tk.BooleanVar(value=cfg.alert_on_known)
        self._cooldown_var = tk.StringVar(value=str(cfg.cooldown_seconds))
        ttk.Checkbutton(tab_f, text="Alerter sur visage inconnu", variable=self._unk_var).pack(
            anchor=tk.W, **pad
        )
        ttk.Checkbutton(tab_f, text="Alerter sur visage connu", variable=self._kn_var).pack(
            anchor=tk.W, **pad
        )
        row = Frame(tab_f)
        row.pack(anchor=tk.W, **pad)
        Label(row, text="Anti-spam (secondes) :").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._cooldown_var, width=6).pack(side=tk.LEFT, padx=4)

        # Boutons
        btn_frame = Frame(self)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 10))
        ttk.Button(btn_frame, text="Annuler", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Sauvegarder", command=self._save).pack(side=tk.RIGHT)

    def _save(self) -> None:
        cfg = self._mgr.config
        cfg.desktop_enabled = self._desk_en.get()
        cfg.email_enabled = self._mail_en.get()
        cfg.smtp_host = self._mail_vars["smtp_host"].get()
        cfg.smtp_port = int(self._mail_vars["smtp_port"].get() or "587")
        cfg.smtp_user = self._mail_vars["smtp_user"].get()
        cfg.smtp_password = self._mail_vars["smtp_password"].get()
        cfg.email_recipients = [
            r.strip() for r in self._mail_vars["email_recipients"].get().split(",") if r.strip()
        ]
        cfg.webhook_enabled = self._wh_en.get()
        cfg.webhook_url = self._wh_webhook_url.get()
        cfg.webhook_secret = self._wh_webhook_secret.get()
        cfg.alert_on_unknown = self._unk_var.get()
        cfg.alert_on_known = self._kn_var.get()
        try:
            cfg.cooldown_seconds = float(self._cooldown_var.get())
        except ValueError:
            pass
        self._mgr.config = cfg
        self.destroy()


def main() -> None:
    import ttkbootstrap as ttk

    root = ttk.Window(themename="solar")
    root.withdraw()  # root caché, le dashboard est la fenêtre visible
    app = SurveillanceDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
