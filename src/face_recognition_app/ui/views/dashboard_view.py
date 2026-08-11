"""
dashboard_view.py
Vue « Tableau de bord » — grille des flux caméra et journal des détections.

Contient les deux composants propres à cette vue :

  - `CameraTile`   : une vignette de flux, avec son badge d'état.
  - `TerminalFeed` : les dernières lignes du journal applicatif, branchées sur
                     le logger racine.

Le rafraîchissement de la grille est suspendu quand la vue n'est pas visible :
inutile de convertir des images pour un panneau caché.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from ... import theme
from ...services.surveillance_engine import SurveillanceEvent
from ..widgets import Canvas, Frame, Label, Text
from .base import AppServices, View, entete_panneau

logger = logging.getLogger(__name__)

# Taille de référence d'une vignette (px) — la grille les étire ensuite.
THUMB_W = 320
THUMB_H = 240

GRID_COLS = 3
GRID_ROWS = 2
MAX_CAMERAS_VISIBLE = GRID_COLS * GRID_ROWS


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


class DashboardView(View):
    """Grille des flux et journal des détections."""

    TITRE = "TABLEAU DE BORD"
    MAX_EVENEMENTS = 100
    REFRESH_MS = 200

    def __init__(self, parent, services: AppServices, on_status) -> None:
        super().__init__(parent, services)
        self._on_status = on_status
        self._tiles: dict[str, CameraTile] = {}
        self._annotated_lock = threading.Lock()
        self._annotated: dict[str, object] = {}
        self._evenements: list[SurveillanceEvent] = []
        self._event_photo_refs: list[ImageTk.PhotoImage] = []
        self._event_count = 0
        self._visible = False
        self._rafraichissement: str | None = None

    # ── Construction ──────────────────────────────────────────────────────────

    def construire(self) -> None:
        body = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            bg=theme.BORDER,
            sashwidth=theme.BORDER_W,
            sashrelief=tk.FLAT,
            bd=0,
        )
        body.pack(fill=tk.BOTH, expand=True)

        # ── Panneau gauche : flux vidéo
        gauche = Frame(body, bg=theme.BG_BASE)
        body.add(gauche, minsize=640)

        entete_flux = entete_panneau(gauche, "FLUX VIDÉO EN DIRECT")
        self._grille_var = tk.StringVar(value="")
        Label(
            entete_flux,
            textvariable=self._grille_var,
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        ).pack(side=tk.RIGHT)

        self._grid_frame = Frame(gauche, bg=theme.BG_BASE, padx=theme.PAD_XS)
        self._grid_frame.pack(fill=tk.BOTH, expand=True)

        # ── Panneau droit : journal
        droite = Frame(body, bg=theme.BG_BASE)
        body.add(droite, minsize=340)

        entete_journal = entete_panneau(droite, "JOURNAL DES ÉVÉNEMENTS")
        self._live_badge = Label(
            entete_journal,
            text=" LIVE ",
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        )
        self._live_badge.pack(side=tk.RIGHT)

        barre_filtres = Frame(droite, bg=theme.BG_BASE, padx=theme.PAD_S, pady=theme.PAD_S)
        barre_filtres.pack(fill=tk.X)
        self._filtre = tk.StringVar(value="TOUT")
        self._chips: dict[str, Label] = {}
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

        cadre_journal = Frame(droite, bg=theme.BG_BASE)
        cadre_journal.pack(fill=tk.BOTH, expand=True)
        self._event_canvas = Canvas(cadre_journal, bg=theme.BG_BASE, highlightthickness=0)
        defilement = ttk.Scrollbar(
            cadre_journal, orient=tk.VERTICAL, command=self._event_canvas.yview
        )
        self._event_canvas.configure(yscrollcommand=defilement.set)
        defilement.pack(side=tk.RIGHT, fill=tk.Y)
        self._event_canvas.pack(fill=tk.BOTH, expand=True)

        self._event_inner = Frame(self._event_canvas, bg=theme.BG_BASE)
        self._event_canvas.create_window((0, 0), window=self._event_inner, anchor=tk.NW)
        self._event_inner.bind(
            "<Configure>",
            lambda e: self._event_canvas.configure(scrollregion=self._event_canvas.bbox("all")),
        )

        self.terminal = TerminalFeed(droite, hauteur=7)
        self.terminal.pack(side=tk.BOTTOM, fill=tk.X)

        self._changer_filtre("TOUT")
        self._rebuild_tiles()
        self.services.cameras.on_change(self._on_cameras_changed)

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def on_show(self) -> None:
        self._visible = True
        self._schedule_refresh()

    def on_hide(self) -> None:
        self._visible = False
        if self._rafraichissement is not None:
            self.after_cancel(self._rafraichissement)
            self._rafraichissement = None

    def fermer(self) -> None:
        self.on_hide()
        if self._construite:
            self.terminal.detacher()

    def marquer_surveillance(self, active: bool) -> None:
        """Le badge LIVE suit l'état du moteur."""
        if not self._construite:
            return
        self._live_badge.configure(
            bg=theme.STATE_DANGER if active else theme.BG_SURFACE,
            fg=theme.TEXT_PRIMARY if active else theme.TEXT_SECONDARY,
        )

    # ── Réception des événements du moteur ────────────────────────────────────

    def encaisser_evenement(self, event: SurveillanceEvent) -> None:
        """Appelé depuis un thread d'analyse : ne fait que mémoriser la frame."""
        if event.frame is not None:
            with self._annotated_lock:
                self._annotated[event.camera_uid] = event.frame

    def _rebuild_tiles(self) -> None:
        """Reconstruit la grille de vignettes à partir des caméras configurées."""
        # Nettoyer
        for w in self._grid_frame.winfo_children():
            w.destroy()
        self._tiles.clear()

        configs = self.services.cameras.list_configs()
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

    def _on_cameras_changed(self) -> None:
        self.after(0, self._rebuild_tiles)

    def ajouter_au_journal(self, event: SurveillanceEvent) -> None:
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
        self._on_status(
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

    def _schedule_refresh(self) -> None:
        if not self._visible:
            return
        self._refresh_tiles()
        self._rafraichissement = self.after(self.REFRESH_MS, self._schedule_refresh)

    def _refresh_tiles(self) -> None:
        """Met à jour chaque tuile avec la dernière frame disponible."""
        with self._annotated_lock:
            annotated_copy = dict(self._annotated)
            self._annotated.clear()

        for uid, tile in self._tiles.items():
            # État réel de la source : le badge affiche le délai de reconnexion
            # plutôt qu'un simple point rouge.
            source = self.services.cameras.get_source(uid)
            if source is None:
                tile.set_status(None)
            else:
                tile.set_status(source.state, source.next_retry_in)

            fps_str = ""
            stats = self.services.engine.get_stats(uid)
            if stats and stats.fps > 0:
                fps_str = f"{stats.fps:.1f} fps"

            annotated = annotated_copy.get(uid)
            frame = annotated if annotated is not None else self.services.cameras.get_frame(uid)
            tile.update_frame(frame, fps=fps_str)

    def _open_fullscreen(self, uid: str) -> None:
        """Ouvre une caméra en plein écran (double-clic sur la vignette)."""
        config = self.services.cameras.get_config(uid)
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
                frame = self.services.cameras.get_frame(uid)
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
