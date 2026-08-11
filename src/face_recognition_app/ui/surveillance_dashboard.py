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


class CameraTile(tk.Frame):
    """
    Vignette affichant le flux d'une caméra dans la grille.

    Structure :
      - en-tête : badge d'état (« EN LIGNE », « RECONNEXION 8 s »…) et FPS
      - corps   : dernière frame annotée, ou le flux brut
      - pied    : nom de la caméra et résumé des détections

    Le badge exploite la machine à états de `CameraSource` : il affiche le délai
    avant la prochaine tentative, au lieu du simple point rouge d'avant.
    """

    def __init__(self, parent: tk.Widget, uid: str, name: str, on_fullscreen=None) -> None:
        super().__init__(parent, bg=theme.BG_CARD, highlightthickness=1)
        self.configure(highlightbackground=theme.BORDER, highlightcolor=theme.BORDER)
        self.uid = uid
        self._on_fullscreen = on_fullscreen
        self._latest_frame = None

        # ── En-tête : badge d'état et cadence
        entete = tk.Frame(self, bg=theme.BG_CARD)
        entete.pack(fill=tk.X, padx=theme.PAD_M, pady=(theme.PAD_S, theme.PAD_XS))

        self._badge_dot = tk.Label(
            entete, text="●", bg=theme.BG_CARD, fg=theme.STATE_DANGER, font=theme.FONT_SMALL
        )
        self._badge_dot.pack(side=tk.LEFT)
        self._badge_label = tk.Label(
            entete,
            text="HORS LIGNE",
            bg=theme.BG_CARD,
            fg=theme.STATE_DANGER,
            font=theme.FONT_BADGE,
        )
        self._badge_label.pack(side=tk.LEFT, padx=theme.PAD_S)

        self._fps_label = tk.Label(
            entete, text="", bg=theme.BG_CARD, fg=theme.ACCENT_AI, font=theme.FONT_BADGE
        )
        self._fps_label.pack(side=tk.RIGHT)

        # ── Corps : le flux
        self._canvas = tk.Canvas(
            self, width=THUMB_W, height=THUMB_H, bg=theme.BG_BASE, highlightthickness=0
        )
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=theme.PAD_XS)
        self._canvas.bind("<Double-Button-1>", self._handle_dblclick)

        # ── Pied : nom et détections
        pied = tk.Frame(self, bg=theme.BG_CARD)
        pied.pack(fill=tk.X, padx=theme.PAD_M, pady=(theme.PAD_XS, theme.PAD_S))
        tk.Label(
            pied, text=name, bg=theme.BG_CARD, fg=theme.TEXT_PRIMARY, font=theme.FONT_SMALL
        ).pack(side=tk.LEFT)
        self._det_label = tk.Label(
            pied, text="", bg=theme.BG_CARD, fg=theme.TEXT_SECONDARY, font=theme.FONT_BADGE
        )
        self._det_label.pack(side=tk.RIGHT)

        self._photo_ref: ImageTk.PhotoImage | None = None
        self._last_annotated: object | None = None

        self._draw_placeholder()

    def _handle_dblclick(self, _event) -> None:
        if self._on_fullscreen:
            self._on_fullscreen(self.uid)

    # ── Mise à jour ───────────────────────────────────────────────────────────

    def set_status(self, state, retry_in: float = 0.0) -> None:
        """Met à jour le badge depuis l'état de la source vidéo."""
        libelle, couleur = theme.connection_badge(state, retry_in)
        self._badge_dot.configure(fg=couleur)
        self._badge_label.configure(text=libelle, fg=couleur)

    def update_frame(self, frame_bgr, detections: str = "", fps: str = "") -> None:
        """Appelé périodiquement depuis le thread Tkinter."""
        self._det_label.configure(text=detections)
        self._fps_label.configure(text=fps)

        if frame_bgr is None:
            return
        self._latest_frame = frame_bgr

        h, w = frame_bgr.shape[:2]
        scale = min(THUMB_W / w, THUMB_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._photo_ref = photo
        self._canvas.delete("all")
        ox = (THUMB_W - nw) // 2
        oy = (THUMB_H - nh) // 2
        self._canvas.create_image(ox, oy, anchor=tk.NW, image=photo)

    def set_annotated(self, frame_bgr) -> None:
        """Reçoit la frame annotée depuis le moteur de surveillance."""
        self._last_annotated = frame_bgr

    def pop_annotated(self):
        """Consomme et retourne la frame annotée (ou None)."""
        f = self._last_annotated
        self._last_annotated = None
        return f

    def _draw_placeholder(self) -> None:
        self._canvas.create_rectangle(0, 0, THUMB_W, THUMB_H, fill=theme.BG_BASE, outline="")
        self._canvas.create_text(
            THUMB_W // 2,
            THUMB_H // 2,
            text="Pas de signal",
            fill=theme.TEXT_SECONDARY,
            font=theme.FONT_SMALL,
        )


# ── Tableau de bord ────────────────────────────────────────────────────────────


class SurveillanceDashboard(tk.Toplevel):
    """
    Fenêtre principale du système de surveillance.

    Doit être créé comme Toplevel d'un root ttkbootstrap vivant.
    Le root est retiré (withdraw) puis détruit quand cette fenêtre se ferme.
    """

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

        # Références Tk images pour le journal
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
        # La couleur ne sert plus qu'à signaler : une seule action est colorée
        # (démarrer / arrêter), tout le reste est neutre. L'ancienne barre
        # alignait neuf boutons de neuf couleurs, où rien ne ressortait.
        topbar = tk.Frame(self, bg=theme.BG_SURFACE, padx=theme.PAD_L, pady=theme.PAD_M)
        topbar.pack(fill=tk.X)

        # ── Ligne 1 : identité, action principale, profil ──────────────────────
        ligne_haut = tk.Frame(topbar, bg=theme.BG_SURFACE)
        ligne_haut.pack(fill=tk.X)

        tk.Label(
            ligne_haut,
            text="◉",
            bg=theme.BG_SURFACE,
            fg=theme.BRAND_PRIMARY,
            font=(theme.FONT_TITLE[0], theme.FONT_TITLE[1]),
        ).pack(side=tk.LEFT)
        tk.Label(
            ligne_haut,
            text="Surveillance",
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_PRIMARY,
            font=theme.FONT_TITLE,
        ).pack(side=tk.LEFT, padx=(theme.PAD_M, theme.PAD_XL))

        self._start_btn = ttk.Button(
            ligne_haut,
            text="▶  Démarrer",
            command=self._start_surveillance,
            bootstyle="success",
            width=14,
        )
        self._start_btn.pack(side=tk.LEFT)

        self._stop_btn = ttk.Button(
            ligne_haut,
            text="■  Arrêter",
            command=self._stop_surveillance,
            state=tk.DISABLED,
            bootstyle="danger",
            width=14,
        )
        self._stop_btn.pack(side=tk.LEFT, padx=theme.PAD_M)

        # Profil, aligné à droite
        self._profile_var = tk.StringVar(value=self._profile_store.active_name)
        self._profile_cb = ttk.Combobox(
            ligne_haut,
            textvariable=self._profile_var,
            values=[p.name for p in self._profile_store.list_profiles()],
            width=10,
            state="readonly",
        )
        self._profile_cb.pack(side=tk.RIGHT)
        self._profile_cb.bind("<<ComboboxSelected>>", self._on_profile_change)
        tk.Label(
            ligne_haut,
            text="Profil",
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_SMALL,
        ).pack(side=tk.RIGHT, padx=theme.PAD_M)

        # ── Ligne 2 : navigation neutre et indicateurs ─────────────────────────
        ligne_bas = tk.Frame(topbar, bg=theme.BG_SURFACE)
        ligne_bas.pack(fill=tk.X, pady=(theme.PAD_M, 0))

        for libelle, commande in (
            ("＋ Caméra", self._add_camera),
            ("Images", self._open_image_importer),
            ("Vidéos", self._open_video_importer),
            ("Historique", self._open_event_browser),
            ("Visages", self._open_encodings_manager),
            ("Alertes", self._open_alerts_config),
        ):
            ttk.Button(
                ligne_bas, text=libelle, command=commande, bootstyle="secondary-outline"
            ).pack(side=tk.LEFT, padx=(0, theme.PAD_S))

        # API — un point coloré porte l'état, le bouton reste neutre
        self._api_var = tk.StringVar(value="API")
        self._api_dot = tk.Label(
            ligne_bas,
            text="●",
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_SMALL,
        )
        self._api_dot.pack(side=tk.RIGHT, padx=(theme.PAD_XS, 0))
        self._api_btn = ttk.Button(
            ligne_bas,
            textvariable=self._api_var,
            command=self._toggle_api,
            bootstyle="secondary-outline",
        )
        self._api_btn.pack(side=tk.RIGHT)

        # ── Bandeau de statut ──────────────────────────────────────────────────
        self._status_var = tk.StringVar(
            value="Prêt. Démarrez la surveillance ou ajoutez une caméra."
        )
        bandeau = tk.Frame(self, bg=theme.BG_CARD)
        bandeau.pack(fill=tk.X)
        tk.Label(
            bandeau,
            textvariable=self._status_var,
            bg=theme.BG_CARD,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_SMALL,
            anchor=tk.W,
        ).pack(fill=tk.X, padx=theme.PAD_L, pady=theme.PAD_S)

        # ── Corps ─────────────────────────────────────────────────────────────
        body = tk.PanedWindow(
            self, orient=tk.HORIZONTAL, bg=theme.BG_BASE, sashwidth=6, sashrelief=tk.FLAT
        )
        body.pack(fill=tk.BOTH, expand=True, padx=theme.PAD_S, pady=theme.PAD_S)

        # ── Panneau gauche : grille caméras + liste ─────────────────────────
        left_pane = tk.Frame(body, bg=theme.BG_BASE)
        body.add(left_pane, minsize=600)

        self._grid_frame = tk.Frame(left_pane, bg=theme.BG_BASE)

        # ── Liste des caméras configurées
        cam_list_lf = ttk.LabelFrame(left_pane, text="Caméras configurées")
        cam_list_lf.pack(
            side=tk.BOTTOM, fill=tk.X, padx=theme.PAD_S, pady=(theme.PAD_S, theme.PAD_XS)
        )

        self._cam_list = ttk.Treeview(
            cam_list_lf, columns=("type", "source", "etat"), show="headings", height=4
        )
        self._cam_list.heading("type", text="Type")
        self._cam_list.heading("source", text="Source")
        self._cam_list.heading("etat", text="État")
        self._cam_list.column("type", width=80, anchor=tk.CENTER)
        self._cam_list.column("source", width=220)
        self._cam_list.column("etat", width=110, anchor=tk.CENTER)
        self._cam_list.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._cam_list.tag_configure("actif", foreground=theme.STATE_OK)
        self._cam_list.tag_configure("attente", foreground=theme.STATE_WARN)
        self._cam_list.tag_configure("inactif", foreground=theme.TEXT_SECONDARY)

        cam_btns = tk.Frame(cam_list_lf, bg=theme.BG_BASE)
        cam_btns.pack(side=tk.RIGHT, fill=tk.Y, padx=theme.PAD_S)
        ttk.Button(cam_btns, text="Modifier", command=self._edit_camera).pack(pady=2, fill=tk.X)
        ttk.Button(cam_btns, text="Supprimer", command=self._remove_camera).pack(pady=2, fill=tk.X)
        ttk.Button(cam_btns, text="Tester", command=self._test_camera).pack(pady=2, fill=tk.X)

        self._grid_frame.pack(fill=tk.BOTH, expand=True)

        # ── Panneau droit : journal des événements
        right_pane = tk.Frame(body, bg=theme.BG_BASE)
        body.add(right_pane, minsize=300)

        entete_journal = tk.Frame(right_pane, bg=theme.BG_BASE)
        entete_journal.pack(fill=tk.X, padx=theme.PAD_M, pady=(theme.PAD_M, theme.PAD_S))
        tk.Label(
            entete_journal,
            text="Détections",
            bg=theme.BG_BASE,
            fg=theme.TEXT_PRIMARY,
            font=theme.FONT_HEADING,
        ).pack(side=tk.LEFT)
        self._event_count_var = tk.StringVar(value="")
        tk.Label(
            entete_journal,
            textvariable=self._event_count_var,
            bg=theme.BG_BASE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_SMALL,
        ).pack(side=tk.RIGHT)

        self._event_canvas = tk.Canvas(right_pane, bg=theme.BG_BASE, highlightthickness=0)
        event_scroll = ttk.Scrollbar(
            right_pane, orient=tk.VERTICAL, command=self._event_canvas.yview
        )
        self._event_canvas.configure(yscrollcommand=event_scroll.set)
        event_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._event_canvas.pack(fill=tk.BOTH, expand=True)

        self._event_inner = tk.Frame(self._event_canvas, bg=theme.BG_BASE)
        self._event_canvas.create_window((0, 0), window=self._event_inner, anchor=tk.NW)
        self._event_inner.bind(
            "<Configure>",
            lambda e: self._event_canvas.configure(scrollregion=self._event_canvas.bbox("all")),
        )

        # Compteur événements
        self._event_count = 0

    # ── Gestion des tuiles (grille) ────────────────────────────────────────────

    def _rebuild_tiles(self) -> None:
        """Reconstruit la grille de vignettes à partir des caméras configurées."""
        # Nettoyer
        for w in self._grid_frame.winfo_children():
            w.destroy()
        self._tiles.clear()

        configs = self._cam_mgr.list_configs()
        if not configs:
            tk.Label(
                self._grid_frame,
                text="Aucune caméra configurée.\nCliquez sur « + Caméra » pour en ajouter une.",
                bg=theme.BG_BASE,
                fg=theme.TEXT_SECONDARY,
                font=("Helvetica", 13),
                justify=tk.CENTER,
            ).pack(expand=True)
            return

        visible = configs[:MAX_CAMERAS_VISIBLE]
        for i, cfg in enumerate(visible):
            row, col = divmod(i, GRID_COLS)
            tile = CameraTile(
                self._grid_frame, cfg.uid, cfg.name, on_fullscreen=self._open_fullscreen
            )
            tile.grid(row=row, column=col, padx=3, pady=3, sticky=tk.NSEW)
            self._tiles[cfg.uid] = tile

        # Poids de grille égaux
        for c in range(GRID_COLS):
            self._grid_frame.columnconfigure(c, weight=1)
        for r in range(GRID_ROWS):
            self._grid_frame.rowconfigure(r, weight=1)

        self._refresh_cam_list()

    def _refresh_cam_list(self) -> None:
        for row in self._cam_list.get_children():
            self._cam_list.delete(row)
        for cfg in self._cam_mgr.list_configs():
            type_label = "Webcam" if cfg.source_type == "webcam" else "IP"
            etat = (
                "Actif"
                if self._cam_mgr.is_running(cfg.uid)
                else ("Désactivé" if not cfg.enabled else "Arrêté")
            )
            self._cam_list.insert(
                "", tk.END, iid=cfg.uid, values=(type_label, str(cfg.source), etat)
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
        self._status_var.set("Surveillance active…")
        self._refresh_cam_list()

    def _stop_surveillance(self) -> None:
        self._engine.stop()
        self._cam_mgr.stop_all()
        self._start_btn.configure(state=tk.NORMAL)
        self._stop_btn.configure(state=tk.DISABLED)
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
        """Insère une entrée dans le panneau journal."""
        self._event_count += 1
        # Limiter à 50 entrées visibles
        children = self._event_inner.winfo_children()
        if len(children) > 50:
            children[0].destroy()

        ts = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        names = ", ".join(event.known_names) if event.known_names else ""
        unknown = (
            f" + {sum(1 for f in event.faces if not f.is_known)} inconnu(s)"
            if event.has_unknown
            else ""
        )

        # Une carte par détection, avec un liseré coloré à gauche : rouge s'il y
        # a un inconnu, vert si tous les visages sont reconnus. La couleur porte
        # l'information la plus importante, lisible sans lire le texte.
        accent = theme.STATE_DANGER if event.has_unknown else theme.STATE_OK

        entry = tk.Frame(self._event_inner, bg=accent)
        entry.pack(fill=tk.X, padx=theme.PAD_M, pady=theme.PAD_XS)

        carte = tk.Frame(entry, bg=theme.BG_CARD)
        carte.pack(fill=tk.X, padx=(3, 0))

        # Miniature snapshot
        if event.frame is not None:
            try:
                thumb = cv2.resize(event.frame, (72, 54))
                rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self._event_photo_refs.append(photo)
                if len(self._event_photo_refs) > 60:
                    self._event_photo_refs.pop(0)
                tk.Label(carte, image=photo, bg=theme.BG_CARD).pack(
                    side=tk.LEFT, padx=theme.PAD_S, pady=theme.PAD_S
                )
            except Exception as exc:
                logger.warning("Miniature du journal non générée : %s", exc)

        info = tk.Frame(carte, bg=theme.BG_CARD)
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=theme.PAD_M, pady=theme.PAD_S)

        ligne_haut = tk.Frame(info, bg=theme.BG_CARD)
        ligne_haut.pack(fill=tk.X)
        tk.Label(
            ligne_haut,
            text=event.camera_name,
            bg=theme.BG_CARD,
            fg=theme.TEXT_PRIMARY,
            font=theme.FONT_HEADING,
            anchor=tk.W,
        ).pack(side=tk.LEFT)
        tk.Label(
            ligne_haut,
            text=ts,
            bg=theme.BG_CARD,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE,
        ).pack(side=tk.RIGHT)

        text = (names + unknown) or "Visage(s) inconnu(s)"
        tk.Label(
            info,
            text=text,
            bg=theme.BG_CARD,
            fg=accent,
            font=theme.FONT_SMALL,
            anchor=tk.W,
        ).pack(anchor=tk.W)

        # Scroller vers le bas
        self._event_canvas.update_idletasks()
        self._event_canvas.yview_moveto(1.0)

        self._event_count_var.set(f"{self._event_count}")
        self._status_var.set(f"Dernière détection : {ts} — {event.camera_name} — {text}")

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

        canvas = tk.Canvas(win, bg=theme.BG_BASE, highlightthickness=0)
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

        tk.Label(
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
            self._api_var.set("API: OFF")
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

        self._api_var.set(f"API: :{self._api_server.port}")
        self._api_dot.configure(fg=theme.STATE_OK)
        self._afficher_cle_api()

    def _afficher_cle_api(self) -> None:
        """Montre l'URL et la clé, avec un bouton de copie."""
        fenetre = tk.Toplevel(self)
        fenetre.title("Accès à l'API")
        fenetre.resizable(False, False)
        fenetre.transient(self)

        tk.Label(
            fenetre,
            text="Envoyez cette clé dans l'en-tête X-API-Key :",
            font=("Helvetica", 10),
        ).pack(padx=16, pady=(14, 6))

        cle = self._api_server.api_key
        champ = ttk.Entry(fenetre, width=52, justify=tk.CENTER)
        champ.insert(0, cle)
        champ.configure(state="readonly")
        champ.pack(padx=16)

        tk.Label(
            fenetre,
            text=f"{self._api_server.url}/api/status",
            fg=theme.TEXT_SECONDARY,
            font=("Helvetica", 9),
        ).pack(pady=(6, 0))

        def copier() -> None:
            self.clipboard_clear()
            self.clipboard_append(cle)

        barre = tk.Frame(fenetre)
        barre.pack(pady=12)
        ttk.Button(barre, text="Copier la clé", command=copier).pack(side=tk.LEFT, padx=4)
        ttk.Button(barre, text="Fermer", command=fenetre.destroy).pack(side=tk.LEFT, padx=4)

    # ── Alertes ───────────────────────────────────────────────────────────────

    def _open_alerts_config(self) -> None:
        _AlertConfigDialog(self, self._alert_mgr)

    # ── Fermeture ─────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
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
        tab_desk = tk.Frame(nb)
        nb.add(tab_desk, text="Bureau")
        self._desk_en = tk.BooleanVar(value=cfg.desktop_enabled)
        ttk.Checkbutton(
            tab_desk, text="Activer les notifications bureau", variable=self._desk_en
        ).pack(anchor=tk.W, **pad)
        ttk.Button(tab_desk, text="Tester", command=lambda: alert_mgr.test_desktop()).pack(
            anchor=tk.W, padx=10
        )

        # ── Onglet Email ───────────────────────────────────────────────────────
        tab_mail = tk.Frame(nb)
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
            row = tk.Frame(tab_mail)
            row.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(row, text=label, width=22, anchor=tk.W).pack(side=tk.LEFT)
            val = getattr(cfg, key)
            if isinstance(val, list):
                val = ", ".join(val)
            var = tk.StringVar(value=str(val))
            self._mail_vars[key] = var
            ttk.Entry(row, textvariable=var, width=28, show="*" if "password" in key else "").pack(
                side=tk.LEFT
            )

        # ── Onglet Webhook ────────────────────────────────────────────────────
        tab_wh = tk.Frame(nb)
        nb.add(tab_wh, text="Webhook")
        self._wh_en = tk.BooleanVar(value=cfg.webhook_enabled)
        ttk.Checkbutton(tab_wh, text="Activer le webhook", variable=self._wh_en).pack(
            anchor=tk.W, **pad
        )
        for label, key in [("URL", "webhook_url"), ("Secret", "webhook_secret")]:
            row = tk.Frame(tab_wh)
            row.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(row, text=label, width=10, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value=getattr(cfg, key))
            setattr(self, f"_wh_{key}", var)
            ttk.Entry(row, textvariable=var, width=36).pack(side=tk.LEFT)

        # ── Onglet Filtres ────────────────────────────────────────────────────
        tab_f = tk.Frame(nb)
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
        row = tk.Frame(tab_f)
        row.pack(anchor=tk.W, **pad)
        tk.Label(row, text="Anti-spam (secondes) :").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._cooldown_var, width=6).pack(side=tk.LEFT, padx=4)

        # Boutons
        btn_frame = tk.Frame(self)
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
