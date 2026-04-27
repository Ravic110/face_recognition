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
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Optional

import cv2
from PIL import Image, ImageTk

from ..services.alert_manager import AlertManager
from ..services.api_server import ApiServer
from ..services.camera_manager import CameraManager
from ..services.camera_source import CameraConfig
from ..services.surveillance_engine import SurveillanceEngine, SurveillanceEvent
from ..services.video_recorder import VideoRecorder
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

    Affiche :
      - La dernière frame annotée ou le flux brut
      - Le nom de la caméra
      - Un indicateur de connexion (vert = OK, rouge = KO)
    """

    def __init__(self, parent: tk.Widget, uid: str, name: str,
                 on_fullscreen=None) -> None:
        super().__init__(parent, bg="#0d0d0d", relief=tk.RAISED, bd=1)
        self.uid = uid
        self._on_fullscreen = on_fullscreen
        self._latest_frame = None

        # Image
        self._canvas = tk.Canvas(self, width=THUMB_W, height=THUMB_H, bg="#0d0d0d",
                                  highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Double-Button-1>", self._handle_dblclick)

        # Pied : nom + statut + FPS
        foot = tk.Frame(self, bg="#1a1a1a")
        foot.pack(fill=tk.X)
        self._status_dot = tk.Label(foot, text="●", fg="#e74c3c", bg="#1a1a1a",
                                     font=("Helvetica", 10))
        self._status_dot.pack(side=tk.LEFT, padx=4)
        tk.Label(foot, text=name, bg="#1a1a1a", fg="#eeeeee",
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        self._fps_label = tk.Label(foot, text="", bg="#1a1a1a", fg="#3498db",
                                    font=("Helvetica", 8))
        self._fps_label.pack(side=tk.RIGHT, padx=2)
        self._det_label = tk.Label(foot, text="", bg="#1a1a1a", fg="#f39c12",
                                    font=("Helvetica", 8))
        self._det_label.pack(side=tk.RIGHT, padx=4)

        self._photo_ref: Optional[ImageTk.PhotoImage] = None
        self._last_annotated: Optional[object] = None

        self._draw_placeholder()

    def _handle_dblclick(self, _event) -> None:
        if self._on_fullscreen:
            self._on_fullscreen(self.uid)

    def update_frame(self, frame_bgr, connected: bool = True,
                     detections: str = "", fps: str = "") -> None:
        """Appelé périodiquement depuis le thread Tkinter."""
        self._status_dot.configure(fg="#2ecc71" if connected else "#e74c3c")
        self._det_label.configure(text=detections)
        self._fps_label.configure(text=fps)
        if frame_bgr is not None:
            self._latest_frame = frame_bgr

        if frame_bgr is None:
            return

        # Redimensionner
        h, w = frame_bgr.shape[:2]
        scale = min(THUMB_W / w, THUMB_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil)
        self._photo_ref = photo
        self._canvas.configure(width=THUMB_W, height=THUMB_H)
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
        self._canvas.create_rectangle(0, 0, THUMB_W, THUMB_H, fill="#0d0d0d", outline="")
        self._canvas.create_text(THUMB_W // 2, THUMB_H // 2, text="Pas de signal",
                                  fill="#444444", font=("Helvetica", 11))


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
        self.configure(bg="#111111")

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

        self._api_server = ApiServer(self._cam_mgr, self._engine, self._event_store, self._recorder)

        # Tiles par uid caméra
        self._tiles: Dict[str, CameraTile] = {}
        self._annotated_lock = threading.Lock()
        self._annotated: Dict[str, object] = {}   # uid → np.ndarray

        # Références Tk images pour le journal
        self._event_photo_refs: List[ImageTk.PhotoImage] = []

        self._build_ui()
        self._rebuild_tiles()
        self._cam_mgr.on_change(self._on_cameras_changed)
        self._engine.add_event_listener(self._on_surveillance_event)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_refresh()

    # ── Interface ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Barre supérieure ───────────────────────────────────────────────────
        topbar = tk.Frame(self, bg="#1a1a1a", padx=8, pady=6)
        topbar.pack(fill=tk.X)

        tk.Label(topbar, text="Surveillance Intelligente", bg="#1a1a1a", fg="white",
                 font=("Helvetica", 15, "bold")).pack(side=tk.LEFT, padx=6)

        # Boutons surveillance
        btn_cfg = {"relief": tk.FLAT, "bg": "#2ecc71", "fg": "white",
                   "font": ("Helvetica", 9, "bold"), "padx": 10, "pady": 4, "bd": 0}
        self._start_btn = tk.Button(topbar, text="▶  Démarrer", command=self._start_surveillance,
                                     **btn_cfg)
        self._start_btn.pack(side=tk.LEFT, padx=6)

        btn_stop_cfg = {**btn_cfg, "bg": "#e74c3c"}
        self._stop_btn = tk.Button(topbar, text="■  Arrêter", command=self._stop_surveillance,
                                    state=tk.DISABLED, **btn_stop_cfg)
        self._stop_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(topbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

        # Caméras
        tk.Button(topbar, text="+ Caméra", command=self._add_camera,
                  relief=tk.FLAT, bg="#3498db", fg="white",
                  font=("Helvetica", 9), padx=8, pady=4, bd=0).pack(side=tk.LEFT, padx=4)

        ttk.Separator(topbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

        # Modules d'import
        tk.Button(topbar, text="Import images", command=self._open_image_importer,
                  relief=tk.FLAT, bg="#8e44ad", fg="white",
                  font=("Helvetica", 9), padx=8, pady=4, bd=0).pack(side=tk.LEFT, padx=4)
        tk.Button(topbar, text="Import vidéos", command=self._open_video_importer,
                  relief=tk.FLAT, bg="#8e44ad", fg="white",
                  font=("Helvetica", 9), padx=8, pady=4, bd=0).pack(side=tk.LEFT, padx=4)

        ttk.Separator(topbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

        # Historique + encodages
        tk.Button(topbar, text="Historique", command=self._open_event_browser,
                  relief=tk.FLAT, bg="#16a085", fg="white",
                  font=("Helvetica", 9), padx=8, pady=4, bd=0).pack(side=tk.LEFT, padx=4)
        tk.Button(topbar, text="Encodages", command=self._open_encodings_manager,
                  relief=tk.FLAT, bg="#16a085", fg="white",
                  font=("Helvetica", 9), padx=8, pady=4, bd=0).pack(side=tk.LEFT, padx=4)

        ttk.Separator(topbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

        # Profil de surveillance (menu déroulant)
        tk.Label(topbar, text="Profil :", bg="#1a1a1a", fg="#aaa",
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        self._profile_var = tk.StringVar(value=self._profile_store.active_name)
        profile_names = [p.name for p in self._profile_store.list_profiles()]
        self._profile_cb = ttk.Combobox(topbar, textvariable=self._profile_var,
                                         values=profile_names, width=10, state="readonly")
        self._profile_cb.pack(side=tk.LEFT, padx=4)
        self._profile_cb.bind("<<ComboboxSelected>>", self._on_profile_change)

        ttk.Separator(topbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

        # API REST
        self._api_var = tk.StringVar(value="API: OFF")
        self._api_btn = tk.Button(topbar, textvariable=self._api_var,
                                   command=self._toggle_api,
                                   relief=tk.FLAT, bg="#555", fg="white",
                                   font=("Helvetica", 9), padx=8, pady=4, bd=0)
        self._api_btn.pack(side=tk.LEFT, padx=4)

        # Alertes
        tk.Button(topbar, text="Alertes", command=self._open_alerts_config,
                  relief=tk.FLAT, bg="#d35400", fg="white",
                  font=("Helvetica", 9), padx=8, pady=4, bd=0).pack(side=tk.LEFT, padx=4)

        # Statut global
        self._status_var = tk.StringVar(value="Prêt. Démarrez la surveillance ou ajoutez des caméras.")
        tk.Label(topbar, textvariable=self._status_var, bg="#1a1a1a", fg="#aaaaaa",
                 font=("Helvetica", 9)).pack(side=tk.RIGHT, padx=8)

        # ── Corps ─────────────────────────────────────────────────────────────
        body = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#111111",
                               sashwidth=6, sashrelief=tk.FLAT)
        body.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Panneau gauche : grille caméras + liste ─────────────────────────
        left_pane = tk.Frame(body, bg="#111111")
        body.add(left_pane, minsize=600)

        # Grille des vignettes
        self._grid_frame = tk.Frame(left_pane, bg="#111111")
        self._grid_frame.pack(fill=tk.BOTH, expand=True)

        # ── Liste des caméras configurées
        cam_list_lf = ttk.LabelFrame(left_pane, text="Caméras configurées")
        cam_list_lf.pack(fill=tk.X, padx=4, pady=(4, 2))

        self._cam_list = ttk.Treeview(cam_list_lf, columns=("type", "source", "etat"),
                                       show="headings", height=4)
        self._cam_list.heading("type", text="Type")
        self._cam_list.heading("source", text="Source")
        self._cam_list.heading("etat", text="État")
        self._cam_list.column("type", width=80, anchor=tk.CENTER)
        self._cam_list.column("source", width=220)
        self._cam_list.column("etat", width=80, anchor=tk.CENTER)
        self._cam_list.pack(side=tk.LEFT, fill=tk.X, expand=True)

        cam_btns = tk.Frame(cam_list_lf)
        cam_btns.pack(side=tk.RIGHT, fill=tk.Y, padx=4)
        ttk.Button(cam_btns, text="Modifier", command=self._edit_camera).pack(pady=2, fill=tk.X)
        ttk.Button(cam_btns, text="Supprimer", command=self._remove_camera).pack(pady=2, fill=tk.X)
        ttk.Button(cam_btns, text="Tester", command=self._test_camera).pack(pady=2, fill=tk.X)

        # ── Panneau droit : journal des événements
        right_pane = tk.Frame(body, bg="#111111")
        body.add(right_pane, minsize=280)

        tk.Label(right_pane, text="Journal de détections", bg="#111111", fg="white",
                 font=("Helvetica", 11, "bold")).pack(anchor=tk.W, padx=6, pady=(6, 2))

        self._event_canvas = tk.Canvas(right_pane, bg="#111111", highlightthickness=0)
        event_scroll = ttk.Scrollbar(right_pane, orient=tk.VERTICAL,
                                      command=self._event_canvas.yview)
        self._event_canvas.configure(yscrollcommand=event_scroll.set)
        event_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._event_canvas.pack(fill=tk.BOTH, expand=True)

        self._event_inner = tk.Frame(self._event_canvas, bg="#111111")
        self._event_canvas.create_window((0, 0), window=self._event_inner, anchor=tk.NW)
        self._event_inner.bind("<Configure>",
                                lambda e: self._event_canvas.configure(
                                    scrollregion=self._event_canvas.bbox("all")))

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
                bg="#111111", fg="#555555",
                font=("Helvetica", 13), justify=tk.CENTER,
            ).pack(expand=True)
            return

        visible = configs[:MAX_CAMERAS_VISIBLE]
        for i, cfg in enumerate(visible):
            row, col = divmod(i, GRID_COLS)
            tile = CameraTile(self._grid_frame, cfg.uid, cfg.name,
                              on_fullscreen=self._open_fullscreen)
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
            etat = "Actif" if self._cam_mgr.is_running(cfg.uid) else ("Désactivé" if not cfg.enabled else "Arrêté")
            self._cam_list.insert("", tk.END, iid=cfg.uid,
                                   values=(type_label, str(cfg.source), etat))

    # ── Callbacks caméras ─────────────────────────────────────────────────────

    def _on_cameras_changed(self) -> None:
        self.after(0, self._rebuild_tiles)

    def _add_camera(self) -> None:
        dlg = CameraConfigDialog(self, title="Ajouter une caméra")
        self.wait_window(dlg)
        if dlg.result:
            self._cam_mgr.add_camera(dlg.result)
            if self._engine._running:
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
            messagebox.showerror("Test",
                                  f"✗ Impossible de se connecter à « {config.name} ».\n"
                                  f"Source : {config.source}", parent=self)

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
        faces_data = [
            {"name": f.name, "confidence": f.confidence, "is_known": f.is_known}
            for f in event.faces
        ]
        self._event_store.record(
            timestamp=event.timestamp,
            camera_uid=event.camera_uid,
            camera_name=event.camera_name,
            faces=faces_data,
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
        unknown = f" + {sum(1 for f in event.faces if not f.is_known)} inconnu(s)" \
            if event.has_unknown else ""

        entry = tk.Frame(self._event_inner, bg="#1e1e1e", relief=tk.GROOVE, bd=1)
        entry.pack(fill=tk.X, padx=4, pady=2)

        # Miniature snapshot
        if event.frame is not None:
            try:
                import cv2 as _cv2
                thumb = _cv2.resize(event.frame, (72, 54))
                rgb = _cv2.cvtColor(thumb, _cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                photo = ImageTk.PhotoImage(pil)
                self._event_photo_refs.append(photo)
                if len(self._event_photo_refs) > 60:
                    self._event_photo_refs.pop(0)
                lbl = tk.Label(entry, image=photo, bg="#1e1e1e")
                lbl.pack(side=tk.LEFT, padx=4, pady=4)
            except Exception:
                pass

        info = tk.Frame(entry, bg="#1e1e1e")
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(info, text=f"[{ts}] {event.camera_name}", bg="#1e1e1e", fg="#3498db",
                 font=("Helvetica", 9, "bold"), anchor=tk.W).pack(anchor=tk.W)
        text = (names + unknown) or "Visage(s) inconnu(s)"
        tk.Label(info, text=text, bg="#1e1e1e", fg="#ecf0f1",
                 font=("Helvetica", 9), anchor=tk.W).pack(anchor=tk.W)

        # Scroller vers le bas
        self._event_canvas.update_idletasks()
        self._event_canvas.yview_moveto(1.0)

        self._status_var.set(
            f"Dernière détection : {ts} — {event.camera_name} — {text}"
        )

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
            connected = self._cam_mgr.is_running(uid)
            annotated = annotated_copy.get(uid)

            # FPS depuis le moteur
            fps_str = ""
            stats = self._engine.get_stats(uid)
            if stats and stats.fps > 0:
                fps_str = f"{stats.fps:.1f} fps"

            if annotated is not None:
                tile.update_frame(annotated, connected=connected, fps=fps_str)
            else:
                frame = self._cam_mgr.get_frame(uid)
                tile.update_frame(frame, connected=connected, fps=fps_str)

        if self._engine._running:
            self._refresh_cam_list()

    # ── Plein écran ───────────────────────────────────────────────────────────

    def _open_fullscreen(self, uid: str) -> None:
        """Ouvre une caméra en plein écran (double-clic sur la vignette)."""
        config = self._cam_mgr.get_config(uid)
        name = config.name if config else uid
        win = tk.Toplevel(self)
        win.title(f"Plein écran — {name}")
        win.configure(bg="#000")
        win.state("zoomed")

        canvas = tk.Canvas(win, bg="#000", highlightthickness=0)
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

        tk.Label(win, text="Appuyez sur Échap ou double-cliquez pour fermer",
                 bg="#000", fg="#555", font=("Helvetica", 9)).pack(side=tk.BOTTOM, pady=4)

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
            self._api_btn.configure(bg="#555")
        else:
            self._api_server.start(host="0.0.0.0", port=5000)
            self._api_var.set("API: :5000")
            self._api_btn.configure(bg="#27ae60")

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

    def __init__(self, parent: tk.Widget, alert_mgr: "AlertManager") -> None:
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
        tab_desk = tk.Frame(nb); nb.add(tab_desk, text="Bureau")
        self._desk_en = tk.BooleanVar(value=cfg.desktop_enabled)
        ttk.Checkbutton(tab_desk, text="Activer les notifications bureau",
                        variable=self._desk_en).pack(anchor=tk.W, **pad)
        ttk.Button(tab_desk, text="Tester",
                   command=lambda: alert_mgr.test_desktop()).pack(anchor=tk.W, padx=10)

        # ── Onglet Email ───────────────────────────────────────────────────────
        tab_mail = tk.Frame(nb); nb.add(tab_mail, text="Email")
        self._mail_en = tk.BooleanVar(value=cfg.email_enabled)
        ttk.Checkbutton(tab_mail, text="Activer les alertes email",
                        variable=self._mail_en).pack(anchor=tk.W, **pad)
        fields = [("Serveur SMTP", "smtp_host"), ("Port", "smtp_port"),
                  ("Utilisateur", "smtp_user"), ("Mot de passe", "smtp_password"),
                  ("Destinataires (virgule)", "email_recipients")]
        self._mail_vars: dict = {}
        for label, key in fields:
            row = tk.Frame(tab_mail); row.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(row, text=label, width=22, anchor=tk.W).pack(side=tk.LEFT)
            val = getattr(cfg, key)
            if isinstance(val, list):
                val = ", ".join(val)
            var = tk.StringVar(value=str(val))
            self._mail_vars[key] = var
            ttk.Entry(row, textvariable=var, width=28,
                      show="*" if "password" in key else "").pack(side=tk.LEFT)

        # ── Onglet Webhook ────────────────────────────────────────────────────
        tab_wh = tk.Frame(nb); nb.add(tab_wh, text="Webhook")
        self._wh_en = tk.BooleanVar(value=cfg.webhook_enabled)
        ttk.Checkbutton(tab_wh, text="Activer le webhook",
                        variable=self._wh_en).pack(anchor=tk.W, **pad)
        for label, key in [("URL", "webhook_url"), ("Secret", "webhook_secret")]:
            row = tk.Frame(tab_wh); row.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(row, text=label, width=10, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value=getattr(cfg, key))
            setattr(self, f"_wh_{key}", var)
            ttk.Entry(row, textvariable=var, width=36).pack(side=tk.LEFT)

        # ── Onglet Filtres ────────────────────────────────────────────────────
        tab_f = tk.Frame(nb); nb.add(tab_f, text="Filtres")
        self._unk_var = tk.BooleanVar(value=cfg.alert_on_unknown)
        self._kn_var = tk.BooleanVar(value=cfg.alert_on_known)
        self._cooldown_var = tk.StringVar(value=str(cfg.cooldown_seconds))
        ttk.Checkbutton(tab_f, text="Alerter sur visage inconnu",
                        variable=self._unk_var).pack(anchor=tk.W, **pad)
        ttk.Checkbutton(tab_f, text="Alerter sur visage connu",
                        variable=self._kn_var).pack(anchor=tk.W, **pad)
        row = tk.Frame(tab_f); row.pack(anchor=tk.W, **pad)
        tk.Label(row, text="Anti-spam (secondes) :").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._cooldown_var, width=6).pack(side=tk.LEFT, padx=4)

        # Boutons
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 10))
        ttk.Button(btn_frame, text="Annuler", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Sauvegarder", command=self._save).pack(side=tk.RIGHT)

    def _save(self) -> None:
        from ..services.alert_manager import AlertConfig
        cfg = self._mgr.config
        cfg.desktop_enabled = self._desk_en.get()
        cfg.email_enabled = self._mail_en.get()
        cfg.smtp_host = self._mail_vars["smtp_host"].get()
        cfg.smtp_port = int(self._mail_vars["smtp_port"].get() or "587")
        cfg.smtp_user = self._mail_vars["smtp_user"].get()
        cfg.smtp_password = self._mail_vars["smtp_password"].get()
        cfg.email_recipients = [r.strip() for r in
                                  self._mail_vars["email_recipients"].get().split(",") if r.strip()]
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
    root.withdraw()          # root caché, le dashboard est la fenêtre visible
    app = SurveillanceDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
