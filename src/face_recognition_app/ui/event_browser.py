"""
event_browser.py
Historique et statistiques des événements de surveillance.

Fonctionnalités :
  - Liste paginée des événements (filtrée par date / caméra / personne)
  - Aperçu du snapshot de chaque événement
  - Statistiques : total, par caméra, par personne
  - Nettoyage de l'historique (supprimer avant une date)
"""

from __future__ import annotations

import base64
import tkinter as tk
from datetime import datetime, timedelta
from io import BytesIO
from tkinter import messagebox, ttk
from typing import List, Optional

from PIL import Image, ImageTk

from ..storage.event_store import EventStore, StoredEvent


class EventBrowserApp(tk.Toplevel):
    """Fenêtre de consultation de l'historique des détections."""

    PAGE_SIZE = 50

    def __init__(self, parent: tk.Widget, event_store: EventStore) -> None:
        super().__init__(parent)
        self.title("Historique des détections")
        self.geometry("1100x680")
        self.minsize(800, 500)

        self._store = event_store
        self._events: List[StoredEvent] = []
        self._photo_refs: List[Optional[ImageTk.PhotoImage]] = []
        self._selected_event: Optional[StoredEvent] = None

        self._build_ui()
        self._load_events()

    # ── Interface ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Barre de filtres
        filter_bar = tk.Frame(self, bg="#2b2b2b", padx=8, pady=6)
        filter_bar.pack(fill=tk.X)

        tk.Label(filter_bar, text="Historique", bg="#2b2b2b", fg="white",
                 font=("Helvetica", 13, "bold")).pack(side=tk.LEFT, padx=6)

        tk.Label(filter_bar, text="Caméra :", bg="#2b2b2b", fg="#aaa").pack(side=tk.LEFT, padx=(12, 2))
        self._cam_var = tk.StringVar(value="Toutes")
        self._cam_cb = ttk.Combobox(filter_bar, textvariable=self._cam_var, width=14, state="readonly")
        self._cam_cb.pack(side=tk.LEFT)
        self._cam_cb.bind("<<ComboboxSelected>>", lambda _: self._load_events())

        tk.Label(filter_bar, text="Personne :", bg="#2b2b2b", fg="#aaa").pack(side=tk.LEFT, padx=(10, 2))
        self._person_var = tk.StringVar()
        ttk.Entry(filter_bar, textvariable=self._person_var, width=14).pack(side=tk.LEFT)

        ttk.Button(filter_bar, text="Filtrer", command=self._load_events).pack(side=tk.LEFT, padx=6)
        ttk.Button(filter_bar, text="Statistiques", command=self._show_stats).pack(side=tk.LEFT, padx=4)
        ttk.Button(filter_bar, text="Nettoyer…", command=self._purge_dialog).pack(side=tk.LEFT, padx=4)

        self._count_var = tk.StringVar(value="")
        tk.Label(filter_bar, textvariable=self._count_var, bg="#2b2b2b", fg="#aaa",
                 font=("Helvetica", 9)).pack(side=tk.RIGHT, padx=10)

        # Corps : liste à gauche, détail à droite
        body = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=5)
        body.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Liste des événements
        left = tk.Frame(body)
        body.add(left, minsize=480)

        cols = ("datetime", "camera", "faces")
        self._tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
        self._tree.heading("datetime", text="Date / Heure")
        self._tree.heading("camera", text="Caméra")
        self._tree.heading("faces", text="Visages détectés")
        self._tree.column("datetime", width=150, anchor=tk.W)
        self._tree.column("camera", width=120, anchor=tk.W)
        self._tree.column("faces", width=210, anchor=tk.W)

        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # Couleurs alternées + connus/inconnus
        self._tree.tag_configure("known", foreground="#2ecc71")
        self._tree.tag_configure("unknown", foreground="#e74c3c")
        self._tree.tag_configure("odd", background="#f9f9f9")

        # ── Détail d'un événement
        right = tk.Frame(body, padx=10, pady=8)
        body.add(right, minsize=280)

        tk.Label(right, text="Détail", font=("Helvetica", 11, "bold")).pack(anchor=tk.W)

        # Snapshot
        self._snap_label = tk.Label(right, bg="#111", width=30, height=12, text="Aucun snapshot",
                                     fg="#555")
        self._snap_label.pack(pady=8)

        # Infos
        info_lf = ttk.LabelFrame(right, text="Informations")
        info_lf.pack(fill=tk.X)
        self._detail_var = tk.StringVar(value="Sélectionnez un événement.")
        tk.Label(info_lf, textvariable=self._detail_var, justify=tk.LEFT,
                 font=("Helvetica", 9), wraplength=240).pack(anchor=tk.W, padx=6, pady=6)

    # ── Chargement ────────────────────────────────────────────────────────────

    def _load_events(self) -> None:
        cam = self._cam_var.get()
        person = self._person_var.get().strip()

        if person:
            events = self._store.get_by_person(person, self.PAGE_SIZE)
        elif cam and cam != "Toutes":
            # Trouver l'uid par le nom
            events = [
                e for e in self._store.get_recent(self.PAGE_SIZE * 3)
                if e.camera_name == cam
            ][:self.PAGE_SIZE]
        else:
            events = self._store.get_recent(self.PAGE_SIZE)

        self._events = events
        self._refresh_tree()
        self._update_camera_list()
        total = self._store.count()
        self._count_var.set(f"{len(events)} affichés / {total} total")

    def _refresh_tree(self) -> None:
        self._tree.delete(*self._tree.get_children())
        for i, evt in enumerate(self._events):
            names = ", ".join(evt.known_names) if evt.known_names else ""
            unknown_n = sum(1 for f in evt.faces if not f.get("is_known", False))
            face_str = names
            if unknown_n:
                face_str += (" + " if names else "") + f"{unknown_n} inconnu(s)"
            if not face_str:
                face_str = f"{len(evt.faces)} visage(s)"

            tag = "known" if evt.known_names and not evt.has_unknown else "unknown"
            if i % 2 == 1:
                self._tree.insert("", tk.END, iid=str(i), tags=(tag, "odd"),
                                   values=(evt.dt.strftime("%Y-%m-%d %H:%M:%S"),
                                           evt.camera_name, face_str))
            else:
                self._tree.insert("", tk.END, iid=str(i), tags=(tag,),
                                   values=(evt.dt.strftime("%Y-%m-%d %H:%M:%S"),
                                           evt.camera_name, face_str))

    def _update_camera_list(self) -> None:
        cameras = sorted({e.camera_name for e in self._store.get_recent(500)})
        self._cam_cb["values"] = ["Toutes"] + cameras

    # ── Sélection ─────────────────────────────────────────────────────────────

    def _on_select(self, _event) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx >= len(self._events):
            return
        evt = self._events[idx]
        self._selected_event = evt
        self._show_event_detail(evt)

    def _show_event_detail(self, evt: StoredEvent) -> None:
        # Snapshot
        if evt.snapshot_b64:
            try:
                img_data = base64.b64decode(evt.snapshot_b64)
                pil = Image.open(BytesIO(img_data))
                pil.thumbnail((320, 200))
                photo = ImageTk.PhotoImage(pil)
                self._snap_label.configure(image=photo, text="", bg="#111")
                self._snap_label.image = photo
            except Exception:
                self._snap_label.configure(image="", text="Snapshot invalide", bg="#111")
        else:
            self._snap_label.configure(image="", text="Pas de snapshot", bg="#111")

        # Infos
        lines = [
            f"ID : {evt.id}",
            f"Date : {evt.dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Caméra : {evt.camera_name}",
            "",
        ]
        for f in evt.faces:
            status = "✓ Connu" if f.get("is_known") else "✗ Inconnu"
            conf = f.get("confidence", 0)
            lines.append(f"  {f.get('name', '?')} — {status} ({conf:.0%})")
        self._detail_var.set("\n".join(lines))

    # ── Statistiques ──────────────────────────────────────────────────────────

    def _show_stats(self) -> None:
        stats = self._store.stats()
        lines = [f"Total d'événements : {stats['total']}", ""]
        for cam, count in stats.get("by_camera", {}).items():
            lines.append(f"  {cam} : {count}")
        messagebox.showinfo("Statistiques", "\n".join(lines), parent=self)

    # ── Nettoyage ─────────────────────────────────────────────────────────────

    def _purge_dialog(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Nettoyer l'historique")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="Supprimer les événements antérieurs à :",
                 font=("Helvetica", 10)).pack(padx=16, pady=(12, 4))

        days_var = tk.IntVar(value=30)
        frame = tk.Frame(dlg)
        frame.pack()
        ttk.Spinbox(frame, from_=1, to=365, textvariable=days_var, width=6).pack(side=tk.LEFT)
        tk.Label(frame, text=" jours").pack(side=tk.LEFT)

        def _do_purge():
            days = days_var.get()
            cutoff = (datetime.now() - timedelta(days=days)).timestamp()
            deleted = self._store.delete_before(cutoff)
            dlg.destroy()
            messagebox.showinfo("Nettoyage", f"{deleted} événement(s) supprimé(s).", parent=self)
            self._load_events()

        btn_frame = tk.Frame(dlg)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Annuler", command=dlg.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Supprimer", command=_do_purge).pack(side=tk.RIGHT)
