"""
image_importer.py
Interface graphique d'import et d'enregistrement de visages depuis des images.

Fonctionnalités :
  - Sélection de plusieurs images (fichier ou glisser-déposer)
  - Affichage des visages détectés avec rectangles numérotés
  - Sélection d'un visage, saisie du nom, sauvegarde dans la base
  - Navigation entre les images
  - Indicateurs : netteté, confiance, doublon
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageTk

from ..core.utils import is_duplicate, save_face_encoding
from ..storage.encodings_store import load_existing_encodings

logger = logging.getLogger(__name__)

# Extensions image acceptées
IMAGE_EXTENSIONS = (
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp",
)


class ImageImporterApp(tk.Toplevel):
    """
    Fenêtre d'import et d'enregistrement de visages depuis des images statiques.
    """

    PREVIEW_W = 720
    PREVIEW_H = 500

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.title("Import d'images — Enregistrement de visages")
        self.geometry("980x680")
        self.minsize(800, 560)

        # État
        self._image_paths: List[Path] = []
        self._current_idx: int = 0
        self._current_image: Optional[np.ndarray] = None   # BGR original
        self._face_locations: List[Tuple] = []
        self._face_encodings: List[np.ndarray] = []
        self._selected_face: int = 0
        self._photo_ref: Optional[ImageTk.PhotoImage] = None
        self._processing = False
        # Initialisé avant _build_ui() pour éviter l'AttributeError si _build_ui() échoue
        self._progress: Optional[ttk.Progressbar] = None

        self._build_ui()

    # ── Interface ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Barre de contrôle supérieure
        top = tk.Frame(self, bg="#2b2b2b", padx=8, pady=6)
        top.pack(fill=tk.X)

        tk.Label(top, text="Import d'images", bg="#2b2b2b", fg="white",
                 font=("Helvetica", 13, "bold")).pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="Sélectionner des images", command=self._browse_images).pack(
            side=tk.LEFT, padx=8)
        ttk.Button(top, text="◀ Précédente", command=self._prev_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Suivante ▶", command=self._next_image).pack(side=tk.LEFT, padx=4)

        self._counter_var = tk.StringVar(value="–")
        tk.Label(top, textvariable=self._counter_var, bg="#2b2b2b", fg="#aaaaaa",
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=8)

        # ── Corps principal
        body = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=5)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Panneau gauche : aperçu image
        left = tk.Frame(body, bg="#1a1a1a")
        body.add(left, minsize=400)

        self._canvas = tk.Canvas(left, bg="#1a1a1a", cursor="crosshair",
                                  width=self.PREVIEW_W, height=self.PREVIEW_H)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Button-1>", self._on_canvas_click)

        # Barre de statut image
        self._img_status = tk.StringVar(value="Sélectionnez des images pour commencer.")
        tk.Label(left, textvariable=self._img_status, bg="#111", fg="#ccc",
                 font=("Helvetica", 9), anchor=tk.W).pack(fill=tk.X, padx=4, pady=2)

        # Panneau droit : sélection et enregistrement
        right = tk.Frame(body, padx=10, pady=8)
        body.add(right, minsize=240)

        tk.Label(right, text="Visages détectés", font=("Helvetica", 11, "bold")).pack(anchor=tk.W)

        # Liste des visages
        list_frame = tk.Frame(right, relief=tk.GROOVE, bd=1)
        list_frame.pack(fill=tk.X, pady=6)

        self._face_listbox = tk.Listbox(list_frame, height=8, selectmode=tk.SINGLE,
                                         activestyle="dotbox")
        self._face_listbox.pack(fill=tk.X)
        self._face_listbox.bind("<<ListboxSelect>>", self._on_face_select)

        # Infos visage sélectionné
        info_lf = ttk.LabelFrame(right, text="Informations")
        info_lf.pack(fill=tk.X, pady=4)
        self._info_var = tk.StringVar(value="–")
        tk.Label(info_lf, textvariable=self._info_var, justify=tk.LEFT,
                 font=("Helvetica", 9), fg="gray").pack(anchor=tk.W, padx=6, pady=4)

        # Aperçu du visage sélectionné
        self._face_preview_label = tk.Label(right, bg="#111", width=10, height=5)
        self._face_preview_label.pack(pady=4)

        # Saisie du nom
        tk.Label(right, text="Nom de la personne :").pack(anchor=tk.W, pady=(8, 2))
        self._name_var = tk.StringVar()
        self._name_entry = ttk.Entry(right, textvariable=self._name_var, width=28)
        self._name_entry.pack(fill=tk.X)

        # Bouton enregistrer
        ttk.Button(right, text="Enregistrer le visage", command=self._save_face).pack(
            pady=10, fill=tk.X)

        # Journal de cette session
        tk.Label(right, text="Journal de session", font=("Helvetica", 10, "bold")).pack(
            anchor=tk.W, pady=(10, 2))
        log_frame = tk.Frame(right, relief=tk.GROOVE, bd=1)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self._log_text = tk.Text(log_frame, height=6, state=tk.DISABLED,
                                  font=("Courier", 8), bg="#f5f5f5")
        log_scroll = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_scroll.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Progression
        self._progress = ttk.Progressbar(self, mode="indeterminate")
        self._progress.pack(fill=tk.X, padx=6, pady=(0, 4))

    # ── Sélection d'images ────────────────────────────────────────────────────

    def _browse_images(self) -> None:
        filetypes = [
            ("Images", " ".join(IMAGE_EXTENSIONS)),
            ("Tous les fichiers", "*.*"),
        ]
        paths = filedialog.askopenfilenames(title="Sélectionner des images", filetypes=filetypes)
        if not paths:
            return
        self._image_paths = [Path(p) for p in paths]
        self._current_idx = 0
        self._load_current_image()

    def _prev_image(self) -> None:
        if not self._image_paths or self._current_idx == 0:
            return
        self._current_idx -= 1
        self._load_current_image()

    def _next_image(self) -> None:
        if not self._image_paths or self._current_idx >= len(self._image_paths) - 1:
            return
        self._current_idx += 1
        self._load_current_image()

    # ── Chargement et analyse d'une image ────────────────────────────────────

    def _load_current_image(self) -> None:
        if self._processing:
            return
        path = self._image_paths[self._current_idx]
        n = len(self._image_paths)
        self._counter_var.set(f"{self._current_idx + 1} / {n}")
        self._img_status.set(f"Analyse en cours : {path.name}…")
        self._face_listbox.delete(0, tk.END)
        self._info_var.set("–")
        self._face_preview_label.configure(image="", bg="#111")
        self._photo_ref = None

        if self._progress is not None:
            self._progress.start()
        self._processing = True
        threading.Thread(target=self._analyse_image, args=(path,), daemon=True).start()

    def _schedule(self, callback) -> None:
        """Planifie un callback dans le thread principal, seulement si la fenêtre existe."""
        try:
            if self.winfo_exists():
                self.after(0, callback)
        except Exception:
            pass

    def _analyse_image(self, path: Path) -> None:
        """Analyse dans un thread secondaire."""
        try:
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                self._schedule(lambda: self._img_status.set(f"Impossible de lire : {path.name}"))
                return

            img_bgr = self._fit_image(img_bgr, 1920, 1080)
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, locations)

            self._schedule(lambda: self._display_results(img_bgr, locations, encodings))
        except Exception as exc:
            logger.exception("Erreur analyse image : %s", exc)
            self._schedule(lambda: self._img_status.set(f"Erreur : {exc}"))
        finally:
            def _cleanup():
                if self._progress is not None:
                    self._progress.stop()
                self._processing = False
            self._schedule(_cleanup)

    def _display_results(
        self,
        img_bgr: np.ndarray,
        locations: list,
        encodings: list,
    ) -> None:
        self._current_image = img_bgr
        self._face_locations = locations
        self._face_encodings = encodings
        self._selected_face = 0

        # Mettre à jour la liste
        self._face_listbox.delete(0, tk.END)
        for i, loc in enumerate(locations):
            self._face_listbox.insert(tk.END, f"Visage {i + 1}")
        if locations:
            self._face_listbox.selection_set(0)
            self._on_face_select(None)

        self._render_canvas()
        n = len(locations)
        path_name = self._image_paths[self._current_idx].name
        self._img_status.set(
            f"{path_name}  —  {n} visage(s) détecté(s). Cliquez sur un visage ou sélectionnez dans la liste."
        )

    # ── Rendu canvas ─────────────────────────────────────────────────────────

    def _render_canvas(self) -> None:
        if self._current_image is None:
            return
        annotated = self._current_image.copy()
        for i, loc in enumerate(self._face_locations):
            top, right, bottom, left = loc
            color = (0, 200, 0) if i == self._selected_face else (200, 200, 200)
            thickness = 3 if i == self._selected_face else 1
            cv2.rectangle(annotated, (left, top), (right, bottom), color, thickness)
            cv2.putText(annotated, str(i + 1), (left + 4, top + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Adapter au canvas
        cw = self._canvas.winfo_width() or self.PREVIEW_W
        ch = self._canvas.winfo_height() or self.PREVIEW_H
        annotated = self._fit_image(annotated, cw, ch)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self._photo_ref = ImageTk.PhotoImage(pil)
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo_ref)

    # ── Sélection de visage ───────────────────────────────────────────────────

    def _on_face_select(self, _event) -> None:
        sel = self._face_listbox.curselection()
        if not sel or not self._face_locations:
            return
        idx = sel[0]
        self._selected_face = idx
        self._render_canvas()
        self._update_face_info(idx)
        self._update_face_preview(idx)

    def _on_canvas_click(self, event: tk.Event) -> None:
        if not self._face_locations or self._current_image is None:
            return
        # Convertir coords canvas → coords image
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        h, w = self._current_image.shape[:2]
        scale = min(cw / w, ch / h)
        off_x = (cw - w * scale) / 2
        off_y = (ch - h * scale) / 2
        img_x = (event.x - off_x) / scale
        img_y = (event.y - off_y) / scale

        for i, (top, right, bottom, left) in enumerate(self._face_locations):
            if left <= img_x <= right and top <= img_y <= bottom:
                self._face_listbox.selection_clear(0, tk.END)
                self._face_listbox.selection_set(i)
                self._on_face_select(None)
                break

    def _update_face_info(self, idx: int) -> None:
        if idx >= len(self._face_encodings):
            self._info_var.set("–")
            return

        enc = self._face_encodings[idx]
        loc = self._face_locations[idx]
        top, right, bottom, left = loc
        h, w = self._current_image.shape[:2]
        area_pct = ((right - left) * (bottom - top)) / (w * h) * 100

        # Netteté (variance du Laplacien)
        face_crop = self._current_image[top:bottom, left:right]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Doublon ?
        existing = load_existing_encodings()
        dup = is_duplicate(enc, existing)
        dup_str = "Oui ⚠" if dup else "Non ✓"

        self._info_var.set(
            f"Taille : {right-left}×{bottom-top} px\n"
            f"Zone image : {area_pct:.1f}%\n"
            f"Netteté : {sharpness:.0f}\n"
            f"Déjà dans la base : {dup_str}"
        )

    def _update_face_preview(self, idx: int) -> None:
        if self._current_image is None or idx >= len(self._face_locations):
            return
        top, right, bottom, left = self._face_locations[idx]
        # Marge
        margin = 20
        top2 = max(0, top - margin)
        left2 = max(0, left - margin)
        bottom2 = min(self._current_image.shape[0], bottom + margin)
        right2 = min(self._current_image.shape[1], right + margin)

        crop = self._current_image[top2:bottom2, left2:right2]
        crop = self._fit_image(crop, 120, 120)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil)
        self._face_preview_label.configure(image=photo, bg="#111")
        self._face_preview_label.image = photo  # garder la référence

    # ── Enregistrement ────────────────────────────────────────────────────────

    def _save_face(self) -> None:
        name = self._name_var.get().strip()
        if not name:
            messagebox.showwarning("Nom manquant", "Veuillez saisir un nom.", parent=self)
            return
        if not self._face_encodings:
            messagebox.showwarning("Aucun visage", "Aucun visage sélectionné.", parent=self)
            return

        idx = self._selected_face
        if idx >= len(self._face_encodings):
            messagebox.showwarning("Sélection", "Sélectionnez un visage dans la liste.", parent=self)
            return

        enc = self._face_encodings[idx]
        loc = self._face_locations[idx]
        top, right, bottom, left = loc
        face_img = self._current_image[top:bottom, left:right]

        # Vérification doublon
        existing = load_existing_encodings()
        if is_duplicate(enc, existing):
            if not messagebox.askyesno(
                "Doublon détecté",
                "Un visage similaire existe déjà dans la base.\nEnregistrer quand même ?",
                parent=self,
            ):
                return

        try:
            save_face_encoding(name, enc, face_img)
            self._log(f"✓ {name} enregistré (visage {idx + 1})")
            self._name_var.set("")
        except Exception as exc:
            logger.exception("Erreur sauvegarde : %s", exc)
            messagebox.showerror("Erreur", str(exc), parent=self)

    # ── Utilitaires ───────────────────────────────────────────────────────────

    @staticmethod
    def _fit_image(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return img

    def _log(self, msg: str) -> None:
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)
