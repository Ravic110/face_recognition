"""
camera_config_dialog.py
Dialogue modal pour ajouter ou modifier une caméra.

Types supportés :
  - Webcam locale  (index 0, 1, 2…)
  - Caméra IP/RTSP (URL complète)
      exemples : rtsp://192.168.1.50:554/live
                 http://192.168.1.100:8080/video  ← Android « IP Webcam »
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

from ..services.camera_source import CameraConfig


class CameraConfigDialog(tk.Toplevel):
    """
    Fenêtre modale de configuration d'une caméra.

    Résultat accessible via la propriété `result` (CameraConfig ou None si annulé).

    Usage :
        dlg = CameraConfigDialog(parent, title="Ajouter une caméra")
        parent.wait_window(dlg)
        if dlg.result:
            camera_manager.add_camera(dlg.result)
    """

    def __init__(
        self,
        parent: tk.Widget,
        title: str = "Configuration de la caméra",
        config: Optional[CameraConfig] = None,   # None = création, sinon édition
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.grab_set()           # modal

        self.result: Optional[CameraConfig] = None
        self._existing = config
        self._type_var = tk.StringVar(value="webcam" if not config else config.source_type)

        self._build_ui()
        if config:
            self._populate(config)

        self.transient(parent)
        self.update_idletasks()
        # Centrer par rapport à la fenêtre parente
        px = parent.winfo_rootx() + parent.winfo_width() // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2
        self.geometry(f"+{px - self.winfo_width() // 2}+{py - self.winfo_height() // 2}")

    # ── Construction de l'interface ───────────────────────────────────────────

    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 6}

        # ── Titre interne
        header = tk.Frame(self, bg="#2b2b2b")
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="Ajouter / Modifier une caméra",
            bg="#2b2b2b", fg="white",
            font=("Helvetica", 13, "bold"),
            pady=10,
        ).pack()

        form = tk.Frame(self, padx=16, pady=12)
        form.pack(fill=tk.BOTH, expand=True)

        # Nom
        tk.Label(form, text="Nom de la caméra :").grid(row=0, column=0, sticky=tk.W, **pad)
        self._name_var = tk.StringVar()
        ttk.Entry(form, textvariable=self._name_var, width=34).grid(row=0, column=1, sticky=tk.EW, **pad)

        # Type
        tk.Label(form, text="Type :").grid(row=1, column=0, sticky=tk.W, **pad)
        type_frame = tk.Frame(form)
        type_frame.grid(row=1, column=1, sticky=tk.W, **pad)
        ttk.Radiobutton(
            type_frame, text="Webcam locale", variable=self._type_var,
            value="webcam", command=self._on_type_change,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(
            type_frame, text="Caméra IP / Smartphone", variable=self._type_var,
            value="ip", command=self._on_type_change,
        ).pack(side=tk.LEFT)

        # Source (index ou URL)
        tk.Label(form, text="Source :").grid(row=2, column=0, sticky=tk.W, **pad)
        self._source_var = tk.StringVar(value="0")
        self._source_entry = ttk.Entry(form, textvariable=self._source_var, width=34)
        self._source_entry.grid(row=2, column=1, sticky=tk.EW, **pad)

        # Aide contextuelle
        self._hint_var = tk.StringVar()
        tk.Label(
            form, textvariable=self._hint_var,
            fg="gray", wraplength=280, justify=tk.LEFT, font=("Helvetica", 9),
        ).grid(row=3, column=1, sticky=tk.W, padx=12, pady=(0, 6))

        # Résolution
        res_frame = tk.Frame(form)
        res_frame.grid(row=4, column=1, sticky=tk.W, **pad)
        tk.Label(form, text="Résolution :").grid(row=4, column=0, sticky=tk.W, **pad)
        self._width_var = tk.StringVar(value="640")
        self._height_var = tk.StringVar(value="480")
        ttk.Entry(res_frame, textvariable=self._width_var, width=6).pack(side=tk.LEFT)
        tk.Label(res_frame, text=" × ").pack(side=tk.LEFT)
        ttk.Entry(res_frame, textvariable=self._height_var, width=6).pack(side=tk.LEFT)

        # Modèle de détection
        tk.Label(form, text="Modèle détection :").grid(row=5, column=0, sticky=tk.W, **pad)
        self._model_var = tk.StringVar(value="hog")
        model_frame = tk.Frame(form)
        model_frame.grid(row=5, column=1, sticky=tk.W, **pad)
        ttk.Radiobutton(model_frame, text="HOG (CPU, rapide)", variable=self._model_var,
                        value="hog").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(model_frame, text="CNN (GPU, précis)", variable=self._model_var,
                        value="cnn").pack(side=tk.LEFT)

        # Zone d'intérêt ROI
        roi_lf = ttk.LabelFrame(form, text="Zone d'intérêt (ROI) — optionnel")
        roi_lf.grid(row=6, column=0, columnspan=2, sticky=tk.EW, padx=12, pady=6)

        tk.Label(roi_lf, text="Laisser vide pour analyser toute l'image.",
                 fg="gray", font=("Helvetica", 8)).grid(row=0, column=0, columnspan=8,
                                                         sticky=tk.W, padx=4)
        self._roi_vars = {k: tk.StringVar(value="") for k in ("x", "y", "w", "h")}
        for i, (lbl, key) in enumerate([("X", "x"), ("Y", "y"), ("L", "w"), ("H", "h")]):
            tk.Label(roi_lf, text=f"{lbl}:").grid(row=1, column=i * 2, padx=(6 if i == 0 else 2, 0))
            ttk.Entry(roi_lf, textvariable=self._roi_vars[key], width=6).grid(
                row=1, column=i * 2 + 1, padx=(0, 4), pady=4)

        # Activée
        self._enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, text="Caméra activée", variable=self._enabled_var).grid(
            row=7, column=1, sticky=tk.W, **pad
        )

        # Boutons
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=16, pady=(4, 14))
        ttk.Button(btn_frame, text="Annuler", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Valider", command=self._validate).pack(side=tk.RIGHT)

        self._on_type_change()

    def _on_type_change(self) -> None:
        t = self._type_var.get()
        if t == "webcam":
            self._source_var.set("0")
            self._hint_var.set("Index de la caméra (0 = caméra par défaut, 1 = deuxième caméra…)")
        else:
            self._source_var.set("http://")
            self._hint_var.set(
                "URL du flux vidéo.\n"
                "• Android « IP Webcam » : http://192.168.x.x:8080/video\n"
                "• Caméra RTSP           : rtsp://192.168.x.x:554/live"
            )

    def _populate(self, config: CameraConfig) -> None:
        self._name_var.set(config.name)
        self._source_var.set(str(config.source))
        self._width_var.set(str(config.width))
        self._height_var.set(str(config.height))
        self._enabled_var.set(config.enabled)
        self._model_var.set(config.detection_model)
        if config.roi:
            x, y, w, h = config.roi
            self._roi_vars["x"].set(str(x))
            self._roi_vars["y"].set(str(y))
            self._roi_vars["w"].set(str(w))
            self._roi_vars["h"].set(str(h))
        self._on_type_change()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        name = self._name_var.get().strip()
        source_raw = self._source_var.get().strip()
        source_type = self._type_var.get()

        if not name:
            messagebox.showwarning("Champ manquant", "Veuillez saisir un nom.", parent=self)
            return
        if not source_raw:
            messagebox.showwarning("Champ manquant", "Veuillez renseigner la source.", parent=self)
            return

        # Convertir l'index webcam en entier
        if source_type == "webcam":
            try:
                source: str | int = int(source_raw)
            except ValueError:
                messagebox.showerror("Erreur", "L'index doit être un entier (0, 1…).", parent=self)
                return
        else:
            source = source_raw

        # Résolution
        try:
            width = int(self._width_var.get())
            height = int(self._height_var.get())
        except ValueError:
            messagebox.showerror("Erreur", "La résolution doit être en pixels entiers.", parent=self)
            return

        # ROI (optionnelle)
        roi = None
        roi_vals = [self._roi_vars[k].get().strip() for k in ("x", "y", "w", "h")]
        if any(roi_vals):
            try:
                roi = tuple(int(v) for v in roi_vals)
                if len(roi) != 4 or any(v < 0 for v in roi):
                    raise ValueError
            except (ValueError, TypeError):
                messagebox.showerror("Erreur",
                                     "ROI invalide. Remplissez les 4 champs X, Y, Largeur, Hauteur "
                                     "(entiers positifs) ou laissez-les vides.", parent=self)
                return

        uid = self._existing.uid if self._existing else None
        self.result = CameraConfig(
            name=name,
            source_type=source_type,
            source=source,
            enabled=self._enabled_var.get(),
            width=width,
            height=height,
            roi=roi,
            detection_model=self._model_var.get(),
            **({"uid": uid} if uid else {}),
        )
        self.destroy()
